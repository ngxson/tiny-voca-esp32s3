// ESP32-S3 backend of tv_kernels.h: PIE assembly kernels, matmuls split over both cores.
#include "tv_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

void qacc_mm_s8(const int8_t *x, const int8_t *wblk, const uint8_t *raw_in, uint8_t *raw_out, int k32);
int32_t attn_max48(const int8_t *q, const int8_t *K, int n, int32_t m);
int32_t attn_weights48(const int8_t *q, const int8_t *K, int n, int32_t m, const uint32_t *lut_lim, int8_t *w);
void attn_pv64(const int8_t *w, const int8_t *v, int32_t *acc);

uint64_t tv_time_us(void) { return (uint64_t)esp_timer_get_time(); }

void *tv_alloc(size_t n) {
    void *p = heap_caps_aligned_alloc(16, n + 64, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!p) p = heap_caps_aligned_alloc(16, n + 64, MALLOC_CAP_8BIT);
    if (!p) {
        printf("tv: out of memory (%u bytes, largest PSRAM block %u)\n", (unsigned)n,
               (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
        abort();
    }
    return p;
}

void tv_free(void *p) { heap_caps_free(p); }

// Internal RAM, as long as 64 KB of it stays free for the system. Inputs read from there leave the
// external memory bus to the matmul weights streaming from flash.
void *tv_alloc_fast(size_t n) {
    void *p = NULL;
    if (heap_caps_get_free_size(MALLOC_CAP_INTERNAL) > n + 64 * 1024)
        p = heap_caps_aligned_alloc(16, n + 64, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    return p ? p : tv_alloc(n);
}

// ---------------------------------------------------------------- QACC raw layout
static void pack_qacc(const int32_t *v, uint8_t *raw) {      // 16 values (20-bit) -> 64 bytes
    memset(raw, 0, 64);
    for (int h = 0; h < 2; h++)
        for (int i = 0; i < 8; i++) {
            uint32_t u = (uint32_t)v[8 * h + i] & 0xFFFFF;
            int bit = 20 * i;
            for (int b = 0; b < 20; b++, bit++)
                if (u >> b & 1) raw[32 * h + bit / 8] |= 1 << (bit % 8);
        }
}

static inline void unpack_qacc(const uint8_t *raw, int32_t *o) {
    for (int h = 0; h < 2; h++, raw += 32, o += 8) {
        const uint32_t *w = (const uint32_t *)raw;
        uint32_t w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3], w4 = w[4];
        o[0] = (int32_t)(w0 << 12) >> 12;
        o[1] = (int32_t)(((w0 >> 20) | (w1 << 12)) << 12) >> 12;
        o[2] = (int32_t)(w1 << 4) >> 12;
        o[3] = (int32_t)(((w1 >> 28) | (w2 << 4)) << 12) >> 12;
        o[4] = (int32_t)(((w2 >> 16) | (w3 << 16)) << 12) >> 12;
        o[5] = (int32_t)(w3 << 8) >> 12;
        o[6] = (int32_t)(((w3 >> 24) | (w4 << 8)) << 12) >> 12;
        o[7] = (int32_t)w4 >> 12;
    }
}

// Backend data per linear: packed preloads [chunks][Npad/16][64 bytes] and the same values
// unpacked [chunks][Npad] (NULL when there is no preload), and the per-channel output scale.
typedef struct {
    uint8_t *raw_pre;
    int32_t *pre;
    float *scale;               // in_scale * ws[n]
} backend;

static const uint8_t zero_raw[64] __attribute__((aligned(16))) = {0};

int tv_linear_prepare(tv_linear *L) {
    if (L->K % (32 * L->chunks)) return -1;
    int nblocks = (L->N + 15) / 16;
    backend *b = calloc(1, sizeof *b);
    if (!b) return -1;
    b->scale = heap_caps_malloc((size_t)nblocks * 16 * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!b->scale) return -1;
    for (int n = 0; n < nblocks * 16; n++) b->scale[n] = n < L->N ? L->in_scale * L->ws[n] : 0.0f;
    if (L->pre) {
        b->raw_pre = heap_caps_aligned_alloc(16, (size_t)L->chunks * nblocks * 64, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        b->pre = heap_caps_malloc((size_t)L->chunks * nblocks * 16 * sizeof(int32_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!b->raw_pre || !b->pre) return -1;
        for (int c = 0; c < L->chunks; c++)
            for (int nb = 0; nb < nblocks; nb++) {
                int32_t *v = b->pre + ((size_t)c * nblocks + nb) * 16;
                for (int j = 0; j < 16; j++) v[j] = nb * 16 + j < L->N ? L->pre[c * L->N + nb * 16 + j] : 0;
                pack_qacc(v, b->raw_pre + ((size_t)c * nblocks + nb) * 64);
            }
    }
    L->backend = b;
    return 0;
}

void tv_linear_release(tv_linear *L) {
    backend *b = L->backend;
    if (b) { heap_caps_free(b->raw_pre); heap_caps_free(b->pre); heap_caps_free(b->scale); free(b); }
    L->backend = NULL;
}

// ---------------------------------------------------------------- matmul over both cores
typedef struct {
    const tv_linear *L;
    const int8_t *x;
    void *out;
    const tv_epilogue *e;
    int T, nb0, nb1;
    long saturations;
} mm_job;

// Lanes sitting on a 20-bit rail (saturated). |v| reaches TV_QACC_MAX only on the rails, so one
// max over the 16 lanes rules out the common case.
static inline long rail_count(const int32_t *v, int width) {
    int32_t m = 0;
    for (int i = 0; i < 16; i++) {
        int32_t u = v[i] ^ (v[i] >> 31);
        m = u > m ? u : m;
    }
    if (m < TV_QACC_MAX) return 0;
    long n = 0;
    for (int i = 0; i < width; i++) n += v[i] == TV_QACC_MAX || v[i] == -TV_QACC_MAX - 1;
    return n;
}

// total[i] = sum over chunks of (lane - preload) for one row of block nb; returns saturated lanes.
static inline __attribute__((always_inline)) long mm_row(const tv_linear *L, const backend *be, const int8_t *xt,
                                                         const int8_t *wb, int nb, int nblocks, int width, int32_t *total) {
    uint8_t raw[64] __attribute__((aligned(16)));
    if (L->chunks == 1 && !be->raw_pre) {       // every layer but project
        qacc_mm_s8(xt, wb, zero_raw, raw, L->K / 32);
        unpack_qacc(raw, total);
        return rail_count(total, width);
    }
    const int chunk = L->K / L->chunks;
    long sat = 0;
    for (int c = 0; c < L->chunks; c++) {
        int32_t lanes[16];
        const size_t at = (size_t)c * nblocks + nb;
        qacc_mm_s8(xt + c * chunk, wb + (size_t)c * chunk * 16, be->raw_pre ? be->raw_pre + at * 64 : zero_raw, raw, chunk / 32);
        unpack_qacc(raw, lanes);
        sat += rail_count(lanes, width);
        const int32_t *pre = be->pre ? be->pre + at * 16 : NULL;
        for (int i = 0; i < 16; i++) total[i] = (c ? total[i] : 0) + lanes[i] - (pre ? pre[i] : 0);
    }
    return sat;
}

static void mm_run(void *arg) {
    mm_job *j = arg;
    const tv_linear *L = j->L;
    const backend *be = L->backend;
    const int K = L->K, N = L->N, nblocks = (N + 15) / 16;
    const tv_epilogue *e = j->e;
    long sat = 0;
    for (int nb = j->nb0; nb < j->nb1; nb++) {
        const int8_t *wb = L->wq + (size_t)nb * K * 16;
        const int width = N - nb * 16 < 16 ? N - nb * 16 : 16;
        const float *scale = be->scale + nb * 16, *bias = L->bias + nb * 16;
        int32_t total[16];
        for (int t = 0; t < j->T; t++) {
            const int8_t *xt = j->x + (size_t)t * K;
            sat += mm_row(L, be, xt, wb, nb, nblocks, width, total);
            const size_t at = (size_t)t * N + nb * 16;
            if (e->mode == TV_EPI_STORE) {
                float *yt = (float *)j->out + at;
                for (int i = 0; i < width; i++) yt[i] = (float)total[i] * scale[i] + bias[i];
            } else if (e->mode == TV_EPI_ADD) {
                float *yt = (float *)j->out + at;
                for (int i = 0; i < width; i++) yt[i] += (float)total[i] * scale[i] + bias[i];
            } else {
                int8_t *ot = (int8_t *)j->out + at;
                for (int i = 0; i < width; i++) {
                    int8_t q = tv_quant((float)total[i] * scale[i] + bias[i], e->inv);
                    ot[i] = e->lut ? e->lut[q + 128] : q;
                }
            }
        }
    }
    j->saturations = sat;
}

static SemaphoreHandle_t go_sem, done_sem;
static void (*worker_fn)(void *);
static void *worker_arg;

static void worker(void *unused) {
    for (;;) {
        xSemaphoreTake(go_sem, portMAX_DELAY);
        worker_fn(worker_arg);
        xSemaphoreGive(done_sem);
    }
}

void tv_parallel2(void (*fn)(void *), void *a, void *b) {
    worker_fn = fn;
    worker_arg = b;
    xSemaphoreGive(go_sem);
    fn(a);
    xSemaphoreTake(done_sem, portMAX_DELAY);
}


void tv_kernels_init(void) {
    if (go_sem) return;
    go_sem = xSemaphoreCreateBinary();
    done_sem = xSemaphoreCreateBinary();
    xTaskCreatePinnedToCore(worker, "tv_worker", 8192, NULL, configMAX_PRIORITIES - 2, NULL, 1);
}

long tv_linear_run(const tv_linear *L, const int8_t *x, int T, void *out, const tv_epilogue *e) {
    int nblocks = (L->N + 15) / 16;
    if (nblocks < 2) {
        mm_job j = {L, x, out, e, T, 0, nblocks, 0};
        mm_run(&j);
        return j.saturations;
    }
    int split = nblocks / 2;
    mm_job a = {L, x, out, e, T, 0, split, 0}, b = {L, x, out, e, T, split, nblocks, 0};
    tv_parallel2(mm_run, &a, &b);
    return a.saturations + b.saturations;
}

// ---------------------------------------------------------------- attention (one head, dim 48)
int32_t tv_attn_max48(const int8_t *q, const int8_t *K, int n, int32_t m) { return attn_max48(q, K, n, m); }

int32_t tv_attn_weights48(const int8_t *q, const int8_t *K, int n, int32_t m, const uint8_t *lut, uint32_t lim, int8_t *w) {
    const uint32_t lut_lim[2] = {(uint32_t)lut, lim};
    return attn_weights48(q, K, n, m, lut_lim, w);
}

static inline int any_nonzero64(const int8_t *w) {
    const uint32_t *p = (const uint32_t *)w;
    uint32_t o = 0;
    for (int i = 0; i < 16; i++) o |= p[i];
    return o != 0;
}

// Most weights of a peaked attention row are 0: skip the 64-key halves that are all zero.
void tv_attn_pv48(const int8_t *w, const int8_t *v, int32_t *acc) {
    if (any_nonzero64(w)) attn_pv64(w, v, acc);
    if (any_nonzero64(w + 64)) attn_pv64(w + 64, v + 64, acc);
}
