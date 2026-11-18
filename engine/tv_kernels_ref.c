// Portable reference kernels (host). Emulates the ESP32-S3 QACC exactly, including
// saturation, so overflow is caught on the host before it can corrupt device output.
#include "tv_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

uint64_t tv_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000u + ts.tv_nsec / 1000;
}

void *tv_alloc(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 16, n + 64)) {
        fprintf(stderr, "tv: out of memory (%zu bytes)\n", n);
        abort();
    }
    return p;
}

void tv_free(void *p) { free(p); }

void *tv_alloc_fast(size_t n) { return tv_alloc(n); }

void tv_kernels_init(void) {}
int tv_linear_prepare(tv_linear *L) { (void)L; return 0; }
void tv_linear_release(tv_linear *L) { (void)L; }

long tv_linear_run(const tv_linear *L, const int8_t *x, int T, void *out, const tv_epilogue *e) {
    long saturations = 0;
    int K = L->K, chunk = K / L->chunks;
    for (int t = 0; t < T; t++) {
        const int8_t *xt = x + (size_t)t * K;
        for (int n = 0; n < L->N; n++) {
            const int8_t *w = L->wq + (size_t)(n / 16) * K * 16 + n % 16;
            int32_t total = 0;
            for (int ch = 0; ch < L->chunks; ch++) {
                int32_t pre = L->pre ? L->pre[ch * L->N + n] : 0, run = pre;
                for (int k = ch * chunk; k < (ch + 1) * chunk; k++) {
                    run += xt[k] * w[k * 16];
                    if (run > TV_QACC_MAX) { run = TV_QACC_MAX; saturations++; }
                    else if (run < -TV_QACC_MAX - 1) { run = -TV_QACC_MAX - 1; saturations++; }
                }
                total += run - pre;
            }
            tv_epilogue_put(e, out, (size_t)t * L->N + n, (float)total * (L->in_scale * L->ws[n]) + L->bias[n]);
        }
    }
    return saturations;
}

void tv_parallel2(void (*fn)(void *), void *a, void *b) { fn(a); fn(b); }

static int32_t dot48(const int8_t *q, const int8_t *k) {
    int32_t s = 0;
    for (int i = 0; i < 48; i++) s += q[i] * k[i];
    return s;
}

int32_t tv_attn_max48(const int8_t *q, const int8_t *K, int n, int32_t m) {
    for (int t = 0; t < n; t++) {
        int32_t d = dot48(q, K + (size_t)t * 48);
        if (d > m) m = d;
    }
    return m;
}

int32_t tv_attn_weights48(const int8_t *q, const int8_t *K, int n, int32_t m, const uint8_t *lut, uint32_t lim, int8_t *w) {
    int32_t total = 0;
    for (int t = 0; t < n; t++) {
        uint32_t gap = (uint32_t)(m - dot48(q, K + (size_t)t * 48));
        w[t] = (int8_t)lut[gap < lim ? gap : lim];
        total += (uint8_t)w[t];
    }
    return total;
}

void tv_attn_pv48(const int8_t *w, const int8_t *v, int32_t *acc) {
    for (int d = 0; d < 48; d++) {
        int32_t s = 0;
        for (int t = 0; t < 128; t++) s += w[t] * v[d * 128 + t];
        acc[d] += s;
    }
}
