// Portable reference kernels (host). Emulates the ESP32-S3 QACC exactly, including
// saturation, so overflow is caught on the host before it can corrupt device output.
#define _POSIX_C_SOURCE 200112L     // posix_memalign, clock_gettime (the host build uses -std=c11)
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

void tv_dot_rows_i8(const int8_t *x, const int8_t *W, int N, int K, int32_t *out) {
    for (int n = 0; n < N; n++) {
        const int8_t *w = W + (size_t)n * K;
        int32_t s = 0;
        for (int k = 0; k < K; k++) s += x[k] * w[k];
        out[n] = s;
    }
}
