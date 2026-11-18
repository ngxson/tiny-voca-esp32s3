// Compute kernels behind the engine. Two backends with identical integer results:
//   tv_kernels_ref.c  portable C; emulates the S3 QACC (20-bit saturating lanes)
//   firmware/main/tv_kernels_s3.c  ESP32-S3 PIE assembly, both cores
#pragma once
#include "tv.h"

#ifdef __cplusplus
extern "C" {
#endif

// 16-byte aligned, with at least 64 bytes of readable slack after n (kernels over-read).
void *tv_alloc(size_t n);
// The same, in faster memory when there is room (small buffers the kernels read a lot).
void *tv_alloc_fast(size_t n);
void tv_free(void *p);

void tv_kernels_init(void);
// Per-linear preparation after loading (e.g. packing QACC preloads). Returns 0 on success.
int tv_linear_prepare(tv_linear *L);
void tv_linear_release(tv_linear *L);

// What happens to each output y = acc * (L->in_scale * L->ws[n]) + L->bias[n]:
enum { TV_EPI_STORE,        // out (float*)[t][n] = y
       TV_EPI_ADD,          // out (float*)[t][n] += y            (residual stream)
       TV_EPI_Q8 };         // out (int8_t*)[t][n] = lut[q + 128] or q, q = round(y * inv) clamped to +-127
typedef struct {
    int mode;
    float inv;
    const int8_t *lut;
} tv_epilogue;

// acc = sum_k x[t][k] * w[n][k] with QACC semantics (per-chunk preload, 20-bit lanes), then the
// epilogue. x: [T][K] rows, K % 32 == 0. Returns the number of saturated lanes seen (must be 0).
long tv_linear_run(const tv_linear *L, const int8_t *x, int T, void *out, const tv_epilogue *e);

// The engine's float -> int8 rounding (half to even), shared by both backends.
static inline int8_t tv_quant(float x, float inv) {
    float q = x * inv;
    q = q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q;
    return (int8_t)((int32_t)(q + 12582912.0f) - 12582912);
}

static inline void tv_epilogue_put(const tv_epilogue *e, void *out, size_t at, float y) {
    if (e->mode == TV_EPI_STORE) ((float *)out)[at] = y;
    else if (e->mode == TV_EPI_ADD) ((float *)out)[at] += y;
    else {
        int8_t q = tv_quant(y, e->inv);
        ((int8_t *)out)[at] = e->lut ? e->lut[q + 128] : q;
    }
}

// Runs fn(a) and fn(b), on the two cores where there are two. fn must not call tv_linear_run.
void tv_parallel2(void (*fn)(void *), void *a, void *b);

// Attention over one head (dimension 48), dots d[t] = q . K[t*48 : t*48+48] for t < n:
// max(m, d[0..n)).
int32_t tv_attn_max48(const int8_t *q, const int8_t *K, int n, int32_t m);
// w[t] = lut[min(m - d[t], lim)] for t < n (lut[lim] must be 0); returns the sum of the w[t].
int32_t tv_attn_weights48(const int8_t *q, const int8_t *K, int n, int32_t m, const uint8_t *lut, uint32_t lim, int8_t *w);
// acc[d] += sum_t w[t] * v[d*128 + t] for d < 48, t < 128 (one key chunk of transposed values).
void tv_attn_pv48(const int8_t *w, const int8_t *v, int32_t *acc);

#ifdef __cplusplus
}
#endif
