// ESP32-S3 PIE kernels for the tiny-voca engine. Vector pointers must be 16-byte
// aligned; kernels read up to 48 bytes past their inputs (buffers carry slack).
    .text

// void qacc_mm_s8(const int8_t *x, const int8_t *wblk, const uint8_t *raw_in, uint8_t *raw_out, int k32)
// 16 outputs: QACC is loaded from raw_in (64-byte layout: QACC_L low128, high32, QACC_H low128,
// high32, each in a 16-byte slot), QACC[j] += sum_k wblk[k*16+j] * x[k] for K = 32*k32,
// then stored to raw_out. Weights loaded 3 instructions ahead, inputs one chunk ahead.
    .align 4
    .global qacc_mm_s8
    .type qacc_mm_s8, @function
qacc_mm_s8:
    entry a1, 32
    mov a8, a4
    ee.ld.qacc_l.l.128.ip a8, 16
    ee.ld.qacc_l.h.32.ip a8, 16
    ee.ld.qacc_h.l.128.ip a8, 16
    ee.ld.qacc_h.h.32.ip a8, 16
    ee.vld.128.ip q0, a2, 16
    ee.vld.128.ip q1, a3, 16
    ee.vld.128.ip q2, a3, 16
    ee.vld.128.ip q3, a3, 16
    loopnez a6, .Lmm_end
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q0, 0
    ee.vld.128.ip q5, a2, 16
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q0, 1
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q0, 2
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q0, 3
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q0, 4
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q0, 5
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q0, 6
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q0, 7
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q0, 8
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q0, 9
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q0, 10
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q0, 11
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q0, 12
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q0, 13
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q0, 14
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q0, 15
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q5, 0
    ee.vld.128.ip q0, a2, 16
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q5, 1
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q5, 2
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q5, 3
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q5, 4
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q5, 5
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q5, 6
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q5, 7
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q5, 8
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q5, 9
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q5, 10
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q5, 11
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q5, 12
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q5, 13
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q5, 14
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q5, 15
.Lmm_end:
    ee.st.qacc_l.l.128.ip a5, 16
    ee.st.qacc_l.h.32.ip a5, 16
    ee.st.qacc_h.l.128.ip a5, 16
    ee.st.qacc_h.h.32.ip a5, 16
    retw

// int32_t attn_max48(const int8_t *q, const int8_t *K, int n, int32_t m)
// max(m, dot(q, K[t*48 : t*48+48]) for t < n), exact in the 40-bit ACCX: the query stays in
// q0..q2, keys stream through q3..q5 with each load issued by the previous multiply. Reads 16
// bytes past the last key row.
    .align 4
    .global attn_max48
    .type attn_max48, @function
attn_max48:
    entry a1, 32
    ee.vld.128.ip q0, a2, 16
    ee.vld.128.ip q1, a2, 16
    ee.vld.128.ip q2, a2, 16
    ee.vld.128.ip q3, a3, 16
    loopnez a4, .Lmx_end
    ee.zero.accx
    ee.vmulas.s8.accx.ld.ip q4, a3, 16, q0, q3
    ee.vmulas.s8.accx.ld.ip q5, a3, 16, q1, q4
    ee.vmulas.s8.accx.ld.ip q3, a3, 16, q2, q5
    rur.accx_0 a6
    max a5, a5, a6
.Lmx_end:
    mov a2, a5
    retw

// int32_t attn_weights48(const int8_t *q, const int8_t *K, int n, int32_t m, const uint32_t *lut_lim, int8_t *w)
// lut_lim = {lut pointer, lim}. w[t] = lut[minu(m - dot_t, lim)], returns the sum of the w[t].
// Same dot products as attn_max48.
    .align 4
    .global attn_weights48
    .type attn_weights48, @function
attn_weights48:
    entry a1, 32
    l32i a8, a6, 0
    l32i a9, a6, 4
    movi a10, 0
    ee.vld.128.ip q0, a2, 16
    ee.vld.128.ip q1, a2, 16
    ee.vld.128.ip q2, a2, 16
    ee.vld.128.ip q3, a3, 16
    loopnez a4, .Lw_end
    ee.zero.accx
    ee.vmulas.s8.accx.ld.ip q4, a3, 16, q0, q3
    ee.vmulas.s8.accx.ld.ip q5, a3, 16, q1, q4
    ee.vmulas.s8.accx.ld.ip q3, a3, 16, q2, q5
    rur.accx_0 a11
    sub a11, a5, a11
    minu a11, a11, a9
    add a11, a11, a8
    l8ui a11, a11, 0
    s8i a11, a7, 0
    add a10, a10, a11
    addi a7, a7, 1
.Lw_end:
    mov a2, a10
    retw

// void attn_pv64(const int8_t *w, const int8_t *v, int32_t *acc)
// acc[d] += dot(w[0:64], v[d*128 : d*128+64]) for d < 48. The 64 weights stay in q0..q3, value
// rows stream through q4..q7 (row stride 128). Reads one vector past the last row.
    .align 4
    .global attn_pv64
    .type attn_pv64, @function
attn_pv64:
    entry a1, 32
    ee.vld.128.ip q0, a2, 16
    ee.vld.128.ip q1, a2, 16
    ee.vld.128.ip q2, a2, 16
    ee.vld.128.ip q3, a2, 16
    movi a8, 80
    movi a9, 48
    ee.vld.128.ip q4, a3, 16
    loopnez a9, .Lpv_end
    ee.zero.accx
    ee.vmulas.s8.accx.ld.ip q5, a3, 16, q0, q4
    ee.vmulas.s8.accx.ld.ip q6, a3, 16, q1, q5
    ee.vmulas.s8.accx.ld.xp q7, a3, a8, q2, q6
    ee.vmulas.s8.accx.ld.ip q4, a3, 16, q3, q7
    rur.accx_0 a10
    l32i a11, a4, 0
    add a11, a11, a10
    s32i a11, a4, 0
    addi a4, a4, 4
.Lpv_end:
    retw
