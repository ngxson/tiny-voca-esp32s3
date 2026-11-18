// Source-filter vocoder of tiny-voca (model.py SourceFilter), batch size 1.
// Matches torch.stft/istft with center=True, constant padding, periodic Hann window.
#include "tv.h"
#include "tv_kernels.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define SYNTH_FFT 512
#define SYNTH_HOP 128
#define NOISE_FFT 256
#define NOISE_HOP 64
#define CEP_CAUSAL 80
#define CEP_ANTI 16
#define NOISE_CEP 32
#define HARMONIC_LIMIT 0.47

// ---------------------------------------------------------------- fast float math
// rintf and floorf are library calls on the S3. These give the same values (round to nearest
// even for |x| < 2^22, where adding 1.5 * 2^23 leaves no fraction bits; floor for 0 <= x < 2^23),
// up to the sign of a zero rint, which none of the callers can see.
static inline float rint_small(float x) { return (x + 12582912.0f) - 12582912.0f; }
static inline float round_even(float x) { return fabsf(x) < 4194304.0f ? rint_small(x) : rintf(x); }
static inline float floor_pos(float x) {
    float t = (float)(int32_t)x;
    return t > x ? t - 1.0f : t;
}

// exp: Cody-Waite reduction + degree-6 polynomial (~2 ulp). sincos: quadrant reduction + minimax.
static inline float fast_expf(float x) {
    if (x < -87.0f) x = -87.0f;
    if (x > 88.0f) x = 88.0f;
    float n = rint_small(x * 1.44269504f);
    float r = x - n * 0.693145752f - n * 1.42860677e-6f;
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.166666672f + r * (0.0416666679f + r * (0.00833333377f + r * 0.00138888892f)))));
    union { float f; int32_t i; } u = {p};
    u.i += (int32_t)n << 23;
    return u.f;
}

static inline void fast_sincosf(float x, float *s, float *c) {
    float k = round_even(x * 0.636619772f);             // quadrant
    float r = x - k * 1.57079601f - k * 3.13916473e-07f - k * 5.39030253e-15f;
    float r2 = r * r;
    float sn = r + r * r2 * (-0.166666657f + r2 * (0.00833330024f + r2 * -0.000198409713f));
    float cs = 1.0f + r2 * (-0.5f + r2 * (0.0416666418f + r2 * (-0.00138867637f + r2 * 2.43904487e-05f)));
    int q = (int)k & 3;
    *s = q == 0 ? sn : q == 1 ? cs : q == 2 ? -sn : -cs;
    *c = q == 0 ? cs : q == 1 ? -sn : q == 2 ? -cs : sn;
}

// ---------------------------------------------------------------- FFT with cached tables
// Complex radix-2 FFT of size m (128 or 256) used for real FFTs of size 2m.
typedef struct {
    int m;
    float *tw_re, *tw_im;       // exp(-2 pi i j / m), j < m/2
    float *rw_re, *rw_im;       // exp(-2 pi i k / 2m), k <= m (real-FFT split)
    uint16_t *rev;
} fft_plan;

static fft_plan plans[2];

static const fft_plan *plan_for(int n) {           // n = real size (256 or 512)
    int m = n / 2;
    fft_plan *p = &plans[m == 256];
    if (p->m == m) return p;
    p->tw_re = malloc(m / 2 * sizeof(float)); p->tw_im = malloc(m / 2 * sizeof(float));
    p->rw_re = malloc((m + 1) * sizeof(float)); p->rw_im = malloc((m + 1) * sizeof(float));
    p->rev = malloc(m * sizeof(uint16_t));
    for (int j = 0; j < m / 2; j++) { p->tw_re[j] = (float)cos(-2 * M_PI * j / m); p->tw_im[j] = (float)sin(-2 * M_PI * j / m); }
    for (int k = 0; k <= m; k++) { p->rw_re[k] = (float)cos(-2 * M_PI * k / n); p->rw_im[k] = (float)sin(-2 * M_PI * k / n); }
    int bits = 0;
    while ((1 << bits) < m) bits++;
    for (int i = 0; i < m; i++) {
        int r = 0;
        for (int b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
        p->rev[i] = (uint16_t)r;
    }
    p->m = m;
    return p;
}

// In place, unscaled; inverse uses conjugate twiddles.
static void cfft(const fft_plan *p, float *re, float *im, int inverse) {
    int m = p->m;
    for (int i = 0; i < m; i++) {
        int j = p->rev[i];
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    const float sign = inverse ? -1.0f : 1.0f;
    for (int len = 2, step = m / 2; len <= m; len <<= 1, step >>= 1) {
        int half = len / 2;
        for (int i = 0; i < m; i += len)
            for (int k = 0; k < half; k++) {
                float wr = p->tw_re[k * step], wi = sign * p->tw_im[k * step];
                int a = i + k, b = a + half;
                float tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr; im[b] = im[a] - ti;
                re[a] += tr; im[a] += ti;
            }
    }
}

// rfft of n real values -> n/2+1 bins, via a complex FFT of size n/2.
static void rfft(const float *x, int n, float *re, float *im, float *zr, float *zi) {
    const fft_plan *p = plan_for(n);
    int m = n / 2;
    for (int k = 0; k < m; k++) { zr[k] = x[2 * k]; zi[k] = x[2 * k + 1]; }
    cfft(p, zr, zi, 0);
    for (int k = 0; k <= m; k++) {
        int a = k % m, b = (m - k) % m;
        float er = 0.5f * (zr[a] + zr[b]), ei = 0.5f * (zi[a] - zi[b]);      // (Z[k] + conj Z[m-k]) / 2
        float orr = 0.5f * (zi[a] + zi[b]), oi = -0.5f * (zr[a] - zr[b]);    // (Z[k] - conj Z[m-k]) / 2i
        float wr = p->rw_re[k], wi = p->rw_im[k];
        re[k] = er + wr * orr - wi * oi;
        im[k] = ei + wr * oi + wi * orr;
    }
}

// irfft: n/2+1 bins -> n real values (imaginary parts of DC and Nyquist ignored, like torch).
static void irfft(const float *re, const float *im, int n, float *x, float *zr, float *zi) {
    const fft_plan *p = plan_for(n);
    int m = n / 2;
    for (int k = 0; k < m; k++) {
        float ar = re[k], ai = k ? im[k] : 0.0f;
        float br = re[m - k], bi = (m - k) == m ? 0.0f : -im[m - k];         // conj X[m-k]
        if (k == 0) bi = 0.0f;
        float er = 0.5f * (ar + br), ei = 0.5f * (ai + bi);
        float dr = 0.5f * (ar - br), di = 0.5f * (ai - bi);
        float wr = p->rw_re[k], wi = -p->rw_im[k];                            // conj W^k
        float orr = dr * wr - di * wi, oi = dr * wi + di * wr;
        zr[k] = er - oi;                                                      // E + i O
        zi[k] = ei + orr;
    }
    cfft(p, zr, zi, 1);
    float inv = 1.0f / m;
    for (int k = 0; k < m; k++) { x[2 * k] = zr[k] * inv; x[2 * k + 1] = zi[k] * inv; }
}

// exp(FFT(cepstrum)) as a complex response: causal part at the start of the buffer,
// anticausal part (reversed) at its end.
static void cepstrum_response(const float *causal, int nc, const float *anti, int na, int n,
                              float *re, float *im, float *buf, float *tr, float *ti) {
    memset(buf, 0, n * sizeof(float));
    memcpy(buf, causal, nc * sizeof(float));
    for (int i = 0; i < na; i++) buf[n - na + i] = anti[na - 1 - i];
    rfft(buf, n, re, im, tr, ti);
    for (int k = 0; k <= n / 2; k++) {
        float mag = fast_expf(re[k] < -14 ? -14 : re[k] > 6 ? 6 : re[k]);
        float sn, cs;
        fast_sincosf(im[k], &sn, &cs);
        re[k] = mag * cs;
        im[k] = mag * sn;
    }
}

static float lifter(int i, int first) {
    if (first == 0 && i == 0) return 2.0f;
    return 1.0f / (1.0f + (float)(first + i) / 4.0f);
}

// Band-limited pulse train with unit power (model.py pulse_train), produced sample by sample.
// The phase is kept in double cycles (exact enough over minutes); everything else is float.
typedef struct {
    const float *f0;
    int T, s;                   // next sample index
    uint64_t n;                 // the phase in units of 2^-52 while `inexact` is 0 (see phase_step)
    double cycles;              // the phase once a step left that range
    int inexact;
    float k0, k1;               // log f0 at this frame and the next
} pulse_gen;

// n < 2^52 times 2^-52, rounded to float (to nearest even), as (float)(double) would.
static float u52_to_float(uint64_t n) {
    if (n < (1u << 24)) return (float)(uint32_t)n * 0x1p-52f;
    uint32_t hi = (uint32_t)(n >> 32);
    int msb = hi ? 63 - __builtin_clz(hi) : 31 - __builtin_clz((uint32_t)n);
    int sh = msb - 23;                                      // 1 .. 28
    uint32_t mant = (uint32_t)(n >> sh);
    uint64_t rem = n & ((1ULL << sh) - 1), half = 1ULL << (sh - 1);
    if (rem > half || (rem == half && (mant & 1))) mant++;  // may reach 2^24, still exact
    union { uint32_t u; float f; } scale = {(uint32_t)(127 + sh - 52) << 23};
    return (float)mant * scale.f;
}

// cycles += step; if (cycles >= 1) cycles -= 1; returns (float)cycles. The S3 has no double
// hardware, so this uses an integer: with every step in [2^-29, 1), all values are multiples of
// 2^-52 below 2 and those double operations are exact, so the integer gives the same phases.
// A step outside that range switches to doubles for good.
static float phase_step(pulse_gen *g, float step) {
    union { float f; uint32_t u; } a = {step};
    uint32_t e = a.u >> 23;                                 // biased exponent; >= 256 if negative
    if (!g->inexact && e >= 98 && e <= 126) {
        g->n += (uint64_t)((a.u & 0x7FFFFF) | 0x800000) << (e - 98);
        if (g->n >= 1ULL << 52) g->n -= 1ULL << 52;
        return u52_to_float(g->n);
    }
    if (!g->inexact) { g->cycles = (double)g->n * 0x1p-52; g->inexact = 1; }
    g->cycles += step;
    if (g->cycles >= 1.0) g->cycles -= 1.0;
    return (float)g->cycles;
}

static float pulse_next(pulse_gen *g) {
    int f = g->s / TV_HOP, i = g->s % TV_HOP;
    if (i == 0) {
        const float *f0 = g->f0;
        int n = f + 1 < g->T ? f + 1 : g->T - 1;
        g->k0 = logf(f0[f] < 1 ? 1.0f : f0[f]);
        g->k1 = logf(f0[n] < 1 ? 1.0f : f0[n]);
    }
    g->s++;
    float w = (float)i / TV_HOP;
    float freq = fast_expf(g->k0 + (g->k1 - g->k0) * w);
    float phase = phase_step(g, freq * (1.0f / TV_SAMPLE_RATE));     // in [0, 1)
    float th = 6.28318531f * phase;
    float harmonics = (float)HARMONIC_LIMIT * TV_SAMPLE_RATE / freq;
    if (harmonics < 1) harmonics = 1;
    float whole = floor_pos(harmonics);
    float s_half, c_half, s_core, c_core, s_top, c_top;
    fast_sincosf(0.5f * th, &s_half, &c_half);
    float u = (whole + 0.5f) * phase, v = (whole + 1.0f) * phase;     // reduce in cycles first
    fast_sincosf(6.28318531f * (u - floor_pos(u)), &s_core, &c_core);
    fast_sincosf(6.28318531f * (v - floor_pos(v)), &s_top, &c_top);
    float den = 2.0f * s_half;
    float core = fabsf(den) < 1e-7f ? whole : s_core / den - 0.5f;
    float wave = core + (harmonics - whole) * c_top;
    return wave * sqrtf(2.0f / harmonics);
}

static float gaussian(uint64_t *state) {     // xorshift64* + Box-Muller (the engine's own noise)
    float u[2];
    for (int i = 0; i < 2; i++) {
        *state ^= *state >> 12; *state ^= *state << 25; *state ^= *state >> 27;
        u[i] = ((float)((*state * 0x2545F4914F6CDD1DULL) >> 40) + 0.5f) / 16777216.0f;
    }
    return sqrtf(-2.0f * logf(u[0])) * cosf(6.2831853f * u[1]);
}

// ---------------------------------------------------------------- streaming STFT filter
// One branch of the source-filter head: torch.stft (center=True, constant padding) of a source
// followed by fft/2 zeros, frame m multiplied by its response, torch.istft, first S samples.
// Frames are processed in order; padded positions [m*hop, (m+1)*hop) are final after frame m,
// so memory is a few windows long. The arithmetic order equals the batch computation.
#define VOC_BLOCK 8192               // samples per parallel step of the two branches
#define FIFO_SIZE (VOC_BLOCK + 1024)
typedef struct {
    int fft, hop, bins, frames, S, P;
    int m;                      // next frame
    int pos;                    // padded index of the next source sample to read
    int harmonic;               // 1: pulses + harmonic response, 0: noise + noise response
    float win[SYNTH_FFT], win2[SYNTH_FFT], window[SYNTH_FFT], ola[SYNTH_FFT], env[SYNTH_FFT];
    float re[SYNTH_FFT / 2 + 1], im[SYNTH_FFT / 2 + 1], rr[SYNTH_FFT / 2 + 1], ri[SYNTH_FFT / 2 + 1];
    float seg[SYNTH_FFT], buf[SYNTH_FFT], tr[SYNTH_FFT], ti[SYNTH_FFT];
    float fifo[FIFO_SIZE];
    int count;                  // samples waiting in fifo
    int produced;               // samples ever appended to fifo
} branch;

typedef struct {
    const float *h, *n;         // head outputs [T][98], [T][36]
    int T;
    pulse_gen pulses;
    const float *noise;         // given noise, or NULL for the generator
    uint64_t rng;
    int noise_pos;
} sources;

static void branch_init(branch *b, int fft_n, int hop, int harmonic, int S) {
    memset(b, 0, sizeof *b);
    b->fft = fft_n, b->hop = hop, b->bins = fft_n / 2 + 1, b->S = S, b->P = fft_n / 2, b->harmonic = harmonic;
    b->frames = 1 + (S + fft_n / 2) / hop;
    for (int i = 0; i < fft_n; i++) {
        b->window[i] = (float)(0.5 - 0.5 * cos(2 * M_PI * i / fft_n));
        b->win2[i] = b->window[i] * b->window[i];
    }
}

static float next_source(branch *b, sources *src) {
    int j = b->pos++;
    int i = j - b->P;                                   // index in the unpadded source
    if (i < 0 || i >= b->S) return 0.0f;                // centre padding and the fft/2 trailing zeros
    if (b->harmonic) return pulse_next(&src->pulses);
    src->noise_pos++;
    return src->noise ? src->noise[i] : gaussian(&src->rng);
}

static void response(branch *b, const sources *src, int j) {
    const int T = src->T;
    if (b->harmonic) {
        const int steps = TV_HOP / SYNTH_HOP;
        float causal[CEP_CAUSAL], anti[CEP_ANTI];
        int sub = j < steps * T - 1 ? j : steps * T - 1;
        int here = sub / steps, next = (sub + 1) / steps < T - 1 ? (sub + 1) / steps : T - 1;
        const float *a = src->h + (size_t)here * 98, *c = src->h + (size_t)next * 98;
        for (int i = 0; i < CEP_CAUSAL; i++) causal[i] = 0.5f * (a[i] + c[i]) * lifter(i, 0);
        for (int i = 0; i < CEP_ANTI; i++) anti[i] = 0.5f * (a[CEP_CAUSAL + i] + c[CEP_CAUSAL + i]) * lifter(i, 1);
        causal[0] += src->h[(size_t)(sub / steps) * 98 + CEP_CAUSAL + CEP_ANTI + sub % steps];
        cepstrum_response(causal, CEP_CAUSAL, anti, CEP_ANTI, SYNTH_FFT, b->rr, b->ri, b->buf, b->tr, b->ti);
    } else {
        const int steps = TV_HOP / NOISE_HOP;
        float ncep[NOISE_CEP];
        int sub = j < steps * T - 1 ? j : steps * T - 1;
        const float *a = src->n + (size_t)(sub / steps) * 36;
        for (int i = 0; i < NOISE_CEP; i++) ncep[i] = a[i] * lifter(i, 0);
        ncep[0] += a[NOISE_CEP + sub % steps];
        cepstrum_response(ncep, NOISE_CEP, NULL, 0, NOISE_FFT, b->rr, b->ri, b->buf, b->tr, b->ti);
    }
}

// Process the next frame; appends the samples that became final to the fifo.
static void branch_step(branch *b, sources *src) {
    // win, ola and env are rings: logical index i is at (o + i) & mask, so nothing is shifted.
    const int fft_n = b->fft, hop = b->hop, m = b->m, mask = fft_n - 1, o = (m * hop) & mask;
    if (m == 0) {
        for (int i = 0; i < fft_n; i++) b->win[i] = next_source(b, src);
    } else {
        for (int i = fft_n - hop; i < fft_n; i++) b->win[(o + i) & mask] = next_source(b, src);
    }
    uint64_t t0 = tv_time_us();
    response(b, src, m);
    tv_prof[TV_P_VOC_CEP] += tv_time_us() - t0;
    t0 = tv_time_us();
    for (int i = 0; i < fft_n; i++) b->seg[i] = b->win[(o + i) & mask] * b->window[i];
    rfft(b->seg, fft_n, b->re, b->im, b->tr, b->ti);
    for (int k = 0; k < b->bins; k++) {
        float x = b->re[k] * b->rr[k] - b->im[k] * b->ri[k], y = b->re[k] * b->ri[k] + b->im[k] * b->rr[k];
        b->re[k] = x; b->im[k] = y;
    }
    irfft(b->re, b->im, fft_n, b->seg, b->tr, b->ti);
    for (int i = 0; i < fft_n; i++) {
        int k = (o + i) & mask;
        b->ola[k] += b->seg[i] * b->window[i];
        b->env[k] += b->win2[i];
    }
    for (int i = 0; i < hop; i++) {                     // padded positions m*hop + i are final
        int out = m * hop + i - b->P, k = (o + i) & mask;
        float e = b->env[k], v = b->ola[k];
        b->ola[k] = 0.0f;                               // logical index fft_n - hop + i next frame
        b->env[k] = 0.0f;
        if (out < 0 || out >= b->S) continue;
        b->fifo[b->count++] = e > 1e-11f ? v / e : v;
        b->produced++;
    }
    b->m++;
    tv_prof[TV_P_VOC_SHAPE] += tv_time_us() - t0;
}

typedef struct {
    branch *b;
    sources *src;
    int target;                 // run until this many samples were produced
} branch_job;

static void branch_run(void *arg) {
    branch_job *j = arg;
    while (j->b->produced < j->target && j->b->m < j->b->frames) branch_step(j->b, j->src);
}

void tv_vocoder_stream(const float *h, const float *n, const float *f0, const float *noise, uint64_t seed, int T,
                       tv_audio_fn sink, void *user) {
    const int S = T * TV_HOP;
    branch *hb = malloc(sizeof(branch)), *nb = malloc(sizeof(branch));
    float *out = malloc(VOC_BLOCK * sizeof(float));
    sources src = {h, n, T, {f0, T, 0, 0, 0.0, 0, 0, 0}, noise, 0x9E3779B97F4A7C15ULL ^ (seed << 1 | 1), 0};
    branch_init(hb, SYNTH_FFT, SYNTH_HOP, 1, S);
    branch_init(nb, NOISE_FFT, NOISE_HOP, 0, S);
    int emitted = 0;
    while (emitted < S) {
        // Both branches advance on their own core until they cover the next block.
        int block = emitted >= VOC_BLOCK / 2 ? VOC_BLOCK : emitted ? VOC_BLOCK / 2 : VOC_BLOCK / 4;  // short first blocks: sound starts sooner
        int target = emitted + block < S ? emitted + block : S;
        branch_job jh = {hb, &src, target}, jn = {nb, &src, target};
        tv_parallel2(branch_run, &jh, &jn);
        int k = target - emitted;
        for (int i = 0; i < k; i++) out[i] = hb->fifo[i] + nb->fifo[i];
        memmove(hb->fifo, hb->fifo + k, (hb->count - k) * sizeof(float)); hb->count -= k;
        memmove(nb->fifo, nb->fifo + k, (nb->count - k) * sizeof(float)); nb->count -= k;
        sink(user, out, k);
        emitted += k;
    }
    free(hb); free(nb); free(out);
}

typedef struct { float *dst; int n; } collect_state;
static void collect(void *user, const float *x, int n) {
    collect_state *c = user;
    memcpy(c->dst + c->n, x, n * sizeof(float));
    c->n += n;
}

void tv_vocoder(const float *h, const float *n, const float *f0, const float *noise, int T, float *audio) {
    collect_state c = {audio, 0};
    tv_vocoder_stream(h, n, f0, noise, 0, T, collect, &c);
}
