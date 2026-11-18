// tiny-voca engine: model loading and the forward pass. Numerics follow
// esp32/tools/tvq.py (the golden model); see that file for the specification.
#include "tv.h"
#include "tv_kernels.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GELU_MIN (-0.16997124f)
#define VOCAB_UNK 4
#define FIRST_PHONE 17
#define FIRST_VOWEL 43

// ---------------------------------------------------------------- container
int tv_file_open(tv_file *f, const void *data, size_t size) {
    const uint8_t *p = data;
    if (size < 16 || memcmp(p, "TVQ1", 4)) return -1;
    uint32_t version, count;
    memcpy(&version, p + 4, 4);
    memcpy(&count, p + 8, 4);
    if (version != 1 || 16 + (size_t)count * sizeof(tv_entry) > size) return -1;
    f->base = p;
    f->entries = (const tv_entry *)(p + 16);
    f->count = count;
    for (uint32_t i = 0; i < count; i++)
        if ((size_t)f->entries[i].offset + f->entries[i].nbytes > size) return -1;
    return 0;
}

const tv_entry *tv_file_find(const tv_file *f, const char *name) {
    for (uint32_t i = 0; i < f->count; i++)
        if (!strncmp(f->entries[i].name, name, 64)) return &f->entries[i];
    return NULL;
}

const void *tv_file_get(const tv_file *f, const char *name, uint32_t dtype, uint32_t nelem) {
    const tv_entry *e = tv_file_find(f, name);
    static const uint32_t size[] = {4, 1, 4};
    if (!e || e->dtype != dtype || e->nbytes != nelem * size[dtype]) return NULL;
    return f->base + e->offset;
}

// ---------------------------------------------------------------- model loading
typedef struct {
    tv_model *m;
    int failed;
} loader;

static const void *get(loader *L, const char *name, uint32_t dtype, uint32_t nelem) {
    const void *p = tv_file_get(&L->m->file, name, dtype, nelem);
    if (!p && !L->failed) {
        snprintf(L->m->error, sizeof L->m->error, "missing or malformed tensor %s (%u elements)", name, (unsigned)nelem);
        L->failed = 1;
    }
    return p;
}

static const float *getf(loader *L, const char *fmt, const char *prefix, uint32_t nelem) {
    char name[96];
    snprintf(name, sizeof name, fmt, prefix);
    return get(L, name, TV_F32, nelem);
}

static float scale_of(loader *L, const char *prefix, const char *point) {
    char name[96];
    snprintf(name, sizeof name, "%s.%s.max", prefix, point);
    const float *v = get(L, name, TV_F32, 1);
    return v ? (float)((double)*v / 127.0) : 1.0f;
}

static void load_linear(loader *L, tv_linear *lin, const char *base, int N, int K, float in_scale, int chunks) {
    char name[96];
    int npad = (N + 15) / 16 * 16;
    lin->N = N, lin->K = K, lin->in_scale = in_scale, lin->chunks = 1, lin->pre = NULL;
    snprintf(name, sizeof name, "%s.wq", base); lin->wq = get(L, name, TV_I8, npad * K);
    snprintf(name, sizeof name, "%s.ws", base); lin->ws = get(L, name, TV_F32, N);
    snprintf(name, sizeof name, "%s.b", base); lin->b = get(L, name, TV_F32, N);
    if (chunks > 1) {
        snprintf(name, sizeof name, "%s.pre", base);
        lin->pre = get(L, name, TV_I32, chunks * N);
        lin->chunks = chunks;
    }
    lin->bias = calloc(N, sizeof(float));
    if (lin->b) memcpy(lin->bias, lin->b, N * sizeof(float));
    if (!L->failed && tv_linear_prepare(lin)) {
        snprintf(L->m->error, sizeof L->m->error, "kernel preparation failed for %s", base);
        L->failed = 1;
    }
}

static float gelu(float x) { return 0.5f * x * (1.0f + erff(x * (float)M_SQRT1_2)); }

static void load_block(loader *L, tv_block *b, const char *prefix) {
    char base[96];
    b->norm1_w = getf(L, "%s.norm1.weight", prefix, TV_C);
    b->norm1_b = getf(L, "%s.norm1.bias", prefix, TV_C);
    b->norm2_w = getf(L, "%s.norm2.weight", prefix, TV_C);
    b->norm2_b = getf(L, "%s.norm2.bias", prefix, TV_C);
    b->dw_w = getf(L, "%s.depthwise.weight", prefix, TV_C * TV_KERNEL);
    b->dw_b = getf(L, "%s.depthwise.bias", prefix, TV_C);
    b->s_ln1 = scale_of(L, prefix, "ln1");
    b->s_q = scale_of(L, prefix, "q");
    b->s_k = scale_of(L, prefix, "k");
    b->s_v = scale_of(L, prefix, "v");
    b->s_attn = scale_of(L, prefix, "attn");
    b->s_dw = scale_of(L, prefix, "dw");
    b->s_pre = scale_of(L, prefix, "pre");
    float gmax = scale_of(L, prefix, "gelu") * 127.0f;     // undo the /127: the stored max
    char name[96];
    snprintf(name, sizeof name, "%s.gelu.max", prefix);
    const float *gm = tv_file_get(&L->m->file, name, TV_F32, 1);
    if (gm) gmax = *gm;
    b->s_gelu = (float)(((double)gmax - (double)GELU_MIN) / 255.0);
    snprintf(base, sizeof base, "%s.attention.qkv", prefix); load_linear(L, &b->qkv, base, 3 * TV_C, TV_C, b->s_ln1, 1);
    snprintf(base, sizeof base, "%s.attention.out", prefix); load_linear(L, &b->out, base, TV_C, TV_C, b->s_attn, 1);
    snprintf(base, sizeof base, "%s.expand", prefix); load_linear(L, &b->expand, base, TV_FF, TV_C, b->s_dw, 1);
    snprintf(base, sizeof base, "%s.project", prefix); load_linear(L, &b->project, base, TV_C, TV_FF, b->s_gelu, 2);
    if (L->failed) return;
    // GELU table: pre code c -> asymmetric gelu code, exactly as the golden model rounds.
    for (int c = -128; c <= 127; c++) {
        float g = gelu((float)(c < -127 ? -127 : c) * b->s_pre);
        float code = rintf((g - GELU_MIN) / b->s_gelu);
        code = code < 0 ? 0 : code > 255 ? 255 : code;
        b->gelu_lut[c + 128] = (int8_t)(code - 128);
    }
    // Attention weights depend only on the integer gap to the row's max dot product.
    double score_scale = (double)b->s_q * b->s_k / sqrt((double)TV_HEAD_DIM);
    b->softmax_n = (int)ceil(log(254.0) / score_scale) + 1;      // 127 * exp(-gap * s) < 0.5 beyond
    b->softmax_lut = malloc(b->softmax_n + 1);
    double v = 127.0, step = exp(-score_scale);             // 127 * exp(-g * s) by repeated multiplication
    for (int g = 0; g < b->softmax_n; g++, v *= step) b->softmax_lut[g] = (uint8_t)lrint(v);
    b->softmax_lut[b->softmax_n] = 0;                       // gaps are clamped to softmax_n
    // project reads codes q standing for (q + 128) * s_gelu + GELU_MIN: fold the constant part.
    tv_linear *p = &b->project;
    for (int n = 0; n < p->N; n++) {
        long rowsum = 0;
        const int8_t *w = p->wq + (n / 16) * p->K * 16 + n % 16;
        for (int k = 0; k < p->K; k++) rowsum += w[k * 16];
        p->bias[n] = p->b[n] + p->ws[n] * (128.0f * b->s_gelu + GELU_MIN) * (float)rowsum;
    }
}

static void load_stack(loader *L, tv_stack *s, tv_block *storage, const char *name, int count) {
    char prefix[96];
    s->blocks = storage;
    s->count = count;
    for (int i = 0; i < count; i++) {
        snprintf(prefix, sizeof prefix, "%s.blocks.%d", name, i);
        load_block(L, &storage[i], prefix);
    }
    s->norm_w = getf(L, "%s.norm.weight", name, TV_C);
    s->norm_b = getf(L, "%s.norm.bias", name, TV_C);
}

int tv_model_init(tv_model *m, const void *data, size_t size) {
    memset(m, 0, sizeof *m);
    tv_kernels_init();
    if (tv_file_open(&m->file, data, size)) {
        snprintf(m->error, sizeof m->error, "not a TVQ1 file");
        return -1;
    }
    loader L = {m, 0};
    m->tokens = get(&L, "tokens.weight", TV_F32, 76 * TV_C);
    m->stress = get(&L, "stress.weight", TV_F32, 3 * TV_C);
    m->voices = get(&L, "voices.weight", TV_F32, 2 * TV_C);
    m->frame_voices = get(&L, "frame_voices.weight", TV_F32, 2 * TV_C);
    load_stack(&L, &m->encoder, m->block_storage, "encoder", 4);
    load_stack(&L, &m->prosody, m->block_storage + 4, "prosody_stack", 2);
    load_stack(&L, &m->acoustic, m->block_storage + 6, "acoustic_stack", 5);
    load_linear(&L, &m->dur1, "duration.conv1", TV_PRED, 3 * TV_C, scale_of(&L, "dur", "in"), 1);
    load_linear(&L, &m->dur2, "duration.conv2", TV_PRED, 3 * TV_PRED, scale_of(&L, "dur", "h1"), 1);
    m->dnorm1_w = get(&L, "duration.norm1.weight", TV_F32, TV_PRED);
    m->dnorm1_b = get(&L, "duration.norm1.bias", TV_F32, TV_PRED);
    m->dnorm2_w = get(&L, "duration.norm2.weight", TV_F32, TV_PRED);
    m->dnorm2_b = get(&L, "duration.norm2.bias", TV_F32, TV_PRED);
    m->dur_out_w = get(&L, "duration.out.weight", TV_F32, 2 * TV_PRED);
    m->dur_out_b = get(&L, "duration.out.bias", TV_F32, 2);
    m->frame_pos_w = get(&L, "frame_position.weight", TV_F32, TV_C * 3);
    m->frame_pos_b = get(&L, "frame_position.bias", TV_F32, TV_C);
    m->prosody_head_w = get(&L, "prosody_head.weight", TV_F32, 3 * TV_C);
    m->prosody_head_b = get(&L, "prosody_head.bias", TV_F32, 3);
    m->prosody_proj_w = get(&L, "prosody_projection.weight", TV_F32, TV_C * 3);
    m->prosody_proj_b = get(&L, "prosody_projection.bias", TV_F32, TV_C);
    m->s_voc = scale_of(&L, "voc", "in");
    load_linear(&L, &m->harmonic, "vocoder.harmonic", 98, TV_C, m->s_voc, 1);
    load_linear(&L, &m->noise, "vocoder.noise", 36, TV_C, m->s_voc, 1);
    return L.failed ? -1 : 0;
}

static void free_linear(tv_linear *l) { tv_linear_release(l); free(l->bias); l->bias = NULL; }

void tv_model_free(tv_model *m) {
    for (int i = 0; i < 11; i++) {
        tv_block *b = &m->block_storage[i];
        free_linear(&b->qkv); free_linear(&b->out); free_linear(&b->expand); free_linear(&b->project);
        free(b->softmax_lut); b->softmax_lut = NULL;
    }
    free_linear(&m->dur1); free_linear(&m->dur2); free_linear(&m->harmonic); free_linear(&m->noise);
}

// ---------------------------------------------------------------- profiling
uint64_t tv_prof[TV_P_COUNT];
static const char *const prof_names[TV_P_COUNT] = {"matmul (int8)", "matmul epilogue", "attention pass 1 (cpu)",
    "softmax", "attention pass 2 (cpu)",
    "layernorm", "depthwise conv", "quantize", "gelu table", "rotary + q/k/v codes", "vocoder cepstrum", "vocoder stft filter",
    "vocoder pulses"};
void tv_profile_reset(void) { memset(tv_prof, 0, sizeof tv_prof); }
void tv_profile_report(void) {
    uint64_t total = 0;
    for (int i = 0; i < TV_P_COUNT; i++) total += tv_prof[i];
    for (int i = 0; i < TV_P_COUNT; i++)
        printf("  %-22s %9.1f ms  %5.1f%%\n", prof_names[i], tv_prof[i] / 1000.0, total ? 100.0 * tv_prof[i] / total : 0.0);
    printf("  %-22s %9.1f ms\n", "(sum of the above)", total / 1000.0);
}

// ---------------------------------------------------------------- ops
typedef struct {
    const tv_options *opt;
    long saturations;
} ctx;

static void dbg(ctx *c, const char *name, int dtype, const void *data, int rows, int cols) {
    if (c->opt->debug) c->opt->debug(c->opt->debug_user, name, dtype, data, rows, cols);
}

#define xmalloc tv_alloc


static void quantize(const float *x, int n, float scale, int8_t *q) {
    uint64_t t0 = tv_time_us();
    const float inv = 1.0f / scale;
    for (int i = 0; i < n; i++) q[i] = tv_quant(x[i], inv);
    tv_prof[TV_P_QUANT] += tv_time_us() - t0;
}

// y = dequantized matmul + bias (float output).
static void linear_f(ctx *c, const tv_linear *L, const int8_t *x, int T, float *y) {
    tv_epilogue e = {TV_EPI_STORE, 0, NULL};
    TV_PROF(TV_P_MATMUL, c->saturations += tv_linear_run(L, x, T, y, &e));
}

// residual[t][n] += linear output
static void linear_add(ctx *c, const tv_linear *L, const int8_t *x, int T, float *residual) {
    tv_epilogue e = {TV_EPI_ADD, 0, NULL};
    TV_PROF(TV_P_MATMUL, c->saturations += tv_linear_run(L, x, T, residual, &e));
}

// out = lut[quantized linear output] (int8)
static void linear_q8(ctx *c, const tv_linear *L, const int8_t *x, int T, int8_t *out, float scale, const int8_t *lut) {
    tv_epilogue e = {TV_EPI_Q8, 1.0f / scale, lut};
    TV_PROF(TV_P_MATMUL, c->saturations += tv_linear_run(L, x, T, out, &e));
}

// LayerNorm straight to int8 codes (same arithmetic as layer_norm followed by quantize).
static void layer_norm_q8_rows(const float *x, int t_begin, int T, int C, const float *w, const float *b, float scale, int8_t *q) {
    const float qinv = 1.0f / scale;
    for (int t = t_begin; t < T; t++) {
        const float *xt = x + (size_t)t * C;
        int8_t *qt = q + (size_t)t * C;
        float mean = 0, var = 0;
        for (int i = 0; i < C; i++) mean += xt[i];
        mean /= C;
        for (int i = 0; i < C; i++) var += (xt[i] - mean) * (xt[i] - mean);
        float inv = 1.0f / sqrtf(var / C + 1e-5f);
        for (int i = 0; i < C; i++) {
            float y = (xt[i] - mean) * inv * w[i] + b[i];
            qt[i] = tv_quant(y, qinv);
        }
    }
}

static void layer_norm_rows(const float *x, int t_begin, int T, int C, const float *w, const float *b, float *y) {
    for (int t = t_begin; t < T; t++) {
        const float *xt = x + (size_t)t * C;
        float *yt = y + (size_t)t * C;
        float mean = 0, var = 0;
        for (int i = 0; i < C; i++) mean += xt[i];
        mean /= C;
        for (int i = 0; i < C; i++) var += (xt[i] - mean) * (xt[i] - mean);
        float inv = 1.0f / sqrtf(var / C + 1e-5f);
        for (int i = 0; i < C; i++) yt[i] = (xt[i] - mean) * inv * w[i] + b[i];
    }
}

static void layer_norm(const float *x, int T, int C, const float *w, const float *b, float *y) {
    TV_PROF(TV_P_NORM, layer_norm_rows(x, 0, T, C, w, b, y));
}

static void windows3(const float *x, int T, int C, float *y) {
    for (int t = 0; t < T; t++)
        for (int j = 0; j < 3; j++) {
            int s = t + j - 1;
            float *dst = y + (size_t)t * 3 * C + j * C;
            if (s < 0 || s >= T) memset(dst, 0, C * sizeof(float));
            else memcpy(dst, x + (size_t)s * C, C * sizeof(float));
        }
}

// Rotary tables for positions t0 .. t0+rows-1.
static void rotary_rows(int t0, int rows, float *cos_t, float *sin_t) {
    const int half = TV_HEAD_DIM / 2;
    for (int i = 0; i < half; i++) {
        float inv = 1.0f / powf(10000.0f, (float)(2 * i) / (float)TV_HEAD_DIM);
        for (int t = 0; t < rows; t++) {
            float a = (float)(t0 + t) * inv;
            cos_t[t * half + i] = cosf(a);
            sin_t[t * half + i] = sinf(a);
        }
    }
}

// ---------------------------------------------------------------- tiled transformer stack
// Each block runs over row tiles of TV_TILE positions. Only the residual stream (the caller's
// buffer) and the int8 queries, keys (per head) and values (per head, transposed) span the whole
// utterance: about 0.6 KB per position here plus the residual, instead of ~10 KB untiled.
#define TV_TILE 64
#define TV_KCH 128                  // keys per chunk: 6 KB of keys and 6 KB of values stay in cache

typedef struct {                    // a debug tensor assembled from tiles (debug runs only)
    uint8_t *data;
} dbg_buf;

static void dbg_tile(ctx *c, dbg_buf *d, const char *name, int dtype, const void *tile, int row0, int rows, int cols, int T) {
    if (!c->opt->debug) return;
    int es = dtype == TV_I8 ? 1 : 4;
    if (!d->data) d->data = xmalloc((size_t)T * cols * es);
    memcpy(d->data + (size_t)row0 * cols * es, tile, (size_t)rows * cols * es);
    if (row0 + rows == T) {
        dbg(c, name, dtype, d->data, T, cols);
        tv_free(d->data);
        d->data = NULL;
    }
}

typedef struct {
    int T, nchunks;                 // nchunks = ceil(T / TV_KCH)
    int8_t *q8;                     // [T][192]
    int8_t *kh;                     // [heads][T][48]
    int8_t *vt;                     // [heads][nchunks][48][TV_KCH] values, transposed per key chunk
    int32_t *vacc[2];               // [TV_TILE][48] per core: weighted value sums of one head
    int32_t *cmax[2];               // [TV_TILE][nchunks] per core: max dot product per key chunk
    int8_t *w8[2];                  // [TV_TILE][TV_KCH] per core: attention weights of one key chunk
    float *cos_t, *sin_t;           // [T][24] rotary tables (whole utterance)
    float *h, *qkv, *ln2;           // tile buffers (h: debug runs only)
    int8_t *i8, *g8;                // tile int8 buffers (matmul input / expand output)
    dbg_buf d_ln1, d_attn, d_dw, d_pre, d_gelu;
} scratch;

static void scratch_alloc(scratch *s, int T, int debug) {
    memset(s, 0, sizeof *s);
    s->T = T;
    s->nchunks = (T + TV_KCH - 1) / TV_KCH;
    size_t vsize = (size_t)TV_HEADS * s->nchunks * TV_HEAD_DIM * TV_KCH;
    s->q8 = xmalloc((size_t)T * TV_C);
    s->kh = xmalloc((size_t)TV_HEADS * T * TV_HEAD_DIM);
    s->vt = xmalloc(vsize + 64);                                        // tv_attn_pv48 reads past a chunk
    memset(s->vt, 0, vsize);                                            // keys past T stay 0
    for (int i = 0; i < 2; i++) s->vacc[i] = xmalloc((size_t)TV_TILE * TV_HEAD_DIM * sizeof(int32_t));
    for (int i = 0; i < 2; i++) s->cmax[i] = xmalloc((size_t)TV_TILE * s->nchunks * sizeof(int32_t));
    for (int i = 0; i < 2; i++) s->w8[i] = tv_alloc_fast((size_t)TV_TILE * TV_KCH);
    if (debug) s->h = xmalloc((size_t)TV_TILE * TV_FF * sizeof(float));     // unfused expand only
    s->qkv = xmalloc((size_t)TV_TILE * 3 * TV_C * sizeof(float));
    s->ln2 = xmalloc((size_t)(TV_TILE + 6) * TV_C * sizeof(float));
    s->cos_t = xmalloc((size_t)T * TV_HEAD_DIM / 2 * sizeof(float));
    s->sin_t = xmalloc((size_t)T * TV_HEAD_DIM / 2 * sizeof(float));
    TV_PROF(TV_P_ROPE, rotary_rows(0, T, s->cos_t, s->sin_t));
    s->i8 = tv_alloc_fast((size_t)TV_TILE * TV_C);                     // matmul inputs
    s->g8 = tv_alloc_fast((size_t)TV_TILE * TV_FF);
}

static void scratch_free(scratch *s) {
    tv_free(s->q8); tv_free(s->kh); tv_free(s->vt);
    for (int i = 0; i < 2; i++) { tv_free(s->vacc[i]); tv_free(s->cmax[i]); tv_free(s->w8[i]); }
    tv_free(s->h); tv_free(s->qkv); tv_free(s->ln2); tv_free(s->cos_t); tv_free(s->sin_t);
    tv_free(s->i8); tv_free(s->g8);
}

static inline size_t vt_index(const scratch *s, int h, int t, int d) {
    return (((size_t)h * s->nchunks + t / TV_KCH) * TV_HEAD_DIM + d) * TV_KCH + t % TV_KCH;
}

// Debug views of q/k/v in the golden layout [head][t][dim].
static void dbg_qkv(ctx *c, const char *prefix, const scratch *s) {
    if (!c->opt->debug) return;
    const int T = s->T;
    char name[96];
    int8_t *tmp = xmalloc((size_t)T * TV_C);
    for (int h = 0; h < TV_HEADS; h++)
        for (int t = 0; t < T; t++)
            memcpy(tmp + ((size_t)h * T + t) * TV_HEAD_DIM, s->q8 + (size_t)t * TV_C + h * TV_HEAD_DIM, TV_HEAD_DIM);
    snprintf(name, sizeof name, "%s.q", prefix); dbg(c, name, TV_I8, tmp, TV_HEADS * T, TV_HEAD_DIM);
    snprintf(name, sizeof name, "%s.k", prefix); dbg(c, name, TV_I8, s->kh, TV_HEADS * T, TV_HEAD_DIM);
    for (int h = 0; h < TV_HEADS; h++)
        for (int t = 0; t < T; t++)
            for (int d = 0; d < TV_HEAD_DIM; d++)
                tmp[((size_t)h * T + t) * TV_HEAD_DIM + d] = s->vt[vt_index(s, h, t, d)];
    snprintf(name, sizeof name, "%s.v", prefix); dbg(c, name, TV_I8, tmp, TV_HEADS * T, TV_HEAD_DIM);
    tv_free(tmp);
}

// Attention for queries [q0, q0+qn) and heads [h0, h1), writing y rows (q - t0). Two passes over
// key chunks, so no score matrix is stored: pass 1 finds each query's max dot product, pass 2
// recomputes the dots, turns them into int8 weights and accumulates weights . values per chunk.
// Integer sums, so the result equals the one-pass computation exactly.
typedef struct {
    const tv_block *b;
    scratch *s;
    int q0, qn, t0, h0, h1, set;
    uint64_t t_qk, t_soft, t_pv;    // time per phase on this job's core
} attn_job;

static void attn_run(void *arg) {
    attn_job *j = arg;
    scratch *s = j->s;
    const tv_block *b = j->b;
    const int T = s->T;
    int32_t *acc = s->vacc[j->set], *cmax = s->cmax[j->set];
    const int nch = s->nchunks;
    int32_t mx[TV_TILE], sum[TV_TILE];
    int8_t *w8 = s->w8[j->set];
    const uint8_t *lut = b->softmax_lut;
    const uint32_t lim = (uint32_t)b->softmax_n;
    for (int h = j->h0; h < j->h1; h++) {
        const int8_t *kh = s->kh + (size_t)h * T * TV_HEAD_DIM;
        const int8_t *vt = s->vt + (size_t)h * s->nchunks * TV_HEAD_DIM * TV_KCH;
        uint64_t t0 = tv_time_us();
        for (int r = 0; r < j->qn; r++) mx[r] = INT32_MIN;
        for (int c = 0; c < s->nchunks; c++) {
            int k0 = c * TV_KCH, kn = T - k0 < TV_KCH ? T - k0 : TV_KCH;
            for (int r = 0; r < j->qn; r++) {
                int32_t cm = tv_attn_max48(s->q8 + (size_t)(j->q0 + r) * TV_C + h * TV_HEAD_DIM, kh + (size_t)k0 * TV_HEAD_DIM, kn, INT32_MIN);
                cmax[r * nch + c] = cm;
                if (cm > mx[r]) mx[r] = cm;
            }
        }
        uint64_t t1 = tv_time_us();
        memset(acc, 0, (size_t)j->qn * TV_HEAD_DIM * sizeof(int32_t));
        memset(sum, 0, sizeof sum);
        for (int c = 0; c < s->nchunks; c++) {
            int k0 = c * TV_KCH, kn = T - k0 < TV_KCH ? T - k0 : TV_KCH;
            const int8_t *vc = vt + (size_t)c * TV_HEAD_DIM * TV_KCH;
            // All weights of the chunk first, then all weights . values: each step streams one
            // chunk (keys, then values) instead of alternating between the two.
            for (int r = 0; r < j->qn; r++) {
                if ((uint32_t)(mx[r] - cmax[r * nch + c]) >= lim) continue;   // every weight of the chunk is 0
                int8_t *w = w8 + (size_t)r * TV_KCH;
                if (kn < TV_KCH) memset(w + kn, 0, TV_KCH - kn);
                // Row-max scaled int8 weights (largest = 127), normalized by their own sum.
                sum[r] += tv_attn_weights48(s->q8 + (size_t)(j->q0 + r) * TV_C + h * TV_HEAD_DIM, kh + (size_t)k0 * TV_HEAD_DIM,
                                            kn, mx[r], lut, lim, w);
            }
            for (int r = 0; r < j->qn; r++)
                if ((uint32_t)(mx[r] - cmax[r * nch + c]) < lim) tv_attn_pv48(w8 + (size_t)r * TV_KCH, vc, acc + (size_t)r * TV_HEAD_DIM);
        }
        const float inv_attn = 1.0f / b->s_attn;
        for (int r = 0; r < j->qn; r++) {           // attention output, straight to the int8 codes of `out`
            const int32_t *ar = acc + (size_t)r * TV_HEAD_DIM;
            int8_t *yr = s->i8 + (size_t)(j->q0 - j->t0 + r) * TV_C + h * TV_HEAD_DIM;
            for (int d = 0; d < TV_HEAD_DIM; d++) yr[d] = tv_quant((float)ar[d] * (b->s_v / (float)sum[r]), inv_attn);
        }
        j->t_qk += t1 - t0; j->t_pv += tv_time_us() - t1;
    }
}

// Row-wise tile work. Every row is computed exactly as before; the rows are just split over the
// two cores.
enum { TILE_LN1, TILE_QKV, TILE_LN2, TILE_DW };
typedef struct {
    int op, r0, r1;                 // rows [r0, r1) of the tile
    const tv_block *b;
    scratch *s;
    const float *x;                 // residual rows of the tile
    int t0, T;
} tile_job;

static void tile_rows(void *arg) {
    const tile_job *j = arg;
    const tv_block *b = j->b;
    scratch *s = j->s;
    const int half = TV_HEAD_DIM / 2, t0 = j->t0, T = j->T;
    switch (j->op) {
    case TILE_LN1:
        layer_norm_q8_rows(j->x, j->r0, j->r1, TV_C, b->norm1_w, b->norm1_b, b->s_ln1, s->i8);
        break;
    case TILE_QKV: {                // rotary on q and k, then int8 q, k (per head) and v (transposed)
        int8_t kq[TV_C], vq[TV_C];
        const float iq = 1.0f / b->s_q, ik = 1.0f / b->s_k, iv = 1.0f / b->s_v;
        const float *cos_t = s->cos_t + (size_t)t0 * half, *sin_t = s->sin_t + (size_t)t0 * half;
        for (int r = j->r0; r < j->r1; r++) {
            const float *row = s->qkv + (size_t)r * 3 * TV_C, *cs = cos_t + r * half, *sn = sin_t + r * half;
            int t = t0 + r;
            int8_t *q = s->q8 + (size_t)t * TV_C;
            for (int part = 0; part < 2; part++) {
                const float *z = row + part * TV_C;
                int8_t *o = part ? kq : q;
                const float inv = part ? ik : iq;
                for (int h = 0; h < TV_HEADS; h++, z += TV_HEAD_DIM, o += TV_HEAD_DIM)
                    for (int i = 0; i < half; i++) {
                        float a = z[2 * i], bb = z[2 * i + 1];
                        o[2 * i] = tv_quant(a * cs[i] - bb * sn[i], inv);
                        o[2 * i + 1] = tv_quant(a * sn[i] + bb * cs[i], inv);
                    }
            }
            for (int i = 0; i < TV_C; i++) vq[i] = tv_quant(row[2 * TV_C + i], iv);
            for (int h = 0; h < TV_HEADS; h++) {
                memcpy(s->kh + ((size_t)h * T + t) * TV_HEAD_DIM, kq + h * TV_HEAD_DIM, TV_HEAD_DIM);
                for (int d = 0; d < TV_HEAD_DIM; d++) s->vt[vt_index(s, h, t, d)] = vq[h * TV_HEAD_DIM + d];
            }
        }
        break;
    }
    case TILE_LN2:                  // ln2 row r + 3 is position t0 + r
        layer_norm_rows(j->x, j->r0, j->r1, TV_C, b->norm2_w, b->norm2_b, s->ln2 + 3 * TV_C);
        break;
    case TILE_DW: {
        const float dw_inv = 1.0f / b->s_dw;
        for (int r = j->r0; r < j->r1; r++) {
            if (t0 + r >= TV_KERNEL / 2 && t0 + r + TV_KERNEL / 2 < T) {
                // All taps inside: no bounds checks, and 4 channels at a time so their sums
                // overlap (each still adds its taps in order).
                const float *x0 = s->ln2 + (size_t)r * TV_C;       // ln2 row r + k is tap k
                int8_t *o = s->i8 + (size_t)r * TV_C;
                for (int ch = 0; ch < TV_C; ch += 4) {
                    const float *w = b->dw_w + ch * TV_KERNEL, *x = x0 + ch;
                    float v0 = b->dw_b[ch], v1 = b->dw_b[ch + 1], v2 = b->dw_b[ch + 2], v3 = b->dw_b[ch + 3];
                    for (int k = 0; k < TV_KERNEL; k++, x += TV_C) {
                        v0 += w[k] * x[0];
                        v1 += w[TV_KERNEL + k] * x[1];
                        v2 += w[2 * TV_KERNEL + k] * x[2];
                        v3 += w[3 * TV_KERNEL + k] * x[3];
                    }
                    o[ch] = tv_quant(v0, dw_inv); o[ch + 1] = tv_quant(v1, dw_inv);
                    o[ch + 2] = tv_quant(v2, dw_inv); o[ch + 3] = tv_quant(v3, dw_inv);
                }
                continue;
            }
            for (int ch = 0; ch < TV_C; ch++) {
                float v = b->dw_b[ch];
                for (int k = 0; k < TV_KERNEL; k++) {
                    int src = t0 + r + k - TV_KERNEL / 2;
                    if (src >= 0 && src < T) v += b->dw_w[ch * TV_KERNEL + k] * s->ln2[(size_t)(src - t0 + 3) * TV_C + ch];
                }
                s->i8[(size_t)r * TV_C + ch] = tv_quant(v, dw_inv);
            }
        }
        break;
    }
    }
}

static void tile_parallel(int prof, int op, int rows, const tv_block *b, scratch *s, const float *x, int t0, int T) {
    uint64_t start = tv_time_us();
    tile_job ja = {op, 0, rows / 2, b, s, x, t0, T}, jb = {op, rows / 2, rows, b, s, x, t0, T};
    tv_parallel2(tile_rows, &ja, &jb);
    tv_prof[prof] += tv_time_us() - start;
}

static void block_forward(ctx *c, const tv_block *b, const char *prefix, float *x, int T, scratch *s) {
    char name[96];
    // ---- pass 1: queries, keys and values of every position
    for (int t0 = 0; t0 < T; t0 += TV_TILE) {
        int rows = T - t0 < TV_TILE ? T - t0 : TV_TILE;
        tile_parallel(TV_P_NORM, TILE_LN1, rows, b, s, x + (size_t)t0 * TV_C, t0, T);
        snprintf(name, sizeof name, "%s.ln1", prefix); dbg_tile(c, &s->d_ln1, name, TV_I8, s->i8, t0, rows, TV_C, T);
        linear_f(c, &b->qkv, s->i8, rows, s->qkv);
        tile_parallel(TV_P_ROPE, TILE_QKV, rows, b, s, NULL, t0, T);
    }
    dbg_qkv(c, prefix, s);
    // ---- pass 2: attention per query tile (heads split over the cores), output projection, residual
    for (int t0 = 0; t0 < T; t0 += TV_TILE) {
        int rows = T - t0 < TV_TILE ? T - t0 : TV_TILE;
        attn_job ja = {b, s, t0, rows, t0, 0, TV_HEADS / 2, 0, 0, 0, 0}, jb = {b, s, t0, rows, t0, TV_HEADS / 2, TV_HEADS, 1, 0, 0, 0};
        tv_parallel2(attn_run, &ja, &jb);
        tv_prof[TV_P_ATTN_DOT] += ja.t_qk + jb.t_qk;     // pass 1
        tv_prof[TV_P_ATTN_PV] += ja.t_pv + jb.t_pv;      // pass 2: dots again, weights, weights . values, codes
        snprintf(name, sizeof name, "%s.attn", prefix); dbg_tile(c, &s->d_attn, name, TV_I8, s->i8, t0, rows, TV_C, T);
        linear_add(c, &b->out, s->i8, rows, x + (size_t)t0 * TV_C);
    }
    // ---- pass 3: depthwise conv over LayerNorm2 (3 rows of context), expand, GELU, project
    float carry[3 * TV_C];                              // LayerNorm2 of the 3 rows before the tile, pre-update
    for (int t0 = 0; t0 < T; t0 += TV_TILE) {
        int rows = T - t0 < TV_TILE ? T - t0 : TV_TILE;
        int last = t0 + rows + 3 < T ? t0 + rows + 3 : T;            // LayerNorm rows [t0, last) are fresh
        // ln2 row r stands for position t0 - 3 + r
        if (t0 > 0) memcpy(s->ln2, carry, sizeof carry);
        tile_parallel(TV_P_NORM, TILE_LN2, last - t0, b, s, x + (size_t)t0 * TV_C, t0, T);
        if (t0 + rows < T) memcpy(carry, s->ln2 + (size_t)rows * TV_C, sizeof carry);  // positions t0+rows-3 ..
        tile_parallel(TV_P_DW, TILE_DW, rows, b, s, NULL, t0, T);
        snprintf(name, sizeof name, "%s.dw", prefix); dbg_tile(c, &s->d_dw, name, TV_I8, s->i8, t0, rows, TV_C, T);
        int8_t *g = s->g8;
        if (c->opt->debug) {        // unfused, so the pre-activation codes can be checked too
            linear_f(c, &b->expand, s->i8, rows, s->h);
            quantize(s->h, rows * TV_FF, b->s_pre, g);
            snprintf(name, sizeof name, "%s.pre", prefix); dbg_tile(c, &s->d_pre, name, TV_I8, g, t0, rows, TV_FF, T);
            TV_PROF(TV_P_GELU, for (int i = 0; i < rows * TV_FF; i++) g[i] = b->gelu_lut[g[i] + 128]);
        } else {
            linear_q8(c, &b->expand, s->i8, rows, g, b->s_pre, b->gelu_lut);
        }
        snprintf(name, sizeof name, "%s.gelu", prefix); dbg_tile(c, &s->d_gelu, name, TV_I8, g, t0, rows, TV_FF, T);
        linear_add(c, &b->project, g, rows, x + (size_t)t0 * TV_C);
    }
}

// x [T][C] is the residual input (overwritten); out receives the final LayerNorm (may alias x).
static void stack_forward(ctx *c, const tv_stack *st, const char *name, float *x, int T, float *out) {
    scratch s;
    scratch_alloc(&s, T, c->opt->debug != NULL);
    char buf[96];
    for (int i = 0; i < st->count; i++) {
        snprintf(buf, sizeof buf, "%s.blocks.%d", name, i);
        block_forward(c, &st->blocks[i], buf, x, T, &s);
        snprintf(buf, sizeof buf, "%s.%d.out", name, i);
        dbg(c, buf, TV_F32, x, T, TV_C);
    }
    scratch_free(&s);
    layer_norm(x, T, TV_C, st->norm_w, st->norm_b, out);
    snprintf(buf, sizeof buf, "%s.final", name);
    dbg(c, buf, TV_F32, out, T, TV_C);
}

static float gelu_f(float x) { return gelu(x); }

static void predict_durations(ctx *c, const tv_model *m, const float *hidden, const int32_t *ids, int N, float speed,
                              int32_t *dur) {
    float *win = xmalloc((size_t)N * 3 * TV_C * sizeof(float));
    float *h1 = xmalloc((size_t)N * TV_PRED * sizeof(float)), *h2 = xmalloc((size_t)N * TV_PRED * sizeof(float));
    int8_t *q = xmalloc((size_t)N * 3 * TV_C);
    float *out = xmalloc((size_t)N * 2 * sizeof(float));
    windows3(hidden, N, TV_C, win);
    quantize(win, N * 3 * TV_C, m->dur1.in_scale, q);
    dbg(c, "dur.in", TV_I8, q, N, 3 * TV_C);
    linear_f(c, &m->dur1, q, N, h1);
    for (int i = 0; i < N * TV_PRED; i++) h1[i] = gelu_f(h1[i]);
    layer_norm(h1, N, TV_PRED, m->dnorm1_w, m->dnorm1_b, h2);
    windows3(h2, N, TV_PRED, win);
    quantize(win, N * 3 * TV_PRED, m->dur2.in_scale, q);
    dbg(c, "dur.h1", TV_I8, q, N, 3 * TV_PRED);
    linear_f(c, &m->dur2, q, N, h1);
    for (int i = 0; i < N * TV_PRED; i++) h1[i] = gelu_f(h1[i]);
    layer_norm(h1, N, TV_PRED, m->dnorm2_w, m->dnorm2_b, h2);
    const float limit = log1pf(400.0f);
    for (int t = 0; t < N; t++) {
        for (int o = 0; o < 2; o++) {
            float v = m->dur_out_b[o];
            for (int i = 0; i < TV_PRED; i++) v += m->dur_out_w[o * TV_PRED + i] * h2[(size_t)t * TV_PRED + i];
            out[t * 2 + o] = v;
        }
        float lf = out[t * 2] < 0 ? 0 : out[t * 2] > limit ? limit : out[t * 2];
        int32_t frames = (int32_t)rintf(expm1f(lf) / speed);
        if (frames < 1) frames = 1;
        int phone = ids[t] == VOCAB_UNK || ids[t] >= FIRST_PHONE;
        dur[t] = (phone || out[t * 2 + 1] > 0) ? frames : 0;
    }
    dbg(c, "dur.out", TV_F32, out, N, 2);
    tv_free(win); tv_free(h1); tv_free(h2); tv_free(q); tv_free(out);
}

static void collect_audio(void *user, const float *x, int n) {
    float **cursor = user;
    memcpy(*cursor, x, n * sizeof(float));
    *cursor += n;
}

long tv_vocode(const tv_model *m, const float *ac, const float *pros, int T, const tv_options *opt, float *audio) {
    ctx c = {opt, 0};
    int8_t *q = xmalloc((size_t)T * TV_C);
    quantize(ac, T * TV_C, m->s_voc, q);
    dbg(&c, "voc.in", TV_I8, q, T, TV_C);
    float *harm = xmalloc((size_t)T * 98 * sizeof(float)), *nz = xmalloc((size_t)T * 36 * sizeof(float));
    float *f0 = xmalloc((size_t)T * sizeof(float));
    linear_f(&c, &m->harmonic, q, T, harm);
    linear_f(&c, &m->noise, q, T, nz);
    for (int f = 0; f < T; f++) {
        float hz = 200.0f * expf(pros[f * 3]);
        f0[f] = hz < 50.0f ? 50.0f : hz > 600.0f ? 600.0f : hz;
    }
    int S = T * TV_HOP;
    if (opt->on_audio) {
        tv_vocoder_stream(harm, nz, f0, opt->noise, opt->seed, T, opt->on_audio, opt->audio_user);
    } else {
        float *cursor = audio;
        tv_vocoder_stream(harm, nz, f0, opt->noise, opt->seed, T, collect_audio, &cursor);
        dbg(&c, "audio", TV_F32, audio, 1, S);
    }
    tv_free(q); tv_free(harm); tv_free(nz); tv_free(f0);
    return c.saturations;
}

int tv_synthesize(const tv_model *m, const int32_t *ids, const int32_t *stress, int N, int voice,
                  const tv_options *opt, tv_result *res) {
    ctx c = {opt, 0};
    memset(res, 0, sizeof *res);
    if (N < 1 || voice < 0 || voice > 1) return -1;
    // ---- text encoder
    float *x = xmalloc((size_t)N * TV_C * sizeof(float)), *hidden = xmalloc((size_t)N * TV_C * sizeof(float));
    for (int t = 0; t < N; t++)
        for (int i = 0; i < TV_C; i++)
            x[t * TV_C + i] = m->tokens[ids[t] * TV_C + i] + m->stress[stress[t] * TV_C + i] + m->voices[voice * TV_C + i];
    stack_forward(&c, &m->encoder, "encoder", x, N, hidden);
    // ---- durations
    int32_t *dur = xmalloc((size_t)N * sizeof(int32_t));
    predict_durations(&c, m, hidden, ids, N, opt->speed > 0 ? opt->speed : 1.0f, dur);
    if (opt->force_durations) memcpy(dur, opt->force_durations, N * sizeof(int32_t));
    dbg(&c, "durations", TV_I32, dur, 1, N);
    int T = 0;
    for (int t = 0; t < N; t++) T += dur[t];
    if (T < 1) { tv_free(x); tv_free(hidden); tv_free(dur); return -2; }
    if (opt->max_frames > 0 && T > opt->max_frames) {
        res->n_frames = T;
        tv_free(x); tv_free(hidden); tv_free(dur);
        return TV_TOO_LONG;
    }
    // ---- repeat token states by duration, add position-in-token and voice
    float *frames = xmalloc((size_t)T * TV_C * sizeof(float)), *ph = xmalloc((size_t)T * TV_C * sizeof(float));
    uint8_t *vowel = xmalloc(T);
    for (int t = 0, f = 0; t < N; t++) {
        int owned = dur[t] < 1 ? 1 : dur[t];
        float feat[3] = {0, 0, log1pf((float)owned) / 4.0f};
        for (int o = 0; o < dur[t]; o++, f++) {
            feat[0] = ((float)o + 0.5f) / (float)owned;
            feat[1] = 1.0f - feat[0];
            vowel[f] = ids[t] >= FIRST_VOWEL;
            for (int i = 0; i < TV_C; i++) {
                const float *w = m->frame_pos_w + i * 3;
                frames[(size_t)f * TV_C + i] = hidden[t * TV_C + i] +
                    (w[0] * feat[0] + w[1] * feat[1] + w[2] * feat[2] + m->frame_pos_b[i]) + m->frame_voices[voice * TV_C + i];
            }
        }
    }
    // ---- prosody
    stack_forward(&c, &m->prosody, "prosody_stack", frames, T, ph);
    float *pros = xmalloc((size_t)T * 3 * sizeof(float));
    float *ac_in = frames;                              // reuse
    for (int f = 0; f < T; f++) {
        const float *row = ph + (size_t)f * TV_C;
        for (int o = 0; o < 3; o++) {
            float v = m->prosody_head_b[o];
            for (int i = 0; i < TV_C; i++) v += m->prosody_head_w[o * TV_C + i] * row[i];
            pros[f * 3 + o] = v;
        }
        if (vowel[f] && pros[f * 3 + 1] < 2.0f) pros[f * 3 + 1] = 2.0f;
        float p0 = pros[f * 3], p2 = pros[f * 3 + 2];
        float cond[3] = {p0 < -2.0f ? -2.0f : p0 > 1.5f ? 1.5f : p0, pros[f * 3 + 1] > 0 ? 1.0f : 0.0f,
                         p2 < -4.5f ? -4.5f : p2 > 3.0f ? 3.0f : p2};
        for (int i = 0; i < TV_C; i++) {
            const float *w = m->prosody_proj_w + i * 3;
            ac_in[(size_t)f * TV_C + i] = row[i] + (w[0] * cond[0] + w[1] * cond[1] + w[2] * cond[2] + m->prosody_proj_b[i]);
        }
    }
    dbg(&c, "prosody", TV_F32, pros, T, 3);
    // ---- acoustic
    float *ac = ph;                                     // final LN of the acoustic stack overwrites ph
    stack_forward(&c, &m->acoustic, "acoustic_stack", ac_in, T, ac);
    dbg(&c, "acoustic", TV_F32, ac, T, TV_C);
    // ---- vocoder heads (int8) and source-filter synthesis (float)
    int S = T * TV_HOP;
    res->audio = opt->on_audio ? NULL : xmalloc((size_t)S * sizeof(float));
    c.saturations += tv_vocode(m, ac, pros, T, opt, res->audio);
    res->n_samples = S;
    res->n_frames = T;
    res->durations = dur;
    res->qacc_saturations = c.saturations;
    tv_free(x); tv_free(hidden); tv_free(frames); tv_free(ph); tv_free(vowel); tv_free(pros);
    return 0;
}
