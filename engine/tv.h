// tiny-voca inference engine: portable C, int8 matmuls (W8A8) with the numerics
// specified by esp32/tools/tvq.py. The same code runs on the host (reference and
// validation) and on the ESP32-S3 (where the matmul is replaced by PIE kernels).
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TV_C 192            // channels
#define TV_HEADS 4
#define TV_HEAD_DIM 48
#define TV_FF 640
#define TV_KERNEL 7
#define TV_PRED 128         // duration predictor channels
#define TV_HOP 256
#define TV_SAMPLE_RATE 24000
#define TV_QACC_MAX 524287  // QACC lanes are 20-bit signed and saturate

// ---------------------------------------------------------------- container (export.py)
typedef struct {
    char name[64];
    uint32_t dtype, ndim, shape[4], offset, nbytes;
} tv_entry;

typedef struct {
    const uint8_t *base;
    const tv_entry *entries;
    uint32_t count;
} tv_file;

enum { TV_F32 = 0, TV_I8 = 1, TV_I32 = 2 };

int tv_file_open(tv_file *f, const void *data, size_t size);       // 0 on success
const tv_entry *tv_file_find(const tv_file *f, const char *name);
const void *tv_file_get(const tv_file *f, const char *name, uint32_t dtype, uint32_t nelem);  // NULL if missing/mismatch

// ---------------------------------------------------------------- model
typedef struct {
    const int8_t *wq;       // [Npad/16][K][16]
    const float *ws, *b;    // [N]
    const int32_t *pre;     // [chunks][N] centring preload, or NULL
    float *bias;            // [N] bias used by the epilogue (folded constants included)
    int N, K, chunks;
    float in_scale;         // scale of the int8 input
    void *backend;          // kernel-backend data (e.g. packed QACC preloads)
} tv_linear;

typedef struct {
    const float *norm1_w, *norm1_b, *norm2_w, *norm2_b, *dw_w, *dw_b;
    tv_linear qkv, out, expand, project;
    float s_ln1, s_q, s_k, s_v, s_attn, s_dw, s_pre, s_gelu;  // int8 scales (gelu: asymmetric step)
    int8_t gelu_lut[256];   // pre code (-127..127) + 128 -> gelu code (-128..127)
    uint8_t *softmax_lut;   // [gap] = round(127 * exp(-gap * score_scale)), gap = max dot - dot
    int softmax_n;          // table length; larger gaps give weight 0
} tv_block;

typedef struct {
    tv_block *blocks;
    int count;
    const float *norm_w, *norm_b;
} tv_stack;

typedef struct {
    tv_file file;
    const float *tokens, *stress, *voices, *frame_voices;   // embeddings
    tv_stack encoder, prosody, acoustic;
    tv_block block_storage[11];
    // duration predictor
    tv_linear dur1, dur2;
    const float *dnorm1_w, *dnorm1_b, *dnorm2_w, *dnorm2_b, *dur_out_w, *dur_out_b;
    // frame-level
    const float *frame_pos_w, *frame_pos_b, *prosody_head_w, *prosody_head_b, *prosody_proj_w, *prosody_proj_b;
    // vocoder heads
    tv_linear harmonic, noise;
    float s_voc;
    // Read from the file so one engine runs any language: token and voice counts come from the
    // embedding shapes, the first vowel id from the optional `meta.first_vowel` (English: 43).
    int n_vocab, n_voices, first_vowel;
    char error[128];
} tv_model;

int tv_model_init(tv_model *m, const void *data, size_t size);   // 0 on success, else see m->error
void tv_model_free(tv_model *m);

// ---------------------------------------------------------------- inference
// Audio sink for streaming output: called with consecutive chunks of samples.
typedef void (*tv_audio_fn)(void *user, const float *samples, int n);

// Debug hook: called with intermediate tensors (names as in the golden dump).
typedef void (*tv_debug_fn)(void *user, const char *name, int dtype, const void *data, int rows, int cols);

typedef struct {
    float speed;                    // 1.0
    const float *noise;             // NULL: engine's own Gaussian noise; else n_frames*TV_HOP values
    uint32_t seed;
    const int32_t *force_durations; // NULL: predicted
    int max_frames;                 // > 0: return TV_TOO_LONG instead of synthesizing more frames
    tv_audio_fn on_audio;           // set: audio is streamed here and tv_result.audio stays NULL
    void *audio_user;
    tv_debug_fn debug;
    void *debug_user;
} tv_options;

typedef struct {
    float *audio;                   // n_samples (caller frees)
    int n_samples, n_frames;
    int32_t *durations;             // n_tokens (caller frees)
    long qacc_saturations;          // must be 0
} tv_result;

#define TV_TOO_LONG (-3)            // predicted frames exceed tv_options.max_frames; split the text
int tv_synthesize(const tv_model *m, const int32_t *ids, const int32_t *stress, int n_tokens, int voice,
                  const tv_options *opt, tv_result *out);

// Last stage alone: acoustic features [T][192] + prosody [T][3] -> audio [T*hop] (or opt->on_audio).
// Returns saturations.
long tv_vocode(const tv_model *m, const float *acoustic, const float *prosody, int T, const tv_options *opt, float *audio);

// ---------------------------------------------------------------- profiling (microseconds per op category)
enum { TV_P_MATMUL, TV_P_EPILOGUE, TV_P_ATTN_DOT, TV_P_SOFTMAX, TV_P_ATTN_PV, TV_P_NORM, TV_P_DW, TV_P_QUANT, TV_P_GELU, TV_P_ROPE,
       TV_P_VOC_CEP, TV_P_VOC_SHAPE, TV_P_VOC_PULSE, TV_P_COUNT };
extern uint64_t tv_prof[TV_P_COUNT];
uint64_t tv_time_us(void);          // provided by the kernel backend
void tv_profile_reset(void);
void tv_profile_report(void);       // printf, one line per category
#define TV_PROF(id, stmt) do { uint64_t tv_t0_ = tv_time_us(); stmt; tv_prof[id] += tv_time_us() - tv_t0_; } while (0)

// ---------------------------------------------------------------- internals shared with the DSP file
void tv_vocoder_stream(const float *harmonic, const float *noise_params, const float *f0, const float *noise,
                       uint64_t seed, int T, tv_audio_fn sink, void *user);
void tv_vocoder(const float *harmonic /*[T][98]*/, const float *noise_params /*[T][36]*/, const float *f0 /*[T]*/,
                const float *noise /*[T*hop]*/, int T, float *audio /*[T*hop]*/);

#ifdef __cplusplus
}
#endif
