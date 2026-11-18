// Host runner: synthesize the test vector with the C engine and compare every
// intermediate tensor with the golden model's dump.
//   tvhost tvoca.bin test0.bin out.wav [--force-durations] [--quiet]
#include "../tv.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = malloc(*size);
    if (fread(p, 1, *size, f) != *size) { perror(path); exit(1); }
    fclose(f);
    return p;
}

typedef struct {
    tv_file test;
    int quiet, compared, int8_tensors, int8_codes, int8_off, float_tensors;
    double worst_float_rel;
    float *prosody, *audio;     // kept for --dump
    int prosody_rows, audio_n;
} checker;

static void check(void *user, const char *name, int dtype, const void *data, int rows, int cols) {
    checker *c = user;
    const tv_entry *e = tv_file_find(&c->test, name);
    size_t n = (size_t)rows * cols;
    if (!e) return;
    static const int size[] = {4, 1, 4};
    if (e->dtype != (uint32_t)dtype || e->nbytes != n * size[dtype]) {
        printf("  %-40s SHAPE/TYPE MISMATCH (engine %zu x dtype %d, golden %u bytes dtype %u)\n", name, n, dtype, e->nbytes, e->dtype);
        return;
    }
    const void *ref = c->test.base + e->offset;
    c->compared++;
    if (!strcmp(name, "prosody")) {
        const float *a = data, *b = ref;
        int flips = 0, worst = -1;
        for (int i = 0; i < rows; i++)
            if ((a[i * 3 + 1] > 0) != (b[i * 3 + 1] > 0)) { flips++; worst = i; }
        printf("  %-40s voicing flips: %d of %d frames%s\n", "prosody[:,1]", flips, rows, flips ? " (e.g. frame shown below)" : "");
        if (worst >= 0) printf("      frame %d: engine p1 %.5f, golden p1 %.5f\n", worst, a[worst * 3 + 1], b[worst * 3 + 1]);
        free(c->prosody); c->prosody = malloc(n * 4); memcpy(c->prosody, data, n * 4); c->prosody_rows = rows;
    }
    if (!strcmp(name, "audio")) { free(c->audio); c->audio = malloc(n * 4); memcpy(c->audio, data, n * 4); c->audio_n = n; }
    if (dtype == TV_I8) {
        const int8_t *a = data, *b = ref;
        size_t off = 0; int worst = 0;
        for (size_t i = 0; i < n; i++) {
            int d = abs(a[i] - b[i]);
            if (d) { off++; if (d > worst) worst = d; }
        }
        c->int8_tensors++; c->int8_codes += n; c->int8_off += off;
        if (!c->quiet || worst > 1) printf("  %-40s int8  %8zu codes, %6zu off (%.3f%%), max diff %d\n", name, n, off, 100.0 * off / n, worst);
    } else if (dtype == TV_I32) {
        const int32_t *a = data, *b = ref;
        size_t off = 0;
        for (size_t i = 0; i < n; i++) off += a[i] != b[i];
        printf("  %-40s int32 %8zu values, %zu differ\n", name, n, off);
    } else {
        const float *a = data, *b = ref;
        double maxd = 0, maxr = 0, dot = 0, aa = 0, bb = 0;
        for (size_t i = 0; i < n; i++) {
            double d = fabs((double)a[i] - b[i]);
            if (d > maxd) maxd = d;
            if (fabs(b[i]) > maxr) maxr = fabs(b[i]);
            dot += (double)a[i] * b[i]; aa += (double)a[i] * a[i]; bb += (double)b[i] * b[i];
        }
        double rel = maxd / (maxr > 0 ? maxr : 1), corr = dot / sqrt(aa * bb + 1e-30);
        c->float_tensors++;
        if (rel > c->worst_float_rel) c->worst_float_rel = rel;
        if (!c->quiet || rel > 1e-2) printf("  %-40s f32   %8zu values, max|diff| %.3g (%.2e of max), corr %.7f\n", name, n, maxd, rel, corr);
    }
}

// Minimal TVQ1 writer (f32 tensors only) for --dump.
static void write_container(const char *path, int count, const char **names, float **data, const int *rows, const int *cols) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return; }
    uint32_t head[4] = {0, 1, (uint32_t)count, 0};
    memcpy(head, "TVQ1", 4);
    fwrite(head, 4, 4, f);
    uint32_t offset = (16 + 96 * count + 15) & ~15u;
    for (int i = 0; i < count; i++) {
        tv_entry e;
        memset(&e, 0, sizeof e);
        strncpy(e.name, names[i], 63);
        e.dtype = TV_F32; e.ndim = 2; e.shape[0] = rows[i]; e.shape[1] = cols[i]; e.shape[2] = e.shape[3] = 1;
        e.offset = offset; e.nbytes = rows[i] * cols[i] * 4;
        fwrite(&e, sizeof e, 1, f);
        offset = (offset + e.nbytes + 15) & ~15u;
    }
    for (int i = 0; i < count; i++) {
        while (ftell(f) % 16) fputc(0, f);
        fwrite(data[i], 4, rows[i] * cols[i], f);
    }
    fclose(f);
}

static void write_wav(const char *path, const float *x, int n) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return; }
    float peak = 0;
    for (int i = 0; i < n; i++) if (fabsf(x[i]) > peak) peak = fabsf(x[i]);
    float gain = peak > 0.99f ? 0.99f / peak : 1.0f;
    uint32_t data = n * 2, riff = 36 + data, rate = TV_SAMPLE_RATE, byte_rate = rate * 2, fmt = 16;
    uint16_t pcm = 1, ch = 1, align = 2, bits = 16;
    fwrite("RIFF", 1, 4, f); fwrite(&riff, 4, 1, f); fwrite("WAVEfmt ", 1, 8, f); fwrite(&fmt, 4, 1, f);
    fwrite(&pcm, 2, 1, f); fwrite(&ch, 2, 1, f); fwrite(&rate, 4, 1, f); fwrite(&byte_rate, 4, 1, f);
    fwrite(&align, 2, 1, f); fwrite(&bits, 2, 1, f); fwrite("data", 1, 4, f); fwrite(&data, 4, 1, f);
    for (int i = 0; i < n; i++) {
        float v = x[i] * gain * 32767.0f;
        int16_t s = (int16_t)lrintf(v > 32767 ? 32767 : v < -32767 ? -32767 : v);
        fwrite(&s, 2, 1, f);
    }
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s tvoca.bin test.bin out.wav [--force-durations] [--quiet] [--lookahead N --past N]\n", argv[0]);
        return 2;
    }
    int force = 0, quiet = 0, lookahead = 0, past = 0;
    const char *dump = NULL;
    int vocoder_only = 0, twice = 0;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--force-durations")) force = 1;
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
        else if (!strcmp(argv[i], "--vocoder-only")) vocoder_only = 1;
        else if (!strcmp(argv[i], "--twice")) twice = 1;
        else if (!strcmp(argv[i], "--lookahead") && i + 1 < argc) lookahead = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--past") && i + 1 < argc) past = atoi(argv[++i]);
    }
    size_t msize, tsize;
    void *mdata = read_file(argv[1], &msize), *tdata = read_file(argv[2], &tsize);
    static tv_model model;
    if (tv_model_init(&model, mdata, msize)) { fprintf(stderr, "model: %s\n", model.error); return 1; }
    checker c = {.quiet = quiet};
    if (tv_file_open(&c.test, tdata, tsize)) { fprintf(stderr, "bad test vector\n"); return 1; }
    const tv_entry *ids_e = tv_file_find(&c.test, "ids");
    int n = ids_e->shape[0];
    const int32_t *ids = tv_file_get(&c.test, "ids", TV_I32, n);
    const int32_t *stress = tv_file_get(&c.test, "stress", TV_I32, n);
    const int32_t *voice = tv_file_get(&c.test, "voice", TV_I32, 1);
    const int32_t *golden_dur = tv_file_get(&c.test, "durations", TV_I32, n);
    const tv_entry *noise_e = tv_file_find(&c.test, "noise");
    tv_options opt = {.speed = 1.0f, .noise = (const float *)(c.test.base + noise_e->offset),
                      .force_durations = force ? golden_dur : NULL, .debug = check, .debug_user = &c,
                      .acoustic_lookahead = lookahead, .acoustic_past = past};
    if (vocoder_only) {     // golden acoustic + prosody in, compare the waveform sample by sample
        const tv_entry *ae = tv_file_find(&c.test, "acoustic");
        int T = ae->shape[0];
        const float *ac = tv_file_get(&c.test, "acoustic", TV_F32, T * TV_C);
        const float *pr = tv_file_get(&c.test, "prosody", TV_F32, T * 3);
        float *audio = malloc((size_t)T * TV_HOP * sizeof(float));
        long sat = tv_vocode(&model, ac, pr, T, &opt, audio);
        printf("vocoder only: %d frames, QACC saturations %ld\n", T, sat);
        write_wav(argv[3], audio, T * TV_HOP);
        free(audio);
        return 0;
    }
    tv_result r;
    tv_profile_reset();
    clock_t t0 = clock();
    int rc = tv_synthesize(&model, ids, stress, n, *voice, &opt, &r);
    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
    if (rc) { fprintf(stderr, "synthesize failed: %d\n", rc); return 1; }
    if (noise_e->nbytes != (uint32_t)r.n_samples * 4) printf("  (noise length %u != %d samples: frame count differs)\n", noise_e->nbytes / 4, r.n_samples);
    write_wav(argv[3], r.audio, r.n_samples);
    printf("\n%d tokens -> %d frames, %.2f s of audio in %.2f s (host, reference C)\n", n, r.n_frames,
           (double)r.n_samples / TV_SAMPLE_RATE, secs);
    printf("compared %d tensors: int8 %d tensors, %.4f%% of codes off; worst float max|diff|/max %.2e\n", c.compared,
           c.int8_tensors, c.int8_codes ? 100.0 * c.int8_off / c.int8_codes : 0.0, c.worst_float_rel);
    printf("QACC saturations: %ld\n", r.qacc_saturations);
    if (twice) {    // the production path (no debug hook: fused epilogues) must give the same audio
        tv_options o2 = opt;
        o2.debug = NULL;
        tv_result r2;
        if (tv_synthesize(&model, ids, stress, n, *voice, &o2, &r2) == 0) {
            int same = r2.n_samples == r.n_samples && !memcmp(r2.audio, r.audio, (size_t)r.n_samples * 4);
            printf("production path (no debug hook): audio %s\n", same ? "bit-identical" : "DIFFERS");
            free(r2.audio); free(r2.durations);
        }
    }
    tv_profile_report();
    if (dump && c.prosody && c.audio) {
        const char *names[] = {"prosody", "audio"};
        float *data[] = {c.prosody, c.audio};
        int rows[] = {c.prosody_rows, 1}, cols[] = {3, c.audio_n};
        write_container(dump, 2, names, data, rows, cols);
    }
    free(c.prosody); free(c.audio);
    free(r.audio); free(r.durations);
    tv_model_free(&model);
    free(mdata); free(tdata);
    return 0;
}
