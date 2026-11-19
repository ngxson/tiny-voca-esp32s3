# AGENTS.md

Technical notes for working on this repo. The user-facing guide is [README.md](README.md).

## What this is

A from-scratch inference engine for tiny-voca on the ESP32-S3 (240 MHz dual-core Xtensa LX7 with
PIE SIMD, 8 MB octal PSRAM, 16 MB quad flash), with int8 weights and activations (W8A8), plus the
whole text-to-speaker path on the board: sentence splitting, the dizione phonemizer, synthesis and
the ES8311 speaker codec.

Users flash prebuilt files from [Hugging Face](https://huggingface.co/ngxson/tiny-voca-esp32s3)
(`hf download ngxson/tiny-voca-esp32s3 --local-dir out`): the merged firmware per board, `tvoca.bin`,
`phonemizer.bin` and the test vector. Only exporting or evaluating the model needs the tiny-voca
source: the tools read `TINY_VOCA_DIR` (a tiny-voca checkout, default: this repo's parent folder) and
`TINY_VOCA_WEIGHTS` (default: `$TINY_VOCA_DIR/weights`, else `$TINY_VOCA_DIR/../models/tiny-voca`),
and write to this repo's `out/`.

## Layout

| Path | What |
|---|---|
| `tools/tvq.py` | **Golden model**: the exact int8 numerics the engine implements. This is the spec. |
| `tools/quant_eval.py` | Renders quantization variants and measures them against fp32 (mel L1, F0 cents, voicing, durations) |
| `tools/export.py` | Calibrates static scales, writes `out/tvoca.bin` and test vectors `out/test0.bin` / `out/test0_dev.bin` |
| `tools/pack_phonemizer.py` | Packs dizione's data files into `out/phonemizer.bin` |
| `tools/compare_dump.py` | Audio-level distance between two dumps (prosody + audio) |
| `tools/serial_dump.py` | Reads the console, decodes `TVDUMP` base64 blocks into a dump; can send a command (`TEST`) |
| `tools/console.py` | Sends console lines and prints replies up to the next `READY` |
| `tools/speak_s3.py`, `tools/dizione_cli.cpp` | Mac phonemizes, board synthesizes (`SYN` command), WAV back |
| `engine/` | Portable C engine (`tv.c` model, `tv_dsp.c` vocoder, `tv_kernels.h` backend interface) |
| `engine/tv_kernels_ref.c` | Host backend: plain C, **emulates the S3 QACC exactly** (20-bit saturation) |
| `engine/host/` | `tvhost`: runs the engine on the host and compares every tensor with the golden dump |
| `firmware/main/` | ESP-IDF app: S3 kernels (`kernels_s3.S`, `tv_kernels_s3.c`), console (`main.c`), phonemizer glue, `board.h` |
| `firmware/boards/<name>/` | One folder per board: `board.c`, `sdkconfig.defaults` (flash, PSRAM), `partitions.csv` |
| `firmware/drivers/` | Hardware code shared by boards: ES8311 codec, I2S player (with a level tap), WS2812 LEDs |
| `bench/` | Microbenchmarks of the S3 (SIMD kernels, memory bandwidth, libm) |
| `third_party/dizione` | C++17 port of the dizione phonemizer, from audio.cpp PR #1311, with small patches (`PROVENANCE.md`) |
| `monitor.sh` | Serial monitor that forwards stdin to the board |

## Numerics (the spec, `tools/tvq.py`)

- Weights int8, symmetric, one scale per output channel. Convolutions run as matmuls over
  `[x[t-1], x[t], x[t+1]]` windows.
- Activations entering a matmul: int8, symmetric, **one static scale per tensor**, calibrated as
  `max|x|` over the calibration sentences (rounded to float32 so C and Python round alike).
- Attention: Q and K (after rotary) and V int8. Weights are `round(127 * exp(score - rowmax))`, so the
  largest weight of every row is exactly 127, normalized by their own integer sum afterwards. Plain
  softmax probabilities on a 1/127 grid cost 31 cents of pitch error; row-max scaling fixed it.
  On the device the weight is a table lookup on the integer gap `maxdot - dot`.
- GELU between `expand` and `project` maps int8 to int8 (256-entry table). Its output is
  **asymmetric**: code `q` stands for `(q + 128) * s + GELU_MIN`, using all 256 codes on
  `[-0.17, max]`; the constant part is folded into `project`'s bias.
- `project` (K = 640) would overflow the 20-bit accumulator with those offset codes (690k vs 524k).
  It runs in 2 K-chunks, each starting from a **centring preload** (minus half the expected drift),
  for a worst case of about 200k. Every other layer peaks around 211k.
- Float: residual stream, LayerNorm, depthwise conv, the tiny 3-channel linears, durations, vocoder.
- Measured against fp32 on the test sentences: mel L1 1.20 dB, F0 11 cents, 97% of durations equal.
  For scale: fp32 with another noise seed differs by 1.41 dB.

## Model file (`out/tvoca.bin`, TVQ1 container)

Header `"TVQ1"`, version 1, count; 96-byte entries `{name[64], dtype, ndim, shape[4], offset, nbytes}`;
data 16-byte aligned. dtype 0 f32, 1 i8, 2 i32. Per int8 linear `<base>`: `.wq` i8 `[Npad/16][K][16]`
(16 output channels interleaved, the QACC kernel layout), `.ws` f32 per-channel scale, `.b` f32 bias,
`.pre` i32 `[chunks][N]` (project only). Activation scales are `<point>.max` f32 scalars. The same
container format is used for the test vectors, the phonemizer data and dumps.

## Engine

`tv_synthesize`: embeddings, encoder stack (4 blocks over tokens), duration predictor, repeat by
duration, prosody stack (2 blocks), prosody head and voicing rules, acoustic stack (5 blocks),
vocoder heads (int8), source-filter vocoder (float).

- **Tiled stacks.** Each block runs over row tiles of `TV_TILE` = 64 positions in three passes:
  (1) LayerNorm1, QKV, rotary, int8 Q / K per head / V transposed per key chunk;
  (2) attention for the tile, output projection added into the residual;
  (3) LayerNorm2 with a 3-row carry (pre-update values for the depthwise conv), expand, GELU, project
  added into the residual. Only the residual and the int8 Q/K/V span the whole utterance.
- **Attention** is two passes over key chunks of 128 with no score matrix: pass 1 finds each query's
  max dot product, pass 2 recomputes the dots, maps them to weights and accumulates weights times
  values. Integer sums, so it equals the one-pass result exactly. Heads are split over the two cores.
- **Streaming vocoder.** Harmonic (FFT 512, hop 128) and noise (FFT 256, hop 64) branches are
  streaming STFT filters: frame responses computed when needed, samples final after their last frame,
  arithmetic order equal to `torch.stft`/`istft`. The two branches run on the two cores in blocks of
  8192 samples. Output goes to `tv_options.on_audio` (the console streams it to the speaker).
- **Epilogues** are fused into the matmul (`tv_linear_run`): store float, add into the residual, or
  quantize to int8 through a table (expand writes final GELU codes). LayerNorm1 and the depthwise
  conv write int8 directly. When a debug hook is set, expand runs unfused so the pre-activation codes
  can be checked; both paths must give bit-identical audio (`tvhost --twice`, firmware `TEST`).
- **Memory:** about 2.7 KB per frame at the peak plus ~0.7 MB of fixed buffers. `tv_options.max_frames`
  makes `tv_synthesize` return `TV_TOO_LONG` before allocating; the console computes the budget from
  free PSRAM (~1,400 frames, 15 s) and falls back to splitting at commas.

## S3 kernels (`firmware/main/kernels_s3.S`)

- `qacc_mm_s8`: 16 outputs per call with `ee.vsmulas.s8.qacc.ld.incp` (one input value broadcast
  against 16 weights). Weights are loaded 3 instructions ahead (q1..q4 rotate), inputs one chunk ahead.
  The accumulator starts from a 64-byte raw state (zero, bias or centring preload) and is stored raw;
  `unpack_qacc` reads the 2 x 8 lanes of 20 bits. 12.6 MAC/cycle kernel-only in the benchmark.
- `dot_rows_s8_accx` / `dot48_rows_accx`: exact dot products in the 40-bit ACCX, for attention
  (Q.K over 48 dims and weights.V could overflow 20 bits).
- QACC lanes **saturate** at +-2^19 (no wrap). `ee.srcmb.s8.qacc` shifts and writes back into QACC;
  `ee.srcmb.s16.qacc` is garbage in 8-bit mode. The host backend emulates saturation and counts it;
  the board flags a lane sitting exactly on the rail. Any saturation is a bug.
- Pointers must be 16-byte aligned; kernels read up to 48 bytes past their inputs (`tv_alloc` adds slack).
- Windowed ABI: never write the top 16 bytes of your own frame from assembly (it is the caller's
  register spill area); this once corrupted a return address.
- `tv_parallel2(fn, a, b)` runs two jobs on the two cores (the worker task on core 1 has an 8 KB stack).

## Phonemizer

`third_party/dizione` is used unchanged except for the patches listed in its `PROVENANCE.md`: the
3.3 MB lexicon is read from mapped flash (`Lexicon::external`), `parse_npz` reads the model from
memory, and the unknown-word GRU is faster (weight rows read once per word, the two directions on
both cores, dot lanes in registers, memoized words), with identical results. Dictionary words cost
about 0.2 ms each, unknown words about 0.3 s. Setup takes 3.5 s at boot.

## Boards

The app talks to the hardware only through `firmware/main/board.h`: `board_init`, `board_play`
(24 kHz mono, takes ownership), `board_set_volume`, `board_beep`, and `board_show_text` (the text being
spoken; boards without a screen ignore it). The board is chosen at build time:

```bash
idf.py -B build/<board> -DBOARD=<board> build flash     # configuration kept in sdkconfig.<board>
```

`firmware/sdkconfig.defaults` holds the settings every board shares; `boards/<board>/sdkconfig.defaults`
adds flash size and mode, PSRAM mode and the partition table path.

To add a board: copy `boards/waveshare_s3_audio`, change the pins and the amplifier/power glue in
`board.c`, set flash and PSRAM in its `sdkconfig.defaults`, and size `partitions.csv` to the flash.
Shared hardware code goes to `drivers/`. Current needs: about 11.4 MB of flash (app 1.2 MB, model
4.85 MB, phonemizer data 5.4 MB, test vector 0.3 MB) and 8 MB of PSRAM (phonemizer ~2.9 MB,
synthesis ~0.7 MB + 2.7 KB per frame; octal PSRAM, quad is slower).

### Waveshare ESP32-S3-AUDIO-Board (`waveshare_s3_audio`)

I2C SDA 11, SCL 10; I2S MCLK 12, BCLK 13, WS 14, DOUT 16 (DIN 15 for the mics, unused). ES8311 at
0x18 (chip id 0x83 0x11 at 0xFD/0xFE), set up as an I2S slave with MCLK = 256 x 24 kHz, 16-bit
(register values from Espressif's es8311 driver). The NS4150B amplifier is enabled by pin 8 of the
TCA9555 expander (0x20): only port 1 bit 0 is ever written, with read-modify-write. **Never drive
expander port 0**: pins 6/7 are the USB/camera mux and the wrong level kills USB.
The ring of 7 WS2812B LEDs on GPIO 38 is a sound level meter (`meter()` in `board.c`): the I2S
player reports the RMS of each 10 ms piece as it is heard (levels delayed by the 4-piece DMA depth),
mapped from -45 to -9 dB onto the LEDs, green, yellow, red, instant rise and a ~0.35 s fall.

Flash: app 2 MB at 0x10000, `model` 6 MB at 0x210000, `testvec` 1 MB at 0x810000, `phonemizer` 6 MB
at 0x910000 (`boards/waveshare_s3_audio/partitions.csv`). The model is memory-mapped from flash;
PSRAM holds the phonemizer (~2.9 MB) and activations.

### Waveshare ESP32-S3-ePaper-1.54 (not supported yet)

Checked 2026-11-17 against [the Waveshare docs](https://docs.waveshare.com/ESP32-S3-ePaper-1.54) and
[demo code](https://github.com/waveshareteam/ESP32-S3-ePaper-1.54). Same ES8311 codec, a 200 x 200
e-paper screen for `board_show_text`. Two versions, and their example code is not interchangeable:
V1 is an ESP32-S3FH4R2 (4 MB flash, 2 MB quad PSRAM): the int8 model alone (4.85 MB) is larger
than the flash, so it would need int4 weights (~2.4 MB) and the Mac doing the phonemizing.
V2 ("V2" printed top-left on the PCB, since Nov 2025) is an ESP32-S3-PICO-1-N8R8 (8 MB flash, 8 MB
PSRAM): it needs the phonemizer data compacted (the 3.3 MB lexicon text, float G2P weights) to fit.

## Exporting the model

```bash
git clone https://github.com/n4ze3m/tiny-voca ../tiny-voca && pip install -e "../tiny-voca"
hf download n4ze3m/tiny-voca --local-dir ../tiny-voca/weights
export TINY_VOCA_DIR=../tiny-voca
python tools/export.py              # calibrates scales; out/tvoca.bin, out/test0.bin, out/test0_dev.bin
python tools/pack_phonemizer.py ../tiny-voca/weights/phonemizer out/phonemizer.bin
```

Calibration is deterministic, so the same checkpoint gives the same `tvoca.bin`.

## Releasing prebuilt files

```bash
cd firmware && idf.py -B build/release -DBOARD=<board> build
cd build/release && esptool --chip esp32s3 merge-bin -o ../../../out/hf/<board>/firmware.bin @flash_args
```

`out/hf/` also holds `tvoca.bin`, `phonemizer.bin`, `test0_dev.bin`, `licenses/` (tiny-voca,
CMUdict and homograph data, whose contents are inside `phonemizer.bin`) and the model card
`README.md`. Flash the exact files once before uploading (`hf upload ngxson/tiny-voca-esp32s3 out/hf .`),
then run `TEST` and compare with the host.

## Validation workflow

Order of trust: fp32 PyTorch, golden model (`tvq.py`), host engine, board.

```bash
python tools/export.py                                  # also writes the golden dump test0.bin
make -C engine/host
engine/host/tvhost out/tvoca.bin out/test0.bin out/c.wav --quiet --twice --dump out/c.bin
python tools/serial_dump.py /dev/ttyACM0 180 out/dev.bin TEST      # macOS: /dev/cu.usbmodem*
python tools/compare_dump.py out/c.bin out/dev.bin       # board vs host: must be 0.00 dB, 0 cents
python tools/compare_dump.py out/test0.bin out/dev.bin   # vs golden: ~0.9-1.0 dB, ~2 cents
```

- Host refactors must keep the host output **bit-identical** (`cmp` of the dumps or `np.array_equal`).
- Host vs golden is not bit-exact: tiny float differences flip int8 codes near rounding boundaries
  and spread through the hard thresholds (rounding, voicing, durations). The golden model nudged by
  1e-6 diverges from itself just as much (~0.94 dB, ~3.7 cents); that is the noise floor.
- Waveform SNR/correlation is meaningless across versions: a few cents of pitch drift the pulse phase.

## Performance (board)

A 3.3 s test sentence: 4.4 s (RTF ~0.75). A 9.25 s sentence: first sound after 10.9 s, then it plays
without gaps (the vocoder alone is faster than real time). Main costs: int8 matmuls (~700 MMAC/s:
weights in flash cost about 2x, see the `mmbench` console command), attention pass 2, vocoder.
The obvious next step is moving the weights to PSRAM, at the cost of the sentence-length budget.

## Working rules

- Every build, flash and serial read runs under a hard timeout (macOS has no `timeout`:
  `perl -e 'alarm shift; exec @ARGV' SECS cmd`); serial readers stop on panic and reboot too.
- ESP-IDF's Python has no numpy: run `compare_dump.py` and the export tools with the project's Python.
- If flashing says the port is busy, check `lsof /dev/ttyACM* /dev/cu.usbmodem*` (a monitor may hold it).
- The firmware build uses `-Werror`; `tv.h` and `tv_kernels.h` have `extern "C"` guards for the C++ glue.
- Commit messages: imperative subject, body explaining what and why; no co-author lines.
