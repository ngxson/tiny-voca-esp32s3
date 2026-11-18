"""Listen to and measure tiny-voca under the quantization schemes the ESP32-S3 engine could use.

Fake quantization (quantize, then dequantize, in float) of:
  * every nn.Linear and non-depthwise Conv1d: weights per output channel (or per group of input
    weights), input activations int8 with one dynamic scale per tensor;
  * attention: Q and K (after rotary) and V int8 per tensor, softmax probabilities uint8.
LayerNorm, softmax, GELU, the depthwise convolution and the source-filter vocoder stay float.

Phonemes come from a plain lookup in the phonemizer lexicon (dizione is not installed here),
so every sentence must use dictionary words. All variants see identical tokens.

Writes one WAV per variant with all sentences in both voices, and prints distances to fp32
measured with the fp32 durations forced, so the waveforms line up frame by frame.

    python tools/quant_eval.py --output out/quant
"""
import argparse, copy, math, os, sys, wave
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]                     # this repo; outputs go to REPO / "out"
# The tiny-voca source (for the model code) and its weights; only needed to export or evaluate.
TINY_VOCA = Path(os.environ.get("TINY_VOCA_DIR", REPO.parent)).resolve()
WEIGHTS = Path(os.environ.get("TINY_VOCA_WEIGHTS",
                              TINY_VOCA / "weights" if (TINY_VOCA / "weights").exists() else TINY_VOCA.parent / "models/tiny-voca"))
sys.path.insert(0, str(TINY_VOCA / "src"))
from tiny_voca.infer import load_model, limit_peak                      # noqa: E402
from tiny_voca.model import Attention, rotate                            # noqa: E402
from tiny_voca.text import chunks, normalize_text, tokenize, parse_espeak  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tvq import Golden                                                    # noqa: E402

SENTENCES = [
    "Hello! This is a small voice that can read to you.",
    "Do you want to go to the market with me, or would you rather stay at home and read a good book?",
    "When the sun went down over the quiet water, the children ran back along the beach, laughing and shouting.",
]

# Calibration set for static activation scales: disjoint from SENTENCES.
CALIBRATION = [
    "The weather today is cold, with rain in the morning and wind in the afternoon.",
    "Seven hundred people came to the old town hall last night, and nobody could find a place to sit.",
    "Yes.",
    "Please call me back when you have a minute; I need to ask you something important.",
    "Why did the little dog run away from the house?",
    "She opened the window, looked at the garden, and smiled.",
    "We will meet at the station at nine, then take the train to the city.",
    "Thank you very much for your help!",
    "It was a dark and quiet night, and the only sound was the river.",
    "How many apples are left in the basket on the kitchen table?",
    "My brother works at a hospital near the park.",
    "Stop! Do not touch that, it is very hot.",
]

# name: (weight bits, weight group size or None for per channel, quantize activations/attention)
# "fp32" is the reference; "fp32-self" must score 0 (checks the forced-duration path);
# "fp32-seed1" only changes the noise seed: the spread between two equally good renders.
VARIANTS = {
    "fp32": None,
    "fp32-self": None,
    "fp32-seed1": None,
    "w8a8": (8, None, True),
    "w4a8": (4, None, True),
    "w4a8-g32": (4, 32, True),
    "golden-fp": "golden-fp",       # golden model, float: must score 0 (checks the re-implementation)
    "golden-w8a8": "golden-w8a8",   # golden model, int8 weights + static activation scales (the engine spec)
}


# ---------------------------------------------------------------- phonemes
def load_lexicon(path):
    lexicon = {}
    for line in open(path, encoding="utf-8"):
        word, phones = line.rstrip("\n").split("\t")
        lexicon.setdefault(word, phones.replace(" ", "_"))
    return lexicon


def phonemize(text, lexicon):
    pieces = []
    for punctuation, piece in chunks(normalize_text(text)):
        if punctuation:
            pieces.append((True, piece))
            continue
        words = []
        for word in piece.lower().split():
            if word not in lexicon:
                raise KeyError(f"{word!r} is not in the lexicon; pick another word")
            words += parse_espeak(lexicon[word])
        pieces.append((False, words))
    return tokenize(pieces)


# ---------------------------------------------------------------- fake quantization
def quant_symmetric(x, bits, dim=None):
    """Round to signed `bits` integers with a symmetric scale (per tensor, or reduced over `dim`)."""
    qmax = 2 ** (bits - 1) - 1
    peak = x.abs().amax() if dim is None else x.abs().amax(dim=dim, keepdim=True)
    scale = (peak / qmax).clamp_min(1e-12)
    return torch.round(x / scale).clamp(-qmax, qmax) * scale


def quant_weight(w, bits, group):
    """Per output channel (dim 0) or per `group` consecutive input weights."""
    flat = w.reshape(w.shape[0], -1)
    if group is None:
        return quant_symmetric(flat, bits, dim=1).reshape(w.shape)
    grouped = flat.reshape(flat.shape[0], -1, group)
    return quant_symmetric(grouped, bits, dim=2).reshape(w.shape)


def quant_attention_forward(self, x, mask, table):
    b, t, c = x.shape
    q, k, v = self.qkv(x).view(b, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
    q, k = quant_symmetric(rotate(q, table), 8), quant_symmetric(rotate(k, table), 8)
    v = quant_symmetric(v, 8)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
    probs = torch.round(scores.softmax(-1) * 255) / 255          # uint8 probabilities
    y = probs @ v
    return self.out(y.transpose(1, 2).reshape(b, t, c))


def quantized_model(model, bits, group, activations):
    model = copy.deepcopy(model)
    for module in model.modules():
        depthwise = isinstance(module, torch.nn.Conv1d) and module.groups > 1
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)) and not depthwise:
            fan_in = module.weight[0].numel()
            g = group if group and fan_in % group == 0 else None
            module.weight.data = quant_weight(module.weight.data, bits, g)
            if activations:
                module.register_forward_pre_hook(lambda m, args: (quant_symmetric(args[0], 8),) + args[1:])
        if activations and isinstance(module, Attention):
            module.forward = quant_attention_forward.__get__(module)
    return model


# ---------------------------------------------------------------- metrics
def log_mel(audio, sample_rate=24000):
    spec = torch.stft(audio, 1024, 256, window=torch.hann_window(1024), return_complex=True).abs() ** 2
    freqs = torch.linspace(0, sample_rate / 2, spec.shape[0])
    mel = 2595 * torch.log10(1 + freqs / 700)
    edges = torch.linspace(0, float(mel[-1]), 82)
    bank = torch.clamp(torch.minimum((mel[None] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None]),
                                     (edges[2:, None] - mel[None]) / (edges[2:, None] - edges[1:-1, None])), min=0)
    return 10 * torch.log10(bank @ spec + 1e-8)


def compare(ref, test):
    """ref/test: dicts with audio, prosody (frames x 3) from the same durations."""
    a, b = ref["audio"], test["audio"]
    ma, mb = log_mel(a), log_mel(b)
    loud = ma.max(0).values > ma.max() - 60                 # skip silent frames
    mel_db = float((ma - mb).abs()[:, loud].mean())
    pa, pb = ref["prosody"], test["prosody"]
    voiced = (pa[:, 1] > 0) & (pb[:, 1] > 0)
    cents = float(((pa[voiced, 0] - pb[voiced, 0]) * 1200 / math.log(2)).pow(2).mean().sqrt())
    flips = float(((pa[:, 1] > 0) != (pb[:, 1] > 0)).float().mean())
    return mel_db, cents, flips


# ---------------------------------------------------------------- main
def run(model, tokens, voice, durations=None, seed=0):
    p = tokens
    if isinstance(model, Golden):
        return model.synthesize(p.ids, p.stress, voice, seed=seed, durations=durations)
    args = (torch.tensor([p.ids]), torch.tensor([p.stress]), torch.tensor([len(p.ids)]), torch.tensor([voice]))
    if durations is None:
        audio, n, info = model.synthesize(*args, seed=seed)
    else:
        original = model.predict_durations
        model.predict_durations = lambda *a, **k: durations
        try:
            audio, n, info = model.synthesize(*args, seed=seed)
        finally:
            model.predict_durations = original
    return {"audio": audio[0, :int(n[0])], "durations": info["durations"], "prosody": info["prosody"][0]}


def write_wav(path, audio, sample_rate=24000):
    pcm = (limit_peak(audio).clamp(-1, 1) * 32767).round().to(torch.int16).numpy().tobytes()
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sample_rate); f.writeframes(pcm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(WEIGHTS / "tiny-voca.pt"))
    parser.add_argument("--lexicon", default=str(WEIGHTS / "phonemizer/lexicon.tsv"))
    parser.add_argument("--output", default=str(REPO / "out/quant"))
    parser.add_argument("--variants", help="comma-separated subset to run (fp32 always runs)")
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    lexicon = load_lexicon(args.lexicon)
    tokens = [phonemize(s, lexicon) for s in SENTENCES]
    calibration = [phonemize(s, lexicon) for s in CALIBRATION]
    base = load_model(args.checkpoint, check_frontend=False)
    gap = torch.zeros(int(0.6 * 24000))

    reference = {}       # (sentence, voice) -> fp32 result
    print(f"{'variant':10s} {'mel L1 dB':>9s} {'F0 cents':>8s} {'voicing':>8s} {'dur. same':>9s}")
    only = set(args.variants.split(",")) if args.variants else None
    for name, spec in VARIANTS.items():
        if only and name not in only and name != "fp32":
            continue
        if spec == "golden-fp":
            model = Golden(base, quantize_weights=False)     # scales None: float everywhere
        elif spec == "golden-w8a8":
            model = Golden(base)
            model.calibrate([(p.ids, p.stress) for p in calibration])
            model.save_scales(out / "scales.json")
        else:
            model = base if spec is None else quantized_model(base, *spec)
        seed = 1 if name == "fp32-seed1" else 0
        clips, scores, same, total = [], [], 0, 0
        for voice in range(len(base.config.speakers)):
            for i, p in enumerate(tokens):
                free = run(model, p, voice, seed=seed)
                clips += [free["audio"], gap]
                if name == "fp32":
                    reference[i, voice] = free
                    continue
                ref = reference[i, voice]
                same += int((free["durations"] == ref["durations"]).sum()); total += ref["durations"].numel()
                scores.append(compare(ref, run(model, p, voice, durations=ref["durations"], seed=seed)))
        if not name.startswith("fp32-"):
            write_wav(out / f"{name}.wav", torch.cat(clips))
        if scores:
            mel, cents, flips = (sum(s[k] for s in scores) / len(scores) for k in range(3))
            print(f"{name:10s} {mel:9.2f} {cents:8.1f} {flips:8.2%} {same / total:9.1%}")
        else:
            print(f"{name:10s} {'reference':>9s}")
    print(f"\nWAVs (F2 then M2, sentences in order): {out}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
