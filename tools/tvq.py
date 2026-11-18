"""Golden model of the ESP32-S3 engine numerics (W8A8, static activation scales).

This is the specification the C engine follows op by op. It runs in float32 but
rounds and clamps exactly where the engine holds int8:

* weights: int8, symmetric, one scale per output channel (convolutions are run as
  matmuls over [x[t-1], x[t], x[t+1]] windows);
* activations entering a matmul: int8, symmetric, one static scale per tensor,
  calibrated as max|x| over calibration sentences; values beyond it saturate;
* attention: Q and K (after rotary) and V int8, scores and softmax in float,
  probabilities int8 in [0, 127] (scale 1/127);
* GELU between `expand` and `project` maps int8 to int8 (a 256-entry table on device);
* float: residual stream, LayerNorm, depthwise conv, softmax, the tiny 3-channel
  linears, duration logic and the source-filter vocoder.

Quantization points (keys of `scales`):
  <stack>.<i>.ln1 .q .k .v .attn .dw .pre .gelu   for every block of encoder/prosody/acoustic
  dur.in dur.h1     inputs of the two duration convolutions
  voc.in            input of the vocoder's harmonic and noise heads
"""
import copy, json, math
import torch
from torch import nn
from torch.nn import functional as F

STACKS = ("encoder", "prosody_stack", "acoustic_stack")
GELU_MIN = -0.16997124   # min over x of x * Phi(x), at x = -0.7518


def fq(x, scale, qmax=127):
    """Fake-quantize: round to the int8 grid of `scale`, saturate at +-qmax."""
    return torch.round(x / scale).clamp(-qmax, qmax) * scale


def weight_int8(w):
    """Per-output-channel int8: returns (int8 values as float, scale per row)."""
    flat = w.reshape(w.shape[0], -1)
    scale = (flat.abs().amax(1) / 127).clamp_min(1e-12)
    return torch.round(flat / scale[:, None]).clamp(-127, 127), scale


def conv_as_matrix(w):
    """Conv1d weight [out, in, 3] -> [out, 3*in] acting on [x[t-1], x[t], x[t+1]]."""
    return w.permute(0, 2, 1).reshape(w.shape[0], -1)


def windows3(x):
    """[T, C] -> [T, 3C]: rows [x[t-1], x[t], x[t+1]] with zero padding."""
    pad = F.pad(x, (0, 0, 1, 1))
    return torch.cat((pad[:-2], pad[1:-1], pad[2:]), dim=1)


class Golden:
    def __init__(self, model, scales=None, quantize_weights=True, only=None):
        self.m = model
        self.c = model.config
        self.scales = scales            # None: calibration mode (records max|x|)
        self.observed = {}
        self.dump = None
        self.only = only                # ablation: quantize only points whose name ends with one of these
        sd = model.state_dict()
        self.w = {}                     # name -> (dequantized float weight [out, in], bias)
        for name, value in sd.items():
            if not name.endswith(".weight") or value.ndim < 2:
                continue
            base = name[:-len(".weight")]
            if base.endswith("depthwise") or value.shape[1] <= 3 or base in ("prosody_head",) or base.startswith(("tokens", "stress", "voices", "frame_voices")):
                continue
            if base == "duration.out":
                continue
            matrix = conv_as_matrix(value) if value.ndim == 3 else value
            if quantize_weights:
                q, s = weight_int8(matrix)
                matrix = q * s[:, None]
            self.w[base] = (matrix, sd[base + ".bias"])

    # ------------------------------------------------------------- helpers
    def act(self, name, x):
        if self.dump is not None:
            self.dump[name + ".float"] = x.detach().clone()
        if self.scales is None:
            self.observed[name] = max(self.observed.get(name, 0.0), float(x.abs().max()))
            return x
        if self.only is not None and not name.endswith(self.only):
            return x
        if name.endswith(".gelu"):
            # Asymmetric: GELU output lies in [GELU_MIN, max], so all 256 codes cover that range.
            # int8 code q in [-128, 127] stands for (q + 128) * scale + GELU_MIN; the engine folds
            # W . (128 * scale + GELU_MIN) into the bias of `project`.
            scale = (self.scales[name] - GELU_MIN) / 255
            code = torch.round((x - GELU_MIN) / scale).clamp(0, 255)
            if self.dump is not None:
                self.dump[name] = (code - 128).to(torch.int8)
            return code * scale + GELU_MIN
        y = fq(x, self.scales[name] / 127)
        if self.dump is not None:
            self.dump[name] = torch.round(y / (self.scales[name] / 127)).to(torch.int8)
        return y

    def linear(self, base, x):
        w, b = self.w[base]
        return x @ w.T + b

    def layer_norm(self, module, x):
        return F.layer_norm(x, (x.shape[-1],), module.weight, module.bias, module.eps)

    # ------------------------------------------------------------- blocks
    def attention(self, prefix, block, x, table):
        t, c = x.shape
        h, d = self.c.heads, c // self.c.heads
        qkv = self.linear(prefix + ".attention.qkv", self.act(prefix + ".ln1", x)).view(t, 3, h, d).permute(1, 2, 0, 3)
        cos, sin = table

        def rope(z):
            a, b = z[..., 0::2], z[..., 1::2]
            return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)
        q = self.act(prefix + ".q", rope(qkv[0]))
        k = self.act(prefix + ".k", rope(qkv[1]))
        v = self.act(prefix + ".v", qkv[2])
        scores = q @ k.transpose(-1, -2) / math.sqrt(d)
        if self.scales is not None and (self.only is None or ".probs" in self.only):
            # Row-max scaling: the largest weight of each row is exactly 127, so diffuse
            # attention keeps its resolution; normalize by the sum of the int8 weights after.
            weights = torch.round(127 * torch.exp(scores - scores.amax(-1, keepdim=True)))
            y = (weights @ v) / weights.sum(-1, keepdim=True)
        else:
            y = scores.softmax(-1) @ v
        y = y.transpose(0, 1).reshape(t, c)
        return self.linear(prefix + ".attention.out", self.act(prefix + ".attn", y))

    def block(self, prefix, block, x, table):
        x = x + self.attention(prefix, block, self.layer_norm(block.norm1, x), table)
        y = self.layer_norm(block.norm2, x)
        dw = block.depthwise
        y = F.conv1d(y.T[None], dw.weight, dw.bias, padding=dw.padding, groups=dw.groups)[0].T
        pre = self.linear(prefix + ".expand", self.act(prefix + ".dw", y))
        g = F.gelu(self.act(prefix + ".pre", pre))
        return x + self.linear(prefix + ".project", self.act(prefix + ".gelu", g))

    def stack(self, name, x):
        module = getattr(self.m, name)
        table = self._rotary(x.shape[0])
        for i, block in enumerate(module.blocks):
            x = self.block(f"{name}.blocks.{i}", block, x, table)
            if self.dump is not None:
                self.dump[f"{name}.{i}.out"] = x.detach().clone()
        x = self.layer_norm(module.norm, x)
        if self.dump is not None:
            self.dump[f"{name}.final"] = x.detach().clone()
        return x

    def _rotary(self, t):
        width = self.c.channels // self.c.heads
        inverse = 1.0 / (10000 ** (torch.arange(0, width, 2, dtype=torch.float32) / width))
        angle = torch.arange(t, dtype=torch.float32)[:, None] * inverse[None]
        return angle.cos(), angle.sin()

    # ------------------------------------------------------------- full model
    def durations(self, hidden, tokens, speed=1.0):
        d, m = self.m.duration, self.m
        x = self.linear("duration.conv1", self.act("dur.in", windows3(hidden)))
        x = self.layer_norm(d.norm1, F.gelu(x))
        x = self.linear("duration.conv2", self.act("dur.h1", windows3(x)))
        x = self.layer_norm(d.norm2, F.gelu(x))
        out = x @ d.out.weight.T + d.out.bias
        if self.dump is not None:
            self.dump["dur.out"] = out.detach().clone()
        frames = torch.round(torch.expm1(out[:, 0].clamp(0, math.log1p(400))) / speed).long().clamp_min(1)
        present = m.phone_tokens[tokens] | (out[:, 1] > 0)
        return torch.where(present, frames, torch.zeros_like(frames))

    @torch.inference_mode()
    def synthesize(self, ids, stress, voice, seed=0, durations=None, dump=False):
        m = self.m
        self.dump = {} if dump else None
        tokens = torch.tensor(ids)
        x = m.tokens.weight[tokens] + m.stress.weight[torch.tensor(stress)] + m.voices.weight[voice]
        hidden = self.stack("encoder", x)
        if durations is None:
            durations = self.durations(hidden, tokens)
        durations = durations.reshape(-1)
        speakers = torch.tensor([voice])
        frames, frame_mask, lengths = m.expand(hidden[None], durations[None], torch.ones(1, len(ids), dtype=torch.bool), speakers)
        frames = frames[0]
        prosody_hidden = self.stack("prosody_stack", frames)
        prosody = prosody_hidden @ m.prosody_head.weight.T + m.prosody_head.bias
        owner = torch.searchsorted(durations.cumsum(0), torch.arange(frames.shape[0]), right=True).clamp(max=len(ids) - 1)
        vowel = m.vowel_tokens[tokens][owner]
        prosody[:, 1] = torch.where(vowel, prosody[:, 1].clamp_min(2.0), prosody[:, 1])
        condition = m.hard_prosody(prosody)
        acoustic = self.stack("acoustic_stack", prosody_hidden + m.prosody_projection(condition))
        # Vocoder: the two linear heads use int8; the rest is the float model.
        vocoder = copy.copy(m.vocoder)
        vocoder._modules = dict(vocoder._modules)   # copy.copy shares the submodule dict: never touch the model's
        golden = self

        class Head(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base

            def forward(self, z):
                return golden.linear(self.base, golden.act("voc.in", z))
        vocoder.harmonic, vocoder.noise = Head("vocoder.harmonic"), Head("vocoder.noise")
        n = int(lengths[0])
        noise = torch.randn(n * self.c.hop, generator=torch.Generator().manual_seed(seed))
        wave = vocoder(acoustic[None], m.frequency(prosody[None, :, 0]), lengths, noise=noise[None])
        if self.dump is not None:
            self.dump.update(durations=durations, prosody=prosody, acoustic=acoustic, noise=noise, audio=wave[0])
        return {"audio": wave[0, :n * self.c.hop], "durations": durations, "prosody": prosody}

    def calibrate(self, utterances):
        """utterances: [(ids, stress)]; every voice. Returns the scales dict (max|x|)."""
        self.scales, self.observed = None, {}
        for ids, stress in utterances:
            for voice in range(len(self.c.speakers)):
                self.synthesize(ids, stress, voice)
        # float32 values: exactly what the export stores, so C and Python round alike
        self.scales = {k: float(torch.tensor(v, dtype=torch.float32)) for k, v in self.observed.items()}
        return self.scales

    def save_scales(self, path):
        json.dump(self.scales, open(path, "w"), indent=1, sort_keys=True)
