"""Text -> phonemes on the Mac -> synthesis on the ESP32-S3 -> WAV, with a keep-up report.

    python tools/speak_s3.py --text "Hello there. How are you today?" --out out/s3.wav

The firmware (firmware/) must be running its command server. Phonemes come from the C++
dizione port (out/bin/dizione_cli, built from third_party/dizione); without that
binary a plain lexicon lookup is used and words missing from the lexicon are skipped.
Keep-up model: sentences are synthesized one after another and each plays as soon as it is
ready and the previous one has finished; any wait between sentences is a gap.
"""
import argparse, base64, glob, os, signal, subprocess, sys, time, wave
from pathlib import Path
import serial

REPO = Path(__file__).resolve().parents[1]                     # this repo; outputs go to REPO / "out"
# The tiny-voca source (for the model code) and its weights; only needed to export or evaluate.
TINY_VOCA = Path(os.environ.get("TINY_VOCA_DIR", REPO.parent)).resolve()
WEIGHTS = Path(os.environ.get("TINY_VOCA_WEIGHTS",
                              TINY_VOCA / "weights" if (TINY_VOCA / "weights").exists() else TINY_VOCA.parent / "models/tiny-voca"))
sys.path.insert(0, str(TINY_VOCA / "src"))
from tiny_voca.infer import sentences, finished                          # noqa: E402
from tiny_voca.text import chunks, normalize_text, tokenize, parse_espeak  # noqa: E402


def default_port():
    """The first /dev/ttyACM* (Linux) or /dev/cu.usbmodem* (macOS): the S3's USB Serial/JTAG."""
    found = sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/cu.usbmodem*"))
    return found[0] if found else "/dev/ttyACM0"


def load_lexicon(path):
    lexicon = {}
    for line in open(path, encoding="utf-8"):
        word, phones = line.rstrip("\n").split("\t")
        lexicon.setdefault(word, phones.replace(" ", "_"))
    return lexicon


def phonemize(text, lexicon):
    pieces, missing = [], []
    for punctuation, piece in chunks(normalize_text(text)):
        if punctuation:
            pieces.append((True, piece))
            continue
        words = []
        for word in piece.lower().split():
            if word in lexicon:
                words += parse_espeak(lexicon[word])
            else:
                missing.append(word)
        pieces.append((False, words))
    return tokenize(pieces), missing


class Dizione:
    """The C++ phonemizer as a subprocess: one sentence per line in, ids and stress out."""
    def __init__(self, binary, data):
        self.p = subprocess.Popen([binary, data], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def __call__(self, sentence):
        self.p.stdin.write(sentence.replace("\n", " ") + "\n")
        self.p.stdin.flush()
        out = self.p.stdout.readline().rstrip("\n")
        if out.startswith("ERR"):
            raise ValueError(out)
        ids, stress = out.split("\t")
        return [int(v) for v in ids.split(",")], [int(v) for v in stress.split(",")]


class Board:
    def __init__(self, port):
        self.s = serial.Serial()
        self.s.port, self.s.baudrate, self.s.timeout = port, 115200, 0.2
        self.s.rts = False  # only RTS: DTR low with RTS high resets the S3
        self.s.open()
        self.buf = b""

    def line(self, deadline):
        while b"\n" not in self.buf:
            if time.time() > deadline:
                raise TimeoutError("no answer from the board")
            self.buf += self.s.read(65536)
        raw, self.buf = self.buf.split(b"\n", 1)
        return raw.decode(errors="replace").rstrip("\r")

    def wait_ready(self, seconds):
        deadline = time.time() + seconds
        self.s.write(b"\n")
        while True:
            line = self.line(deadline)
            if line == "READY":
                return
            if "Guru Meditation" in line or "abort()" in line:
                raise RuntimeError(line)

    def synthesize(self, ids, stress, voice, seed, seconds):
        cmd = f"SYN {voice} {seed} {len(ids)} {','.join(map(str, ids))} {','.join(map(str, stress))}\n"
        self.s.write(cmd.encode())
        deadline = time.time() + seconds
        result, pcm, chunks_, in_dump = None, None, [], False
        while True:
            line = self.line(deadline)
            if line.startswith("RESULT "):
                result = dict(kv.split("=") for kv in line.split()[1:])
            elif line.startswith("TVDUMP pcm"):
                in_dump, chunks_ = True, []
            elif line.startswith("TVDUMP_END"):
                pcm, in_dump = base64.b64decode("".join(chunks_)), False
            elif in_dump:
                chunks_.append(line)
            elif line.startswith("ERR") or "Guru Meditation" in line or "abort()" in line:
                raise RuntimeError(line)
            elif line == "DONE":
                return result, pcm
            elif line.startswith("  ") and args.profile:
                print("   ", line.strip())


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", choices=("F2", "M2"), default="M2")
    parser.add_argument("--port", default=default_port())
    parser.add_argument("--out", default=str(REPO / "out/s3.wav"))
    parser.add_argument("--phonemizer", default=str(WEIGHTS / "phonemizer"))
    parser.add_argument("--dizione", default=str(REPO / "out/bin/dizione_cli"))
    parser.add_argument("--timeout", type=int, default=600, help="hard limit for the whole run, seconds")
    parser.add_argument("--profile", action="store_true", help="print the board's per-op profile")
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, lambda *_: sys.exit("[speak_s3: hard timeout]"))
    signal.alarm(args.timeout)

    if Path(args.dizione).exists():
        dizione = Dizione(args.dizione, args.phonemizer)
        to_tokens = lambda text: (dizione(text), [])
    else:
        print("(dizione_cli not built: falling back to a plain lexicon lookup)")
        lexicon = load_lexicon(str(Path(args.phonemizer) / "lexicon.tsv"))
        def to_tokens(text):
            p, missing = phonemize(text, lexicon)
            return (list(p.ids), list(p.stress)), missing
    voice = 0 if args.voice == "F2" else 1
    board = Board(args.port)
    board.wait_ready(30)
    pcm_all, rows, gap = [], [], int(0.25 * 24000)
    print(f"{'#':>2} {'tokens':>6} {'audio s':>7} {'synth s':>7} {'RTF':>5}  text")
    for i, sentence in enumerate(sentences(finished(args.text))):
        (ids, stress), missing = to_tokens(sentence)
        if missing:
            print(f"   (skipped words not in the lexicon: {', '.join(missing)})")
        result, pcm = board.synthesize(ids, stress, voice, seed=i, seconds=300)
        board.wait_ready(10)
        audio_s, synth_s = float(result["audio_ms"]) / 1000, float(result["synth_ms"]) / 1000
        rows.append((audio_s, synth_s))
        print(f"{i:2d} {len(ids):6d} {audio_s:7.2f} {synth_s:7.2f} {audio_s / synth_s:5.2f}  {sentence}"
              + (f"  [saturations {result['saturations']}]" if result["saturations"] != "0" else ""))
        pcm_all.append(pcm + b"\0\0" * gap)
    with wave.open(args.out, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(24000); f.writeframes(b"".join(pcm_all))
    # keep-up: sequential synthesis, playback starts as soon as a sentence is ready
    ready, play_end, gaps = 0.0, None, 0.0
    for audio_s, synth_s in rows:
        ready += synth_s
        start = ready if play_end is None else max(ready, play_end)
        if play_end is not None:
            gaps += start - play_end
        play_end = start + audio_s
    total_audio, total_synth = sum(a for a, _ in rows), sum(s for _, s in rows)
    print(f"\ntotal: {total_audio:.2f} s of audio in {total_synth:.2f} s (RTF {total_audio / total_synth:.2f}); "
          f"first audio after {rows[0][1]:.2f} s; gaps while playing: {gaps:.2f} s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
