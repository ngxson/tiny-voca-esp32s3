"""Send lines to the firmware console and print the replies (each line waits for the next READY).
Hard deadline. With --boot, first resets the board and prints its boot log up to the first READY.

    python tools/console.py --boot "Hello there." "voice F2" "How are you?"
"""
import argparse, glob, signal, sys, time
import serial


def default_port():
    """The first /dev/ttyACM* (Linux) or /dev/cu.usbmodem* (macOS): the S3's USB Serial/JTAG."""
    found = sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/cu.usbmodem*"))
    return found[0] if found else "/dev/ttyACM0"


ap = argparse.ArgumentParser()
ap.add_argument("lines", nargs="*")
ap.add_argument("--port", default=default_port())
ap.add_argument("--boot", action="store_true", help="reset the board and show the boot log")
ap.add_argument("--timeout", type=int, default=300)
args = ap.parse_args()
signal.signal(signal.SIGALRM, lambda *_: sys.exit("\n[console: hard timeout]"))
signal.alarm(args.timeout)


def open_port():
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = args.port, 115200, 0.2
    s.rts = False  # only RTS: DTR low with RTS high resets the S3
    for _ in range(50):
        try:
            s.open(); break
        except Exception:
            time.sleep(0.2)
    return s


s = open_port()
if args.boot:   # opening the port no longer resets the board: reset it explicitly
    s.dtr = False; s.rts = True; time.sleep(0.1); s.rts = False
    s.close(); time.sleep(0.5)
    s = open_port()     # the USB device comes back after the reset
buf = b""

def until_ready(echo=True):
    global buf
    while True:
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            line = raw.decode(errors="replace").rstrip("\r")
            if line.startswith("tts> "):
                line = line[5:]
            if line == "READY":
                return
            if echo and not line.startswith(("I (", "W (")):
                print(line, flush=True)
            if "Guru Meditation" in line or "abort()" in line:
                time.sleep(1); print(s.read(8192).decode(errors="replace")); sys.exit(1)
        try:
            buf += s.read(4096)
        except Exception:
            time.sleep(0.3)

if args.boot:
    until_ready()
else:
    s.write(b"\n"); until_ready(echo=False)
    # A board still booting answers twice (its boot READY, then the empty line): drop the second.
    time.sleep(0.3); s.reset_input_buffer(); buf = b""
for line in args.lines:
    print(f">>> {line}", flush=True)
    t0 = time.time()
    s.write(line.encode() + b"\n")
    until_ready()
    print(f"    ({time.time() - t0:.1f} s)", flush=True)
