#!/usr/bin/env bash
# Print the ESP32 serial log and send stdin to the board (Ctrl+C to quit).
# Usage: ./monitor.sh [-r] [port]
#   -r    reset the board first, to get the boot log
#   port  defaults to the first /dev/ttyACM* (Linux) or /dev/cu.usbmodem* (macOS)
# Typed keys go to the board as they are pressed (the board echoes them and handles
# backspace); piped input is forwarded too: echo "Hello there." | ./monitor.sh
set -euo pipefail
cd "$(dirname "$0")"

RESET=0
if [ "${1:-}" = "-r" ]; then
    RESET=1
    shift
fi

PORT="${1:-$(ls /dev/ttyACM* /dev/cu.usbmodem* 2>/dev/null | head -n1 || true)}"
if [ -z "$PORT" ]; then
    echo "No /dev/ttyACM* or /dev/cu.usbmodem* found - is the board plugged in?" >&2
    exit 1
fi

# ESP-IDF's Python if installed, else python3 (needs pyserial: pip install pyserial)
PY="${PY:-$(ls -d "$HOME"/.espressif/python_env/idf*_env/bin/python 2>/dev/null | tail -n1)}"
PY="${PY:-python3}"
if ! "$PY" -c "import serial" 2>/dev/null; then
    echo "$PY has no pyserial: pip install pyserial" >&2
    exit 1
fi
# A killed idf.py monitor can leave the terminal raw (no Ctrl+C): reset it
if [ -t 0 ]; then
    stty sane
fi
echo "port=$PORT (Ctrl+C to quit)" >&2

exec "$PY" -I -u -c '
import os, signal, sys, termios, threading, time, serial
port, reset = sys.argv[1], sys.argv[2] == "1"
stop = False
def on_signal(*_):
    global stop
    if stop:
        os._exit(130)  # second Ctrl+C: quit immediately
    stop = True
# Install handlers explicitly: SIGINT may be inherited as ignored
for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, on_signal)
try:
    # exclusive: a second monitor/keymap/flash must fail loudly, not share the port
    s = serial.Serial(port, 115200, timeout=0.2, exclusive=True)
except serial.SerialException as e:
    print(f"Cannot open {port}: {e}\nIs another monitor.sh / keymap.sh / idf.py using it?", file=sys.stderr)
    sys.exit(1)
if reset:
    s.dtr = False; s.rts = True; time.sleep(0.1); s.rts = False

def forward_stdin():
    while not stop:
        try:
            data = os.read(0, 1024)
        except OSError:
            return
        if not data:          # end of piped input: keep showing the board output
            return
        s.write(data.replace(b"\r\n", b"\n"))

# On a terminal, send keys one by one without local echo (the board echoes them);
# Ctrl+C keeps working. The terminal is restored on the way out.
saved = None
if os.isatty(0):
    saved = termios.tcgetattr(0)
    mode = termios.tcgetattr(0)
    mode[3] &= ~(termios.ECHO | termios.ICANON)
    mode[6][termios.VMIN], mode[6][termios.VTIME] = 1, 0
    termios.tcsetattr(0, termios.TCSANOW, mode)
try:
    threading.Thread(target=forward_stdin, daemon=True).start()
    while not stop:
        d = s.read(4096)
        if d:
            sys.stdout.buffer.write(d); sys.stdout.flush()
finally:
    if saved:
        termios.tcsetattr(0, termios.TCSANOW, saved)
s.close()
print("\n[monitor stopped]", file=sys.stderr)
' "$PORT" "$RESET"
