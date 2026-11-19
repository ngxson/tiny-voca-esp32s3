"""Read the firmware's console: print the log, decode TVDUMP base64 blocks into a TVQ1 file.
Hard deadline; stops on TVDUMP_DONE, "tv: finished", a failed TEST, a panic or a reboot.

    python tools/serial_dump.py /dev/cu.usbmodem1201 180 out/dev_test0.bin
"""
import base64, signal, struct, sys, time
import serial

port, timeout, out_path = sys.argv[1], float(sys.argv[2]), sys.argv[3]
send = sys.argv[4].encode() + b"\n" if len(sys.argv) > 4 else None     # command to send once READY


def bail(*_):
    print("\n[serial_dump: hard timeout]")
    sys.exit(2)


signal.signal(signal.SIGALRM, bail)
signal.alarm(int(timeout))

s = None
while s is None:
    try:
        s = serial.Serial()
        s.port, s.baudrate, s.timeout = port, 115200, 0.2
        s.dtr = s.rts = False
        s.open()
    except Exception:
        s = None
        time.sleep(0.2)
if send:
    s.write(b"\n")

dumps, current, chunks, pending = {}, None, [], b""
finished = (b"TVDUMP_DONE", b"tv: finished", b"no test vector", b"synthesize failed")     # the last two: TEST failed
panic = (b"Guru Meditation", b"Rebooting...", b"abort() was called")
done = panicked = False
while not done:
    try:
        data = s.read(65536)
    except Exception:
        time.sleep(0.3)
        try:
            s = serial.Serial(port, 115200, timeout=0.2)
        except Exception:
            pass
        continue
    pending += data
    while b"\n" in pending:
        raw, pending = pending.split(b"\n", 1)
        line = raw.decode(errors="replace").rstrip("\r")
        if send and line.endswith("READY"):
            s.write(send); send = None
            continue
        if line.startswith("TVDUMP_END"):
            name, rows, cols = current or ("?", 0, 0)
            dumps[name] = (base64.b64decode("".join(chunks)), rows, cols)
            print(f"[decoded {name}: {rows}x{cols}]")
            current = None
        elif line.startswith("TVDUMP ") and not line.startswith("TVDUMP_"):
            _, name, rows, cols = line.split()
            current, chunks = (name, int(rows), int(cols)), []
        elif current is not None:
            chunks.append(line)
        else:
            print(line)
        panicked = panicked or any(m in raw for m in panic)
        done = done or panicked or any(m in raw for m in finished)
if panicked:                    # print the rest of the panic report
    time.sleep(1.0)
    print((pending + s.read(16384)).decode(errors="replace"))

if dumps:
    header = 16 + 96 * len(dumps)
    offset = (header + 15) & ~15
    entries, blobs = [], []
    for name, (blob, rows, cols) in dumps.items():
        entries.append(struct.pack("<64sII4III", name.encode(), 0, 2, rows, cols, 1, 1, offset, len(blob)))
        blobs.append((offset, blob))
        offset = (offset + len(blob) + 15) & ~15
    with open(out_path, "wb") as f:
        f.write(b"TVQ1" + struct.pack("<III", 1, len(dumps), 0) + b"".join(entries))
        for off, blob in blobs:
            f.write(b"\0" * (off - f.tell()))
            f.write(blob)
    print(f"[wrote {out_path}: {', '.join(dumps)}]")
