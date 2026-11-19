#!/bin/bash
# Build, flash (app; model + test vector only with --data), run TEST and capture its dump to
# out/dev_test0.bin. Hard timeouts everywhere. ESP-IDF: on PATH already, or from $IDF_PATH.
#   BOARD=waveshare_s3_audio ./run.sh [--data] [capture seconds]
set -o pipefail
to() { local s=$1; shift; perl -e 'alarm shift; exec @ARGV' "$s" "$@"; }
PORT=/dev/cu.usbmodem1201
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/../out
DATA=0; [ "$1" = "--data" ] && { DATA=1; shift; }
CAPTURE=${1:-180}
BOARD=${BOARD:-waveshare_s3_audio}
IDF=(idf.py -B "build/$BOARD" -DBOARD="$BOARD")
cd "$HERE" || exit 1
command -v idf.py >/dev/null || . "${IDF_PATH:?ESP-IDF not found: set IDF_PATH or source its export.sh}/export.sh" >/dev/null 2>&1
to 400 "${IDF[@]}" build 2>&1 | grep -E "error:|Project build complete" || exit 1
to 180 "${IDF[@]}" -p $PORT flash 2>&1 | grep -E "error|Hash of data verified|Done" | tail -2 || exit 1
if [ $DATA = 1 ]; then
  to 300 esptool --port $PORT write-flash 0x210000 "$OUT/tvoca.bin" 0x810000 "$OUT/test0_dev.bin" 2>&1 | grep -E "Wrote|error|Hash" | tail -3 || exit 1
fi
to $((CAPTURE + 10)) python -I "$HERE/../tools/serial_dump.py" $PORT "$CAPTURE" "$OUT/dev_test0.bin" TEST | grep -av "^I (\|^W (" | sed -n '/TEST/,$p'
