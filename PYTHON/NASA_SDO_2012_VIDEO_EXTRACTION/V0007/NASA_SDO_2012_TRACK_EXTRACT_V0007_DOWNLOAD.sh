#!/usr/bin/env bash
set -euo pipefail

BASE="https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/PYTHON/NASA_SDO_2012_VIDEO_EXTRACTION"
TARGET="${1:-NASA_SDO_2012_TRACK_EXTRACT.py}"
PATCH_FILE="NASA_SDO_2012_TRACK_EXTRACT_V0007_PATCH_FIXED.py"

curl -fsSL --retry 3 --retry-delay 1 \
  "$BASE/V0006/NASA_SDO_2012_TRACK_EXTRACT_V0006_DOWNLOAD.sh" \
  | bash -s -- "$TARGET"

curl -fsSL --retry 3 --retry-delay 1 \
  "$BASE/V0007/$PATCH_FILE" \
  -o "$PATCH_FILE"

python "$PATCH_FILE" "$TARGET"
rm -f "$PATCH_FILE"
python -m py_compile "$TARGET"

SHA256="$(sha256sum "$TARGET" | awk '{print $1}')"
printf 'Downloaded from IERS repository: %s\n' "$TARGET"
printf 'Version: V0007\n'
printf 'Analysis window: 46.0-53.0 seconds\n'
printf 'SHA-256: %s\n' "$SHA256"
