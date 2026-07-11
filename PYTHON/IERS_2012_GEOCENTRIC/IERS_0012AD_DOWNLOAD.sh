#!/usr/bin/env bash
set -euo pipefail

BASE="https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/PYTHON/IERS_2012_GEOCENTRIC"
TARGET="${1:-IERS_0012AD_2012_VIDEO_VS_JPL_ROTATED.py}"
EXPECTED_SHA256="a939c81698efc9f88fd1671d16c122ad61d3448a2b7d4b235bc477a63c801d21"

curl -fsSL --retry 3 --retry-delay 1 \
  "$BASE/IERS_0012AD_2012_VIDEO_VS_JPL_ROTATED.py" \
  -o "$TARGET"

ACTUAL_SHA256="$(sha256sum "$TARGET" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "SHA-256 verification failed." >&2
  echo "Expected: $EXPECTED_SHA256" >&2
  echo "Actual  : $ACTUAL_SHA256" >&2
  rm -f "$TARGET"
  exit 1
fi

python -m py_compile "$TARGET"
printf 'Downloaded from IERS repository: %s\n' "$TARGET"
printf 'Version: IERS-0012AD\n'
printf 'Outputs: inline PNG overlay + inline PNG results + Drive CSV\n'
printf 'SHA-256: %s\n' "$ACTUAL_SHA256"
