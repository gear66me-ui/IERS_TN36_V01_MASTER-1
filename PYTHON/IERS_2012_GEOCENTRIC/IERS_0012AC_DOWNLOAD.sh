#!/usr/bin/env bash
set -euo pipefail

BASE="https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/PYTHON/IERS_2012_GEOCENTRIC"
TARGET="${1:-IERS_0012AC_2012_GEOCENTRIC_TRACK_PNG.py}"
EXPECTED_SHA256="565767c0706df46f49abf78f6cd4f967681e3d4b57ad8f7523a3c0d9165b1c19"

curl -fsSL --retry 3 --retry-delay 1 \
  "$BASE/IERS_0012AC_2012_GEOCENTRIC_TRACK_PNG.py" \
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
printf 'Version: IERS-0012AC\n'
printf 'Outputs: PNG track + PNG results sheet\n'
printf 'SHA-256: %s\n' "$ACTUAL_SHA256"
