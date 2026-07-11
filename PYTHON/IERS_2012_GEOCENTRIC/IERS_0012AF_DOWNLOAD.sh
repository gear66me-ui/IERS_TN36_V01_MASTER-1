#!/usr/bin/env bash
set -euo pipefail

BASE="https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/PYTHON/IERS_2012_GEOCENTRIC"
AC_TARGET="IERS_0012AC_2012_GEOCENTRIC_TRACK_PNG.py"
AF_TARGET="${1:-IERS_0012AF_2012_VIDEO_VS_JPL_ALIGNED.py}"
EXPECTED_SHA256="b70010068495d00597585a156808f28339c00dd982ad9247464b48c128bbb42d"

curl -fsSL --retry 3 --retry-delay 1 \
  "$BASE/IERS_0012AC_2012_GEOCENTRIC_TRACK_PNG.py" \
  -o "$AC_TARGET"

curl -fsSL --retry 3 --retry-delay 1 \
  "$BASE/IERS_0012AF_2012_VIDEO_VS_JPL_ALIGNED.py" \
  -o "$AF_TARGET"

ACTUAL_SHA256="$(sha256sum "$AF_TARGET" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "IERS-0012AF SHA-256 verification failed." >&2
  echo "Expected: $EXPECTED_SHA256" >&2
  echo "Actual  : $ACTUAL_SHA256" >&2
  rm -f "$AF_TARGET"
  exit 1
fi

python -m py_compile "$AC_TARGET" "$AF_TARGET"
printf 'Downloaded from IERS repository: %s\n' "$AF_TARGET"
printf 'Version: IERS-0012AF\n'
printf 'Data: V0007 video CSV + JPL Horizons geocenter vectors\n'
printf 'Outputs: inline Matplotlib PNG overlay + inline PNG results + Drive CSV\n'
printf 'SHA-256: %s\n' "$ACTUAL_SHA256"
