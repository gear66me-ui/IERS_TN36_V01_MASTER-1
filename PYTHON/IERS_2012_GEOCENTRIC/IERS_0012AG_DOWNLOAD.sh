#!/usr/bin/env bash
set -euo pipefail

BASE="https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/PYTHON/IERS_2012_GEOCENTRIC"
AC_TARGET="IERS_0012AC_2012_GEOCENTRIC_TRACK_PNG.py"
AG_TARGET="${1:-IERS_0012AG_2012_VIDEO_VS_JPL_NORMALIZED.py}"
EXPECTED_SHA256="469bf723a2d2fab5319032f2e7d2bcb32fffefc76227c4d8a7f75eeefe0198d5"

curl -fsSL --retry 3 --retry-delay 1 \
  "$BASE/IERS_0012AC_2012_GEOCENTRIC_TRACK_PNG.py" \
  -o "$AC_TARGET"

curl -fsSL --retry 3 --retry-delay 1 \
  "$BASE/IERS_0012AG_2012_VIDEO_VS_JPL_NORMALIZED.py" \
  -o "$AG_TARGET"

ACTUAL_SHA256="$(sha256sum "$AG_TARGET" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "IERS-0012AG SHA-256 verification failed." >&2
  echo "Expected: $EXPECTED_SHA256" >&2
  echo "Actual  : $ACTUAL_SHA256" >&2
  rm -f "$AG_TARGET"
  exit 1
fi

python -m py_compile "$AC_TARGET" "$AG_TARGET"
printf 'Downloaded from IERS repository: %s\n' "$AG_TARGET"
printf 'Version: IERS-0012AG\n'
printf 'Normalization: temporal registration + uniform scale + rotation + translation\n'
printf 'Data: V0007 video CSV + JPL Horizons geocenter vectors\n'
printf 'Outputs: inline Matplotlib PNG overlay + inline PNG results + Drive CSV\n'
printf 'SHA-256: %s\n' "$ACTUAL_SHA256"
