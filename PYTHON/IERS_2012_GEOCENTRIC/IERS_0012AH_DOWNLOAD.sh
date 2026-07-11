#!/usr/bin/env bash
set -euo pipefail

BASE="https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/PYTHON/IERS_2012_GEOCENTRIC"
AC_TARGET="IERS_0012AC_2012_GEOCENTRIC_TRACK_PNG.py"
AH_TARGET="${1:-IERS_0012AH_2012_VIDEO_NORMALIZED_TO_JPL.py}"
EXPECTED_SHA256="030a90f4eb89db4c09bfcd12e6cabea179711d9c144b95c4eb813c185fa9b3c1"

curl -fsSL --retry 3 --retry-delay 1 \
  "$BASE/IERS_0012AC_2012_GEOCENTRIC_TRACK_PNG.py" \
  -o "$AC_TARGET"

curl -fsSL --retry 3 --retry-delay 1 \
  "$BASE/IERS_0012AH_2012_VIDEO_NORMALIZED_TO_JPL.py" \
  -o "$AH_TARGET"

ACTUAL_SHA256="$(sha256sum "$AH_TARGET" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "IERS-0012AH SHA-256 verification failed." >&2
  echo "Expected: $EXPECTED_SHA256" >&2
  echo "Actual  : $ACTUAL_SHA256" >&2
  rm -f "$AH_TARGET"
  exit 1
fi

python -m py_compile "$AC_TARGET" "$AH_TARGET"
printf 'Downloaded from IERS repository: %s\n' "$AH_TARGET"
printf 'Version: IERS-0012AH\n'
printf 'Gold standard: JPL geocentric track unchanged\n'
printf 'Normalization: SDO video track only\n'
printf 'Angles: degrees from horizontal (-90 to +90)\n'
printf 'Outputs: IERS-0012AA-style plot PNG + table PNG + Drive CSV\n'
printf 'SHA-256: %s\n' "$ACTUAL_SHA256"
