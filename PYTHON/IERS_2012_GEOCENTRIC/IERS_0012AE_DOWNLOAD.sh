#!/usr/bin/env bash
set -euo pipefail

BASE="https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/PYTHON/IERS_2012_GEOCENTRIC"
AD_TARGET="IERS_0012AD_2012_VIDEO_VS_JPL_ROTATED.py"
AE_TARGET="${1:-IERS_0012AE_2012_VIDEO_VS_JPL_ROTATED_FIXED.py}"
EXPECTED_SHA256="5360b5ca22371e7801163a8883f830d59857dbe2ec4e3f7d5855487451cac24b"

curl -fsSL --retry 3 --retry-delay 1 \
  "$BASE/IERS_0012AD_DOWNLOAD.sh" \
  | bash -s -- "$AD_TARGET"

curl -fsSL --retry 3 --retry-delay 1 \
  "$BASE/IERS_0012AE_2012_VIDEO_VS_JPL_ROTATED_FIXED.py" \
  -o "$AE_TARGET"

ACTUAL_SHA256="$(sha256sum "$AE_TARGET" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "IERS-0012AE SHA-256 verification failed." >&2
  echo "Expected: $EXPECTED_SHA256" >&2
  echo "Actual  : $ACTUAL_SHA256" >&2
  rm -f "$AE_TARGET"
  exit 1
fi

python -m py_compile "$AD_TARGET" "$AE_TARGET"
printf 'Downloaded from IERS repository: %s\n' "$AE_TARGET"
printf 'Version: IERS-0012AE\n'
printf 'Fix: Python 3.12 dataclass module registration\n'
printf 'Outputs: inline PNG overlay + inline PNG results + Drive CSV\n'
printf 'SHA-256: %s\n' "$ACTUAL_SHA256"
