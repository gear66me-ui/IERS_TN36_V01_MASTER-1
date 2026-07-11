#!/usr/bin/env bash
set -euo pipefail

BASE="https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/PYTHON/IERS_2012_GEOCENTRIC"
AJ_TARGET="IERS_0012AJ_CAPE_TOWN_SITE_COORD_VS_VIDEO.py"
AP_TARGET="IERS_0012AP_CAPE_TOWN_UPPER_HALF_CUBIC_R2_RMS.py"
AQ_TARGET="${1:-IERS_0012AQ_CAPE_TOWN_UPPER_HALF_X_FLIP.py}"

curl -fsSL --retry 3 --retry-delay 1 "$BASE/$AJ_TARGET" -o "$AJ_TARGET"
curl -fsSL --retry 3 --retry-delay 1 "$BASE/$AP_TARGET" -o "$AP_TARGET"
curl -fsSL --retry 3 --retry-delay 1 "$BASE/IERS_0012AQ_CAPE_TOWN_UPPER_HALF_X_FLIP.py" -o "$AQ_TARGET"

python -m py_compile "$AJ_TARGET" "$AP_TARGET" "$AQ_TARGET"
printf 'Downloaded from IERS repository: %s\n' "$AQ_TARGET"
printf 'Version: IERS-0012AQ\n'
printf 'Display transform: x sign reversed, y unchanged\n'
printf 'Verified orientation: C1 left/high; C4 right/lower; upper half-Sun\n'
printf 'Statistics: cubic R-squared, RMS, delta-beta, delta-m only\n'
printf 'Outputs: inline Matplotlib PNG + Drive CSV\n'
