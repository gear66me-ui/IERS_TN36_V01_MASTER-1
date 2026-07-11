#!/usr/bin/env bash
set -euo pipefail

BASE="https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/PYTHON/IERS_2012_GEOCENTRIC"
AJ_TARGET="IERS_0012AJ_CAPE_TOWN_SITE_COORD_VS_VIDEO.py"
AN_TARGET="${1:-IERS_0012AN_CAPE_TOWN_UPPER_HALF_R2_RMS.py}"

curl -fsSL --retry 3 --retry-delay 1 "$BASE/$AJ_TARGET" -o "$AJ_TARGET"
curl -fsSL --retry 3 --retry-delay 1 "$BASE/IERS_0012AN_CAPE_TOWN_UPPER_HALF_R2_RMS.py" -o "$AN_TARGET"

python -m py_compile "$AJ_TARGET" "$AN_TARGET"
printf 'Downloaded from IERS repository: %s\n' "$AN_TARGET"
printf 'Version: IERS-0012AN\n'
printf 'Display: true upper half-Sun; C1 left; C4 lower-right\n'
printf 'Statistics: R-squared and RMS only\n'
printf 'Outputs: inline Matplotlib PNG + Drive CSV\n'
