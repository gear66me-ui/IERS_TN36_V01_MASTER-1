#!/usr/bin/env bash
set -euo pipefail

BASE="https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/PYTHON/IERS_2012_GEOCENTRIC"
AJ_TARGET="IERS_0012AJ_CAPE_TOWN_SITE_COORD_VS_VIDEO.py"
AU_TARGET="${1:-IERS_0012AU_CAPE_TOWN_COMPLETE_WIDGET.py}"

curl -fsSL --retry 3 --retry-delay 1 "$BASE/$AJ_TARGET" -o "$AJ_TARGET"
curl -fsSL --retry 3 --retry-delay 1 "$BASE/IERS_0012AU_CAPE_TOWN_COMPLETE_WIDGET.py" -o "$AU_TARGET"

python -m py_compile "$AJ_TARGET" "$AU_TARGET"
printf 'Downloaded from IERS repository: %s\n' "$AU_TARGET"
printf 'Version: IERS-0012AU\n'
printf 'Orientation: upper half-Sun; C1 left/high; C4 right/lower\n'
printf 'Table: angles, delta-beta, delta-m, ratio, cubic R-squared, RMS, site, C1-C4 UTC\n'
printf 'Outputs: inline Matplotlib PNG + Drive CSV\n'
