#!/usr/bin/env bash
set -euo pipefail

BASE="https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/PYTHON/IERS_2012_GEOCENTRIC"
AJ_TARGET="IERS_0012AJ_CAPE_TOWN_SITE_COORD_VS_VIDEO.py"
AU_TARGET="IERS_0012AU_CAPE_TOWN_COMPLETE_WIDGET.py"
AW_TARGET="${1:-IERS_0012AW_USER_SOLAR_IMAGE_WIDGET.py}"

curl -fsSL --retry 3 --retry-delay 1 "$BASE/$AJ_TARGET" -o "$AJ_TARGET"
curl -fsSL --retry 3 --retry-delay 1 "$BASE/$AU_TARGET" -o "$AU_TARGET"
curl -fsSL --retry 3 --retry-delay 1 "$BASE/IERS_0012AW_USER_SOLAR_IMAGE_WIDGET.py" -o "$AW_TARGET"

python -m py_compile "$AJ_TARGET" "$AU_TARGET" "$AW_TARGET"
printf 'Downloaded from IERS repository: %s\n' "$AW_TARGET"
printf 'Version: IERS-0012AW\n'
printf 'Image input: upload the second solar image when prompted\n'
printf 'Rendering: Python/Pillow/Matplotlib only; no AI images\n'
printf 'Venus disks: black; C1 and C4 outlined in white\n'
printf 'Orientation: upper half-Sun; C1 left/high; C4 right/lower\n'
printf 'Outputs: inline PNG + Drive CSV\n'
