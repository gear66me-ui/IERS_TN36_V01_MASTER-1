#!/usr/bin/env bash
set -euo pipefail

BASE="https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/PYTHON/IERS_2012_GEOCENTRIC"
AJ_TARGET="IERS_0012AJ_CAPE_TOWN_SITE_COORD_VS_VIDEO.py"
AU_TARGET="IERS_0012AU_CAPE_TOWN_COMPLETE_WIDGET.py"
AV_TARGET="${1:-IERS_0012AV_SDO_FRAME_STYLED_WIDGET.py}"

curl -fsSL --retry 3 --retry-delay 1 "$BASE/$AJ_TARGET" -o "$AJ_TARGET"
curl -fsSL --retry 3 --retry-delay 1 "$BASE/$AU_TARGET" -o "$AU_TARGET"
curl -fsSL --retry 3 --retry-delay 1 "$BASE/IERS_0012AV_SDO_FRAME_STYLED_WIDGET.py" -o "$AV_TARGET"

python -m py_compile "$AJ_TARGET" "$AU_TARGET" "$AV_TARGET"
printf 'Downloaded from IERS repository: %s\n' "$AV_TARGET"
printf 'Version: IERS-0012AV\n'
printf 'Image source: actual NASA SDO video frame at 47.5 seconds\n'
printf 'Rendering: Python/OpenCV/Matplotlib only; no AI images\n'
printf 'Style: orange field, dark textured solar disk, black Venus disks\n'
printf 'Orientation: upper half-Sun; C1 left/high; C4 right/lower\n'
printf 'Outputs: inline PNG + Drive CSV\n'
