#!/usr/bin/env bash
set -euo pipefail

BASE="https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/PYTHON/IERS_2012_GEOCENTRIC"
AJ_TARGET="IERS_0012AJ_CAPE_TOWN_SITE_COORD_VS_VIDEO.py"
AU_TARGET="IERS_0012AU_CAPE_TOWN_COMPLETE_WIDGET.py"
AX_TARGET="IERS_0012AX_REGISTERED_SOLAR_TEXTURE_WIDGET.py"
AY_TARGET="${1:-IERS_0012AY_CAPE_TOWN_TRANSIT_VIDEO.py}"

curl -fsSL --retry 3 --retry-delay 1 "$BASE/$AJ_TARGET" -o "$AJ_TARGET"
curl -fsSL --retry 3 --retry-delay 1 "$BASE/$AU_TARGET" -o "$AU_TARGET"
curl -fsSL --retry 3 --retry-delay 1 "$BASE/$AX_TARGET" -o "$AX_TARGET"
curl -fsSL --retry 3 --retry-delay 1 "$BASE/IERS_0012AY_CAPE_TOWN_TRANSIT_VIDEO.py" -o "$AY_TARGET"

python -m py_compile "$AJ_TARGET" "$AU_TARGET" "$AX_TARGET" "$AY_TARGET"
printf 'Downloaded from IERS repository: %s\n' "$AY_TARGET"
printf 'Version: IERS-0012AY\n'
printf 'Animation: Cape Town JPL Venus transit, one-minute cadence\n'
printf 'Image: uploaded second solar image, limb-fitted and registered to R_sun=1\n'
printf 'Video: 1280x720 MP4, contact holds, UTC clock, progress panel\n'
printf 'Rendering: Python/OpenCV/Matplotlib only; no AI images\n'
printf 'Outputs: inline MP4 + Drive frame CSV\n'
