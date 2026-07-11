#!/usr/bin/env bash
set -euo pipefail

BASE="https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/PYTHON/IERS_2012_GEOCENTRIC"
AJ_TARGET="IERS_0012AJ_CAPE_TOWN_SITE_COORD_VS_VIDEO.py"
AK_TARGET="IERS_0012AK_CAPE_TOWN_FLIPPED_AA_FORMAT.py"
AL_TARGET="${1:-IERS_0012AL_CAPE_TOWN_TRUE_HALF_SUN.py}"

curl -fsSL --retry 3 --retry-delay 1 "$BASE/$AJ_TARGET" -o "$AJ_TARGET"
curl -fsSL --retry 3 --retry-delay 1 "$BASE/$AK_TARGET" -o "$AK_TARGET"
curl -fsSL --retry 3 --retry-delay 1 "$BASE/IERS_0012AL_CAPE_TOWN_TRUE_HALF_SUN.py" -o "$AL_TARGET"

python -m py_compile "$AJ_TARGET" "$AK_TARGET" "$AL_TARGET"
printf 'Downloaded from IERS repository: %s\n' "$AL_TARGET"
printf 'Version: IERS-0012AL\n'
printf 'Display: true lower half-Sun; C1 left; C4 lower-right\n'
printf 'Format: compact IERS-0012AA color-coded table inside plot\n'
printf 'Outputs: inline Matplotlib PNG + Drive CSV\n'
