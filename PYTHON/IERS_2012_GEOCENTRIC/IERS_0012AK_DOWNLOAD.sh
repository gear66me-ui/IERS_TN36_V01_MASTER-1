#!/usr/bin/env bash
set -euo pipefail

BASE="https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/PYTHON/IERS_2012_GEOCENTRIC"
AJ_TARGET="IERS_0012AJ_CAPE_TOWN_SITE_COORD_VS_VIDEO.py"
AK_TARGET="${1:-IERS_0012AK_CAPE_TOWN_FLIPPED_AA_FORMAT.py}"
EXPECTED_AJ_SHA256="c5aefcc1123663b5e6331efc9402459a328af44434eb7189e8de7469f6415f53"
EXPECTED_AK_SHA256="2df9bed2f930cdf63464a3dde7267b93b217b7af836c7da91bd80a24a75c0d19"

curl -fsSL --retry 3 --retry-delay 1 \
  "$BASE/$AJ_TARGET" \
  -o "$AJ_TARGET"

curl -fsSL --retry 3 --retry-delay 1 \
  "$BASE/IERS_0012AK_CAPE_TOWN_FLIPPED_AA_FORMAT.py" \
  -o "$AK_TARGET"

ACTUAL_AJ_SHA256="$(sha256sum "$AJ_TARGET" | awk '{print $1}')"
ACTUAL_AK_SHA256="$(sha256sum "$AK_TARGET" | awk '{print $1}')"

if [[ "$ACTUAL_AJ_SHA256" != "$EXPECTED_AJ_SHA256" ]]; then
  echo "IERS-0012AJ SHA-256 verification failed." >&2
  echo "Expected: $EXPECTED_AJ_SHA256" >&2
  echo "Actual  : $ACTUAL_AJ_SHA256" >&2
  rm -f "$AJ_TARGET" "$AK_TARGET"
  exit 1
fi

if [[ "$ACTUAL_AK_SHA256" != "$EXPECTED_AK_SHA256" ]]; then
  echo "IERS-0012AK SHA-256 verification failed." >&2
  echo "Expected: $EXPECTED_AK_SHA256" >&2
  echo "Actual  : $ACTUAL_AK_SHA256" >&2
  rm -f "$AK_TARGET"
  exit 1
fi

python -m py_compile "$AJ_TARGET" "$AK_TARGET"
printf 'Downloaded from IERS repository: %s\n' "$AK_TARGET"
printf 'Version: IERS-0012AK\n'
printf 'Display: 180-degree rotation; C1 left; C4 lower-right\n'
printf 'Format: IERS-0012AA color-coded in-plot statistics table\n'
printf 'Outputs: inline Matplotlib PNG + Drive CSV\n'
printf 'SHA-256: %s\n' "$ACTUAL_AK_SHA256"
