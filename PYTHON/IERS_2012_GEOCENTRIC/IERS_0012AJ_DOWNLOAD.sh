#!/usr/bin/env bash
set -euo pipefail

BASE="https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/PYTHON/IERS_2012_GEOCENTRIC"
TARGET="${1:-IERS_0012AJ_CAPE_TOWN_SITE_COORD_VS_VIDEO.py}"
EXPECTED_SHA256="c5aefcc1123663b5e6331efc9402459a328af44434eb7189e8de7469f6415f53"

curl -fsSL --retry 3 --retry-delay 1 \
  "$BASE/IERS_0012AJ_CAPE_TOWN_SITE_COORD_VS_VIDEO.py" \
  -o "$TARGET"

ACTUAL_SHA256="$(sha256sum "$TARGET" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "IERS-0012AJ SHA-256 verification failed." >&2
  echo "Expected: $EXPECTED_SHA256" >&2
  echo "Actual  : $ACTUAL_SHA256" >&2
  rm -f "$TARGET"
  exit 1
fi

python -m py_compile "$TARGET"
printf 'Downloaded from IERS repository: %s\n' "$TARGET"
printf 'Version: IERS-0012AJ\n'
printf 'Plot: Cape Town SITE_COORD JPL + dashed SDO video TLS reference\n'
printf 'Outputs: inline half-Sun PNG + Drive CSV\n'
printf 'SHA-256: %s\n' "$ACTUAL_SHA256"
