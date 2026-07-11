#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/PYTHON/NASA_SDO_2012_VIDEO_EXTRACTION/V0005"
OUTPUT_FILE="${1:-NASA_SDO_2012_TRACK_EXTRACT.py}"
TEMP_FILE="${OUTPUT_FILE}.tmp"
EXPECTED_SHA256="e548e71e71652fb7e8b11d8d022540f7e4dd0a16ce8e69996cfe2339ce54ef46"

: > "${TEMP_FILE}"
for PART in 00 01 02 03 04; do
    curl -fsSL --retry 3 --retry-delay 1 \
        "${BASE_URL}/NASA_SDO_2012_TRACK_EXTRACT_V0005_PART_${PART}.py" \
        >> "${TEMP_FILE}"
done

ACTUAL_SHA256="$(sha256sum "${TEMP_FILE}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
    echo "V0005 SHA-256 verification failed." >&2
    echo "Expected: ${EXPECTED_SHA256}" >&2
    echo "Actual  : ${ACTUAL_SHA256}" >&2
    rm -f "${TEMP_FILE}"
    exit 1
fi

mv "${TEMP_FILE}" "${OUTPUT_FILE}"
python -m py_compile "${OUTPUT_FILE}"
printf 'Downloaded from IERS repository: %s\n' "${OUTPUT_FILE}"
printf 'Version: V0005\n'
printf 'SHA-256: %s\n' "${ACTUAL_SHA256}"
