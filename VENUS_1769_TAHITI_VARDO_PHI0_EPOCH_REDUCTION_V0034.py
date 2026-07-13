# V0034
# Audit reference: Lock all Tahiti–Vardø 1769 Venus reductions to the project φ0 closest-approach epoch.
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

VERSION = "V0034"
ROOT = Path("/content")
FULL_PATH = ROOT / "VENUS_1769_TAHITI_VARDO_PHI0_EPOCH_REDUCTION_V0034_FULL.py"
SOURCE_COMMIT = "50874991edc928a67fca708d9f1a7a47e7b6aa2f"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/"
    f"{SOURCE_COMMIT}/VENUS_1769_TAHITI_VARDO_CLEAN_REDUCTION_V0033.py"
)
PHI0_UTC = "1769-06-03 22:19:15.599"


def fetch_source() -> str:
    request = urllib.request.Request(
        f"{SOURCE_URL}?cache={time.time_ns()}",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        source = response.read().decode("utf-8")
    if not source.startswith("# V0033\n"):
        raise RuntimeError("Pinned V0033 source header audit failed.")
    if not source.rstrip().endswith("# V0033"):
        raise RuntimeError("Pinned V0033 source footer audit failed.")
    return source


def build_v0034(source: str) -> str:
    source = source.replace("# V0033", "# V0034")
    source = source.replace(
        "# Audit reference: Clean Tahiti–Vardø 1769 Venus report with individual A′, B′, A, B coordinates and explicit classical/exact reductions.",
        "# Audit reference: Clean Tahiti–Vardø 1769 Venus report locked to the project φ0 closest-approach epoch.",
    )
    source = source.replace('VERSION = "V0033"', 'VERSION = "V0034"')
    source = source.replace(
        'VENUS_1769_TAHITI_VARDO_CLEAN_REDUCTION_V0033_OUTPUT',
        'VENUS_1769_TAHITI_VARDO_PHI0_EPOCH_REDUCTION_V0034_OUTPUT',
    )
    source = source.replace(
        'VENUS_1769_TAHITI_VARDO_CLEAN_REDUCTION_V0033.csv',
        'VENUS_1769_TAHITI_VARDO_PHI0_EPOCH_REDUCTION_V0034.csv',
    )
    source = source.replace(
        'VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0033.csv',
        'VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0034.csv',
    )
    old_epoch = '    jd_reference = float(reference_epoch(cache))\n'
    new_epoch = (
        '    jd_reference = float(base["Time"]('
        f'"{PHI0_UTC}", format="iso", scale="utc").tdb.jd)\n'
    )
    if old_epoch not in source:
        raise RuntimeError("V0033 epoch assignment was not found.")
    source = source.replace(old_epoch, new_epoch, 1)
    old_inputs = '            ["Reference UTC", utc_reference + " UTC"],\n'
    new_inputs = (
        '            ["Epoch source", "PROJECT φ0 CLOSEST APPROACH"],\n'
        '            ["Reference UTC", utc_reference + " UTC"],\n'
    )
    if old_inputs not in source:
        raise RuntimeError("V0033 input table insertion point was not found.")
    source = source.replace(old_inputs, new_inputs, 1)
    old_comment = (
        '    print("A′, B′, A, and B are midpoint-centered reporting coordinates; each separation is unchanged.")\n'
    )
    new_comment = (
        '    print("All JPL vectors, distances, A′B′, AB, and reduction factors use the identical project φ0 epoch.")\n'
        '    print("A′, B′, A, and B are midpoint-centered reporting coordinates; each separation is unchanged.")\n'
    )
    if old_comment not in source:
        raise RuntimeError("V0033 comment insertion point was not found.")
    source = source.replace(old_comment, new_comment, 1)
    compile(source, str(FULL_PATH), "exec")
    return source


def main() -> None:
    source = build_v0034(fetch_source())
    FULL_PATH.write_text(source, encoding="utf-8")
    namespace = {
        "__name__": "__main__",
        "__file__": str(FULL_PATH),
    }
    exec(compile(source, str(FULL_PATH), "exec"), namespace)


if __name__ == "__main__":
    main()
# V0034
