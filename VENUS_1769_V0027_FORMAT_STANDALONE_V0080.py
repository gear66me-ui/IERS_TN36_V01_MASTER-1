# V0080
# Audit reference: Plot-only correction from verified V0079; moves PV labels farther up and Vardø labels farther down; geometry unchanged.
from __future__ import annotations

import re
import urllib.request

VERSION = "V0080"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "VENUS_1769_V0027_FORMAT_STANDALONE_V0079.py"
)

MAIN_PV_LABEL_Y = 36.0
MAIN_VARDO_LABEL_Y = -36.0
ZOOM_PV_LABEL_Y = 21.0
ZOOM_VARDO_LABEL_Y = -21.0


def fetch_source() -> str:
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": VERSION})
    with urllib.request.urlopen(request, timeout=60) as response:
        source = response.read().decode("utf-8")
    if "# V0079" not in source or "MAIN_PV_LABEL_Y" not in source:
        raise RuntimeError("Verified V0079 source was not loaded correctly.")
    return source


def promote_to_v0080(source: str) -> str:
    replacements = {
        r"MAIN_PV_LABEL_Y\s*=\s*-?[0-9]+(?:\.[0-9]+)?": f"MAIN_PV_LABEL_Y = {MAIN_PV_LABEL_Y:.1f}",
        r"MAIN_VARDO_LABEL_Y\s*=\s*-?[0-9]+(?:\.[0-9]+)?": f"MAIN_VARDO_LABEL_Y = {MAIN_VARDO_LABEL_Y:.1f}",
        r"ZOOM_PV_LABEL_Y\s*=\s*-?[0-9]+(?:\.[0-9]+)?": f"ZOOM_PV_LABEL_Y = {ZOOM_PV_LABEL_Y:.1f}",
        r"ZOOM_VARDO_LABEL_Y\s*=\s*-?[0-9]+(?:\.[0-9]+)?": f"ZOOM_VARDO_LABEL_Y = {ZOOM_VARDO_LABEL_Y:.1f}",
    }
    for pattern, replacement in replacements.items():
        source, count = re.subn(pattern, replacement, source, count=1)
        if count != 1:
            raise RuntimeError(f"Label-offset constant was not patched: {pattern}")
    source = source.replace("V0079", VERSION)
    source = source.replace(
        "# Audit reference: Plot-only correction from verified V0067; redder stronger solar fill, track-colored Venus paint outlines, corrected label offsets, delta track-angle row, geometry unchanged.",
        "# Audit reference: Plot-only correction from verified V0079; moves PV labels farther up and Vardø labels farther down; geometry unchanged.",
    )
    source = source.replace(
        "VENUS_1769_V0027_FORMAT_STANDALONE_V0079.py",
        "VENUS_1769_V0027_FORMAT_STANDALONE_V0080.py",
    )
    return source


def main() -> None:
    source = promote_to_v0080(fetch_source())
    namespace = {"__name__": "__main__", "__file__": "VENUS_1769_V0027_FORMAT_STANDALONE_V0080.py"}
    exec(compile(source, "VENUS_1769_V0027_FORMAT_STANDALONE_V0080.py", "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
# V0080
