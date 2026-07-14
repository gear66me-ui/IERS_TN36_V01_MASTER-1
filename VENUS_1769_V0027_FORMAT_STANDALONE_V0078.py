# V0078
# Audit reference: Plot-only correction from verified V0077; redder solar fill, stronger opacity, PV labels up, Vardø labels down, geometry unchanged.
from __future__ import annotations

import re
import urllib.request

VERSION = "V0078"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "VENUS_1769_V0027_FORMAT_STANDALONE_V0077.py"
)
SOLAR_FILL_COLOR_NEW = "#D95A1B"
SOLAR_FILL_ALPHA_NEW = 0.260
MAIN_PV_LABEL_Y = 28.0
MAIN_VARDO_LABEL_Y = -28.0
ZOOM_PV_LABEL_Y = 15.5
ZOOM_VARDO_LABEL_Y = -15.5


def fetch_source() -> str:
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": VERSION})
    with urllib.request.urlopen(request, timeout=60) as response:
        source = response.read().decode("utf-8")
    if "# V0077" not in source or "def promote_to_v0077(" not in source:
        raise RuntimeError("Verified V0077 source was not loaded correctly.")
    return source


def patch_solar_fill_tone(source: str) -> str:
    source = re.sub(
        r'SOLAR_FILL_COLOR\s*=\s*"#[0-9A-Fa-f]{6}"',
        f'SOLAR_FILL_COLOR = "{SOLAR_FILL_COLOR_NEW}"',
        source,
        count=1,
    )
    source = re.sub(
        r'SOLAR_FILL_ALPHA\s*=\s*[0-9]+(?:\.[0-9]+)?',
        f'SOLAR_FILL_ALPHA = {SOLAR_FILL_ALPHA_NEW:.3f}',
        source,
        count=1,
    )
    return source


def inject_label_offset_patch(source: str) -> str:
    insertion_point = "\n\ndef promote_to_v0077(source: str) -> str:\n"
    patch_function = r'''

def patch_contact_label_offsets(source: str) -> str:
    target = ''' + "'''" + r'''        if main:
            above = short == "PV"
            y_shift = 20.0 if above else -20.0
        else:
            above = short == "PV"
            y_shift = 11.0 if above else -11.0
''' + "'''" + r'''
    replacement = ''' + "'''" + f'''        if main:
            above = short == "PV"
            y_shift = {MAIN_PV_LABEL_Y:.1f} if above else {MAIN_VARDO_LABEL_Y:.1f}
        else:
            above = short == "PV"
            y_shift = {ZOOM_PV_LABEL_Y:.1f} if above else {ZOOM_VARDO_LABEL_Y:.1f}
''' + "'''" + r'''
    if target not in source:
        if "y_shift = 28.0 if above else -28.0" in source:
            return source
        raise RuntimeError("Contact-label y-shift block was not found; no unsafe partial patch applied.")
    return source.replace(target, replacement, 1)
'''
    if patch_function.strip() not in source:
        if insertion_point not in source:
            raise RuntimeError("V0077 promote insertion point was not found.")
        source = source.replace(insertion_point, patch_function + insertion_point, 1)
    old_call = "    source = inject_delta_track_angle_patch(source)\n"
    new_call = "    source = inject_delta_track_angle_patch(source)\n    source = patch_contact_label_offsets(source)\n"
    if new_call not in source:
        if old_call not in source:
            raise RuntimeError("V0077 delta patch call was not found.")
        source = source.replace(old_call, new_call, 1)
    return source


def promote_to_v0078(source: str) -> str:
    source = patch_solar_fill_tone(source)
    source = inject_label_offset_patch(source)
    source = source.replace("V0077", VERSION)
    source = source.replace(
        "# Audit reference: Plot-only correction from verified V0076; Venus disk paint outlines retained, zoom disks outlined, derivation-table last row changed from average track angle to delta track angle, geometry unchanged.",
        "# Audit reference: Plot-only correction from verified V0077; redder solar fill, stronger opacity, PV labels up, Vardø labels down, geometry unchanged.",
    )
    source = source.replace(
        "VENUS_1769_V0027_FORMAT_STANDALONE_V0077.py",
        "VENUS_1769_V0027_FORMAT_STANDALONE_V0078.py",
    )
    return source


def main() -> None:
    source = promote_to_v0078(fetch_source())
    namespace = {"__name__": "__main__", "__file__": "VENUS_1769_V0027_FORMAT_STANDALONE_V0078.py"}
    exec(compile(source, "VENUS_1769_V0027_FORMAT_STANDALONE_V0078.py", "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
# V0078
