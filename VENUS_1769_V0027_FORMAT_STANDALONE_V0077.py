# V0077
# Audit reference: Plot-only correction from verified V0076; Venus disk paint outlines retained, zoom disks outlined, derivation-table last row changed from average track angle to delta track angle, geometry unchanged.
from __future__ import annotations

import re
import urllib.request

VERSION = "V0077"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "VENUS_1769_V0027_FORMAT_STANDALONE_V0076.py"
)


def fetch_source() -> str:
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": VERSION})
    with urllib.request.urlopen(request, timeout=60) as response:
        source = response.read().decode("utf-8")
    if "# V0076" not in source or "def build_v0076_source()" not in source:
        raise RuntimeError("Verified V0076 source was not loaded correctly.")
    return source


def inject_delta_track_angle_patch(source: str) -> str:
    insertion_point = "\n\ndef build_v0076_source() -> str:\n"
    patch_function = r'''

def patch_delta_track_angle_table(source: str) -> str:
    target = ''' + "'''" + r'''    rows = [
        ["Quantity", "Definition", "Arcseconds", "Kilometers"],
        ["A′B′", "JPL separate-ray derived", f"{float(geometry['A_prime_B_prime_arcsec']):.6f}", f"{float(geometry['A_prime_B_prime_km']):,.6f}"],
        ["AB", "JPL projected baseline", f"{float(geometry['AB_arcsec']):.6f}", f"{float(geometry['AB_km']):,.6f}"],
        ["α PV", "Point Venus, Tahiti track angle (degrees)", f"{point_angle:.6f}°", ""],
        ["α V", "Vardo, Norway track angle (degrees)", f"{vardo_angle:.6f}°", ""],
        ["ᾱ", "Average track angle (degrees)", f"{average_angle:.6f}°", ""],
    ]
''' + "'''" + r'''
    replacement = ''' + "'''" + r'''    delta_angle = abs(vardo_angle - point_angle)
    rows = [
        ["Quantity", "Definition", "Arcseconds", "Kilometers"],
        ["A′B′", "JPL separate-ray derived", f"{float(geometry['A_prime_B_prime_arcsec']):.6f}", f"{float(geometry['A_prime_B_prime_km']):,.6f}"],
        ["AB", "JPL projected baseline", f"{float(geometry['AB_arcsec']):.6f}", f"{float(geometry['AB_km']):,.6f}"],
        ["α PV", "Point Venus, Tahiti track angle (degrees)", f"{point_angle:.6f}°", ""],
        ["α V", "Vardo, Norway track angle (degrees)", f"{vardo_angle:.6f}°", ""],
        ["Δα", "Delta track angle, |αV − αPV| (degrees)", f"{delta_angle:.6f}°", ""],
    ]
''' + "'''" + r'''
    if target not in source:
        if "Delta track angle, |αV − αPV|" in source:
            return source
        raise RuntimeError("Derivation-table average-angle row was not found; no unsafe partial patch applied.")
    return source.replace(target, replacement, 1)
'''
    if patch_function.strip() not in source:
        if insertion_point not in source:
            raise RuntimeError("V0076 build insertion point was not found.")
        source = source.replace(insertion_point, patch_function + insertion_point, 1)

    old_call = "    source = patch_venus_disk_paint_lines(source)\n"
    new_call = "    source = patch_venus_disk_paint_lines(source)\n    source = patch_delta_track_angle_table(source)\n"
    if new_call not in source:
        if old_call not in source:
            raise RuntimeError("V0076 Venus disk patch call was not found.")
        source = source.replace(old_call, new_call, 1)
    return source


def promote_to_v0077(source: str) -> str:
    source = inject_delta_track_angle_patch(source)
    source = source.replace("V0076", VERSION)
    source = source.replace(
        "# Audit reference: Plot-only correction from verified V0067; warm solar fill/limb plus opaque track-colored paint lines on every Venus disk, geometry unchanged.",
        "# Audit reference: Plot-only correction from verified V0076; Venus disk paint outlines retained, zoom disks outlined, derivation-table last row changed from average track angle to delta track angle, geometry unchanged.",
    )
    source = source.replace(
        "VENUS_1769_V0027_FORMAT_STANDALONE_V0076.py",
        "VENUS_1769_V0027_FORMAT_STANDALONE_V0077.py",
    )
    return source


def main() -> None:
    source = promote_to_v0077(fetch_source())
    namespace = {"__name__": "__main__", "__file__": "VENUS_1769_V0027_FORMAT_STANDALONE_V0077.py"}
    exec(compile(source, "VENUS_1769_V0027_FORMAT_STANDALONE_V0077.py", "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
# V0077
