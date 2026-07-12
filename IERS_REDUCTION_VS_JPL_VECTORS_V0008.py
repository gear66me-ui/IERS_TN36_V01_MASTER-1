# V0008
# Audit reference: Full rerun of the original IERS-0012N Tahiti–Vardø JPL geometry with exact IAU-1976 reduction constants.
from __future__ import annotations

import py_compile
import runpy
from pathlib import Path
from urllib.request import Request, urlopen

VERSION = "V0008"
PROGRAM = "IERS_REDUCTION_VS_JPL_VECTORS_V0008.py"
ROOT = Path("/content")
SOURCE_PATH = ROOT / "IERS_0012N_SOURCE_V0008.py"
PATCHED_PATH = ROOT / "IERS_0012N_TAHITI_VARDO_IAU1976_RECALC_V0008.py"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "IERS_0012N_VARDO_POINT_VENUS_ENGINEERING_TRACK_PLOT_PI_SUN.py?v=8"
)


def download_source() -> str:
    request = Request(SOURCE_URL, headers={"User-Agent": PROGRAM})
    with urlopen(request, timeout=180) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError("The IERS-0012N source download was empty.")
    SOURCE_PATH.write_bytes(payload)
    return payload.decode("utf-8")


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Patch audit failed for {label}: expected exactly one occurrence, found {count}."
        )
    return text.replace(old, new, 1)


def build_patched_engine(source: str) -> str:
    patched = source
    patched = patched.replace("IERS-0012N", "V0008")
    patched = replace_exact(
        patched,
        'PROGRAM_NAME = "IERS_0012N_VARDO_POINT_VENUS_ENGINEERING_TRACK_PLOT_PI_SUN.py"',
        'PROGRAM_NAME = "IERS_0012N_TAHITI_VARDO_IAU1976_RECALC_V0008.py"',
        "program name",
    )
    patched = replace_exact(
        patched,
        "AU_KM = 149_597_870.7",
        (
            "AU_KM = 149_597_870.7\n"
            "C_KM_S = 299_792.458\n"
            "TAU_A_S = 499.004782\n"
            "IAU1976_REDUCTION_AU_KM = C_KM_S * TAU_A_S"
        ),
        "IAU-1976 astronomical unit insertion",
    )
    patched = replace_exact(
        patched,
        "EARTH_RADIUS_KM = 6_378.137",
        "EARTH_RADIUS_KM = 6_378.140",
        "IAU-1976 Earth radius",
    )
    patched = replace_exact(
        patched,
        "pi_sun = raw_phi * (es / AU_KM)",
        "pi_sun = raw_phi * (es / IAU1976_REDUCTION_AU_KM)",
        "IAU-1976 parallax normalization",
    )
    patched = replace_exact(
        patched,
        '"D_ES_AU": es / AU_KM',
        '"D_ES_AU": es / IAU1976_REDUCTION_AU_KM',
        "IAU-1976 Earth-Sun normalization display",
    )
    patched = replace_exact(
        patched,
        '"D_ES_source": "|GEOCENTER_SUN| / AU_KM"',
        '"D_ES_source": "|GEOCENTER_SUN| / (c tau_A)"',
        "IAU-1976 source label",
    )
    patched = patched.replace(
        "A′B′ = solar-screen chord; AB = projected baseline; D ES is JPL |Sun|/AU.",
        "A′B′ = solar-screen chord; AB = projected baseline; D ES is JPL |Sun|/(c tau_A).",
    )
    return patched


def main() -> None:
    source = download_source()
    patched = build_patched_engine(source)
    PATCHED_PATH.write_text(patched, encoding="utf-8")
    py_compile.compile(str(PATCHED_PATH), doraise=True)

    print(f"CODE OUTPUT: {VERSION}")
    print("CODE INPUTS")
    print(f"Program: {PROGRAM}")
    print("Source engine: original IERS-0012N from GitHub")
    print("COMMENTS")
    print("All geocenter, Tahiti, and Vardø JPL Horizons vector series are downloaded and recalculated from scratch.")
    print("Only the reduction convention is changed: a = 6378.140 km and A = c tau_A.")
    print("RESULTS")
    print("Executing full Tahiti–Vardø geometry and parallax engine now.")
    print("OUTPUT SUMMARY")
    print(f"Patched full engine: {PATCHED_PATH}")
    print("PAPER COMPARISON")
    print("Reference target: IAU-1976 exact solar horizontal parallax 8.794148 arcsec.")
    print("EQUATION STATUS")
    print("Patch audit and compile check: PASS")

    runpy.run_path(str(PATCHED_PATH), run_name="__main__")


if __name__ == "__main__":
    main()
# V0008
