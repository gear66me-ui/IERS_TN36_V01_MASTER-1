# V0013
# Audit reference: Exact IERS-0012N visual replica with only the IAU-1976 constants and parallax equations changed.
from __future__ import annotations

import py_compile
import runpy
from pathlib import Path
from urllib.request import Request, urlopen

VERSION = "V0013"
PROGRAM = "IERS_0012N_EXACT_REPLICA_IAU1976_V0013.py"
ROOT = Path("/content")
ORIGINAL = ROOT / "IERS_0012N_ORIGINAL_V0013.py"
PATCHED = ROOT / "IERS_0012N_IAU1976_EXACT_REPLICA_V0013.py"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "IERS_0012N_VARDO_POINT_VENUS_ENGINEERING_TRACK_PLOT_PI_SUN.py?v=13"
)


def download_original() -> str:
    request = Request(SOURCE_URL, headers={"User-Agent": PROGRAM})
    with urlopen(request, timeout=180) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError("The original IERS-0012N source download was empty.")
    ORIGINAL.write_bytes(payload)
    return payload.decode("utf-8")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Patch audit failed for {label}: expected one occurrence, found {count}."
        )
    return text.replace(old, new, 1)


def build_exact_replica(source: str) -> str:
    patched = source

    patched = replace_once(
        patched,
        '# IERS-0012N\n# Audit reference: GitHubDelivery@IERS-0012N; 1769 Vardo-Point Venus engineering half-Sun plot using JPL Horizons SITE_COORD vectors.',
        '# V0013\n# Audit reference: Exact IERS-0012N visual replica using JPL vectors and IAU-1976 normalization.',
        "header",
    )
    patched = replace_once(
        patched,
        'VERSION = "IERS-0012N"',
        'VERSION = "V0013"',
        "version",
    )
    patched = replace_once(
        patched,
        'PROGRAM_NAME = "IERS_0012N_VARDO_POINT_VENUS_ENGINEERING_TRACK_PLOT_PI_SUN.py"',
        'PROGRAM_NAME = "IERS_0012N_IAU1976_EXACT_REPLICA_V0013.py"',
        "program name",
    )

    constant_block_old = '''AU_KM = 149_597_870.7
ARCSEC_PER_RAD = 206_264.80624709636
EARTH_RADIUS_KM = 6_378.137
SUN_RADIUS_KM = 695_700.0
VENUS_RADIUS_KM = 6_051.8
PI_SUN_REFERENCE_ARCSEC = 8.794148'''
    constant_block_new = '''AU_KM = 149_597_870.7
ARCSEC_PER_RAD = 206_264.80624709636
EARTH_RADIUS_KM = 6_378.140
C_KM_S = 299_792.458
TAU_A_S = 499.004782
IAU1976_AU_KM = C_KM_S * TAU_A_S
SUN_RADIUS_KM = 695_700.0
VENUS_RADIUS_KM = 6_051.8
PI_SUN_REFERENCE_ARCSEC = math.asin(EARTH_RADIUS_KM / IAU1976_AU_KM) * ARCSEC_PER_RAD'''
    patched = replace_once(
        patched,
        constant_block_old,
        constant_block_new,
        "IAU-1976 constants",
    )

    formula_old = '''    abp_km = math.tan(abp_arcsec / ARCSEC_PER_RAD) * es
    ab_km = abp_km * ev / vs
    ab_arcsec = math.atan2(ab_km, es) * ARCSEC_PER_RAD
    halley_ratio = abp_km / ab_km
    raw_phi = rho_arcsec * (ev / vs) * (EARTH_RADIUS_KM / ab_km)
    pi_sun = raw_phi * (es / AU_KM)
    rho_scaled = rho_arcsec * EARTH_RADIUS_KM / ab_km
    return {"aprime": aprime, "bprime": bprime, "A_prime_B_prime_arcsec": abp_arcsec, "A_prime_B_prime_km": abp_km, "rho_arcsec": rho_arcsec, "rho_scaled_arcsec": rho_scaled, "AB_arcsec": ab_arcsec, "AB_km": ab_km, "halley_ratio": halley_ratio, "raw_phi_arcsec": raw_phi, "pi_sun_arcsec": pi_sun, "pi_sun_residual_arcsec": pi_sun - PI_SUN_REFERENCE_ARCSEC, "pi_sun_residual_percent": 100.0 * (pi_sun - PI_SUN_REFERENCE_ARCSEC) / PI_SUN_REFERENCE_ARCSEC, "pi_sun_reference_arcsec": PI_SUN_REFERENCE_ARCSEC, "D_ES_AU": es / AU_KM, "D_EV_D_VS": ev / vs, "D_VS_D_EV": vs / ev, "D_ES_source": "|GEOCENTER_SUN| / AU_KM"}'''

    formula_new = '''    theta_rad = abp_arcsec / ARCSEC_PER_RAD
    abp_km = math.tan(theta_rad) * IAU1976_AU_KM
    ab_km = abp_km * ev / vs
    ab_arcsec = math.atan2(ab_km, IAU1976_AU_KM) * ARCSEC_PER_RAD
    halley_ratio = abp_km / ab_km
    raw_phi = rho_arcsec * (ev / vs) * (EARTH_RADIUS_KM / ab_km)
    chord_normal_factor = abp_arcsec / rho_arcsec
    tangent_factor = math.tan(theta_rad) / theta_rad
    horizontal_ratio = EARTH_RADIUS_KM / IAU1976_AU_KM
    exact_arcsine_factor = math.asin(horizontal_ratio) / horizontal_ratio
    pi_sun = raw_phi * chord_normal_factor * tangent_factor * exact_arcsine_factor
    rho_scaled = rho_arcsec * EARTH_RADIUS_KM / ab_km
    return {"aprime": aprime, "bprime": bprime, "A_prime_B_prime_arcsec": abp_arcsec, "A_prime_B_prime_km": abp_km, "rho_arcsec": rho_arcsec, "rho_scaled_arcsec": rho_scaled, "AB_arcsec": ab_arcsec, "AB_km": ab_km, "halley_ratio": halley_ratio, "raw_phi_arcsec": raw_phi, "pi_sun_arcsec": pi_sun, "pi_sun_residual_arcsec": pi_sun - PI_SUN_REFERENCE_ARCSEC, "pi_sun_residual_percent": 100.0 * (pi_sun - PI_SUN_REFERENCE_ARCSEC) / PI_SUN_REFERENCE_ARCSEC, "pi_sun_reference_arcsec": PI_SUN_REFERENCE_ARCSEC, "D_ES_AU": 1.0, "D_EV_D_VS": ev / vs, "D_VS_D_EV": vs / ev, "D_ES_source": "IAU 1976 c tau_A / c tau_A"}'''
    patched = replace_once(
        patched,
        formula_old,
        formula_new,
        "parallax equation",
    )

    patched = replace_once(
        patched,
        'A′B′ = solar-screen chord; AB = projected baseline; D ES is JPL |Sun|/AU.',
        'A′B′ = solar-screen chord; AB = projected baseline; D ES uses IAU 1976 cτA.',
        "plot note",
    )

    patched = replace_once(
        patched,
        '# IERS-0012N',
        '# V0013',
        "final version marker",
    )

    return patched


def main() -> None:
    source = download_original()
    patched = build_exact_replica(source)
    PATCHED.write_text(patched, encoding="utf-8")
    py_compile.compile(str(PATCHED), doraise=True)
    runpy.run_path(str(PATCHED), run_name="__main__")


if __name__ == "__main__":
    main()
# V0013
