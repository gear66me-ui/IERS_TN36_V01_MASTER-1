# V0033
# Audit reference: Clean Tahiti–Vardø 1769 Venus report with individual A′, B′, A, B coordinates and explicit classical/exact reductions.
from __future__ import annotations

import math
import subprocess
import sys
import time
import urllib.request
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def ensure_package(import_name: str, pip_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "-q", "install", pip_name]
        )


for _import_name, _pip_name in (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("astropy", "astropy"),
    ("astroquery", "astroquery"),
):
    ensure_package(_import_name, _pip_name)

try:
    from erfa import ErfaWarning

    warnings.filterwarnings("ignore", category=ErfaWarning)
except Exception:
    warnings.filterwarnings("ignore", message=".*dubious year.*")

import numpy as np
import pandas as pd

VERSION = "V0033"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUTPUT_DIR = ROOT / "VENUS_1769_TAHITI_VARDO_CLEAN_REDUCTION_V0033_OUTPUT"
OUTPUT_CSV = OUTPUT_DIR / "VENUS_1769_TAHITI_VARDO_CLEAN_REDUCTION_V0033.csv"
MASTER_CSV = ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0033.csv"

BASE_COMMIT = "d55bc2274359a3014c19f257e42cc149bc458d57"
BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/"
    f"{BASE_COMMIT}/VENUS_1769_TAHITI_VARDO_HALLEY_THUMB_CHECK_V0031.py"
)
BASE_PATH = ROOT / "VENUS_1769_TAHITI_VARDO_HALLEY_THUMB_CHECK_V0031_BASE.py"


def load_base_namespace() -> dict[str, object]:
    request = urllib.request.Request(
        f"{BASE_URL}?cache={time.time_ns()}",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        source = response.read().decode("utf-8")
    if not source.startswith("# V0031\n") or not source.rstrip().endswith("# V0031"):
        raise RuntimeError("Pinned V0031 source-boundary audit failed.")
    compile(source, str(BASE_PATH), "exec")
    BASE_PATH.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {
        "__name__": "venus_v0031_base",
        "__file__": str(BASE_PATH),
    }
    exec(compile(source, str(BASE_PATH), "exec"), namespace)
    return namespace


def compatible_master(path: Path, required_columns: list[str]) -> bool:
    if not path.is_file():
        return False
    try:
        sample = pd.read_csv(path, nrows=3)
    except Exception:
        return False
    return all(column in sample.columns for column in required_columns)


def load_or_build_master(
    base: dict[str, object], required_columns: list[str]
) -> tuple[pd.DataFrame, str]:
    if compatible_master(MASTER_CSV, required_columns):
        return pd.read_csv(MASTER_CSV), str(MASTER_CSV)
    master = base["build_master"]()
    master.to_csv(MASTER_CSV, index=False, float_format="%.15f")
    return master, "NEW JPL HORIZONS DOWNLOAD"


def format_number(value: float) -> str:
    return f"{float(value):,.12f}"


def table_text(frame: pd.DataFrame) -> str:
    formatters = {
        column: format_number
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    }
    return frame.to_string(index=False, formatters=formatters)


def main() -> None:
    base = load_base_namespace()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prefixes = tuple(base["PREFIXES"])
    required_columns = ["JD_TDB"] + [
        f"{prefix}_{axis}_KM" for prefix in prefixes for axis in "XYZ"
    ]
    master, source = load_or_build_master(base, required_columns)

    build_cache = base["build_cache"]
    vector_at = base["vector_at"]
    reference_epoch = base["reference_epoch"]
    common_basis = base["common_basis"]
    external_contacts = base["external_contacts"]
    relative_position_arcsec = base["relative_position_arcsec"]
    fitted_direction = base["fitted_direction"]
    norm = base["norm"]
    unit = base["unit"]
    utc_text = base["utc_text"]

    arcsec_per_rad = float(base["ARCSEC_PER_RAD"])
    au_km = float(base["AU_KM"])
    earth_radius_km = float(base["EARTH_EQUATORIAL_RADIUS_KM"])

    cache = build_cache(master)
    jd_reference = float(reference_epoch(cache))
    utc_reference = str(utc_text(jd_reference))

    geocentric_sun = vector_at(cache, "GEOCENTER_SUN", jd_reference)
    geocentric_venus = vector_at(cache, "GEOCENTER_VENUS", jd_reference)
    venus_to_sun = geocentric_sun - geocentric_venus
    _center, xi, eta = common_basis(geocentric_sun)
    center = unit(geocentric_sun)

    contacts_t = external_contacts(cache, "TAHITI")
    contacts_v = external_contacts(cache, "VARDO")
    common_start = max(float(contacts_t[0]), float(contacts_v[0]))
    common_stop = min(float(contacts_t[1]), float(contacts_v[1]))
    minute_jds = np.asarray(cache["JD_TDB"], dtype=float)
    selected_jds = minute_jds[
        (minute_jds >= common_start) & (minute_jds <= common_stop)
    ]
    if len(selected_jds) < 20:
        raise RuntimeError("Too few synchronized in-transit JPL samples.")

    tahiti_track = np.array(
        [
            relative_position_arcsec(cache, "TAHITI", jd, center, xi, eta)
            for jd in selected_jds
        ],
        dtype=float,
    )
    vardo_track = np.array(
        [
            relative_position_arcsec(cache, "VARDO", jd, center, xi, eta)
            for jd in selected_jds
        ],
        dtype=float,
    )

    direction_t = fitted_direction(tahiti_track)
    direction_v = fitted_direction(vardo_track)
    if float(np.dot(direction_t, direction_v)) < 0.0:
        direction_v = -direction_v
    common_direction = unit(direction_t + direction_v)
    common_normal = np.array(
        [-common_direction[1], common_direction[0]], dtype=float
    )
    common_track_angle_deg = math.degrees(
        math.atan2(common_direction[1], common_direction[0])
    )

    apparent_t = relative_position_arcsec(
        cache, "TAHITI", jd_reference, center, xi, eta
    )
    apparent_v = relative_position_arcsec(
        cache, "VARDO", jd_reference, center, xi, eta
    )
    signed_apparent_difference = float(
        np.dot(apparent_v - apparent_t, common_normal)
    )
    if signed_apparent_difference < 0.0:
        common_normal = -common_normal
        signed_apparent_difference = -signed_apparent_difference

    aprime_bprime_arcsec = signed_apparent_difference

    topocentric_sun_t = vector_at(cache, "TAHITI_SUN", jd_reference)
    topocentric_sun_v = vector_at(cache, "VARDO_SUN", jd_reference)
    observer_t = geocentric_sun - topocentric_sun_t
    observer_v = geocentric_sun - topocentric_sun_v
    observer_baseline = observer_v - observer_t
    baseline_plane = np.array(
        [
            float(np.dot(observer_baseline, xi)),
            float(np.dot(observer_baseline, eta)),
        ],
        dtype=float,
    )
    ab_direct_km = abs(float(np.dot(baseline_plane, common_normal)))

    earth_sun_km = norm(geocentric_sun)
    earth_venus_km = norm(geocentric_venus)
    venus_sun_km = norm(venus_to_sun)

    km_per_arcsec = earth_sun_km / arcsec_per_rad
    aprime_bprime_km = aprime_bprime_arcsec * km_per_arcsec
    ab_direct_arcsec = ab_direct_km / km_per_arcsec

    a_prime_arcsec = -0.5 * aprime_bprime_arcsec
    b_prime_arcsec = +0.5 * aprime_bprime_arcsec
    a_prime_km = a_prime_arcsec * km_per_arcsec
    b_prime_km = b_prime_arcsec * km_per_arcsec

    a_arcsec = -0.5 * ab_direct_arcsec
    b_arcsec = +0.5 * ab_direct_arcsec
    a_km = -0.5 * ab_direct_km
    b_km = +0.5 * ab_direct_km

    classical_factor = earth_venus_km / venus_sun_km
    es_vs_ratio = earth_sun_km / venus_sun_km
    ev_es_ratio = earth_venus_km / earth_sun_km

    ab_classical_arcsec = aprime_bprime_arcsec * classical_factor
    ab_classical_km = aprime_bprime_km * classical_factor
    classical_residual_arcsec = ab_direct_arcsec - ab_classical_arcsec
    classical_residual_km = ab_direct_km - ab_classical_km
    classical_residual_percent = (
        100.0 * classical_residual_arcsec / ab_direct_arcsec
    )

    exact_transfer_factor = ab_direct_arcsec / aprime_bprime_arcsec
    ab_exact_arcsec = aprime_bprime_arcsec * exact_transfer_factor
    ab_exact_km = aprime_bprime_km * exact_transfer_factor
    exact_residual_arcsec = ab_direct_arcsec - ab_exact_arcsec
    exact_residual_km = ab_direct_km - ab_exact_km
    exact_residual_percent = 100.0 * exact_residual_arcsec / ab_direct_arcsec

    factor_difference = exact_transfer_factor - classical_factor
    factor_difference_percent = 100.0 * factor_difference / classical_factor

    solar_parallax_arcsec = (
        math.asin(earth_radius_km / au_km) * arcsec_per_rad
    )

    inputs_frame = pd.DataFrame(
        [
            ["JPL source", source],
            ["Reference UTC", utc_reference + " UTC"],
            ["Reference JD TDB", f"{jd_reference:.12f}"],
            ["AU conversion", f"{au_km:,.6f} km/AU"],
            ["Common track angle", f"{common_track_angle_deg:.6f} deg"],
            ["Solar horizontal parallax", f"{solar_parallax_arcsec:.12f} arcsec"],
        ],
        columns=["Input", "Value"],
    )

    points_frame = pd.DataFrame(
        [
            ["A′", "Tahiti apparent Venus", a_prime_arcsec, a_prime_km],
            ["B′", "Vardø apparent Venus", b_prime_arcsec, b_prime_km],
            ["A", "Tahiti projected observer", a_arcsec, a_km],
            ["B", "Vardø projected observer", b_arcsec, b_km],
        ],
        columns=["Point", "Definition", "Arcseconds", "Kilometers"],
    )

    separations_frame = pd.DataFrame(
        [
            ["A′B′ direct JPL", aprime_bprime_arcsec, aprime_bprime_km],
            ["AB direct JPL", ab_direct_arcsec, ab_direct_km],
        ],
        columns=["Separation", "Arcseconds", "Kilometers"],
    )

    distances_frame = pd.DataFrame(
        [
            ["Earth → Venus", "EV", earth_venus_km],
            ["Venus → Sun", "VS", venus_sun_km],
            ["Earth → Sun", "ES", earth_sun_km],
        ],
        columns=["JPL vector distance", "Symbol", "Kilometers"],
    )

    ratios_frame = pd.DataFrame(
        [
            [
                "Classical Halley factor",
                "EV / VS",
                f"{earth_venus_km:,.12f} / {venus_sun_km:,.12f}",
                classical_factor,
            ],
            [
                "Earth–Sun / Venus–Sun",
                "ES / VS",
                f"{earth_sun_km:,.12f} / {venus_sun_km:,.12f}",
                es_vs_ratio,
            ],
            [
                "Earth–Venus / Earth–Sun",
                "EV / ES",
                f"{earth_venus_km:,.12f} / {earth_sun_km:,.12f}",
                ev_es_ratio,
            ],
            [
                "Exact JPL transfer factor",
                "AB / A′B′",
                f"{ab_direct_arcsec:.12f} / {aprime_bprime_arcsec:.12f}",
                exact_transfer_factor,
            ],
        ],
        columns=["Factor", "Definition", "Calculation", "Value"],
    )

    classical_frame = pd.DataFrame(
        [
            [
                "Angular reduction",
                f"{aprime_bprime_arcsec:.12f} × {classical_factor:.12f}",
                ab_classical_arcsec,
                "arcsec",
            ],
            [
                "Linear reduction",
                f"{aprime_bprime_km:.12f} × {classical_factor:.12f}",
                ab_classical_km,
                "km",
            ],
            [
                "Direct JPL AB",
                "projected observer baseline",
                ab_direct_arcsec,
                "arcsec",
            ],
            [
                "Direct JPL AB",
                "projected observer baseline",
                ab_direct_km,
                "km",
            ],
            [
                "Classical residual",
                "direct JPL − classical Halley",
                classical_residual_arcsec,
                "arcsec",
            ],
            [
                "Classical residual",
                "direct JPL − classical Halley",
                classical_residual_km,
                "km",
            ],
            [
                "Classical residual",
                "100 × residual / direct JPL",
                classical_residual_percent,
                "%",
            ],
        ],
        columns=["Quantity", "Calculation", "Value", "Unit"],
    )

    exact_frame = pd.DataFrame(
        [
            [
                "Exact angular reduction",
                f"{aprime_bprime_arcsec:.12f} × {exact_transfer_factor:.12f}",
                ab_exact_arcsec,
                "arcsec",
            ],
            [
                "Exact linear reduction",
                f"{aprime_bprime_km:.12f} × {exact_transfer_factor:.12f}",
                ab_exact_km,
                "km",
            ],
            ["Exact residual", "direct JPL − exact", exact_residual_arcsec, "arcsec"],
            ["Exact residual", "direct JPL − exact", exact_residual_km, "km"],
            ["Exact residual", "100 × residual / direct JPL", exact_residual_percent, "%"],
            ["Factor difference", "exact − classical", factor_difference, "ratio"],
            [
                "Factor difference",
                "100 × (exact − classical) / classical",
                factor_difference_percent,
                "%",
            ],
        ],
        columns=["Quantity", "Calculation", "Value", "Unit"],
    )

    audit_frames = [
        ("POINTS", points_frame),
        ("SEPARATIONS", separations_frame),
        ("DISTANCES", distances_frame),
        ("RATIOS", ratios_frame),
        ("CLASSICAL", classical_frame),
        ("EXACT", exact_frame),
    ]
    audit_rows: list[dict[str, object]] = []
    for section, frame in audit_frames:
        for row_index, row in frame.iterrows():
            record: dict[str, object] = {
                "section": section,
                "row": int(row_index),
            }
            record.update({str(key): value for key, value in row.items()})
            audit_rows.append(record)
    pd.DataFrame(audit_rows).to_csv(
        OUTPUT_CSV, index=False, float_format="%.15f"
    )

    print("CODE INPUTS")
    print(inputs_frame.to_string(index=False))
    print()

    print("COMMENTS")
    print("A′, B′, A, and B are midpoint-centered reporting coordinates; each separation is unchanged.")
    print("A′B′ comes from JPL topocentric Venus-minus-Sun vectors in one common tangent basis and common normal.")
    print("AB comes from the JPL Tahiti–Vardø observer baseline projected onto that same common normal.")
    print()

    print("RESULTS")
    print("JPL POINT COORDINATES")
    print(table_text(points_frame))
    print()
    print("JPL SEPARATIONS")
    print(table_text(separations_frame))
    print()
    print("JPL VECTOR DISTANCES")
    print(table_text(distances_frame))
    print()
    print("DISTANCE AND TRANSFER FACTORS")
    print(table_text(ratios_frame))
    print()
    print("CLASSICAL HALLEY REDUCTION")
    print(table_text(classical_frame))
    print()
    print("EXACT JPL VECTOR CLOSURE")
    print(table_text(exact_frame))
    print()

    print("OUTPUT SUMMARY")
    print(str(OUTPUT_CSV))
    print(str(MASTER_CSV))
    print()

    print("PAPER COMPARISON")
    comparison_frame = pd.DataFrame(
        [
            ["AB direct JPL", ab_direct_arcsec, ab_direct_km],
            ["AB classical Halley", ab_classical_arcsec, ab_classical_km],
            ["AB exact JPL reduction", ab_exact_arcsec, ab_exact_km],
            ["Classical minus direct", -classical_residual_arcsec, -classical_residual_km],
            ["Exact minus direct", -exact_residual_arcsec, -exact_residual_km],
        ],
        columns=["Result", "Arcseconds", "Kilometers"],
    )
    print(table_text(comparison_frame))
    print()

    print("EQUATION STATUS")
    status_frame = pd.DataFrame(
        [
            ["A′B′ × EV/VS = AB", "CLASSICAL APPROXIMATION", classical_residual_arcsec],
            ["A′B′ × exact JPL factor = AB", "PASS", exact_residual_arcsec],
        ],
        columns=["Equation", "Status", "Residual arcsec"],
    )
    print(table_text(status_frame))
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0033
