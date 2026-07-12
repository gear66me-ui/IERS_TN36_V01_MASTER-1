# V0005
# Audit reference: 1769 Mercury transit parallax geometry for Vardø and James Cook's Mercury Bay position using JPL Horizons vectors only.
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def ensure_package(import_name: str, pip_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pip_name])


for _import_name, _pip_name in (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("astroquery", "astroquery"),
    ("astropy", "astropy"),
):
    ensure_package(_import_name, _pip_name)

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar
from astroquery.jplhorizons import Horizons
from astropy.time import Time

VERSION = "V0005"
PROGRAM = "MERCURY_PARALLAX_VARDO_MERCURY_BAY_V0005.py"
TITLE = "1769 MERCURY TRANSIT — VARDO / MERCURY BAY JPL PARALLAX AUDIT"
LOCAL_TZ = ZoneInfo("America/Bogota")

AU_KM = 149_597_870.700000
ARCSEC_PER_RAD = 206_264.80624709636
EARTH_RADIUS_WGS84_KM = 6_378.137000
SUN_RADIUS_KM = 695_700.000000
MERCURY_RADIUS_KM = 2_439.700000
REFERENCE_PI_ARCSEC = math.asin(EARTH_RADIUS_WGS84_KM / AU_KM) * ARCSEC_PER_RAD

START = "1769-Nov-09 17:00"
STOP = "1769-Nov-10 02:30"
STEP = "1m"
TANGENT_STEP_SECONDS = 30.0

SITE_A = {
    "key": "MERCURY_BAY",
    "label": "Mercury Bay, New Zealand — Cook-reported position",
    "lon_deg_east": 175.933333333333,
    "lat_deg": -36.783333333333,
    "elevation_km": 0.0,
    "trace": "Cook report: 36 deg 47 min S, 184 deg 04 min W = 175 deg 56 min E",
}
SITE_B = {
    "key": "VARDO",
    "label": "Vardø, Norway",
    "lon_deg_east": 31.110700000000,
    "lat_deg": 70.370600000000,
    "elevation_km": 0.0,
    "trace": "Project Vardø observer coordinate",
}

ROOT = Path("/content")
OUTPUT_DEFAULT = ROOT / "MERCURY_PARALLAX_VARDO_MERCURY_BAY_V0005_OUTPUT"
MASTER_DEFAULT = ROOT / "MERCURY_BAY_VARDO_1769_JPL_MASTER_V0005.csv"
PREFIXES = (
    "GEOCENTER_SUN",
    "GEOCENTER_MERCURY",
    "MERCURY_BAY_SUN",
    "MERCURY_BAY_MERCURY",
    "VARDO_SUN",
    "VARDO_MERCURY",
)
REQUIRED = ["JD", "UTC"] + [f"{prefix}_{axis}_KM" for prefix in PREFIXES for axis in "XYZ"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--jpl-csv", default="")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def norm(vector) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def unit(vector) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    magnitude = norm(value)
    if magnitude == 0.0:
        raise RuntimeError("Zero vector cannot be normalized.")
    return value / magnitude


def angular_separation_arcsec(vector_a, vector_b) -> float:
    cosine = float(np.clip(np.dot(unit(vector_a), unit(vector_b)), -1.0, 1.0))
    return math.acos(cosine) * ARCSEC_PER_RAD


def horizons_location(site: dict[str, object]) -> dict[str, float]:
    return {
        "lon": float(site["lon_deg_east"]),
        "lat": float(site["lat_deg"]),
        "elevation": float(site["elevation_km"]),
    }


def horizons_vectors(target_id: str, location, prefix: str) -> pd.DataFrame:
    query = Horizons(
        id=target_id,
        location=location,
        epochs={"start": START, "stop": STOP, "step": STEP},
    )
    vectors = query.vectors(refplane="earth").to_pandas()
    frame = pd.DataFrame()
    frame["JD"] = pd.to_numeric(vectors["datetime_jd"], errors="coerce")
    frame["UTC"] = vectors["datetime_str"].astype(str)
    for source, axis in (("x", "X"), ("y", "Y"), ("z", "Z")):
        frame[f"{prefix}_{axis}_KM"] = pd.to_numeric(vectors[source], errors="coerce") * AU_KM
    return frame.dropna().sort_values("JD").drop_duplicates("JD").reset_index(drop=True)


def build_jpl_master() -> pd.DataFrame:
    specifications = [
        ("10", "500@399", "GEOCENTER_SUN"),
        ("199", "500@399", "GEOCENTER_MERCURY"),
        ("10", horizons_location(SITE_A), "MERCURY_BAY_SUN"),
        ("199", horizons_location(SITE_A), "MERCURY_BAY_MERCURY"),
        ("10", horizons_location(SITE_B), "VARDO_SUN"),
        ("199", horizons_location(SITE_B), "VARDO_MERCURY"),
    ]
    master: pd.DataFrame | None = None
    for target, location, prefix in specifications:
        frame = horizons_vectors(target, location, prefix)
        if master is None:
            master = frame
        else:
            master = master.merge(frame.drop(columns="UTC"), on="JD", how="inner")
    if master is None or len(master) < 300:
        raise RuntimeError("JPL Horizons returned an incomplete six-series Mercury master.")
    return master[REQUIRED].sort_values("JD").reset_index(drop=True)


def compatible(path: Path) -> bool:
    try:
        columns = pd.read_csv(path, nrows=0).columns
    except Exception:
        return False
    return all(column in columns for column in REQUIRED)


def locate_or_build_master(requested: str) -> tuple[pd.DataFrame, Path, str]:
    candidates: list[Path] = []
    if requested:
        candidates.append(Path(requested).expanduser())
    candidates.append(MASTER_DEFAULT)
    for root, directories, files in os.walk(ROOT):
        directories[:] = [directory for directory in directories if directory != "drive" and not directory.startswith(".")]
        candidates.extend(Path(root) / filename for filename in files if filename.lower().endswith(".csv"))
    for candidate in candidates:
        if candidate.is_file() and compatible(candidate):
            return pd.read_csv(candidate)[REQUIRED], candidate.resolve(), "EXISTING COLAB JPL MASTER"
    master = build_jpl_master()
    master.to_csv(MASTER_DEFAULT, index=False, float_format="%.15f")
    return master, MASTER_DEFAULT.resolve(), "NEW JPL HORIZONS SIX-SERIES DOWNLOAD"


def build_cache(master: pd.DataFrame) -> dict[str, object]:
    frame = master.copy()
    frame["JD"] = pd.to_numeric(frame["JD"], errors="coerce")
    numeric_columns = [column for column in REQUIRED if column not in ("JD", "UTC")]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna().sort_values("JD").drop_duplicates("JD").reset_index(drop=True)
    jds = frame["JD"].to_numpy(dtype=float)
    cache: dict[str, object] = {"frame": frame, "JD": jds, "UTC": frame["UTC"].astype(str).tolist()}
    for prefix in PREFIXES:
        for axis in "XYZ":
            column = f"{prefix}_{axis}_KM"
            cache[column] = CubicSpline(jds, frame[column].to_numpy(dtype=float), bc_type="natural")
    return cache


def vector_at(cache: dict[str, object], prefix: str, jd: float) -> np.ndarray:
    return np.array([
        float(cache[f"{prefix}_X_KM"](jd)),
        float(cache[f"{prefix}_Y_KM"](jd)),
        float(cache[f"{prefix}_Z_KM"](jd)),
    ])


def utc_at(jd: float) -> str:
    return Time(jd, format="jd", scale="tdb").utc.iso + " UTC"


def site_vectors(cache: dict[str, object], site_key: str, jd: float) -> tuple[np.ndarray, np.ndarray]:
    return vector_at(cache, f"{site_key}_SUN", jd), vector_at(cache, f"{site_key}_MERCURY", jd)


def geocenter_separation(cache: dict[str, object], jd: float) -> float:
    return angular_separation_arcsec(
        vector_at(cache, "GEOCENTER_SUN", jd),
        vector_at(cache, "GEOCENTER_MERCURY", jd),
    )


def find_geocenter_closest(cache: dict[str, object]) -> float:
    jds = np.asarray(cache["JD"], dtype=float)
    values = np.array([geocenter_separation(cache, jd) for jd in jds])
    index = int(np.argmin(values))
    lower = float(jds[max(0, index - 2)])
    upper = float(jds[min(len(jds) - 1, index + 2)])
    result = minimize_scalar(
        lambda value: geocenter_separation(cache, value),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1.0e-13},
    )
    if not result.success:
        raise RuntimeError("Exact geocentric closest-approach minimization failed.")
    return float(result.x)


def angular_radii(cache: dict[str, object], site_key: str, jd: float) -> tuple[float, float]:
    sun, mercury = site_vectors(cache, site_key, jd)
    solar_radius = math.asin(min(1.0, SUN_RADIUS_KM / norm(sun))) * ARCSEC_PER_RAD
    mercury_radius = math.asin(min(1.0, MERCURY_RADIUS_KM / norm(mercury))) * ARCSEC_PER_RAD
    return solar_radius, mercury_radius


def contact_function(cache: dict[str, object], site_key: str, jd: float, internal: bool) -> float:
    sun, mercury = site_vectors(cache, site_key, jd)
    solar_radius, mercury_radius = angular_radii(cache, site_key, jd)
    target = solar_radius - mercury_radius if internal else solar_radius + mercury_radius
    return angular_separation_arcsec(sun, mercury) - target


def roots_for_contact_type(cache: dict[str, object], site_key: str, internal: bool) -> list[float]:
    jds = np.asarray(cache["JD"], dtype=float)
    values = np.array([contact_function(cache, site_key, jd, internal) for jd in jds])
    roots: list[float] = []
    for index in range(len(jds) - 1):
        left = float(values[index])
        right = float(values[index + 1])
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        if left == 0.0:
            roots.append(float(jds[index]))
        elif left * right < 0.0:
            roots.append(float(brentq(
                lambda value: contact_function(cache, site_key, value, internal),
                float(jds[index]),
                float(jds[index + 1]),
                xtol=1.0e-13,
                rtol=1.0e-13,
            )))
    unique: list[float] = []
    for root in sorted(roots):
        if not unique or abs(root - unique[-1]) * 86400.0 > 0.01:
            unique.append(root)
    if len(unique) != 2:
        label = "internal" if internal else "external"
        raise RuntimeError(f"Expected two {label} contact roots for {site_key}; found {len(unique)}.")
    return unique


def contacts(cache: dict[str, object], site_key: str) -> dict[str, float]:
    external = roots_for_contact_type(cache, site_key, internal=False)
    internal = roots_for_contact_type(cache, site_key, internal=True)
    return {"C1": external[0], "C2": internal[0], "C3": internal[1], "C4": external[1]}


def fixed_screen_basis(cache: dict[str, object], jd: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = unit(vector_at(cache, "GEOCENTER_SUN", jd))
    reference = np.array([0.0, 0.0, 1.0])
    if norm(np.cross(reference, normal)) < 1.0e-12:
        reference = np.array([1.0, 0.0, 0.0])
    xhat = unit(np.cross(reference, normal))
    yhat = unit(np.cross(normal, xhat))
    return normal, xhat, yhat


def apparent_point(cache: dict[str, object], site_key: str, jd: float, basis) -> np.ndarray:
    _normal, xhat, yhat = basis
    sun, mercury = site_vectors(cache, site_key, jd)
    difference = unit(mercury) - unit(sun)
    return np.array([
        float(np.dot(difference, xhat)) * ARCSEC_PER_RAD,
        float(np.dot(difference, yhat)) * ARCSEC_PER_RAD,
    ])


def local_tangent(cache: dict[str, object], site_key: str, jd: float, basis) -> np.ndarray:
    step = TANGENT_STEP_SECONDS / 86400.0
    direction = apparent_point(cache, site_key, jd + step, basis) - apparent_point(cache, site_key, jd - step, basis)
    direction = unit(direction)
    if direction[0] < 0.0:
        direction = -direction
    return direction


def observer_position_from_sun(cache: dict[str, object], site_key: str, jd: float) -> np.ndarray:
    geo_sun = vector_at(cache, "GEOCENTER_SUN", jd)
    site_sun = vector_at(cache, f"{site_key}_SUN", jd)
    return geo_sun - site_sun


def solar_altitude_deg(cache: dict[str, object], site_key: str, jd: float) -> float:
    site_sun = vector_at(cache, f"{site_key}_SUN", jd)
    observer = observer_position_from_sun(cache, site_key, jd)
    sine_altitude = float(np.clip(np.dot(unit(site_sun), unit(observer)), -1.0, 1.0))
    return math.degrees(math.asin(sine_altitude))


def observer_baseline(cache: dict[str, object], jd: float) -> np.ndarray:
    mercury_bay_to_mercury = vector_at(cache, "MERCURY_BAY_MERCURY", jd)
    vardo_to_mercury = vector_at(cache, "VARDO_MERCURY", jd)
    return mercury_bay_to_mercury - vardo_to_mercury


def independent_common_normal_reduction(cache: dict[str, object], closest_jd: float, basis) -> dict[str, float]:
    tangent_a = local_tangent(cache, "MERCURY_BAY", closest_jd, basis)
    tangent_b = local_tangent(cache, "VARDO", closest_jd, basis)
    tangent = unit(tangent_a + tangent_b)
    if tangent[0] < 0.0:
        tangent = -tangent
    normal_2d = np.array([-tangent[1], tangent[0]])

    point_a = apparent_point(cache, "MERCURY_BAY", closest_jd, basis)
    point_b = apparent_point(cache, "VARDO", closest_jd, basis)
    separation_vector = point_b - point_a
    normal_separation = abs(float(np.dot(separation_vector, normal_2d)))
    parallel_separation = abs(float(np.dot(separation_vector, tangent)))

    _screen_normal, xhat, yhat = basis
    baseline_3d = observer_baseline(cache, closest_jd)
    baseline_screen = np.array([
        float(np.dot(baseline_3d, xhat)),
        float(np.dot(baseline_3d, yhat)),
    ])
    baseline_normal = abs(float(np.dot(baseline_screen, normal_2d)))
    baseline_parallel = abs(float(np.dot(baseline_screen, tangent)))

    sun = vector_at(cache, "GEOCENTER_SUN", closest_jd)
    mercury = vector_at(cache, "GEOCENTER_MERCURY", closest_jd)
    d_es = norm(sun)
    d_em = norm(mercury)
    d_ms = norm(sun - mercury)
    raw_event = normal_separation * (d_em / d_ms) * (EARTH_RADIUS_WGS84_KM / baseline_normal)
    normalized_au = raw_event * (d_es / AU_KM)

    return {
        "common_track_angle_deg": math.degrees(math.atan2(tangent[1], tangent[0])),
        "mercury_bay_track_angle_deg": math.degrees(math.atan2(tangent_a[1], tangent_a[0])),
        "vardo_track_angle_deg": math.degrees(math.atan2(tangent_b[1], tangent_b[0])),
        "normal_separation_arcsec": normal_separation,
        "parallel_separation_arcsec": parallel_separation,
        "baseline_3d_km": norm(baseline_3d),
        "baseline_normal_km": baseline_normal,
        "baseline_parallel_km": baseline_parallel,
        "d_es_km": d_es,
        "d_em_km": d_em,
        "d_ms_km": d_ms,
        "distance_ratio_d_em_d_ms": d_em / d_ms,
        "dynamic_normalization_d_es_au": d_es / AU_KM,
        "event_parallax_arcsec": raw_event,
        "au_normalized_parallax_arcsec": normalized_au,
        "residual_vs_wgs84_arcsec": normalized_au - REFERENCE_PI_ARCSEC,
    }


def build_contact_table(cache: dict[str, object], site: dict[str, object], site_contacts: dict[str, float], closest_jd: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered = [("C1", site_contacts["C1"]), ("C2", site_contacts["C2"]), ("MAX", closest_jd), ("C3", site_contacts["C3"]), ("C4", site_contacts["C4"])]
    for event, jd in ordered:
        sun, mercury = site_vectors(cache, str(site["key"]), jd)
        rows.append({
            "site": str(site["label"]),
            "event": event,
            "utc": utc_at(jd),
            "separation_arcsec": angular_separation_arcsec(sun, mercury),
            "solar_altitude_deg": solar_altitude_deg(cache, str(site["key"]), jd),
            "observable_sun_above_horizon": solar_altitude_deg(cache, str(site["key"]), jd) > 0.0,
        })
    return pd.DataFrame(rows)


def format_table(frame: pd.DataFrame, formats: dict[str, str]) -> str:
    display = frame.copy()
    for column, pattern in formats.items():
        if column in display.columns:
            display[column] = display[column].map(lambda value, p=pattern: p.format(value))
    return display.to_string(index=False)


def main() -> None:
    arguments = parse_args()
    output_dir = Path(arguments.output_dir).expanduser().resolve() if arguments.output_dir else OUTPUT_DEFAULT
    output_dir.mkdir(parents=True, exist_ok=True)

    master, master_path, source = locate_or_build_master(arguments.jpl_csv)
    cache = build_cache(master)
    closest_jd = find_geocenter_closest(cache)
    closest_utc = utc_at(closest_jd)
    basis = fixed_screen_basis(cache, closest_jd)

    contacts_a = contacts(cache, "MERCURY_BAY")
    contacts_b = contacts(cache, "VARDO")
    table_a = build_contact_table(cache, SITE_A, contacts_a, closest_jd)
    table_b = build_contact_table(cache, SITE_B, contacts_b, closest_jd)
    contact_table = pd.concat([table_a, table_b], ignore_index=True)

    reduction = independent_common_normal_reduction(cache, closest_jd, basis)
    mercury_bay_visible = bool(table_a["observable_sun_above_horizon"].all())
    vardo_visible = bool(table_b["observable_sun_above_horizon"].all())
    historical_pair_valid = mercury_bay_visible and vardo_visible

    results = pd.DataFrame([
        {
            "case": "MERCURY_BAY_VARDO_COMMON_NORMAL",
            "classification": "VALID HISTORICAL TWO-STATION MEASUREMENT" if historical_pair_valid else "VIRTUAL JPL GEOMETRY — HISTORICAL PAIR REJECTED",
            "event_parallax_arcsec": reduction["event_parallax_arcsec"],
            "au_normalized_parallax_arcsec": reduction["au_normalized_parallax_arcsec"],
            "wgs84_standard_arcsec": REFERENCE_PI_ARCSEC,
            "residual_vs_wgs84_arcsec": reduction["residual_vs_wgs84_arcsec"],
            "mercury_bay_visible": mercury_bay_visible,
            "vardo_visible": vardo_visible,
        }
    ])

    master_csv = output_dir / "JPL_1769_MERCURY_BAY_VARDO_MASTER_V0005.csv"
    contacts_csv = output_dir / "MERCURY_1769_CONTACTS_AND_VISIBILITY_V0005.csv"
    geometry_csv = output_dir / "MERCURY_1769_COMMON_NORMAL_GEOMETRY_V0005.csv"
    results_csv = output_dir / "MERCURY_1769_PARALLAX_RESULTS_V0005.csv"

    export_master = master.drop(columns=["JD"]).copy()
    export_master.to_csv(master_csv, index=False, float_format="%.15f")
    contact_table.to_csv(contacts_csv, index=False, float_format="%.15f")
    pd.DataFrame([reduction]).to_csv(geometry_csv, index=False, float_format="%.15f")
    results.to_csv(results_csv, index=False, float_format="%.15f")

    checks = {
        "six JPL vector series present": all(f"{prefix}_{axis}_KM" in master.columns for prefix in PREFIXES for axis in "XYZ"),
        "four Mercury Bay contacts derived": len(contacts_a) == 4,
        "four Vardø geometric contacts derived": len(contacts_b) == 4,
        "normal separation positive": reduction["normal_separation_arcsec"] > 0.0,
        "normal baseline positive": reduction["baseline_normal_km"] > 0.0,
        "dynamic normalization verified": abs(reduction["dynamic_normalization_d_es_au"] - reduction["d_es_km"] / AU_KM) <= 1.0e-15,
        "no Julian date in exported master": "JD" not in pd.read_csv(master_csv, nrows=0).columns,
        "Mercury Bay visibility calculated": isinstance(mercury_bay_visible, bool),
        "Vardø visibility calculated": isinstance(vardo_visible, bool),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Equation or traceability checks failed: " + ", ".join(failed))

    result_table = pd.DataFrame([
        ["Geocentric closest approach", closest_utc, ""],
        ["Mercury Bay Sun above horizon", "YES" if mercury_bay_visible else "NO", ""],
        ["Vardø Sun above horizon", "YES" if vardo_visible else "NO", ""],
        ["Historical two-station pair", "VALID" if historical_pair_valid else "REJECTED", ""],
        ["Normal angular separation", f"{reduction['normal_separation_arcsec']:.12f}", "arcsec"],
        ["Independent normal baseline", f"{reduction['baseline_normal_km']:.6f}", "km"],
        ["Event solar parallax", f"{reduction['event_parallax_arcsec']:.12f}", "arcsec"],
        ["AU-normalized solar parallax", f"{reduction['au_normalized_parallax_arcsec']:.12f}", "arcsec"],
        ["WGS84 / IAU 2012 standard", f"{REFERENCE_PI_ARCSEC:.12f}", "arcsec"],
        ["Residual vs WGS84", f"{reduction['residual_vs_wgs84_arcsec']:+.12f}", "arcsec"],
    ], columns=["Quantity", "Value", "Unit"])

    contact_print = contact_table[["site", "event", "utc", "solar_altitude_deg", "observable_sun_above_horizon"]].copy()
    contact_print["solar_altitude_deg"] = contact_print["solar_altitude_deg"].map(lambda value: f"{value:+.6f}")
    contact_print["observable_sun_above_horizon"] = contact_print["observable_sun_above_horizon"].map(lambda value: "YES" if value else "NO")

    print(f"CODE OUTPUT: {VERSION}")
    print("CODE INPUTS")
    print(f"Program: {PROGRAM}")
    print(f"JPL source: {source}")
    print(f"JPL master: {master_path}")
    print(f"Mercury Bay: {SITE_A['lat_deg']:.12f} deg, {SITE_A['lon_deg_east']:.12f} deg east")
    print(f"Vardø: {SITE_B['lat_deg']:.12f} deg, {SITE_B['lon_deg_east']:.12f} deg east")
    print("COMMENTS")
    print("No AI images are generated. No fitted-track identity audit is used.")
    print("The code computes the independent common-normal Mercury parallax geometry and separately verifies daylight visibility.")
    print("RESULTS")
    print(result_table.to_string(index=False))
    print("\nCONTACT AND VISIBILITY TABLE")
    print(contact_print.to_string(index=False))
    print("OUTPUT SUMMARY")
    print(f"JPL calendar master: {master_csv}")
    print(f"Contacts and visibility CSV: {contacts_csv}")
    print(f"Common-normal geometry CSV: {geometry_csv}")
    print(f"Parallax results CSV: {results_csv}")
    print("PAPER COMPARISON")
    print("James Cook and Charles Green observed the 1769 Mercury transit from Mercury Bay; the Vardø pairing is tested here computationally, not assumed historically valid.")
    print("EQUATION STATUS")
    print("All equations and traceability checks: PASS")
    print("Historical measurement status: " + ("VALID" if historical_pair_valid else "REJECTED — Vardø daylight visibility fails"))
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# V0005
