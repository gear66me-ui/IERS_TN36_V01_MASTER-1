# V0004
# Audit reference: Independent Tahiti–Vardø common-normal reduction versus fitted-track and standards audits from JPL vectors.
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


for _import, _pip in (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("astroquery", "astroquery"),
    ("astropy", "astropy"),
):
    ensure_package(_import, _pip)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar
from astroquery.jplhorizons import Horizons
from astropy.time import Time

VERSION = "V0004"
PROGRAM = "IERS_REDUCTION_VS_JPL_VECTORS_V0004.py"
TITLE = "SOLAR HORIZONTAL PARALLAX — HISTORICAL REDUCTION AND JPL VECTOR AUDIT"
LOCAL_TZ = ZoneInfo("America/Bogota")

AU_KM = 149_597_870.700000
ARCSEC_PER_RAD = 206_264.80624709636
EARTH_RADIUS_WGS84_KM = 6_378.137000
EARTH_RADIUS_IAU1976_KM = 6_378.140000
EARTH_RADIUS_IERS2010_KM = 6_378.136600
C_KM_S = 299_792.458000
TAU_A_S = 499.004782000
IAU1976_EXACT_AU_KM = C_KM_S * TAU_A_S
SUN_RADIUS_KM = 695_700.000000
VENUS_RADIUS_KM = 6_051.840000
REFERENCE_PI_ARCSEC = 8.794148
RECALLED_VALUE_ARCSEC = 8.791444

START = "1769-Jun-03 18:00"
STOP = "1769-Jun-04 04:00"
STEP = "1m"
TANGENT_STEP_SECONDS = 30.0

SITE_A = {
    "key": "TAHITI",
    "label": "Tahiti Point Venus",
    "lon_deg_east": -149.4970,
    "lat_deg": -17.4950,
    "elevation_km": 0.0,
}
SITE_B = {
    "key": "VARDO",
    "label": "Vardø, Norway",
    "lon_deg_east": 31.1107,
    "lat_deg": 70.3706,
    "elevation_km": 0.0,
}

ROOT = Path("/content")
OUTPUT_DEFAULT = ROOT / "IERS_REDUCTION_VS_JPL_VECTORS_V0004_OUTPUT"
MASTER_DEFAULT = ROOT / "TAHITI_VARDO_1769_JPL_MASTER_V0004.csv"
PREFIXES = (
    "GEOCENTER_SUN", "GEOCENTER_VENUS",
    "TAHITI_SUN", "TAHITI_VENUS",
    "VARDO_SUN", "VARDO_VENUS",
)
REQUIRED = ["JD", "UTC"] + [f"{p}_{axis}_KM" for p in PREFIXES for axis in "XYZ"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--jpl-csv", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--dpi", type=int, default=420)
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


def horizons_location(site: dict[str, object]):
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
    result = pd.DataFrame()
    result["JD"] = pd.to_numeric(vectors["datetime_jd"], errors="coerce")
    result["UTC"] = vectors["datetime_str"].astype(str)
    for source, axis in (("x", "X"), ("y", "Y"), ("z", "Z")):
        result[f"{prefix}_{axis}_KM"] = pd.to_numeric(vectors[source], errors="coerce") * AU_KM
    return result.dropna().sort_values("JD").drop_duplicates("JD").reset_index(drop=True)


def build_jpl_master() -> pd.DataFrame:
    specifications = [
        ("10", "500@399", "GEOCENTER_SUN"),
        ("299", "500@399", "GEOCENTER_VENUS"),
        ("10", horizons_location(SITE_A), "TAHITI_SUN"),
        ("299", horizons_location(SITE_A), "TAHITI_VENUS"),
        ("10", horizons_location(SITE_B), "VARDO_SUN"),
        ("299", horizons_location(SITE_B), "VARDO_VENUS"),
    ]
    master = None
    for target, location, prefix in specifications:
        frame = horizons_vectors(target, location, prefix)
        if master is None:
            master = frame
        else:
            master = master.merge(frame.drop(columns="UTC"), on="JD", how="inner")
    if master is None or len(master) < 100:
        raise RuntimeError("JPL Horizons returned an incomplete Tahiti–Vardø vector master.")
    return master[REQUIRED].sort_values("JD").reset_index(drop=True)


def compatible(path: Path) -> bool:
    try:
        columns = pd.read_csv(path, nrows=0).columns
    except Exception:
        return False
    return all(column in columns for column in REQUIRED)


def locate_or_build_master(requested: str) -> tuple[pd.DataFrame, Path, str]:
    candidates = []
    if requested:
        candidates.append(Path(requested).expanduser())
    candidates.extend([
        ROOT / "O6_TAHITI_VARDO_1769_1MIN_MASTER.csv",
        MASTER_DEFAULT,
    ])
    for root, directories, files in os.walk(ROOT):
        directories[:] = [d for d in directories if d != "drive" and not d.startswith(".")]
        candidates.extend(Path(root) / f for f in files if f.lower().endswith(".csv"))
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
    return Time(jd, format="jd", scale="tdb").utc.iso.replace(" ", " ") + " UTC"


def geocenter_separation(cache: dict[str, object], jd: float) -> float:
    return angular_separation_arcsec(
        vector_at(cache, "GEOCENTER_SUN", jd),
        vector_at(cache, "GEOCENTER_VENUS", jd),
    )


def find_geocenter_closest(cache: dict[str, object]) -> float:
    jds = np.asarray(cache["JD"], dtype=float)
    values = np.array([geocenter_separation(cache, jd) for jd in jds])
    index = int(np.argmin(values))
    lower = jds[max(0, index - 2)]
    upper = jds[min(len(jds) - 1, index + 2)]
    result = minimize_scalar(
        lambda value: geocenter_separation(cache, value),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1.0e-13},
    )
    return float(result.x)


def site_vectors(cache: dict[str, object], site_key: str, jd: float) -> tuple[np.ndarray, np.ndarray]:
    return vector_at(cache, f"{site_key}_SUN", jd), vector_at(cache, f"{site_key}_VENUS", jd)


def angular_radii(cache: dict[str, object], site_key: str, jd: float) -> tuple[float, float]:
    sun, venus = site_vectors(cache, site_key, jd)
    return (
        math.atan2(SUN_RADIUS_KM, norm(sun)) * ARCSEC_PER_RAD,
        math.atan2(VENUS_RADIUS_KM, norm(venus)) * ARCSEC_PER_RAD,
    )


def contact_function(cache: dict[str, object], site_key: str, jd: float) -> float:
    sun, venus = site_vectors(cache, site_key, jd)
    solar_radius, venus_radius = angular_radii(cache, site_key, jd)
    return angular_separation_arcsec(sun, venus) - (solar_radius + venus_radius)


def contact_roots(cache: dict[str, object], site_key: str) -> list[float]:
    jds = np.asarray(cache["JD"], dtype=float)
    values = np.array([contact_function(cache, site_key, jd) for jd in jds])
    roots: list[float] = []
    for index in range(len(jds) - 1):
        left, right = values[index], values[index + 1]
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        if left == 0.0:
            roots.append(float(jds[index]))
        elif left * right < 0.0:
            roots.append(float(brentq(
                lambda value: contact_function(cache, site_key, value),
                jds[index], jds[index + 1], xtol=1.0e-13, rtol=1.0e-13,
            )))
    if len(roots) < 2:
        raise RuntimeError(f"Could not derive external contacts for {site_key}.")
    return sorted(roots)


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
    sun, venus = site_vectors(cache, site_key, jd)
    difference = unit(venus) - unit(sun)
    return np.array([
        float(np.dot(difference, xhat)) * ARCSEC_PER_RAD,
        float(np.dot(difference, yhat)) * ARCSEC_PER_RAD,
    ])


def ray_screen_point(cache: dict[str, object], site_key: str, jd: float, basis) -> np.ndarray:
    normal, xhat, yhat = basis
    geo_sun = vector_at(cache, "GEOCENTER_SUN", jd)
    site_sun, site_venus = site_vectors(cache, site_key, jd)
    observer = geo_sun - site_sun
    denominator = float(np.dot(site_venus, normal))
    if abs(denominator) < 1.0e-14:
        raise RuntimeError("Topocentric Venus ray is parallel to the solar screen.")
    scale = float(np.dot(geo_sun - observer, normal) / denominator)
    hit = observer + scale * site_venus
    screen_vector = hit - geo_sun
    earth_sun_distance = norm(geo_sun)
    return np.array([
        math.atan2(float(np.dot(screen_vector, xhat)), earth_sun_distance) * ARCSEC_PER_RAD,
        math.atan2(float(np.dot(screen_vector, yhat)), earth_sun_distance) * ARCSEC_PER_RAD,
    ])


def local_tangent(cache, site_key: str, jd: float, basis, point_function) -> np.ndarray:
    step = TANGENT_STEP_SECONDS / 86400.0
    direction = point_function(cache, site_key, jd + step, basis) - point_function(cache, site_key, jd - step, basis)
    direction = unit(direction)
    if direction[0] < 0.0:
        direction = -direction
    return direction


def observer_baseline(cache: dict[str, object], jd: float) -> np.ndarray:
    tahiti_to_venus = vector_at(cache, "TAHITI_VENUS", jd)
    vardo_to_venus = vector_at(cache, "VARDO_VENUS", jd)
    return tahiti_to_venus - vardo_to_venus


def o6_common_normal_reduction(cache: dict[str, object], closest_jd: float, basis) -> dict[str, float]:
    tangent_a = local_tangent(cache, "TAHITI", closest_jd, basis, apparent_point)
    tangent_b = local_tangent(cache, "VARDO", closest_jd, basis, apparent_point)
    tangent = unit(tangent_a + tangent_b)
    if tangent[0] < 0.0:
        tangent = -tangent
    normal = np.array([-tangent[1], tangent[0]])
    point_a = apparent_point(cache, "TAHITI", closest_jd, basis)
    point_b = apparent_point(cache, "VARDO", closest_jd, basis)
    separation_vector = point_b - point_a
    separation_normal = abs(float(np.dot(separation_vector, normal)))
    separation_parallel = abs(float(np.dot(separation_vector, tangent)))

    _screen_normal, xhat, yhat = basis
    baseline_3d = observer_baseline(cache, closest_jd)
    baseline_screen = np.array([float(np.dot(baseline_3d, xhat)), float(np.dot(baseline_3d, yhat))])
    baseline_normal = abs(float(np.dot(baseline_screen, normal)))
    baseline_parallel = abs(float(np.dot(baseline_screen, tangent)))

    sun = vector_at(cache, "GEOCENTER_SUN", closest_jd)
    venus = vector_at(cache, "GEOCENTER_VENUS", closest_jd)
    d_es = norm(sun)
    d_ev = norm(venus)
    d_vs = norm(venus - sun)
    raw = separation_normal * (d_ev / d_vs) * (EARTH_RADIUS_WGS84_KM / baseline_normal)
    normalized = raw * (d_es / AU_KM)
    return {
        "track_angle_deg": math.degrees(math.atan2(tangent[1], tangent[0])),
        "tahiti_angle_deg": math.degrees(math.atan2(tangent_a[1], tangent_a[0])),
        "vardo_angle_deg": math.degrees(math.atan2(tangent_b[1], tangent_b[0])),
        "normal_separation_arcsec": separation_normal,
        "parallel_separation_arcsec": separation_parallel,
        "baseline_3d_km": norm(baseline_3d),
        "baseline_normal_km": baseline_normal,
        "baseline_parallel_km": baseline_parallel,
        "d_es_km": d_es,
        "d_ev_km": d_ev,
        "d_vs_km": d_vs,
        "dynamic_normalization": d_es / AU_KM,
        "raw_arcsec": raw,
        "normalized_arcsec": normalized,
        "residual_arcsec": normalized - REFERENCE_PI_ARCSEC,
    }


def pca_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    mean = np.mean(points, axis=0)
    centered = points - mean
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    direction = unit(vt[0])
    if direction[0] < 0.0:
        direction = -direction
    projection = centered @ direction
    reconstructed = np.outer(projection, direction)
    rms = float(np.sqrt(np.mean(np.sum((centered - reconstructed) ** 2, axis=1))))
    return mean, direction, rms


def line_normal_intersection(mean, direction, midpoint, normal) -> np.ndarray:
    matrix = np.column_stack([direction, -normal])
    solution, *_ = np.linalg.lstsq(matrix, midpoint - mean, rcond=None)
    return mean + solution[0] * direction


def fitted_track_reduction(cache: dict[str, object], closest_jd: float, basis) -> dict[str, float]:
    contacts_a = contact_roots(cache, "TAHITI")
    contacts_b = contact_roots(cache, "VARDO")
    interval_start = min(contacts_a[0], contacts_b[0])
    interval_stop = max(contacts_a[-1], contacts_b[-1])
    jds = np.asarray(cache["JD"], dtype=float)
    use_jds = jds[(jds >= interval_start) & (jds <= interval_stop)]
    if len(use_jds) < 20:
        raise RuntimeError("Fitted-track interval contains fewer than twenty JPL epochs.")

    points_a = np.array([ray_screen_point(cache, "TAHITI", jd, basis) for jd in use_jds])
    points_b = np.array([ray_screen_point(cache, "VARDO", jd, basis) for jd in use_jds])
    mean_a, direction_a, rms_a = pca_line(points_a)
    mean_b, direction_b, rms_b = pca_line(points_b)
    common_tangent = unit(direction_a + direction_b)
    if common_tangent[0] < 0.0:
        common_tangent = -common_tangent
    common_normal = np.array([-common_tangent[1], common_tangent[0]])
    midpoint = 0.5 * (mean_a + mean_b)
    aprime = line_normal_intersection(mean_a, direction_a, midpoint, common_normal)
    bprime = line_normal_intersection(mean_b, direction_b, midpoint, common_normal)
    ab_vector = bprime - aprime
    rho = abs(float(np.dot(ab_vector, common_normal)))
    ab_arcsec = norm(ab_vector)

    sun = vector_at(cache, "GEOCENTER_SUN", closest_jd)
    venus = vector_at(cache, "GEOCENTER_VENUS", closest_jd)
    d_es = norm(sun)
    d_ev = norm(venus)
    d_vs = norm(venus - sun)
    ab_km = math.tan(ab_arcsec / ARCSEC_PER_RAD) * d_es
    projected_baseline = ab_km * d_ev / d_vs
    raw = rho * (d_ev / d_vs) * (EARTH_RADIUS_WGS84_KM / projected_baseline)
    normalized = raw * d_es / AU_KM
    return {
        "interval_start": interval_start,
        "interval_stop": interval_stop,
        "interval_start_utc": utc_at(interval_start),
        "interval_stop_utc": utc_at(interval_stop),
        "row_count": len(use_jds),
        "tahiti_angle_deg": math.degrees(math.atan2(direction_a[1], direction_a[0])),
        "vardo_angle_deg": math.degrees(math.atan2(direction_b[1], direction_b[0])),
        "common_angle_deg": math.degrees(math.atan2(common_tangent[1], common_tangent[0])),
        "rho_arcsec": rho,
        "aprime_bprime_arcsec": ab_arcsec,
        "aprime_bprime_km": ab_km,
        "projected_baseline_km": projected_baseline,
        "fit_rms_tahiti_arcsec": rms_a,
        "fit_rms_vardo_arcsec": rms_b,
        "raw_arcsec": raw,
        "normalized_arcsec": normalized,
        "residual_arcsec": normalized - REFERENCE_PI_ARCSEC,
        "identity_gap_microarcsec": (rho - ab_arcsec) * 1_000_000.0,
    }


def standard_values(cache: dict[str, object], closest_jd: float) -> dict[str, float]:
    d_es = norm(vector_at(cache, "GEOCENTER_SUN", closest_jd))
    return {
        "jpl_distance_km": d_es,
        "jpl_epoch_parallax_wgs84": math.asin(EARTH_RADIUS_WGS84_KM / d_es) * ARCSEC_PER_RAD,
        "iau1976_case2": math.asin(EARTH_RADIUS_IAU1976_KM / IAU1976_EXACT_AU_KM) * ARCSEC_PER_RAD,
        "modern_wgs84": math.asin(EARTH_RADIUS_WGS84_KM / AU_KM) * ARCSEC_PER_RAD,
        "iers2010": math.asin(EARTH_RADIUS_IERS2010_KM / AU_KM) * ARCSEC_PER_RAD,
    }


def save_vectors(master: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    calendar = master.copy()
    paths = {
        "geocenter": output_dir / "JPL_1769_GEOCENTER_VECTORS_V0004.csv",
        "tahiti": output_dir / "JPL_1769_TAHITI_VECTORS_V0004.csv",
        "vardo": output_dir / "JPL_1769_VARDO_VECTORS_V0004.csv",
        "master": output_dir / "JPL_1769_TAHITI_VARDO_MASTER_V0004.csv",
    }
    calendar[["UTC", *[f"GEOCENTER_{body}_{axis}_KM" for body in ("SUN", "VENUS") for axis in "XYZ"]]].to_csv(paths["geocenter"], index=False, float_format="%.15f")
    calendar[["UTC", *[f"TAHITI_{body}_{axis}_KM" for body in ("SUN", "VENUS") for axis in "XYZ"]]].to_csv(paths["tahiti"], index=False, float_format="%.15f")
    calendar[["UTC", *[f"VARDO_{body}_{axis}_KM" for body in ("SUN", "VENUS") for axis in "XYZ"]]].to_csv(paths["vardo"], index=False, float_format="%.15f")
    calendar[["UTC", *[column for column in REQUIRED if column not in ("JD", "UTC")]]].to_csv(paths["master"], index=False, float_format="%.15f")
    return paths


def panel(axis, title: str) -> None:
    axis.set_axis_off()
    axis.add_patch(FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        transform=axis.transAxes, linewidth=0.8,
        edgecolor="white", facecolor="black", clip_on=False, zorder=-10,
    ))
    axis.text(0.025, 0.94, title, transform=axis.transAxes, ha="left", va="top", fontsize=11.2, fontweight="bold", color="white")


def render_plate(standards, o6, fitted, closest_utc, source, paths, output_png: Path, dpi: int) -> None:
    plt.close("all")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "figure.facecolor": "black",
        "savefig.facecolor": "black",
        "text.color": "white",
    })
    figure = plt.figure(figsize=(18, 12), facecolor="black")
    grid = figure.add_gridspec(4, 2, height_ratios=(0.17, 0.93, 1.08, 0.88), hspace=0.17, wspace=0.06, left=0.035, right=0.965, top=0.965, bottom=0.05)
    title_axis = figure.add_subplot(grid[0, :]); title_axis.axis("off")
    title_axis.text(0.5, 0.74, TITLE, ha="center", fontsize=18.5, fontweight="bold")
    title_axis.text(0.5, 0.22, "1769 Tahiti–Vardø independent baseline reduction • fitted-track audit • JPL Horizons vectors", ha="center", fontsize=10.8, color="#D1D5DB")
    a1, a2, a3, a4 = figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1]), figure.add_subplot(grid[2, :]), figure.add_subplot(grid[3, :])

    panel(a1, "I. WHAT THE TWO JPL CALCULATIONS MEAN")
    statements = [
        ("Standards reduction", "Uses |Earth–Sun| from JPL, then changes the distance/radius convention. It is not a two-station transit measurement."),
        ("Tahiti–Vardø common normal", "Uses the observed angular track separation and an independently projected physical Earth baseline."),
        ("Fitted-track A′B′ audit", "Uses fitted screen tracks. When ρ ≈ A′B′, the algebra nearly collapses to the adopted radius/AU standard."),
        ("Hard-coded 1.01474", "REJECTED / NOT USED. The event normalization is computed directly as D_ES / AU."),
    ]
    y = 0.79
    for heading, text in statements:
        a1.text(0.05, y, heading, transform=a1.transAxes, fontsize=9.7, fontweight="bold")
        a1.text(0.05, y - 0.06, text, transform=a1.transAxes, fontsize=8.5, color="#D1D5DB", wrap=True)
        y -= 0.20

    panel(a2, "II. TAHITI–VARDØ GEOMETRY FROM SIX JPL VECTOR SERIES")
    geometry_rows = [
        ("Calendar closest approach", closest_utc, ""),
        (r"$D_{ES}$", f"{o6['d_es_km']:,.6f}", "km"),
        (r"$D_{ES}/AU$", f"{o6['dynamic_normalization']:.12f}", ""),
        (r"$\rho_N$", f"{o6['normal_separation_arcsec']:.9f}", "arcsec"),
        (r"$B_N$", f"{o6['baseline_normal_km']:,.6f}", "km"),
        ("Common track angle", f"{o6['track_angle_deg']:.9f}", "deg"),
        ("Tahiti track angle", f"{o6['tahiti_angle_deg']:.9f}", "deg"),
        ("Vardø track angle", f"{o6['vardo_angle_deg']:.9f}", "deg"),
        (r"$D_{EV}/D_{VS}$", f"{o6['d_ev_km']/o6['d_vs_km']:.12f}", ""),
    ]
    y = 0.80
    for label, value, unit in geometry_rows:
        a2.text(0.05, y, label, transform=a2.transAxes, fontsize=9.0)
        a2.text(0.37, y, value, transform=a2.transAxes, fontsize=8.6, family="monospace")
        a2.text(0.87, y, unit, transform=a2.transAxes, fontsize=8.4, color="#D1D5DB")
        y -= 0.075

    panel(a3, "III. SOLAR-PARALLAX RESULTS — INDEPENDENT TRANSIT REDUCTION VS FITTED-TRACK AUDIT")
    comparison = pd.DataFrame([
        ["IAU 1976", "Exact cτ_A standard", standards["iau1976_case2"], standards["iau1976_case2"] - REFERENCE_PI_ARCSEC, "STANDARD"],
        ["IAU 2012", "WGS84 radius / exact AU", standards["modern_wgs84"], standards["modern_wgs84"] - REFERENCE_PI_ARCSEC, "STANDARD"],
        ["Tahiti–Vardø", "Independent common-normal baseline", o6["normalized_arcsec"], o6["residual_arcsec"], "MEASURED"],
        ["Tahiti–Vardø", "Fitted-track A′B′ consistency audit", fitted["normalized_arcsec"], fitted["residual_arcsec"], "AUDIT"],
    ], columns=["Case", "Method / convention", "π⊙ (arcsec)", "Δ vs 8.794148 (arcsec)", "Class"])
    display = comparison.copy()
    display["π⊙ (arcsec)"] = display["π⊙ (arcsec)"].map(lambda value: f"{value:.12f}")
    display["Δ vs 8.794148 (arcsec)"] = display["Δ vs 8.794148 (arcsec)"].map(lambda value: f"{value:+.12f}")
    table = a3.table(cellText=display.values, colLabels=display.columns, cellLoc="left", colLoc="center", bbox=(0.025, 0.13, 0.95, 0.70), colWidths=(0.13, 0.39, 0.17, 0.21, 0.10))
    table.set_zorder(5); table.auto_set_font_size(False); table.set_fontsize(8.5)
    for (row, _column), cell in table.get_celld().items():
        cell.set_linewidth(0.45); cell.set_edgecolor("white")
        cell.set_facecolor("black" if row == 0 else ("#111111" if row % 2 else "#202020"))
        cell.set_text_props(color="white", weight="bold" if row == 0 else "normal")
    a3.text(0.04, 0.055, f"Project search found 8.794144″, not 8.791444″. The fresh calculation above is authoritative for this run.", transform=a3.transAxes, fontsize=9.0, color="#D1D5DB")

    panel(a4, "IV. EQUATION AUDIT AND TRACEABILITY")
    equations = [
        rf"$\pi_{{\odot,\mathrm{{TV}}}}=\rho_N\,(D_{{EV}}/D_{{VS}})\,(R_\oplus/B_N)\,(D_{{ES}}/AU)={o6['normalized_arcsec']:.12f}^{{\prime\prime}}$",
        rf"$\pi_{{\odot,\mathrm{{fit}}}}=\rho\,(D_{{EV}}/D_{{VS}})\,(R_\oplus/B_{{proj}})\,(D_{{ES}}/AU)={fitted['normalized_arcsec']:.12f}^{{\prime\prime}}$",
        rf"$\rho={fitted['rho_arcsec']:.9f}^{{\prime\prime}},\quad A^\prime B^\prime={fitted['aprime_bprime_arcsec']:.9f}^{{\prime\prime}},\quad \rho-A^\prime B^\prime={fitted['identity_gap_microarcsec']:.3f}\ \mu\mathrm{{as}}$",
    ]
    y = 0.75
    for equation in equations:
        a4.text(0.04, y, equation, transform=a4.transAxes, fontsize=12.0)
        y -= 0.18
    a4.text(0.04, 0.16, "Generated vector files:", transform=a4.transAxes, fontsize=9.5, fontweight="bold")
    a4.text(0.20, 0.16, "  •  ".join(path.name for path in paths.values()), transform=a4.transAxes, fontsize=8.2, family="monospace", color="#D1D5DB")
    a4.text(0.04, 0.07, f"JPL source: {source}. No published parallax value is used to compute either Tahiti–Vardø result.", transform=a4.transAxes, fontsize=8.8, color="#D1D5DB")

    figure.text(0.5, 0.016, "Figure V0004. The independent baseline reduction and fitted-track audit are intentionally reported separately; the latter is a consistency check, not an independent measurement.", ha="center", fontsize=8.3, color="#D1D5DB")
    figure.savefig(output_png, dpi=max(240, int(dpi)), bbox_inches="tight", pad_inches=0.08, facecolor="black")
    plt.close(figure)


def main() -> None:
    arguments = parse_args()
    output_dir = Path(arguments.output_dir).expanduser().resolve() if arguments.output_dir else OUTPUT_DEFAULT
    output_dir.mkdir(parents=True, exist_ok=True)
    master, master_path, source = locate_or_build_master(arguments.jpl_csv)
    cache = build_cache(master)
    closest_jd = find_geocenter_closest(cache)
    closest_utc = utc_at(closest_jd)
    basis = fixed_screen_basis(cache, closest_jd)
    standards = standard_values(cache, closest_jd)
    o6 = o6_common_normal_reduction(cache, closest_jd, basis)
    fitted = fitted_track_reduction(cache, closest_jd, basis)
    vector_paths = save_vectors(master, output_dir)

    results = pd.DataFrame([
        {"case": "IAU1976_CASE2", "method": "Exact c tau_A standard", "pi_arcsec": standards["iau1976_case2"], "residual_arcsec": standards["iau1976_case2"] - REFERENCE_PI_ARCSEC},
        {"case": "MODERN_WGS84", "method": "WGS84 / exact AU", "pi_arcsec": standards["modern_wgs84"], "residual_arcsec": standards["modern_wgs84"] - REFERENCE_PI_ARCSEC},
        {"case": "TAHITI_VARDO_COMMON_NORMAL", "method": "Independent O6 baseline reduction", "pi_arcsec": o6["normalized_arcsec"], "residual_arcsec": o6["residual_arcsec"]},
        {"case": "TAHITI_VARDO_FITTED_TRACK", "method": "Fitted-track A-prime B-prime audit", "pi_arcsec": fitted["normalized_arcsec"], "residual_arcsec": fitted["residual_arcsec"]},
    ])
    results_csv = output_dir / "IERS_REDUCTION_VS_JPL_VECTORS_V0004_RESULTS.csv"
    geometry_csv = output_dir / "IERS_REDUCTION_VS_JPL_VECTORS_V0004_TAHITI_VARDO_GEOMETRY.csv"
    publication_png = output_dir / "SOLAR_PARALLAX_HISTORICAL_REDUCTION_JPL_AUDIT_V0004.png"
    results.to_csv(results_csv, index=False, float_format="%.15f")
    pd.DataFrame([{**{f"O6_{k}": v for k, v in o6.items()}, **{f"FIT_{k}": v for k, v in fitted.items()}}]).to_csv(geometry_csv, index=False, float_format="%.15f")
    render_plate(standards, o6, fitted, closest_utc, source, vector_paths, publication_png, arguments.dpi)

    checks = {
        "six JPL vector series present": all(f"{prefix}_{axis}_KM" in master.columns for prefix in PREFIXES for axis in "XYZ"),
        "dynamic normalization not hard-coded": abs(o6["dynamic_normalization"] - o6["d_es_km"] / AU_KM) <= 1.0e-15,
        "independent baseline positive": o6["baseline_normal_km"] > 0.0,
        "common-normal separation positive": o6["normal_separation_arcsec"] > 0.0,
        "fitted-track interval valid": fitted["row_count"] >= 20,
        "publication image generated": publication_png.is_file(),
        "calendar-only vector exports": all("JD" not in pd.read_csv(path, nrows=0).columns for path in vector_paths.values()),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Equation checks failed: " + ", ".join(failed))

    try:
        from IPython.display import Image, display
        display(Image(filename=str(publication_png)))
    except Exception:
        pass

    print(f"CODE OUTPUT: {VERSION}")
    print("CODE INPUTS")
    print(f"Program: {PROGRAM}")
    print(f"JPL source: {source}")
    print(f"JPL master: {master_path}")
    print("COMMENTS")
    print("The independent Tahiti–Vardø common-normal baseline reduction is separated from the fitted-track identity audit.")
    print("The hard-coded historical factor 1.01474 is REJECTED / NOT USED; D_ES/AU is calculated from JPL vectors.")
    print("RESULTS")
    print(f"Calendar closest approach: {closest_utc}")
    print(f"Tahiti–Vardø independent normalized π⊙: {o6['normalized_arcsec']:.12f} arcsec")
    print(f"Tahiti–Vardø fitted-track audit π⊙: {fitted['normalized_arcsec']:.12f} arcsec")
    print("OUTPUT SUMMARY")
    print(f"Publication image: {publication_png}")
    print(f"Results CSV: {results_csv}")
    print(f"Geometry CSV: {geometry_csv}")
    print(f"JPL combined master: {vector_paths['master']}")
    print("PAPER COMPARISON")
    print(f"Verified project record: 8.794144 arcsec; user-recalled 8.791444 arcsec is comparison-only and was not found in the project artifacts.")
    print(f"IAU-1976 exact Case 2: {standards['iau1976_case2']:.12f} arcsec")
    print("EQUATION STATUS")
    print("All checks: PASS")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# V0004
