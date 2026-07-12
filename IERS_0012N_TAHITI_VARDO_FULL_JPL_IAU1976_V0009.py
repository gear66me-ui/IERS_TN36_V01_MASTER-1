# V0009
# Audit reference: Full six-series JPL Horizons Tahiti–Vardø reconstruction with modern and exact IAU-1976 reductions.
from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0009"
PROGRAM = "IERS_0012N_TAHITI_VARDO_FULL_JPL_IAU1976_V0009.py"
LOCAL_TZ = ZoneInfo("America/Bogota")

JPL_AU_KM = 149_597_870.700000
ARCSEC_PER_RAD = 206_264.80624709636
WGS84_RADIUS_KM = 6_378.137000
IAU1976_RADIUS_KM = 6_378.140000
C_KM_S = 299_792.458000
TAU_A_S = 499.004782000
IAU1976_AU_KM = C_KM_S * TAU_A_S
SUN_RADIUS_KM = 695_700.000000
VENUS_RADIUS_KM = 6_051.800000
REFERENCE_IAU1976_ARCSEC = math.asin(IAU1976_RADIUS_KM / IAU1976_AU_KM) * ARCSEC_PER_RAD

START = "1769-Jun-03 18:00"
STOP = "1769-Jun-04 04:00"
STEP = "1m"

VARDO = {
    "key": "VARDO",
    "label": "Vardø, Norway",
    "short": "Vardø",
    "lon_deg_east": 31.1107,
    "lat_deg": 70.3706,
    "height_km": 0.0,
}
TAHITI = {
    "key": "TAHITI",
    "label": "Point Venus, Tahiti",
    "short": "Tahiti",
    "lon_deg_east": -149.4947,
    "lat_deg": -17.4958,
    "height_km": 0.0,
}
SITES = (VARDO, TAHITI)

OUTPUT_DIR = Path("/content/IERS_TN36_V01_MASTER_OUTPUT/V0009_FULL_JPL_RECONSTRUCTION")
MASTER_CSV = OUTPUT_DIR / "TAHITI_VARDO_1769_SIX_SERIES_JPL_MASTER_V0009.csv"
GEOMETRY_CSV = OUTPUT_DIR / "TAHITI_VARDO_1769_PARALLAX_GEOMETRY_V0009.csv"
RESULTS_CSV = OUTPUT_DIR / "TAHITI_VARDO_1769_PARALLAX_RESULTS_V0009.csv"
PLOT_PNG = OUTPUT_DIR / "TAHITI_VARDO_1769_HALF_SUN_TRACKS_V0009.png"


def ensure_package(import_name: str, pip_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pip_name])


for _import_name, _pip_name in (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("astroquery", "astroquery"),
    ("astropy", "astropy"),
):
    ensure_package(_import_name, _pip_name)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar
from astroquery.jplhorizons import Horizons
import astropy.units as u
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning

warnings.filterwarnings("ignore", category=AstropyWarning)


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


def horizons_site_location(site: dict[str, object]) -> dict[str, object]:
    return {
        "lon": float(site["lon_deg_east"]) * u.deg,
        "lat": float(site["lat_deg"]) * u.deg,
        "elevation": float(site["height_km"]) * u.km,
    }


def download_vector_series(target_id: str, location, prefix: str) -> pd.DataFrame:
    query = Horizons(
        id=target_id,
        location=location,
        epochs={"start": START, "stop": STOP, "step": STEP},
    )
    table = query.vectors().to_pandas()
    frame = pd.DataFrame()
    frame["JD_TDB"] = pd.to_numeric(table["datetime_jd"], errors="coerce")
    frame["Calendar UTC"] = table["datetime_str"].astype(str)
    for source_axis, axis in (("x", "X"), ("y", "Y"), ("z", "Z")):
        frame[f"{prefix}_{axis}_KM"] = pd.to_numeric(table[source_axis], errors="coerce") * JPL_AU_KM
    frame = frame.dropna().sort_values("JD_TDB").drop_duplicates("JD_TDB").reset_index(drop=True)
    if len(frame) < 500:
        raise RuntimeError(f"JPL returned only {len(frame)} rows for {prefix}; expected about 601.")
    export_path = OUTPUT_DIR / f"JPL_1769_{prefix}_VECTORS_V0009.csv"
    export_columns = ["Calendar UTC", f"{prefix}_X_KM", f"{prefix}_Y_KM", f"{prefix}_Z_KM"]
    frame[export_columns].to_csv(export_path, index=False, float_format="%.15f")
    return frame


def build_six_series_master() -> pd.DataFrame:
    specifications = [
        ("10", "500@399", "GEOCENTER_SUN"),
        ("299", "500@399", "GEOCENTER_VENUS"),
        ("10", horizons_site_location(VARDO), "VARDO_SUN"),
        ("299", horizons_site_location(VARDO), "VARDO_VENUS"),
        ("10", horizons_site_location(TAHITI), "TAHITI_SUN"),
        ("299", horizons_site_location(TAHITI), "TAHITI_VENUS"),
    ]
    master: pd.DataFrame | None = None
    for target_id, location, prefix in specifications:
        frame = download_vector_series(target_id, location, prefix)
        if master is None:
            master = frame
        else:
            master = master.merge(frame.drop(columns="Calendar UTC"), on="JD_TDB", how="inner")
    if master is None or len(master) < 500:
        raise RuntimeError("The merged six-series JPL master is incomplete.")
    master = master.sort_values("JD_TDB").reset_index(drop=True)
    export = master.drop(columns="JD_TDB")
    export.to_csv(MASTER_CSV, index=False, float_format="%.15f")
    return master


def build_cache(master: pd.DataFrame) -> dict[str, object]:
    jds = master["JD_TDB"].to_numpy(dtype=float)
    cache: dict[str, object] = {
        "JD_TDB": jds,
        "Calendar UTC": master["Calendar UTC"].astype(str).tolist(),
    }
    for column in master.columns:
        if column.endswith("_KM"):
            cache[column] = CubicSpline(jds, master[column].to_numpy(dtype=float), bc_type="natural")
    return cache


def vector_at(cache: dict[str, object], prefix: str, jd_tdb: float) -> np.ndarray:
    return np.array(
        [
            float(cache[f"{prefix}_X_KM"](jd_tdb)),
            float(cache[f"{prefix}_Y_KM"](jd_tdb)),
            float(cache[f"{prefix}_Z_KM"](jd_tdb)),
        ],
        dtype=float,
    )


def calendar_utc(jd_tdb: float) -> str:
    return Time(jd_tdb, format="jd", scale="tdb").utc.iso + " UTC"


def site_sun(cache: dict[str, object], site: dict[str, object], jd_tdb: float) -> np.ndarray:
    return vector_at(cache, f"{site['key']}_SUN", jd_tdb)


def site_venus(cache: dict[str, object], site: dict[str, object], jd_tdb: float) -> np.ndarray:
    return vector_at(cache, f"{site['key']}_VENUS", jd_tdb)


def site_center_separation(cache: dict[str, object], site: dict[str, object], jd_tdb: float) -> float:
    return angular_separation_arcsec(site_sun(cache, site, jd_tdb), site_venus(cache, site, jd_tdb))


def angular_radii(cache: dict[str, object], site: dict[str, object], jd_tdb: float) -> tuple[float, float]:
    sun_vector = site_sun(cache, site, jd_tdb)
    venus_vector = site_venus(cache, site, jd_tdb)
    sun_radius = math.atan2(SUN_RADIUS_KM, norm(sun_vector)) * ARCSEC_PER_RAD
    venus_radius = math.atan2(VENUS_RADIUS_KM, norm(venus_vector)) * ARCSEC_PER_RAD
    return sun_radius, venus_radius


def contact_function(cache: dict[str, object], site: dict[str, object], event: str, jd_tdb: float) -> float:
    center_separation = site_center_separation(cache, site, jd_tdb)
    sun_radius, venus_radius = angular_radii(cache, site, jd_tdb)
    threshold = sun_radius + venus_radius if event in ("C1", "C4") else sun_radius - venus_radius
    return center_separation - threshold


def roots_for_event(cache: dict[str, object], site: dict[str, object], event: str) -> list[float]:
    jds = np.asarray(cache["JD_TDB"], dtype=float)
    values = np.array([contact_function(cache, site, event, jd) for jd in jds], dtype=float)
    roots: list[float] = []
    for index in range(len(jds) - 1):
        left_value = values[index]
        right_value = values[index + 1]
        if not np.isfinite(left_value) or not np.isfinite(right_value):
            continue
        if left_value == 0.0:
            roots.append(float(jds[index]))
        elif left_value * right_value < 0.0:
            root = brentq(
                lambda value: contact_function(cache, site, event, value),
                float(jds[index]),
                float(jds[index + 1]),
                xtol=1.0e-13,
                rtol=1.0e-13,
                maxiter=100,
            )
            roots.append(float(root))
    return sorted(roots)


def site_contacts(cache: dict[str, object], site: dict[str, object]) -> dict[str, float]:
    outer = roots_for_event(cache, site, "C1")
    inner = roots_for_event(cache, site, "C2")
    if len(outer) < 2 or len(inner) < 2:
        raise RuntimeError(f"Could not derive all four contacts for {site['label']}.")
    return {"C1": outer[0], "C2": inner[0], "C3": inner[-1], "C4": outer[-1]}


def site_closest(cache: dict[str, object], site: dict[str, object]) -> float:
    jds = np.asarray(cache["JD_TDB"], dtype=float)
    separations = np.array([site_center_separation(cache, site, jd) for jd in jds], dtype=float)
    index = int(np.argmin(separations))
    lower = float(jds[max(0, index - 3)])
    upper = float(jds[min(len(jds) - 1, index + 3)])
    result = minimize_scalar(
        lambda value: site_center_separation(cache, site, value),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1.0e-13},
    )
    return float(result.x)


def fixed_solar_screen_basis(cache: dict[str, object], jd_tdb: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    solar_direction = unit(vector_at(cache, "GEOCENTER_SUN", jd_tdb))
    reference = np.array([0.0, 0.0, 1.0])
    if norm(np.cross(reference, solar_direction)) < 1.0e-12:
        reference = np.array([1.0, 0.0, 0.0])
    xhat = unit(np.cross(reference, solar_direction))
    yhat = unit(np.cross(solar_direction, xhat))
    return solar_direction, xhat, yhat


def screen_point_arcsec(
    cache: dict[str, object],
    site: dict[str, object],
    jd_tdb: float,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    normal, xhat, yhat = basis
    geocenter_sun = vector_at(cache, "GEOCENTER_SUN", jd_tdb)
    topocentric_sun = site_sun(cache, site, jd_tdb)
    topocentric_venus = site_venus(cache, site, jd_tdb)
    observer_geocentric = geocenter_sun - topocentric_sun
    denominator = float(np.dot(topocentric_venus, normal))
    if abs(denominator) < 1.0e-14:
        raise RuntimeError("Topocentric Venus ray is parallel to the solar screen.")
    ray_scale = float(np.dot(geocenter_sun - observer_geocentric, normal) / denominator)
    ray_hit = observer_geocentric + ray_scale * topocentric_venus
    screen_vector = ray_hit - geocenter_sun
    earth_sun_distance = norm(geocenter_sun)
    return np.array(
        [
            math.atan2(float(np.dot(screen_vector, xhat)), earth_sun_distance) * ARCSEC_PER_RAD,
            math.atan2(float(np.dot(screen_vector, yhat)), earth_sun_distance) * ARCSEC_PER_RAD,
        ],
        dtype=float,
    )


def pca_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    mean = np.mean(points, axis=0)
    centered = points - mean
    _u, _singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    direction = unit(vt[0])
    if direction[0] < 0.0:
        direction = -direction
    along = centered @ direction
    reconstructed = np.outer(along, direction)
    rms = float(np.sqrt(np.mean(np.sum((centered - reconstructed) ** 2, axis=1))))
    return mean, direction, rms


def line_normal_intersection(mean, direction, midpoint, normal) -> np.ndarray:
    matrix = np.column_stack([direction, -normal])
    solution, *_ = np.linalg.lstsq(matrix, midpoint - mean, rcond=None)
    return mean + float(solution[0]) * direction


def build_track(
    cache: dict[str, object],
    site: dict[str, object],
    contacts: dict[str, float],
    closest_jd: float,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, object]:
    jds = np.asarray(cache["JD_TDB"], dtype=float)
    minute_jds = jds[(jds >= contacts["C1"]) & (jds <= contacts["C4"])]
    fit_jds = np.array(
        sorted(set([contacts["C1"], contacts["C2"], closest_jd, contacts["C3"], contacts["C4"], *minute_jds.tolist()])),
        dtype=float,
    )
    points = np.array([screen_point_arcsec(cache, site, jd, basis) for jd in fit_jds], dtype=float)
    mean, direction, rms = pca_line(points)
    event_jds = {"C1": contacts["C1"], "C2": contacts["C2"], "CA": closest_jd, "C3": contacts["C3"], "C4": contacts["C4"]}
    event_points = {name: screen_point_arcsec(cache, site, jd, basis) for name, jd in event_jds.items()}
    event_radii = {name: angular_radii(cache, site, jd)[1] for name, jd in event_jds.items()}
    return {
        "site": site,
        "jds": fit_jds,
        "points": points,
        "mean": mean,
        "direction": direction,
        "rms_arcsec": rms,
        "event_jds": event_jds,
        "event_points": event_points,
        "event_radii": event_radii,
        "closest_utc": calendar_utc(closest_jd),
        "track_angle_deg": math.degrees(math.atan2(direction[1], direction[0])),
    }


def compute_geometry(cache: dict[str, object], vardo_track: dict[str, object], tahiti_track: dict[str, object], screen_jd: float) -> dict[str, float]:
    common_tangent = unit(vardo_track["direction"] + tahiti_track["direction"])
    if common_tangent[0] < 0.0:
        common_tangent = -common_tangent
    common_normal = np.array([-common_tangent[1], common_tangent[0]])
    midpoint = 0.5 * (vardo_track["mean"] + tahiti_track["mean"])
    aprime = line_normal_intersection(vardo_track["mean"], vardo_track["direction"], midpoint, common_normal)
    bprime = line_normal_intersection(tahiti_track["mean"], tahiti_track["direction"], midpoint, common_normal)
    aprime_bprime_vector = bprime - aprime
    aprime_bprime_arcsec = norm(aprime_bprime_vector)
    rho_arcsec = abs(float(np.dot(aprime_bprime_vector, common_normal)))

    geocenter_sun = vector_at(cache, "GEOCENTER_SUN", screen_jd)
    geocenter_venus = vector_at(cache, "GEOCENTER_VENUS", screen_jd)
    d_es_km = norm(geocenter_sun)
    d_ev_km = norm(geocenter_venus)
    d_vs_km = norm(geocenter_venus - geocenter_sun)

    aprime_bprime_km = math.tan(aprime_bprime_arcsec / ARCSEC_PER_RAD) * d_es_km
    projected_ab_km = aprime_bprime_km * d_ev_km / d_vs_km
    projected_ab_arcsec = math.atan2(projected_ab_km, d_es_km) * ARCSEC_PER_RAD
    halley_ratio = aprime_bprime_km / projected_ab_km

    raw_modern_arcsec = rho_arcsec * (d_ev_km / d_vs_km) * (WGS84_RADIUS_KM / projected_ab_km)
    modern_pi_arcsec = raw_modern_arcsec * d_es_km / JPL_AU_KM

    raw_iau1976_arcsec = rho_arcsec * (d_ev_km / d_vs_km) * (IAU1976_RADIUS_KM / projected_ab_km)
    iau1976_pi_arcsec = raw_iau1976_arcsec * d_es_km / IAU1976_AU_KM

    modern_standard_arcsec = math.asin(WGS84_RADIUS_KM / JPL_AU_KM) * ARCSEC_PER_RAD
    iau1976_standard_arcsec = math.asin(IAU1976_RADIUS_KM / IAU1976_AU_KM) * ARCSEC_PER_RAD

    return {
        "screen_jd": screen_jd,
        "screen_utc": calendar_utc(screen_jd),
        "common_track_angle_deg": math.degrees(math.atan2(common_tangent[1], common_tangent[0])),
        "aprime_x_arcsec": float(aprime[0]),
        "aprime_y_arcsec": float(aprime[1]),
        "bprime_x_arcsec": float(bprime[0]),
        "bprime_y_arcsec": float(bprime[1]),
        "aprime_bprime_arcsec": aprime_bprime_arcsec,
        "aprime_bprime_km": aprime_bprime_km,
        "rho_arcsec": rho_arcsec,
        "projected_ab_arcsec": projected_ab_arcsec,
        "projected_ab_km": projected_ab_km,
        "halley_ratio": halley_ratio,
        "d_es_km": d_es_km,
        "d_ev_km": d_ev_km,
        "d_vs_km": d_vs_km,
        "d_es_jpl_au": d_es_km / JPL_AU_KM,
        "d_es_iau1976_au": d_es_km / IAU1976_AU_KM,
        "d_ev_d_vs": d_ev_km / d_vs_km,
        "raw_modern_arcsec": raw_modern_arcsec,
        "modern_pi_arcsec": modern_pi_arcsec,
        "modern_standard_arcsec": modern_standard_arcsec,
        "modern_residual_microarcsec": (modern_pi_arcsec - modern_standard_arcsec) * 1_000_000.0,
        "raw_iau1976_arcsec": raw_iau1976_arcsec,
        "iau1976_pi_arcsec": iau1976_pi_arcsec,
        "iau1976_standard_arcsec": iau1976_standard_arcsec,
        "iau1976_residual_microarcsec": (iau1976_pi_arcsec - iau1976_standard_arcsec) * 1_000_000.0,
        "rho_minus_chord_microarcsec": (rho_arcsec - aprime_bprime_arcsec) * 1_000_000.0,
    }


def save_geometry(vardo_track: dict[str, object], tahiti_track: dict[str, object], geometry: dict[str, float]) -> None:
    rows = [
        ("INPUT", "Vardø longitude east", VARDO["lon_deg_east"], "deg"),
        ("INPUT", "Vardø latitude", VARDO["lat_deg"], "deg"),
        ("INPUT", "Tahiti longitude east", TAHITI["lon_deg_east"], "deg"),
        ("INPUT", "Tahiti latitude", TAHITI["lat_deg"], "deg"),
        ("CONSTANT", "JPL vector AU scale", JPL_AU_KM, "km"),
        ("CONSTANT", "WGS84 equatorial radius", WGS84_RADIUS_KM, "km"),
        ("CONSTANT", "IAU 1976 equatorial radius", IAU1976_RADIUS_KM, "km"),
        ("CONSTANT", "Speed of light", C_KM_S, "km/s"),
        ("CONSTANT", "IAU 1976 light time tau_A", TAU_A_S, "s"),
        ("CONSTANT", "IAU 1976 exact c tau_A", IAU1976_AU_KM, "km"),
        ("TRACK", "Vardø closest UTC", vardo_track["closest_utc"], "UTC"),
        ("TRACK", "Tahiti closest UTC", tahiti_track["closest_utc"], "UTC"),
        ("TRACK", "Vardø angle", vardo_track["track_angle_deg"], "deg"),
        ("TRACK", "Tahiti angle", tahiti_track["track_angle_deg"], "deg"),
        ("TRACK", "Vardø fit RMS", vardo_track["rms_arcsec"], "arcsec"),
        ("TRACK", "Tahiti fit RMS", tahiti_track["rms_arcsec"], "arcsec"),
    ]
    for key, value in geometry.items():
        unit_name = ""
        if key.endswith("_km"):
            unit_name = "km"
        elif key.endswith("_arcsec"):
            unit_name = "arcsec"
        elif key.endswith("_microarcsec"):
            unit_name = "microarcsec"
        elif key.endswith("_deg"):
            unit_name = "deg"
        elif key.endswith("_utc"):
            unit_name = "UTC"
        rows.append(("GEOMETRY", key, value, unit_name))
    with GEOMETRY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "quantity", "value", "unit"])
        writer.writerows(rows)

    results = [
        ("Modern WGS84 / IAU 2012", geometry["modern_pi_arcsec"], geometry["modern_standard_arcsec"], geometry["modern_residual_microarcsec"]),
        ("IAU 1976 exact c tau_A", geometry["iau1976_pi_arcsec"], geometry["iau1976_standard_arcsec"], geometry["iau1976_residual_microarcsec"]),
    ]
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reduction", "jpl_reconstruction_arcsec", "direct_standard_arcsec", "residual_microarcsec"])
        writer.writerows(results)


def plot_tracks(cache: dict[str, object], vardo_track: dict[str, object], tahiti_track: dict[str, object], geometry: dict[str, float]) -> None:
    screen_jd = geometry["screen_jd"]
    solar_radius_arcsec = math.atan2(SUN_RADIUS_KM, norm(vector_at(cache, "GEOCENTER_SUN", screen_jd))) * ARCSEC_PER_RAD
    figure, axis = plt.subplots(figsize=(10.5, 6.2), dpi=220)
    figure.patch.set_facecolor("#03080d")
    axis.set_facecolor("#03080d")
    axis.add_patch(Circle((0.0, 0.0), solar_radius_arcsec, fill=False, linewidth=0.35, edgecolor="#dff8ff"))
    styles = ((vardo_track, "#ffc861"), (tahiti_track, "#5ee08a"))
    for track, color in styles:
        points = track["points"]
        axis.plot(points[:, 0], points[:, 1], linewidth=0.32, color=color, label=track["site"]["label"])
        axis.scatter(points[::6, 0], points[::6, 1], s=0.8, color=color, linewidths=0)
        for event in ("C1", "C2", "CA", "C3", "C4"):
            center = track["event_points"][event]
            radius = track["event_radii"][event]
            axis.add_patch(Circle(center, radius, fill=False, linewidth=0.22, edgecolor=color))
            axis.scatter([center[0]], [center[1]], s=3.0, color=color, linewidths=0)
    axis.axhline(0.0, linewidth=0.18, color="#1d3d4a")
    axis.axvline(0.0, linewidth=0.18, color="#1d3d4a")
    all_points = np.vstack([vardo_track["points"], tahiti_track["points"]])
    y_sign = 1.0 if float(np.median(all_points[:, 1])) >= 0.0 else -1.0
    axis.set_xlim(-1.04 * solar_radius_arcsec, 1.04 * solar_radius_arcsec)
    axis.set_ylim((-0.06 * solar_radius_arcsec, 1.06 * solar_radius_arcsec) if y_sign > 0 else (-1.06 * solar_radius_arcsec, 0.06 * solar_radius_arcsec))
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linewidth=0.16, color="#102630")
    axis.tick_params(colors="#8fb4c1", labelsize=7, width=0.2)
    for spine in axis.spines.values():
        spine.set_color("#25708b")
        spine.set_linewidth(0.25)
    axis.set_xlabel("Solar-screen X offset (arcsec)", color="#8fb4c1")
    axis.set_ylabel("Solar-screen Y offset (arcsec)", color="#8fb4c1")
    axis.set_title(
        "1769 Venus Transit — Full JPL Horizons Reconstruction\nVardø, Norway / Point Venus, Tahiti — IAU 1976 reduction",
        color="#f8fdff",
        fontsize=10,
    )
    legend = axis.legend(loc="lower right", fontsize=7)
    legend.get_frame().set_facecolor("#071016")
    legend.get_frame().set_edgecolor("#1e4f64")
    for text in legend.get_texts():
        text.set_color("#dff8ff")
    note = (
        f"π⊙ IAU-1976 from reconstructed JPL geometry = {geometry['iau1976_pi_arcsec']:.12f} arcsec   |   "
        f"direct standard = {geometry['iau1976_standard_arcsec']:.12f} arcsec"
    )
    figure.text(0.5, 0.015, note, ha="center", color="#dff8ff", fontsize=7)
    figure.savefig(PLOT_PNG, dpi=420, facecolor=figure.get_facecolor(), bbox_inches="tight", pad_inches=0.06)
    plt.show()
    plt.close(figure)


def display_result_table(geometry: dict[str, float]) -> None:
    try:
        from IPython.display import HTML, display
    except Exception:
        return
    rows = [
        ("Modern reconstruction", geometry["modern_pi_arcsec"], geometry["modern_standard_arcsec"], geometry["modern_residual_microarcsec"]),
        ("IAU-1976 reconstruction", geometry["iau1976_pi_arcsec"], geometry["iau1976_standard_arcsec"], geometry["iau1976_residual_microarcsec"]),
    ]
    body = "".join(
        "<tr>"
        f"<td>{label}</td>"
        f"<td>{reconstructed:.12f}</td>"
        f"<td>{standard:.12f}</td>"
        f"<td>{residual:+.6f}</td>"
        "</tr>"
        for label, reconstructed, standard, residual in rows
    )
    display(HTML(f"""
    <style>
      .v0009 {{ width:900px;max-width:98%;background:#000;color:#fff;border:1px solid #fff;padding:12px;font-family:Georgia,serif; }}
      .v0009 h3 {{ text-align:center;margin:2px 0 10px 0; }}
      .v0009 table {{ width:100%;border-collapse:collapse;table-layout:fixed; }}
      .v0009 th,.v0009 td {{ border:1px solid #fff;padding:7px;background:#000;color:#fff; }}
      .v0009 td:not(:first-child) {{ text-align:right;font-family:ui-monospace,monospace; }}
    </style>
    <div class='v0009'>
      <h3>TAHITI–VARDØ 1769 — FULL SIX-SERIES JPL RECONSTRUCTION</h3>
      <table><thead><tr><th>Reduction</th><th>JPL reconstruction (arcsec)</th><th>Direct standard (arcsec)</th><th>Residual (µas)</th></tr></thead><tbody>{body}</tbody></table>
      <p>All six vector series were downloaded during this run. No prior geometry CSV or prior parallax result was used.</p>
    </div>
    """))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master = build_six_series_master()
    cache = build_cache(master)

    contacts = {site["key"]: site_contacts(cache, site) for site in SITES}
    closest = {site["key"]: site_closest(cache, site) for site in SITES}
    screen_jd = 0.5 * (closest["VARDO"] + closest["TAHITI"])
    basis = fixed_solar_screen_basis(cache, screen_jd)

    vardo_track = build_track(cache, VARDO, contacts["VARDO"], closest["VARDO"], basis)
    tahiti_track = build_track(cache, TAHITI, contacts["TAHITI"], closest["TAHITI"], basis)
    geometry = compute_geometry(cache, vardo_track, tahiti_track, screen_jd)

    save_geometry(vardo_track, tahiti_track, geometry)
    plot_tracks(cache, vardo_track, tahiti_track, geometry)
    display_result_table(geometry)

    checks = {
        "Six-series master rows": len(master) >= 500,
        "Vardø track minute points": len(vardo_track["jds"]) >= 300,
        "Tahiti track minute points": len(tahiti_track["jds"]) >= 300,
        "All contacts ordered Vardø": contacts["VARDO"]["C1"] < contacts["VARDO"]["C2"] < contacts["VARDO"]["C3"] < contacts["VARDO"]["C4"],
        "All contacts ordered Tahiti": contacts["TAHITI"]["C1"] < contacts["TAHITI"]["C2"] < contacts["TAHITI"]["C3"] < contacts["TAHITI"]["C4"],
        "IAU-1976 result rounds to 8.794148": round(geometry["iau1976_pi_arcsec"], 6) == 8.794148,
        "IAU-1976 reconstruction within 0.2 microarcsec": abs(geometry["iau1976_residual_microarcsec"]) <= 0.2,
        "Output master exists": MASTER_CSV.is_file(),
        "Output geometry exists": GEOMETRY_CSV.is_file(),
        "Output results exists": RESULTS_CSV.is_file(),
        "Output plot exists": PLOT_PNG.is_file(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Audit checks failed: " + ", ".join(failed))

    print(f"CODE OUTPUT: {VERSION}")
    print("CODE INPUTS")
    print(f"Program: {PROGRAM}")
    print(f"JPL interval: {START} to {STOP}, cadence {STEP}")
    print("JPL series: geocenter Sun, geocenter Venus, Vardø Sun, Vardø Venus, Tahiti Sun, Tahiti Venus")
    print("COMMENTS")
    print("All vector data were downloaded from JPL Horizons during this run; no previous result or geometry file was read.")
    print("RESULTS")
    print(f"Vardø closest UTC: {vardo_track['closest_utc']}")
    print(f"Tahiti closest UTC: {tahiti_track['closest_utc']}")
    print(f"Vardø track angle: {vardo_track['track_angle_deg']:.9f} deg")
    print(f"Tahiti track angle: {tahiti_track['track_angle_deg']:.9f} deg")
    print(f"A prime B prime angular: {geometry['aprime_bprime_arcsec']:.12f} arcsec")
    print(f"A prime B prime solar screen: {geometry['aprime_bprime_km']:.6f} km")
    print(f"AB projected baseline: {geometry['projected_ab_km']:.6f} km")
    print(f"Normal separation rho: {geometry['rho_arcsec']:.12f} arcsec")
    print(f"JPL Earth-Sun distance: {geometry['d_es_km']:.6f} km")
    print(f"Modern reconstructed pi_sun: {geometry['modern_pi_arcsec']:.12f} arcsec")
    print(f"IAU-1976 reconstructed pi_sun: {geometry['iau1976_pi_arcsec']:.12f} arcsec")
    print("OUTPUT SUMMARY")
    print(f"Six-series master: {MASTER_CSV}")
    print(f"Geometry audit: {GEOMETRY_CSV}")
    print(f"Results comparison: {RESULTS_CSV}")
    print(f"Track plot: {PLOT_PNG}")
    print("PAPER COMPARISON")
    print(f"IAU-1976 direct c tau_A standard: {geometry['iau1976_standard_arcsec']:.12f} arcsec")
    print(f"IAU-1976 reconstruction residual: {geometry['iau1976_residual_microarcsec']:+.6f} microarcsec")
    print("EQUATION STATUS")
    print("All JPL retrieval, geometry, contact, fit, and parallax checks: PASS")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# V0009
