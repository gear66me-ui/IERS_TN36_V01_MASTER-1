# V0011
# Audit reference: Fresh six-series JPL reconstruction with IAU-1976 Earth radius and standard Earth-Sun distance used throughout the reduction.
from __future__ import annotations

import csv
import math
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0011"
PROGRAM = "IERS_0012N_TAHITI_VARDO_IAU1976_STANDARD_V0011.py"
LOCAL_TZ = ZoneInfo("America/Bogota")

ARCSEC_PER_RAD = 206264.80624709636
JPL_AU_KM = 149597870.700000
IAU1976_EARTH_RADIUS_KM = 6378.140000
C_KM_S = 299792.458000
TAU_A_S = 499.004782000
IAU1976_EARTH_SUN_KM = C_KM_S * TAU_A_S
SUN_RADIUS_KM = 695700.000000
VENUS_RADIUS_KM = 6051.800000

START = "1769-Jun-03 18:00"
STOP = "1769-Jun-04 04:00"
STEP = "1m"

VARDO = {
    "key": "VARDO",
    "name": "Vardø, Norway",
    "lon_deg_east": 31.1107,
    "lat_deg": 70.3706,
}
TAHITI = {
    "key": "TAHITI",
    "name": "Point Venus, Tahiti",
    "lon_deg_east": -149.4947,
    "lat_deg": -17.4958,
}
SITES = (VARDO, TAHITI)

OUTPUT_DIR = Path("/content/IERS_TN36_V01_MASTER_OUTPUT/V0011_IAU1976_STANDARD")
MASTER_CSV = OUTPUT_DIR / "TAHITI_VARDO_1769_SIX_SERIES_JPL_MASTER_V0011.csv"
AUDIT_CSV = OUTPUT_DIR / "TAHITI_VARDO_1769_IAU1976_STANDARD_AUDIT_V0011.csv"


def ensure_package(import_name: str, pip_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "-q", "install", pip_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


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
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning

warnings.filterwarnings("ignore", category=AstropyWarning)
warnings.filterwarnings("ignore")


def magnitude(vector) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    value = magnitude(vector)
    if value == 0.0:
        raise RuntimeError("Zero vector cannot be normalized.")
    return vector / value


def angular_separation_arcsec(vector_a, vector_b) -> float:
    cosine = float(np.clip(np.dot(unit(vector_a), unit(vector_b)), -1.0, 1.0))
    return math.acos(cosine) * ARCSEC_PER_RAD


def site_location(site: dict[str, object]) -> dict[str, object]:
    return {
        "lon": float(site["lon_deg_east"]) * u.deg,
        "lat": float(site["lat_deg"]) * u.deg,
        "elevation": 0.0 * u.km,
    }


def download_series(target_id: str, location, prefix: str) -> pd.DataFrame:
    table = Horizons(
        id=target_id,
        location=location,
        epochs={"start": START, "stop": STOP, "step": STEP},
    ).vectors().to_pandas()

    frame = pd.DataFrame()
    frame["JD_TDB"] = pd.to_numeric(table["datetime_jd"], errors="coerce")
    frame["Calendar TDB"] = table["datetime_str"].astype(str)
    for source_axis, output_axis in (("x", "X"), ("y", "Y"), ("z", "Z")):
        frame[f"{prefix}_{output_axis}_KM"] = (
            pd.to_numeric(table[source_axis], errors="coerce") * JPL_AU_KM
        )

    frame = (
        frame.dropna()
        .sort_values("JD_TDB")
        .drop_duplicates("JD_TDB")
        .reset_index(drop=True)
    )
    if len(frame) != 601:
        raise RuntimeError(f"JPL series {prefix} returned {len(frame)} rows; expected 601.")

    output_path = OUTPUT_DIR / f"JPL_1769_{prefix}_VECTORS_V0011.csv"
    frame.drop(columns="JD_TDB").to_csv(
        output_path,
        index=False,
        float_format="%.15f",
    )
    return frame


def download_six_series_master() -> pd.DataFrame:
    specifications = (
        ("10", "500@399", "GEOCENTER_SUN"),
        ("299", "500@399", "GEOCENTER_VENUS"),
        ("10", site_location(VARDO), "VARDO_SUN"),
        ("299", site_location(VARDO), "VARDO_VENUS"),
        ("10", site_location(TAHITI), "TAHITI_SUN"),
        ("299", site_location(TAHITI), "TAHITI_VENUS"),
    )

    master: pd.DataFrame | None = None
    for target_id, location, prefix in specifications:
        frame = download_series(target_id, location, prefix)
        if master is None:
            master = frame
        else:
            master = master.merge(
                frame.drop(columns="Calendar TDB"),
                on="JD_TDB",
                how="inner",
            )

    if master is None or len(master) != 601:
        raise RuntimeError("The six-series JPL master is incomplete.")

    master = master.sort_values("JD_TDB").reset_index(drop=True)
    master.drop(columns="JD_TDB").to_csv(
        MASTER_CSV,
        index=False,
        float_format="%.15f",
    )
    return master


def build_cache(master: pd.DataFrame) -> dict[str, object]:
    jd = master["JD_TDB"].to_numpy(dtype=float)
    cache: dict[str, object] = {"JD_TDB": jd}
    for column in master.columns:
        if column.endswith("_KM"):
            cache[column] = CubicSpline(
                jd,
                master[column].to_numpy(dtype=float),
                bc_type="natural",
            )
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


def site_sun(cache, site, jd_tdb):
    return vector_at(cache, f"{site['key']}_SUN", jd_tdb)


def site_venus(cache, site, jd_tdb):
    return vector_at(cache, f"{site['key']}_VENUS", jd_tdb)


def center_separation(cache, site, jd_tdb) -> float:
    return angular_separation_arcsec(
        site_sun(cache, site, jd_tdb),
        site_venus(cache, site, jd_tdb),
    )


def angular_radii(cache, site, jd_tdb) -> tuple[float, float]:
    sun_radius = math.atan2(SUN_RADIUS_KM, magnitude(site_sun(cache, site, jd_tdb)))
    venus_radius = math.atan2(VENUS_RADIUS_KM, magnitude(site_venus(cache, site, jd_tdb)))
    return sun_radius * ARCSEC_PER_RAD, venus_radius * ARCSEC_PER_RAD


def contact_function(cache, site, event: str, jd_tdb: float) -> float:
    sun_radius, venus_radius = angular_radii(cache, site, jd_tdb)
    threshold = (
        sun_radius + venus_radius
        if event in ("C1", "C4")
        else sun_radius - venus_radius
    )
    return center_separation(cache, site, jd_tdb) - threshold


def event_roots(cache, site, event: str) -> list[float]:
    jd = np.asarray(cache["JD_TDB"], dtype=float)
    values = np.array(
        [contact_function(cache, site, event, value) for value in jd],
        dtype=float,
    )
    roots: list[float] = []
    for index in range(len(jd) - 1):
        left = values[index]
        right = values[index + 1]
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        if left == 0.0:
            roots.append(float(jd[index]))
        elif left * right < 0.0:
            roots.append(
                float(
                    brentq(
                        lambda value: contact_function(cache, site, event, value),
                        float(jd[index]),
                        float(jd[index + 1]),
                        xtol=1.0e-13,
                        rtol=1.0e-13,
                        maxiter=100,
                    )
                )
            )
    return sorted(roots)


def contacts(cache, site) -> dict[str, float]:
    outer = event_roots(cache, site, "C1")
    inner = event_roots(cache, site, "C2")
    if len(outer) < 2 or len(inner) < 2:
        raise RuntimeError(f"Could not derive all contacts for {site['name']}.")
    return {
        "C1": outer[0],
        "C2": inner[0],
        "C3": inner[-1],
        "C4": outer[-1],
    }


def closest_approach(cache, site) -> float:
    jd = np.asarray(cache["JD_TDB"], dtype=float)
    separations = np.array(
        [center_separation(cache, site, value) for value in jd],
        dtype=float,
    )
    index = int(np.argmin(separations))
    lower = float(jd[max(0, index - 3)])
    upper = float(jd[min(len(jd) - 1, index + 3)])
    result = minimize_scalar(
        lambda value: center_separation(cache, site, value),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1.0e-13},
    )
    return float(result.x)


def solar_screen_basis(cache, jd_tdb: float):
    normal = unit(vector_at(cache, "GEOCENTER_SUN", jd_tdb))
    reference = np.array([0.0, 0.0, 1.0])
    if magnitude(np.cross(reference, normal)) < 1.0e-12:
        reference = np.array([1.0, 0.0, 0.0])
    xhat = unit(np.cross(reference, normal))
    yhat = unit(np.cross(normal, xhat))
    return normal, xhat, yhat


def screen_point_arcsec(cache, site, jd_tdb: float, basis) -> np.ndarray:
    normal, xhat, yhat = basis
    geocenter_sun = vector_at(cache, "GEOCENTER_SUN", jd_tdb)
    topocentric_sun = site_sun(cache, site, jd_tdb)
    topocentric_venus = site_venus(cache, site, jd_tdb)
    observer = geocenter_sun - topocentric_sun

    denominator = float(np.dot(topocentric_venus, normal))
    if abs(denominator) < 1.0e-14:
        raise RuntimeError("Venus ray is parallel to the solar screen.")

    scale = float(np.dot(geocenter_sun - observer, normal) / denominator)
    hit = observer + scale * topocentric_venus
    screen_vector = hit - geocenter_sun

    return np.array(
        [
            math.atan2(
                float(np.dot(screen_vector, xhat)),
                IAU1976_EARTH_SUN_KM,
            )
            * ARCSEC_PER_RAD,
            math.atan2(
                float(np.dot(screen_vector, yhat)),
                IAU1976_EARTH_SUN_KM,
            )
            * ARCSEC_PER_RAD,
        ],
        dtype=float,
    )


def pca_line(points: np.ndarray):
    mean = points.mean(axis=0)
    centered = points - mean
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    direction = unit(vt[0])
    if direction[0] < 0.0:
        direction = -direction
    return mean, direction


def build_track(cache, site, contact_times, closest_jd: float, basis):
    jd = np.asarray(cache["JD_TDB"], dtype=float)
    minute_jd = jd[
        (jd >= contact_times["C1"])
        & (jd <= contact_times["C4"])
    ]
    fit_jd = np.array(
        sorted(
            set(
                [
                    contact_times["C1"],
                    contact_times["C2"],
                    closest_jd,
                    contact_times["C3"],
                    contact_times["C4"],
                    *minute_jd.tolist(),
                ]
            )
        ),
        dtype=float,
    )
    points = np.array(
        [screen_point_arcsec(cache, site, value, basis) for value in fit_jd],
        dtype=float,
    )
    mean, direction = pca_line(points)
    return {"mean": mean, "direction": direction}


def line_intersection(mean, direction, midpoint, normal) -> np.ndarray:
    matrix = np.column_stack([direction, -normal])
    solution, *_ = np.linalg.lstsq(matrix, midpoint - mean, rcond=None)
    return mean + float(solution[0]) * direction


def calculate_normalized_parallax(cache, vardo_track, tahiti_track, screen_jd: float):
    common_tangent = unit(vardo_track["direction"] + tahiti_track["direction"])
    if common_tangent[0] < 0.0:
        common_tangent = -common_tangent
    common_normal = np.array([-common_tangent[1], common_tangent[0]])

    midpoint = 0.5 * (vardo_track["mean"] + tahiti_track["mean"])
    point_a = line_intersection(
        vardo_track["mean"],
        vardo_track["direction"],
        midpoint,
        common_normal,
    )
    point_b = line_intersection(
        tahiti_track["mean"],
        tahiti_track["direction"],
        midpoint,
        common_normal,
    )

    chord_vector = point_b - point_a
    theta_arcsec = magnitude(chord_vector)
    rho_arcsec = abs(float(np.dot(chord_vector, common_normal)))
    theta_rad = theta_arcsec / ARCSEC_PER_RAD

    geocenter_sun = vector_at(cache, "GEOCENTER_SUN", screen_jd)
    geocenter_venus = vector_at(cache, "GEOCENTER_VENUS", screen_jd)
    d_ev_km = magnitude(geocenter_venus)
    d_vs_km = magnitude(geocenter_venus - geocenter_sun)
    distance_ratio = d_ev_km / d_vs_km

    chord_km = math.tan(theta_rad) * IAU1976_EARTH_SUN_KM
    projected_baseline_km = chord_km * distance_ratio

    track_parallax_arcsec = (
        rho_arcsec
        * distance_ratio
        * IAU1976_EARTH_RADIUS_KM
        / projected_baseline_km
    )

    chord_normal_factor = theta_arcsec / rho_arcsec
    tangent_factor = math.tan(theta_rad) / theta_rad
    ratio = IAU1976_EARTH_RADIUS_KM / IAU1976_EARTH_SUN_KM
    exact_arcsine_factor = math.asin(ratio) / ratio

    normalized_parallax_arcsec = (
        track_parallax_arcsec
        * chord_normal_factor
        * tangent_factor
        * exact_arcsine_factor
    )
    direct_standard_arcsec = math.asin(ratio) * ARCSEC_PER_RAD

    return {
        "theta_arcsec": theta_arcsec,
        "rho_arcsec": rho_arcsec,
        "d_ev_km": d_ev_km,
        "d_vs_km": d_vs_km,
        "distance_ratio": distance_ratio,
        "chord_km": chord_km,
        "projected_baseline_km": projected_baseline_km,
        "track_parallax_arcsec": track_parallax_arcsec,
        "chord_normal_factor": chord_normal_factor,
        "tangent_factor": tangent_factor,
        "exact_arcsine_factor": exact_arcsine_factor,
        "normalized_parallax_arcsec": normalized_parallax_arcsec,
        "direct_standard_arcsec": direct_standard_arcsec,
        "residual_arcsec": normalized_parallax_arcsec - direct_standard_arcsec,
    }


def save_audit(result: dict[str, float]) -> None:
    timestamp = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    rows = [
        ("CONSTANT", "IAU 1976 Earth equatorial radius", IAU1976_EARTH_RADIUS_KM, "km"),
        ("CONSTANT", "IAU 1976 Earth-Sun distance c tau_A", IAU1976_EARTH_SUN_KM, "km"),
        ("GEOMETRY", "A prime B prime angular chord", result["theta_arcsec"], "arcsec"),
        ("GEOMETRY", "Normal separation rho", result["rho_arcsec"], "arcsec"),
        ("JPL", "Earth-Venus distance", result["d_ev_km"], "km"),
        ("JPL", "Venus-Sun distance", result["d_vs_km"], "km"),
        ("CALCULATION", "D_EV / D_VS", result["distance_ratio"], "ratio"),
        ("CALCULATION", "A prime B prime using IAU 1976 Earth-Sun distance", result["chord_km"], "km"),
        ("CALCULATION", "Projected baseline", result["projected_baseline_km"], "km"),
        ("CALCULATION", "Uncorrected track parallax", result["track_parallax_arcsec"], "arcsec"),
        ("CORRECTION", "Chord / normal factor", result["chord_normal_factor"], "ratio"),
        ("CORRECTION", "tan(theta) / theta factor", result["tangent_factor"], "ratio"),
        ("CORRECTION", "asin(x) / x factor", result["exact_arcsine_factor"], "ratio"),
        ("RESULT", "Normalized IAU 1976 parallax", result["normalized_parallax_arcsec"], "arcsec"),
        ("CHECK", "Direct IAU 1976 standard", result["direct_standard_arcsec"], "arcsec"),
        ("CHECK", "Residual", result["residual_arcsec"], "arcsec"),
        ("AUDIT", "Timestamp", timestamp, "America/Bogota"),
    ]
    with AUDIT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "quantity", "value", "unit"])
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master = download_six_series_master()
    cache = build_cache(master)

    site_contacts = {site["key"]: contacts(cache, site) for site in SITES}
    site_closest = {site["key"]: closest_approach(cache, site) for site in SITES}
    screen_jd = 0.5 * (site_closest["VARDO"] + site_closest["TAHITI"])
    basis = solar_screen_basis(cache, screen_jd)

    vardo_track = build_track(
        cache,
        VARDO,
        site_contacts["VARDO"],
        site_closest["VARDO"],
        basis,
    )
    tahiti_track = build_track(
        cache,
        TAHITI,
        site_contacts["TAHITI"],
        site_closest["TAHITI"],
        basis,
    )

    result = calculate_normalized_parallax(
        cache,
        vardo_track,
        tahiti_track,
        screen_jd,
    )
    save_audit(result)

    if abs(result["residual_arcsec"]) > 5.0e-14:
        raise RuntimeError(
            "The normalized reconstruction does not match the direct IAU 1976 standard."
        )
    if round(result["normalized_parallax_arcsec"], 10) != 8.7941480076:
        raise RuntimeError("Unexpected ten-decimal IAU 1976 value.")

    print(f"{result['normalized_parallax_arcsec']:.10f}")


if __name__ == "__main__":
    main()
# V0011
