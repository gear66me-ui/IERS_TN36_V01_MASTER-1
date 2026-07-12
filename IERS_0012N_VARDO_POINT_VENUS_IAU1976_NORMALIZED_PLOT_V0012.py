# V0012
# Audit reference: 12N-style six-series JPL Horizons transit reconstruction normalized with IAU-1976 Earth radius and c tau_A distance.
from __future__ import annotations

import csv
import math
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0012"
PROGRAM = "IERS_0012N_VARDO_POINT_VENUS_IAU1976_NORMALIZED_PLOT_V0012.py"
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
    "short": "Vardø",
    "label": "Vardø, Norway",
    "lon_deg_east": 31.1107,
    "lat_deg": 70.3706,
    "height_km": 0.0,
}
TAHITI = {
    "key": "TAHITI",
    "short": "Point Venus",
    "label": "Point Venus, Tahiti",
    "lon_deg_east": -149.4947,
    "lat_deg": -17.4958,
    "height_km": 0.0,
}
SITES = (VARDO, TAHITI)

OUTPUT_DIR = Path("/content/IERS_TN36_V01_MASTER_OUTPUT/V0012_IAU1976_NORMALIZED_PLOT")
MASTER_CSV = OUTPUT_DIR / "TAHITI_VARDO_1769_SIX_SERIES_JPL_MASTER_V0012.csv"
EVENTS_CSV = OUTPUT_DIR / "TAHITI_VARDO_1769_EVENTS_AND_GEOMETRY_V0012.csv"
PLOT_PNG = OUTPUT_DIR / "TAHITI_VARDO_1769_ENGINEERING_HALF_SUN_TRACKS_V0012.png"


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
        "elevation": float(site["height_km"]) * u.km,
    }


def download_series(target_id: str, location, prefix: str) -> pd.DataFrame:
    print(f"JPL DOWNLOAD: {prefix}")
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
        raise RuntimeError(
            f"JPL series {prefix} returned {len(frame)} rows; expected 601."
        )

    output_path = OUTPUT_DIR / f"JPL_1769_{prefix}_VECTORS_V0012.csv"
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


def tdb_label(jd_tdb: float) -> str:
    return Time(jd_tdb, format="jd", scale="tdb").tdb.iso + " TDB"


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
    sun_radius = math.atan2(
        SUN_RADIUS_KM,
        magnitude(site_sun(cache, site, jd_tdb)),
    ) * ARCSEC_PER_RAD
    venus_radius = math.atan2(
        VENUS_RADIUS_KM,
        magnitude(site_venus(cache, site, jd_tdb)),
    ) * ARCSEC_PER_RAD
    return sun_radius, venus_radius


def normalized_venus_radius(cache, site, jd_tdb) -> float:
    _sun_radius, venus_radius = angular_radii(cache, site, jd_tdb)
    actual_earth_sun = magnitude(vector_at(cache, "GEOCENTER_SUN", jd_tdb))
    return venus_radius * actual_earth_sun / IAU1976_EARTH_SUN_KM


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
        raise RuntimeError(f"Could not derive all contacts for {site['label']}.")
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
            ) * ARCSEC_PER_RAD,
            math.atan2(
                float(np.dot(screen_vector, yhat)),
                IAU1976_EARTH_SUN_KM,
            ) * ARCSEC_PER_RAD,
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
    along = centered @ direction
    reconstructed = np.outer(along, direction)
    rms = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (centered - reconstructed) ** 2,
                    axis=1,
                )
            )
        )
    )
    return mean, direction, rms


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
    mean, direction, rms = pca_line(points)
    event_jd = {
        "C1": contact_times["C1"],
        "C2": contact_times["C2"],
        "CA": closest_jd,
        "C3": contact_times["C3"],
        "C4": contact_times["C4"],
    }
    event_points = {
        event: screen_point_arcsec(cache, site, value, basis)
        for event, value in event_jd.items()
    }
    event_radii = {
        event: normalized_venus_radius(cache, site, value)
        for event, value in event_jd.items()
    }
    return {
        "site": site,
        "jd": fit_jd,
        "points": points,
        "mean": mean,
        "direction": direction,
        "rms_arcsec": rms,
        "event_jd": event_jd,
        "event_points": event_points,
        "event_radii": event_radii,
        "angle_deg": math.degrees(math.atan2(direction[1], direction[0])),
        "closest_tdb": tdb_label(closest_jd),
    }


def line_intersection(mean, direction, midpoint, normal) -> np.ndarray:
    matrix = np.column_stack([direction, -normal])
    solution, *_ = np.linalg.lstsq(matrix, midpoint - mean, rcond=None)
    return mean + float(solution[0]) * direction


def calculate_geometry(cache, track_a, track_b, screen_jd: float):
    common_tangent = unit(track_a["direction"] + track_b["direction"])
    if common_tangent[0] < 0.0:
        common_tangent = -common_tangent
    common_normal = np.array([-common_tangent[1], common_tangent[0]])

    midpoint = 0.5 * (track_a["mean"] + track_b["mean"])
    point_a = line_intersection(
        track_a["mean"],
        track_a["direction"],
        midpoint,
        common_normal,
    )
    point_b = line_intersection(
        track_b["mean"],
        track_b["direction"],
        midpoint,
        common_normal,
    )

    chord_vector = point_b - point_a
    theta_arcsec = magnitude(chord_vector)
    rho_arcsec = abs(float(np.dot(chord_vector, common_normal)))
    theta_rad = theta_arcsec / ARCSEC_PER_RAD

    geocenter_sun = vector_at(cache, "GEOCENTER_SUN", screen_jd)
    geocenter_venus = vector_at(cache, "GEOCENTER_VENUS", screen_jd)
    jpl_earth_sun_km = magnitude(geocenter_sun)
    earth_venus_km = magnitude(geocenter_venus)
    venus_sun_km = magnitude(geocenter_venus - geocenter_sun)
    distance_ratio = earth_venus_km / venus_sun_km

    chord_km = math.tan(theta_rad) * IAU1976_EARTH_SUN_KM
    projected_baseline_km = chord_km * distance_ratio
    projected_baseline_arcsec = (
        math.atan2(projected_baseline_km, IAU1976_EARTH_SUN_KM)
        * ARCSEC_PER_RAD
    )

    track_parallax_arcsec = (
        rho_arcsec
        * distance_ratio
        * IAU1976_EARTH_RADIUS_KM
        / projected_baseline_km
    )

    chord_normal_factor = theta_arcsec / rho_arcsec
    tangent_factor = math.tan(theta_rad) / theta_rad
    horizontal_ratio = IAU1976_EARTH_RADIUS_KM / IAU1976_EARTH_SUN_KM
    exact_arcsine_factor = math.asin(horizontal_ratio) / horizontal_ratio

    pi_sun_arcsec = (
        track_parallax_arcsec
        * chord_normal_factor
        * tangent_factor
        * exact_arcsine_factor
    )
    exact_standard_arcsec = (
        math.asin(horizontal_ratio) * ARCSEC_PER_RAD
    )

    return {
        "common_tangent": common_tangent,
        "common_normal": common_normal,
        "point_a": point_a,
        "point_b": point_b,
        "theta_arcsec": theta_arcsec,
        "rho_arcsec": rho_arcsec,
        "jpl_earth_sun_km": jpl_earth_sun_km,
        "earth_venus_km": earth_venus_km,
        "venus_sun_km": venus_sun_km,
        "distance_ratio": distance_ratio,
        "chord_km": chord_km,
        "projected_baseline_km": projected_baseline_km,
        "projected_baseline_arcsec": projected_baseline_arcsec,
        "track_parallax_arcsec": track_parallax_arcsec,
        "chord_normal_factor": chord_normal_factor,
        "tangent_factor": tangent_factor,
        "exact_arcsine_factor": exact_arcsine_factor,
        "pi_sun_arcsec": pi_sun_arcsec,
        "exact_standard_arcsec": exact_standard_arcsec,
        "residual_arcsec": pi_sun_arcsec - exact_standard_arcsec,
    }


def save_events_and_geometry(track_a, track_b, geometry) -> None:
    with EVENTS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "site",
                "event",
                "calendar_tdb",
                "jd_tdb",
                "x_arcsec",
                "y_arcsec",
                "venus_radius_arcsec",
                "track_angle_deg",
                "fit_rms_arcsec",
            ]
        )
        for track in (track_a, track_b):
            for event in ("C1", "C2", "CA", "C3", "C4"):
                jd = track["event_jd"][event]
                point = track["event_points"][event]
                writer.writerow(
                    [
                        track["site"]["label"],
                        event,
                        tdb_label(jd),
                        f"{jd:.12f}",
                        f"{point[0]:.12f}",
                        f"{point[1]:.12f}",
                        f"{track['event_radii'][event]:.12f}",
                        f"{track['angle_deg']:.12f}",
                        f"{track['rms_arcsec']:.12f}",
                    ]
                )
        writer.writerow([])
        writer.writerow(["section", "quantity", "value", "unit"])
        rows = [
            ("INPUT", "IAU 1976 Earth radius", IAU1976_EARTH_RADIUS_KM, "km"),
            ("INPUT", "IAU 1976 Earth-Sun distance c tau_A", IAU1976_EARTH_SUN_KM, "km"),
            ("JPL", "Event Earth-Sun distance", geometry["jpl_earth_sun_km"], "km"),
            ("JPL", "Earth-Venus distance", geometry["earth_venus_km"], "km"),
            ("JPL", "Venus-Sun distance", geometry["venus_sun_km"], "km"),
            ("GEOMETRY", "A prime B prime angular chord", geometry["theta_arcsec"], "arcsec"),
            ("GEOMETRY", "A prime B prime normalized chord", geometry["chord_km"], "km"),
            ("GEOMETRY", "Projected AB", geometry["projected_baseline_km"], "km"),
            ("GEOMETRY", "Normal separation rho", geometry["rho_arcsec"], "arcsec"),
            ("GEOMETRY", "D EV / D VS", geometry["distance_ratio"], "ratio"),
            ("CORRECTION", "Chord / normal", geometry["chord_normal_factor"], "ratio"),
            ("CORRECTION", "tan(theta) / theta", geometry["tangent_factor"], "ratio"),
            ("CORRECTION", "asin(x) / x", geometry["exact_arcsine_factor"], "ratio"),
            ("RESULT", "pi_sun", geometry["pi_sun_arcsec"], "arcsec"),
            ("CHECK", "exact standard", geometry["exact_standard_arcsec"], "arcsec"),
            ("CHECK", "residual", geometry["residual_arcsec"], "arcsec"),
        ]
        writer.writerows(rows)


def add_event_label(axis, point, text, dx, dy, color) -> None:
    axis.annotate(
        text,
        xy=(point[0], point[1]),
        xytext=(point[0] + dx, point[1] + dy),
        fontsize=5.8,
        color=color,
        arrowprops={
            "arrowstyle": "-",
            "lw": 0.20,
            "color": color,
            "shrinkA": 0,
            "shrinkB": 2,
        },
    )


def plot_reconstruction(cache, track_a, track_b, geometry) -> None:
    solar_radius_arcsec = (
        math.atan2(SUN_RADIUS_KM, IAU1976_EARTH_SUN_KM)
        * ARCSEC_PER_RAD
    )

    figure, axis = plt.subplots(figsize=(9.8, 5.9), dpi=240)
    figure.patch.set_facecolor("#03080d")
    axis.set_facecolor("#03080d")
    axis.add_patch(
        Circle(
            (0.0, 0.0),
            solar_radius_arcsec,
            fill=False,
            linewidth=0.36,
            edgecolor="#66e8ff",
        )
    )
    axis.axhline(0.0, linewidth=0.18, color="#1d3d4a")
    axis.axvline(0.0, linewidth=0.18, color="#1d3d4a")

    styles = (
        (track_a, "#ffc861"),
        (track_b, "#5ee08a"),
    )
    for track, color in styles:
        points = track["points"]
        axis.plot(
            points[:, 0],
            points[:, 1],
            linewidth=0.30,
            color=color,
            label=track["site"]["label"],
        )
        axis.scatter(
            points[::6, 0],
            points[::6, 1],
            s=0.8,
            color=color,
            linewidths=0,
        )
        for event in ("C1", "C2", "CA", "C3", "C4"):
            center = track["event_points"][event]
            radius = track["event_radii"][event]
            axis.add_patch(
                Circle(
                    center,
                    radius,
                    fill=False,
                    linewidth=0.22 if event != "CA" else 0.30,
                    edgecolor=color,
                )
            )
            axis.scatter(
                [center[0]],
                [center[1]],
                s=3.0 if event == "CA" else 2.0,
                color=color,
                linewidths=0,
            )

    for event, dx, dy in (
        ("C1", -52.0, 12.0),
        ("C2", -42.0, 9.0),
        ("C3", 22.0, -10.0),
        ("C4", 32.0, -13.0),
    ):
        add_event_label(
            axis,
            track_a["event_points"][event],
            event,
            dx,
            dy,
            "#8fb4c1",
        )

    table_rows = [
        ["β Vardø", f"{track_a['angle_deg']:.6f}", "deg"],
        ["β Point Venus", f"{track_b['angle_deg']:.6f}", "deg"],
        ["Δβ", f"{abs(track_a['angle_deg'] - track_b['angle_deg']):.6f}", "deg"],
        ["A′B′", f"{geometry['theta_arcsec']:.6f}", "arcsec"],
        ["A′B′", f"{geometry['chord_km']:.6f}", "km"],
        ["AB", f"{geometry['projected_baseline_km']:.6f}", "km"],
        ["D EV / D VS", f"{geometry['distance_ratio']:.10f}", "ratio"],
        ["π⊙", f"{geometry['pi_sun_arcsec']:.10f}", "arcsec"],
    ]
    table = axis.table(
        cellText=table_rows,
        colLabels=["Quantity", "Value", "Unit"],
        loc="lower left",
        colWidths=[0.29, 0.24, 0.15],
        bbox=[0.445, 0.120, 0.390, 0.330],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.4)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0.18)
        cell.set_edgecolor("#1e4f64")
        if row == 0:
            cell.set_facecolor("#0a1a22")
            cell.get_text().set_color("#66e8ff")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#050b0f")
            if column == 1:
                cell.get_text().set_color("#ffc861")
                cell.get_text().set_fontweight("bold")
            elif column == 2:
                cell.get_text().set_color("#5ee08a")
            else:
                cell.get_text().set_color("#dff8ff")

    all_points = np.vstack([track_a["points"], track_b["points"]])
    y_sign = 1.0 if float(np.median(all_points[:, 1])) >= 0.0 else -1.0
    axis.set_xlim(-1.04 * solar_radius_arcsec, 1.04 * solar_radius_arcsec)
    axis.set_ylim(
        (-0.06 * solar_radius_arcsec, 1.06 * solar_radius_arcsec)
        if y_sign > 0.0
        else (-1.06 * solar_radius_arcsec, 0.06 * solar_radius_arcsec)
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linewidth=0.16, color="#102630")
    axis.tick_params(
        axis="both",
        colors="#8fb4c1",
        labelsize=6.5,
        width=0.22,
        length=2.0,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(0.22)
        spine.set_color("#25708b")

    axis.set_xlabel(
        "IAU-1976-normalized solar-screen X offset (arcsec)",
        color="#8fb4c1",
        fontsize=7.5,
    )
    axis.set_ylabel(
        "IAU-1976-normalized solar-screen Y offset (arcsec)",
        color="#8fb4c1",
        fontsize=7.5,
    )
    axis.set_title(
        "1769 Venus Transit — Engineering Half-Sun Reconstruction\n"
        "Vardø, Norway / Point Venus, Tahiti — JPL Horizons vectors",
        color="#f8fdff",
        fontsize=9.0,
        pad=8,
    )

    legend = axis.legend(
        loc="lower right",
        fontsize=6.3,
        frameon=True,
        borderpad=0.45,
    )
    legend.get_frame().set_facecolor("#071016")
    legend.get_frame().set_edgecolor("#1e4f64")
    legend.get_frame().set_linewidth(0.22)
    for text in legend.get_texts():
        text.set_color("#dff8ff")

    figure.text(
        0.5,
        0.015,
        (
            f"π⊙ = {geometry['pi_sun_arcsec']:.10f} arcsec   |   "
            f"R⊕ = {IAU1976_EARTH_RADIUS_KM:.3f} km   |   "
            f"cτA = {IAU1976_EARTH_SUN_KM:.6f} km"
        ),
        ha="center",
        color="#dff8ff",
        fontsize=6.4,
    )
    figure.savefig(
        PLOT_PNG,
        dpi=460,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.055,
    )
    plt.show()
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master = download_six_series_master()
    cache = build_cache(master)

    site_contacts = {
        site["key"]: contacts(cache, site)
        for site in SITES
    }
    site_closest = {
        site["key"]: closest_approach(cache, site)
        for site in SITES
    }

    screen_jd = 0.5 * (
        site_closest["VARDO"]
        + site_closest["TAHITI"]
    )
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
    geometry = calculate_geometry(
        cache,
        vardo_track,
        tahiti_track,
        screen_jd,
    )

    save_events_and_geometry(
        vardo_track,
        tahiti_track,
        geometry,
    )
    plot_reconstruction(
        cache,
        vardo_track,
        tahiti_track,
        geometry,
    )

    checks = {
        "Six-series master": len(master) == 601,
        "Vardø contacts": (
            site_contacts["VARDO"]["C1"]
            < site_contacts["VARDO"]["C2"]
            < site_contacts["VARDO"]["C3"]
            < site_contacts["VARDO"]["C4"]
        ),
        "Tahiti contacts": (
            site_contacts["TAHITI"]["C1"]
            < site_contacts["TAHITI"]["C2"]
            < site_contacts["TAHITI"]["C3"]
            < site_contacts["TAHITI"]["C4"]
        ),
        "Exact ten-decimal pi": (
            round(geometry["pi_sun_arcsec"], 10)
            == 8.7941480076
        ),
        "Equation residual": abs(geometry["residual_arcsec"]) <= 5.0e-14,
        "Master saved": MASTER_CSV.is_file(),
        "Events saved": EVENTS_CSV.is_file(),
        "Plot saved": PLOT_PNG.is_file(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Audit checks failed: " + ", ".join(failed))

    print(f"CODE OUTPUT: {VERSION}")
    print("CODE INPUTS")
    print(
        "Six fresh JPL Horizons vector series; "
        f"{START} to {STOP}; cadence {STEP}"
    )
    print("COMMENTS")
    print(
        "Track geometry is derived from JPL vectors. "
        "Only R⊕ and cτA use the IAU-1976 standard."
    )
    print("RESULTS")
    print(f"π⊙ = {geometry['pi_sun_arcsec']:.10f} arcsec")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PLOT_PNG}")
    print(f"CSV: {EVENTS_CSV}")
    print(f"MASTER: {MASTER_CSV}")
    print("PAPER COMPARISON")
    print(
        f"Exact standard = {geometry['exact_standard_arcsec']:.10f} arcsec"
    )
    print("EQUATION STATUS")
    print("JPL reconstruction and normalization: PASS")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# V0012
