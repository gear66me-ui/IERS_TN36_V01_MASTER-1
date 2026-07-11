# IERS-0012AC
# Audit reference: 2012 geocentric JPL Venus-transit recreation in IERS-0012AA engineering PNG style.
from __future__ import annotations

import math
import os
import subprocess
import sys
from dataclasses import dataclass
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
from astropy.time import Time


VERSION = "IERS-0012AC"
PROGRAM_NAME = "IERS_0012AC_2012_GEOCENTRIC_TRACK_PNG.py"

AU_KM = 149_597_870.7
ARCSEC_PER_RAD = 206_264.80624709636
SUN_RADIUS_KM = 695_700.0
VENUS_RADIUS_KM = 6_051.8
LOCAL_TZ = ZoneInfo("America/Bogota")

START = "2012-Jun-05 20:00"
STOP = "2012-Jun-06 07:30"
STEP = "1m"
JPL_LOCATION = "500@399"

VIDEO_TLS_COMPARISON_DEG = 8.4340601435
VIDEO_OLS_COMPARISON_DEG = 8.4116632810
VIDEO_COMPARISON_VERSION = "V0007"
SATELLITE_ESTIMATE = "NASA Solar Dynamics Observatory (SDO)"

DRIVE_ROOT = Path("/content/drive/MyDrive/Colab Notebooks")
if DRIVE_ROOT.exists():
    PROJECT_ROOT = DRIVE_ROOT / "IERS_TN36_OUTPUT"
else:
    PROJECT_ROOT = Path("/content/IERS_TN36_OUTPUT")
OUTPUT_PNG_DIR = PROJECT_ROOT / "OUTPUT_PNG"
OUTPUT_CSV_DIR = PROJECT_ROOT / "OUTPUT_CSV"
OUTPUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)

TRACK_PNG = OUTPUT_PNG_DIR / f"{VERSION}_2012_GEOCENTRIC_ENGINEERING_HALF_SUN.png"
RESULTS_PNG = OUTPUT_PNG_DIR / f"{VERSION}_2012_GEOCENTRIC_RESULTS.png"
TRACK_CSV = OUTPUT_CSV_DIR / f"{VERSION}_2012_GEOCENTRIC_TRACK.csv"

BG = "#03080d"
PANEL = "#071016"
GRID = "#102630"
SPINE = "#25708b"
TEXT = "#dff8ff"
MUTED = "#8fb4c1"
SOLAR = "#57c7e3"
TRACK = "#58d68d"
VENUS = "#b7ffcf"
ACCENT = "#ffd166"
REJECTED = "#ef8354"


@dataclass(frozen=True)
class Event:
    name: str
    jd_tdb: float
    utc: str
    point: np.ndarray
    venus_radius_arcsec: float
    sun_radius_arcsec: float


def norm(vector: np.ndarray) -> float:
    return float(np.sqrt(np.dot(vector, vector)))


def unit(vector: np.ndarray) -> np.ndarray:
    magnitude = norm(vector)
    if magnitude == 0.0:
        raise RuntimeError("Zero vector encountered.")
    return np.asarray(vector, dtype=float) / magnitude


def angular_sep_arcsec(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(unit(vector_a), unit(vector_b)), -1.0, 1.0))
    return math.acos(cosine) * ARCSEC_PER_RAD


def utc_at(jd_tdb: float) -> str:
    return Time(jd_tdb, format="jd", scale="tdb").utc.iso


def horizons_geocenter_vectors(target_id: str, prefix: str) -> pd.DataFrame:
    result = Horizons(
        id=target_id,
        location=JPL_LOCATION,
        epochs={"start": START, "stop": STOP, "step": STEP},
    ).vectors().to_pandas()
    table = pd.DataFrame()
    table["jd_tdb"] = result["datetime_jd"].astype(float)
    table["utc"] = result["datetime_str"].astype(str)
    for axis in ("x", "y", "z"):
        table[f"{prefix}_{axis}_km"] = result[axis].astype(float) * AU_KM
    return table


def build_master() -> pd.DataFrame:
    sun = horizons_geocenter_vectors("10", "GEOCENTER_SUN")
    venus = horizons_geocenter_vectors("299", "GEOCENTER_VENUS")
    master = sun.merge(venus, on=["jd_tdb", "utc"], how="inner")
    if len(master) < 100:
        raise RuntimeError("JPL geocentric master contains too few rows.")
    return master


def build_cache(master: pd.DataFrame) -> dict[str, object]:
    cache: dict[str, object] = {
        "jd_tdb": master["jd_tdb"].to_numpy(dtype=float),
        "utc": master["utc"].astype(str).tolist(),
    }
    for column in master.columns:
        if column.endswith("_km"):
            cache[column] = CubicSpline(
                cache["jd_tdb"],
                master[column].to_numpy(dtype=float),
                bc_type="natural",
            )
    return cache


def vector_at(cache: dict[str, object], prefix: str, jd_tdb: float) -> np.ndarray:
    return np.array(
        [
            float(cache[f"{prefix}_x_km"](jd_tdb)),
            float(cache[f"{prefix}_y_km"](jd_tdb)),
            float(cache[f"{prefix}_z_km"](jd_tdb)),
        ],
        dtype=float,
    )


def distances(cache: dict[str, object], jd_tdb: float) -> tuple[float, float, float]:
    sun = vector_at(cache, "GEOCENTER_SUN", jd_tdb)
    venus = vector_at(cache, "GEOCENTER_VENUS", jd_tdb)
    return norm(sun), norm(venus), norm(sun - venus)


def apparent_radii_arcsec(cache: dict[str, object], jd_tdb: float) -> tuple[float, float]:
    earth_sun_km, earth_venus_km, _ = distances(cache, jd_tdb)
    sun_radius = math.atan2(SUN_RADIUS_KM, earth_sun_km) * ARCSEC_PER_RAD
    venus_radius = math.atan2(VENUS_RADIUS_KM, earth_venus_km) * ARCSEC_PER_RAD
    return sun_radius, venus_radius


def separation_arcsec(cache: dict[str, object], jd_tdb: float) -> float:
    return angular_sep_arcsec(
        vector_at(cache, "GEOCENTER_SUN", jd_tdb),
        vector_at(cache, "GEOCENTER_VENUS", jd_tdb),
    )


def contact_value(cache: dict[str, object], kind: str, jd_tdb: float) -> float:
    separation = separation_arcsec(cache, jd_tdb)
    sun_radius, venus_radius = apparent_radii_arcsec(cache, jd_tdb)
    threshold = sun_radius + venus_radius if kind == "OUTER" else sun_radius - venus_radius
    return separation - threshold


def roots_for(cache: dict[str, object], kind: str) -> list[float]:
    jds = np.asarray(cache["jd_tdb"], dtype=float)
    values = np.array([contact_value(cache, kind, jd) for jd in jds], dtype=float)
    roots: list[float] = []
    for index in range(len(jds) - 1):
        first = values[index]
        second = values[index + 1]
        if not np.isfinite(first) or not np.isfinite(second):
            continue
        if first == 0.0:
            roots.append(float(jds[index]))
        elif first * second < 0.0:
            root = brentq(
                lambda value: contact_value(cache, kind, value),
                float(jds[index]),
                float(jds[index + 1]),
                xtol=1.0e-13,
                rtol=1.0e-13,
                maxiter=100,
            )
            roots.append(float(root))
    return sorted(roots)


def contacts(cache: dict[str, object]) -> dict[str, float]:
    outer = roots_for(cache, "OUTER")
    inner = roots_for(cache, "INNER")
    if len(outer) < 2 or len(inner) < 2:
        raise RuntimeError(
            f"Could not derive four geocentric contacts: outer={len(outer)}, inner={len(inner)}"
        )
    return {"C1": outer[0], "C2": inner[0], "C3": inner[-1], "C4": outer[-1]}


def closest_approach(cache: dict[str, object]) -> float:
    jds = np.asarray(cache["jd_tdb"], dtype=float)
    values = np.array([separation_arcsec(cache, jd) for jd in jds], dtype=float)
    minimum_index = int(np.argmin(values))
    lower = float(jds[max(0, minimum_index - 3)])
    upper = float(jds[min(len(jds) - 1, minimum_index + 3)])
    result = minimize_scalar(
        lambda value: separation_arcsec(cache, value),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1.0e-13},
    )
    if not result.success:
        raise RuntimeError("Geocentric closest-approach minimization failed.")
    return float(result.x)


def fixed_screen_basis(cache: dict[str, object], jd_tdb: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sun = vector_at(cache, "GEOCENTER_SUN", jd_tdb)
    normal = unit(sun)
    north_reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if norm(np.cross(north_reference, normal)) < 1.0e-12:
        north_reference = np.array([1.0, 0.0, 0.0], dtype=float)
    x_axis = unit(np.cross(north_reference, normal))
    y_axis = unit(np.cross(normal, x_axis))
    return normal, x_axis, y_axis


def screen_point_arcsec(
    cache: dict[str, object],
    jd_tdb: float,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    normal, x_axis, y_axis = basis
    sun = vector_at(cache, "GEOCENTER_SUN", jd_tdb)
    venus = vector_at(cache, "GEOCENTER_VENUS", jd_tdb)
    denominator = float(np.dot(venus, normal))
    if abs(denominator) < 1.0e-14:
        raise RuntimeError("Venus ray is nearly parallel to the geocentric solar screen.")
    ray_scale = float(np.dot(sun, normal) / denominator)
    hit = ray_scale * venus
    offset = hit - sun
    earth_sun_km = norm(sun)
    x_arcsec = math.atan2(float(np.dot(offset, x_axis)), earth_sun_km) * ARCSEC_PER_RAD
    y_arcsec = math.atan2(float(np.dot(offset, y_axis)), earth_sun_km) * ARCSEC_PER_RAD
    return np.array([x_arcsec, y_arcsec], dtype=float)


def pca_direction(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    centered = points - center
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    direction = unit(right_vectors[0])
    if direction[0] < 0.0:
        direction = -direction
    return center, direction


def r_squared(observed: np.ndarray, fitted: np.ndarray) -> float:
    residual = float(np.sum((observed - fitted) ** 2))
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    if total <= 0.0:
        return 1.0
    return 1.0 - residual / total


def build_geometry(cache: dict[str, object]) -> dict[str, object]:
    event_jds = contacts(cache)
    closest_jd = closest_approach(cache)
    basis = fixed_screen_basis(cache, closest_jd)

    minute_jds = np.asarray(cache["jd_tdb"], dtype=float)
    transit_minute_jds = minute_jds[
        (minute_jds >= event_jds["C1"]) & (minute_jds <= event_jds["C4"])
    ]
    all_jds = sorted(
        set(
            transit_minute_jds.tolist()
            + [event_jds["C1"], event_jds["C2"], closest_jd, event_jds["C3"], event_jds["C4"]]
        )
    )
    points = np.array([screen_point_arcsec(cache, jd, basis) for jd in all_jds], dtype=float)
    center, direction = pca_direction(points)
    normal = np.array([-direction[1], direction[0]], dtype=float)

    x_values = points[:, 0]
    y_values = points[:, 1]
    linear_slope, linear_intercept = np.polyfit(x_values, y_values, 1)
    y_linear = linear_slope * x_values + linear_intercept
    ols_angle = math.degrees(math.atan(linear_slope))
    tls_angle = math.degrees(math.atan2(direction[1], direction[0]))

    centered = points - center
    along = centered @ direction
    cross = centered @ normal
    rms_perpendicular = float(np.sqrt(np.mean(cross**2)))
    rms_vertical = float(np.sqrt(np.mean((y_values - y_linear) ** 2)))

    quadratic = np.polyfit(along, cross, 2)
    quadratic_fit = np.polyval(quadratic, along)
    quadratic_r2 = r_squared(cross, quadratic_fit)
    linear_cross_fit = np.polyval(np.polyfit(along, cross, 1), along)
    linear_cross_r2 = r_squared(cross, linear_cross_fit)
    cubic_fit = np.polyval(np.polyfit(along, cross, 3), along)
    cubic_r2 = r_squared(cross, cubic_fit)
    curvature = float(2.0 * quadratic[0] / (1.0 + quadratic[1] ** 2) ** 1.5)

    event_names = ["C1", "C2", "CA", "C3", "C4"]
    event_times = {
        "C1": event_jds["C1"],
        "C2": event_jds["C2"],
        "CA": closest_jd,
        "C3": event_jds["C3"],
        "C4": event_jds["C4"],
    }
    events: dict[str, Event] = {}
    for name in event_names:
        jd = event_times[name]
        sun_radius, venus_radius = apparent_radii_arcsec(cache, jd)
        events[name] = Event(
            name=name,
            jd_tdb=jd,
            utc=utc_at(jd),
            point=screen_point_arcsec(cache, jd, basis),
            venus_radius_arcsec=venus_radius,
            sun_radius_arcsec=sun_radius,
        )

    earth_sun_km, earth_venus_km, venus_sun_km = distances(cache, closest_jd)
    minimum_separation = separation_arcsec(cache, closest_jd)
    track_chord_c1_c4 = norm(events["C4"].point - events["C1"].point)
    interior_chord_c2_c3 = norm(events["C3"].point - events["C2"].point)

    track_table = pd.DataFrame(
        {
            "version": VERSION,
            "jd_tdb": all_jds,
            "utc": [utc_at(jd) for jd in all_jds],
            "x_arcsec": points[:, 0],
            "y_arcsec": points[:, 1],
            "along_track_arcsec": along,
            "cross_track_arcsec": cross,
            "linear_y_arcsec": y_linear,
            "linear_vertical_residual_arcsec": y_values - y_linear,
        }
    )
    track_table.to_csv(TRACK_CSV, index=False, float_format="%.12f")

    return {
        "basis": basis,
        "jds": np.asarray(all_jds, dtype=float),
        "points": points,
        "center": center,
        "direction": direction,
        "normal": normal,
        "events": events,
        "tls_angle_deg": tls_angle,
        "ols_angle_deg": ols_angle,
        "angle_delta_deg": abs(tls_angle - ols_angle),
        "linear_slope": float(linear_slope),
        "linear_intercept": float(linear_intercept),
        "linear_r2": r_squared(y_values, y_linear),
        "linear_cross_r2": linear_cross_r2,
        "quadratic_r2": quadratic_r2,
        "cubic_r2": cubic_r2,
        "rms_perpendicular_arcsec": rms_perpendicular,
        "rms_vertical_arcsec": rms_vertical,
        "curvature_arcsec_inverse": curvature,
        "closest_jd": closest_jd,
        "closest_utc": utc_at(closest_jd),
        "minimum_separation_arcsec": minimum_separation,
        "sun_radius_arcsec": events["CA"].sun_radius_arcsec,
        "venus_radius_arcsec": events["CA"].venus_radius_arcsec,
        "earth_sun_km": earth_sun_km,
        "earth_sun_au": earth_sun_km / AU_KM,
        "earth_venus_km": earth_venus_km,
        "venus_sun_km": venus_sun_km,
        "track_chord_c1_c4_arcsec": track_chord_c1_c4,
        "interior_chord_c2_c3_arcsec": interior_chord_c2_c3,
        "video_tls_delta_deg": tls_angle - VIDEO_TLS_COMPARISON_DEG,
        "video_ols_delta_deg": ols_angle - VIDEO_OLS_COMPARISON_DEG,
    }


def add_label(ax: plt.Axes, point: np.ndarray, text: str, dx: float, dy: float, color: str) -> None:
    ax.annotate(
        text,
        xy=(point[0], point[1]),
        xytext=(point[0] + dx, point[1] + dy),
        textcoords="data",
        fontsize=5.7,
        color=color,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "lw": 0.20, "color": color, "shrinkA": 0, "shrinkB": 2},
        zorder=10,
    )


def format_value(value: object, digits: int = 6) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def add_compact_summary(ax: plt.Axes, geometry: dict[str, object]) -> None:
    rows = [
        ["TLS track angle", format_value(geometry["tls_angle_deg"]), "deg"],
        ["OLS track angle", format_value(geometry["ols_angle_deg"]), "deg"],
        ["TLS-OLS delta", format_value(geometry["angle_delta_deg"]), "deg"],
        ["Linear R²", format_value(geometry["linear_r2"], 9), ""],
        ["Perpendicular RMS", format_value(geometry["rms_perpendicular_arcsec"], 6), "arcsec"],
        ["Closest separation", format_value(geometry["minimum_separation_arcsec"], 6), "arcsec"],
        ["C1-C4 chord", format_value(geometry["track_chord_c1_c4_arcsec"], 6), "arcsec"],
        ["D ES", format_value(geometry["earth_sun_au"], 9), "AU"],
        ["SDO-video TLS", format_value(VIDEO_TLS_COMPARISON_DEG), "deg"],
        ["JPL - video TLS", format_value(geometry["video_tls_delta_deg"]), "deg"],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["Quantity", "Value", "Unit"],
        loc="lower left",
        colWidths=[0.32, 0.22, 0.12],
        bbox=[0.545, 0.095, 0.405, 0.36],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.6)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(SPINE)
        cell.set_linewidth(0.20)
        cell.set_facecolor(PANEL if row > 0 else "#0b1a22")
        cell.get_text().set_color(TEXT if row == 0 else MUTED)
        if row == 0:
            cell.get_text().set_weight("bold")


def axis_limits_half_sun(solar_radius: float, points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    x_limits = (-1.04 * solar_radius, 1.04 * solar_radius)
    median_y = float(np.median(points[:, 1]))
    if median_y >= 0.0:
        y_limits = (-0.06 * solar_radius, 1.06 * solar_radius)
    else:
        y_limits = (-1.06 * solar_radius, 0.06 * solar_radius)
    minimum_y = float(np.min(points[:, 1]))
    maximum_y = float(np.max(points[:, 1]))
    padding = 0.08 * solar_radius
    y_limits = (min(y_limits[0], minimum_y - padding), max(y_limits[1], maximum_y + padding))
    return x_limits, y_limits


def render_track_png(geometry: dict[str, object]) -> None:
    points = np.asarray(geometry["points"], dtype=float)
    events: dict[str, Event] = geometry["events"]
    solar_radius = float(geometry["sun_radius_arcsec"])
    theta = np.linspace(0.0, 2.0 * np.pi, 1800)

    figure, ax = plt.subplots(figsize=(10.5, 7.2), dpi=180)
    figure.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.plot(
        solar_radius * np.cos(theta),
        solar_radius * np.sin(theta),
        color=SOLAR,
        lw=0.34,
        alpha=0.95,
        zorder=1,
    )
    ax.axhline(0.0, color=SPINE, lw=0.18, alpha=0.55, zorder=0)
    ax.axvline(0.0, color=SPINE, lw=0.18, alpha=0.55, zorder=0)

    ax.plot(
        points[:, 0],
        points[:, 1],
        color=TRACK,
        lw=0.34,
        solid_capstyle="round",
        label="JPL geocenter",
        zorder=3,
    )
    ax.scatter(
        points[::6, 0],
        points[::6, 1],
        s=0.9,
        color=TRACK,
        alpha=0.72,
        linewidths=0,
        zorder=4,
    )

    for name in ("C1", "C2", "CA", "C3", "C4"):
        event = events[name]
        line_width = 0.30 if name == "CA" else 0.22
        ax.add_patch(
            Circle(
                (event.point[0], event.point[1]),
                event.venus_radius_arcsec,
                fill=False,
                lw=line_width,
                ec=VENUS,
                alpha=0.96,
                zorder=5,
            )
        )
        ax.scatter(
            [event.point[0]],
            [event.point[1]],
            s=4.0 if name == "CA" else 2.4,
            color=TRACK,
            edgecolors=BG,
            linewidths=0.18,
            zorder=6,
        )

    label_offsets = {
        "C1": (-55.0, 16.0),
        "C2": (-43.0, 12.0),
        "CA": (20.0, 17.0),
        "C3": (22.0, -12.0),
        "C4": (34.0, -16.0),
    }
    for name, (dx, dy) in label_offsets.items():
        label = "Geocenter CA" if name == "CA" else name
        add_label(ax, events[name].point, label, dx, dy, ACCENT if name == "CA" else MUTED)

    add_compact_summary(ax, geometry)
    x_limits, y_limits = axis_limits_half_sun(solar_radius, points)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal", adjustable="box")

    for spine in ax.spines.values():
        spine.set_linewidth(0.22)
        spine.set_color(SPINE)
    ax.tick_params(axis="both", colors=MUTED, labelsize=6.5, width=0.22, length=2.0)
    ax.grid(True, color=GRID, linewidth=0.16, alpha=0.55)
    ax.set_xlabel("Solar-screen X offset (arcsec)", color=MUTED, fontsize=7.5)
    ax.set_ylabel("Solar-screen Y offset (arcsec)", color=MUTED, fontsize=7.5)
    ax.set_title(
        "2012 Venus Transit — Geocentric Engineering Half-Sun Reconstruction\n"
        "IERS-0012AA visual format — JPL Horizons 500@399 — one fixed solar screen",
        color="#f8fdff",
        fontsize=9.2,
        pad=8,
    )
    legend = ax.legend(loc="lower right", fontsize=6.3, frameon=True, borderpad=0.45)
    legend.get_frame().set_facecolor(PANEL)
    legend.get_frame().set_edgecolor(SPINE)
    legend.get_frame().set_linewidth(0.22)
    for legend_text in legend.get_texts():
        legend_text.set_color(TEXT)

    ax.text(
        0.015,
        0.018,
        "Venus disks plotted to JPL-derived apparent angular scale.  Minute-by-minute C1-C4 fit.",
        transform=ax.transAxes,
        fontsize=5.5,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    figure.savefig(TRACK_PNG, dpi=320, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)


def results_rows(geometry: dict[str, object]) -> list[tuple[str, str, str, str]]:
    events: dict[str, Event] = geometry["events"]
    rows: list[tuple[str, str, str, str]] = [
        ("CODE INPUTS", "JPL observer", JPL_LOCATION, "VERIFIED"),
        ("CODE INPUTS", "JPL interval", f"{START} to {STOP}", "INPUT"),
        ("CODE INPUTS", "JPL step", STEP, "INPUT"),
        ("COMMENTS", "Geometry", "Geocenter; one fixed solar screen", "VERIFIED"),
        ("COMMENTS", "Video spacecraft estimate", SATELLITE_ESTIMATE, "COMPARISON ONLY"),
        ("COMMENTS", "SDO observer geometry", "NOT USED in this geocentric solution", "NOT USED"),
        ("RESULTS", "TLS track angle", f"{geometry['tls_angle_deg']:.9f} deg", "VERIFIED"),
        ("RESULTS", "OLS track angle", f"{geometry['ols_angle_deg']:.9f} deg", "VERIFIED"),
        ("RESULTS", "TLS-OLS delta", f"{geometry['angle_delta_deg']:.9f} deg", "VERIFIED"),
        ("RESULTS", "Linear slope y/x", f"{geometry['linear_slope']:.12f}", "VERIFIED"),
        ("RESULTS", "Linear R²", f"{geometry['linear_r2']:.12f}", "VERIFIED"),
        ("RESULTS", "Cross-track linear R²", f"{geometry['linear_cross_r2']:.12f}", "VERIFIED"),
        ("RESULTS", "Quadratic R²", f"{geometry['quadratic_r2']:.12f}", "VERIFIED"),
        ("RESULTS", "Cubic R²", f"{geometry['cubic_r2']:.12f}", "VERIFIED"),
        ("RESULTS", "Perpendicular RMS", f"{geometry['rms_perpendicular_arcsec']:.9f} arcsec", "VERIFIED"),
        ("RESULTS", "Vertical RMS", f"{geometry['rms_vertical_arcsec']:.9f} arcsec", "VERIFIED"),
        ("RESULTS", "Curvature", f"{geometry['curvature_arcsec_inverse']:.12e} arcsec^-1", "DIAGNOSTIC"),
        ("RESULTS", "Closest approach UTC", str(geometry["closest_utc"]), "VERIFIED"),
        ("RESULTS", "Minimum center separation", f"{geometry['minimum_separation_arcsec']:.9f} arcsec", "VERIFIED"),
        ("RESULTS", "Solar angular radius at CA", f"{geometry['sun_radius_arcsec']:.9f} arcsec", "VERIFIED"),
        ("RESULTS", "Venus angular radius at CA", f"{geometry['venus_radius_arcsec']:.9f} arcsec", "VERIFIED"),
        ("RESULTS", "C1-C4 center chord", f"{geometry['track_chord_c1_c4_arcsec']:.9f} arcsec", "VERIFIED"),
        ("RESULTS", "C2-C3 center chord", f"{geometry['interior_chord_c2_c3_arcsec']:.9f} arcsec", "VERIFIED"),
        ("RESULTS", "Earth-Sun distance", f"{geometry['earth_sun_km']:.6f} km", "JPL"),
        ("RESULTS", "Earth-Sun distance", f"{geometry['earth_sun_au']:.12f} AU", "JPL"),
        ("RESULTS", "Earth-Venus distance", f"{geometry['earth_venus_km']:.6f} km", "JPL"),
        ("RESULTS", "Venus-Sun distance", f"{geometry['venus_sun_km']:.6f} km", "JPL"),
    ]
    for name in ("C1", "C2", "CA", "C3", "C4"):
        rows.append(("RESULTS", f"{name} UTC", events[name].utc, "JPL-DERIVED"))
    rows.extend(
        [
            ("PAPER COMPARISON", f"{VIDEO_COMPARISON_VERSION} SDO-video TLS", f"{VIDEO_TLS_COMPARISON_DEG:.9f} deg", "COMPARISON"),
            ("PAPER COMPARISON", "JPL geocenter - video TLS", f"{geometry['video_tls_delta_deg']:+.9f} deg", "COMPARISON"),
            ("PAPER COMPARISON", f"{VIDEO_COMPARISON_VERSION} SDO-video OLS", f"{VIDEO_OLS_COMPARISON_DEG:.9f} deg", "COMPARISON"),
            ("PAPER COMPARISON", "JPL geocenter - video OLS", f"{geometry['video_ols_delta_deg']:+.9f} deg", "COMPARISON"),
            ("EQUATION STATUS", "Contact roots", "sep = R_sun ± R_venus", "VERIFIED"),
            ("EQUATION STATUS", "Screen projection", "Venus ray to fixed Sun plane", "VERIFIED"),
            ("EQUATION STATUS", "Track direction", "SVD/PCA over minute-by-minute C1-C4", "VERIFIED"),
            ("EQUATION STATUS", "Angle", "atan2(direction_y, direction_x)", "VERIFIED"),
            ("EQUATION STATUS", "No topocentric observer", "Geocenter only", "VERIFIED"),
        ]
    )
    return rows


def render_results_png(geometry: dict[str, object]) -> None:
    rows = results_rows(geometry)
    figure = plt.figure(figsize=(12.5, 15.5), dpi=170, facecolor=BG)
    ax = figure.add_axes([0.035, 0.025, 0.93, 0.94])
    ax.set_facecolor(BG)
    ax.axis("off")

    ax.text(
        0.0,
        1.025,
        "2012 VENUS TRANSIT — GEOCENTRIC JPL RESULTS",
        transform=ax.transAxes,
        color="#f8fdff",
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.0,
        0.992,
        "IERS-0012AA engineering format | all reported values rendered to PNG",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9,
        va="top",
    )

    cell_text = [[section, quantity, value, status] for section, quantity, value, status in rows]
    table = ax.table(
        cellText=cell_text,
        colLabels=["Section", "Quantity", "Value", "Status"],
        colWidths=[0.16, 0.29, 0.37, 0.18],
        bbox=[0.0, 0.02, 1.0, 0.94],
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.1)
    section_colors = {
        "CODE INPUTS": "#0c2530",
        "COMMENTS": "#10231d",
        "RESULTS": "#071016",
        "PAPER COMPARISON": "#241d0f",
        "EQUATION STATUS": "#11172b",
    }
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(SPINE)
        cell.set_linewidth(0.22)
        if row == 0:
            cell.set_facecolor("#0b1a22")
            cell.get_text().set_color(TEXT)
            cell.get_text().set_weight("bold")
        else:
            section = cell_text[row - 1][0]
            cell.set_facecolor(section_colors.get(section, PANEL))
            cell.get_text().set_color(TEXT if column in (0, 1) else MUTED)
            if column == 3:
                status = cell_text[row - 1][3]
                if "REJECTED" in status or "NOT USED" in status:
                    cell.get_text().set_color(REJECTED)
                elif "VERIFIED" in status or "JPL" in status:
                    cell.get_text().set_color(TRACK)
                else:
                    cell.get_text().set_color(ACCENT)

    ax.text(
        0.0,
        0.0,
        f"Generated {datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S %z')}  |  {VERSION}",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=7,
        va="bottom",
    )
    figure.savefig(RESULTS_PNG, dpi=300, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    master = build_master()
    cache = build_cache(master)
    geometry = build_geometry(cache)
    render_track_png(geometry)
    render_results_png(geometry)

    print("CODE INPUTS")
    print("JPL geocenter 500@399; 2012 transit; one-minute vectors")
    print("COMMENTS")
    print("All numerical tables are rendered inside PNG files; SDO estimate is comparison only")
    print("RESULTS")
    print(f"Track PNG: {TRACK_PNG}")
    print(f"Results PNG: {RESULTS_PNG}")
    print("OUTPUT SUMMARY")
    print(f"PNG files: {TRACK_PNG} | {RESULTS_PNG}")
    print("PAPER COMPARISON")
    print(f"SDO video comparison rendered in {RESULTS_PNG}")
    print("EQUATION STATUS")
    print("All equations and statuses rendered in results PNG")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# IERS-0012AC
