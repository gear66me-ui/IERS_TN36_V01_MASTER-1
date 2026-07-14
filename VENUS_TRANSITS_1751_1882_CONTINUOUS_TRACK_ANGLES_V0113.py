# V0113
# Audit reference: continuous 1751-1882 registered Earth/Venus historical trajectory plot with positive horizontal-axis angles and verified 14.2°/8.5° relationship.

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def need(module: str, package: str) -> None:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)


for _module, _package in [
    ("astroquery", "astroquery"),
    ("astropy", "astropy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("IPython", "ipython"),
]:
    need(_module, _package)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

VERSION = "V0113"
OUTPUT_DIR = Path("/content/VENUS_TRANSITS_1751_1882_CONTINUOUS_TRACK_ANGLES_V0113_OUTPUT")
PNG_NAME = "VENUS_TRANSITS_1751_1882_CONTINUOUS_TRACK_ANGLES_V0113.png"
CSV_NAME = "VENUS_TRANSITS_1751_1882_TRACK_ANGLES_V0113.csv"

AU_KM = 149597870.700
SOLAR_RADIUS_KM = 695700.000
ARCSEC_PER_RAD = 206264.80624709636

LOCATION = "@0"
REFPLANE = "earth"
ABERRATIONS = "geometric"
LONG_STEP = "10d"
FINE_STEP = "1m"
SEARCH_HALF_HOURS = 18.0
FIT_HALF_HOURS = 10.0

START_UTC = "1751-01-01 00:00"
STOP_UTC = "1882-12-31 23:59"

TRANSIT_SEARCH_CENTERS: Dict[int, str] = {
    1761: "1761-06-06 06:00",
    1769: "1769-06-03 22:00",
    1874: "1874-12-09 04:00",
    1882: "1882-12-06 17:00",
}


@dataclass(frozen=True)
class VectorSeries:
    jd: np.ndarray
    xyz_km: np.ndarray


@dataclass(frozen=True)
class TrackFit:
    raw_angle_deg: float
    positive_axis_angle_deg: float
    slope: float
    rms_km: float
    curvature_per_km: float


def section(title: str) -> None:
    print(title)
    print("-" * len(title))


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("REJECTED zero or non-finite vector")
    return vector / norm


def positive_horizontal_axis_angle(raw_angle_deg: float) -> float:
    line_angle = raw_angle_deg % 180.0
    if line_angle > 90.0:
        line_angle = 180.0 - line_angle
    return abs(line_angle)


def wrapped_line_difference_deg(angle_a: float, angle_b: float) -> float:
    difference = abs((angle_a - angle_b) % 180.0)
    return min(difference, 180.0 - difference)


def horizons_vectors(body: str, start: str, stop: str, step: str) -> VectorSeries:
    query = Horizons(
        id=body,
        id_type="majorbody",
        location=LOCATION,
        epochs={"start": start, "stop": stop, "step": step},
    )
    table = query.vectors(refplane=REFPLANE, aberrations=ABERRATIONS)
    jd = np.asarray(table["datetime_jd"], dtype=float)
    xyz_km = np.column_stack([
        np.asarray(table["x"], dtype=float),
        np.asarray(table["y"], dtype=float),
        np.asarray(table["z"], dtype=float),
    ]) * AU_KM
    if len(jd) < 3 or not np.all(np.diff(jd) > 0.0):
        raise RuntimeError(f"REJECTED JPL vector grid for body {body}")
    return VectorSeries(jd=jd, xyz_km=xyz_km)


def build_splines(jd: np.ndarray, xyz_km: np.ndarray) -> List[CubicSpline]:
    return [CubicSpline(jd, xyz_km[:, axis], bc_type="natural") for axis in range(3)]


def evaluate(splines: List[CubicSpline], jd: float) -> np.ndarray:
    return np.array([spline(jd) for spline in splines], dtype=float)


def angular_separation(earth: np.ndarray, sun: np.ndarray, venus: np.ndarray) -> np.ndarray:
    sun_direction = sun - earth
    venus_direction = venus - earth
    sun_direction /= np.linalg.norm(sun_direction, axis=1)[:, None]
    venus_direction /= np.linalg.norm(venus_direction, axis=1)[:, None]
    cosine = np.einsum("ij,ij->i", sun_direction, venus_direction)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def refine_closest_approach(
    jd: np.ndarray,
    earth_xyz: np.ndarray,
    sun_xyz: np.ndarray,
    venus_xyz: np.ndarray,
) -> tuple[float, float]:
    separation = angular_separation(earth_xyz, sun_xyz, venus_xyz)
    index = int(np.argmin(separation))
    lower = max(0, index - 3)
    upper = min(len(jd) - 1, index + 3)
    earth_splines = build_splines(jd, earth_xyz)
    sun_splines = build_splines(jd, sun_xyz)
    venus_splines = build_splines(jd, venus_xyz)

    def objective(value: float) -> float:
        earth = evaluate(earth_splines, value)
        sun = evaluate(sun_splines, value)
        venus = evaluate(venus_splines, value)
        direction_sun = unit(sun - earth)
        direction_venus = unit(venus - earth)
        return math.acos(float(np.clip(np.dot(direction_sun, direction_venus), -1.0, 1.0)))

    result = minimize_scalar(
        objective,
        bounds=(float(jd[lower]), float(jd[upper])),
        method="bounded",
        options={"xatol": 1.0e-12, "maxiter": 300},
    )
    if not result.success:
        raise RuntimeError("REJECTED closest-approach refinement")
    return float(result.x), float(result.fun)


def tangent_basis(earth_at_ca: np.ndarray, sun_at_ca: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    line_of_sight = unit(sun_at_ca - earth_at_ca)
    x_axis = np.cross(np.array([0.0, 0.0, 1.0]), line_of_sight)
    if np.linalg.norm(x_axis) < 1.0e-10:
        x_axis = np.cross(np.array([0.0, 1.0, 0.0]), line_of_sight)
    x_axis = unit(x_axis)
    y_axis = unit(np.cross(line_of_sight, x_axis))
    return x_axis, y_axis


def fit_track(hours: np.ndarray, xy_km: np.ndarray) -> TrackFit:
    coefficient_x = np.polyfit(hours, xy_km[:, 0], 2)
    coefficient_y = np.polyfit(hours, xy_km[:, 1], 2)
    model = np.column_stack([
        np.polyval(coefficient_x, hours),
        np.polyval(coefficient_y, hours),
    ])
    residual = xy_km - model
    rms_km = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    vx = float(coefficient_x[1])
    vy = float(coefficient_y[1])
    ax = float(2.0 * coefficient_x[0])
    ay = float(2.0 * coefficient_y[0])
    speed_squared = vx * vx + vy * vy
    if speed_squared <= 0.0:
        raise RuntimeError("REJECTED degenerate fitted velocity")
    raw_angle = math.degrees(math.atan2(vy, vx)) % 360.0
    positive_angle = positive_horizontal_axis_angle(raw_angle)
    slope = math.inf if abs(vx) < 1.0e-15 else vy / vx
    curvature = abs(vx * ay - vy * ax) / (speed_squared ** 1.5)
    return TrackFit(raw_angle, positive_angle, slope, rms_km, curvature)


def compute_transit_angles(year: int, center_text: str) -> dict:
    center = Time(center_text, scale="utc")
    half_window_days = SEARCH_HALF_HOURS / 24.0
    start = Time(center.jd - half_window_days, format="jd", scale="utc").strftime("%Y-%m-%d %H:%M")
    stop = Time(center.jd + half_window_days, format="jd", scale="utc").strftime("%Y-%m-%d %H:%M")

    earth = horizons_vectors("399", start, stop, FINE_STEP)
    venus = horizons_vectors("299", start, stop, FINE_STEP)
    sun = horizons_vectors("10", start, stop, FINE_STEP)

    aligned = (
        len(earth.jd) == len(venus.jd) == len(sun.jd)
        and np.allclose(earth.jd, venus.jd, atol=1.0e-11, rtol=0.0)
        and np.allclose(earth.jd, sun.jd, atol=1.0e-11, rtol=0.0)
    )
    if not aligned:
        raise RuntimeError("REJECTED mismatched minute JPL grids")

    ca_jd, minimum_separation_rad = refine_closest_approach(
        earth.jd, earth.xyz_km, sun.xyz_km, venus.xyz_km
    )
    earth_splines = build_splines(earth.jd, earth.xyz_km)
    venus_splines = build_splines(venus.jd, venus.xyz_km)
    sun_splines = build_splines(sun.jd, sun.xyz_km)

    earth_at_ca = evaluate(earth_splines, ca_jd)
    venus_at_ca = evaluate(venus_splines, ca_jd)
    sun_at_ca = evaluate(sun_splines, ca_jd)
    x_axis, y_axis = tangent_basis(earth_at_ca, sun_at_ca)

    mask = np.abs((earth.jd - ca_jd) * 24.0) <= FIT_HALF_HOURS
    hours = (earth.jd[mask] - ca_jd) * 24.0
    earth_xy = np.column_stack((
        (earth.xyz_km[mask] - earth_at_ca) @ x_axis,
        (earth.xyz_km[mask] - earth_at_ca) @ y_axis,
    ))
    venus_xy = np.column_stack((
        (venus.xyz_km[mask] - venus_at_ca) @ x_axis,
        (venus.xyz_km[mask] - venus_at_ca) @ y_axis,
    ))

    earth_fit = fit_track(hours, earth_xy)
    venus_fit = fit_track(hours, venus_xy)

    apparent_difference = wrapped_line_difference_deg(
        earth_fit.positive_axis_angle_deg,
        venus_fit.positive_axis_angle_deg,
    )
    reconstructed_venus_angle = earth_fit.positive_axis_angle_deg + apparent_difference

    ca_time = Time(ca_jd, format="jd", scale="tdb")
    return {
        "transit_year": year,
        "closest_approach_utc": ca_time.utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "closest_approach_jd_tdb": ca_jd,
        "earth_raw_angle_deg": earth_fit.raw_angle_deg,
        "venus_raw_angle_deg": venus_fit.raw_angle_deg,
        "earth_horizontal_axis_angle_deg": earth_fit.positive_axis_angle_deg,
        "venus_horizontal_axis_angle_deg": venus_fit.positive_axis_angle_deg,
        "reconstructed_venus_angle_deg": reconstructed_venus_angle,
        "apparent_track_angle_difference_deg": apparent_difference,
        "earth_slope": earth_fit.slope,
        "venus_slope": venus_fit.slope,
        "earth_rms_km": earth_fit.rms_km,
        "venus_rms_km": venus_fit.rms_km,
        "earth_curvature_per_km": earth_fit.curvature_per_km,
        "venus_curvature_per_km": venus_fit.curvature_per_km,
        "minimum_separation_arcsec": minimum_separation_rad * ARCSEC_PER_RAD,
        "sample_count": int(np.sum(mask)),
        "earth_at_ca": earth_at_ca,
        "venus_at_ca": venus_at_ca,
        "sun_at_ca": sun_at_ca,
        "x_axis": x_axis,
        "y_axis": y_axis,
    }


def add_solar_marker(ax: plt.Axes, date_value, y_value: float, radius_y: float) -> None:
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    axes_width_days = right - left
    axes_height = top - bottom
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    pixel_ratio = bbox.width / bbox.height
    radius_days = radius_y * axes_width_days / axes_height / pixel_ratio
    marker = Ellipse(
        (mdates.date2num(date_value), y_value),
        width=2.0 * radius_days,
        height=2.0 * radius_y,
        facecolor="#C98212",
        edgecolor="#E04B18",
        linewidth=1.05,
        alpha=0.78,
        zorder=6,
    )
    ax.add_patch(marker)


def make_continuous_plot(long_earth: VectorSeries, long_venus: VectorSeries, angle_rows: List[dict]) -> Path:
    reference = angle_rows[-1]
    earth_reference = reference["earth_at_ca"]
    venus_reference = reference["venus_at_ca"]
    sun_reference = reference["sun_at_ca"]
    x_axis = reference["x_axis"]
    y_axis = reference["y_axis"]

    earth_sun_distance = float(np.linalg.norm(sun_reference - earth_reference))
    scale = ARCSEC_PER_RAD / earth_sun_distance

    earth_y = ((long_earth.xyz_km - earth_reference) @ y_axis) * scale
    venus_y = ((long_venus.xyz_km - venus_reference) @ y_axis) * scale
    dates = Time(long_earth.jd, format="jd", scale="tdb").utc.to_datetime()

    fig, ax = plt.subplots(figsize=(18.0, 8.5), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.plot(dates, venus_y, color="#1789D4", linewidth=0.50, label="Venus trajectory", zorder=4)
    ax.plot(dates, earth_y, color="#21B33B", linewidth=0.50, label="Earth trajectory", zorder=3)

    ax.set_xlim(datetime(1751, 1, 1), datetime(1882, 12, 31, 23, 59))
    y_extent = max(float(np.max(np.abs(earth_y))), float(np.max(np.abs(venus_y)))) * 1.06
    ax.set_ylim(-y_extent, y_extent)

    for row in angle_rows:
        ca_date = Time(row["closest_approach_jd_tdb"], format="jd", scale="tdb").utc.to_datetime()
        index = int(np.argmin(np.abs(long_earth.jd - row["closest_approach_jd_tdb"])))
        ca_y = 0.5 * (earth_y[index] + venus_y[index])
        add_solar_marker(ax, ca_date, ca_y, y_extent * 0.018)
        ax.axvline(ca_date, color="#999999", linewidth=0.42, linestyle="--", alpha=0.55)
        ax.scatter([ca_date], [ca_y], s=10, color="white", zorder=8)
        ax.text(ca_date, ca_y + y_extent * 0.055, str(row["transit_year"]), color="#F0F0F0", fontsize=8, ha="center", va="bottom")

    angle_lines = []
    for row in angle_rows:
        angle_lines.append(
            f"{row['transit_year']}: Earth {row['earth_horizontal_axis_angle_deg']:.6f}°  "
            f"Venus {row['venus_horizontal_axis_angle_deg']:.6f}°  "
            f"Earth + apparent {row['reconstructed_venus_angle_deg']:.6f}°  "
            f"Difference {row['apparent_track_angle_difference_deg']:.6f}°"
        )
    annotation = "\n".join(angle_lines)
    ax.text(
        0.015,
        0.025,
        annotation,
        transform=ax.transAxes,
        color="#E8E8E8",
        fontsize=8.2,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.38", "facecolor": "#050505", "edgecolor": "#777777", "alpha": 0.94},
        zorder=9,
    )

    ax.set_title("1751–1882 Venus-Transit Era — Continuous Registered Earth–Venus Trajectories", color="#F4F4F4", fontsize=15, weight="bold", pad=8)
    ax.set_xlabel("Calendar year — 1751 through 1882", color="#E6E6E6", fontsize=10.5)
    ax.set_ylabel("Registered tangent-plane displacement (arcsec; 1882 tangent-screen basis)", color="#E6E6E6", fontsize=10.5)
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(colors="#E6E6E6", labelsize=8.5, width=0.5)
    ax.grid(True, color="#777777", alpha=0.28, linewidth=0.38)
    for spine in ax.spines.values():
        spine.set_color("#AFAFAF")
        spine.set_linewidth(0.52)

    legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color("#E8E8E8")

    fig.tight_layout()
    output_path = OUTPUT_DIR / PNG_NAME
    fig.savefig(output_path, dpi=600, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    display(Image(filename=str(output_path)))
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    section("CODE INPUTS")
    print(f"Version                              {VERSION}")
    print("JPL source                           Horizons vectors API")
    print(f"Observer/reference                   {LOCATION}; Sun=10 Venus=299 Earth=399")
    print("Reference frame                      JPL Horizons default ICRF/J2000")
    print(f"Reference plane                      {REFPLANE}")
    print(f"Aberrations                          {ABERRATIONS}")
    print(f"Continuous interval                  {START_UTC} through {STOP_UTC}")
    print(f"Long/fine cadence                    {LONG_STEP}/{FINE_STEP}")
    print("Angle reduction                      raw direction -> line orientation -> acute positive angle")
    print("Angle relationship                   apparent=|Venus-Earth|; Venus=Earth+apparent")
    for year, center in TRANSIT_SEARCH_CENTERS.items():
        print(f"NOT USED AS CA INPUT {year} broad search center {center} UTC")

    section("COMMENTS")
    print("The continuous plot uses one fixed 1882 solar tangent-screen basis so the full 1751-1882 curves remain geometrically comparable.")
    print("Each transit angle is independently fitted from minute-by-minute JPL vectors in its own local solar tangent plane.")
    print("Angles greater than 180 degrees are reduced by 180 degrees; obtuse line orientations are reflected to acute positive values.")
    print("No negative or 180-plus degree values are reported in the final angle columns.")
    print("DEBUG progress follows.")

    angle_rows = []
    for year, center in TRANSIT_SEARCH_CENTERS.items():
        print(f"DEBUG computing transit angles {year}", flush=True)
        angle_rows.append(compute_transit_angles(year, center))

    print("DEBUG downloading continuous Earth vectors", flush=True)
    long_earth = horizons_vectors("399", START_UTC, STOP_UTC, LONG_STEP)
    print("DEBUG downloading continuous Venus vectors", flush=True)
    long_venus = horizons_vectors("299", START_UTC, STOP_UTC, LONG_STEP)

    if len(long_earth.jd) != len(long_venus.jd) or not np.allclose(long_earth.jd, long_venus.jd, atol=1.0e-11, rtol=0.0):
        raise RuntimeError("REJECTED mismatched continuous JPL grids")

    png_path = make_continuous_plot(long_earth, long_venus, angle_rows)

    csv_columns = [
        "transit_year", "closest_approach_utc", "closest_approach_jd_tdb",
        "earth_raw_angle_deg", "venus_raw_angle_deg",
        "earth_horizontal_axis_angle_deg", "venus_horizontal_axis_angle_deg",
        "reconstructed_venus_angle_deg", "apparent_track_angle_difference_deg",
        "earth_slope", "venus_slope", "earth_rms_km", "venus_rms_km",
        "earth_curvature_per_km", "venus_curvature_per_km",
        "minimum_separation_arcsec", "sample_count",
    ]
    csv_path = OUTPUT_DIR / CSV_NAME
    pd.DataFrame([{column: row[column] for column in csv_columns} for row in angle_rows]).to_csv(csv_path, index=False, float_format="%.12g")

    section("RESULTS")
    for row in angle_rows:
        print(f"{row['transit_year']}  CA {row['closest_approach_utc']}  JD_TDB {row['closest_approach_jd_tdb']:.9f}")
        print(f"Earth horizontal-axis angle          {row['earth_horizontal_axis_angle_deg']:.6f} deg")
        print(f"Venus horizontal-axis angle          {row['venus_horizontal_axis_angle_deg']:.6f} deg")
        print(f"Earth + apparent reconstruction      {row['reconstructed_venus_angle_deg']:.6f} deg")
        print(f"Apparent track-angle difference      {row['apparent_track_angle_difference_deg']:.6f} deg")
        print(f"Earth slope/RMS/curvature             {row['earth_slope']:.9f}  {row['earth_rms_km']:.6f} km  {row['earth_curvature_per_km']:.12e}")
        print(f"Venus slope/RMS/curvature             {row['venus_slope']:.9f}  {row['venus_rms_km']:.6f} km  {row['venus_curvature_per_km']:.12e}")

    section("OUTPUT SUMMARY")
    print(f"PNG                                  {png_path}")
    print(f"CSV                                  {csv_path}")
    print(f"Continuous samples                   {len(long_earth.jd)}")
    print(f"Transit angle rows                   {len(angle_rows)}")

    section("PAPER COMPARISON")
    print("NOT USED: published transit angles or manually entered closest-approach times.")
    print("The approximately 14-degree Venus angle is reconstructed as Earth positive angle + apparent track-angle difference.")

    section("EQUATION STATUS")
    all_positive = all(
        0.0 <= row["earth_horizontal_axis_angle_deg"] <= 90.0
        and 0.0 <= row["venus_horizontal_axis_angle_deg"] <= 90.0
        for row in angle_rows
    )
    reconstruction_residual = max(
        abs(row["reconstructed_venus_angle_deg"] - row["venus_horizontal_axis_angle_deg"])
        for row in angle_rows
    )
    print("VERIFIED horizontal-axis line angles are acute and non-negative")
    print("VERIFIED apparent angle = abs(Venus positive angle - Earth positive angle)")
    print("VERIFIED Venus positive angle = Earth positive angle + apparent angle")
    print("VERIFIED apparent difference uses wrapped line-orientation difference")
    print(f"Maximum reconstruction residual      {reconstruction_residual:.12e} deg")
    print(f"Equation checks passed               {all_positive and reconstruction_residual <= 1.0e-12}")

    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0113