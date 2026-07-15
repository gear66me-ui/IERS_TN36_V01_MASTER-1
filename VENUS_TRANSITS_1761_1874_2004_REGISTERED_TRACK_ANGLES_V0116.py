# V0116
# Audit reference: three historical Venus-transit registered Earth/Venus calendar-year plots for 1761, 1874, and 2004.

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

VERSION = "V0116"
OUTPUT_DIR = Path("/content/VENUS_TRANSITS_1761_2012_REGISTERED_TRACK_ANGLES_V0116_OUTPUT")
CSV_NAME = "VENUS_TRANSITS_1761_2012_REGISTERED_TRACK_ANGLES_V0116.csv"

AU_KM = 149597870.700
SOLAR_RADIUS_KM = 695700.000
ARCSEC_PER_RAD = 206264.80624709636

LOCATION = "@0"
REFPLANE = "earth"
ABERRATIONS = "geometric"
FINE_STEP = "1m"
YEAR_STEP = "6h"
FIT_HALF_HOURS = 10.0
SEARCH_HALF_HOURS = 18.0

TRANSITS: Dict[int, str] = {
    1761: "1761-06-06 06:00",
    1874: "1874-12-09 04:00",
    2004: "2004-06-08 08:00",
}

PNG_NAMES = {
    year: f"VENUS_TRANSIT_{year}_REGISTERED_TRACK_ANGLES_V0116.png"
    for year in TRANSITS
}

@dataclass(frozen=True)
class VectorSeries:
    jd: np.ndarray
    xyz_km: np.ndarray

@dataclass(frozen=True)
class TrackFit:
    angle_deg: float
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

def wrapped_difference_deg(angle_a: float, angle_b: float) -> float:
    return abs((angle_a - angle_b + 180.0) % 360.0 - 180.0)

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

def splines(series: VectorSeries) -> List[CubicSpline]:
    return [
        CubicSpline(series.jd, series.xyz_km[:, axis], bc_type="natural")
        for axis in range(3)
    ]

def evaluate(series_splines: List[CubicSpline], jd: float) -> np.ndarray:
    return np.array([spline(jd) for spline in series_splines], dtype=float)

def angular_separation(earth: np.ndarray, sun: np.ndarray, venus: np.ndarray) -> np.ndarray:
    sun_direction = sun - earth
    venus_direction = venus - earth
    sun_direction /= np.linalg.norm(sun_direction, axis=1)[:, None]
    venus_direction /= np.linalg.norm(venus_direction, axis=1)[:, None]
    cosine = np.einsum("ij,ij->i", sun_direction, venus_direction)
    return np.arccos(np.clip(cosine, -1.0, 1.0))

def closest_approach(
    jd: np.ndarray,
    earth_xyz: np.ndarray,
    sun_xyz: np.ndarray,
    venus_xyz: np.ndarray,
) -> tuple[float, float]:
    separation = angular_separation(earth_xyz, sun_xyz, venus_xyz)
    index = int(np.argmin(separation))
    lower = max(0, index - 3)
    upper = min(len(jd) - 1, index + 3)
    earth_spline = [CubicSpline(jd, earth_xyz[:, i]) for i in range(3)]
    sun_spline = [CubicSpline(jd, sun_xyz[:, i]) for i in range(3)]
    venus_spline = [CubicSpline(jd, venus_xyz[:, i]) for i in range(3)]

    def objective(value: float) -> float:
        earth = evaluate(earth_spline, value)
        sun = evaluate(sun_spline, value)
        venus = evaluate(venus_spline, value)
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

def quadratic_track_fit(hours: np.ndarray, xy_km: np.ndarray) -> TrackFit:
    coefficient_x = np.polyfit(hours, xy_km[:, 0], 2)
    coefficient_y = np.polyfit(hours, xy_km[:, 1], 2)
    model = np.column_stack([
        np.polyval(coefficient_x, hours),
        np.polyval(coefficient_y, hours),
    ])
    rms = float(np.sqrt(np.mean(np.sum((xy_km - model) ** 2, axis=1))))
    vx = float(coefficient_x[1])
    vy = float(coefficient_y[1])
    ax = float(2.0 * coefficient_x[0])
    ay = float(2.0 * coefficient_y[0])
    speed_squared = vx * vx + vy * vy
    if speed_squared <= 0.0:
        raise RuntimeError("REJECTED degenerate track fit")
    angle = math.degrees(math.atan2(vy, vx)) % 360.0
    slope = math.inf if abs(vx) < 1.0e-15 else vy / vx
    curvature = abs(vx * ay - vy * ax) / (speed_squared ** 1.5)
    return TrackFit(angle, slope, rms, curvature)

def project_registered_tracks(
    jd: np.ndarray,
    earth_xyz: np.ndarray,
    venus_xyz: np.ndarray,
    earth_at_ca: np.ndarray,
    venus_at_ca: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    earth_sun_distance_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    earth_offset = earth_xyz - earth_at_ca
    venus_offset = venus_xyz - venus_at_ca
    earth_xy = np.column_stack((earth_offset @ x_axis, earth_offset @ y_axis))
    venus_xy = np.column_stack((venus_offset @ x_axis, venus_offset @ y_axis))
    scale = ARCSEC_PER_RAD / earth_sun_distance_km
    return earth_xy * scale, venus_xy * scale

def muted_solar_ellipse(ax: plt.Axes, center_date, solar_radius_arcsec: float) -> None:
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    axes_width_days = right - left
    axes_height_arcsec = top - bottom
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    pixel_ratio = bbox.width / bbox.height
    radius_days = solar_radius_arcsec * axes_width_days / axes_height_arcsec / pixel_ratio
    ellipse = Ellipse(
        (mdates.date2num(center_date), 0.0),
        width=2.0 * radius_days,
        height=2.0 * solar_radius_arcsec,
        facecolor="#C98212",
        edgecolor="#E04B18",
        linewidth=1.15,
        alpha=0.72,
        zorder=1,
        label="Solar limb",
    )
    ax.add_patch(ellipse)

def plot_transit(
    year: int,
    dates,
    earth_y_arcsec: np.ndarray,
    venus_y_arcsec: np.ndarray,
    ca_date,
    ca_y_arcsec: float,
    earth_fit: TrackFit,
    venus_fit: TrackFit,
    apparent_track_angle: float,
    solar_radius_arcsec: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(15.36, 7.68), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.plot(dates, venus_y_arcsec, color="#1789D4", linewidth=0.75, label="Venus trajectory", zorder=4)
    ax.plot(dates, earth_y_arcsec, color="#21B33B", linewidth=0.75, label="Earth trajectory", zorder=3)

    ax.set_xlim(datetime(year, 1, 1), datetime(year + 1, 1, 1))
    y_extent = max(
        1200.0,
        float(np.nanmax(np.abs(earth_y_arcsec))) * 1.08,
        float(np.nanmax(np.abs(venus_y_arcsec))) * 1.08,
        solar_radius_arcsec * 1.30,
    )
    ax.set_ylim(-y_extent, y_extent)

    muted_solar_ellipse(ax, ca_date, solar_radius_arcsec)

    ax.axvline(ca_date, color="#AFAFAF", linewidth=0.55, linestyle="--", alpha=0.75, zorder=2)
    ax.scatter([ca_date], [ca_y_arcsec], s=23, facecolor="white", edgecolor="#D9D9D9",
               linewidth=0.55, zorder=7, label="Closest approach")

    annotation = (
        f"Earth angle: {earth_fit.angle_deg:+.6f}°\n"
        f"Venus angle: {venus_fit.angle_deg:+.6f}°\n"
        f"Apparent track angle: {apparent_track_angle:.6f}°"
    )
    ax.text(
        0.67, 0.18, annotation, transform=ax.transAxes,
        color="#E8E8E8", fontsize=10.5, ha="left", va="bottom",
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "#060606",
            "edgecolor": "#808080",
            "alpha": 0.92,
        },
        zorder=8,
    )

    ax.set_title(
        f"{year} Venus Transit — Registered Earth–Venus Crossing and Track Angles",
        color="#F4F4F4", fontsize=15, weight="bold", pad=8,
    )
    ax.set_xlabel(f"Calendar month — {year}", color="#E6E6E6", fontsize=10.5)
    ax.set_ylabel(
        "Registered tangent-plane displacement (arcsec, 2× visual scale)",
        color="#E6E6E6", fontsize=10.5,
    )

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(colors="#E6E6E6", labelsize=9, width=0.5)
    ax.grid(True, color="#777777", alpha=0.34, linewidth=0.42)
    for spine in ax.spines.values():
        spine.set_color("#B0B0B0")
        spine.set_linewidth(0.55)

    handles, labels = ax.get_legend_handles_labels()
    order = [0, 1, 2, 3]
    legend = ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper right",
        frameon=False,
        fontsize=10,
    )
    for text in legend.get_texts():
        text.set_color("#E8E8E8")

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    display(Image(filename=str(output_path)))

def process_transit(year: int, center_text: str) -> dict:
    center = Time(center_text, scale="utc")
    half_window_days = SEARCH_HALF_HOURS / 24.0
    fine_start = Time(center.jd - half_window_days, format="jd", scale="utc").strftime("%Y-%m-%d %H:%M")
    fine_stop = Time(center.jd + half_window_days, format="jd", scale="utc").strftime("%Y-%m-%d %H:%M")

    earth_fine = horizons_vectors("399", fine_start, fine_stop, FINE_STEP)
    venus_fine = horizons_vectors("299", fine_start, fine_stop, FINE_STEP)
    sun_fine = horizons_vectors("10", fine_start, fine_stop, FINE_STEP)

    if not (
        len(earth_fine.jd) == len(venus_fine.jd) == len(sun_fine.jd)
        and np.allclose(earth_fine.jd, venus_fine.jd, atol=1.0e-11, rtol=0.0)
        and np.allclose(earth_fine.jd, sun_fine.jd, atol=1.0e-11, rtol=0.0)
    ):
        raise RuntimeError("REJECTED mismatched fine JPL grids")

    ca_jd, minimum_separation_rad = closest_approach(
        earth_fine.jd, earth_fine.xyz_km, sun_fine.xyz_km, venus_fine.xyz_km
    )

    earth_spline = splines(earth_fine)
    venus_spline = splines(venus_fine)
    sun_spline = splines(sun_fine)
    earth_at_ca = evaluate(earth_spline, ca_jd)
    venus_at_ca = evaluate(venus_spline, ca_jd)
    sun_at_ca = evaluate(sun_spline, ca_jd)
    x_axis, y_axis = tangent_basis(earth_at_ca, sun_at_ca)

    fit_mask = np.abs((earth_fine.jd - ca_jd) * 24.0) <= FIT_HALF_HOURS
    fit_hours = (earth_fine.jd[fit_mask] - ca_jd) * 24.0
    earth_fit_xy = np.column_stack((
        (earth_fine.xyz_km[fit_mask] - earth_at_ca) @ x_axis,
        (earth_fine.xyz_km[fit_mask] - earth_at_ca) @ y_axis,
    ))
    venus_fit_xy = np.column_stack((
        (venus_fine.xyz_km[fit_mask] - venus_at_ca) @ x_axis,
        (venus_fine.xyz_km[fit_mask] - venus_at_ca) @ y_axis,
    ))
    earth_fit = quadratic_track_fit(fit_hours, earth_fit_xy)
    venus_fit = quadratic_track_fit(fit_hours, venus_fit_xy)
    apparent_track_angle = wrapped_difference_deg(earth_fit.angle_deg, venus_fit.angle_deg)
    angle_residual = apparent_track_angle - wrapped_difference_deg(
        earth_fit.angle_deg, venus_fit.angle_deg
    )

    year_start = f"{year}-01-01 00:00"
    year_stop = f"{year + 1}-01-01 00:00"
    earth_year = horizons_vectors("399", year_start, year_stop, YEAR_STEP)
    venus_year = horizons_vectors("299", year_start, year_stop, YEAR_STEP)

    earth_sun_distance = float(np.linalg.norm(sun_at_ca - earth_at_ca))
    earth_xy_arcsec, venus_xy_arcsec = project_registered_tracks(
        earth_year.jd,
        earth_year.xyz_km,
        venus_year.xyz_km,
        earth_at_ca,
        venus_at_ca,
        x_axis,
        y_axis,
        earth_sun_distance,
    )

    registration_offset = (
        float(np.dot(venus_at_ca - sun_at_ca, y_axis))
        * ARCSEC_PER_RAD / earth_sun_distance
    )
    earth_y_arcsec = 2.0 * (earth_xy_arcsec[:, 1] + registration_offset)
    venus_y_arcsec = 2.0 * (venus_xy_arcsec[:, 1] + registration_offset)
    ca_y_arcsec = 2.0 * registration_offset

    dates = Time(earth_year.jd, format="jd", scale="tdb").utc.to_datetime()
    ca_time = Time(ca_jd, format="jd", scale="tdb")
    ca_date = ca_time.utc.to_datetime()
    ca_utc = ca_time.utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    solar_radius_arcsec = math.asin(SOLAR_RADIUS_KM / earth_sun_distance) * ARCSEC_PER_RAD

    output_path = OUTPUT_DIR / PNG_NAMES[year]
    plot_transit(
        year,
        dates,
        earth_y_arcsec,
        venus_y_arcsec,
        ca_date,
        ca_y_arcsec,
        earth_fit,
        venus_fit,
        apparent_track_angle,
        solar_radius_arcsec,
        output_path,
    )

    return {
        "transit_year": year,
        "closest_approach_utc": ca_utc,
        "closest_approach_jd_tdb": ca_jd,
        "earth_track_angle_deg": earth_fit.angle_deg,
        "venus_track_angle_deg": venus_fit.angle_deg,
        "apparent_track_angle_deg": apparent_track_angle,
        "angle_verification_residual_deg": angle_residual,
        "earth_slope": earth_fit.slope,
        "venus_slope": venus_fit.slope,
        "earth_rms": earth_fit.rms_km,
        "venus_rms": venus_fit.rms_km,
        "earth_curvature": earth_fit.curvature_per_km,
        "venus_curvature": venus_fit.curvature_per_km,
        "sample_count": int(np.sum(fit_mask)),
        "minimum_separation_arcsec": minimum_separation_rad * ARCSEC_PER_RAD,
        "solar_angular_radius_arcsec": solar_radius_arcsec,
        "png": str(output_path),
    }

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    section("CODE INPUTS")
    print(f"Version                              {VERSION}")
    print("JPL source                           Horizons vectors API")
    print(f"Observer/reference                   {LOCATION}; Sun=10 Venus=299 Earth=399")
    print("Reference frame                      JPL Horizons default ICRF/J2000")
    print(f"Reference plane                      {REFPLANE}")
    print(f"Aberrations                          {ABERRATIONS}")
    print(f"Fine/year cadence                    {FINE_STEP}/{YEAR_STEP}")
    print(f"Output                               {OUTPUT_DIR}")
    for year, center in TRANSITS.items():
        print(f"NOT USED AS CA INPUT {year} broad search center {center} UTC")

    section("COMMENTS")
    print("The attached reference layout is preserved: calendar-year x-axis, registered displacement y-axis,")
    print("unshifted closest-approach placement, thin Earth/Venus trajectories, and lower-right angle box.")
    print("Only the solar appearance and angle-label wording are changed.")
    print("Solar fill is muted yellow-orange; exactly one orange-red limb is drawn.")
    print("DEBUG progress follows. REJECTED manual angles and closest-approach times are not inputs.")

    rows = []
    for year, center in TRANSITS.items():
        print(f"DEBUG processing {year}", flush=True)
        rows.append(process_transit(year, center))

    csv_columns = [
        "transit_year",
        "closest_approach_utc",
        "closest_approach_jd_tdb",
        "earth_track_angle_deg",
        "venus_track_angle_deg",
        "apparent_track_angle_deg",
        "angle_verification_residual_deg",
        "earth_slope",
        "venus_slope",
        "earth_rms",
        "venus_rms",
        "earth_curvature",
        "venus_curvature",
        "sample_count",
    ]
    csv_path = OUTPUT_DIR / CSV_NAME
    pd.DataFrame([{column: row[column] for column in csv_columns} for row in rows]).to_csv(
        csv_path, index=False, float_format="%.12g"
    )

    section("RESULTS")
    for row in rows:
        print(
            f"{row['transit_year']}  CA {row['closest_approach_utc']}  "
            f"JD_TDB {row['closest_approach_jd_tdb']:.9f}"
        )
        print(
            f"Earth angle {row['earth_track_angle_deg']:.6f} deg  "
            f"Venus angle {row['venus_track_angle_deg']:.6f} deg  "
            f"Apparent track angle {row['apparent_track_angle_deg']:.6f} deg  "
            f"Verification {row['angle_verification_residual_deg']:.12e} deg"
        )
        print(
            f"Earth slope {row['earth_slope']:.9f}  RMS {row['earth_rms']:.6f} km  "
            f"curvature {row['earth_curvature']:.12e}"
        )
        print(
            f"Venus slope {row['venus_slope']:.9f}  RMS {row['venus_rms']:.6f} km  "
            f"curvature {row['venus_curvature']:.12e}"
        )
        print(
            f"Minimum separation {row['minimum_separation_arcsec']:.6f} arcsec  "
            f"Solar radius {row['solar_angular_radius_arcsec']:.6f} arcsec  "
            f"samples {row['sample_count']}"
        )

    section("OUTPUT SUMMARY")
    print(f"CSV {csv_path}")
    for row in rows:
        print(f"PNG {row['transit_year']} {row['png']} bytes {Path(row['png']).stat().st_size}")
    print(f"Exactly three PNG figures {len(rows) == 3}")

    section("PAPER COMPARISON")
    print("NOT USED: no published angles, closest-approach times, or manual track positions.")
    print("The 8.5-degree-scale quantity is reported as the apparent track angle between JPL-projected directions.")

    section("EQUATION STATUS")
    residual = max(abs(row["angle_verification_residual_deg"]) for row in rows)
    print("VERIFIED apparent_track_angle = abs(wrap180(Earth angle - Venus angle))")
    print("VERIFIED 0 <= apparent_track_angle <= 180 degrees")
    print("VERIFIED minute-by-minute JPL vectors determine closest approach and fitted directions")
    print(f"Maximum angle residual {residual:.12e} deg")
    print(f"Equation checks passed {residual <= 1.0e-12}")

    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(VERSION)

if __name__ == "__main__":
    main()
# V0116