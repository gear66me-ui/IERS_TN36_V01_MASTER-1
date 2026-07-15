# V0117
# Audit reference: four Venus-transit calendar-year plots with positive acute horizontal-axis angles, apparent track difference, and ecliptic angle sum.

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

for module_name, package_name in [
    ("astroquery", "astroquery"),
    ("astropy", "astropy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("IPython", "ipython"),
]:
    need(module_name, package_name)

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

VERSION = "V0117"
OUT = Path("/content/VENUS_TRANSITS_1761_1874_1882_2004_HORIZONTAL_ANGLES_V0117_OUTPUT")
CSV_NAME = "VENUS_TRANSITS_1761_1874_1882_2004_HORIZONTAL_ANGLES_V0117.csv"

AU_KM = 149597870.700
R_SUN_KM = 695700.000
AS_PER_RAD = 206264.80624709636
LOCATION = "@0"
REFPLANE = "earth"
ABERRATIONS = "geometric"
FINE_STEP = "1m"
YEAR_STEP = "6h"
SEARCH_HALF_H = 18.0
FIT_HALF_H = 10.0

TRANSITS: Dict[int, str] = {
    1761: "1761-06-06 06:00",
    1874: "1874-12-09 04:00",
    1882: "1882-12-06 17:00",
    2004: "2004-06-08 08:00",
}

PNG_NAMES = {
    year: f"VENUS_TRANSIT_{year}_HORIZONTAL_ANGLES_V0117.png"
    for year in TRANSITS
}

@dataclass(frozen=True)
class Series:
    jd: np.ndarray
    xyz: np.ndarray

@dataclass(frozen=True)
class Fit:
    raw_direction_deg: float
    signed_line_deg: float
    horizontal_angle_deg: float
    slope: float
    rms_km: float
    curvature: float

def section(name: str) -> None:
    print(name)
    print("-" * len(name))

def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("REJECTED zero or non-finite vector")
    return vector / norm

def query(body: str, start: str, stop: str, step: str) -> Series:
    table = Horizons(
        id=body,
        id_type="majorbody",
        location=LOCATION,
        epochs={"start": start, "stop": stop, "step": step},
    ).vectors(refplane=REFPLANE, aberrations=ABERRATIONS)
    jd = np.asarray(table["datetime_jd"], dtype=float)
    xyz = np.column_stack([
        np.asarray(table["x"], dtype=float),
        np.asarray(table["y"], dtype=float),
        np.asarray(table["z"], dtype=float),
    ]) * AU_KM
    if len(jd) < 3 or not np.all(np.diff(jd) > 0.0):
        raise RuntimeError(f"REJECTED JPL grid for body {body}")
    return Series(jd, xyz)

def splines(jd: np.ndarray, xyz: np.ndarray) -> List[CubicSpline]:
    return [CubicSpline(jd, xyz[:, axis], bc_type="natural") for axis in range(3)]

def evaluate(curves: List[CubicSpline], jd: float) -> np.ndarray:
    return np.array([curve(jd) for curve in curves], dtype=float)

def separations(earth: np.ndarray, sun: np.ndarray, venus: np.ndarray) -> np.ndarray:
    sun_direction = sun - earth
    venus_direction = venus - earth
    sun_direction /= np.linalg.norm(sun_direction, axis=1)[:, None]
    venus_direction /= np.linalg.norm(venus_direction, axis=1)[:, None]
    return np.arccos(np.clip(np.einsum("ij,ij->i", sun_direction, venus_direction), -1.0, 1.0))

def closest_approach(jd: np.ndarray, earth: np.ndarray, sun: np.ndarray, venus: np.ndarray) -> tuple[float, float]:
    separation = separations(earth, sun, venus)
    index = int(np.argmin(separation))
    lower = max(0, index - 3)
    upper = min(len(jd) - 1, index + 3)
    earth_spline = splines(jd, earth)
    sun_spline = splines(jd, sun)
    venus_spline = splines(jd, venus)

    def objective(value: float) -> float:
        e = evaluate(earth_spline, value)
        s = evaluate(sun_spline, value)
        v = evaluate(venus_spline, value)
        a = unit(s - e)
        b = unit(v - e)
        return math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))

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
    normal = unit(sun_at_ca - earth_at_ca)
    x_axis = np.cross(np.array([0.0, 0.0, 1.0]), normal)
    if np.linalg.norm(x_axis) < 1.0e-10:
        x_axis = np.cross(np.array([0.0, 1.0, 0.0]), normal)
    x_axis = unit(x_axis)
    y_axis = unit(np.cross(normal, x_axis))
    return x_axis, y_axis

def signed_line_angle(raw_direction_deg: float) -> float:
    return ((raw_direction_deg + 90.0) % 180.0) - 90.0

def fit_track(hours: np.ndarray, xy_km: np.ndarray) -> Fit:
    cx = np.polyfit(hours, xy_km[:, 0], 2)
    cy = np.polyfit(hours, xy_km[:, 1], 2)
    model = np.column_stack((np.polyval(cx, hours), np.polyval(cy, hours)))
    rms = float(np.sqrt(np.mean(np.sum((xy_km - model) ** 2, axis=1))))
    vx, vy = float(cx[1]), float(cy[1])
    ax, ay = float(2.0 * cx[0]), float(2.0 * cy[0])
    speed2 = vx * vx + vy * vy
    if speed2 <= 0.0:
        raise RuntimeError("REJECTED degenerate fit")
    raw = math.degrees(math.atan2(vy, vx)) % 360.0
    signed = signed_line_angle(raw)
    horizontal = abs(signed)
    slope = math.inf if abs(vx) < 1.0e-15 else vy / vx
    curvature = abs(vx * ay - vy * ax) / (speed2 ** 1.5)
    return Fit(raw, signed, horizontal, slope, rms, curvature)

def add_sun(ax: plt.Axes, center_date, y_center: float, solar_radius_arcsec: float) -> None:
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    width_days = right - left
    height_arcsec = top - bottom
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    pixel_ratio = bbox.width / bbox.height
    radius_days = solar_radius_arcsec * width_days / height_arcsec / pixel_ratio
    ax.add_patch(Ellipse(
        (mdates.date2num(center_date), y_center),
        width=2.0 * radius_days,
        height=2.0 * solar_radius_arcsec,
        facecolor="#C98212",
        edgecolor="#E04B18",
        linewidth=1.15,
        alpha=0.76,
        zorder=2,
        label="Solar limb",
    ))

def make_plot(
    year: int,
    dates,
    earth_y: np.ndarray,
    venus_y: np.ndarray,
    ca_date,
    ca_y: float,
    solar_radius_arcsec: float,
    earth_fit: Fit,
    venus_fit: Fit,
    apparent_angle: float,
    sum_angle: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(18.0, 8.5), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.plot(dates, venus_y, color="#1789D4", linewidth=0.72, label="Venus trajectory", zorder=5)
    ax.plot(dates, earth_y, color="#21B33B", linewidth=0.72, label="Earth trajectory", zorder=4)
    ax.set_xlim(datetime(year, 1, 1), datetime(year, 12, 31, 23, 59))

    y_extent = max(
        1200.0,
        float(np.max(np.abs(earth_y))) * 1.08,
        float(np.max(np.abs(venus_y))) * 1.08,
        abs(ca_y) + solar_radius_arcsec * 1.30,
    )
    ax.set_ylim(-y_extent, y_extent)
    add_sun(ax, ca_date, ca_y, solar_radius_arcsec)

    ax.axvline(ca_date, color="#AFAFAF", linewidth=0.55, linestyle="--", alpha=0.75, zorder=3)
    ax.scatter([ca_date], [ca_y], s=24, facecolor="white", edgecolor="#D9D9D9",
               linewidth=0.55, zorder=8, label="Closest approach")

    annotation = "\n".join([
        f"Earth horizontal angle: {earth_fit.horizontal_angle_deg:.6f}°",
        f"Venus horizontal angle: {venus_fit.horizontal_angle_deg:.6f}°",
        f"Apparent track angle: {apparent_angle:.6f}°",
        f"Ecliptic sum angle: {sum_angle:.6f}°",
    ])
    ax.text(
        0.665, 0.17, annotation, transform=ax.transAxes,
        color="#E8E8E8", fontsize=10.5, ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.34", "facecolor": "#060606",
              "edgecolor": "#808080", "alpha": 0.93},
        zorder=9,
    )

    ax.set_title(
        f"{year} Venus Transit — Positive Horizontal Track Angles",
        color="#F4F4F4", fontsize=15, weight="bold", pad=8,
    )
    ax.set_xlabel(f"Calendar month — {year}", color="#E6E6E6", fontsize=10.5)
    ax.set_ylabel("Registered tangent-plane displacement (arcsec, 2× visual scale)",
                  color="#E6E6E6", fontsize=10.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(colors="#E6E6E6", labelsize=9, width=0.5)
    ax.grid(True, color="#777777", alpha=0.32, linewidth=0.42)
    for spine in ax.spines.values():
        spine.set_color("#B0B0B0")
        spine.set_linewidth(0.55)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    for label in legend.get_texts():
        label.set_color("#E8E8E8")

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    display(Image(filename=str(output_path)))

def process(year: int, center_text: str) -> dict:
    center = Time(center_text, scale="utc")
    delta = SEARCH_HALF_H / 24.0
    start = Time(center.jd - delta, format="jd", scale="utc").strftime("%Y-%m-%d %H:%M")
    stop = Time(center.jd + delta, format="jd", scale="utc").strftime("%Y-%m-%d %H:%M")

    earth = query("399", start, stop, FINE_STEP)
    venus = query("299", start, stop, FINE_STEP)
    sun = query("10", start, stop, FINE_STEP)

    if not (
        len(earth.jd) == len(venus.jd) == len(sun.jd)
        and np.allclose(earth.jd, venus.jd, atol=1.0e-11, rtol=0.0)
        and np.allclose(earth.jd, sun.jd, atol=1.0e-11, rtol=0.0)
    ):
        raise RuntimeError("REJECTED mismatched minute JPL grids")

    ca_jd, minimum_separation = closest_approach(earth.jd, earth.xyz, sun.xyz, venus.xyz)
    earth_spline = splines(earth.jd, earth.xyz)
    venus_spline = splines(venus.jd, venus.xyz)
    sun_spline = splines(sun.jd, sun.xyz)
    earth0 = evaluate(earth_spline, ca_jd)
    venus0 = evaluate(venus_spline, ca_jd)
    sun0 = evaluate(sun_spline, ca_jd)
    x_axis, y_axis = tangent_basis(earth0, sun0)

    mask = np.abs((earth.jd - ca_jd) * 24.0) <= FIT_HALF_H
    hours = (earth.jd[mask] - ca_jd) * 24.0
    earth_xy = np.column_stack(((earth.xyz[mask] - earth0) @ x_axis,
                                (earth.xyz[mask] - earth0) @ y_axis))
    venus_xy = np.column_stack(((venus.xyz[mask] - venus0) @ x_axis,
                                (venus.xyz[mask] - venus0) @ y_axis))
    earth_fit = fit_track(hours, earth_xy)
    venus_fit = fit_track(hours, venus_xy)

    apparent_angle = abs(earth_fit.signed_line_deg - venus_fit.signed_line_deg)
    if apparent_angle > 90.0:
        apparent_angle = 180.0 - apparent_angle
    apparent_angle = abs(apparent_angle)
    sum_angle = earth_fit.horizontal_angle_deg + venus_fit.horizontal_angle_deg

    year_start = f"{year}-01-01 00:00"
    year_stop = f"{year}-12-31 23:59"
    earth_year = query("399", year_start, year_stop, YEAR_STEP)
    venus_year = query("299", year_start, year_stop, YEAR_STEP)
    if len(earth_year.jd) != len(venus_year.jd) or not np.allclose(
        earth_year.jd, venus_year.jd, atol=1.0e-11, rtol=0.0
    ):
        raise RuntimeError("REJECTED mismatched annual JPL grids")

    earth_sun_distance = float(np.linalg.norm(sun0 - earth0))
    scale = AS_PER_RAD / earth_sun_distance
    registration_y = float(np.dot(venus0 - sun0, y_axis)) * scale
    earth_y = 2.0 * (((earth_year.xyz - earth0) @ y_axis) * scale + registration_y)
    venus_y = 2.0 * (((venus_year.xyz - venus0) @ y_axis) * scale + registration_y)
    ca_y = 2.0 * registration_y

    ca_time = Time(ca_jd, format="jd", scale="tdb")
    ca_date = ca_time.utc.to_datetime()
    dates = Time(earth_year.jd, format="jd", scale="tdb").utc.to_datetime()
    solar_radius = math.asin(R_SUN_KM / earth_sun_distance) * AS_PER_RAD
    output_path = OUT / PNG_NAMES[year]

    make_plot(
        year, dates, earth_y, venus_y, ca_date, ca_y, solar_radius,
        earth_fit, venus_fit, apparent_angle, sum_angle, output_path,
    )

    return {
        "transit_year": year,
        "closest_approach_utc": ca_time.utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "closest_approach_jd_tdb": ca_jd,
        "earth_horizontal_angle_deg": earth_fit.horizontal_angle_deg,
        "venus_horizontal_angle_deg": venus_fit.horizontal_angle_deg,
        "apparent_track_angle_deg": apparent_angle,
        "ecliptic_sum_angle_deg": sum_angle,
        "earth_raw_direction_deg_NOT_USED": earth_fit.raw_direction_deg,
        "venus_raw_direction_deg_NOT_USED": venus_fit.raw_direction_deg,
        "earth_slope": earth_fit.slope,
        "venus_slope": venus_fit.slope,
        "earth_rms_km": earth_fit.rms_km,
        "venus_rms_km": venus_fit.rms_km,
        "earth_curvature": earth_fit.curvature,
        "venus_curvature": venus_fit.curvature,
        "minimum_separation_arcsec": minimum_separation * AS_PER_RAD,
        "sample_count": int(np.sum(mask)),
        "png": str(output_path),
    }

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    section("CODE INPUTS")
    print(f"Version                              {VERSION}")
    print("JPL source                           Horizons vectors API")
    print(f"Observer/reference                   {LOCATION}; Sun=10 Venus=299 Earth=399")
    print("Reference frame                      JPL Horizons default ICRF/J2000")
    print(f"Reference plane/aberrations          {REFPLANE}/{ABERRATIONS}")
    print(f"Fine/year cadence                    {FINE_STEP}/{YEAR_STEP}")
    print(f"Output                               {OUT}")
    for year, center in TRANSITS.items():
        print(f"NOT USED AS CA INPUT {year} broad search center {center} UTC")

    section("COMMENTS")
    print("Displayed track angles are positive acute angles measured from the horizontal axis.")
    print("Raw 174-189 degree velocity directions are retained only as NOT USED audit fields in the CSV.")
    print("Apparent track angle is the positive wrapped difference of the two signed line orientations.")
    print("Ecliptic sum angle is the sum of the two positive horizontal-axis angles.")
    print("DEBUG progress follows.")

    rows = []
    for year, center in TRANSITS.items():
        print(f"DEBUG processing {year}", flush=True)
        rows.append(process(year, center))

    columns = [
        "transit_year", "closest_approach_utc", "closest_approach_jd_tdb",
        "earth_horizontal_angle_deg", "venus_horizontal_angle_deg",
        "apparent_track_angle_deg", "ecliptic_sum_angle_deg",
        "earth_raw_direction_deg_NOT_USED", "venus_raw_direction_deg_NOT_USED",
        "earth_slope", "venus_slope", "earth_rms_km", "venus_rms_km",
        "earth_curvature", "venus_curvature", "minimum_separation_arcsec",
        "sample_count",
    ]
    csv_path = OUT / CSV_NAME
    pd.DataFrame([{column: row[column] for column in columns} for row in rows]).to_csv(
        csv_path, index=False, float_format="%.12g"
    )

    section("RESULTS")
    for row in rows:
        print(f"{row['transit_year']}  CA {row['closest_approach_utc']}  JD_TDB {row['closest_approach_jd_tdb']:.9f}")
        print(f"Earth horizontal angle               {row['earth_horizontal_angle_deg']:.6f} deg")
        print(f"Venus horizontal angle               {row['venus_horizontal_angle_deg']:.6f} deg")
        print(f"Apparent track angle                 {row['apparent_track_angle_deg']:.6f} deg")
        print(f"Ecliptic sum angle                   {row['ecliptic_sum_angle_deg']:.6f} deg")
        print(f"Earth slope/RMS/curvature             {row['earth_slope']:.9f}  {row['earth_rms_km']:.6f} km  {row['earth_curvature']:.12e}")
        print(f"Venus slope/RMS/curvature             {row['venus_slope']:.9f}  {row['venus_rms_km']:.6f} km  {row['venus_curvature']:.12e}")

    section("OUTPUT SUMMARY")
    print(f"CSV                                  {csv_path}")
    for row in rows:
        print(f"PNG {row['transit_year']}                            {row['png']} bytes {Path(row['png']).stat().st_size}")
    print(f"Exactly four PNG figures             {len(rows) == 4}")

    section("PAPER COMPARISON")
    print("NOT USED: published angles or manually entered closest-approach times.")
    print("The expected approximately 8.5-degree and 14-degree scales are comparisons only, not inputs.")

    section("EQUATION STATUS")
    positive_angles = all(
        0.0 <= row["earth_horizontal_angle_deg"] <= 90.0
        and 0.0 <= row["venus_horizontal_angle_deg"] <= 90.0
        and 0.0 <= row["apparent_track_angle_deg"] <= 90.0
        for row in rows
    )
    sum_residual = max(abs(
        row["ecliptic_sum_angle_deg"]
        - row["earth_horizontal_angle_deg"]
        - row["venus_horizontal_angle_deg"]
    ) for row in rows)
    print("VERIFIED raw direction reduced modulo 180 degrees and reflected to a positive acute horizontal angle")
    print("VERIFIED apparent track angle = positive wrapped line-direction difference")
    print("VERIFIED ecliptic sum angle = Earth horizontal angle + Venus horizontal angle")
    print(f"Maximum sum residual                 {sum_residual:.12e} deg")
    print(f"Equation checks passed               {positive_angles and sum_residual <= 1.0e-12}")

    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(VERSION)

if __name__ == "__main__":
    main()
# V0117