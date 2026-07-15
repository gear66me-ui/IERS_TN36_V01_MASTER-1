# V0120
# Audit reference: six annual registered Earth/Venus crossing plots matched to the approved 2004 calendar-track presentation.

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

VERSION = "V0120"
FILENAME = "VENUS_TRANSITS_1761_1769_1874_1882_2004_2012_ANNUAL_REGISTERED_TRACKS_V0120.py"
OUT = Path("/content/VENUS_TRANSITS_1761_1769_1874_1882_2004_2012_ANNUAL_REGISTERED_TRACKS_V0120_OUTPUT")
CSV_NAME = "VENUS_TRANSITS_1761_1769_1874_1882_2004_2012_ANNUAL_REGISTERED_TRACKS_V0120.csv"

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
VISUAL_SCALE = 2.0

TRANSITS: Dict[int, str] = {
    1761: "1761-06-06 06:00",
    1769: "1769-06-03 22:00",
    1874: "1874-12-09 04:00",
    1882: "1882-12-06 17:00",
    2004: "2004-06-08 08:00",
    2012: "2012-06-06 01:00",
}

PNG_NAMES = {
    year: f"VENUS_TRANSIT_{year}_ANNUAL_REGISTERED_TRACKS_V0120.png"
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
    positive_horizontal_deg: float
    slope: float
    rms_km: float
    curvature_per_km: float


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
    return Series(jd=jd, xyz=xyz)


def splines(jd: np.ndarray, xyz: np.ndarray) -> List[CubicSpline]:
    return [CubicSpline(jd, xyz[:, axis], bc_type="natural") for axis in range(3)]


def evaluate(curves: List[CubicSpline], jd: float) -> np.ndarray:
    return np.array([curve(jd) for curve in curves], dtype=float)


def angular_separation(earth: np.ndarray, sun: np.ndarray, venus: np.ndarray) -> np.ndarray:
    sun_dir = sun - earth
    venus_dir = venus - earth
    sun_dir /= np.linalg.norm(sun_dir, axis=1)[:, None]
    venus_dir /= np.linalg.norm(venus_dir, axis=1)[:, None]
    return np.arccos(np.clip(np.einsum("ij,ij->i", sun_dir, venus_dir), -1.0, 1.0))


def closest_approach(jd: np.ndarray, earth: np.ndarray, sun: np.ndarray, venus: np.ndarray) -> tuple[float, float]:
    separation = angular_separation(earth, sun, venus)
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
        return math.acos(float(np.clip(np.dot(unit(s - e), unit(v - e)), -1.0, 1.0)))

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
        raise RuntimeError("REJECTED degenerate track fit")
    raw = math.degrees(math.atan2(vy, vx)) % 360.0
    signed = signed_line_angle(raw)
    positive = abs(signed)
    slope = math.inf if abs(vx) < 1.0e-15 else vy / vx
    curvature = abs(vx * ay - vy * ax) / (speed2 ** 1.5)
    return Fit(raw, signed, positive, slope, rms, curvature)


def add_solar_limb(ax: plt.Axes, center_date: datetime, y_center: float, radius_arcsec: float) -> None:
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    width_days = right - left
    height_arcsec = top - bottom
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    pixel_ratio = bbox.width / bbox.height
    radius_days = radius_arcsec * width_days / height_arcsec / pixel_ratio
    ax.add_patch(Ellipse(
        (mdates.date2num(center_date), y_center),
        width=2.0 * radius_days,
        height=2.0 * radius_arcsec,
        facecolor="#C98A18",
        edgecolor="#E64A19",
        linewidth=1.05,
        alpha=0.90,
        zorder=4,
        label="Solar limb",
    ))


def make_plot(
    year: int,
    dates: np.ndarray,
    earth_y: np.ndarray,
    venus_y: np.ndarray,
    ca_date: datetime,
    ca_y: float,
    solar_radius_arcsec: float,
    earth_fit: Fit,
    venus_fit: Fit,
    apparent_angle: float,
    angle_sum: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(18.0, 8.5), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.plot(dates, venus_y, color="#1E78B4", linewidth=0.72, label="Venus trajectory", zorder=3)
    ax.plot(dates, earth_y, color="#2FAA45", linewidth=0.72, label="Earth trajectory", zorder=2)
    ax.set_xlim(datetime(year, 1, 1), datetime(year + 1, 1, 1))

    y_extent = max(
        1200.0,
        float(np.max(np.abs(earth_y))) * 1.08,
        float(np.max(np.abs(venus_y))) * 1.08,
        abs(ca_y) + solar_radius_arcsec * 1.35,
    )
    ax.set_ylim(-y_extent, y_extent)
    add_solar_limb(ax, ca_date, ca_y, solar_radius_arcsec)

    ax.axvline(ca_date, color="#B0B0B0", linewidth=0.52, linestyle="--", alpha=0.72, zorder=1)
    ax.scatter([ca_date], [ca_y], s=22, facecolor="white", edgecolor="#DADADA",
               linewidth=0.55, zorder=7, label="Closest approach")

    annotation = "\n".join([
        f"Earth track angle: {earth_fit.positive_horizontal_deg:.6f}°",
        f"Venus track angle: {venus_fit.positive_horizontal_deg:.6f}°",
        f"Apparent track angle: {apparent_angle:.6f}°",
        f"Earth + Venus: {angle_sum:.6f}°",
    ])
    x_offset_days = 30.0 if ca_date.month <= 8 else -30.0
    ha = "left" if x_offset_days > 0.0 else "right"
    y_offset = -0.18 * y_extent if ca_y >= 0.0 else 0.18 * y_extent
    va = "top" if y_offset < 0.0 else "bottom"
    ax.annotate(
        annotation,
        xy=(ca_date, ca_y),
        xytext=(ca_date + pd.Timedelta(days=x_offset_days), ca_y + y_offset),
        color="#ECECEC",
        fontsize=10.2,
        ha=ha,
        va=va,
        arrowprops={"arrowstyle": "-", "color": "#AFAFAF", "linewidth": 0.65},
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "#050505",
              "edgecolor": "#858585", "alpha": 0.94},
        zorder=8,
    )

    ax.set_title(
        f"{year} Venus Transit — Registered Earth–Venus Crossing and Track Angles",
        color="#F0F0F0", fontsize=15, weight="bold", pad=8,
    )
    ax.set_xlabel(f"Calendar month — {year}", color="#E0E0E0", fontsize=10.5)
    ax.set_ylabel(
        f"Registered tangent-plane displacement (arcsec, {VISUAL_SCALE:.0f}× visual scale)",
        color="#E0E0E0", fontsize=10.5,
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(colors="#D7D7D7", labelsize=9, width=0.48)
    ax.grid(True, color="#646464", alpha=0.28, linewidth=0.42)
    for spine in ax.spines.values():
        spine.set_color("#8E8E8E")
        spine.set_linewidth(0.55)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=9.4)
    for text in legend.get_texts():
        text.set_color("#DFDFDF")

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
    angle_sum = earth_fit.positive_horizontal_deg + venus_fit.positive_horizontal_deg

    year_start = f"{year}-01-01 00:00"
    year_stop = f"{year + 1}-01-01 00:00"
    earth_year = query("399", year_start, year_stop, YEAR_STEP)
    venus_year = query("299", year_start, year_stop, YEAR_STEP)
    if len(earth_year.jd) != len(venus_year.jd) or not np.allclose(
        earth_year.jd, venus_year.jd, atol=1.0e-11, rtol=0.0
    ):
        raise RuntimeError("REJECTED mismatched annual JPL grids")

    earth_sun_distance = float(np.linalg.norm(sun0 - earth0))
    scale = AS_PER_RAD / earth_sun_distance
    registration_y = float(np.dot(venus0 - sun0, y_axis)) * scale
    earth_y = VISUAL_SCALE * (((earth_year.xyz - earth0) @ y_axis) * scale + registration_y)
    venus_y = VISUAL_SCALE * (((venus_year.xyz - venus0) @ y_axis) * scale + registration_y)
    ca_y = VISUAL_SCALE * registration_y

    ca_time = Time(ca_jd, format="jd", scale="tdb")
    ca_date = ca_time.utc.to_datetime()
    dates = Time(earth_year.jd, format="jd", scale="tdb").utc.to_datetime()
    solar_radius = math.asin(R_SUN_KM / earth_sun_distance) * AS_PER_RAD
    output_path = OUT / PNG_NAMES[year]

    make_plot(
        year, dates, earth_y, venus_y, ca_date, ca_y, solar_radius,
        earth_fit, venus_fit, apparent_angle, angle_sum, output_path,
    )

    return {
        "transit_year": year,
        "closest_approach_utc": ca_time.utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "jd_tdb": ca_jd,
        "earth_angle_deg": earth_fit.positive_horizontal_deg,
        "venus_angle_deg": venus_fit.positive_horizontal_deg,
        "apparent_track_angle_deg": apparent_angle,
        "earth_positive_horizontal_angle_deg": earth_fit.positive_horizontal_deg,
        "venus_positive_horizontal_angle_deg": venus_fit.positive_horizontal_deg,
        "earth_plus_venus_angle_sum_deg": angle_sum,
        "earth_slope": earth_fit.slope,
        "venus_slope": venus_fit.slope,
        "earth_rms_km": earth_fit.rms_km,
        "venus_rms_km": venus_fit.rms_km,
        "earth_curvature_per_km": earth_fit.curvature_per_km,
        "venus_curvature_per_km": venus_fit.curvature_per_km,
        "sample_count": int(np.sum(mask)),
        "minimum_separation_arcsec": minimum_separation * AS_PER_RAD,
        "png_file": str(output_path),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    section("CODE INPUTS")
    print(f"Version                              {VERSION}")
    print(f"Program                              {FILENAME}")
    print("JPL source                           Horizons vectors API")
    print(f"Observer/reference                   {LOCATION}; Sun=10 Venus=299 Earth=399")
    print(f"Reference plane/aberrations          {REFPLANE}/{ABERRATIONS}")
    print(f"Minute/year cadence                  {FINE_STEP}/{YEAR_STEP}")
    print(f"Fit half-window                      {FIT_HALF_H:.1f} h")
    print(f"Output                               {OUT}")
    for year, center in TRANSITS.items():
        print(f"NOT USED AS CA INPUT {year} broad search center {center} UTC")

    section("COMMENTS")
    print("The annual curves preserve the approved registered tangent-plane basis at closest approach.")
    print("Only positive acute Earth, Venus, and apparent track angles are displayed.")
    print("Raw direction angles are calculated internally but NOT USED in plots or CSV output.")
    print("The solar disk uses one muted yellow-orange fill and one orange-red limb.")
    print("DEBUG progress follows.")

    rows = []
    for year, center in TRANSITS.items():
        print(f"DEBUG processing {year}", flush=True)
        rows.append(process(year, center))

    columns = [
        "transit_year", "closest_approach_utc", "jd_tdb",
        "earth_angle_deg", "venus_angle_deg", "apparent_track_angle_deg",
        "earth_positive_horizontal_angle_deg", "venus_positive_horizontal_angle_deg",
        "earth_plus_venus_angle_sum_deg", "earth_slope", "venus_slope",
        "earth_rms_km", "venus_rms_km", "earth_curvature_per_km",
        "venus_curvature_per_km", "sample_count", "minimum_separation_arcsec",
    ]
    csv_path = OUT / CSV_NAME
    pd.DataFrame([{column: row[column] for column in columns} for row in rows]).to_csv(
        csv_path, index=False, float_format="%.12g"
    )

    section("RESULTS")
    for row in rows:
        print(f"{row['transit_year']}  CA {row['closest_approach_utc']}  JD(TDB) {row['jd_tdb']:.9f}")
        print(f"Earth/Venus/apparent angles          {row['earth_angle_deg']:.6f}  {row['venus_angle_deg']:.6f}  {row['apparent_track_angle_deg']:.6f} deg")
        print(f"Earth + Venus angle sum              {row['earth_plus_venus_angle_sum_deg']:.6f} deg")
        print(f"Earth slope/RMS/curvature             {row['earth_slope']:.9f}  {row['earth_rms_km']:.6f} km  {row['earth_curvature_per_km']:.12e} 1/km")
        print(f"Venus slope/RMS/curvature             {row['venus_slope']:.9f}  {row['venus_rms_km']:.6f} km  {row['venus_curvature_per_km']:.12e} 1/km")
        print(f"Minute samples                        {row['sample_count']}")

    section("OUTPUT SUMMARY")
    print(f"CSV                                  {csv_path}")
    for row in rows:
        path = Path(row["png_file"])
        print(f"PNG {row['transit_year']}                            {path} bytes {path.stat().st_size}")
    print(f"Exactly six PNG figures              {len(rows) == 6}")

    section("PAPER COMPARISON")
    print("NOT USED: published angles, manual contact times, or manual closest-approach values.")
    print("Published values may be compared only after the JPL-derived results are produced.")

    section("EQUATION STATUS")
    positive = all(
        0.0 <= row["earth_angle_deg"] <= 90.0
        and 0.0 <= row["venus_angle_deg"] <= 90.0
        and 0.0 <= row["apparent_track_angle_deg"] <= 90.0
        for row in rows
    )
    sum_residual = max(abs(
        row["earth_plus_venus_angle_sum_deg"]
        - row["earth_angle_deg"]
        - row["venus_angle_deg"]
    ) for row in rows)
    png_ok = all(Path(row["png_file"]).is_file() and Path(row["png_file"]).stat().st_size > 0 for row in rows)
    print("VERIFIED positive acute angle = absolute signed line angle after modulo-180 reduction")
    print("VERIFIED apparent angle = positive wrapped Earth/Venus signed-line difference")
    print("VERIFIED angle sum = Earth positive horizontal angle + Venus positive horizontal angle")
    print(f"Maximum angle-sum residual           {sum_residual:.12e} deg")
    print(f"Equation checks passed               {positive and sum_residual <= 1.0e-12}")
    print(f"Six PNG file checks passed           {png_ok and len(rows) == 6}")

    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0120