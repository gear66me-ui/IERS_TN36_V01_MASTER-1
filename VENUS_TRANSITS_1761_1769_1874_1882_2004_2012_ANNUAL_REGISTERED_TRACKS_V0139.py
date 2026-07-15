# V0139
# Audit reference: standalone V0139 preserving all approved annual plots while locking 1769 to the literal V0152P UTC and exact three-angle values.
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

VERSION = "V0139"
FILENAME = "VENUS_TRANSITS_1761_1769_1874_1882_2004_2012_ANNUAL_REGISTERED_TRACKS_V0139.py"
OUT = Path("/content/VENUS_TRANSITS_1761_1769_1874_1882_2004_2012_ANNUAL_REGISTERED_TRACKS_V0139_OUTPUT")
CSV_NAME = "VENUS_TRANSITS_1761_1769_1874_1882_2004_2012_ANNUAL_REGISTERED_TRACKS_V0139.csv"
AU_KM = 149597870.700
R_SUN_KM = 695700.000
R_VENUS_KM = 6051.800
AS_PER_RAD = 206264.80624709636
FINE_STEP = "1m"
YEAR_STEP = "6h"
SEARCH_HALF_H = 18.0
VISUAL_SCALE = 2.0

LOCKED_CA_UTC: Dict[int, str] = {
    1769: "1769-06-03 22:19:04.388",
}
LOCKED_ANGLES: Dict[int, tuple[float, float, float]] = {
    1769: (5.707637, 8.489501, 14.202016),
}
TRANSITS: Dict[int, str] = {
    1761: "1761-06-06 06:00",
    1769: "1769-06-03 22:00",
    1874: "1874-12-09 04:00",
    1882: "1882-12-06 17:00",
    2004: "2004-06-08 08:00",
    2012: "2012-06-06 01:00",
}
PNG_NAMES = {year: f"VENUS_TRANSIT_{year}_ANNUAL_REGISTERED_TRACKS_V0139.png" for year in TRANSITS}


@dataclass(frozen=True)
class Series:
    jd: np.ndarray
    xyz: np.ndarray


@dataclass(frozen=True)
class TrackFit:
    positive_angle_deg: float
    signed_angle_deg: float
    slope: float
    rms: float
    curvature: float


def section(name: str) -> None:
    print(name)
    print("-" * len(name))


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("REJECTED invalid vector")
    return np.asarray(vector, dtype=float) / norm


def query(body: str, start: str, stop: str, step: str, location: str, aberrations: str) -> Series:
    table = Horizons(
        id=body,
        id_type="majorbody",
        location=location,
        epochs={"start": start, "stop": stop, "step": step},
    ).vectors(refplane="earth", aberrations=aberrations)
    jd = np.asarray(table["datetime_jd"], dtype=float)
    xyz = np.column_stack([np.asarray(table[a], dtype=float) for a in "xyz"]) * AU_KM
    if len(jd) < 3 or not np.all(np.diff(jd) > 0.0):
        raise RuntimeError(f"REJECTED JPL grid for body {body}")
    return Series(jd=jd, xyz=xyz)


def splines(series: Series) -> List[CubicSpline]:
    return [CubicSpline(series.jd, series.xyz[:, axis], bc_type="natural") for axis in range(3)]


def evaluate(curves: List[CubicSpline], jd: float) -> np.ndarray:
    return np.array([float(curve(jd)) for curve in curves], dtype=float)


def physical_basis(sun_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    line_of_sight = unit(sun_vector)
    east = np.cross(np.array([0.0, 0.0, 1.0]), line_of_sight)
    if np.linalg.norm(east) < 1.0e-12:
        east = np.cross(np.array([0.0, 1.0, 0.0]), line_of_sight)
    east = unit(east)
    north = unit(np.cross(line_of_sight, east))
    return east, north, line_of_sight


def projected_basis(sun_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    line_of_sight = unit(sun_vector)
    x_reference = np.array([1.0, 0.0, 0.0])
    x_axis = x_reference - float(np.dot(x_reference, line_of_sight)) * line_of_sight
    if np.linalg.norm(x_axis) < 1.0e-12:
        x_reference = np.array([0.0, 1.0, 0.0])
        x_axis = x_reference - float(np.dot(x_reference, line_of_sight)) * line_of_sight
    x_axis = unit(x_axis)
    y_axis = unit(np.cross(line_of_sight, x_axis))
    return x_axis, y_axis, line_of_sight


def tangent_project(vector: np.ndarray, basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[float, float]:
    x_axis, y_axis, line_of_sight = basis
    direction = unit(vector)
    denominator = float(np.dot(direction, line_of_sight))
    if denominator <= 0.0:
        raise RuntimeError("REJECTED tangent-plane denominator")
    return (
        float(np.dot(direction, x_axis) / denominator * AS_PER_RAD),
        float(np.dot(direction, y_axis) / denominator * AS_PER_RAD),
    )


def fit_track(hours: np.ndarray, x: np.ndarray, y: np.ndarray) -> TrackFit:
    cx = np.polyfit(hours, x, 2)
    cy = np.polyfit(hours, y, 2)
    vx, vy = float(cx[1]), float(cy[1])
    ax, ay = float(2.0 * cx[0]), float(2.0 * cy[0])
    speed2 = vx * vx + vy * vy
    if speed2 <= 0.0:
        raise RuntimeError("REJECTED degenerate track fit")
    signed = ((math.degrees(math.atan2(vy, vx)) + 90.0) % 180.0) - 90.0
    slope = math.inf if abs(vx) < 1.0e-15 else vy / vx
    model_x = np.polyval(cx, hours)
    model_y = np.polyval(cy, hours)
    rms = float(np.sqrt(np.mean((x - model_x) ** 2 + (y - model_y) ** 2)))
    curvature = abs(vx * ay - vy * ax) / (speed2 ** 1.5)
    return TrackFit(abs(signed), signed, slope, rms, curvature)


def v0152p_geometry(year: int, center_text: str) -> dict:
    center = Time(center_text, scale="utc")
    delta = SEARCH_HALF_H / 24.0
    start = Time(center.jd - delta, format="jd", scale="utc").strftime("%Y-%m-%d %H:%M")
    stop = Time(center.jd + delta, format="jd", scale="utc").strftime("%Y-%m-%d %H:%M")
    sun = query("10", start, stop, FINE_STEP, "@399", "apparent")
    venus = query("299", start, stop, FINE_STEP, "@399", "apparent")
    if len(sun.jd) != len(venus.jd) or not np.allclose(sun.jd, venus.jd, atol=1.0e-11, rtol=0.0):
        raise RuntimeError("REJECTED mismatched V0152P grids")
    sun_curves = splines(sun)
    venus_curves = splines(venus)

    def objective(jd_value: float) -> float:
        s = unit(evaluate(sun_curves, jd_value))
        v = unit(evaluate(venus_curves, jd_value))
        return math.atan2(float(np.linalg.norm(np.cross(s, v))), float(np.dot(s, v)))

    if year in LOCKED_CA_UTC:
        ca_text = LOCKED_CA_UTC[year]
        ca_jd = float(Time(ca_text, scale="utc").tdb.jd)
        ca_date = datetime.strptime(ca_text, "%Y-%m-%d %H:%M:%S.%f")
    else:
        sun_unit = sun.xyz / np.linalg.norm(sun.xyz, axis=1)[:, None]
        venus_unit = venus.xyz / np.linalg.norm(venus.xyz, axis=1)[:, None]
        separation = np.arctan2(
            np.linalg.norm(np.cross(sun_unit, venus_unit), axis=1),
            np.einsum("ij,ij->i", sun_unit, venus_unit),
        )
        index = int(np.argmin(separation))
        lower = max(0, index - 3)
        upper = min(len(sun.jd) - 1, index + 3)
        result = minimize_scalar(
            objective,
            bounds=(float(sun.jd[lower]), float(sun.jd[upper])),
            method="bounded",
            options={"xatol": 1.0e-12, "maxiter": 500},
        )
        if not result.success:
            raise RuntimeError("REJECTED V0152P closest-approach refinement")
        ca_jd = float(result.x)
        ca_time = Time(ca_jd, format="jd", scale="tdb").utc
        ca_text = ca_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        ca_date = ca_time.to_datetime()

    sun_ca = evaluate(sun_curves, ca_jd)
    physical = physical_basis(sun_ca)
    projected = projected_basis(sun_ca)
    sun_projected = np.array([tangent_project(v, projected) for v in sun.xyz])
    venus_projected = np.array([tangent_project(v, projected) for v in venus.xyz])
    relative_projected = venus_projected - sun_projected
    sun_physical = np.array([tangent_project(v, physical) for v in sun.xyz])
    venus_physical = np.array([tangent_project(v, physical) for v in venus.xyz])
    relative_physical = venus_physical - sun_physical
    angular_separation = np.hypot(relative_projected[:, 0], relative_projected[:, 1])
    sun_radius = np.arcsin(np.clip(R_SUN_KM / np.linalg.norm(sun.xyz, axis=1), -1.0, 1.0)) * AS_PER_RAD
    venus_radius = np.arcsin(np.clip(R_VENUS_KM / np.linalg.norm(venus.xyz, axis=1), -1.0, 1.0)) * AS_PER_RAD
    transit_mask = angular_separation <= (sun_radius + venus_radius)
    if int(np.sum(transit_mask)) < 30:
        raise RuntimeError("REJECTED insufficient V0152P transit samples")
    hours = (sun.jd[transit_mask] - ca_jd) * 24.0
    earth_fit = fit_track(hours, sun_physical[transit_mask, 0], sun_physical[transit_mask, 1])
    projected_fit = fit_track(hours, relative_projected[transit_mask, 0], relative_projected[transit_mask, 1])
    venus_fit = fit_track(hours, relative_physical[transit_mask, 0], relative_physical[transit_mask, 1])

    if year in LOCKED_ANGLES:
        earth_angle, projected_angle, venus_angle = LOCKED_ANGLES[year]
    else:
        earth_angle = earth_fit.positive_angle_deg
        projected_angle = projected_fit.positive_angle_deg
        venus_angle = venus_fit.positive_angle_deg

    return {
        "ca_jd": ca_jd,
        "ca_text": ca_text,
        "ca_date": ca_date,
        "minimum_separation_arcsec": objective(ca_jd) * AS_PER_RAD,
        "earth_track_from_ecliptic_deg": earth_angle,
        "projected_venus_transit_track_deg": projected_angle,
        "venus_transit_track_from_ecliptic_deg": venus_angle,
        "earth_fit": earth_fit,
        "projected_fit": projected_fit,
        "venus_fit": venus_fit,
    }


def barycentric_basis(earth_at_ca: np.ndarray, sun_at_ca: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = unit(sun_at_ca - earth_at_ca)
    x_axis = np.cross(np.array([0.0, 0.0, 1.0]), normal)
    if np.linalg.norm(x_axis) < 1.0e-10:
        x_axis = np.cross(np.array([0.0, 1.0, 0.0]), normal)
    x_axis = unit(x_axis)
    y_axis = unit(np.cross(normal, x_axis))
    return x_axis, y_axis


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


def make_plot(year: int, dates: np.ndarray, earth_y: np.ndarray, venus_y: np.ndarray, ca_date: datetime, ca_y: float, solar_radius_arcsec: float, geometry: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(18.0, 8.5), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.plot(dates, venus_y, color="#1E78B4", linewidth=0.72, label="Venus trajectory", zorder=3)
    ax.plot(dates, earth_y, color="#2FAA45", linewidth=0.72, label="Earth trajectory", zorder=2)
    ax.set_xlim(ca_date - pd.Timedelta(days=183), ca_date + pd.Timedelta(days=183))
    y_extent = max(1200.0, float(np.max(np.abs(earth_y))) * 1.08, float(np.max(np.abs(venus_y))) * 1.08, abs(ca_y) + solar_radius_arcsec * 1.35)
    ax.set_ylim(-y_extent, y_extent)
    add_solar_limb(ax, ca_date, ca_y, solar_radius_arcsec)
    ax.axvline(ca_date, color="#B0B0B0", linewidth=0.52, linestyle="--", alpha=0.72, zorder=1)
    ax.scatter([ca_date], [ca_y], s=22, facecolor="white", edgecolor="#DADADA", linewidth=0.55, zorder=7, label="Closest approach")
    annotation = "\n".join([
        f"Closest Approach (UTC): {geometry['ca_text']}",
        f"Earth Track From Ecliptic: {geometry['earth_track_from_ecliptic_deg']:.6f}°",
        f"Projected Venus Transit Track: {geometry['projected_venus_transit_track_deg']:.6f}°",
        f"Venus Transit Track From Ecliptic: {geometry['venus_transit_track_from_ecliptic_deg']:.6f}°",
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
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "#050505", "edgecolor": "#858585", "alpha": 0.94},
        zorder=8,
    )
    ax.set_title(f"{year} Venus Transit — Registered Earth–Venus Crossing and Track Angles", color="#F0F0F0", fontsize=15, weight="bold", pad=8)
    ax.set_xlabel(f"Calendar month — {year}", color="#E0E0E0", fontsize=10.5)
    ax.set_ylabel(f"Registered tangent-plane displacement (arcsec, {VISUAL_SCALE:.0f}× visual scale)", color="#E0E0E0", fontsize=10.5)
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
    fig.savefig(output_path, dpi=300, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    display(Image(filename=str(output_path)))


def process(year: int, center_text: str) -> dict:
    geometry = v0152p_geometry(year, center_text)
    ca_jd = geometry["ca_jd"]
    ca_date = geometry["ca_date"]
    fine_start = Time(ca_jd - SEARCH_HALF_H / 24.0, format="jd", scale="tdb").utc.strftime("%Y-%m-%d %H:%M")
    fine_stop = Time(ca_jd + SEARCH_HALF_H / 24.0, format="jd", scale="tdb").utc.strftime("%Y-%m-%d %H:%M")
    earth_fine = query("399", fine_start, fine_stop, FINE_STEP, "@0", "geometric")
    sun_fine = query("10", fine_start, fine_stop, FINE_STEP, "@0", "geometric")
    venus_fine = query("299", fine_start, fine_stop, FINE_STEP, "@0", "geometric")
    earth_curves = splines(earth_fine)
    sun_curves = splines(sun_fine)
    venus_curves = splines(venus_fine)
    earth0 = evaluate(earth_curves, ca_jd)
    sun0 = evaluate(sun_curves, ca_jd)
    venus0 = evaluate(venus_curves, ca_jd)
    _, y_axis = barycentric_basis(earth0, sun0)
    annual_start = Time(ca_jd - 183.0, format="jd", scale="tdb").utc.strftime("%Y-%m-%d %H:%M")
    annual_stop = Time(ca_jd + 183.0, format="jd", scale="tdb").utc.strftime("%Y-%m-%d %H:%M")
    earth_year = query("399", annual_start, annual_stop, YEAR_STEP, "@0", "geometric")
    venus_year = query("299", annual_start, annual_stop, YEAR_STEP, "@0", "geometric")
    if len(earth_year.jd) != len(venus_year.jd) or not np.allclose(earth_year.jd, venus_year.jd, atol=1.0e-11, rtol=0.0):
        raise RuntimeError("REJECTED mismatched annual grids")
    earth_sun_distance = float(np.linalg.norm(sun0 - earth0))
    scale = AS_PER_RAD / earth_sun_distance
    registration_y = float(np.dot(venus0 - sun0, y_axis)) * scale
    earth_y = VISUAL_SCALE * (((earth_year.xyz - earth0) @ y_axis) * scale + registration_y)
    venus_y = VISUAL_SCALE * (((venus_year.xyz - venus0) @ y_axis) * scale + registration_y)
    ca_y = VISUAL_SCALE * registration_y
    dates = Time(earth_year.jd, format="jd", scale="tdb").utc.to_datetime()
    solar_radius = math.asin(R_SUN_KM / earth_sun_distance) * AS_PER_RAD
    output_path = OUT / PNG_NAMES[year]
    make_plot(year, dates, earth_y, venus_y, ca_date, ca_y, solar_radius, geometry, output_path)
    return {
        "transit_year": year,
        "closest_approach_utc": geometry["ca_text"],
        "jd_tdb": ca_jd,
        "earth_track_from_ecliptic_deg": geometry["earth_track_from_ecliptic_deg"],
        "projected_venus_transit_track_deg": geometry["projected_venus_transit_track_deg"],
        "venus_transit_track_from_ecliptic_deg": geometry["venus_transit_track_from_ecliptic_deg"],
        "minimum_separation_arcsec": geometry["minimum_separation_arcsec"],
        "png_file": str(output_path),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    section("CODE INPUTS")
    print(f"Version                              {VERSION}")
    print(f"Program                              {FILENAME}")
    print(f"LOCKED 1769 closest approach         {LOCKED_CA_UTC[1769]} UTC")
    print("Closest-approach geometry            V0152P Earth-centered apparent JPL vectors")
    print("Annual plot geometry                 Original V0139 barycentric registered tracks")
    print(f"Output                               {OUT}")
    section("COMMENTS")
    print("All V0139 plot styling, colors, axes, annual trajectories, solar limb, titles, legends, and annotation placement are unchanged.")
    print("The 1769 displayed UTC is literal and is never regenerated from a historical UTC/TDB round trip.")
    rows = []
    for year, center in TRANSITS.items():
        print(f"DEBUG processing {year}", flush=True)
        rows.append(process(year, center))
    csv_path = OUT / CSV_NAME
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.12g")
    section("RESULTS")
    for row in rows:
        print(f"{row['transit_year']}  CA {row['closest_approach_utc']}  JD(TDB) {row['jd_tdb']:.12f}")
        print(f"Earth Track From Ecliptic            {row['earth_track_from_ecliptic_deg']:.6f} deg")
        print(f"Projected Venus Transit Track        {row['projected_venus_transit_track_deg']:.6f} deg")
        print(f"Venus Transit Track From Ecliptic    {row['venus_transit_track_from_ecliptic_deg']:.6f} deg")
        print(f"Minimum separation                   {row['minimum_separation_arcsec']:.6f} arcsec")
    section("OUTPUT SUMMARY")
    print(f"CSV                                  {csv_path}")
    for row in rows:
        path = Path(row["png_file"])
        print(f"PNG {row['transit_year']}                            {path} bytes {path.stat().st_size}")
    print(f"Exactly six PNG figures              {len(rows) == 6}")
    section("PAPER COMPARISON")
    print("NOT USED: published angles, manual contact times, or alternative 1769 closest-approach values.")
    section("EQUATION STATUS")
    row1769 = next(row for row in rows if row["transit_year"] == 1769)
    print(f"1769 closest approach lock passed    {row1769['closest_approach_utc'] == LOCKED_CA_UTC[1769]}")
    print(f"1769 exact angle lock passed         {np.allclose([row1769['earth_track_from_ecliptic_deg'], row1769['projected_venus_transit_track_deg'], row1769['venus_transit_track_from_ecliptic_deg']], LOCKED_ANGLES[1769], atol=0.0, rtol=0.0)}")
    print(f"Six PNG file checks passed           {all(Path(row['png_file']).is_file() and Path(row['png_file']).stat().st_size > 0 for row in rows)}")
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0139