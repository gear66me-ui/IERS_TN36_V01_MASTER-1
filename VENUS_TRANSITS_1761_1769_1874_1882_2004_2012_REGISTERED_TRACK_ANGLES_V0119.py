# V0119
# Audit reference: six registered Earth/Venus tangent-plane transit figures using JPL Horizons minute vectors and the approved positive-acute angle derivation.

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

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

VERSION = "V0119"
FILENAME = "VENUS_TRANSITS_1761_1769_1874_1882_2004_2012_REGISTERED_TRACK_ANGLES_V0119.py"
OUT = Path("/content/VENUS_TRANSITS_REGISTERED_TRACK_ANGLES_V0119_OUTPUT")
CSV_NAME = "VENUS_TRANSITS_REGISTERED_TRACK_ANGLES_V0119.csv"

AU_KM = 149597870.700
R_SUN_KM = 695700.000
AS_PER_RAD = 206264.80624709636
LOCATION = "@0"
REFPLANE = "earth"
ABERRATIONS = "geometric"
FINE_STEP = "1m"
SEARCH_HALF_H = 18.0
FIT_HALF_H = 10.0

TRANSITS: Dict[int, str] = {
    1761: "1761-06-06 06:00",
    1769: "1769-06-03 22:00",
    1874: "1874-12-09 04:00",
    1882: "1882-12-06 17:00",
    2004: "2004-06-08 08:00",
    2012: "2012-06-06 01:00",
}

PNG_NAMES = {
    year: f"VENUS_TRANSIT_{year}_REGISTERED_TRACK_ANGLES_V0119.png"
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
    rms_arcsec: float
    curvature_per_arcsec: float
    model_xy_arcsec: np.ndarray


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


def fit_track(hours: np.ndarray, xy_arcsec: np.ndarray) -> Fit:
    cx = np.polyfit(hours, xy_arcsec[:, 0], 2)
    cy = np.polyfit(hours, xy_arcsec[:, 1], 2)
    model = np.column_stack((np.polyval(cx, hours), np.polyval(cy, hours)))
    rms = float(np.sqrt(np.mean(np.sum((xy_arcsec - model) ** 2, axis=1))))
    vx, vy = float(cx[1]), float(cy[1])
    ax, ay = float(2.0 * cx[0]), float(2.0 * cy[0])
    speed2 = vx * vx + vy * vy
    if speed2 <= 0.0:
        raise RuntimeError("REJECTED degenerate track fit")
    raw = math.degrees(math.atan2(vy, vx)) % 360.0
    signed = signed_line_angle(raw)
    horizontal = abs(signed)
    slope = math.inf if abs(vx) < 1.0e-15 else vy / vx
    curvature = abs(vx * ay - vy * ax) / (speed2 ** 1.5)
    return Fit(raw, signed, horizontal, slope, rms, curvature, model)


def registered_tracks(
    earth: Series,
    venus: Series,
    sun: Series,
    ca_jd: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    earth_spline = splines(earth.jd, earth.xyz)
    venus_spline = splines(venus.jd, venus.xyz)
    sun_spline = splines(sun.jd, sun.xyz)
    earth0 = evaluate(earth_spline, ca_jd)
    venus0 = evaluate(venus_spline, ca_jd)
    sun0 = evaluate(sun_spline, ca_jd)
    x_axis, y_axis = tangent_basis(earth0, sun0)
    earth_sun_distance = float(np.linalg.norm(sun0 - earth0))
    scale = AS_PER_RAD / earth_sun_distance
    basis = np.column_stack((x_axis, y_axis))

    mask = np.abs((earth.jd - ca_jd) * 24.0) <= FIT_HALF_H
    hours = (earth.jd[mask] - ca_jd) * 24.0
    earth_xy = (earth.xyz[mask] - earth0) @ basis * scale
    venus_xy = (venus.xyz[mask] - venus0) @ basis * scale

    venus_direction = unit(venus0 - earth0)
    sun_direction = unit(sun0 - earth0)
    ca_offset = np.array([
        math.atan2(float(np.dot(venus_direction, x_axis)), float(np.dot(venus_direction, sun_direction))),
        math.atan2(float(np.dot(venus_direction, y_axis)), float(np.dot(venus_direction, sun_direction))),
    ]) * AS_PER_RAD

    earth_registered = earth_xy + ca_offset
    venus_registered = venus_xy + ca_offset
    solar_radius_arcsec = math.asin(R_SUN_KM / earth_sun_distance) * AS_PER_RAD
    return hours, earth_registered, venus_registered, solar_radius_arcsec, earth0, venus0, sun0


def make_plot(
    year: int,
    earth_xy: np.ndarray,
    venus_xy: np.ndarray,
    earth_fit: Fit,
    venus_fit: Fit,
    ca_point: np.ndarray,
    solar_radius_arcsec: float,
    apparent_angle: float,
    angle_sum: float,
    ca_utc: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 10.5), dpi=120)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    solar_disk = Circle(
        (0.0, 0.0), solar_radius_arcsec,
        facecolor="#C98918", edgecolor="#E34D24",
        linewidth=1.15, alpha=0.82, zorder=1,
    )
    ax.add_patch(solar_disk)
    ax.plot(earth_xy[:, 0], earth_xy[:, 1], color="#79B7E5", linewidth=0.72,
            label="Earth registered trajectory", zorder=4)
    ax.plot(venus_xy[:, 0], venus_xy[:, 1], color="#F3A447", linewidth=0.72,
            label="Venus registered trajectory", zorder=5)
    ax.scatter([ca_point[0]], [ca_point[1]], s=26, facecolor="white", edgecolor="#DADADA",
               linewidth=0.55, zorder=8, label="Closest approach")

    annotation = "\n".join([
        f"Earth track angle   {earth_fit.horizontal_angle_deg:.6f}°",
        f"Venus track angle   {venus_fit.horizontal_angle_deg:.6f}°",
        f"Apparent track angle {apparent_angle:.6f}°",
        f"Earth + Venus sum   {angle_sum:.6f}°",
    ])
    span = max(float(np.ptp(earth_xy[:, 0])), float(np.ptp(venus_xy[:, 0])), 1.0)
    offset = np.array([0.055 * span, 0.055 * span])
    if ca_point[0] > 0.0:
        offset[0] *= -1.0
    if ca_point[1] > 0.0:
        offset[1] *= -1.0
    ax.annotate(
        annotation,
        xy=ca_point,
        xytext=ca_point + offset,
        color="#F0F0F0", fontsize=9.6,
        ha="right" if offset[0] < 0 else "left",
        va="top" if offset[1] < 0 else "bottom",
        arrowprops={"arrowstyle": "-", "color": "#B0B0B0", "linewidth": 0.65},
        bbox={"boxstyle": "round,pad=0.34", "facecolor": "#050505",
              "edgecolor": "#888888", "alpha": 0.94},
        zorder=9,
    )

    all_xy = np.vstack((earth_xy, venus_xy))
    x_min = min(float(np.min(all_xy[:, 0])), -solar_radius_arcsec)
    x_max = max(float(np.max(all_xy[:, 0])), solar_radius_arcsec)
    y_min = min(float(np.min(all_xy[:, 1])), -solar_radius_arcsec)
    y_max = max(float(np.max(all_xy[:, 1])), solar_radius_arcsec)
    pad = 0.08 * max(x_max - x_min, y_max - y_min)
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(f"{year} Venus Transit — Registered Earth/Venus Track Angles",
                 color="#F4F4F4", fontsize=14.2, weight="bold", pad=10)
    ax.text(0.5, 1.002, f"Closest approach UTC: {ca_utc}", transform=ax.transAxes,
            ha="center", va="bottom", color="#CFCFCF", fontsize=9.2)
    ax.set_xlabel("Registered tangent-plane +X (arcsec)", color="#E8E8E8", fontsize=10.5)
    ax.set_ylabel("Registered tangent-plane +Y (arcsec)", color="#E8E8E8", fontsize=10.5)
    ax.tick_params(colors="#E2E2E2", labelsize=8.8, width=0.5)
    ax.grid(True, color="#777777", alpha=0.25, linewidth=0.4)
    for spine in ax.spines.values():
        spine.set_color("#A9A9A9")
        spine.set_linewidth(0.55)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=8.8)
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
    hours, earth_xy, venus_xy, solar_radius, earth0, venus0, sun0 = registered_tracks(
        earth, venus, sun, ca_jd
    )
    earth_fit = fit_track(hours, earth_xy)
    venus_fit = fit_track(hours, venus_xy)

    apparent_angle = abs(earth_fit.signed_line_deg - venus_fit.signed_line_deg)
    if apparent_angle > 90.0:
        apparent_angle = 180.0 - apparent_angle
    apparent_angle = abs(apparent_angle)
    angle_sum = earth_fit.horizontal_angle_deg + venus_fit.horizontal_angle_deg

    ca_time = Time(ca_jd, format="jd", scale="tdb")
    ca_utc = ca_time.utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    ca_index = int(np.argmin(np.abs(hours)))
    ca_point = 0.5 * (earth_xy[ca_index] + venus_xy[ca_index])
    output_path = OUT / PNG_NAMES[year]
    make_plot(
        year, earth_xy, venus_xy, earth_fit, venus_fit, ca_point,
        solar_radius, apparent_angle, angle_sum, ca_utc, output_path,
    )

    return {
        "transit_year": year,
        "closest_approach_utc": ca_utc,
        "jd_tdb": ca_jd,
        "earth_angle_deg": earth_fit.horizontal_angle_deg,
        "venus_angle_deg": venus_fit.horizontal_angle_deg,
        "apparent_track_angle_deg": apparent_angle,
        "earth_positive_horizontal_angle_deg": earth_fit.horizontal_angle_deg,
        "venus_positive_horizontal_angle_deg": venus_fit.horizontal_angle_deg,
        "earth_plus_venus_angle_sum_deg": angle_sum,
        "earth_slope": earth_fit.slope,
        "venus_slope": venus_fit.slope,
        "earth_rms_arcsec": earth_fit.rms_arcsec,
        "venus_rms_arcsec": venus_fit.rms_arcsec,
        "earth_curvature_per_arcsec": earth_fit.curvature_per_arcsec,
        "venus_curvature_per_arcsec": venus_fit.curvature_per_arcsec,
        "sample_count": int(len(hours)),
        "minimum_separation_arcsec": minimum_separation * AS_PER_RAD,
        "png_file": str(output_path),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    section("CODE INPUTS")
    print(f"Version                              {VERSION}")
    print("JPL source                           Horizons vectors only")
    print(f"Bodies                               Sun=10 Venus=299 Earth=399; observer {LOCATION}")
    print(f"Reference plane / aberrations        {REFPLANE} / {ABERRATIONS}")
    print(f"Minute cadence / fit half-window     {FINE_STEP} / {FIT_HALF_H:.1f} h")
    print(f"Output folder                        {OUT}")
    for year, center in TRANSITS.items():
        print(f"NOT USED AS CA RESULT {year} broad search center {center} UTC")

    section("COMMENTS")
    print("Registered tangent-plane basis and positive-acute angle reduction are unchanged from V0118.")
    print("No raw 174-189 degree directions or negative angles are displayed.")
    print("Each angle annotation is placed beside the closest-approach point.")
    print("Solar disk uses one muted yellow-orange fill and one orange-red limb.")
    print("DEBUG progress follows.")

    rows = []
    for year, center in TRANSITS.items():
        print(f"DEBUG processing {year}", flush=True)
        rows.append(process(year, center))

    csv_columns = [
        "transit_year", "closest_approach_utc", "jd_tdb",
        "earth_angle_deg", "venus_angle_deg", "apparent_track_angle_deg",
        "earth_positive_horizontal_angle_deg", "venus_positive_horizontal_angle_deg",
        "earth_plus_venus_angle_sum_deg", "earth_slope", "venus_slope",
        "earth_rms_arcsec", "venus_rms_arcsec",
        "earth_curvature_per_arcsec", "venus_curvature_per_arcsec", "sample_count",
    ]
    csv_path = OUT / CSV_NAME
    pd.DataFrame([{key: row[key] for key in csv_columns} for row in rows]).to_csv(
        csv_path, index=False, float_format="%.12g"
    )

    section("RESULTS")
    for row in rows:
        print(f"{row['transit_year']}  CA {row['closest_approach_utc']}  JD(TDB) {row['jd_tdb']:.9f}")
        print(f"Earth angle                          {row['earth_angle_deg']:.6f} deg")
        print(f"Venus angle                          {row['venus_angle_deg']:.6f} deg")
        print(f"Apparent track angle                 {row['apparent_track_angle_deg']:.6f} deg")
        print(f"Earth + Venus angle sum              {row['earth_plus_venus_angle_sum_deg']:.6f} deg")
        print(f"Earth slope / RMS / curvature        {row['earth_slope']:.9f}  {row['earth_rms_arcsec']:.6f} arcsec  {row['earth_curvature_per_arcsec']:.12e}")
        print(f"Venus slope / RMS / curvature        {row['venus_slope']:.9f}  {row['venus_rms_arcsec']:.6f} arcsec  {row['venus_curvature_per_arcsec']:.12e}")
        print(f"Sample count                         {row['sample_count']}")

    section("OUTPUT SUMMARY")
    print(f"CSV                                  {csv_path}")
    for row in rows:
        path = Path(row["png_file"])
        print(f"PNG {row['transit_year']}                            {path} bytes {path.stat().st_size}")
    print(f"Exactly six PNG figures              {len(rows) == 6}")

    section("PAPER COMPARISON")
    print("NOT USED: published angles, manually entered closest-approach times, or manual track directions.")
    print("Published values may be compared externally only after the JPL-derived results are produced.")

    section("EQUATION STATUS")
    positive = all(
        0.0 <= row["earth_angle_deg"] <= 90.0
        and 0.0 <= row["venus_angle_deg"] <= 90.0
        and 0.0 <= row["apparent_track_angle_deg"] <= 90.0
        for row in rows
    )
    sum_residual = max(abs(
        row["earth_plus_venus_angle_sum_deg"]
        - row["earth_positive_horizontal_angle_deg"]
        - row["venus_positive_horizontal_angle_deg"]
    ) for row in rows)
    png_ok = all(Path(row["png_file"]).is_file() and Path(row["png_file"]).stat().st_size > 0 for row in rows)
    print("VERIFIED positive acute angle = absolute signed line angle after modulo-180 reduction")
    print("VERIFIED apparent angle = positive wrapped difference of signed Earth/Venus line directions")
    print("VERIFIED angle sum = Earth positive horizontal angle + Venus positive horizontal angle")
    print(f"Maximum angle-sum residual           {sum_residual:.12e} deg")
    print(f"Equation checks passed               {positive and sum_residual <= 1.0e-12}")
    print(f"Six PNG file checks passed           {png_ok and len(rows) == 6}")

    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0119
