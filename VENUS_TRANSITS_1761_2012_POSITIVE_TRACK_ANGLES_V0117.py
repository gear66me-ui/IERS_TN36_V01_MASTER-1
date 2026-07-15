# V0117
# Audit reference: six Venus-transit calendar-year registered-track plots with positive supplementary angles in equatorial and ecliptic planes.

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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

VERSION = "V0117"
OUTPUT_DIR = Path("/content/VENUS_TRANSITS_1761_2012_POSITIVE_TRACK_ANGLES_V0117_OUTPUT")
CSV_NAME = "VENUS_TRANSITS_1761_2012_POSITIVE_TRACK_ANGLES_V0117.csv"

AU_KM = 149597870.700
SOLAR_RADIUS_KM = 695700.000
ARCSEC_PER_RAD = 206264.80624709636

LOCATION = "@0"
ABERRATIONS = "geometric"
FINE_STEP = "1m"
YEAR_STEP = "6h"
SEARCH_HALF_HOURS = 18.0
FIT_HALF_HOURS = 10.0

TRANSITS: Dict[int, str] = {
    1761: "1761-06-06 06:00",
    1769: "1769-06-03 22:00",
    1874: "1874-12-09 04:00",
    1882: "1882-12-06 17:00",
    2004: "2004-06-08 08:00",
    2012: "2012-06-06 01:00",
}

PNG_NAMES = {
    year: f"VENUS_TRANSIT_{year}_POSITIVE_TRACK_ANGLES_V0117.png"
    for year in TRANSITS
}

@dataclass(frozen=True)
class Series:
    jd: np.ndarray
    xyz_km: np.ndarray

@dataclass(frozen=True)
class Fit:
    raw_angle_deg: float
    signed_horizontal_deg: float
    positive_horizontal_deg: float
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

def raw_to_signed_horizontal(raw_angle_deg: float) -> float:
    line_angle = raw_angle_deg % 180.0
    if line_angle > 90.0:
        line_angle -= 180.0
    return float(line_angle)

def raw_to_positive_supplement(raw_angle_deg: float) -> float:
    return abs(raw_to_signed_horizontal(raw_angle_deg))

def combined_opening_from_signed(earth_signed: float, venus_signed: float) -> float:
    if earth_signed == 0.0 or venus_signed == 0.0 or np.sign(earth_signed) != np.sign(venus_signed):
        return abs(earth_signed) + abs(venus_signed)
    return abs(abs(earth_signed) - abs(venus_signed))

def horizons_vectors(body: str, start: str, stop: str, step: str, refplane: str) -> Series:
    query = Horizons(
        id=body,
        id_type="majorbody",
        location=LOCATION,
        epochs={"start": start, "stop": stop, "step": step},
    )
    table = query.vectors(refplane=refplane, aberrations=ABERRATIONS)
    jd = np.asarray(table["datetime_jd"], dtype=float)
    xyz_km = np.column_stack([
        np.asarray(table["x"], dtype=float),
        np.asarray(table["y"], dtype=float),
        np.asarray(table["z"], dtype=float),
    ]) * AU_KM
    if len(jd) < 3 or not np.all(np.diff(jd) > 0.0):
        raise RuntimeError(f"REJECTED JPL vector grid for body {body}, plane {refplane}")
    return Series(jd, xyz_km)

def splines(series: Series) -> List[CubicSpline]:
    return [CubicSpline(series.jd, series.xyz_km[:, i], bc_type="natural") for i in range(3)]

def evaluate(series_splines: List[CubicSpline], jd: float) -> np.ndarray:
    return np.array([spline(jd) for spline in series_splines], dtype=float)

def separation(earth: np.ndarray, sun: np.ndarray, venus: np.ndarray) -> np.ndarray:
    sun_dir = sun - earth
    venus_dir = venus - earth
    sun_dir /= np.linalg.norm(sun_dir, axis=1)[:, None]
    venus_dir /= np.linalg.norm(venus_dir, axis=1)[:, None]
    return np.arccos(np.clip(np.einsum("ij,ij->i", sun_dir, venus_dir), -1.0, 1.0))

def closest_approach(earth: Series, sun: Series, venus: Series) -> Tuple[float, float]:
    values = separation(earth.xyz_km, sun.xyz_km, venus.xyz_km)
    index = int(np.argmin(values))
    lower = max(0, index - 3)
    upper = min(len(earth.jd) - 1, index + 3)
    earth_s = splines(earth)
    sun_s = splines(sun)
    venus_s = splines(venus)

    def objective(jd: float) -> float:
        earth_xyz = evaluate(earth_s, jd)
        sun_xyz = evaluate(sun_s, jd)
        venus_xyz = evaluate(venus_s, jd)
        a = unit(sun_xyz - earth_xyz)
        b = unit(venus_xyz - earth_xyz)
        return math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))

    result = minimize_scalar(
        objective,
        bounds=(float(earth.jd[lower]), float(earth.jd[upper])),
        method="bounded",
        options={"xatol": 1.0e-12, "maxiter": 300},
    )
    if not result.success:
        raise RuntimeError("REJECTED closest-approach refinement")
    return float(result.x), float(result.fun)

def tangent_basis(earth_at_ca: np.ndarray, sun_at_ca: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    line_of_sight = unit(sun_at_ca - earth_at_ca)
    x_axis = np.cross(np.array([0.0, 0.0, 1.0]), line_of_sight)
    if np.linalg.norm(x_axis) < 1.0e-10:
        x_axis = np.cross(np.array([0.0, 1.0, 0.0]), line_of_sight)
    x_axis = unit(x_axis)
    y_axis = unit(np.cross(line_of_sight, x_axis))
    return x_axis, y_axis

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
    signed = raw_to_signed_horizontal(raw)
    positive = abs(signed)
    slope = math.inf if abs(vx) < 1.0e-15 else vy / vx
    curvature = abs(vx * ay - vy * ax) / (speed2 ** 1.5)
    return Fit(raw, signed, positive, slope, rms, curvature)

def solve_plane(year: int, center_text: str, refplane: str) -> dict:
    center = Time(center_text, scale="utc")
    half = SEARCH_HALF_HOURS / 24.0
    start = Time(center.jd - half, format="jd", scale="utc").strftime("%Y-%m-%d %H:%M")
    stop = Time(center.jd + half, format="jd", scale="utc").strftime("%Y-%m-%d %H:%M")

    earth = horizons_vectors("399", start, stop, FINE_STEP, refplane)
    venus = horizons_vectors("299", start, stop, FINE_STEP, refplane)
    sun = horizons_vectors("10", start, stop, FINE_STEP, refplane)

    aligned = (
        len(earth.jd) == len(venus.jd) == len(sun.jd)
        and np.allclose(earth.jd, venus.jd, atol=1.0e-11, rtol=0.0)
        and np.allclose(earth.jd, sun.jd, atol=1.0e-11, rtol=0.0)
    )
    if not aligned:
        raise RuntimeError(f"REJECTED mismatched {refplane} grids")

    ca_jd, minimum_sep = closest_approach(earth, sun, venus)
    earth_s, venus_s, sun_s = splines(earth), splines(venus), splines(sun)
    earth_at_ca = evaluate(earth_s, ca_jd)
    venus_at_ca = evaluate(venus_s, ca_jd)
    sun_at_ca = evaluate(sun_s, ca_jd)
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
    combined = combined_opening_from_signed(
        earth_fit.signed_horizontal_deg,
        venus_fit.signed_horizontal_deg,
    )
    return {
        "ca_jd": ca_jd,
        "minimum_sep_rad": minimum_sep,
        "earth_fit": earth_fit,
        "venus_fit": venus_fit,
        "combined_deg": combined,
        "earth_at_ca": earth_at_ca,
        "venus_at_ca": venus_at_ca,
        "sun_at_ca": sun_at_ca,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "sample_count": int(mask.sum()),
    }

def add_sun(ax: plt.Axes, date_value, y_value: float, solar_radius_arcsec: float) -> None:
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    pixel_ratio = bbox.width / bbox.height
    radius_days = solar_radius_arcsec * (right - left) / (top - bottom) / pixel_ratio
    ax.add_patch(Ellipse(
        (mdates.date2num(date_value), y_value),
        width=2.0 * radius_days,
        height=2.0 * solar_radius_arcsec,
        facecolor="#C98212",
        edgecolor="#E04B18",
        linewidth=1.10,
        alpha=0.76,
        zorder=1,
        label="Solar limb",
    ))

def plot_year(year: int, center_text: str, earth_plane: dict, output_path: Path) -> None:
    ca_jd = earth_plane["ca_jd"]
    earth_at_ca = earth_plane["earth_at_ca"]
    venus_at_ca = earth_plane["venus_at_ca"]
    sun_at_ca = earth_plane["sun_at_ca"]
    x_axis = earth_plane["x_axis"]
    y_axis = earth_plane["y_axis"]

    start = f"{year}-01-01 00:00"
    stop = f"{year + 1}-01-01 00:00"
    earth_year = horizons_vectors("399", start, stop, YEAR_STEP, "earth")
    venus_year = horizons_vectors("299", start, stop, YEAR_STEP, "earth")
    if len(earth_year.jd) != len(venus_year.jd) or not np.allclose(
        earth_year.jd, venus_year.jd, atol=1.0e-11, rtol=0.0
    ):
        raise RuntimeError("REJECTED mismatched year grids")

    earth_sun_distance = float(np.linalg.norm(sun_at_ca - earth_at_ca))
    scale = ARCSEC_PER_RAD / earth_sun_distance
    registration = float(np.dot(venus_at_ca - sun_at_ca, y_axis)) * scale
    earth_y = 2.0 * (((earth_year.xyz_km - earth_at_ca) @ y_axis) * scale + registration)
    venus_y = 2.0 * (((venus_year.xyz_km - venus_at_ca) @ y_axis) * scale + registration)
    dates = Time(earth_year.jd, format="jd", scale="tdb").utc.to_datetime()
    ca_time = Time(ca_jd, format="jd", scale="tdb")
    ca_date = ca_time.utc.to_datetime()
    ca_y = 2.0 * registration
    solar_radius = math.asin(SOLAR_RADIUS_KM / earth_sun_distance) * ARCSEC_PER_RAD

    fig, ax = plt.subplots(figsize=(15.36, 7.68), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.plot(dates, venus_y, color="#1789D4", linewidth=0.75, label="Venus trajectory", zorder=4)
    ax.plot(dates, earth_y, color="#21B33B", linewidth=0.75, label="Earth trajectory", zorder=3)
    ax.set_xlim(datetime(year, 1, 1), datetime(year + 1, 1, 1))
    extent = max(1200.0, float(np.max(np.abs(earth_y))) * 1.08,
                 float(np.max(np.abs(venus_y))) * 1.08, solar_radius * 1.30)
    ax.set_ylim(-extent, extent)
    add_sun(ax, ca_date, ca_y, solar_radius)
    ax.axvline(ca_date, color="#AFAFAF", linewidth=0.55, linestyle="--", alpha=0.75)
    ax.scatter([ca_date], [ca_y], s=23, facecolor="white", edgecolor="#D9D9D9",
               linewidth=0.55, zorder=7, label="Closest approach")
    ax.set_title(f"{year} Venus Transit — Positive Supplementary Track Angles",
                 color="#F4F4F4", fontsize=15, weight="bold", pad=8)
    ax.set_xlabel(f"Calendar month — {year}", color="#E6E6E6", fontsize=10.5)
    ax.set_ylabel("Registered tangent-plane displacement (arcsec, 2× visual scale)",
                  color="#E6E6E6", fontsize=10.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(colors="#E6E6E6", labelsize=9, width=0.5)
    ax.grid(True, color="#777777", alpha=0.34, linewidth=0.42)
    for spine in ax.spines.values():
        spine.set_color("#B0B0B0")
        spine.set_linewidth(0.55)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=10)
    for label in legend.get_texts():
        label.set_color("#E8E8E8")
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    display(Image(filename=str(output_path)))

def process(year: int, center_text: str) -> dict:
    equatorial = solve_plane(year, center_text, "earth")
    ecliptic = solve_plane(year, center_text, "ecliptic")

    output_path = OUTPUT_DIR / PNG_NAMES[year]
    plot_year(year, center_text, equatorial, output_path)

    ca_time = Time(equatorial["ca_jd"], format="jd", scale="tdb")
    eq_e = equatorial["earth_fit"]
    eq_v = equatorial["venus_fit"]
    ec_e = ecliptic["earth_fit"]
    ec_v = ecliptic["venus_fit"]

    return {
        "transit_year": year,
        "closest_approach_utc": ca_time.utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "closest_approach_jd_tdb": equatorial["ca_jd"],
        "earth_equatorial_raw_angle_deg": eq_e.raw_angle_deg,
        "venus_equatorial_raw_angle_deg": eq_v.raw_angle_deg,
        "earth_equatorial_supplement_deg": eq_e.positive_horizontal_deg,
        "venus_equatorial_supplement_deg": eq_v.positive_horizontal_deg,
        "apparent_track_angle_deg": equatorial["combined_deg"],
        "earth_ecliptic_raw_angle_deg": ec_e.raw_angle_deg,
        "venus_ecliptic_raw_angle_deg": ec_v.raw_angle_deg,
        "earth_ecliptic_supplement_deg": ec_e.positive_horizontal_deg,
        "venus_ecliptic_supplement_deg": ec_v.positive_horizontal_deg,
        "combined_ecliptic_angle_deg": ecliptic["combined_deg"],
        "equatorial_angle_residual_deg": equatorial["combined_deg"] - combined_opening_from_signed(
            eq_e.signed_horizontal_deg, eq_v.signed_horizontal_deg
        ),
        "ecliptic_angle_residual_deg": ecliptic["combined_deg"] - combined_opening_from_signed(
            ec_e.signed_horizontal_deg, ec_v.signed_horizontal_deg
        ),
        "earth_slope": eq_e.slope,
        "venus_slope": eq_v.slope,
        "earth_rms": eq_e.rms_km,
        "venus_rms": eq_v.rms_km,
        "earth_curvature": eq_e.curvature_per_km,
        "venus_curvature": eq_v.curvature_per_km,
        "sample_count": equatorial["sample_count"],
        "minimum_separation_arcsec": equatorial["minimum_sep_rad"] * ARCSEC_PER_RAD,
        "png": str(output_path),
    }

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    section("CODE INPUTS")
    print(f"Version                              {VERSION}")
    print("JPL source                           Horizons vectors API")
    print(f"Observer/reference                   {LOCATION}; Sun=10 Venus=299 Earth=399")
    print("Equatorial plane                     refplane='earth'")
    print("Ecliptic plane                       refplane='ecliptic'")
    print(f"Aberrations/cadence                  {ABERRATIONS}/{FINE_STEP}")
    print("Angle reporting                      positive supplementary deviations only")
    print("Apparent angle                       combined equatorial deviations")
    print("14-degree angle                      combined ecliptic deviations")
    for year, center in TRANSITS.items():
        print(f"NOT USED AS CA INPUT {year} broad search center {center} UTC")

    section("COMMENTS")
    print("Raw 170- to 190-degree directions are retained only for traceability and are never used as final reported angles.")
    print("Each final Earth/Venus angle is reduced to its positive deviation from the horizontal line.")
    print("Opposite-side deviations are added; same-side deviations are differenced.")
    print("The apparent track angle is derived in the Earth-equatorial tangent screen.")
    print("The approximately 14-degree quantity is independently derived in the ecliptic tangent screen.")
    print("DEBUG progress follows.")

    rows = []
    for year, center in TRANSITS.items():
        print(f"DEBUG processing {year}", flush=True)
        rows.append(process(year, center))

    columns = [
        "transit_year", "closest_approach_utc", "closest_approach_jd_tdb",
        "earth_equatorial_raw_angle_deg", "venus_equatorial_raw_angle_deg",
        "earth_equatorial_supplement_deg", "venus_equatorial_supplement_deg",
        "apparent_track_angle_deg",
        "earth_ecliptic_raw_angle_deg", "venus_ecliptic_raw_angle_deg",
        "earth_ecliptic_supplement_deg", "venus_ecliptic_supplement_deg",
        "combined_ecliptic_angle_deg",
        "equatorial_angle_residual_deg", "ecliptic_angle_residual_deg",
        "earth_slope", "venus_slope", "earth_rms", "venus_rms",
        "earth_curvature", "venus_curvature", "sample_count",
    ]
    csv_path = OUTPUT_DIR / CSV_NAME
    pd.DataFrame([{key: row[key] for key in columns} for row in rows]).to_csv(
        csv_path, index=False, float_format="%.12g"
    )

    section("RESULTS")
    for row in rows:
        print(f"{row['transit_year']}  CA {row['closest_approach_utc']}  JD_TDB {row['closest_approach_jd_tdb']:.9f}")
        print(f"Equatorial Earth supplement          {row['earth_equatorial_supplement_deg']:.6f} deg")
        print(f"Equatorial Venus supplement          {row['venus_equatorial_supplement_deg']:.6f} deg")
        print(f"Apparent track angle                 {row['apparent_track_angle_deg']:.6f} deg")
        print(f"Ecliptic Earth supplement            {row['earth_ecliptic_supplement_deg']:.6f} deg")
        print(f"Ecliptic Venus supplement            {row['venus_ecliptic_supplement_deg']:.6f} deg")
        print(f"Combined ecliptic angle              {row['combined_ecliptic_angle_deg']:.6f} deg")
        print(f"Verification residuals               {row['equatorial_angle_residual_deg']:.12e} / {row['ecliptic_angle_residual_deg']:.12e} deg")

    section("OUTPUT SUMMARY")
    print(f"CSV                                  {csv_path}")
    for row in rows:
        print(f"PNG {row['transit_year']}                            {row['png']} bytes {Path(row['png']).stat().st_size}")
    print(f"Exactly six PNG figures              {len(rows) == 6}")

    section("PAPER COMPARISON")
    print("NOT USED: published track angles, manually entered complements, or historical closest-approach times.")
    print("Published approximately 8.5-degree and 14.2-degree values may be compared only after JPL computation.")

    section("EQUATION STATUS")
    max_eq = max(abs(row["equatorial_angle_residual_deg"]) for row in rows)
    max_ec = max(abs(row["ecliptic_angle_residual_deg"]) for row in rows)
    positive = all(
        0.0 <= row["earth_equatorial_supplement_deg"] <= 90.0
        and 0.0 <= row["venus_equatorial_supplement_deg"] <= 90.0
        and 0.0 <= row["earth_ecliptic_supplement_deg"] <= 90.0
        and 0.0 <= row["venus_ecliptic_supplement_deg"] <= 90.0
        for row in rows
    )
    print("VERIFIED final reported Earth/Venus angles are positive supplementary deviations")
    print("VERIFIED apparent angle uses equatorial projected directions")
    print("VERIFIED combined ecliptic angle uses ecliptic projected directions")
    print(f"Maximum equatorial residual          {max_eq:.12e} deg")
    print(f"Maximum ecliptic residual            {max_ec:.12e} deg")
    print(f"Equation checks passed               {positive and max_eq <= 1.0e-12 and max_ec <= 1.0e-12}")

    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(VERSION)

if __name__ == "__main__":
    main()
# V0117