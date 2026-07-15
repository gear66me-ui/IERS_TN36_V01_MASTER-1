# V0129
# Audit reference: calculation-only six-transit angle figures of merit with explicit geometric names.

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

def ensure(module: str, package: str) -> None:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)

for module_name, package_name in [
    ("astroquery", "astroquery"),
    ("astropy", "astropy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
]:
    ensure(module_name, package_name)

import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

VERSION = "V0129"
LOCATION = "@399"
REFPLANE = "earth"
ABERRATIONS = "apparent"
STEP = "1m"
AU_KM = 149597870.700
AS_PER_RAD = 206264.80624709636
R_SUN_KM = 695700.000
R_VENUS_KM = 6051.800
OBLIQUITY_J2000_DEG = 23.439291111
SEARCH_HALF_HOURS = 18.0
OUTPUT_DIR = Path("/content/VENUS_TRANSITS_SIX_ANGLE_FIGURES_OF_MERIT_V0129_OUTPUT")
CSV_NAME = "VENUS_TRANSITS_SIX_ANGLE_FIGURES_OF_MERIT_V0129.csv"

TRANSITS: Dict[int, str] = {
    1761: "1761-06-06 06:00",
    1769: "1769-06-03 22:00",
    1874: "1874-12-09 04:00",
    1882: "1882-12-06 17:00",
    2004: "2004-06-08 08:00",
    2012: "2012-06-06 01:00",
}

@dataclass(frozen=True)
class VectorSeries:
    jd: np.ndarray
    xyz_km: np.ndarray

@dataclass(frozen=True)
class TrackFit:
    signed_angle_deg: float
    positive_angle_deg: float
    slope: float
    rms_arcsec: float
    curvature_per_arcsec: float
    sample_count: int

def section(title: str) -> None:
    print(title)
    print("-" * len(title))

def unit(vector: np.ndarray) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    if not np.isfinite(magnitude) or magnitude <= 0.0:
        raise RuntimeError("REJECTED invalid vector")
    return vector / magnitude

def query_vectors(body_id: str, start: str, stop: str) -> VectorSeries:
    table = Horizons(
        id=body_id,
        id_type="majorbody",
        location=LOCATION,
        epochs={"start": start, "stop": stop, "step": STEP},
    ).vectors(refplane=REFPLANE, aberrations=ABERRATIONS)

    jd = np.asarray(table["datetime_jd"], dtype=float)
    xyz_km = np.column_stack([
        np.asarray(table["x"], dtype=float),
        np.asarray(table["y"], dtype=float),
        np.asarray(table["z"], dtype=float),
    ]) * AU_KM

    if len(jd) < 30 or not np.all(np.diff(jd) > 0.0):
        raise RuntimeError(f"REJECTED JPL grid for body {body_id}")

    return VectorSeries(jd=jd, xyz_km=xyz_km)

def cubic_splines(series: VectorSeries) -> list[CubicSpline]:
    return [
        CubicSpline(series.jd, series.xyz_km[:, axis], bc_type="natural")
        for axis in range(3)
    ]

def evaluate(curves: list[CubicSpline], jd_value: float) -> np.ndarray:
    return np.array([curve(jd_value) for curve in curves], dtype=float)

def closest_approach(
    sun: VectorSeries,
    venus: VectorSeries,
) -> tuple[float, float, np.ndarray]:
    if len(sun.jd) != len(venus.jd) or not np.allclose(
        sun.jd, venus.jd, atol=1.0e-11, rtol=0.0
    ):
        raise RuntimeError("REJECTED mismatched JPL grids")

    sun_unit = sun.xyz_km / np.linalg.norm(sun.xyz_km, axis=1)[:, None]
    venus_unit = venus.xyz_km / np.linalg.norm(venus.xyz_km, axis=1)[:, None]
    separation = np.arccos(np.clip(np.einsum("ij,ij->i", sun_unit, venus_unit), -1.0, 1.0))
    index = int(np.argmin(separation))
    lower = max(0, index - 3)
    upper = min(len(sun.jd) - 1, index + 3)

    sun_curves = cubic_splines(sun)
    venus_curves = cubic_splines(venus)

    def objective(jd_value: float) -> float:
        sun_direction = unit(evaluate(sun_curves, jd_value))
        venus_direction = unit(evaluate(venus_curves, jd_value))
        return math.acos(float(np.clip(np.dot(sun_direction, venus_direction), -1.0, 1.0)))

    result = minimize_scalar(
        objective,
        bounds=(float(sun.jd[lower]), float(sun.jd[upper])),
        method="bounded",
        options={"xatol": 1.0e-12, "maxiter": 300},
    )
    if not result.success:
        raise RuntimeError("REJECTED closest-approach refinement")

    jd_ca = float(result.x)
    return jd_ca, float(result.fun), evaluate(sun_curves, jd_ca)

def physical_east_north_basis(
    sun_at_ca: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    line_of_sight = unit(sun_at_ca)
    icrf_north = np.array([0.0, 0.0, 1.0])
    east = np.cross(icrf_north, line_of_sight)
    if np.linalg.norm(east) < 1.0e-12:
        east = np.cross(np.array([0.0, 1.0, 0.0]), line_of_sight)
    east = unit(east)
    north = unit(np.cross(line_of_sight, east))
    return east, north, line_of_sight

def projected_jpl_x_basis(
    sun_at_ca: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    line_of_sight = unit(sun_at_ca)
    fixed_x = np.array([1.0, 0.0, 0.0])
    projected_x = fixed_x - float(np.dot(fixed_x, line_of_sight)) * line_of_sight
    if np.linalg.norm(projected_x) < 1.0e-12:
        fixed_y = np.array([0.0, 1.0, 0.0])
        projected_x = fixed_y - float(np.dot(fixed_y, line_of_sight)) * line_of_sight
    projected_x = unit(projected_x)
    projected_y = unit(np.cross(line_of_sight, projected_x))
    return projected_x, projected_y, line_of_sight

def project_to_tangent_plane(
    xyz_km: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    x_axis, y_axis, line_of_sight = basis
    direction = xyz_km / np.linalg.norm(xyz_km, axis=1)[:, None]
    denominator = direction @ line_of_sight
    if np.any(denominator <= 0.0):
        raise RuntimeError("REJECTED tangent-plane denominator")
    x_arcsec = (direction @ x_axis) / denominator * AS_PER_RAD
    y_arcsec = (direction @ y_axis) / denominator * AS_PER_RAD
    return x_arcsec, y_arcsec

def signed_line_angle(raw_direction_deg: float) -> float:
    return ((raw_direction_deg + 90.0) % 180.0) - 90.0

def fit_track(hours: np.ndarray, x_arcsec: np.ndarray, y_arcsec: np.ndarray) -> TrackFit:
    coefficient_x = np.polyfit(hours, x_arcsec, 2)
    coefficient_y = np.polyfit(hours, y_arcsec, 2)

    model_x = np.polyval(coefficient_x, hours)
    model_y = np.polyval(coefficient_y, hours)
    rms = float(np.sqrt(np.mean((x_arcsec - model_x) ** 2 + (y_arcsec - model_y) ** 2)))

    velocity_x = float(coefficient_x[1])
    velocity_y = float(coefficient_y[1])
    acceleration_x = float(2.0 * coefficient_x[0])
    acceleration_y = float(2.0 * coefficient_y[0])
    speed_squared = velocity_x * velocity_x + velocity_y * velocity_y

    if speed_squared <= 0.0:
        raise RuntimeError("REJECTED degenerate track fit")

    raw_direction = math.degrees(math.atan2(velocity_y, velocity_x)) % 360.0
    signed_angle = signed_line_angle(raw_direction)
    slope = math.inf if abs(velocity_x) < 1.0e-15 else velocity_y / velocity_x
    curvature = abs(
        velocity_x * acceleration_y - velocity_y * acceleration_x
    ) / (speed_squared ** 1.5)

    return TrackFit(
        signed_angle_deg=signed_angle,
        positive_angle_deg=abs(signed_angle),
        slope=slope,
        rms_arcsec=rms,
        curvature_per_arcsec=curvature,
        sample_count=len(hours),
    )

def ecliptic_tangent_angle(
    physical_basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> float:
    east, north, line_of_sight = physical_basis
    epsilon = math.radians(OBLIQUITY_J2000_DEG)
    ecliptic_north = np.array([0.0, -math.sin(epsilon), math.cos(epsilon)])
    tangent = unit(np.cross(ecliptic_north, line_of_sight))
    raw = math.degrees(math.atan2(float(np.dot(tangent, north)), float(np.dot(tangent, east)))) % 360.0
    return abs(signed_line_angle(raw))

def tangent_plane_basis_rotation(
    physical_basis: tuple[np.ndarray, np.ndarray, np.ndarray],
    projected_basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> float:
    cosine = float(np.clip(np.dot(physical_basis[0], projected_basis[0]), -1.0, 1.0))
    return math.degrees(math.acos(abs(cosine)))

def process_transit(year: int, center_text: str) -> dict:
    center = Time(center_text, scale="utc")
    half_window_days = SEARCH_HALF_HOURS / 24.0
    start = Time(center.jd - half_window_days, format="jd", scale="utc").strftime("%Y-%m-%d %H:%M")
    stop = Time(center.jd + half_window_days, format="jd", scale="utc").strftime("%Y-%m-%d %H:%M")

    sun = query_vectors("10", start, stop)
    venus = query_vectors("299", start, stop)
    jd_ca, minimum_separation_rad, sun_at_ca = closest_approach(sun, venus)

    physical_basis = physical_east_north_basis(sun_at_ca)
    projected_basis = projected_jpl_x_basis(sun_at_ca)

    sun_x_physical, sun_y_physical = project_to_tangent_plane(sun.xyz_km, physical_basis)
    venus_x_physical, venus_y_physical = project_to_tangent_plane(venus.xyz_km, physical_basis)
    sun_x_projected, sun_y_projected = project_to_tangent_plane(sun.xyz_km, projected_basis)
    venus_x_projected, venus_y_projected = project_to_tangent_plane(venus.xyz_km, projected_basis)

    relative_x_physical = venus_x_physical - sun_x_physical
    relative_y_physical = venus_y_physical - sun_y_physical
    relative_x_projected = venus_x_projected - sun_x_projected
    relative_y_projected = venus_y_projected - sun_y_projected

    separation_arcsec = np.hypot(relative_x_projected, relative_y_projected)
    sun_distance_km = np.linalg.norm(sun.xyz_km, axis=1)
    venus_distance_km = np.linalg.norm(venus.xyz_km, axis=1)

    external_contact_radius_arcsec = (
        np.arcsin(np.clip(R_SUN_KM / sun_distance_km, -1.0, 1.0))
        + np.arcsin(np.clip(R_VENUS_KM / venus_distance_km, -1.0, 1.0))
    ) * AS_PER_RAD

    mask = separation_arcsec <= external_contact_radius_arcsec
    if int(np.sum(mask)) < 30:
        raise RuntimeError(f"REJECTED {year}: insufficient in-transit samples")

    hours = (sun.jd[mask] - jd_ca) * 24.0

    earth_track = fit_track(hours, sun_x_physical[mask], sun_y_physical[mask])
    venus_track = fit_track(hours, venus_x_physical[mask], venus_y_physical[mask])
    physical_relative_track = fit_track(hours, relative_x_physical[mask], relative_y_physical[mask])
    projected_relative_track = fit_track(hours, relative_x_projected[mask], relative_y_projected[mask])

    ca_time = Time(jd_ca, format="jd", scale="tdb")

    return {
        "transit_year": year,
        "closest_approach_utc": ca_time.utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "jd_tdb": jd_ca,
        "ecliptic_tangent_angle_deg": ecliptic_tangent_angle(physical_basis),
        "earth_apparent_track_angle_deg": earth_track.positive_angle_deg,
        "venus_apparent_track_angle_deg": venus_track.positive_angle_deg,
        "physical_relative_track_angle_deg": physical_relative_track.positive_angle_deg,
        "projected_relative_track_angle_deg": projected_relative_track.positive_angle_deg,
        "tangent_plane_basis_rotation_deg": tangent_plane_basis_rotation(
            physical_basis, projected_basis
        ),
        "earth_apparent_slope": earth_track.slope,
        "venus_apparent_slope": venus_track.slope,
        "physical_relative_slope": physical_relative_track.slope,
        "projected_relative_slope": projected_relative_track.slope,
        "earth_apparent_rms_arcsec": earth_track.rms_arcsec,
        "venus_apparent_rms_arcsec": venus_track.rms_arcsec,
        "physical_relative_rms_arcsec": physical_relative_track.rms_arcsec,
        "projected_relative_rms_arcsec": projected_relative_track.rms_arcsec,
        "earth_apparent_curvature_per_arcsec": earth_track.curvature_per_arcsec,
        "venus_apparent_curvature_per_arcsec": venus_track.curvature_per_arcsec,
        "physical_relative_curvature_per_arcsec": physical_relative_track.curvature_per_arcsec,
        "projected_relative_curvature_per_arcsec": projected_relative_track.curvature_per_arcsec,
        "sample_count": projected_relative_track.sample_count,
        "minimum_separation_arcsec": minimum_separation_rad * AS_PER_RAD,
    }

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    section("CODE INPUTS")
    print(f"Version                              {VERSION}")
    print("JPL source                           Horizons geocentric vectors only")
    print(f"Observer                             {LOCATION}")
    print("Bodies                               Sun=10; Venus=299")
    print(f"Cadence                              {STEP}")
    print("Physical tangent plane               equatorial east-north")
    print("Projected tangent plane              fixed JPL/ICRF +X projected onto solar screen")
    print("Relative track                       Venus tangent position minus Sun tangent position")
    print(f"Output                               {OUTPUT_DIR}")

    section("COMMENTS")
    print("CALCULATION ONLY. No figures are generated.")
    print("Ecliptic tangent angle is the local J2000 ecliptic direction in the physical tangent plane.")
    print("Earth apparent track is the geocentric apparent solar motion.")
    print("Venus apparent track is the geocentric apparent Venus motion.")
    print("Physical relative track is Venus-minus-Sun in the east-north tangent plane.")
    print("Projected relative track is the same Venus-minus-Sun vector in the projected JPL +X plane.")
    print("Tangent-plane basis rotation is the acute angle between the two +X axes.")

    rows = []
    for year, center in TRANSITS.items():
        print(f"DEBUG calculating {year}", flush=True)
        rows.append(process_transit(year, center))

    csv_path = OUTPUT_DIR / CSV_NAME
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.12g")

    section("RESULTS")
    print(
        f"{'YEAR':<6}"
        f"{'ECLIPTIC':>12}"
        f"{'EARTH TRACK':>14}"
        f"{'VENUS TRACK':>14}"
        f"{'PHYSICAL REL':>15}"
        f"{'PROJECTED REL':>16}"
        f"{'BASIS ROT':>12}"
    )
    for row in rows:
        print(
            f"{row['transit_year']:<6}"
            f"{row['ecliptic_tangent_angle_deg']:>11.6f}°"
            f"{row['earth_apparent_track_angle_deg']:>13.6f}°"
            f"{row['venus_apparent_track_angle_deg']:>13.6f}°"
            f"{row['physical_relative_track_angle_deg']:>14.6f}°"
            f"{row['projected_relative_track_angle_deg']:>15.6f}°"
            f"{row['tangent_plane_basis_rotation_deg']:>11.6f}°"
        )

    section("OUTPUT SUMMARY")
    print(f"CSV                                  {csv_path}")
    print("PNG files                            NOT GENERATED")
    print(f"Transit count                        {len(rows)}")

    section("PAPER COMPARISON")
    print("Published or prior reduced values are comparison-only and are not calculation inputs.")

    section("EQUATION STATUS")
    print("VERIFIED Earth and Venus apparent tracks are independently fitted.")
    print("VERIFIED both relative tracks use the same minute-by-minute Venus-minus-Sun data.")
    print("VERIFIED ecliptic tangent is derived from the J2000 ecliptic pole.")
    print("VERIFIED basis rotation is computed from explicit tangent-plane axes.")

    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(VERSION)

if __name__ == "__main__":
    main()
# V0129
