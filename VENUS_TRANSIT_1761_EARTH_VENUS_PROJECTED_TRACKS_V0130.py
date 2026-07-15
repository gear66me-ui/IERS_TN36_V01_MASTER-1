# V0130
# Audit reference: 1761-only JPL geocentric plot of Earth apparent, Venus apparent, and projected Venus-minus-Sun track directions.

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

def ensure(module: str, package: str) -> None:
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
    ensure(module_name, package_name)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

VERSION = "V0130"
YEAR = 1761
CENTER_UTC = "1761-06-06 06:00"
LOCATION = "@399"
STEP = "1m"
REFPLANE = "earth"
ABERRATIONS = "apparent"
AU_KM = 149597870.700
AS_PER_RAD = 206264.80624709636
R_SUN_KM = 695700.000
R_VENUS_KM = 6051.800
SEARCH_HALF_HOURS = 18.0

OUTPUT_DIR = Path("/content/VENUS_TRANSIT_1761_EARTH_VENUS_PROJECTED_TRACKS_V0130_OUTPUT")
PNG_NAME = "VENUS_TRANSIT_1761_EARTH_VENUS_PROJECTED_TRACKS_V0130.png"
CSV_NAME = "VENUS_TRANSIT_1761_EARTH_VENUS_PROJECTED_TRACKS_V0130.csv"

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

def line_segment(
    center_x: float,
    center_y: float,
    signed_angle_deg: float,
    half_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(signed_angle_deg)
    dx = half_length * math.cos(angle)
    dy = half_length * math.sin(angle)
    return (
        np.array([center_x - dx, center_x + dx]),
        np.array([center_y - dy, center_y + dy]),
    )

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    center = Time(CENTER_UTC, scale="utc")
    half_window_days = SEARCH_HALF_HOURS / 24.0
    start = Time(center.jd - half_window_days, format="jd", scale="utc").strftime("%Y-%m-%d %H:%M")
    stop = Time(center.jd + half_window_days, format="jd", scale="utc").strftime("%Y-%m-%d %H:%M")

    section("CODE INPUTS")
    print(f"Version                              {VERSION}")
    print("Transit                              1761")
    print("JPL source                           Horizons geocentric vectors only")
    print("Observer                             @399")
    print("Bodies                               Sun=10; Venus=299")
    print("Cadence                              1 minute")
    print("Earth and Venus tracks               physical east-north tangent plane")
    print("Projected relative track             projected fixed JPL/ICRF +X tangent plane")
    print(f"Output                               {OUTPUT_DIR}")

    sun = query_vectors("10", start, stop)
    venus = query_vectors("299", start, stop)
    jd_ca, minimum_separation_rad, sun_at_ca = closest_approach(sun, venus)

    physical_basis = physical_east_north_basis(sun_at_ca)
    projected_basis = projected_jpl_x_basis(sun_at_ca)

    sun_x_physical, sun_y_physical = project_to_tangent_plane(sun.xyz_km, physical_basis)
    venus_x_physical, venus_y_physical = project_to_tangent_plane(venus.xyz_km, physical_basis)
    sun_x_projected, sun_y_projected = project_to_tangent_plane(sun.xyz_km, projected_basis)
    venus_x_projected, venus_y_projected = project_to_tangent_plane(venus.xyz_km, projected_basis)

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
        raise RuntimeError("REJECTED insufficient in-transit samples")

    hours = (sun.jd[mask] - jd_ca) * 24.0

    earth_track = fit_track(hours, sun_x_physical[mask], sun_y_physical[mask])
    venus_track = fit_track(hours, venus_x_physical[mask], venus_y_physical[mask])
    projected_relative_track = fit_track(
        hours,
        relative_x_projected[mask],
        relative_y_projected[mask],
    )

    index_ca = int(np.argmin(np.abs(sun.jd - jd_ca)))
    ca_x = float(relative_x_projected[index_ca])
    ca_y = float(relative_y_projected[index_ca])
    solar_radius_arcsec = float(math.asin(R_SUN_KM / sun_distance_km[index_ca]) * AS_PER_RAD)
    venus_radius_arcsec = float(math.asin(R_VENUS_KM / venus_distance_km[index_ca]) * AS_PER_RAD)

    fig, ax = plt.subplots(figsize=(10.5, 10.5), dpi=120)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.add_patch(Circle(
        (0.0, 0.0),
        solar_radius_arcsec,
        facecolor="#C98A18",
        edgecolor="#E64A19",
        linewidth=1.15,
        alpha=0.92,
        zorder=1,
    ))

    half_length = 0.88 * solar_radius_arcsec
    line_specs = [
        ("Earth apparent track", earth_track.signed_angle_deg, "#3EA6FF"),
        ("Projected Venus−Sun relative track", projected_relative_track.signed_angle_deg, "#F5F5F5"),
        ("Venus apparent track", venus_track.signed_angle_deg, "#EF5350"),
    ]

    for label, angle, color in line_specs:
        x_line, y_line = line_segment(ca_x, ca_y, angle, half_length)
        ax.plot(
            x_line,
            y_line,
            color=color,
            linewidth=0.95,
            label=label,
            zorder=5,
        )

    ax.scatter(
        [ca_x],
        [ca_y],
        s=24,
        facecolor="white",
        edgecolor="#DADADA",
        linewidth=0.55,
        zorder=7,
        label="Closest approach",
    )
    ax.add_patch(Circle(
        (ca_x, ca_y),
        venus_radius_arcsec,
        facecolor="none",
        edgecolor="white",
        linewidth=0.65,
        zorder=6,
    ))

    annotation = "\n".join([
        f"Earth apparent track: {earth_track.positive_angle_deg:.6f}°",
        f"Projected relative track: {projected_relative_track.positive_angle_deg:.6f}°",
        f"Venus apparent track: {venus_track.positive_angle_deg:.6f}°",
    ])

    offset_x = 0.18 * solar_radius_arcsec if ca_x <= 0.0 else -0.18 * solar_radius_arcsec
    offset_y = 0.16 * solar_radius_arcsec if ca_y <= 0.0 else -0.16 * solar_radius_arcsec

    ax.annotate(
        annotation,
        xy=(ca_x, ca_y),
        xytext=(ca_x + offset_x, ca_y + offset_y),
        color="#F0F0F0",
        fontsize=9.6,
        ha="left" if offset_x > 0.0 else "right",
        va="bottom" if offset_y > 0.0 else "top",
        arrowprops={
            "arrowstyle": "-",
            "color": "#B0B0B0",
            "linewidth": 0.65,
        },
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "#050505",
            "edgecolor": "#858585",
            "alpha": 0.94,
        },
        zorder=8,
    )

    extent = 1.10 * solar_radius_arcsec
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "1761 Venus Transit — Earth, Venus, and Projected Relative Tracks",
        color="#F4F4F4",
        fontsize=14.5,
        weight="bold",
        pad=10,
    )
    ax.set_xlabel("Registered tangent-plane X (arcsec)", color="#E4E4E4")
    ax.set_ylabel("Registered tangent-plane Y (arcsec)", color="#E4E4E4")
    ax.tick_params(colors="#D8D8D8", labelsize=9, width=0.5)
    ax.grid(True, color="#686868", alpha=0.25, linewidth=0.42)

    for spine in ax.spines.values():
        spine.set_color("#999999")
        spine.set_linewidth(0.55)

    legend = ax.legend(loc="upper right", frameon=False, fontsize=9.0)
    for label in legend.get_texts():
        label.set_color("#E6E6E6")

    png_path = OUTPUT_DIR / PNG_NAME
    fig.tight_layout()
    fig.savefig(png_path, dpi=600, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    display(Image(filename=str(png_path)))

    ca_time = Time(jd_ca, format="jd", scale="tdb")
    result = {
        "transit_year": YEAR,
        "closest_approach_utc": ca_time.utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "jd_tdb": jd_ca,
        "earth_apparent_track_angle_deg": earth_track.positive_angle_deg,
        "venus_apparent_track_angle_deg": venus_track.positive_angle_deg,
        "projected_relative_track_angle_deg": projected_relative_track.positive_angle_deg,
        "earth_apparent_slope": earth_track.slope,
        "venus_apparent_slope": venus_track.slope,
        "projected_relative_slope": projected_relative_track.slope,
        "earth_apparent_rms_arcsec": earth_track.rms_arcsec,
        "venus_apparent_rms_arcsec": venus_track.rms_arcsec,
        "projected_relative_rms_arcsec": projected_relative_track.rms_arcsec,
        "earth_apparent_curvature_per_arcsec": earth_track.curvature_per_arcsec,
        "venus_apparent_curvature_per_arcsec": venus_track.curvature_per_arcsec,
        "projected_relative_curvature_per_arcsec": projected_relative_track.curvature_per_arcsec,
        "sample_count": projected_relative_track.sample_count,
        "minimum_separation_arcsec": minimum_separation_rad * AS_PER_RAD,
    }

    csv_path = OUTPUT_DIR / CSV_NAME
    pd.DataFrame([result]).to_csv(csv_path, index=False, float_format="%.12g")

    section("COMMENTS")
    print("Only the requested 1761 test figure is generated.")
    print("Earth and Venus apparent tracks use the physical east-north tangent plane.")
    print("Projected relative track uses the projected JPL/ICRF +X tangent plane.")
    print("No manual angles are inserted.")

    section("RESULTS")
    print(f"Closest approach UTC                 {result['closest_approach_utc']}")
    print(f"JD(TDB)                              {result['jd_tdb']:.9f}")
    print(f"Earth apparent track angle           {result['earth_apparent_track_angle_deg']:.6f} deg")
    print(f"Projected relative track angle       {result['projected_relative_track_angle_deg']:.6f} deg")
    print(f"Venus apparent track angle           {result['venus_apparent_track_angle_deg']:.6f} deg")
    print(f"Earth apparent slope                 {result['earth_apparent_slope']:.9f}")
    print(f"Projected relative slope             {result['projected_relative_slope']:.9f}")
    print(f"Venus apparent slope                 {result['venus_apparent_slope']:.9f}")
    print(f"Sample count                         {result['sample_count']}")

    section("OUTPUT SUMMARY")
    print(f"PNG                                  {png_path}")
    print(f"CSV                                  {csv_path}")

    section("PAPER COMPARISON")
    print("Published or previously reduced values are comparison-only.")

    section("EQUATION STATUS")
    print("VERIFIED JPL geocentric vectors only")
    print("VERIFIED one-minute in-transit fitting")
    print("VERIFIED equal-aspect solar tangent-plane plot")
    print(f"PNG exists                           {png_path.is_file() and png_path.stat().st_size > 0}")

    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(VERSION)

if __name__ == "__main__":
    main()
# V0130
