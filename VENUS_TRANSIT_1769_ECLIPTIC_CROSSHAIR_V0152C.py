# V0152C
# Audit reference: standalone 1769 ecliptic-crosshair plot using V0102C geometric vectors, d-rho/dt-zero CA, and only the local plus/minus-30-minute curved transit segment.
from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def ensure(module: str, package: str) -> None:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)


for module_name, package_name in [
    ("astroquery", "astroquery"),
    ("astropy", "astropy"),
    ("scipy", "scipy"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("IPython", "ipython"),
]:
    ensure(module_name, package_name)

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from matplotlib.patches import Circle
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar

warnings.filterwarnings("ignore", message=".*dubious year.*")
warnings.filterwarnings("ignore", message=".*id_type.*deprecated.*")

VERSION = "V0152C"
LOCAL_TZ = ZoneInfo("America/Bogota")
YEAR = 1769
CENTER_UTC = "1769-06-03 22:00"
LOCATION = "500@399"
STEP = "1m"
REFPLANE = "ecliptic"
ABERRATIONS = "geometric"
AU_KM = 149_597_870.700
AS_PER_RAD = 206_264.80624709636
R_SUN_KM = 695_700.000
R_VENUS_KM = 6_051.800
SEARCH_HALF_HOURS = 3.0
LOCAL_HALF_MINUTES = 30.0
OUTPUT_DIR = Path("/content/VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152C_OUTPUT")
PNG = OUTPUT_DIR / "VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152C.png"


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 0.0:
        raise RuntimeError("REJECTED invalid vector")
    return np.asarray(v, dtype=float) / n


def query_vectors(body_id: str, start: str, stop: str) -> tuple[np.ndarray, np.ndarray]:
    table = Horizons(
        id=body_id,
        id_type="majorbody",
        location=LOCATION,
        epochs={"start": start, "stop": stop, "step": STEP},
    ).vectors(refplane=REFPLANE, aberrations=ABERRATIONS, cache=False)
    jd = np.asarray(table["datetime_jd"], dtype=float)
    xyz = np.column_stack([
        np.asarray(table["x"], dtype=float),
        np.asarray(table["y"], dtype=float),
        np.asarray(table["z"], dtype=float),
    ]) * AU_KM
    if len(jd) < 60 or not np.all(np.diff(jd) > 0.0):
        raise RuntimeError(f"REJECTED JPL vector grid for body {body_id}")
    return jd, xyz


def splines(jd: np.ndarray, xyz: np.ndarray) -> list[CubicSpline]:
    return [CubicSpline(jd, xyz[:, i], bc_type="natural") for i in range(3)]


def evaluate(curves: list[CubicSpline], jd: float) -> np.ndarray:
    return np.array([float(c(jd)) for c in curves], dtype=float)


def rho_rad(sun_curves: list[CubicSpline], venus_curves: list[CubicSpline], jd: float) -> float:
    su = unit(evaluate(sun_curves, jd))
    vu = unit(evaluate(venus_curves, jd))
    return math.atan2(float(np.linalg.norm(np.cross(su, vu))), float(np.dot(su, vu)))


def rho_dot_per_min(sun_curves: list[CubicSpline], venus_curves: list[CubicSpline], jd: float) -> float:
    h = 0.5 / 1440.0
    return rho_rad(sun_curves, venus_curves, jd + h) - rho_rad(sun_curves, venus_curves, jd - h)


def solve_ca(jd: np.ndarray, sun_curves: list[CubicSpline], venus_curves: list[CubicSpline]) -> float:
    sampled = np.array([rho_rad(sun_curves, venus_curves, float(t)) for t in jd], dtype=float)
    i = int(np.argmin(sampled))
    lo_seed = float(jd[max(0, i - 3)])
    hi_seed = float(jd[min(len(jd) - 1, i + 3)])
    seed = minimize_scalar(
        lambda t: rho_rad(sun_curves, venus_curves, float(t)),
        bounds=(lo_seed, hi_seed),
        method="bounded",
        options={"xatol": 1.0e-13, "maxiter": 600},
    )
    if not seed.success:
        raise RuntimeError("REJECTED rho minimum seed")
    lo = float(seed.x) - 5.0 / 1440.0
    hi = float(seed.x) + 5.0 / 1440.0
    f_lo = rho_dot_per_min(sun_curves, venus_curves, lo)
    f_hi = rho_dot_per_min(sun_curves, venus_curves, hi)
    if f_lo * f_hi > 0.0:
        raise RuntimeError("REJECTED d rho/dt zero not bracketed")
    return float(brentq(
        lambda t: rho_dot_per_min(sun_curves, venus_curves, float(t)),
        lo,
        hi,
        xtol=1.0e-14,
        rtol=1.0e-14,
        maxiter=200,
    ))


def projected_basis(sun_at_ca: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    los = unit(sun_at_ca)
    fixed_x = np.array([1.0, 0.0, 0.0])
    px = fixed_x - float(np.dot(fixed_x, los)) * los
    if np.linalg.norm(px) < 1.0e-12:
        fixed_y = np.array([0.0, 1.0, 0.0])
        px = fixed_y - float(np.dot(fixed_y, los)) * los
    px = unit(px)
    py = unit(np.cross(los, px))
    return px, py, los


def physical_basis(sun_at_ca: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    los = unit(sun_at_ca)
    north_ref = np.array([0.0, 0.0, 1.0])
    east = np.cross(north_ref, los)
    if np.linalg.norm(east) < 1.0e-12:
        east = np.cross(np.array([0.0, 1.0, 0.0]), los)
    east = unit(east)
    north = unit(np.cross(los, east))
    return east, north, los


def project(direction_vector: np.ndarray, basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[float, float]:
    x_axis, y_axis, los = basis
    d = unit(direction_vector)
    den = float(np.dot(d, los))
    if den <= 0.0:
        raise RuntimeError("REJECTED tangent-plane denominator")
    return (
        float(np.dot(d, x_axis) / den * AS_PER_RAD),
        float(np.dot(d, y_axis) / den * AS_PER_RAD),
    )


def fit_angle(x: np.ndarray, y: np.ndarray, minutes: np.ndarray) -> tuple[float, float, float, float]:
    cx = np.polyfit(minutes, x, 2)
    cy = np.polyfit(minutes, y, 2)
    mx = np.polyval(cx, minutes)
    my = np.polyval(cy, minutes)
    vx = float(cx[1])
    vy = float(cy[1])
    ax = float(2.0 * cx[0])
    ay = float(2.0 * cy[0])
    speed2 = vx * vx + vy * vy
    angle = ((math.degrees(math.atan2(vy, vx)) + 90.0) % 180.0) - 90.0
    slope = math.inf if abs(vx) < 1.0e-15 else vy / vx
    rms = float(np.sqrt(np.mean((x - mx) ** 2 + (y - my) ** 2)))
    curvature = abs(vx * ay - vy * ax) / (speed2 ** 1.5)
    return abs(angle), slope, rms, curvature


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    center = Time(CENTER_UTC, scale="utc")
    start = Time(center.jd - SEARCH_HALF_HOURS / 24.0, format="jd", scale="utc").strftime("%Y-%m-%d %H:%M")
    stop = Time(center.jd + SEARCH_HALF_HOURS / 24.0, format="jd", scale="utc").strftime("%Y-%m-%d %H:%M")

    print("CODE INPUTS")
    print(f"Version                              {VERSION}")
    print(f"Transit                              {YEAR}")
    print("JPL source                           Horizons geocentric geometric vectors")
    print(f"Observer                             {LOCATION}")
    print(f"Local curved segment                 ±{LOCAL_HALF_MINUTES:.0f} minutes")
    print("COMMENTS")
    print("Closest approach is the raw central-difference d rho/dt zero from the V0102C geometry.")
    print("Only the local ±30-minute projected Venus-Sun parabola is plotted; no rho circle is drawn.")

    sun_jd, sun_xyz = query_vectors("10", start, stop)
    venus_jd, venus_xyz = query_vectors("299", start, stop)
    if len(sun_jd) != len(venus_jd) or not np.allclose(sun_jd, venus_jd, atol=1.0e-11, rtol=0.0):
        raise RuntimeError("REJECTED mismatched Sun/Venus grids")

    sun_curves = splines(sun_jd, sun_xyz)
    venus_curves = splines(venus_jd, venus_xyz)
    jd_ca = solve_ca(sun_jd, sun_curves, venus_curves)
    sun_ca = evaluate(sun_curves, jd_ca)
    venus_ca = evaluate(venus_curves, jd_ca)
    proj_basis = projected_basis(sun_ca)
    phys_basis = physical_basis(sun_ca)

    minutes = np.linspace(-LOCAL_HALF_MINUTES, LOCAL_HALF_MINUTES, 121)
    jds = jd_ca + minutes / 1440.0
    rel_x = []
    rel_y = []
    earth_x = []
    earth_y = []
    physical_rel_x = []
    physical_rel_y = []
    for t in jds:
        s = evaluate(sun_curves, float(t))
        v = evaluate(venus_curves, float(t))
        sx, sy = project(s, proj_basis)
        vx, vy = project(v, proj_basis)
        rel_x.append(vx - sx)
        rel_y.append(vy - sy)
        ex, ey = project(s, phys_basis)
        pvx, pvy = project(v, phys_basis)
        earth_x.append(ex)
        earth_y.append(ey)
        physical_rel_x.append(pvx - ex)
        physical_rel_y.append(pvy - ey)

    rel_x = np.asarray(rel_x)
    rel_y = np.asarray(rel_y)
    earth_x = np.asarray(earth_x)
    earth_y = np.asarray(earth_y)
    physical_rel_x = np.asarray(physical_rel_x)
    physical_rel_y = np.asarray(physical_rel_y)

    ca_x, ca_y = project(venus_ca, proj_basis)
    sun_x_ca, sun_y_ca = project(sun_ca, proj_basis)
    ca_x -= sun_x_ca
    ca_y -= sun_y_ca
    rho_ca = float(math.hypot(ca_x, ca_y))
    sun_radius = float(math.asin(R_SUN_KM / np.linalg.norm(sun_ca)) * AS_PER_RAD)
    venus_radius = float(math.asin(R_VENUS_KM / np.linalg.norm(venus_ca)) * AS_PER_RAD)

    earth_angle, earth_slope, earth_rms, earth_curv = fit_angle(earth_x, earth_y, minutes)
    projected_angle, projected_slope, projected_rms, projected_curv = fit_angle(rel_x, rel_y, minutes)
    venus_angle, venus_slope, venus_rms, venus_curv = fit_angle(physical_rel_x, physical_rel_y, minutes)

    fit_x = np.polyfit(minutes, rel_x, 2)
    fit_y = np.polyfit(minutes, rel_y, 2)
    parabola_x = np.polyval(fit_x, minutes)
    parabola_y = np.polyval(fit_y, minutes)

    fig, ax = plt.subplots(figsize=(10.5, 10.5), dpi=120)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.add_patch(Circle((0.0, 0.0), sun_radius, facecolor="#C98A18", edgecolor="#E64A19", linewidth=1.15, alpha=0.92, zorder=1))
    extent_cross = 1.02 * sun_radius
    ax.plot([-extent_cross, extent_cross], [0.0, 0.0], color="#000000", linewidth=0.72, zorder=2)
    ax.plot([0.0, 0.0], [-extent_cross, extent_cross], color="#000000", linewidth=0.72, zorder=2)

    ax.plot(parabola_x, parabola_y, color="#FF2020", linewidth=4.8, alpha=0.18, solid_capstyle="round", zorder=6)
    ax.plot(parabola_x, parabola_y, color="#FF3030", linewidth=1.85, alpha=0.98, solid_capstyle="round", zorder=7, label="ρ parabola: ±30 min about CA")
    ax.scatter(parabola_x[::8], parabola_y[::8], s=7, color="#FF3030", zorder=8)
    ax.scatter([ca_x], [ca_y], s=78, facecolor="#FF3030", edgecolor="#FFF0F0", linewidth=0.85, zorder=10, label="dρ/dt = 0 closest-approach vertex")
    ax.add_patch(Circle((ca_x, ca_y), venus_radius, facecolor="none", edgecolor="#FFF0F0", linewidth=0.75, zorder=9))

    box_lines = [
        TextArea(f"Closest Approach (UTC): {Time(jd_ca, format='jd', scale='tdb').utc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}", textprops={"color":"#FF4A4A","fontsize":9.8,"fontweight":"bold"}),
        TextArea(f"ρ at CA: {rho_ca:.9f} arcsec", textprops={"color":"#FF4A4A","fontsize":9.6}),
        TextArea("Local parabola window: ±30.000 min", textprops={"color":"#FFB0B0","fontsize":9.6}),
        TextArea(f"Projected Track Angle: {projected_angle:.6f}°", textprops={"color":"#F5F5F5","fontsize":9.6}),
        TextArea(f"Projected Curvature: {projected_curv:.12e} arcsec⁻¹", textprops={"color":"#FF4A4A","fontsize":9.6}),
        TextArea("Ecliptic Reference: 0.000°", textprops={"color":"#000000","fontsize":9.6}),
    ]
    packed = VPacker(children=box_lines, align="left", pad=0.0, sep=2.0)
    angle_box = AnchoredOffsetbox(loc="upper right", child=packed, pad=0.45, frameon=True, borderpad=0.45)
    angle_box.patch.set_facecolor("#050505")
    angle_box.patch.set_edgecolor("#FF3030")
    angle_box.patch.set_linewidth(0.85)
    angle_box.patch.set_alpha(0.95)
    ax.add_artist(angle_box)

    extent = 1.10 * sun_radius
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("1769 Venus Transit — Ecliptic Crosshair and Local Closest-Approach Parabola", color="#F4F4F4", fontsize=14.5, weight="bold", pad=10)
    ax.set_xlabel("Registered tangent-plane X (arcsec)", color="#E4E4E4")
    ax.set_ylabel("Registered tangent-plane Y (arcsec)", color="#E4E4E4")
    ax.tick_params(colors="#D8D8D8", labelsize=9, width=0.5)
    ax.grid(True, color="#686868", alpha=0.25, linewidth=0.42)
    for spine in ax.spines.values():
        spine.set_color("#999999")
        spine.set_linewidth(0.55)
    legend = ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    for text in legend.get_texts():
        text.set_color("#E6E6E6")
    fig.tight_layout()
    fig.savefig(PNG, dpi=300, facecolor="black", bbox_inches="tight")
    display(Image(filename=str(PNG)))

    print("RESULTS")
    print(f"Closest approach UTC                 {Time(jd_ca, format='jd', scale='tdb').utc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    print(f"JD(TDB)                              {jd_ca:.12f}")
    print(f"rho at CA                            {rho_ca:.12f} arcsec")
    print(f"Raw d rho/dt at CA                   {rho_dot_per_min(sun_curves, venus_curves, jd_ca)*AS_PER_RAD:.15e} arcsec/min")
    print(f"Earth orbit angle                    {earth_angle:.6f} deg")
    print(f"Projected relative track angle       {projected_angle:.6f} deg")
    print(f"Venus Transit Track From Ecliptic    {venus_angle:.6f} deg")
    print(f"Earth orbit slope                    {earth_slope:.9f}")
    print(f"Projected relative slope             {projected_slope:.9f}")
    print(f"Venus Transit Ecliptic Slope         {venus_slope:.9f}")
    print(f"Projected RMS                        {projected_rms:.12f} arcsec")
    print(f"Projected curvature                  {projected_curv:.12e} arcsec^-1")
    print(f"Local plotted samples                {len(minutes)}")
    print("OUTPUT SUMMARY")
    print(f"PNG                                  {PNG}")
    print("PAPER COMPARISON")
    print("NOT USED: JPL-only internal geometry audit.")
    print("EQUATION STATUS")
    print("PASS: CA is the root of the same geometric d rho/dt used by V0102C; only the red quadratic ±30-minute local segment is plotted.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0152C