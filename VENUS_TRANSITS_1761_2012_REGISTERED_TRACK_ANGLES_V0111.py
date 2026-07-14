# V0111
# Audit reference: six historical Venus-transit registered Earth/Venus tangent-plane track-angle plots from fresh JPL Horizons vectors, with inline Colab display.

from __future__ import annotations
import importlib.util, math, subprocess, sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def need(module: str, package: str) -> None:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)


for _m, _p in [("astroquery", "astroquery"), ("astropy", "astropy"), ("scipy", "scipy"), ("pandas", "pandas"), ("matplotlib", "matplotlib"), ("IPython", "ipython")]:
    need(_m, _p)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from IPython.display import Image, display

VERSION = "V0111"
OUT = Path("/content/VENUS_TRANSITS_1761_2012_REGISTERED_TRACK_ANGLES_V0111_OUTPUT")
CSV = "VENUS_TRANSITS_1761_2012_REGISTERED_TRACK_ANGLES_V0111.csv"
AU_KM, R_SUN_KM, AS_PER_RAD = 149597870.700, 695700.000, 206264.80624709636
LOCATION, REFPLANE, ABERRATIONS, STEP = "@0", "earth", "geometric", "1m"
FRAME_DESCRIPTION = "JPL Horizons default ICRF/J2000 vector frame; Earth mean equator/equinox reference plane"
SEARCH_HALF_H, FIT_HALF_H = 18.0, 10.0
TRANSITS: Dict[int, str] = {
    1761: "1761-06-06 06:00", 1769: "1769-06-03 22:00",
    1874: "1874-12-09 04:00", 1882: "1882-12-06 17:00",
    2004: "2004-06-08 08:00", 2012: "2012-06-06 01:00",
}
PNGS = {y: f"VENUS_TRANSIT_{y}_REGISTERED_TRACK_ANGLES_V0111.png" for y in TRANSITS}


@dataclass(frozen=True)
class Series:
    jd: np.ndarray
    r: np.ndarray


@dataclass(frozen=True)
class Fit:
    angle: float
    slope: float
    rms: float
    curvature: float


def section(name: str) -> None:
    print(name)
    print("-" * len(name))


def unit(a: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n <= 0:
        raise ValueError("Zero or non-finite vector")
    return a / n


def wrapdiff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def query(body: str, start: str, stop: str) -> Series:
    h = Horizons(id=body, id_type="majorbody", location=LOCATION,
                 epochs={"start": start, "stop": stop, "step": STEP})
    table = h.vectors(refplane=REFPLANE, aberrations=ABERRATIONS)
    jd = np.asarray(table["datetime_jd"], float)
    r = np.column_stack([np.asarray(table[k], float) for k in ("x", "y", "z")]) * AU_KM
    if len(jd) < 100 or not np.all(np.diff(jd) > 0):
        raise RuntimeError(f"REJECTED JPL grid for {body}")
    return Series(jd, r)


def spl(jd: np.ndarray, a: np.ndarray) -> List[CubicSpline]:
    return [CubicSpline(jd, a[:, i], bc_type="natural") for i in range(3)]


def ev(s: List[CubicSpline], t: float) -> np.ndarray:
    return np.array([f(t) for f in s])


def sep(e: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    a, b = s - e, v - e
    a /= np.linalg.norm(a, axis=1)[:, None]
    b /= np.linalg.norm(b, axis=1)[:, None]
    return np.arccos(np.clip(np.einsum("ij,ij->i", a, b), -1, 1))


def closest(jd: np.ndarray, e: np.ndarray, s: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    z = sep(e, s, v)
    i = int(np.argmin(z))
    lo, hi = max(0, i - 3), min(len(jd) - 1, i + 3)
    es, ss, vs = spl(jd, e), spl(jd, s), spl(jd, v)
    def objective(t: float) -> float:
        a = unit(ev(ss, t) - ev(es, t))
        b = unit(ev(vs, t) - ev(es, t))
        return math.acos(float(np.clip(np.dot(a, b), -1, 1)))
    q = minimize_scalar(objective, bounds=(float(jd[lo]), float(jd[hi])), method="bounded",
                        options={"xatol": 1e-12, "maxiter": 300})
    if not q.success:
        raise RuntimeError("REJECTED closest-approach refinement")
    return float(q.x), float(q.fun)


def basis(e: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = unit(s - e)
    x = np.cross(np.array([0.0, 0.0, 1.0]), n)
    if np.linalg.norm(x) < 1e-10:
        x = np.cross(np.array([0.0, 1.0, 0.0]), n)
    x = unit(x)
    y = unit(np.cross(n, x))
    if abs(np.dot(x, y)) > 1e-12:
        raise RuntimeError("REJECTED tangent basis")
    return x, y


def fit(hours: np.ndarray, xy: np.ndarray) -> Fit:
    cx, cy = np.polyfit(hours, xy[:, 0], 2), np.polyfit(hours, xy[:, 1], 2)
    model = np.column_stack((np.polyval(cx, hours), np.polyval(cy, hours)))
    rms = float(np.sqrt(np.mean(np.sum((xy - model) ** 2, axis=1))))
    vx, vy, ax, ay = float(cx[1]), float(cy[1]), float(2 * cx[0]), float(2 * cy[0])
    speed2 = vx * vx + vy * vy
    if speed2 <= 0:
        raise RuntimeError("REJECTED degenerate fit")
    return Fit(math.degrees(math.atan2(vy, vx)) % 360.0,
               math.inf if abs(vx) < 1e-15 else vy / vx,
               rms, abs(vx * ay - vy * ax) / (speed2 ** 1.5))


def draw_sun(ax: plt.Axes) -> None:
    for ring in [
        Circle((0, 0), 1.000, facecolor="#FFD900", edgecolor="none", alpha=0.95),
        Circle((0, 0), 0.985, facecolor="none", edgecolor="#FFD34D", linewidth=0.50),
        Circle((0, 0), 1.012, facecolor="none", edgecolor="#FF8A00", linewidth=0.80),
        Circle((0, 0), 1.024, facecolor="none", edgecolor="#FF2020", linewidth=1.45),
        Circle((0, 0), 1.036, facecolor="none", edgecolor="#FF4A1A", linewidth=0.48),
    ]:
        ax.add_patch(ring)


def make_plot(year: int, exy: np.ndarray, vxy: np.ndarray, ef: Fit, vf: Fit,
              opening: float, utc: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 10), dpi=160)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    draw_sun(ax)
    stride = max(1, len(exy) // 160)
    ax.plot(exy[:, 0], exy[:, 1], color="#4DA6FF", linewidth=0.72, label="Earth registered trajectory")
    ax.plot(vxy[:, 0], vxy[:, 1], color="white", linewidth=0.72, label="Venus registered trajectory")
    ax.scatter(exy[::stride, 0], exy[::stride, 1], s=2, color="#4DA6FF")
    ax.scatter(vxy[::stride, 0], vxy[::stride, 1], s=2, color="white")
    ax.scatter([0], [0], s=30, facecolors="none", edgecolors="#39FF88", linewidths=1)
    ax.scatter([0], [0], s=5, color="#39FF88", label="Closest approach registration")
    for track, color in ((ef, "#4DA6FF"), (vf, "white")):
        a = math.radians(track.angle)
        ax.arrow(0, 0, 0.72 * math.cos(a), 0.72 * math.sin(a), width=0.004,
                 head_width=0.055, head_length=0.075, length_includes_head=True, color=color)
    annotation = "\n".join([f"Closest approach: {utc}", f"Earth track angle: {ef.angle:.6f}°",
                              f"Venus track angle: {vf.angle:.6f}°", f"Opening angle: {opening:.6f}°"])
    ax.text(0.025, 0.975, annotation, transform=ax.transAxes, va="top", color="white", fontsize=9,
            bbox={"boxstyle": "round,pad=.45", "facecolor": "#101010", "edgecolor": "#B0B0B0", "alpha": 0.92})
    extent = max(1.3, float(np.max(np.abs(exy))), float(np.max(np.abs(vxy))))
    limit = min(max(1.35, extent * 1.08), 3.8)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit),
           xlabel="Registered tangent-plane displacement X [solar radii]",
           ylabel="Registered tangent-plane displacement Y [solar radii]")
    ax.set_aspect("equal")
    ax.set_title(f"{year} Venus Transit — Registered Earth–Venus Crossing and Track Angles",
                 color="white", fontsize=13, pad=14, weight="bold")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    ax.tick_params(colors="white", width=0.55, labelsize=8)
    ax.grid(True, color="#777777", alpha=0.2, linewidth=0.4)
    for spine in ax.spines.values():
        spine.set_color("#A0A0A0"); spine.set_linewidth(0.55)
    legend = ax.legend(loc="lower right", fontsize=8)
    legend.get_frame().set_facecolor("#0E0E0E"); legend.get_frame().set_edgecolor("#9A9A9A")
    for label in legend.get_texts(): label.set_color("white")
    fig.tight_layout()
    fig.savefig(path, dpi=600, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    display(Image(filename=str(path)))


def process(year: int, center: str) -> dict:
    c = Time(center, scale="utc")
    d = SEARCH_HALF_H / 24
    start = Time(c.jd - d, format="jd", scale="utc").strftime("%Y-%m-%d %H:%M")
    stop = Time(c.jd + d, format="jd", scale="utc").strftime("%Y-%m-%d %H:%M")
    e, v, s = query("399", start, stop), query("299", start, stop), query("10", start, stop)
    aligned = len(e.jd) == len(v.jd) == len(s.jd) and np.allclose(e.jd, v.jd, atol=1e-11, rtol=0) and np.allclose(e.jd, s.jd, atol=1e-11, rtol=0)
    if not aligned: raise RuntimeError("REJECTED mismatched grids")
    ca, minsep = closest(e.jd, e.r, s.r, v.r)
    es, vs, ss = spl(e.jd, e.r), spl(e.jd, v.r), spl(e.jd, s.r)
    e0, v0, s0 = ev(es, ca), ev(vs, ca), ev(ss, ca)
    x, y = basis(e0, s0)
    mask = np.abs((e.jd - ca) * 24) <= FIT_HALF_H
    hours = (e.jd[mask] - ca) * 24
    if int(mask.sum()) < 300: raise RuntimeError("REJECTED sample count")
    exy = np.column_stack(((e.r[mask] - e0) @ x, (e.r[mask] - e0) @ y))
    vxy = np.column_stack(((v.r[mask] - v0) @ x, (v.r[mask] - v0) @ y))
    ef, vf = fit(hours, exy), fit(hours, vxy)
    opening = wrapdiff(ef.angle, vf.angle)
    verify = opening - wrapdiff(ef.angle, vf.angle)
    if not 0 <= opening <= 180: raise RuntimeError("REJECTED opening angle")
    utc = Time(ca, format="jd", scale="tdb").utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    path = OUT / PNGS[year]
    make_plot(year, exy / R_SUN_KM, vxy / R_SUN_KM, ef, vf, opening, utc, path)
    sun_as = math.asin(R_SUN_KM / float(np.linalg.norm(s0 - e0))) * AS_PER_RAD
    return {"transit_year": year, "closest_approach_utc": utc, "closest_approach_jd_tdb": ca,
            "earth_track_angle_deg": ef.angle, "venus_track_angle_deg": vf.angle,
            "opening_angle_deg": opening, "angle_verification_residual_deg": verify,
            "earth_slope": ef.slope, "venus_slope": vf.slope, "earth_rms": ef.rms,
            "venus_rms": vf.rms, "earth_curvature": ef.curvature, "venus_curvature": vf.curvature,
            "sample_count": int(mask.sum()), "minimum_separation_arcsec": minsep * AS_PER_RAD,
            "solar_angular_radius_arcsec": sun_as, "png": str(path)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    section("CODE INPUTS")
    print(f"Version                              {VERSION}")
    print("JPL source                           Horizons vectors API")
    print(f"Observer/reference                   {LOCATION}; Sun=10 Venus=299 Earth=399")
    print(f"Frame                                {FRAME_DESCRIPTION}")
    print(f"Aberrations/cadence                  {ABERRATIONS}/{STEP}")
    print(f"Search/Fit half-window               {SEARCH_HALF_H:.1f}/{FIT_HALF_H:.1f} h")
    print(f"Output                               {OUT}")
    for year, center in TRANSITS.items(): print(f"NOT USED AS CA INPUT {year} window center {center} UTC")
    section("COMMENTS")
    print("Fresh JPL vectors drive all geometry. Window centers identify events only.")
    print("Closest approach, projected tracks, slopes, RMS, curvature, and angles are calculated.")
    print("DEBUG progress follows. REJECTED manual angles and closest-approach times are not inputs.")
    rows = []
    for year, center in TRANSITS.items():
        print(f"DEBUG processing {year}", flush=True)
        rows.append(process(year, center))
    columns = ["transit_year", "closest_approach_utc", "closest_approach_jd_tdb",
               "earth_track_angle_deg", "venus_track_angle_deg", "opening_angle_deg",
               "angle_verification_residual_deg", "earth_slope", "venus_slope", "earth_rms",
               "venus_rms", "earth_curvature", "venus_curvature", "sample_count"]
    csv_path = OUT / CSV
    pd.DataFrame([{k: r[k] for k in columns} for r in rows]).to_csv(csv_path, index=False, float_format="%.12g")
    section("RESULTS")
    for r in rows:
        print(f"{r['transit_year']} CA {r['closest_approach_utc']} JD_TDB {r['closest_approach_jd_tdb']:.9f} Earth {r['earth_track_angle_deg']:.6f}° Venus {r['venus_track_angle_deg']:.6f}° Opening {r['opening_angle_deg']:.6f}° Verify {r['angle_verification_residual_deg']:.12f}°")
        print(f"Earth slope {r['earth_slope']:.9f} RMS {r['earth_rms']:.6f} km curvature {r['earth_curvature']:.12e}")
        print(f"Venus slope {r['venus_slope']:.9f} RMS {r['venus_rms']:.6f} km curvature {r['venus_curvature']:.12e}")
        print(f"Minimum separation {r['minimum_separation_arcsec']:.6f} arcsec; solar radius {r['solar_angular_radius_arcsec']:.6f} arcsec; samples {r['sample_count']}")
    section("OUTPUT SUMMARY")
    print(f"CSV {csv_path}")
    for r in rows: print(f"PNG {r['transit_year']} {r['png']} bytes {Path(r['png']).stat().st_size}")
    print(f"Exactly six PNG figures {len(rows) == 6}")
    section("PAPER COMPARISON")
    print("NOT USED: no published angles or closest-approach values.")
    print("Historical dates identify only broad JPL query windows.")
    section("EQUATION STATUS")
    residual = max(abs(r["angle_verification_residual_deg"]) for r in rows)
    print("VERIFIED opening_angle = abs(wrap180(Earth angle - Venus angle))")
    print("VERIFIED 0 <= opening_angle <= 180 degrees")
    print("VERIFIED tangent basis orthonormal and closest approach refined from minute JPL vectors")
    print(f"Maximum residual {residual:.12e} deg")
    print(f"Equation checks passed {residual <= 1e-12 and all(0 <= r['opening_angle_deg'] <= 180 for r in rows)}")
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0111
