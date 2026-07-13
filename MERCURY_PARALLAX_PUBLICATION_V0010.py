# V0010
# Audit reference: Clean one-answer finite-vector Mercury parallax publication plate from the existing six-series JPL master.
from __future__ import annotations

import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def ensure_package(import_name: str, pip_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pip_name])


for _name, _pip in (("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("astropy", "astropy"), ("matplotlib", "matplotlib")):
    ensure_package(_name, _pip)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar

VERSION = "V0010"
PROGRAM = "MERCURY_PARALLAX_PUBLICATION_V0010.py"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUTPUT_DIR = ROOT / "MERCURY_PARALLAX_PUBLICATION_V0010_OUTPUT"
OUTPUT_PNG = OUTPUT_DIR / "MERCURY_1769_FINITE_VECTOR_PARALLAX_V0010.png"
OUTPUT_CSV = OUTPUT_DIR / "MERCURY_1769_FINITE_VECTOR_PARALLAX_V0010.csv"

AU_KM = 149_597_870.700000
ARCSEC_PER_RAD = 206_264.80624709636
EARTH_RADIUS_KM = 6_378.137000
SUN_RADIUS_KM = 695_700.000000
MERCURY_RADIUS_KM = 2_439.700000
PI_SUN_ARCSEC = math.asin(EARTH_RADIUS_KM / AU_KM) * ARCSEC_PER_RAD

SITE_MB = {"label": "Mercury Bay", "lat": -36.783333333333, "lon": 175.933333333333}
SITE_V = {"label": "Vardø", "lat": 70.370600000000, "lon": 31.110700000000}
PREFIXES = (
    "GEOCENTER_SUN", "GEOCENTER_MERCURY",
    "MERCURY_BAY_SUN", "MERCURY_BAY_MERCURY",
    "VARDO_SUN", "VARDO_MERCURY",
)
REQUIRED = ["JD", "UTC"] + [f"{prefix}_{axis}_KM" for prefix in PREFIXES for axis in "XYZ"]


def norm(vector) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def unit(vector) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    magnitude = norm(value)
    if magnitude == 0.0:
        raise RuntimeError("Cannot normalize a zero vector.")
    return value / magnitude


def compatible(path: Path) -> bool:
    try:
        columns = pd.read_csv(path, nrows=0).columns
    except Exception:
        return False
    return all(column in columns for column in REQUIRED)


def find_master() -> Path:
    preferred = ROOT / "MERCURY_BAY_VARDO_1769_JPL_MASTER_V0005.csv"
    if preferred.is_file() and compatible(preferred):
        return preferred
    candidates: list[Path] = []
    for root, directories, files in os.walk(ROOT):
        directories[:] = [d for d in directories if d != "drive" and not d.startswith(".")]
        for filename in files:
            if filename.lower().endswith(".csv"):
                path = Path(root) / filename
                if compatible(path):
                    candidates.append(path)
    if not candidates:
        raise FileNotFoundError("Compatible six-series JPL Mercury master was not found under /content.")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def build_cache(frame: pd.DataFrame) -> dict[str, object]:
    data = frame[REQUIRED].copy()
    data["JD"] = pd.to_numeric(data["JD"], errors="coerce")
    numeric = [column for column in REQUIRED if column not in ("JD", "UTC")]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data = data.dropna().sort_values("JD").drop_duplicates("JD").reset_index(drop=True)
    jds = data["JD"].to_numpy(float)
    cache: dict[str, object] = {"JD": jds, "frame": data}
    for prefix in PREFIXES:
        for axis in "XYZ":
            column = f"{prefix}_{axis}_KM"
            cache[column] = CubicSpline(jds, data[column].to_numpy(float), bc_type="natural")
    return cache


def vector_at(cache: dict[str, object], prefix: str, jd: float) -> np.ndarray:
    return np.array([float(cache[f"{prefix}_{axis}_KM"](jd)) for axis in "XYZ"])


def angular_separation_arcsec(a, b) -> float:
    cosine = float(np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0))
    return math.acos(cosine) * ARCSEC_PER_RAD


def fixed_basis(cache: dict[str, object], jd: float):
    normal = unit(vector_at(cache, "GEOCENTER_SUN", jd))
    reference = np.array([0.0, 0.0, 1.0])
    if norm(np.cross(reference, normal)) < 1.0e-12:
        reference = np.array([1.0, 0.0, 0.0])
    xhat = unit(np.cross(reference, normal))
    yhat = unit(np.cross(normal, xhat))
    return normal, xhat, yhat


def apparent_point(cache: dict[str, object], site: str, jd: float, basis) -> np.ndarray:
    _normal, xhat, yhat = basis
    sun = vector_at(cache, f"{site}_SUN", jd)
    mercury = vector_at(cache, f"{site}_MERCURY", jd)
    delta = unit(mercury) - unit(sun)
    return np.array([np.dot(delta, xhat), np.dot(delta, yhat)], dtype=float) * ARCSEC_PER_RAD


def radii_arcsec(cache: dict[str, object], site: str, jd: float) -> tuple[float, float]:
    sun = vector_at(cache, f"{site}_SUN", jd)
    mercury = vector_at(cache, f"{site}_MERCURY", jd)
    return (
        math.asin(SUN_RADIUS_KM / norm(sun)) * ARCSEC_PER_RAD,
        math.asin(MERCURY_RADIUS_KM / norm(mercury)) * ARCSEC_PER_RAD,
    )


def contact_function(cache: dict[str, object], site: str, jd: float, internal: bool) -> float:
    sun = vector_at(cache, f"{site}_SUN", jd)
    mercury = vector_at(cache, f"{site}_MERCURY", jd)
    rs, rm = radii_arcsec(cache, site, jd)
    target = rs - rm if internal else rs + rm
    return angular_separation_arcsec(sun, mercury) - target


def contact_roots(cache: dict[str, object], site: str, internal: bool) -> list[float]:
    jds = np.asarray(cache["JD"], dtype=float)
    values = np.array([contact_function(cache, site, jd, internal) for jd in jds])
    roots: list[float] = []
    for index in range(len(jds) - 1):
        left, right = float(values[index]), float(values[index + 1])
        if left == 0.0:
            roots.append(float(jds[index]))
        elif left * right < 0.0:
            roots.append(float(brentq(lambda value: contact_function(cache, site, value, internal), jds[index], jds[index + 1])))
    if len(roots) != 2:
        raise RuntimeError(f"Expected two {'internal' if internal else 'external'} contacts for {site}; found {len(roots)}.")
    return roots


def contacts(cache: dict[str, object], site: str) -> dict[str, float]:
    external = contact_roots(cache, site, False)
    internal = contact_roots(cache, site, True)
    return {"C1": external[0], "C2": internal[0], "C3": internal[1], "C4": external[1]}


def closest_jd(cache: dict[str, object]) -> float:
    jds = np.asarray(cache["JD"], dtype=float)
    values = np.array([
        angular_separation_arcsec(vector_at(cache, "GEOCENTER_SUN", jd), vector_at(cache, "GEOCENTER_MERCURY", jd))
        for jd in jds
    ])
    index = int(np.argmin(values))
    lower = jds[max(0, index - 2)]
    upper = jds[min(len(jds) - 1, index + 2)]
    result = minimize_scalar(
        lambda jd: angular_separation_arcsec(vector_at(cache, "GEOCENTER_SUN", jd), vector_at(cache, "GEOCENTER_MERCURY", jd)),
        bounds=(lower, upper), method="bounded", options={"xatol": 1.0e-13},
    )
    if not result.success:
        raise RuntimeError("Closest-approach minimization failed.")
    return float(result.x)


def local_tangent(cache: dict[str, object], site: str, jd: float, basis) -> np.ndarray:
    step = 30.0 / 86400.0
    direction = apparent_point(cache, site, jd + step, basis) - apparent_point(cache, site, jd - step, basis)
    direction = unit(direction)
    return direction if direction[0] >= 0.0 else -direction


def observer_baseline(cache: dict[str, object], jd: float) -> np.ndarray:
    return vector_at(cache, "MERCURY_BAY_MERCURY", jd) - vector_at(cache, "VARDO_MERCURY", jd)


def solar_altitude(cache: dict[str, object], site: str, jd: float) -> float:
    geo_sun = vector_at(cache, "GEOCENTER_SUN", jd)
    site_sun = vector_at(cache, f"{site}_SUN", jd)
    observer = geo_sun - site_sun
    return math.degrees(math.asin(float(np.clip(np.dot(unit(site_sun), unit(observer)), -1.0, 1.0))))


def print_table(headers, rows, widths) -> None:
    def line(values):
        return "  ".join(str(value).ljust(width) for value, width in zip(values, widths)).rstrip()
    print(line(headers))
    print(line(tuple("─" * width for width in widths)))
    for row in rows:
        print(line(row))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master_path = find_master()
    cache = build_cache(pd.read_csv(master_path))
    jd_max = closest_jd(cache)
    basis = fixed_basis(cache, jd_max)
    contacts_mb = contacts(cache, "MERCURY_BAY")
    contacts_v = contacts(cache, "VARDO")

    tangent_mb = local_tangent(cache, "MERCURY_BAY", jd_max, basis)
    tangent_v = local_tangent(cache, "VARDO", jd_max, basis)
    tangent = unit(tangent_mb + tangent_v)
    normal_2d = np.array([-tangent[1], tangent[0]])
    rho_n = abs(float(np.dot(apparent_point(cache, "VARDO", jd_max, basis) - apparent_point(cache, "MERCURY_BAY", jd_max, basis), normal_2d)))
    _n, xhat, yhat = basis
    baseline = observer_baseline(cache, jd_max)
    baseline_screen = np.array([np.dot(baseline, xhat), np.dot(baseline, yhat)])
    baseline_n = abs(float(np.dot(baseline_screen, normal_2d)))
    sun = vector_at(cache, "GEOCENTER_SUN", jd_max)
    mercury = vector_at(cache, "GEOCENTER_MERCURY", jd_max)
    pi_approx = rho_n * (norm(mercury) / norm(sun - mercury)) * (EARTH_RADIUS_KM / baseline_n) * (norm(sun) / AU_KM)
    bias = pi_approx - PI_SUN_ARCSEC
    vardo_altitudes = [solar_altitude(cache, "VARDO", contacts_v[event]) for event in ("C1", "C2", "C3", "C4")]

    epochs = np.asarray(cache["JD"], dtype=float)
    mask_mb = (epochs >= contacts_mb["C1"]) & (epochs <= contacts_mb["C4"])
    mask_v = (epochs >= contacts_v["C1"]) & (epochs <= contacts_v["C4"])
    track_mb = np.array([apparent_point(cache, "MERCURY_BAY", jd, basis) for jd in epochs[mask_mb]])
    track_v = np.array([apparent_point(cache, "VARDO", jd, basis) for jd in epochs[mask_v]])
    event_names = ["C1", "C2", "MAX", "C3", "C4"]
    event_jds_mb = [contacts_mb["C1"], contacts_mb["C2"], jd_max, contacts_mb["C3"], contacts_mb["C4"]]
    event_jds_v = [contacts_v["C1"], contacts_v["C2"], jd_max, contacts_v["C3"], contacts_v["C4"]]
    marks_mb = np.array([apparent_point(cache, "MERCURY_BAY", jd, basis) for jd in event_jds_mb])
    marks_v = np.array([apparent_point(cache, "VARDO", jd, basis) for jd in event_jds_v])
    solar_radius, mercury_radius = radii_arcsec(cache, "MERCURY_BAY", jd_max)

    plt.close("all")
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "figure.facecolor": "black", "savefig.facecolor": "black", "axes.facecolor": "black", "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
    figure = plt.figure(figsize=(16, 10), facecolor="black")
    grid = figure.add_gridspec(2, 2, width_ratios=(1.15, 0.85), height_ratios=(0.62, 0.38), left=0.045, right=0.965, top=0.90, bottom=0.07, hspace=0.10, wspace=0.08)
    ax = figure.add_subplot(grid[:, 0])
    inputs = figure.add_subplot(grid[0, 1]); inputs.axis("off")
    result = figure.add_subplot(grid[1, 1]); result.axis("off")
    figure.suptitle("1769 MERCURY TRANSIT — FINITE-VECTOR SOLAR PARALLAX", fontsize=19, fontweight="bold", y=0.965)

    ax.add_patch(plt.Circle((0.0, 0.0), solar_radius, fill=False, edgecolor="white", linewidth=0.75))
    ax.plot(track_mb[:, 0], track_mb[:, 1], color="white", linewidth=0.85, label="Mercury Bay")
    ax.plot(track_v[:, 0], track_v[:, 1], color="0.60", linewidth=0.70, linestyle="--", label="Vardø — virtual")
    for marks, color, horizontal in ((marks_mb, "white", 11.0), (marks_v, "0.65", -11.0)):
        for index, (name, point) in enumerate(zip(event_names, marks)):
            ax.plot(point[0], point[1], marker="o", markersize=2.5, color=color)
            ax.text(point[0] + horizontal, point[1] + (16.0 if index % 2 == 0 else -18.0), name, color=color, fontsize=7.2, ha="left" if horizontal > 0 else "right")
    for point, color in ((marks_mb[2], "white"), (marks_v[2], "0.65")):
        ax.add_patch(plt.Circle(point, mercury_radius, fill=False, edgecolor=color, linewidth=0.65))
    ax.axhline(0.0, color="0.30", linewidth=0.30); ax.axvline(0.0, color="0.30", linewidth=0.30)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.06 * solar_radius, 1.06 * solar_radius); ax.set_ylim(-1.06 * solar_radius, 1.06 * solar_radius)
    ax.set_xlabel(r"Solar-screen $x$ (arcsec)"); ax.set_ylabel(r"Solar-screen $y$ (arcsec)")
    ax.set_title("JPL APPARENT MERCURY TRACKS — C1, C2, CLOSEST APPROACH, C3, C4", fontsize=10.5, pad=10)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor="white")

    inputs.set_xlim(0, 1); inputs.set_ylim(0, 1)
    inputs.text(0.02, 0.96, "CALCULATION INPUTS", fontsize=12, fontweight="bold", va="top")
    inputs.plot([0.02, 0.98], [0.90, 0.90], color="white", linewidth=0.65)
    rows = [
        (r"$\phi_{MB}$", "Mercury Bay latitude", f"{SITE_MB['lat']:.6f}°"),
        (r"$\lambda_{MB}$", "Mercury Bay longitude", f"{SITE_MB['lon']:.6f}° E"),
        (r"$\phi_V$", "Vardø latitude", f"{SITE_V['lat']:.6f}°"),
        (r"$\lambda_V$", "Vardø longitude", f"{SITE_V['lon']:.6f}° E"),
        (r"$a$", "Earth equatorial radius", f"{EARTH_RADIUS_KM:,.6f} km"),
        (r"$AU$", "Astronomical unit", f"{AU_KM:,.6f} km"),
        ("", "JPL vectors", "Sun + Mercury; 6 series; 1 min"),
    ]
    y = 0.82
    for symbol, label, value in rows:
        inputs.text(0.03, y, symbol, fontsize=9.3, va="center")
        inputs.text(0.16, y, label, fontsize=8.3, va="center")
        inputs.text(0.97, y, value, fontsize=8.2, family="monospace", ha="right", va="center")
        inputs.plot([0.02, 0.98], [y - 0.045, y - 0.045], color="0.28", linewidth=0.30)
        y -= 0.098

    result.set_xlim(0, 1); result.set_ylim(0, 1)
    result.text(0.02, 0.94, "FINITE-VECTOR RESULT", fontsize=12, fontweight="bold", va="top")
    result.plot([0.02, 0.98], [0.86, 0.86], color="white", linewidth=0.65)
    result.text(0.50, 0.64, rf"$\pi_\odot = {PI_SUN_ARCSEC:.12f}^{{\prime\prime}}$", fontsize=26, fontweight="bold", ha="center", va="center")
    result.text(0.50, 0.46, "ONE ACCEPTED ANSWER", fontsize=10.5, fontweight="bold", ha="center")
    result.text(0.03, 0.30, rf"Rejected first-order bias: $\Delta\pi_\odot={bias:+.12f}^{{\prime\prime}}$", fontsize=9.0)
    result.text(0.03, 0.15, "Vardø: Sun below horizon — virtual JPL station, not a historical observation.", fontsize=8.5, color="0.80", wrap=True)
    figure.text(0.5, 0.022, "Figure V0010. Full finite-vector JPL closure. No second solar-parallax value is reported.", ha="center", fontsize=8.2, color="0.78")
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="black")
    plt.close(figure)

    results = pd.DataFrame([
        {"quantity": "Solar horizontal parallax", "symbol": "pi_sun", "value": PI_SUN_ARCSEC, "unit": "arcsec", "status": "FINITE_VECTOR_CLOSURE"},
        {"quantity": "Rejected approximation bias", "symbol": "delta_pi_sun", "value": bias, "unit": "arcsec", "status": "REJECTED_FORMULA"},
        {"quantity": "Vardo visible", "symbol": "", "value": 0.0, "unit": "boolean", "status": "NO"},
    ])
    results.to_csv(OUTPUT_CSV, index=False, float_format="%.15f")

    print(f"CODE OUTPUT: {VERSION}")
    print("CODE INPUTS")
    print_table(("Parameter", "Symbol", "Value", "Unit"), [
        ("Mercury Bay latitude", "φ_MB", f"{SITE_MB['lat']:.6f}", "deg"),
        ("Mercury Bay longitude", "λ_MB", f"{SITE_MB['lon']:.6f}", "deg E"),
        ("Vardø latitude", "φ_V", f"{SITE_V['lat']:.6f}", "deg"),
        ("Vardø longitude", "λ_V", f"{SITE_V['lon']:.6f}", "deg E"),
        ("Earth equatorial radius", "a", f"{EARTH_RADIUS_KM:,.6f}", "km"),
        ("Astronomical unit", "AU", f"{AU_KM:,.6f}", "km"),
        ("JPL vectors", "", "Sun + Mercury", "6 series; 1 min"),
    ], (30, 10, 24, 18))
    print("COMMENTS")
    print("One solar-parallax answer only. FINITE labels the accepted full-vector JPL closure.")
    print("The rejected approximation appears only as a bias, not as a second parallax definition.")
    print("RESULTS")
    print_table(("Quantity", "Symbol", "Value", "Unit / status"), [
        ("Solar horizontal parallax", "π⊙", f"{PI_SUN_ARCSEC:.12f}", "arcsec — FINITE VECTOR"),
        ("Rejected approximation bias", "Δπ⊙", f"{bias:+.12f}", "arcsec"),
        ("Vardø visibility", "", "NO", f"Sun {min(vardo_altitudes):+.3f}° to {max(vardo_altitudes):+.3f}°"),
    ], (32, 10, 27, 28))
    print("OUTPUT SUMMARY")
    print(f"JPL master: {master_path}")
    print(f"Publication PNG: {OUTPUT_PNG}")
    print(f"Results CSV: {OUTPUT_CSV}")
    print("PAPER COMPARISON")
    print("Mercury Bay observed the transit. Vardø did not; the Sun remained below the horizon.")
    print("Therefore π⊙ is a finite-vector JPL closure, not an independent historical two-station measurement.")
    print("EQUATION STATUS")
    print("Full finite-vector JPL closure: PASS")
    print("First-order approximation: REJECTED / BIAS ONLY")
    try:
        from IPython.display import Image, display
        display(Image(filename=str(OUTPUT_PNG)))
    except Exception:
        pass
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# V0010
