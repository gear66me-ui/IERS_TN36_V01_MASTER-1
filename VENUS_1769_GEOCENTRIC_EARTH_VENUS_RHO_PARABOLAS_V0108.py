# V0108
# Audit reference: clean geocentric Earth/Venus tangent-plane rho-minus-rhomin parabolas only; Python/Matplotlib/JPL only; no AI images.
from __future__ import annotations

import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0108"
LOCAL_TZ = ZoneInfo("America/Bogota")
OUT = Path("/content/VENUS_1769_GEOCENTRIC_EARTH_VENUS_RHO_PARABOLAS_V0108_OUTPUT")
PNG = OUT / "VENUS_1769_GEOCENTRIC_EARTH_VENUS_RHO_PARABOLAS_V0108.png"
CSV = OUT / "VENUS_1769_GEOCENTRIC_EARTH_VENUS_RHO_PARABOLAS_V0108.csv"

ARCSEC_PER_RAD = 206_264.80624709636
AU_KM = 149_597_870.700000
START = "1769-06-03 21:40"
STOP = "1769-06-03 23:00"
STEP = "1m"
WINDOW_MIN = 30.0
SUN_TARGET = "10"
VENUS_TARGET = "299"
EARTH_TARGET = "399"
CENTER_LOCATION = "500@0"
BG = "#000000"
FG = "#F8FAFC"
MUTED = "#B8CBD6"
GRID = "#263A4B"
CYAN = "#42D7C3"
GOLD = "#D89B18"
BLUE = "#23466F"
TEAL = "#164B55"
BROWN = "#563B0B"


def require(import_name: str, package_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", package_name])


for _import, _package in (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("astropy", "astropy"),
    ("astroquery", "astroquery"),
    ("matplotlib", "matplotlib"),
    ("IPython", "ipython"),
):
    require(_import, _package)

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore", message=".*id_type.*deprecated.*")
warnings.filterwarnings("ignore", message=".*dubious year.*")


def norm(v) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = norm(a)
    if n <= 0.0:
        raise RuntimeError("Zero vector cannot be normalized.")
    return a / n


def utc_from_jd(jd: float) -> str:
    return Time(float(jd), format="jd", scale="tdb").utc.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def download(prefix: str, target_id: str) -> pd.DataFrame:
    last = None
    for attempt in range(4):
        try:
            table = Horizons(
                id=target_id,
                location=CENTER_LOCATION,
                epochs={"start": START, "stop": STOP, "step": STEP},
                id_type=None,
            ).vectors(refplane="ecliptic", aberrations="geometric", cache=False)
            raw = table.to_pandas()
            df = pd.DataFrame({"JD_TDB": pd.to_numeric(raw["datetime_jd"], errors="coerce")})
            for ax in "xyz":
                df[f"{prefix}_{ax.upper()}_KM"] = pd.to_numeric(raw[ax], errors="coerce") * AU_KM
            return df.dropna().drop_duplicates("JD_TDB").sort_values("JD_TDB").reset_index(drop=True)
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"JPL Horizons download failed for {prefix}: {last}")


def build_master() -> pd.DataFrame:
    sun = download("SUN", SUN_TARGET)
    venus = download("VENUS", VENUS_TARGET)
    earth = download("EARTH", EARTH_TARGET)
    master = sun.merge(venus, on="JD_TDB", how="inner").merge(earth, on="JD_TDB", how="inner")
    if len(master) < 60:
        raise RuntimeError(f"Insufficient JPL samples: {len(master)}")
    return master


def splines(df: pd.DataFrame) -> dict[str, object]:
    jd = df["JD_TDB"].to_numpy(float)
    c: dict[str, object] = {"JD_TDB": jd}
    for col in df.columns:
        if col != "JD_TDB":
            c[col] = CubicSpline(jd, df[col].to_numpy(float), bc_type="natural")
    return c


def vec(c: dict[str, object], prefix: str, jd: float) -> np.ndarray:
    return np.array([
        float(c[f"{prefix}_X_KM"](jd)),
        float(c[f"{prefix}_Y_KM"](jd)),
        float(c[f"{prefix}_Z_KM"](jd)),
    ], dtype=float)


def basis_from_sun(c: dict[str, object], jd: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sun = vec(c, "SUN", jd)
    z = unit(sun)
    pole = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(z, pole))) > 0.95:
        pole = np.array([0.0, 1.0, 0.0], dtype=float)
    east = unit(np.cross(pole, z))
    north = unit(np.cross(z, east))
    return east, north, z


def relative_screen_xy_arcsec(body_vec: np.ndarray, sun_vec: np.ndarray, basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[float, float]:
    east, north, z = basis
    rel = body_vec - sun_vec
    denom = abs(float(np.dot(rel, z)))
    if denom <= 0.0:
        raise RuntimeError("Invalid Sun-screen denominator.")
    x = ARCSEC_PER_RAD * float(np.dot(rel, east)) / denom
    y = ARCSEC_PER_RAD * float(np.dot(rel, north)) / denom
    return x, y


def venus_rho_for_ca(c: dict[str, object], jd: float) -> float:
    basis = basis_from_sun(c, jd)
    sun = vec(c, "SUN", jd)
    venus = vec(c, "VENUS", jd)
    x, y = relative_screen_xy_arcsec(venus, sun, basis)
    return float(math.hypot(x, y))


def solve_venus_ca(c: dict[str, object]) -> float:
    jds = np.asarray(c["JD_TDB"], dtype=float)
    rhos = np.array([venus_rho_for_ca(c, float(jd)) for jd in jds], dtype=float)
    i = int(np.argmin(rhos))
    lo = float(jds[max(0, i - 2)])
    hi = float(jds[min(len(jds) - 1, i + 2)])
    res = minimize_scalar(lambda x: venus_rho_for_ca(c, float(x)), bounds=(lo, hi), method="bounded", options={"xatol": 1e-13, "maxiter": 600})
    if not res.success:
        raise RuntimeError("Venus closest-approach minimization failed.")
    return float(res.x)


def solve_local_minute_minimum(minutes: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    i = int(np.argmin(y))
    if i <= 0 or i >= len(y) - 1:
        return float(minutes[i]), float(y[i])
    x0, x1, x2 = float(minutes[i - 1]), float(minutes[i]), float(minutes[i + 1])
    y0, y1, y2 = float(y[i - 1]), float(y[i]), float(y[i + 1])
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(denom) < 1e-15:
        return x1, y1
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
    c0 = (x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1 + x0 * x1 * (x0 - x1) * y2) / denom
    if abs(a) < 1e-15:
        return x1, y1
    xv = -b / (2.0 * a)
    yv = a * xv * xv + b * xv + c0
    return float(xv), float(yv)


def analyze() -> tuple[pd.DataFrame, dict[str, float | str]]:
    master = build_master()
    c = splines(master)
    ca_jd = solve_venus_ca(c)
    basis_fixed = basis_from_sun(c, ca_jd)
    grid_minutes = np.linspace(-WINDOW_MIN, WINDOW_MIN, int(2 * WINDOW_MIN) + 1)
    rows = []
    for minute in grid_minutes:
        jd = ca_jd + float(minute) / 1440.0
        sun = vec(c, "SUN", jd)
        venus = vec(c, "VENUS", jd)
        earth = vec(c, "EARTH", jd)
        vx, vy = relative_screen_xy_arcsec(venus, sun, basis_fixed)
        ex, ey = relative_screen_xy_arcsec(earth, sun, basis_fixed)
        venus_rho = float(math.hypot(vx, vy))
        earth_rho = float(math.hypot(ex, ey))
        rows.append({
            "minute_from_venus_ca": float(minute),
            "jd_tdb": float(jd),
            "utc": utc_from_jd(jd),
            "venus_x_arcsec": vx,
            "venus_y_arcsec": vy,
            "earth_x_arcsec": ex,
            "earth_y_arcsec": ey,
            "venus_rho_arcsec": venus_rho,
            "earth_rho_arcsec": earth_rho,
        })
    df = pd.DataFrame(rows)
    v_min = float(df["venus_rho_arcsec"].min())
    e_min = float(df["earth_rho_arcsec"].min())
    df["venus_rho_minus_rhomin_arcsec"] = df["venus_rho_arcsec"] - v_min
    df["earth_rho_minus_rhomin_arcsec"] = df["earth_rho_arcsec"] - e_min
    v_span = float(df["venus_rho_minus_rhomin_arcsec"].max())
    e_span = float(df["earth_rho_minus_rhomin_arcsec"].max())
    df["venus_normalized_rho_minus_rhomin"] = df["venus_rho_minus_rhomin_arcsec"] / v_span if v_span != 0.0 else 0.0
    df["earth_normalized_rho_minus_rhomin"] = df["earth_rho_minus_rhomin_arcsec"] / e_span if e_span != 0.0 else 0.0
    v_min_minute, _ = solve_local_minute_minimum(df["minute_from_venus_ca"].to_numpy(float), df["venus_rho_arcsec"].to_numpy(float))
    e_min_minute, _ = solve_local_minute_minimum(df["minute_from_venus_ca"].to_numpy(float), df["earth_rho_arcsec"].to_numpy(float))
    stats: dict[str, float | str] = {
        "ca_utc": utc_from_jd(ca_jd),
        "ca_jd_tdb": ca_jd,
        "venus_rho_min_arcsec": v_min,
        "earth_rho_min_arcsec": e_min,
        "venus_minute_of_min": v_min_minute,
        "earth_minute_of_min": e_min_minute,
        "venus_excess_plus30": float(df.loc[df["minute_from_venus_ca"] == 30.0, "venus_rho_minus_rhomin_arcsec"].iloc[0]),
        "earth_excess_plus30": float(df.loc[df["minute_from_venus_ca"] == 30.0, "earth_rho_minus_rhomin_arcsec"].iloc[0]),
        "samples": len(df),
    }
    return df, stats


def style_axis(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(MUTED)
        s.set_linewidth(0.7)
    ax.tick_params(colors=FG, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.45, alpha=0.45)
    ax.axvline(0.0, color=FG, linestyle="--", linewidth=0.8, alpha=0.65)
    ax.axhline(0.0, color=MUTED, linestyle=":", linewidth=0.65, alpha=0.70)


def make_plot(df: pd.DataFrame, stats: dict[str, float | str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False, float_format="%.15f")
    fig = plt.figure(figsize=(14.5, 9.0), facecolor=BG)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.46], left=0.07, right=0.985, top=0.88, bottom=0.09, hspace=0.26)
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_norm = fig.add_subplot(gs[1, 0])
    ax_tbl = fig.add_subplot(gs[2, 0])
    ax_tbl.axis("off")
    x = df["minute_from_venus_ca"].to_numpy(float)
    style_axis(ax_raw)
    style_axis(ax_norm)
    ax_raw.plot(x, df["venus_rho_minus_rhomin_arcsec"], color=CYAN, linewidth=1.25, marker="o", markersize=2.4, label="Venus ρ − ρmin")
    ax_raw.plot(x, df["earth_rho_minus_rhomin_arcsec"], color=GOLD, linewidth=1.25, linestyle="--", marker="x", markersize=3.0, label="Earth ρ − ρmin")
    ax_raw.set_title("Raw tangent-plane angular distance excess: ρ − ρmin", color=FG, fontsize=15, fontweight="bold")
    ax_raw.set_ylabel("arcsec", color=FG, fontsize=10)
    leg = ax_raw.legend(loc="upper center", ncol=2, fontsize=9, frameon=True)
    leg.get_frame().set_facecolor("#101A2E")
    leg.get_frame().set_edgecolor("#35516E")
    for txt in leg.get_texts():
        txt.set_color(FG)
    ax_norm.plot(x, df["venus_normalized_rho_minus_rhomin"], color=CYAN, linewidth=1.25, marker="o", markersize=2.4, label="Venus normalized ρ − ρmin")
    ax_norm.plot(x, df["earth_normalized_rho_minus_rhomin"], color=GOLD, linewidth=1.25, linestyle="--", marker="x", markersize=3.0, label="Earth normalized ρ − ρmin")
    ax_norm.set_title("Normalized parabolas: each curve divided by its own ±30 min maximum excess", color=FG, fontsize=15, fontweight="bold")
    ax_norm.set_ylabel("normalized", color=FG, fontsize=10)
    ax_norm.set_xlabel("minutes from geocentric Venus closest approach", color=FG, fontsize=10)
    leg2 = ax_norm.legend(loc="upper center", ncol=2, fontsize=9, frameon=True)
    leg2.get_frame().set_facecolor("#101A2E")
    leg2.get_frame().set_edgecolor("#35516E")
    for txt in leg2.get_texts():
        txt.set_color(FG)
    rows = [
        ["Quantity", "Value", "Unit / trace"],
        ["Geocentric Venus CA UTC", str(stats["ca_utc"]), "JPL solve; min Venus ρ"],
        ["Venus ρmin", f"{float(stats['venus_rho_min_arcsec']):.12f}", "arcsec"],
        ["Earth ρmin", f"{float(stats['earth_rho_min_arcsec']):.12f}", "arcsec"],
        ["Venus minimum minute", f"{float(stats['venus_minute_of_min']):+.6f}", "minutes from Venus CA"],
        ["Earth minimum minute", f"{float(stats['earth_minute_of_min']):+.6f}", "minutes from Venus CA"],
        ["Venus excess at +30 min", f"{float(stats['venus_excess_plus30']):.12f}", "arcsec"],
        ["Earth excess at +30 min", f"{float(stats['earth_excess_plus30']):.12f}", "arcsec"],
        ["Samples", f"{int(stats['samples'])}", "one-minute plotted window"],
    ]
    table = ax_tbl.table(cellText=rows[1:], colLabels=rows[0], cellLoc="left", colLoc="left", loc="center", colWidths=[0.36, 0.32, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1.0, 1.18)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#47627D")
        cell.set_linewidth(0.45)
        if r == 0:
            cell.set_facecolor(BLUE)
            cell.get_text().set_color(FG)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(TEAL if r % 2 == 1 else BROWN)
            cell.get_text().set_color(FG)
    fig.suptitle("1769 Venus Transit — Clean Geocentric Earth/Venus ρ − ρmin Parabolas V0108", color=FG, fontsize=20, fontweight="bold", y=0.965)
    fig.text(0.5, 0.927, "Fixed Sun-screen tangent plane at geocentric Venus closest approach; no limb, no Point Venus, no ρ².", ha="center", color=MUTED, fontsize=10)
    fig.text(0.5, 0.032, f"File: VENUS_1769_GEOCENTRIC_EARTH_VENUS_RHO_PARABOLAS_V0108.py | Output: {PNG.name} | CSV: {CSV.name}", ha="center", color=MUTED, fontsize=8)
    fig.savefig(PNG, dpi=500, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    display(Image(filename=str(PNG)))


def main() -> None:
    print("CODE INPUTS")
    print(f"Version : {VERSION}")
    print(f"Window  : {START} to {STOP}, step {STEP}")
    print(f"Center  : {CENTER_LOCATION} heliocentric vectors; geocentric Sun-screen basis from Earth-Sun direction")
    print("COMMENTS")
    print("Plots only rho minus rho_min. No rho squared. Two panels only: raw arcsec excess and independent normalized excess.")
    print("RESULTS")
    df, stats = analyze()
    print(f"Geocentric Venus CA UTC : {stats['ca_utc']}")
    print(f"Venus rho_min           : {float(stats['venus_rho_min_arcsec']):.12f} arcsec")
    print(f"Earth rho_min           : {float(stats['earth_rho_min_arcsec']):.12f} arcsec")
    print(f"Venus min minute        : {float(stats['venus_minute_of_min']):+.6f} min")
    print(f"Earth min minute        : {float(stats['earth_minute_of_min']):+.6f} min")
    make_plot(df, stats)
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    print("PAPER COMPARISON")
    print("NOT USED: visualization-only geocentric tangent-plane rho audit.")
    print("EQUATION STATUS")
    print("PASS: plotted quantities are rho-rho_min only; no rho^2 terms are used.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0108
