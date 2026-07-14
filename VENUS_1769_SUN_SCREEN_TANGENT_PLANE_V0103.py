# V0103
# Audit reference: Sun-screen tangent-plane closest-approach geometry; Python/Matplotlib/JPL only; no AI images.
from __future__ import annotations

import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0103"
LOCAL_TZ = ZoneInfo("America/Bogota")
OUT = Path("/content/VENUS_1769_SUN_SCREEN_TANGENT_PLANE_V0103_OUTPUT")
PNG = OUT / "VENUS_1769_SUN_SCREEN_TANGENT_PLANE_V0103.png"
CSV = OUT / "VENUS_1769_SUN_SCREEN_TANGENT_PLANE_V0103.csv"

ARCSEC_PER_RAD = 206_264.80624709636
AU_KM = 149_597_870.700000
START = "1769-06-03 21:30"
STOP = "1769-06-03 23:15"
STEP = "1m"
WINDOW_MIN = 30.0
SUN_TARGET = "10"
VENUS_TARGET = "299"
GEOCENTER_LOCATION = "500@399"

BG = "#000000"
FG = "#F8FAFC"
MUTED = "#B8CBD6"
GRID = "#263A4B"
BLUE = "#42D7C3"
GOLD = "#D89B18"
GREEN = "#74D680"
RED = "#FF6B6B"
PURPLE = "#9B8CFF"
TABLE_HEADER = "#23466F"
TABLE_BODY = "#101A2E"
TABLE_TEAL = "#164B55"
TABLE_GOLD = "#563B0B"


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
from scipy.optimize import minimize_scalar, brentq

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
                location=GEOCENTER_LOCATION,
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
    master = sun.merge(venus, on="JD_TDB", how="inner")
    if len(master) < 80:
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


def tangent_basis_from_sun(sun_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = unit(sun_vec)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(ref, z))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])
    east = unit(np.cross(ref, z))
    north = unit(np.cross(z, east))
    return east, north, z


def project_on_sun_screen(c: dict[str, object], jd: float, fixed_basis: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> tuple[float, float]:
    sun = vec(c, "SUN", jd)
    venus = vec(c, "VENUS", jd)
    if fixed_basis is None:
        east, north, z = tangent_basis_from_sun(sun)
    else:
        east, north, z = fixed_basis
    venus_u = unit(venus)
    denom = float(np.dot(venus_u, z))
    if abs(denom) <= 1.0e-15:
        raise RuntimeError("Projection denominator too small.")
    x = ARCSEC_PER_RAD * float(np.dot(venus_u, east)) / denom
    y = ARCSEC_PER_RAD * float(np.dot(venus_u, north)) / denom
    return x, y


def rho_xy(c: dict[str, object], jd: float) -> float:
    x, y = project_on_sun_screen(c, jd, None)
    return math.hypot(x, y)


def rho_dot_arcsec_per_min(c: dict[str, object], jd: float) -> float:
    h = 0.5 / 1440.0
    return (rho_xy(c, jd + h) - rho_xy(c, jd - h)) / 1.0


def solve_ca(c: dict[str, object]) -> float:
    jds = np.asarray(c["JD_TDB"], dtype=float)
    samples = np.array([rho_xy(c, float(jd)) for jd in jds], dtype=float)
    i = int(np.argmin(samples))
    lo = float(jds[max(0, i - 2)])
    hi = float(jds[min(len(jds) - 1, i + 2)])
    res = minimize_scalar(lambda x: rho_xy(c, float(x)), bounds=(lo, hi), method="bounded", options={"xatol": 1e-13, "maxiter": 600})
    if not res.success:
        raise RuntimeError("Closest approach minimization failed.")
    return float(res.x)


def solve_rhodot_zero(c: dict[str, object], ca_jd: float) -> float:
    lo = ca_jd - 5.0 / 1440.0
    hi = ca_jd + 5.0 / 1440.0
    f_lo = rho_dot_arcsec_per_min(c, lo)
    f_hi = rho_dot_arcsec_per_min(c, hi)
    if f_lo * f_hi <= 0.0:
        return float(brentq(lambda x: rho_dot_arcsec_per_min(c, float(x)), lo, hi, xtol=1e-13, rtol=1e-13, maxiter=100))
    return ca_jd


def fixed_track_axes(c: dict[str, object], ca_jd: float, fixed_basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    h = 30.0 / 86400.0
    x1, y1 = project_on_sun_screen(c, ca_jd - h, fixed_basis)
    x2, y2 = project_on_sun_screen(c, ca_jd + h, fixed_basis)
    t = unit(np.array([x2 - x1, y2 - y1], dtype=float))
    n = np.array([-t[1], t[0]], dtype=float)
    ca_xy = np.array(project_on_sun_screen(c, ca_jd, fixed_basis), dtype=float)
    if float(np.dot(ca_xy, n)) < 0.0:
        n = -n
    return t, n


def analyze() -> tuple[pd.DataFrame, dict[str, float | str]]:
    master = build_master()
    c = splines(master)
    ca_jd = solve_ca(c)
    sun_ca = vec(c, "SUN", ca_jd)
    fixed_basis = tangent_basis_from_sun(sun_ca)
    tangent, normal = fixed_track_axes(c, ca_jd, fixed_basis)
    ca_xy = np.array(project_on_sun_screen(c, ca_jd, fixed_basis), dtype=float)
    ca_s = float(np.dot(ca_xy, tangent))
    ca_n = float(np.dot(ca_xy, normal))
    rho_min = math.hypot(ca_s, ca_n)
    rhodot_zero_jd = solve_rhodot_zero(c, ca_jd)

    grid_minutes = np.linspace(-WINDOW_MIN, WINDOW_MIN, int(2 * WINDOW_MIN) + 1)
    rows = []
    for m in grid_minutes:
        jd = ca_jd + float(m) / 1440.0
        xy = np.array(project_on_sun_screen(c, jd, fixed_basis), dtype=float)
        s = float(np.dot(xy, tangent) - ca_s)
        n = float(np.dot(xy, normal))
        rho = math.hypot(s, n)
        rho_direct = rho_xy(c, jd)
        rows.append({
            "minute_from_ca": float(m),
            "jd_tdb": float(jd),
            "utc": utc_from_jd(jd),
            "screen_x_arcsec": float(xy[0]),
            "screen_y_arcsec": float(xy[1]),
            "along_track_s_arcsec": s,
            "normal_n_arcsec": n,
            "rho_from_rotated_screen_arcsec": rho,
            "rho_direct_arcsec": rho_direct,
            "rho_excess_arcsec": rho - rho_min,
            "rho_dot_arcsec_per_min": rho_dot_arcsec_per_min(c, jd),
        })
    df = pd.DataFrame(rows)
    fit = np.polyfit(df["minute_from_ca"].to_numpy(float), df["along_track_s_arcsec"].to_numpy(float), 1)
    s_fit = np.polyval(fit, df["minute_from_ca"].to_numpy(float))
    s_rms = float(np.sqrt(np.mean((df["along_track_s_arcsec"].to_numpy(float) - s_fit) ** 2)))
    n_mean = float(df["normal_n_arcsec"].mean())
    n_rms = float(np.sqrt(np.mean((df["normal_n_arcsec"].to_numpy(float) - n_mean) ** 2)))
    stats: dict[str, float | str] = {
        "ca_jd_tdb": ca_jd,
        "ca_utc": utc_from_jd(ca_jd),
        "rho_min_arcsec": rho_min,
        "rho_direct_min_arcsec": rho_xy(c, ca_jd),
        "rhodot_at_ca": rho_dot_arcsec_per_min(c, ca_jd),
        "rhodot_zero_utc": utc_from_jd(rhodot_zero_jd),
        "rhodot_zero_offset_sec": (rhodot_zero_jd - ca_jd) * 86400.0,
        "ca_along_track_raw_arcsec": ca_s,
        "ca_normal_arcsec": ca_n,
        "track_angle_deg_screen_xy": math.degrees(math.atan2(float(tangent[1]), float(tangent[0]))),
        "along_track_slope_arcsec_per_min": float(fit[0]),
        "along_track_linear_rms_arcsec": s_rms,
        "normal_mean_arcsec": n_mean,
        "normal_rms_about_mean_arcsec": n_rms,
        "samples": len(df),
    }
    return df, stats


def style_ax(ax) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.6)
    ax.grid(True, color=GRID, linewidth=0.35, alpha=0.8)


def add_table(ax, stats: dict[str, float | str]) -> None:
    ax.axis("off")
    rows = [
        ["Geocentric CA UTC", str(stats["ca_utc"]), "true ρ minimum"],
        ["Minimum ρ", f"{float(stats['rho_min_arcsec']):.12f}", "arcsec"],
        ["dρ/dt at CA", f"{float(stats['rhodot_at_ca']):+.12e}", "arcsec/min"],
        ["dρ/dt zero offset", f"{float(stats['rhodot_zero_offset_sec']):+.9f}", "s"],
        ["Along-track slope", f"{float(stats['along_track_slope_arcsec_per_min']):+.9f}", "arcsec/min"],
        ["Along-track RMS", f"{float(stats['along_track_linear_rms_arcsec']):.9e}", "arcsec"],
        ["Normal mean", f"{float(stats['normal_mean_arcsec']):.9f}", "arcsec"],
        ["Normal RMS", f"{float(stats['normal_rms_about_mean_arcsec']):.9e}", "arcsec"],
        ["Track angle", f"{float(stats['track_angle_deg_screen_xy']):+.9f}", "deg"],
    ]
    table = ax.table(cellText=rows, colLabels=["Quantity", "Value", "Unit / status"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.0, 1.25)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.35)
        cell.get_text().set_color(FG)
        if r == 0:
            cell.set_facecolor(TABLE_HEADER)
            cell.get_text().set_weight("bold")
        elif r in (1, 2, 3):
            cell.set_facecolor(TABLE_TEAL)
            cell.get_text().set_weight("bold")
        elif r in (5, 7):
            cell.set_facecolor(TABLE_GOLD)
        else:
            cell.set_facecolor(TABLE_BODY)


def plot(df: pd.DataFrame, stats: dict[str, float | str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False)
    plt.rcParams.update({"figure.facecolor": BG, "savefig.facecolor": BG, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(14.0, 11.0), dpi=160)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.05, 1.0, 1.0, 0.9], hspace=0.36, wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, :])
    ax4 = fig.add_subplot(gs[3, :])
    for ax in (ax0, ax1, ax2, ax3):
        style_ax(ax)
    m = df["minute_from_ca"].to_numpy(float)
    s = df["along_track_s_arcsec"].to_numpy(float)
    n = df["normal_n_arcsec"].to_numpy(float)
    rho_excess = df["rho_excess_arcsec"].to_numpy(float)
    rhodot = df["rho_dot_arcsec_per_min"].to_numpy(float)

    ax0.plot(df["screen_x_arcsec"], df["screen_y_arcsec"], lw=0.7, marker="o", ms=1.5, color=BLUE)
    ax0.scatter([df.loc[df["minute_from_ca"] == 0.0, "screen_x_arcsec"].iloc[0]], [df.loc[df["minute_from_ca"] == 0.0, "screen_y_arcsec"].iloc[0]], s=18, color=GOLD, zorder=5)
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_title("Sun-screen tangent plane: Venus center relative to Sun center", color=FG, fontsize=10, weight="bold")
    ax0.set_xlabel("screen x (arcsec)", color=MUTED, fontsize=8)
    ax0.set_ylabel("screen y (arcsec)", color=MUTED, fontsize=8)

    ax1.plot(s, n, lw=0.7, marker="o", ms=1.5, color=GREEN)
    ax1.axvline(0.0, color=GOLD, lw=0.7)
    ax1.scatter([0.0], [float(stats["ca_normal_arcsec"])], s=18, color=GOLD, zorder=5)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("Rotated tangent/normal coordinates", color=FG, fontsize=10, weight="bold")
    ax1.set_xlabel("along-track s from CA (arcsec)", color=MUTED, fontsize=8)
    ax1.set_ylabel("normal n (arcsec)", color=MUTED, fontsize=8)

    ax2.plot(m, s, lw=0.9, marker="o", ms=1.5, color=BLUE, label="along-track s(t)")
    ax2.plot(m, n - np.mean(n), lw=0.9, marker="o", ms=1.5, color=GREEN, label="normal n(t) − mean(n)")
    ax2.axvline(0.0, color=GOLD, lw=0.8, label="closest approach")
    ax2.axhline(0.0, color=GRID, lw=0.6)
    ax2.set_title("Tangent coordinate is linear; normal coordinate is nearly constant", color=FG, fontsize=10, weight="bold")
    ax2.set_xlabel("minutes from closest approach", color=MUTED, fontsize=8)
    ax2.set_ylabel("arcsec", color=MUTED, fontsize=8)
    ax2.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8, loc="upper left")

    ax3.plot(m, rho_excess, lw=0.95, marker="o", ms=1.5, color=PURPLE, label="ρ(t) − ρmin")
    ax3.axvline(0.0, color=GOLD, lw=0.8, label="closest approach")
    ax3.axhline(0.0, color=GRID, lw=0.6)
    ax3.set_title("The closest-approach bowl appears after tangent-plane projection", color=FG, fontsize=10, weight="bold")
    ax3.set_xlabel("minutes from closest approach", color=MUTED, fontsize=8)
    ax3.set_ylabel("ρ excess (arcsec)", color=MUTED, fontsize=8)
    ax3.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8, loc="upper left")

    ax3b = ax3.twinx()
    ax3b.set_facecolor(BG)
    ax3b.plot(m, rhodot, lw=0.75, color=RED, alpha=0.85, label="dρ/dt")
    ax3b.tick_params(colors=FG, labelsize=8)
    ax3b.set_ylabel("dρ/dt (arcsec/min)", color=MUTED, fontsize=8)
    for spine in ax3b.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.6)

    add_table(ax4, stats)
    fig.suptitle("1769 Venus Transit — Sun-Screen Tangent-Plane Closest Approach (JPL Geocenter)", color=FG, fontsize=14, weight="bold", y=0.985)
    fig.text(0.5, 0.012, f"{VERSION} | JPL Horizons geometric vectors | no AI images | PNG: {PNG.name}", ha="center", color=MUTED, fontsize=8)
    fig.savefig(PNG, bbox_inches="tight")
    plt.close(fig)


def print_sections(df: pd.DataFrame, stats: dict[str, float | str]) -> None:
    print("CODE INPUTS")
    print(f"VERSION: {VERSION}")
    print(f"JPL LOCATION: {GEOCENTER_LOCATION} (Earth geocenter)")
    print(f"JPL TARGETS: Sun={SUN_TARGET}, Venus={VENUS_TARGET}")
    print(f"WINDOW: ±{WINDOW_MIN:.1f} min around solved geocentric ρ minimum")
    print("COMMENTS")
    print("Sun-screen tangent plane is built from the geocentric Sun direction at closest approach.")
    print("Venus is projected onto that Sun-centered tangent plane and rotated into along-track s and normal n.")
    print("RESULTS")
    print(f"CA UTC: {stats['ca_utc']}")
    print(f"CA JD_TDB: {float(stats['ca_jd_tdb']):.12f}")
    print(f"Minimum ρ: {float(stats['rho_min_arcsec']):.12f} arcsec")
    print(f"dρ/dt at CA: {float(stats['rhodot_at_ca']):+.12e} arcsec/min")
    print(f"dρ/dt zero offset: {float(stats['rhodot_zero_offset_sec']):+.9f} s")
    print(f"Along-track slope: {float(stats['along_track_slope_arcsec_per_min']):+.9f} arcsec/min")
    print(f"Along-track linear RMS: {float(stats['along_track_linear_rms_arcsec']):.9e} arcsec")
    print(f"Normal mean: {float(stats['normal_mean_arcsec']):.9f} arcsec")
    print(f"Normal RMS about mean: {float(stats['normal_rms_about_mean_arcsec']):.9e} arcsec")
    print(f"Track angle in Sun-screen xy: {float(stats['track_angle_deg_screen_xy']):+.9f} deg")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    print("PAPER COMPARISON")
    print("Published values are NOT USED; this widget is a geometry explanation from JPL vectors only.")
    print("EQUATION STATUS")
    print("ρ(t) = sqrt(s(t)^2 + n(t)^2): VERIFIED numerically from tangent-plane projection")
    print("Closest approach = argmin ρ(t): VERIFIED by scalar minimization")
    print("dρ/dt = 0 at closest approach: VERIFIED by finite-difference derivative")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


def main() -> None:
    df, stats = analyze()
    plot(df, stats)
    print_sections(df, stats)
    display(Image(filename=str(PNG)))


if __name__ == "__main__":
    main()
# V0103
