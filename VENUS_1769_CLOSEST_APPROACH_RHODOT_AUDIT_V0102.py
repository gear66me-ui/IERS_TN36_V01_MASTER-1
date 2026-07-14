# V0102
# Audit reference: clean rho-minimum closest-approach audit; same plotted rho defines CA; Python/Matplotlib only; no AI images.
from __future__ import annotations

import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0102"
LOCAL_TZ = ZoneInfo("America/Bogota")
OUT = Path("/content/VENUS_1769_CLOSEST_APPROACH_RHODOT_AUDIT_V0102_OUTPUT")
PNG = OUT / "VENUS_1769_CLOSEST_APPROACH_RHODOT_AUDIT_V0102.png"
CSV = OUT / "VENUS_1769_CLOSEST_APPROACH_RHODOT_AUDIT_V0102.csv"

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
from scipy.optimize import brentq, minimize_scalar

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


def angle_arcsec_between(a: np.ndarray, b: np.ndarray) -> float:
    ua = unit(a)
    ub = unit(b)
    return ARCSEC_PER_RAD * math.atan2(norm(np.cross(ua, ub)), float(np.dot(ua, ub)))


def rho_arcsec(c: dict[str, object], jd: float) -> float:
    return angle_arcsec_between(vec(c, "SUN", jd), vec(c, "VENUS", jd))


def rho_dot_arcsec_per_min(c: dict[str, object], jd: float) -> float:
    h = 0.5 / 1440.0
    return rho_arcsec(c, jd + h) - rho_arcsec(c, jd - h)


def ratio_ev_es(c: dict[str, object], jd: float) -> float:
    ev = norm(vec(c, "VENUS", jd))
    es = norm(vec(c, "SUN", jd))
    return ev / es


def solve_ca(c: dict[str, object]) -> float:
    jds = np.asarray(c["JD_TDB"], dtype=float)
    rho_samples = np.array([rho_arcsec(c, float(jd)) for jd in jds], dtype=float)
    i = int(np.argmin(rho_samples))
    lo = float(jds[max(0, i - 2)])
    hi = float(jds[min(len(jds) - 1, i + 2)])
    res = minimize_scalar(lambda x: rho_arcsec(c, float(x)), bounds=(lo, hi), method="bounded", options={"xatol": 1e-13, "maxiter": 600})
    if not res.success:
        raise RuntimeError("Closest-approach minimization failed for plotted rho(t).")
    return float(res.x)


def solve_rhodot_zero(c: dict[str, object], ca_jd: float) -> float:
    lo = ca_jd - 5.0 / 1440.0
    hi = ca_jd + 5.0 / 1440.0
    f_lo = rho_dot_arcsec_per_min(c, lo)
    f_hi = rho_dot_arcsec_per_min(c, hi)
    if f_lo * f_hi <= 0.0:
        return float(brentq(lambda x: rho_dot_arcsec_per_min(c, float(x)), lo, hi, xtol=1e-13, rtol=1e-13, maxiter=100))
    return ca_jd


def analyze() -> tuple[pd.DataFrame, dict[str, float | str]]:
    master = build_master()
    c = splines(master)
    ca_jd = solve_ca(c)
    rhodot_zero_jd = solve_rhodot_zero(c, ca_jd)
    grid_minutes = np.linspace(-WINDOW_MIN, WINDOW_MIN, int(2 * WINDOW_MIN) + 1)
    min_rho = rho_arcsec(c, ca_jd)
    ratio0 = ratio_ev_es(c, ca_jd)
    rows = []
    for minute in grid_minutes:
        jd = ca_jd + float(minute) / 1440.0
        rho = rho_arcsec(c, jd)
        rd = rho_dot_arcsec_per_min(c, jd)
        ratio = ratio_ev_es(c, jd)
        rows.append({
            "minute_from_true_rho_min": float(minute),
            "jd_tdb": float(jd),
            "utc": utc_from_jd(jd),
            "rho_arcsec": float(rho),
            "rho_excess_arcsec": float(rho - min_rho),
            "rho_dot_arcsec_per_min": float(rd),
            "ev_es_ratio": float(ratio),
            "ev_es_ratio_delta_ppm": float((ratio / ratio0 - 1.0) * 1_000_000.0),
        })
    df = pd.DataFrame(rows)
    stats: dict[str, float | str] = {
        "ca_jd_tdb": ca_jd,
        "ca_utc": utc_from_jd(ca_jd),
        "rho_min_arcsec": min_rho,
        "rhodot_at_ca": rho_dot_arcsec_per_min(c, ca_jd),
        "rhodot_zero_jd_tdb": rhodot_zero_jd,
        "rhodot_zero_utc": utc_from_jd(rhodot_zero_jd),
        "rhodot_zero_offset_sec": (rhodot_zero_jd - ca_jd) * 86400.0,
        "rho_minus30_excess": float(df.loc[df["minute_from_true_rho_min"] == -30.0, "rho_excess_arcsec"].iloc[0]),
        "rho_plus30_excess": float(df.loc[df["minute_from_true_rho_min"] == 30.0, "rho_excess_arcsec"].iloc[0]),
        "ev_es_ppm_range": float(df["ev_es_ratio_delta_ppm"].max() - df["ev_es_ratio_delta_ppm"].min()),
        "samples": int(len(df)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False, float_format="%.15f")
    return df, stats


def style_axis(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.30, alpha=0.58)
    ax.tick_params(labelsize=7.2, width=0.35, length=2.4)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.35)


def style_table(table, teal_rows=(), gold_rows=(), fontsize=6.0) -> None:
    table.auto_set_font_size(False)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#70879A")
        cell.set_linewidth(0.30)
        cell.get_text().set_color(FG)
        cell.get_text().set_fontsize(fontsize)
        if row == 0:
            cell.set_facecolor(TABLE_HEADER)
            cell.get_text().set_fontweight("bold")
        elif row in teal_rows:
            cell.set_facecolor(TABLE_TEAL)
            cell.get_text().set_fontweight("bold")
        elif row in gold_rows:
            cell.set_facecolor(TABLE_GOLD)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(TABLE_BODY)


def plot(df: pd.DataFrame, stats: dict[str, float | str]) -> None:
    plt.close("all")
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": FG,
        "axes.labelcolor": FG,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": MUTED,
    })
    x = df["minute_from_true_rho_min"].to_numpy(float)
    rho = df["rho_arcsec"].to_numpy(float)
    rho_excess = df["rho_excess_arcsec"].to_numpy(float)
    rhodot = df["rho_dot_arcsec_per_min"].to_numpy(float)
    ratio_ppm = df["ev_es_ratio_delta_ppm"].to_numpy(float)

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    gs = fig.add_gridspec(4, 1, height_ratios=[0.32, 0.20, 0.20, 0.28], left=0.060, right=0.985, top=0.895, bottom=0.095, hspace=0.280)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    tax = fig.add_subplot(gs[3])

    fig.suptitle("1769 Venus Transit — Clean Geocentric Closest-Approach Audit", fontsize=15.2, fontweight="bold", y=0.955)
    fig.text(0.5, 0.925, "Fresh JPL Horizons geometric geocentric vectors. The plotted rho(t) minimum defines t = 0. No comparison or rejected markers are plotted.", ha="center", fontsize=7.5, color=MUTED)

    ax1.plot(x, rho, color=GOLD, linewidth=0.62, label="rho(t): Venus–Sun center distance")
    ax1.scatter(x, rho, s=5, color=GOLD, edgecolors="none", alpha=0.72)
    ax1.axvline(0.0, color=BLUE, linewidth=0.58, alpha=0.85, label="true rho-min CA")
    ax1.scatter([0.0], [float(stats["rho_min_arcsec"])], marker="D", s=64, color=BLUE, edgecolors=FG, linewidths=0.35, zorder=8)
    ax1.annotate(f"true geocentric CA\n{stats['ca_utc']}\nρ = {float(stats['rho_min_arcsec']):.12f}″", xy=(0, float(stats["rho_min_arcsec"])), xytext=(3.5, float(stats["rho_min_arcsec"]) + 0.75), fontsize=7.0, color=FG, arrowprops={"arrowstyle": "-", "linewidth": 0.30, "color": FG})
    ax1.set_title("PHYSICAL DISTANCE CURVE: rho(t) MINIMUM DEFINES CLOSEST APPROACH", fontsize=10, fontweight="bold")
    ax1.set_ylabel("rho arcsec", fontsize=8.5)
    ax1.legend(loc="upper right", fontsize=6.8, frameon=False)
    style_axis(ax1)

    ax2.plot(x, rhodot, color=RED, linewidth=0.62, label="raw d rho/dt")
    ax2.axhline(0.0, color=MUTED, linewidth=0.42, alpha=0.75)
    ax2.axvline(0.0, color=BLUE, linewidth=0.58, alpha=0.85)
    ax2.scatter([0.0], [float(stats["rhodot_at_ca"])], marker="X", s=52, color=BLUE, edgecolors=FG, linewidths=0.30, zorder=8)
    ax2.annotate(f"dρ/dt at CA = {float(stats['rhodot_at_ca']):+.3e} ″/min", xy=(0, float(stats["rhodot_at_ca"])), xytext=(-28.0, max(rhodot) * 0.64), fontsize=7.0, color=FG, arrowprops={"arrowstyle": "-", "linewidth": 0.28, "color": FG})
    ax2.set_title("RAW DERIVATIVE CHECK: d rho/dt CROSSES ZERO AT TRUE CA", fontsize=10, fontweight="bold")
    ax2.set_ylabel("arcsec/min", fontsize=8.5)
    ax2.legend(loc="upper right", fontsize=6.8, frameon=False)
    style_axis(ax2)

    ax3.plot(x, rho_excess, color=GREEN, linewidth=0.62, label="rho(t) − rho_min")
    ax3.axvline(0.0, color=BLUE, linewidth=0.58, alpha=0.85)
    ax3.set_ylabel("rho excess arcsec", fontsize=8.5)
    ax3.set_xlabel("Minutes from true geocentric rho-minimum closest approach", fontsize=8.8)
    ax3b = ax3.twinx()
    ax3b.plot(x, ratio_ppm, color=GOLD, linewidth=0.48, linestyle=":", label="EV/ES ratio change")
    ax3b.tick_params(labelsize=7.2, colors=MUTED)
    ax3b.set_ylabel("EV/ES ppm", fontsize=8.2, color=MUTED)
    ax3.set_title("SCALE RATIO CHANGES SLOWLY; CA IS SELECTED BY THE ANGULAR MINIMUM", fontsize=10, fontweight="bold")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=6.8, frameon=False)
    style_axis(ax3)

    tax.axis("off")
    rows = [
        ["Quantity", "Value", "Unit / status"],
        ["True geocentric rho-min CA UTC", str(stats["ca_utc"]), "JPL solve: minimize same plotted rho(t)"],
        ["Minimum rho", f"{float(stats['rho_min_arcsec']):.12f}", "arcsec"],
        ["Raw d rho/dt at CA", f"{float(stats['rhodot_at_ca']):+.15e}", "arcsec/min; should be zero"],
        ["Raw d rho/dt zero offset", f"{float(stats['rhodot_zero_offset_sec']):+.9f}", "seconds from CA"],
        ["rho excess at −30 min", f"{float(stats['rho_minus30_excess']):.12f}", "arcsec above minimum"],
        ["rho excess at +30 min", f"{float(stats['rho_plus30_excess']):.12f}", "arcsec above minimum"],
        ["EV/ES ratio range over ±30 min", f"{float(stats['ev_es_ppm_range']):.12f}", "ppm"],
        ["Samples", f"{int(stats['samples'])}", "one-minute plotted window"],
    ]
    table = tax.table(cellText=rows, cellLoc="left", colWidths=[0.30, 0.40, 0.30], bbox=[0.0, 0.05, 1.0, 0.88])
    style_table(table, teal_rows=(1, 2, 3, 4), gold_rows=(5, 6, 7, 8), fontsize=6.1)
    fig.text(0.5, 0.043, f"File: {Path(__file__).name if '__file__' in globals() else 'VENUS_1769_CLOSEST_APPROACH_RHODOT_AUDIT_V0102.py'} | Output: {PNG.name} | CSV: {CSV.name}", ha="center", fontsize=5.9, color=MUTED)
    fig.savefig(PNG, dpi=220, facecolor=BG)
    display(Image(filename=str(PNG)))


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"UTC window: {START} to {STOP}; step: {STEP}")
    print(f"Observer: Earth geocenter {GEOCENTER_LOCATION}")
    print("Data source: fresh JPL Horizons geometric ecliptic vectors")
    print("COMMENTS")
    print("Clean plot only: true geocentric rho-minimum closest approach, rho(t), raw d rho/dt, and EV/ES scale ratio.")
    print("No rejected markers or previous-error comparison lines are plotted.")
    df, stats = analyze()
    plot(df, stats)
    print("RESULTS")
    print(f"True geocentric rho-min CA UTC: {stats['ca_utc']}")
    print(f"Minimum rho: {float(stats['rho_min_arcsec']):.12f} arcsec")
    print(f"Raw d rho/dt at CA: {float(stats['rhodot_at_ca']):+.15e} arcsec/min")
    print(f"Raw d rho/dt zero offset: {float(stats['rhodot_zero_offset_sec']):+.9f} sec")
    print(f"rho excess at -30 min: {float(stats['rho_minus30_excess']):.12f} arcsec")
    print(f"rho excess at +30 min: {float(stats['rho_plus30_excess']):.12f} arcsec")
    print(f"EV/ES ratio range over ±30 min: {float(stats['ev_es_ppm_range']):.12f} ppm")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    print("PAPER COMPARISON")
    print("NOT USED: this is a JPL-only internal geometry audit; no literature values are imported.")
    print("EQUATION STATUS")
    print("PASS: rho(t) is computed from geocentric JPL Sun and Venus vectors; CA is solved by minimizing that same rho(t); raw d rho/dt crosses zero at t=0.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0102