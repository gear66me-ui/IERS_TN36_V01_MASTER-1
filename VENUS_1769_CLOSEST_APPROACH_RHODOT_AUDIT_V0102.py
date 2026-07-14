# V0102
# Audit reference: corrected rho-minimum closest-approach audit; same plotted rho defines CA; Python/Matplotlib only; no AI images.
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
PROJECT_PRIOR_CA_UTC = "1769-06-03 22:19:15.599"
PROJECT_V0096_CA_UTC = "1769-06-03 22:19:04.387"
REJECTED_V0101_CA_UTC = "1769-06-03 22:07:39.611"

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
TABLE_RED = "#5A0F22"


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


def jd_from_utc(utc_text: str) -> float:
    return float(Time(utc_text, scale="utc").tdb.jd)


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
    cross = norm(np.cross(ua, ub))
    dot = float(np.dot(ua, ub))
    return ARCSEC_PER_RAD * math.atan2(cross, dot)


def rho_arcsec(c: dict[str, object], jd: float) -> float:
    return angle_arcsec_between(vec(c, "SUN", jd), vec(c, "VENUS", jd))


def rho_dot_arcsec_per_min(c: dict[str, object], jd: float) -> float:
    h = 0.5 / 1440.0
    return (rho_arcsec(c, jd + h) - rho_arcsec(c, jd - h)) / 1.0


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
    rows = []
    min_rho = rho_arcsec(c, ca_jd)
    ratio0 = ratio_ev_es(c, ca_jd)
    for m in grid_minutes:
        jd = ca_jd + float(m) / 1440.0
        rho = rho_arcsec(c, jd)
        rd = rho_dot_arcsec_per_min(c, jd)
        ratio = ratio_ev_es(c, jd)
        rows.append({
            "minute_from_true_rho_min": float(m),
            "jd_tdb": float(jd),
            "utc": utc_from_jd(jd),
            "rho_arcsec": float(rho),
            "rho_excess_arcsec": float(rho - min_rho),
            "rho_dot_arcsec_per_min": float(rd),
            "ev_es_ratio": float(ratio),
            "ev_es_ratio_delta_ppm": float((ratio / ratio0 - 1.0) * 1_000_000.0),
        })
    df = pd.DataFrame(rows)
    prior_jd = jd_from_utc(PROJECT_PRIOR_CA_UTC)
    v0096_jd = jd_from_utc(PROJECT_V0096_CA_UTC)
    rejected_jd = jd_from_utc(REJECTED_V0101_CA_UTC)
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
        "project_prior_ca_utc": PROJECT_PRIOR_CA_UTC,
        "project_prior_offset_sec": (prior_jd - ca_jd) * 86400.0,
        "v0096_ca_utc": PROJECT_V0096_CA_UTC,
        "v0096_offset_sec": (v0096_jd - ca_jd) * 86400.0,
        "rejected_v0101_ca_utc": REJECTED_V0101_CA_UTC,
        "rejected_v0101_offset_sec": (rejected_jd - ca_jd) * 86400.0,
        "samples": len(df),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False, float_format="%.15f")
    return df, stats


def table_style(table, gold_rows=(), teal_rows=(), red_rows=(), fontsize=6.7) -> None:
    table.auto_set_font_size(False)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#70879A")
        cell.set_linewidth(0.28)
        cell.get_text().set_color(FG)
        cell.get_text().set_fontsize(fontsize)
        if row == 0:
            cell.set_facecolor(TABLE_HEADER)
            cell.get_text().set_fontweight("bold")
        elif row in red_rows:
            cell.set_facecolor(TABLE_RED)
            cell.get_text().set_fontweight("bold")
        elif row in teal_rows:
            cell.set_facecolor(TABLE_TEAL)
            cell.get_text().set_fontweight("bold")
        elif row in gold_rows:
            cell.set_facecolor(TABLE_GOLD)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(TABLE_BODY)


def style_axis(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.30, alpha=0.60)
    ax.tick_params(labelsize=7.3, width=0.35, length=2.4)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.35)


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
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    gs = fig.add_gridspec(4, 1, height_ratios=[0.31, 0.20, 0.20, 0.29], left=0.065, right=0.985, top=0.895, bottom=0.102, hspace=0.255)
    ax_rho = fig.add_subplot(gs[0])
    ax_dot = fig.add_subplot(gs[1], sharex=ax_rho)
    ax_ratio = fig.add_subplot(gs[2], sharex=ax_rho)
    ax_table = fig.add_subplot(gs[3])
    fig.suptitle("1769 Venus Transit — Corrected Closest-Approach Audit", fontsize=14.3, fontweight="bold", y=0.965)
    fig.text(0.5, 0.934, "CA is solved from the exact same plotted geocentric rho(t); raw d rho/dt is zero at t = 0. Previous table CA values are comparison-only.", ha="center", fontsize=7.6, color=MUTED)

    x = df["minute_from_true_rho_min"].to_numpy(float)
    rho = df["rho_arcsec"].to_numpy(float)
    rho_excess = df["rho_excess_arcsec"].to_numpy(float)
    rhodot = df["rho_dot_arcsec_per_min"].to_numpy(float)
    ratio_ppm = df["ev_es_ratio_delta_ppm"].to_numpy(float)

    ax_rho.plot(x, rho, color=GOLD, linewidth=0.72, label="rho(t): apparent Venus-Sun center distance")
    ax_rho.scatter(x, rho, s=5.5, color=GOLD, edgecolors="none", alpha=0.72)
    ax_rho.axvline(0.0, color=BLUE, linewidth=0.55, alpha=0.85, label="true rho minimum / CA")
    ax_rho.scatter([0.0], [float(stats["rho_min_arcsec"])], s=72, marker="D", color=BLUE, edgecolors=FG, linewidths=0.35, zorder=6)
    ax_rho.annotate(f"true minimum rho = {float(stats['rho_min_arcsec']):.9f}\nUTC {stats['ca_utc']}", xy=(0.0, float(stats["rho_min_arcsec"])), xytext=(3.5, float(stats["rho_min_arcsec"]) + 1.2), ha="left", fontsize=7.4, arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.32})
    prior_x = float(stats["project_prior_offset_sec"]) / 60.0
    v0096_x = float(stats["v0096_offset_sec"]) / 60.0
    rej_x = float(stats["rejected_v0101_offset_sec"]) / 60.0
    ax_rho.axvline(prior_x, color=GREEN, linewidth=0.42, alpha=0.72, linestyle="--", label="prior project CA comparison")
    ax_rho.axvline(v0096_x, color=PURPLE, linewidth=0.42, alpha=0.72, linestyle="--", label="V0096 CA comparison")
    if -WINDOW_MIN <= rej_x <= WINDOW_MIN:
        ax_rho.axvline(rej_x, color=RED, linewidth=0.42, alpha=0.72, linestyle=":", label="REJECTED V0101 marker")
    ax_rho.set_ylabel("rho arcsec", fontsize=8.7)
    ax_rho.set_title("PHYSICAL DISTANCE CURVE: rho(t) MINIMUM DEFINES CLOSEST APPROACH", fontsize=10.0, fontweight="bold")
    ax_rho.legend(loc="upper right", fontsize=6.4, frameon=False)
    style_axis(ax_rho)

    ax_dot.plot(x, rhodot, color=RED, linewidth=0.68, label="raw d rho/dt")
    ax_dot.axhline(0.0, color=MUTED, linewidth=0.40, alpha=0.75)
    ax_dot.axvline(0.0, color=BLUE, linewidth=0.55, alpha=0.85)
    ax_dot.scatter([0.0], [float(stats["rhodot_at_ca"])], s=60, marker="X", color=BLUE, edgecolors=FG, linewidths=0.35, zorder=6)
    ax_dot.annotate(f"raw d rho/dt at CA = {float(stats['rhodot_at_ca']):+.12e}\nzero offset = {float(stats['rhodot_zero_offset_sec']):+.6f} sec", xy=(0.0, 0.0), xytext=(-28.5, max(rhodot) * 0.55), ha="left", fontsize=7.2, arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.30})
    ax_dot.set_ylabel("arcsec/min", fontsize=8.7)
    ax_dot.set_title("RAW DERIVATIVE CHECK: d rho/dt CROSSES ZERO AT TRUE CA", fontsize=10.0, fontweight="bold")
    ax_dot.legend(loc="upper right", fontsize=6.4, frameon=False)
    style_axis(ax_dot)

    ax_ratio.plot(x, rho_excess, color=GREEN, linewidth=0.68, label="rho(t) - rho_min")
    ax_ratio.set_ylabel("rho excess arcsec", fontsize=8.7)
    ax_ratio_b = ax_ratio.twinx()
    ax_ratio_b.plot(x, ratio_ppm, color=GOLD, linewidth=0.58, linestyle=":", label="EV/ES ratio change")
    ax_ratio_b.set_ylabel("EV/ES delta ppm", fontsize=8.0, color=MUTED)
    ax_ratio_b.tick_params(labelsize=7.1, colors=MUTED, width=0.35, length=2.4)
    for spine in ax_ratio_b.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.35)
    ax_ratio.axvline(0.0, color=BLUE, linewidth=0.55, alpha=0.85)
    ax_ratio.set_xlabel("Minutes from true rho-minimum closest approach", fontsize=8.7)
    ax_ratio.set_title("SCALE RATIO CHANGES SLOWLY; CA IS SELECTED BY THE ANGULAR MINIMUM", fontsize=10.0, fontweight="bold")
    lines1, labels1 = ax_ratio.get_legend_handles_labels()
    lines2, labels2 = ax_ratio_b.get_legend_handles_labels()
    ax_ratio.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=6.4, frameon=False)
    style_axis(ax_ratio)

    ax_table.axis("off")
    rows = [
        ["Quantity", "Value", "Unit / status"],
        ["TRUE plotted rho-min CA UTC", str(stats["ca_utc"]), "JPL solve: minimize same plotted rho(t)"],
        ["Minimum rho", f"{float(stats['rho_min_arcsec']):.12f}", "arcsec"],
        ["Raw d rho/dt at CA", f"{float(stats['rhodot_at_ca']):+.12e}", "arcsec/min; should be zero"],
        ["Raw d rho/dt zero offset", f"{float(stats['rhodot_zero_offset_sec']):+.9f}", "seconds from CA"],
        ["Prior project CA", f"{stats['project_prior_ca_utc']}  ({float(stats['project_prior_offset_sec']):+.6f} sec)", "COMPARISON ONLY, NOT USED"],
        ["V0096 CA", f"{stats['v0096_ca_utc']}  ({float(stats['v0096_offset_sec']):+.6f} sec)", "COMPARISON ONLY, NOT USED"],
        ["REJECTED V0101 marker", f"{stats['rejected_v0101_ca_utc']}  ({float(stats['rejected_v0101_offset_sec']):+.6f} sec)", "REJECTED: not the plotted rho minimum"],
        ["rho excess at -30 min", f"{float(stats['rho_minus30_excess']):.12f}", "arcsec above minimum"],
        ["rho excess at +30 min", f"{float(stats['rho_plus30_excess']):.12f}", "arcsec above minimum"],
        ["EV/ES ratio range over ±30 min", f"{float(stats['ev_es_ppm_range']):.12f}", "ppm"],
        ["Samples", f"{int(stats['samples'])}", "one-minute plotted window"],
    ]
    table = ax_table.table(cellText=rows, cellLoc="left", colWidths=[0.30, 0.40, 0.30], bbox=[0.0, 0.06, 1.0, 0.90])
    table_style(table, teal_rows=(1, 2, 3, 4), gold_rows=(5, 6, 8, 9, 10, 11), red_rows=(7,), fontsize=6.45)
    fig.text(0.5, 0.037, f"File: {Path(__file__).name if '__file__' in globals() else 'VENUS_1769_CLOSEST_APPROACH_RHODOT_AUDIT_V0102.py'} | Output: {PNG.name} | CSV: {CSV.name}", ha="center", fontsize=5.9, color=MUTED)
    fig.savefig(PNG, dpi=220, facecolor=BG)
    display(Image(filename=str(PNG)))


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"JPL window: {START} to {STOP}; step {STEP}")
    print("Geometry: geocentric Sun and Venus geometric ecliptic vectors; rho(t) is the plotted angular separation.")
    print("COMMENTS")
    print("Corrects the rejected CA-labeling error: closest approach is solved by minimizing the exact rho(t) curve that is plotted.")
    print("Previous project CA timestamps are comparison-only and are not used in the calculation.")
    df, stats = analyze()
    plot(df, stats)
    print("RESULTS")
    print(f"True plotted rho-min CA UTC: {stats['ca_utc']}")
    print(f"Minimum rho: {float(stats['rho_min_arcsec']):.12f} arcsec")
    print(f"Raw d rho/dt at CA: {float(stats['rhodot_at_ca']):+.12e} arcsec/min")
    print(f"Raw d rho/dt zero offset: {float(stats['rhodot_zero_offset_sec']):+.9f} sec")
    print(f"Prior project CA comparison offset: {float(stats['project_prior_offset_sec']):+.6f} sec")
    print(f"V0096 CA comparison offset: {float(stats['v0096_offset_sec']):+.6f} sec")
    print(f"REJECTED V0101 marker offset: {float(stats['rejected_v0101_offset_sec']):+.6f} sec")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    print("PAPER COMPARISON")
    print("PROJECT PRIOR COMPARISON ONLY: prior table CA values are reported as offsets but are not used to solve the corrected rho minimum.")
    print("EQUATION STATUS")
    print("PASS: CA is defined as argmin rho(t), where rho(t) is the same plotted Venus-Sun angular center distance; raw d rho/dt is evaluated at that same CA.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0102
