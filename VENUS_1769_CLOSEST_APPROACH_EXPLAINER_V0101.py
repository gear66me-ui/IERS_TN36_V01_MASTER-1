# V0101
# Audit reference: standalone closest-approach explainer widget; Python/Matplotlib only; no AI images.
from __future__ import annotations

import math
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0101"
LOCAL_TZ = ZoneInfo("America/Bogota")
OUT = Path("/content/VENUS_1769_CLOSEST_APPROACH_EXPLAINER_V0101_OUTPUT")
PNG = OUT / "VENUS_1769_CLOSEST_APPROACH_EXPLAINER_V0101.png"
CSV = OUT / "VENUS_1769_CLOSEST_APPROACH_EXPLAINER_V0101.csv"

ARCSEC_PER_RAD = 206_264.80624709636
AU_KM = 149_597_870.700000
START = "1769-06-03 21:30"
STOP = "1769-06-03 23:10"
STEP = "1m"
WINDOW_MIN = 30.0
BG = "#000000"
FG = "#F8FAFC"
MUTED = "#B8CBD6"
GRID = "#263A4B"
BLUE = "#42D7C3"
GOLD = "#D89B18"
GREEN = "#74D680"
RED = "#FF6B6B"
PURPLE = "#B79CFF"
TABLE_HEADER = "#23466F"
TABLE_BODY = "#101A2E"
TABLE_TEAL = "#164B55"
TABLE_GOLD = "#563B0B"
TABLE_RED = "#5A081A"


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


def utc(jd: float) -> str:
    return Time(float(jd), format="jd", scale="tdb").utc.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def download(prefix: str, target_id: str, location) -> pd.DataFrame:
    last = None
    for attempt in range(4):
        try:
            tab = Horizons(id=target_id, location=location, epochs={"start": START, "stop": STOP, "step": STEP}, id_type=None).vectors(refplane="ecliptic", aberrations="geometric", cache=False)
            raw = tab.to_pandas()
            df = pd.DataFrame({"JD_TDB": pd.to_numeric(raw["datetime_jd"], errors="coerce")})
            for ax in "xyz":
                df[f"{prefix}_{ax.upper()}_KM"] = pd.to_numeric(raw[ax], errors="coerce") * AU_KM
            return df.dropna().drop_duplicates("JD_TDB").sort_values("JD_TDB")
        except Exception as exc:
            last = exc
            import time
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"JPL Horizons download failed for {prefix}: {last}")


def build_master() -> pd.DataFrame:
    frames = [
        download("SUN", "10", "500@399"),
        download("VENUS", "299", "500@399"),
    ]
    master = frames[0]
    for frame in frames[1:]:
        master = pd.merge(master, frame, on="JD_TDB", how="inner")
    if len(master) < 60:
        raise RuntimeError("Insufficient JPL samples for closest-approach explainer.")
    return master.sort_values("JD_TDB").reset_index(drop=True)


def splines(master: pd.DataFrame) -> dict[str, object]:
    jd = master["JD_TDB"].to_numpy(float)
    c: dict[str, object] = {"JD_TDB": jd}
    for col in master.columns:
        if col != "JD_TDB":
            c[col] = CubicSpline(jd, master[col].to_numpy(float))
    return c


def vec(c: dict[str, object], prefix: str, jd: float) -> np.ndarray:
    return np.array([float(c[f"{prefix}_X_KM"](jd)), float(c[f"{prefix}_Y_KM"](jd)), float(c[f"{prefix}_Z_KM"](jd))])


def rho_arcsec(c: dict[str, object], jd: float) -> float:
    sun = vec(c, "SUN", jd)
    ven = vec(c, "VENUS", jd)
    cosang = float(np.clip(np.dot(unit(sun), unit(ven)), -1.0, 1.0))
    return math.degrees(math.acos(cosang)) * 3600.0


def geocentric_ca(c: dict[str, object]) -> float:
    jd = np.asarray(c["JD_TDB"], dtype=float)
    res = minimize_scalar(lambda t: rho_arcsec(c, float(t)), bounds=(float(jd.min()), float(jd.max())), method="bounded", options={"xatol": 1e-11, "maxiter": 1000})
    if not res.success:
        raise RuntimeError("Closest approach solve failed.")
    return float(res.x)


def derivative_per_minute(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.gradient(y, x)


def zero_root(x: np.ndarray, y: np.ndarray) -> float:
    spl = CubicSpline(x, y)
    roots = []
    for a, b in zip(x[:-1], x[1:]):
        fa = float(spl(a))
        fb = float(spl(b))
        if fa == 0.0:
            roots.append(float(a))
        elif fa * fb < 0.0:
            roots.append(float(brentq(lambda z: float(spl(z)), float(a), float(b))))
    if not roots:
        return float("nan")
    return min(roots, key=abs)


def analyze() -> tuple[pd.DataFrame, dict[str, float | str]]:
    OUT.mkdir(parents=True, exist_ok=True)
    master = build_master()
    c = splines(master)
    ca_jd = geocentric_ca(c)
    ca_rho = rho_arcsec(c, ca_jd)
    start = ca_jd - WINDOW_MIN / 1440.0
    stop = ca_jd + WINDOW_MIN / 1440.0
    sample_jd = np.linspace(start, stop, int(2 * WINDOW_MIN) + 1)
    rows = []
    for jd in sample_jd:
        sun = vec(c, "SUN", float(jd))
        ven = vec(c, "VENUS", float(jd))
        des = norm(sun)
        dev = norm(ven)
        rho = rho_arcsec(c, float(jd))
        rows.append({
            "jd_tdb": float(jd),
            "utc": utc(float(jd)),
            "minute_from_ca": (float(jd) - ca_jd) * 1440.0,
            "rho_arcsec": rho,
            "rho_minus_min_arcsec": rho - ca_rho,
            "earth_sun_km": des,
            "earth_venus_km": dev,
            "earth_venus_over_earth_sun": dev / des,
            "earth_venus_over_earth_sun_delta_ppm": 1.0e6 * ((dev / des) - (norm(vec(c, "VENUS", ca_jd)) / norm(vec(c, "SUN", ca_jd)))),
        })
    df = pd.DataFrame(rows)
    df["drho_dt_arcsec_per_min"] = derivative_per_minute(df["minute_from_ca"].to_numpy(float), df["rho_arcsec"].to_numpy(float))
    df["abs_drho_dt_arcsec_per_min"] = np.abs(df["drho_dt_arcsec_per_min"].to_numpy(float))
    rho_spl = CubicSpline(df["minute_from_ca"].to_numpy(float), df["rho_arcsec"].to_numpy(float))
    fine_x = np.linspace(-WINDOW_MIN, WINDOW_MIN, 2401)
    fine_rho = rho_spl(fine_x)
    min_i = int(np.argmin(fine_rho))
    slope_zero = zero_root(df["minute_from_ca"].to_numpy(float), df["drho_dt_arcsec_per_min"].to_numpy(float))
    ratio = df["earth_venus_over_earth_sun"].to_numpy(float)
    stats: dict[str, float | str] = {
        "ca_jd": ca_jd,
        "ca_utc": utc(ca_jd),
        "ca_rho_arcsec": ca_rho,
        "rho_min_fine_minute": float(fine_x[min_i]),
        "rho_min_fine_arcsec": float(fine_rho[min_i]),
        "slope_zero_minute": float(slope_zero),
        "slope_at_ca": float(CubicSpline(df["minute_from_ca"].to_numpy(float), df["drho_dt_arcsec_per_min"].to_numpy(float))(0.0)),
        "rho_plus_30": float(df.loc[df["minute_from_ca"].idxmax(), "rho_arcsec"]),
        "rho_minus_30": float(df.loc[df["minute_from_ca"].idxmin(), "rho_arcsec"]),
        "rho_excess_plus_30": float(df.loc[df["minute_from_ca"].idxmax(), "rho_minus_min_arcsec"]),
        "rho_excess_minus_30": float(df.loc[df["minute_from_ca"].idxmin(), "rho_minus_min_arcsec"]),
        "ratio_ca": float(norm(vec(c, "VENUS", ca_jd)) / norm(vec(c, "SUN", ca_jd))),
        "ratio_ppm_range": float(1.0e6 * (ratio.max() - ratio.min())),
        "samples": float(len(df)),
    }
    df.to_csv(CSV, index=False, float_format="%.15f")
    return df, stats


def style_axis(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.32, alpha=0.55)
    ax.tick_params(labelsize=7.5, width=0.35, length=2.6)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.35)


def table_style(table, teal_rows=(), gold_rows=(), red_rows=(), fontsize=6.3) -> None:
    table.auto_set_font_size(False)
    for (r, _c), cell in table.get_celld().items():
        cell.set_edgecolor("#70879A")
        cell.set_linewidth(0.30)
        cell.get_text().set_color(FG)
        cell.get_text().set_fontsize(fontsize)
        if r == 0:
            cell.set_facecolor(TABLE_HEADER)
            cell.get_text().set_fontweight("bold")
        elif r in teal_rows:
            cell.set_facecolor(TABLE_TEAL)
            cell.get_text().set_fontweight("bold")
        elif r in gold_rows:
            cell.set_facecolor(TABLE_GOLD)
            cell.get_text().set_fontweight("bold")
        elif r in red_rows:
            cell.set_facecolor(TABLE_RED)
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
    x = df["minute_from_ca"].to_numpy(float)
    rho = df["rho_arcsec"].to_numpy(float)
    excess = df["rho_minus_min_arcsec"].to_numpy(float)
    slope = df["drho_dt_arcsec_per_min"].to_numpy(float)
    ratio_ppm = df["earth_venus_over_earth_sun_delta_ppm"].to_numpy(float)

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    gs = fig.add_gridspec(4, 1, height_ratios=[0.38, 0.20, 0.20, 0.22], left=0.060, right=0.985, top=0.895, bottom=0.105, hspace=0.26)
    ax_rho = fig.add_subplot(gs[0, 0])
    ax_excess = fig.add_subplot(gs[1, 0], sharex=ax_rho)
    ax_slope = fig.add_subplot(gs[2, 0], sharex=ax_rho)
    tab_ax = fig.add_subplot(gs[3, 0])
    fig.suptitle("1769 Venus Transit — Why Closest Approach Is the Physical Reference", fontsize=14.5, fontweight="bold", y=0.958)
    fig.text(0.5, 0.925, "Fresh JPL Horizons geocentric geometric vectors. Window is ±30 minutes around geocentric closest approach; no AI images.", ha="center", fontsize=7.5, color=MUTED)

    ax_rho.plot(x, rho, color=GOLD, linewidth=0.72, label="ρ(t): apparent Venus–Sun center distance")
    ax_rho.scatter(x, rho, s=5.0, color=GOLD, edgecolors="none", alpha=0.75)
    ax_rho.axvline(0.0, color=BLUE, linewidth=0.50, alpha=0.85, label="geocentric CA")
    ax_rho.scatter([0.0], [float(stats["ca_rho_arcsec"])], s=72, marker="D", color=BLUE, edgecolors=FG, linewidths=0.38, zorder=7)
    ax_rho.annotate(f"minimum ρ = {float(stats['ca_rho_arcsec']):.9f}″\nCA UTC {stats['ca_utc']}", xy=(0.0, float(stats["ca_rho_arcsec"])), xytext=(4.0, float(stats["ca_rho_arcsec"]) + 0.015), fontsize=7.1, color=FG, ha="left", arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.32})
    ax_rho.set_ylabel("ρ arcsec", fontsize=8.8)
    ax_rho.set_title("PHYSICAL CLOSEST APPROACH: ρ(t) HAS ITS MINIMUM AT CA", fontsize=9.8, fontweight="bold")
    ax_rho.legend(loc="upper right", fontsize=6.8, frameon=False)
    style_axis(ax_rho)

    ax_excess.plot(x, excess, color=GREEN, linewidth=0.68, label="ρ(t) − ρmin")
    ax_excess.plot(x, np.abs(slope) / max(np.max(np.abs(slope)), 1e-30) * max(np.max(excess), 1e-30), color=PURPLE, linewidth=0.56, linestyle="--", label="|dρ/dt| normalized to same height")
    ax_excess.axvline(0.0, color=BLUE, linewidth=0.45, alpha=0.78)
    ax_excess.axhline(0.0, color=MUTED, linewidth=0.35, alpha=0.62)
    ax_excess.set_ylabel("ρ excess arcsec", fontsize=8.8)
    ax_excess.set_title("CA ALSO ZEROES THE RADIAL MOTION: dρ/dt = 0", fontsize=9.4, fontweight="bold")
    ax_excess.legend(loc="upper right", fontsize=6.5, frameon=False)
    style_axis(ax_excess)

    ax_slope.plot(x, slope, color=RED, linewidth=0.68, label="dρ/dt")
    ax_slope.axvline(0.0, color=BLUE, linewidth=0.45, alpha=0.78)
    ax_slope.axhline(0.0, color=MUTED, linewidth=0.38, alpha=0.72)
    ax_slope_twin = ax_slope.twinx()
    ax_slope_twin.plot(x, ratio_ppm, color=GOLD, linewidth=0.50, linestyle=":", label="EV/ES ratio change")
    ax_slope_twin.tick_params(labelsize=7.2, colors=MUTED, width=0.35, length=2.4)
    for spine in ax_slope_twin.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.35)
    ax_slope.set_ylabel("dρ/dt arcsec/min", fontsize=8.8)
    ax_slope_twin.set_ylabel("EV/ES Δ ppm", fontsize=8.5, color=MUTED)
    ax_slope.set_xlabel("Minutes from geocentric closest approach", fontsize=8.8)
    ax_slope.set_title("DISTANCE-SCALE RATIO CHANGES SLOWLY; THE CA SELECTION IS DRIVEN BY ANGULAR GEOMETRY", fontsize=9.4, fontweight="bold")
    lines1, labels1 = ax_slope.get_legend_handles_labels()
    lines2, labels2 = ax_slope_twin.get_legend_handles_labels()
    ax_slope.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=6.5, frameon=False)
    style_axis(ax_slope)

    tab_ax.axis("off")
    rows = [
        ["Quantity", "Value", "Unit / trace"],
        ["Geocentric CA UTC", str(stats["ca_utc"]), "JPL solve: minimum ρ"],
        ["Minimum ρ", f"{float(stats['ca_rho_arcsec']):.12f}", "arcsec"],
        ["dρ/dt at CA", f"{float(stats['slope_at_ca']):+.12e}", "arcsec/min, should be near zero"],
        ["ρ excess at −30 min", f"{float(stats['rho_excess_minus_30']):.12f}", "arcsec above minimum"],
        ["ρ excess at +30 min", f"{float(stats['rho_excess_plus_30']):.12f}", "arcsec above minimum"],
        ["EV/ES at CA", f"{float(stats['ratio_ca']):.12f}", "Earth–Venus / Earth–Sun"],
        ["EV/ES range over ±30 min", f"{float(stats['ratio_ppm_range']):.12f}", "ppm"],
        ["Samples", f"{int(float(stats['samples']))}", "one-minute JPL samples"],
    ]
    table = tab_ax.table(cellText=rows, cellLoc="left", colWidths=[0.31, 0.31, 0.38], bbox=[0.0, 0.06, 1.0, 0.88])
    table_style(table, teal_rows=(1, 2, 3), gold_rows=(4, 5, 6, 7, 8), fontsize=6.5)
    fig.text(0.5, 0.043, f"File: {Path(__file__).name if '__file__' in globals() else 'VENUS_1769_CLOSEST_APPROACH_EXPLAINER_V0101.py'} | Output: {PNG.name} | CSV: {CSV.name}", ha="center", fontsize=5.8, color=MUTED)
    fig.savefig(PNG, dpi=220, facecolor=BG)
    display(Image(filename=str(PNG)))


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"JPL window UTC: {START} to {STOP}; plotted window: ±{WINDOW_MIN:.1f} minutes around geocentric CA")
    print("Data source: fresh JPL Horizons geocentric geometric ecliptic vectors for Sun and Venus")
    print("COMMENTS")
    print("Plots physical closest-approach distance ρ(t), radial derivative dρ/dt, and Earth–Venus/Earth–Sun scale ratio around CA.")
    print("No AI images; Python/Matplotlib plot only.")
    df, stats = analyze()
    plot(df, stats)
    print("RESULTS")
    print(f"Geocentric CA UTC: {stats['ca_utc']}")
    print(f"Minimum ρ: {float(stats['ca_rho_arcsec']):.12f} arcsec")
    print(f"dρ/dt at CA: {float(stats['slope_at_ca']):+.12e} arcsec/min")
    print(f"ρ excess at -30 min: {float(stats['rho_excess_minus_30']):.12f} arcsec")
    print(f"ρ excess at +30 min: {float(stats['rho_excess_plus_30']):.12f} arcsec")
    print(f"EV/ES at CA: {float(stats['ratio_ca']):.12f}")
    print(f"EV/ES range over ±30 min: {float(stats['ratio_ppm_range']):.12f} ppm")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    print("PAPER COMPARISON")
    print("NOT USED: this is an internal JPL-vector geometry explainer only.")
    print("EQUATION STATUS")
    print("PASS: ρ is computed from geocentric Sun and Venus unit-vector angular separation; CA is solved by minimizing ρ(t); dρ/dt is derived numerically from one-minute JPL samples.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0101
