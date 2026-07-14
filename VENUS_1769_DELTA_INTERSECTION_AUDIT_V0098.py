# V0098
# Audit reference: V0096 CSV bottom-panel intersection audit; Python/Matplotlib only; no AI images.
from __future__ import annotations

import math
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


def require(import_name: str, package_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", package_name])


for _import, _package in (("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("matplotlib", "matplotlib"), ("IPython", "ipython")):
    require(_import, _package)

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar
from IPython.display import Image, display

VERSION = "V0098"
LOCAL_TZ = ZoneInfo("America/Bogota")
OUT = Path("/content/VENUS_1769_DELTA_INTERSECTION_AUDIT_V0098_OUTPUT")
PNG = OUT / "VENUS_1769_DELTA_INTERSECTION_AUDIT_V0098.png"
RESULT_CSV = OUT / "VENUS_1769_DELTA_INTERSECTION_AUDIT_V0098.csv"
TARGET_CSV = "VENUS_1769_APRIME_BPRIME_AB_FOUR_CURVE_DELTA_V0096.csv"

BG = "#000000"
FG = "#F8FAFC"
MUTED = "#B8CBD6"
GRID = "#263A4B"
BLUE = "#42D7C3"
GOLD = "#D89B18"
GREEN = "#74D680"
RED = "#FF6B6B"
PURPLE = "#A78BFA"
TABLE_HEADER = "#23466F"
TABLE_BODY = "#101A2E"
TABLE_TEAL = "#164B55"
TABLE_GOLD = "#563B0B"


def find_source_csv() -> Path:
    candidates = [
        Path.cwd() / TARGET_CSV,
        Path("/content") / TARGET_CSV,
        Path("/content/VENUS_1769_APRIME_BPRIME_AB_FOUR_CURVE_DELTA_V0096_OUTPUT") / TARGET_CSV,
        Path("/content/GitHub_Sandbox_DELETE_1769/VENUS_1769_APRIME_BPRIME_AB_FOUR_CURVE_DELTA_V0096_OUTPUT") / TARGET_CSV,
    ]
    matches = []
    for c in candidates:
        if c.exists():
            matches.append(c)
    matches.extend(Path("/content").rglob(TARGET_CSV))
    unique = []
    seen = set()
    for p in matches:
        rp = str(p.resolve())
        if rp not in seen:
            unique.append(p)
            seen.add(rp)
    if not unique:
        raise FileNotFoundError(f"Could not find {TARGET_CSV}. Run V0096 first, then run this audit.")
    unique.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return unique[0]


def required_columns(df: pd.DataFrame) -> None:
    required = [
        "jd_tdb",
        "utc",
        "minute_from_start",
        "apbp_fixed_arcsec",
        "apbp_instant_arcsec",
        "ab_fixed_arcsec",
        "ab_instant_arcsec",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required V0096 columns: {missing}. Available columns: {list(df.columns)}")


def sign_roots(x: np.ndarray, y: np.ndarray, spline: CubicSpline) -> list[float]:
    roots: list[float] = []
    for i in range(len(x) - 1):
        y0 = float(y[i])
        y1 = float(y[i + 1])
        if not np.isfinite(y0) or not np.isfinite(y1):
            continue
        if abs(y0) < 1e-15:
            roots.append(float(x[i]))
        elif y0 * y1 < 0.0:
            roots.append(float(brentq(lambda t: float(spline(t)), float(x[i]), float(x[i + 1]), xtol=1e-13, rtol=1e-14)))
    clean: list[float] = []
    for r in roots:
        if not clean or abs(r - clean[-1]) > 1e-8:
            clean.append(r)
    return clean


def nearest_value(x: np.ndarray, y: np.ndarray, target: float) -> tuple[float, float]:
    idx = int(np.argmin(np.abs(x - target)))
    return float(x[idx]), float(y[idx])


def extremum_root(x: np.ndarray, spline: CubicSpline) -> float:
    lo = float(x.min())
    hi = float(x.max())
    res = minimize_scalar(lambda t: abs(float(spline(t))), bounds=(lo, hi), method="bounded", options={"xatol": 1e-10, "maxiter": 1000})
    if not res.success:
        raise RuntimeError("Minimum residual search failed.")
    return float(res.x)


def interpolate_utc(df: pd.DataFrame, minute: float) -> str:
    x = df["minute_from_start"].to_numpy(float)
    j = df["jd_tdb"].to_numpy(float)
    jd = float(np.interp(float(minute), x, j))
    try:
        from astropy.time import Time
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "astropy"])
        from astropy.time import Time
    return Time(jd, format="jd", scale="tdb").utc.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def locate_geo_ca(df: pd.DataFrame) -> float:
    # The V0096 CSV may not carry a dedicated geocentric-CA flag; infer from the point where |2ΔA′B′| is smallest near CA.
    x = df["minute_from_start"].to_numpy(float)
    y = df["delta_apbp_arcsec"].to_numpy(float) if "delta_apbp_arcsec" in df.columns else (df["apbp_instant_arcsec"].to_numpy(float) - df["apbp_fixed_arcsec"].to_numpy(float))
    center_mask = (x > x.min() + 0.35 * (x.max() - x.min())) & (x < x.min() + 0.65 * (x.max() - x.min()))
    idxs = np.where(center_mask)[0]
    if len(idxs) == 0:
        idxs = np.arange(len(x))
    idx = idxs[int(np.argmin(np.abs(y[idxs])))]
    return float(x[idx])


def analyze() -> tuple[pd.DataFrame, dict[str, float | str | list[float]], Path]:
    source = find_source_csv()
    df = pd.read_csv(source)
    required_columns(df)
    df = df.sort_values("minute_from_start").reset_index(drop=True)
    x = df["minute_from_start"].to_numpy(float)
    df["delta_apbp_arcsec"] = df["apbp_instant_arcsec"].to_numpy(float) - df["apbp_fixed_arcsec"].to_numpy(float)
    df["delta_ab_arcsec"] = df["ab_instant_arcsec"].to_numpy(float) - df["ab_fixed_arcsec"].to_numpy(float)
    df["delta_delta_arcsec"] = df["delta_apbp_arcsec"] - df["delta_ab_arcsec"]
    df["delta_apbp_2x_arcsec"] = 2.0 * df["delta_apbp_arcsec"]
    df["delta_ab_2x_arcsec"] = 2.0 * df["delta_ab_arcsec"]
    df["delta_delta_2x_arcsec"] = 2.0 * df["delta_delta_arcsec"]

    ap_spline = CubicSpline(x, df["delta_apbp_arcsec"].to_numpy(float), bc_type="natural")
    ab_spline = CubicSpline(x, df["delta_ab_arcsec"].to_numpy(float), bc_type="natural")
    dd_spline = CubicSpline(x, df["delta_delta_arcsec"].to_numpy(float), bc_type="natural")

    ap_zero = sign_roots(x, df["delta_apbp_arcsec"].to_numpy(float), ap_spline)
    ab_zero = sign_roots(x, df["delta_ab_arcsec"].to_numpy(float), ab_spline)
    cross = sign_roots(x, df["delta_delta_arcsec"].to_numpy(float), dd_spline)

    if not ap_zero:
        ap_zero = [extremum_root(x, ap_spline)]
    if not ab_zero:
        ab_zero = [extremum_root(x, ab_spline)]
    if not cross:
        cross = [extremum_root(x, dd_spline)]

    geo_minute = locate_geo_ca(df)
    stats: dict[str, float | str | list[float]] = {
        "source": str(source),
        "ap_zero_minute": float(ap_zero[0]),
        "ab_zero_minute": float(ab_zero[0]),
        "intersection_minute": float(cross[0]),
        "geo_marker_minute": float(geo_minute),
        "ap_zero_utc": interpolate_utc(df, float(ap_zero[0])),
        "ab_zero_utc": interpolate_utc(df, float(ab_zero[0])),
        "intersection_utc": interpolate_utc(df, float(cross[0])),
        "geo_marker_utc": interpolate_utc(df, float(geo_minute)),
        "ap_zero_delta_ap": float(ap_spline(float(ap_zero[0]))),
        "ab_zero_delta_ab": float(ab_spline(float(ab_zero[0]))),
        "intersection_delta_ap": float(ap_spline(float(cross[0]))),
        "intersection_delta_ab": float(ab_spline(float(cross[0]))),
        "intersection_residual": float(dd_spline(float(cross[0]))),
        "geo_delta_ap": float(ap_spline(float(geo_minute))),
        "geo_delta_ab": float(ab_spline(float(geo_minute))),
        "geo_minus_intersection_min": float(geo_minute - float(cross[0])),
        "mean_delta_ap": float(df["delta_apbp_arcsec"].mean()),
        "mean_delta_ab": float(df["delta_ab_arcsec"].mean()),
        "rms_delta_ap": float(np.sqrt(np.mean(df["delta_apbp_arcsec"].to_numpy(float) ** 2))),
        "rms_delta_ab": float(np.sqrt(np.mean(df["delta_ab_arcsec"].to_numpy(float) ** 2))),
        "samples": int(len(df)),
    }
    return df, stats, source


def table_style(table, teal_rows=(), gold_rows=(), fontsize=6.3) -> None:
    table.auto_set_font_size(False)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#70879A")
        cell.set_linewidth(0.32)
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


def plot(df: pd.DataFrame, stats: dict[str, float | str | list[float]], source: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result_rows = [
        {"quantity": "ΔA′B′ zero crossing", "minute": stats["ap_zero_minute"], "utc": stats["ap_zero_utc"], "delta_apbp_arcsec": stats["ap_zero_delta_ap"], "delta_ab_arcsec": np.nan, "residual_arcsec": 0.0},
        {"quantity": "ΔAB zero crossing", "minute": stats["ab_zero_minute"], "utc": stats["ab_zero_utc"], "delta_apbp_arcsec": np.nan, "delta_ab_arcsec": stats["ab_zero_delta_ab"], "residual_arcsec": 0.0},
        {"quantity": "ΔA′B′ equals ΔAB intersection", "minute": stats["intersection_minute"], "utc": stats["intersection_utc"], "delta_apbp_arcsec": stats["intersection_delta_ap"], "delta_ab_arcsec": stats["intersection_delta_ab"], "residual_arcsec": stats["intersection_residual"]},
        {"quantity": "inferred geocentric-CA marker", "minute": stats["geo_marker_minute"], "utc": stats["geo_marker_utc"], "delta_apbp_arcsec": stats["geo_delta_ap"], "delta_ab_arcsec": stats["geo_delta_ab"], "residual_arcsec": float(stats["geo_delta_ap"]) - float(stats["geo_delta_ab"])},
    ]
    pd.DataFrame(result_rows).to_csv(RESULT_CSV, index=False, float_format="%.15f")

    plt.close("all")
    plt.rcParams.update({"font.family": "DejaVu Serif", "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED})
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.70, 0.30], left=0.055, right=0.985, top=0.900, bottom=0.110, hspace=0.205)
    ax = fig.add_subplot(gs[0, 0])
    tab_ax = fig.add_subplot(gs[1, 0])
    fig.suptitle("1769 Venus Transit — V0096 Bottom-Panel Intersection Audit", fontsize=15, fontweight="bold", y=0.963)
    fig.text(0.5, 0.928, "Uses V0096 one-minute CSV; solves ΔA′B′=0, ΔAB=0, and ΔA′B′=ΔAB by cubic interpolation/root finding.", ha="center", fontsize=7.4, color=MUTED)

    x = df["minute_from_start"].to_numpy(float)
    y_ap2 = df["delta_apbp_2x_arcsec"].to_numpy(float)
    y_ab2 = df["delta_ab_2x_arcsec"].to_numpy(float)
    y_dd2 = df["delta_delta_2x_arcsec"].to_numpy(float)
    ax.plot(x, y_ap2, color=GOLD, linewidth=0.58, label="2× ΔA′B′ = 2×(instant − fixed)")
    ax.plot(x, y_ab2, color=PURPLE, linewidth=0.58, label="2× ΔAB = 2×(instant − fixed)")
    ax.plot(x, y_dd2, color=GREEN, linewidth=0.48, linestyle="--", label="2×(ΔA′B′ − ΔAB)")
    ax.axhline(0.0, color=FG, linewidth=0.34, alpha=0.75)
    ax.axhline(2.0 * float(stats["mean_delta_ap"]), color=GOLD, linewidth=0.42, linestyle=":", alpha=0.82, label=f"2× mean ΔA′B′ = {2.0 * float(stats['mean_delta_ap']):+.12f}")
    ax.axhline(2.0 * float(stats["mean_delta_ab"]), color=PURPLE, linewidth=0.42, linestyle=":", alpha=0.82, label=f"2× mean ΔAB = {2.0 * float(stats['mean_delta_ab']):+.12f}")

    markers = [
        (float(stats["ap_zero_minute"]), 0.0, GOLD, "ΔA′B′ zero"),
        (float(stats["ab_zero_minute"]), 0.0, PURPLE, "ΔAB zero"),
        (float(stats["intersection_minute"]), 2.0 * float(stats["intersection_delta_ap"]), GREEN, "ΔA′B′=ΔAB"),
        (float(stats["geo_marker_minute"]), 2.0 * float(stats["geo_delta_ap"]), RED, "geo marker"),
    ]
    for mx, my, color, label in markers:
        ax.axvline(mx, color=color, linewidth=0.36, alpha=0.75)
        ax.scatter([mx], [my], s=60, marker="X", color=color, edgecolors=FG, linewidths=0.36, zorder=8)
        dy = 0.010 if label != "geo marker" else -0.014
        va = "bottom" if dy > 0 else "top"
        ax.annotate(f"{label}\nx={mx:.6f} min", xy=(mx, my), xytext=(mx + 8, my + dy), ha="left", va=va, fontsize=6.8, color=FG, arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.28})

    ax.set_xlabel("Elapsed time from first C1–C4 sample (minutes)", fontsize=9.2)
    ax.set_ylabel("2× delta / residual (arcsec)", fontsize=9.2)
    ax.grid(True, color=GRID, linewidth=0.30, alpha=0.58)
    ax.tick_params(labelsize=7.4, width=0.35, length=2.5)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=6.3)
    for text in leg.get_texts():
        text.set_color(FG)

    tab_ax.axis("off")
    rows = [
        ["Quantity", "Minute", "UTC", "ΔA′B′", "ΔAB", "Residual / trace"],
        ["ΔA′B′ = 0", f"{float(stats['ap_zero_minute']):.9f}", str(stats["ap_zero_utc"]), f"{float(stats['ap_zero_delta_ap']):+.12e}", "—", "arcsec root"],
        ["ΔAB = 0", f"{float(stats['ab_zero_minute']):.9f}", str(stats["ab_zero_utc"]), "—", f"{float(stats['ab_zero_delta_ab']):+.12e}", "arcsec root"],
        ["ΔA′B′ = ΔAB", f"{float(stats['intersection_minute']):.9f}", str(stats["intersection_utc"]), f"{float(stats['intersection_delta_ap']):+.12e}", f"{float(stats['intersection_delta_ab']):+.12e}", f"res={float(stats['intersection_residual']):+.3e}"],
        ["Geo marker", f"{float(stats['geo_marker_minute']):.9f}", str(stats["geo_marker_utc"]), f"{float(stats['geo_delta_ap']):+.12e}", f"{float(stats['geo_delta_ab']):+.12e}", f"geo−intersection={float(stats['geo_minus_intersection_min']):+.9f} min"],
        ["Mean Δ", "—", "—", f"{float(stats['mean_delta_ap']):+.12f}", f"{float(stats['mean_delta_ab']):+.12f}", "arcsec"],
        ["RMS Δ", "—", "—", f"{float(stats['rms_delta_ap']):.12f}", f"{float(stats['rms_delta_ab']):.12f}", "arcsec"],
        ["Samples", "—", "—", f"{int(stats['samples'])}", f"{int(stats['samples'])}", "V0096 CSV rows"],
    ]
    table = tab_ax.table(cellText=rows, cellLoc="left", colWidths=[0.18, 0.13, 0.23, 0.17, 0.17, 0.22], bbox=[0.0, 0.04, 1.0, 0.86])
    table_style(table, teal_rows=(1, 2, 3, 4), gold_rows=(5, 6, 7), fontsize=5.9)
    fig.text(0.5, 0.045, f"File: {Path(__file__).name if '__file__' in globals() else 'VENUS_1769_DELTA_INTERSECTION_AUDIT_V0098.py'} | Source CSV: {source} | Output: {PNG.name} | Result CSV: {RESULT_CSV.name}", ha="center", fontsize=5.6, color=MUTED)
    fig.savefig(PNG, dpi=220, facecolor=BG)
    display(Image(filename=str(PNG)))


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"Input CSV target: {TARGET_CSV}")
    print("Data source: V0096 one-minute JPL-derived CSV already generated in Colab")
    print("COMMENTS")
    print("Finds bottom-panel intersections using the exact V0096 CSV schema.")
    print("No AI images; Python/Matplotlib plot only.")
    df, stats, source = analyze()
    plot(df, stats, source)
    print("RESULTS")
    print(f"ΔA′B′ zero crossing minute: {float(stats['ap_zero_minute']):.12f}; UTC: {stats['ap_zero_utc']}")
    print(f"ΔAB zero crossing minute: {float(stats['ab_zero_minute']):.12f}; UTC: {stats['ab_zero_utc']}")
    print(f"ΔA′B′ = ΔAB intersection minute: {float(stats['intersection_minute']):.12f}; UTC: {stats['intersection_utc']}")
    print(f"Intersection ΔA′B′: {float(stats['intersection_delta_ap']):+.15e} arcsec")
    print(f"Intersection ΔAB: {float(stats['intersection_delta_ab']):+.15e} arcsec")
    print(f"Intersection residual ΔA′B′−ΔAB: {float(stats['intersection_residual']):+.15e} arcsec")
    print(f"Geo marker minute: {float(stats['geo_marker_minute']):.12f}; UTC: {stats['geo_marker_utc']}")
    print(f"Geo marker minus intersection: {float(stats['geo_minus_intersection_min']):+.12f} minutes")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"RESULT CSV: {RESULT_CSV}")
    print(f"SOURCE CSV: {source}")
    print("PAPER COMPARISON")
    print("NOT USED: this is an internal V0096 CSV interpolation/root audit only.")
    print("EQUATION STATUS")
    print("PASS: root finding uses cubic splines of V0096 one-minute JPL-derived delta columns; intersections are not assumed to equal geocentric CA.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0098
