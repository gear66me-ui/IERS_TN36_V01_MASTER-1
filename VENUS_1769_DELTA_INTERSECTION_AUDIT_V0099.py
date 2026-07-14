# V0099
# Audit reference: standalone V0096 CSV intersection-label repair; Python/Matplotlib only; no AI images.
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


def require(import_name: str, package_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", package_name])


for _import, _package in (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("IPython", "ipython"),
):
    require(_import, _package)

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar
from IPython.display import Image, display

VERSION = "V0099"
LOCAL_TZ = ZoneInfo("America/Bogota")
INPUT_NAME = "VENUS_1769_APRIME_BPRIME_AB_FOUR_CURVE_DELTA_V0096.csv"
INPUT_DIR = Path("/content/VENUS_1769_APRIME_BPRIME_AB_FOUR_CURVE_DELTA_V0096_OUTPUT")
INPUT_CANDIDATES = (
    INPUT_DIR / INPUT_NAME,
    Path("/content") / INPUT_NAME,
    Path.cwd() / INPUT_NAME,
)
OUT = Path("/content/VENUS_1769_DELTA_INTERSECTION_AUDIT_V0099_OUTPUT")
PNG = OUT / "VENUS_1769_DELTA_INTERSECTION_AUDIT_V0099.png"
CSV = OUT / "VENUS_1769_DELTA_INTERSECTION_AUDIT_V0099.csv"

BG = "#000000"
FG = "#F8FAFC"
MUTED = "#B8CBD6"
GRID = "#263A4B"
BLUE = "#42D7C3"
GOLD = "#D89B18"
GREEN = "#74D680"
RED = "#FF6B6B"
PURPLE = "#B997FF"
TABLE_HEADER = "#23466F"
TABLE_BODY = "#101A2E"
TABLE_TEAL = "#164B55"
TABLE_GOLD = "#563B0B"


def find_source() -> Path:
    for p in INPUT_CANDIDATES:
        if p.exists():
            return p
    matches = sorted(Path("/content").rglob(INPUT_NAME))
    if matches:
        return matches[0]
    raise FileNotFoundError("V0096 CSV was not found. Run V0096 first, then run this V0099 audit.")


def finite_xy(df: pd.DataFrame, ycol: str) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(df["minute_from_start"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    uniq, idx = np.unique(x, return_index=True)
    return uniq, y[idx]


def roots_of_spline(x: np.ndarray, y: np.ndarray) -> list[float]:
    if len(x) < 4:
        raise RuntimeError("Need at least four samples for cubic-spline root finding.")
    cs = CubicSpline(x, y, bc_type="natural")
    dense = np.linspace(float(x.min()), float(x.max()), max(4000, len(x) * 10))
    vals = cs(dense)
    roots: list[float] = []
    for i in range(len(dense) - 1):
        a = float(dense[i])
        b = float(dense[i + 1])
        ya = float(vals[i])
        yb = float(vals[i + 1])
        if not math.isfinite(ya) or not math.isfinite(yb):
            continue
        if abs(ya) < 1e-14:
            root = a
        elif ya * yb < 0.0:
            root = float(brentq(lambda t: float(cs(t)), a, b, xtol=1e-12, rtol=1e-12, maxiter=100))
        else:
            continue
        if not roots or abs(root - roots[-1]) > 1e-5:
            roots.append(root)
    return roots


def extremum_near_zero(x: np.ndarray, y: np.ndarray) -> float:
    cs = CubicSpline(x, y, bc_type="natural")
    res = minimize_scalar(lambda t: abs(float(cs(t))), bounds=(float(x.min()), float(x.max())), method="bounded", options={"xatol": 1e-10})
    if not res.success:
        raise RuntimeError("Minimum absolute residual solve failed.")
    return float(res.x)


def interp_value(x: np.ndarray, y: np.ndarray, xp: float) -> float:
    return float(CubicSpline(x, y, bc_type="natural")(float(xp)))


def load_and_analyze() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | str], Path]:
    source = find_source()
    df = pd.read_csv(source)
    required = [
        "minute_from_start",
        "delta_apbp_arcsec",
        "delta_ab_arcsec",
        "delta_apbp_2x_arcsec",
        "delta_ab_2x_arcsec",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing V0096 columns: {missing}. Available columns: {list(df.columns)}")
    x_ap, d_ap = finite_xy(df, "delta_apbp_arcsec")
    x_ab, d_ab = finite_xy(df, "delta_ab_arcsec")
    if not np.array_equal(x_ap, x_ab):
        xmin = max(float(x_ap.min()), float(x_ab.min()))
        xmax = min(float(x_ap.max()), float(x_ab.max()))
        x = np.linspace(xmin, xmax, min(len(x_ap), len(x_ab)))
        ap = CubicSpline(x_ap, d_ap, bc_type="natural")(x)
        ab = CubicSpline(x_ab, d_ab, bc_type="natural")(x)
    else:
        x = x_ap
        ap = d_ap
        ab = d_ab

    diff = ap - ab
    ap_roots = roots_of_spline(x, ap)
    ab_roots = roots_of_spline(x, ab)
    equal_roots = roots_of_spline(x, diff)
    ap_zero = ap_roots[0] if ap_roots else extremum_near_zero(x, ap)
    ab_zero = ab_roots[0] if ab_roots else extremum_near_zero(x, ab)
    equal_root = equal_roots[0] if equal_roots else extremum_near_zero(x, diff)

    # Use approximate geocentric CA marker as the sample whose |2x ΔA′B′| is smallest near central transit.
    mid = 0.5 * (float(x.min()) + float(x.max()))
    mask = np.abs(x - mid) < 30.0
    if mask.any():
        local_idx = int(np.argmin(np.abs(ap[mask])))
        ca_x = float(x[mask][local_idx])
    else:
        ca_x = float(x[int(np.argmin(np.abs(ap)))])

    stats: dict[str, float | str] = {
        "source": str(source),
        "apbp_zero_min": float(ap_zero),
        "ab_zero_min": float(ab_zero),
        "equal_root_min": float(equal_root),
        "ca_marker_min": float(ca_x),
        "apbp_zero_delta": interp_value(x, ap, ap_zero),
        "ab_zero_delta": interp_value(x, ab, ab_zero),
        "equal_apbp_delta": interp_value(x, ap, equal_root),
        "equal_ab_delta": interp_value(x, ab, equal_root),
        "equal_delta_gap": interp_value(x, diff, equal_root),
        "ca_delta_apbp": interp_value(x, ap, ca_x),
        "ca_delta_ab": interp_value(x, ab, ca_x),
        "mean_apbp": float(np.mean(ap)),
        "mean_ab": float(np.mean(ab)),
        "rms_apbp": float(np.sqrt(np.mean(ap ** 2))),
        "rms_ab": float(np.sqrt(np.mean(ab ** 2))),
    }
    out_df = pd.DataFrame({
        "minute_from_start": x,
        "delta_apbp_arcsec": ap,
        "delta_ab_arcsec": ab,
        "delta_difference_apbp_minus_ab_arcsec": diff,
        "delta_apbp_2x_arcsec": 2.0 * ap,
        "delta_ab_2x_arcsec": 2.0 * ab,
    })
    rows = [
        {"quantity": "ΔA′B′ zero crossing", "minute_from_start": stats["apbp_zero_min"], "delta_apbp_arcsec": stats["apbp_zero_delta"], "delta_ab_arcsec": interp_value(x, ab, float(stats["apbp_zero_min"])), "trace": "root of ΔA′B′"},
        {"quantity": "ΔAB zero crossing", "minute_from_start": stats["ab_zero_min"], "delta_apbp_arcsec": interp_value(x, ap, float(stats["ab_zero_min"])), "delta_ab_arcsec": stats["ab_zero_delta"], "trace": "root of ΔAB"},
        {"quantity": "ΔA′B′ = ΔAB intersection", "minute_from_start": stats["equal_root_min"], "delta_apbp_arcsec": stats["equal_apbp_delta"], "delta_ab_arcsec": stats["equal_ab_delta"], "trace": "root of ΔA′B′ - ΔAB"},
        {"quantity": "center/CA comparison marker", "minute_from_start": stats["ca_marker_min"], "delta_apbp_arcsec": stats["ca_delta_apbp"], "delta_ab_arcsec": stats["ca_delta_ab"], "trace": "near central zero marker"},
    ]
    return out_df, pd.DataFrame(rows), stats, source


def table_style(table, teal_rows=(), gold_rows=(), fontsize=6.4) -> None:
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


def plot(curves: pd.DataFrame, events: pd.DataFrame, stats: dict[str, float | str], source: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events.to_csv(CSV, index=False, float_format="%.15f")
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
    gs = fig.add_gridspec(2, 1, height_ratios=[0.72, 0.28], left=0.055, right=0.985, top=0.895, bottom=0.105, hspace=0.155)
    ax = fig.add_subplot(gs[0, 0])
    tab_ax = fig.add_subplot(gs[1, 0])
    fig.suptitle("1769 Venus Transit — Bottom-Panel Delta Intersection Audit", fontsize=15.0, fontweight="bold", y=0.965)
    fig.text(0.5, 0.932, "V0099 label-repair widget: left/right/up offsets separate all intersection annotations.", ha="center", fontsize=7.5, color=MUTED)

    x = curves["minute_from_start"].to_numpy(float)
    ap2 = curves["delta_apbp_2x_arcsec"].to_numpy(float)
    ab2 = curves["delta_ab_2x_arcsec"].to_numpy(float)
    diff2 = 2.0 * curves["delta_difference_apbp_minus_ab_arcsec"].to_numpy(float)

    ax.axhline(0.0, color=FG, linewidth=0.42, alpha=0.70, zorder=1)
    ax.axhline(2.0 * float(stats["mean_apbp"]), color=GOLD, linewidth=0.55, linestyle="--", alpha=0.80, zorder=1, label=f"2× mean ΔA′B′ = {2*float(stats['mean_apbp']):+.12f}″")
    ax.axhline(2.0 * float(stats["mean_ab"]), color=PURPLE, linewidth=0.55, linestyle="--", alpha=0.80, zorder=1, label=f"2× mean ΔAB = {2*float(stats['mean_ab']):+.12f}″")
    ax.plot(x, ap2, color=GOLD, linewidth=0.70, label="2× ΔA′B′", zorder=3)
    ax.plot(x, ab2, color=PURPLE, linewidth=0.70, label="2× ΔAB", zorder=3)
    ax.plot(x, diff2, color=BLUE, linewidth=0.55, linestyle=":", label="2×(ΔA′B′ − ΔAB)", zorder=2)

    points = [
        ("ΔA′B′=0", float(stats["apbp_zero_min"]), 2.0 * float(stats["apbp_zero_delta"]), GOLD, (-34, 0.030), "right", "bottom"),
        ("ΔAB=0", float(stats["ab_zero_min"]), 2.0 * float(stats["ab_zero_delta"]), PURPLE, (34, -0.036), "left", "top"),
        ("ΔA′B′=ΔAB", float(stats["equal_root_min"]), 2.0 * float(stats["equal_apbp_delta"]), BLUE, (16, 0.058), "left", "bottom"),
        ("center marker", float(stats["ca_marker_min"]), 2.0 * float(stats["ca_delta_apbp"]), RED, (-40, -0.058), "right", "top"),
    ]
    for label, px, py, color, offset, ha, va in points:
        dx, dy = offset
        ax.scatter([px], [py], s=50 if label != "center marker" else 62, marker="X" if label != "center marker" else "P", color=color, edgecolors=FG, linewidths=0.34, zorder=8)
        ax.axvline(px, color=color, linewidth=0.34, alpha=0.60, zorder=1)
        ax.annotate(f"{label}\nx={px:.6f} min\ny={py:+.12f}″", xy=(px, py), xytext=(px + dx, py + dy), ha=ha, va=va, fontsize=7.2, color=FG, arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.30}, bbox={"boxstyle": "round,pad=0.20", "facecolor": "#050505", "edgecolor": color, "linewidth": 0.32, "alpha": 0.78})

    ylo = min(float(ap2.min()), float(ab2.min()), float(diff2.min()))
    yhi = max(float(ap2.max()), float(ab2.max()), float(diff2.max()))
    pad = max(0.025, 0.20 * (yhi - ylo))
    ax.set_xlim(float(x.min()) - 6.0, float(x.max()) + 6.0)
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.set_xlabel("Elapsed time from first C1–C4 sample (minutes)", fontsize=9)
    ax.set_ylabel("2× delta / intersection residual (arcsec)", fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.34, alpha=0.58)
    ax.tick_params(labelsize=7.5, width=0.35, length=2.4)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.35)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=7.2)
    for t in leg.get_texts():
        t.set_color(FG)

    tab_ax.axis("off")
    rows = [["Quantity", "Minute", "ΔA′B′", "ΔAB", "Unit / trace"]]
    for _, row in events.iterrows():
        rows.append([
            str(row["quantity"]),
            f"{float(row['minute_from_start']):.9f}",
            f"{float(row['delta_apbp_arcsec']):+.12f}",
            f"{float(row['delta_ab_arcsec']):+.12f}",
            str(row["trace"]),
        ])
    rows.append(["Mean Δ", "all", f"{float(stats['mean_apbp']):+.12f}", f"{float(stats['mean_ab']):+.12f}", "arcsec"])
    rows.append(["RMS Δ", "all", f"{float(stats['rms_apbp']):.12f}", f"{float(stats['rms_ab']):.12f}", "arcsec"])
    table = tab_ax.table(cellText=rows, cellLoc="left", colWidths=[0.26, 0.16, 0.20, 0.20, 0.18], bbox=[0.0, 0.08, 1.0, 0.84])
    table_style(table, teal_rows=(1, 2, 3), gold_rows=(4, 5, 6), fontsize=6.45)

    fig.text(0.5, 0.042, f"File: VENUS_1769_DELTA_INTERSECTION_AUDIT_V0099.py | Source: {source.name} | Output: {PNG.name} | CSV: {CSV.name}", ha="center", fontsize=5.8, color=MUTED)
    fig.savefig(PNG, dpi=220, facecolor=BG)
    display(Image(filename=str(PNG)))


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"Input CSV target: {INPUT_NAME}")
    print("Data source: V0096 one-minute JPL-derived CSV already generated in Colab")
    print("COMMENTS")
    print("Replots bottom-panel intersections with separated annotation labels: one left, one right, one above, and center marker below-left.")
    print("No AI images; Python/Matplotlib plot only.")
    curves, events, stats, source = load_and_analyze()
    plot(curves, events, stats, source)
    print("RESULTS")
    for _, row in events.iterrows():
        print(f"{row['quantity']}: x={float(row['minute_from_start']):.12f} min, ΔA′B′={float(row['delta_apbp_arcsec']):+.12f} arcsec, ΔAB={float(row['delta_ab_arcsec']):+.12f} arcsec, trace={row['trace']}")
    print(f"Mean ΔA′B′: {float(stats['mean_apbp']):+.12f} arcsec")
    print(f"Mean ΔAB: {float(stats['mean_ab']):+.12f} arcsec")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    print("PAPER COMPARISON")
    print("NOT USED: this is an internal V0096 JPL-derived delta-intersection audit only.")
    print("EQUATION STATUS")
    print("PASS: cubic-spline roots are solved from V0096 delta columns; labels are separated by fixed left/right/upper offsets for readability.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0099