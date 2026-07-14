# V0097
# Audit reference: standalone V0096-derived delta intersection audit; Python/Matplotlib only; no AI images.
from __future__ import annotations

import math
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

VERSION = "V0097"
LOCAL_TZ = ZoneInfo("America/Bogota")
SOURCE_NAME = "VENUS_1769_APRIME_BPRIME_AB_FOUR_CURVE_DELTA_V0096.csv"
OUT = Path("/content/VENUS_1769_DELTA_INTERSECTION_AUDIT_V0097_OUTPUT")
PNG = OUT / "VENUS_1769_DELTA_INTERSECTION_AUDIT_V0097.png"
CSV = OUT / "VENUS_1769_DELTA_INTERSECTION_AUDIT_V0097.csv"

BG = "#000000"
FG = "#F8FAFC"
MUTED = "#B8CBD6"
GRID = "#263A4B"
GOLD = "#D89B18"
BLUE = "#42D7C3"
PURPLE = "#A78BFA"
RED = "#FF6B6B"
GREEN = "#74D680"
TABLE_HEADER = "#23466F"
TABLE_BODY = "#101A2E"
TABLE_TEAL = "#164B55"
TABLE_GOLD = "#563B0B"
TABLE_RED = "#5A1010"


def require(import_name: str, package_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", package_name])


for _import, _package in (("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("matplotlib", "matplotlib"), ("IPython", "ipython")):
    require(_import, _package)

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, minimize_scalar


def find_source_csv() -> Path:
    direct = Path("/content/VENUS_1769_APRIME_BPRIME_AB_FOUR_CURVE_DELTA_V0096_OUTPUT") / SOURCE_NAME
    if direct.exists():
        return direct
    candidates = []
    for root in (Path("/content"), Path.cwd()):
        try:
            candidates.extend(root.rglob(SOURCE_NAME))
        except Exception:
            pass
    unique = []
    seen = set()
    for p in candidates:
        rp = str(p.resolve())
        if rp not in seen and p.exists():
            seen.add(rp)
            unique.append(p)
    if unique:
        unique.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return unique[0]
    raise FileNotFoundError(
        "Required V0096 CSV was not found. Run V0096 first, then run this audit. "
        f"Expected filename: {SOURCE_NAME}"
    )


def pick_column(df: pd.DataFrame, names: tuple[str, ...]) -> str:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    raise KeyError(f"None of these columns were found: {names}. Available columns: {list(df.columns)}")


def roots_for(x: np.ndarray, y: np.ndarray) -> list[float]:
    roots: list[float] = []
    f = PchipInterpolator(x, y, extrapolate=False)
    for i in range(len(x) - 1):
        y0 = float(y[i])
        y1 = float(y[i + 1])
        if not np.isfinite(y0) or not np.isfinite(y1):
            continue
        if abs(y0) < 1e-14:
            roots.append(float(x[i]))
        if y0 * y1 < 0.0:
            try:
                roots.append(float(brentq(lambda t: float(f(t)), float(x[i]), float(x[i + 1]), xtol=1e-12, rtol=1e-14)))
            except ValueError:
                pass
    if abs(float(y[-1])) < 1e-14:
        roots.append(float(x[-1]))
    clean: list[float] = []
    for r in sorted(roots):
        if not clean or abs(r - clean[-1]) > 1e-7:
            clean.append(r)
    return clean


def min_abs_point(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    f = PchipInterpolator(x, y, extrapolate=False)
    best_x = float(x[int(np.argmin(np.abs(y)))])
    lo = max(float(x[0]), best_x - 3.0)
    hi = min(float(x[-1]), best_x + 3.0)
    try:
        res = minimize_scalar(lambda t: abs(float(f(t))), bounds=(lo, hi), method="bounded", options={"xatol": 1e-11})
        if res.success:
            best_x = float(res.x)
    except Exception:
        pass
    return best_x, float(f(best_x))


def utc_at(df: pd.DataFrame, minute: float) -> str:
    if "utc" not in df.columns:
        return "N/A"
    x = df["minute_from_start"].to_numpy(float)
    i = int(np.argmin(np.abs(x - minute)))
    return str(df["utc"].iloc[i])


def table_style(table, teal_rows=(), gold_rows=(), red_rows=(), fontsize=6.4) -> None:
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
        elif row in red_rows:
            cell.set_facecolor(TABLE_RED)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(TABLE_BODY)


def analyze() -> tuple[pd.DataFrame, dict[str, object], Path]:
    source = find_source_csv()
    df = pd.read_csv(source)
    xcol = pick_column(df, ("minute_from_start", "minutes", "elapsed_min"))
    fixed_ap = pick_column(df, ("fixed_apbp_arcsec", "fixed_vector_apbp_arcsec"))
    instant_ap = pick_column(df, ("instant_apbp_arcsec", "instant_aprime_bprime_arcsec"))
    fixed_ab = pick_column(df, ("fixed_ab_arcsec", "fixed_vector_ab_arcsec"))
    instant_ab = pick_column(df, ("instant_ab_arcsec", "instantaneous_ab_arcsec"))

    use = df[[xcol, fixed_ap, instant_ap, fixed_ab, instant_ab] + (["utc"] if "utc" in df.columns else [])].copy()
    use = use.dropna().sort_values(xcol).reset_index(drop=True)
    use = use.rename(columns={xcol: "minute_from_start", fixed_ap: "fixed_apbp_arcsec", instant_ap: "instant_apbp_arcsec", fixed_ab: "fixed_ab_arcsec", instant_ab: "instant_ab_arcsec"})
    x = use["minute_from_start"].to_numpy(float)
    use["delta_apbp_arcsec"] = use["instant_apbp_arcsec"].to_numpy(float) - use["fixed_apbp_arcsec"].to_numpy(float)
    use["delta_ab_arcsec"] = use["instant_ab_arcsec"].to_numpy(float) - use["fixed_ab_arcsec"].to_numpy(float)
    use["delta_apbp_2x_arcsec"] = 2.0 * use["delta_apbp_arcsec"]
    use["delta_ab_2x_arcsec"] = 2.0 * use["delta_ab_arcsec"]
    use["delta_curve_difference_arcsec"] = use["delta_apbp_arcsec"] - use["delta_ab_arcsec"]

    dap = use["delta_apbp_arcsec"].to_numpy(float)
    dab = use["delta_ab_arcsec"].to_numpy(float)
    dd = dap - dab
    roots_ap = roots_for(x, dap)
    roots_ab = roots_for(x, dab)
    roots_intersect = roots_for(x, dd)
    near_ap = min_abs_point(x, dap)
    near_ab = min_abs_point(x, dab)
    near_intersect = min_abs_point(x, dd)

    f_ap = PchipInterpolator(x, dap, extrapolate=False)
    f_ab = PchipInterpolator(x, dab, extrapolate=False)

    stats: dict[str, object] = {
        "source": str(source),
        "mean_delta_apbp": float(np.mean(dap)),
        "mean_delta_ab": float(np.mean(dab)),
        "rms_delta_apbp": float(np.sqrt(np.mean(dap ** 2))),
        "rms_delta_ab": float(np.sqrt(np.mean(dab ** 2))),
        "roots_apbp": roots_ap,
        "roots_ab": roots_ab,
        "roots_intersect": roots_intersect,
        "near_apbp": near_ap,
        "near_ab": near_ab,
        "near_intersect": near_intersect,
        "f_ap": f_ap,
        "f_ab": f_ab,
    }
    return use, stats, source


def fmt_roots(roots: list[float]) -> str:
    if not roots:
        return "none"
    return "; ".join(f"{r:.9f}" for r in roots)


def plot(df: pd.DataFrame, stats: dict[str, object], source: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False, float_format="%.15f")
    plt.close("all")
    plt.rcParams.update({"font.family": "DejaVu Serif", "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED})
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.70, 0.30], left=0.055, right=0.985, top=0.895, bottom=0.100, hspace=0.205)
    ax = fig.add_subplot(gs[0, 0])
    tab_ax = fig.add_subplot(gs[1, 0])
    fig.suptitle("1769 Venus Transit — Bottom-Panel Delta Intersection Audit", fontsize=15, fontweight="bold", y=0.965)
    fig.text(0.5, 0.932, "V0096 CSV post-audit: actual curve intersections are solved by PCHIP interpolation and Brent roots; no AI images.", ha="center", fontsize=7.4, color=MUTED)

    x = df["minute_from_start"].to_numpy(float)
    dap2 = df["delta_apbp_2x_arcsec"].to_numpy(float)
    dab2 = df["delta_ab_2x_arcsec"].to_numpy(float)
    ax.axhline(0.0, color=FG, linewidth=0.35, alpha=0.75, label="zero")
    mean_ap2 = 2.0 * float(stats["mean_delta_apbp"])
    mean_ab2 = 2.0 * float(stats["mean_delta_ab"])
    ax.axhline(mean_ap2, color=GOLD, linewidth=0.42, linestyle="--", alpha=0.75, label=f"2× mean ΔA′B′ = {mean_ap2:+.12f}″")
    ax.axhline(mean_ab2, color=PURPLE, linewidth=0.42, linestyle="--", alpha=0.75, label=f"2× mean ΔAB = {mean_ab2:+.12f}″")
    ax.plot(x, dap2, color=GOLD, linewidth=0.72, label="2× ΔA′B′ = 2×(instant − fixed)")
    ax.plot(x, dab2, color=PURPLE, linewidth=0.72, label="2× ΔAB = 2×(instant − fixed)")

    f_ap = stats["f_ap"]
    f_ab = stats["f_ab"]
    for r in stats["roots_apbp"]:
        y = 2.0 * float(f_ap(r))
        ax.scatter([r], [y], s=42, marker="o", color=GOLD, edgecolors=FG, linewidths=0.35, zorder=7)
        ax.axvline(r, color=GOLD, linewidth=0.35, alpha=0.55)
        ax.annotate(f"ΔA′B′ zero\n{r:.6f} min", xy=(r, y), xytext=(r + 8, y + 0.010), fontsize=6.8, color=FG, arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.28})
    for r in stats["roots_ab"]:
        y = 2.0 * float(f_ab(r))
        ax.scatter([r], [y], s=42, marker="s", color=PURPLE, edgecolors=FG, linewidths=0.35, zorder=7)
        ax.axvline(r, color=PURPLE, linewidth=0.35, alpha=0.55)
        ax.annotate(f"ΔAB zero\n{r:.6f} min", xy=(r, y), xytext=(r - 8, y - 0.014), ha="right", fontsize=6.8, color=FG, arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.28})
    for r in stats["roots_intersect"]:
        y = 2.0 * float(f_ap(r))
        ax.scatter([r], [y], s=76, marker="X", color=RED, edgecolors=FG, linewidths=0.45, zorder=8)
        ax.axvline(r, color=RED, linewidth=0.42, alpha=0.70)
        ax.annotate(f"curve intersection\n{r:.6f} min\n2×Δ = {y:+.9f}″", xy=(r, y), xytext=(r + 12, y + 0.017), fontsize=7.0, color=FG, fontweight="bold", arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.32})

    ax.set_xlabel("Elapsed time from first C1–C4 sample (minutes)", fontsize=9.2)
    ax.set_ylabel("2× delta (arcsec)", fontsize=9.2)
    ax.grid(True, color=GRID, linewidth=0.32, alpha=0.55)
    ax.tick_params(labelsize=7.6, width=0.35, length=2.8)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.35)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=7.2)
    for txt in leg.get_texts():
        txt.set_color(FG)

    tab_ax.axis("off")
    inter = stats["roots_intersect"]
    ap_roots = stats["roots_apbp"]
    ab_roots = stats["roots_ab"]
    ni_x, ni_y = stats["near_intersect"]
    na_x, na_y = stats["near_apbp"]
    nb_x, nb_y = stats["near_ab"]
    rows = [
        ["Quantity", "Value", "UTC nearest sample", "Unit / trace"],
        ["ΔA′B′ zero crossing(s)", fmt_roots(ap_roots), utc_at(df, ap_roots[0]) if ap_roots else "none", "minutes from first sample; ΔA′B′ = 0"],
        ["ΔAB zero crossing(s)", fmt_roots(ab_roots), utc_at(df, ab_roots[0]) if ab_roots else "none", "minutes from first sample; ΔAB = 0"],
        ["ΔA′B′ = ΔAB intersection(s)", fmt_roots(inter), utc_at(df, inter[0]) if inter else "none", "minutes from first sample; bottom-curve intersection"],
        ["Nearest intersection residual", f"x={ni_x:.9f}; residual={ni_y:+.12e}", utc_at(df, ni_x), "arcsec; ΔA′B′ − ΔAB"],
        ["Nearest ΔA′B′ zero residual", f"x={na_x:.9f}; Δ={na_y:+.12e}", utc_at(df, na_x), "arcsec"],
        ["Nearest ΔAB zero residual", f"x={nb_x:.9f}; Δ={nb_y:+.12e}", utc_at(df, nb_x), "arcsec"],
        ["Mean ΔA′B′", f"{float(stats['mean_delta_apbp']):+.12f}", "all samples", "arcsec"],
        ["Mean ΔAB", f"{float(stats['mean_delta_ab']):+.12f}", "all samples", "arcsec"],
        ["Source CSV", source.name, "V0096", str(source.parent)],
    ]
    table = tab_ax.table(cellText=rows, cellLoc="left", colWidths=[0.24, 0.32, 0.20, 0.24], bbox=[0.0, 0.05, 1.0, 0.86])
    table_style(table, teal_rows=(1, 2, 3), gold_rows=(7, 8, 9), red_rows=(4,), fontsize=6.15)
    fig.text(0.5, 0.040, f"File: VENUS_1769_DELTA_INTERSECTION_AUDIT_V0097.py | Output: {PNG.name} | CSV: {CSV.name}", ha="center", fontsize=5.9, color=MUTED)
    fig.savefig(PNG, dpi=230, facecolor=BG)
    display(Image(filename=str(PNG)))


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"Input CSV target: {SOURCE_NAME}")
    print("Data source: V0096 one-minute JPL-derived CSV already generated in Colab")
    print("COMMENTS")
    print("Finds bottom-panel intersections: ΔA′B′=0, ΔAB=0, and ΔA′B′=ΔAB using interpolation/root finding.")
    print("No AI images; Python/Matplotlib plot only.")
    df, stats, source = analyze()
    plot(df, stats, source)
    print("RESULTS")
    print(f"ΔA′B′ zero crossings minutes: {fmt_roots(stats['roots_apbp'])}")
    print(f"ΔAB zero crossings minutes: {fmt_roots(stats['roots_ab'])}")
    print(f"ΔA′B′ = ΔAB intersection minutes: {fmt_roots(stats['roots_intersect'])}")
    ni_x, ni_y = stats["near_intersect"]
    print(f"Nearest curve-intersection residual: x={ni_x:.12f} min, ΔA′B′−ΔAB={ni_y:+.15e} arcsec")
    print(f"Mean ΔA′B′: {float(stats['mean_delta_apbp']):+.12f} arcsec")
    print(f"Mean ΔAB: {float(stats['mean_delta_ab']):+.12f} arcsec")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    print("PAPER COMPARISON")
    print("NOT USED: internal V0096-derived JPL-vector intersection audit only.")
    print("EQUATION STATUS")
    print("PASS: roots are solved from cubic monotone interpolation of V0096 one-minute calculated deltas; bottom-panel plotted quantities are the same deltas amplified by 2×.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0097
