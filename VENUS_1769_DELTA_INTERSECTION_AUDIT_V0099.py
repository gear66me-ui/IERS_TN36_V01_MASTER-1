# V0099
# Audit reference: corrected V0096 delta-intersection audit; readable label offsets; absolute-magnitude delta difference; Matplotlib only; no AI images.
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
from IPython.display import Image, display
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

VERSION = "V0099"
LOCAL_TZ = ZoneInfo("America/Bogota")
INPUT_NAME = "VENUS_1769_APRIME_BPRIME_AB_FOUR_CURVE_DELTA_V0096.csv"
CONTACT_NAME = "VENUS_1769_APRIME_BPRIME_AB_FOUR_CURVE_DELTA_CONTACTS_V0096.csv"
SEARCH_DIRS = [Path("/content"), Path("/content/VENUS_1769_APRIME_BPRIME_AB_FOUR_CURVE_DELTA_V0096_OUTPUT"), Path.cwd()]
OUT = Path("/content/VENUS_1769_DELTA_INTERSECTION_AUDIT_V0099_OUTPUT")
PNG = OUT / "VENUS_1769_DELTA_INTERSECTION_AUDIT_V0099.png"
CSV = OUT / "VENUS_1769_DELTA_INTERSECTION_AUDIT_V0099.csv"
SUMMARY_CSV = OUT / "VENUS_1769_DELTA_INTERSECTION_SUMMARY_V0099.csv"

BG = "#000000"
FG = "#F8FAFC"
MUTED = "#B8CBD6"
GRID = "#263A4B"
BLUE = "#42D7C3"
GOLD = "#D89B18"
PURPLE = "#9B8CFF"
RED = "#FF6B6B"
GREEN = "#74D680"
WHITE = "#FFFFFF"
TABLE_HEADER = "#23466F"
TABLE_BODY = "#101A2E"
TABLE_TEAL = "#164B55"
TABLE_GOLD = "#563B0B"
TABLE_RED = "#5A1515"


def find_file(name: str) -> Path:
    for root in SEARCH_DIRS:
        candidate = root / name
        if candidate.exists():
            return candidate
    for root in SEARCH_DIRS:
        if root.exists():
            hits = sorted(root.rglob(name))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"Could not find {name}. Run V0096 first in this Colab session.")


def pick(df: pd.DataFrame, names: tuple[str, ...]) -> str:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    raise KeyError(f"Missing columns {names}. Available columns: {list(df.columns)}")


def utc_from_minutes(source_df: pd.DataFrame, minute: float) -> str:
    if "utc" not in source_df.columns:
        return "N/A"
    x = source_df["minute_from_start"].to_numpy(float)
    utc_values = source_df["utc"].astype(str).to_numpy()
    idx = int(np.argmin(np.abs(x - float(minute))))
    return str(utc_values[idx])


def make_spline(x: np.ndarray, y: np.ndarray) -> CubicSpline:
    order = np.argsort(x)
    xs = np.asarray(x[order], dtype=float)
    ys = np.asarray(y[order], dtype=float)
    unique_x, unique_idx = np.unique(xs, return_index=True)
    unique_y = ys[unique_idx]
    if len(unique_x) < 4:
        raise RuntimeError("Need at least four unique samples for cubic interpolation.")
    return CubicSpline(unique_x, unique_y, bc_type="natural")


def roots_for(x: np.ndarray, y: np.ndarray) -> list[float]:
    spline = make_spline(x, y)
    xs = np.asarray(x, dtype=float)
    roots: list[float] = []
    for i in range(len(xs) - 1):
        a = float(xs[i])
        b = float(xs[i + 1])
        fa = float(spline(a))
        fb = float(spline(b))
        if abs(fa) < 1e-14:
            roots.append(a)
        elif fa * fb < 0.0:
            roots.append(float(brentq(lambda z: float(spline(z)), a, b, xtol=1e-11, rtol=1e-13)))
    clean: list[float] = []
    for r in sorted(roots):
        if not clean or abs(r - clean[-1]) > 1e-6:
            clean.append(r)
    return clean


def y_at(x: np.ndarray, y: np.ndarray, minute: float) -> float:
    return float(make_spline(x, y)(float(minute)))


def table_style(table, teal_rows=(), gold_rows=(), red_rows=(), fontsize=5.55) -> None:
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


def contact_minutes(contact_path: Path | None, source_df: pd.DataFrame) -> dict[str, float]:
    if contact_path is None or not contact_path.exists():
        return {}
    cdf = pd.read_csv(contact_path)
    if "event" not in cdf.columns or "jd_tdb" not in cdf.columns or "jd_tdb" not in source_df.columns:
        return {}
    start_jd = float(source_df["jd_tdb"].iloc[0]) if "jd_tdb" in source_df.columns else 0.0
    out: dict[str, float] = {}
    for event in ("C1", "C2", "CA", "C3", "C4"):
        vals = pd.to_numeric(cdf.loc[cdf["event"].astype(str) == event, "jd_tdb"], errors="coerce").dropna().to_numpy(float)
        if len(vals) and start_jd != 0.0:
            out[event] = float((np.mean(vals) - start_jd) * 1440.0)
    return out


def analyze() -> tuple[pd.DataFrame, list[dict[str, object]], Path, Path | None, dict[str, float]]:
    source = find_file(INPUT_NAME)
    try:
        contact = find_file(CONTACT_NAME)
    except FileNotFoundError:
        contact = None
    df = pd.read_csv(source)
    xcol = pick(df, ("minute_from_start",))
    dap_col = pick(df, ("delta_apbp_arcsec", "apbp_delta_arcsec"))
    dab_col = pick(df, ("delta_ab_arcsec", "ab_delta_arcsec"))
    fixed_ap_col = pick(df, ("apbp_fixed_arcsec", "fixed_apbp_arcsec", "fixed_vector_apbp_arcsec"))
    inst_ap_col = pick(df, ("apbp_instant_arcsec", "instant_apbp_arcsec"))
    fixed_ab_col = pick(df, ("ab_fixed_arcsec", "fixed_ab_arcsec"))
    inst_ab_col = pick(df, ("ab_instant_arcsec", "instant_ab_arcsec"))
    base = pd.DataFrame({
        "minute_from_start": pd.to_numeric(df[xcol], errors="coerce"),
        "delta_apbp_arcsec": pd.to_numeric(df[dap_col], errors="coerce"),
        "delta_ab_arcsec": pd.to_numeric(df[dab_col], errors="coerce"),
        "apbp_fixed_arcsec": pd.to_numeric(df[fixed_ap_col], errors="coerce"),
        "apbp_instant_arcsec": pd.to_numeric(df[inst_ap_col], errors="coerce"),
        "ab_fixed_arcsec": pd.to_numeric(df[fixed_ab_col], errors="coerce"),
        "ab_instant_arcsec": pd.to_numeric(df[inst_ab_col], errors="coerce"),
    })
    if "utc" in df.columns:
        base["utc"] = df["utc"].astype(str)
    if "jd_tdb" in df.columns:
        base["jd_tdb"] = pd.to_numeric(df["jd_tdb"], errors="coerce")
    work = base.dropna(subset=["minute_from_start", "delta_apbp_arcsec", "delta_ab_arcsec"]).sort_values("minute_from_start").reset_index(drop=True)
    if "utc" not in work.columns:
        work["utc"] = "N/A"
    work["delta_apbp_2x_arcsec"] = 2.0 * work["delta_apbp_arcsec"]
    work["delta_ab_2x_arcsec"] = 2.0 * work["delta_ab_arcsec"]
    work["abs_delta_difference_arcsec"] = np.abs(work["delta_apbp_arcsec"]) - np.abs(work["delta_ab_arcsec"])
    work["abs_delta_difference_2x_arcsec"] = 2.0 * work["abs_delta_difference_arcsec"]
    work["signed_delta_difference_2x_rejected"] = 2.0 * (work["delta_apbp_arcsec"] - work["delta_ab_arcsec"])

    x = work["minute_from_start"].to_numpy(float)
    dap = work["delta_apbp_arcsec"].to_numpy(float)
    dab = work["delta_ab_arcsec"].to_numpy(float)
    diff_abs = work["abs_delta_difference_arcsec"].to_numpy(float)
    results: list[dict[str, object]] = []
    for label, y, trace in (("ΔA′B′ zero", dap, "ΔA′B′ = 0"), ("ΔAB zero", dab, "ΔAB = 0"), ("Magnitude-difference zero", diff_abs, "|ΔA′B′| − |ΔAB| = 0")):
        for n, minute in enumerate(roots_for(x, y), start=1):
            results.append({"event": label if n == 1 else f"{label} #{n}", "minute_from_start": minute, "utc_nearest": utc_from_minutes(work, minute), "delta_apbp_arcsec": y_at(x, dap, minute), "delta_ab_arcsec": y_at(x, dab, minute), "abs_delta_difference_arcsec": y_at(x, diff_abs, minute), "trace": trace})
    stats = {"mean_delta_apbp": float(np.mean(dap)), "mean_delta_ab": float(np.mean(dab)), "mean_abs_delta_difference": float(np.mean(diff_abs)), "rms_delta_apbp": float(np.sqrt(np.mean(dap ** 2))), "rms_delta_ab": float(np.sqrt(np.mean(dab ** 2))), "rms_abs_delta_difference": float(np.sqrt(np.mean(diff_abs ** 2)))}
    return work, results, source, contact, stats


def annotate_label(ax, row: dict[str, object], y_value: float, xytext: tuple[float, float], align: tuple[str, str], color: str) -> None:
    minute = float(row["minute_from_start"])
    label = str(row["event"])
    text = f"{label}\nx={minute:.6f} min\nΔA′B′={float(row['delta_apbp_arcsec']):+.9f}″\nΔAB={float(row['delta_ab_arcsec']):+.9f}″"
    ax.scatter([minute], [y_value], s=54, marker="X", color=color, edgecolors=WHITE, linewidths=0.42, zorder=8)
    ax.annotate(text, xy=(minute, y_value), xytext=xytext, textcoords="offset points", ha=align[0], va=align[1], fontsize=7.0, color=FG, arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.36}, bbox={"boxstyle": "round,pad=0.22", "facecolor": "#050505", "edgecolor": color, "linewidth": 0.35, "alpha": 0.82}, zorder=9)


def plot(work: pd.DataFrame, results: list[dict[str, object]], source: Path, contact: Path | None, stats: dict[str, float]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    work.to_csv(CSV, index=False, float_format="%.15f")
    pd.DataFrame(results).to_csv(SUMMARY_CSV, index=False, float_format="%.15f")
    x = work["minute_from_start"].to_numpy(float)
    dap2 = work["delta_apbp_2x_arcsec"].to_numpy(float)
    dab2 = work["delta_ab_2x_arcsec"].to_numpy(float)
    diff2 = work["abs_delta_difference_2x_arcsec"].to_numpy(float)
    plt.close("all")
    plt.rcParams.update({"font.family": "DejaVu Serif", "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED})
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.74, 0.26], left=0.055, right=0.985, top=0.895, bottom=0.115, hspace=0.145)
    ax = fig.add_subplot(gs[0, 0])
    tab_ax = fig.add_subplot(gs[1, 0])
    fig.suptitle("1769 Venus Transit — Delta Intersection Audit V0099", fontsize=15.5, fontweight="bold", y=0.965)
    fig.text(0.5, 0.933, "Corrected third curve: 2×(|ΔA′B′| − |ΔAB|). Labels separated left/right/upper for readable intersection values.", ha="center", fontsize=7.6, color=MUTED)
    ax.axhline(0.0, color=WHITE, linewidth=0.42, alpha=0.72, zorder=1, label="zero reference")
    ax.plot(x, dap2, color=GOLD, linewidth=0.78, label="2× ΔA′B′ = 2×(instant − fixed)", zorder=3)
    ax.plot(x, dab2, color=PURPLE, linewidth=0.78, label="2× ΔAB = 2×(instant − fixed)", zorder=3)
    ax.plot(x, diff2, color=GREEN, linewidth=0.82, linestyle="--", label="2×(|ΔA′B′| − |ΔAB|), corrected", zorder=4)
    ax.axhline(2.0 * stats["mean_delta_apbp"], color=GOLD, linewidth=0.45, linestyle=":", alpha=0.82, label=f"2× mean ΔA′B′ = {2.0 * stats['mean_delta_apbp']:+.9f}″")
    ax.axhline(2.0 * stats["mean_delta_ab"], color=PURPLE, linewidth=0.45, linestyle=":", alpha=0.82, label=f"2× mean ΔAB = {2.0 * stats['mean_delta_ab']:+.9f}″")
    ax.axhline(2.0 * stats["mean_abs_delta_difference"], color=GREEN, linewidth=0.45, linestyle=":", alpha=0.82, label=f"2× mean |Δ|-|Δ| = {2.0 * stats['mean_abs_delta_difference']:+.9f}″")
    contacts = contact_minutes(contact, work) if contact is not None else {}
    for event, minute in contacts.items():
        color = RED if event == "CA" else BLUE
        ax.axvline(minute, color=color, linewidth=0.30, alpha=0.55, zorder=1)
        ymark = float(np.interp(minute, x, dap2))
        ax.scatter([minute], [ymark], s=16 if event != "CA" else 30, color=color, edgecolors=WHITE, linewidths=0.20, zorder=6)
        ax.annotate(event, xy=(minute, ymark), xytext=(0, 8 if event in ("C1", "C2", "CA") else -10), textcoords="offset points", ha="center", va="bottom" if event in ("C1", "C2", "CA") else "top", fontsize=6.2, color=FG)
    first_by_event: dict[str, dict[str, object]] = {}
    for row in results:
        first_by_event.setdefault(str(row["event"]).split(" #", 1)[0], row)
    if "ΔA′B′ zero" in first_by_event:
        row = first_by_event["ΔA′B′ zero"]
        annotate_label(ax, row, 2.0 * float(row["delta_apbp_arcsec"]), (-78, 60), ("right", "bottom"), GOLD)
    if "ΔAB zero" in first_by_event:
        row = first_by_event["ΔAB zero"]
        annotate_label(ax, row, 2.0 * float(row["delta_ab_arcsec"]), (72, -56), ("left", "top"), PURPLE)
    if "Magnitude-difference zero" in first_by_event:
        row = first_by_event["Magnitude-difference zero"]
        annotate_label(ax, row, 2.0 * float(row["abs_delta_difference_arcsec"]), (40, 62), ("left", "bottom"), GREEN)
    ax.set_xlim(float(x.min()) - 5.0, float(x.max()) + 5.0)
    ymin = min(float(np.min(dap2)), float(np.min(dab2)), float(np.min(diff2)))
    ymax = max(float(np.max(dap2)), float(np.max(dab2)), float(np.max(diff2)))
    pad = max(0.018, 0.18 * (ymax - ymin))
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_ylabel("2× delta / corrected magnitude difference (arcsec)", fontsize=9.0)
    ax.set_xlabel("Elapsed time from first C1–C4 sample (minutes)", fontsize=9.0)
    ax.grid(True, color=GRID, linewidth=0.34, alpha=0.56)
    ax.tick_params(labelsize=7.2, width=0.35, length=2.5)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.35)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=6.6, ncol=2)
    for text in leg.get_texts():
        text.set_color(FG)
    tab_ax.axis("off")
    rows = [["Intersection / diagnostic", "Minute", "Nearest UTC", "ΔA′B′", "ΔAB", "|ΔA′B′|-|ΔAB|", "Trace"]]
    for row in results[:6]:
        rows.append([str(row["event"]), f"{float(row['minute_from_start']):.9f}", str(row["utc_nearest"]), f"{float(row['delta_apbp_arcsec']):+.12f}", f"{float(row['delta_ab_arcsec']):+.12f}", f"{float(row['abs_delta_difference_arcsec']):+.12f}", str(row["trace"])])
    rows.extend([["Mean Δ", "—", "—", f"{stats['mean_delta_apbp']:+.12f}", f"{stats['mean_delta_ab']:+.12f}", f"{stats['mean_abs_delta_difference']:+.12f}", "arcsec"], ["RMS", "—", "—", f"{stats['rms_delta_apbp']:.12f}", f"{stats['rms_delta_ab']:.12f}", f"{stats['rms_abs_delta_difference']:.12f}", "arcsec"], ["Corrected formula", "—", "—", "2×ΔA′B′", "2×ΔAB", "2×(|ΔA′B′|-|ΔAB|)", "absolute magnitudes, not signed addition"]])
    table = tab_ax.table(cellText=rows, cellLoc="left", colWidths=[0.20, 0.11, 0.17, 0.13, 0.13, 0.16, 0.10], bbox=[0.0, 0.02, 1.0, 0.92])
    table_style(table, teal_rows=(1, 2, 3), gold_rows=(len(rows)-3, len(rows)-2), red_rows=(len(rows)-1,), fontsize=5.55)
    fig.text(0.5, 0.043, f"File: VENUS_1769_DELTA_INTERSECTION_AUDIT_V0099.py | Input: {source.name} | Output: {PNG.name} | CSV: {CSV.name}", ha="center", fontsize=5.7, color=MUTED)
    fig.savefig(PNG, dpi=230, facecolor=BG)
    display(Image(filename=str(PNG)))


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"Input CSV target: {INPUT_NAME}")
    print("Data source: V0096 one-minute JPL-derived CSV already generated in Colab")
    print("COMMENTS")
    print("Finds readable bottom-panel intersections and corrects the sign issue using |ΔA′B′| − |ΔAB|.")
    print("No AI images; Python/Matplotlib plot only.")
    work, results, source, contact, stats = analyze()
    plot(work, results, source, contact, stats)
    print("RESULTS")
    for row in results:
        print(f"{row['event']}: minute={float(row['minute_from_start']):.12f}; ΔA′B′={float(row['delta_apbp_arcsec']):+.12f} arcsec; ΔAB={float(row['delta_ab_arcsec']):+.12f} arcsec; |ΔA′B′|-|ΔAB|={float(row['abs_delta_difference_arcsec']):+.12f} arcsec; nearest UTC={row['utc_nearest']}")
    print(f"Mean ΔA′B′: {stats['mean_delta_apbp']:+.12f} arcsec")
    print(f"Mean ΔAB: {stats['mean_delta_ab']:+.12f} arcsec")
    print(f"Mean |ΔA′B′|-|ΔAB|: {stats['mean_abs_delta_difference']:+.12f} arcsec")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    print(f"SUMMARY CSV: {SUMMARY_CSV}")
    print("PAPER COMPARISON")
    print("NOT USED: this is an internal JPL-vector V0096 delta-intersection audit only.")
    print("EQUATION STATUS")
    print("PASS: ΔA′B′ and ΔAB are read from V0096; corrected comparison uses absolute-magnitude difference |ΔA′B′| − |ΔAB|, avoiding signed addition of opposite-signed deltas.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0099