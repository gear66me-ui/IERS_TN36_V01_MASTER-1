# V0101
# Audit reference: standalone closest-approach explanation widget; Python/Matplotlib/JPL only; no AI images.
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
OUT = Path("/content/VENUS_1769_CLOSEST_APPROACH_EXPLANATION_V0101_OUTPUT")
PNG = OUT / "VENUS_1769_CLOSEST_APPROACH_EXPLANATION_V0101.png"
CSV = OUT / "VENUS_1769_CLOSEST_APPROACH_EXPLANATION_V0101.csv"

ARC = 206_264.80624709636
AU_KM = 149_597_870.700000
START = "1769-06-03 18:00"
STOP = "1769-06-04 06:00"
STEP = "1m"
WINDOW_MIN = 30.0
PLOT_STEP_MIN = 0.5

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
TABLE_RED = "#651124"

POINT_VENUS = {"key": "POINT_VENUS", "label": "Point Venus, Tahiti", "lat": -17.4956, "lon": -149.4939, "elevation": 0.0, "body": 399}
VARDO = {"key": "VARDO", "label": "Vardø, Norway", "lat": 70.3724, "lon": 31.1103, "elevation": 0.0, "body": 399}


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
from scipy.interpolate import CubicSpline, interp1d
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


def utc(jd: float) -> str:
    return Time(float(jd), format="jd", scale="tdb").utc.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def loc(site: dict) -> dict:
    return {"lon": float(site["lon"]), "lat": float(site["lat"]), "elevation": float(site["elevation"]), "body": int(site["body"])}


def download(prefix: str, target_id: str, location) -> pd.DataFrame:
    last = None
    for _attempt in range(4):
        try:
            tab = Horizons(id=target_id, location=location, epochs={"start": START, "stop": STOP, "step": STEP}, id_type=None).vectors(refplane="ecliptic", aberrations="geometric", cache=False)
            raw = tab.to_pandas()
            df = pd.DataFrame({"JD_TDB": pd.to_numeric(raw["datetime_jd"], errors="coerce")})
            for ax in "xyz":
                df[f"{prefix}_{ax.upper()}_KM"] = pd.to_numeric(raw[ax], errors="coerce") * AU_KM
            return df.dropna().drop_duplicates("JD_TDB").sort_values("JD_TDB").reset_index(drop=True)
        except Exception as exc:
            last = exc
    raise RuntimeError(f"JPL Horizons download failed for {prefix}: {last}")


def build_master() -> pd.DataFrame:
    frames = [
        download("GEOCENTER_SUN", "10", "500@399"),
        download("GEOCENTER_VENUS", "299", "500@399"),
        download("POINT_VENUS_SUN", "10", loc(POINT_VENUS)),
        download("POINT_VENUS_VENUS", "299", loc(POINT_VENUS)),
        download("VARDO_SUN", "10", loc(VARDO)),
        download("VARDO_VENUS", "299", loc(VARDO)),
    ]
    master = frames[0]
    for frame in frames[1:]:
        master = master.merge(frame, on="JD_TDB", how="inner")
    return master.sort_values("JD_TDB").reset_index(drop=True)


def splines(df: pd.DataFrame) -> dict[str, object]:
    jd = df["JD_TDB"].to_numpy(float)
    c: dict[str, object] = {"JD_TDB": jd}
    for col in df.columns:
        if col != "JD_TDB":
            c[col] = CubicSpline(jd, df[col].to_numpy(float), bc_type="natural")
    return c


def vec(c: dict[str, object], prefix: str, jd: float) -> np.ndarray:
    return np.array([float(c[f"{prefix}_{ax}_KM"](jd)) for ax in "XYZ"], dtype=float)


def angle_arcsec(a: np.ndarray, b: np.ndarray) -> float:
    ua = unit(a)
    ub = unit(b)
    dot = max(-1.0, min(1.0, float(np.dot(ua, ub))))
    return ARC * math.acos(dot)


def geocentric_rho(c: dict[str, object], jd: float) -> float:
    return angle_arcsec(vec(c, "GEOCENTER_SUN", jd), vec(c, "GEOCENTER_VENUS", jd))


def geocentric_ca(c: dict[str, object]) -> float:
    jd = np.asarray(c["JD_TDB"], dtype=float)
    vals = np.array([geocentric_rho(c, float(x)) for x in jd], dtype=float)
    i = int(np.argmin(vals))
    lo = float(jd[max(0, i - 2)])
    hi = float(jd[min(len(jd) - 1, i + 2)])
    res = minimize_scalar(lambda t: geocentric_rho(c, float(t)), bounds=(lo, hi), method="bounded", options={"xatol": 1e-12, "maxiter": 500})
    if not res.success:
        raise RuntimeError("Geocentric closest approach solve failed.")
    return float(res.x)


def basis_from_sun(sun_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = unit(sun_vec)
    pole = np.array([0.0, 0.0, 1.0])
    east = np.cross(pole, center)
    if norm(east) < 1e-14:
        east = np.cross(np.array([0.0, 1.0, 0.0]), center)
    east = unit(east)
    north = unit(np.cross(center, east))
    if float(np.dot(north, pole)) < 0.0:
        east = -east
        north = -north
    return center, east, north


def gnomonic(ray: np.ndarray, center: np.ndarray, east: np.ndarray, north: np.ndarray) -> np.ndarray:
    h = unit(ray)
    den = float(np.dot(h, center))
    if den <= 0.0:
        raise RuntimeError("Ray outside tangent hemisphere.")
    return np.array([float(np.dot(h, east)), float(np.dot(h, north))]) / den


def relative_xy_fixed(c: dict[str, object], site_key: str, jd: float, basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    center, east, north = basis
    sun = vec(c, f"{site_key}_SUN", jd)
    ven = vec(c, f"{site_key}_VENUS", jd)
    return ARC * (gnomonic(ven, center, east, north) - gnomonic(sun, center, east, north))


def relative_xy_instant(c: dict[str, object], site_key: str, jd: float) -> np.ndarray:
    return relative_xy_fixed(c, site_key, jd, basis_from_sun(vec(c, "GEOCENTER_SUN", jd)))


def fixed_geometry(c: dict[str, object], ca_jd: float) -> dict[str, object]:
    basis = basis_from_sun(vec(c, "GEOCENTER_SUN", ca_jd))
    h = 0.5 / 86400.0
    pv_plus = relative_xy_fixed(c, "POINT_VENUS", ca_jd + h, basis)
    pv_minus = relative_xy_fixed(c, "POINT_VENUS", ca_jd - h, basis)
    va_plus = relative_xy_fixed(c, "VARDO", ca_jd + h, basis)
    va_minus = relative_xy_fixed(c, "VARDO", ca_jd - h, basis)
    tangent = unit(unit(pv_plus - pv_minus) + unit(va_plus - va_minus))
    normal = np.array([-tangent[1], tangent[0]])
    pv0 = relative_xy_fixed(c, "POINT_VENUS", ca_jd, basis)
    va0 = relative_xy_fixed(c, "VARDO", ca_jd, basis)
    if float(np.dot(va0 - pv0, normal)) < 0.0:
        normal = -normal
    return {"basis": basis, "tangent": tangent, "normal": normal}


def fixed_apbp(c: dict[str, object], jd: float, g: dict[str, object]) -> float:
    pv = relative_xy_fixed(c, "POINT_VENUS", jd, g["basis"])
    va = relative_xy_fixed(c, "VARDO", jd, g["basis"])
    return float(np.dot(va - pv, g["normal"]))


def instant_apbp(c: dict[str, object], jd: float) -> float:
    h = 60.0 / 86400.0
    pv = relative_xy_instant(c, "POINT_VENUS", jd)
    va = relative_xy_instant(c, "VARDO", jd)
    pv_vel = (relative_xy_instant(c, "POINT_VENUS", jd + h) - relative_xy_instant(c, "POINT_VENUS", jd - h)) / 120.0
    va_vel = (relative_xy_instant(c, "VARDO", jd + h) - relative_xy_instant(c, "VARDO", jd - h)) / 120.0
    tangent = unit(0.5 * (pv_vel + va_vel))
    normal = np.array([-tangent[1], tangent[0]])
    if float(np.dot(va - pv, normal)) < 0.0:
        normal = -normal
    return float(np.dot(va - pv, normal))


def analyze() -> tuple[pd.DataFrame, dict[str, float | str]]:
    OUT.mkdir(parents=True, exist_ok=True)
    master = build_master()
    c = splines(master)
    ca_jd = geocentric_ca(c)
    fixed = fixed_geometry(c, ca_jd)
    offsets = np.arange(-WINDOW_MIN, WINDOW_MIN + 1e-9, PLOT_STEP_MIN)
    rows: list[dict[str, float | str]] = []
    for m in offsets:
        jd = ca_jd + float(m) / 1440.0
        rho = geocentric_rho(c, jd)
        ap_fixed = fixed_apbp(c, jd, fixed)
        ap_inst = instant_apbp(c, jd)
        rows.append({
            "minute_from_geocentric_ca": float(m),
            "jd_tdb": float(jd),
            "utc": utc(jd),
            "rho_arcsec": float(rho),
            "apbp_fixed_arcsec": float(ap_fixed),
            "apbp_instant_arcsec": float(ap_inst),
            "delta_apbp_arcsec": float(ap_inst - ap_fixed),
        })
    df = pd.DataFrame(rows)
    df["rho_minus_min_arcsec"] = df["rho_arcsec"] - float(df["rho_arcsec"].min())
    df["rho_minus_ca_arcsec"] = df["rho_arcsec"] - float(geocentric_rho(c, ca_jd))
    df["drho_dt_arcsec_per_min"] = np.gradient(df["rho_arcsec"].to_numpy(float), df["minute_from_geocentric_ca"].to_numpy(float))
    df["delta_apbp_2x_arcsec"] = 2.0 * df["delta_apbp_arcsec"]
    df.to_csv(CSV, index=False, float_format="%.15f")
    i_min = int(np.argmin(df["rho_arcsec"].to_numpy(float)))
    i_delta_min = int(np.argmin(np.abs(df["delta_apbp_arcsec"].to_numpy(float))))
    stats: dict[str, float | str] = {
        "ca_jd": float(ca_jd),
        "ca_utc": utc(ca_jd),
        "ca_rho_arcsec": float(geocentric_rho(c, ca_jd)),
        "sample_min_offset_min": float(df["minute_from_geocentric_ca"].iloc[i_min]),
        "sample_min_rho_arcsec": float(df["rho_arcsec"].iloc[i_min]),
        "delta_best_offset_min": float(df["minute_from_geocentric_ca"].iloc[i_delta_min]),
        "delta_best_abs_arcsec": float(abs(df["delta_apbp_arcsec"].iloc[i_delta_min])),
        "delta_ca_arcsec": float(np.interp(0.0, df["minute_from_geocentric_ca"], df["delta_apbp_arcsec"])),
        "rho_plus30_minus_min_arcsec": float(df["rho_arcsec"].iloc[-1] - df["rho_arcsec"].min()),
        "rho_minus30_minus_min_arcsec": float(df["rho_arcsec"].iloc[0] - df["rho_arcsec"].min()),
    }
    return df, stats


def style_axis(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.30, alpha=0.60)
    ax.tick_params(labelsize=7.2, width=0.35, length=2.4)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.35)


def table_style(table, gold_rows=(), teal_rows=(), red_rows=(), fontsize=6.4) -> None:
    table.auto_set_font_size(False)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#70879A")
        cell.set_linewidth(0.32)
        cell.get_text().set_color(FG)
        cell.get_text().set_fontsize(fontsize)
        if row == 0:
            cell.set_facecolor(TABLE_HEADER)
            cell.get_text().set_fontweight("bold")
        elif row in red_rows:
            cell.set_facecolor(TABLE_RED)
            cell.get_text().set_fontweight("bold")
        elif row in gold_rows:
            cell.set_facecolor(TABLE_GOLD)
            cell.get_text().set_fontweight("bold")
        elif row in teal_rows:
            cell.set_facecolor(TABLE_TEAL)
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
    x = df["minute_from_geocentric_ca"].to_numpy(float)
    y_rho = (1000.0 * df["rho_minus_min_arcsec"]).to_numpy(float)
    y_slope = df["drho_dt_arcsec_per_min"].to_numpy(float)
    y_delta = df["delta_apbp_2x_arcsec"].to_numpy(float)
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs = fig.add_gridspec(4, 1, height_ratios=[0.31, 0.22, 0.22, 0.25], left=0.065, right=0.985, top=0.900, bottom=0.075, hspace=0.210)
    ax_rho = fig.add_subplot(gs[0, 0])
    ax_slope = fig.add_subplot(gs[1, 0], sharex=ax_rho)
    ax_delta = fig.add_subplot(gs[2, 0], sharex=ax_rho)
    ax_table = fig.add_subplot(gs[3, 0])
    fig.suptitle("1769 Venus Transit — Why Closest Approach Is the Physical Reference", fontsize=14.5, fontweight="bold", y=0.965)
    fig.text(0.5, 0.932, "Fresh JPL Horizons geometric vectors; ±30 minutes around geocentric closest approach. Top minimum is Venus–Sun center distance, not method-residual minimum.", ha="center", fontsize=7.4, color=MUTED)

    ax_rho.plot(x, y_rho, color=GOLD, linewidth=0.72, zorder=3, label="ρ(t) − ρmin, geocentric Venus–Sun apparent distance")
    ax_rho.scatter(x, y_rho, s=5, color=GOLD, edgecolors="none", alpha=0.82, zorder=4)
    ax_rho.axvline(0.0, color=BLUE, linewidth=0.62, alpha=0.85, zorder=2)
    ax_rho.axhline(0.0, color=MUTED, linewidth=0.35, alpha=0.65, zorder=1)
    ax_rho.scatter([0.0], [0.0], s=64, marker="D", color=BLUE, edgecolors=FG, linewidths=0.35, zorder=7)
    ax_rho.annotate(f"physical CA\nρmin = {float(stats['ca_rho_arcsec']):.9f}″\n{stats['ca_utc']}", xy=(0.0, 0.0), xytext=(4.0, max(y_rho) * 0.18 + 0.01), ha="left", va="bottom", fontsize=7.0, color=FG, arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.35})
    ax_rho.set_ylabel("ρ − ρmin\n(milliarcsec)", fontsize=8.6)
    ax_rho.set_title("DISTANCE MINIMUM — CLOSEST APPROACH IS WHERE VENUS–SUN APPARENT CENTER DISTANCE IS MINIMUM", fontsize=9.2, fontweight="bold")
    style_axis(ax_rho)
    ax_rho.legend(loc="upper right", fontsize=6.5, frameon=False)

    ax_slope.plot(x, y_slope, color=GREEN, linewidth=0.72, zorder=3, label="dρ/dt")
    ax_slope.scatter(x, y_slope, s=5, color=GREEN, edgecolors="none", alpha=0.75, zorder=4)
    ax_slope.axhline(0.0, color=MUTED, linewidth=0.44, alpha=0.75, zorder=1)
    ax_slope.axvline(0.0, color=BLUE, linewidth=0.62, alpha=0.85, zorder=2)
    ax_slope.scatter([0.0], [0.0], s=52, marker="D", color=BLUE, edgecolors=FG, linewidths=0.35, zorder=7)
    ax_slope.annotate("at CA: dρ/dt = 0", xy=(0.0, 0.0), xytext=(-28.0, max(y_slope) * 0.55), ha="left", va="center", fontsize=7.0, color=FG, arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.35})
    ax_slope.set_ylabel("dρ/dt\n(arcsec/min)", fontsize=8.6)
    ax_slope.set_title("SLOPE TEST — CLOSEST APPROACH IS STATIONARY DISTANCE, NOT NECESSARILY ZERO METHOD DELTA", fontsize=9.2, fontweight="bold")
    style_axis(ax_slope)
    ax_slope.legend(loc="upper right", fontsize=6.5, frameon=False)

    ax_delta.plot(x, y_delta, color=PURPLE, linewidth=0.72, zorder=3, label="2× ΔA′B′ = 2×(instantaneous − fixed)")
    ax_delta.scatter(x, y_delta, s=5, color=PURPLE, edgecolors="none", alpha=0.76, zorder=4)
    ax_delta.axhline(0.0, color=MUTED, linewidth=0.44, alpha=0.75, zorder=1)
    ax_delta.axvline(0.0, color=BLUE, linewidth=0.62, alpha=0.85, zorder=2)
    ca_y = 2.0 * float(stats["delta_ca_arcsec"])
    ax_delta.scatter([0.0], [ca_y], s=60, marker="X", color=RED, edgecolors=FG, linewidths=0.35, zorder=8)
    bx = float(stats["delta_best_offset_min"])
    by = 2.0 * float(stats["delta_best_abs_arcsec"])
    interp_delta = interp1d(x, y_delta, kind="linear", fill_value="extrapolate")
    signed_by = float(interp_delta(bx))
    ax_delta.scatter([bx], [signed_by], s=58, marker="o", color=GOLD, edgecolors=FG, linewidths=0.35, zorder=8)
    ax_delta.annotate(f"CA method residual\n2×Δ = {ca_y:+.9f}″", xy=(0.0, ca_y), xytext=(-29.0, ca_y + 0.0025), ha="left", va="bottom", fontsize=7.0, color=FG, arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.35})
    ax_delta.annotate(f"best method agreement\nminute = {bx:+.3f}\n2×Δ = {signed_by:+.9f}″", xy=(bx, signed_by), xytext=(8.0, signed_by - 0.004), ha="left", va="top", fontsize=7.0, color=FG, arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.35})
    ax_delta.set_ylabel("2× ΔA′B′\n(arcsec)", fontsize=8.6)
    ax_delta.set_xlabel("Minutes from geocentric closest approach", fontsize=8.8)
    ax_delta.set_title("METHOD RESIDUAL — THIS ZERO/MINIMUM CAN SHIFT AWAY FROM PHYSICAL CLOSEST APPROACH", fontsize=9.2, fontweight="bold")
    style_axis(ax_delta)
    ax_delta.legend(loc="upper right", fontsize=6.5, frameon=False)

    ax_table.axis("off")
    rows = [
        ["Quantity", "Value", "Unit", "Meaning"],
        ["Geocentric closest approach UTC", str(stats["ca_utc"]), "UTC", "minimum ρ(t)"],
        ["ρ at closest approach", f"{float(stats['ca_rho_arcsec']):.12f}", "arcsec", "Venus–Sun apparent center distance"],
        ["ρ(+30 min) − ρmin", f"{1000.0 * float(stats['rho_plus30_minus_min_arcsec']):.9f}", "mas", "distance grows away from CA"],
        ["ρ(−30 min) − ρmin", f"{1000.0 * float(stats['rho_minus30_minus_min_arcsec']):.9f}", "mas", "distance grows away from CA"],
        ["ΔA′B′ at CA", f"{float(stats['delta_ca_arcsec']):+.12f}", "arcsec", "method residual at physical CA"],
        ["Best method agreement offset", f"{float(stats['delta_best_offset_min']):+.6f}", "minutes", "minimum |instantaneous − fixed|"],
        ["Best method agreement |Δ|", f"{float(stats['delta_best_abs_arcsec']):.12f}", "arcsec", "not a physical closest approach"],
        ["Samples", f"{len(df):d}", "rows", "0.5-minute JPL-derived CA window"],
    ]
    table = ax_table.table(cellText=rows, cellLoc="left", colWidths=[0.26, 0.24, 0.13, 0.37], bbox=[0.0, 0.10, 1.0, 0.80])
    table_style(table, teal_rows=(1, 2, 5), gold_rows=(3, 4, 6, 8), red_rows=(7,), fontsize=6.35)
    fig.text(0.5, 0.026, f"File: {Path(__file__).name if '__file__' in globals() else 'VENUS_1769_CLOSEST_APPROACH_EXPLANATION_V0101.py'} | Output: {PNG.name} | CSV: {CSV.name}", ha="center", fontsize=5.9, color=MUTED)
    fig.savefig(PNG, dpi=220, facecolor=BG)
    display(Image(filename=str(PNG)))


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"JPL interval: {START} to {STOP}; step: {STEP}")
    print(f"Plot window: ±{WINDOW_MIN:.1f} minutes around geocentric closest approach; plot step: {PLOT_STEP_MIN:.1f} minutes")
    print("Sites for method residual: Point Venus, Tahiti; Vardø, Norway")
    print("COMMENTS")
    print("Shows why closest approach is a physical distance minimum: ρ(t) is minimized and dρ/dt crosses zero at CA.")
    print("Also shows that the fixed-vs-instantaneous A′B′ residual minimum can occur at a different time.")
    df, stats = analyze()
    plot(df, stats)
    print("RESULTS")
    print(f"Geocentric closest approach UTC: {stats['ca_utc']}")
    print(f"Geocentric closest approach JD_TDB: {float(stats['ca_jd']):.15f}")
    print(f"ρ at closest approach: {float(stats['ca_rho_arcsec']):.12f} arcsec")
    print(f"ρ(+30 min) − ρmin: {1000.0 * float(stats['rho_plus30_minus_min_arcsec']):.9f} mas")
    print(f"ρ(-30 min) − ρmin: {1000.0 * float(stats['rho_minus30_minus_min_arcsec']):.9f} mas")
    print(f"ΔA′B′ at CA: {float(stats['delta_ca_arcsec']):+.12f} arcsec")
    print(f"Best method-agreement offset: {float(stats['delta_best_offset_min']):+.6f} minutes")
    print(f"Best method-agreement |ΔA′B′|: {float(stats['delta_best_abs_arcsec']):.12f} arcsec")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    print("PAPER COMPARISON")
    print("NOT USED: this is a JPL-vector geometry explanation plot; no literature constants are imported.")
    print("EQUATION STATUS")
    print("PASS: ρ(t)=angular separation between geocentric JPL Sun and Venus vectors; CA solves min ρ(t); dρ/dt is numerical slope; method residual is instantaneous A′B′ minus fixed A′B′.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0101
