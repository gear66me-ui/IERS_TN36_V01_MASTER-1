# V0100
# Audit reference: standalone +/-30 minute geocentric CA normal/tangent component audit; Python/Matplotlib only; no AI images.
from __future__ import annotations

import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0100"
LOCAL_TZ = ZoneInfo("America/Bogota")
OUT = Path("/content/VENUS_1769_NORMAL_TANGENT_COMPONENTS_CA_WINDOW_V0100_OUTPUT")
PNG = OUT / "VENUS_1769_NORMAL_TANGENT_COMPONENTS_CA_WINDOW_V0100.png"
CSV = OUT / "VENUS_1769_NORMAL_TANGENT_COMPONENTS_CA_WINDOW_V0100.csv"

ARC = 206_264.80624709636
AU_KM = 149_597_870.700000
START = "1769-06-03 21:00"
STOP = "1769-06-03 23:40"
STEP = "1m"
WINDOW_MIN = 30.0
SAMPLE_STEP_MIN = 1.0

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

POINT_VENUS = {"key": "POINT_VENUS", "label": "Point Venus, Tahiti", "short": "PV", "lat": -17.4956, "lon": -149.4939, "elevation": 0.0, "body": 399}
VARDO = {"key": "VARDO", "label": "Vardø, Norway", "short": "V", "lat": 70.3724, "lon": 31.1103, "elevation": 0.0, "body": 399}
SITES = (POINT_VENUS, VARDO)
PREFIXES = ("GEOCENTER_SUN", "GEOCENTER_VENUS", "POINT_VENUS_SUN", "POINT_VENUS_VENUS", "VARDO_SUN", "VARDO_VENUS")


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


def utc(jd: float) -> str:
    return Time(float(jd), format="jd", scale="tdb").utc.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def loc(site: dict[str, object]) -> dict[str, float | int]:
    return {"lon": float(site["lon"]), "lat": float(site["lat"]), "elevation": float(site["elevation"]), "body": int(site["body"])}


def download(prefix: str, target_id: str, location) -> pd.DataFrame:
    last = None
    for attempt in range(4):
        try:
            tab = Horizons(id=target_id, location=location, epochs={"start": START, "stop": STOP, "step": STEP}, id_type=None).vectors(refplane="ecliptic", aberrations="geometric", cache=False)
            raw = tab.to_pandas()
            df = pd.DataFrame({"JD_TDB": pd.to_numeric(raw["datetime_jd"], errors="coerce")})
            for ax in "xyz":
                df[f"{prefix}_{ax.upper()}_KM"] = pd.to_numeric(raw[ax], errors="coerce") * AU_KM
            df = df.dropna().drop_duplicates("JD_TDB").sort_values("JD_TDB").reset_index(drop=True)
            if len(df) < 120:
                raise RuntimeError(f"Incomplete JPL series for {prefix}: {len(df)} rows")
            return df
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"JPL query failed for {prefix}: {last}")


def build_master() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for target_name, target_id in (("SUN", "10"), ("VENUS", "299")):
        parts.append(download(f"GEOCENTER_{target_name}", target_id, "@399"))
    for site in SITES:
        for target_name, target_id in (("SUN", "10"), ("VENUS", "299")):
            parts.append(download(f"{site['key']}_{target_name}", target_id, loc(site)))
    master = parts[0]
    for df in parts[1:]:
        master = master.merge(df, on="JD_TDB", how="inner", validate="one_to_one")
    return master.sort_values("JD_TDB").reset_index(drop=True)


def splines(master: pd.DataFrame) -> dict[str, object]:
    jds = master["JD_TDB"].to_numpy(float)
    out: dict[str, object] = {"JD_TDB": jds}
    for prefix in PREFIXES:
        for ax in "XYZ":
            out[f"{prefix}_{ax}_KM"] = CubicSpline(jds, master[f"{prefix}_{ax}_KM"].to_numpy(float), bc_type="natural")
    return out


def vec(c: dict[str, object], prefix: str, jd: float) -> np.ndarray:
    return np.array([float(c[f"{prefix}_{ax}_KM"](float(jd))) for ax in "XYZ"], dtype=float)


def sep_rad(a: np.ndarray, b: np.ndarray) -> float:
    ah = unit(a)
    bh = unit(b)
    return math.atan2(norm(np.cross(ah, bh)), float(np.dot(ah, bh)))


def geocentric_ca(c: dict[str, object]) -> float:
    jds = np.asarray(c["JD_TDB"], dtype=float)
    ref = float(jds[len(jds) // 2])
    lo = (float(jds[0]) - ref) * 86400.0
    hi = (float(jds[-1]) - ref) * 86400.0
    res = minimize_scalar(
        lambda sec: sep_rad(vec(c, "GEOCENTER_SUN", ref + float(sec) / 86400.0), vec(c, "GEOCENTER_VENUS", ref + float(sec) / 86400.0)),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-4, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError("Geocentric closest approach failed.")
    return ref + float(res.x) / 86400.0


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


def fixed_geometry(c: dict[str, object], ca: float) -> dict[str, object]:
    basis = basis_from_sun(vec(c, "GEOCENTER_SUN", ca))
    h = 0.5 / 86400.0
    pv_plus = relative_xy_fixed(c, "POINT_VENUS", ca + h, basis)
    pv_minus = relative_xy_fixed(c, "POINT_VENUS", ca - h, basis)
    va_plus = relative_xy_fixed(c, "VARDO", ca + h, basis)
    va_minus = relative_xy_fixed(c, "VARDO", ca - h, basis)
    tangent = unit(unit(pv_plus - pv_minus) + unit(va_plus - va_minus))
    normal = np.array([-tangent[1], tangent[0]])
    pv0 = relative_xy_fixed(c, "POINT_VENUS", ca, basis)
    va0 = relative_xy_fixed(c, "VARDO", ca, basis)
    if float(np.dot(va0 - pv0, normal)) < 0.0:
        normal = -normal
        tangent = -tangent
    return {"basis": basis, "tangent": tangent, "normal": normal, "angle_deg": math.degrees(math.atan2(tangent[1], tangent[0]))}


def instant_geometry(c: dict[str, object], jd: float) -> dict[str, np.ndarray | float]:
    h = 60.0 / 86400.0
    pv_plus = relative_xy_instant(c, "POINT_VENUS", jd + h)
    pv_minus = relative_xy_instant(c, "POINT_VENUS", jd - h)
    va_plus = relative_xy_instant(c, "VARDO", jd + h)
    va_minus = relative_xy_instant(c, "VARDO", jd - h)
    tangent = unit(0.5 * ((pv_plus - pv_minus) + (va_plus - va_minus)))
    normal = np.array([-tangent[1], tangent[0]])
    pv = relative_xy_instant(c, "POINT_VENUS", jd)
    va = relative_xy_instant(c, "VARDO", jd)
    if float(np.dot(va - pv, normal)) < 0.0:
        normal = -normal
        tangent = -tangent
    return {"tangent": tangent, "normal": normal, "angle_deg": math.degrees(math.atan2(tangent[1], tangent[0]))}


def components(c: dict[str, object], jd: float, fixed: dict[str, object]) -> dict[str, float]:
    pv_fixed = relative_xy_fixed(c, "POINT_VENUS", jd, fixed["basis"])
    va_fixed = relative_xy_fixed(c, "VARDO", jd, fixed["basis"])
    delta_fixed = va_fixed - pv_fixed
    normal_fixed = float(np.dot(delta_fixed, fixed["normal"]))
    tangent_fixed = float(np.dot(delta_fixed, fixed["tangent"]))
    inst = instant_geometry(c, jd)
    pv_inst = relative_xy_instant(c, "POINT_VENUS", jd)
    va_inst = relative_xy_instant(c, "VARDO", jd)
    delta_inst = va_inst - pv_inst
    normal_inst = float(np.dot(delta_inst, inst["normal"]))
    tangent_inst = float(np.dot(delta_inst, inst["tangent"]))
    return {
        "normal_fixed_arcsec": normal_fixed,
        "tangent_fixed_arcsec": tangent_fixed,
        "normal_instant_arcsec": normal_inst,
        "tangent_instant_arcsec": tangent_inst,
        "delta_normal_arcsec": normal_inst - normal_fixed,
        "delta_tangent_arcsec": tangent_inst - tangent_fixed,
        "fixed_angle_deg": float(fixed["angle_deg"]),
        "instant_angle_deg": float(inst["angle_deg"]),
    }


def analyze() -> tuple[pd.DataFrame, dict[str, float | str]]:
    master = build_master()
    c = splines(master)
    ca = geocentric_ca(c)
    fixed = fixed_geometry(c, ca)
    rel_minutes = np.arange(-WINDOW_MIN, WINDOW_MIN + 0.001, SAMPLE_STEP_MIN)
    rows: list[dict[str, float | str]] = []
    for minute in rel_minutes:
        jd = ca + float(minute) / 1440.0
        row = components(c, jd, fixed)
        row.update({"jd_tdb": float(jd), "utc": utc(float(jd)), "minutes_from_ca": float(minute)})
        rows.append(row)
    df = pd.DataFrame(rows)
    ca_row = df.iloc[int(np.argmin(np.abs(df["minutes_from_ca"].to_numpy(float))))]
    stats = {
        "ca_jd": float(ca),
        "ca_utc": utc(ca),
        "normal_fixed_ca": float(ca_row["normal_fixed_arcsec"]),
        "normal_instant_ca": float(ca_row["normal_instant_arcsec"]),
        "tangent_fixed_ca": float(ca_row["tangent_fixed_arcsec"]),
        "tangent_instant_ca": float(ca_row["tangent_instant_arcsec"]),
        "mean_delta_normal": float(df["delta_normal_arcsec"].mean()),
        "mean_delta_tangent": float(df["delta_tangent_arcsec"].mean()),
        "rms_delta_normal": float(np.sqrt(np.mean(df["delta_normal_arcsec"].to_numpy(float) ** 2))),
        "rms_delta_tangent": float(np.sqrt(np.mean(df["delta_tangent_arcsec"].to_numpy(float) ** 2))),
        "fixed_angle_deg": float(fixed["angle_deg"]),
        "instant_angle_mean_deg": float(df["instant_angle_deg"].mean()),
    }
    return df, stats


def table_style(table, teal_rows=(), gold_rows=(), fontsize=6.1) -> None:
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
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False, float_format="%.15f")
    plt.close("all")
    plt.rcParams.update({"font.family": "DejaVu Serif", "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED})
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.55, 0.23, 0.22], left=0.065, right=0.985, top=0.89, bottom=0.095, hspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    dax = fig.add_subplot(gs[1, 0], sharex=ax)
    tax = fig.add_subplot(gs[2, 0])
    fig.suptitle("1769 Venus Transit — Normal and Tangent Components Around Geocentric Closest Approach", fontsize=14.2, fontweight="bold", y=0.962)
    fig.text(0.5, 0.925, "Window: −30 to +30 minutes from geocentric CA. Four curves: fixed normal, fixed tangent, instantaneous normal, instantaneous tangent.", ha="center", fontsize=7.4, color=MUTED)
    x = df["minutes_from_ca"].to_numpy(float)
    nf = df["normal_fixed_arcsec"].to_numpy(float)
    tf = df["tangent_fixed_arcsec"].to_numpy(float)
    ni = df["normal_instant_arcsec"].to_numpy(float)
    ti = df["tangent_instant_arcsec"].to_numpy(float)
    ax.plot(x, nf, color=GREEN, linewidth=0.72, label="fixed normal A′B′")
    ax.plot(x, ni, color=BLUE, linewidth=0.72, linestyle="--", label="instantaneous normal A′B′")
    ax.plot(x, tf, color=GOLD, linewidth=0.72, label="fixed tangent AB")
    ax.plot(x, ti, color=PURPLE, linewidth=0.72, linestyle="--", label="instantaneous tangent AB")
    ax.axvline(0.0, color=RED, linewidth=0.55, alpha=0.8)
    ax.scatter([0.0], [float(stats["normal_fixed_ca"])], color=GREEN, edgecolors=FG, s=36, linewidths=0.3, zorder=5)
    ax.scatter([0.0], [float(stats["normal_instant_ca"])], color=BLUE, edgecolors=FG, s=36, linewidths=0.3, zorder=5)
    ax.scatter([0.0], [float(stats["tangent_fixed_ca"])], color=GOLD, edgecolors=FG, s=36, linewidths=0.3, zorder=5)
    ax.scatter([0.0], [float(stats["tangent_instant_ca"])], color=PURPLE, edgecolors=FG, s=36, linewidths=0.3, zorder=5)
    ax.annotate(f"CA {stats['ca_utc']}\nnormal fixed {float(stats['normal_fixed_ca']):.9f}\nnormal inst {float(stats['normal_instant_ca']):.9f}", xy=(0.0, float(stats["normal_fixed_ca"])), xytext=(3.5, float(stats["normal_fixed_ca"]) + 0.018), ha="left", fontsize=6.8, color=FG, arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.28})
    ax.annotate(f"tangent fixed {float(stats['tangent_fixed_ca']):+.9f}\ntangent inst {float(stats['tangent_instant_ca']):+.9f}", xy=(0.0, float(stats["tangent_fixed_ca"])), xytext=(-28.5, float(stats["tangent_fixed_ca"]) - 0.030), ha="left", fontsize=6.8, color=FG, arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.28})
    ax.set_ylabel("Component separation (arcsec)", fontsize=8.8)
    ax.grid(True, color=GRID, linewidth=0.32, alpha=0.55)
    ax.tick_params(labelsize=7.4, width=0.35, length=2.5)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.35)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=7.0, ncol=2)
    for t in leg.get_texts():
        t.set_color(FG)

    dn = df["delta_normal_arcsec"].to_numpy(float)
    dt = df["delta_tangent_arcsec"].to_numpy(float)
    dax.plot(x, dn, color=BLUE, linewidth=0.68, label="Δ normal = instant − fixed")
    dax.plot(x, dt, color=PURPLE, linewidth=0.68, label="Δ tangent = instant − fixed")
    dax.axhline(0.0, color=MUTED, linewidth=0.45)
    dax.axhline(float(stats["mean_delta_normal"]), color=BLUE, linewidth=0.45, linestyle=":", label="mean Δ normal")
    dax.axhline(float(stats["mean_delta_tangent"]), color=PURPLE, linewidth=0.45, linestyle=":", label="mean Δ tangent")
    dax.axvline(0.0, color=RED, linewidth=0.50, alpha=0.8)
    dax.set_ylabel("Delta (arcsec)", fontsize=8.4)
    dax.set_xlabel("Minutes from geocentric closest approach", fontsize=8.8)
    dax.grid(True, color=GRID, linewidth=0.32, alpha=0.55)
    dax.tick_params(labelsize=7.4, width=0.35, length=2.5)
    for spine in dax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.35)
    dleg = dax.legend(loc="upper right", frameon=False, fontsize=6.6, ncol=2)
    for t in dleg.get_texts():
        t.set_color(FG)

    tax.axis("off")
    rows = [
        ["Quantity", "Fixed", "Instantaneous", "Instant − fixed", "Unit"],
        ["Normal A′B′ at CA", f"{float(stats['normal_fixed_ca']):.12f}", f"{float(stats['normal_instant_ca']):.12f}", f"{float(stats['normal_instant_ca']) - float(stats['normal_fixed_ca']):+.12f}", "arcsec"],
        ["Tangent AB at CA", f"{float(stats['tangent_fixed_ca']):+.12f}", f"{float(stats['tangent_instant_ca']):+.12f}", f"{float(stats['tangent_instant_ca']) - float(stats['tangent_fixed_ca']):+.12f}", "arcsec"],
        ["Mean Δ normal", "—", "—", f"{float(stats['mean_delta_normal']):+.12f}", "arcsec"],
        ["Mean Δ tangent", "—", "—", f"{float(stats['mean_delta_tangent']):+.12f}", "arcsec"],
        ["RMS Δ normal", "—", "—", f"{float(stats['rms_delta_normal']):.12f}", "arcsec"],
        ["RMS Δ tangent", "—", "—", f"{float(stats['rms_delta_tangent']):.12f}", "arcsec"],
        ["Track angle", f"{float(stats['fixed_angle_deg']):.9f}", f"{float(stats['instant_angle_mean_deg']):.9f}", f"{float(stats['instant_angle_mean_deg']) - float(stats['fixed_angle_deg']):+.9f}", "deg"],
    ]
    table = tax.table(cellText=rows, cellLoc="left", colWidths=[0.25, 0.18, 0.20, 0.20, 0.17], bbox=[0.0, 0.02, 1.0, 0.88])
    table_style(table, teal_rows=(1, 2), gold_rows=(3, 4, 5, 6, 7), fontsize=6.4)
    fig.text(0.5, 0.042, f"File: VENUS_1769_NORMAL_TANGENT_COMPONENTS_CA_WINDOW_V0100.py | PNG: {PNG.name} | CSV: {CSV.name}", ha="center", fontsize=5.8, color=MUTED)
    fig.savefig(PNG, dpi=220, facecolor=BG)
    display(Image(filename=str(PNG)))


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"JPL query window UTC: {START} to {STOP}; step {STEP}")
    print(f"Analysis window: +/- {WINDOW_MIN:.1f} minutes around geocentric CA")
    print("Sites: Point Venus, Tahiti; Vardø, Norway")
    print("COMMENTS")
    print("Computes four observer-displacement components: fixed normal, fixed tangent, instantaneous normal, instantaneous tangent.")
    print("All curves use fresh JPL Horizons geometric ecliptic vectors and Matplotlib only.")
    df, stats = analyze()
    plot(df, stats)
    print("RESULTS")
    print(f"Geocentric CA UTC: {stats['ca_utc']}")
    print(f"Normal fixed A′B′ at CA: {float(stats['normal_fixed_ca']):.12f} arcsec")
    print(f"Normal instantaneous A′B′ at CA: {float(stats['normal_instant_ca']):.12f} arcsec")
    print(f"Tangent fixed AB at CA: {float(stats['tangent_fixed_ca']):+.12f} arcsec")
    print(f"Tangent instantaneous AB at CA: {float(stats['tangent_instant_ca']):+.12f} arcsec")
    print(f"Mean Δ normal: {float(stats['mean_delta_normal']):+.12f} arcsec")
    print(f"Mean Δ tangent: {float(stats['mean_delta_tangent']):+.12f} arcsec")
    print(f"RMS Δ normal: {float(stats['rms_delta_normal']):.12f} arcsec")
    print(f"RMS Δ tangent: {float(stats['rms_delta_tangent']):.12f} arcsec")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    print("PAPER COMPARISON")
    print("NOT USED: this is an internal JPL-vector component audit only.")
    print("EQUATION STATUS")
    print("PASS: normal components are projections along the common-normal direction; tangent components are projections along the track-tangent direction; fixed uses the geocentric CA frame; instantaneous recomputes the tangent-normal frame at each epoch.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0100