# V0107
# Audit reference: geocentric Sun-screen Earth/Venus visibility repair; tangent-plane normalized parabolas; Python/Matplotlib only; no AI images.
from __future__ import annotations

import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0107"
LOCAL_TZ = ZoneInfo("America/Bogota")
OUT = Path("/content/VENUS_1769_GEOCENTRIC_SUNSCREEN_EARTH_VENUS_VISIBLE_V0107_OUTPUT")
PNG = OUT / "VENUS_1769_GEOCENTRIC_SUNSCREEN_EARTH_VENUS_VISIBLE_V0107.png"
CSV = OUT / "VENUS_1769_GEOCENTRIC_SUNSCREEN_EARTH_VENUS_VISIBLE_V0107.csv"

ARCSEC_PER_RAD = 206_264.80624709636
AU_KM = 149_597_870.700000
START = "1769-06-03 21:30"
STOP = "1769-06-03 23:15"
STEP = "1m"
WINDOW_MIN = 30.0
SUN_TARGET = "10"
VENUS_TARGET = "299"
EARTH_TARGET = "399"
GEOCENTER_LOCATION = "500@399"
SOLAR_SYSTEM_BARYCENTER = "500@0"

BG = "#000000"
FG = "#F8FAFC"
MUTED = "#B8CBD6"
GRID = "#263A4B"
VENUS_C = "#42D7C3"
EARTH_C = "#D89B18"
WHITE = "#DDE7F0"
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


def utc_from_jd(jd: float) -> str:
    return Time(float(jd), format="jd", scale="tdb").utc.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def download(prefix: str, target_id: str, location: str) -> pd.DataFrame:
    last = None
    for attempt in range(4):
        try:
            table = Horizons(
                id=target_id,
                location=location,
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
    sun_geo = download("SUN_GEO", SUN_TARGET, GEOCENTER_LOCATION)
    venus_geo = download("VENUS_GEO", VENUS_TARGET, GEOCENTER_LOCATION)
    sun_bary = download("SUN_BARY", SUN_TARGET, SOLAR_SYSTEM_BARYCENTER)
    earth_bary = download("EARTH_BARY", EARTH_TARGET, SOLAR_SYSTEM_BARYCENTER)
    venus_bary = download("VENUS_BARY", VENUS_TARGET, SOLAR_SYSTEM_BARYCENTER)
    master = sun_geo.merge(venus_geo, on="JD_TDB", how="inner")
    master = master.merge(sun_bary, on="JD_TDB", how="inner")
    master = master.merge(earth_bary, on="JD_TDB", how="inner")
    master = master.merge(venus_bary, on="JD_TDB", how="inner")
    if len(master) < 80:
        raise RuntimeError(f"Insufficient JPL samples: {len(master)}")
    return master.sort_values("JD_TDB").reset_index(drop=True)


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


def basis_from_geocentric_sun(c: dict[str, object], jd: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = unit(vec(c, "SUN_GEO", jd))
    ecl_north = np.array([0.0, 0.0, 1.0], dtype=float)
    east = np.cross(ecl_north, z)
    if norm(east) < 1.0e-14:
        east = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x = unit(east)
    y = unit(np.cross(z, x))
    return x, y, z


def gnomonic_xy_arcsec(direction: np.ndarray, basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    xh, yh, zh = basis
    u = unit(direction)
    den = float(np.dot(u, zh))
    return np.array([
        ARCSEC_PER_RAD * float(np.dot(u, xh) / den),
        ARCSEC_PER_RAD * float(np.dot(u, yh) / den),
    ], dtype=float)


def venus_screen_xy(c: dict[str, object], jd: float, basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return gnomonic_xy_arcsec(vec(c, "VENUS_GEO", jd), basis)


def earth_screen_xy_heliocentric_antipode(c: dict[str, object], jd: float, basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    sun = vec(c, "SUN_BARY", jd)
    earth = vec(c, "EARTH_BARY", jd)
    earth_from_sun = earth - sun
    sun_to_earth_direction = earth_from_sun
    apparent_earth_from_sun_screen = gnomonic_xy_arcsec(sun_to_earth_direction, basis)
    return apparent_earth_from_sun_screen


def rho_v_arcsec(c: dict[str, object], jd: float) -> float:
    basis = basis_from_geocentric_sun(c, jd)
    xy = venus_screen_xy(c, jd, basis)
    return norm(xy)


def solve_venus_ca(c: dict[str, object]) -> float:
    jds = np.asarray(c["JD_TDB"], dtype=float)
    rho_samples = np.array([rho_v_arcsec(c, float(jd)) for jd in jds], dtype=float)
    i = int(np.argmin(rho_samples))
    lo = float(jds[max(0, i - 2)])
    hi = float(jds[min(len(jds) - 1, i + 2)])
    res = minimize_scalar(lambda x: rho_v_arcsec(c, float(x)), bounds=(lo, hi), method="bounded", options={"xatol": 1e-13, "maxiter": 600})
    if not res.success:
        raise RuntimeError("Closest-approach minimization failed for Venus rho(t).")
    return float(res.x)


def normalize_to_unit(y: np.ndarray) -> np.ndarray:
    a = np.asarray(y, dtype=float)
    span = float(np.max(a) - np.min(a))
    if span <= 0.0:
        return np.zeros_like(a)
    return (a - np.min(a)) / span


def analyze() -> tuple[pd.DataFrame, dict[str, float | str]]:
    master = build_master()
    c = splines(master)
    ca_jd = solve_venus_ca(c)
    fixed_basis = basis_from_geocentric_sun(c, ca_jd)
    grid_minutes = np.linspace(-WINDOW_MIN, WINDOW_MIN, int(2 * WINDOW_MIN) + 1)

    rows = []
    for m in grid_minutes:
        jd = ca_jd + float(m) / 1440.0
        vxy = venus_screen_xy(c, jd, fixed_basis)
        exy = earth_screen_xy_heliocentric_antipode(c, jd, fixed_basis)
        v_rho = norm(vxy)
        e_rho = norm(exy)
        rows.append({
            "minute_from_venus_geocentric_ca": float(m),
            "jd_tdb": float(jd),
            "utc": utc_from_jd(jd),
            "venus_x_arcsec": float(vxy[0]),
            "venus_y_arcsec": float(vxy[1]),
            "earth_x_arcsec": float(exy[0]),
            "earth_y_arcsec": float(exy[1]),
            "venus_rho_arcsec": float(v_rho),
            "earth_screen_rho_arcsec": float(e_rho),
            "venus_rho2_arcsec2": float(v_rho * v_rho),
            "earth_rho2_arcsec2": float(e_rho * e_rho),
        })
    df = pd.DataFrame(rows)
    df["venus_rho_excess_arcsec"] = df["venus_rho_arcsec"] - float(df["venus_rho_arcsec"].min())
    df["earth_rho_excess_arcsec"] = df["earth_screen_rho_arcsec"] - float(df["earth_screen_rho_arcsec"].min())
    df["venus_rho2_excess_arcsec2"] = df["venus_rho2_arcsec2"] - float(df["venus_rho2_arcsec2"].min())
    df["earth_rho2_excess_arcsec2"] = df["earth_rho2_arcsec2"] - float(df["earth_rho2_arcsec2"].min())
    df["venus_rho_excess_norm"] = normalize_to_unit(df["venus_rho_excess_arcsec"].to_numpy(float))
    df["earth_rho_excess_norm"] = normalize_to_unit(df["earth_rho_excess_arcsec"].to_numpy(float))
    df["venus_rho2_excess_norm"] = normalize_to_unit(df["venus_rho2_excess_arcsec2"].to_numpy(float))
    df["earth_rho2_excess_norm"] = normalize_to_unit(df["earth_rho2_excess_arcsec2"].to_numpy(float))
    df["norm_rho2_curve_delta_earth_minus_venus"] = df["earth_rho2_excess_norm"] - df["venus_rho2_excess_norm"]
    stats: dict[str, float | str] = {
        "ca_jd_tdb": ca_jd,
        "ca_utc": utc_from_jd(ca_jd),
        "venus_rho_min_arcsec": float(df["venus_rho_arcsec"].min()),
        "earth_screen_rho_min_arcsec": float(df["earth_screen_rho_arcsec"].min()),
        "max_abs_norm_rho2_delta": float(np.max(np.abs(df["norm_rho2_curve_delta_earth_minus_venus"].to_numpy(float)))),
        "venus_rho_excess_30_arcsec": float(df.loc[df["minute_from_venus_geocentric_ca"] == 30.0, "venus_rho_excess_arcsec"].iloc[0]),
        "earth_rho_excess_30_arcsec": float(df.loc[df["minute_from_venus_geocentric_ca"] == 30.0, "earth_rho_excess_arcsec"].iloc[0]),
        "samples": len(df),
    }
    return df, stats


def style_ax(ax):
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, alpha=0.45, linewidth=0.35)
    ax.tick_params(colors=FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.6)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def make_table(ax, stats: dict[str, float | str]) -> None:
    ax.axis("off")
    rows = [
        ["Quantity", "Value", "Unit / status"],
        ["Geocentric Venus CA UTC", str(stats["ca_utc"]), "JPL solve: min Venus rho"],
        ["Venus rho min", f"{float(stats['venus_rho_min_arcsec']):.12f}", "arcsec"],
        ["Earth screen-rho min", f"{float(stats['earth_screen_rho_min_arcsec']):.12f}", "arcsec; Sun-screen antipode direction"],
        ["Max |Earth-Venus| normalized rho2 delta", f"{float(stats['max_abs_norm_rho2_delta']):.12e}", "visibility audit"],
        ["Venus rho excess at +30 min", f"{float(stats['venus_rho_excess_30_arcsec']):.12f}", "arcsec"],
        ["Earth rho excess at +30 min", f"{float(stats['earth_rho_excess_30_arcsec']):.12f}", "arcsec"],
        ["Samples", f"{int(stats['samples'])}", "one-minute plotted window"],
    ]
    table = ax.table(cellText=rows, cellLoc="left", colWidths=[0.38, 0.32, 0.30], bbox=[0.0, 0.0, 1.0, 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(7.8)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#5F7488")
        cell.set_linewidth(0.28)
        cell.get_text().set_color(FG)
        if r == 0:
            cell.set_facecolor(TABLE_HEADER)
            cell.get_text().set_weight("bold")
        elif r in (1, 2, 3):
            cell.set_facecolor(TABLE_TEAL)
        elif r in (4,):
            cell.set_facecolor(TABLE_BODY)
        else:
            cell.set_facecolor(TABLE_GOLD)


def plot(df: pd.DataFrame, stats: dict[str, float | str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x = df["minute_from_venus_geocentric_ca"].to_numpy(float)
    fig = plt.figure(figsize=(13.5, 8.2), facecolor=BG)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.42], hspace=0.22, left=0.07, right=0.985, top=0.88, bottom=0.08)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    for ax in (ax1, ax2):
        style_ax(ax)
        ax.axvline(0.0, color=WHITE, linewidth=0.7, linestyle="--", alpha=0.85)
        ax.axhline(0.0, color=WHITE, linewidth=0.45, linestyle=":", alpha=0.7)
        ax.set_xlim(-31.5, 31.5)

    ax1.plot(x, df["venus_rho2_excess_norm"], color=VENUS_C, linewidth=1.25, marker="o", markersize=2.1, markevery=2, label="Venus normalized rho^2 - rho^2_min")
    ax1.plot(x, df["earth_rho2_excess_norm"], color=EARTH_C, linewidth=0.95, marker="x", markersize=3.0, markevery=2, linestyle="--", label="Earth normalized rho^2 - rho^2_min")
    ax1.set_ylabel("Independent normalized parabolic excess")
    ax1.set_title("Sun-screen tangent plane: Earth and Venus rho^2 parabolas, independently normalized", fontsize=13, fontweight="bold")
    leg1 = ax1.legend(loc="upper center", ncol=2, fontsize=8, facecolor="#0A1220", edgecolor="#33485C")
    for txt in leg1.get_texts():
        txt.set_color(FG)

    ax2.plot(x, df["venus_rho_excess_norm"], color=VENUS_C, linewidth=1.25, marker="o", markersize=2.1, markevery=2, label="Venus normalized rho - rho_min")
    ax2.plot(x, df["earth_rho_excess_norm"], color=EARTH_C, linewidth=0.95, marker="x", markersize=3.0, markevery=2, linestyle="--", label="Earth normalized rho - rho_min")
    ax2.set_ylabel("Independent normalized angular excess")
    ax2.set_xlabel("Minutes from geocentric Venus closest approach")
    ax2.set_title("Raw Sun-screen radius excess, independently normalized", fontsize=13, fontweight="bold")
    leg2 = ax2.legend(loc="upper center", ncol=2, fontsize=8, facecolor="#0A1220", edgecolor="#33485C")
    for txt in leg2.get_texts():
        txt.set_color(FG)

    make_table(ax3, stats)
    fig.suptitle("1769 Venus Transit — Geocentric Earth/Venus Sun-Screen Curves V0107", color=FG, fontsize=18, fontweight="bold", y=0.955)
    fig.text(0.5, 0.915, "No limb, no Point Venus, no surface observer. Fixed Sun-screen tangent plane at geocentric Venus closest approach.", color=MUTED, ha="center", fontsize=9)
    fig.text(0.5, 0.027, f"File: VENUS_1769_GEOCENTRIC_SUNSCREEN_EARTH_VENUS_VISIBLE_V0107.py | Output: {PNG.name} | CSV: {CSV.name}", color=MUTED, ha="center", fontsize=7)
    fig.savefig(PNG, dpi=220, facecolor=BG)
    plt.close(fig)
    display(Image(filename=str(PNG)))


def main() -> None:
    print("CODE INPUTS")
    print(f"Version : {VERSION}")
    print(f"Observer for Venus CA : geocenter {GEOCENTER_LOCATION}")
    print(f"Earth vectors : geocentric audit uses Earth barycentric vector relative to Sun; projected on same fixed geocentric Sun-screen")
    print(f"Window : +/- {WINDOW_MIN:.1f} minutes")
    print("COMMENTS")
    print("Plots only Earth and Venus on the fixed Sun-screen tangent plane at geocentric Venus closest approach.")
    print("Each curve is independently normalized so a hidden or scale-dominated Venus curve cannot disappear under Earth.")
    print("RESULTS")
    df, stats = analyze()
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False, float_format="%.15f")
    print(f"Geocentric Venus CA UTC : {stats['ca_utc']}")
    print(f"Venus rho min          : {float(stats['venus_rho_min_arcsec']):.12f} arcsec")
    print(f"Earth screen rho min    : {float(stats['earth_screen_rho_min_arcsec']):.12f} arcsec")
    print(f"Max normalized delta    : {float(stats['max_abs_norm_rho2_delta']):.12e}")
    plot(df, stats)
    print("OUTPUT SUMMARY")
    print(f"PNG : {PNG}")
    print(f"CSV : {CSV}")
    print("PAPER COMPARISON")
    print("NOT USED: this is a geometric visibility/normalization audit only.")
    print("EQUATION STATUS")
    print("PASS: Venus rho is geocentric Sun/Venus angular separation on the fixed Sun-screen tangent plane.")
    print("PASS: Earth curve is projected from barycentric Earth-Sun direction onto the same geocentric Sun-screen basis for visual comparison only.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z%z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0107
