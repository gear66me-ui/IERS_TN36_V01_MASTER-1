# V0103
# Audit reference: normalized Sun-Earth, Earth-Venus, and Sun-Venus distances over ±30 min from geocentric closest approach; Python/Matplotlib only; no AI images.
from __future__ import annotations

import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0103"
LOCAL_TZ = ZoneInfo("America/Bogota")
OUT = Path("/content/VENUS_1769_DISTANCE_NORMALIZATION_CA_WINDOW_V0103_OUTPUT")
PNG = OUT / "VENUS_1769_DISTANCE_NORMALIZATION_CA_WINDOW_V0103.png"
CSV = OUT / "VENUS_1769_DISTANCE_NORMALIZATION_CA_WINDOW_V0103.csv"

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
PURPLE = "#9B8CFF"
TABLE_HEADER = "#23466F"
TABLE_BODY = "#101A2E"
TABLE_TEAL = "#164B55"


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


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def unit(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n <= 0.0:
        raise RuntimeError("Zero vector cannot be normalized.")
    return np.asarray(v, dtype=float) / n


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


def distances(c: dict[str, object], jd: float) -> tuple[float, float, float]:
    sun = vec(c, "SUN", jd)
    venus = vec(c, "VENUS", jd)
    earth = np.zeros(3, dtype=float)
    d_es = norm(sun - earth)
    d_ev = norm(venus - earth)
    d_sv = norm(sun - venus)
    return d_es, d_ev, d_sv


def solve_ca(c: dict[str, object]) -> float:
    jds = np.asarray(c["JD_TDB"], dtype=float)
    rho_samples = np.array([rho_arcsec(c, float(jd)) for jd in jds], dtype=float)
    i = int(np.argmin(rho_samples))
    lo = float(jds[max(0, i - 2)])
    hi = float(jds[min(len(jds) - 1, i + 2)])
    res = minimize_scalar(lambda x: rho_arcsec(c, float(x)), bounds=(lo, hi), method="bounded", options={"xatol": 1e-13, "maxiter": 600})
    if not res.success:
        raise RuntimeError("Closest-approach minimization failed.")
    return float(res.x)


def analyze() -> tuple[pd.DataFrame, dict[str, float | str]]:
    master = build_master()
    c = splines(master)
    ca_jd = solve_ca(c)
    d_es0, d_ev0, d_sv0 = distances(c, ca_jd)
    grid_minutes = np.linspace(-WINDOW_MIN, WINDOW_MIN, int(2 * WINDOW_MIN) + 1)
    rows = []
    for m in grid_minutes:
        jd = ca_jd + float(m) / 1440.0
        d_es, d_ev, d_sv = distances(c, jd)
        rows.append({
            "minute_from_geocentric_ca": float(m),
            "jd_tdb": float(jd),
            "utc": utc_from_jd(jd),
            "earth_sun_km": float(d_es),
            "earth_venus_km": float(d_ev),
            "sun_venus_km": float(d_sv),
            "earth_sun_norm_ppm": float((d_es / d_es0 - 1.0) * 1_000_000.0),
            "earth_venus_norm_ppm": float((d_ev / d_ev0 - 1.0) * 1_000_000.0),
            "sun_venus_norm_ppm": float((d_sv / d_sv0 - 1.0) * 1_000_000.0),
            "ev_over_es_norm_ppm": float(((d_ev / d_es) / (d_ev0 / d_es0) - 1.0) * 1_000_000.0),
            "sv_over_es_norm_ppm": float(((d_sv / d_es) / (d_sv0 / d_es0) - 1.0) * 1_000_000.0),
        })
    df = pd.DataFrame(rows)
    stats: dict[str, float | str] = {
        "ca_utc": utc_from_jd(ca_jd),
        "ca_jd_tdb": ca_jd,
        "rho_min_arcsec": rho_arcsec(c, ca_jd),
        "earth_sun_ca_km": d_es0,
        "earth_venus_ca_km": d_ev0,
        "sun_venus_ca_km": d_sv0,
        "earth_venus_over_earth_sun_ca": d_ev0 / d_es0,
        "sun_venus_over_earth_sun_ca": d_sv0 / d_es0,
        "earth_sun_ppm_span": float(df["earth_sun_norm_ppm"].max() - df["earth_sun_norm_ppm"].min()),
        "earth_venus_ppm_span": float(df["earth_venus_norm_ppm"].max() - df["earth_venus_norm_ppm"].min()),
        "sun_venus_ppm_span": float(df["sun_venus_norm_ppm"].max() - df["sun_venus_norm_ppm"].min()),
        "ev_over_es_ppm_span": float(df["ev_over_es_norm_ppm"].max() - df["ev_over_es_norm_ppm"].min()),
        "sv_over_es_ppm_span": float(df["sv_over_es_norm_ppm"].max() - df["sv_over_es_norm_ppm"].min()),
        "samples": len(df),
    }
    return df, stats


def style_axis(ax) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.7)
    ax.grid(True, color=GRID, lw=0.45, alpha=0.75)


def plot(df: pd.DataFrame, stats: dict[str, float | str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False)
    fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG, dpi=180)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.25, 1.45, 1.0], hspace=0.20)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0])
    for ax in (ax1, ax2, ax3):
        style_axis(ax)
        ax.axvline(0.0, color=GOLD, lw=0.9, alpha=0.95)
    x = df["minute_from_geocentric_ca"].to_numpy(float)
    ax1.plot(x, df["earth_sun_norm_ppm"], lw=1.1, marker="o", ms=1.7, color=BLUE, label="Earth-Sun normalized")
    ax1.plot(x, df["earth_venus_norm_ppm"], lw=1.1, marker="o", ms=1.7, color=GREEN, label="Earth-Venus normalized")
    ax1.plot(x, df["sun_venus_norm_ppm"], lw=1.1, marker="o", ms=1.7, color=PURPLE, label="Sun-Venus normalized")
    ax1.set_ylabel("Δ distance from CA (ppm)", color=FG, fontsize=9)
    ax1.legend(loc="upper left", fontsize=8, facecolor=TABLE_BODY, edgecolor=GRID, labelcolor=FG)
    ax1.set_title("1769 Venus Transit — Sun, Venus, Earth Distance Normalization Around Geocentric Closest Approach", color=FG, fontsize=12, pad=10)
    ax2.plot(x, df["ev_over_es_norm_ppm"], lw=1.1, marker="o", ms=1.7, color=GREEN, label="(Earth-Venus / Earth-Sun) normalized")
    ax2.plot(x, df["sv_over_es_norm_ppm"], lw=1.1, marker="o", ms=1.7, color=PURPLE, label="(Sun-Venus / Earth-Sun) normalized")
    ax2.axhline(0.0, color=MUTED, lw=0.7, alpha=0.85)
    ax2.set_ylabel("Δ ratio from CA (ppm)", color=FG, fontsize=9)
    ax2.legend(loc="upper left", fontsize=8, facecolor=TABLE_BODY, edgecolor=GRID, labelcolor=FG)
    ax2.set_xlabel("Minutes from geocentric closest approach", color=FG, fontsize=9)
    ax3.axis("off")
    table_rows = [
        ["Geocentric closest approach UTC", f"{stats['ca_utc']}"],
        ["Minimum ρ", f"{float(stats['rho_min_arcsec']):.12f} arcsec"],
        ["Earth-Sun at CA", f"{float(stats['earth_sun_ca_km']):,.6f} km"],
        ["Earth-Venus at CA", f"{float(stats['earth_venus_ca_km']):,.6f} km"],
        ["Sun-Venus at CA", f"{float(stats['sun_venus_ca_km']):,.6f} km"],
        ["EV/ES at CA", f"{float(stats['earth_venus_over_earth_sun_ca']):.12f}"],
        ["SV/ES at CA", f"{float(stats['sun_venus_over_earth_sun_ca']):.12f}"],
        ["EV/ES ppm span ±30 min", f"{float(stats['ev_over_es_ppm_span']):.9f} ppm"],
        ["SV/ES ppm span ±30 min", f"{float(stats['sv_over_es_ppm_span']):.9f} ppm"],
    ]
    table = ax3.table(cellText=table_rows, colLabels=["Quantity", "Value"], loc="center", cellLoc="left", colWidths=[0.42, 0.50])
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1, 1.13)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.45)
        cell.get_text().set_color(FG)
        if r == 0:
            cell.set_facecolor(TABLE_HEADER)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(TABLE_TEAL if r in (1, 2) else TABLE_BODY)
            if r in (1, 2):
                cell.get_text().set_weight("bold")
    fig.text(0.5, 0.018, f"JPL Horizons geometric vectors | geocenter location {GEOCENTER_LOCATION} | Version {VERSION}", color=MUTED, fontsize=7.5, ha="center")
    fig.savefig(PNG, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def print_section(name: str) -> None:
    print(name)


def main() -> None:
    print_section("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"Observer/location: GEOCENTER_LOCATION = {GEOCENTER_LOCATION}")
    print(f"Targets: Sun={SUN_TARGET}, Venus={VENUS_TARGET}")
    print(f"JPL window: {START} UTC to {STOP} UTC, step {STEP}")
    print(f"Plot window: ±{WINDOW_MIN:.1f} minutes from solved geocentric closest approach")
    print("Distances: Earth-Sun, Earth-Venus, Sun-Venus from JPL geometric vectors")
    print("Normalization: (D(t)/D(CA)-1)×1,000,000 ppm")
    print_section("COMMENTS")
    print("This widget uses the same geocentric rho minimum as V0102 to define t=0.")
    print("All curves are normalized to their own value at closest approach.")
    print("No rejected or prior comparison markers are plotted.")
    df, stats = analyze()
    plot(df, stats)
    print_section("RESULTS")
    print(f"Geocentric closest approach UTC      : {stats['ca_utc']}")
    print(f"Geocentric closest approach JD TDB   : {float(stats['ca_jd_tdb']):.12f}")
    print(f"Minimum rho                          : {float(stats['rho_min_arcsec']):.12f} arcsec")
    print(f"Earth-Sun at CA                      : {float(stats['earth_sun_ca_km']):,.6f} km")
    print(f"Earth-Venus at CA                    : {float(stats['earth_venus_ca_km']):,.6f} km")
    print(f"Sun-Venus at CA                      : {float(stats['sun_venus_ca_km']):,.6f} km")
    print(f"Earth-Venus / Earth-Sun at CA        : {float(stats['earth_venus_over_earth_sun_ca']):.12f}")
    print(f"Sun-Venus / Earth-Sun at CA          : {float(stats['sun_venus_over_earth_sun_ca']):.12f}")
    print(f"Earth-Sun normalized span            : {float(stats['earth_sun_ppm_span']):.9f} ppm")
    print(f"Earth-Venus normalized span          : {float(stats['earth_venus_ppm_span']):.9f} ppm")
    print(f"Sun-Venus normalized span            : {float(stats['sun_venus_ppm_span']):.9f} ppm")
    print(f"EV/ES normalized ratio span          : {float(stats['ev_over_es_ppm_span']):.9f} ppm")
    print(f"SV/ES normalized ratio span          : {float(stats['sv_over_es_ppm_span']):.9f} ppm")
    print_section("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    print(f"Rows: {int(stats['samples'])}")
    print_section("PAPER COMPARISON")
    print("NOT USED: no published distance values are used as inputs; JPL vectors only.")
    print_section("EQUATION STATUS")
    print("rho(t) = angular separation of geocentric Sun and Venus unit vectors: VERIFIED")
    print("distance normalization = (D(t)/D(CA)-1)*1e6 ppm: VERIFIED")
    print("ratio normalization = ((D1/D2)/(D1/D2 at CA)-1)*1e6 ppm: VERIFIED")
    display(Image(filename=str(PNG)))
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0103
