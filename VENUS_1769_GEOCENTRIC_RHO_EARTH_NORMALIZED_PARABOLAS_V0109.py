# V0109
# Audit reference: clean geocentric Venus rho and Earth-distance-normalized Sun-screen rho parabolas; JPL Horizons; Python/Matplotlib only.
from __future__ import annotations

import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0109"
LOCAL_TZ = ZoneInfo("America/Bogota")
OUT = Path("/content/VENUS_1769_GEOCENTRIC_RHO_EARTH_NORMALIZED_PARABOLAS_V0109_OUTPUT")
PNG = OUT / "VENUS_1769_GEOCENTRIC_RHO_EARTH_NORMALIZED_PARABOLAS_V0109.png"
CSV = OUT / "VENUS_1769_GEOCENTRIC_RHO_EARTH_NORMALIZED_PARABOLAS_V0109.csv"

ARCSEC_PER_RAD = 206_264.80624709636
AU_KM = 149_597_870.700000
START = "1769-06-03 21:30"
STOP = "1769-06-03 23:15"
STEP = "1m"
WINDOW_MIN = 30
SUN_TARGET = "10"
VENUS_TARGET = "299"
GEOCENTER_LOCATION = "500@399"

BG = "#000000"
FG = "#F8FAFC"
GRID = "#263A4B"
VENUS_COLOR = "#42D7C3"
EARTH_COLOR = "#D89B18"
CA_COLOR = "#E8EEF4"
TABLE_HEADER = "#23466F"
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


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def unit(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = norm(a)
    if n <= 0.0:
        raise RuntimeError("Zero vector cannot be normalized.")
    return a / n


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
            for axis in "xyz":
                df[f"{prefix}_{axis.upper()}_KM"] = pd.to_numeric(raw[axis], errors="coerce") * AU_KM
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
    result: dict[str, object] = {"JD_TDB": jd}
    for col in df.columns:
        if col != "JD_TDB":
            result[col] = CubicSpline(jd, df[col].to_numpy(float), bc_type="natural")
    return result


def vec(c: dict[str, object], prefix: str, jd: float) -> np.ndarray:
    return np.array([
        float(c[f"{prefix}_X_KM"](jd)),
        float(c[f"{prefix}_Y_KM"](jd)),
        float(c[f"{prefix}_Z_KM"](jd)),
    ], dtype=float)


def angular_separation_rad(c: dict[str, object], jd: float) -> float:
    sun_u = unit(vec(c, "SUN", jd))
    venus_u = unit(vec(c, "VENUS", jd))
    return math.atan2(norm(np.cross(sun_u, venus_u)), float(np.dot(sun_u, venus_u)))


def rho_arcsec(c: dict[str, object], jd: float) -> float:
    return angular_separation_rad(c, jd) * ARCSEC_PER_RAD


def earth_sun_distance_km(c: dict[str, object], jd: float) -> float:
    return norm(vec(c, "SUN", jd))


def solve_ca(c: dict[str, object]) -> float:
    jds = np.asarray(c["JD_TDB"], dtype=float)
    sampled = np.array([rho_arcsec(c, float(jd)) for jd in jds], dtype=float)
    i = int(np.argmin(sampled))
    lo = float(jds[max(0, i - 2)])
    hi = float(jds[min(len(jds) - 1, i + 2)])
    result = minimize_scalar(
        lambda x: rho_arcsec(c, float(x)),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-13, "maxiter": 600},
    )
    if not result.success:
        raise RuntimeError("Geocentric closest-approach minimization failed.")
    return float(result.x)


def earth_normalized_rho_arcsec(c: dict[str, object], jd: float, ca_jd: float) -> float:
    rho_rad = angular_separation_rad(c, jd)
    d_es_now = earth_sun_distance_km(c, jd)
    d_es_ca = earth_sun_distance_km(c, ca_jd)
    transverse_screen_km = math.tan(rho_rad) * d_es_now
    equivalent_angle_rad = math.atan2(transverse_screen_km, d_es_ca)
    return equivalent_angle_rad * ARCSEC_PER_RAD


def analyze() -> tuple[pd.DataFrame, dict[str, float | str]]:
    master = build_master()
    c = splines(master)
    ca_jd = solve_ca(c)
    minutes = np.arange(-WINDOW_MIN, WINDOW_MIN + 1, dtype=float)

    rows: list[dict[str, float | str]] = []
    for minute in minutes:
        jd = ca_jd + minute / 1440.0
        venus_rho = rho_arcsec(c, jd)
        earth_norm_rho = earth_normalized_rho_arcsec(c, jd, ca_jd)
        rows.append({
            "minute_from_ca": float(minute),
            "jd_tdb": float(jd),
            "utc": utc_from_jd(jd),
            "venus_rho_arcsec": float(venus_rho),
            "earth_normalized_rho_arcsec": float(earth_norm_rho),
            "earth_sun_distance_km": float(earth_sun_distance_km(c, jd)),
        })

    df = pd.DataFrame(rows)
    venus_min = float(df["venus_rho_arcsec"].min())
    earth_norm_min = float(df["earth_normalized_rho_arcsec"].min())
    df["venus_rho_minus_min_arcsec"] = df["venus_rho_arcsec"] - venus_min
    df["earth_normalized_rho_minus_min_arcsec"] = df["earth_normalized_rho_arcsec"] - earth_norm_min

    stats: dict[str, float | str] = {
        "ca_jd_tdb": ca_jd,
        "ca_utc": utc_from_jd(ca_jd),
        "venus_rho_min_arcsec": venus_min,
        "earth_normalized_rho_min_arcsec": earth_norm_min,
        "earth_sun_distance_ca_km": earth_sun_distance_km(c, ca_jd),
        "venus_minus30_arcsec": float(df.loc[df["minute_from_ca"] == -30.0, "venus_rho_minus_min_arcsec"].iloc[0]),
        "venus_plus30_arcsec": float(df.loc[df["minute_from_ca"] == 30.0, "venus_rho_minus_min_arcsec"].iloc[0]),
        "earth_minus30_arcsec": float(df.loc[df["minute_from_ca"] == -30.0, "earth_normalized_rho_minus_min_arcsec"].iloc[0]),
        "earth_plus30_arcsec": float(df.loc[df["minute_from_ca"] == 30.0, "earth_normalized_rho_minus_min_arcsec"].iloc[0]),
        "samples": len(df),
    }
    return df, stats


def make_plot(df: pd.DataFrame, stats: dict[str, float | str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x = df["minute_from_ca"].to_numpy(float)
    venus = df["venus_rho_minus_min_arcsec"].to_numpy(float)
    earth = df["earth_normalized_rho_minus_min_arcsec"].to_numpy(float)

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.5, 1.25], hspace=0.16, left=0.07, right=0.98, top=0.88, bottom=0.07)
    ax = fig.add_subplot(gs[0, 0], facecolor=BG)
    ax_table = fig.add_subplot(gs[1, 0], facecolor=BG)
    ax_table.axis("off")

    ax.plot(x, venus, color=VENUS_COLOR, lw=1.3, marker="o", markersize=2.8, label="Venus geocentric ρ − ρmin")
    ax.plot(x, earth, color=EARTH_COLOR, lw=1.3, ls="--", marker="x", markersize=3.2, label="Earth-distance-normalized ρ − ρmin")
    ax.axvline(0.0, color=CA_COLOR, lw=0.8, ls="--", alpha=0.85)
    ax.axhline(0.0, color=CA_COLOR, lw=0.55, ls=":", alpha=0.55)
    ax.set_xlim(-30.0, 30.0)
    ax.set_xlabel("Minutes from true geocentric closest approach", color=FG, fontsize=12)
    ax.set_ylabel("ρ − ρmin (arcsec)", color=FG, fontsize=12)
    ax.set_title("Two Geocentric Sun-Screen Parabolas: Venus and Earth-Distance Normalization", color=FG, fontsize=18, fontweight="bold", pad=12)
    ax.grid(True, color=GRID, lw=0.45, alpha=0.55)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color("#A9B7C4")
    legend = ax.legend(loc="upper center", ncol=2, fontsize=10, frameon=True)
    legend.get_frame().set_facecolor("#102035")
    legend.get_frame().set_edgecolor("#36516B")
    for text in legend.get_texts():
        text.set_color(FG)

    table_rows = [
        ["Geocentric closest approach UTC", str(stats["ca_utc"]), "JPL solve; minimum of plotted Venus ρ(t)"],
        ["Venus ρmin", f"{float(stats['venus_rho_min_arcsec']):.12f}", "arcsec"],
        ["Earth-normalized ρmin", f"{float(stats['earth_normalized_rho_min_arcsec']):.12f}", "arcsec at CA Earth–Sun distance"],
        ["Earth–Sun distance at CA", f"{float(stats['earth_sun_distance_ca_km']):,.6f}", "km"],
        ["Samples", str(stats["samples"]), "one-minute samples over ±30 minutes"],
    ]
    table = ax_table.table(
        cellText=table_rows,
        colLabels=["Quantity", "Value", "Unit / trace"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.34, 0.28, 0.38],
        bbox=[0.0, 0.03, 1.0, 0.92],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#47627D")
        cell.set_linewidth(0.55)
        if row == 0:
            cell.set_facecolor(TABLE_HEADER)
            cell.get_text().set_color(FG)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(TABLE_TEAL if row % 2 else TABLE_GOLD)
            cell.get_text().set_color(FG)

    fig.suptitle("1769 Venus Transit — Clean Geocentric ρ − ρmin Comparison", color=FG, fontsize=24, fontweight="bold", y=0.965)
    fig.text(0.5, 0.915, "Fixed geocentric Sun-screen geometry; no ρ², no limb, no surface observer", color="#B8CBD6", ha="center", fontsize=10)
    fig.savefig(PNG, dpi=500, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"Observer: Earth geocenter ({GEOCENTER_LOCATION})")
    print(f"Window: ±{WINDOW_MIN} minutes around solved geocentric closest approach")
    print("Quantities: Venus geocentric rho-rho_min and Earth-distance-normalized rho-rho_min")

    print("COMMENTS")
    print("The Earth-normalized curve converts the same physical Sun-screen offset to the fixed Earth-Sun distance at closest approach.")
    print("No rho-squared quantity is used anywhere in the calculation or plot.")

    df, stats = analyze()
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False, float_format="%.15f")
    make_plot(df, stats)

    print("RESULTS")
    print(f"Geocentric closest approach UTC: {stats['ca_utc']}")
    print(f"Venus rho minimum: {float(stats['venus_rho_min_arcsec']):.12f} arcsec")
    print(f"Earth-normalized rho minimum: {float(stats['earth_normalized_rho_min_arcsec']):.12f} arcsec")
    print(f"Venus rho excess at -30 min: {float(stats['venus_minus30_arcsec']):.12f} arcsec")
    print(f"Venus rho excess at +30 min: {float(stats['venus_plus30_arcsec']):.12f} arcsec")
    print(f"Earth-normalized rho excess at -30 min: {float(stats['earth_minus30_arcsec']):.12f} arcsec")
    print(f"Earth-normalized rho excess at +30 min: {float(stats['earth_plus30_arcsec']):.12f} arcsec")

    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")

    print("PAPER COMPARISON")
    print("NOT USED: this is a JPL geometric interpretation plot only.")

    print("EQUATION STATUS")
    print("PASS: both plotted curves use rho-rho_min; rho-squared is not used.")
    print("PASS: closest approach is solved from the same geocentric Venus rho(t) plotted here.")

    display(Image(filename=str(PNG)))
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z%z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0109