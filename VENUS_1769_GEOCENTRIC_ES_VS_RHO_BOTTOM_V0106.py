# V0106
# Audit reference: Geocentric Earth-Sun and Venus-Sun distance plot with geocentric Venus-Sun rho minus rho-min bottom panel; no topocentric observer.
from __future__ import annotations

import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0106"
LOCAL_TZ = ZoneInfo("America/Bogota")
OUT = Path("/content/VENUS_1769_GEOCENTRIC_ES_VS_RHO_BOTTOM_V0106_OUTPUT")
PNG = OUT / "VENUS_1769_GEOCENTRIC_ES_VS_RHO_BOTTOM_V0106.png"
CSV = OUT / "VENUS_1769_GEOCENTRIC_ES_VS_RHO_BOTTOM_V0106.csv"

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
BLUE = "#4CC9F0"
ORANGE = "#F8961E"
PURPLE = "#B18CFF"
GREEN = "#84DCC6"


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


def d_es_km(c: dict[str, object], jd: float) -> float:
    return norm(vec(c, "SUN", jd))


def d_vs_km(c: dict[str, object], jd: float) -> float:
    return norm(vec(c, "VENUS", jd) - vec(c, "SUN", jd))


def rho_arcsec(c: dict[str, object], jd: float) -> float:
    sun_u = unit(vec(c, "SUN", jd))
    venus_u = unit(vec(c, "VENUS", jd))
    cross = norm(np.cross(sun_u, venus_u))
    dot = float(np.dot(sun_u, venus_u))
    return ARCSEC_PER_RAD * math.atan2(cross, dot)


def rho_dot_arcsec_per_min(c: dict[str, object], jd: float) -> float:
    h = 0.5 / 1440.0
    return rho_arcsec(c, jd + h) - rho_arcsec(c, jd - h)


def solve_ca(c: dict[str, object]) -> float:
    jds = np.asarray(c["JD_TDB"], dtype=float)
    rho_samples = np.array([rho_arcsec(c, float(jd)) for jd in jds], dtype=float)
    i = int(np.argmin(rho_samples))
    lo = float(jds[max(0, i - 2)])
    hi = float(jds[min(len(jds) - 1, i + 2)])
    res = minimize_scalar(
        lambda x: rho_arcsec(c, float(x)),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-13, "maxiter": 600},
    )
    if not res.success:
        raise RuntimeError("Closest-approach minimization failed for geocentric rho(t).")
    return float(res.x)


def solve_rhodot_zero(c: dict[str, object], ca_jd: float) -> float:
    lo = ca_jd - 5.0 / 1440.0
    hi = ca_jd + 5.0 / 1440.0
    f_lo = rho_dot_arcsec_per_min(c, lo)
    f_hi = rho_dot_arcsec_per_min(c, hi)
    if f_lo * f_hi <= 0.0:
        return float(brentq(lambda x: rho_dot_arcsec_per_min(c, float(x)), lo, hi, xtol=1e-13, rtol=1e-13, maxiter=100))
    return ca_jd


def analyze() -> tuple[pd.DataFrame, dict[str, float | str]]:
    master = build_master()
    c = splines(master)
    ca_jd = solve_ca(c)
    rhodot_zero_jd = solve_rhodot_zero(c, ca_jd)
    rho_min = rho_arcsec(c, ca_jd)
    es0 = d_es_km(c, ca_jd)
    vs0 = d_vs_km(c, ca_jd)

    minutes = np.linspace(-WINDOW_MIN, WINDOW_MIN, int(2 * WINDOW_MIN) + 1)
    rows = []
    for m in minutes:
        jd = ca_jd + float(m) / 1440.0
        es = d_es_km(c, jd)
        vs = d_vs_km(c, jd)
        rho = rho_arcsec(c, jd)
        rows.append({
            "minute_from_geocentric_ca": float(m),
            "jd_tdb": float(jd),
            "utc": utc_from_jd(jd),
            "earth_sun_km": float(es),
            "venus_sun_km": float(vs),
            "earth_sun_delta_km": float(es - es0),
            "venus_sun_delta_km": float(vs - vs0),
            "earth_sun_delta_ppm": float((es / es0 - 1.0) * 1_000_000.0),
            "venus_sun_delta_ppm": float((vs / vs0 - 1.0) * 1_000_000.0),
            "rho_arcsec": float(rho),
            "rho_minus_rho_min_arcsec": float(rho - rho_min),
            "rho_dot_arcsec_per_min": float(rho_dot_arcsec_per_min(c, jd)),
        })
    df = pd.DataFrame(rows)
    stats: dict[str, float | str] = {
        "ca_jd_tdb": ca_jd,
        "ca_utc": utc_from_jd(ca_jd),
        "rho_min_arcsec": rho_min,
        "rhodot_zero_utc": utc_from_jd(rhodot_zero_jd),
        "rhodot_zero_offset_sec": (rhodot_zero_jd - ca_jd) * 86400.0,
        "earth_sun_ca_km": es0,
        "venus_sun_ca_km": vs0,
        "earth_sun_ppm_range": float(df["earth_sun_delta_ppm"].max() - df["earth_sun_delta_ppm"].min()),
        "venus_sun_ppm_range": float(df["venus_sun_delta_ppm"].max() - df["venus_sun_delta_ppm"].min()),
        "rho_minus_min_plus30_arcsec": float(df.loc[df["minute_from_geocentric_ca"] == 30.0, "rho_minus_rho_min_arcsec"].iloc[0]),
        "rho_minus_min_minus30_arcsec": float(df.loc[df["minute_from_geocentric_ca"] == -30.0, "rho_minus_rho_min_arcsec"].iloc[0]),
    }
    return df, stats


def set_dark(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.6)
    ax.grid(True, color=GRID, linewidth=0.35, alpha=0.72)


def make_plot(df: pd.DataFrame, stats: dict[str, float | str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.6, 8.2), dpi=180, facecolor=BG)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.55], hspace=0.20)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0])

    for ax in (ax1, ax2, ax3):
        set_dark(ax)

    x = df["minute_from_geocentric_ca"].to_numpy(float)
    ax1.plot(x, df["earth_sun_delta_ppm"], linewidth=0.90, marker=".", markersize=2.3, color=BLUE, label="Earth-Sun distance: ΔD/D_CA (ppm)")
    ax1.plot(x, df["venus_sun_delta_ppm"], linewidth=0.90, marker=".", markersize=2.3, color=ORANGE, label="Venus-Sun distance: ΔD/D_CA (ppm)")
    ax1.axvline(0.0, color=GREEN, linewidth=0.8, linestyle="--", label="Geocentric CA")
    ax1.axhline(0.0, color=MUTED, linewidth=0.55, linestyle=":")
    ax1.set_ylabel("Distance change (ppm)", color=FG, fontsize=9)
    ax1.legend(loc="best", fontsize=7.8, facecolor=BG, edgecolor=GRID, labelcolor=FG)
    ax1.set_title("Geocentric physical Sun distances: Earth-Sun and Venus-Sun", color=FG, fontsize=12, weight="bold")

    ax2.plot(x, df["rho_minus_rho_min_arcsec"], linewidth=1.05, marker=".", markersize=2.5, color=PURPLE, label="Geocentric Venus-Sun ρ(t) − ρmin")
    ax2.axvline(0.0, color=GREEN, linewidth=0.8, linestyle="--", label="Geocentric CA")
    ax2.axhline(0.0, color=MUTED, linewidth=0.55, linestyle=":")
    ax2.set_ylabel("ρ(t) − ρmin (arcsec)", color=FG, fontsize=9)
    ax2.legend(loc="best", fontsize=7.8, facecolor=BG, edgecolor=GRID, labelcolor=FG)
    ax2.set_title("Geocentric apparent Venus-Sun separation bowl", color=FG, fontsize=12, weight="bold")

    ax3.axis("off")
    rows = [
        ["Version", VERSION],
        ["Observer", "Earth geocenter only: 500@399"],
        ["CA UTC", str(stats["ca_utc"])],
        ["Minimum ρ", f"{float(stats['rho_min_arcsec']):.12f} arcsec"],
        ["dρ/dt zero offset", f"{float(stats['rhodot_zero_offset_sec']):+.9f} s"],
        ["Earth-Sun at CA", f"{float(stats['earth_sun_ca_km']):,.6f} km"],
        ["Venus-Sun at CA", f"{float(stats['venus_sun_ca_km']):,.6f} km"],
        ["Window", "±30 minutes from geocentric ρ-min CA"],
    ]
    table = ax3.table(cellText=rows, colWidths=[0.28, 0.72], cellLoc="left", bbox=[0.02, 0.03, 0.96, 0.92])
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    for (r, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.45)
        cell.set_facecolor("#0D1829" if r % 2 else "#111E33")
        cell.get_text().set_color(FG)
        if col == 0:
            cell.get_text().set_weight("bold")

    fig.suptitle("1769 Transit Geometry — Geocentric D_ES, D_VS, and ρ(t) − ρmin", color=FG, fontsize=14, weight="bold")
    ax2.set_xlabel("Minutes from geocentric closest approach", color=FG, fontsize=9)
    fig.savefig(PNG, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("CODE INPUTS")
    print(f"Version              : {VERSION}")
    print(f"Observer location    : {GEOCENTER_LOCATION} (Earth geocenter only)")
    print(f"JPL targets          : Sun={SUN_TARGET}, Venus={VENUS_TARGET}")
    print(f"UTC range            : {START} to {STOP}; step={STEP}")
    print(f"Window plotted       : ±{WINDOW_MIN:.1f} min from solved geocentric rho minimum")

    print("COMMENTS")
    print("Computes Earth-Sun and Venus-Sun physical distances from geocentric JPL vectors.")
    print("Computes geocentric apparent Venus-Sun angular separation rho(t) and plots rho(t)-rho_min in the bottom panel.")
    print("No Point Venus, Tahiti, Vardø, topocentric observer, solar limb, or AI image generation is used.")

    df, stats = analyze()
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False)
    make_plot(df, stats)

    print("RESULTS")
    print(f"Geocentric CA UTC        : {stats['ca_utc']}")
    print(f"Geocentric CA JD(TDB)    : {float(stats['ca_jd_tdb']):.15f}")
    print(f"Minimum rho              : {float(stats['rho_min_arcsec']):.12f} arcsec")
    print(f"d rho/dt zero offset     : {float(stats['rhodot_zero_offset_sec']):+.9f} s")
    print(f"Earth-Sun at CA          : {float(stats['earth_sun_ca_km']):,.6f} km")
    print(f"Venus-Sun at CA          : {float(stats['venus_sun_ca_km']):,.6f} km")
    print(f"Earth-Sun ppm range      : {float(stats['earth_sun_ppm_range']):.12f} ppm")
    print(f"Venus-Sun ppm range      : {float(stats['venus_sun_ppm_range']):.12f} ppm")
    print(f"rho-rho_min at -30 min   : {float(stats['rho_minus_min_minus30_arcsec']):.12f} arcsec")
    print(f"rho-rho_min at +30 min   : {float(stats['rho_minus_min_plus30_arcsec']):.12f} arcsec")

    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    try:
        display(Image(filename=str(PNG)))
    except Exception as exc:
        print(f"INLINE DISPLAY WARNING: {exc}")

    print("PAPER COMPARISON")
    print("NOT USED: this is a geocentric JPL geometry diagnostic, not a historical paper comparison.")

    print("EQUATION STATUS")
    print("PASS: rho(t)-rho_min is computed from the same geocentric Venus-Sun angular separation used to solve closest approach.")
    print("PASS: Earth-Sun and Venus-Sun distances are both geocentric physical distances from JPL vectors.")

    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z%z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0106
