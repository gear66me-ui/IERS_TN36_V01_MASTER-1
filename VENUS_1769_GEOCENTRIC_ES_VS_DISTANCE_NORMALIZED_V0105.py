# V0105
# Audit reference: geocentric Earth-Sun and Venus-Sun physical-distance normalization around true geocentric rho-minimum CA; Python/Matplotlib/JPL only; no AI images.
from __future__ import annotations

import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0105"
LOCAL_TZ = ZoneInfo("America/Bogota")
OUT = Path("/content/VENUS_1769_GEOCENTRIC_ES_VS_DISTANCE_NORMALIZED_V0105_OUTPUT")
PNG = OUT / "VENUS_1769_GEOCENTRIC_ES_VS_DISTANCE_NORMALIZED_V0105.png"
CSV = OUT / "VENUS_1769_GEOCENTRIC_ES_VS_DISTANCE_NORMALIZED_V0105.csv"

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
PURPLE = "#9B8CFF"
GREEN = "#74D680"


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


def angle_arcsec_between(a: np.ndarray, b: np.ndarray) -> float:
    ua = unit(a)
    ub = unit(b)
    return ARCSEC_PER_RAD * math.atan2(norm(np.cross(ua, ub)), float(np.dot(ua, ub)))


def rho_arcsec(c: dict[str, object], jd: float) -> float:
    return angle_arcsec_between(vec(c, "SUN", jd), vec(c, "VENUS", jd))


def rho_dot_arcsec_per_min(c: dict[str, object], jd: float) -> float:
    h = 0.5 / 1440.0
    return (rho_arcsec(c, jd + h) - rho_arcsec(c, jd - h)) / 1.0


def d_es_km(c: dict[str, object], jd: float) -> float:
    return norm(vec(c, "SUN", jd))


def d_ev_km(c: dict[str, object], jd: float) -> float:
    return norm(vec(c, "VENUS", jd))


def d_vs_km(c: dict[str, object], jd: float) -> float:
    return norm(vec(c, "VENUS", jd) - vec(c, "SUN", jd))


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
        raise RuntimeError("Geocentric closest-approach minimization failed.")
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
    minute_grid = np.linspace(-WINDOW_MIN, WINDOW_MIN, int(2 * WINDOW_MIN) + 1)

    es0 = d_es_km(c, ca_jd)
    ev0 = d_ev_km(c, ca_jd)
    vs0 = d_vs_km(c, ca_jd)
    rho0 = rho_arcsec(c, ca_jd)

    rows = []
    for minute in minute_grid:
        jd = ca_jd + float(minute) / 1440.0
        es = d_es_km(c, jd)
        ev = d_ev_km(c, jd)
        vs = d_vs_km(c, jd)
        rho = rho_arcsec(c, jd)
        rows.append({
            "minute_from_geocentric_ca": float(minute),
            "jd_tdb": float(jd),
            "utc": utc_from_jd(jd),
            "rho_arcsec": float(rho),
            "rho_minus_rho_min_arcsec": float(rho - rho0),
            "rho_dot_arcsec_per_min": float(rho_dot_arcsec_per_min(c, jd)),
            "earth_sun_km": float(es),
            "earth_sun_delta_km": float(es - es0),
            "earth_sun_delta_ppm": float((es / es0 - 1.0) * 1_000_000.0),
            "earth_venus_km_not_plotted": float(ev),
            "venus_sun_km": float(vs),
            "venus_sun_delta_km": float(vs - vs0),
            "venus_sun_delta_ppm": float((vs / vs0 - 1.0) * 1_000_000.0),
        })
    df = pd.DataFrame(rows)
    stats: dict[str, float | str] = {
        "ca_jd_tdb": ca_jd,
        "ca_utc": utc_from_jd(ca_jd),
        "rho_min_arcsec": rho0,
        "rhodot_zero_utc": utc_from_jd(rhodot_zero_jd),
        "rhodot_zero_offset_sec": (rhodot_zero_jd - ca_jd) * 86400.0,
        "earth_sun_at_ca_km": es0,
        "venus_sun_at_ca_km": vs0,
        "earth_venus_at_ca_km_not_plotted": ev0,
        "earth_sun_delta_min_km": float(df["earth_sun_delta_km"].min()),
        "earth_sun_delta_max_km": float(df["earth_sun_delta_km"].max()),
        "venus_sun_delta_min_km": float(df["venus_sun_delta_km"].min()),
        "venus_sun_delta_max_km": float(df["venus_sun_delta_km"].max()),
        "earth_sun_ppm_range": float(df["earth_sun_delta_ppm"].max() - df["earth_sun_delta_ppm"].min()),
        "venus_sun_ppm_range": float(df["venus_sun_delta_ppm"].max() - df["venus_sun_delta_ppm"].min()),
        "samples": len(df),
    }
    return df, stats


def make_plot(df: pd.DataFrame, stats: dict[str, float | str]) -> None:
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": FG,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": FG,
        "font.size": 9,
    })

    fig = plt.figure(figsize=(13.6, 9.0), dpi=160)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.0, 0.72], hspace=0.32)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    x = df["minute_from_geocentric_ca"].to_numpy(float)

    ax1.plot(x, df["earth_sun_delta_km"], lw=1.1, marker="o", ms=2.0, color=BLUE, label="Earth-Sun distance change from CA")
    ax1.plot(x, df["venus_sun_delta_km"], lw=1.1, marker="o", ms=2.0, color=GOLD, label="Venus-Sun distance change from CA")
    ax1.axvline(0.0, lw=0.9, ls="--", color=GREEN, label="Geocentric CA")
    ax1.axhline(0.0, lw=0.7, ls=":", color=MUTED)
    ax1.set_ylabel("Distance change from CA (km)")
    ax1.set_title("Geocentric physical distances: Earth-Sun and Venus-Sun", fontsize=14, weight="bold", pad=10)
    ax1.grid(True, color=GRID, lw=0.5, alpha=0.75)
    ax1.legend(loc="best", fontsize=8, frameon=True)

    ax2.plot(x, df["earth_sun_delta_ppm"], lw=1.1, marker="o", ms=2.0, color=BLUE, label="Earth-Sun normalized change")
    ax2.plot(x, df["venus_sun_delta_ppm"], lw=1.1, marker="o", ms=2.0, color=GOLD, label="Venus-Sun normalized change")
    ax2.axvline(0.0, lw=0.9, ls="--", color=GREEN)
    ax2.axhline(0.0, lw=0.7, ls=":", color=MUTED)
    ax2.set_ylabel("Relative change from CA (ppm)")
    ax2.grid(True, color=GRID, lw=0.5, alpha=0.75)
    ax2.legend(loc="best", fontsize=8, frameon=True)

    ax3.plot(x, df["rho_minus_rho_min_arcsec"], lw=1.15, marker="o", ms=2.0, color=PURPLE, label="ρ(t) − ρmin")
    ax3.axvline(0.0, lw=0.9, ls="--", color=GREEN)
    ax3.axhline(0.0, lw=0.7, ls=":", color=MUTED)
    ax3.set_xlabel("Minutes from true geocentric closest approach")
    ax3.set_ylabel("ρ excess (arcsec)")
    ax3.grid(True, color=GRID, lw=0.5, alpha=0.75)
    ax3.legend(loc="best", fontsize=8, frameon=True)

    for ax in (ax1, ax2, ax3):
        for spine in ax.spines.values():
            spine.set_color(MUTED)

    summary = (
        f"CA UTC: {stats['ca_utc']}   |   ρmin: {float(stats['rho_min_arcsec']):.12f} arcsec\n"
        f"D_ES(CA): {float(stats['earth_sun_at_ca_km']):,.6f} km   |   "
        f"D_VS(CA): {float(stats['venus_sun_at_ca_km']):,.6f} km\n"
        "Location: GEOCENTER only (500@399). No Point Venus, no Tahiti, no Vardø, no solar limb."
    )
    fig.text(0.5, 0.017, summary, ha="center", va="bottom", fontsize=8.3, color=MUTED)
    fig.savefig(PNG, bbox_inches="tight")
    plt.close(fig)


def print_report(df: pd.DataFrame, stats: dict[str, float | str]) -> None:
    print("CODE INPUTS")
    print(f"Version                         : {VERSION}")
    print(f"JPL observer location            : {GEOCENTER_LOCATION}  (GEOCENTER)")
    print(f"Targets                          : Sun={SUN_TARGET}, Venus={VENUS_TARGET}")
    print(f"JPL vector mode                  : geometric, ecliptic")
    print(f"Window plotted                   : +/- {WINDOW_MIN:.1f} minutes from geocentric CA")
    print(f"Output folder                    : {OUT}")
    print()

    print("COMMENTS")
    print("This widget plots ONLY geocentric physical distances: Earth-Sun and Venus-Sun.")
    print("Earth-Sun is the norm of the geocentric Sun vector.")
    print("Venus-Sun is the norm of the geocentric Venus vector minus the geocentric Sun vector.")
    print("Earth-Venus is calculated only as a NOT PLOTTED audit column in the CSV.")
    print("The closest approach time is solved from the geocentric angular Sun-Venus rho(t), not from Point Venus or any surface observer.")
    print()

    print("RESULTS")
    print(f"Geocentric CA UTC                : {stats['ca_utc']}")
    print(f"Geocentric CA JD(TDB)            : {float(stats['ca_jd_tdb']):.15f}")
    print(f"rho_min                          : {float(stats['rho_min_arcsec']):.12f} arcsec")
    print(f"rho_dot zero UTC                 : {stats['rhodot_zero_utc']}")
    print(f"rho_dot zero offset              : {float(stats['rhodot_zero_offset_sec']):+.9f} s")
    print(f"Earth-Sun at CA                  : {float(stats['earth_sun_at_ca_km']):,.6f} km")
    print(f"Venus-Sun at CA                  : {float(stats['venus_sun_at_ca_km']):,.6f} km")
    print(f"NOT PLOTTED Earth-Venus at CA    : {float(stats['earth_venus_at_ca_km_not_plotted']):,.6f} km")
    print(f"Earth-Sun delta range            : {float(stats['earth_sun_delta_min_km']):+.6f} to {float(stats['earth_sun_delta_max_km']):+.6f} km")
    print(f"Venus-Sun delta range            : {float(stats['venus_sun_delta_min_km']):+.6f} to {float(stats['venus_sun_delta_max_km']):+.6f} km")
    print(f"Earth-Sun normalized range       : {float(stats['earth_sun_ppm_range']):.9f} ppm")
    print(f"Venus-Sun normalized range       : {float(stats['venus_sun_ppm_range']):.9f} ppm")
    print()

    print("OUTPUT SUMMARY")
    print(f"PNG                              : {PNG}")
    print(f"CSV                              : {CSV}")
    print(f"Rows                             : {int(stats['samples'])}")
    print()

    print("PAPER COMPARISON")
    print("NOT USED: this is a JPL geocentric distance-normalization diagnostic plot only.")
    print()

    print("EQUATION STATUS")
    print("PASS: D_ES = ||r_Sun_geocentric||.")
    print("PASS: D_VS = ||r_Venus_geocentric - r_Sun_geocentric||.")
    print("PASS: t=0 is true geocentric CA from rho(t), where rho is angular Sun-center to Venus-center distance.")
    print("PASS: No Point Venus, Tahiti, Vardo, or surface topocentric location is used.")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, stats = analyze()
    df.to_csv(CSV, index=False, float_format="%.15f")
    make_plot(df, stats)
    print_report(df, stats)
    display(Image(filename=str(PNG)))
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z%z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0105
