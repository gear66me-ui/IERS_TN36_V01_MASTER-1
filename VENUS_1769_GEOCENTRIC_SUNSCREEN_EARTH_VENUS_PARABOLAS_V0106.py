# V0106
# Audit reference: geocentric Sun-screen Earth/Venus tangent-plane normalized parabolic audit; Python/Matplotlib only; no AI images.
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
OUT = Path("/content/VENUS_1769_GEOCENTRIC_SUNSCREEN_EARTH_VENUS_PARABOLAS_V0106_OUTPUT")
PNG = OUT / "VENUS_1769_GEOCENTRIC_SUNSCREEN_EARTH_VENUS_PARABOLAS_V0106.png"
CSV = OUT / "VENUS_1769_GEOCENTRIC_SUNSCREEN_EARTH_VENUS_PARABOLAS_V0106.csv"

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
VENUS_COLOR = "#42D7C3"
EARTH_COLOR = "#D89B18"
ZERO_COLOR = "#FFFFFF"


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
                df[f"{prefix}_{ax.upper()}_AU"] = pd.to_numeric(raw[ax], errors="coerce")
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
        float(c[f"{prefix}_X_AU"](jd)),
        float(c[f"{prefix}_Y_AU"](jd)),
        float(c[f"{prefix}_Z_AU"](jd)),
    ], dtype=float)


def basis_from_axis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zhat = unit(axis)
    ecl_north = np.array([0.0, 0.0, 1.0], dtype=float)
    xhat = np.cross(ecl_north, zhat)
    if norm(xhat) < 1.0e-12:
        xhat = np.cross(np.array([0.0, 1.0, 0.0], dtype=float), zhat)
    xhat = unit(xhat)
    yhat = unit(np.cross(zhat, xhat))
    return xhat, yhat, zhat


def gnomonic_screen_arcsec(vector: np.ndarray, basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    xhat, yhat, zhat = basis
    u = unit(vector)
    den = float(np.dot(u, zhat))
    if abs(den) < 1.0e-14:
        raise RuntimeError("Projection denominator is too small.")
    return np.array([
        ARCSEC_PER_RAD * float(np.dot(u, xhat) / den),
        ARCSEC_PER_RAD * float(np.dot(u, yhat) / den),
    ], dtype=float)


def geocentric_venus_rho_arcsec(c: dict[str, object], jd: float) -> float:
    sun_geo = vec(c, "SUN", jd)
    venus_geo = vec(c, "VENUS", jd)
    basis = basis_from_axis(sun_geo)
    xy = gnomonic_screen_arcsec(venus_geo, basis)
    return norm(xy)


def solve_geocentric_venus_ca(c: dict[str, object]) -> float:
    jds = np.asarray(c["JD_TDB"], dtype=float)
    rho_samples = np.array([geocentric_venus_rho_arcsec(c, float(jd)) for jd in jds], dtype=float)
    i = int(np.argmin(rho_samples))
    lo = float(jds[max(0, i - 2)])
    hi = float(jds[min(len(jds) - 1, i + 2)])
    res = minimize_scalar(
        lambda x: geocentric_venus_rho_arcsec(c, float(x)),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1.0e-13, "maxiter": 600},
    )
    if not res.success:
        raise RuntimeError("Closest approach minimization failed.")
    return float(res.x)


def heliocentric_projected_rhos(c: dict[str, object], jd: float, ca_basis: tuple[np.ndarray, np.ndarray, np.ndarray], ca_earth_xy: np.ndarray, ca_venus_xy: np.ndarray) -> tuple[float, float, float, float]:
    sun_geo = vec(c, "SUN", jd)
    venus_geo = vec(c, "VENUS", jd)
    earth_helio = -sun_geo
    venus_helio = venus_geo - sun_geo
    earth_xy = gnomonic_screen_arcsec(earth_helio, ca_basis) - ca_earth_xy
    venus_xy = gnomonic_screen_arcsec(venus_helio, ca_basis) - ca_venus_xy
    return norm(earth_xy), norm(venus_xy), earth_xy[0], venus_xy[0]


def safe_normalize(series: np.ndarray) -> np.ndarray:
    mx = float(np.nanmax(np.abs(series)))
    if mx <= 0.0:
        return np.zeros_like(series, dtype=float)
    return series / mx


def analyze() -> tuple[pd.DataFrame, dict[str, float | str]]:
    master = build_master()
    c = splines(master)
    ca_jd = solve_geocentric_venus_ca(c)
    ca_sun_geo = vec(c, "SUN", ca_jd)
    ca_basis = basis_from_axis(ca_sun_geo)
    ca_earth_helio = -vec(c, "SUN", ca_jd)
    ca_venus_helio = vec(c, "VENUS", ca_jd) - vec(c, "SUN", ca_jd)
    ca_earth_xy = gnomonic_screen_arcsec(ca_earth_helio, ca_basis)
    ca_venus_xy = gnomonic_screen_arcsec(ca_venus_helio, ca_basis)

    minutes = np.linspace(-WINDOW_MIN, WINDOW_MIN, int(2 * WINDOW_MIN) + 1)
    rows = []
    for m in minutes:
        jd = ca_jd + float(m) / 1440.0
        rho_e, rho_v, s_e, s_v = heliocentric_projected_rhos(c, jd, ca_basis, ca_earth_xy, ca_venus_xy)
        rows.append({
            "minute_from_geocentric_venus_ca": float(m),
            "jd_tdb": float(jd),
            "utc": utc_from_jd(jd),
            "earth_sunscreen_rho_arcsec": float(rho_e),
            "venus_sunscreen_rho_arcsec": float(rho_v),
            "earth_sunscreen_s_arcsec": float(s_e),
            "venus_sunscreen_s_arcsec": float(s_v),
        })
    df = pd.DataFrame(rows)

    earth_rho_min = float(df["earth_sunscreen_rho_arcsec"].min())
    venus_rho_min = float(df["venus_sunscreen_rho_arcsec"].min())
    df["earth_rho_minus_min_arcsec"] = df["earth_sunscreen_rho_arcsec"] - earth_rho_min
    df["venus_rho_minus_min_arcsec"] = df["venus_sunscreen_rho_arcsec"] - venus_rho_min
    df["earth_rho2_minus_min2_arcsec2"] = df["earth_sunscreen_rho_arcsec"] ** 2 - earth_rho_min ** 2
    df["venus_rho2_minus_min2_arcsec2"] = df["venus_sunscreen_rho_arcsec"] ** 2 - venus_rho_min ** 2
    df["earth_rho_minus_min_norm"] = safe_normalize(df["earth_rho_minus_min_arcsec"].to_numpy(float))
    df["venus_rho_minus_min_norm"] = safe_normalize(df["venus_rho_minus_min_arcsec"].to_numpy(float))
    df["earth_rho2_minus_min2_norm"] = safe_normalize(df["earth_rho2_minus_min2_arcsec2"].to_numpy(float))
    df["venus_rho2_minus_min2_norm"] = safe_normalize(df["venus_rho2_minus_min2_arcsec2"].to_numpy(float))

    stats: dict[str, float | str] = {
        "ca_jd_tdb": ca_jd,
        "ca_utc": utc_from_jd(ca_jd),
        "geocentric_venus_rho_min_arcsec": geocentric_venus_rho_arcsec(c, ca_jd),
        "earth_sunscreen_rho_min_arcsec": earth_rho_min,
        "venus_sunscreen_rho_min_arcsec": venus_rho_min,
        "earth_rho_plus30_arcsec": float(df.loc[df["minute_from_geocentric_venus_ca"] == 30.0, "earth_sunscreen_rho_arcsec"].iloc[0]),
        "venus_rho_plus30_arcsec": float(df.loc[df["minute_from_geocentric_venus_ca"] == 30.0, "venus_sunscreen_rho_arcsec"].iloc[0]),
        "earth_rho_minus30_arcsec": float(df.loc[df["minute_from_geocentric_venus_ca"] == -30.0, "earth_sunscreen_rho_arcsec"].iloc[0]),
        "venus_rho_minus30_arcsec": float(df.loc[df["minute_from_geocentric_venus_ca"] == -30.0, "venus_sunscreen_rho_arcsec"].iloc[0]),
        "samples": len(df),
    }
    return df, stats


def make_plot(df: pd.DataFrame, stats: dict[str, float | str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": FG,
        "axes.labelcolor": FG,
        "axes.edgecolor": MUTED,
        "xtick.color": FG,
        "ytick.color": FG,
        "font.size": 9,
    })
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 8.2), sharex=True)
    minutes = df["minute_from_geocentric_venus_ca"].to_numpy(float)

    ax = axes[0]
    ax.plot(minutes, df["venus_rho2_minus_min2_norm"], lw=1.2, marker="o", ms=2.0, color=VENUS_COLOR, label="Venus: normalized ρ² − ρ²min")
    ax.plot(minutes, df["earth_rho2_minus_min2_norm"], lw=1.2, marker="o", ms=2.0, color=EARTH_COLOR, label="Earth: normalized ρ² − ρ²min")
    ax.axvline(0.0, lw=0.8, ls="--", color=ZERO_COLOR)
    ax.axhline(0.0, lw=0.6, ls=":", color=MUTED)
    ax.grid(True, lw=0.35, color=GRID, alpha=0.85)
    ax.set_ylabel("Normalized parabolic excess")
    ax.set_title("Geocentric fixed Sun-screen tangent plane: Earth and Venus parabolic screen-radius excess", fontsize=12, weight="bold")
    ax.legend(loc="upper center", ncols=2, fontsize=8, frameon=True, facecolor="#0F172A", edgecolor=GRID)

    ax = axes[1]
    ax.plot(minutes, df["venus_rho_minus_min_norm"], lw=1.2, marker="o", ms=2.0, color=VENUS_COLOR, label="Venus: normalized ρ − ρmin")
    ax.plot(minutes, df["earth_rho_minus_min_norm"], lw=1.2, marker="o", ms=2.0, color=EARTH_COLOR, label="Earth: normalized ρ − ρmin")
    ax.axvline(0.0, lw=0.8, ls="--", color=ZERO_COLOR)
    ax.axhline(0.0, lw=0.6, ls=":", color=MUTED)
    ax.grid(True, lw=0.35, color=GRID, alpha=0.85)
    ax.set_xlabel("Minutes from geocentric Venus closest approach")
    ax.set_ylabel("Normalized angular excess")
    ax.set_title("Raw screen-radius excess on same fixed Sun-screen", fontsize=12, weight="bold")
    ax.legend(loc="upper center", ncols=2, fontsize=8, frameon=True, facecolor="#0F172A", edgecolor=GRID)

    note = (
        f"CA UTC {stats['ca_utc']}  |  "
        f"Venus geocentric ρmin {float(stats['geocentric_venus_rho_min_arcsec']):.9f} arcsec  |  "
        "Fixed tangent plane: Sun direction at geocentric Venus CA"
    )
    fig.suptitle("1769 Venus Transit — Geocentric Earth/Venus Sun-Screen Normalized Curves", fontsize=14, weight="bold", y=0.985)
    fig.text(0.5, 0.02, note, ha="center", va="bottom", color=MUTED, fontsize=8)
    fig.tight_layout(rect=[0.035, 0.055, 0.985, 0.955])
    fig.savefig(PNG, dpi=240)
    plt.close(fig)


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"Observer frame: geocentric JPL Horizons location {GEOCENTER_LOCATION}")
    print(f"Targets used: Sun={SUN_TARGET}, Venus={VENUS_TARGET}; Earth is derived as -Sun geocentric vector")
    print(f"Time range: {START} to {STOP}; step={STEP}; plot window=±{WINDOW_MIN:.1f} min")
    print("COMMENTS")
    print("No Point Venus, Tahiti, Vardo, or surface observer is used.")
    print("No solar limb is plotted.")
    print("The fixed Sun-screen tangent plane is defined by the geocentric Sun direction at Venus closest approach.")
    print("Earth and Venus are projected as heliocentric vectors onto that same fixed Sun-screen plane and normalized to their own minima.")
    df, stats = analyze()
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False, float_format="%.15f")
    make_plot(df, stats)
    print("RESULTS")
    print(f"Geocentric Venus CA UTC                 : {stats['ca_utc']}")
    print(f"Geocentric Venus CA JD_TDB              : {float(stats['ca_jd_tdb']):.15f}")
    print(f"Geocentric Venus apparent ρmin          : {float(stats['geocentric_venus_rho_min_arcsec']):.12f} arcsec")
    print(f"Earth fixed-Sun-screen ρmin             : {float(stats['earth_sunscreen_rho_min_arcsec']):.12f} arcsec")
    print(f"Venus fixed-Sun-screen heliocentric ρmin: {float(stats['venus_sunscreen_rho_min_arcsec']):.12f} arcsec")
    print(f"Earth fixed-Sun-screen ρ(-30m)          : {float(stats['earth_rho_minus30_arcsec']):.12f} arcsec")
    print(f"Earth fixed-Sun-screen ρ(+30m)          : {float(stats['earth_rho_plus30_arcsec']):.12f} arcsec")
    print(f"Venus fixed-Sun-screen ρ(-30m)          : {float(stats['venus_rho_minus30_arcsec']):.12f} arcsec")
    print(f"Venus fixed-Sun-screen ρ(+30m)          : {float(stats['venus_rho_plus30_arcsec']):.12f} arcsec")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    try:
        display(Image(filename=str(PNG)))
    except Exception:
        pass
    print("PAPER COMPARISON")
    print("NOT USED: this is a geocentric tangent-plane interpretation audit, not a paper-value comparison.")
    print("EQUATION STATUS")
    print("PASS: all plotted curves are derived from JPL vectors and normalized from calculated minima only.")
    print("PASS: plotted Earth is derived geocentrically from the negative Sun vector; plotted Venus is derived from the Sun-to-Venus heliocentric vector.")
    local_ts = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(local_ts)
    print(VERSION)


if __name__ == "__main__":
    main()
# V0106
