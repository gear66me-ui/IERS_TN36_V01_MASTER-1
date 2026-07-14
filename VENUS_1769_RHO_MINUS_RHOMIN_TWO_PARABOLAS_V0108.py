# V0108
# Audit reference: geocentric tangent-plane rho-minus-rhomin only; two panels; two curves; no rho-squared; no AI images.
from __future__ import annotations

import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0108"
LOCAL_TZ = ZoneInfo("America/Bogota")
OUT = Path("/content/VENUS_1769_RHO_MINUS_RHOMIN_TWO_PARABOLAS_V0108_OUTPUT")
PNG = OUT / "VENUS_1769_RHO_MINUS_RHOMIN_TWO_PARABOLAS_V0108.png"
CSV = OUT / "VENUS_1769_RHO_MINUS_RHOMIN_TWO_PARABOLAS_V0108.csv"

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
RED = "#FF6B6B"
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
    cross = norm(np.cross(ua, ub))
    dot = float(np.dot(ua, ub))
    return ARCSEC_PER_RAD * math.atan2(cross, dot)


def rho_venus_from_earth_arcsec(c: dict[str, object], jd: float) -> float:
    # Apparent angular Sun-center to Venus-center separation as seen from Earth geocenter.
    return angle_arcsec_between(vec(c, "SUN", jd), vec(c, "VENUS", jd))


def rho_earth_from_venus_arcsec(c: dict[str, object], jd: float) -> float:
    # Apparent angular Sun-center to Earth-center separation as seen from Venus center.
    # This uses the same geocentric JPL vectors, transformed algebraically; no surface observer is used.
    sun_from_earth = vec(c, "SUN", jd)
    venus_from_earth = vec(c, "VENUS", jd)
    earth_from_venus = -venus_from_earth
    sun_from_venus = sun_from_earth - venus_from_earth
    return angle_arcsec_between(sun_from_venus, earth_from_venus)


def solve_ca(c: dict[str, object]) -> float:
    jds = np.asarray(c["JD_TDB"], dtype=float)
    samples = np.array([rho_venus_from_earth_arcsec(c, float(jd)) for jd in jds], dtype=float)
    i = int(np.argmin(samples))
    lo = float(jds[max(0, i - 2)])
    hi = float(jds[min(len(jds) - 1, i + 2)])
    res = minimize_scalar(lambda x: rho_venus_from_earth_arcsec(c, float(x)), bounds=(lo, hi), method="bounded", options={"xatol": 1e-13, "maxiter": 600})
    if not res.success:
        raise RuntimeError("Closest-approach minimization failed.")
    return float(res.x)


def normalized_excess(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    mn = float(np.min(v))
    ex = v - mn
    mx = float(np.max(ex))
    if mx <= 0.0:
        return np.zeros_like(ex)
    return ex / mx


def analyze() -> tuple[pd.DataFrame, dict[str, float | str]]:
    master = build_master()
    c = splines(master)
    ca_jd = solve_ca(c)
    minutes = np.linspace(-WINDOW_MIN, WINDOW_MIN, int(2 * WINDOW_MIN) + 1)
    rows = []
    for minute in minutes:
        jd = ca_jd + float(minute) / 1440.0
        rho_v = rho_venus_from_earth_arcsec(c, jd)
        rho_e = rho_earth_from_venus_arcsec(c, jd)
        rows.append({
            "minute_from_venus_geocentric_ca": float(minute),
            "jd_tdb": float(jd),
            "utc": utc_from_jd(jd),
            "rho_venus_seen_from_earth_arcsec": float(rho_v),
            "rho_earth_seen_from_venus_arcsec": float(rho_e),
        })
    df = pd.DataFrame(rows)
    df["venus_rho_minus_rhomin_arcsec"] = df["rho_venus_seen_from_earth_arcsec"] - float(df["rho_venus_seen_from_earth_arcsec"].min())
    df["earth_rho_minus_rhomin_arcsec"] = df["rho_earth_seen_from_venus_arcsec"] - float(df["rho_earth_seen_from_venus_arcsec"].min())
    df["venus_rho_minus_rhomin_normalized"] = normalized_excess(df["rho_venus_seen_from_earth_arcsec"].to_numpy(float))
    df["earth_rho_minus_rhomin_normalized"] = normalized_excess(df["rho_earth_seen_from_venus_arcsec"].to_numpy(float))
    stats: dict[str, float | str] = {
        "ca_utc": utc_from_jd(ca_jd),
        "ca_jd_tdb": ca_jd,
        "venus_rhomin_arcsec": float(df["rho_venus_seen_from_earth_arcsec"].min()),
        "earth_rhomin_arcsec": float(df["rho_earth_seen_from_venus_arcsec"].min()),
        "venus_excess_plus30_arcsec": float(df.loc[df["minute_from_venus_geocentric_ca"] == 30.0, "venus_rho_minus_rhomin_arcsec"].iloc[0]),
        "earth_excess_plus30_arcsec": float(df.loc[df["minute_from_venus_geocentric_ca"] == 30.0, "earth_rho_minus_rhomin_arcsec"].iloc[0]),
        "samples": len(df),
    }
    return df, stats


def style_axes(ax) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.6)
    ax.grid(True, color=GRID, lw=0.35, alpha=0.45)
    ax.axvline(0.0, color=MUTED, lw=0.7, ls="--")
    ax.axhline(0.0, color=MUTED, lw=0.45, ls=":")


def make_plot(df: pd.DataFrame, stats: dict[str, float | str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False, float_format="%.15f")

    x = df["minute_from_venus_geocentric_ca"].to_numpy(float)
    fig = plt.figure(figsize=(13.8, 8.2), dpi=180, facecolor=BG)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.42], hspace=0.22, left=0.07, right=0.985, top=0.89, bottom=0.08)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax_tbl = fig.add_subplot(gs[2, 0])

    style_axes(ax1)
    style_axes(ax2)

    ax1.plot(x, df["venus_rho_minus_rhomin_arcsec"], color=BLUE, lw=1.0, marker="o", ms=2.2, markevery=3, label="Venus: ρ − ρmin, seen from Earth geocenter")
    ax1.plot(x, df["earth_rho_minus_rhomin_arcsec"], color=GOLD, lw=0.95, ls="--", marker="s", ms=2.0, markevery=3, label="Earth: ρ − ρmin, seen from Venus center")
    ax1.set_ylabel("ρ − ρmin (arcsec)", color=FG, fontsize=9)
    ax1.set_title("Raw angular screen-distance excess: ρ − ρmin only", color=FG, fontsize=13, weight="bold", pad=8)
    leg1 = ax1.legend(loc="upper center", ncols=2, fontsize=8, frameon=True)
    leg1.get_frame().set_facecolor("#102034")
    leg1.get_frame().set_edgecolor("#37516B")
    for txt in leg1.get_texts():
        txt.set_color(FG)

    ax2.plot(x, df["venus_rho_minus_rhomin_normalized"], color=BLUE, lw=1.0, marker="o", ms=2.2, markevery=3, label="Venus normalized")
    ax2.plot(x, df["earth_rho_minus_rhomin_normalized"], color=GOLD, lw=0.95, ls="--", marker="s", ms=2.0, markevery=3, label="Earth normalized")
    ax2.set_ylabel("Normalized excess", color=FG, fontsize=9)
    ax2.set_xlabel("Minutes from Venus geocentric closest approach", color=FG, fontsize=9)
    ax2.set_title("Same two parabolas normalized independently to 0–1", color=FG, fontsize=13, weight="bold", pad=8)
    leg2 = ax2.legend(loc="upper center", ncols=2, fontsize=8, frameon=True)
    leg2.get_frame().set_facecolor("#102034")
    leg2.get_frame().set_edgecolor("#37516B")
    for txt in leg2.get_texts():
        txt.set_color(FG)

    ax_tbl.axis("off")
    ax_tbl.set_facecolor(BG)
    rows = [
        ["Quantity", "Value", "Status"],
        ["CA UTC", str(stats["ca_utc"]), "Venus geocentric ρ minimum"],
        ["Venus ρmin", f"{float(stats['venus_rhomin_arcsec']):.12f}", "arcsec"],
        ["Earth ρmin", f"{float(stats['earth_rhomin_arcsec']):.12f}", "arcsec"],
        ["Venus +30 min excess", f"{float(stats['venus_excess_plus30_arcsec']):.12f}", "arcsec"],
        ["Earth +30 min excess", f"{float(stats['earth_excess_plus30_arcsec']):.12f}", "arcsec"],
        ["Samples", f"{int(stats['samples'])}", "one-minute JPL window"],
    ]
    table = ax_tbl.table(cellText=rows, cellLoc="left", loc="center", colWidths=[0.25, 0.32, 0.43])
    table.auto_set_font_size(False)
    table.set_fontsize(7.8)
    table.scale(1, 1.28)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        cell.set_edgecolor("#7F8FA6")
        cell.get_text().set_color(FG)
        if r == 0:
            cell.set_facecolor(TABLE_HEADER)
            cell.get_text().set_weight("bold")
        elif r in (1, 2, 3):
            cell.set_facecolor(TABLE_TEAL)
        else:
            cell.set_facecolor(TABLE_GOLD)

    fig.suptitle("1769 Venus Transit — Two ρ − ρmin Parabolas Only", color=FG, fontsize=17, weight="bold", y=0.965)
    fig.text(0.5, 0.925, "No ρ²; no raw 3D range lines; geocentric JPL vectors; tangent-plane angular distance excess", color=MUTED, ha="center", fontsize=8.5)
    fig.text(0.5, 0.018, f"File: {Path(__file__).name} | Output: {PNG.name} | CSV: {CSV.name}", color=MUTED, ha="center", fontsize=6.5)
    fig.savefig(PNG, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    display(Image(filename=str(PNG)))


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"Observer/source vectors: geocentric JPL Horizons, location={GEOCENTER_LOCATION}")
    print(f"Window: ±{WINDOW_MIN:.1f} minutes around Venus geocentric closest approach")
    print("Quantities: Venus ρ−ρmin and Earth ρ−ρmin, plus independently normalized versions")
    print("COMMENTS")
    print("This widget removes rho-squared entirely and does not plot raw 3D range distances.")
    print("Top panel reports rho minus rho-min in arcseconds. Bottom panel reports the same two curves normalized 0 to 1.")
    print("RESULTS")
    df, stats = analyze()
    make_plot(df, stats)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:34s}: {value:.12f}")
        else:
            print(f"{key:34s}: {value}")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    print("PAPER COMPARISON")
    print("NOT USED: visualization-only geometry audit; no published values inserted into calculations.")
    print("EQUATION STATUS")
    print("PASS: plotted values are rho-rhomin, not rho^2-rho^2min; normalized panel derives only from same rho-rhomin arrays.")
    local_ts = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(local_ts)
    print(VERSION)


if __name__ == "__main__":
    main()
# V0108
