# V0103
# Audit reference: standalone tangent-plane Sun-screen plot around geocentric closest approach; Python/Matplotlib/JPL only; no AI images.
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
OUT = Path("/content/VENUS_1769_TANGENT_PLANE_SCREEN_V0103_OUTPUT")
PNG = OUT / "VENUS_1769_TANGENT_PLANE_SCREEN_V0103.png"
CSV = OUT / "VENUS_1769_TANGENT_PLANE_SCREEN_V0103.csv"

ARCSEC_PER_RAD = 206_264.80624709636
AU_KM = 149_597_870.700000
SUN_RADIUS_KM = 695_700.0
START = "1769-06-03 21:00"
STOP = "1769-06-03 23:30"
STEP = "1m"
GEOCENTER_LOCATION = "500@399"
SUN_TARGET = "10"
VENUS_TARGET = "299"
WINDOW_MIN = 30.0

BG = "#000000"
FG = "#F8FAFC"
MUTED = "#B8CBD6"
GRID = "#263A4B"
BLUE = "#42D7C3"
GOLD = "#D89B18"
GREEN = "#74D680"
RED = "#FF6B6B"
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
from matplotlib.patches import Circle
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
    if len(master) < 120:
        raise RuntimeError(f"Insufficient JPL samples: {len(master)}")
    return master


def build_splines(df: pd.DataFrame) -> dict[str, object]:
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


def sun_basis(c: dict[str, object], jd: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sun = vec(c, "SUN", jd)
    zhat = unit(sun)
    ecliptic_north = np.array([0.0, 0.0, 1.0], dtype=float)
    xhat = np.cross(ecliptic_north, zhat)
    if norm(xhat) < 1.0e-14:
        xhat = np.cross(np.array([0.0, 1.0, 0.0], dtype=float), zhat)
    xhat = unit(xhat)
    yhat = unit(np.cross(zhat, xhat))
    return xhat, yhat, zhat


def tangent_xy_arcsec(c: dict[str, object], jd: float) -> tuple[float, float, float]:
    xhat, yhat, zhat = sun_basis(c, jd)
    vhat = unit(vec(c, "VENUS", jd))
    denom = float(np.dot(vhat, zhat))
    x = ARCSEC_PER_RAD * float(np.dot(vhat, xhat) / denom)
    y = ARCSEC_PER_RAD * float(np.dot(vhat, yhat) / denom)
    rho = float(math.hypot(x, y))
    return x, y, rho


def rho_arcsec(c: dict[str, object], jd: float) -> float:
    return tangent_xy_arcsec(c, jd)[2]


def solve_ca(c: dict[str, object]) -> float:
    jds = np.asarray(c["JD_TDB"], dtype=float)
    samples = np.array([rho_arcsec(c, float(jd)) for jd in jds], dtype=float)
    i = int(np.argmin(samples))
    lo = float(jds[max(0, i - 3)])
    hi = float(jds[min(len(jds) - 1, i + 3)])
    res = minimize_scalar(lambda x: rho_arcsec(c, float(x)), bounds=(lo, hi), method="bounded", options={"xatol": 1e-13, "maxiter": 600})
    if not res.success:
        raise RuntimeError("Closest-approach minimization failed for plotted rho(t).")
    return float(res.x)


def solar_radius_arcsec(c: dict[str, object], jd: float) -> float:
    es = norm(vec(c, "SUN", jd))
    return ARCSEC_PER_RAD * math.asin(SUN_RADIUS_KM / es)


def finite_velocity_xy(c: dict[str, object], jd: float, half_seconds: float = 30.0) -> np.ndarray:
    h = half_seconds / 86400.0
    x1, y1, _ = tangent_xy_arcsec(c, jd - h)
    x2, y2, _ = tangent_xy_arcsec(c, jd + h)
    return np.array([(x2 - x1) / (2.0 * half_seconds), (y2 - y1) / (2.0 * half_seconds)], dtype=float)


def analyze() -> tuple[pd.DataFrame, dict[str, float | str | np.ndarray]]:
    master = build_master()
    c = build_splines(master)
    ca_jd = solve_ca(c)
    ca_utc = utc_from_jd(ca_jd)
    x0, y0, rho0 = tangent_xy_arcsec(c, ca_jd)
    t_hat = unit(finite_velocity_xy(c, ca_jd, 30.0))
    n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)
    if float(np.dot(np.array([x0, y0]), n_hat)) < 0.0:
        n_hat *= -1.0
    minutes = np.linspace(-WINDOW_MIN, WINDOW_MIN, int(2 * WINDOW_MIN * 4) + 1)
    rows = []
    for m in minutes:
        jd = ca_jd + float(m) / 1440.0
        x, y, rho = tangent_xy_arcsec(c, jd)
        xy = np.array([x, y], dtype=float)
        xy_ca = np.array([x0, y0], dtype=float)
        dxy = xy - xy_ca
        s_centered = float(np.dot(dxy, t_hat))
        n_centered = float(np.dot(dxy, n_hat))
        n_absolute = float(np.dot(xy, n_hat))
        rows.append({
            "minute_from_ca": float(m),
            "jd_tdb": float(jd),
            "utc": utc_from_jd(jd),
            "x_arcsec": float(x),
            "y_arcsec": float(y),
            "rho_arcsec": float(rho),
            "rho_minus_min_arcsec": float(rho - rho0),
            "s_centered_tangent_arcsec": s_centered,
            "n_centered_normal_arcsec": n_centered,
            "n_absolute_normal_arcsec": n_absolute,
        })
    df = pd.DataFrame(rows)
    stats: dict[str, float | str | np.ndarray] = {
        "ca_jd_tdb": ca_jd,
        "ca_utc": ca_utc,
        "x_ca_arcsec": x0,
        "y_ca_arcsec": y0,
        "rho_min_arcsec": rho0,
        "solar_radius_arcsec": solar_radius_arcsec(c, ca_jd),
        "track_angle_deg": math.degrees(math.atan2(float(t_hat[1]), float(t_hat[0]))),
        "normal_at_ca_arcsec": float(np.dot(np.array([x0, y0], dtype=float), n_hat)),
        "tangent_centered_span_arcsec": float(df["s_centered_tangent_arcsec"].max() - df["s_centered_tangent_arcsec"].min()),
        "normal_centered_span_arcsec": float(df["n_centered_normal_arcsec"].max() - df["n_centered_normal_arcsec"].min()),
        "rho_excess_plus30_arcsec": float(df.loc[df["minute_from_ca"] == 30.0, "rho_minus_min_arcsec"].iloc[0]),
        "rho_excess_minus30_arcsec": float(df.loc[df["minute_from_ca"] == -30.0, "rho_minus_min_arcsec"].iloc[0]),
    }
    return df, stats


def style_axis(ax) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.8)
    ax.grid(True, color=GRID, alpha=0.65, linewidth=0.35)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def draw(df: pd.DataFrame, stats: dict[str, float | str | np.ndarray]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False, float_format="%.15f")
    fig = plt.figure(figsize=(14.0, 8.5), facecolor=BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0], left=0.055, right=0.985, top=0.90, bottom=0.08, wspace=0.22, hspace=0.28)
    ax_screen = fig.add_subplot(gs[:, 0])
    ax_comp = fig.add_subplot(gs[0, 1])
    ax_rho = fig.add_subplot(gs[1, 1])
    for ax in (ax_screen, ax_comp, ax_rho):
        style_axis(ax)
    ax_screen.set_aspect("equal", adjustable="box")
    radius = float(stats["solar_radius_arcsec"])
    ax_screen.add_patch(Circle((0.0, 0.0), radius, fill=False, edgecolor=GOLD, linewidth=0.95, alpha=0.95))
    ax_screen.plot(df["x_arcsec"], df["y_arcsec"], color=BLUE, linewidth=1.0, label="Venus center track")
    i0 = int(np.argmin(np.abs(df["minute_from_ca"].to_numpy(float))))
    x0 = float(df.loc[i0, "x_arcsec"])
    y0 = float(df.loc[i0, "y_arcsec"])
    ax_screen.scatter([x0], [y0], s=18, marker="o", color=RED, zorder=5, label="Closest approach")
    ax_screen.scatter([float(df["x_arcsec"].iloc[0])], [float(df["y_arcsec"].iloc[0])], s=12, marker="o", color=GREEN, zorder=4, label="−30 min")
    ax_screen.scatter([float(df["x_arcsec"].iloc[-1])], [float(df["y_arcsec"].iloc[-1])], s=12, marker="o", color=PURPLE, zorder=4, label="+30 min")
    # Local tangent and normal through closest approach.
    half = 145.0
    s = df["s_centered_tangent_arcsec"].to_numpy(float)
    n = df["n_centered_normal_arcsec"].to_numpy(float)
    x = df["x_arcsec"].to_numpy(float)
    y = df["y_arcsec"].to_numpy(float)
    # Estimate tangent/normal from centered coordinates using endpoint xy displacement.
    v = unit(np.array([x[-1] - x[0], y[-1] - y[0]], dtype=float))
    nn = np.array([-v[1], v[0]], dtype=float)
    ax_screen.plot([x0 - half*v[0], x0 + half*v[0]], [y0 - half*v[1], y0 + half*v[1]], color=MUTED, linewidth=0.75, linestyle="--", label="Tangent at CA")
    ax_screen.plot([x0 - half*nn[0], x0 + half*nn[0]], [y0 - half*nn[1], y0 + half*nn[1]], color=GOLD, linewidth=0.75, linestyle=":", label="Normal at CA")
    ax_screen.set_xlim(-radius * 1.05, radius * 1.05)
    ax_screen.set_ylim(-radius * 1.05, radius * 1.05)
    ax_screen.set_xlabel("Sun-screen tangent-plane X (arcsec)")
    ax_screen.set_ylabel("Sun-screen tangent-plane Y (arcsec)")
    ax_screen.set_title("Sun-screen tangent plane: Venus track across solar disk", fontsize=11, fontweight="bold")
    leg = ax_screen.legend(loc="upper right", fontsize=7.6, frameon=True)
    leg.get_frame().set_facecolor("#08111F")
    leg.get_frame().set_edgecolor(GRID)
    for text in leg.get_texts():
        text.set_color(FG)
    m = df["minute_from_ca"].to_numpy(float)
    ax_comp.plot(m, df["s_centered_tangent_arcsec"], color=BLUE, linewidth=0.95, label="s(t), along tangent")
    ax_comp.plot(m, df["n_centered_normal_arcsec"], color=GOLD, linewidth=0.95, label="n(t) − n(CA), normal drift")
    ax_comp.axvline(0.0, color=RED, linewidth=0.75, linestyle="--")
    ax_comp.axhline(0.0, color=MUTED, linewidth=0.55, linestyle=":")
    ax_comp.set_xlabel("Minutes from geocentric closest approach")
    ax_comp.set_ylabel("Centered coordinate (arcsec)")
    ax_comp.set_title("Rotated tangent/normal coordinates", fontsize=10.5, fontweight="bold")
    leg2 = ax_comp.legend(loc="best", fontsize=7.6, frameon=True)
    leg2.get_frame().set_facecolor("#08111F")
    leg2.get_frame().set_edgecolor(GRID)
    for text in leg2.get_texts():
        text.set_color(FG)
    ax_rho.plot(m, df["rho_minus_min_arcsec"], color=PURPLE, linewidth=1.05, label="ρ(t) − ρmin")
    ax_rho.axvline(0.0, color=RED, linewidth=0.75, linestyle="--", label="CA")
    ax_rho.axhline(0.0, color=MUTED, linewidth=0.55, linestyle=":")
    ax_rho.set_xlabel("Minutes from geocentric closest approach")
    ax_rho.set_ylabel("Angular excess (arcsec)")
    ax_rho.set_title("Parabolic closest-approach bowl on the Sun screen", fontsize=10.5, fontweight="bold")
    leg3 = ax_rho.legend(loc="best", fontsize=7.6, frameon=True)
    leg3.get_frame().set_facecolor("#08111F")
    leg3.get_frame().set_edgecolor(GRID)
    for text in leg3.get_texts():
        text.set_color(FG)
    title = "1769 Venus Transit — Tangent-Plane Sun-Screen Geometry (Geocenter)"
    subtitle = f"CA UTC {stats['ca_utc']}   |   ρmin = {float(stats['rho_min_arcsec']):.12f} arcsec   |   R☉ = {radius:.6f} arcsec"
    fig.suptitle(title, color=FG, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.925, subtitle, ha="center", va="center", color=MUTED, fontsize=9.5)
    fig.savefig(PNG, dpi=220, facecolor=BG)
    plt.close(fig)


def print_sections(df: pd.DataFrame, stats: dict[str, float | str | np.ndarray]) -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"Observer: Geocenter ({GEOCENTER_LOCATION})")
    print(f"JPL window: {START} to {STOP}, step {STEP}")
    print(f"Plot window: ±{WINDOW_MIN:.1f} minutes around solved geocentric closest approach")
    print("COMMENTS")
    print("This widget projects Venus onto the Sun-centered tangent plane and rotates the result into tangent and normal components at closest approach.")
    print("The parabolic closest-approach behavior belongs to ρ(t) on the Sun screen, not to raw 3D Earth-Sun/Venus range distances.")
    print("RESULTS")
    print(f"Closest approach UTC: {stats['ca_utc']}")
    print(f"Closest approach JD(TDB): {float(stats['ca_jd_tdb']):.15f}")
    print(f"Minimum ρ: {float(stats['rho_min_arcsec']):.12f} arcsec")
    print(f"Solar angular radius at CA: {float(stats['solar_radius_arcsec']):.6f} arcsec")
    print(f"Track angle at CA: {float(stats['track_angle_deg']):.6f} deg")
    print(f"Normal coordinate at CA: {float(stats['normal_at_ca_arcsec']):.12f} arcsec")
    print(f"Along-track span over ±30 min: {float(stats['tangent_centered_span_arcsec']):.12f} arcsec")
    print(f"Normal drift span over ±30 min: {float(stats['normal_centered_span_arcsec']):.12f} arcsec")
    print(f"ρ excess at −30 min: {float(stats['rho_excess_minus30_arcsec']):.12f} arcsec")
    print(f"ρ excess at +30 min: {float(stats['rho_excess_plus30_arcsec']):.12f} arcsec")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    print("PAPER COMPARISON")
    print("NOT USED: this widget is a geometric explanation plot; published values are not inputs.")
    print("EQUATION STATUS")
    print("PASS: ρ(t) is computed on the Sun-centered tangent plane; CA is solved as argmin ρ(t); s and n are derived from the local CA tangent vector.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z%z"))
    print(VERSION)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, stats = analyze()
    draw(df, stats)
    print_sections(df, stats)
    display(Image(filename=str(PNG)))


if __name__ == "__main__":
    main()
# V0103
