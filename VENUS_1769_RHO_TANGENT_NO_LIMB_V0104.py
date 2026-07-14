# V0104
# Audit reference: no-limb tangent-plane rho audit; Earth/geocenter, Vardo A, and Point Venus B rho-minus-minimum curves from JPL Horizons only.
from __future__ import annotations

import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0104"
LOCAL_TZ = ZoneInfo("America/Bogota")
OUT = Path("/content/VENUS_1769_RHO_TANGENT_NO_LIMB_V0104_OUTPUT")
PNG = OUT / "VENUS_1769_RHO_TANGENT_NO_LIMB_V0104.png"
CSV = OUT / "VENUS_1769_RHO_TANGENT_NO_LIMB_V0104.csv"
CONTACT_CSV = OUT / "VENUS_1769_RHO_TANGENT_NO_LIMB_CONTACTS_V0104.csv"

ARCSEC_PER_RAD = 206_264.80624709636
AU_KM = 149_597_870.700000
START = "1769-06-03 21:30"
STOP = "1769-06-03 23:15"
STEP = "1m"
WINDOW_MIN = 30.0
SUN_TARGET = "10"
VENUS_TARGET = "299"

OBSERVERS = {
    "Earth geocenter": {"location": "500@399", "short": "E", "kind": "geocenter"},
    "Vardo A": {"location": {"lon": 31.1103, "lat": 70.3724, "elevation": 0.0}, "short": "A", "kind": "topocenter"},
    "Point Venus B": {"location": {"lon": -149.4939, "lat": -17.4956, "elevation": 0.0}, "short": "B", "kind": "topocenter"},
}

BG = "#000000"
FG = "#F8FAFC"
MUTED = "#B8CBD6"
GRID = "#263A4B"
BLUE = "#42D7C3"
RED = "#FF6B6B"
GOLD = "#D89B18"
PURPLE = "#9B8CFF"
GREEN = "#74D680"
TABLE_HEADER = "#23466F"
TABLE_BODY = "#101A2E"
TABLE_ALT = "#13233C"


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


def download(prefix: str, target_id: str, location) -> pd.DataFrame:
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


def build_observer_master(observer_name: str, location) -> pd.DataFrame:
    sun = download(f"{observer_name}_SUN", SUN_TARGET, location)
    venus = download(f"{observer_name}_VENUS", VENUS_TARGET, location)
    master = sun.merge(venus, on="JD_TDB", how="inner")
    if len(master) < 80:
        raise RuntimeError(f"Insufficient JPL samples for {observer_name}: {len(master)}")
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


def basis_from_sun(r_sun: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = unit(r_sun)
    k = np.array([0.0, 0.0, 1.0], dtype=float)
    east = np.cross(k, z)
    if norm(east) < 1.0e-14:
        east = np.cross(np.array([0.0, 1.0, 0.0], dtype=float), z)
    east = unit(east)
    north = unit(np.cross(z, east))
    return east, north, z


def tangent_xy_arcsec(r_sun: np.ndarray, r_venus: np.ndarray) -> tuple[float, float, float]:
    east, north, z = basis_from_sun(r_sun)
    u_v = unit(r_venus)
    denom = float(np.dot(u_v, z))
    x = ARCSEC_PER_RAD * float(np.dot(u_v, east) / denom)
    y = ARCSEC_PER_RAD * float(np.dot(u_v, north) / denom)
    rho = float(math.hypot(x, y))
    return x, y, rho


def xy_rho(c: dict[str, object], observer_name: str, jd: float) -> tuple[float, float, float]:
    sun_prefix = f"{observer_name}_SUN"
    venus_prefix = f"{observer_name}_VENUS"
    return tangent_xy_arcsec(vec(c, sun_prefix, jd), vec(c, venus_prefix, jd))


def rho(c: dict[str, object], observer_name: str, jd: float) -> float:
    return xy_rho(c, observer_name, jd)[2]


def rho_dot(c: dict[str, object], observer_name: str, jd: float) -> float:
    h = 0.5 / 1440.0
    return rho(c, observer_name, jd + h) - rho(c, observer_name, jd - h)


def solve_ca(c: dict[str, object], observer_name: str) -> float:
    jds = np.asarray(c["JD_TDB"], dtype=float)
    samples = np.array([rho(c, observer_name, float(jd)) for jd in jds], dtype=float)
    i = int(np.argmin(samples))
    lo = float(jds[max(0, i - 3)])
    hi = float(jds[min(len(jds) - 1, i + 3)])
    res = minimize_scalar(lambda x: rho(c, observer_name, float(x)), bounds=(lo, hi), method="bounded", options={"xatol": 1e-13, "maxiter": 600})
    if not res.success:
        raise RuntimeError(f"Closest-approach minimization failed for {observer_name}.")
    return float(res.x)


def solve_rhodot_zero(c: dict[str, object], observer_name: str, ca_jd: float) -> float:
    lo = ca_jd - 5.0 / 1440.0
    hi = ca_jd + 5.0 / 1440.0
    f_lo = rho_dot(c, observer_name, lo)
    f_hi = rho_dot(c, observer_name, hi)
    if f_lo * f_hi <= 0.0:
        return float(brentq(lambda x: rho_dot(c, observer_name, float(x)), lo, hi, xtol=1e-13, rtol=1e-13, maxiter=100))
    return ca_jd


def build_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    OUT.mkdir(parents=True, exist_ok=True)
    curve_frames = []
    contact_rows = []
    observer_data = {}

    for name, meta in OBSERVERS.items():
        master = build_observer_master(name, meta["location"])
        c = splines(master)
        ca_jd = solve_ca(c, name)
        zero_jd = solve_rhodot_zero(c, name, ca_jd)
        rho_min = rho(c, name, ca_jd)
        x_ca, y_ca, _ = xy_rho(c, name, ca_jd)
        observer_data[name] = {"c": c, "ca_jd": ca_jd, "rho_min": rho_min, "x_ca": x_ca, "y_ca": y_ca, "zero_jd": zero_jd}
        contact_rows.append({
            "observer": name,
            "short": meta["short"],
            "ca_utc": utc_from_jd(ca_jd),
            "ca_jd_tdb": ca_jd,
            "rho_min_arcsec": rho_min,
            "rhodot_zero_utc": utc_from_jd(zero_jd),
            "rhodot_zero_offset_sec": (zero_jd - ca_jd) * 86400.0,
        })

    earth_ca = float(observer_data["Earth geocenter"]["ca_jd"])
    minutes = np.linspace(-WINDOW_MIN, WINDOW_MIN, int(2 * WINDOW_MIN * 4) + 1)

    for name, meta in OBSERVERS.items():
        c = observer_data[name]["c"]
        ca_jd = float(observer_data[name]["ca_jd"])
        rho_min = float(observer_data[name]["rho_min"])
        for m in minutes:
            jd = earth_ca + float(m) / 1440.0
            x, y, rr = xy_rho(c, name, jd)
            curve_frames.append({
                "observer": name,
                "short": meta["short"],
                "minute_from_earth_geocenter_ca": float(m),
                "minute_from_own_ca": float((jd - ca_jd) * 1440.0),
                "jd_tdb": float(jd),
                "utc": utc_from_jd(jd),
                "x_arcsec": float(x),
                "y_arcsec": float(y),
                "rho_arcsec": float(rr),
                "rho_min_arcsec": float(rho_min),
                "rho_minus_min_arcsec": float(rr - rho_min),
                "rho_dot_arcsec_per_min_raw": float(rho_dot(c, name, jd)),
            })

    curves = pd.DataFrame(curve_frames)
    contacts = pd.DataFrame(contact_rows)
    curves.to_csv(CSV, index=False, float_format="%.15f")
    contacts.to_csv(CONTACT_CSV, index=False, float_format="%.15f")
    return curves, contacts, observer_data


def make_plot(curves: pd.DataFrame, contacts: pd.DataFrame) -> None:
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": FG,
        "axes.labelcolor": FG,
        "xtick.color": FG,
        "ytick.color": FG,
        "axes.edgecolor": FG,
        "font.size": 9,
    })

    colors = {"Earth geocenter": BLUE, "Vardo A": RED, "Point Venus B": GOLD}
    fig = plt.figure(figsize=(15.2, 9.2), dpi=180)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.58, 1.0], height_ratios=[1.0, 1.0], left=0.055, right=0.975, bottom=0.075, top=0.900, wspace=0.20, hspace=0.28)
    ax_rho = fig.add_subplot(gs[0, 0])
    ax_xy = fig.add_subplot(gs[1, 0])
    ax_dot = fig.add_subplot(gs[0, 1])
    ax_tbl = fig.add_subplot(gs[1, 1])

    for name in OBSERVERS:
        sub = curves[curves["observer"] == name].copy()
        ax_rho.plot(sub["minute_from_earth_geocenter_ca"], sub["rho_minus_min_arcsec"], lw=0.95, color=colors[name], label=f"{name}: ρ − ρmin")
        ax_dot.plot(sub["minute_from_earth_geocenter_ca"], sub["rho_dot_arcsec_per_min_raw"], lw=0.90, color=colors[name], label=f"{name}: raw dρ/dt")
        ax_xy.plot(sub["x_arcsec"], sub["y_arcsec"], lw=0.85, color=colors[name], label=name)
        row = contacts[contacts["observer"] == name].iloc[0]
        ca_min = (float(row["ca_jd_tdb"]) - float(contacts[contacts["observer"] == "Earth geocenter"].iloc[0]["ca_jd_tdb"])) * 1440.0
        ax_rho.axvline(ca_min, lw=0.55, ls="--", color=colors[name], alpha=0.90)
        ax_dot.axvline(ca_min, lw=0.55, ls="--", color=colors[name], alpha=0.90)
        ca_xy = sub.iloc[(sub["minute_from_own_ca"].abs()).argmin()]
        ax_xy.scatter([ca_xy["x_arcsec"]], [ca_xy["y_arcsec"]], s=18, color=colors[name], zorder=4)
        ax_xy.annotate(row["short"], (ca_xy["x_arcsec"], ca_xy["y_arcsec"]), xytext=(5, 5), textcoords="offset points", fontsize=8, color=colors[name], weight="bold")

    ax_rho.axhline(0.0, lw=0.65, ls=":", color=MUTED)
    ax_rho.axvline(0.0, lw=0.70, ls="-", color=FG, alpha=0.70)
    ax_rho.grid(color=GRID, lw=0.35, alpha=0.75)
    ax_rho.set_xlabel("Minutes from Earth/geocenter closest approach")
    ax_rho.set_ylabel("ρ − ρmin (arcsec)")
    ax_rho.set_title("Closest-approach bowl: ρ(t) − ρmin, no solar limb", weight="bold")
    ax_rho.legend(loc="upper center", fontsize=7.7, ncol=1, frameon=True, facecolor=BG, edgecolor=GRID)

    ax_dot.axhline(0.0, lw=0.65, ls=":", color=MUTED)
    ax_dot.axvline(0.0, lw=0.70, ls="-", color=FG, alpha=0.70)
    ax_dot.grid(color=GRID, lw=0.35, alpha=0.75)
    ax_dot.set_xlabel("Minutes from Earth/geocenter closest approach")
    ax_dot.set_ylabel("raw dρ/dt (arcsec/min)")
    ax_dot.set_title("Raw derivative crosses zero at each observer CA", weight="bold")
    ax_dot.legend(loc="best", fontsize=7.2, frameon=True, facecolor=BG, edgecolor=GRID)

    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(color=GRID, lw=0.35, alpha=0.75)
    ax_xy.set_xlabel("Sun-screen tangent X (arcsec)")
    ax_xy.set_ylabel("Sun-screen tangent Y (arcsec)")
    ax_xy.set_title("Tangent-plane tracks only — solar limb removed", weight="bold")
    ax_xy.legend(loc="best", fontsize=7.7, frameon=True, facecolor=BG, edgecolor=GRID)

    ax_tbl.axis("off")
    earth_ca = float(contacts[contacts["observer"] == "Earth geocenter"].iloc[0]["ca_jd_tdb"])
    table_rows = [["Observer", "CA UTC", "Δ from Earth CA", "ρmin"]]
    for _, row in contacts.iterrows():
        offset = (float(row["ca_jd_tdb"]) - earth_ca) * 86400.0
        table_rows.append([
            str(row["short"]),
            str(row["ca_utc"])[11:],
            f"{offset:+.3f} s",
            f"{float(row['rho_min_arcsec']):.9f}″",
        ])
    table_rows.extend([
        ["", "", "", ""],
        ["ρ", "Sun-center to Venus-center angular separation", "", "arcsec"],
        ["A", "Vardo/Norway", "topocentric", "JPL"],
        ["B", "Point Venus/Tahiti", "topocentric", "JPL"],
        ["E", "Earth geocenter", "geocentric", "JPL"],
    ])
    tbl = ax_tbl.table(cellText=table_rows, cellLoc="left", colWidths=[0.15, 0.42, 0.23, 0.20], bbox=[0.00, 0.02, 1.00, 0.94])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.1)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.45)
        cell.get_text().set_color(FG)
        if r == 0:
            cell.set_facecolor(TABLE_HEADER)
            cell.get_text().set_weight("bold")
        elif r in (1, 2, 3):
            cell.set_facecolor(TABLE_BODY if r % 2 else TABLE_ALT)
        else:
            cell.set_facecolor(BG)

    fig.suptitle("1769 Venus Transit — ρ(t) Tangent-Plane Closest-Approach Audit", fontsize=15, weight="bold", y=0.975)
    fig.text(0.5, 0.940, "Solar limb deleted. Curves show Earth/geocenter ρ and observer A/B ρ − ρmin on the Sun-screen tangent plane.", ha="center", color=MUTED, fontsize=9.2)
    fig.savefig(PNG, dpi=260)
    display(Image(filename=str(PNG)))


def print_sections(curves: pd.DataFrame, contacts: pd.DataFrame) -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"JPL Horizons targets: Sun={SUN_TARGET}, Venus={VENUS_TARGET}")
    print(f"Window: {START} to {STOP}, step {STEP}; plotted ±{WINDOW_MIN:.1f} minutes from Earth/geocenter CA")
    print("Observers: Earth geocenter, Vardo A, Point Venus B")
    print("Solar limb: NOT USED / removed by request")
    print()
    print("COMMENTS")
    print("ρ is the Sun-center to Venus-center angular separation on the observer-specific tangent plane.")
    print("Each ρ − ρmin curve is minimized at that observer's own closest approach; vertical dashed lines mark those CA times.")
    print()
    print("RESULTS")
    earth_ca = float(contacts[contacts["observer"] == "Earth geocenter"].iloc[0]["ca_jd_tdb"])
    for _, row in contacts.iterrows():
        offset = (float(row["ca_jd_tdb"]) - earth_ca) * 86400.0
        print(f"{row['observer']:<17} CA UTC {row['ca_utc']}   ΔE {offset:+.6f} s   ρmin {float(row['rho_min_arcsec']):.12f} arcsec   dρ/dt zero offset {float(row['rhodot_zero_offset_sec']):+.9f} s")
    print()
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    print(f"CONTACT CSV: {CONTACT_CSV}")
    print()
    print("PAPER COMPARISON")
    print("NOT USED: this is a JPL-derived geometry visualization only.")
    print()
    print("EQUATION STATUS")
    print("PASS: ρ(t) is computed on the Sun-screen tangent plane from JPL vectors.")
    print("PASS: closest approach for each observer is argmin ρ(t), and raw dρ/dt crosses zero at that CA.")
    print("PASS: solar limb plot is NOT USED and not drawn.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


def main() -> None:
    curves, contacts, observer_data = build_data()
    make_plot(curves, contacts)
    print_sections(curves, contacts)


if __name__ == "__main__":
    main()
# V0104
