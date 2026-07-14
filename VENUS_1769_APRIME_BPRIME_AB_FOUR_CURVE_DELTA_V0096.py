# V0096
# Audit reference: standalone JPL Horizons four-curve A-prime/B-prime and AB fixed-vs-instantaneous Matplotlib widget; no AI images.
from __future__ import annotations

import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0096"
LOCAL_TZ = ZoneInfo("America/Bogota")
OUT = Path("/content/VENUS_1769_APRIME_BPRIME_AB_FOUR_CURVE_DELTA_V0096_OUTPUT")
PNG = OUT / "VENUS_1769_APRIME_BPRIME_AB_FOUR_CURVE_DELTA_V0096.png"
CSV = OUT / "VENUS_1769_APRIME_BPRIME_AB_FOUR_CURVE_DELTA_V0096.csv"
CONTACT_CSV = OUT / "VENUS_1769_APRIME_BPRIME_AB_FOUR_CURVE_DELTA_CONTACTS_V0096.csv"

ARC = 206_264.80624709636
AU_KM = 149_597_870.700000
SUN_RADIUS_KM = 695_700.000000
VENUS_RADIUS_KM = 6_051.800000
START = "1769-06-03 18:00"
STOP = "1769-06-04 06:00"
STEP = "1m"
EVENTS = ("C1", "C2", "CA", "C3", "C4")

BG = "#000000"
FG = "#F8FAFC"
MUTED = "#B8CBD6"
GRID = "#263A4B"
BLUE = "#42D7C3"
GOLD = "#D89B18"
GREEN = "#74D680"
RED = "#FF6B6B"
PURPLE = "#B28CFF"
ORANGE = "#FFB454"
TABLE_HEADER = "#23466F"
TABLE_BODY = "#101A2E"
TABLE_TEAL = "#164B55"
TABLE_GOLD = "#563B0B"

POINT_VENUS = {"key": "POINT_VENUS", "label": "Point Venus, Tahiti", "short": "PV", "lat": -17.4956, "lon": -149.4939, "elevation": 0.0, "body": 399, "color": BLUE}
VARDO = {"key": "VARDO", "label": "Vardø, Norway", "short": "V", "lat": 70.3724, "lon": 31.1103, "elevation": 0.0, "body": 399, "color": GOLD}
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
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import brentq, minimize_scalar

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


def loc(site: dict) -> dict:
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
            if len(df) < 600:
                raise RuntimeError(f"Incomplete JPL series for {prefix}: {len(df)} rows")
            return df
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"JPL query failed for {prefix}: {last}")


def build_master() -> pd.DataFrame:
    parts = []
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
    c: dict[str, object] = {"JD_TDB": jds}
    for p in PREFIXES:
        for ax in "XYZ":
            c[f"{p}_{ax}_KM"] = CubicSpline(jds, master[f"{p}_{ax}_KM"].to_numpy(float), bc_type="natural")
    return c


def vec(c: dict[str, object], prefix: str, jd: float) -> np.ndarray:
    return np.array([float(c[f"{prefix}_{ax}_KM"](float(jd))) for ax in "XYZ"], dtype=float)


def sep_rad(a: np.ndarray, b: np.ndarray) -> float:
    ah, bh = unit(a), unit(b)
    return math.atan2(norm(np.cross(ah, bh)), float(np.dot(ah, bh)))


def radii(c: dict[str, object], site_key: str, jd: float) -> tuple[float, float]:
    sun = norm(vec(c, f"{site_key}_SUN", jd))
    ven = norm(vec(c, f"{site_key}_VENUS", jd))
    return math.asin(SUN_RADIUS_KM / sun), math.asin(VENUS_RADIUS_KM / ven)


def residual(c: dict[str, object], site_key: str, jd: float, internal: bool) -> float:
    s = sep_rad(vec(c, f"{site_key}_SUN", jd), vec(c, f"{site_key}_VENUS", jd))
    rs, rv = radii(c, site_key, jd)
    return s - (rs - rv if internal else rs + rv)


def roots(c: dict[str, object], site_key: str, internal: bool) -> list[float]:
    jds = np.asarray(c["JD_TDB"], dtype=float)
    vals = np.array([residual(c, site_key, jd, internal) for jd in jds], dtype=float)
    out = []
    for i in range(len(jds) - 1):
        if not np.isfinite(vals[i]) or not np.isfinite(vals[i + 1]):
            continue
        if vals[i] == 0.0:
            out.append(float(jds[i]))
        elif vals[i] * vals[i + 1] < 0.0:
            out.append(float(brentq(lambda x: residual(c, site_key, x, internal), float(jds[i]), float(jds[i + 1]), xtol=1e-13, rtol=1e-14)))
    clean = []
    for r in sorted(out):
        if not clean or abs(r - clean[-1]) > 0.2 / 86400.0:
            clean.append(r)
    if len(clean) != 2:
        raise RuntimeError(f"Expected two contact roots for {site_key}; found {len(clean)}")
    return clean


def local_ca(c: dict[str, object], site_key: str, a: float, b: float) -> float:
    ref = 0.5 * (a + b)
    lo = (a - ref) * 86400.0
    hi = (b - ref) * 86400.0
    res = minimize_scalar(lambda seconds: sep_rad(vec(c, f"{site_key}_SUN", ref + float(seconds) / 86400.0), vec(c, f"{site_key}_VENUS", ref + float(seconds) / 86400.0)), bounds=(lo, hi), method="bounded", options={"xatol": 1e-4, "maxiter": 500})
    if not res.success:
        raise RuntimeError(f"Closest approach failed for {site_key}")
    return ref + float(res.x) / 86400.0


def geocentric_ca(c: dict[str, object]) -> float:
    jds = np.asarray(c["JD_TDB"], dtype=float)
    vals = np.array([sep_rad(vec(c, "GEOCENTER_SUN", jd), vec(c, "GEOCENTER_VENUS", jd)) for jd in jds], dtype=float)
    i = int(np.argmin(vals))
    lo_jd = float(jds[max(0, i - 3)])
    hi_jd = float(jds[min(len(jds) - 1, i + 3)])
    ref = 0.5 * (lo_jd + hi_jd)
    res = minimize_scalar(lambda seconds: sep_rad(vec(c, "GEOCENTER_SUN", ref + float(seconds) / 86400.0), vec(c, "GEOCENTER_VENUS", ref + float(seconds) / 86400.0)), bounds=((lo_jd - ref) * 86400.0, (hi_jd - ref) * 86400.0), method="bounded", options={"xatol": 1e-4, "maxiter": 500})
    if not res.success:
        raise RuntimeError("Geocentric closest approach failed.")
    return ref + float(res.x) / 86400.0


def events_for(c: dict[str, object], site: dict) -> dict[str, float]:
    ext = roots(c, str(site["key"]), False)
    inn = roots(c, str(site["key"]), True)
    ca = local_ca(c, str(site["key"]), inn[0], inn[1])
    ev = {"C1": ext[0], "C2": inn[0], "CA": ca, "C3": inn[1], "C4": ext[1]}
    if not (ev["C1"] < ev["C2"] < ev["CA"] < ev["C3"] < ev["C4"]):
        raise RuntimeError(f"Bad contact order for {site['label']}")
    return ev


def basis_from_sun(sun_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = unit(sun_vec)
    pole = np.array([0.0, 0.0, 1.0])
    east = np.cross(pole, center)
    if norm(east) < 1e-14:
        east = np.cross(np.array([0.0, 1.0, 0.0]), center)
    east = unit(east)
    north = unit(np.cross(center, east))
    if float(np.dot(north, pole)) < 0.0:
        east, north = -east, -north
    return center, east, north


def gnomonic(ray: np.ndarray, center: np.ndarray, east: np.ndarray, north: np.ndarray) -> np.ndarray:
    h = unit(ray)
    den = float(np.dot(h, center))
    if den <= 0.0:
        raise RuntimeError("Ray outside tangent hemisphere.")
    return np.array([float(np.dot(h, east)), float(np.dot(h, north))]) / den


def relative_xy_fixed(c: dict[str, object], site_key: str, jd: float, basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    center, east, north = basis
    return ARC * (gnomonic(vec(c, f"{site_key}_VENUS", jd), center, east, north) - gnomonic(vec(c, f"{site_key}_SUN", jd), center, east, north))


def relative_xy_instant(c: dict[str, object], site_key: str, jd: float) -> np.ndarray:
    return relative_xy_fixed(c, site_key, jd, basis_from_sun(vec(c, "GEOCENTER_SUN", jd)))


def fixed_geometry(c: dict[str, object], geo_ca_jd: float) -> dict[str, object]:
    basis = basis_from_sun(vec(c, "GEOCENTER_SUN", geo_ca_jd))
    h = 0.5 / 86400.0
    pv_t = relative_xy_fixed(c, "POINT_VENUS", geo_ca_jd + h, basis) - relative_xy_fixed(c, "POINT_VENUS", geo_ca_jd - h, basis)
    va_t = relative_xy_fixed(c, "VARDO", geo_ca_jd + h, basis) - relative_xy_fixed(c, "VARDO", geo_ca_jd - h, basis)
    tangent2 = unit(unit(pv_t) + unit(va_t))
    normal2 = np.array([-tangent2[1], tangent2[0]])
    pv0 = relative_xy_fixed(c, "POINT_VENUS", geo_ca_jd, basis)
    va0 = relative_xy_fixed(c, "VARDO", geo_ca_jd, basis)
    if float(np.dot(va0 - pv0, normal2)) < 0.0:
        tangent2, normal2 = -tangent2, -normal2
    center, east, north = basis
    normal3 = unit(normal2[0] * east + normal2[1] * north)
    return {"basis": basis, "tangent2": tangent2, "normal2": normal2, "normal3": normal3, "angle_deg": math.degrees(math.atan2(tangent2[1], tangent2[0]))}


def instantaneous_geometry(c: dict[str, object], jd: float) -> dict[str, object]:
    basis = basis_from_sun(vec(c, "GEOCENTER_SUN", jd))
    h = 60.0 / 86400.0
    pv = relative_xy_fixed(c, "POINT_VENUS", jd, basis)
    va = relative_xy_fixed(c, "VARDO", jd, basis)
    pv_v = (relative_xy_instant(c, "POINT_VENUS", jd + h) - relative_xy_instant(c, "POINT_VENUS", jd - h)) / 120.0
    va_v = (relative_xy_instant(c, "VARDO", jd + h) - relative_xy_instant(c, "VARDO", jd - h)) / 120.0
    tangent2 = unit(0.5 * (pv_v + va_v))
    normal2 = np.array([-tangent2[1], tangent2[0]])
    if float(np.dot(va - pv, normal2)) < 0.0:
        tangent2, normal2 = -tangent2, -normal2
    center, east, north = basis
    normal3 = unit(normal2[0] * east + normal2[1] * north)
    return {"basis": basis, "tangent2": tangent2, "normal2": normal2, "normal3": normal3, "angle_deg": math.degrees(math.atan2(tangent2[1], tangent2[0]))}


def site_position_from_sun(c: dict[str, object], site_key: str, jd: float) -> np.ndarray:
    return vec(c, "GEOCENTER_SUN", jd) - vec(c, f"{site_key}_SUN", jd)


def compute_row(c: dict[str, object], jd: float, fixed: dict[str, object]) -> dict[str, float | str]:
    inst = instantaneous_geometry(c, jd)
    fixed_basis = fixed["basis"]
    pv_fixed = relative_xy_fixed(c, "POINT_VENUS", jd, fixed_basis)
    va_fixed = relative_xy_fixed(c, "VARDO", jd, fixed_basis)
    pv_inst = relative_xy_fixed(c, "POINT_VENUS", jd, inst["basis"])
    va_inst = relative_xy_fixed(c, "VARDO", jd, inst["basis"])
    delta_fixed = va_fixed - pv_fixed
    delta_inst = va_inst - pv_inst
    geo_sun = vec(c, "GEOCENTER_SUN", jd)
    km_per_arcsec = norm(geo_sun) / ARC
    baseline = site_position_from_sun(c, "VARDO", jd) - site_position_from_sun(c, "POINT_VENUS", jd)
    apbp_fixed = float(np.dot(delta_fixed, fixed["normal2"]))
    apbp_inst = float(np.dot(delta_inst, inst["normal2"]))
    ab_fixed = float(np.dot(baseline, fixed["normal3"])) / km_per_arcsec
    ab_inst = float(np.dot(baseline, inst["normal3"])) / km_per_arcsec
    return {
        "jd_tdb": float(jd),
        "utc": utc(jd),
        "minute_from_start": 0.0,
        "apbp_fixed_arcsec": apbp_fixed,
        "apbp_instant_arcsec": apbp_inst,
        "ab_fixed_arcsec": ab_fixed,
        "ab_instant_arcsec": ab_inst,
        "delta_apbp_arcsec": apbp_inst - apbp_fixed,
        "delta_ab_arcsec": ab_inst - ab_fixed,
        "delta_apbp_2x_arcsec": 2.0 * (apbp_inst - apbp_fixed),
        "delta_ab_2x_arcsec": 2.0 * (ab_inst - ab_fixed),
        "fixed_angle_deg": float(fixed["angle_deg"]),
        "instant_angle_deg": float(inst["angle_deg"]),
    }


def closest_abs(df: pd.DataFrame, column: str) -> dict[str, float | str]:
    y = df[column].to_numpy(float)
    i = int(np.argmax(np.abs(y)))
    return {"jd": float(df["jd_tdb"].iloc[i]), "utc": str(df["utc"].iloc[i]), "minute": float(df["minute_from_start"].iloc[i]), "value": float(y[i])}


def analyze() -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, float | str]]:
    OUT.mkdir(parents=True, exist_ok=True)
    master = build_master()
    c = splines(master)
    evs = {"Point Venus, Tahiti": events_for(c, POINT_VENUS), "Vardø, Norway": events_for(c, VARDO)}
    start = min(evs["Point Venus, Tahiti"]["C1"], evs["Vardø, Norway"]["C1"])
    stop = max(evs["Point Venus, Tahiti"]["C4"], evs["Vardø, Norway"]["C4"])
    geo_ca_jd = geocentric_ca(c)
    fixed = fixed_geometry(c, geo_ca_jd)
    minute_jds = np.asarray(c["JD_TDB"], dtype=float)
    minute_jds = minute_jds[(minute_jds >= start) & (minute_jds <= stop)]
    rows = [compute_row(c, float(jd), fixed) for jd in minute_jds]
    df = pd.DataFrame(rows)
    df["minute_from_start"] = (df["jd_tdb"] - float(df["jd_tdb"].iloc[0])) * 1440.0
    df.to_csv(CSV, index=False, float_format="%.15f")
    contacts = []
    for station, ev in evs.items():
        for event in EVENTS:
            contacts.append({"station": station, "event": event, "utc": utc(ev[event]), "jd_tdb": ev[event]})
    pd.DataFrame(contacts).to_csv(CONTACT_CSV, index=False, float_format="%.15f")
    apmax = closest_abs(df, "delta_apbp_arcsec")
    abmax = closest_abs(df, "delta_ab_arcsec")
    start_jd = float(df["jd_tdb"].iloc[0])
    stats: dict[str, float | str] = {
        "geo_ca_jd": float(geo_ca_jd),
        "geo_ca_utc": utc(geo_ca_jd),
        "geo_ca_minute": (geo_ca_jd - start_jd) * 1440.0,
        "mean_delta_apbp": float(df["delta_apbp_arcsec"].mean()),
        "mean_delta_ab": float(df["delta_ab_arcsec"].mean()),
        "rms_delta_apbp": float(np.sqrt(np.mean(df["delta_apbp_arcsec"].to_numpy(float) ** 2))),
        "rms_delta_ab": float(np.sqrt(np.mean(df["delta_ab_arcsec"].to_numpy(float) ** 2))),
        "max_abs_delta_apbp": float(apmax["value"]),
        "max_abs_delta_apbp_utc": str(apmax["utc"]),
        "max_abs_delta_ab": float(abmax["value"]),
        "max_abs_delta_ab_utc": str(abmax["utc"]),
        "geo_delta_apbp": float(np.interp(geo_ca_jd, df["jd_tdb"].to_numpy(float), df["delta_apbp_arcsec"].to_numpy(float))),
        "geo_delta_ab": float(np.interp(geo_ca_jd, df["jd_tdb"].to_numpy(float), df["delta_ab_arcsec"].to_numpy(float))),
        "samples": int(len(df)),
    }
    return df, evs, stats


def style_axis(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.30, alpha=0.58)
    ax.tick_params(labelsize=7.0, width=0.35, length=2.4)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.35)


def mark_events(ax, df: pd.DataFrame, evs: dict[str, dict[str, float]], column: str) -> None:
    start_jd = float(df["jd_tdb"].iloc[0])
    interp = interp1d(df["jd_tdb"].to_numpy(float), df[column].to_numpy(float), kind="linear", bounds_error=False, fill_value="extrapolate")
    mean_events = {event: 0.5 * (evs["Point Venus, Tahiti"][event] + evs["Vardø, Norway"][event]) for event in EVENTS}
    for event in EVENTS:
        jd = float(mean_events[event])
        xm = (jd - start_jd) * 1440.0
        ym = float(interp(jd))
        color = GOLD if event == "CA" else BLUE
        ax.axvline(xm, color=color, linewidth=0.26, alpha=0.56, zorder=1)
        ax.scatter([xm], [ym], s=13 if event != "CA" else 22, color=color, edgecolors=FG, linewidths=0.20, zorder=5)
        ax.annotate(event, xy=(xm, ym), xytext=(xm, ym + (0.018 if event in ("C1", "C2", "CA") else -0.020)), ha="center", va="bottom" if event in ("C1", "C2", "CA") else "top", fontsize=5.8, color=FG, arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.20})


def table_style(table, gold_rows=(), teal_rows=(), fontsize=5.9) -> None:
    table.auto_set_font_size(False)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#70879A")
        cell.set_linewidth(0.30)
        cell.get_text().set_color(FG)
        cell.get_text().set_fontsize(fontsize)
        if row == 0:
            cell.set_facecolor(TABLE_HEADER)
            cell.get_text().set_fontweight("bold")
        elif row in gold_rows:
            cell.set_facecolor(TABLE_GOLD)
            cell.get_text().set_fontweight("bold")
        elif row in teal_rows:
            cell.set_facecolor(TABLE_TEAL)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(TABLE_BODY)


def plot(df: pd.DataFrame, evs: dict[str, dict[str, float]], stats: dict[str, float | str]) -> None:
    plt.close("all")
    plt.rcParams.update({"font.family": "DejaVu Serif", "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED})
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.48, 0.30, 0.22], left=0.055, right=0.985, top=0.900, bottom=0.092, hspace=0.175)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_delta = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_table = fig.add_subplot(gs[2, 0])
    fig.suptitle("1769 Venus Transit — A′B′ and AB Fixed-Vector vs Instantaneous Normal", fontsize=14.5, fontweight="bold", y=0.965)
    fig.text(0.5, 0.932, "Fresh JPL Horizons geometric vectors; four curves in arcsec with 2× amplified deltas below.", ha="center", fontsize=7.2, color=MUTED)
    x = df["minute_from_start"].to_numpy(float)
    ax_main.plot(x, df["apbp_fixed_arcsec"], color=GREEN, linewidth=0.50, label="A′B′ fixed-vector")
    ax_main.plot(x, df["apbp_instant_arcsec"], color=GOLD, linewidth=0.50, label="A′B′ instantaneous")
    ax_main.plot(x, df["ab_fixed_arcsec"], color=BLUE, linewidth=0.50, linestyle="--", label="AB fixed-vector")
    ax_main.plot(x, df["ab_instant_arcsec"], color=PURPLE, linewidth=0.50, linestyle="--", label="AB instantaneous")
    for col in ("apbp_fixed_arcsec", "apbp_instant_arcsec", "ab_fixed_arcsec", "ab_instant_arcsec"):
        mark_events(ax_main, df, evs, col)
    ax_main.set_ylabel("Separation / projected baseline (arcsec)", fontsize=8.4)
    ax_main.set_title("FOUR CURVES — A′B′ AND AB", fontsize=9.7, fontweight="bold")
    style_axis(ax_main)
    leg = ax_main.legend(loc="upper right", frameon=False, fontsize=7.0, ncol=2)
    for text in leg.get_texts():
        text.set_color(FG)

    d_ap = df["delta_apbp_2x_arcsec"].to_numpy(float)
    d_ab = df["delta_ab_2x_arcsec"].to_numpy(float)
    mean_ap_2x = 2.0 * float(stats["mean_delta_apbp"])
    mean_ab_2x = 2.0 * float(stats["mean_delta_ab"])
    ax_delta.axhline(0.0, color=MUTED, linewidth=0.55, alpha=0.85, label="zero")
    ax_delta.plot(x, d_ap, color=GOLD, linewidth=0.62, label="2× ΔA′B′ = 2×(instant − fixed)")
    ax_delta.plot(x, d_ab, color=PURPLE, linewidth=0.62, label="2× ΔAB = 2×(instant − fixed)")
    ax_delta.axhline(mean_ap_2x, color=GOLD, linewidth=0.54, linestyle="--", alpha=0.86, label=f"2× average ΔA′B′ = {mean_ap_2x:+.9f}″")
    ax_delta.axhline(mean_ab_2x, color=PURPLE, linewidth=0.54, linestyle="--", alpha=0.86, label=f"2× average ΔAB = {mean_ab_2x:+.9f}″")
    ax_delta.scatter([float(stats["geo_ca_minute"])], [2.0 * float(stats["geo_delta_apbp"])], s=42, marker="X", color=RED, edgecolors=FG, linewidths=0.32, zorder=6)
    ax_delta.annotate(f"geo CA 2×ΔA′B′ = {2.0 * float(stats['geo_delta_apbp']):+.9f}″", xy=(float(stats["geo_ca_minute"]), 2.0 * float(stats["geo_delta_apbp"])), xytext=(float(stats["geo_ca_minute"]) + 16, 2.0 * float(stats["geo_delta_apbp"]) + 0.015), ha="left", fontsize=6.4, color=FG, arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.25})
    ax_delta.set_ylabel("2× delta (arcsec)", fontsize=8.4)
    ax_delta.set_xlabel("Elapsed time from first C1–C4 sample (minutes)", fontsize=8.4)
    ax_delta.set_title("BOTTOM PANEL — 2× AMPLIFIED DELTAS AND AVERAGE LINES", fontsize=9.7, fontweight="bold")
    style_axis(ax_delta)
    leg2 = ax_delta.legend(loc="upper right", frameon=False, fontsize=6.6, ncol=2)
    for text in leg2.get_texts():
        text.set_color(FG)

    ax_table.axis("off")
    rows = [
        ["Quantity", "A′B′", "AB", "Unit / trace"],
        ["Mean Δ instant−fixed", f"{float(stats['mean_delta_apbp']):+.12f}", f"{float(stats['mean_delta_ab']):+.12f}", "arcsec"],
        ["RMS Δ", f"{float(stats['rms_delta_apbp']):.12f}", f"{float(stats['rms_delta_ab']):.12f}", "arcsec"],
        ["Max |Δ|", f"{float(stats['max_abs_delta_apbp']):+.12f}", f"{float(stats['max_abs_delta_ab']):+.12f}", "arcsec"],
        ["Max |Δ| UTC", str(stats["max_abs_delta_apbp_utc"]), str(stats["max_abs_delta_ab_utc"]), "UTC"],
        ["Geocentric CA Δ", f"{float(stats['geo_delta_apbp']):+.12f}", f"{float(stats['geo_delta_ab']):+.12f}", str(stats["geo_ca_utc"])],
        ["Samples", str(int(stats["samples"])), str(int(stats["samples"])), "one-minute JPL C1–C4 span"],
    ]
    table = ax_table.table(cellText=rows, cellLoc="left", colWidths=[0.25, 0.24, 0.24, 0.27], bbox=[0.0, 0.08, 1.0, 0.78])
    table_style(table, teal_rows=(1, 5), gold_rows=(2, 3, 4, 6), fontsize=6.4)
    fig.text(0.5, 0.042, f"File: {Path(__file__).name if '__file__' in globals() else 'VENUS_1769_APRIME_BPRIME_AB_FOUR_CURVE_DELTA_V0096.py'} | Output: {PNG.name} | CSV: {CSV.name}", ha="center", fontsize=5.8, color=MUTED)
    fig.savefig(PNG, dpi=220, facecolor=BG)
    display(Image(filename=str(PNG)))


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"Start UTC: {START}; stop UTC: {STOP}; step: {STEP}")
    print("Sites: Point Venus, Tahiti; Vardø, Norway")
    print("Data source: fresh JPL Horizons geometric ecliptic vectors")
    print("COMMENTS")
    print("Four-curve plot: A′B′ fixed, A′B′ instantaneous, AB fixed, AB instantaneous.")
    print("Bottom panel: 2× amplified deltas with average delta lines. No AI images; Matplotlib only.")
    df, evs, stats = analyze()
    plot(df, evs, stats)
    print("RESULTS")
    print(f"Mean ΔA′B′ instant-fixed: {float(stats['mean_delta_apbp']):+.12f} arcsec")
    print(f"Mean ΔAB instant-fixed: {float(stats['mean_delta_ab']):+.12f} arcsec")
    print(f"RMS ΔA′B′: {float(stats['rms_delta_apbp']):.12f} arcsec")
    print(f"RMS ΔAB: {float(stats['rms_delta_ab']):.12f} arcsec")
    print(f"Geocentric CA ΔA′B′: {float(stats['geo_delta_apbp']):+.12f} arcsec")
    print(f"Geocentric CA ΔAB: {float(stats['geo_delta_ab']):+.12f} arcsec")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    print(f"CONTACT CSV: {CONTACT_CSV}")
    print("PAPER COMPARISON")
    print("NOT USED: this is an internal JPL-vector derivation comparison only.")
    print("EQUATION STATUS")
    print("PASS: A′B′ curves use angular common-normal projection; AB curves use JPL observer baseline projected onto matching fixed or instantaneous 3D normal and converted to arcsec.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0096
