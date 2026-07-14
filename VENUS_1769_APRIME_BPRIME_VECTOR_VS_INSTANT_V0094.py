# V0094
# Audit reference: standalone JPL Horizons A-prime/B-prime fixed-vector versus instantaneous tangent-normal CA label comparison.
from __future__ import annotations

import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0094"
LOCAL_TZ = ZoneInfo("America/Bogota")
OUT = Path("/content/VENUS_1769_APRIME_BPRIME_VECTOR_VS_INSTANT_V0094_OUTPUT")
PNG = OUT / "VENUS_1769_APRIME_BPRIME_VECTOR_VS_INSTANT_V0094.png"
CSV = OUT / "VENUS_1769_APRIME_BPRIME_VECTOR_VS_INSTANT_V0094.csv"
CONTACT_CSV = OUT / "VENUS_1769_APRIME_BPRIME_VECTOR_VS_INSTANT_CONTACTS_V0094.csv"

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


for _import_name, _package_name in (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("astropy", "astropy"),
    ("astroquery", "astroquery"),
    ("matplotlib", "matplotlib"),
    ("IPython", "ipython"),
):
    require(_import_name, _package_name)

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
            tab = Horizons(
                id=target_id,
                location=location,
                epochs={"start": START, "stop": STOP, "step": STEP},
                id_type=None,
            ).vectors(refplane="ecliptic", aberrations="geometric", cache=False)
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
    for name, target_id in (("SUN", "10"), ("VENUS", "299")):
        parts.append(download(f"GEOCENTER_{name}", target_id, "@399"))
    for site in SITES:
        for name, target_id in (("SUN", "10"), ("VENUS", "299")):
            parts.append(download(f"{site['key']}_{name}", target_id, loc(site)))
    master = parts[0]
    for df in parts[1:]:
        master = master.merge(df, on="JD_TDB", how="inner", validate="one_to_one")
    if len(master) < 600:
        raise RuntimeError(f"Synchronized JPL master too short: {len(master)} rows")
    return master.sort_values("JD_TDB").reset_index(drop=True)


def splines(master: pd.DataFrame) -> dict:
    jds = master["JD_TDB"].to_numpy(float)
    c = {"JD_TDB": jds}
    for prefix in PREFIXES:
        for ax in "XYZ":
            c[f"{prefix}_{ax}_KM"] = CubicSpline(jds, master[f"{prefix}_{ax}_KM"].to_numpy(float), bc_type="natural")
    return c


def vec(c: dict, prefix: str, jd: float) -> np.ndarray:
    return np.array([float(c[f"{prefix}_{ax}_KM"](float(jd))) for ax in "XYZ"], dtype=float)


def sep_rad(a: np.ndarray, b: np.ndarray) -> float:
    ah = unit(a)
    bh = unit(b)
    return math.atan2(norm(np.cross(ah, bh)), float(np.dot(ah, bh)))


def radii(c: dict, site_key: str, jd: float) -> tuple[float, float]:
    sun = norm(vec(c, f"{site_key}_SUN", jd))
    ven = norm(vec(c, f"{site_key}_VENUS", jd))
    return math.asin(SUN_RADIUS_KM / sun), math.asin(VENUS_RADIUS_KM / ven)


def residual(c: dict, site_key: str, jd: float, internal: bool) -> float:
    d = sep_rad(vec(c, f"{site_key}_SUN", jd), vec(c, f"{site_key}_VENUS", jd))
    rs, rv = radii(c, site_key, jd)
    return d - (rs - rv if internal else rs + rv)


def roots(c: dict, site_key: str, internal: bool) -> list[float]:
    jds = np.asarray(c["JD_TDB"], dtype=float)
    vals = np.array([residual(c, site_key, jd, internal) for jd in jds], dtype=float)
    found = []
    for i in range(len(jds) - 1):
        if not np.isfinite(vals[i]) or not np.isfinite(vals[i + 1]):
            continue
        if vals[i] == 0.0:
            found.append(float(jds[i]))
        elif vals[i] * vals[i + 1] < 0.0:
            found.append(float(brentq(lambda x: residual(c, site_key, x, internal), float(jds[i]), float(jds[i + 1]), xtol=1e-13, rtol=1e-14)))
    clean = []
    for r in sorted(found):
        if not clean or abs(r - clean[-1]) > 0.2 / 86400.0:
            clean.append(r)
    if len(clean) != 2:
        raise RuntimeError(f"Expected two {'internal' if internal else 'external'} contact roots for {site_key}; found {len(clean)}")
    return clean


def local_ca(c: dict, site_key: str, a: float, b: float) -> float:
    ref = 0.5 * (a + b)
    lo = (a - ref) * 86400.0
    hi = (b - ref) * 86400.0
    res = minimize_scalar(
        lambda seconds: sep_rad(vec(c, f"{site_key}_SUN", ref + float(seconds) / 86400.0), vec(c, f"{site_key}_VENUS", ref + float(seconds) / 86400.0)),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-4, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"Closest approach failed for {site_key}")
    return ref + float(res.x) / 86400.0


def geocentric_ca(c: dict) -> float:
    jds = np.asarray(c["JD_TDB"], dtype=float)
    vals = np.array([sep_rad(vec(c, "GEOCENTER_SUN", jd), vec(c, "GEOCENTER_VENUS", jd)) for jd in jds], dtype=float)
    i = int(np.argmin(vals))
    lo_jd = float(jds[max(0, i - 3)])
    hi_jd = float(jds[min(len(jds) - 1, i + 3)])
    ref = 0.5 * (lo_jd + hi_jd)
    lo_s = (lo_jd - ref) * 86400.0
    hi_s = (hi_jd - ref) * 86400.0
    res = minimize_scalar(
        lambda seconds: sep_rad(vec(c, "GEOCENTER_SUN", ref + float(seconds) / 86400.0), vec(c, "GEOCENTER_VENUS", ref + float(seconds) / 86400.0)),
        bounds=(lo_s, hi_s),
        method="bounded",
        options={"xatol": 1e-4, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError("Geocentric closest approach failed.")
    return ref + float(res.x) / 86400.0


def events_for(c: dict, site: dict) -> dict[str, float]:
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
        east = -east
        north = -north
    return center, east, north


def gnomonic(ray: np.ndarray, center: np.ndarray, east: np.ndarray, north: np.ndarray) -> np.ndarray:
    h = unit(ray)
    den = float(np.dot(h, center))
    if den <= 0.0:
        raise RuntimeError("Ray outside tangent hemisphere.")
    return np.array([float(np.dot(h, east)), float(np.dot(h, north))]) / den


def relative_xy_fixed(c: dict, site_key: str, jd: float, basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    center, east, north = basis
    sun = vec(c, f"{site_key}_SUN", jd)
    ven = vec(c, f"{site_key}_VENUS", jd)
    return ARC * (gnomonic(ven, center, east, north) - gnomonic(sun, center, east, north))


def relative_xy_instant(c: dict, site_key: str, jd: float) -> np.ndarray:
    return relative_xy_fixed(c, site_key, jd, basis_from_sun(vec(c, "GEOCENTER_SUN", jd)))


def fixed_algorithm_geometry(c: dict, geo_ca_jd: float) -> dict:
    basis = basis_from_sun(vec(c, "GEOCENTER_SUN", geo_ca_jd))
    h = 0.5 / 86400.0
    pv_plus = relative_xy_fixed(c, "POINT_VENUS", geo_ca_jd + h, basis)
    pv_minus = relative_xy_fixed(c, "POINT_VENUS", geo_ca_jd - h, basis)
    va_plus = relative_xy_fixed(c, "VARDO", geo_ca_jd + h, basis)
    va_minus = relative_xy_fixed(c, "VARDO", geo_ca_jd - h, basis)
    tangent = unit(unit(pv_plus - pv_minus) + unit(va_plus - va_minus))
    normal = np.array([-tangent[1], tangent[0]])
    pv0 = relative_xy_fixed(c, "POINT_VENUS", geo_ca_jd, basis)
    va0 = relative_xy_fixed(c, "VARDO", geo_ca_jd, basis)
    if float(np.dot(va0 - pv0, normal)) < 0.0:
        normal = -normal
    angle = math.degrees(math.atan2(tangent[1], tangent[0]))
    return {"basis": basis, "tangent": tangent, "normal": normal, "angle_deg": angle}


def fixed_apbp(c: dict, jd: float, geom: dict) -> dict[str, float]:
    pv = relative_xy_fixed(c, "POINT_VENUS", jd, geom["basis"])
    va = relative_xy_fixed(c, "VARDO", jd, geom["basis"])
    delta = va - pv
    return {"apbp_arcsec": float(np.dot(delta, geom["normal"])), "along_arcsec": float(np.dot(delta, geom["tangent"]))}


def instant_apbp(c: dict, jd: float) -> dict[str, float]:
    h = 60.0 / 86400.0
    pv = relative_xy_instant(c, "POINT_VENUS", jd)
    va = relative_xy_instant(c, "VARDO", jd)
    pv_vel = (relative_xy_instant(c, "POINT_VENUS", jd + h) - relative_xy_instant(c, "POINT_VENUS", jd - h)) / 120.0
    va_vel = (relative_xy_instant(c, "VARDO", jd + h) - relative_xy_instant(c, "VARDO", jd - h)) / 120.0
    tangent = unit(0.5 * (pv_vel + va_vel))
    normal = np.array([-tangent[1], tangent[0]])
    delta = va - pv
    if float(np.dot(delta, normal)) < 0.0:
        normal = -normal
    return {"apbp_arcsec": float(np.dot(delta, normal)), "along_arcsec": float(np.dot(delta, tangent)), "track_angle_deg": math.degrees(math.atan2(tangent[1], tangent[0]))}


def closest_sample(df: pd.DataFrame, column: str) -> dict[str, float | str]:
    y = df[column].to_numpy(float)
    i = int(np.argmin(y))
    return {"jd": float(df["jd_tdb"].iloc[i]), "utc": str(df["utc"].iloc[i]), "minute": float(df["minute_from_start"].iloc[i]), "value": float(y[i])}


def analyze() -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, float | str]]:
    OUT.mkdir(parents=True, exist_ok=True)
    master = build_master()
    c = splines(master)
    evs = {"Point Venus, Tahiti": events_for(c, POINT_VENUS), "Vardø, Norway": events_for(c, VARDO)}
    start = min(evs["Point Venus, Tahiti"]["C1"], evs["Vardø, Norway"]["C1"])
    stop = max(evs["Point Venus, Tahiti"]["C4"], evs["Vardø, Norway"]["C4"])
    geo_ca_jd = geocentric_ca(c)
    fixed_geom = fixed_algorithm_geometry(c, geo_ca_jd)
    minute_jds = np.asarray(c["JD_TDB"], dtype=float)
    minute_jds = minute_jds[(minute_jds >= start) & (minute_jds <= stop)]
    rows = []
    for jd in minute_jds:
        fixed = fixed_apbp(c, float(jd), fixed_geom)
        instant = instant_apbp(c, float(jd))
        rows.append({
            "jd_tdb": float(jd),
            "utc": utc(float(jd)),
            "minute_from_start": 0.0,
            "fixed_vector_apbp_arcsec": fixed["apbp_arcsec"],
            "fixed_vector_along_arcsec": fixed["along_arcsec"],
            "instant_apbp_arcsec": instant["apbp_arcsec"],
            "instant_along_arcsec": instant["along_arcsec"],
            "instant_track_angle_deg": instant["track_angle_deg"],
        })
    df = pd.DataFrame(rows)
    df["minute_from_start"] = (df["jd_tdb"] - float(df["jd_tdb"].iloc[0])) * 1440.0
    fixed_geo = fixed_apbp(c, geo_ca_jd, fixed_geom)
    instant_geo = instant_apbp(c, geo_ca_jd)
    fixed_min = closest_sample(df, "fixed_vector_apbp_arcsec")
    instant_min = closest_sample(df, "instant_apbp_arcsec")
    stats: dict[str, float | str] = {
        "geo_ca_jd": float(geo_ca_jd),
        "geo_ca_utc": utc(float(geo_ca_jd)),
        "geo_ca_minute": float((geo_ca_jd - float(df["jd_tdb"].iloc[0])) * 1440.0),
        "fixed_at_geo_ca": float(fixed_geo["apbp_arcsec"]),
        "instant_at_geo_ca": float(instant_geo["apbp_arcsec"]),
        "fixed_angle_deg": float(fixed_geom["angle_deg"]),
        "instant_angle_mean_deg": float(df["instant_track_angle_deg"].mean()),
        "delta_at_geo_ca": float(instant_geo["apbp_arcsec"] - fixed_geo["apbp_arcsec"]),
        "fixed_min": float(df["fixed_vector_apbp_arcsec"].min()),
        "fixed_max": float(df["fixed_vector_apbp_arcsec"].max()),
        "instant_min": float(df["instant_apbp_arcsec"].min()),
        "instant_max": float(df["instant_apbp_arcsec"].max()),
        "fixed_range": float(df["fixed_vector_apbp_arcsec"].max() - df["fixed_vector_apbp_arcsec"].min()),
        "instant_range": float(df["instant_apbp_arcsec"].max() - df["instant_apbp_arcsec"].min()),
        "rms_difference": float(np.sqrt(np.mean((df["instant_apbp_arcsec"].to_numpy(float) - df["fixed_vector_apbp_arcsec"].to_numpy(float)) ** 2))),
        "fixed_sample_min_jd": float(fixed_min["jd"]),
        "fixed_sample_min_utc": str(fixed_min["utc"]),
        "fixed_sample_min_minute": float(fixed_min["minute"]),
        "fixed_sample_min_value": float(fixed_min["value"]),
        "instant_sample_min_jd": float(instant_min["jd"]),
        "instant_sample_min_utc": str(instant_min["utc"]),
        "instant_sample_min_minute": float(instant_min["minute"]),
        "instant_sample_min_value": float(instant_min["value"]),
    }
    contacts = []
    for station, ev in evs.items():
        for event in EVENTS:
            contacts.append({"station": station, "event": event, "utc": utc(ev[event]), "jd_tdb": ev[event]})
    pd.DataFrame(contacts).to_csv(CONTACT_CSV, index=False, float_format="%.15f")
    return df, evs, stats


def style_axis(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.30, alpha=0.58)
    ax.tick_params(labelsize=7.2, width=0.35, length=2.4)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.35)


def mark_contact_lines(ax, df: pd.DataFrame, evs: dict[str, dict[str, float]], column: str) -> None:
    start_jd = float(df["jd_tdb"].iloc[0])
    interp = interp1d(df["jd_tdb"].to_numpy(float), df[column].to_numpy(float), kind="linear", bounds_error=False, fill_value="extrapolate")
    mean_events = {event: 0.5 * (evs["Point Venus, Tahiti"][event] + evs["Vardø, Norway"][event]) for event in EVENTS}
    for event in EVENTS:
        jd = float(mean_events[event])
        xm = (jd - start_jd) * 1440.0
        ym = float(interp(jd))
        color = GOLD if event == "CA" else BLUE
        ax.axvline(xm, color=color, linewidth=0.32, alpha=0.72, zorder=1)
        ax.scatter([xm], [ym], s=26 if event == "CA" else 14, color=color, edgecolors=FG, linewidths=0.22, zorder=5)
        dy = 0.030 if event in ("C1", "C2", "CA") else -0.034
        ax.annotate(event, xy=(xm, ym), xytext=(xm, ym + dy), ha="center", va="bottom" if dy > 0 else "top", fontsize=6.8, color=FG, fontweight="bold", arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.25})


def table_style(table, gold_rows=(), teal_rows=(), fontsize=6.1) -> None:
    table.auto_set_font_size(False)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#70879A")
        cell.set_linewidth(0.32)
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
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False, float_format="%.15f")
    plt.close("all")
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": FG,
        "axes.labelcolor": FG,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": MUTED,
    })
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.70, 0.30], left=0.055, right=0.985, top=0.900, bottom=0.105, wspace=0.075, hspace=0.190)
    ax_fixed = fig.add_subplot(gs[0, 0])
    ax_instant = fig.add_subplot(gs[0, 1], sharey=ax_fixed)
    tab_ax = fig.add_subplot(gs[1, :])
    fig.suptitle("1769 Venus Transit — A′B′ Fixed-Vector vs Instantaneous Tangent-Normal", fontsize=14.5, fontweight="bold", y=0.962)
    fig.text(0.5, 0.928, "Fresh JPL Horizons geometric ecliptic vectors; C1–C4 comparison of fixed geocentric common-normal A′B′ and instantaneous tangent-velocity A′B′.", ha="center", fontsize=7.4, color=MUTED)

    x = df["minute_from_start"].to_numpy(float)
    y_fixed = df["fixed_vector_apbp_arcsec"].to_numpy(float)
    y_instant = df["instant_apbp_arcsec"].to_numpy(float)
    ymin = min(float(y_fixed.min()), float(y_instant.min()))
    ymax = max(float(y_fixed.max()), float(y_instant.max()))
    pad = max(0.035, 0.25 * (ymax - ymin))

    ax_fixed.plot(x, y_fixed, color=GREEN, linewidth=0.58, zorder=3, label="fixed JPL vector common-normal A′B′")
    ax_fixed.scatter(x, y_fixed, s=3.5, color=GREEN, edgecolors="none", alpha=0.75, zorder=4)
    mark_contact_lines(ax_fixed, df, evs, "fixed_vector_apbp_arcsec")
    gx = float(stats["geo_ca_minute"])
    gy = float(stats["fixed_at_geo_ca"])
    ax_fixed.scatter([gx], [gy], s=58, marker="X", color=GOLD, edgecolors=FG, linewidths=0.35, zorder=7)
    ax_fixed.annotate(f"fixed geo CA A′B′ = {gy:.9f}″\n{stats['geo_ca_utc']}", xy=(gx, gy), xytext=(gx + 22, gy + 0.035), ha="left", va="bottom", fontsize=7.2, color=FG, arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.32})
    ax_fixed.set_title("FIXED JPL VECTOR ALGORITHM", fontsize=10, fontweight="bold")
    ax_fixed.set_xlabel("Elapsed time from first C1–C4 sample (minutes)", fontsize=8.8)
    ax_fixed.set_ylabel("A′B′ normal separation (arcsec)", fontsize=8.8)
    ax_fixed.set_xlim(x.min() - 5, x.max() + 5)
    ax_fixed.set_ylim(ymin - pad, ymax + pad)
    style_axis(ax_fixed)

    ax_instant.plot(x, y_instant, color=GOLD, linewidth=0.58, zorder=3, label="instantaneous tangent-velocity A′B′")
    ax_instant.scatter(x, y_instant, s=3.5, color=GOLD, edgecolors="none", alpha=0.75, zorder=4)
    mark_contact_lines(ax_instant, df, evs, "instant_apbp_arcsec")
    igx = float(stats["geo_ca_minute"])
    igy = float(stats["instant_at_geo_ca"])
    ax_instant.scatter([igx], [igy], s=62, marker="X", color=GOLD, edgecolors=FG, linewidths=0.38, zorder=8)
    ax_instant.annotate(f"instant geo CA A′B′ = {igy:.9f}″\n{stats['geo_ca_utc']}", xy=(igx, igy), xytext=(igx + 21, igy + 0.045), ha="left", va="bottom", fontsize=7.2, color=FG, arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.32})
    imx = float(stats["instant_sample_min_minute"])
    imy = float(stats["instant_sample_min_value"])
    ax_instant.scatter([imx], [imy], s=58, marker="o", color=RED, edgecolors=FG, linewidths=0.35, zorder=7)
    ax_instant.annotate(f"instant min A′B′ = {imy:.9f}″\n{stats['instant_sample_min_utc']}", xy=(imx, imy), xytext=(imx - 24, imy - 0.050), ha="right", va="top", fontsize=7.2, color=FG, arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.32})
    ax_instant.set_title("INSTANTANEOUS TANGENT-VELOCITY NORMAL", fontsize=10, fontweight="bold")
    ax_instant.set_xlabel("Elapsed time from first C1–C4 sample (minutes)", fontsize=8.8)
    ax_instant.set_xlim(x.min() - 5, x.max() + 5)
    style_axis(ax_instant)

    tab_ax.axis("off")
    rows = [
        ["Quantity", "Fixed JPL vector", "Instantaneous", "Difference", "Unit / trace"],
        ["A′B′ at geocentric CA", f"{float(stats['fixed_at_geo_ca']):.9f}", f"{float(stats['instant_at_geo_ca']):.9f}", f"{float(stats['delta_at_geo_ca']):+.9f}", "arcsec; same JD"],
        ["Minimum one-minute A′B′", f"{float(stats['fixed_sample_min_value']):.9f}", f"{float(stats['instant_sample_min_value']):.9f}", f"{float(stats['instant_sample_min_value']) - float(stats['fixed_sample_min_value']):+.9f}", "arcsec; each method min"],
        ["Maximum one-minute A′B′", f"{float(stats['fixed_max']):.9f}", f"{float(stats['instant_max']):.9f}", f"{float(stats['instant_max']) - float(stats['fixed_max']):+.9f}", "arcsec"],
        ["C1–C4 range", f"{float(stats['fixed_range']):.9f}", f"{float(stats['instant_range']):.9f}", f"{float(stats['instant_range']) - float(stats['fixed_range']):+.9f}", "arcsec"],
        ["RMS difference", "—", f"{float(stats['rms_difference']):.9f}", "—", "arcsec; instant minus fixed"],
        ["Track angle", f"{float(stats['fixed_angle_deg']):.9f}", f"{float(stats['instant_angle_mean_deg']):.9f}", f"{float(stats['instant_angle_mean_deg']) - float(stats['fixed_angle_deg']):+.9f}", "deg"],
        ["Geocentric CA UTC", str(stats["geo_ca_utc"]), str(stats["geo_ca_utc"]), "0", "fixed comparison timestamp"],
        ["Samples", f"{len(df):d}", f"{len(df):d}", "0", "one-minute JPL C1–C4 span"],
    ]
    table = tab_ax.table(cellText=rows, cellLoc="left", colWidths=[0.23, 0.18, 0.18, 0.16, 0.25], bbox=[0.0, 0.03, 1.0, 0.86])
    table_style(table, teal_rows=(1, 2, 5), gold_rows=(3, 4, 6, 7, 8), fontsize=6.4)

    fig.text(0.5, 0.043, f"File: {Path(__file__).name if '__file__' in globals() else 'VENUS_1769_APRIME_BPRIME_VECTOR_VS_INSTANT_V0094.py'} | Output: {PNG.name} | CSV: {CSV.name}", ha="center", fontsize=5.8, color=MUTED)
    fig.savefig(PNG, dpi=220, facecolor=BG)
    display(Image(filename=str(PNG)))


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"Start UTC: {START}; stop UTC: {STOP}; step: {STEP}")
    print("Sites: Point Venus, Tahiti; Vardø, Norway")
    print("Data source: fresh JPL Horizons geometric ecliptic vectors")
    print("COMMENTS")
    print("Compares fixed geocentric common-normal A′B′ against instantaneous tangent-velocity normal A′B′ over C1–C4.")
    print("Right panel now labels both instantaneous A′B′ at geocentric CA and instantaneous one-minute minimum.")
    df, evs, stats = analyze()
    plot(df, evs, stats)
    print("RESULTS")
    print(f"Fixed vector A′B′ at geocentric CA: {float(stats['fixed_at_geo_ca']):.12f} arcsec")
    print(f"Instantaneous A′B′ at geocentric CA: {float(stats['instant_at_geo_ca']):.12f} arcsec")
    print(f"Difference at geocentric CA: {float(stats['delta_at_geo_ca']):+.12f} arcsec")
    print(f"Fixed minimum one-minute A′B′: {float(stats['fixed_sample_min_value']):.12f} arcsec at {stats['fixed_sample_min_utc']}")
    print(f"Instantaneous minimum one-minute A′B′: {float(stats['instant_sample_min_value']):.12f} arcsec at {stats['instant_sample_min_utc']}")
    print(f"RMS instantaneous-minus-fixed difference: {float(stats['rms_difference']):.12f} arcsec")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    print(f"CONTACT CSV: {CONTACT_CSV}")
    print("PAPER COMPARISON")
    print("NOT USED: comparison is internal JPL-vector audit only; no literature values are imported.")
    print("EQUATION STATUS")
    print("PASS: fixed A′B′ uses geocentric CA basis and fixed tangent normal; instantaneous A′B′ uses per-minute basis and tangent-velocity normal; both use minute-by-minute JPL vectors.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0094
