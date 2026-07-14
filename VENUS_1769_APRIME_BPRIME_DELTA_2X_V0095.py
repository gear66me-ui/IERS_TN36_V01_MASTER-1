# V0095
# Audit reference: standalone JPL Horizons A-prime/B-prime delta audit; 2x amplified Matplotlib plot; no AI images.
from __future__ import annotations

import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0095"
LOCAL_TZ = ZoneInfo("America/Bogota")
OUT = Path("/content/VENUS_1769_APRIME_BPRIME_DELTA_2X_V0095_OUTPUT")
PNG = OUT / "VENUS_1769_APRIME_BPRIME_DELTA_2X_V0095.png"
CSV = OUT / "VENUS_1769_APRIME_BPRIME_DELTA_2X_V0095.csv"
CONTACT_CSV = OUT / "VENUS_1769_APRIME_BPRIME_DELTA_CONTACTS_V0095.csv"

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


def utc_time(jd: float) -> str:
    return utc(jd).split(" ", 1)[1]


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
    for target_name, target_id in (("SUN", "10"), ("VENUS", "299")):
        parts.append(download(f"GEOCENTER_{target_name}", target_id, "@399"))
    for site in SITES:
        for target_name, target_id in (("SUN", "10"), ("VENUS", "299")):
            parts.append(download(f"{site['key']}_{target_name}", target_id, loc(site)))
    master = parts[0]
    for df in parts[1:]:
        master = master.merge(df, on="JD_TDB", how="inner", validate="one_to_one")
    master = master.sort_values("JD_TDB").reset_index(drop=True)
    if len(master) < 600:
        raise RuntimeError(f"Synchronized master frame too short: {len(master)} rows")
    return master


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
    ah = unit(a)
    bh = unit(b)
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
    out: list[float] = []
    for i in range(len(jds) - 1):
        if not np.isfinite(vals[i]) or not np.isfinite(vals[i + 1]):
            continue
        if vals[i] == 0.0:
            out.append(float(jds[i]))
        elif vals[i] * vals[i + 1] < 0.0:
            out.append(float(brentq(lambda x: residual(c, site_key, x, internal), float(jds[i]), float(jds[i + 1]), xtol=1e-13, rtol=1e-14)))
    clean: list[float] = []
    for r in sorted(out):
        if not clean or abs(r - clean[-1]) > 0.2 / 86400.0:
            clean.append(r)
    if len(clean) != 2:
        raise RuntimeError(f"Expected two {'internal' if internal else 'external'} roots for {site_key}; found {len(clean)}")
    return clean


def local_ca(c: dict[str, object], site_key: str, a: float, b: float) -> float:
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


def geocentric_ca(c: dict[str, object]) -> float:
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
        east = -east
        north = -north
    return center, east, north


def gnomonic(ray: np.ndarray, center: np.ndarray, east: np.ndarray, north: np.ndarray) -> np.ndarray:
    h = unit(ray)
    den = float(np.dot(h, center))
    if den <= 0.0:
        raise RuntimeError("Ray outside tangent hemisphere.")
    return np.array([float(np.dot(h, east)), float(np.dot(h, north))]) / den


def relative_xy_fixed(c: dict[str, object], site_key: str, jd: float, basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    center, east, north = basis
    sun = vec(c, f"{site_key}_SUN", jd)
    ven = vec(c, f"{site_key}_VENUS", jd)
    return ARC * (gnomonic(ven, center, east, north) - gnomonic(sun, center, east, north))


def relative_xy_instant(c: dict[str, object], site_key: str, jd: float) -> np.ndarray:
    return relative_xy_fixed(c, site_key, jd, basis_from_sun(vec(c, "GEOCENTER_SUN", jd)))


def fixed_algorithm_geometry(c: dict[str, object], geo_ca_jd: float) -> dict[str, object]:
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


def fixed_apbp(c: dict[str, object], jd: float, g: dict[str, object]) -> float:
    pv = relative_xy_fixed(c, "POINT_VENUS", jd, g["basis"])
    va = relative_xy_fixed(c, "VARDO", jd, g["basis"])
    return float(np.dot(va - pv, g["normal"]))


def instant_apbp(c: dict[str, object], jd: float) -> tuple[float, float]:
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
    angle = math.degrees(math.atan2(tangent[1], tangent[0]))
    return float(np.dot(delta, normal)), angle


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
    rows: list[dict[str, float | str]] = []
    for jd in minute_jds:
        fixed_value = fixed_apbp(c, float(jd), fixed_geom)
        instant_value, instant_angle = instant_apbp(c, float(jd))
        delta = instant_value - fixed_value
        rows.append({
            "jd_tdb": float(jd),
            "utc": utc(float(jd)),
            "minute_from_start": 0.0,
            "fixed_vector_apbp_arcsec": fixed_value,
            "instant_apbp_arcsec": instant_value,
            "delta_arcsec": delta,
            "delta_2x_arcsec": 2.0 * delta,
            "instant_track_angle_deg": instant_angle,
        })
    df = pd.DataFrame(rows)
    df["minute_from_start"] = (df["jd_tdb"] - float(df["jd_tdb"].iloc[0])) * 1440.0
    fixed_geo = fixed_apbp(c, geo_ca_jd, fixed_geom)
    instant_geo, instant_geo_angle = instant_apbp(c, geo_ca_jd)
    delta_geo = instant_geo - fixed_geo
    delta_array = df["delta_arcsec"].to_numpy(float)
    amplified = df["delta_2x_arcsec"].to_numpy(float)
    abs_index = int(np.argmax(np.abs(delta_array)))
    stats: dict[str, float | str] = {
        "geo_ca_jd": float(geo_ca_jd),
        "geo_ca_utc": utc(float(geo_ca_jd)),
        "fixed_at_geo_ca": float(fixed_geo),
        "instant_at_geo_ca": float(instant_geo),
        "delta_at_geo_ca": float(delta_geo),
        "delta_2x_at_geo_ca": float(2.0 * delta_geo),
        "delta_mean": float(np.mean(delta_array)),
        "delta_2x_mean": float(np.mean(amplified)),
        "delta_min": float(np.min(delta_array)),
        "delta_max": float(np.max(delta_array)),
        "delta_range": float(np.max(delta_array) - np.min(delta_array)),
        "delta_rms": float(np.sqrt(np.mean(delta_array ** 2))),
        "delta_abs_max": float(delta_array[abs_index]),
        "delta_abs_max_2x": float(amplified[abs_index]),
        "delta_abs_max_utc": str(df["utc"].iloc[abs_index]),
        "delta_abs_max_minute": float(df["minute_from_start"].iloc[abs_index]),
        "fixed_angle_deg": float(fixed_geom["angle_deg"]),
        "instant_geo_angle_deg": float(instant_geo_angle),
        "instant_angle_mean_deg": float(df["instant_track_angle_deg"].mean()),
        "samples": float(len(df)),
    }
    contacts: list[dict[str, float | str]] = []
    for station, ev in evs.items():
        for event in EVENTS:
            contacts.append({"station": station, "event": event, "utc": utc(ev[event]), "jd_tdb": ev[event]})
    pd.DataFrame(contacts).to_csv(CONTACT_CSV, index=False, float_format="%.15f")
    df.to_csv(CSV, index=False, float_format="%.15f")
    return df, evs, stats


def style_axis(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.30, alpha=0.58)
    ax.tick_params(labelsize=7.5, width=0.35, length=2.4)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.35)


def table_style(table, gold_rows=(), teal_rows=(), red_rows=(), fontsize=6.0) -> None:
    table.auto_set_font_size(False)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#70879A")
        cell.set_linewidth(0.32)
        cell.get_text().set_color(FG)
        cell.get_text().set_fontsize(fontsize)
        if row == 0:
            cell.set_facecolor(TABLE_HEADER)
            cell.get_text().set_fontweight("bold")
        elif row in red_rows:
            cell.set_facecolor("#5A1018")
            cell.get_text().set_fontweight("bold")
        elif row in gold_rows:
            cell.set_facecolor(TABLE_GOLD)
            cell.get_text().set_fontweight("bold")
        elif row in teal_rows:
            cell.set_facecolor(TABLE_TEAL)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(TABLE_BODY)


def mark_events(ax, df: pd.DataFrame, evs: dict[str, dict[str, float]], y_column: str) -> None:
    start_jd = float(df["jd_tdb"].iloc[0])
    interp = interp1d(df["jd_tdb"].to_numpy(float), df[y_column].to_numpy(float), kind="linear", bounds_error=False, fill_value="extrapolate")
    mean_events = {event: 0.5 * (evs["Point Venus, Tahiti"][event] + evs["Vardø, Norway"][event]) for event in EVENTS}
    for event in EVENTS:
        jd = float(mean_events[event])
        x = (jd - start_jd) * 1440.0
        y = float(interp(jd))
        color = GOLD if event == "CA" else BLUE
        ax.axvline(x, color=color, linewidth=0.30, alpha=0.66, zorder=1)
        ax.scatter([x], [y], s=26 if event == "CA" else 13, color=color, edgecolors=FG, linewidths=0.20, zorder=6)
        dy = 0.10 * max(1e-12, float(np.ptp(df[y_column].to_numpy(float))))
        dy = dy if event in ("C1", "C2", "CA") else -dy
        ax.annotate(event, xy=(x, y), xytext=(x, y + dy), ha="center", va="bottom" if dy > 0 else "top", fontsize=6.8, color=FG, fontweight="bold", arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.24})


def plot(df: pd.DataFrame, evs: dict[str, dict[str, float]], stats: dict[str, float | str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
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
    gs = fig.add_gridspec(2, 1, height_ratios=[0.69, 0.31], left=0.065, right=0.985, top=0.895, bottom=0.100, hspace=0.175)
    ax = fig.add_subplot(gs[0, 0])
    tab_ax = fig.add_subplot(gs[1, 0])
    fig.suptitle("1769 Venus Transit — Delta Between Fixed-Vector and Instantaneous A′B′", fontsize=14.5, fontweight="bold", y=0.962)
    fig.text(0.5, 0.928, "Plotted value is 2× amplified: 2·(A′B′instantaneous − A′B′fixed). Average line is also amplified 2×; table reports raw and amplified values.", ha="center", fontsize=7.4, color=MUTED)

    x = df["minute_from_start"].to_numpy(float)
    y_raw = df["delta_arcsec"].to_numpy(float)
    y = df["delta_2x_arcsec"].to_numpy(float)
    y_mean = float(stats["delta_2x_mean"])
    y_zero = 0.0
    y_absmax = float(stats["delta_abs_max_2x"])
    x_absmax = float(stats["delta_abs_max_minute"])
    pad = max(0.00001, 0.24 * max(1e-12, float(y.max() - y.min())))

    ax.axhline(y_zero, color=MUTED, linewidth=0.38, alpha=0.75, zorder=1, label="zero delta")
    ax.axhline(y_mean, color=RED, linewidth=0.72, linestyle="--", alpha=0.95, zorder=2, label=f"2× average delta = {y_mean:+.9f}″")
    ax.plot(x, y, color=GREEN, linewidth=0.55, zorder=4, label="2× amplified delta")
    ax.scatter(x, y, s=4.2, color=GOLD, edgecolors="none", alpha=0.76, zorder=5, label="one-minute JPL samples")
    mark_events(ax, df, evs, "delta_2x_arcsec")
    ax.scatter([x_absmax], [y_absmax], s=64, marker="X", color=RED, edgecolors=FG, linewidths=0.35, zorder=8)
    ax.annotate(f"max |2×Δ| = {y_absmax:+.9f}″\nraw Δ = {float(stats['delta_abs_max']):+.9f}″\n{stats['delta_abs_max_utc']}", xy=(x_absmax, y_absmax), xytext=(x_absmax + 18.0, y_absmax + (0.22 * pad if y_absmax >= y_mean else -0.22 * pad)), ha="left", va="bottom" if y_absmax >= y_mean else "top", fontsize=7.4, color=FG, arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.32})
    gx = (float(stats["geo_ca_jd"]) - float(df["jd_tdb"].iloc[0])) * 1440.0
    gy = float(stats["delta_2x_at_geo_ca"])
    ax.scatter([gx], [gy], s=54, marker="D", color=BLUE, edgecolors=FG, linewidths=0.32, zorder=8)
    ax.annotate(f"geo CA 2×Δ = {gy:+.9f}″\nraw Δ = {float(stats['delta_at_geo_ca']):+.9f}″", xy=(gx, gy), xytext=(gx - 20.0, gy - (0.40 * pad)), ha="right", va="top", fontsize=7.4, color=FG, arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.32})

    ax.set_xlabel("Elapsed time from first C1–C4 sample (minutes)", fontsize=9.2)
    ax.set_ylabel("2× amplified ΔA′B′ (arcsec)", fontsize=9.2)
    ax.set_xlim(x.min() - 5.0, x.max() + 5.0)
    ax.set_ylim(float(y.min()) - pad, float(y.max()) + pad)
    style_axis(ax)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=7.6)
    for text in leg.get_texts():
        text.set_color(FG)

    tab_ax.axis("off")
    rows = [
        ["Quantity", "Raw Δ arcsec", "2× plotted arcsec", "Trace / status"],
        ["Δ at geocentric CA", f"{float(stats['delta_at_geo_ca']):+.12f}", f"{float(stats['delta_2x_at_geo_ca']):+.12f}", "same JD; plotted blue diamond"],
        ["Average Δ", f"{float(stats['delta_mean']):+.12f}", f"{float(stats['delta_2x_mean']):+.12f}", "horizontal dashed red line"],
        ["Minimum Δ", f"{float(stats['delta_min']):+.12f}", f"{2.0 * float(stats['delta_min']):+.12f}", "one-minute JPL samples"],
        ["Maximum Δ", f"{float(stats['delta_max']):+.12f}", f"{2.0 * float(stats['delta_max']):+.12f}", "one-minute JPL samples"],
        ["Range Δ", f"{float(stats['delta_range']):+.12f}", f"{2.0 * float(stats['delta_range']):+.12f}", "max − min"],
        ["RMS Δ", f"{float(stats['delta_rms']):+.12f}", f"{2.0 * float(stats['delta_rms']):+.12f}", "root mean square"],
        ["Max |Δ|", f"{float(stats['delta_abs_max']):+.12f}", f"{float(stats['delta_abs_max_2x']):+.12f}", f"{stats['delta_abs_max_utc']}"],
        ["Samples", f"{int(float(stats['samples']))}", f"{int(float(stats['samples']))}", "one-minute C1–C4 span"],
    ]
    table = tab_ax.table(cellText=rows, cellLoc="left", colWidths=[0.22, 0.22, 0.22, 0.34], bbox=[0.0, 0.05, 1.0, 0.86])
    table_style(table, teal_rows=(1, 2), gold_rows=(3, 4, 5, 6, 8), red_rows=(7,), fontsize=6.4)
    fig.text(0.5, 0.043, f"File: {Path(__file__).name if '__file__' in globals() else 'VENUS_1769_APRIME_BPRIME_DELTA_2X_V0095.py'} | Output: {PNG.name} | CSV: {CSV.name}", ha="center", fontsize=5.8, color=MUTED)
    fig.savefig(PNG, dpi=230, facecolor=BG)
    display(Image(filename=str(PNG)))


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"Start UTC: {START}; stop UTC: {STOP}; step: {STEP}")
    print("Sites: Point Venus, Tahiti; Vardø, Norway")
    print("Data source: fresh JPL Horizons geometric ecliptic vectors")
    print("Amplification: plotted delta = 2 × (instantaneous A′B′ − fixed-vector A′B′)")
    print("COMMENTS")
    print("Quantifies the delta between fixed geocentric common-normal A′B′ and instantaneous tangent-velocity normal A′B′ over C1–C4.")
    print("No external project file is fetched; calculations are recomputed from JPL vectors; output is Matplotlib only.")
    df, evs, stats = analyze()
    plot(df, evs, stats)
    print("RESULTS")
    print(f"Delta at geocentric CA: {float(stats['delta_at_geo_ca']):+.12f} arcsec")
    print(f"2x delta at geocentric CA: {float(stats['delta_2x_at_geo_ca']):+.12f} arcsec")
    print(f"Average delta: {float(stats['delta_mean']):+.12f} arcsec")
    print(f"2x average delta: {float(stats['delta_2x_mean']):+.12f} arcsec")
    print(f"Delta min/max: {float(stats['delta_min']):+.12f} / {float(stats['delta_max']):+.12f} arcsec")
    print(f"Delta RMS: {float(stats['delta_rms']):+.12f} arcsec")
    print(f"Max absolute delta: {float(stats['delta_abs_max']):+.12f} arcsec at {stats['delta_abs_max_utc']}")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    print(f"CONTACT CSV: {CONTACT_CSV}")
    print("PAPER COMPARISON")
    print("NOT USED: comparison is internal JPL-vector audit only; no literature values are imported.")
    print("EQUATION STATUS")
    print("PASS: delta is computed point-by-point as A′B′_instantaneous minus A′B′_fixed; plotted line is exactly 2x delta; average line is exactly 2x mean delta.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0095
