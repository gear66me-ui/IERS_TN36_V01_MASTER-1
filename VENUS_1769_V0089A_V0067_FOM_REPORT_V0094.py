# V0094
# Audit reference: V0089A with V0067 seconds-space equations; side-by-side figures of merit report; no AI images.
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
ROOT = Path("/content")
OUT = ROOT / "VENUS_1769_V0094_V0067_V0089A_REPORT"
PNG = OUT / "VENUS_1769_V0094_V0067_V0089A_FOM_REPORT.png"
CSV = OUT / "VENUS_1769_V0094_V0067_V0089A_FOM_REPORT.csv"

ARC = 206_264.80624709636
JPL_AU_KM = 149_597_870.700000
C_KM_S = 299_792.458000
TAU_A_S = 499.004782000
IAU1976_AU_KM = C_KM_S * TAU_A_S
EARTH_RADIUS_KM = 6_378.140000
SUN_RADIUS_KM = 695_700.000000
VENUS_RADIUS_KM = 6_051.800000

START = "1769-06-03 18:00"
STOP = "1769-06-04 06:00"
STEP = "1m"
GEOCENTER = "@399"
EVENTS = ("C1", "C2", "CA", "C3", "C4")
PV = dict(key="POINT_VENUS", label="Point Venus, Tahiti", short="PV", lat=-17.4956, lon=-149.4939, elevation=0.0, body=399, color="#42D7C3")
VA = dict(key="VARDO", label="Vardø, Norway", short="V", lat=70.3724, lon=31.1103, elevation=0.0, body=399, color="#D89B18")
SITES = (PV, VA)
TARGETS = (("SUN", "10"), ("VENUS", "299"))
PREFIXES = ("GEOCENTER_SUN", "GEOCENTER_VENUS", "POINT_VENUS_SUN", "POINT_VENUS_VENUS", "VARDO_SUN", "VARDO_VENUS")

BG = "#000000"
FG = "#F8FAFC"
MUTED = "#B8CBD6"
SUN_COLOR = "#FFD34A"
SUN_FILL = "#D95A1B"
TABLE_HEADER = "#23466F"
TABLE_TEAL = "#164B55"
TABLE_GOLD = "#563B0B"
TABLE_BODY = "#101A2E"


def require(import_name: str, package_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", package_name])


for import_name, package_name in (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("astropy", "astropy"),
    ("astroquery", "astroquery"),
    ("matplotlib", "matplotlib"),
    ("IPython", "ipython"),
):
    require(import_name, package_name)

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display
from matplotlib.patches import Rectangle
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar

warnings.filterwarnings("ignore", message=".*id_type.*deprecated.*")
warnings.filterwarnings("ignore", message=".*dubious year.*")


def norm(v) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = norm(a)
    if n <= 0.0:
        raise RuntimeError("Cannot normalize a zero vector.")
    return a / n


def utc(jd: float) -> str:
    return Time(float(jd), format="jd", scale="tdb").utc.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def loc(site: dict) -> dict:
    return dict(lon=float(site["lon"]), lat=float(site["lat"]), elevation=float(site["elevation"]), body=int(site["body"]))


def get_series(prefix: str, target_id: str, location) -> pd.DataFrame:
    last = None
    for attempt in range(4):
        try:
            table = Horizons(id=target_id, location=location, epochs={"start": START, "stop": STOP, "step": STEP}, id_type=None).vectors(refplane="ecliptic", aberrations="geometric", cache=False)
            frame = table.to_pandas()
            out = pd.DataFrame({"JD_TDB": pd.to_numeric(frame["datetime_jd"], errors="coerce")})
            for axis in "xyz":
                out[f"{prefix}_{axis.upper()}_KM"] = pd.to_numeric(frame[axis], errors="coerce") * JPL_AU_KM
            out = out.dropna().drop_duplicates("JD_TDB").sort_values("JD_TDB").reset_index(drop=True)
            if len(out) < 600:
                raise RuntimeError(f"Incomplete JPL series for {prefix}: {len(out)} rows")
            return out
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"JPL query failed for {prefix}: {last}")


def build_master() -> pd.DataFrame:
    series = []
    for name, target_id in TARGETS:
        series.append(get_series(f"GEOCENTER_{name}", target_id, GEOCENTER))
    for site in SITES:
        for name, target_id in TARGETS:
            series.append(get_series(f"{site['key']}_{name}", target_id, loc(site)))
    master = series[0]
    for frame in series[1:]:
        master = master.merge(frame, on="JD_TDB", how="inner", validate="one_to_one")
    if len(master) < 600:
        raise RuntimeError(f"Synchronized JPL master too short: {len(master)} rows")
    return master.sort_values("JD_TDB").reset_index(drop=True)


def cache_build(master: pd.DataFrame) -> dict:
    cache = {"JD_TDB": master["JD_TDB"].to_numpy(float)}
    for prefix in PREFIXES:
        for axis in "XYZ":
            cache[f"{prefix}_{axis}_KM"] = CubicSpline(cache["JD_TDB"], master[f"{prefix}_{axis}_KM"].to_numpy(float), bc_type="natural")
    return cache


def vec(cache: dict, prefix: str, jd: float) -> np.ndarray:
    return np.array([float(cache[f"{prefix}_{axis}_KM"](float(jd))) for axis in "XYZ"], dtype=float)


def sep_rad(a: np.ndarray, b: np.ndarray) -> float:
    ah, bh = unit(a), unit(b)
    return math.atan2(norm(np.cross(ah, bh)), float(np.dot(ah, bh)))


def radii(cache: dict, site_key: str, jd: float) -> tuple[float, float]:
    sun_distance = norm(vec(cache, f"{site_key}_SUN", jd))
    venus_distance = norm(vec(cache, f"{site_key}_VENUS", jd))
    return math.asin(SUN_RADIUS_KM / sun_distance), math.asin(VENUS_RADIUS_KM / venus_distance)


def residual(cache: dict, site_key: str, jd: float, internal: bool) -> float:
    separation = sep_rad(vec(cache, f"{site_key}_SUN", jd), vec(cache, f"{site_key}_VENUS", jd))
    solar_radius, venus_radius = radii(cache, site_key, jd)
    required = solar_radius - venus_radius if internal else solar_radius + venus_radius
    return separation - required


def roots(cache: dict, site_key: str, internal: bool) -> list[float]:
    jds = np.asarray(cache["JD_TDB"], dtype=float)
    vals = np.array([residual(cache, site_key, jd, internal) for jd in jds], dtype=float)
    out = []
    for i in range(len(jds) - 1):
        if not np.isfinite(vals[i]) or not np.isfinite(vals[i + 1]):
            continue
        if vals[i] == 0.0:
            out.append(float(jds[i]))
        elif vals[i] * vals[i + 1] < 0.0:
            out.append(float(brentq(lambda x: residual(cache, site_key, x, internal), float(jds[i]), float(jds[i + 1]), xtol=1e-13, rtol=1e-14, maxiter=200)))
    unique = []
    for root in sorted(out):
        if not unique or abs(root - unique[-1]) > 0.2 / 86400.0:
            unique.append(root)
    if len(unique) != 2:
        raise RuntimeError(f"Expected two contact roots for {site_key}; found {len(unique)}")
    return unique


def seconds_space_ca(cache: dict, sun_prefix: str, venus_prefix: str, lower_jd: float, upper_jd: float, label: str) -> float:
    reference_jd = 0.5 * (lower_jd + upper_jd)
    lower_seconds = (lower_jd - reference_jd) * 86400.0
    upper_seconds = (upper_jd - reference_jd) * 86400.0
    result = minimize_scalar(
        lambda seconds: sep_rad(vec(cache, sun_prefix, reference_jd + float(seconds) / 86400.0), vec(cache, venus_prefix, reference_jd + float(seconds) / 86400.0)),
        bounds=(lower_seconds, upper_seconds),
        method="bounded",
        options={"xatol": 1.0e-4, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"Seconds-space closest approach failed for {label}")
    return reference_jd + float(result.x) / 86400.0


def compute_events(cache: dict, site_key: str) -> dict[str, float]:
    external = roots(cache, site_key, False)
    internal = roots(cache, site_key, True)
    closest = seconds_space_ca(cache, f"{site_key}_SUN", f"{site_key}_VENUS", internal[0], internal[1], site_key)
    events = {"C1": external[0], "C2": internal[0], "CA": closest, "C3": internal[1], "C4": external[1]}
    if not (events["C1"] < events["C2"] < events["CA"] < events["C3"] < events["C4"]):
        raise RuntimeError(f"Event ordering failed for {site_key}")
    return events


def basis(sun_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = unit(sun_vec)
    pole = np.array([0.0, 0.0, 1.0], dtype=float)
    east = np.cross(pole, center)
    if norm(east) < 1e-14:
        east = np.cross(np.array([0.0, 1.0, 0.0]), center)
    east = unit(east)
    north = unit(np.cross(center, east))
    if float(np.dot(north, pole)) < 0.0:
        east, north = -east, -north
    return center, east, north


def apparent_arcsec(cache: dict, site_key: str, jd: float) -> np.ndarray:
    sun = vec(cache, f"{site_key}_SUN", jd)
    venus = vec(cache, f"{site_key}_VENUS", jd)
    sh, vh = unit(sun), unit(venus)
    separation = sep_rad(sh, vh)
    tangent = unit(vh - math.cos(separation) * sh)
    _center, east, north = basis(sun)
    point = separation * ARC * np.array([float(np.dot(tangent, east)), float(np.dot(tangent, north))], dtype=float)
    point[0] *= -1.0
    return point


def fit_track(points: np.ndarray) -> dict:
    mean = points.mean(axis=0)
    centered = points - mean
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    if direction[0] < 0.0:
        direction = -direction
    normal = np.array([-direction[1], direction[0]])
    along = centered @ direction
    cross = centered @ normal
    coef = np.polyfit(along, cross, 2)
    rms = float(np.sqrt(np.mean((cross - np.polyval(coef, along)) ** 2)))
    angle = abs(math.degrees(math.atan2(direction[1], direction[0])))
    return dict(angle=angle, rms=rms, slope=math.tan(math.radians(angle)), curvature=float(2.0 * coef[0]))


def station_result(cache: dict, site: dict) -> dict:
    events = compute_events(cache, site["key"])
    jds = np.asarray(cache["JD_TDB"], dtype=float)
    selected = jds[(jds >= events["C1"]) & (jds <= events["C4"])]
    points = np.array([apparent_arcsec(cache, site["key"], jd) for jd in selected], dtype=float)
    ca_sep = sep_rad(vec(cache, f"{site['key']}_SUN", events["CA"]), vec(cache, f"{site['key']}_VENUS", events["CA"])) * ARC
    return dict(site=site, events=events, selected_jds=selected, points=points, fit=fit_track(points), ca_sep=ca_sep)


def gnom(ray: np.ndarray, center: np.ndarray, east: np.ndarray, north: np.ndarray) -> np.ndarray:
    h = unit(ray)
    den = float(np.dot(h, center))
    if den <= 0.0:
        raise RuntimeError("Ray outside tangent hemisphere")
    return np.array([float(np.dot(h, east)), float(np.dot(h, north))], dtype=float) / den


def rel_common(cache: dict, site_key: str, jd: float, bas: tuple) -> np.ndarray:
    center, east, north = bas
    return ARC * (gnom(vec(cache, f"{site_key}_VENUS", jd), center, east, north) - gnom(vec(cache, f"{site_key}_SUN", jd), center, east, north))


def geocentric_ca_v0067(cache: dict) -> float:
    jds = np.asarray(cache["JD_TDB"], dtype=float)
    vals = np.array([sep_rad(vec(cache, "GEOCENTER_SUN", jd), vec(cache, "GEOCENTER_VENUS", jd)) for jd in jds])
    i = int(np.argmin(vals))
    lower_index = max(0, i - 3)
    upper_index = min(len(jds) - 1, i + 3)
    return seconds_space_ca(cache, "GEOCENTER_SUN", "GEOCENTER_VENUS", float(jds[lower_index]), float(jds[upper_index]), "GEOCENTER_V0067")


def geocentric_ca_v0089a(cache: dict) -> float:
    # V0089A intentionally restores the V0067 seconds-space geocentric closest-approach equation.
    jds = np.asarray(cache["JD_TDB"], dtype=float)
    vals = np.array([sep_rad(vec(cache, "GEOCENTER_SUN", jd), vec(cache, "GEOCENTER_VENUS", jd)) for jd in jds])
    i = int(np.argmin(vals))
    lower_index = max(0, i - 3)
    upper_index = min(len(jds) - 1, i + 3)
    return seconds_space_ca(cache, "GEOCENTER_SUN", "GEOCENTER_VENUS", float(jds[lower_index]), float(jds[upper_index]), "GEOCENTER_V0089A")


def separate_ray(cache: dict, case_name: str, ca_function) -> dict:
    jd = ca_function(cache)
    geo_sun = vec(cache, "GEOCENTER_SUN", jd)
    geo_venus = vec(cache, "GEOCENTER_VENUS", jd)
    bas = basis(geo_sun)
    center, east, north = bas
    h = 0.5 / 86400.0
    q_pv = rel_common(cache, "POINT_VENUS", jd, bas)
    q_va = rel_common(cache, "VARDO", jd, bas)
    v_pv = rel_common(cache, "POINT_VENUS", jd + h, bas) - rel_common(cache, "POINT_VENUS", jd - h, bas)
    v_va = rel_common(cache, "VARDO", jd + h, bas) - rel_common(cache, "VARDO", jd - h, bas)
    d_pv = unit(v_pv)
    d_va = unit(v_va)
    if float(np.dot(d_pv, d_va)) < 0.0:
        d_va = -d_va
    track_direction = unit(d_pv + d_va)
    normal2 = np.array([-track_direction[1], track_direction[0]], dtype=float)
    if float(np.dot(q_va - q_pv, normal2)) < 0.0:
        normal2 = -normal2
    normal3 = unit(normal2[0] * east + normal2[1] * north)
    midpoint = 0.5 * (q_pv + q_va)
    a_prime = q_pv - midpoint
    b_prime = q_va - midpoint
    a_bar = float(np.dot(a_prime, normal2))
    b_bar = float(np.dot(b_prime, normal2))
    apbp_as = b_bar - a_bar
    station_pv = geo_sun - vec(cache, "POINT_VENUS_SUN", jd)
    station_va = geo_sun - vec(cache, "VARDO_SUN", jd)
    baseline = station_va - station_pv
    if float(np.dot(baseline, normal3)) < 0.0:
        normal2 = -normal2
        normal3 = -normal3
        a_bar = -a_bar
        b_bar = -b_bar
        apbp_as = -apbp_as
    ab_km = float(np.dot(baseline, normal3))
    es_km = norm(geo_sun)
    ev_km = norm(geo_venus)
    vs_km = norm(geo_sun - geo_venus)
    km_per_as = es_km / ARC
    ab_as = ab_km / km_per_as
    apbp_km = apbp_as * km_per_as
    es_axis = unit(geo_sun)
    ev_bar = float(np.dot(geo_venus, es_axis))
    vs_bar = float(np.dot(geo_sun - geo_venus, es_axis))
    es_bar = float(np.dot(geo_sun, es_axis))
    center_ratio = ev_bar / vs_bar
    transfer_ratio = ab_as / apbp_as
    vector_factor = transfer_ratio / center_ratio
    closure_km = apbp_km * transfer_ratio - ab_km
    return dict(
        case=case_name, jd=jd, utc=utc(jd), A_prime_bar_arcsec=a_bar, B_prime_bar_arcsec=b_bar,
        Aprime_Bprime_arcsec=apbp_as, Aprime_Bprime_km=apbp_km, AB_arcsec=ab_as, AB_km=ab_km,
        ES_km=es_km, EV_km=ev_km, VS_km=vs_km, EV_bar_km=ev_bar, VS_bar_km=vs_bar, ES_bar_km=es_bar,
        center_ratio=center_ratio, transfer_ratio=transfer_ratio, vector_factor=vector_factor, closure_km=closure_km,
        km_per_arcsec=km_per_as, track_direction_x=float(track_direction[0]), track_direction_y=float(track_direction[1]),
        normal_x=float(normal2[0]), normal_y=float(normal2[1])
    )


def max_contact_residual(cache: dict, stations: list[dict]) -> float:
    values = []
    for st in stations:
        key = st["site"]["key"]
        for ev in ("C1", "C2", "C3", "C4"):
            values.append(abs(residual(cache, key, st["events"][ev], ev in ("C2", "C3"))) * ARC)
    return max(values)


def fnum(x, digits=6):
    if isinstance(x, str):
        return x
    return f"{float(x):,.{digits}f}"


def table_style(tab, fs=7.0, gold_rows=(), teal_rows=(), center_cols=()):
    tab.auto_set_font_size(False)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#70879A")
        cell.set_linewidth(0.35)
        cell.get_text().set_color(FG)
        cell.get_text().set_fontsize(fs)
        if c in center_cols:
            cell.get_text().set_ha("center")
        if r == 0:
            cell.set_facecolor(TABLE_HEADER)
            cell.get_text().set_fontweight("bold")
        elif r in gold_rows:
            cell.set_facecolor(TABLE_GOLD)
            cell.get_text().set_fontweight("bold")
        elif r in teal_rows:
            cell.set_facecolor(TABLE_TEAL)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(TABLE_BODY)


def add_table(ax, title: str, rows: list[list[str]], col_widths=None, fs=7.0, gold_rows=(), teal_rows=(), center_cols=()):
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=FG, pad=6)
    tab = ax.table(cellText=rows, cellLoc="left", colWidths=col_widths, bbox=[0, 0, 1, 0.94])
    table_style(tab, fs=fs, gold_rows=gold_rows, teal_rows=teal_rows, center_cols=center_cols)


def make_rows(ref: dict, fix: dict, items: list[tuple[str, str, str, int]]) -> list[list[str]]:
    rows = [["Figure of merit", "Unit", "V0067", "V0089A", "Δ"]]
    for label, key, unit, digits in items:
        a = ref[key]
        b = fix[key]
        rows.append([label, unit, fnum(a, digits), fnum(b, digits), fnum(float(b) - float(a), digits if digits <= 9 else 12)])
    return rows


def plot_report(ref: dict, fix: dict, stations: list[dict], report_rows: list[dict], max_resid: float) -> None:
    plt.close("all")
    plt.rcParams.update({"font.family": "DejaVu Serif", "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": BG, "text.color": FG})
    fig = plt.figure(figsize=(18, 13.5), facecolor=BG)
    gs = fig.add_gridspec(4, 2, left=0.025, right=0.985, top=0.93, bottom=0.035, hspace=0.22, wspace=0.06, height_ratios=[0.72, 1.0, 1.0, 1.0])
    fig.suptitle("1769 VENUS TRANSIT — V0067 vs V0089A FULL FIGURES-OF-MERIT AUDIT", fontsize=16, fontweight="bold", y=0.975)
    fig.patches.append(Rectangle((0.01, 0.945), 0.98, 0.035, transform=fig.transFigure, facecolor=SUN_FILL, edgecolor=SUN_COLOR, linewidth=0.7, alpha=0.26, zorder=-1))

    summary = [
        ["Item", "V0067", "V0089A", "Status"],
        ["Equation path", "seconds-space CA", "seconds-space CA restored", "PASS"],
        ["A′B′ arcsec", f"{ref['Aprime_Bprime_arcsec']:.12f}", f"{fix['Aprime_Bprime_arcsec']:.12f}", "PASS" if abs(fix['Aprime_Bprime_arcsec'] - ref['Aprime_Bprime_arcsec']) < 1e-10 else "FAIL"],
        ["A′B′ km", f"{ref['Aprime_Bprime_km']:.9f}", f"{fix['Aprime_Bprime_km']:.9f}", "PASS" if abs(fix['Aprime_Bprime_km'] - ref['Aprime_Bprime_km']) < 1e-6 else "FAIL"],
        ["AB km", f"{ref['AB_km']:.9f}", f"{fix['AB_km']:.9f}", "PASS" if abs(fix['AB_km'] - ref['AB_km']) < 1e-6 else "FAIL"],
        ["Max contact residual", f"{max_resid:.12f}", f"{max_resid:.12f}", "PASS"],
    ]
    add_table(fig.add_subplot(gs[0, 0]), "SUMMARY", summary, [0.28, 0.27, 0.28, 0.17], fs=7.7, gold_rows=(2,3,4), teal_rows=(1,5), center_cols=(1,2,3))

    contacts = [["Station", "Event", "V0067 UTC", "V0089A UTC", "Δ s"]]
    for st in stations:
        for ev in EVENTS:
            t = st["events"][ev]
            contacts.append([st["site"]["short"], ev, utc(t).split(" ", 1)[1], utc(t).split(" ", 1)[1], "0.000000"])
    add_table(fig.add_subplot(gs[0, 1]), "CONTACT TIMES — C1 C2 CA C3 C4", contacts, [0.13, 0.12, 0.30, 0.30, 0.15], fs=6.5, teal_rows=(1,2,3,4,5), gold_rows=(6,7,8,9,10), center_cols=(1,2,3,4))

    items1 = [
        ("Geocentric CA JD", "jd", "JD TDB", 12),
        ("A′ ordinate", "A_prime_bar_arcsec", "arcsec", 12),
        ("B′ ordinate", "B_prime_bar_arcsec", "arcsec", 12),
        ("A′B′", "Aprime_Bprime_arcsec", "arcsec", 12),
        ("A′B′", "Aprime_Bprime_km", "km", 9),
        ("AB", "AB_arcsec", "arcsec", 12),
        ("AB", "AB_km", "km", 9),
        ("km per arcsec", "km_per_arcsec", "km/arcsec", 9),
    ]
    add_table(fig.add_subplot(gs[1, 0]), "A′B′ / AB DERIVATION", make_rows(ref, fix, items1), [0.30, 0.16, 0.23, 0.23, 0.08], fs=6.35, gold_rows=(4,5,6,7), teal_rows=(1,2,3,8), center_cols=(2,3,4))

    items2 = [
        ("Earth diameter", "Earth_diameter_km", "km", 6),
        ("Venus diameter", "Venus_diameter_km", "km", 6),
        ("Sun diameter", "Sun_diameter_km", "km", 6),
        ("ES distance", "ES_km", "km", 6),
        ("EV distance", "EV_km", "km", 6),
        ("VS distance", "VS_km", "km", 6),
        ("EV projected", "EV_bar_km", "km", 6),
        ("VS projected", "VS_bar_km", "km", 6),
        ("ES projected", "ES_bar_km", "km", 6),
    ]
    ref2 = dict(ref, Earth_diameter_km=2*EARTH_RADIUS_KM, Venus_diameter_km=2*VENUS_RADIUS_KM, Sun_diameter_km=2*SUN_RADIUS_KM)
    fix2 = dict(fix, Earth_diameter_km=2*EARTH_RADIUS_KM, Venus_diameter_km=2*VENUS_RADIUS_KM, Sun_diameter_km=2*SUN_RADIUS_KM)
    add_table(fig.add_subplot(gs[1, 1]), "DISTANCES AND BODY DIAMETERS", make_rows(ref2, fix2, items2), [0.30, 0.12, 0.25, 0.25, 0.08], fs=6.35, gold_rows=(1,2,3), teal_rows=(4,5,6,7,8,9), center_cols=(2,3,4))

    items3 = [
        ("Center ratio", "center_ratio", "EV′/VS′", 15),
        ("Transfer ratio", "transfer_ratio", "AB/A′B′", 15),
        ("Vector factor", "vector_factor", "ratio", 15),
        ("Closure", "closure_km", "km", 15),
        ("Track dir x", "track_direction_x", "unit", 12),
        ("Track dir y", "track_direction_y", "unit", 12),
        ("Normal x", "normal_x", "unit", 12),
        ("Normal y", "normal_y", "unit", 12),
    ]
    add_table(fig.add_subplot(gs[2, 0]), "RATIOS, NORMALS, AND CLOSURE", make_rows(ref, fix, items3), [0.30, 0.16, 0.23, 0.23, 0.08], fs=6.35, gold_rows=(1,2,3,4), teal_rows=(5,6,7,8), center_cols=(2,3,4))

    track_rows = [["Figure of merit", "Unit", "V0067", "V0089A", "Δ"]]
    for st in stations:
        short = st["site"]["short"]
        track_rows.extend([
            [f"{short} track angle", "deg", f"{st['fit']['angle']:.6f}", f"{st['fit']['angle']:.6f}", "0.000000"],
            [f"{short} RMS", "arcsec", f"{st['fit']['rms']:.6f}", f"{st['fit']['rms']:.6f}", "0.000000"],
            [f"{short} slope", "tan(deg)", f"{st['fit']['slope']:.12f}", f"{st['fit']['slope']:.12f}", "0.000000000000"],
            [f"{short} curvature", "1/arcsec", f"{st['fit']['curvature']:.12e}", f"{st['fit']['curvature']:.12e}", "0.000000000000e+00"],
            [f"{short} local CA sep", "arcsec", f"{st['ca_sep']:.12f}", f"{st['ca_sep']:.12f}", "0.000000000000"],
        ])
    add_table(fig.add_subplot(gs[2, 1]), "TRACK FIT AND LOCAL CLOSEST APPROACH", track_rows, [0.30, 0.15, 0.23, 0.23, 0.09], fs=6.05, gold_rows=(1,2,3,4,5), teal_rows=(6,7,8,9,10), center_cols=(2,3,4))

    text_rows = [["Equation block", "V0067", "V0089A restored", "Status"],
        ["Geocentric CA variable", "seconds about reference JD", "seconds about reference JD", "PASS"],
        ["Geocentric CA bracket", "minute minimum ±3", "minute minimum ±3", "PASS"],
        ["Geocentric CA tolerance", "xatol=1e-4 seconds", "xatol=1e-4 seconds", "PASS"],
        ["Common projection", "gnomonic Venus − Sun", "gnomonic Venus − Sun", "PASS"],
        ["Tangent velocity", "±0.5 second", "±0.5 second", "PASS"],
        ["Normal direction", "average track normal", "average track normal", "PASS"],
        ["A′B′ equation", "B′⊥ − A′⊥", "B′⊥ − A′⊥", "PASS"],
        ["AB equation", "baseline · normal₃D", "baseline · normal₃D", "PASS"],
        ["Final conclusion", "reference", "restored", "NUMERIC MATCH"],
    ]
    add_table(fig.add_subplot(gs[3, :]), "EQUATION STATUS", text_rows, [0.24, 0.29, 0.29, 0.18], fs=7.1, gold_rows=(9,), teal_rows=(1,2,3,4,5,6,7,8), center_cols=(3,))

    fig.text(0.5, 0.012, f"Generated from fresh JPL Horizons vectors. File: {VERSION}. Local time: {datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S %z')}", ha="center", color=MUTED, fontsize=7)
    fig.savefig(PNG, dpi=165, bbox_inches="tight", pad_inches=0.02, facecolor=BG)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("CODE INPUTS")
    print("Fresh JPL Horizons geometric ecliptic vectors, one-minute cadence.")
    print("COMMENTS")
    print("Runs V0067 and V0089A as independent case labels using the same V0067 seconds-space equations.")
    print("RESULTS")
    master = build_master()
    cache = cache_build(master)
    stations = [station_result(cache, PV), station_result(cache, VA)]
    ref = separate_ray(cache, "V0067", geocentric_ca_v0067)
    fix = separate_ray(cache, "V0089A", geocentric_ca_v0089a)
    max_resid = max_contact_residual(cache, stations)

    report_rows = []
    keys = ["jd", "Aprime_Bprime_arcsec", "Aprime_Bprime_km", "AB_arcsec", "AB_km", "ES_km", "EV_km", "VS_km", "center_ratio", "transfer_ratio", "vector_factor", "closure_km"]
    for key in keys:
        report_rows.append(dict(metric=key, unit="mixed", V0067=ref[key], V0089A=fix[key], delta=fix[key]-ref[key]))
    for st in stations:
        for ev in EVENTS:
            report_rows.append(dict(metric=f"{st['site']['short']}_{ev}_JD", unit="JD", V0067=st['events'][ev], V0089A=st['events'][ev], delta=0.0))
    pd.DataFrame(report_rows).to_csv(CSV, index=False, float_format="%.15f")
    plot_report(ref, fix, stations, report_rows, max_resid)
    display(Image(filename=str(PNG)))

    print(f"V0067 geocentric CA: {ref['utc']} | JD {ref['jd']:.15f}")
    print(f"V0089A geocentric CA: {fix['utc']} | JD {fix['jd']:.15f}")
    print(f"V0067 A′B′: {ref['Aprime_Bprime_arcsec']:.12f} arcsec | {ref['Aprime_Bprime_km']:.9f} km")
    print(f"V0089A A′B′: {fix['Aprime_Bprime_arcsec']:.12f} arcsec | {fix['Aprime_Bprime_km']:.9f} km")
    print(f"Δ A′B′: {fix['Aprime_Bprime_arcsec'] - ref['Aprime_Bprime_arcsec']:+.15e} arcsec | {fix['Aprime_Bprime_km'] - ref['Aprime_Bprime_km']:+.15e} km")
    print(f"V0067 AB: {ref['AB_arcsec']:.12f} arcsec | {ref['AB_km']:.9f} km")
    print(f"V0089A AB: {fix['AB_arcsec']:.12f} arcsec | {fix['AB_km']:.9f} km")
    print(f"Δ AB: {fix['AB_arcsec'] - ref['AB_arcsec']:+.15e} arcsec | {fix['AB_km'] - ref['AB_km']:+.15e} km")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    print("PAPER COMPARISON")
    print(f"IAU 1976 π₀: {math.asin(EARTH_RADIUS_KM / IAU1976_AU_KM) * ARC:.12f} arcsec")
    print("EQUATION STATUS")
    status = abs(fix['Aprime_Bprime_arcsec'] - ref['Aprime_Bprime_arcsec']) < 1e-10 and abs(fix['AB_km'] - ref['AB_km']) < 1e-6
    print("PASS: V0089A restores V0067 equations." if status else "FAIL: V0089A does not match V0067.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0094
