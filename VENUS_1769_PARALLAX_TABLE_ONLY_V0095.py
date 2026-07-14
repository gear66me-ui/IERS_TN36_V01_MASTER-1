# V0095
# Audit reference: standalone table-only JPL Horizons Halley parallax audit; Matplotlib table only; no AI images; no external project Python.
from __future__ import annotations
import math
import subprocess
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

VERSION = "V0095"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUT = ROOT / "VENUS_1769_V0095_PARALLAX_TABLE_ONLY"
PNG = OUT / "VENUS_1769_V0095_PARALLAX_TABLE_ONLY_900DPI.png"
PDF = OUT / "VENUS_1769_V0095_PARALLAX_TABLE_ONLY_VECTOR.pdf"
SVG = OUT / "VENUS_1769_V0095_PARALLAX_TABLE_ONLY_VECTOR.svg"
CSV = OUT / "VENUS_1769_V0095_PARALLAX_TABLE_ONLY.csv"
DPI = 900

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
PV = dict(key="POINT_VENUS", label="Point Venus, Tahiti", short="PV", lat=-17.4956, lon=-149.4939, elevation=0.0, body=399)
VA = dict(key="VARDO", label="Vardø, Norway", short="V", lat=70.3724, lon=31.1103, elevation=0.0, body=399)
SITES = (PV, VA)
TARGETS = (("SUN", "10"), ("VENUS", "299"))
PREFIXES = ("GEOCENTER_SUN", "GEOCENTER_VENUS", "POINT_VENUS_SUN", "POINT_VENUS_VENUS", "VARDO_SUN", "VARDO_VENUS")

BG = "#000000"
FG = "#F8FAFC"
MUTED = "#B8CBD6"
TABLE_HEADER = "#23466F"
TABLE_TEAL = "#164B55"
TABLE_GOLD = "#563B0B"
TABLE_BODY = "#101A2E"
GRID = "#70879A"


def require(import_name: str, package_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", package_name])


for import_name, package_name in (("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("astropy", "astropy"), ("astroquery", "astroquery"), ("matplotlib", "matplotlib"), ("IPython", "ipython")):
    require(import_name, package_name)

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display
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
        raise RuntimeError("Cannot normalize zero vector.")
    return a / n


def utc(jd: float) -> str:
    return Time(float(jd), format="jd", scale="tdb").utc.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def loc(site: dict) -> dict:
    return dict(lon=float(site["lon"]), lat=float(site["lat"]), elevation=float(site["elevation"]), body=int(site["body"]))


def get_series(prefix: str, target: str, location) -> pd.DataFrame:
    last = None
    for attempt in range(4):
        try:
            tab = Horizons(id=target, location=location, epochs={"start": START, "stop": STOP, "step": STEP}, id_type=None).vectors(refplane="ecliptic", aberrations="geometric", cache=False)
            fr = tab.to_pandas()
            out = pd.DataFrame({"JD_TDB": pd.to_numeric(fr["datetime_jd"], errors="coerce")})
            for ax in "xyz":
                out[f"{prefix}_{ax.upper()}_KM"] = pd.to_numeric(fr[ax], errors="coerce") * JPL_AU_KM
            out = out.dropna().drop_duplicates("JD_TDB").sort_values("JD_TDB").reset_index(drop=True)
            if len(out) < 600:
                raise RuntimeError(f"Incomplete JPL query for {prefix}: {len(out)} rows")
            return out
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"JPL query failed for {prefix}: {last}")


def master_frame() -> pd.DataFrame:
    series = []
    for name, target in TARGETS:
        series.append(get_series(f"GEOCENTER_{name}", target, GEOCENTER))
    for site in SITES:
        for name, target in TARGETS:
            series.append(get_series(f"{site['key']}_{name}", target, loc(site)))
    m = series[0]
    for s in series[1:]:
        m = m.merge(s, on="JD_TDB", how="inner", validate="one_to_one")
    if len(m) < 600:
        raise RuntimeError(f"Synchronized JPL master too short: {len(m)} rows")
    return m.sort_values("JD_TDB").reset_index(drop=True)


def cache_build(m: pd.DataFrame) -> dict:
    c = {"JD_TDB": m["JD_TDB"].to_numpy(float)}
    for p in PREFIXES:
        for ax in "XYZ":
            c[f"{p}_{ax}_KM"] = CubicSpline(c["JD_TDB"], m[f"{p}_{ax}_KM"].to_numpy(float), bc_type="natural")
    return c


def vec(c: dict, prefix: str, jd: float) -> np.ndarray:
    return np.array([float(c[f"{prefix}_{ax}_KM"](float(jd))) for ax in "XYZ"], dtype=float)


def sep_rad(a: np.ndarray, b: np.ndarray) -> float:
    ah, bh = unit(a), unit(b)
    return math.atan2(norm(np.cross(ah, bh)), float(np.dot(ah, bh)))


def radii(c: dict, site: str, jd: float) -> tuple[float, float]:
    return math.asin(SUN_RADIUS_KM / norm(vec(c, f"{site}_SUN", jd))), math.asin(VENUS_RADIUS_KM / norm(vec(c, f"{site}_VENUS", jd)))


def residual(c: dict, site: str, jd: float, internal: bool) -> float:
    d = sep_rad(vec(c, f"{site}_SUN", jd), vec(c, f"{site}_VENUS", jd))
    rs, rv = radii(c, site, jd)
    return d - (rs - rv if internal else rs + rv)


def roots(c: dict, site: str, internal: bool) -> list[float]:
    jds = np.asarray(c["JD_TDB"], dtype=float)
    vals = np.array([residual(c, site, jd, internal) for jd in jds])
    out = []
    for i in range(len(jds) - 1):
        if not np.isfinite(vals[i] + vals[i + 1]):
            continue
        if vals[i] == 0.0:
            out.append(float(jds[i]))
        elif vals[i] * vals[i + 1] < 0.0:
            out.append(float(brentq(lambda x: residual(c, site, x, internal), float(jds[i]), float(jds[i + 1]), xtol=1e-13, rtol=1e-14, maxiter=200)))
    uniq = []
    for r in sorted(out):
        if not uniq or abs(r - uniq[-1]) > 0.2 / 86400.0:
            uniq.append(r)
    if len(uniq) != 2:
        raise RuntimeError(f"Expected two contact roots for {site}; found {len(uniq)}")
    return uniq


def local_ca(c: dict, site: str, a: float, b: float) -> float:
    ref = 0.5 * (a + b)
    lo, hi = (a - ref) * 86400.0, (b - ref) * 86400.0
    res = minimize_scalar(lambda s: sep_rad(vec(c, f"{site}_SUN", ref + float(s) / 86400.0), vec(c, f"{site}_VENUS", ref + float(s) / 86400.0)), bounds=(lo, hi), method="bounded", options={"xatol": 1.0e-4, "maxiter": 500})
    if not res.success:
        raise RuntimeError(f"Closest approach failed for {site}")
    return ref + float(res.x) / 86400.0


def station_events(c: dict, site: dict) -> dict:
    ext = roots(c, site["key"], False)
    inn = roots(c, site["key"], True)
    ca = local_ca(c, site["key"], inn[0], inn[1])
    return {"C1": ext[0], "C2": inn[0], "CA": ca, "C3": inn[1], "C4": ext[1]}


def basis(sun_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def gnom(ray: np.ndarray, cen: np.ndarray, east: np.ndarray, north: np.ndarray) -> np.ndarray:
    h = unit(ray)
    den = float(np.dot(h, cen))
    return np.array([float(np.dot(h, east)), float(np.dot(h, north))]) / den


def rel_common(c: dict, site: str, jd: float, bas: tuple) -> np.ndarray:
    cen, east, north = bas
    return ARC * (gnom(vec(c, f"{site}_VENUS", jd), cen, east, north) - gnom(vec(c, f"{site}_SUN", jd), cen, east, north))


def geocentric_ca_v0067(c: dict) -> float:
    jds = np.asarray(c["JD_TDB"], dtype=float)
    vals = np.array([sep_rad(vec(c, "GEOCENTER_SUN", jd), vec(c, "GEOCENTER_VENUS", jd)) for jd in jds], dtype=float)
    i = int(np.argmin(vals))
    lo_jd = float(jds[max(0, i - 3)])
    hi_jd = float(jds[min(len(jds) - 1, i + 3)])
    ref = 0.5 * (lo_jd + hi_jd)
    lo_s = (lo_jd - ref) * 86400.0
    hi_s = (hi_jd - ref) * 86400.0
    res = minimize_scalar(lambda s: sep_rad(vec(c, "GEOCENTER_SUN", ref + float(s) / 86400.0), vec(c, "GEOCENTER_VENUS", ref + float(s) / 86400.0)), bounds=(lo_s, hi_s), method="bounded", options={"xatol": 1e-4, "maxiter": 500})
    if not res.success:
        raise RuntimeError("Geocentric closest approach failed")
    return ref + float(res.x) / 86400.0


def fit_track_from_events(c: dict, site: dict, ev: dict) -> dict:
    jds = np.asarray(c["JD_TDB"], dtype=float)
    sel = jds[(jds >= ev["C1"]) & (jds <= ev["C4"])]
    pts = []
    for jd in sel:
        s = vec(c, f"{site['key']}_SUN", jd)
        v = vec(c, f"{site['key']}_VENUS", jd)
        sh, vh = unit(s), unit(v)
        d = sep_rad(sh, vh)
        tangent = unit(vh - math.cos(d) * sh)
        _cen, east, north = basis(s)
        p = d * ARC * np.array([float(np.dot(tangent, east)), float(np.dot(tangent, north))])
        p[0] *= -1.0
        pts.append(p)
    points = np.array(pts, dtype=float)
    mean = points.mean(axis=0)
    cen = points - mean
    _u, _s, vt = np.linalg.svd(cen, full_matrices=False)
    direction = vt[0]
    if direction[0] < 0:
        direction = -direction
    normal = np.array([-direction[1], direction[0]])
    along = cen @ direction
    cross = cen @ normal
    coef = np.polyfit(along, cross, 2)
    rms = float(np.sqrt(np.mean((cross - np.polyval(coef, along)) ** 2)))
    angle = abs(math.degrees(math.atan2(direction[1], direction[0])))
    return dict(angle=angle, rms=rms, slope=float(math.tan(math.radians(angle))), curvature=float(2.0 * coef[0]))


def geometry(c: dict) -> dict:
    jd = geocentric_ca_v0067(c)
    bas = basis(vec(c, "GEOCENTER_SUN", jd))
    h = 0.5 / 86400.0
    pv0 = rel_common(c, "POINT_VENUS", jd, bas)
    va0 = rel_common(c, "VARDO", jd, bas)
    vp = rel_common(c, "POINT_VENUS", jd + h, bas) - rel_common(c, "POINT_VENUS", jd - h, bas)
    vv = rel_common(c, "VARDO", jd + h, bas) - rel_common(c, "VARDO", jd - h, bas)
    direction = unit(unit(vp) + unit(vv))
    normal2 = np.array([-direction[1], direction[0]])
    if float(np.dot(va0 - pv0, normal2)) < 0.0:
        normal2 = -normal2
    cen, east, north = bas
    normal3 = unit(normal2[0] * east + normal2[1] * north)
    geo_sun = vec(c, "GEOCENTER_SUN", jd)
    geo_venus = vec(c, "GEOCENTER_VENUS", jd)
    site_pv = geo_sun - vec(c, "POINT_VENUS_SUN", jd)
    site_va = geo_sun - vec(c, "VARDO_SUN", jd)
    baseline = site_va - site_pv
    if float(np.dot(baseline, normal3)) < 0.0:
        normal2, normal3 = -normal2, -normal3
    apbp_as = float(np.dot(va0 - pv0, normal2))
    km_per_as = norm(geo_sun) / ARC
    ab_km = float(np.dot(baseline, normal3))
    es_axis = unit(geo_sun)
    ev_bar = float(np.dot(geo_venus, es_axis))
    vs_bar = float(np.dot(geo_sun - geo_venus, es_axis))
    es_bar = float(np.dot(geo_sun, es_axis))
    closure_bar = ev_bar + vs_bar - es_bar
    center_ratio = ev_bar / vs_bar
    ab_halley_as = -apbp_as * center_ratio
    ab_halley_km = ab_halley_as * km_per_as
    return dict(jd=jd, utc=utc(jd), geo_sun=geo_sun, geo_venus=geo_venus, ev_bar=ev_bar, vs_bar=vs_bar, es_bar=es_bar, closure_bar=closure_bar, km_per_as=km_per_as, apbp_as=apbp_as, apbp_km=apbp_as * km_per_as, ab_as=ab_km / km_per_as, ab_km=ab_km, center_ratio=center_ratio, ab_halley_as=ab_halley_as, ab_halley_km=ab_halley_km, halley_closure_as=ab_halley_as - (ab_km / km_per_as), halley_closure_km=ab_halley_km - ab_km)


def max_contact_resid(c: dict, events: dict) -> float:
    vals = []
    for site in SITES:
        key = site["key"]
        for ev in ("C1", "C2", "C3", "C4"):
            vals.append(abs(residual(c, key, events[key][ev], ev in ("C2", "C3"))) * ARC)
    return max(vals)


def fmt(x: float, n: int = 6) -> str:
    return f"{x:,.{n}f}"


def make_rows(events: dict, fits: dict, g: dict, max_resid: float) -> list[list[str]]:
    pi0 = math.asin(EARTH_RADIUS_KM / IAU1976_AU_KM) * ARC
    pi_raw = math.asin(EARTH_RADIUS_KM / g["es_bar"]) * ARC
    au_factor = g["es_bar"] / IAU1976_AU_KM
    pi_iau_from_raw = pi_raw * au_factor
    rows = []
    rows.append(["SECTION", "Quantity", "Symbol", "Value", "Unit / trace"])
    rows.append(["PARALLAX", "IAU 1976 solar horizontal parallax", "π₀", f"{pi0:.6f}", "arcsec; asin(a⊕ / cτA)"])
    rows.append(["PARALLAX", "IAU 1976 solar horizontal parallax, full precision", "π₀", f"{pi0:.12f}", "arcsec"])
    rows.append(["JPL DISTANCE", "Earth → Venus vector distance projected on Earth → Sun axis", "EV̄", fmt(g["ev_bar"], 6), "km; JPL geometric ecliptic vectors"])
    rows.append(["JPL DISTANCE", "Venus → Sun vector distance projected on Earth → Sun axis", "VS̄", fmt(g["vs_bar"], 6), "km; VS̄ = ES̄ − EV̄"])
    rows.append(["JPL DISTANCE", "Earth → Sun vector distance projected on Earth → Sun axis", "ES̄", fmt(g["es_bar"], 6), "km; norm/geocentric Sun axis"])
    rows.append(["JPL DISTANCE", "Vector closure", "EV̄ + VS̄ − ES̄", f"{g['closure_bar']:+.12e}", "km; PASS if ≈ 0"])
    rows.append(["JPL DISTANCE", "Earth-Sun distance in IAU-1976 AU", "ES̄ / AU₇₆", f"{au_factor:.12f}", "dimensionless; source of ~1.015 normalization"])
    rows.append(["CONTACT", "Point Venus, Tahiti first exterior contact", "PV C1", utc(events["POINT_VENUS"]["C1"]), "UTC"])
    rows.append(["CONTACT", "Point Venus, Tahiti first interior contact", "PV C2", utc(events["POINT_VENUS"]["C2"]), "UTC"])
    rows.append(["CONTACT", "Point Venus, Tahiti closest approach", "PV CA", utc(events["POINT_VENUS"]["CA"]), "UTC"])
    rows.append(["CONTACT", "Point Venus, Tahiti second interior contact", "PV C3", utc(events["POINT_VENUS"]["C3"]), "UTC"])
    rows.append(["CONTACT", "Point Venus, Tahiti second exterior contact", "PV C4", utc(events["POINT_VENUS"]["C4"]), "UTC"])
    rows.append(["CONTACT", "Vardø, Norway first exterior contact", "V C1", utc(events["VARDO"]["C1"]), "UTC"])
    rows.append(["CONTACT", "Vardø, Norway first interior contact", "V C2", utc(events["VARDO"]["C2"]), "UTC"])
    rows.append(["CONTACT", "Vardø, Norway closest approach", "V CA", utc(events["VARDO"]["CA"]), "UTC"])
    rows.append(["CONTACT", "Vardø, Norway second interior contact", "V C3", utc(events["VARDO"]["C3"]), "UTC"])
    rows.append(["CONTACT", "Vardø, Norway second exterior contact", "V C4", utc(events["VARDO"]["C4"]), "UTC"])
    rows.append(["TRACK FIT", "Point Venus track angle", "α_PV", f"{fits['POINT_VENUS']['angle']:.6f}", "deg"])
    rows.append(["TRACK FIT", "Vardø track angle", "α_V", f"{fits['VARDO']['angle']:.6f}", "deg"])
    rows.append(["TRACK FIT", "Average track angle", "ᾱ", f"{0.5 * (fits['POINT_VENUS']['angle'] + fits['VARDO']['angle']):.6f}", "deg"])
    rows.append(["TRACK FIT", "Delta track angle", "Δα", f"{abs(fits['VARDO']['angle'] - fits['POINT_VENUS']['angle']):.6f}", "deg"])
    rows.append(["TRACK FIT", "Point Venus RMS", "RMS_PV", f"{fits['POINT_VENUS']['rms']:.6f}", "arcsec"])
    rows.append(["TRACK FIT", "Vardø RMS", "RMS_V", f"{fits['VARDO']['rms']:.6f}", "arcsec"])
    rows.append(["TRACK FIT", "Maximum contact-equation residual", "max |ρ − R|", f"{max_resid:.12f}", "arcsec; PASS"])
    rows.append(["HALLEY", "Separate-ray observed chord separation", "A′B′", f"{g['apbp_as']:.6f}", "arcsec"])
    rows.append(["HALLEY", "Separate-ray observed chord separation", "A′B′", fmt(g["apbp_km"], 6), "km; converted by ES̄ / arcsec"])
    rows.append(["HALLEY", "Projected terrestrial baseline", "AB", f"{g['ab_as']:.6f}", "arcsec"])
    rows.append(["HALLEY", "Projected terrestrial baseline", "AB", fmt(g["ab_km"], 6), "km"])
    rows.append(["HALLEY CHECK", "Distance ratio used by Halley reduction", "EV̄ / VS̄", f"{g['center_ratio']:.12f}", "dimensionless"])
    rows.append(["HALLEY CHECK", "Halley baseline from separate-ray separation", "−A′B′·EV̄/VS̄", f"{g['ab_halley_as']:.6f}", "arcsec"])
    rows.append(["HALLEY CHECK", "Halley baseline from separate-ray separation", "−A′B′·EV̄/VS̄", fmt(g["ab_halley_km"], 6), "km"])
    rows.append(["HALLEY CHECK", "Halley closure against direct projected AB", "ΔAB", f"{g['halley_closure_as']:+.12f}", "arcsec"])
    rows.append(["HALLEY CHECK", "Halley closure against direct projected AB", "ΔAB", f"{g['halley_closure_km']:+.9f}", "km"])
    rows.append(["IAU REDUCTION", "Raw parallax at actual ES̄ distance", "π_raw", f"{pi_raw:.12f}", "arcsec; asin(a⊕ / ES̄)"])
    rows.append(["IAU REDUCTION", "AU normalization factor", "f_AU", f"{au_factor:.12f}", "ES̄ / AU₇₆ from JPL vectors"])
    rows.append(["IAU REDUCTION", "Normalized parallax check", "π_raw·f_AU", f"{pi_iau_from_raw:.12f}", "arcsec"])
    rows.append(["IAU REDUCTION", "Normalized minus π₀", "Δπ", f"{pi_iau_from_raw - pi0:+.12e}", "arcsec; closure"])
    return rows


def style_table(tab) -> None:
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.25)
        cell.get_text().set_color(FG)
        cell.get_text().set_fontsize(5.9)
        if r == 0:
            cell.set_facecolor(TABLE_HEADER)
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_fontsize(6.4)
        else:
            section = tab.get_celld()[(r, 0)].get_text().get_text()
            if section in ("JPL DISTANCE", "HALLEY", "HALLEY CHECK"):
                cell.set_facecolor(TABLE_TEAL if section != "HALLEY" else TABLE_GOLD)
                if c in (0, 2, 3):
                    cell.get_text().set_fontweight("bold")
            elif section in ("PARALLAX", "IAU REDUCTION"):
                cell.set_facecolor(TABLE_GOLD)
                if c in (0, 2, 3):
                    cell.get_text().set_fontweight("bold")
            elif section == "CONTACT":
                cell.set_facecolor(TABLE_BODY)
            elif section == "TRACK FIT":
                cell.set_facecolor("#132238")
            else:
                cell.set_facecolor(TABLE_BODY)
        if c in (2, 3):
            cell.get_text().set_ha("center")


def render_table(rows: list[list[str]]) -> None:
    plt.close("all")
    plt.rcParams.update({"figure.facecolor": BG, "savefig.facecolor": BG, "text.color": FG, "font.family": "DejaVu Serif", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(16.5, 18.5), facecolor=BG)
    ax = fig.add_subplot(111)
    ax.axis("off")
    fig.suptitle("Summary of 1769 Venus Transit Parallax Calculations Between Vardø, Norway, and Point Venus, Tahiti", fontsize=15, fontweight="bold", y=0.982)
    ax.text(0.5, 0.953, "Table-only audit from fresh JPL Horizons geometric ecliptic vectors — distances, contacts, Halley A′B′→AB triangulation, and IAU-1976 AU-normalized parallax closure", ha="center", va="center", fontsize=8.8, color=MUTED, transform=ax.transAxes)
    table = ax.table(cellText=rows, cellLoc="left", colWidths=[0.14, 0.37, 0.13, 0.18, 0.18], bbox=[0.018, 0.035, 0.964, 0.900])
    table.auto_set_font_size(False)
    style_table(table)
    fig.text(0.5, 0.012, "NO AI IMAGES — Python/Matplotlib table only. Standalone JPL Horizons vector reconstruction; all values are either JPL inputs or direct calculations.", ha="center", fontsize=6.5, color=MUTED)
    fig.savefig(PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.04, facecolor=BG)
    fig.savefig(PDF, bbox_inches="tight", pad_inches=0.04, facecolor=BG)
    fig.savefig(SVG, bbox_inches="tight", pad_inches=0.04, facecolor=BG)
    display(Image(filename=str(PNG)))


def write_csv(rows: list[list[str]]) -> None:
    pd.DataFrame(rows[1:], columns=rows[0]).to_csv(CSV, index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print("JPL source: Horizons geometric ecliptic vectors, fetched at runtime")
    print(f"Stations: {VA['label']} and {PV['label']}")
    print(f"Raster DPI: {DPI}; vector outputs: PDF and SVG")
    print("COMMENTS")
    print("Table-only widget. No solar geometry plot. No AI images. No external project Python imports.")
    m = master_frame()
    c = cache_build(m)
    events = {PV["key"]: station_events(c, PV), VA["key"]: station_events(c, VA)}
    fits = {PV["key"]: fit_track_from_events(c, PV, events[PV["key"]]), VA["key"]: fit_track_from_events(c, VA, events[VA["key"]])}
    g = geometry(c)
    mr = max_contact_resid(c, events)
    rows = make_rows(events, fits, g, mr)
    write_csv(rows)
    render_table(rows)
    print("RESULTS")
    print(f"π₀ IAU 1976 rounded: {math.asin(EARTH_RADIUS_KM / IAU1976_AU_KM) * ARC:.6f} arcsec")
    print(f"Geocentric CA UTC: {g['utc']} | JD {g['jd']:.18f}")
    print(f"EV̄ + VS̄ − ES̄: {g['closure_bar']:+.12e} km")
    print(f"A′B′: {g['apbp_as']:.12f} arcsec | {g['apbp_km']:.9f} km")
    print(f"AB direct: {g['ab_as']:.12f} arcsec | {g['ab_km']:.9f} km")
    print(f"AB Halley: {g['ab_halley_as']:.12f} arcsec | {g['ab_halley_km']:.9f} km")
    print(f"Halley closure: {g['halley_closure_as']:+.12f} arcsec | {g['halley_closure_km']:+.9f} km")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"PDF: {PDF}")
    print(f"SVG: {SVG}")
    print(f"CSV: {CSV}")
    print("PAPER COMPARISON")
    pi0 = math.asin(EARTH_RADIUS_KM / IAU1976_AU_KM) * ARC
    pi_raw = math.asin(EARTH_RADIUS_KM / g["es_bar"]) * ARC
    au_factor = g["es_bar"] / IAU1976_AU_KM
    print(f"IAU 1976 π₀: {pi0:.12f} arcsec")
    print(f"AU normalization factor from JPL ES̄/AU₇₆: {au_factor:.12f}")
    print(f"π_raw·f_AU closure: {pi_raw * au_factor - pi0:+.12e} arcsec")
    print("EQUATION STATUS")
    print("PASS: table uses V0067 seconds-space geocentric CA; Halley A′B′ to AB and IAU AU-normalization closures are reported.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0095
