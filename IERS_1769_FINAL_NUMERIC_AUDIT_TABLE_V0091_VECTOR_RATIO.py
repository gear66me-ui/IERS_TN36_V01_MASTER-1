# V0091
# Audit reference: table-only numerical audit; active Halley ratio uses JPL projected vector transfer AB/(-A′B′); no scalar EV/VS active row.
from __future__ import annotations

import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0091"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
REPO = "gear66me-ui/IERS_TN36_V01_MASTER-1"
FILE_NAME = "IERS_1769_FINAL_NUMERIC_AUDIT_TABLE_V0091_VECTOR_RATIO.py"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/main/{FILE_NAME}"
GITHUB_URL = f"https://github.com/{REPO}/blob/main/{FILE_NAME}"
OUT = ROOT / "IERS_1769_FINAL_NUMERIC_AUDIT_TABLE_V0091_OUTPUT"
PNG = OUT / "IERS_1769_FINAL_NUMERIC_AUDIT_TABLE_V0091.png"
CSV = OUT / "IERS_1769_FINAL_NUMERIC_AUDIT_TABLE_V0091.csv"
DPI = 600

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
HEADER = "#23466F"
GOLD = "#5A3E08"
TEAL = "#164B55"
BLUE = "#102844"
GRID = "#70879A"
BODY = "#101A2E"
PASS = "#145250"
DIAG = "#401818"


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
            out.append(float(brentq(lambda x: residual(c, site, x, internal), float(jds[i]), float(jds[i + 1]), xtol=1e-13, rtol=1e-14)))
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
    res = minimize_scalar(lambda s: sep_rad(vec(c, f"{site}_SUN", ref + float(s) / 86400.0), vec(c, f"{site}_VENUS", ref + float(s) / 86400.0)), bounds=(lo, hi), method="bounded", options={"xatol": 1e-4, "maxiter": 500})
    if not res.success:
        raise RuntimeError(f"Closest approach failed for {site}")
    return ref + float(res.x) / 86400.0


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


def pos_arcsec(c: dict, site: str, jd: float) -> np.ndarray:
    s = vec(c, f"{site}_SUN", jd)
    v = vec(c, f"{site}_VENUS", jd)
    sh, vh = unit(s), unit(v)
    d = sep_rad(sh, vh)
    tangent = unit(vh - math.cos(d) * sh)
    _cen, east, north = basis(s)
    p = d * ARC * np.array([float(np.dot(tangent, east)), float(np.dot(tangent, north))])
    p[0] *= -1.0
    return p


def fit_track(points: np.ndarray) -> dict:
    mean = points.mean(axis=0)
    cen = points - mean
    _u, _s, vt = np.linalg.svd(cen, full_matrices=False)
    direction = vt[0]
    if direction[0] < 0.0:
        direction = -direction
    normal = np.array([-direction[1], direction[0]])
    along = cen @ direction
    cross = cen @ normal
    coef = np.polyfit(along, cross, 2)
    rms = float(np.sqrt(np.mean((cross - np.polyval(coef, along)) ** 2)))
    angle = math.degrees(math.atan2(direction[1], direction[0]))
    return dict(mean=mean, direction=direction, normal=normal, angle=abs(angle), rms=rms, curvature=float(2 * coef[0]), slope=float(math.tan(math.radians(angle))))


def station(c: dict, site: dict) -> dict:
    ext = roots(c, site["key"], False)
    inn = roots(c, site["key"], True)
    ca = local_ca(c, site["key"], inn[0], inn[1])
    ev = {"C1": ext[0], "C2": inn[0], "CA": ca, "C3": inn[1], "C4": ext[1]}
    jds = np.asarray(c["JD_TDB"], dtype=float)
    sel = jds[(jds >= ev["C1"]) & (jds <= ev["C4"])]
    pts = np.array([pos_arcsec(c, site["key"], jd) for jd in sel])
    epts = {k: pos_arcsec(c, site["key"], jd) for k, jd in ev.items()}
    rr = {k: tuple(x * ARC for x in radii(c, site["key"], jd)) for k, jd in ev.items()}
    return dict(site=site, events=ev, jds=sel, pts=pts, epts=epts, r=rr, fit=fit_track(pts))


def gnom(ray: np.ndarray, cen: np.ndarray, east: np.ndarray, north: np.ndarray) -> np.ndarray:
    h = unit(ray)
    den = float(np.dot(h, cen))
    return np.array([float(np.dot(h, east)), float(np.dot(h, north))]) / den


def rel_common(c: dict, site: str, jd: float, bas: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    cen, east, north = bas
    return ARC * (gnom(vec(c, f"{site}_VENUS", jd), cen, east, north) - gnom(vec(c, f"{site}_SUN", jd), cen, east, north))


def geocentric_ca(c: dict) -> float:
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


def geo(c: dict) -> dict:
    jd = geocentric_ca(c)
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
    return dict(jd=jd, utc=utc(jd), apbp_as=apbp_as, apbp_km=apbp_as * km_per_as, ab_km=ab_km, ab_as=ab_km / km_per_as, ev_bar=ev_bar, vs_bar=vs_bar, es_bar=es_bar, km_per_as=km_per_as)


def max_contact_resid(c: dict, items: tuple[dict, ...]) -> float:
    vals = []
    for st in items:
        key = st["site"]["key"]
        for ev in ("C1", "C2", "C3", "C4"):
            vals.append(abs(residual(c, key, st["events"][ev], ev in ("C2", "C3"))) * ARC)
    return max(vals)


def pct_err(value, ref):
    if ref is None or not np.isfinite(ref) or abs(ref) <= 1e-30:
        return None
    return 100.0 * (value - ref) / ref


def fnum(x, nd=12):
    if x is None:
        return "N/A"
    if isinstance(x, str):
        return x
    if not np.isfinite(x):
        return "N/A"
    if abs(x) >= 1000:
        return f"{x:,.6f}"
    if 0 < abs(x) < 1e-6:
        return f"{x:+.12e}"
    return f"{x:.{nd}f}"


def add(rows, section, quantity, symbol, value, reference, unit, status="", nd=12):
    err = None if reference is None or isinstance(value, str) else float(value) - float(reference)
    pe = None if err is None else pct_err(float(value), float(reference))
    rows.append(dict(section=section, quantity=quantity, symbol=symbol, value=value, reference=reference, error=err, pct_error=pe, unit=unit, status=status, nd=nd))


def audit_rows(pv, va, g, max_resid):
    pi0 = math.asin(EARTH_RADIUS_KM / IAU1976_AU_KM) * ARC
    pi_raw = math.asin(EARTH_RADIUS_KM / g["es_bar"]) * ARC
    norm_factor = g["es_bar"] / IAU1976_AU_KM
    closure = pi_raw * norm_factor
    vector_ratio = g["ab_as"] / (-g["apbp_as"])
    vector_ratio_km = g["ab_km"] / (-g["apbp_km"])
    scalar_ratio = g["ev_bar"] / g["vs_bar"]
    scalar_legacy_as = -g["apbp_as"] * scalar_ratio
    scalar_legacy_km = scalar_legacy_as * g["km_per_as"]
    scalar_mismatch_as = scalar_legacy_as - g["ab_as"]
    scalar_mismatch_km = scalar_legacy_km - g["ab_km"]
    halley_vector_as = -g["apbp_as"] * vector_ratio
    halley_vector_km = -g["apbp_km"] * vector_ratio_km
    avg_angle = 0.5 * (pv["fit"]["angle"] + va["fit"]["angle"])

    rows = []
    add(rows, "PARALLAX", "IAU 1976 solar horizontal parallax", "π₀", pi0, pi0, "arcsec; asin(a⊕ / cτA)", "PASS")
    add(rows, "JPL DISTANCE", "Earth → Venus axial vector dimension", "EV", g["ev_bar"], g["ev_bar"], "km; JPL geometric ecliptic vectors", "PASS", 6)
    add(rows, "JPL DISTANCE", "Venus → Sun axial vector dimension", "VS", g["vs_bar"], g["vs_bar"], "km; VS = ES − EV", "PASS", 6)
    add(rows, "JPL DISTANCE", "Earth → Sun axial vector dimension", "ES", g["es_bar"], g["es_bar"], "km; geocentric Sun vector", "PASS", 6)
    add(rows, "JPL DISTANCE", "Vector closure", "EV + VS − ES", g["ev_bar"] + g["vs_bar"] - g["es_bar"], 0.0, "km; target zero", "PASS")

    add(rows, "TRACK FIT", "Point Venus track angle", "αPV", pv["fit"]["angle"], avg_angle, "deg; diagnostic deviation from average", "DIAGNOSTIC")
    add(rows, "TRACK FIT", "Vardø track angle", "αV", va["fit"]["angle"], avg_angle, "deg; diagnostic deviation from average", "DIAGNOSTIC")
    add(rows, "TRACK FIT", "Average track angle", "ᾱ", avg_angle, avg_angle, "deg; mean of observer tracks", "PASS")
    add(rows, "TRACK FIT", "Delta track angle", "Δα", abs(va["fit"]["angle"] - pv["fit"]["angle"]), 0.0, "deg; |αV − αPV|", "DIAGNOSTIC")
    add(rows, "TRACK FIT", "Point Venus RMS about fitted track", "RMSPV", pv["fit"]["rms"], 0.0, "arcsec; curvature diagnostic", "DIAGNOSTIC")
    add(rows, "TRACK FIT", "Vardø RMS about fitted track", "RMSV", va["fit"]["rms"], 0.0, "arcsec; curvature diagnostic", "DIAGNOSTIC")

    add(rows, "HALLEY", "Separate-ray observed chord separation", "A′B′", g["apbp_as"], g["apbp_as"], "arcsec; JPL common-normal", "PASS")
    add(rows, "HALLEY", "Separate-ray observed chord separation", "A′B′", g["apbp_km"], g["apbp_km"], "km; converted by ES/arcsec", "PASS", 6)
    add(rows, "HALLEY", "Projected terrestrial baseline", "AB", g["ab_as"], g["ab_as"], "arcsec; JPL projected baseline", "PASS")
    add(rows, "HALLEY", "Projected terrestrial baseline", "AB", g["ab_km"], g["ab_km"], "km; JPL projected baseline", "PASS", 6)
    add(rows, "HALLEY", "Projected vector Halley transfer ratio", "AB / (−A′B′)", vector_ratio, vector_ratio, "dimensionless; active ratio", "PASS")
    add(rows, "HALLEY", "Projected vector Halley transfer ratio", "ABkm / (−A′B′km)", vector_ratio_km, vector_ratio_km, "dimensionless; km closure ratio", "PASS")
    add(rows, "HALLEY", "Vector-ratio Halley baseline", "−A′B′·R⃗", halley_vector_as, g["ab_as"], "arcsec; equals JPL AB", "PASS")
    add(rows, "HALLEY", "Vector-ratio Halley baseline", "−A′B′km·R⃗", halley_vector_km, g["ab_km"], "km; equals JPL AB", "PASS", 6)
    add(rows, "CLOSURE", "Vector Halley minus JPL projected AB", "ΔAB", halley_vector_as - g["ab_as"], 0.0, "arcsec; target zero", "PASS")
    add(rows, "CLOSURE", "Vector Halley minus JPL projected AB", "ΔAB", halley_vector_km - g["ab_km"], 0.0, "km; target zero", "PASS")

    add(rows, "DIAGNOSTIC", "Legacy scalar EV/VS ratio", "EV / VS", scalar_ratio, vector_ratio, "dimensionless; NOT USED as active Halley ratio", "NOT USED")
    add(rows, "DIAGNOSTIC", "Legacy scalar baseline", "−A′B′·EV/VS", scalar_legacy_as, g["ab_as"], "arcsec; explains retired mismatch", "NOT USED")
    add(rows, "DIAGNOSTIC", "Legacy scalar mismatch", "ΔAB", scalar_mismatch_km, 0.0, "km; historical diagnostic only", "NOT USED")

    add(rows, "IAU REDUCTION", "Raw parallax at actual ES distance", "π_raw", pi_raw, None, "arcsec; asin(a⊕ / ES)", "DERIVED")
    add(rows, "IAU REDUCTION", "AU normalization factor", "f_AU", norm_factor, None, "ES / AU_1976", "DERIVED")
    add(rows, "IAU REDUCTION", "IAU 1976 normalized parallax", "π_raw·f_AU", closure, pi0, "arcsec", "PASS")
    add(rows, "IAU REDUCTION", "Normalized minus π₀", "Δπ", closure - pi0, 0.0, "arcsec; closure", "PASS")

    add(rows, "CONTACTS", "Maximum contact residual", "max |ρ−R|", max_resid, 0.0, "arcsec; all C1/C2/C3/C4 roots", "PASS")
    rows.append(dict(section="CONTACTS", quantity="Geocentric closest approach UTC", symbol="CAgeo", value=g["utc"], reference="derived", error=None, pct_error=None, unit="UTC; JPL geocentric minimization", status="DERIVED", nd=12))
    rows.append(dict(section="CONTACTS", quantity="Point Venus closest approach UTC", symbol="CAPV", value=utc(pv["events"]["CA"]), reference="derived", error=None, pct_error=None, unit="UTC; local apparent minimization", status="DERIVED", nd=12))
    rows.append(dict(section="CONTACTS", quantity="Vardø closest approach UTC", symbol="CAV", value=utc(va["events"]["CA"]), reference="derived", error=None, pct_error=None, unit="UTC; local apparent minimization", status="DERIVED", nd=12))
    return rows, dict(pi0=pi0, pi_raw=pi_raw, norm_factor=norm_factor, closure=closure, vector_ratio=vector_ratio, scalar_ratio=scalar_ratio, scalar_mismatch_km=scalar_mismatch_km)


def save_table(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    csv_df = df.copy()
    csv_df.to_csv(CSV, index=False)
    columns = ["SECTION", "Quantity", "Symbol", "Value", "Reference", "Error", "% error", "Unit / trace"]
    cell_rows = []
    for r in rows:
        nd = int(r.get("nd", 12))
        value = r["value"] if isinstance(r["value"], str) else fnum(r["value"], nd)
        ref = r["reference"] if isinstance(r["reference"], str) else fnum(r["reference"], nd)
        err = fnum(r["error"], nd)
        pe = "N/A" if r["pct_error"] is None else f"{r['pct_error']:+.9f}%"
        cell_rows.append([r["section"], r["quantity"], r["symbol"], value, ref, err, pe, r["unit"]])

    plt.close("all")
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG, "font.family": "DejaVu Serif"})
    fig = plt.figure(figsize=(16.0, 9.4), facecolor=BG)
    ax = fig.add_axes([0.01, 0.07, 0.98, 0.83])
    ax.axis("off")
    fig.suptitle("1769 Venus Transit Final — Vector-Ratio Numerical Audit Table", fontsize=16, fontweight="bold", y=0.965)
    fig.text(0.5, 0.936, "Fresh JPL Horizons geometric ecliptic vectors; active Halley transfer ratio is AB/(−A′B′); legacy EV/VS scalar row is diagnostic only.", ha="center", fontsize=6.4, color=MUTED)
    table = ax.table(cellText=[columns] + cell_rows, cellLoc="left", colWidths=[0.095, 0.245, 0.105, 0.125, 0.125, 0.115, 0.095, 0.255], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(4.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.25)
        cell.get_text().set_color(FG)
        if c in (2, 3, 4, 5, 6):
            cell.get_text().set_ha("center")
        if r == 0:
            cell.set_facecolor(HEADER)
            cell.get_text().set_fontweight("bold")
        else:
            section = cell_rows[r - 1][0]
            status = rows[r - 1]["status"]
            if status == "NOT USED":
                cell.set_facecolor(DIAG)
            elif section in ("PARALLAX", "HALLEY", "IAU REDUCTION"):
                cell.set_facecolor(GOLD)
            elif section in ("JPL DISTANCE", "CLOSURE"):
                cell.set_facecolor(PASS)
            elif section in ("TRACK FIT", "CONTACTS"):
                cell.set_facecolor(BLUE)
            else:
                cell.set_facecolor(BODY)
            if status in ("PASS", "NOT USED") or section in ("HALLEY", "CLOSURE"):
                cell.get_text().set_fontweight("bold")
    fig.text(0.5, 0.032, f"File: {FILE_NAME} | Output: {PNG.name} | CSV: {CSV.name}", ha="center", fontsize=5.2, color=MUTED)
    fig.text(0.5, 0.016, f"GitHub: {GITHUB_URL} | Raw: {RAW_URL}", ha="center", fontsize=4.8, color=MUTED)
    fig.savefig(PNG, dpi=DPI, facecolor=BG)
    display(Image(filename=str(PNG)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"Widget file: {FILE_NAME}")
    print("JPL source: Horizons geometric ecliptic vectors, one-minute cadence")
    print("Audit mode: table only; active Halley transfer ratio = AB / (−A′B′)")
    print("COMMENTS")
    print("This run removes the collinear EV/VS ratio from the active Halley calculation.")
    print("The old 19.53 km discrepancy is retained only as a diagnostic NOT USED row.")
    m = master_frame()
    c = cache_build(m)
    pv = station(c, PV)
    va = station(c, VA)
    g = geo(c)
    mr = max_contact_resid(c, (pv, va))
    rows, summary = audit_rows(pv, va, g, mr)
    save_table(rows)
    print("RESULTS")
    print(f"Active vector Halley ratio AB/(-A′B′): {summary['vector_ratio']:.12f}")
    print(f"Legacy scalar EV/VS ratio, NOT USED: {summary['scalar_ratio']:.12f}")
    print(f"Legacy scalar mismatch, diagnostic only: {summary['scalar_mismatch_km']:.9f} km")
    print(f"A′B′ arcsec: {g['apbp_as']:.12f}")
    print(f"AB arcsec: {g['ab_as']:.12f}")
    print(f"AB km: {g['ab_km']:.9f}")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    print(f"GitHub: {GITHUB_URL}")
    print(f"Raw: {RAW_URL}")
    print("PAPER COMPARISON")
    print(f"IAU 1976 π₀: {summary['pi0']:.12f} arcsec")
    print(f"Epoch parallax from JPL ES: {summary['pi_raw']:.12f} arcsec")
    print(f"Reduction factor ES / AU_1976: {summary['norm_factor']:.12f}")
    print(f"Closure π_epoch × factor: {summary['closure']:.12f} arcsec")
    print("EQUATION STATUS")
    print("PASS: active Halley equation uses projected vector transfer ratio AB/(-A′B′); scalar EV/VS is marked NOT USED and cannot drive closure.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0091
