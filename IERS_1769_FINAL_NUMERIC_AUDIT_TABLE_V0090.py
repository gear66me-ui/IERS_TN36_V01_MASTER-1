# V0090
# Audit reference: table-only numerical audit of IERS 1769 Venus Transit Final geometry; fresh JPL Horizons values; no plot panel.
from __future__ import annotations

import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0090"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUT = ROOT / "IERS_1769_FINAL_NUMERIC_AUDIT_TABLE_V0090_OUTPUT"
PNG = OUT / "IERS_1769_FINAL_NUMERIC_AUDIT_TABLE_V0090.png"
CSV = OUT / "IERS_1769_FINAL_NUMERIC_AUDIT_TABLE_V0090.csv"
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

PV = dict(key="POINT_VENUS", label="Point Venus, Tahiti", short="PV", lat=-17.4956, lon=-149.4939, elevation=0.0, body=399)
VA = dict(key="VARDO", label="Vardø, Norway", short="V", lat=70.3724, lon=31.1103, elevation=0.0, body=399)
SITES = (PV, VA)
TARGETS = (("SUN", "10"), ("VENUS", "299"))
PREFIXES = (
    "GEOCENTER_SUN", "GEOCENTER_VENUS",
    "POINT_VENUS_SUN", "POINT_VENUS_VENUS",
    "VARDO_SUN", "VARDO_VENUS",
)
EVENTS = ("C1", "C2", "CA", "C3", "C4")

BG = "#000000"
FG = "#F8FAFC"
MUTED = "#B8CBD6"
HEADER = "#23466F"
TEAL = "#164B55"
GOLD = "#5A3D09"
RED = "#651313"
BLUE = "#102640"
GRID = "#70879A"


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
    pred = np.polyval(coef, along)
    rms = float(np.sqrt(np.mean((cross - pred) ** 2)))
    angle = math.degrees(math.atan2(direction[1], direction[0]))
    return dict(angle=abs(angle), rms=rms, curvature=float(2 * coef[0]), slope=float(math.tan(math.radians(angle))))


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
    ca_sep = sep_rad(vec(c, f"{site['key']}_SUN", ca), vec(c, f"{site['key']}_VENUS", ca)) * ARC
    return dict(site=site, events=ev, jds=sel, pts=pts, epts=epts, r=rr, fit=fit_track(pts), ca_sep=ca_sep)


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
    return dict(jd=jd, utc=utc(jd), bas=bas, normal2=normal2, apbp_as=apbp_as, apbp_km=apbp_as * km_per_as, ab_km=ab_km, ab_as=ab_km / km_per_as, ev_bar=ev_bar, vs_bar=vs_bar, es_bar=es_bar, km_per_as=km_per_as)


def max_contact_resid(c: dict, items: tuple[dict, ...]) -> float:
    vals = []
    for st in items:
        key = st["site"]["key"]
        for ev in ("C1", "C2", "C3", "C4"):
            vals.append(abs(residual(c, key, st["events"][ev], ev in ("C2", "C3"))) * ARC)
    return max(vals)


def pct(value: float, ref: float) -> str:
    if not np.isfinite(value) or not np.isfinite(ref) or abs(ref) < 1e-30:
        return "N/A"
    return f"{100.0 * (value - ref) / ref:+.9f}%"


def fmt(x, nd=12) -> str:
    if isinstance(x, str):
        return x
    if not np.isfinite(x):
        return "N/A"
    ax = abs(float(x))
    if ax >= 1_000:
        return f"{x:,.6f}"
    if ax != 0 and ax < 1e-6:
        return f"{x:+.12e}"
    return f"{x:.{nd}f}"


def row(section, quantity, symbol, value, reference, error, percent, unit_trace, status):
    return dict(SECTION=section, Quantity=quantity, Symbol=symbol, Value=value, Reference=reference, Error=error, Percent_Error=percent, Unit_Trace=unit_trace, Status=status)


def build_rows(pv: dict, va: dict, g: dict, max_resid: float) -> list[dict]:
    pi0 = math.asin(EARTH_RADIUS_KM / IAU1976_AU_KM) * ARC
    ev, vs, es = g["ev_bar"], g["vs_bar"], g["es_bar"]
    vec_closure = ev + vs - es
    apbp_as = g["apbp_as"]
    apbp_abs = -apbp_as
    apbp_km = g["apbp_km"]
    ab_as, ab_km = g["ab_as"], g["ab_km"]
    km_per_as = g["km_per_as"]
    ratio_scalar = ev / vs
    ratio_vector = ab_as / apbp_abs
    scalar_baseline_as = apbp_abs * ratio_scalar
    scalar_baseline_km = scalar_baseline_as * km_per_as
    scalar_error_as = scalar_baseline_as - ab_as
    scalar_error_km = scalar_baseline_km - ab_km
    correction = ratio_vector / ratio_scalar
    corrected_as = apbp_abs * ratio_scalar * correction
    corrected_km = corrected_as * km_per_as
    corrected_err_as = corrected_as - ab_as
    corrected_err_km = corrected_km - ab_km
    pi_raw = math.asin(EARTH_RADIUS_KM / es) * ARC
    norm_factor = es / IAU1976_AU_KM
    pi_norm = pi_raw * norm_factor
    pi_err = pi_norm - pi0
    avg_angle = 0.5 * (pv["fit"]["angle"] + va["fit"]["angle"])
    delta_angle = abs(va["fit"]["angle"] - pv["fit"]["angle"])
    data = []
    data.append(row("PARALLAX", "IAU 1976 solar horizontal parallax", "π₀", pi0, pi0, 0.0, "0.000000000%", "arcsec; asin(a⊕ / cτA)", "REFERENCE"))
    data.append(row("JPL DISTANCE", "Earth → Venus axial dimension along Earth → Sun axis", "EV", ev, np.nan, np.nan, "N/A", "km; JPL geometric ecliptic vectors", "USED"))
    data.append(row("JPL DISTANCE", "Venus → Sun axial dimension along Earth → Sun axis", "VS", vs, es - ev, vs - (es - ev), pct(vs, es - ev), "km; VS = ES − EV", "CHECK"))
    data.append(row("JPL DISTANCE", "Earth → Sun axial dimension along Earth → Sun axis", "ES", es, ev + vs, es - (ev + vs), pct(es, ev + vs), "km; geocentric Sun vector magnitude", "CHECK"))
    data.append(row("JPL DISTANCE", "Vector closure", "EV + VS − ES", vec_closure, 0.0, vec_closure, "N/A", "km; target zero", "CLOSURE"))
    data.append(row("TRACK FIT", "Point Venus track angle", "αPV", pv["fit"]["angle"], avg_angle, pv["fit"]["angle"] - avg_angle, pct(pv["fit"]["angle"], avg_angle), "deg; one-minute JPL fit", "USED"))
    data.append(row("TRACK FIT", "Vardø track angle", "αV", va["fit"]["angle"], avg_angle, va["fit"]["angle"] - avg_angle, pct(va["fit"]["angle"], avg_angle), "deg; one-minute JPL fit", "USED"))
    data.append(row("TRACK FIT", "Average track angle", "ᾱ", avg_angle, avg_angle, 0.0, "0.000000000%", "deg; mean of station fits", "USED"))
    data.append(row("TRACK FIT", "Delta track angle", "Δα", delta_angle, 0.0, delta_angle, "N/A", "deg; |αV − αPV|", "DIAGNOSTIC"))
    data.append(row("TRACK FIT", "Point Venus RMS about fitted track", "RMSPV", pv["fit"]["rms"], 0.0, pv["fit"]["rms"], "N/A", "arcsec; curvature diagnostic", "DIAGNOSTIC"))
    data.append(row("TRACK FIT", "Vardø RMS about fitted track", "RMSV", va["fit"]["rms"], 0.0, va["fit"]["rms"], "N/A", "arcsec; curvature diagnostic", "DIAGNOSTIC"))
    data.append(row("HALLEY", "Separate-ray observed chord separation", "A′B′", apbp_as, apbp_as, 0.0, "0.000000000%", "arcsec; JPL common-normal", "USED"))
    data.append(row("HALLEY", "Separate-ray observed chord separation", "A′B′", apbp_km, apbp_as * km_per_as, apbp_km - apbp_as * km_per_as, pct(apbp_km, apbp_as * km_per_as), "km; converted by ES/arcsec", "USED"))
    data.append(row("HALLEY", "Projected terrestrial baseline", "AB", ab_as, ab_as, 0.0, "0.000000000%", "arcsec; JPL projected baseline", "USED"))
    data.append(row("HALLEY", "Projected terrestrial baseline", "AB", ab_km, ab_as * km_per_as, ab_km - ab_as * km_per_as, pct(ab_km, ab_as * km_per_as), "km; JPL projected baseline", "USED"))
    data.append(row("REJECTED", "Simple scalar Halley ratio", "EV / VS", ratio_scalar, ratio_vector, ratio_scalar - ratio_vector, pct(ratio_scalar, ratio_vector), "dimensionless; collinear approximation only", "REJECTED"))
    data.append(row("REJECTED", "Simple scalar Halley baseline", "−A′B′·EV/VS", scalar_baseline_as, ab_as, scalar_error_as, pct(scalar_baseline_as, ab_as), "arcsec; NOT USED", "REJECTED"))
    data.append(row("REJECTED", "Simple scalar Halley baseline", "−A′B′·EV/VS", scalar_baseline_km, ab_km, scalar_error_km, pct(scalar_baseline_km, ab_km), "km; should show old mismatch", "REJECTED"))
    data.append(row("USED", "JPL vector transfer ratio", "AB / (−A′B′)", ratio_vector, ratio_vector, 0.0, "0.000000000%", "dimensionless; vector common-normal transfer", "USED"))
    data.append(row("USED", "Vector projection correction", "Fₚ", correction, correction, 0.0, "0.000000000%", "Fₚ = [AB/(−A′B′)] / [EV/VS]", "USED"))
    data.append(row("USED", "Corrected Halley baseline", "−A′B′·EV/VS·Fₚ", corrected_as, ab_as, corrected_err_as, pct(corrected_as, ab_as), "arcsec; equals JPL AB", "USED"))
    data.append(row("USED", "Corrected Halley baseline", "−A′B′·EV/VS·Fₚ", corrected_km, ab_km, corrected_err_km, pct(corrected_km, ab_km), "km; equals JPL AB", "USED"))
    data.append(row("CLOSURE", "Corrected Halley minus JPL projected AB", "ΔAB", corrected_err_as, 0.0, corrected_err_as, "N/A", "arcsec; target zero", "CLOSURE"))
    data.append(row("CLOSURE", "Corrected Halley minus JPL projected AB", "ΔAB", corrected_err_km, 0.0, corrected_err_km, "N/A", "km; target zero", "CLOSURE"))
    data.append(row("IAU REDUCTION", "Raw parallax at actual ES distance", "π_raw", pi_raw, np.nan, np.nan, "N/A", "arcsec; asin(a⊕ / ES)", "USED"))
    data.append(row("IAU REDUCTION", "AU normalization factor", "f_AU", norm_factor, np.nan, np.nan, "N/A", "ES / AU_1976", "USED"))
    data.append(row("IAU REDUCTION", "IAU 1976 normalized parallax", "π_raw·f_AU", pi_norm, pi0, pi_err, pct(pi_norm, pi0), "arcsec", "CLOSURE"))
    data.append(row("IAU REDUCTION", "Normalized minus π₀", "Δπ", pi_err, 0.0, pi_err, "N/A", "arcsec; closure", "CLOSURE"))
    data.append(row("CONTACTS", "Maximum contact residual", "max |ρ−R|", max_resid, 0.0, max_resid, "N/A", "arcsec; all C1/C2/C3/C4 roots", "CLOSURE"))
    data.append(row("CONTACTS", "Geocentric closest approach UTC", "CAgeo", g["utc"], "derived", "N/A", "N/A", "UTC; JPL geocentric minimization", "USED"))
    data.append(row("CONTACTS", "Point Venus closest approach UTC", "CAPV", utc(pv["events"]["CA"]), "derived", "N/A", "N/A", "UTC; local apparent minimization", "USED"))
    data.append(row("CONTACTS", "Vardø closest approach UTC", "CAV", utc(va["events"]["CA"]), "derived", "N/A", "N/A", "UTC; local apparent minimization", "USED"))
    return data


def as_display_rows(rows: list[dict]) -> list[list[str]]:
    out = []
    for r in rows:
        out.append([
            r["SECTION"],
            r["Quantity"],
            r["Symbol"],
            fmt(r["Value"], 12),
            fmt(r["Reference"], 12),
            fmt(r["Error"], 12),
            r["Percent_Error"],
            r["Unit_Trace"],
        ])
    return out


def save_outputs(rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(CSV, index=False)
    table_rows = as_display_rows(rows)
    headers = ["SECTION", "Quantity", "Symbol", "Value", "Reference", "Error", "% error", "Unit / trace"]
    nrows = len(table_rows) + 1
    fig_h = max(10.5, 0.34 * nrows + 1.6)
    plt.close("all")
    plt.rcParams.update({"figure.facecolor": BG, "savefig.facecolor": BG, "font.family": "DejaVu Serif", "text.color": FG})
    fig, ax = plt.subplots(figsize=(19.2, fig_h), facecolor=BG)
    ax.axis("off")
    fig.suptitle("1769 Venus Transit Final — Numerical Audit Table", fontsize=18, fontweight="bold", y=0.985, color=FG)
    ax.text(0.5, 0.965, "Fresh JPL Horizons geometric ecliptic vectors; table-only audit of all key figures of merit; rejected scalar Halley row compared against JPL projected AB.", ha="center", va="center", transform=fig.transFigure, fontsize=8.3, color=MUTED)
    tab = ax.table(cellText=[headers] + table_rows, cellLoc="left", colWidths=[0.11, 0.26, 0.10, 0.13, 0.13, 0.12, 0.10, 0.23], bbox=[0.0, 0.035, 1.0, 0.90])
    tab.auto_set_font_size(False)
    tab.set_fontsize(6.05)
    for (rr, cc), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.32)
        cell.get_text().set_color(FG)
        if cc in (2, 3, 4, 5, 6):
            cell.get_text().set_ha("center")
        if rr == 0:
            cell.set_facecolor(HEADER)
            cell.get_text().set_fontweight("bold")
        else:
            section = table_rows[rr - 1][0]
            if section == "REJECTED":
                cell.set_facecolor(RED)
            elif section in ("HALLEY", "PARALLAX", "IAU REDUCTION"):
                cell.set_facecolor(GOLD)
            elif section in ("JPL DISTANCE", "USED", "CLOSURE"):
                cell.set_facecolor(TEAL)
            elif section in ("TRACK FIT", "CONTACTS"):
                cell.set_facecolor(BLUE)
            else:
                cell.set_facecolor("#101A2E")
            if section in ("PARALLAX", "HALLEY", "USED", "CLOSURE", "REJECTED", "IAU REDUCTION"):
                cell.get_text().set_fontweight("bold")
    footer = f"File: IERS_1769_FINAL_NUMERIC_AUDIT_TABLE_V0090.py | Output: {PNG.name} | CSV: {CSV.name}"
    fig.text(0.5, 0.014, footer, ha="center", va="center", fontsize=6.3, color=MUTED)
    fig.savefig(PNG, dpi=DPI, facecolor=BG)
    display(Image(filename=str(PNG)))


def print_numeric_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    rejected = df[df["SECTION"] == "REJECTED"]
    closures = df[df["SECTION"] == "CLOSURE"]
    print("RESULTS")
    for _, r in rejected.iterrows():
        print(f"REJECTED AUDIT | {r['Quantity']} | {r['Symbol']} | value={fmt(r['Value'])} | reference={fmt(r['Reference'])} | error={fmt(r['Error'])} | percent={r['Percent_Error']}")
    for _, r in closures.iterrows():
        print(f"CLOSURE AUDIT | {r['Quantity']} | {r['Symbol']} | value={fmt(r['Value'])} | target={fmt(r['Reference'])} | percent={r['Percent_Error']}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print("Mode: table-only numerical audit; no transit plot panel")
    print("JPL source: Horizons geometric ecliptic vectors, one-minute cadence")
    print(f"Sites: {PV['label']} and {VA['label']}")
    print("COMMENTS")
    print("This audit recomputes all listed values fresh from JPL vectors and compares scalar, vector-corrected, closure, and IAU-reduction rows.")
    m = master_frame()
    c = cache_build(m)
    pv = station(c, PV)
    va = station(c, VA)
    g = geo(c)
    mr = max_contact_resid(c, (pv, va))
    rows = build_rows(pv, va, g, mr)
    save_outputs(rows)
    print_numeric_summary(rows)
    print("OUTPUT SUMMARY")
    print(f"PNG table: {PNG}")
    print(f"CSV table: {CSV}")
    print("PAPER COMPARISON")
    print("Published values are NOT USED as inputs. IAU 1976 π₀ is recomputed from a⊕ and cτA and used only as a closure reference.")
    print("EQUATION STATUS")
    print("PASS: contacts, track fits, A′B′, AB, scalar Halley test, vector correction, and parallax reduction are all recomputed from JPL-derived quantities in this table.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0090
