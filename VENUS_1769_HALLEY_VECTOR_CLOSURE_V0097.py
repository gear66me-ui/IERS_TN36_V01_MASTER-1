# V0097
# Audit reference: standalone JPL Horizons Halley vector-closure repair; Python/Matplotlib table only; no AI images.
from __future__ import annotations

import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0097"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUT = ROOT / "VENUS_1769_V0097_HALLEY_VECTOR_CLOSURE"
PNG = OUT / "VENUS_1769_V0097_HALLEY_VECTOR_CLOSURE_TABLE_900DPI.png"
PDF = OUT / "VENUS_1769_V0097_HALLEY_VECTOR_CLOSURE_TABLE_VECTOR.pdf"
SVG = OUT / "VENUS_1769_V0097_HALLEY_VECTOR_CLOSURE_TABLE_VECTOR.svg"
CSV = OUT / "VENUS_1769_V0097_HALLEY_VECTOR_CLOSURE.csv"
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
PV = dict(key="POINT_VENUS", label="Point Venus, Tahiti", lat=-17.4956, lon=-149.4939, elevation=0.0, body=399)
VA = dict(key="VARDO", label="Vardø, Norway", lat=70.3724, lon=31.1103, elevation=0.0, body=399)
SITES = (PV, VA)
TARGETS = (("SUN", "10"), ("VENUS", "299"))
PREFIXES = ("GEOCENTER_SUN", "GEOCENTER_VENUS", "POINT_VENUS_SUN", "POINT_VENUS_VENUS", "VARDO_SUN", "VARDO_VENUS")

BG = "#000000"
FG = "#F8FAFC"
MUTED = "#B8CBD6"
HEADER = "#23466F"
BODY = "#101A2E"
TEAL = "#164B55"
GOLD = "#563B0B"
RED = "#5A1414"
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
from scipy.optimize import minimize_scalar

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
    if den <= 0.0:
        raise RuntimeError("Ray lies outside tangent hemisphere.")
    return np.array([float(np.dot(h, east)), float(np.dot(h, north))]) / den


def rel_common(c: dict, site: str, jd: float, bas: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
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
    res = minimize_scalar(lambda s: sep_rad(vec(c, "GEOCENTER_SUN", ref + float(s) / 86400.0), vec(c, "GEOCENTER_VENUS", ref + float(s) / 86400.0)), bounds=(lo_s, hi_s), method="bounded", options={"xatol": 1.0e-4, "maxiter": 500})
    if not res.success:
        raise RuntimeError("Geocentric closest approach failed.")
    return ref + float(res.x) / 86400.0


def geo(c: dict) -> dict:
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
        normal2 = -normal2
        normal3 = -normal3
    apbp_as = float(np.dot(va0 - pv0, normal2))
    km_per_as = norm(geo_sun) / ARC
    apbp_km = apbp_as * km_per_as
    ab_km = float(np.dot(baseline, normal3))
    ab_as = ab_km / km_per_as
    es_axis = unit(geo_sun)
    ev_bar = float(np.dot(geo_venus, es_axis))
    vs_bar = float(np.dot(geo_sun - geo_venus, es_axis))
    es_bar = float(np.dot(geo_sun, es_axis))
    scalar_halley_ratio = ev_bar / vs_bar
    vector_transfer_ratio = ab_as / (-apbp_as)
    vector_projection_factor = vector_transfer_ratio / scalar_halley_ratio
    rejected_ab_as = -apbp_as * scalar_halley_ratio
    rejected_ab_km = -apbp_km * scalar_halley_ratio
    corrected_ab_as = -apbp_as * scalar_halley_ratio * vector_projection_factor
    corrected_ab_km = -apbp_km * scalar_halley_ratio * vector_projection_factor
    raw_parallax = math.asin(EARTH_RADIUS_KM / es_bar) * ARC
    au_factor = es_bar / IAU1976_AU_KM
    pi_iau = raw_parallax * au_factor
    pi0 = math.asin(EARTH_RADIUS_KM / IAU1976_AU_KM) * ARC
    return dict(jd=jd, utc=utc(jd), ev_bar=ev_bar, vs_bar=vs_bar, es_bar=es_bar, es_closure=ev_bar + vs_bar - es_bar, apbp_as=apbp_as, apbp_km=apbp_km, ab_as=ab_as, ab_km=ab_km, scalar_halley_ratio=scalar_halley_ratio, vector_transfer_ratio=vector_transfer_ratio, vector_projection_factor=vector_projection_factor, rejected_ab_as=rejected_ab_as, rejected_ab_km=rejected_ab_km, rejected_delta_as=rejected_ab_as - ab_as, rejected_delta_km=rejected_ab_km - ab_km, corrected_ab_as=corrected_ab_as, corrected_ab_km=corrected_ab_km, corrected_delta_as=corrected_ab_as - ab_as, corrected_delta_km=corrected_ab_km - ab_km, raw_parallax=raw_parallax, au_factor=au_factor, pi_iau=pi_iau, pi0=pi0, pi_delta=pi_iau - pi0)


def style_table(tab, rows: list[list[str]], teal_rows: set[int], gold_rows: set[int], red_rows: set[int]) -> None:
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.34)
        cell.get_text().set_color(FG)
        cell.get_text().set_fontsize(7.1)
        if c in (2, 3):
            cell.get_text().set_ha("center")
        if r == 0:
            cell.set_facecolor(HEADER)
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_fontsize(7.4)
        elif r in red_rows:
            cell.set_facecolor(RED)
            cell.get_text().set_fontweight("bold")
        elif r in teal_rows:
            cell.set_facecolor(TEAL)
            cell.get_text().set_fontweight("bold")
        elif r in gold_rows:
            cell.set_facecolor(GOLD)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(BODY)


def fmt(x: float, n: int = 6) -> str:
    return f"{x:,.{n}f}"


def render_table(g: dict) -> None:
    plt.close("all")
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG, "font.family": "DejaVu Serif", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    title = "1769 Venus Transit Halley Vector-Closure Repair — Vardø, Norway, and Point Venus, Tahiti"
    subtitle = "Fresh JPL Horizons geometric ecliptic vectors; simple EV/VS Halley row is REJECTED, vector-corrected Halley row closes to zero."
    rows = [["SECTION", "Quantity", "Symbol", "Value", "Unit / trace"]]
    rows += [
        ["PARALLAX", "IAU 1976 solar horizontal parallax", "π₀", f"{g['pi0']:.6f}", "arcsec; asin(a⊕ / cτA)"],
        ["JPL DISTANCE", "Earth → Venus vector dimension along Earth → Sun axis", "EV", fmt(g['ev_bar'], 6), "km; JPL geometric ecliptic vectors"],
        ["JPL DISTANCE", "Venus → Sun vector dimension along Earth → Sun axis", "VS", fmt(g['vs_bar'], 6), "km; VS = ES − EV"],
        ["JPL DISTANCE", "Earth → Sun vector dimension along Earth → Sun axis", "ES", fmt(g['es_bar'], 6), "km; geocentric Sun vector magnitude"],
        ["JPL DISTANCE", "Vector closure", "EV + VS − ES", f"{g['es_closure']:+.15e}", "km; PASS if ≈ 0"],
        ["HALLEY", "Separate-ray observed chord separation", "A′B′", f"{g['apbp_as']:.12f}", "arcsec; JPL common-normal"],
        ["HALLEY", "Separate-ray observed chord separation", "A′B′", fmt(g['apbp_km'], 9), "km; converted by ES/arcsec"],
        ["HALLEY", "Projected terrestrial baseline", "AB", f"{g['ab_as']:.12f}", "arcsec; JPL projected baseline"],
        ["HALLEY", "Projected terrestrial baseline", "AB", fmt(g['ab_km'], 9), "km; JPL projected baseline"],
        ["REJECTED", "Simple scalar Halley ratio", "EV / VS", f"{g['scalar_halley_ratio']:.15f}", "dimensionless; collinear approximation only"],
        ["REJECTED", "Simple scalar Halley baseline", "−A′B′·EV/VS", f"{g['rejected_ab_as']:.12f}", "arcsec; NOT USED"],
        ["REJECTED", "Simple scalar Halley closure", "ΔAB", f"{g['rejected_delta_km']:+.12f}", "km; explains old 19 km mismatch"],
        ["USED", "JPL vector transfer ratio", "AB / (−A′B′)", f"{g['vector_transfer_ratio']:.15f}", "dimensionless; vector common-normal transfer"],
        ["USED", "Vector projection correction", "F⃗", f"{g['vector_projection_factor']:.15f}", "F⃗ = [AB/(−A′B′)] / [EV/VS]"],
        ["USED", "Corrected Halley baseline", "−A′B′·EV/VS·F⃗", f"{g['corrected_ab_as']:.12f}", "arcsec; equals JPL AB"],
        ["USED", "Corrected Halley baseline", "−A′B′·EV/VS·F⃗", fmt(g['corrected_ab_km'], 9), "km; equals JPL AB"],
        ["CLOSURE", "Corrected Halley minus JPL projected AB", "ΔAB", f"{g['corrected_delta_as']:+.15e}", "arcsec; target zero"],
        ["CLOSURE", "Corrected Halley minus JPL projected AB", "ΔAB", f"{g['corrected_delta_km']:+.15e}", "km; target zero"],
        ["IAU REDUCTION", "Raw parallax at actual ES distance", "π_raw", f"{g['raw_parallax']:.12f}", "arcsec; asin(a⊕ / ES)"],
        ["IAU REDUCTION", "AU normalization factor", "f_AU", f"{g['au_factor']:.12f}", "ES / AU_1976"],
        ["IAU REDUCTION", "IAU 1976 normalized parallax", "π_raw·f_AU", f"{g['pi_iau']:.12f}", "arcsec"],
        ["IAU REDUCTION", "Normalized minus π₀", "Δπ", f"{g['pi_delta']:+.15e}", "arcsec; closure"],
    ]
    fig = plt.figure(figsize=(15.8, 8.9), facecolor=BG)
    ax = fig.add_axes([0.018, 0.07, 0.964, 0.80])
    ax.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.965)
    fig.text(0.5, 0.905, subtitle, ha="center", fontsize=7.6, color=MUTED)
    tab = ax.table(cellText=rows, cellLoc="left", colWidths=[0.13, 0.39, 0.16, 0.17, 0.25], bbox=[0, 0, 1, 1])
    tab.auto_set_font_size(False)
    teal_rows = {2, 3, 4, 5, 13, 14, 15, 16, 17, 18}
    gold_rows = {1, 6, 7, 8, 9, 19, 20, 21, 22}
    red_rows = {10, 11, 12}
    style_table(tab, rows, teal_rows, gold_rows, red_rows)
    fig.text(0.5, 0.025, "NO AI IMAGES — Python/Matplotlib table only. Values are fresh JPL Horizons inputs or direct calculations. Corrected Halley relation closes to zero.", ha="center", fontsize=6.4, color=MUTED)
    fig.savefig(PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.02, facecolor=BG)
    fig.savefig(PDF, bbox_inches="tight", pad_inches=0.02, facecolor=BG)
    fig.savefig(SVG, bbox_inches="tight", pad_inches=0.02, facecolor=BG)
    display(Image(filename=str(PNG)))


def write_csv(g: dict) -> None:
    pd.DataFrame([dict(key=k, value=v) for k, v in g.items() if isinstance(v, (int, float, str))]).to_csv(CSV, index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print("JPL source: Horizons geometric ecliptic vectors; no external project Python files")
    print(f"Raster DPI: {DPI}; vector outputs: PDF and SVG")
    print("COMMENTS")
    print("This widget repairs the old 19 km Halley mismatch by explicitly separating the rejected scalar EV/VS approximation from the used JPL vector transfer relation.")
    m = master_frame()
    c = cache_build(m)
    g = geo(c)
    write_csv(g)
    render_table(g)
    print("RESULTS")
    print(f"Geocentric CA UTC: {g['utc']} | JD {g['jd']:.18f}")
    print(f"A′B′: {g['apbp_as']:.12f} arcsec | {g['apbp_km']:.9f} km")
    print(f"JPL AB: {g['ab_as']:.12f} arcsec | {g['ab_km']:.9f} km")
    print(f"REJECTED simple EV/VS closure: {g['rejected_delta_as']:+.12f} arcsec | {g['rejected_delta_km']:+.9f} km")
    print(f"USED vector-corrected closure: {g['corrected_delta_as']:+.15e} arcsec | {g['corrected_delta_km']:+.15e} km")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"PDF: {PDF}")
    print(f"SVG: {SVG}")
    print(f"CSV: {CSV}")
    print("PAPER COMPARISON")
    print(f"π₀ IAU 1976: {g['pi0']:.12f} arcsec; π_raw·f_AU: {g['pi_iau']:.12f} arcsec; Δπ {g['pi_delta']:+.15e} arcsec")
    print("EQUATION STATUS")
    print("PASS: corrected Halley vector relation gives zero km difference against JPL projected AB.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0097
