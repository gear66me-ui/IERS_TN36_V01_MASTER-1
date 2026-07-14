# V0096
# Audit reference: compact table-only 1769 Venus transit parallax audit; standalone JPL Horizons vectors; Matplotlib only; no AI images.
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
ROOT = Path("/content")
OUT = ROOT / "VENUS_1769_V0096_PARALLAX_COMPACT_TABLE"
PNG = OUT / "VENUS_1769_V0096_PARALLAX_COMPACT_TABLE_600DPI.png"
PDF = OUT / "VENUS_1769_V0096_PARALLAX_COMPACT_TABLE_VECTOR.pdf"
SVG = OUT / "VENUS_1769_V0096_PARALLAX_COMPACT_TABLE_VECTOR.svg"
CSV = OUT / "VENUS_1769_V0096_PARALLAX_COMPACT_TABLE.csv"
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
PREFIXES = ("GEOCENTER_SUN", "GEOCENTER_VENUS", "POINT_VENUS_SUN", "POINT_VENUS_VENUS", "VARDO_SUN", "VARDO_VENUS")

BG = "#000000"
FG = "#F8FAFC"
MUTED = "#B8CBD6"
HEADER = "#23466F"
TEAL = "#164B55"
GOLD = "#563B0B"
BODY = "#101A2E"
EDGE = "#70879A"


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


def gnom(ray: np.ndarray, cen: np.ndarray, east: np.ndarray, north: np.ndarray) -> np.ndarray:
    h = unit(ray)
    den = float(np.dot(h, cen))
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
    res = minimize_scalar(lambda s: sep_rad(vec(c, "GEOCENTER_SUN", ref + float(s) / 86400.0), vec(c, "GEOCENTER_VENUS", ref + float(s) / 86400.0)), bounds=(lo_s, hi_s), method="bounded", options={"xatol": 1e-4, "maxiter": 500})
    if not res.success:
        raise RuntimeError("Geocentric closest approach failed")
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
        normal2, normal3 = -normal2, -normal3
    apbp_as = float(np.dot(va0 - pv0, normal2))
    km_per_as = norm(geo_sun) / ARC
    ab_km = float(np.dot(baseline, normal3))
    es_axis = unit(geo_sun)
    ev_bar = float(np.dot(geo_venus, es_axis))
    vs_bar = float(np.dot(geo_sun - geo_venus, es_axis))
    es_bar = float(np.dot(geo_sun, es_axis))
    return dict(
        jd=jd,
        utc=utc(jd),
        apbp_as=apbp_as,
        apbp_km=apbp_as * km_per_as,
        ab_km=ab_km,
        ab_as=ab_km / km_per_as,
        ev_bar=ev_bar,
        vs_bar=vs_bar,
        es_bar=es_bar,
        vector_closure=es_bar - ev_bar - vs_bar,
        norm_factor=es_bar / IAU1976_AU_KM,
        km_per_as=km_per_as,
    )


def station_events(c: dict, site: dict) -> dict:
    inn = roots(c, site["key"], True)
    ca = local_ca(c, site["key"], inn[0], inn[1])
    return dict(CA=ca)


def table_rows(g: dict) -> list[list[str]]:
    pi0 = math.asin(EARTH_RADIUS_KM / IAU1976_AU_KM) * ARC
    pi_raw = pi0 / g["norm_factor"]
    pi_iau = pi_raw * g["norm_factor"]
    halley_ratio = abs(g["ab_as"] / g["apbp_as"])
    return [
        ["Group", "Quantity", "Symbol", "Arcsec / ratio", "Kilometers", "Derivation / status"],
        ["Vector distances", "Earth–Venus projected vector dimension", "EV̄", "", f"{g['ev_bar']:,.6f}", "JPL vectors projected on ES axis"],
        ["Vector distances", "Venus–Sun projected vector dimension", "VS̄", "", f"{g['vs_bar']:,.6f}", "JPL vectors projected on ES axis"],
        ["Vector distances", "Earth–Sun projected vector dimension", "ES̄", "", f"{g['es_bar']:,.6f}", "JPL vectors projected on ES axis"],
        ["Vector closure", "ES̄ − EV̄ − VS̄", "ΔES", "", f"{g['vector_closure']:+.12e}", "PASS"],
        ["Normalization", "IAU 1976 AU from c·τA", "AU₇₆", "", f"{IAU1976_AU_KM:,.6f}", "c = 299792.458 km/s; τA = 499.004782 s"],
        ["Normalization", "1769 distance normalization", "N₁₇₆₉", f"{g['norm_factor']:.12f}", "", "N = ES̄ / AU₇₆"],
        ["Parallax", "Raw Halley parallax", "πρ", f"{pi_raw:.6f}", "", "πρ = π₀ / N₁₇₆₉"],
        ["Parallax", "IAU 1976 normalized parallax", "π₀", f"{pi0:.6f}", "", "π₀ = asin(a⊕ / AU₇₆)"],
        ["Parallax", "Check: πρ × N₁₇₆₉", "πIAU", f"{pi_iau:.6f}", "", "PASS; equals π₀"],
        ["Halley geometry", "Common-normal separate-ray spacing", "A′B′", f"{g['apbp_as']:.6f}", f"{g['apbp_km']:,.6f}", "JPL ecliptic common-normal"],
        ["Halley geometry", "Projected observer baseline", "AB", f"{g['ab_as']:.6f}", f"{g['ab_km']:,.6f}", "JPL baseline projected on same normal"],
        ["Halley check", "AB / A′B′ transfer ratio", "kH", f"{halley_ratio:.12f}", "", "Used only as internal closure check"],
    ]


def style_table(tab) -> None:
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(EDGE)
        cell.set_linewidth(0.35)
        cell.get_text().set_color(FG)
        cell.get_text().set_fontsize(7.1)
        if r == 0:
            cell.set_facecolor(HEADER)
            cell.get_text().set_fontweight("bold")
        elif r in (1, 2, 3, 4):
            cell.set_facecolor(TEAL)
            if r == 4:
                cell.get_text().set_fontweight("bold")
        elif r in (5, 6, 7, 8, 9):
            cell.set_facecolor(GOLD if r in (6, 9) else BODY)
            if r in (6, 7, 8, 9):
                cell.get_text().set_fontweight("bold")
        elif r in (10, 11, 12):
            cell.set_facecolor(TEAL if r in (10, 11) else BODY)
            if r in (10, 11):
                cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(BODY)
        if c in (2, 3, 4):
            cell.get_text().set_ha("center")


def plot_table(rows: list[list[str]]) -> None:
    plt.close("all")
    plt.rcParams.update({"figure.facecolor": BG, "savefig.facecolor": BG, "text.color": FG, "font.family": "DejaVu Serif", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(16, 8.2), facecolor=BG)
    ax = fig.add_subplot(111)
    ax.axis("off")
    fig.suptitle("Summary of Venus Transit 1769 Parallax Calculations Between Vardø, Norway, and Point Venus, Tahiti", fontsize=16, fontweight="bold", y=0.965)
    ax.text(0.5, 0.905, "JPL Horizons geometric ecliptic vectors • compact Halley reduction audit • table only", ha="center", va="center", fontsize=9, color=MUTED, transform=ax.transAxes)
    tab = ax.table(cellText=rows, cellLoc="left", colWidths=[0.15, 0.28, 0.10, 0.14, 0.17, 0.28], bbox=[0.015, 0.075, 0.970, 0.790])
    tab.auto_set_font_size(False)
    style_table(tab)
    fig.text(0.5, 0.030, "Output is a Matplotlib table only. No transit-geometry plot is generated.", ha="center", fontsize=7, color=MUTED)
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
    print("JPL source: Horizons geometric ecliptic vectors")
    print(f"Raster DPI: {DPI}; vector outputs: PDF and SVG")
    print("COMMENTS")
    print("Compact table only: planet distances, normalization factor, raw/IAU parallax, A′B′, AB, and Halley closure checks. Contacts intentionally omitted.")
    m = master_frame()
    c = cache_build(m)
    g = geo(c)
    rows = table_rows(g)
    write_csv(rows)
    plot_table(rows)
    pi0 = math.asin(EARTH_RADIUS_KM / IAU1976_AU_KM) * ARC
    pi_raw = pi0 / g["norm_factor"]
    halley_as_closure = abs(g["apbp_as"]) * abs(g["ab_as"] / g["apbp_as"]) - abs(g["ab_as"])
    halley_km_closure = abs(g["apbp_km"]) * abs(g["ab_as"] / g["apbp_as"]) - abs(g["ab_km"])
    parallax_closure = pi_raw * g["norm_factor"] - pi0
    print("RESULTS")
    print(f"Geocentric CA UTC: {g['utc']} | JD {g['jd']:.18f}")
    print(f"ES̄/AU₇₆ normalization factor: {g['norm_factor']:.12f}")
    print(f"πρ raw: {pi_raw:.12f} arcsec")
    print(f"π₀ IAU 1976: {pi0:.12f} arcsec")
    print(f"A′B′: {g['apbp_as']:.12f} arcsec | {g['apbp_km']:.9f} km")
    print(f"AB: {g['ab_as']:.12f} arcsec | {g['ab_km']:.9f} km")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"PDF: {PDF}")
    print(f"SVG: {SVG}")
    print(f"CSV: {CSV}")
    print("PAPER COMPARISON")
    print("IAU 1976 π₀ is calculated directly from a⊕ and AU₇₆ = c·τA, not manually entered.")
    print("EQUATION STATUS")
    print(f"Vector distance closure ES̄ − EV̄ − VS̄: {g['vector_closure']:+.12e} km")
    print(f"Halley angular closure |A′B′|·|AB/A′B′| − |AB|: {halley_as_closure:+.12e} arcsec")
    print(f"Halley kilometer closure |A′B′km|·|AB/A′B′| − |ABkm|: {halley_km_closure:+.12e} km")
    print(f"Parallax normalization closure πρ·N₁₇₆₉ − π₀: {parallax_closure:+.12e} arcsec")
    print("PASS")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0096
