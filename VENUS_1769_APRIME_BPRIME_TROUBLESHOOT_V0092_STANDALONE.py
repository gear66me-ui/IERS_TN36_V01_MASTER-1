# V0092
# Audit reference: Standalone A′B′ routine audit; runs V0067/V0080 seconds-space and V0089 JD-space reductions from fresh JPL vectors; no AI images.
from __future__ import annotations

import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0092"
LOCAL_TZ = ZoneInfo("America/Bogota")
OUT = Path("/content/VENUS_1769_V0092_APBP_AUDIT")
PNG = OUT / "VENUS_1769_APBP_80_89_AUDIT_V0092.png"
CSV = OUT / "VENUS_1769_APBP_80_89_AUDIT_V0092.csv"

ARC = 206_264.80624709636
JPL_AU_KM = 149_597_870.700000
SUN_RADIUS_KM = 695_700.000000
VENUS_RADIUS_KM = 6_051.800000
START = "1769-06-03 18:00"
STOP = "1769-06-04 06:00"
STEP = "1m"
GEOCENTER = "@399"
PV = dict(key="POINT_VENUS", lat=-17.4956, lon=-149.4939, elevation=0.0, body=399)
VA = dict(key="VARDO", lat=70.3724, lon=31.1103, elevation=0.0, body=399)
SITES = (PV, VA)
TARGETS = (("SUN", "10"), ("VENUS", "299"))
PREFIXES = ("GEOCENTER_SUN", "GEOCENTER_VENUS", "POINT_VENUS_SUN", "POINT_VENUS_VENUS", "VARDO_SUN", "VARDO_VENUS")

BG = "#000000"
FG = "#F8FAFC"
MUTED = "#B8CBD6"
BLUE = "#42D7C3"
GOLD = "#D89B18"
ORANGE = "#D95A1B"
SUN = "#FFD34A"
HEADER = "#23466F"
BODY = "#101A2E"
TEAL = "#164B55"
BROWN = "#563B0B"


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
        raise RuntimeError("Zero vector normalization requested.")
    return a / n


def utc(jd: float) -> str:
    return Time(float(jd), format="jd", scale="tdb").utc.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def loc(site: dict) -> dict:
    return dict(lon=float(site["lon"]), lat=float(site["lat"]), elevation=float(site["elevation"]), body=int(site["body"]))


def download_series(prefix: str, target_id: str, location) -> pd.DataFrame:
    last = None
    for attempt in range(4):
        try:
            table = Horizons(id=target_id, location=location, epochs={"start": START, "stop": STOP, "step": STEP}, id_type=None).vectors(refplane="ecliptic", aberrations="geometric", cache=False)
            raw = table.to_pandas()
            out = pd.DataFrame({"JD_TDB": pd.to_numeric(raw["datetime_jd"], errors="coerce")})
            for axis in "xyz":
                out[f"{prefix}_{axis.upper()}_KM"] = pd.to_numeric(raw[axis], errors="coerce") * JPL_AU_KM
            out = out.dropna().drop_duplicates("JD_TDB").sort_values("JD_TDB").reset_index(drop=True)
            if len(out) < 600:
                raise RuntimeError(f"Incomplete JPL query for {prefix}: {len(out)} rows")
            return out
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"JPL query failed for {prefix}: {last}")


def build_master() -> pd.DataFrame:
    series = []
    for name, target_id in TARGETS:
        series.append(download_series(f"GEOCENTER_{name}", target_id, GEOCENTER))
    for site in SITES:
        for name, target_id in TARGETS:
            series.append(download_series(f"{site['key']}_{name}", target_id, loc(site)))
    master = series[0]
    for frame in series[1:]:
        master = master.merge(frame, on="JD_TDB", how="inner", validate="one_to_one")
    if len(master) < 600:
        raise RuntimeError(f"Synchronized JPL master too short: {len(master)} rows")
    return master.sort_values("JD_TDB").reset_index(drop=True)


def build_cache(master: pd.DataFrame) -> dict:
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


def gnom(ray: np.ndarray, center: np.ndarray, east: np.ndarray, north: np.ndarray) -> np.ndarray:
    h = unit(ray)
    den = float(np.dot(h, center))
    if den <= 0.0:
        raise RuntimeError("Ray outside tangent hemisphere.")
    return np.array([float(np.dot(h, east)), float(np.dot(h, north))]) / den


def common_relative(cache: dict, site_key: str, jd: float, bas: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    center, east, north = bas
    return ARC * (gnom(vec(cache, f"{site_key}_VENUS", jd), center, east, north) - gnom(vec(cache, f"{site_key}_SUN", jd), center, east, north))


def geocentric_ca_seconds_space(cache: dict) -> float:
    jds = np.asarray(cache["JD_TDB"], dtype=float)
    separations = np.array([sep_rad(vec(cache, "GEOCENTER_SUN", jd), vec(cache, "GEOCENTER_VENUS", jd)) for jd in jds], dtype=float)
    i = int(np.argmin(separations))
    lower_jd = float(jds[max(0, i - 3)])
    upper_jd = float(jds[min(len(jds) - 1, i + 3)])
    ref = 0.5 * (lower_jd + upper_jd)
    lo = (lower_jd - ref) * 86400.0
    hi = (upper_jd - ref) * 86400.0
    result = minimize_scalar(lambda seconds: sep_rad(vec(cache, "GEOCENTER_SUN", ref + float(seconds) / 86400.0), vec(cache, "GEOCENTER_VENUS", ref + float(seconds) / 86400.0)), bounds=(lo, hi), method="bounded", options={"xatol": 1.0e-4, "maxiter": 500})
    if not result.success:
        raise RuntimeError("V0067/V0080 seconds-space CA failed.")
    return ref + float(result.x) / 86400.0


def geocentric_ca_jd_space(cache: dict) -> float:
    jds = np.asarray(cache["JD_TDB"], dtype=float)
    separations = np.array([sep_rad(vec(cache, "GEOCENTER_SUN", jd), vec(cache, "GEOCENTER_VENUS", jd)) for jd in jds], dtype=float)
    i = int(np.argmin(separations))
    result = minimize_scalar(lambda jd: sep_rad(vec(cache, "GEOCENTER_SUN", jd), vec(cache, "GEOCENTER_VENUS", jd)), bounds=(float(jds[max(0, i - 2)]), float(jds[min(len(jds) - 1, i + 2)])), method="bounded")
    if not result.success:
        raise RuntimeError("V0089 JD-space CA failed.")
    return float(result.x)


def apbp_reduction(cache: dict, jd: float, label: str) -> dict:
    geo_sun = vec(cache, "GEOCENTER_SUN", jd)
    bas = basis(geo_sun)
    center, east, north = bas
    h = 0.5 / 86400.0
    q_pv = common_relative(cache, "POINT_VENUS", jd, bas)
    q_va = common_relative(cache, "VARDO", jd, bas)
    v_pv = common_relative(cache, "POINT_VENUS", jd + h, bas) - common_relative(cache, "POINT_VENUS", jd - h, bas)
    v_va = common_relative(cache, "VARDO", jd + h, bas) - common_relative(cache, "VARDO", jd - h, bas)
    direction_pv = unit(v_pv)
    direction_va = unit(v_va)
    if float(np.dot(direction_pv, direction_va)) < 0.0:
        direction_va = -direction_va
    track_direction = unit(direction_pv + direction_va)
    normal2 = np.array([-track_direction[1], track_direction[0]])
    if float(np.dot(q_va - q_pv, normal2)) < 0.0:
        normal2 = -normal2
    normal3 = unit(normal2[0] * east + normal2[1] * north)
    site_pv = geo_sun - vec(cache, "POINT_VENUS_SUN", jd)
    site_va = geo_sun - vec(cache, "VARDO_SUN", jd)
    baseline = site_va - site_pv
    if float(np.dot(baseline, normal3)) < 0.0:
        normal2 = -normal2
        normal3 = -normal3
    apbp_arcsec = float(np.dot(q_va - q_pv, normal2))
    km_per_arcsec = norm(geo_sun) / ARC
    ab_km = float(np.dot(baseline, normal3))
    ab_arcsec = ab_km / km_per_arcsec
    angle = math.degrees(math.atan2(track_direction[1], track_direction[0]))
    return dict(label=label, jd=jd, utc=utc(jd), apbp_arcsec=apbp_arcsec, apbp_km=apbp_arcsec * km_per_arcsec, ab_arcsec=ab_arcsec, ab_km=ab_km, km_per_arcsec=km_per_arcsec, track_angle_deg=angle)


def style_table(table, fs=8.0) -> None:
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#70879A")
        cell.set_linewidth(0.4)
        cell.get_text().set_color(FG)
        cell.get_text().set_fontsize(fs)
        if c >= 2:
            cell.get_text().set_ha("center")
        if r == 0:
            cell.set_facecolor(HEADER)
            cell.get_text().set_fontweight("bold")
        elif r in (1,):
            cell.set_facecolor(TEAL)
            cell.get_text().set_fontweight("bold")
        elif r in (2,):
            cell.set_facecolor(BROWN)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(BODY)


def plot(results: list[dict], delta: dict) -> None:
    plt.close("all")
    plt.rcParams.update({"font.family": "DejaVu Serif", "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": BG, "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED})
    fig = plt.figure(figsize=(15.5, 8.0), facecolor=BG)
    gs = fig.add_gridspec(2, 2, width_ratios=(1.12, 1.0), height_ratios=(0.72, 0.28), left=0.055, right=0.975, top=0.90, bottom=0.09, wspace=0.12, hspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    tabax = fig.add_subplot(gs[:, 1])
    fig.suptitle("1769 VENUS TRANSIT — A′B′ REDUCTION TROUBLESHOOT", fontsize=16, fontweight="bold", y=0.965)

    labels = ["A′B′ 80\nseconds-space CA", "A′B′ 89\nJD-space CA"]
    y = [results[0]["apbp_arcsec"], results[1]["apbp_arcsec"]]
    colors = [BLUE, ORANGE]
    ax.plot([0, 1], y, color=SUN, linewidth=0.9, alpha=0.80, zorder=1)
    ax.scatter([0, 1], y, s=90, color=colors, edgecolor=FG, linewidth=0.7, zorder=3)
    for i, res in enumerate(results):
        ax.annotate(f"{labels[i]}\n{res['apbp_arcsec']:.12f} arcsec\n{res['apbp_km']:,.6f} km", xy=(i, res["apbp_arcsec"]), xytext=(i, res["apbp_arcsec"] + (0.00045 if i == 0 else -0.00058)), ha="center", va="bottom" if i == 0 else "top", fontsize=8.7, fontweight="bold", color=colors[i], arrowprops=dict(arrowstyle="-", linewidth=0.55, color=colors[i]))
    ax.set_xlim(-0.35, 1.35)
    pad = max(abs(y[0] - y[1]) * 0.85, 0.0016)
    ax.set_ylim(min(y) - pad, max(y) + pad)
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel("A′B′ common-normal separation (arcsec)")
    ax.grid(color="#263A4B", linewidth=0.35, alpha=0.55)
    ax.set_title("Same fresh JPL vectors; only the geocentric closest-approach routine changes", fontsize=10, pad=9)

    ax2.axis("off")
    ax2.text(0.0, 0.82, "Δ(V0089 − V0080)", fontsize=10, fontweight="bold", color=SUN)
    ax2.text(0.0, 0.52, f"A′B′ difference: {delta['apbp_arcsec']:+.12f} arcsec    {delta['apbp_km']:+,.6f} km", fontsize=10, color=FG)
    ax2.text(0.0, 0.25, f"AB difference:   {delta['ab_arcsec']:+.12f} arcsec    {delta['ab_km']:+,.6f} km", fontsize=10, color=FG)
    ax2.text(0.0, 0.00, "Cause under test: V0089 changed the CA minimization from seconds-space ±3-min bracket to direct JD-space ±2-min bracket.", fontsize=8.6, color=MUTED)

    tabax.axis("off")
    rows = [["Metric", "V0080 / V0067 routine", "V0089 routine", "Δ 89−80"], ["CA UTC", results[0]["utc"], results[1]["utc"], f"{(results[1]['jd'] - results[0]['jd']) * 86400.0:+.6f} s"], ["A′B′", f"{results[0]['apbp_arcsec']:.12f}″", f"{results[1]['apbp_arcsec']:.12f}″", f"{delta['apbp_arcsec']:+.12f}″"], ["A′B′ km", f"{results[0]['apbp_km']:,.6f}", f"{results[1]['apbp_km']:,.6f}", f"{delta['apbp_km']:+,.6f}"], ["AB", f"{results[0]['ab_arcsec']:.12f}″", f"{results[1]['ab_arcsec']:.12f}″", f"{delta['ab_arcsec']:+.12f}″"], ["AB km", f"{results[0]['ab_km']:,.6f}", f"{results[1]['ab_km']:,.6f}", f"{delta['ab_km']:+,.6f}"], ["km / arcsec", f"{results[0]['km_per_arcsec']:.9f}", f"{results[1]['km_per_arcsec']:.9f}", f"{results[1]['km_per_arcsec']-results[0]['km_per_arcsec']:+.9f}"], ["Track angle", f"{results[0]['track_angle_deg']:.9f}°", f"{results[1]['track_angle_deg']:.9f}°", f"{results[1]['track_angle_deg']-results[0]['track_angle_deg']:+.9f}°"]]
    table = tabax.table(cellText=rows, cellLoc="left", colWidths=[0.22, 0.32, 0.27, 0.19], bbox=[0.0, 0.17, 1.0, 0.78])
    table.auto_set_font_size(False)
    style_table(table, fs=7.1)
    tabax.text(0.0, 0.09, "NO AI IMAGES — Python/Matplotlib only. Both A′B′ values are recomputed from fresh JPL Horizons vectors in this run.", fontsize=7.2, color=MUTED)
    fig.savefig(PNG, dpi=160, bbox_inches="tight", pad_inches=0.02, facecolor=BG)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{VERSION} A′B′ AUDIT — downloading fresh JPL vectors")
    master = build_master()
    cache = build_cache(master)
    res80 = apbp_reduction(cache, geocentric_ca_seconds_space(cache), "A′B′ 80 / V0067 seconds-space CA")
    res89 = apbp_reduction(cache, geocentric_ca_jd_space(cache), "A′B′ 89 / V0089 JD-space CA")
    delta = dict(apbp_arcsec=res89["apbp_arcsec"] - res80["apbp_arcsec"], apbp_km=res89["apbp_km"] - res80["apbp_km"], ab_arcsec=res89["ab_arcsec"] - res80["ab_arcsec"], ab_km=res89["ab_km"] - res80["ab_km"])
    pd.DataFrame([res80, res89]).to_csv(CSV, index=False)
    plot([res80, res89], delta)
    display(Image(filename=str(PNG)))
    print("RESULTS")
    print(f"A′B′ 80 / V0067 routine | CA {res80['utc']} | {res80['apbp_arcsec']:.12f} arcsec | {res80['apbp_km']:,.6f} km")
    print(f"A′B′ 89 / V0089 routine | CA {res89['utc']} | {res89['apbp_arcsec']:.12f} arcsec | {res89['apbp_km']:,.6f} km")
    print(f"DELTA 89-80 | {delta['apbp_arcsec']:+.12f} arcsec | {delta['apbp_km']:+,.6f} km")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0092
