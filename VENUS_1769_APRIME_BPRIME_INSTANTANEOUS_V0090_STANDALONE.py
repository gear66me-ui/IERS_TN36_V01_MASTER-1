# V0090
# Audit reference: standalone JPL Horizons minute-by-minute instantaneous A′B′ curvature audit; Matplotlib only; no AI images; no external Python file references.
from __future__ import annotations

import math
import subprocess
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


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

VERSION = "V0090"
LOCAL_TZ = ZoneInfo("America/Bogota")
OUT = Path("/content/VENUS_1769_APRIME_BPRIME_INSTANTANEOUS_V0090_OUTPUT")
PNG = OUT / "VENUS_1769_APRIME_BPRIME_INSTANTANEOUS_CURVATURE_V0090.png"
CSV = OUT / "VENUS_1769_APRIME_BPRIME_INSTANTANEOUS_CURVATURE_V0090.csv"
CONTACT_CSV = OUT / "VENUS_1769_APRIME_BPRIME_CONTACTS_V0090.csv"

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
SUN_FILL = "#D95A1B"
SUN_EDGE = "#FFD34A"
TABLE_HEADER = "#23466F"
TABLE_BODY = "#101A2E"
TABLE_TEAL = "#164B55"
TABLE_GOLD = "#563B0B"

POINT_VENUS = {"key": "POINT_VENUS", "label": "Point Venus, Tahiti", "short": "PV", "lat": -17.4956, "lon": -149.4939, "elevation": 0.0, "body": 399, "color": BLUE}
VARDO = {"key": "VARDO", "label": "Vardø, Norway", "short": "V", "lat": 70.3724, "lon": 31.1103, "elevation": 0.0, "body": 399, "color": GOLD}
SITES = (POINT_VENUS, VARDO)
PREFIXES = ("GEOCENTER_SUN", "GEOCENTER_VENUS", "POINT_VENUS_SUN", "POINT_VENUS_VENUS", "VARDO_SUN", "VARDO_VENUS")


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def unit(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = norm(a)
    if n <= 0.0:
        raise RuntimeError("Zero vector cannot be normalized.")
    return a / n


def utc(jd: float) -> str:
    return Time(float(jd), format="jd", scale="tdb").utc.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def utc_time(jd: float) -> str:
    return utc(jd).split(" ", 1)[1]


def loc(site: dict[str, object]) -> dict[str, float | int]:
    return {"lon": float(site["lon"]), "lat": float(site["lat"]), "elevation": float(site["elevation"]), "body": int(site["body"])}


def download(prefix: str, target_id: str, location: str | dict[str, float | int]) -> pd.DataFrame:
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
    parts: list[pd.DataFrame] = []
    for target_name, target_id in (("SUN", "10"), ("VENUS", "299")):
        parts.append(download(f"GEOCENTER_{target_name}", target_id, "@399"))
    for site in SITES:
        for target_name, target_id in (("SUN", "10"), ("VENUS", "299")):
            parts.append(download(f"{site['key']}_{target_name}", target_id, loc(site)))
    master = parts[0]
    for df in parts[1:]:
        master = master.merge(df, on="JD_TDB", how="inner", validate="one_to_one")
    master = master.sort_values("JD_TDB").reset_index(drop=True)
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


def events_for(c: dict[str, object], site: dict[str, object]) -> dict[str, float]:
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


def gnomonic(v: np.ndarray, center: np.ndarray, east: np.ndarray, north: np.ndarray) -> np.ndarray:
    d = unit(v)
    den = float(np.dot(d, center))
    if den <= 0.0:
        raise RuntimeError("Ray outside tangent hemisphere")
    return np.array([float(np.dot(d, east)), float(np.dot(d, north))]) / den


def relative_xy(c: dict[str, object], site_key: str, jd: float) -> np.ndarray:
    basis = basis_from_sun(vec(c, "GEOCENTER_SUN", jd))
    center, east, north = basis
    sun = vec(c, f"{site_key}_SUN", jd)
    ven = vec(c, f"{site_key}_VENUS", jd)
    return ARC * (gnomonic(ven, center, east, north) - gnomonic(sun, center, east, north))


def instant_apbp(c: dict[str, object], jd: float) -> dict[str, float]:
    h = 60.0 / 86400.0
    pv = relative_xy(c, "POINT_VENUS", jd)
    va = relative_xy(c, "VARDO", jd)
    pv_vel = (relative_xy(c, "POINT_VENUS", jd + h) - relative_xy(c, "POINT_VENUS", jd - h)) / 120.0
    va_vel = (relative_xy(c, "VARDO", jd + h) - relative_xy(c, "VARDO", jd - h)) / 120.0
    tangent = unit(0.5 * (pv_vel + va_vel))
    normal = np.array([-tangent[1], tangent[0]])
    delta = va - pv
    if float(np.dot(delta, normal)) < 0.0:
        normal = -normal
    apbp = float(np.dot(delta, normal))
    along = float(np.dot(delta, tangent))
    angle = math.degrees(math.atan2(tangent[1], tangent[0]))
    return {"jd_tdb": jd, "utc": utc(jd), "minute_from_start": 0.0, "apbp_arcsec": apbp, "along_arcsec": along, "track_angle_deg": angle, "pv_x": float(pv[0]), "pv_y": float(pv[1]), "vardo_x": float(va[0]), "vardo_y": float(va[1])}


def table_style(table, gold_rows=(), teal_rows=(), fontsize=6.2) -> None:
    table.auto_set_font_size(False)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#70879A")
        cell.set_linewidth(0.35)
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


def analyze() -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, float], dict[str, float]]:
    master = build_master()
    c = splines(master)
    evs = {"Point Venus, Tahiti": events_for(c, POINT_VENUS), "Vardø, Norway": events_for(c, VARDO)}
    start = min(evs["Point Venus, Tahiti"]["C1"], evs["Vardø, Norway"]["C1"])
    stop = max(evs["Point Venus, Tahiti"]["C4"], evs["Vardø, Norway"]["C4"])
    minute_jds = np.asarray(c["JD_TDB"], dtype=float)
    minute_jds = minute_jds[(minute_jds >= start) & (minute_jds <= stop)]
    rows = [instant_apbp(c, float(jd)) for jd in minute_jds]
    df = pd.DataFrame(rows)
    df["minute_from_start"] = (df["jd_tdb"] - float(df["jd_tdb"].iloc[0])) * 1440.0
    x = df["minute_from_start"].to_numpy(float)
    y = df["apbp_arcsec"].to_numpy(float)
    quad = np.polyfit(x - x.mean(), y, 2)
    fit = np.polyval(quad, x - x.mean())
    stats = {
        "min_arcsec": float(y.min()),
        "max_arcsec": float(y.max()),
        "range_arcsec": float(y.max() - y.min()),
        "mean_arcsec": float(y.mean()),
        "ca_mean_utc": utc(0.5 * (evs["Point Venus, Tahiti"]["CA"] + evs["Vardø, Norway"]["CA"])),
        "quadratic_arcsec_per_min2": float(quad[0]),
        "linear_arcsec_per_min": float(quad[1]),
        "rms_arcsec": float(np.sqrt(np.mean((y - fit) ** 2))),
    }
    mean_events = {event: 0.5 * (evs["Point Venus, Tahiti"][event] + evs["Vardø, Norway"][event]) for event in EVENTS}
    return df, evs, mean_events, stats


def plot(df: pd.DataFrame, evs: dict[str, dict[str, float]], mean_events: dict[str, float], stats: dict[str, float]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False, float_format="%.15f")
    contact_rows = []
    for site, ev in evs.items():
        for event in EVENTS:
            contact_rows.append({"station": site, "event": event, "utc": utc(ev[event]), "jd_tdb": ev[event]})
    pd.DataFrame(contact_rows).to_csv(CONTACT_CSV, index=False, float_format="%.15f")

    plt.close("all")
    plt.rcParams.update({"font.family": "DejaVu Serif", "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED})
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.08, 1.0], left=0.045, right=0.985, top=0.91, bottom=0.095, wspace=0.045)
    ax = fig.add_subplot(gs[0, 0])
    side = gs[0, 1].subgridspec(2, 1, height_ratios=[0.58, 0.42], hspace=0.16)
    contact_ax = fig.add_subplot(side[0, 0])
    stats_ax = fig.add_subplot(side[1, 0])
    fig.suptitle("1769 VENUS TRANSIT — INSTANTANEOUS A′B′ CURVATURE AUDIT", fontsize=15, fontweight="bold", y=0.965)

    x = df["minute_from_start"].to_numpy(float)
    y = df["apbp_arcsec"].to_numpy(float)
    ax.axhspan(y.min() - 0.04, y.max() + 0.04, color=SUN_FILL, alpha=0.16, zorder=0)
    ax.plot(x, y, color=SUN_EDGE, linewidth=1.10, zorder=3, label="instantaneous A′B′ normal separation")
    ax.scatter(x, y, s=8, color=BLUE, edgecolors=BG, linewidths=0.18, zorder=4, label="one-minute JPL samples")
    xfit = np.linspace(x.min(), x.max(), 600)
    quad = np.polyfit(x - x.mean(), y, 2)
    yfit = np.polyval(quad, xfit - x.mean())
    ax.plot(xfit, yfit, color=GOLD, linewidth=0.90, linestyle="--", zorder=2, label="quadratic curvature fit")

    interp = interp1d(df["jd_tdb"].to_numpy(float), y, kind="linear", bounds_error=False, fill_value="extrapolate")
    start_jd = float(df["jd_tdb"].iloc[0])
    for event in EVENTS:
        jd = mean_events[event]
        xm = (jd - start_jd) * 1440.0
        ym = float(interp(jd))
        ax.axvline(xm, color=GOLD if event == "CA" else BLUE, linewidth=0.45, alpha=0.70, zorder=1)
        ax.scatter([xm], [ym], s=42 if event == "CA" else 30, color=GOLD if event == "CA" else BLUE, edgecolors=FG, linewidths=0.38, zorder=5)
        dy = 0.055 if event in ("C1", "C2", "CA") else -0.065
        ax.annotate(event, xy=(xm, ym), xytext=(xm, ym + dy), ha="center", va="bottom" if dy > 0 else "top", fontsize=8.5, color=FG, fontweight="bold", arrowprops={"arrowstyle": "-", "color": FG, "linewidth": 0.35})

    ax.set_xlabel("Transit elapsed time from first one-minute sample (minutes)", fontsize=10)
    ax.set_ylabel("Instantaneous A′B′ normal separation (arcsec)", fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.35, alpha=0.55)
    ax.tick_params(labelsize=8, width=0.35, length=3)
    ax.set_xlim(x.min() - 6, x.max() + 6)
    pad = max(0.06, 0.22 * (y.max() - y.min()))
    ax.set_ylim(y.min() - pad, y.max() + pad)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=8)
    for text in leg.get_texts():
        text.set_color(FG)

    contact_ax.axis("off")
    contact_ax.set_title("RECOMPUTED CONTACT TIMES — UTC", loc="left", fontsize=9.5, fontweight="bold", pad=6)
    contact_rows = [["Station", "C1", "C2", "CA", "C3", "C4"]]
    for site in ("Point Venus, Tahiti", "Vardø, Norway"):
        contact_rows.append([site] + [utc_time(evs[site][event]) for event in EVENTS])
    contact_table = contact_ax.table(cellText=contact_rows, cellLoc="center", colWidths=[0.29, 0.142, 0.142, 0.142, 0.142, 0.142], bbox=[0.0, 0.42, 1.0, 0.45])
    table_style(contact_table, teal_rows=(1,), gold_rows=(2,), fontsize=5.9)
    formula_rows = [["Event", "Plot marker", "Meaning"], ["C1/C4", "external tangency", "ρ = R☉ + R♀"], ["C2/C3", "internal tangency", "ρ = R☉ − R♀"], ["CA", "local midpoint marker", "mean of PV/Vardø local CA"]]
    formula_table = contact_ax.table(cellText=formula_rows, cellLoc="left", colWidths=[0.18, 0.32, 0.50], bbox=[0.0, 0.02, 1.0, 0.30])
    table_style(formula_table, teal_rows=(3,), gold_rows=(), fontsize=6.0)

    stats_ax.axis("off")
    stats_ax.set_title("A′B′ CURVATURE RESULTS", loc="left", fontsize=9.5, fontweight="bold", pad=6)
    stat_rows = [
        ["Quantity", "Value", "Unit"],
        ["A′B′ minimum", f"{stats['min_arcsec']:.9f}", "arcsec"],
        ["A′B′ maximum", f"{stats['max_arcsec']:.9f}", "arcsec"],
        ["A′B′ range", f"{stats['range_arcsec']:.9f}", "arcsec"],
        ["A′B′ mean", f"{stats['mean_arcsec']:.9f}", "arcsec"],
        ["Quadratic curvature", f"{stats['quadratic_arcsec_per_min2']:+.12e}", "arcsec/min²"],
        ["Linear slope", f"{stats['linear_arcsec_per_min']:+.12e}", "arcsec/min"],
        ["Quadratic RMS", f"{stats['rms_arcsec']:.9f}", "arcsec"],
        ["Samples", f"{len(df):d}", "one-minute JPL points"],
    ]
    stat_table = stats_ax.table(cellText=stat_rows, cellLoc="left", colWidths=[0.44, 0.34, 0.22], bbox=[0.0, 0.05, 1.0, 0.86])
    table_style(stat_table, teal_rows=(1, 2, 3, 4), gold_rows=(5, 6, 7, 8), fontsize=6.3)

    fig.text(0.50, 0.035, "NO AI IMAGES — Python/Matplotlib only. Standalone JPL Horizons geometric-vector computation; instantaneous A′B′ derived minute-by-minute from the local tangent-velocity normal.", ha="center", fontsize=6.3, color=MUTED)
    fig.savefig(PNG, dpi=170, bbox_inches="tight", pad_inches=0.02, facecolor=BG)
    display(Image(filename=str(PNG)))
    print("V0090")
    print(f"PNG: {PNG}")
    print(f"CSV: {CSV}")
    print(f"CONTACTS: {CONTACT_CSV}")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))


def main() -> None:
    df, evs, mean_events, stats = analyze()
    plot(df, evs, mean_events, stats)


if __name__ == "__main__":
    main()
# V0090
