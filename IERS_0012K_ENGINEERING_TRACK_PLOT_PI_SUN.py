# IERS-0012K
# Audit reference: GitHubDelivery@IERS-0012K; 0012J plot plus 12G-style summary table after plot.

import os
import sys
import math
import csv
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

VERSION = "IERS-0012K"
PROGRAM_NAME = "IERS_0012K_ENGINEERING_TRACK_PLOT_PI_SUN.py"

AU_KM = 149_597_870.7
ARCSEC_PER_RAD = 206_264.80624709636
EARTH_RADIUS_KM = 6_378.137
SUN_RADIUS_KM = 695_700.0
VENUS_RADIUS_KM = 6_051.8
PI_SUN_REFERENCE_ARCSEC = 8.794148
LOCAL_TZ = ZoneInfo("America/Bogota")

START = "2012-Jun-05 20:00"
STOP = "2012-Jun-06 07:30"
STEP = "1m"

SITE_A = {"key": "WELLINGTON_NZ", "short": "Wellington", "label": "Wellington NZ", "lon_deg_east": 174.77557, "lat_deg": -41.28664, "height_m": 0.0}
SITE_B = {"key": "ALAEJOS_SPAIN", "short": "Alaejos", "label": "Alaejos Spain", "lon_deg_east": -5.22443, "lat_deg": 41.28664, "height_m": 0.0}
TRACK_COLORS = {"Wellington NZ": "#ff3434", "Alaejos Spain": "#2f75ff"}


def ensure_package(import_name, pip_name):
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pip_name])


for import_name, pip_name in [("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("matplotlib", "matplotlib"), ("astroquery", "astroquery"), ("astropy", "astropy")]:
    ensure_package(import_name, pip_name)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar
from astroquery.jplhorizons import Horizons
import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.utils import iers

try:
    from IPython.display import HTML, display
except Exception:
    HTML = None
    display = None

iers.conf.auto_download = True
iers.conf.iers_degraded_accuracy = "warn"


def norm(v):
    return float(np.sqrt(np.dot(v, v)))


def unit(v):
    n = norm(v)
    if n == 0.0:
        raise RuntimeError("Zero vector.")
    return np.asarray(v, dtype=float) / n


def angular_sep_arcsec(a, b):
    c = float(np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0))
    return math.acos(c) * ARCSEC_PER_RAD


def horizons_geocenter_vectors(target_id, prefix):
    obj = Horizons(id=target_id, location="500@399", epochs={"start": START, "stop": STOP, "step": STEP})
    vec = obj.vectors().to_pandas()
    out = pd.DataFrame()
    out["jd_tdb"] = vec["datetime_jd"].astype(float)
    out["utc"] = vec["datetime_str"].astype(str)
    out[f"{prefix}_x_km"] = vec["x"].astype(float) * AU_KM
    out[f"{prefix}_y_km"] = vec["y"].astype(float) * AU_KM
    out[f"{prefix}_z_km"] = vec["z"].astype(float) * AU_KM
    return out


def build_master_geocenter():
    sun = horizons_geocenter_vectors("10", "GEOCENTER_SUN")
    venus = horizons_geocenter_vectors("299", "GEOCENTER_VENUS")
    return sun.merge(venus, on=["jd_tdb", "utc"], how="inner")


def build_cache(df):
    cache = {"jd_tdb": df["jd_tdb"].to_numpy(dtype=float), "utc": df["utc"].astype(str).tolist()}
    for col in df.columns:
        if col.endswith("_km"):
            cache[col] = CubicSpline(cache["jd_tdb"], df[col].to_numpy(dtype=float), bc_type="natural")
    return cache


def vec_at(cache, prefix, jd_tdb):
    return np.array([float(cache[f"{prefix}_x_km"](jd_tdb)), float(cache[f"{prefix}_y_km"](jd_tdb)), float(cache[f"{prefix}_z_km"](jd_tdb))], dtype=float)


def utc_at(cache, jd_tdb):
    return Time(jd_tdb, format="jd", scale="tdb").utc.iso.replace(" ", " ")


def earth_location(site):
    return EarthLocation.from_geodetic(lon=site["lon_deg_east"] * u.deg, lat=site["lat_deg"] * u.deg, height=site["height_m"] * u.m)


def observer_gcrs_km(site, jd_tdb):
    t = Time(jd_tdb, format="jd", scale="tdb")
    loc = earth_location(site)
    pos, _vel = loc.get_gcrs_posvel(t)
    return np.array([pos.x.to_value(u.km), pos.y.to_value(u.km), pos.z.to_value(u.km)], dtype=float)


def topocentric_body_vector(cache, site, body_prefix, jd_tdb):
    return vec_at(cache, body_prefix, jd_tdb) - observer_gcrs_km(site, jd_tdb)


def site_sep_arcsec(cache, site, jd_tdb):
    sun = topocentric_body_vector(cache, site, "GEOCENTER_SUN", jd_tdb)
    venus = topocentric_body_vector(cache, site, "GEOCENTER_VENUS", jd_tdb)
    return angular_sep_arcsec(sun, venus)


def angular_radii_arcsec(cache, site, jd_tdb):
    sun = topocentric_body_vector(cache, site, "GEOCENTER_SUN", jd_tdb)
    venus = topocentric_body_vector(cache, site, "GEOCENTER_VENUS", jd_tdb)
    rs = math.atan2(SUN_RADIUS_KM, norm(sun)) * ARCSEC_PER_RAD
    rv = math.atan2(VENUS_RADIUS_KM, norm(venus)) * ARCSEC_PER_RAD
    return rs, rv


def contact_function(cache, site, event, jd_tdb):
    sep = site_sep_arcsec(cache, site, jd_tdb)
    rs, rv = angular_radii_arcsec(cache, site, jd_tdb)
    threshold = rs + rv if event in ["C1", "C4"] else rs - rv
    return sep - threshold


def find_event_roots(cache, site, event):
    jds = cache["jd_tdb"]
    vals = np.array([contact_function(cache, site, event, jd) for jd in jds], dtype=float)
    roots = []
    for i in range(len(jds) - 1):
        if not np.isfinite(vals[i]) or not np.isfinite(vals[i + 1]):
            continue
        if vals[i] == 0.0:
            roots.append(float(jds[i]))
        elif vals[i] * vals[i + 1] < 0.0:
            roots.append(float(brentq(lambda x: contact_function(cache, site, event, x), jds[i], jds[i + 1], xtol=1e-13, rtol=1e-13, maxiter=100)))
    return sorted(roots)


def find_site_contacts(cache, site):
    outer = find_event_roots(cache, site, "C1")
    inner = find_event_roots(cache, site, "C2")
    if len(outer) < 2 or len(inner) < 2:
        raise RuntimeError(f"Could not derive four contacts for {site['label']}.")
    return {"C1": outer[0], "C2": inner[0], "C3": inner[-1], "C4": outer[-1]}


def find_site_closest(cache, site):
    jds = cache["jd_tdb"]
    vals = [site_sep_arcsec(cache, site, jd) for jd in jds]
    i = int(np.argmin(vals))
    a = jds[max(0, i - 3)]
    b = jds[min(len(jds) - 1, i + 3)]
    res = minimize_scalar(lambda jd: site_sep_arcsec(cache, site, jd), bounds=(a, b), method="bounded", options={"xatol": 1e-13})
    return float(res.x)


def fixed_geocenter_basis(cache, jd_tdb):
    sun = vec_at(cache, "GEOCENTER_SUN", jd_tdb)
    n = unit(sun)
    ref = np.array([0.0, 0.0, 1.0])
    if norm(np.cross(ref, n)) < 1e-12:
        ref = np.array([1.0, 0.0, 0.0])
    xhat = unit(np.cross(ref, n))
    yhat = unit(np.cross(n, xhat))
    return n, xhat, yhat


def ray_screen_point_arcsec(cache, site, jd_tdb, basis):
    n, xhat, yhat = basis
    obs = observer_gcrs_km(site, jd_tdb)
    sun_geo = vec_at(cache, "GEOCENTER_SUN", jd_tdb)
    venus_geo = vec_at(cache, "GEOCENTER_VENUS", jd_tdb)
    ray = venus_geo - obs
    denom = float(np.dot(ray, n))
    if abs(denom) < 1e-14:
        raise RuntimeError("Ray nearly parallel to solar screen.")
    tau = float(np.dot(sun_geo - obs, n) / denom)
    hit = obs + tau * ray
    screen_vec = hit - sun_geo
    es = norm(sun_geo)
    x = math.atan2(float(np.dot(screen_vec, xhat)), es) * ARCSEC_PER_RAD
    y = math.atan2(float(np.dot(screen_vec, yhat)), es) * ARCSEC_PER_RAD
    return np.array([x, y], dtype=float)


def pca_direction(points):
    pts = np.asarray(points, dtype=float)
    mu = pts.mean(axis=0)
    centered = pts - mu
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    d = vt[0]
    if d[0] < 0:
        d = -d
    return mu, unit(d)


def line_intersection(mu, d, mid, normal):
    a = np.column_stack([d, -normal])
    b = mid - mu
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    return mu + sol[0] * d


def distances_at(cache, jd_tdb):
    es = norm(vec_at(cache, "GEOCENTER_SUN", jd_tdb))
    ev = norm(vec_at(cache, "GEOCENTER_VENUS", jd_tdb))
    vs = norm(vec_at(cache, "GEOCENTER_VENUS", jd_tdb) - vec_at(cache, "GEOCENTER_SUN", jd_tdb))
    return es, ev, vs


def sun_radius_arcsec(cache, jd_tdb):
    es = norm(vec_at(cache, "GEOCENTER_SUN", jd_tdb))
    return math.atan2(SUN_RADIUS_KM, es) * ARCSEC_PER_RAD


def site_track(cache, site, contacts, closest_jd, basis):
    jds = cache["jd_tdb"]
    mask = (jds >= contacts["C1"]) & (jds <= contacts["C4"])
    use_jds = sorted(set([contacts["C1"], contacts["C2"], closest_jd, contacts["C3"], contacts["C4"]] + list(jds[mask])))
    pts = np.array([ray_screen_point_arcsec(cache, site, jd, basis) for jd in use_jds], dtype=float)
    mu, direction = pca_direction(pts)
    event_jds = {"C1": contacts["C1"], "C2": contacts["C2"], "CA": closest_jd, "C3": contacts["C3"], "C4": contacts["C4"]}
    event_pts = {name: ray_screen_point_arcsec(cache, site, jd, basis) for name, jd in event_jds.items()}
    event_radii = {name: angular_radii_arcsec(cache, site, jd)[1] for name, jd in event_jds.items()}
    return {
        "site": site,
        "jds": np.array(use_jds, dtype=float),
        "pts": pts,
        "mu": mu,
        "direction": direction,
        "event_jds": event_jds,
        "event_pts": event_pts,
        "event_radii": event_radii,
        "closest_jd": closest_jd,
        "closest_utc": utc_at(cache, closest_jd),
        "track_angle_deg": math.degrees(math.atan2(direction[1], direction[0])),
    }


def compute_parallax_geometry(cache, track_a, track_b, screen_jd):
    tangent = unit(track_a["direction"] + track_b["direction"])
    if tangent[0] < 0:
        tangent = -tangent
    normal = np.array([-tangent[1], tangent[0]])
    mid = 0.5 * (track_a["mu"] + track_b["mu"])
    aprime = line_intersection(track_a["mu"], track_a["direction"], mid, normal)
    bprime = line_intersection(track_b["mu"], track_b["direction"], mid, normal)
    abp_vec = bprime - aprime
    abp_arcsec = float(np.sqrt(np.sum(abp_vec * abp_vec)))
    ab_arcsec = abs(float(np.dot(abp_vec, normal)))
    es, ev, vs = distances_at(cache, screen_jd)
    abp_km = math.tan(abp_arcsec / ARCSEC_PER_RAD) * es
    ab_km = abp_km * ev / vs
    raw_phi = ab_arcsec * (ev / vs) * (EARTH_RADIUS_KM / ab_km)
    pi_sun = raw_phi * (es / AU_KM)
    return {
        "aprime": aprime,
        "bprime": bprime,
        "A_prime_B_prime_arcsec": abp_arcsec,
        "A_prime_B_prime_km": abp_km,
        "AB_arcsec": ab_arcsec,
        "AB_km": ab_km,
        "raw_phi_arcsec": raw_phi,
        "pi_sun_arcsec": pi_sun,
        "pi_sun_residual_arcsec": pi_sun - PI_SUN_REFERENCE_ARCSEC,
        "pi_sun_reference_arcsec": PI_SUN_REFERENCE_ARCSEC,
        "D_ES_AU": es / AU_KM,
        "D_EV_D_VS": ev / vs,
    }


def fmt(value, decimals=6):
    if isinstance(value, str):
        return value
    return f"{float(value):.{decimals}f}"


def axis_limits_for_half_sun(radius, track_a, track_b):
    all_pts = np.vstack([track_a["pts"], track_b["pts"]])
    y_med = float(np.median(all_pts[:, 1]))
    sign = 1.0 if y_med >= 0.0 else -1.0
    xlim = (-1.04 * radius, 1.04 * radius)
    ylim = (-0.06 * radius, 1.06 * radius) if sign > 0.0 else (-1.06 * radius, 0.06 * radius)
    y_min = float(np.min(all_pts[:, 1]))
    y_max = float(np.max(all_pts[:, 1]))
    if y_min < ylim[0] or y_max > ylim[1]:
        pad = 0.08 * radius
        ylim = (min(ylim[0], y_min - pad), max(ylim[1], y_max + pad))
    return xlim, ylim


def add_label(ax, xy, text, dx, dy, color):
    ax.annotate(text, xy=(xy[0], xy[1]), xytext=(xy[0] + dx, xy[1] + dy), textcoords="data", fontsize=5.7, color=color, ha="left", va="center", arrowprops=dict(arrowstyle="-", lw=0.22, color=color, shrinkA=0, shrinkB=2))


def summary_rows(track_a, track_b, geometry):
    delta_angle = abs(track_a["track_angle_deg"] - track_b["track_angle_deg"])
    return [
        ("β Wellington", track_a["track_angle_deg"], "deg"),
        ("β Alaejos", track_b["track_angle_deg"], "deg"),
        ("Δβ", delta_angle, "deg"),
        ("π⊙", geometry["pi_sun_arcsec"], "arcsec"),
        ("π⊙ Residual", geometry["pi_sun_residual_arcsec"], "arcsec"),
        ("A′B′", geometry["A_prime_B_prime_arcsec"], "arcsec"),
        ("A′B′", geometry["A_prime_B_prime_km"], "km"),
        ("AB", geometry["AB_arcsec"], "arcsec"),
        ("AB", geometry["AB_km"], "km"),
        ("D ES", geometry["D_ES_AU"], "AU"),
        ("D EV / D VS", geometry["D_EV_D_VS"], "ratio"),
        ("Wellington CA", track_a["closest_utc"], "UTC"),
        ("Alaejos CA", track_b["closest_utc"], "UTC"),
    ]


def add_summary_table_on_plot(ax, track_a, track_b, geometry):
    rows = [[q, fmt(v), u] for q, v, u in summary_rows(track_a, track_b, geometry)[:9]]
    table = ax.table(cellText=rows, colLabels=["Quantity", "Value", "Unit"], loc="upper right", colWidths=[0.28, 0.20, 0.16], bbox=[0.604, 0.646, 0.374, 0.328])
    table.auto_set_font_size(False)
    table.set_fontsize(5.6)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.22)
        cell.set_edgecolor("#1e4f64")
        if row == 0:
            cell.set_facecolor("#0a1a22")
            cell.get_text().set_color("#66e8ff")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#050b0f")
            if col == 1:
                cell.get_text().set_color("#ffc861")
                cell.get_text().set_fontweight("bold")
            elif col == 2:
                cell.get_text().set_color("#5ee08a")
            else:
                cell.get_text().set_color("#dff8ff")
    ax.text(0.605, 0.628, "A′B′ = solar-screen chord; AB = normal/projection baseline geometry.", transform=ax.transAxes, color="#8fb4c1", fontsize=5.6, ha="left", va="top")


def html_escape(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def html_label(text):
    return html_escape(text).replace("π⊙", "π<sub>⊙</sub>")


def display_html_summary(track_a, track_b, geometry, csv_path):
    if display is None or HTML is None:
        return False
    rows = summary_rows(track_a, track_b, geometry)
    body = []
    for q, v, u in rows:
        body.append(f"<tr><td class='quantity-cell'>{html_label(q)}</td><td class='value-cell'>{html_escape(fmt(v))}</td><td class='unit-cell'>{html_escape(u)}</td></tr>")
    html = f"""
    <style>
    .iers-wrap {{ background:#03080d; color:#e8f7ff; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; width:650px; max-width:98%; border:1px solid #1e4f64; border-radius:8px; padding:8px 8px 10px 8px; margin:8px 0 14px 0; }}
    .iers-title {{ color:#66e8ff; font-size:10px; font-weight:800; letter-spacing:0.06em; text-align:center; border-top:1px solid #25708b; border-bottom:1px solid #25708b; padding:5px 0; margin:5px 0 5px 0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .iers-table {{ border-collapse:collapse; width:100%; table-layout:fixed; font-size:10px; color:#dff8ff; background:#050b0f; margin:0 0 6px 0; }}
    .iers-table th {{ color:#66e8ff; background:#0a1a22; border-top:1px solid #1d3d4a; border-bottom:1px solid #1d3d4a; padding:4px 5px; text-align:left; font-weight:800; }}
    .iers-table td {{ border-bottom:1px solid #102630; padding:4px 5px; }}
    .iers-table th:nth-child(1), .iers-table td:nth-child(1) {{ width:42%; }}
    .iers-table th:nth-child(2), .iers-table td:nth-child(2) {{ width:40%; }}
    .iers-table th:nth-child(3), .iers-table td:nth-child(3) {{ width:18%; }}
    .quantity-cell {{ color:#dff8ff; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .value-cell {{ color:#ffc861; text-align:left; font-weight:800; white-space:nowrap; }}
    .unit-cell {{ color:#5ee08a; white-space:nowrap; }}
    .iers-note {{ color:#8fb4c1; font-size:9px; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    </style>
    <div class="iers-wrap">
      <div class="iers-title">ENGINEERING GEOMETRY SUMMARY — {html_escape(VERSION)}</div>
      <table class='iers-table'><thead><tr><th>Quantity</th><th>Value</th><th>Units</th></tr></thead><tbody>{''.join(body)}</tbody></table>
      <div class="iers-note">CSV: {html_escape(csv_path)}</div>
    </div>
    """
    display(HTML(html))
    return True


def write_event_csv(track_a, track_b, geometry, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([VERSION, "ENGINEERING TRACK PLOT EVENT AND GEOMETRY DATA"])
        writer.writerow([])
        writer.writerow(["site", "event", "utc", "jd_tdb", "x_arcsec", "y_arcsec", "venus_radius_arcsec", "track_angle_deg"])
        for track in [track_a, track_b]:
            site = track["site"]
            for event in ["C1", "C2", "CA", "C3", "C4"]:
                jd = track["event_jds"][event]
                xy = track["event_pts"][event]
                writer.writerow([site["label"], event, utc_at_from_jd(jd), f"{jd:.12f}", f"{xy[0]:.6f}", f"{xy[1]:.6f}", f"{track['event_radii'][event]:.6f}", f"{track['track_angle_deg']:.6f}"])
        writer.writerow([])
        writer.writerow(["quantity", "value", "unit"])
        for q, v, u in summary_rows(track_a, track_b, geometry):
            writer.writerow([q, fmt(v), u])


def utc_at_from_jd(jd):
    return Time(jd, format="jd", scale="tdb").utc.iso.replace(" ", " ")


def plot_engineering_track(cache, track_a, track_b, screen_jd, geometry, png_path):
    radius = sun_radius_arcsec(cache, screen_jd)
    fig, ax = plt.subplots(figsize=(9.6, 5.8), dpi=240)
    fig.patch.set_facecolor("#03080d")
    ax.set_facecolor("#03080d")
    ax.add_patch(Circle((0.0, 0.0), radius, fill=False, lw=0.38, ec="#66e8ff", alpha=0.95))
    ax.axhline(0.0, lw=0.20, color="#1d3d4a", alpha=0.72)
    ax.axvline(0.0, lw=0.20, color="#1d3d4a", alpha=0.72)

    for track in [track_a, track_b]:
        site_label = track["site"]["label"]
        color = TRACK_COLORS[site_label]
        pts = track["pts"]
        ax.plot(pts[:, 0], pts[:, 1], lw=0.34, color=color, solid_capstyle="round", label=site_label, zorder=3)
        ax.scatter(pts[::6, 0], pts[::6, 1], s=0.9, color=color, alpha=0.72, linewidths=0, zorder=4)
        for event in ["C1", "C2", "CA", "C3", "C4"]:
            xy = track["event_pts"][event]
            r = track["event_radii"][event]
            ax.add_patch(Circle((xy[0], xy[1]), r, fill=False, lw=0.24 if event != "CA" else 0.32, ec=color, alpha=0.92, zorder=2))
            ax.scatter([xy[0]], [xy[1]], s=4.2 if event == "CA" else 2.6, color=color, edgecolors="#03080d", linewidths=0.18, zorder=5)
        ca = track["event_pts"]["CA"]
        dy = 15.0 if site_label.startswith("Wellington") else -15.0
        add_label(ax, ca, f"{track['site']['short']} CA", 18.0, dy, color)

    for event, dx, dy in [("C1", -48.0, 12.0), ("C2", -38.0, 9.0), ("C3", 20.0, -10.0), ("C4", 30.0, -13.0)]:
        add_label(ax, track_a["event_pts"][event], event, dx, dy, "#8fb4c1")

    add_summary_table_on_plot(ax, track_a, track_b, geometry)
    xlim, ylim = axis_limits_for_half_sun(radius, track_a, track_b)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_linewidth(0.24)
        spine.set_color("#25708b")
    ax.tick_params(axis="both", colors="#8fb4c1", labelsize=6.5, width=0.24, length=2.0)
    ax.grid(True, color="#102630", linewidth=0.18, alpha=0.55)
    ax.set_xlabel("Solar-screen X offset (arcsec)", color="#8fb4c1", fontsize=7.5)
    ax.set_ylabel("Solar-screen Y offset (arcsec)", color="#8fb4c1", fontsize=7.5)
    ax.set_title("2012 Venus Transit — Engineering Half-Sun Track Reconstruction\nJPL Horizons Sun/Venus vectors + IERS/Astropy GCRS observer geometry", color="#f8fdff", fontsize=9.0, pad=8)
    legend = ax.legend(loc="lower right", fontsize=6.3, frameon=True, borderpad=0.45)
    legend.get_frame().set_facecolor("#071016")
    legend.get_frame().set_edgecolor("#1e4f64")
    legend.get_frame().set_linewidth(0.24)
    for text in legend.get_texts():
        text.set_color("#dff8ff")
    note = f"Venus disks are plotted to scale at C1, C2, closest approach, C3, and C4.  Wellington CA: {track_a['closest_utc']}   Alaejos CA: {track_b['closest_utc']}"
    fig.text(0.5, 0.016, note, ha="center", va="bottom", fontsize=6.2, color="#8fb4c1")
    fig.savefig(png_path, dpi=460, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.055)
    plt.show()
    plt.close(fig)


def main():
    print(f"CODE OUTPUT: {VERSION}")
    print(f"PROGRAM: {PROGRAM_NAME}")
    print()
    out_dir = "/content/IERS_TN36_V01_MASTER_OUTPUT"
    os.makedirs(out_dir, exist_ok=True)
    geo_df = build_master_geocenter()
    cache = build_cache(geo_df)
    contacts_a = find_site_contacts(cache, SITE_A)
    contacts_b = find_site_contacts(cache, SITE_B)
    closest_a = find_site_closest(cache, SITE_A)
    closest_b = find_site_closest(cache, SITE_B)
    screen_jd = 0.5 * (closest_a + closest_b)
    basis = fixed_geocenter_basis(cache, screen_jd)
    track_a = site_track(cache, SITE_A, contacts_a, closest_a, basis)
    track_b = site_track(cache, SITE_B, contacts_b, closest_b, basis)
    geometry = compute_parallax_geometry(cache, track_a, track_b, screen_jd)
    png_path = os.path.join(out_dir, f"{VERSION}_WELLINGTON_ALAEJOS_ENGINEERING_HALF_SUN_TRACKS.png")
    csv_path = os.path.join(out_dir, f"{VERSION}_WELLINGTON_ALAEJOS_ENGINEERING_HALF_SUN_EVENTS_AND_GEOMETRY.csv")
    write_event_csv(track_a, track_b, geometry, csv_path)
    plot_engineering_track(cache, track_a, track_b, screen_jd, geometry, png_path)
    display_html_summary(track_a, track_b, geometry, csv_path)
    print("RESULTS")
    print(f"Wellington closest UTC : {track_a['closest_utc']}")
    print(f"Alaejos closest UTC    : {track_b['closest_utc']}")
    print(f"Wellington track angle : {track_a['track_angle_deg']:.6f} deg")
    print(f"Alaejos track angle    : {track_b['track_angle_deg']:.6f} deg")
    print(f"Track angle delta abs  : {abs(track_a['track_angle_deg'] - track_b['track_angle_deg']):.6f} deg")
    print(f"A prime B prime        : {geometry['A_prime_B_prime_arcsec']:.6f} arcsec")
    print(f"A prime B prime        : {geometry['A_prime_B_prime_km']:.6f} km")
    print(f"AB                     : {geometry['AB_arcsec']:.6f} arcsec")
    print(f"AB                     : {geometry['AB_km']:.6f} km")
    print(f"Pi sun                 : {geometry['pi_sun_arcsec']:.6f} arcsec")
    print(f"PNG output             : {png_path}")
    print(f"CSV output             : {csv_path}")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# IERS-0012K
