# IERS-0012O
# Audit reference: GitHubDelivery@IERS-0012O; 1769 North Pole / South Pole engineering half-Sun plot using JPL Horizons SITE_COORD vectors.

import os
import sys
import math
import csv
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

VERSION = "IERS-0012O"
PROGRAM_NAME = "IERS_0012O_NORTH_SOUTH_POLE_ENGINEERING_TRACK_PLOT_PI_SUN.py"

AU_KM = 149_597_870.7
ARCSEC_PER_RAD = 206_264.80624709636
EARTH_RADIUS_KM = 6_378.137
SUN_RADIUS_KM = 695_700.0
VENUS_RADIUS_KM = 6_051.8
PI_SUN_REFERENCE_ARCSEC = 8.794148

LOCAL_TZ = ZoneInfo("America/Bogota")

START = "1769-Jun-03 18:00"
STOP = "1769-Jun-04 04:00"
STEP = "1m"

SITE_A = {
    "key": "NORTH_POLE_0E",
    "short": "North Pole",
    "label": "North Pole 0E",
    "lon_deg_east": 0.0,
    "lat_deg": 90.0,
    "height_m": 0.0,
}

SITE_B = {
    "key": "SOUTH_POLE_0E",
    "short": "South Pole",
    "label": "South Pole 0E",
    "lon_deg_east": 0.0,
    "lat_deg": -90.0,
    "height_m": 0.0,
}

TRACK_COLORS = {
    "North Pole 0E": "#ffc861",
    "South Pole 0E": "#5ee08a",
}


def ensure_package(import_name, pip_name):
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pip_name])


for import_name, pip_name in [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("astroquery", "astroquery"),
    ("astropy", "astropy"),
]:
    ensure_package(import_name, pip_name)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar
from astroquery.jplhorizons import Horizons
import astropy.units as u
from astropy.time import Time


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


def horizons_site_location(site):
    return {"lon": site["lon_deg_east"] * u.deg, "lat": site["lat_deg"] * u.deg, "elevation": (site["height_m"] / 1000.0) * u.km}


def horizons_site_vectors(target_id, site, prefix):
    obj = Horizons(id=target_id, location=horizons_site_location(site), epochs={"start": START, "stop": STOP, "step": STEP})
    vec = obj.vectors().to_pandas()
    out = pd.DataFrame()
    out["jd_tdb"] = vec["datetime_jd"].astype(float)
    out["utc"] = vec["datetime_str"].astype(str)
    out[f"{prefix}_x_km"] = vec["x"].astype(float) * AU_KM
    out[f"{prefix}_y_km"] = vec["y"].astype(float) * AU_KM
    out[f"{prefix}_z_km"] = vec["z"].astype(float) * AU_KM
    return out


def build_geocenter_master():
    sun = horizons_geocenter_vectors("10", "GEOCENTER_SUN")
    venus = horizons_geocenter_vectors("299", "GEOCENTER_VENUS")
    return sun.merge(venus, on=["jd_tdb", "utc"], how="inner")


def build_sitecoord_master(site_a, site_b):
    frames = []
    for site in [site_a, site_b]:
        key = site["key"]
        frames.append(horizons_site_vectors("10", site, f"{key}_SUN"))
        frames.append(horizons_site_vectors("299", site, f"{key}_VENUS"))
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["jd_tdb", "utc"], how="inner")
    return merged


def build_cache(df):
    cache = {"jd_tdb": df["jd_tdb"].to_numpy(dtype=float), "utc": df["utc"].astype(str).tolist()}
    for col in df.columns:
        if col.endswith("_km"):
            cache[col] = CubicSpline(cache["jd_tdb"], df[col].to_numpy(dtype=float), bc_type="natural")
    return cache


def vec_at(cache, prefix, jd_tdb):
    return np.array([float(cache[f"{prefix}_x_km"](jd_tdb)), float(cache[f"{prefix}_y_km"](jd_tdb)), float(cache[f"{prefix}_z_km"](jd_tdb))], dtype=float)


def utc_at(jd_tdb):
    return Time(jd_tdb, format="jd", scale="tdb").utc.iso.replace(" ", " ")


def site_sun_vector(topo_cache, site, jd_tdb):
    return vec_at(topo_cache, f"{site['key']}_SUN", jd_tdb)


def site_venus_vector(topo_cache, site, jd_tdb):
    return vec_at(topo_cache, f"{site['key']}_VENUS", jd_tdb)


def site_sep_arcsec(topo_cache, site, jd_tdb):
    return angular_sep_arcsec(site_sun_vector(topo_cache, site, jd_tdb), site_venus_vector(topo_cache, site, jd_tdb))


def angular_radii_arcsec(topo_cache, site, jd_tdb):
    sun = site_sun_vector(topo_cache, site, jd_tdb)
    venus = site_venus_vector(topo_cache, site, jd_tdb)
    rs = math.atan2(SUN_RADIUS_KM, norm(sun)) * ARCSEC_PER_RAD
    rv = math.atan2(VENUS_RADIUS_KM, norm(venus)) * ARCSEC_PER_RAD
    return rs, rv


def contact_function(topo_cache, site, event, jd_tdb):
    sep = site_sep_arcsec(topo_cache, site, jd_tdb)
    rs, rv = angular_radii_arcsec(topo_cache, site, jd_tdb)
    threshold = rs + rv if event in ["C1", "C4"] else rs - rv
    return sep - threshold


def find_event_roots(topo_cache, site, event):
    jds = topo_cache["jd_tdb"]
    vals = np.array([contact_function(topo_cache, site, event, jd) for jd in jds], dtype=float)
    roots = []
    for i in range(len(jds) - 1):
        if not np.isfinite(vals[i]) or not np.isfinite(vals[i + 1]):
            continue
        if vals[i] == 0.0:
            roots.append(float(jds[i]))
        elif vals[i] * vals[i + 1] < 0.0:
            roots.append(float(brentq(lambda x: contact_function(topo_cache, site, event, x), jds[i], jds[i + 1], xtol=1e-13, rtol=1e-13, maxiter=100)))
    return sorted(roots)


def find_site_contacts(topo_cache, site):
    outer = find_event_roots(topo_cache, site, "C1")
    inner = find_event_roots(topo_cache, site, "C2")
    if len(outer) < 2 or len(inner) < 2:
        raise RuntimeError(f"Could not derive four contacts for {site['label']}.")
    return {"C1": outer[0], "C2": inner[0], "C3": inner[-1], "C4": outer[-1]}


def find_site_closest(topo_cache, site):
    jds = topo_cache["jd_tdb"]
    vals = [site_sep_arcsec(topo_cache, site, jd) for jd in jds]
    i = int(np.argmin(vals))
    a = jds[max(0, i - 3)]
    b = jds[min(len(jds) - 1, i + 3)]
    res = minimize_scalar(lambda jd: site_sep_arcsec(topo_cache, site, jd), bounds=(a, b), method="bounded", options={"xatol": 1e-13})
    return float(res.x)


def fixed_geocenter_basis(geo_cache, jd_tdb):
    sun = vec_at(geo_cache, "GEOCENTER_SUN", jd_tdb)
    n = unit(sun)
    ref = np.array([0.0, 0.0, 1.0])
    if norm(np.cross(ref, n)) < 1e-12:
        ref = np.array([1.0, 0.0, 0.0])
    xhat = unit(np.cross(ref, n))
    yhat = unit(np.cross(n, xhat))
    return n, xhat, yhat


def ray_screen_point_arcsec_sitecoord(geo_cache, topo_cache, site, jd_tdb, basis):
    n, xhat, yhat = basis
    sun_geo = vec_at(geo_cache, "GEOCENTER_SUN", jd_tdb)
    sun_topo = site_sun_vector(topo_cache, site, jd_tdb)
    venus_topo = site_venus_vector(topo_cache, site, jd_tdb)
    obs_geo = sun_geo - sun_topo
    ray = venus_topo
    denom = float(np.dot(ray, n))
    if abs(denom) < 1e-14:
        raise RuntimeError("SITE_COORD ray nearly parallel to solar screen.")
    tau = float(np.dot(sun_geo - obs_geo, n) / denom)
    hit = obs_geo + tau * ray
    screen_vec = hit - sun_geo
    es = norm(sun_geo)
    x = math.atan2(float(np.dot(screen_vec, xhat)), es) * ARCSEC_PER_RAD
    y = math.atan2(float(np.dot(screen_vec, yhat)), es) * ARCSEC_PER_RAD
    return np.array([x, y], dtype=float)


def pca_direction(points):
    pts = np.asarray(points, dtype=float)
    mu = pts.mean(axis=0)
    centered = pts - mu
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    d = vt[0]
    if d[0] < 0:
        d = -d
    return mu, unit(d)


def line_intersection(mu, d, mid, normal):
    a = np.column_stack([d, -normal])
    b = mid - mu
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    return mu + sol[0] * d


def distances_at(geo_cache, jd_tdb):
    es = norm(vec_at(geo_cache, "GEOCENTER_SUN", jd_tdb))
    ev = norm(vec_at(geo_cache, "GEOCENTER_VENUS", jd_tdb))
    vs = norm(vec_at(geo_cache, "GEOCENTER_VENUS", jd_tdb) - vec_at(geo_cache, "GEOCENTER_SUN", jd_tdb))
    return es, ev, vs


def sun_radius_arcsec(geo_cache, jd_tdb):
    es = norm(vec_at(geo_cache, "GEOCENTER_SUN", jd_tdb))
    return math.atan2(SUN_RADIUS_KM, es) * ARCSEC_PER_RAD


def site_track(geo_cache, topo_cache, site, contacts, closest_jd, basis):
    jds = topo_cache["jd_tdb"]
    mask = (jds >= contacts["C1"]) & (jds <= contacts["C4"])
    use_jds = sorted(set([contacts["C1"], contacts["C2"], closest_jd, contacts["C3"], contacts["C4"]] + list(jds[mask])))
    pts = np.array([ray_screen_point_arcsec_sitecoord(geo_cache, topo_cache, site, jd, basis) for jd in use_jds], dtype=float)
    mu, direction = pca_direction(pts)
    event_jds = {"C1": contacts["C1"], "C2": contacts["C2"], "CA": closest_jd, "C3": contacts["C3"], "C4": contacts["C4"]}
    event_pts = {name: ray_screen_point_arcsec_sitecoord(geo_cache, topo_cache, site, jd, basis) for name, jd in event_jds.items()}
    event_radii = {name: angular_radii_arcsec(topo_cache, site, jd)[1] for name, jd in event_jds.items()}
    return {"site": site, "jds": np.array(use_jds, dtype=float), "pts": pts, "mu": mu, "direction": direction, "event_jds": event_jds, "event_pts": event_pts, "event_radii": event_radii, "closest_jd": closest_jd, "closest_utc": utc_at(closest_jd), "track_angle_deg": math.degrees(math.atan2(direction[1], direction[0]))}


def compute_parallax_geometry(geo_cache, track_a, track_b, screen_jd):
    tangent = unit(track_a["direction"] + track_b["direction"])
    if tangent[0] < 0:
        tangent = -tangent
    normal = np.array([-tangent[1], tangent[0]])
    mid = 0.5 * (track_a["mu"] + track_b["mu"])
    aprime = line_intersection(track_a["mu"], track_a["direction"], mid, normal)
    bprime = line_intersection(track_b["mu"], track_b["direction"], mid, normal)
    abp_vec = bprime - aprime
    abp_arcsec = float(np.sqrt(np.sum(abp_vec * abp_vec)))
    rho_arcsec = abs(float(np.dot(abp_vec, normal)))
    es, ev, vs = distances_at(geo_cache, screen_jd)
    abp_km = math.tan(abp_arcsec / ARCSEC_PER_RAD) * es
    ab_km = abp_km * ev / vs
    ab_arcsec = math.atan2(ab_km, es) * ARCSEC_PER_RAD
    halley_ratio = abp_km / ab_km
    raw_phi = rho_arcsec * (ev / vs) * (EARTH_RADIUS_KM / ab_km)
    pi_sun = raw_phi * (es / AU_KM)
    rho_scaled = rho_arcsec * EARTH_RADIUS_KM / ab_km
    return {"aprime": aprime, "bprime": bprime, "A_prime_B_prime_arcsec": abp_arcsec, "A_prime_B_prime_km": abp_km, "rho_arcsec": rho_arcsec, "rho_scaled_arcsec": rho_scaled, "AB_arcsec": ab_arcsec, "AB_km": ab_km, "halley_ratio": halley_ratio, "raw_phi_arcsec": raw_phi, "pi_sun_arcsec": pi_sun, "pi_sun_residual_arcsec": pi_sun - PI_SUN_REFERENCE_ARCSEC, "pi_sun_residual_percent": 100.0 * (pi_sun - PI_SUN_REFERENCE_ARCSEC) / PI_SUN_REFERENCE_ARCSEC, "pi_sun_reference_arcsec": PI_SUN_REFERENCE_ARCSEC, "D_ES_AU": es / AU_KM, "D_EV_D_VS": ev / vs, "D_VS_D_EV": vs / ev, "D_ES_source": "|GEOCENTER_SUN| / AU_KM"}


def decimals_for_quantity(quantity, unit):
    if quantity in ["D ES"]:
        return 12
    if quantity in ["Computed π⊙", "Reference π⊙", "Residual π⊙", "Raw φ"]:
        return 9
    if quantity in ["A′B′ / AB", "D EV / D VS", "D VS / D EV"]:
        return 10
    if unit in ["UTC", "JPL"]:
        return None
    return 6


def fmt_value(quantity, value, unit):
    if isinstance(value, str):
        return value
    decimals = decimals_for_quantity(quantity, unit)
    if decimals is None:
        return str(value)
    return f"{float(value):.{decimals}f}"


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
    ax.annotate(text, xy=(xy[0], xy[1]), xytext=(xy[0] + dx, xy[1] + dy), textcoords="data", fontsize=5.7, color=color, ha="left", va="center", arrowprops=dict(arrowstyle="-", lw=0.20, color=color, shrinkA=0, shrinkB=2))


def trigonometry_rows(track_a, track_b):
    delta_angle = abs(track_a["track_angle_deg"] - track_b["track_angle_deg"])
    beta_average = 0.5 * (track_a["track_angle_deg"] + track_b["track_angle_deg"])
    return [("β North Pole", track_a["track_angle_deg"], "deg"), ("β South Pole", track_b["track_angle_deg"], "deg"), ("Δβ", delta_angle, "deg"), ("β Average", beta_average, "deg")]


def geometric_rows(track_a, track_b, geometry):
    return [("Closest North Pole UTC", track_a["closest_utc"], "UTC"), ("Closest South Pole UTC", track_b["closest_utc"], "UTC"), ("A′B′ Angular Chord", geometry["A_prime_B_prime_arcsec"], "arcsec"), ("A′B′ Solar-Screen Chord", geometry["A_prime_B_prime_km"], "km"), ("AB Angular Projection", geometry["AB_arcsec"], "arcsec"), ("AB Projected Baseline", geometry["AB_km"], "km"), ("A′B′ / AB", geometry["halley_ratio"], "ratio"), ("Normal Separation ρ", geometry["rho_arcsec"], "arcsec"), ("ρ Scaled To R⊕", geometry["rho_scaled_arcsec"], "arcsec"), ("D ES", geometry["D_ES_AU"], "AU"), ("D ES Source", geometry["D_ES_source"], "JPL"), ("D EV / D VS", geometry["D_EV_D_VS"], "ratio"), ("D VS / D EV", geometry["D_VS_D_EV"], "ratio"), ("Raw φ", geometry["raw_phi_arcsec"], "arcsec"), ("Computed π⊙", geometry["pi_sun_arcsec"], "arcsec"), ("Reference π⊙", geometry["pi_sun_reference_arcsec"], "arcsec"), ("Residual π⊙", geometry["pi_sun_residual_arcsec"], "arcsec"), ("Residual π⊙", geometry["pi_sun_residual_percent"], "percent")]


def add_summary_table_on_plot(ax, track_a, track_b, geometry):
    compact_rows = [("β North Pole", track_a["track_angle_deg"], "deg"), ("β South Pole", track_b["track_angle_deg"], "deg"), ("Δβ", abs(track_a["track_angle_deg"] - track_b["track_angle_deg"]), "deg"), ("π⊙", geometry["pi_sun_arcsec"], "arcsec"), ("A′B′ / AB", geometry["halley_ratio"], "ratio"), ("A′B′", geometry["A_prime_B_prime_arcsec"], "arcsec"), ("A′B′", geometry["A_prime_B_prime_km"], "km"), ("AB", geometry["AB_arcsec"], "arcsec"), ("AB", geometry["AB_km"], "km"), ("D ES", geometry["D_ES_AU"], "AU")]
    rows = [[q, fmt_value(q, v, u), u] for q, v, u in compact_rows]
    table = ax.table(cellText=rows, colLabels=["Quantity", "Value", "Unit"], loc="lower left", colWidths=[0.29, 0.23, 0.15], bbox=[0.438, 0.122, 0.380, 0.345])
    table.auto_set_font_size(False)
    table.set_fontsize(5.30)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.18)
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
    ax.text(0.440, 0.101, "A′B′ = solar-screen chord; AB = projected baseline; D ES is JPL |Sun|/AU.", transform=ax.transAxes, color="#8fb4c1", fontsize=5.25, ha="left", va="top")


def html_escape(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def html_label(text):
    return html_escape(text).replace("π⊙", "π<sub>⊙</sub>")


def html_table(headers, rows):
    head = "".join(f"<th>{html_label(h)}</th>" for h in headers)
    body = []
    for q, v, u in rows:
        body.append(f"<tr><td class='quantity-cell'>{html_label(q)}</td><td class='value-cell'>{html_escape(fmt_value(q, v, u))}</td><td class='unit-cell'>{html_escape(u)}</td></tr>")
    return f"<table class='iers-table'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def display_html_12g_style(track_a, track_b, geometry, csv_path):
    try:
        from IPython.display import HTML, display
    except Exception:
        return False
    trig = html_table(["Quantity", "Value", "Units"], trigonometry_rows(track_a, track_b))
    geom = html_table(["Quantity", "Value", "Units"], geometric_rows(track_a, track_b, geometry))
    title = "North Pole 0E → South Pole 0E — JPL HORIZONS SITE_COORD"
    html = f"""
    <style>
    .iers-wrap {{ background:#03080d; color:#e8f7ff; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; width:700px; max-width:98%; border:1px solid #1e4f64; border-radius:8px; padding:8px 8px 10px 8px; margin:8px 0 14px 0; }}
    .iers-title {{ color:#66e8ff; font-size:10px; font-weight:800; letter-spacing:0.06em; text-align:center; border-top:1px solid #25708b; border-bottom:1px solid #25708b; padding:5px 0; margin:5px 0 5px 0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .iers-title span {{ color:#f8fdff; }}
    .iers-table {{ border-collapse:collapse; width:100%; table-layout:fixed; font-size:10px; color:#dff8ff; background:#050b0f; margin:0 0 6px 0; }}
    .iers-table th {{ color:#66e8ff; background:#0a1a22; border-top:1px solid #1d3d4a; border-bottom:1px solid #1d3d4a; padding:4px 5px; text-align:left; font-weight:800; }}
    .iers-table td {{ border-bottom:1px solid #102630; padding:4px 5px; }}
    .iers-table th:nth-child(1), .iers-table td:nth-child(1) {{ width:50%; }}
    .iers-table th:nth-child(2), .iers-table td:nth-child(2) {{ width:34%; }}
    .iers-table th:nth-child(3), .iers-table td:nth-child(3) {{ width:16%; }}
    .quantity-cell {{ color:#dff8ff; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .value-cell {{ color:#ffc861; text-align:right; font-weight:800; white-space:nowrap; }}
    .unit-cell {{ color:#5ee08a; white-space:nowrap; }}
    .iers-note {{ color:#8fb4c1; font-size:9px; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    </style>
    <div class="iers-wrap"><div class="iers-title">TRIGONOMETRY — <span>{html_escape(title)}</span></div>{trig}<div class="iers-title">π<sub>⊙</sub> GEOMETRIC SOLUTION — <span>{html_escape(title)}</span></div>{geom}<div class="iers-note">CSV: {html_escape(csv_path)}</div></div>
    """
    display(HTML(html))
    return True


def write_event_csv(track_a, track_b, geometry, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([VERSION, "1769 NORTH SOUTH POLE ENGINEERING TRACK PLOT EVENT AND GEOMETRY DATA"])
        writer.writerow([])
        writer.writerow(["site", "event", "utc", "jd_tdb", "x_arcsec", "y_arcsec", "venus_radius_arcsec", "track_angle_deg"])
        for track in [track_a, track_b]:
            site = track["site"]
            for event in ["C1", "C2", "CA", "C3", "C4"]:
                jd = track["event_jds"][event]
                xy = track["event_pts"][event]
                writer.writerow([site["label"], event, utc_at(jd), f"{jd:.12f}", f"{xy[0]:.6f}", f"{xy[1]:.6f}", f"{track['event_radii'][event]:.6f}", f"{track['track_angle_deg']:.6f}"])
        writer.writerow([])
        writer.writerow(["section", "quantity", "value", "unit"])
        for row in trigonometry_rows(track_a, track_b):
            writer.writerow(["TRIGONOMETRY", row[0], fmt_value(row[0], row[1], row[2]), row[2]])
        for row in geometric_rows(track_a, track_b, geometry):
            writer.writerow(["PI_SUN_GEOMETRIC_SOLUTION", row[0], fmt_value(row[0], row[1], row[2]), row[2]])


def plot_engineering_track(geo_cache, track_a, track_b, screen_jd, geometry, png_path):
    radius = sun_radius_arcsec(geo_cache, screen_jd)
    fig, ax = plt.subplots(figsize=(9.6, 5.8), dpi=240)
    fig.patch.set_facecolor("#03080d")
    ax.set_facecolor("#03080d")
    ax.add_patch(Circle((0.0, 0.0), radius, fill=False, lw=0.36, ec="#66e8ff", alpha=0.95))
    ax.axhline(0.0, lw=0.18, color="#1d3d4a", alpha=0.72)
    ax.axvline(0.0, lw=0.18, color="#1d3d4a", alpha=0.72)
    for track in [track_a, track_b]:
        site_label = track["site"]["label"]
        color = TRACK_COLORS[site_label]
        pts = track["pts"]
        ax.plot(pts[:, 0], pts[:, 1], lw=0.30, color=color, solid_capstyle="round", label=site_label, zorder=3)
        ax.scatter(pts[::6, 0], pts[::6, 1], s=0.75, color=color, alpha=0.70, linewidths=0, zorder=4)
        for event in ["C1", "C2", "CA", "C3", "C4"]:
            xy = track["event_pts"][event]
            r = track["event_radii"][event]
            ax.add_patch(Circle((xy[0], xy[1]), r, fill=False, lw=0.20 if event != "CA" else 0.28, ec=color, alpha=0.92, zorder=2))
            ax.scatter([xy[0]], [xy[1]], s=3.8 if event == "CA" else 2.2, color=color, edgecolors="#03080d", linewidths=0.16, zorder=5)
        ca = track["event_pts"]["CA"]
        dy = 15.0 if site_label.startswith("North") else -15.0
        add_label(ax, ca, f"{track['site']['short']} CA", 18.0, dy, color)
    for event, dx, dy in [("C1", -48.0, 12.0), ("C2", -38.0, 9.0), ("C3", 20.0, -10.0), ("C4", 30.0, -13.0)]:
        add_label(ax, track_a["event_pts"][event], event, dx, dy, "#8fb4c1")
    add_summary_table_on_plot(ax, track_a, track_b, geometry)
    xlim, ylim = axis_limits_for_half_sun(radius, track_a, track_b)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_linewidth(0.22)
        spine.set_color("#25708b")
    ax.tick_params(axis="both", colors="#8fb4c1", labelsize=6.5, width=0.22, length=2.0)
    ax.grid(True, color="#102630", linewidth=0.16, alpha=0.55)
    ax.set_xlabel("Solar-screen X offset (arcsec)", color="#8fb4c1", fontsize=7.5)
    ax.set_ylabel("Solar-screen Y offset (arcsec)", color="#8fb4c1", fontsize=7.5)
    ax.set_title("1769 Venus Transit — Engineering Half-Sun Track Reconstruction\nNorth Pole 0E / South Pole 0E — JPL Horizons SITE_COORD geometry", color="#f8fdff", fontsize=9.0, pad=8)
    legend = ax.legend(loc="lower right", fontsize=6.3, frameon=True, borderpad=0.45)
    legend.get_frame().set_facecolor("#071016")
    legend.get_frame().set_edgecolor("#1e4f64")
    legend.get_frame().set_linewidth(0.22)
    for text in legend.get_texts():
        text.set_color("#dff8ff")
    note = f"Venus disks are plotted to scale at C1, C2, closest approach, C3, and C4.  North Pole CA: {track_a['closest_utc']}   South Pole CA: {track_b['closest_utc']}"
    fig.text(0.5, 0.016, note, ha="center", va="bottom", fontsize=6.2, color="#8fb4c1")
    fig.savefig(png_path, dpi=460, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.055)
    plt.show()
    plt.close(fig)


def print_plain_12g_style(track_a, track_b, geometry, csv_path):
    print("TRIGONOMETRY — North Pole 0E → South Pole 0E — JPL HORIZONS SITE_COORD")
    print("Quantity | Value | Units")
    for q, v, u in trigonometry_rows(track_a, track_b):
        print(f"{q} | {fmt_value(q, v, u)} | {u}")
    print()
    print("π⊙ GEOMETRIC SOLUTION — North Pole 0E → South Pole 0E — JPL HORIZONS SITE_COORD")
    print("Quantity | Value | Units")
    for q, v, u in geometric_rows(track_a, track_b, geometry):
        print(f"{q} | {fmt_value(q, v, u)} | {u}")
    print()
    print(f"CSV: {csv_path}")
    print()


def main():
    print(f"CODE OUTPUT: {VERSION}")
    print(f"PROGRAM: {PROGRAM_NAME}")
    print()
    print("RUN PAIR: North Pole 0E → South Pole 0E")
    print(f"TIME RANGE: {START} TO {STOP} STEP {STEP}")
    print("SOURCE DATA: JPL Horizons geocenter vectors and JPL Horizons SITE_COORD topocentric vectors")
    print()
    out_dir = "/content/IERS_TN36_V01_MASTER_OUTPUT"
    os.makedirs(out_dir, exist_ok=True)
    geo_df = build_geocenter_master()
    topo_df = build_sitecoord_master(SITE_A, SITE_B)
    geo_cache = build_cache(geo_df)
    topo_cache = build_cache(topo_df)
    contacts_a = find_site_contacts(topo_cache, SITE_A)
    contacts_b = find_site_contacts(topo_cache, SITE_B)
    closest_a = find_site_closest(topo_cache, SITE_A)
    closest_b = find_site_closest(topo_cache, SITE_B)
    screen_jd = 0.5 * (closest_a + closest_b)
    basis = fixed_geocenter_basis(geo_cache, screen_jd)
    track_a = site_track(geo_cache, topo_cache, SITE_A, contacts_a, closest_a, basis)
    track_b = site_track(geo_cache, topo_cache, SITE_B, contacts_b, closest_b, basis)
    geometry = compute_parallax_geometry(geo_cache, track_a, track_b, screen_jd)
    png_path = os.path.join(out_dir, f"{VERSION}_NORTH_SOUTH_POLE_ENGINEERING_HALF_SUN_TRACKS.png")
    csv_path = os.path.join(out_dir, f"{VERSION}_NORTH_SOUTH_POLE_ENGINEERING_HALF_SUN_EVENTS_AND_GEOMETRY.csv")
    write_event_csv(track_a, track_b, geometry, csv_path)
    plot_engineering_track(geo_cache, track_a, track_b, screen_jd, geometry, png_path)
    rendered = display_html_12g_style(track_a, track_b, geometry, csv_path)
    if not rendered:
        print_plain_12g_style(track_a, track_b, geometry, csv_path)
    print("RESULTS")
    print(f"D ES source            : {geometry['D_ES_source']}")
    print(f"D ES                   : {geometry['D_ES_AU']:.12f} AU")
    print(f"A prime B prime / AB   : {geometry['halley_ratio']:.10f}")
    print(f"North Pole closest UTC : {track_a['closest_utc']}")
    print(f"South Pole closest UTC : {track_b['closest_utc']}")
    print(f"North Pole track angle : {track_a['track_angle_deg']:.6f} deg")
    print(f"South Pole angle       : {track_b['track_angle_deg']:.6f} deg")
    print(f"Track angle delta abs  : {abs(track_a['track_angle_deg'] - track_b['track_angle_deg']):.6f} deg")
    print(f"A prime B prime        : {geometry['A_prime_B_prime_arcsec']:.6f} arcsec")
    print(f"A prime B prime        : {geometry['A_prime_B_prime_km']:.6f} km")
    print(f"AB                     : {geometry['AB_arcsec']:.6f} arcsec")
    print(f"AB                     : {geometry['AB_km']:.6f} km")
    print(f"rho                    : {geometry['rho_arcsec']:.6f} arcsec")
    print(f"Pi sun                 : {geometry['pi_sun_arcsec']:.9f} arcsec")
    print(f"Pi sun residual        : {geometry['pi_sun_residual_arcsec']:.9f} arcsec")
    print(f"PNG output             : {png_path}")
    print(f"CSV output             : {csv_path}")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# IERS-0012O
