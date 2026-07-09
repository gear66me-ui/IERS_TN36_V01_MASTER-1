# IERS-0012D
# Audit reference: GitHubDelivery@IERS-0012D; based on IERS-0012C with report-only cleanup.

import os
import sys
import math
import csv
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

VERSION = "IERS-0012D"
PROGRAM_NAME = "IERS_0012D_ANTIPODAL_CITY_PAIR_TRIG_FIRST_PI_SUN.py"

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

SITE_A = {
    "key": "WELLINGTON_NZ",
    "label": "Wellington NZ",
    "lon_deg_east": 174.77557,
    "lat_deg": -41.28664,
    "height_m": 0.0,
}

SITE_B = {
    "key": "ALAEJOS_SPAIN",
    "label": "Alaejos Spain",
    "lon_deg_east": -5.22443,
    "lat_deg": 41.28664,
    "height_m": 0.0,
}

PAIR_RUNS = [
    {
        "title": "Wellington NZ → Alaejos Spain",
        "start": "2012-Jun-05 20:00",
        "stop": "2012-Jun-06 07:30",
        "step": "1m",
        "site_a": {
            "key": "WELLINGTON_NZ",
            "label": "Wellington NZ",
            "lon_deg_east": 174.77557,
            "lat_deg": -41.28664,
            "height_m": 0.0,
        },
        "site_b": {
            "key": "ALAEJOS_SPAIN",
            "label": "Alaejos Spain",
            "lon_deg_east": -5.22443,
            "lat_deg": 41.28664,
            "height_m": 0.0,
        },
    },
]


def ensure_package(import_name, pip_name):
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pip_name])


ensure_package("numpy", "numpy")
ensure_package("pandas", "pandas")
ensure_package("scipy", "scipy")
ensure_package("astroquery", "astroquery")
ensure_package("astropy", "astropy")

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar, brentq
from astroquery.jplhorizons import Horizons
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation
from astropy.utils import iers

try:
    from IPython.display import display, HTML
except Exception:
    display = None
    HTML = None

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
    obj = Horizons(
        id=target_id,
        location="500@399",
        epochs={"start": START, "stop": STOP, "step": STEP},
    )
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
    return np.array([
        float(cache[f"{prefix}_x_km"](jd_tdb)),
        float(cache[f"{prefix}_y_km"](jd_tdb)),
        float(cache[f"{prefix}_z_km"](jd_tdb)),
    ], dtype=float)


def utc_at(cache, jd_tdb):
    try:
        return Time(jd_tdb, format="jd", scale="tdb").utc.iso.replace(" ", " ")
    except Exception:
        jds = cache["jd_tdb"]
        idx = int(np.argmin(np.abs(jds - jd_tdb)))
        return cache["utc"][idx]


def earth_location(site):
    return EarthLocation.from_geodetic(
        lon=site["lon_deg_east"] * u.deg,
        lat=site["lat_deg"] * u.deg,
        height=site["height_m"] * u.m,
    )


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


def geocenter_sep_arcsec(cache, jd_tdb):
    return angular_sep_arcsec(vec_at(cache, "GEOCENTER_SUN", jd_tdb), vec_at(cache, "GEOCENTER_VENUS", jd_tdb))


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


def find_roots(cache, site, event):
    jds = cache["jd_tdb"]
    vals = np.array([contact_function(cache, site, event, jd) for jd in jds], dtype=float)
    roots = []
    for i in range(len(jds) - 1):
        if not np.isfinite(vals[i]) or not np.isfinite(vals[i + 1]):
            continue
        if vals[i] == 0.0:
            roots.append(float(jds[i]))
        elif vals[i] * vals[i + 1] < 0.0:
            roots.append(float(brentq(
                lambda x: contact_function(cache, site, event, x),
                jds[i],
                jds[i + 1],
                xtol=1e-13,
                rtol=1e-13,
                maxiter=100,
            )))
    return roots


def contact_range(cache):
    roots = []
    for site in [SITE_A, SITE_B]:
        roots.extend(find_roots(cache, site, "C1"))
        roots.extend(find_roots(cache, site, "C4"))
    if not roots:
        raise RuntimeError("No C1/C4 contacts found.")
    return min(roots), max(roots)


def find_closest(cache):
    jds = cache["jd_tdb"]
    vals = [geocenter_sep_arcsec(cache, jd) for jd in jds]
    i = int(np.argmin(vals))
    a = jds[max(0, i - 3)]
    b = jds[min(len(jds) - 1, i + 3)]
    res = minimize_scalar(
        lambda jd: geocenter_sep_arcsec(cache, jd),
        bounds=(a, b),
        method="bounded",
        options={"xatol": 1e-13},
    )
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
    screen_point = sun_geo
    denom = float(np.dot(ray, n))
    if abs(denom) < 1e-14:
        raise RuntimeError("Ray nearly parallel to solar screen.")
    tau = float(np.dot(screen_point - obs, n) / denom)
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


def fit_r2(points, degree):
    pts = np.asarray(points, dtype=float)
    x = pts[:, 0] - np.mean(pts[:, 0])
    y = pts[:, 1]
    coeff = np.polyfit(x, y, degree)
    yhat = np.polyval(coeff, x)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def distances_at(cache, jd_tdb):
    es = norm(vec_at(cache, "GEOCENTER_SUN", jd_tdb))
    ev = norm(vec_at(cache, "GEOCENTER_VENUS", jd_tdb))
    vs = norm(vec_at(cache, "GEOCENTER_VENUS", jd_tdb) - vec_at(cache, "GEOCENTER_SUN", jd_tdb))
    return es, ev, vs


def compute_iers_guideline_method(cache):
    ca_jd = find_closest(cache)
    c1, c4 = contact_range(cache)
    use_jds = cache["jd_tdb"][(cache["jd_tdb"] >= c1) & (cache["jd_tdb"] <= c4)]
    basis = fixed_geocenter_basis(cache, ca_jd)

    mus = {}
    dirs = {}
    track_rows = []

    for site in [SITE_A, SITE_B]:
        pts = np.array([ray_screen_point_arcsec(cache, site, jd, basis) for jd in use_jds], dtype=float)
        mu, d = pca_direction(pts)
        mus[site["key"]] = mu
        dirs[site["key"]] = d
        angle = math.degrees(math.atan2(d[1], d[0]))
        track_rows.append({
            "track": site["label"],
            "lat": site["lat_deg"],
            "lon": site["lon_deg_east"],
            "angle_deg": angle,
            "linear_R2": fit_r2(pts, 1),
            "quad_R2": fit_r2(pts, 2),
            "cubic_R2": fit_r2(pts, 3),
            "parabolic_R2": fit_r2(pts, 2),
        })

    tangent = unit(dirs[SITE_A["key"]] + dirs[SITE_B["key"]])
    if tangent[0] < 0:
        tangent = -tangent
    normal = np.array([-tangent[1], tangent[0]])
    common_angle = math.degrees(math.atan2(tangent[1], tangent[0]))

    mid = 0.5 * (mus[SITE_A["key"]] + mus[SITE_B["key"]])
    aprime = line_intersection(mus[SITE_A["key"]], dirs[SITE_A["key"]], mid, normal)
    bprime = line_intersection(mus[SITE_B["key"]], dirs[SITE_B["key"]], mid, normal)

    abp_vec = bprime - aprime
    abp_arcsec = float(np.sqrt(np.sum(abp_vec * abp_vec)))
    sep_arcsec = abs(float(np.dot(abp_vec, normal)))

    es, ev, vs = distances_at(cache, ca_jd)
    abp_km = math.tan(abp_arcsec / ARCSEC_PER_RAD) * es
    baseline_proj_km = abp_km * ev / vs
    raw_phi = sep_arcsec * (ev / vs) * (EARTH_RADIUS_KM / baseline_proj_km)
    pi_sun_phi = raw_phi * (es / AU_KM)
    delta = pi_sun_phi - PI_SUN_REFERENCE_ARCSEC
    theta_mean = 0.5 * (track_rows[0]["angle_deg"] + track_rows[1]["angle_deg"])
    delta_percent = 100.0 * delta / PI_SUN_REFERENCE_ARCSEC
    rho_R_earth_arcsec = sep_arcsec * EARTH_RADIUS_KM / baseline_proj_km

    parallax = {
        "baseline_proj_km": baseline_proj_km,
        "A_prime_B_prime_arcsec": abp_arcsec,
        "A_prime_B_prime_km": abp_km,
        "sep_arcsec": sep_arcsec,
        "rho_R_earth_arcsec": rho_R_earth_arcsec,
        "raw_phi_arcsec": raw_phi,
        "pi_sun_phi_arcsec": pi_sun_phi,
        "pi_sun_delta_arcsec": delta,
        "pi_sun_delta_percent": delta_percent,
        "common_angle_deg": common_angle,
        "theta_mean_deg": theta_mean,
        "D_ES_AU": es / AU_KM,
        "D_EV_D_VS": ev / vs,
        "R_EARTH_KM": EARTH_RADIUS_KM,
        "AU_KM": AU_KM,
        "PI_SUN_TARGET_ARCSEC": PI_SUN_REFERENCE_ARCSEC,
        "CA_utc": utc_at(cache, ca_jd),
        "CA_jd_tdb": ca_jd,
    }
    return parallax, track_rows


def horizons_site_location(site):
    return {
        "lon": site["lon_deg_east"] * u.deg,
        "lat": site["lat_deg"] * u.deg,
        "elevation": (site["height_m"] / 1000.0) * u.km,
    }


def horizons_site_vectors(target_id, site, prefix):
    obj = Horizons(
        id=target_id,
        location=horizons_site_location(site),
        epochs={"start": START, "stop": STOP, "step": STEP},
    )
    vec = obj.vectors().to_pandas()
    out = pd.DataFrame()
    out["jd_tdb"] = vec["datetime_jd"].astype(float)
    out["utc"] = vec["datetime_str"].astype(str)
    out[f"{prefix}_x_km"] = vec["x"].astype(float) * AU_KM
    out[f"{prefix}_y_km"] = vec["y"].astype(float) * AU_KM
    out[f"{prefix}_z_km"] = vec["z"].astype(float) * AU_KM
    return out


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


def ray_screen_point_arcsec_sitecoord(geo_cache, topo_cache, site, jd_tdb, basis):
    n, xhat, yhat = basis
    key = site["key"]
    sun_geo = vec_at(geo_cache, "GEOCENTER_SUN", jd_tdb)
    sun_topo = vec_at(topo_cache, f"{key}_SUN", jd_tdb)
    venus_topo = vec_at(topo_cache, f"{key}_VENUS", jd_tdb)
    obs = sun_geo - sun_topo
    ray = venus_topo
    denom = float(np.dot(ray, n))
    if abs(denom) < 1e-14:
        raise RuntimeError("SITE_COORD ray nearly parallel to solar screen.")
    tau = float(np.dot(sun_geo - obs, n) / denom)
    hit = obs + tau * ray
    screen_vec = hit - sun_geo
    es = norm(sun_geo)
    x = math.atan2(float(np.dot(screen_vec, xhat)), es) * ARCSEC_PER_RAD
    y = math.atan2(float(np.dot(screen_vec, yhat)), es) * ARCSEC_PER_RAD
    return np.array([x, y], dtype=float)


def compute_iers_guideline_method_sitecoord(geo_cache, topo_cache):
    ca_jd = find_closest(geo_cache)
    c1, c4 = contact_range(geo_cache)
    use_jds = geo_cache["jd_tdb"][(geo_cache["jd_tdb"] >= c1) & (geo_cache["jd_tdb"] <= c4)]
    basis = fixed_geocenter_basis(geo_cache, ca_jd)

    mus = {}
    dirs = {}
    track_rows = []

    for site in [SITE_A, SITE_B]:
        pts = np.array([ray_screen_point_arcsec_sitecoord(geo_cache, topo_cache, site, jd, basis) for jd in use_jds], dtype=float)
        mu, d = pca_direction(pts)
        mus[site["key"]] = mu
        dirs[site["key"]] = d
        angle = math.degrees(math.atan2(d[1], d[0]))
        track_rows.append({
            "track": site["label"],
            "lat": site["lat_deg"],
            "lon": site["lon_deg_east"],
            "angle_deg": angle,
            "linear_R2": fit_r2(pts, 1),
            "quad_R2": fit_r2(pts, 2),
            "cubic_R2": fit_r2(pts, 3),
            "parabolic_R2": fit_r2(pts, 2),
        })

    tangent = unit(dirs[SITE_A["key"]] + dirs[SITE_B["key"]])
    if tangent[0] < 0:
        tangent = -tangent
    normal = np.array([-tangent[1], tangent[0]])
    common_angle = math.degrees(math.atan2(tangent[1], tangent[0]))

    mid = 0.5 * (mus[SITE_A["key"]] + mus[SITE_B["key"]])
    aprime = line_intersection(mus[SITE_A["key"]], dirs[SITE_A["key"]], mid, normal)
    bprime = line_intersection(mus[SITE_B["key"]], dirs[SITE_B["key"]], mid, normal)

    abp_vec = bprime - aprime
    abp_arcsec = float(np.sqrt(np.sum(abp_vec * abp_vec)))
    sep_arcsec = abs(float(np.dot(abp_vec, normal)))

    es, ev, vs = distances_at(geo_cache, ca_jd)
    abp_km = math.tan(abp_arcsec / ARCSEC_PER_RAD) * es
    baseline_proj_km = abp_km * ev / vs
    raw_phi = sep_arcsec * (ev / vs) * (EARTH_RADIUS_KM / baseline_proj_km)
    pi_sun_phi = raw_phi * (es / AU_KM)
    delta = pi_sun_phi - PI_SUN_REFERENCE_ARCSEC
    delta_percent = 100.0 * delta / PI_SUN_REFERENCE_ARCSEC
    theta_mean = 0.5 * (track_rows[0]["angle_deg"] + track_rows[1]["angle_deg"])
    rho_R_earth_arcsec = sep_arcsec * EARTH_RADIUS_KM / baseline_proj_km

    return {
        "baseline_proj_km": baseline_proj_km,
        "A_prime_B_prime_arcsec": abp_arcsec,
        "A_prime_B_prime_km": abp_km,
        "sep_arcsec": sep_arcsec,
        "rho_R_earth_arcsec": rho_R_earth_arcsec,
        "raw_phi_arcsec": raw_phi,
        "pi_sun_phi_arcsec": pi_sun_phi,
        "pi_sun_delta_arcsec": delta,
        "pi_sun_delta_percent": delta_percent,
        "common_angle_deg": common_angle,
        "theta_mean_deg": theta_mean,
        "D_ES_AU": es / AU_KM,
        "D_EV_D_VS": ev / vs,
        "R_EARTH_KM": EARTH_RADIUS_KM,
        "AU_KM": AU_KM,
        "PI_SUN_TARGET_ARCSEC": PI_SUN_REFERENCE_ARCSEC,
        "CA_utc": utc_at(geo_cache, ca_jd),
        "CA_jd_tdb": ca_jd,
    }, track_rows


def fmt(value, decimals=6):
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "nan"
        return f"{float(value):.{decimals}f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def build_report_rows(parallax, track_rows):
    trigonometry_rows = [
        ("closest approach UTC", parallax["CA_utc"], "UTC"),
        ("closest approach JD_TDB", parallax["CA_jd_tdb"], "d"),
        ("mean track angle θ", parallax["theta_mean_deg"], "deg"),
        ("common tangent angle θ", parallax["common_angle_deg"], "deg"),
        ("A′B′ angular chord", parallax["A_prime_B_prime_arcsec"], "arcsec"),
        ("A′B′ solar-screen chord", parallax["A_prime_B_prime_km"], "km"),
        ("normal separation ρ", parallax["sep_arcsec"], "arcsec"),
        ("baseline projection", parallax["baseline_proj_km"], "km"),
        ("D_ES", parallax["D_ES_AU"], "AU"),
        ("D_EV / D_VS", parallax["D_EV_D_VS"], "ratio"),
        ("ρ scaled to R⊕", parallax["rho_R_earth_arcsec"], "arcsec"),
    ]
    track_rows_out = []
    for row in track_rows:
        track_rows_out.append((
            row["track"],
            row["lat"],
            row["lon"],
            row["angle_deg"],
            row["linear_R2"],
            row["quad_R2"],
            row["cubic_R2"],
        ))
    solution_rows = [
        ("raw φ", parallax["raw_phi_arcsec"], "arcsec"),
        ("π⊙ computed", parallax["pi_sun_phi_arcsec"], "arcsec"),
        ("π⊙ reference", parallax["PI_SUN_TARGET_ARCSEC"], "arcsec"),
        ("π⊙ residual", parallax["pi_sun_delta_arcsec"], "arcsec"),
        ("π⊙ residual", parallax["pi_sun_delta_percent"], "percent"),
        ("R⊕", parallax["R_EARTH_KM"], "km"),
        ("AU", parallax["AU_KM"], "km"),
    ]
    return trigonometry_rows, track_rows_out, solution_rows


def html_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def html_table(headers, rows, numeric_start=1):
    out = ["<table class='iers-table'>"]
    out.append("<thead><tr>" + "".join(f"<th>{html_escape(h)}</th>" for h in headers) + "</tr></thead>")
    out.append("<tbody>")
    for row in rows:
        out.append("<tr>")
        for i, cell in enumerate(row):
            text = fmt(cell, 6) if i >= numeric_start else str(cell)
            out.append(f"<td>{html_escape(text)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def display_html_report(parallax, track_rows, csv_path, title):
    if display is None or HTML is None:
        return False

    trig_rows, track_rows_out, solution_rows = build_report_rows(parallax, track_rows)
    html = f"""
    <style>
    .iers-wrap {{
        background:#000000;
        color:#ff4040;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
        padding: 12px 14px;
        width: 860px;
        border: 1px solid #666;
        margin: 12px 0 18px 0;
    }}
    .iers-title {{
        color:#ffffff;
        font-weight:700;
        letter-spacing:0.05em;
        text-align:center;
        border-top:1px solid #aaa;
        border-bottom:1px solid #aaa;
        padding:7px 0;
        margin-bottom:11px;
    }}
    .iers-subtitle {{
        color:#ff4040;
        font-weight:700;
        margin: 12px 0 5px 0;
        letter-spacing:0.03em;
    }}
    .iers-table {{
        border-collapse:collapse;
        width:100%;
        font-size:12px;
        color:#ff4040;
        margin: 0 0 10px 0;
    }}
    .iers-table th {{
        border:1px solid #555;
        padding:4px 6px;
        color:#ff4040;
        background:#101010;
        text-align:left;
    }}
    .iers-table td {{
        border:1px solid #333;
        padding:4px 6px;
        color:#ff4040;
        background:#000000;
    }}
    .iers-note {{
        color:#ff4040;
        font-size:12px;
        margin-top:8px;
    }}
    </style>
    <div class="iers-wrap">
      <div class="iers-title">{html_escape(VERSION)} — {html_escape(title)}</div>

      <div class="iers-subtitle">TRIGONOMETRY TABLE</div>
      {html_table(["quantity", "value", "unit"], trig_rows, numeric_start=1)}

      <div class="iers-subtitle">TRACK FIT TABLE</div>
      {html_table(["track", "lat_deg", "lon_deg", "angle_deg", "linear_R2", "quad_R2", "cubic_R2"], track_rows_out, numeric_start=1)}

      <div class="iers-subtitle">π⊙ SOLUTION TABLE</div>
      {html_table(["quantity", "value", "unit"], solution_rows, numeric_start=1)}

      <div class="iers-note">CSV: {html_escape(csv_path)}</div>
    </div>
    """
    display(HTML(html))
    return True


def print_plain_section(title, rows):
    print(title)
    print("-" * len(title))
    for row in rows:
        print(" | ".join(fmt(x, 6) for x in row))
    print()


def print_plain_track_section(track_rows_out):
    print_plain_section(
        "TRACK FIT TABLE",
        [("track", "lat_deg", "lon_deg", "angle_deg", "linear_R2", "quad_R2", "cubic_R2")] + track_rows_out,
    )


def write_csv(parallax, track_rows, csv_path):
    trig_rows, track_rows_out, solution_rows = build_report_rows(parallax, track_rows)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([VERSION, "CLEANED ENGINEERING REPORT"])
        writer.writerow([])
        writer.writerow(["TRIGONOMETRY TABLE"])
        writer.writerow(["quantity", "value", "unit"])
        for row in trig_rows:
            writer.writerow(row)
        writer.writerow([])
        writer.writerow(["TRACK FIT TABLE"])
        writer.writerow(["track", "lat_deg", "lon_deg", "angle_deg", "linear_R2", "quad_R2", "cubic_R2"])
        for row in track_rows_out:
            writer.writerow(row)
        writer.writerow([])
        writer.writerow(["π⊙ SOLUTION TABLE"])
        writer.writerow(["quantity", "value", "unit"])
        for row in solution_rows:
            writer.writerow(row)


def safe_title_from_pair_title(title):
    return (
        title
        .replace(" ", "_")
        .replace("→", "TO")
        .replace("ø", "o")
        .replace("Ø", "O")
        .replace("í", "i")
        .replace("Í", "I")
        .replace("/", "_")
    )


def run_pair(pair, out_dir):
    global START, STOP, STEP, SITE_A, SITE_B

    START = pair["start"]
    STOP = pair["stop"]
    STEP = pair["step"]
    SITE_A = pair["site_a"]
    SITE_B = pair["site_b"]

    print(f"RUN PAIR: {pair['title']}")
    print(f"TIME RANGE: {START} TO {STOP} STEP {STEP}")
    print("SOURCE DATA: JPL Horizons vectors and Astropy/ERFA Earth-orientation transform")
    print()

    safe_title = safe_title_from_pair_title(pair["title"])

    geo_df = build_master_geocenter()
    geo_cache = build_cache(geo_df)

    parallax, track_rows = compute_iers_guideline_method(geo_cache)
    iers_csv = os.path.join(out_dir, f"{VERSION}_{safe_title}_PI_SUN_IERS_GCRS_REPORT.csv")
    write_csv(parallax, track_rows, iers_csv)
    rendered = display_html_report(parallax, track_rows, iers_csv, pair["title"] + " — IERS GCRS")
    if not rendered:
        trig_rows, track_rows_out, solution_rows = build_report_rows(parallax, track_rows)
        print_plain_section("TRIGONOMETRY TABLE", [("quantity", "value", "unit")] + trig_rows)
        print_plain_track_section(track_rows_out)
        print_plain_section("π⊙ SOLUTION TABLE", [("quantity", "value", "unit")] + solution_rows)
        print(f"CSV π⊙ IERS GCRS: {iers_csv}")

    topo_df = build_sitecoord_master(SITE_A, SITE_B)
    topo_cache = build_cache(topo_df)
    sitecoord_parallax, sitecoord_track_rows = compute_iers_guideline_method_sitecoord(geo_cache, topo_cache)
    sitecoord_csv = os.path.join(out_dir, f"{VERSION}_{safe_title}_PI_SUN_SITE_COORD_REPORT.csv")
    write_csv(sitecoord_parallax, sitecoord_track_rows, sitecoord_csv)
    rendered = display_html_report(sitecoord_parallax, sitecoord_track_rows, sitecoord_csv, pair["title"] + " — HORIZONS SITE_COORD")
    if not rendered:
        trig_rows, track_rows_out, solution_rows = build_report_rows(sitecoord_parallax, sitecoord_track_rows)
        print_plain_section("TRIGONOMETRY TABLE", [("quantity", "value", "unit")] + trig_rows)
        print_plain_track_section(track_rows_out)
        print_plain_section("π⊙ SOLUTION TABLE", [("quantity", "value", "unit")] + solution_rows)
        print(f"CSV π⊙ SITE_COORD: {sitecoord_csv}")


def main():
    print(f"CODE OUTPUT: {VERSION}")
    print(f"PROGRAM: {PROGRAM_NAME}")
    print()

    out_dir = "/content/IERS_TN36_V01_MASTER_OUTPUT"
    os.makedirs(out_dir, exist_ok=True)

    for pair in PAIR_RUNS:
        run_pair(pair, out_dir)

    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# IERS-0012D
