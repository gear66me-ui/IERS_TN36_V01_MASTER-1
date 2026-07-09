# IERS-0012G
# Audit reference: GitHubDelivery@IERS-0012G; 12D narrow block table restored with E accent colors and beta-only trigonometry block.

import os
import sys
import math
import csv
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

VERSION = "IERS-0012G"
PROGRAM_NAME = "IERS_0012G_ANTIPODAL_CITY_PAIR_NARROW_PI_SUN.py"

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
    "short": "Wellington",
    "label": "Wellington NZ",
    "lon_deg_east": 174.77557,
    "lat_deg": -41.28664,
    "height_m": 0.0,
}

SITE_B = {
    "key": "ALAEJOS_SPAIN",
    "short": "Alaejos",
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
        "site_a": SITE_A,
        "site_b": SITE_B,
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
    return np.array(
        [
            float(cache[f"{prefix}_x_km"](jd_tdb)),
            float(cache[f"{prefix}_y_km"](jd_tdb)),
            float(cache[f"{prefix}_z_km"](jd_tdb)),
        ],
        dtype=float,
    )


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
            roots.append(
                float(
                    brentq(
                        lambda x: contact_function(cache, site, event, x),
                        jds[i],
                        jds[i + 1],
                        xtol=1e-13,
                        rtol=1e-13,
                        maxiter=100,
                    )
                )
            )
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
        track_rows.append({"track": site["label"], "short": site["short"], "angle_deg": angle})

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
    delta_percent = 100.0 * delta / PI_SUN_REFERENCE_ARCSEC
    theta_mean = 0.5 * (track_rows[0]["angle_deg"] + track_rows[1]["angle_deg"])
    delta_beta_abs = abs(track_rows[0]["angle_deg"] - track_rows[1]["angle_deg"])
    rho_R_earth_arcsec = sep_arcsec * EARTH_RADIUS_KM / baseline_proj_km

    parallax = {
        "A_prime_B_prime_arcsec": abp_arcsec,
        "A_prime_B_prime_km": abp_km,
        "baseline_proj_km": baseline_proj_km,
        "sep_arcsec": sep_arcsec,
        "rho_R_earth_arcsec": rho_R_earth_arcsec,
        "raw_phi_arcsec": raw_phi,
        "pi_sun_phi_arcsec": pi_sun_phi,
        "pi_sun_delta_arcsec": delta,
        "pi_sun_delta_percent": delta_percent,
        "common_angle_deg": common_angle,
        "theta_mean_deg": theta_mean,
        "delta_beta_abs_deg": delta_beta_abs,
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
        track_rows.append({"track": site["label"], "short": site["short"], "angle_deg": angle})
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
    delta_beta_abs = abs(track_rows[0]["angle_deg"] - track_rows[1]["angle_deg"])
    rho_R_earth_arcsec = sep_arcsec * EARTH_RADIUS_KM / baseline_proj_km
    return {
        "A_prime_B_prime_arcsec": abp_arcsec,
        "A_prime_B_prime_km": abp_km,
        "baseline_proj_km": baseline_proj_km,
        "sep_arcsec": sep_arcsec,
        "rho_R_earth_arcsec": rho_R_earth_arcsec,
        "raw_phi_arcsec": raw_phi,
        "pi_sun_phi_arcsec": pi_sun_phi,
        "pi_sun_delta_arcsec": delta,
        "pi_sun_delta_percent": delta_percent,
        "common_angle_deg": common_angle,
        "theta_mean_deg": theta_mean,
        "delta_beta_abs_deg": delta_beta_abs,
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


def html_escape(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def html_label(text):
    return html_escape(text).replace("π⊙", "π<sub>⊙</sub>")


def build_trig_rows(parallax, track_rows):
    return [
        (f"β {track_rows[0]['short']}", track_rows[0]["angle_deg"], "deg"),
        (f"β {track_rows[1]['short']}", track_rows[1]["angle_deg"], "deg"),
        ("Δβ", parallax["delta_beta_abs_deg"], "deg"),
        ("β Average", parallax["theta_mean_deg"], "deg"),
    ]


def build_solution_rows(parallax):
    return [
        ("Closest Approach UTC", parallax["CA_utc"], "UTC"),
        ("A′B′ Angular Chord", parallax["A_prime_B_prime_arcsec"], "arcsec"),
        ("A′B′ Solar-Screen Chord", parallax["A_prime_B_prime_km"], "km"),
        ("Projected Baseline", parallax["baseline_proj_km"], "km"),
        ("Normal Separation ρ", parallax["sep_arcsec"], "arcsec"),
        ("ρ Scaled To R⊕", parallax["rho_R_earth_arcsec"], "arcsec"),
        ("D ES", parallax["D_ES_AU"], "AU"),
        ("D EV / D VS", parallax["D_EV_D_VS"], "ratio"),
        ("Raw φ", parallax["raw_phi_arcsec"], "arcsec"),
        ("Computed π⊙", parallax["pi_sun_phi_arcsec"], "arcsec"),
        ("Reference π⊙", parallax["PI_SUN_TARGET_ARCSEC"], "arcsec"),
        ("Residual π⊙", parallax["pi_sun_delta_arcsec"], "arcsec"),
        ("Residual π⊙", parallax["pi_sun_delta_percent"], "percent"),
    ]


def html_three_column_table(headers, rows):
    out = ["<table class='iers-table'>"]
    out.append("<thead><tr>")
    for header in headers:
        out.append(f"<th>{html_label(header)}</th>")
    out.append("</tr></thead><tbody>")
    for quantity, value, unit in rows:
        out.append("<tr>")
        out.append(f"<td class='quantity-cell'>{html_label(quantity)}</td>")
        out.append(f"<td class='value-cell'>{html_escape(fmt(value, 6))}</td>")
        out.append(f"<td class='unit-cell'>{html_escape(unit)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def display_html_report(parallax, track_rows, csv_path, title):
    if display is None or HTML is None:
        return False
    trig_rows = build_trig_rows(parallax, track_rows)
    solution_rows = build_solution_rows(parallax)
    html = f"""
    <style>
    .iers-wrap {{ background:#03080d; color:#e8f7ff; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; width:680px; max-width:98%; border:1px solid #1e4f64; border-radius:8px; padding:8px 8px 10px 8px; margin:8px 0 14px 0; }}
    .iers-title {{ color:#66e8ff; font-size:10px; font-weight:800; letter-spacing:0.06em; text-align:center; border-top:1px solid #25708b; border-bottom:1px solid #25708b; padding:5px 0; margin:5px 0 5px 0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .iers-title span {{ color:#f8fdff; }}
    .iers-table {{ border-collapse:collapse; width:100%; table-layout:fixed; font-size:10px; color:#dff8ff; background:#050b0f; margin:0 0 6px 0; }}
    .iers-table th {{ color:#66e8ff; background:#0a1a22; border-top:1px solid #1d3d4a; border-bottom:1px solid #1d3d4a; padding:4px 5px; text-align:left; font-weight:800; }}
    .iers-table td {{ border-bottom:1px solid #102630; padding:4px 5px; }}
    .iers-table th:nth-child(1), .iers-table td:nth-child(1) {{ width:52%; }}
    .iers-table th:nth-child(2), .iers-table td:nth-child(2) {{ width:31%; }}
    .iers-table th:nth-child(3), .iers-table td:nth-child(3) {{ width:17%; }}
    .quantity-cell {{ color:#dff8ff; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .value-cell {{ color:#ffc861; text-align:right; font-weight:800; white-space:nowrap; }}
    .unit-cell {{ color:#5ee08a; white-space:nowrap; }}
    .iers-note {{ color:#8fb4c1; font-size:9px; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    </style>
    <div class="iers-wrap">
      <div class="iers-title">TRIGONOMETRY — <span>{html_escape(title)}</span></div>
      {html_three_column_table(["Quantity", "Value", "Units"], trig_rows)}
      <div class="iers-title">π<sub>⊙</sub> GEOMETRIC SOLUTION — <span>{html_escape(title)}</span></div>
      {html_three_column_table(["Quantity", "Value", "Units"], solution_rows)}
      <div class="iers-note">CSV: {html_escape(csv_path)}</div>
    </div>
    """
    display(HTML(html))
    return True


def print_plain_report(parallax, track_rows, csv_path, title):
    print(f"TRIGONOMETRY — {title}")
    print("Quantity | Value | Units")
    for quantity, value, unit in build_trig_rows(parallax, track_rows):
        print(f"{quantity} | {fmt(value, 6)} | {unit}")
    print()
    print(f"π⊙ GEOMETRIC SOLUTION — {title}")
    print("Quantity | Value | Units")
    for quantity, value, unit in build_solution_rows(parallax):
        print(f"{quantity} | {fmt(value, 6)} | {unit}")
    print()
    print(f"CSV: {csv_path}")
    print()


def write_csv(parallax, track_rows, csv_path, title):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([VERSION, title])
        writer.writerow([])
        writer.writerow(["TRIGONOMETRY"])
        writer.writerow(["Quantity", "Value", "Units"])
        for row in build_trig_rows(parallax, track_rows):
            writer.writerow(row)
        writer.writerow([])
        writer.writerow(["π⊙ GEOMETRIC SOLUTION"])
        writer.writerow(["Quantity", "Value", "Units"])
        for row in build_solution_rows(parallax):
            writer.writerow(row)


def safe_title_from_pair_title(title):
    return title.replace(" ", "_").replace("→", "TO").replace("ø", "o").replace("Ø", "O").replace("í", "i").replace("Í", "I").replace("/", "_")


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
    iers_title = pair["title"] + " — IERS GCRS"
    iers_csv = os.path.join(out_dir, f"{VERSION}_{safe_title}_NARROW_PI_SUN_IERS_GCRS.csv")
    write_csv(parallax, track_rows, iers_csv, iers_title)
    rendered = display_html_report(parallax, track_rows, iers_csv, iers_title)
    if not rendered:
        print_plain_report(parallax, track_rows, iers_csv, iers_title)
    topo_df = build_sitecoord_master(SITE_A, SITE_B)
    topo_cache = build_cache(topo_df)
    sitecoord_parallax, sitecoord_track_rows = compute_iers_guideline_method_sitecoord(geo_cache, topo_cache)
    sitecoord_title = pair["title"] + " — HORIZONS SITE_COORD"
    sitecoord_csv = os.path.join(out_dir, f"{VERSION}_{safe_title}_NARROW_PI_SUN_SITE_COORD.csv")
    write_csv(sitecoord_parallax, sitecoord_track_rows, sitecoord_csv, sitecoord_title)
    rendered = display_html_report(sitecoord_parallax, sitecoord_track_rows, sitecoord_csv, sitecoord_title)
    if not rendered:
        print_plain_report(sitecoord_parallax, sitecoord_track_rows, sitecoord_csv, sitecoord_title)


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
# IERS-0012G
