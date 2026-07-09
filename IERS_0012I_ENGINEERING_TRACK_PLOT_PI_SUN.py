# IERS-0012I
# Audit reference: GitHubDelivery@IERS-0012I; publication-style half-Sun engineering track plot from JPL-derived geometry.

import os
import sys
import math
import csv
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

VERSION = "IERS-0012I"
PROGRAM_NAME = "IERS_0012I_ENGINEERING_TRACK_PLOT_PI_SUN.py"

AU_KM = 149_597_870.7
ARCSEC_PER_RAD = 206_264.80624709636
SUN_RADIUS_KM = 695_700.0
VENUS_RADIUS_KM = 6_051.8

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


def ensure_package(import_name, pip_name):
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pip_name])


ensure_package("numpy", "numpy")
ensure_package("pandas", "pandas")
ensure_package("scipy", "scipy")
ensure_package("matplotlib", "matplotlib")
ensure_package("astroquery", "astroquery")
ensure_package("astropy", "astropy")

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
    return Time(jd_tdb, format="jd", scale="tdb").utc.iso.replace(" ", " ")


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
    return sorted(roots)


def find_site_contacts(cache, site):
    outer = find_event_roots(cache, site, "C1")
    inner = find_event_roots(cache, site, "C2")
    if len(outer) < 2 or len(inner) < 2:
        raise RuntimeError(f"Could not derive four contacts for {site['label']}.")
    return {
        "C1": outer[0],
        "C2": inner[0],
        "C3": inner[-1],
        "C4": outer[-1],
    }


def find_site_closest(cache, site):
    jds = cache["jd_tdb"]
    vals = [site_sep_arcsec(cache, site, jd) for jd in jds]
    i = int(np.argmin(vals))
    a = jds[max(0, i - 3)]
    b = jds[min(len(jds) - 1, i + 3)]
    res = minimize_scalar(
        lambda jd: site_sep_arcsec(cache, site, jd),
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


def pca_angle_deg(points):
    pts = np.asarray(points, dtype=float)
    centered = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    d = vt[0]
    if d[0] < 0:
        d = -d
    return math.degrees(math.atan2(d[1], d[0]))


def site_track(cache, site, contacts, closest_jd, basis):
    jds = cache["jd_tdb"]
    mask = (jds >= contacts["C1"]) & (jds <= contacts["C4"])
    use_jds = list(jds[mask])
    use_jds = sorted(set([contacts["C1"], contacts["C2"], closest_jd, contacts["C3"], contacts["C4"]] + use_jds))
    pts = np.array([ray_screen_point_arcsec(cache, site, jd, basis) for jd in use_jds], dtype=float)
    event_pts = {name: ray_screen_point_arcsec(cache, site, jd, basis) for name, jd in contacts.items()}
    event_pts["CA"] = ray_screen_point_arcsec(cache, site, closest_jd, basis)
    _rs, rv = angular_radii_arcsec(cache, site, closest_jd)
    return {
        "site": site,
        "jds": np.array(use_jds, dtype=float),
        "pts": pts,
        "event_pts": event_pts,
        "contacts": contacts,
        "closest_jd": closest_jd,
        "closest_utc": utc_at(cache, closest_jd),
        "venus_radius_arcsec": rv,
        "track_angle_deg": pca_angle_deg(pts),
    }


def sun_radius_arcsec(cache, jd_tdb):
    es = norm(vec_at(cache, "GEOCENTER_SUN", jd_tdb))
    return math.atan2(SUN_RADIUS_KM, es) * ARCSEC_PER_RAD


def write_event_csv(track_a, track_b, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([VERSION, "ENGINEERING TRACK PLOT EVENT COORDINATES"])
        writer.writerow(["site", "event", "utc", "jd_tdb", "x_arcsec", "y_arcsec", "venus_radius_arcsec", "track_angle_deg"])
        for track in [track_a, track_b]:
            site = track["site"]
            for event in ["C1", "C2", "CA", "C3", "C4"]:
                jd = track["closest_jd"] if event == "CA" else track["contacts"][event]
                xy = track["event_pts"][event]
                writer.writerow([
                    site["label"],
                    event,
                    utc_at_from_track(track, jd),
                    f"{jd:.12f}",
                    f"{xy[0]:.6f}",
                    f"{xy[1]:.6f}",
                    f"{track['venus_radius_arcsec']:.6f}",
                    f"{track['track_angle_deg']:.6f}",
                ])


def utc_at_from_track(track, jd):
    if abs(jd - track["closest_jd"]) < 1e-10:
        return track["closest_utc"]
    return Time(jd, format="jd", scale="tdb").utc.iso.replace(" ", " ")


def axis_limits_for_half_sun(radius, track_a, track_b):
    all_pts = np.vstack([track_a["pts"], track_b["pts"]])
    y_med = float(np.median(all_pts[:, 1]))
    sign = 1.0 if y_med >= 0.0 else -1.0
    xlim = (-1.04 * radius, 1.04 * radius)
    if sign > 0.0:
        ylim = (-0.06 * radius, 1.06 * radius)
    else:
        ylim = (-1.06 * radius, 0.06 * radius)
    y_min = float(np.min(all_pts[:, 1]))
    y_max = float(np.max(all_pts[:, 1]))
    if y_min < ylim[0] or y_max > ylim[1]:
        pad = 0.08 * radius
        ylim = (min(ylim[0], y_min - pad), max(ylim[1], y_max + pad))
    return xlim, ylim


def add_label(ax, xy, text, dx, dy, color):
    ax.annotate(
        text,
        xy=(xy[0], xy[1]),
        xytext=(xy[0] + dx, xy[1] + dy),
        textcoords="data",
        fontsize=6.5,
        color=color,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", lw=0.35, color=color, shrinkA=0, shrinkB=2),
    )


def plot_engineering_track(cache, track_a, track_b, screen_jd, png_path):
    radius = sun_radius_arcsec(cache, screen_jd)
    fig, ax = plt.subplots(figsize=(9.2, 5.7), dpi=220)
    fig.patch.set_facecolor("#03080d")
    ax.set_facecolor("#03080d")

    limb = Circle((0.0, 0.0), radius, fill=False, lw=0.55, ec="#66e8ff", alpha=0.95)
    ax.add_patch(limb)

    ax.axhline(0.0, lw=0.32, color="#1d3d4a", alpha=0.7)
    ax.axvline(0.0, lw=0.32, color="#1d3d4a", alpha=0.7)

    colors = {
        "Wellington NZ": "#ffc861",
        "Alaejos Spain": "#5ee08a",
    }

    for track in [track_a, track_b]:
        site_label = track["site"]["label"]
        color = colors.get(site_label, "#ffffff")
        pts = track["pts"]
        ax.plot(pts[:, 0], pts[:, 1], lw=0.58, color=color, solid_capstyle="round", label=site_label)
        ax.scatter(pts[::8, 0], pts[::8, 1], s=1.6, color=color, alpha=0.70, linewidths=0)
        ca = track["event_pts"]["CA"]
        venus = Circle((ca[0], ca[1]), track["venus_radius_arcsec"], fill=False, lw=0.55, ec=color, alpha=0.95)
        ax.add_patch(venus)
        for event in ["C1", "C2", "CA", "C3", "C4"]:
            xy = track["event_pts"][event]
            ax.scatter([xy[0]], [xy[1]], s=8 if event == "CA" else 5, color=color, edgecolors="#03080d", linewidths=0.30, zorder=5)
        dy = 17.0 if site_label.startswith("Wellington") else -17.0
        add_label(ax, ca, f"{track['site']['short']} CA", 20.0, dy, color)

    for event, dx in [("C1", -52.0), ("C2", -40.0), ("C3", 20.0), ("C4", 32.0)]:
        xy = track_a["event_pts"][event]
        add_label(ax, xy, event, dx, 10.0, "#8fb4c1")

    xlim, ylim = axis_limits_for_half_sun(radius, track_a, track_b)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")

    for spine in ax.spines.values():
        spine.set_linewidth(0.35)
        spine.set_color("#25708b")

    ax.tick_params(axis="both", colors="#8fb4c1", labelsize=7, width=0.35, length=2.5)
    ax.grid(True, color="#102630", linewidth=0.25, alpha=0.58)
    ax.set_xlabel("Solar-screen X offset (arcsec)", color="#8fb4c1", fontsize=8)
    ax.set_ylabel("Solar-screen Y offset (arcsec)", color="#8fb4c1", fontsize=8)

    title = "2012 Venus Transit — Wellington NZ / Alaejos Spain"
    subtitle = "JPL Horizons geocentric Sun/Venus vectors + IERS/Astropy GCRS observer transform"
    ax.set_title(title + "\n" + subtitle, color="#f8fdff", fontsize=9.5, pad=9)

    legend = ax.legend(loc="lower right", fontsize=7, frameon=True, borderpad=0.55)
    legend.get_frame().set_facecolor("#071016")
    legend.get_frame().set_edgecolor("#1e4f64")
    for text in legend.get_texts():
        text.set_color("#dff8ff")

    note = (
        f"Venus disks are plotted to scale at observer-specific closest approach.  "
        f"Wellington CA: {track_a['closest_utc']}   Alaejos CA: {track_b['closest_utc']}"
    )
    fig.text(0.5, 0.018, note, ha="center", va="bottom", fontsize=6.8, color="#8fb4c1")
    fig.savefig(png_path, dpi=420, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.06)
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

    png_path = os.path.join(out_dir, f"{VERSION}_WELLINGTON_ALAEJOS_ENGINEERING_HALF_SUN_TRACKS.png")
    csv_path = os.path.join(out_dir, f"{VERSION}_WELLINGTON_ALAEJOS_ENGINEERING_HALF_SUN_EVENTS.csv")

    write_event_csv(track_a, track_b, csv_path)
    plot_engineering_track(cache, track_a, track_b, screen_jd, png_path)

    print("RESULTS")
    print(f"Wellington closest UTC : {track_a['closest_utc']}")
    print(f"Alaejos closest UTC    : {track_b['closest_utc']}")
    print(f"Wellington track angle : {track_a['track_angle_deg']:.6f} deg")
    print(f"Alaejos track angle    : {track_b['track_angle_deg']:.6f} deg")
    print(f"Track angle delta abs  : {abs(track_a['track_angle_deg'] - track_b['track_angle_deg']):.6f} deg")
    print(f"PNG output             : {png_path}")
    print(f"CSV output             : {csv_path}")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# IERS-0012I
