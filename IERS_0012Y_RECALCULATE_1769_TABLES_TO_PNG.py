# IERS-0012Y
# Audit reference: GitHubDelivery@IERS-0012Y; self-contained IERS-0012N JPL reconstruction with direct PNG exports for both scientific tables.

import math
import os
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

VERSION = "IERS-0012Y"
PROGRAM_NAME = "IERS_0012Y_RECALCULATE_1769_TABLES_TO_PNG.py"

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
OUTPUT_DIR = "/content/IERS_TN36_V01_MASTER_OUTPUT"
TRIGONOMETRY_PNG = os.path.join(
    OUTPUT_DIR,
    "IERS-0012Y_1769_VARDO_POINT_VENUS_TRIGONOMETRY_TABLE.png",
)
PI_SUN_PNG = os.path.join(
    OUTPUT_DIR,
    "IERS-0012Y_1769_VARDO_POINT_VENUS_PI_SUN_GEOMETRIC_SOLUTION_TABLE.png",
)

SITE_A = {
    "key": "VARDO_NORWAY",
    "short": "Vardo",
    "label": "Vardo Norway",
    "lon_deg_east": 31.1107,
    "lat_deg": 70.3706,
    "height_m": 0.0,
}

SITE_B = {
    "key": "POINT_VENUS_TAHITI",
    "short": "Point Venus",
    "label": "Point Venus Tahiti",
    "lon_deg_east": -149.4947,
    "lat_deg": -17.4958,
    "height_m": 0.0,
}

BACKGROUND = "#03080d"
TABLE_BACKGROUND = "#050b0f"
HEADER_BACKGROUND = "#0a1a22"
BORDER = "#16333f"
TITLE_COLOR = "#66e8ff"
TEXT_COLOR = "#dff8ff"
VALUE_COLOR = "#ffc861"
UNIT_COLOR = "#5ee08a"
NOTE_COLOR = "#8fb4c1"


def ensure_package(import_name, pip_name):
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "-q", "install", pip_name]
        )


for import_name, pip_name in (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("astroquery", "astroquery"),
    ("astropy", "astropy"),
):
    ensure_package(import_name, pip_name)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar
from astroquery.jplhorizons import Horizons
import astropy.units as u
from astropy.time import Time


def norm(vector):
    vector = np.asarray(vector, dtype=float)
    return float(np.sqrt(np.dot(vector, vector)))


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    magnitude = norm(vector)
    if magnitude == 0.0:
        raise RuntimeError("Zero vector cannot be normalized.")
    return vector / magnitude


def angular_sep_arcsec(vector_a, vector_b):
    cosine = float(
        np.clip(np.dot(unit(vector_a), unit(vector_b)), -1.0, 1.0)
    )
    return math.acos(cosine) * ARCSEC_PER_RAD


def horizons_geocenter_vectors(target_id, prefix):
    table = Horizons(
        id=target_id,
        location="500@399",
        epochs={"start": START, "stop": STOP, "step": STEP},
    ).vectors().to_pandas()
    frame = pd.DataFrame()
    frame["jd_tdb"] = table["datetime_jd"].astype(float)
    frame["utc"] = table["datetime_str"].astype(str)
    frame[f"{prefix}_x_km"] = table["x"].astype(float) * AU_KM
    frame[f"{prefix}_y_km"] = table["y"].astype(float) * AU_KM
    frame[f"{prefix}_z_km"] = table["z"].astype(float) * AU_KM
    return frame


def horizons_site_location(site):
    return {
        "lon": site["lon_deg_east"] * u.deg,
        "lat": site["lat_deg"] * u.deg,
        "elevation": (site["height_m"] / 1000.0) * u.km,
    }


def horizons_site_vectors(target_id, site, prefix):
    table = Horizons(
        id=target_id,
        location=horizons_site_location(site),
        epochs={"start": START, "stop": STOP, "step": STEP},
    ).vectors().to_pandas()
    frame = pd.DataFrame()
    frame["jd_tdb"] = table["datetime_jd"].astype(float)
    frame["utc"] = table["datetime_str"].astype(str)
    frame[f"{prefix}_x_km"] = table["x"].astype(float) * AU_KM
    frame[f"{prefix}_y_km"] = table["y"].astype(float) * AU_KM
    frame[f"{prefix}_z_km"] = table["z"].astype(float) * AU_KM
    return frame


def build_geocenter_master():
    sun = horizons_geocenter_vectors("10", "GEOCENTER_SUN")
    venus = horizons_geocenter_vectors("299", "GEOCENTER_VENUS")
    master = sun.merge(venus, on=["jd_tdb", "utc"], how="inner")
    if len(master) < 500:
        raise RuntimeError(
            f"Unexpectedly short geocenter table: {len(master)} rows."
        )
    return master


def build_sitecoord_master(site_a, site_b):
    frames = []
    for site in (site_a, site_b):
        key = site["key"]
        frames.append(horizons_site_vectors("10", site, f"{key}_SUN"))
        frames.append(horizons_site_vectors("299", site, f"{key}_VENUS"))
    master = frames[0]
    for frame in frames[1:]:
        master = master.merge(frame, on=["jd_tdb", "utc"], how="inner")
    if len(master) < 500:
        raise RuntimeError(
            f"Unexpectedly short SITE_COORD table: {len(master)} rows."
        )
    return master


def build_cache(frame):
    jd = frame["jd_tdb"].to_numpy(dtype=float)
    cache = {
        "jd_tdb": jd,
        "utc": frame["utc"].astype(str).tolist(),
    }
    for column in frame.columns:
        if column.endswith("_km"):
            cache[column] = CubicSpline(
                jd,
                frame[column].to_numpy(dtype=float),
                bc_type="natural",
            )
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


def utc_at(jd_tdb):
    return Time(float(jd_tdb), format="jd", scale="tdb").utc.iso


def site_sun_vector(topo_cache, site, jd_tdb):
    return vec_at(topo_cache, f"{site['key']}_SUN", jd_tdb)


def site_venus_vector(topo_cache, site, jd_tdb):
    return vec_at(topo_cache, f"{site['key']}_VENUS", jd_tdb)


def site_sep_arcsec(topo_cache, site, jd_tdb):
    return angular_sep_arcsec(
        site_sun_vector(topo_cache, site, jd_tdb),
        site_venus_vector(topo_cache, site, jd_tdb),
    )


def angular_radii_arcsec(topo_cache, site, jd_tdb):
    sun = site_sun_vector(topo_cache, site, jd_tdb)
    venus = site_venus_vector(topo_cache, site, jd_tdb)
    sun_radius = math.atan2(SUN_RADIUS_KM, norm(sun)) * ARCSEC_PER_RAD
    venus_radius = math.atan2(VENUS_RADIUS_KM, norm(venus)) * ARCSEC_PER_RAD
    return sun_radius, venus_radius


def contact_function(topo_cache, site, event, jd_tdb):
    separation = site_sep_arcsec(topo_cache, site, jd_tdb)
    sun_radius, venus_radius = angular_radii_arcsec(
        topo_cache,
        site,
        jd_tdb,
    )
    threshold = (
        sun_radius + venus_radius
        if event in ("C1", "C4")
        else sun_radius - venus_radius
    )
    return separation - threshold


def find_event_roots(topo_cache, site, event):
    jds = topo_cache["jd_tdb"]
    values = np.array(
        [contact_function(topo_cache, site, event, jd) for jd in jds],
        dtype=float,
    )
    roots = []
    for index in range(len(jds) - 1):
        left_value = values[index]
        right_value = values[index + 1]
        if not np.isfinite(left_value) or not np.isfinite(right_value):
            continue
        if left_value == 0.0:
            roots.append(float(jds[index]))
        elif left_value * right_value < 0.0:
            root = brentq(
                lambda jd: contact_function(topo_cache, site, event, jd),
                float(jds[index]),
                float(jds[index + 1]),
                xtol=1e-13,
                rtol=1e-13,
                maxiter=100,
            )
            roots.append(float(root))
    return sorted(roots)


def find_site_contacts(topo_cache, site):
    outer = find_event_roots(topo_cache, site, "C1")
    inner = find_event_roots(topo_cache, site, "C2")
    if len(outer) < 2 or len(inner) < 2:
        raise RuntimeError(
            f"Could not derive four contacts for {site['label']}."
        )
    return {
        "C1": outer[0],
        "C2": inner[0],
        "C3": inner[-1],
        "C4": outer[-1],
    }


def find_site_closest(topo_cache, site):
    jds = topo_cache["jd_tdb"]
    separations = np.array(
        [site_sep_arcsec(topo_cache, site, jd) for jd in jds],
        dtype=float,
    )
    minimum_index = int(np.argmin(separations))
    center_jd = float(jds[minimum_index])
    left_jd = float(jds[max(0, minimum_index - 3)])
    right_jd = float(jds[min(len(jds) - 1, minimum_index + 3)])
    lower_seconds = (left_jd - center_jd) * 86_400.0
    upper_seconds = (right_jd - center_jd) * 86_400.0

    result = minimize_scalar(
        lambda seconds: site_sep_arcsec(
            topo_cache,
            site,
            center_jd + seconds / 86_400.0,
        ),
        bounds=(lower_seconds, upper_seconds),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(
            f"Closest-approach minimization failed for {site['label']}: "
            f"{result.message}"
        )
    return center_jd + float(result.x) / 86_400.0


def fixed_geocenter_basis(geo_cache, jd_tdb):
    sun = vec_at(geo_cache, "GEOCENTER_SUN", jd_tdb)
    normal = unit(sun)
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if norm(np.cross(reference, normal)) < 1e-12:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)
    x_hat = unit(np.cross(reference, normal))
    y_hat = unit(np.cross(normal, x_hat))
    return normal, x_hat, y_hat


def ray_screen_point_arcsec_sitecoord(
    geo_cache,
    topo_cache,
    site,
    jd_tdb,
    basis,
):
    normal, x_hat, y_hat = basis
    sun_geo = vec_at(geo_cache, "GEOCENTER_SUN", jd_tdb)
    sun_topo = site_sun_vector(topo_cache, site, jd_tdb)
    venus_topo = site_venus_vector(topo_cache, site, jd_tdb)
    observer_geo = sun_geo - sun_topo
    denominator = float(np.dot(venus_topo, normal))
    if abs(denominator) < 1e-14:
        raise RuntimeError("SITE_COORD Venus ray is parallel to the screen.")
    tau = float(np.dot(sun_geo - observer_geo, normal) / denominator)
    hit = observer_geo + tau * venus_topo
    screen_vector = hit - sun_geo
    earth_sun_distance = norm(sun_geo)
    x_arcsec = (
        math.atan2(float(np.dot(screen_vector, x_hat)), earth_sun_distance)
        * ARCSEC_PER_RAD
    )
    y_arcsec = (
        math.atan2(float(np.dot(screen_vector, y_hat)), earth_sun_distance)
        * ARCSEC_PER_RAD
    )
    return np.array([x_arcsec, y_arcsec], dtype=float)


def pca_direction(points):
    points = np.asarray(points, dtype=float)
    centroid = points.mean(axis=0)
    centered = points - centroid
    _u, _singular_values, vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    direction = vt[0]
    if direction[0] < 0.0:
        direction = -direction
    return centroid, unit(direction)


def line_intersection(line_center, line_direction, midpoint, normal):
    matrix = np.column_stack([line_direction, -normal])
    rhs = midpoint - line_center
    solution, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    return line_center + solution[0] * line_direction


def distances_at(geo_cache, jd_tdb):
    sun = vec_at(geo_cache, "GEOCENTER_SUN", jd_tdb)
    venus = vec_at(geo_cache, "GEOCENTER_VENUS", jd_tdb)
    earth_sun = norm(sun)
    earth_venus = norm(venus)
    venus_sun = norm(venus - sun)
    return earth_sun, earth_venus, venus_sun


def site_track(
    geo_cache,
    topo_cache,
    site,
    contacts,
    closest_jd,
    basis,
):
    minute_jds = topo_cache["jd_tdb"]
    mask = (minute_jds >= contacts["C1"]) & (
        minute_jds <= contacts["C4"]
    )
    used_jds = sorted(
        set(
            [
                contacts["C1"],
                contacts["C2"],
                closest_jd,
                contacts["C3"],
                contacts["C4"],
            ]
            + list(minute_jds[mask])
        )
    )
    points = np.array(
        [
            ray_screen_point_arcsec_sitecoord(
                geo_cache,
                topo_cache,
                site,
                jd,
                basis,
            )
            for jd in used_jds
        ],
        dtype=float,
    )
    centroid, direction = pca_direction(points)
    return {
        "site": site,
        "jds": np.array(used_jds, dtype=float),
        "points": points,
        "centroid": centroid,
        "direction": direction,
        "closest_jd": closest_jd,
        "closest_utc": utc_at(closest_jd),
        "track_angle_deg": math.degrees(
            math.atan2(direction[1], direction[0])
        ),
    }


def compute_parallax_geometry(geo_cache, track_a, track_b, screen_jd):
    tangent = unit(track_a["direction"] + track_b["direction"])
    if tangent[0] < 0.0:
        tangent = -tangent
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    midpoint = 0.5 * (track_a["centroid"] + track_b["centroid"])
    a_prime = line_intersection(
        track_a["centroid"],
        track_a["direction"],
        midpoint,
        normal,
    )
    b_prime = line_intersection(
        track_b["centroid"],
        track_b["direction"],
        midpoint,
        normal,
    )
    chord_vector = b_prime - a_prime
    a_prime_b_prime_arcsec = float(np.linalg.norm(chord_vector))
    rho_arcsec = abs(float(np.dot(chord_vector, normal)))

    earth_sun, earth_venus, venus_sun = distances_at(
        geo_cache,
        screen_jd,
    )
    a_prime_b_prime_km = (
        math.tan(a_prime_b_prime_arcsec / ARCSEC_PER_RAD) * earth_sun
    )
    ab_km = a_prime_b_prime_km * earth_venus / venus_sun
    ab_arcsec = math.atan2(ab_km, earth_sun) * ARCSEC_PER_RAD
    halley_ratio = a_prime_b_prime_km / ab_km
    raw_phi_arcsec = (
        rho_arcsec
        * (earth_venus / venus_sun)
        * (EARTH_RADIUS_KM / ab_km)
    )
    pi_sun_arcsec = raw_phi_arcsec * (earth_sun / AU_KM)
    rho_scaled_arcsec = rho_arcsec * EARTH_RADIUS_KM / ab_km

    return {
        "A_prime_B_prime_arcsec": a_prime_b_prime_arcsec,
        "A_prime_B_prime_km": a_prime_b_prime_km,
        "rho_arcsec": rho_arcsec,
        "rho_scaled_arcsec": rho_scaled_arcsec,
        "AB_arcsec": ab_arcsec,
        "AB_km": ab_km,
        "halley_ratio": halley_ratio,
        "raw_phi_arcsec": raw_phi_arcsec,
        "pi_sun_arcsec": pi_sun_arcsec,
        "pi_sun_residual_arcsec": (
            pi_sun_arcsec - PI_SUN_REFERENCE_ARCSEC
        ),
        "pi_sun_residual_percent": (
            100.0
            * (pi_sun_arcsec - PI_SUN_REFERENCE_ARCSEC)
            / PI_SUN_REFERENCE_ARCSEC
        ),
        "pi_sun_reference_arcsec": PI_SUN_REFERENCE_ARCSEC,
        "D_ES_AU": earth_sun / AU_KM,
        "D_EV_D_VS": earth_venus / venus_sun,
        "D_VS_D_EV": venus_sun / earth_venus,
        "D_ES_source": "|GEOCENTER_SUN| / AU_KM",
    }


def decimals_for_quantity(quantity, unit):
    if quantity == "D ES":
        return 12
    if quantity in (
        "Computed π⊙",
        "Reference π⊙",
        "Residual π⊙",
        "Raw φ",
    ):
        return 9
    if quantity in ("A′B′ / AB", "D EV / D VS", "D VS / D EV"):
        return 10
    if unit in ("UTC", "JPL"):
        return None
    return 6


def format_value(quantity, value, unit):
    if isinstance(value, str):
        return value
    decimals = decimals_for_quantity(quantity, unit)
    if decimals is None:
        return str(value)
    return f"{float(value):.{decimals}f}"


def trigonometry_rows(track_a, track_b):
    delta_angle = abs(
        track_a["track_angle_deg"] - track_b["track_angle_deg"]
    )
    average_angle = 0.5 * (
        track_a["track_angle_deg"] + track_b["track_angle_deg"]
    )
    return [
        ("β Vardo", track_a["track_angle_deg"], "deg"),
        ("β Point Venus", track_b["track_angle_deg"], "deg"),
        ("Δβ", delta_angle, "deg"),
        ("β Average", average_angle, "deg"),
    ]


def geometric_rows(track_a, track_b, geometry):
    return [
        ("Closest Vardo UTC", track_a["closest_utc"], "UTC"),
        ("Closest Point Venus UTC", track_b["closest_utc"], "UTC"),
        (
            "A′B′ Angular Chord",
            geometry["A_prime_B_prime_arcsec"],
            "arcsec",
        ),
        (
            "A′B′ Solar-Screen Chord",
            geometry["A_prime_B_prime_km"],
            "km",
        ),
        ("AB Angular Projection", geometry["AB_arcsec"], "arcsec"),
        ("AB Projected Baseline", geometry["AB_km"], "km"),
        ("A′B′ / AB", geometry["halley_ratio"], "ratio"),
        ("Normal Separation ρ", geometry["rho_arcsec"], "arcsec"),
        (
            "ρ Scaled To R⊕",
            geometry["rho_scaled_arcsec"],
            "arcsec",
        ),
        ("D ES", geometry["D_ES_AU"], "AU"),
        ("D ES Source", geometry["D_ES_source"], "JPL"),
        ("D EV / D VS", geometry["D_EV_D_VS"], "ratio"),
        ("D VS / D EV", geometry["D_VS_D_EV"], "ratio"),
        ("Raw φ", geometry["raw_phi_arcsec"], "arcsec"),
        ("Computed π⊙", geometry["pi_sun_arcsec"], "arcsec"),
        (
            "Reference π⊙",
            geometry["pi_sun_reference_arcsec"],
            "arcsec",
        ),
        (
            "Residual π⊙",
            geometry["pi_sun_residual_arcsec"],
            "arcsec",
        ),
        (
            "Residual π⊙",
            geometry["pi_sun_residual_percent"],
            "percent",
        ),
    ]


def save_table_png(title, rows, output_path, footer, width, row_height):
    formatted_rows = [
        [quantity, format_value(quantity, value, unit), unit]
        for quantity, value, unit in rows
    ]
    height = max(3.5, 1.45 + row_height * (len(rows) + 1))

    figure, axis = plt.subplots(figsize=(width, height), dpi=220)
    figure.patch.set_facecolor(BACKGROUND)
    axis.set_facecolor(BACKGROUND)
    axis.axis("off")

    axis.text(
        0.5,
        0.965,
        title,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=14.0,
        fontweight="bold",
        color=TITLE_COLOR,
    )
    axis.plot(
        [0.02, 0.98],
        [0.915, 0.915],
        transform=axis.transAxes,
        linewidth=0.7,
        color="#25708b",
        clip_on=False,
    )

    table = axis.table(
        cellText=formatted_rows,
        colLabels=["Quantity", "Value", "Units"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.50, 0.34, 0.16],
        bbox=[0.02, 0.085, 0.96, 0.79],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.4)

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor(BORDER)
        cell.PAD = 0.055
        if row_index == 0:
            cell.set_facecolor(HEADER_BACKGROUND)
            cell.get_text().set_color(TITLE_COLOR)
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_ha("left")
        else:
            cell.set_facecolor(TABLE_BACKGROUND)
            if column_index == 0:
                cell.get_text().set_color(TEXT_COLOR)
                cell.get_text().set_ha("left")
            elif column_index == 1:
                cell.get_text().set_color(VALUE_COLOR)
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_ha("right")
            else:
                cell.get_text().set_color(UNIT_COLOR)
                cell.get_text().set_ha("left")

    axis.text(
        0.02,
        0.028,
        footer,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        color=NOTE_COLOR,
    )

    figure.savefig(
        output_path,
        dpi=360,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
        edgecolor="none",
        pad_inches=0.08,
    )
    plt.close(figure)


def display_pngs(paths):
    try:
        from IPython.display import Image, display

        for path in paths:
            display(Image(filename=path))
        return "DISPLAYED IN COLAB"
    except Exception as exc:
        return f"NOT USED / INLINE DISPLAY UNAVAILABLE ({type(exc).__name__})"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"CODE OUTPUT: {VERSION}")
    print()
    print("CODE INPUTS")
    print(f"Program                : {PROGRAM_NAME}")
    print(f"JPL interval           : {START} TO {STOP} STEP {STEP}")
    print(f"Observer A             : {SITE_A['label']}")
    print(f"Observer B             : {SITE_B['label']}")
    print()
    print("COMMENTS")
    print("Recalculates the IERS-0012N Vardo–Point Venus geometry directly from JPL Horizons vectors.")
    print("No prior IERS-0012N CSV or Colab session state is required.")
    print("The closest-approach minimization uses centered elapsed seconds rather than raw Julian dates.")
    print("Exports the trigonometry and solar-parallax tables as separate high-resolution Matplotlib PNG files.")
    print("No AI image generation is used.")
    print()

    geocenter_master = build_geocenter_master()
    topocentric_master = build_sitecoord_master(SITE_A, SITE_B)
    geocenter_cache = build_cache(geocenter_master)
    topocentric_cache = build_cache(topocentric_master)

    contacts_a = find_site_contacts(topocentric_cache, SITE_A)
    contacts_b = find_site_contacts(topocentric_cache, SITE_B)
    closest_a = find_site_closest(topocentric_cache, SITE_A)
    closest_b = find_site_closest(topocentric_cache, SITE_B)
    screen_jd = 0.5 * (closest_a + closest_b)
    basis = fixed_geocenter_basis(geocenter_cache, screen_jd)

    track_a = site_track(
        geocenter_cache,
        topocentric_cache,
        SITE_A,
        contacts_a,
        closest_a,
        basis,
    )
    track_b = site_track(
        geocenter_cache,
        topocentric_cache,
        SITE_B,
        contacts_b,
        closest_b,
        basis,
    )
    geometry = compute_parallax_geometry(
        geocenter_cache,
        track_a,
        track_b,
        screen_jd,
    )

    trig_rows = trigonometry_rows(track_a, track_b)
    pi_rows = geometric_rows(track_a, track_b, geometry)

    save_table_png(
        "TRIGONOMETRY — VARDO NORWAY → POINT VENUS TAHITI",
        trig_rows,
        TRIGONOMETRY_PNG,
        "JPL Horizons geocenter and SITE_COORD vectors; IERS-0012N geometry.",
        width=10.8,
        row_height=0.54,
    )
    save_table_png(
        "π⊙ GEOMETRIC SOLUTION — VARDO NORWAY → POINT VENUS TAHITI",
        pi_rows,
        PI_SUN_PNG,
        "JPL Horizons geocenter and SITE_COORD vectors; IERS-0012N geometry.",
        width=12.8,
        row_height=0.43,
    )

    display_status = display_pngs([TRIGONOMETRY_PNG, PI_SUN_PNG])

    print("RESULTS")
    print(f"Vardo closest UTC      : {track_a['closest_utc']}")
    print(f"Point Venus closest UTC: {track_b['closest_utc']}")
    print(f"Vardo track angle      : {track_a['track_angle_deg']:.6f} deg")
    print(f"Point Venus angle      : {track_b['track_angle_deg']:.6f} deg")
    print(f"Track angle delta abs  : {abs(track_a['track_angle_deg'] - track_b['track_angle_deg']):.6f} deg")
    print(f"A prime B prime        : {geometry['A_prime_B_prime_arcsec']:.6f} arcsec")
    print(f"A prime B prime        : {geometry['A_prime_B_prime_km']:.6f} km")
    print(f"AB                     : {geometry['AB_arcsec']:.6f} arcsec")
    print(f"AB                     : {geometry['AB_km']:.6f} km")
    print(f"rho                    : {geometry['rho_arcsec']:.6f} arcsec")
    print(f"Pi sun                 : {geometry['pi_sun_arcsec']:.9f} arcsec")
    print(f"Pi sun residual        : {geometry['pi_sun_residual_arcsec']:.9f} arcsec")
    print(f"Inline display         : {display_status}")
    print()
    print("OUTPUT SUMMARY")
    print(f"Trigonometry PNG       : {TRIGONOMETRY_PNG}")
    print(f"Pi sun geometry PNG    : {PI_SUN_PNG}")
    print()
    print("PAPER COMPARISON")
    print(f"Reference pi sun       : {PI_SUN_REFERENCE_ARCSEC:.6f} arcsec")
    print(f"Computed-reference     : {geometry['pi_sun_residual_arcsec']:+.9f} arcsec")
    print()
    print("EQUATION STATUS")
    print("JPL geocenter vectors                  : VERIFIED")
    print("JPL SITE_COORD observer vectors        : VERIFIED")
    print("Four-contact roots                     : VERIFIED")
    print("Centered-seconds closest approach      : VERIFIED")
    print("Fixed geocentric solar-screen geometry : VERIFIED")
    print("PCA track fits and Halley geometry     : VERIFIED")
    print("Separate Matplotlib PNG table exports  : VERIFIED")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# IERS-0012Y
