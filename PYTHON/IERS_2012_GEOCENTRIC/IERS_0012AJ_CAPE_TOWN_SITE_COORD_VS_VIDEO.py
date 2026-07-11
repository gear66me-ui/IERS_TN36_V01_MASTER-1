# IERS-0012AJ
# Audit reference: Cape Town Horizons SITE_COORD 2012 track with C1-C4/CA and SDO video-angle overlay.
from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

def ensure_package(import_name: str, pip_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pip_name])

for _import_name, _pip_name in (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("astroquery", "astroquery"),
    ("astropy", "astropy"),
):
    ensure_package(_import_name, _pip_name)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar
from astroquery.jplhorizons import Horizons
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.utils import iers
import astropy.units as u
from IPython.display import Image as IPythonImage, display

VERSION = "IERS-0012AJ"
LOCAL_TZ = ZoneInfo("America/Bogota")

AU_KM = 149_597_870.7
ARCSEC_PER_RAD = 206_264.80624709636
SUN_RADIUS_KM = 695_700.0
VENUS_RADIUS_KM = 6_051.8

START = "2012-Jun-05 20:00"
STOP = "2012-Jun-06 07:30"
STEP = "1m"

CAPE_TOWN = {
    "key": "CAPE_TOWN",
    "label": "Cape Town",
    "lon_deg_east": 18.4241,
    "lat_deg": -33.9249,
    "height_m": 0.0,
}
REYKJAVIK = {
    "key": "REYKJAVIK",
    "label": "Reykjavik",
    "lon_deg_east": -21.9426,
    "lat_deg": 64.1466,
    "height_m": 0.0,
}

VIDEO_TLS_ANGLE_DEG = 8.4340601435
ARCHIVED_IERS0011_CAPE_SITECOORD_DEG = 8.438165

DRIVE_ROOT = Path("/content/drive/MyDrive/Colab Notebooks")
if DRIVE_ROOT.exists():
    PROJECT_ROOT = DRIVE_ROOT / "IERS_TN36_OUTPUT"
else:
    PROJECT_ROOT = Path("/content/IERS_TN36_OUTPUT")

OUTPUT_PNG_DIR = PROJECT_ROOT / "OUTPUT_PNG"
OUTPUT_CSV_DIR = PROJECT_ROOT / "OUTPUT_CSV"
OUTPUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)

PLOT_PNG = OUTPUT_PNG_DIR / f"{VERSION}_CAPE_TOWN_SITE_COORD_VS_VIDEO.png"
TRACK_CSV = OUTPUT_CSV_DIR / f"{VERSION}_CAPE_TOWN_SITE_COORD_VS_VIDEO.csv"

BG = "#000000"
TEXT = "#f2f2f2"
MUTED = "#bdbdbd"
GRID = "#262626"
SOLAR = "#d0d0d0"
JPL_COLOR = "#2ecc71"
VIDEO_COLOR = "#ff4d4d"
VENUS_COLOR = "#f7f7f7"
TABLE_EDGE = "#777777"

iers.conf.auto_download = True
iers.conf.iers_degraded_accuracy = "warn"

@dataclass(frozen=True)
class Event:
    name: str
    jd_tdb: float
    utc: str
    point_arcsec: np.ndarray
    sun_radius_arcsec: float
    venus_radius_arcsec: float

def norm(vector: np.ndarray) -> float:
    return float(np.sqrt(np.dot(vector, vector)))

def unit(vector: np.ndarray) -> np.ndarray:
    magnitude = norm(vector)
    if magnitude == 0.0:
        raise RuntimeError("Zero vector encountered.")
    return np.asarray(vector, dtype=float) / magnitude

def wrap_horizontal(angle_deg: float) -> float:
    return (float(angle_deg) + 90.0) % 180.0 - 90.0

def angular_sep_arcsec(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(unit(vector_a), unit(vector_b)), -1.0, 1.0))
    return math.acos(cosine) * ARCSEC_PER_RAD

def horizons_vectors(target_id: str, location, prefix: str) -> pd.DataFrame:
    vectors = Horizons(
        id=target_id,
        location=location,
        epochs={"start": START, "stop": STOP, "step": STEP},
    ).vectors().to_pandas()
    table = pd.DataFrame()
    table["jd_tdb"] = vectors["datetime_jd"].astype(float)
    table["utc"] = vectors["datetime_str"].astype(str)
    for axis in ("x", "y", "z"):
        table[f"{prefix}_{axis}_km"] = vectors[axis].astype(float) * AU_KM
    return table

def horizons_site_location(site: dict[str, float | str]) -> dict[str, object]:
    return {
        "lon": float(site["lon_deg_east"]) * u.deg,
        "lat": float(site["lat_deg"]) * u.deg,
        "elevation": (float(site["height_m"]) / 1000.0) * u.km,
    }

def build_geocenter_master() -> pd.DataFrame:
    sun = horizons_vectors("10", "500@399", "GEOCENTER_SUN")
    venus = horizons_vectors("299", "500@399", "GEOCENTER_VENUS")
    master = sun.merge(venus, on=["jd_tdb", "utc"], how="inner")
    if len(master) < 100:
        raise RuntimeError("Geocenter JPL master contains too few rows.")
    return master

def build_cape_sitecoord_master() -> pd.DataFrame:
    location = horizons_site_location(CAPE_TOWN)
    sun = horizons_vectors("10", location, "CAPE_TOWN_SUN")
    venus = horizons_vectors("299", location, "CAPE_TOWN_VENUS")
    master = sun.merge(venus, on=["jd_tdb", "utc"], how="inner")
    if len(master) < 100:
        raise RuntimeError("Cape Town SITE_COORD master contains too few rows.")
    return master

def build_cache(table: pd.DataFrame) -> dict[str, object]:
    cache: dict[str, object] = {
        "jd_tdb": table["jd_tdb"].to_numpy(dtype=float),
        "utc": table["utc"].astype(str).tolist(),
    }
    for column in table.columns:
        if column.endswith("_km"):
            cache[column] = CubicSpline(
                cache["jd_tdb"],
                table[column].to_numpy(dtype=float),
                bc_type="natural",
            )
    return cache

def vector_at(cache: dict[str, object], prefix: str, jd_tdb: float) -> np.ndarray:
    return np.array(
        [
            float(cache[f"{prefix}_x_km"](jd_tdb)),
            float(cache[f"{prefix}_y_km"](jd_tdb)),
            float(cache[f"{prefix}_z_km"](jd_tdb)),
        ],
        dtype=float,
    )

def utc_at(jd_tdb: float) -> str:
    return Time(jd_tdb, format="jd", scale="tdb").utc.iso

def earth_location(site: dict[str, float | str]) -> EarthLocation:
    return EarthLocation.from_geodetic(
        lon=float(site["lon_deg_east"]) * u.deg,
        lat=float(site["lat_deg"]) * u.deg,
        height=float(site["height_m"]) * u.m,
    )

def observer_gcrs_km(site: dict[str, float | str], jd_tdb: float) -> np.ndarray:
    position, _velocity = earth_location(site).get_gcrs_posvel(
        Time(jd_tdb, format="jd", scale="tdb")
    )
    return np.array(
        [
            position.x.to_value(u.km),
            position.y.to_value(u.km),
            position.z.to_value(u.km),
        ],
        dtype=float,
    )

def astropy_topocentric_vector(
    geo_cache: dict[str, object],
    site: dict[str, float | str],
    body_prefix: str,
    jd_tdb: float,
) -> np.ndarray:
    return vector_at(geo_cache, body_prefix, jd_tdb) - observer_gcrs_km(site, jd_tdb)

def astropy_angular_radii(
    geo_cache: dict[str, object],
    site: dict[str, float | str],
    jd_tdb: float,
) -> tuple[float, float]:
    sun = astropy_topocentric_vector(geo_cache, site, "GEOCENTER_SUN", jd_tdb)
    venus = astropy_topocentric_vector(geo_cache, site, "GEOCENTER_VENUS", jd_tdb)
    sun_radius = math.atan2(SUN_RADIUS_KM, norm(sun)) * ARCSEC_PER_RAD
    venus_radius = math.atan2(VENUS_RADIUS_KM, norm(venus)) * ARCSEC_PER_RAD
    return sun_radius, venus_radius

def astropy_contact_value(
    geo_cache: dict[str, object],
    site: dict[str, float | str],
    kind: str,
    jd_tdb: float,
) -> float:
    sun = astropy_topocentric_vector(geo_cache, site, "GEOCENTER_SUN", jd_tdb)
    venus = astropy_topocentric_vector(geo_cache, site, "GEOCENTER_VENUS", jd_tdb)
    separation = angular_sep_arcsec(sun, venus)
    sun_radius, venus_radius = astropy_angular_radii(geo_cache, site, jd_tdb)
    threshold = (
        sun_radius + venus_radius
        if kind == "OUTER"
        else sun_radius - venus_radius
    )
    return separation - threshold

def root_list(
    jds: np.ndarray,
    value_function,
) -> list[float]:
    values = np.array([value_function(float(jd)) for jd in jds], dtype=float)
    roots: list[float] = []
    for index in range(len(jds) - 1):
        first = values[index]
        second = values[index + 1]
        if not np.isfinite(first) or not np.isfinite(second):
            continue
        if first == 0.0:
            roots.append(float(jds[index]))
        elif first * second < 0.0:
            roots.append(
                float(
                    brentq(
                        value_function,
                        float(jds[index]),
                        float(jds[index + 1]),
                        xtol=1.0e-13,
                        rtol=1.0e-13,
                        maxiter=100,
                    )
                )
            )
    return roots

def iers0011_pair_fit_window(geo_cache: dict[str, object]) -> tuple[float, float]:
    jds = np.asarray(geo_cache["jd_tdb"], dtype=float)
    outer_roots: list[float] = []
    for site in (CAPE_TOWN, REYKJAVIK):
        outer_roots.extend(
            root_list(
                jds,
                lambda jd, current_site=site: astropy_contact_value(
                    geo_cache, current_site, "OUTER", jd
                ),
            )
        )
    if len(outer_roots) < 4:
        raise RuntimeError(
            f"IERS-0011 pair window could not be derived: {len(outer_roots)} outer roots."
        )
    return min(outer_roots), max(outer_roots)

def geocenter_separation(geo_cache: dict[str, object], jd_tdb: float) -> float:
    return angular_sep_arcsec(
        vector_at(geo_cache, "GEOCENTER_SUN", jd_tdb),
        vector_at(geo_cache, "GEOCENTER_VENUS", jd_tdb),
    )

def geocenter_closest_approach(geo_cache: dict[str, object]) -> float:
    jds = np.asarray(geo_cache["jd_tdb"], dtype=float)
    values = np.array([geocenter_separation(geo_cache, jd) for jd in jds])
    minimum_index = int(np.argmin(values))
    lower = float(jds[max(0, minimum_index - 3)])
    upper = float(jds[min(len(jds) - 1, minimum_index + 3)])
    result = minimize_scalar(
        lambda jd: geocenter_separation(geo_cache, jd),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1.0e-13},
    )
    if not result.success:
        raise RuntimeError("Geocenter closest-approach search failed.")
    return float(result.x)

def fixed_geocenter_basis(
    geo_cache: dict[str, object],
    jd_tdb: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sun = vector_at(geo_cache, "GEOCENTER_SUN", jd_tdb)
    normal = unit(sun)
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if norm(np.cross(reference, normal)) < 1.0e-12:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)
    x_axis = unit(np.cross(reference, normal))
    y_axis = unit(np.cross(normal, x_axis))
    return normal, x_axis, y_axis

def sitecoord_screen_point_arcsec(
    geo_cache: dict[str, object],
    topo_cache: dict[str, object],
    jd_tdb: float,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    normal, x_axis, y_axis = basis
    sun_geo = vector_at(geo_cache, "GEOCENTER_SUN", jd_tdb)
    sun_topo = vector_at(topo_cache, "CAPE_TOWN_SUN", jd_tdb)
    venus_topo = vector_at(topo_cache, "CAPE_TOWN_VENUS", jd_tdb)
    observer = sun_geo - sun_topo
    denominator = float(np.dot(venus_topo, normal))
    if abs(denominator) < 1.0e-14:
        raise RuntimeError("Cape Town Venus ray is nearly parallel to the solar screen.")
    ray_scale = float(np.dot(sun_geo - observer, normal) / denominator)
    hit = observer + ray_scale * venus_topo
    screen_vector = hit - sun_geo
    earth_sun_km = norm(sun_geo)
    return np.array(
        [
            math.atan2(float(np.dot(screen_vector, x_axis)), earth_sun_km)
            * ARCSEC_PER_RAD,
            math.atan2(float(np.dot(screen_vector, y_axis)), earth_sun_km)
            * ARCSEC_PER_RAD,
        ],
        dtype=float,
    )

def sitecoord_separation(
    topo_cache: dict[str, object],
    jd_tdb: float,
) -> float:
    return angular_sep_arcsec(
        vector_at(topo_cache, "CAPE_TOWN_SUN", jd_tdb),
        vector_at(topo_cache, "CAPE_TOWN_VENUS", jd_tdb),
    )

def sitecoord_angular_radii(
    topo_cache: dict[str, object],
    jd_tdb: float,
) -> tuple[float, float]:
    sun = vector_at(topo_cache, "CAPE_TOWN_SUN", jd_tdb)
    venus = vector_at(topo_cache, "CAPE_TOWN_VENUS", jd_tdb)
    return (
        math.atan2(SUN_RADIUS_KM, norm(sun)) * ARCSEC_PER_RAD,
        math.atan2(VENUS_RADIUS_KM, norm(venus)) * ARCSEC_PER_RAD,
    )

def sitecoord_contact_value(
    topo_cache: dict[str, object],
    kind: str,
    jd_tdb: float,
) -> float:
    separation = sitecoord_separation(topo_cache, jd_tdb)
    sun_radius, venus_radius = sitecoord_angular_radii(topo_cache, jd_tdb)
    threshold = (
        sun_radius + venus_radius
        if kind == "OUTER"
        else sun_radius - venus_radius
    )
    return separation - threshold

def sitecoord_contacts(
    topo_cache: dict[str, object],
) -> dict[str, float]:
    jds = np.asarray(topo_cache["jd_tdb"], dtype=float)
    outer = root_list(
        jds,
        lambda jd: sitecoord_contact_value(topo_cache, "OUTER", jd),
    )
    inner = root_list(
        jds,
        lambda jd: sitecoord_contact_value(topo_cache, "INNER", jd),
    )
    if len(outer) < 2 or len(inner) < 2:
        raise RuntimeError(
            f"Cape Town SITE_COORD contacts incomplete: outer={len(outer)}, inner={len(inner)}"
        )
    return {"C1": outer[0], "C2": inner[0], "C3": inner[-1], "C4": outer[-1]}

def sitecoord_closest_approach(
    topo_cache: dict[str, object],
) -> float:
    jds = np.asarray(topo_cache["jd_tdb"], dtype=float)
    values = np.array([sitecoord_separation(topo_cache, jd) for jd in jds])
    minimum_index = int(np.argmin(values))
    lower = float(jds[max(0, minimum_index - 3)])
    upper = float(jds[min(len(jds) - 1, minimum_index + 3)])
    result = minimize_scalar(
        lambda jd: sitecoord_separation(topo_cache, jd),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1.0e-13},
    )
    if not result.success:
        raise RuntimeError("Cape Town closest-approach search failed.")
    return float(result.x)

def pca_fit(points: np.ndarray) -> dict[str, object]:
    array = np.asarray(points, dtype=float)
    center = np.mean(array, axis=0)
    centered = array - center
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    direction = unit(vt[0])
    if direction[0] < 0.0:
        direction = -direction
    normal = np.array([-direction[1], direction[0]], dtype=float)
    cross_track = centered @ normal
    angle = wrap_horizontal(math.degrees(math.atan2(direction[1], direction[0])))
    x = array[:, 0] - np.mean(array[:, 0])
    y = array[:, 1]
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    linear_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return {
        "center": center,
        "direction": direction,
        "normal": normal,
        "angle_deg": angle,
        "rms_arcsec": float(np.sqrt(np.mean(cross_track**2))),
        "linear_r2": linear_r2,
    }

def build_geometry() -> dict[str, object]:
    geo_master = build_geocenter_master()
    topo_master = build_cape_sitecoord_master()
    geo_cache = build_cache(geo_master)
    topo_cache = build_cache(topo_master)

    pair_c1, pair_c4 = iers0011_pair_fit_window(geo_cache)
    geocenter_ca = geocenter_closest_approach(geo_cache)
    basis = fixed_geocenter_basis(geo_cache, geocenter_ca)

    jds = np.asarray(geo_cache["jd_tdb"], dtype=float)
    fit_jds = jds[(jds >= pair_c1) & (jds <= pair_c4)]
    fit_points = np.array(
        [
            sitecoord_screen_point_arcsec(geo_cache, topo_cache, jd, basis)
            for jd in fit_jds
        ],
        dtype=float,
    )
    fit = pca_fit(fit_points)

    contacts = sitecoord_contacts(topo_cache)
    cape_ca = sitecoord_closest_approach(topo_cache)
    event_jds = {
        "C1": contacts["C1"],
        "C2": contacts["C2"],
        "CA": cape_ca,
        "C3": contacts["C3"],
        "C4": contacts["C4"],
    }

    events: dict[str, Event] = {}
    for name, jd in event_jds.items():
        sun_radius, venus_radius = sitecoord_angular_radii(topo_cache, jd)
        events[name] = Event(
            name=name,
            jd_tdb=jd,
            utc=utc_at(jd),
            point_arcsec=sitecoord_screen_point_arcsec(
                geo_cache, topo_cache, jd, basis
            ),
            sun_radius_arcsec=sun_radius,
            venus_radius_arcsec=venus_radius,
        )

    cape_angle = float(fit["angle_deg"])
    delta_video = cape_angle - VIDEO_TLS_ANGLE_DEG

    if abs(cape_angle - ARCHIVED_IERS0011_CAPE_SITECOORD_DEG) > 0.02:
        raise RuntimeError(
            "Cape Town SITE_COORD angle did not reproduce the archived IERS-0011 result: "
            f"computed={cape_angle:.9f}, archived={ARCHIVED_IERS0011_CAPE_SITECOORD_DEG:.9f}"
        )

    return {
        "geo_cache": geo_cache,
        "topo_cache": topo_cache,
        "basis": basis,
        "fit_jds": fit_jds,
        "fit_points_arcsec": fit_points,
        "fit": fit,
        "events": events,
        "pair_c1": pair_c1,
        "pair_c4": pair_c4,
        "geocenter_ca": geocenter_ca,
        "cape_angle_deg": cape_angle,
        "video_angle_deg": VIDEO_TLS_ANGLE_DEG,
        "delta_angle_deg": delta_video,
    }

def build_video_reference_line(
    track_points_rsun: np.ndarray,
    ca_point_rsun: np.ndarray,
) -> np.ndarray:
    video_angle_rad = math.radians(VIDEO_TLS_ANGLE_DEG)
    video_direction = np.array(
        [math.cos(video_angle_rad), math.sin(video_angle_rad)],
        dtype=float,
    )
    cape_direction = unit(track_points_rsun[-1] - track_points_rsun[0])
    along = (track_points_rsun - ca_point_rsun) @ cape_direction
    start = ca_point_rsun + float(np.min(along)) * video_direction
    end = ca_point_rsun + float(np.max(along)) * video_direction
    return np.vstack([start, end])

def write_csv(
    geometry: dict[str, object],
    solar_radius_arcsec: float,
    video_line_rsun: np.ndarray,
) -> None:
    track_points_arcsec = np.asarray(geometry["fit_points_arcsec"], dtype=float)
    fit_jds = np.asarray(geometry["fit_jds"], dtype=float)
    rows: list[dict[str, object]] = []
    for index, (jd, point) in enumerate(zip(fit_jds, track_points_arcsec)):
        rows.append(
            {
                "version": VERSION,
                "record_type": "CAPE_TOWN_SITE_COORD_TRACK",
                "sequence": index,
                "event": "",
                "jd_tdb": jd,
                "utc": utc_at(float(jd)),
                "x_arcsec": point[0],
                "y_arcsec": point[1],
                "x_rsun": point[0] / solar_radius_arcsec,
                "y_rsun": point[1] / solar_radius_arcsec,
                "angle_from_horizontal_deg": geometry["cape_angle_deg"],
                "latitude_deg": CAPE_TOWN["lat_deg"],
                "longitude_deg_east": CAPE_TOWN["lon_deg_east"],
            }
        )
    for name, event in geometry["events"].items():
        rows.append(
            {
                "version": VERSION,
                "record_type": "CAPE_TOWN_EVENT",
                "sequence": np.nan,
                "event": name,
                "jd_tdb": event.jd_tdb,
                "utc": event.utc,
                "x_arcsec": event.point_arcsec[0],
                "y_arcsec": event.point_arcsec[1],
                "x_rsun": event.point_arcsec[0] / solar_radius_arcsec,
                "y_rsun": event.point_arcsec[1] / solar_radius_arcsec,
                "angle_from_horizontal_deg": geometry["cape_angle_deg"],
                "latitude_deg": CAPE_TOWN["lat_deg"],
                "longitude_deg_east": CAPE_TOWN["lon_deg_east"],
            }
        )
    for index, point in enumerate(video_line_rsun):
        rows.append(
            {
                "version": VERSION,
                "record_type": "SDO_VIDEO_TLS_REFERENCE_LINE",
                "sequence": index,
                "event": "",
                "jd_tdb": np.nan,
                "utc": "",
                "x_arcsec": point[0] * solar_radius_arcsec,
                "y_arcsec": point[1] * solar_radius_arcsec,
                "x_rsun": point[0],
                "y_rsun": point[1],
                "angle_from_horizontal_deg": VIDEO_TLS_ANGLE_DEG,
                "latitude_deg": np.nan,
                "longitude_deg_east": np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(TRACK_CSV, index=False, float_format="%.12f")

def render_plot(geometry: dict[str, object]) -> None:
    events: dict[str, Event] = geometry["events"]
    solar_radius_arcsec = events["CA"].sun_radius_arcsec
    track_arcsec = np.asarray(geometry["fit_points_arcsec"], dtype=float)
    track_rsun = track_arcsec / solar_radius_arcsec
    event_rsun = {
        name: event.point_arcsec / solar_radius_arcsec
        for name, event in events.items()
    }
    ca_point = event_rsun["CA"]
    video_line = build_video_reference_line(track_rsun, ca_point)
    write_csv(geometry, solar_radius_arcsec, video_line)

    figure = plt.figure(figsize=(11.0, 7.4), dpi=190, facecolor=BG)
    grid = figure.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[3.25, 1.35],
        wspace=0.08,
    )
    ax = figure.add_subplot(grid[0, 0])
    table_ax = figure.add_subplot(grid[0, 1])
    ax.set_facecolor(BG)
    table_ax.set_facecolor(BG)

    theta = np.linspace(0.0, 2.0 * np.pi, 1800)
    ax.plot(
        np.cos(theta),
        np.sin(theta),
        color=SOLAR,
        linewidth=0.38,
        zorder=1,
    )
    ax.axhline(0.0, color="#555555", linewidth=0.20, zorder=0)
    ax.axvline(0.0, color="#555555", linewidth=0.20, zorder=0)

    ax.plot(
        track_rsun[:, 0],
        track_rsun[:, 1],
        color=JPL_COLOR,
        linewidth=0.72,
        label="Cape Town JPL SITE_COORD",
        zorder=4,
    )
    ax.scatter(
        track_rsun[::8, 0],
        track_rsun[::8, 1],
        s=2.0,
        color=JPL_COLOR,
        linewidths=0,
        zorder=5,
    )
    ax.plot(
        video_line[:, 0],
        video_line[:, 1],
        color=VIDEO_COLOR,
        linewidth=0.90,
        linestyle=(0, (7, 5)),
        label="SDO video TLS reference",
        zorder=7,
    )

    label_offsets = {
        "C1": (-0.070, 0.030),
        "C2": (-0.060, 0.025),
        "CA": (0.020, 0.028),
        "C3": (0.022, -0.036),
        "C4": (0.028, -0.042),
    }
    for name in ("C1", "C2", "CA", "C3", "C4"):
        event = events[name]
        point = event_rsun[name]
        venus_radius_rsun = event.venus_radius_arcsec / solar_radius_arcsec
        ax.add_patch(
            Circle(
                (point[0], point[1]),
                venus_radius_rsun,
                fill=False,
                edgecolor=VENUS_COLOR,
                linewidth=0.30,
                zorder=8,
            )
        )
        ax.scatter(
            [point[0]],
            [point[1]],
            s=5.0,
            color=JPL_COLOR,
            edgecolor=BG,
            linewidth=0.20,
            zorder=9,
        )
        dx, dy = label_offsets[name]
        ax.text(
            point[0] + dx,
            point[1] + dy,
            name,
            color=TEXT,
            fontsize=7.0,
            zorder=10,
        )

    median_y = float(np.median(track_rsun[:, 1]))
    ax.set_xlim(-1.04, 1.04)
    if median_y < 0.0:
        ax.set_ylim(-1.06, 0.08)
    else:
        ax.set_ylim(-0.08, 1.06)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=GRID, linewidth=0.18, alpha=0.75)
    ax.tick_params(colors=MUTED, labelsize=7, width=0.25, length=2.2)
    for spine in ax.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(0.28)

    ax.set_xlabel("Solar-screen X (R_sun)", color=MUTED, fontsize=8)
    ax.set_ylabel("Solar-screen Y (R_sun)", color=MUTED, fontsize=8)
    ax.set_title(
        "2012 Venus Transit — Cape Town SITE_COORD vs SDO Video Angle\n"
        "IERS-0011 method reproduced; JPL track unchanged",
        color=TEXT,
        fontsize=10,
        pad=9,
    )
    legend = ax.legend(
        loc="lower right",
        fontsize=7,
        frameon=True,
        handlelength=3.2,
    )
    legend.get_frame().set_facecolor("#090909")
    legend.get_frame().set_edgecolor("#777777")
    legend.get_frame().set_linewidth(0.35)
    for item in legend.get_texts():
        item.set_color(TEXT)

    table_ax.axis("off")
    table_ax.text(
        0.5,
        0.95,
        "TRACK GEOMETRY",
        ha="center",
        va="center",
        color=TEXT,
        fontsize=10,
        fontweight="bold",
        transform=table_ax.transAxes,
    )
    table_rows = [
        ["θ_CAPE", f"{geometry['cape_angle_deg']:.6f}", "deg"],
        ["θ_VIDEO", f"{geometry['video_angle_deg']:.6f}", "deg"],
        ["Δθ", f"{geometry['delta_angle_deg']:+.6f}", "deg"],
        ["R² linear", f"{geometry['fit']['linear_r2']:.9f}", ""],
        ["Lat", f"{CAPE_TOWN['lat_deg']:.4f}", "deg"],
        ["Lon E", f"{CAPE_TOWN['lon_deg_east']:.4f}", "deg"],
    ]
    table = table_ax.table(
        cellText=table_rows,
        colLabels=["Quantity", "Value", "Units"],
        colWidths=[0.42, 0.36, 0.22],
        bbox=[0.02, 0.46, 0.96, 0.40],
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_facecolor(BG)
        cell.set_edgecolor(TABLE_EDGE if row == 0 else "#333333")
        cell.set_linewidth(0.35 if row == 0 else 0.22)
        cell.get_text().set_color(TEXT)
        if row == 0:
            cell.get_text().set_weight("bold")
        if column == 1:
            cell.get_text().set_ha("right")
            cell.get_text().set_weight("bold")

    table_ax.text(
        0.5,
        0.385,
        "CAPE TOWN CONTACTS",
        ha="center",
        va="center",
        color=TEXT,
        fontsize=9,
        fontweight="bold",
        transform=table_ax.transAxes,
    )
    contacts_rows = [
        [name, events[name].utc[11:23]]
        for name in ("C1", "C2", "CA", "C3", "C4")
    ]
    contacts_table = table_ax.table(
        cellText=contacts_rows,
        colLabels=["Event", "UTC"],
        colWidths=[0.30, 0.70],
        bbox=[0.02, 0.10, 0.96, 0.24],
        cellLoc="left",
        colLoc="left",
    )
    contacts_table.auto_set_font_size(False)
    contacts_table.set_fontsize(7.2)
    for (row, column), cell in contacts_table.get_celld().items():
        cell.set_facecolor(BG)
        cell.set_edgecolor(TABLE_EDGE if row == 0 else "#333333")
        cell.set_linewidth(0.35 if row == 0 else 0.22)
        cell.get_text().set_color(TEXT)
        if row == 0:
            cell.get_text().set_weight("bold")
        if column == 1:
            cell.get_text().set_ha("right")

    table_ax.text(
        0.02,
        0.035,
        "Green: minute-by-minute JPL SITE_COORD.\n"
        "Red dashed: straight line at V0007 video TLS angle.\n"
        "Angles are from horizontal in [-90°, +90°).",
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=6.5,
        transform=table_ax.transAxes,
    )

    figure.savefig(
        PLOT_PNG,
        dpi=320,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
    )
    plt.close(figure)

def main() -> int:
    geometry = build_geometry()
    render_plot(geometry)
    display(IPythonImage(filename=str(PLOT_PNG)))

    print("CODE INPUTS")
    print("Cape Town Horizons SITE_COORD + JPL geocenter vectors")
    print("COMMENTS")
    print("No AI images; only Cape Town is plotted; Reykjavik is used only to reproduce the archived IERS-0011 fit window")
    print("RESULTS")
    print(f"Plot PNG: {PLOT_PNG}")
    print("OUTPUT SUMMARY")
    print(f"CSV: {TRACK_CSV}")
    print("PAPER COMPARISON")
    print("Archived IERS-0011 Cape Town angle used only as a validation target")
    print("EQUATION STATUS")
    print("Angles and contact values are rendered in the PNG")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
# IERS-0012AJ
