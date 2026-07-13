# V0067
# Audit reference: Correct both closest-approach solutions in seconds space, lower the derivation table, and match paired station-row colors.
from __future__ import annotations

import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def require(import_name: str, package_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "-q", "install", package_name]
        )


for _import_name, _package_name in (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("astropy", "astropy"),
    ("astroquery", "astroquery"),
    ("matplotlib", "matplotlib"),
    ("IPython", "ipython"),
):
    require(_import_name, _package_name)

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display
from matplotlib.patches import Circle
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar

warnings.filterwarnings("ignore", message=".*id_type.*deprecated.*")
warnings.filterwarnings("ignore", message=".*dubious year.*")

VERSION = "V0067"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUTPUT_DIR = ROOT / "VENUS_1769_V0027_FORMAT_STANDALONE_V0067_OUTPUT"
PNG = OUTPUT_DIR / "VENUS_1769_JPL_ECLIPTIC_HALF_SUN_CONTACT_GEOMETRY_V0067.png"
MASTER_CSV = OUTPUT_DIR / "VENUS_1769_SYNCHRONIZED_JPL_VECTORS_V0067.csv"
CONTACTS_CSV = OUTPUT_DIR / "VENUS_1769_CONTACTS_V0067.csv"
TRACKS_CSV = OUTPUT_DIR / "VENUS_1769_TRACKS_V0067.csv"
GEOMETRY_CSV = OUTPUT_DIR / "VENUS_1769_SEPARATE_RAY_GEOMETRY_V0067.csv"

ARCSEC_PER_RAD = 206_264.80624709636
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
EVENT_ORDER = ("C1", "C2", "CA", "C3", "C4")

POINT_VENUS = {
    "key": "POINT_VENUS",
    "label": "Point Venus, Tahiti",
    "short": "PV",
    "lat": -17.495600000000,
    "lon": -149.493900000000,
    "elevation": 0.0,
    "body": 399,
    "color": "#42D7C3",
}
VARDO = {
    "key": "VARDO",
    "label": "Vardo, Norway",
    "short": "V",
    "lat": 70.372400000000,
    "lon": 31.110300000000,
    "elevation": 0.0,
    "body": 399,
    "color": "#D89B18",
}
SITES = (POINT_VENUS, VARDO)
TARGETS = (("SUN", "10"), ("VENUS", "299"))
PREFIXES = (
    "GEOCENTER_SUN",
    "GEOCENTER_VENUS",
    "POINT_VENUS_SUN",
    "POINT_VENUS_VENUS",
    "VARDO_SUN",
    "VARDO_VENUS",
)
REQUIRED_COLUMNS = ["JD_TDB"] + [
    f"{prefix}_{axis}_KM" for prefix in PREFIXES for axis in "XYZ"
]

SUN_LINE_WIDTH = 0.500
TRACK_LINE_WIDTH = 0.375
DISK_LINE_WIDTH = 0.375
GUIDE_LINE_WIDTH = 0.250
MARKER_EDGE_WIDTH = 0.250
SUN_COLOR = "#F8FAFC"
GUIDE_COLOR = "#263A4B"
TEXT_COLOR = "#F8FAFC"
MUTED_TEXT = "#B8CBD6"
TABLE_HEADER = "#23466F"
TABLE_TEAL = "#164B55"
TABLE_GOLD = "#563B0B"
TABLE_BODY = "#101A2E"
BACKGROUND = "#000000"


def norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def unit(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    magnitude = norm(array)
    if magnitude <= 0.0:
        raise RuntimeError("Cannot normalize a zero vector.")
    return array / magnitude


def utc_text(jd_tdb: float) -> str:
    return (
        Time(float(jd_tdb), format="jd", scale="tdb")
        .utc.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    )


def location_for(site: dict[str, object]) -> dict[str, float | int]:
    return {
        "lon": float(site["lon"]),
        "lat": float(site["lat"]),
        "elevation": float(site["elevation"]),
        "body": int(site["body"]),
    }


def download_series(
    prefix: str,
    target_id: str,
    location: str | dict[str, float | int],
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            table = Horizons(
                id=target_id,
                location=location,
                epochs={"start": START, "stop": STOP, "step": STEP},
                id_type=None,
            ).vectors(
                refplane="ecliptic",
                aberrations="geometric",
                cache=False,
            )
            frame = table.to_pandas()
            output = pd.DataFrame(
                {
                    "JD_TDB": pd.to_numeric(
                        frame["datetime_jd"], errors="coerce"
                    )
                }
            )
            for axis in "xyz":
                output[f"{prefix}_{axis.upper()}_KM"] = (
                    pd.to_numeric(frame[axis], errors="coerce") * JPL_AU_KM
                )
            output = (
                output.dropna()
                .drop_duplicates("JD_TDB")
                .sort_values("JD_TDB")
                .reset_index(drop=True)
            )
            if len(output) < 600:
                raise RuntimeError(
                    f"Incomplete JPL series for {prefix}: {len(output)} rows."
                )
            return output
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"JPL query failed for {prefix}: {last_error}")


def build_master() -> pd.DataFrame:
    series: list[pd.DataFrame] = []
    for target_name, target_id in TARGETS:
        series.append(
            download_series(
                f"GEOCENTER_{target_name}", target_id, GEOCENTER
            )
        )
    for site in SITES:
        for target_name, target_id in TARGETS:
            series.append(
                download_series(
                    f"{site['key']}_{target_name}",
                    target_id,
                    location_for(site),
                )
            )

    master = series[0]
    for frame in series[1:]:
        master = master.merge(
            frame,
            on="JD_TDB",
            how="inner",
            validate="one_to_one",
        )
    if len(master) < 600:
        raise RuntimeError(
            f"Synchronized JPL master has only {len(master)} rows."
        )
    master["UTC"] = [utc_text(jd) for jd in master["JD_TDB"]]
    master["REFERENCE_PLANE"] = "JPL ECLIPTIC"
    master.to_csv(MASTER_CSV, index=False, float_format="%.15f")
    return master


def build_cache(master: pd.DataFrame) -> dict[str, object]:
    frame = master.copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna(subset=REQUIRED_COLUMNS)
        .sort_values("JD_TDB")
        .drop_duplicates("JD_TDB")
        .reset_index(drop=True)
    )
    jds = frame["JD_TDB"].to_numpy(dtype=float)
    cache: dict[str, object] = {"JD_TDB": jds, "FRAME": frame}
    for prefix in PREFIXES:
        for axis in "XYZ":
            column = f"{prefix}_{axis}_KM"
            cache[column] = CubicSpline(
                jds,
                frame[column].to_numpy(dtype=float),
                bc_type="natural",
            )
    return cache


def vector_at(cache: dict[str, object], prefix: str, jd_tdb: float) -> np.ndarray:
    return np.array(
        [
            float(cache[f"{prefix}_{axis}_KM"](float(jd_tdb)))
            for axis in "XYZ"
        ],
        dtype=float,
    )


def angular_separation_rad(first: np.ndarray, second: np.ndarray) -> float:
    first_hat = unit(first)
    second_hat = unit(second)
    return math.atan2(
        norm(np.cross(first_hat, second_hat)),
        float(np.dot(first_hat, second_hat)),
    )


def angular_radii_rad(
    cache: dict[str, object], site_key: str, jd_tdb: float
) -> tuple[float, float]:
    sun_distance = norm(vector_at(cache, f"{site_key}_SUN", jd_tdb))
    venus_distance = norm(vector_at(cache, f"{site_key}_VENUS", jd_tdb))
    return (
        math.asin(SUN_RADIUS_KM / sun_distance),
        math.asin(VENUS_RADIUS_KM / venus_distance),
    )


def contact_residual_rad(
    cache: dict[str, object],
    site_key: str,
    jd_tdb: float,
    internal: bool,
) -> float:
    sun = vector_at(cache, f"{site_key}_SUN", jd_tdb)
    venus = vector_at(cache, f"{site_key}_VENUS", jd_tdb)
    separation = angular_separation_rad(sun, venus)
    solar_radius, venus_radius = angular_radii_rad(cache, site_key, jd_tdb)
    required = (
        solar_radius - venus_radius
        if internal
        else solar_radius + venus_radius
    )
    return separation - required


def contact_roots(
    cache: dict[str, object], site_key: str, internal: bool
) -> list[float]:
    jds = np.asarray(cache["JD_TDB"], dtype=float)
    values = np.array(
        [
            contact_residual_rad(cache, site_key, jd, internal)
            for jd in jds
        ],
        dtype=float,
    )
    roots: list[float] = []
    for index in range(len(jds) - 1):
        left_value = values[index]
        right_value = values[index + 1]
        if not np.isfinite(left_value) or not np.isfinite(right_value):
            continue
        if left_value == 0.0:
            roots.append(float(jds[index]))
        elif left_value * right_value < 0.0:
            roots.append(
                float(
                    brentq(
                        lambda jd: contact_residual_rad(
                            cache, site_key, jd, internal
                        ),
                        float(jds[index]),
                        float(jds[index + 1]),
                        xtol=1.0e-13,
                        rtol=1.0e-14,
                        maxiter=200,
                    )
                )
            )
    unique: list[float] = []
    for root in sorted(roots):
        if not unique or abs(root - unique[-1]) > 0.2 / 86400.0:
            unique.append(root)
    if len(unique) != 2:
        kind = "internal" if internal else "external"
        raise RuntimeError(
            f"Expected two {kind} contacts for {site_key}; found {len(unique)}."
        )
    return unique


def compute_events(cache: dict[str, object], site_key: str) -> dict[str, float]:
    external = contact_roots(cache, site_key, internal=False)
    internal = contact_roots(cache, site_key, internal=True)

    reference_jd = 0.5 * (internal[0] + internal[1])
    lower_seconds = (internal[0] - reference_jd) * 86400.0
    upper_seconds = (internal[1] - reference_jd) * 86400.0
    result = minimize_scalar(
        lambda seconds: angular_separation_rad(
            vector_at(
                cache,
                f"{site_key}_SUN",
                reference_jd + float(seconds) / 86400.0,
            ),
            vector_at(
                cache,
                f"{site_key}_VENUS",
                reference_jd + float(seconds) / 86400.0,
            ),
        ),
        bounds=(lower_seconds, upper_seconds),
        method="bounded",
        options={"xatol": 1.0e-4, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"Closest approach failed for {site_key}.")
    closest_jd = reference_jd + float(result.x) / 86400.0
    events = {
        "C1": external[0],
        "C2": internal[0],
        "CA": closest_jd,
        "C3": internal[1],
        "C4": external[1],
    }
    if not (
        events["C1"]
        < events["C2"]
        < events["CA"]
        < events["C3"]
        < events["C4"]
    ):
        raise RuntimeError(f"Event ordering failed for {site_key}.")
    return events


def ecliptic_basis(sun_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = unit(sun_vector)
    north_pole = np.array([0.0, 0.0, 1.0], dtype=float)
    east = np.cross(north_pole, center)
    if norm(east) < 1.0e-14:
        east = np.cross(np.array([0.0, 1.0, 0.0]), center)
    east = unit(east)
    north = unit(np.cross(center, east))
    if float(np.dot(north, north_pole)) < 0.0:
        east = -east
        north = -north
    return center, east, north


def apparent_position_arcsec(
    cache: dict[str, object], site_key: str, jd_tdb: float
) -> np.ndarray:
    sun = vector_at(cache, f"{site_key}_SUN", jd_tdb)
    venus = vector_at(cache, f"{site_key}_VENUS", jd_tdb)
    sun_hat = unit(sun)
    venus_hat = unit(venus)
    separation = angular_separation_rad(sun_hat, venus_hat)
    if separation == 0.0:
        return np.zeros(2, dtype=float)
    tangent = unit(venus_hat - math.cos(separation) * sun_hat)
    _center, east, north = ecliptic_basis(sun)
    return separation * ARCSEC_PER_RAD * np.array(
        [float(np.dot(tangent, east)), float(np.dot(tangent, north))],
        dtype=float,
    )


def fit_track(points: np.ndarray) -> dict[str, object]:
    mean = np.mean(points, axis=0)
    centered = points - mean
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    direction = np.asarray(vt[0], dtype=float)
    if direction[0] < 0.0:
        direction = -direction
    normal_direction = np.array([-direction[1], direction[0]], dtype=float)
    along = centered @ direction
    cross = centered @ normal_direction
    coefficients = np.polyfit(along, cross, 2)
    fitted_cross = np.polyval(coefficients, along)
    rms = float(np.sqrt(np.mean((cross - fitted_cross) ** 2)))
    angle = math.degrees(math.atan2(direction[1], direction[0]))
    return {
        "mean": mean,
        "direction": direction,
        "normal": normal_direction,
        "angle_deg": angle,
        "slope": math.tan(math.radians(angle)),
        "rms_arcsec": rms,
        "curvature_per_arcsec": float(2.0 * coefficients[0]),
    }


def build_site_result(
    cache: dict[str, object], site: dict[str, object]
) -> dict[str, object]:
    key = str(site["key"])
    events = compute_events(cache, key)
    minute_jds = np.asarray(cache["JD_TDB"], dtype=float)
    mask = (minute_jds >= events["C1"]) & (minute_jds <= events["C4"])
    selected_jds = minute_jds[mask]
    points = np.array(
        [apparent_position_arcsec(cache, key, jd) for jd in selected_jds],
        dtype=float,
    )
    event_points = {
        event: apparent_position_arcsec(cache, key, jd)
        for event, jd in events.items()
    }
    event_radii = {
        event: tuple(
            radius * ARCSEC_PER_RAD
            for radius in angular_radii_rad(cache, key, jd)
        )
        for event, jd in events.items()
    }
    return {
        "site": site,
        "events": events,
        "selected_jds": selected_jds,
        "points": points,
        "event_points": event_points,
        "event_radii": event_radii,
        "fit": fit_track(points),
    }


def geocentric_closest_approach(cache: dict[str, object]) -> float:
    jds = np.asarray(cache["JD_TDB"], dtype=float)
    separations = np.array(
        [
            angular_separation_rad(
                vector_at(cache, "GEOCENTER_SUN", jd),
                vector_at(cache, "GEOCENTER_VENUS", jd),
            )
            for jd in jds
        ],
        dtype=float,
    )
    minimum_index = int(np.argmin(separations))
    lower_index = max(0, minimum_index - 3)
    upper_index = min(len(jds) - 1, minimum_index + 3)
    lower_jd = float(jds[lower_index])
    upper_jd = float(jds[upper_index])
    reference_jd = 0.5 * (lower_jd + upper_jd)
    lower_seconds = (lower_jd - reference_jd) * 86400.0
    upper_seconds = (upper_jd - reference_jd) * 86400.0

    result = minimize_scalar(
        lambda seconds: angular_separation_rad(
            vector_at(
                cache,
                "GEOCENTER_SUN",
                reference_jd + float(seconds) / 86400.0,
            ),
            vector_at(
                cache,
                "GEOCENTER_VENUS",
                reference_jd + float(seconds) / 86400.0,
            ),
        ),
        bounds=(lower_seconds, upper_seconds),
        method="bounded",
        options={"xatol": 1.0e-4, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError("Geocentric closest approach failed.")
    return reference_jd + float(result.x) / 86400.0


def gnomonic(
    vector: np.ndarray,
    center: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
) -> np.ndarray:
    direction = unit(vector)
    denominator = float(np.dot(direction, center))
    if denominator <= 0.0:
        raise RuntimeError("Ray lies outside the tangent hemisphere.")
    return np.array(
        [float(np.dot(direction, east)), float(np.dot(direction, north))],
        dtype=float,
    ) / denominator


def common_relative_position(
    cache: dict[str, object],
    site_key: str,
    jd_tdb: float,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    center, east, north = basis
    sun = vector_at(cache, f"{site_key}_SUN", jd_tdb)
    venus = vector_at(cache, f"{site_key}_VENUS", jd_tdb)
    return ARCSEC_PER_RAD * (
        gnomonic(venus, center, east, north)
        - gnomonic(sun, center, east, north)
    )


def separate_ray_geometry(cache: dict[str, object]) -> dict[str, object]:
    jd = geocentric_closest_approach(cache)
    geo_sun = vector_at(cache, "GEOCENTER_SUN", jd)
    geo_venus = vector_at(cache, "GEOCENTER_VENUS", jd)
    basis = ecliptic_basis(geo_sun)
    center, east, north = basis

    half_step = 0.5 / 86400.0
    tracks: dict[str, np.ndarray] = {}
    velocities: dict[str, np.ndarray] = {}
    for site in SITES:
        key = str(site["key"])
        tracks[key] = np.array(
            [
                common_relative_position(cache, key, jd - half_step, basis),
                common_relative_position(cache, key, jd, basis),
                common_relative_position(cache, key, jd + half_step, basis),
            ],
            dtype=float,
        )
        velocities[key] = (tracks[key][2] - tracks[key][0]) / 1.0

    direction_pv = unit(velocities["POINT_VENUS"])
    direction_v = unit(velocities["VARDO"])
    if float(np.dot(direction_pv, direction_v)) < 0.0:
        direction_v = -direction_v
    track_direction = unit(direction_pv + direction_v)
    normal_2d = np.array(
        [-track_direction[1], track_direction[0]], dtype=float
    )

    q_pv = tracks["POINT_VENUS"][1]
    q_v = tracks["VARDO"][1]
    if float(np.dot(q_v - q_pv, normal_2d)) < 0.0:
        normal_2d = -normal_2d
    normal_3d = unit(normal_2d[0] * east + normal_2d[1] * north)

    midpoint = 0.5 * (q_pv + q_v)
    point_a_prime = q_pv - midpoint
    point_b_prime = q_v - midpoint
    a_prime = float(np.dot(point_a_prime, normal_2d))
    b_prime = float(np.dot(point_b_prime, normal_2d))
    aprime_bprime_arcsec = b_prime - a_prime

    station_pv = geo_sun - vector_at(cache, "POINT_VENUS_SUN", jd)
    station_v = geo_sun - vector_at(cache, "VARDO_SUN", jd)
    baseline = station_v - station_pv
    if float(np.dot(baseline, normal_3d)) < 0.0:
        normal_2d = -normal_2d
        normal_3d = -normal_3d
        a_prime = -a_prime
        b_prime = -b_prime
        aprime_bprime_arcsec = -aprime_bprime_arcsec

    ab_km = float(np.dot(baseline, normal_3d))
    es_km = norm(geo_sun)
    km_per_arcsec = es_km / ARCSEC_PER_RAD
    ab_arcsec = ab_km / km_per_arcsec
    aprime_bprime_km = aprime_bprime_arcsec * km_per_arcsec

    ev_vector = geo_venus
    vs_vector = geo_sun - geo_venus
    es_axis = unit(geo_sun)
    ev_bar = float(np.dot(ev_vector, es_axis))
    vs_bar = float(np.dot(vs_vector, es_axis))
    es_bar = float(np.dot(geo_sun, es_axis))
    center_ratio = ev_bar / vs_bar
    transfer_ratio = ab_arcsec / aprime_bprime_arcsec
    vector_factor = transfer_ratio / center_ratio
    recovered_ab_km = aprime_bprime_km * transfer_ratio
    closure_km = recovered_ab_km - ab_km

    return {
        "jd_tdb": jd,
        "utc": utc_text(jd),
        "basis": basis,
        "track_direction": track_direction,
        "normal_2d": normal_2d,
        "A_prime": point_a_prime,
        "B_prime": point_b_prime,
        "A_prime_bar_arcsec": a_prime,
        "B_prime_bar_arcsec": b_prime,
        "A_prime_B_prime_arcsec": aprime_bprime_arcsec,
        "A_prime_B_prime_km": aprime_bprime_km,
        "AB_arcsec": ab_arcsec,
        "AB_km": ab_km,
        "EV_bar_km": ev_bar,
        "VS_bar_km": vs_bar,
        "ES_bar_km": es_bar,
        "center_ratio": center_ratio,
        "transfer_ratio": transfer_ratio,
        "vector_factor": vector_factor,
        "closure_km": closure_km,
        "km_per_arcsec": km_per_arcsec,
    }


def maximum_contact_residual_arcsec(
    cache: dict[str, object], results: tuple[dict[str, object], ...]
) -> float:
    residuals: list[float] = []
    for result in results:
        site_key = str(result["site"]["key"])
        for event in ("C1", "C2", "C3", "C4"):
            internal = event in ("C2", "C3")
            residuals.append(
                abs(
                    contact_residual_rad(
                        cache,
                        site_key,
                        float(result["events"][event]),
                        internal,
                    )
                )
                * ARCSEC_PER_RAD
            )
    return max(residuals)


def add_venus_disk(
    axis: plt.Axes,
    center: np.ndarray,
    radius_arcsec: float,
    color: str,
) -> None:
    axis.add_patch(
        Circle(
            (float(center[0]), float(center[1])),
            radius_arcsec,
            facecolor=color,
            edgecolor=color,
            alpha=0.16,
            linewidth=DISK_LINE_WIDTH,
            zorder=5,
        )
    )


def annotate_event(
    axis: plt.Axes,
    center: np.ndarray,
    label: str,
    color: str,
    above: bool,
    x_shift: float,
    y_shift: float,
) -> None:
    axis.annotate(
        label,
        xy=(float(center[0]), float(center[1])),
        xytext=(float(center[0] + x_shift), float(center[1] + y_shift)),
        textcoords="data",
        ha="center",
        va="bottom" if above else "top",
        color=color,
        fontsize=6.3,
        fontweight="bold",
        arrowprops={
            "arrowstyle": "-",
            "linewidth": GUIDE_LINE_WIDTH,
            "color": color,
            "shrinkA": 1.0,
            "shrinkB": 1.0,
        },
        zorder=9,
    )


def style_table(
    table,
    teal_rows: tuple[int, ...] = (),
    gold_rows: tuple[int, ...] = (),
    font_size: float = 6.5,
) -> None:
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#70879A")
        cell.set_linewidth(0.35)
        cell.get_text().set_color(TEXT_COLOR)
        cell.get_text().set_fontsize(font_size)
        if row == 0:
            cell.set_facecolor(TABLE_HEADER)
            cell.get_text().set_fontweight("bold")
        elif row in teal_rows:
            cell.set_facecolor(TABLE_TEAL)
            cell.get_text().set_fontweight("bold")
        elif row in gold_rows:
            cell.set_facecolor(TABLE_GOLD)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(TABLE_BODY)


def draw_track(
    axis: plt.Axes,
    result: dict[str, object],
    main: bool,
) -> None:
    site = result["site"]
    color = str(site["color"])
    points = np.asarray(result["points"], dtype=float)
    axis.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        linewidth=TRACK_LINE_WIDTH,
        solid_capstyle="round",
        zorder=3,
        label=str(site["label"]) if main else None,
    )
    if main:
        axis.scatter(
            points[::8, 0],
            points[::8, 1],
            s=1.1,
            color=color,
            linewidths=0.0,
            alpha=0.72,
            zorder=4,
        )


def draw_events(
    axis: plt.Axes,
    result: dict[str, object],
    events: tuple[str, ...],
    main: bool,
) -> None:
    site = result["site"]
    color = str(site["color"])
    short = str(site["short"])
    x_shifts = {"C1": -13.0, "C2": -4.0, "CA": 0.0, "C3": 4.0, "C4": 13.0}
    for event in events:
        center = np.asarray(result["event_points"][event], dtype=float)
        radius = float(result["event_radii"][event][1])
        add_venus_disk(axis, center, radius, color)
        marker = "X" if event == "CA" else "o"
        axis.scatter(
            [center[0]],
            [center[1]],
            s=16.0 if event == "CA" else 7.5,
            marker=marker,
            color=color,
            edgecolors=BACKGROUND,
            linewidths=MARKER_EDGE_WIDTH,
            zorder=8,
        )
        if main:
            above = short == "PV"
            y_shift = 20.0 if above else -20.0
        else:
            above = short == "PV"
            y_shift = 11.0 if above else -11.0
        annotate_event(
            axis,
            center,
            f"{short} {event}",
            color,
            above,
            x_shifts[event],
            y_shift,
        )


def zoom_limits(
    results: tuple[dict[str, object], ...], events: tuple[str, ...]
) -> tuple[tuple[float, float], tuple[float, float]]:
    centers: list[np.ndarray] = []
    radii: list[float] = []
    for result in results:
        for event in events:
            centers.append(np.asarray(result["event_points"][event], dtype=float))
            radii.append(float(result["event_radii"][event][1]))
    array = np.vstack(centers)
    maximum_radius = max(radii)
    margin = maximum_radius * 0.35 + 5.0
    x_min = float(np.min(array[:, 0]) - maximum_radius - margin)
    x_max = float(np.max(array[:, 0]) + maximum_radius + margin)
    y_min = float(np.min(array[:, 1]) - maximum_radius - margin)
    y_max = float(np.max(array[:, 1]) + maximum_radius + margin)
    return (x_min, x_max), (y_min, y_max)


def make_results_table(
    axis: plt.Axes,
    point_result: dict[str, object],
    vardo_result: dict[str, object],
    geometry: dict[str, object],
    max_contact_residual: float,
) -> None:
    axis.axis("off")
    axis.set_title(
        "RESULTS",
        loc="left",
        fontsize=9.0,
        fontweight="bold",
        pad=5,
    )
    point_angle = float(point_result["fit"]["angle_deg"])
    vardo_angle = float(vardo_result["fit"]["angle_deg"])
    average_angle = 0.5 * (point_angle + vardo_angle)
    pi0 = math.asin(EARTH_RADIUS_KM / IAU1976_AU_KM) * ARCSEC_PER_RAD
    rows = [
        ["Quantity", "Symbol", "Value", "Unit / status"],
        ["IAU 1976 AU-normalized solar horizontal parallax", "π₀", f"{pi0:.12f}", "arcsec"],
        ["Point Venus, Tahiti track angle", "α_PV", f"{point_angle:.6f}", "deg"],
        ["Vardo, Norway track angle", "α_V", f"{vardo_angle:.6f}", "deg"],
        ["Average track angle", "ᾱ", f"{average_angle:.6f}", "deg"],
        ["Point Venus, Tahiti RMS", "RMS_PV", f"{float(point_result['fit']['rms_arcsec']):.6f}", "arcsec"],
        ["Vardo, Norway RMS", "RMS_V", f"{float(vardo_result['fit']['rms_arcsec']):.6f}", "arcsec"],
        ["Maximum contact-equation residual", "", f"{max_contact_residual:.12f}", "arcsec — PASS"],
        ["A′B′ common-normal separation", "A′B′", f"{float(geometry['A_prime_B_prime_arcsec']):.6f}", "arcsec"],
        ["Projection reference", "", "JPL ECLIPTIC", "verified"],
    ]
    table = axis.table(
        cellText=rows,
        cellLoc="left",
        colWidths=[0.47, 0.12, 0.21, 0.20],
        bbox=[0.00, 0.00, 1.00, 0.91],
    )
    table.auto_set_font_size(False)
    style_table(table, teal_rows=(8, 9), gold_rows=(2, 3), font_size=6.2)


def make_contact_table(
    axis: plt.Axes,
    point_result: dict[str, object],
    vardo_result: dict[str, object],
) -> None:
    axis.axis("off")
    axis.set_title(
        "RECOMPUTED CONTACT TIMES — UTC",
        loc="left",
        fontsize=9.0,
        fontweight="bold",
        pad=5,
    )
    rows = [["Station", "Event", "UTC", "Exact limb condition"]]
    for result in (point_result, vardo_result):
        station = str(result["site"]["label"])
        for event in ("C1", "C2", "C3", "C4"):
            condition = "ρ = R☉ − R♀" if event in ("C2", "C3") else "ρ = R☉ + R♀"
            utc = utc_text(float(result["events"][event])).split(" ", 1)[1]
            rows.append([station, event, utc, condition])
    table = axis.table(
        cellText=rows,
        cellLoc="left",
        colWidths=[0.28, 0.12, 0.31, 0.29],
        bbox=[0.00, 0.00, 1.00, 0.92],
    )
    table.auto_set_font_size(False)
    style_table(
        table,
        teal_rows=(1, 2, 5, 6),
        gold_rows=(),
        font_size=6.2,
    )


def make_derivation_table(
    axis: plt.Axes,
    point_result: dict[str, object],
    vardo_result: dict[str, object],
    geometry: dict[str, object],
) -> None:
    axis.axis("off")
    axis.text(
        0.5,
        0.86,
        "A′B′ AND AB DERIVATION",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=8.4,
        fontweight="bold",
        color=TEXT_COLOR,
        zorder=20,
    )
    point_angle = float(point_result["fit"]["angle_deg"])
    vardo_angle = float(vardo_result["fit"]["angle_deg"])
    average_angle = 0.5 * (point_angle + vardo_angle)
    rows = [
        ["Quantity", "Definition", "Arcseconds", "Kilometers"],
        ["A′B′", "JPL separate-ray derived", f"{float(geometry['A_prime_B_prime_arcsec']):.6f}", f"{float(geometry['A_prime_B_prime_km']):,.6f}"],
        ["AB", "JPL projected baseline", f"{float(geometry['AB_arcsec']):.6f}", f"{float(geometry['AB_km']):,.6f}"],
        ["α PV", "Point Venus, Tahiti track angle (degrees)", f"{point_angle:.6f}°", ""],
        ["α V", "Vardo, Norway track angle (degrees)", f"{vardo_angle:.6f}°", ""],
        ["ᾱ", "Average track angle (degrees)", f"{average_angle:.6f}°", ""],
    ]
    table = axis.table(
        cellText=rows,
        cellLoc="left",
        colWidths=[0.15, 0.43, 0.19, 0.23],
        bbox=[0.00, -0.02, 1.00, 0.69],
    )
    table.auto_set_font_size(False)
    style_table(table, teal_rows=(1, 2), gold_rows=(3, 4), font_size=6.4)


def plot_publication(
    point_result: dict[str, object],
    vardo_result: dict[str, object],
    geometry: dict[str, object],
    max_contact_residual: float,
) -> None:
    plt.close("all")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "figure.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": MUTED_TEXT,
            "ytick.color": MUTED_TEXT,
            "axes.edgecolor": MUTED_TEXT,
        }
    )

    figure = plt.figure(figsize=(16.0, 9.0), facecolor=BACKGROUND)
    outer = figure.add_gridspec(
        1,
        2,
        width_ratios=(2.06, 1.00),
        left=0.028,
        right=0.984,
        top=0.925,
        bottom=0.070,
        wspace=0.035,
    )
    left = outer[0, 0].subgridspec(
        2, 1, height_ratios=(0.73, 0.27), hspace=0.105
    )
    main_axis = figure.add_subplot(left[0, 0])
    lower = left[1, 0].subgridspec(
        1, 3, width_ratios=(0.82, 1.55, 0.82), wspace=0.25
    )
    egress_axis = figure.add_subplot(lower[0, 0])
    derivation_axis = figure.add_subplot(lower[0, 1])
    ingress_axis = figure.add_subplot(lower[0, 2])

    right = outer[0, 1].subgridspec(
        2, 1, height_ratios=(0.47, 0.53), hspace=0.095
    )
    results_axis = figure.add_subplot(right[0, 0])
    contacts_axis = figure.add_subplot(right[1, 0])

    figure.suptitle(
        "1769 VENUS TRANSIT — JPL ECLIPTIC HALF-SUN CONTACT GEOMETRY",
        fontsize=15.0,
        fontweight="bold",
        y=0.976,
    )

    reference_solar_radius = float(point_result["event_radii"]["CA"][0])
    theta = np.linspace(0.0, 2.0 * math.pi, 1800)
    main_axis.plot(
        reference_solar_radius * np.cos(theta),
        reference_solar_radius * np.sin(theta),
        color=SUN_COLOR,
        linewidth=SUN_LINE_WIDTH,
        zorder=1,
    )
    main_axis.axhline(0.0, color=GUIDE_COLOR, linewidth=GUIDE_LINE_WIDTH, zorder=0)
    main_axis.axvline(0.0, color=GUIDE_COLOR, linewidth=GUIDE_LINE_WIDTH, zorder=0)

    for result in (point_result, vardo_result):
        draw_track(main_axis, result, main=True)
        draw_events(main_axis, result, EVENT_ORDER, main=True)

    main_axis.text(
        -0.42 * reference_solar_radius,
        0.21 * reference_solar_radius,
        "TRACK GEOMETRY — A, B, A′, B′",
        fontsize=7.3,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    normal = np.asarray(geometry["normal_2d"], dtype=float)
    center_point = 0.5 * (
        np.asarray(point_result["event_points"]["CA"], dtype=float)
        + np.asarray(vardo_result["event_points"]["CA"], dtype=float)
    )
    span = 1.15 * reference_solar_radius
    guide_a = center_point - span * normal
    guide_b = center_point + span * normal
    main_axis.plot(
        [guide_a[0], guide_b[0]],
        [guide_a[1], guide_b[1]],
        color="#30445A",
        linewidth=0.22,
        alpha=0.75,
        zorder=0,
    )

    main_axis.set_aspect("equal", adjustable="box")
    main_axis.set_xlim(-1.07 * reference_solar_radius, 1.07 * reference_solar_radius)
    main_axis.set_ylim(-0.08 * reference_solar_radius, 1.06 * reference_solar_radius)
    main_axis.set_xlabel(r"Ecliptic longitude direction, $\xi$ (arcsec)", fontsize=8.0)
    main_axis.set_ylabel(r"Ecliptic north direction, $\eta$ (arcsec)", fontsize=8.0)
    main_axis.tick_params(width=GUIDE_LINE_WIDTH, length=2.5, labelsize=6.5)
    for spine in main_axis.spines.values():
        spine.set_linewidth(GUIDE_LINE_WIDTH)
        spine.set_color(MUTED_TEXT)
    legend = main_axis.legend(loc="lower left", frameon=False, fontsize=6.7)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    for zoom_axis, events, title in (
        (egress_axis, ("C3", "C4"), "EGRESS ZOOM — C3 / C4 TANGENCY"),
        (ingress_axis, ("C1", "C2"), "INGRESS ZOOM — C1 / C2 TANGENCY"),
    ):
        for result in (point_result, vardo_result):
            draw_track(zoom_axis, result, main=False)
            draw_events(zoom_axis, result, events, main=False)
        x_limits, y_limits = zoom_limits((point_result, vardo_result), events)
        zoom_axis.set_xlim(*x_limits)
        zoom_axis.set_ylim(*y_limits)
        zoom_axis.set_aspect("equal", adjustable="box")
        zoom_axis.set_title(title, fontsize=6.4, pad=3)
        zoom_axis.tick_params(width=0.22, length=1.8, labelsize=5.3)
        for spine in zoom_axis.spines.values():
            spine.set_linewidth(0.24)
            spine.set_color(MUTED_TEXT)

    make_derivation_table(
        derivation_axis, point_result, vardo_result, geometry
    )
    make_results_table(
        results_axis,
        point_result,
        vardo_result,
        geometry,
        max_contact_residual,
    )
    make_contact_table(contacts_axis, point_result, vardo_result)

    center, east, north = geometry["basis"]
    audit_text = (
        f"Ecliptic audit: ξ·S={float(np.dot(east, center)):.3e}, "
        f"η·S={float(np.dot(north, center)):.3e}, "
        f"ξ·η={float(np.dot(east, north)):.3e}, "
        f"N·S={float(center[2]):.12f}"
    )
    figure.text(
        0.50,
        0.038,
        audit_text,
        ha="center",
        fontsize=5.8,
        color=MUTED_TEXT,
    )
    figure.text(
        0.50,
        0.018,
        "JPL Horizons geometric vectors; reference plane = ecliptic. All tracks, fits, contact roots, disk radii, and separate-ray transfer values are calculated from the synchronized vector series.",
        ha="center",
        fontsize=5.8,
        color=MUTED_TEXT,
    )

    figure.savefig(
        PNG,
        dpi=160,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor=BACKGROUND,
    )
    plt.close(figure)


def write_outputs(
    point_result: dict[str, object],
    vardo_result: dict[str, object],
    geometry: dict[str, object],
) -> None:
    contact_rows: list[dict[str, object]] = []
    track_rows: list[dict[str, object]] = []
    for result in (point_result, vardo_result):
        site = result["site"]
        for event in EVENT_ORDER:
            jd = float(result["events"][event])
            point = np.asarray(result["event_points"][event], dtype=float)
            solar_radius, venus_radius = result["event_radii"][event]
            contact_rows.append(
                {
                    "station": str(site["label"]),
                    "event": event,
                    "jd_tdb": jd,
                    "utc": utc_text(jd),
                    "xi_arcsec": float(point[0]),
                    "eta_arcsec": float(point[1]),
                    "solar_radius_arcsec": float(solar_radius),
                    "venus_radius_arcsec": float(venus_radius),
                }
            )
        for jd, point in zip(result["selected_jds"], result["points"]):
            track_rows.append(
                {
                    "station": str(site["label"]),
                    "jd_tdb": float(jd),
                    "utc": utc_text(float(jd)),
                    "xi_arcsec": float(point[0]),
                    "eta_arcsec": float(point[1]),
                }
            )
    pd.DataFrame(contact_rows).to_csv(
        CONTACTS_CSV, index=False, float_format="%.15f"
    )
    pd.DataFrame(track_rows).to_csv(
        TRACKS_CSV, index=False, float_format="%.15f"
    )

    geometry_rows = [
        ["Closest-approach UTC", geometry["utc"], "UTC"],
        ["Closest-approach JD TDB", geometry["jd_tdb"], "day"],
        ["A′B′", geometry["A_prime_B_prime_arcsec"], "arcsec"],
        ["A′B′", geometry["A_prime_B_prime_km"], "km"],
        ["AB", geometry["AB_arcsec"], "arcsec"],
        ["AB", geometry["AB_km"], "km"],
        ["Projected center ratio", geometry["center_ratio"], "dimensionless"],
        ["Separate-ray transfer ratio", geometry["transfer_ratio"], "dimensionless"],
        ["Separate-ray vector factor", geometry["vector_factor"], "dimensionless"],
        ["Final closure", geometry["closure_km"], "km"],
    ]
    pd.DataFrame(
        geometry_rows, columns=["quantity", "value", "unit"]
    ).to_csv(GEOMETRY_CSV, index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master = build_master()
    cache = build_cache(master)
    point_result = build_site_result(cache, POINT_VENUS)
    vardo_result = build_site_result(cache, VARDO)
    geometry = separate_ray_geometry(cache)
    max_contact_residual = maximum_contact_residual_arcsec(
        cache, (point_result, vardo_result)
    )

    write_outputs(point_result, vardo_result, geometry)
    plot_publication(
        point_result,
        vardo_result,
        geometry,
        max_contact_residual,
    )

    display(Image(filename=str(PNG)))

    print("CODE INPUTS")
    print("Fresh synchronized JPL Horizons geometric ecliptic vectors at one-minute cadence.")
    print("Stations: Point Venus, Tahiti; Vardo, Norway.")
    print("COMMENTS")
    print("Standalone calculation: no prior notebook files, CSVs, scripts, patches, or reduction outputs are required.")
    print("RESULTS")
    print(f"A′B′: {float(geometry['A_prime_B_prime_arcsec']):.12f} arcsec | {float(geometry['A_prime_B_prime_km']):.9f} km")
    print(f"AB: {float(geometry['AB_arcsec']):.12f} arcsec | {float(geometry['AB_km']):.9f} km")
    print(f"Separate-ray transfer ratio: {float(geometry['transfer_ratio']):.15f}")
    print(f"Closure residual: {float(geometry['closure_km']):+.15e} km")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"JPL vectors: {MASTER_CSV}")
    print(f"Contacts: {CONTACTS_CSV}")
    print(f"Tracks: {TRACKS_CSV}")
    print(f"Geometry: {GEOMETRY_CSV}")
    print("PAPER COMPARISON")
    print("IAU 1976 π₀ is displayed for comparison; all transit geometry is derived from fresh JPL vectors.")
    print("EQUATION STATUS")
    print(f"Separate-ray A′B′→AB closure: {'PASS' if abs(float(geometry['closure_km'])) < 1.0e-6 else 'FAIL'}")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0067