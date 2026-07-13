# V0021
# Audit reference: JPL ecliptic Mercury contact plot with visible scale-true disks, ordered labels, closest approach, and compact A/B/A-prime/B-prime table.
from __future__ import annotations

import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def ensure_package(import_name: str, pip_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "-q", "install", pip_name]
        )


for _import_name, _pip_name in (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("astropy", "astropy"),
    ("astroquery", "astroquery"),
    ("matplotlib", "matplotlib"),
):
    ensure_package(_import_name, _pip_name)

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar

VERSION = "V0021"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUTPUT_DIR = ROOT / "MERCURY_PARALLAX_PUBLICATION_V0021_OUTPUT"
OUTPUT_PNG = OUTPUT_DIR / "MERCURY_1769_CONTACT_GEOMETRY_V0021.png"
OUTPUT_CONTACTS_CSV = OUTPUT_DIR / "MERCURY_1769_CONTACTS_V0021.csv"
OUTPUT_TRACKS_CSV = OUTPUT_DIR / "MERCURY_1769_TRACKS_V0021.csv"
OUTPUT_GEOMETRY_CSV = OUTPUT_DIR / "MERCURY_1769_AB_APRIME_BPRIME_V0021.csv"
MASTER_CSV = ROOT / "MERCURY_BAY_VARDO_1769_JPL_ECLIPTIC_V0021.csv"

ARCSEC_PER_RAD = 206_264.80624709636
AU_KM = 149_597_870.000000
SUN_RADIUS_KM = 695_700.000000
MERCURY_RADIUS_KM = 2_439.700000

START = "1769-11-09 12:00"
STOP = "1769-11-10 08:00"
STEP = "1m"
GEOCENTER = "500@399"

SITE_BLUE = {
    "label": "Mercury Bay",
    "short": "MB",
    "key": "MERCURY_BAY",
    "lat": -36.783333333333,
    "lon": 175.933333333333,
    "elevation": 0.0,
    "body": 399,
}
SITE_YELLOW = {
    "label": "Vardø",
    "short": "V",
    "key": "VARDO",
    "lat": 70.370600000000,
    "lon": 31.110700000000,
    "elevation": 0.0,
    "body": 399,
}
SITES = (SITE_BLUE, SITE_YELLOW)
TARGETS = (("SUN", "10"), ("MERCURY", "199"))
PREFIXES = (
    "GEOCENTER_SUN",
    "GEOCENTER_MERCURY",
    "MERCURY_BAY_SUN",
    "MERCURY_BAY_MERCURY",
    "VARDO_SUN",
    "VARDO_MERCURY",
)
REQUIRED_COLUMNS = ["JD_TDB"] + [
    f"{prefix}_{axis}_KM" for prefix in PREFIXES for axis in "XYZ"
]

SUN_LINE_WIDTH = 0.500
TRACK_LINE_WIDTH = 0.375
DISK_LINE_WIDTH = 0.375
GUIDE_LINE_WIDTH = 0.250
MARKER_EDGE_WIDTH = 0.250

BLUE = "#35A7FF"
YELLOW = "#FFD54F"
SUN_COLOR = "#F8FAFC"
GUIDE_COLOR = "#375364"
TEXT_COLOR = "#F8FAFC"
MUTED_TEXT = "#B8CBD6"
TABLE_HEADER = "#173A4C"
TABLE_BLUE = "#123B52"
TABLE_YELLOW = "#4A3A10"
TABLE_BODY = "#0B1720"
BACKGROUND = "#020609"

EVENT_ORDER = ("C1", "C2", "CA", "C3", "C4")


def norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    magnitude = norm(vector)
    if magnitude == 0.0:
        raise RuntimeError("Cannot normalize a zero vector.")
    return vector / magnitude


def site_location(site: dict[str, object]) -> dict[str, object]:
    return {
        "lon": float(site["lon"]),
        "lat": float(site["lat"]),
        "elevation": float(site["elevation"]),
        "body": int(site["body"]),
    }


def utc_text(jd_tdb: float) -> str:
    return (
        Time(jd_tdb, format="jd", scale="tdb")
        .utc.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    )


def download_series(
    prefix: str,
    target_id: str,
    location: str | dict[str, object],
) -> pd.DataFrame:
    query = Horizons(
        id=target_id,
        location=location,
        epochs={"start": START, "stop": STOP, "step": STEP},
    )
    table = query.vectors(
        refplane="ecliptic",
        aberrations="geometric",
    )
    frame = table.to_pandas()
    output = pd.DataFrame(
        {"JD_TDB": pd.to_numeric(frame["datetime_jd"], errors="coerce")}
    )
    for axis in "xyz":
        output[f"{prefix}_{axis.upper()}_KM"] = (
            pd.to_numeric(frame[axis], errors="coerce") * AU_KM
        )
    return (
        output.dropna()
        .drop_duplicates("JD_TDB")
        .sort_values("JD_TDB")
        .reset_index(drop=True)
    )


def build_master() -> pd.DataFrame:
    series: list[pd.DataFrame] = []
    for target_name, target_id in TARGETS:
        series.append(
            download_series(
                f"GEOCENTER_{target_name}",
                target_id,
                GEOCENTER,
            )
        )
    for site in SITES:
        location = site_location(site)
        for target_name, target_id in TARGETS:
            series.append(
                download_series(
                    f"{site['key']}_{target_name}",
                    target_id,
                    location,
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
    master["REFERENCE_PLANE"] = "JPL ECLIPTIC"
    master["UTC"] = [utc_text(value) for value in master["JD_TDB"]]
    master.to_csv(MASTER_CSV, index=False, float_format="%.15f")
    return master


def master_is_compatible(path: Path) -> bool:
    try:
        sample = pd.read_csv(path, nrows=3)
    except Exception:
        return False
    if not all(column in sample.columns for column in REQUIRED_COLUMNS):
        return False
    if "REFERENCE_PLANE" in sample.columns:
        joined = " ".join(sample["REFERENCE_PLANE"].astype(str)).upper()
        return "ECLIPTIC" in joined
    return "ECLIPTIC" in path.name.upper()


def load_or_build_master() -> tuple[pd.DataFrame, str]:
    if MASTER_CSV.is_file() and master_is_compatible(MASTER_CSV):
        return pd.read_csv(MASTER_CSV), str(MASTER_CSV)

    candidates: list[Path] = []
    for root, directories, files in os.walk(ROOT):
        directories[:] = [
            directory
            for directory in directories
            if directory != "drive" and not directory.startswith(".")
        ]
        for filename in files:
            path = Path(root) / filename
            if (
                filename.lower().endswith(".csv")
                and master_is_compatible(path)
            ):
                candidates.append(path)

    if candidates:
        selected = max(candidates, key=lambda item: item.stat().st_mtime)
        return pd.read_csv(selected), str(selected)

    return build_master(), "NEW JPL HORIZONS ECLIPTIC DOWNLOAD"


def build_cache(frame: pd.DataFrame) -> dict[str, object]:
    data = frame.copy()
    data["JD_TDB"] = pd.to_numeric(data["JD_TDB"], errors="coerce")
    for column in REQUIRED_COLUMNS[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = (
        data.dropna(subset=REQUIRED_COLUMNS)
        .sort_values("JD_TDB")
        .drop_duplicates("JD_TDB")
        .reset_index(drop=True)
    )
    if len(data) < 20:
        raise RuntimeError(
            "The synchronized JPL ecliptic master contains too few rows."
        )

    jds = data["JD_TDB"].to_numpy(dtype=float)
    cache: dict[str, object] = {
        "JD_TDB": jds,
        "FRAME": data,
    }
    for prefix in PREFIXES:
        for axis in "XYZ":
            column = f"{prefix}_{axis}_KM"
            cache[column] = CubicSpline(
                jds,
                data[column].to_numpy(dtype=float),
                bc_type="natural",
            )
    return cache


def vector_at(
    cache: dict[str, object],
    prefix: str,
    jd_tdb: float,
) -> np.ndarray:
    return np.array(
        [
            float(cache[f"{prefix}_{axis}_KM"](jd_tdb))
            for axis in "XYZ"
        ],
        dtype=float,
    )


def angular_separation_rad(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    cosine = float(
        np.clip(np.dot(unit(first), unit(second)), -1.0, 1.0)
    )
    return math.acos(cosine)


def angular_radii_rad(
    cache: dict[str, object],
    site_key: str,
    jd_tdb: float,
) -> tuple[float, float]:
    sun_distance = norm(
        vector_at(cache, f"{site_key}_SUN", jd_tdb)
    )
    mercury_distance = norm(
        vector_at(cache, f"{site_key}_MERCURY", jd_tdb)
    )
    return (
        math.asin(SUN_RADIUS_KM / sun_distance),
        math.asin(MERCURY_RADIUS_KM / mercury_distance),
    )


def contact_residual(
    cache: dict[str, object],
    site_key: str,
    jd_tdb: float,
    internal: bool,
) -> float:
    sun = vector_at(cache, f"{site_key}_SUN", jd_tdb)
    mercury = vector_at(cache, f"{site_key}_MERCURY", jd_tdb)
    separation = angular_separation_rad(sun, mercury)
    solar_radius, mercury_radius = angular_radii_rad(
        cache,
        site_key,
        jd_tdb,
    )
    required = (
        solar_radius - mercury_radius
        if internal
        else solar_radius + mercury_radius
    )
    return separation - required


def contact_roots(
    cache: dict[str, object],
    site_key: str,
    internal: bool,
) -> list[float]:
    jds = np.asarray(cache["JD_TDB"], dtype=float)
    residuals = np.array(
        [
            contact_residual(cache, site_key, jd, internal)
            for jd in jds
        ],
        dtype=float,
    )
    roots: list[float] = []
    for index in range(len(jds) - 1):
        left_value = residuals[index]
        right_value = residuals[index + 1]
        if not np.isfinite(left_value) or not np.isfinite(right_value):
            continue
        if left_value == 0.0:
            roots.append(float(jds[index]))
        elif left_value * right_value < 0.0:
            roots.append(
                float(
                    brentq(
                        lambda value: contact_residual(
                            cache,
                            site_key,
                            value,
                            internal,
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
            f"Expected two {kind} contacts for {site_key}; "
            f"found {len(unique)}."
        )
    return unique


def compute_events(
    cache: dict[str, object],
    site_key: str,
) -> dict[str, float]:
    external = contact_roots(cache, site_key, internal=False)
    internal = contact_roots(cache, site_key, internal=True)
    contacts = {
        "C1": external[0],
        "C2": internal[0],
        "C3": internal[1],
        "C4": external[1],
    }

    result = minimize_scalar(
        lambda jd: angular_separation_rad(
            vector_at(cache, f"{site_key}_SUN", jd),
            vector_at(cache, f"{site_key}_MERCURY", jd),
        ),
        bounds=(contacts["C2"], contacts["C3"]),
        method="bounded",
        options={"xatol": 1.0e-13, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(
            f"Closest-approach solution failed for {site_key}."
        )

    events = {
        "C1": contacts["C1"],
        "C2": contacts["C2"],
        "CA": float(result.x),
        "C3": contacts["C3"],
        "C4": contacts["C4"],
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


def ecliptic_local_basis(
    sun_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sun_hat = unit(sun_vector)
    ecliptic_north = np.array([0.0, 0.0, 1.0], dtype=float)
    xi = unit(np.cross(ecliptic_north, sun_hat))
    eta = unit(np.cross(sun_hat, xi))
    if float(np.dot(eta, ecliptic_north)) < 0.0:
        xi = -xi
        eta = -eta
    return sun_hat, xi, eta


def apparent_position_arcsec(
    cache: dict[str, object],
    site_key: str,
    jd_tdb: float,
) -> np.ndarray:
    sun = vector_at(cache, f"{site_key}_SUN", jd_tdb)
    mercury = vector_at(cache, f"{site_key}_MERCURY", jd_tdb)
    sun_hat = unit(sun)
    mercury_hat = unit(mercury)
    separation = angular_separation_rad(sun_hat, mercury_hat)
    if separation == 0.0:
        return np.zeros(2, dtype=float)

    tangent_direction = unit(
        mercury_hat - math.cos(separation) * sun_hat
    )
    _sun_hat, xi, eta = ecliptic_local_basis(sun_hat)
    return (
        separation
        * ARCSEC_PER_RAD
        * np.array(
            [
                float(np.dot(tangent_direction, xi)),
                float(np.dot(tangent_direction, eta)),
            ],
            dtype=float,
        )
    )


def fit_track(points: np.ndarray) -> dict[str, object]:
    mean = np.mean(points, axis=0)
    centered = points - mean
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    if direction[0] < 0.0:
        direction = -direction
    normal_direction = np.array(
        [-direction[1], direction[0]],
        dtype=float,
    )
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
    cache: dict[str, object],
    site: dict[str, object],
) -> dict[str, object]:
    key = str(site["key"])
    events = compute_events(cache, key)
    minute_jds = np.asarray(cache["JD_TDB"], dtype=float)
    mask = (
        (minute_jds >= events["C1"])
        & (minute_jds <= events["C4"])
    )
    selected_jds = minute_jds[mask]
    points = np.array(
        [
            apparent_position_arcsec(cache, key, jd)
            for jd in selected_jds
        ],
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


def geocentric_closest_approach(
    cache: dict[str, object],
) -> float:
    jds = np.asarray(cache["JD_TDB"], dtype=float)
    result = minimize_scalar(
        lambda jd: angular_separation_rad(
            vector_at(cache, "GEOCENTER_SUN", jd),
            vector_at(cache, "GEOCENTER_MERCURY", jd),
        ),
        bounds=(float(jds[0]), float(jds[-1])),
        method="bounded",
        options={"xatol": 1.0e-13, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(
            "Geocentric closest-approach solution failed."
        )
    return float(result.x)


def project_vector_km(
    vector: np.ndarray,
    xi: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            float(np.dot(vector, xi)),
            float(np.dot(vector, eta)),
        ],
        dtype=float,
    )


def apparent_mercury_common_basis(
    topocentric_sun: np.ndarray,
    topocentric_mercury: np.ndarray,
    xi: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    sun_hat = unit(topocentric_sun)
    mercury_hat = unit(topocentric_mercury)
    separation = angular_separation_rad(sun_hat, mercury_hat)
    if separation == 0.0:
        return np.zeros(2, dtype=float)
    tangent_direction = unit(
        mercury_hat - math.cos(separation) * sun_hat
    )
    tangent_vector = separation * tangent_direction
    return (
        ARCSEC_PER_RAD
        * np.array(
            [
                float(np.dot(tangent_vector, xi)),
                float(np.dot(tangent_vector, eta)),
            ],
            dtype=float,
        )
    )


def centered_pair(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    midpoint = 0.5 * (first + second)
    return first - midpoint, second - midpoint


def common_epoch_geometry(
    cache: dict[str, object],
) -> dict[str, object]:
    jd_common = geocentric_closest_approach(cache)
    geocentric_sun = vector_at(
        cache,
        "GEOCENTER_SUN",
        jd_common,
    )
    _sun_hat, xi, eta = ecliptic_local_basis(geocentric_sun)

    sun_blue = vector_at(
        cache,
        "MERCURY_BAY_SUN",
        jd_common,
    )
    sun_yellow = vector_at(cache, "VARDO_SUN", jd_common)
    mercury_blue = vector_at(
        cache,
        "MERCURY_BAY_MERCURY",
        jd_common,
    )
    mercury_yellow = vector_at(
        cache,
        "VARDO_MERCURY",
        jd_common,
    )

    observer_blue = geocentric_sun - sun_blue
    observer_yellow = geocentric_sun - sun_yellow
    a_absolute = project_vector_km(observer_blue, xi, eta)
    b_absolute = project_vector_km(observer_yellow, xi, eta)
    point_a, point_b = centered_pair(a_absolute, b_absolute)

    a_prime_absolute = apparent_mercury_common_basis(
        sun_blue,
        mercury_blue,
        xi,
        eta,
    )
    b_prime_absolute = apparent_mercury_common_basis(
        sun_yellow,
        mercury_yellow,
        xi,
        eta,
    )
    point_a_prime, point_b_prime = centered_pair(
        a_prime_absolute,
        b_prime_absolute,
    )

    return {
        "jd_tdb": jd_common,
        "utc": utc_text(jd_common),
        "A": point_a,
        "B": point_b,
        "A_prime": point_a_prime,
        "B_prime": point_b_prime,
        "AB_km": norm(point_b - point_a),
        "A_prime_B_prime_arcsec": norm(
            point_b_prime - point_a_prime
        ),
    }


def add_mercury_disk(
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
) -> None:
    y_shift = 24.0 if above else -24.0
    vertical_alignment = "bottom" if above else "top"
    axis.annotate(
        label,
        xy=(float(center[0]), float(center[1])),
        xytext=(
            float(center[0] + x_shift),
            float(center[1] + y_shift),
        ),
        textcoords="data",
        ha="center",
        va=vertical_alignment,
        color=color,
        fontsize=6.7,
        fontweight="bold",
        arrowprops={
            "arrowstyle": "-",
            "linewidth": GUIDE_LINE_WIDTH,
            "color": color,
            "shrinkA": 1.5,
            "shrinkB": 1.5,
        },
        zorder=9,
    )


def style_table(table) -> None:
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#537080")
        cell.set_linewidth(GUIDE_LINE_WIDTH)
        cell.get_text().set_color(TEXT_COLOR)
        cell.get_text().set_fontsize(7.0)
        if row == 0:
            cell.set_facecolor(TABLE_HEADER)
            cell.get_text().set_fontweight("bold")
        elif row in (1, 3, 5):
            cell.set_facecolor(TABLE_BLUE)
        elif row in (2, 4, 6):
            cell.set_facecolor(TABLE_YELLOW)
        else:
            cell.set_facecolor(TABLE_BODY)


def make_geometry_table(
    axis: plt.Axes,
    geometry: dict[str, object],
    blue_result: dict[str, object],
    yellow_result: dict[str, object],
) -> None:
    axis.axis("off")
    point_a = np.asarray(geometry["A"], dtype=float)
    point_b = np.asarray(geometry["B"], dtype=float)
    point_a_prime = np.asarray(
        geometry["A_prime"],
        dtype=float,
    )
    point_b_prime = np.asarray(
        geometry["B_prime"],
        dtype=float,
    )
    blue_angle = float(blue_result["fit"]["angle_deg"])
    yellow_angle = float(yellow_result["fit"]["angle_deg"])
    average_angle = 0.5 * (blue_angle + yellow_angle)

    rows = [
        ["Symbol", "Definition", "ξ / Value", "η", "Unit"],
        [
            "A",
            "Mercury Bay projected observer point",
            f"{point_a[0]:,.6f}",
            f"{point_a[1]:,.6f}",
            "km",
        ],
        [
            "B",
            "Vardø projected observer point",
            f"{point_b[0]:,.6f}",
            f"{point_b[1]:,.6f}",
            "km",
        ],
        [
            "A′",
            "Mercury Bay apparent Mercury point",
            f"{point_a_prime[0]:.9f}",
            f"{point_a_prime[1]:.9f}",
            "arcsec",
        ],
        [
            "B′",
            "Vardø apparent Mercury point",
            f"{point_b_prime[0]:.9f}",
            f"{point_b_prime[1]:.9f}",
            "arcsec",
        ],
        [
            "α blue",
            "Mercury Bay plotted track angle",
            f"{blue_angle:.6f}",
            "",
            "deg",
        ],
        [
            "α yellow",
            "Vardø plotted track angle",
            f"{yellow_angle:.6f}",
            "",
            "deg",
        ],
        [
            "ᾱ",
            "Average plotted track angle",
            f"{average_angle:.6f}",
            "",
            "deg",
        ],
        [
            "A′B′",
            "Simultaneous apparent separation",
            f"{float(geometry['A_prime_B_prime_arcsec']):.9f}",
            "",
            "arcsec",
        ],
    ]

    table = axis.table(
        cellText=rows,
        cellLoc="left",
        colWidths=[0.09, 0.47, 0.18, 0.16, 0.10],
        bbox=[0.035, 0.02, 0.93, 0.96],
    )
    table.auto_set_font_size(False)
    style_table(table)


def plot_publication(
    blue_result: dict[str, object],
    yellow_result: dict[str, object],
    geometry: dict[str, object],
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

    figure = plt.figure(figsize=(15.8, 10.2), facecolor=BACKGROUND)
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=(0.80, 0.20),
        left=0.055,
        right=0.975,
        top=0.915,
        bottom=0.055,
        hspace=0.055,
    )
    axis = figure.add_subplot(grid[0, 0])
    table_axis = figure.add_subplot(grid[1, 0])

    figure.suptitle(
        "1769 MERCURY TRANSIT — JPL ECLIPTIC CONTACT GEOMETRY",
        fontsize=16,
        fontweight="bold",
        y=0.970,
    )

    reference_solar_radius = float(
        blue_result["event_radii"]["CA"][0]
    )
    theta = np.linspace(0.0, 2.0 * math.pi, 1800)
    axis.plot(
        reference_solar_radius * np.cos(theta),
        reference_solar_radius * np.sin(theta),
        color=SUN_COLOR,
        linewidth=SUN_LINE_WIDTH,
        zorder=1,
    )
    axis.axhline(
        0.0,
        color=GUIDE_COLOR,
        linewidth=GUIDE_LINE_WIDTH,
        zorder=0,
    )
    axis.axvline(
        0.0,
        color=GUIDE_COLOR,
        linewidth=GUIDE_LINE_WIDTH,
        zorder=0,
    )

    x_shifts = {
        "C1": -18.0,
        "C2": -7.0,
        "CA": 0.0,
        "C3": 7.0,
        "C4": 18.0,
    }

    for result, color, above in (
        (blue_result, BLUE, True),
        (yellow_result, YELLOW, False),
    ):
        points = np.asarray(result["points"], dtype=float)
        axis.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=TRACK_LINE_WIDTH,
            solid_capstyle="round",
            zorder=3,
            label=str(result["site"]["label"]),
        )
        axis.scatter(
            points[::8, 0],
            points[::8, 1],
            s=1.2,
            color=color,
            linewidths=0.0,
            alpha=0.72,
            zorder=4,
        )

        for event in EVENT_ORDER:
            center = np.asarray(
                result["event_points"][event],
                dtype=float,
            )
            mercury_radius = float(
                result["event_radii"][event][1]
            )
            add_mercury_disk(
                axis,
                center,
                mercury_radius,
                color,
            )
            marker = "X" if event == "CA" else "o"
            marker_size = 18.0 if event == "CA" else 9.0
            axis.scatter(
                [center[0]],
                [center[1]],
                s=marker_size,
                marker=marker,
                color=color,
                edgecolors=BACKGROUND,
                linewidths=MARKER_EDGE_WIDTH,
                zorder=8,
            )
            annotate_event(
                axis,
                center,
                event,
                color,
                above,
                x_shifts[event],
            )

    all_points = np.vstack(
        [blue_result["points"], yellow_result["points"]]
    )
    mean_y = float(np.mean(all_points[:, 1]))
    if mean_y >= 0.0:
        y_limits = (
            -0.08 * reference_solar_radius,
            1.06 * reference_solar_radius,
        )
        legend_location = "lower left"
    else:
        y_limits = (
            -1.06 * reference_solar_radius,
            0.08 * reference_solar_radius,
        )
        legend_location = "upper left"

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(
        -1.07 * reference_solar_radius,
        1.07 * reference_solar_radius,
    )
    axis.set_ylim(*y_limits)
    axis.set_xlabel(
        r"Ecliptic longitude direction, $\xi$ (arcsec)",
        fontsize=9,
    )
    axis.set_ylabel(
        r"Ecliptic north direction, $\eta$ (arcsec)",
        fontsize=9,
    )
    axis.set_title(
        "SCALE-TRUE MERCURY DISKS AT C1, C2, CLOSEST APPROACH, C3, AND C4",
        fontsize=10,
        pad=8,
    )
    axis.tick_params(
        width=GUIDE_LINE_WIDTH,
        length=2.5,
        labelsize=7.5,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(GUIDE_LINE_WIDTH)
        spine.set_color(MUTED_TEXT)

    legend = axis.legend(
        loc=legend_location,
        frameon=False,
        fontsize=8,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    make_geometry_table(
        table_axis,
        geometry,
        blue_result,
        yellow_result,
    )

    figure.text(
        0.5,
        0.018,
        f"Common A/B epoch: {geometry['utc']} UTC | "
        "JPL Horizons geometric vectors | ecliptic reference plane",
        ha="center",
        fontsize=7.5,
        color=MUTED_TEXT,
    )

    figure.savefig(
        OUTPUT_PNG,
        dpi=320,
        bbox_inches="tight",
        pad_inches=0.06,
        facecolor=BACKGROUND,
    )
    plt.close(figure)


def write_csv_outputs(
    blue_result: dict[str, object],
    yellow_result: dict[str, object],
    geometry: dict[str, object],
) -> None:
    contact_rows: list[dict[str, object]] = []
    track_rows: list[dict[str, object]] = []

    for result in (blue_result, yellow_result):
        site = result["site"]
        for event in EVENT_ORDER:
            jd_tdb = float(result["events"][event])
            point = np.asarray(
                result["event_points"][event],
                dtype=float,
            )
            solar_radius, mercury_radius = result["event_radii"][event]
            contact_rows.append(
                {
                    "station": str(site["label"]),
                    "event": event,
                    "jd_tdb": jd_tdb,
                    "utc": utc_text(jd_tdb),
                    "x_arcsec": float(point[0]),
                    "y_arcsec": float(point[1]),
                    "solar_radius_arcsec": float(solar_radius),
                    "mercury_radius_arcsec": float(mercury_radius),
                }
            )

        for jd_tdb, point in zip(
            result["selected_jds"],
            result["points"],
        ):
            track_rows.append(
                {
                    "station": str(site["label"]),
                    "jd_tdb": float(jd_tdb),
                    "utc": utc_text(float(jd_tdb)),
                    "x_arcsec": float(point[0]),
                    "y_arcsec": float(point[1]),
                }
            )

    pd.DataFrame(contact_rows).to_csv(
        OUTPUT_CONTACTS_CSV,
        index=False,
        float_format="%.15f",
    )
    pd.DataFrame(track_rows).to_csv(
        OUTPUT_TRACKS_CSV,
        index=False,
        float_format="%.15f",
    )

    geometry_rows = []
    for symbol, key, unit in (
        ("A", "A", "km"),
        ("B", "B", "km"),
        ("A′", "A_prime", "arcsec"),
        ("B′", "B_prime", "arcsec"),
    ):
        point = np.asarray(geometry[key], dtype=float)
        geometry_rows.append(
            {
                "symbol": symbol,
                "x": float(point[0]),
                "y": float(point[1]),
                "unit": unit,
                "common_jd_tdb": float(geometry["jd_tdb"]),
                "common_utc": str(geometry["utc"]),
            }
        )
    pd.DataFrame(geometry_rows).to_csv(
        OUTPUT_GEOMETRY_CSV,
        index=False,
        float_format="%.15f",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master, source = load_or_build_master()
    cache = build_cache(master)

    blue_result = build_site_result(cache, SITE_BLUE)
    yellow_result = build_site_result(cache, SITE_YELLOW)
    geometry = common_epoch_geometry(cache)

    write_csv_outputs(
        blue_result,
        yellow_result,
        geometry,
    )
    plot_publication(
        blue_result,
        yellow_result,
        geometry,
    )

    from IPython.display import Image, display
    display(Image(filename=str(OUTPUT_PNG)))

    print(f"JPL source: {source}")
    print(f"PNG: {OUTPUT_PNG}")
    print(
        datetime.now(LOCAL_TZ).strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        )
    )
    print(VERSION)


if __name__ == "__main__":
    main()
# V0021
