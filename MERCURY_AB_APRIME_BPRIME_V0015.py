# V0015
# Audit reference: Plot the common-epoch observer pair A/B in kilometers and apparent Mercury pair A-prime/B-prime in arcseconds from JPL ecliptic vectors.
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
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pip_name])


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
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

VERSION = "V0015"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUTPUT_DIR = ROOT / "MERCURY_PARALLAX_PUBLICATION_V0015_OUTPUT"
OUTPUT_PNG = OUTPUT_DIR / "MERCURY_AB_APRIME_BPRIME_COMMON_EPOCH_V0015.png"
OUTPUT_CSV = OUTPUT_DIR / "MERCURY_AB_APRIME_BPRIME_COMMON_EPOCH_V0015.csv"
MASTER_CSV = ROOT / "MERCURY_BAY_VARDO_1769_JPL_ECLIPTIC_V0015.csv"

AU_KM = 149_597_870.000000
ARCSEC_PER_RAD = 206_264.80624709636
START = "1769-11-09 12:00"
STOP = "1769-11-10 08:00"
STEP = "1m"
GEOCENTER = "500@399"

SITE_MB = {
    "label": "Mercury Bay",
    "key": "MERCURY_BAY",
    "lat": -36.783333333333,
    "lon": 175.933333333333,
    "elevation": 0.0,
    "body": 399,
}
SITE_V = {
    "label": "Vardø",
    "key": "VARDO",
    "lat": 70.370600000000,
    "lon": 31.110700000000,
    "elevation": 0.0,
    "body": 399,
}
SITES = (SITE_MB, SITE_V)
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

COLOR_A = "#22D3EE"
COLOR_B = "#F59E0B"
COLOR_SEGMENT = "#E2E8F0"
COLOR_GUIDE = "#475569"
COLOR_HEADER = "#1E3A5F"
COLOR_BODY = "#0F172A"
LINE_WIDTH = 0.75
POINT_SIZE = 52.0


def norm(vector) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    magnitude = norm(vector)
    if magnitude == 0.0:
        raise RuntimeError("Cannot normalize a zero vector.")
    return vector / magnitude


def site_location(site: dict[str, object]):
    return {
        "lon": float(site["lon"]),
        "lat": float(site["lat"]),
        "elevation": float(site["elevation"]),
        "body": int(site["body"]),
    }


def download_series(prefix: str, target_id: str, location) -> pd.DataFrame:
    query = Horizons(
        id=target_id,
        location=location,
        epochs={"start": START, "stop": STOP, "step": STEP},
    )
    table = query.vectors(refplane="ecliptic", aberrations="geometric")
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
            download_series(f"GEOCENTER_{target_name}", target_id, GEOCENTER)
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
    master["UTC"] = [
        Time(value, format="jd", scale="tdb").utc.datetime.strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]
        for value in master["JD_TDB"]
    ]
    master.to_csv(MASTER_CSV, index=False, float_format="%.15f")
    return master


def master_is_compatible(path: Path) -> bool:
    try:
        sample = pd.read_csv(path, nrows=2)
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
            if filename.lower().endswith(".csv") and master_is_compatible(path):
                candidates.append(path)
    if candidates:
        selected = max(candidates, key=lambda path: path.stat().st_mtime)
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
        raise RuntimeError("The synchronized ecliptic master contains too few rows.")

    jds = data["JD_TDB"].to_numpy(float)
    cache: dict[str, object] = {"JD_TDB": jds}
    for prefix in PREFIXES:
        for axis in "XYZ":
            column = f"{prefix}_{axis}_KM"
            cache[column] = CubicSpline(
                jds,
                data[column].to_numpy(float),
                bc_type="natural",
            )
    return cache


def vector_at(cache: dict[str, object], prefix: str, jd_tdb: float) -> np.ndarray:
    return np.array(
        [
            float(cache[f"{prefix}_{axis}_KM"](jd_tdb))
            for axis in "XYZ"
        ],
        dtype=float,
    )


def angular_separation(first, second) -> float:
    cosine = float(np.clip(np.dot(unit(first), unit(second)), -1.0, 1.0))
    return math.acos(cosine)


def common_epoch(cache: dict[str, object]) -> float:
    jds = np.asarray(cache["JD_TDB"], dtype=float)
    result = minimize_scalar(
        lambda jd: angular_separation(
            vector_at(cache, "GEOCENTER_SUN", jd),
            vector_at(cache, "GEOCENTER_MERCURY", jd),
        ),
        bounds=(float(jds[0]), float(jds[-1])),
        method="bounded",
        options={"xatol": 1.0e-13, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError("Geocentric closest-approach epoch solution failed.")
    return float(result.x)


def ecliptic_solar_basis(
    geocentric_sun_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sun_hat = unit(geocentric_sun_vector)
    ecliptic_north = np.array([0.0, 0.0, 1.0], dtype=float)
    xi = unit(np.cross(ecliptic_north, sun_hat))
    eta = unit(np.cross(sun_hat, xi))
    if float(np.dot(eta, ecliptic_north)) < 0.0:
        xi = -xi
        eta = -eta
    return sun_hat, xi, eta


def project_vector_km(
    vector: np.ndarray,
    xi: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    return np.array(
        [float(np.dot(vector, xi)), float(np.dot(vector, eta))],
        dtype=float,
    )


def apparent_mercury_arcsec(
    topocentric_sun: np.ndarray,
    topocentric_mercury: np.ndarray,
    xi: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    sun_hat = unit(topocentric_sun)
    mercury_hat = unit(topocentric_mercury)
    separation = angular_separation(sun_hat, mercury_hat)
    if separation == 0.0:
        return np.zeros(2, dtype=float)

    tangent_direction = unit(
        mercury_hat - math.cos(separation) * sun_hat
    )
    tangent_vector = separation * tangent_direction
    return ARCSEC_PER_RAD * np.array(
        [
            float(np.dot(tangent_vector, xi)),
            float(np.dot(tangent_vector, eta)),
        ],
        dtype=float,
    )


def center_pair(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    midpoint = 0.5 * (first + second)
    return first - midpoint, second - midpoint


def pair_angle(first: np.ndarray, second: np.ndarray) -> float:
    difference = second - first
    return math.degrees(math.atan2(difference[1], difference[0]))


def padded_limits(first: np.ndarray, second: np.ndarray, minimum_span: float):
    values_x = np.array([first[0], second[0]], dtype=float)
    values_y = np.array([first[1], second[1]], dtype=float)
    span_x = max(float(np.ptp(values_x)), minimum_span)
    span_y = max(float(np.ptp(values_y)), minimum_span)
    half_x = 0.72 * max(span_x, span_y)
    half_y = 0.72 * max(span_x, span_y)
    return (-half_x, half_x), (-half_y, half_y)


def annotate_point(ax, point: np.ndarray, label: str, color: str, offset):
    ax.scatter(
        [point[0]],
        [point[1]],
        s=POINT_SIZE,
        color=color,
        edgecolor="white",
        linewidth=0.75,
        zorder=5,
    )
    ax.annotate(
        label,
        xy=(point[0], point[1]),
        xytext=offset,
        textcoords="offset points",
        color=color,
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "color": color,
            "linewidth": 0.75,
        },
        zorder=6,
    )


def style_axis(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor("black")
    ax.axhline(0.0, color=COLOR_GUIDE, linewidth=0.5, zorder=0)
    ax.axvline(0.0, color=COLOR_GUIDE, linewidth=0.5, zorder=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors="#CBD5E1", width=0.5, length=3, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.75)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master, source = load_or_build_master()
    cache = build_cache(master)
    jd_common = common_epoch(cache)
    utc_common = Time(jd_common, format="jd", scale="tdb").utc.datetime.strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )[:-3]

    geocentric_sun = vector_at(cache, "GEOCENTER_SUN", jd_common)
    _sun_hat, xi, eta = ecliptic_solar_basis(geocentric_sun)

    topocentric_sun_mb = vector_at(cache, "MERCURY_BAY_SUN", jd_common)
    topocentric_sun_v = vector_at(cache, "VARDO_SUN", jd_common)
    topocentric_mercury_mb = vector_at(
        cache, "MERCURY_BAY_MERCURY", jd_common
    )
    topocentric_mercury_v = vector_at(cache, "VARDO_MERCURY", jd_common)

    observer_mb = geocentric_sun - topocentric_sun_mb
    observer_v = geocentric_sun - topocentric_sun_v

    a_absolute = project_vector_km(observer_mb, xi, eta)
    b_absolute = project_vector_km(observer_v, xi, eta)
    a, b = center_pair(a_absolute, b_absolute)

    a_prime_absolute = apparent_mercury_arcsec(
        topocentric_sun_mb,
        topocentric_mercury_mb,
        xi,
        eta,
    )
    b_prime_absolute = apparent_mercury_arcsec(
        topocentric_sun_v,
        topocentric_mercury_v,
        xi,
        eta,
    )
    a_prime, b_prime = center_pair(a_prime_absolute, b_prime_absolute)

    ab_vector = b - a
    aprime_bprime_vector = b_prime - a_prime
    ab_distance_km = norm(ab_vector)
    aprime_bprime_distance_arcsec = norm(aprime_bprime_vector)
    ab_angle_deg = pair_angle(a, b)
    aprime_bprime_angle_deg = pair_angle(a_prime, b_prime)

    audit = {
        "A_plus_B_midpoint_km": norm(0.5 * (a + b)),
        "Aprime_plus_Bprime_midpoint_arcsec": norm(
            0.5 * (a_prime + b_prime)
        ),
        "xi_dot_eta": abs(float(np.dot(xi, eta))),
        "xi_norm_error": abs(norm(xi) - 1.0),
        "eta_norm_error": abs(norm(eta) - 1.0),
    }

    rows = [
        {
            "point": "A",
            "meaning": "Mercury Bay projected observer point",
            "x": a[0],
            "y": a[1],
            "unit": "km",
        },
        {
            "point": "B",
            "meaning": "Vardø projected observer point",
            "x": b[0],
            "y": b[1],
            "unit": "km",
        },
        {
            "point": "A′",
            "meaning": "Mercury Bay apparent Mercury point",
            "x": a_prime[0],
            "y": a_prime[1],
            "unit": "arcsec",
        },
        {
            "point": "B′",
            "meaning": "Vardø apparent Mercury point",
            "x": b_prime[0],
            "y": b_prime[1],
            "unit": "arcsec",
        },
    ]
    result_frame = pd.DataFrame(rows)
    result_frame["common_jd_tdb"] = jd_common
    result_frame["common_utc"] = utc_common
    result_frame.to_csv(OUTPUT_CSV, index=False, float_format="%.12f")

    plt.close("all")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "figure.facecolor": "black",
            "savefig.facecolor": "black",
            "axes.facecolor": "black",
            "text.color": "white",
            "axes.labelcolor": "white",
            "xtick.color": "#CBD5E1",
            "ytick.color": "#CBD5E1",
        }
    )

    figure = plt.figure(figsize=(17.0, 9.8), facecolor="black")
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(0.78, 0.22),
        left=0.055,
        right=0.970,
        top=0.900,
        bottom=0.065,
        wspace=0.18,
        hspace=0.10,
    )
    ax_ab = figure.add_subplot(grid[0, 0])
    ax_ap = figure.add_subplot(grid[0, 1])
    ax_table = figure.add_subplot(grid[1, :])
    ax_table.axis("off")

    figure.suptitle(
        "1769 MERCURY TRANSIT — COMMON-EPOCH A/B AND A′/B′ GEOMETRY",
        fontsize=17,
        fontweight="bold",
        y=0.965,
    )

    style_axis(
        ax_ab,
        r"Ecliptic $\xi$ (km)",
        r"Ecliptic $\eta$ (km)",
        "PROJECTED OBSERVER PAIR — EARTH PLANE",
    )
    ax_ab.plot(
        [a[0], b[0]],
        [a[1], b[1]],
        color=COLOR_SEGMENT,
        linewidth=LINE_WIDTH,
        zorder=2,
    )
    annotate_point(ax_ab, a, "A\nMercury Bay", COLOR_A, (-10, 25))
    annotate_point(ax_ab, b, "B\nVardø", COLOR_B, (10, -28))
    limits_x, limits_y = padded_limits(a, b, 1000.0)
    ax_ab.set_xlim(*limits_x)
    ax_ab.set_ylim(*limits_y)
    ax_ab.text(
        0.03,
        0.04,
        f"AB = {ab_distance_km:,.6f} km\n"
        f"segment angle = {ab_angle_deg:.6f}°",
        transform=ax_ab.transAxes,
        fontsize=10,
        color="#E2E8F0",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#111827",
            "edgecolor": "#64748B",
            "linewidth": 0.75,
        },
    )

    style_axis(
        ax_ap,
        r"Ecliptic $\xi$ (arcsec)",
        r"Ecliptic $\eta$ (arcsec)",
        "APPARENT MERCURY PAIR — SKY PLANE",
    )
    ax_ap.plot(
        [a_prime[0], b_prime[0]],
        [a_prime[1], b_prime[1]],
        color=COLOR_SEGMENT,
        linewidth=LINE_WIDTH,
        zorder=2,
    )
    annotate_point(
        ax_ap,
        a_prime,
        "A′\nMercury Bay",
        COLOR_A,
        (-10, 25),
    )
    annotate_point(
        ax_ap,
        b_prime,
        "B′\nVardø",
        COLOR_B,
        (10, -28),
    )
    limits_x, limits_y = padded_limits(a_prime, b_prime, 1.0)
    ax_ap.set_xlim(*limits_x)
    ax_ap.set_ylim(*limits_y)
    ax_ap.text(
        0.03,
        0.04,
        f"A′B′ = {aprime_bprime_distance_arcsec:.12f} arcsec\n"
        f"segment angle = {aprime_bprime_angle_deg:.6f}°",
        transform=ax_ap.transAxes,
        fontsize=10,
        color="#E2E8F0",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#111827",
            "edgecolor": "#64748B",
            "linewidth": 0.75,
        },
    )

    table_rows = [
        ["Point", "Definition", "ξ", "η", "Unit"],
        [
            "A",
            "Mercury Bay projected observer; midpoint-centered",
            f"{a[0]:,.6f}",
            f"{a[1]:,.6f}",
            "km",
        ],
        [
            "B",
            "Vardø projected observer; midpoint-centered",
            f"{b[0]:,.6f}",
            f"{b[1]:,.6f}",
            "km",
        ],
        [
            "A′",
            "Mercury Bay apparent Mercury; same epoch",
            f"{a_prime[0]:.9f}",
            f"{a_prime[1]:.9f}",
            "arcsec",
        ],
        [
            "B′",
            "Vardø apparent Mercury; same epoch",
            f"{b_prime[0]:.9f}",
            f"{b_prime[1]:.9f}",
            "arcsec",
        ],
        [
            "AB",
            "Projected observer separation",
            f"{ab_distance_km:,.6f}",
            "",
            "km",
        ],
        [
            "A′B′",
            "Apparent Mercury separation",
            f"{aprime_bprime_distance_arcsec:.12f}",
            "",
            "arcsec",
        ],
    ]
    table = ax_table.table(
        cellText=table_rows,
        cellLoc="left",
        colWidths=[0.08, 0.47, 0.17, 0.17, 0.11],
        bbox=[0.04, 0.02, 0.92, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        cell.set_linewidth(0.75)
        cell.get_text().set_color("white")
        if row == 0:
            cell.set_facecolor(COLOR_HEADER)
            cell.get_text().set_fontweight("bold")
        elif row in (1, 3):
            cell.set_facecolor("#123B48")
        elif row in (2, 4):
            cell.set_facecolor("#4A3510")
        else:
            cell.set_facecolor(COLOR_BODY)
            if row in (5, 6):
                cell.get_text().set_fontweight("bold")

    figure.text(
        0.5,
        0.025,
        f"Common epoch: {utc_common} UTC | JPL ecliptic geometric vectors | "
        "both pairs midpoint-centered only for display; separations are unchanged.",
        ha="center",
        fontsize=8.2,
        color="#CBD5E1",
    )

    figure.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="black",
    )
    plt.close(figure)

    print(f"CODE OUTPUT: {VERSION}")
    print("CODE INPUTS")
    print(f"JPL source: {source}")
    print("Reference plane: JPL ECLIPTIC")
    print(f"Common epoch: {utc_common} UTC")
    print("COMMENTS")
    print("A and B are projected observer points in the common geocentric solar tangent plane.")
    print("A′ and B′ are apparent Mercury points evaluated at the identical common epoch.")
    print("Each pair is midpoint-centered for plotting; the AB and A′B′ separations are invariant.")
    print("RESULTS")
    print(f"A  = ({a[0]:+.9f}, {a[1]:+.9f}) km")
    print(f"B  = ({b[0]:+.9f}, {b[1]:+.9f}) km")
    print(f"AB = {ab_distance_km:.9f} km")
    print(f"A′ = ({a_prime[0]:+.12f}, {a_prime[1]:+.12f}) arcsec")
    print(f"B′ = ({b_prime[0]:+.12f}, {b_prime[1]:+.12f}) arcsec")
    print(f"A′B′ = {aprime_bprime_distance_arcsec:.12f} arcsec")
    print("OUTPUT SUMMARY")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"CSV: {OUTPUT_CSV}")
    print("PAPER COMPARISON")
    print("The plotted A′B′ value is the simultaneous two-observer apparent separation, not an absolute Sun-centered coordinate.")
    print("EQUATION STATUS")
    print(f"Observer midpoint audit: {audit['A_plus_B_midpoint_km']:.3e} km")
    print(f"Apparent midpoint audit: {audit['Aprime_plus_Bprime_midpoint_arcsec']:.3e} arcsec")
    print(f"Basis orthogonality audit: {audit['xi_dot_eta']:.3e}")
    print("Common-epoch geometry: PASS")
    try:
        from IPython.display import Image, display
        display(Image(filename=str(OUTPUT_PNG)))
    except Exception as display_error:
        print(f"Inline PNG display unavailable: {display_error}")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# V0015
