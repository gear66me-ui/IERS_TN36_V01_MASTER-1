# V0002
# Audit reference: Black-and-white summary plate for IERS reduction versus JPL vectors.
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

VERSION = "V0002"
PROGRAM = "IERS_REDUCTION_VS_JPL_VECTORS_V0002.py"
TITLE = "SOLAR HORIZONTAL PARALLAX — HISTORICAL REDUCTION AND JPL VECTOR AUDIT"
LOCAL_TZ = ZoneInfo("America/Bogota")

ARCSEC_PER_RAD = 206264.80624709636
TARGET_ARCSEC = 8.794148
PASS_TOLERANCE_ARCSEC = 0.000007

C_M_S = 299792458.000000
TAU_A_S = 499.004782000
IAU1976_RADIUS_M = 6378140.000000
IAU1976_RADIUS_SIGMA_M = 5.000000
IAU1976_PUBLISHED_AU_M = 149597870000.000000
IAU1976_PUBLISHED_AU_SIGMA_M = 2000.000000
WGS84_RADIUS_M = 6378137.000000
IERS2010_RADIUS_M = 6378136.600000
IAU2012_AU_M = 149597870700.000000

ROOT = Path("/content")
OUTPUT_DIR_DEFAULT = ROOT / "IERS_REDUCTION_VS_JPL_VECTORS_V0002_OUTPUT"
RUNTIME_MASTER = ROOT / "O6_1769_GEOCENTER_HORIZONS_V0002.csv"
HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"
VECTOR_COLUMNS = [
    "JD",
    "GEOCENTER_SUN_X_KM",
    "GEOCENTER_SUN_Y_KM",
    "GEOCENTER_SUN_Z_KM",
    "GEOCENTER_VENUS_X_KM",
    "GEOCENTER_VENUS_Y_KM",
    "GEOCENTER_VENUS_Z_KM",
]
RADIUS_MODES = {
    "WGS84": WGS84_RADIUS_M,
    "IAU1976": IAU1976_RADIUS_M,
    "IERS2010": IERS2010_RADIUS_M,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--jpl-csv", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--earth-radius-mode",
        choices=("WGS84", "IAU1976", "IERS2010", "CUSTOM"),
        default="WGS84",
    )
    parser.add_argument("--earth-radius-m", type=float)
    parser.add_argument("--dpi", type=int, default=420)
    return parser.parse_args()


def selected_radius(arguments: argparse.Namespace) -> tuple[str, float]:
    mode = arguments.earth_radius_mode
    if mode == "CUSTOM":
        value = arguments.earth_radius_m
        if value is None or not math.isfinite(value) or value <= 0.0:
            raise ValueError("CUSTOM mode requires a positive --earth-radius-m value.")
        return mode, float(value)
    return mode, RADIUS_MODES[mode]


def is_compatible_master(path: Path) -> bool:
    try:
        columns = pd.read_csv(path, nrows=0).columns
    except Exception:
        return False
    return all(column in columns for column in VECTOR_COLUMNS)


def runtime_csv_candidates() -> list[Path]:
    candidates: list[Path] = []
    if not ROOT.exists():
        return candidates
    for root, directories, filenames in os.walk(ROOT):
        directories[:] = [
            directory
            for directory in directories
            if directory != "drive" and not directory.startswith(".")
        ]
        candidates.extend(
            Path(root) / filename
            for filename in filenames
            if filename.lower().endswith(".csv")
        )
    return sorted(set(candidates))


def request_horizons_vectors(target: str, label: str) -> pd.DataFrame:
    parameters = {
        "format": "json",
        "COMMAND": f"'{target}'",
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": "'500@399'",
        "START_TIME": "'1769-Jun-03 18:00'",
        "STOP_TIME": "'1769-Jun-04 03:00'",
        "STEP_SIZE": "'1m'",
        "TIME_TYPE": "'UT'",
        "TIME_DIGITS": "'FRACSEC'",
        "CAL_TYPE": "'GREGORIAN'",
        "REF_PLANE": "'FRAME'",
        "REF_SYSTEM": "'ICRF'",
        "OUT_UNITS": "'KM-S'",
        "VEC_TABLE": "'1'",
        "VEC_CORR": "'NONE'",
        "CSV_FORMAT": "'YES'",
        "VEC_LABELS": "'NO'",
    }
    url = HORIZONS_API + "?" + urlencode(parameters)
    payload = None
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": PROGRAM})
            with urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except Exception as error:
            last_error = error
            time.sleep(attempt + 1)
    if payload is None:
        raise RuntimeError(f"JPL Horizons request failed for {label}: {last_error}")
    if payload.get("error"):
        raise RuntimeError(f"JPL Horizons returned an error for {label}: {payload['error']}")
    if "JPL" not in str(payload.get("signature", {}).get("source", "")).upper():
        raise RuntimeError(f"Unverified Horizons response for {label}.")
    result_text = str(payload.get("result", ""))
    if "$$SOE" not in result_text or "$$EOE" not in result_text:
        raise RuntimeError(f"No JPL vector table was returned for {label}.")

    rows: list[dict[str, float]] = []
    vector_text = result_text.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    for raw_row in csv.reader(io.StringIO(vector_text)):
        fields = [field.strip() for field in raw_row if field.strip()]
        try:
            julian_date = float(fields[0].replace("D", "E"))
        except (IndexError, ValueError):
            continue
        numeric_values: list[float] = []
        for field in fields[1:]:
            try:
                numeric_values.append(float(field.replace("D", "E")))
            except ValueError:
                continue
        if len(numeric_values) < 3:
            raise RuntimeError(f"Cannot decode the {label} vector row: {raw_row}")
        rows.append(
            {
                "JD": julian_date,
                f"GEOCENTER_{label}_X_KM": numeric_values[0],
                f"GEOCENTER_{label}_Y_KM": numeric_values[1],
                f"GEOCENTER_{label}_Z_KM": numeric_values[2],
            }
        )
    frame = pd.DataFrame(rows).sort_values("JD").drop_duplicates("JD").reset_index(drop=True)
    if len(frame) < 7:
        raise RuntimeError(f"Only {len(frame)} usable JPL vector rows were returned for {label}.")
    return frame


def build_runtime_master() -> tuple[Path, str]:
    sun = request_horizons_vectors("10", "SUN")
    venus = request_horizons_vectors("299", "VENUS")
    sun["JOIN_KEY"] = sun["JD"].round(10)
    venus["JOIN_KEY"] = venus["JD"].round(10)
    master = (
        sun.merge(venus.drop(columns="JD"), on="JOIN_KEY", validate="one_to_one")
        .drop(columns="JOIN_KEY")[VECTOR_COLUMNS]
    )
    master.to_csv(RUNTIME_MASTER, index=False, float_format="%.15f")
    return RUNTIME_MASTER.resolve(), "OFFICIAL JPL HORIZONS API"


def locate_jpl_master(requested: str) -> tuple[Path, str]:
    if requested:
        explicit = Path(requested).expanduser()
        if explicit.is_file() and is_compatible_master(explicit):
            return explicit.resolve(), "EXPLICIT JPL VECTOR CSV"
    preferred = [
        ROOT / "O6_TAHITI_VARDO_1769_1MIN_MASTER.csv",
        ROOT / "O6_1769_GEOCENTER_HORIZONS_V0013.csv",
        ROOT / "O6_1769_GEOCENTER_HORIZONS_V0014.csv",
        RUNTIME_MASTER,
    ]
    for candidate in preferred + runtime_csv_candidates():
        if candidate.is_file() and is_compatible_master(candidate):
            return candidate.resolve(), "COLAB RUNTIME JPL VECTOR CSV"
    return build_runtime_master()


def julian_date_to_utc_text(julian_date: float) -> str:
    shifted = float(julian_date) + 0.5
    integer = int(math.floor(shifted))
    fraction = shifted - integer
    if integer >= 2299161:
        alpha = int((integer - 1867216.25) / 36524.25)
        adjusted = integer + 1 + alpha - int(alpha / 4)
    else:
        adjusted = integer
    b_value = adjusted + 1524
    c_value = int((b_value - 122.1) / 365.25)
    d_value = int(365.25 * c_value)
    e_value = int((b_value - d_value) / 30.6001)
    day_decimal = b_value - d_value - int(30.6001 * e_value) + fraction
    day = int(day_decimal)
    month = e_value - 1 if e_value < 14 else e_value - 13
    year = c_value - 4716 if month > 2 else c_value - 4715
    date_time = datetime(year, month, day, tzinfo=timezone.utc) + timedelta(
        microseconds=round((day_decimal - day) * 86400.0 * 1_000_000.0)
    )
    return date_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"


def golden_section_minimum(function, lower: float, upper: float) -> tuple[float, float]:
    inverse_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c_value = upper - inverse_phi * (upper - lower)
    d_value = lower + inverse_phi * (upper - lower)
    function_c = function(c_value)
    function_d = function(d_value)
    for _ in range(240):
        if upper - lower <= 0.000001:
            break
        if function_c < function_d:
            upper = d_value
            d_value = c_value
            function_d = function_c
            c_value = upper - inverse_phi * (upper - lower)
            function_c = function(c_value)
        else:
            lower = c_value
            c_value = d_value
            function_c = function_d
            d_value = lower + inverse_phi * (upper - lower)
            function_d = function(d_value)
    optimum = 0.5 * (lower + upper)
    return optimum, function(optimum)


def derive_jpl_geometry(master_path: Path) -> dict[str, object]:
    frame = pd.read_csv(master_path)[VECTOR_COLUMNS].apply(pd.to_numeric, errors="coerce")
    frame = frame[np.isfinite(frame.to_numpy(dtype=float)).all(axis=1)]
    frame = frame.sort_values("JD").drop_duplicates("JD").reset_index(drop=True)
    julian_dates = frame["JD"].to_numpy(dtype=float)
    sun_vectors = frame[VECTOR_COLUMNS[1:4]].to_numpy(dtype=float)
    venus_vectors = frame[VECTOR_COLUMNS[4:7]].to_numpy(dtype=float)
    separations = np.arctan2(
        np.linalg.norm(np.cross(sun_vectors, venus_vectors), axis=1),
        np.sum(sun_vectors * venus_vectors, axis=1),
    )
    nearest_index = int(np.argmin(separations))
    if nearest_index == 0 or nearest_index == len(frame) - 1:
        raise RuntimeError("The closest-approach epoch lies at a JPL data boundary.")

    lower_index = max(0, nearest_index - 3)
    upper_index = min(len(frame), nearest_index + 4)
    center_jd = julian_dates[nearest_index]
    seconds = (julian_dates[lower_index:upper_index] - center_jd) * 86400.0
    degree = min(3, len(seconds) - 1)
    sun_polynomials = [
        np.poly1d(np.polyfit(seconds, sun_vectors[lower_index:upper_index, axis], degree))
        for axis in range(3)
    ]
    venus_polynomials = [
        np.poly1d(np.polyfit(seconds, venus_vectors[lower_index:upper_index, axis], degree))
        for axis in range(3)
    ]

    def evaluate(polynomials, offset_seconds: float) -> np.ndarray:
        return np.array([float(polynomial(offset_seconds)) for polynomial in polynomials])

    def separation(offset_seconds: float) -> float:
        sun_vector = evaluate(sun_polynomials, offset_seconds)
        venus_vector = evaluate(venus_polynomials, offset_seconds)
        return math.atan2(
            float(np.linalg.norm(np.cross(sun_vector, venus_vector))),
            float(np.dot(sun_vector, venus_vector)),
        )

    lower_seconds = (julian_dates[nearest_index - 1] - center_jd) * 86400.0
    upper_seconds = (julian_dates[nearest_index + 1] - center_jd) * 86400.0
    offset_seconds, separation_radians = golden_section_minimum(
        separation,
        lower_seconds,
        upper_seconds,
    )
    selected_jd = center_jd + offset_seconds / 86400.0
    sun_vector = evaluate(sun_polynomials, offset_seconds)
    venus_vector = evaluate(venus_polynomials, offset_seconds)
    return {
        "frame": frame,
        "jd": selected_jd,
        "epoch_utc": julian_date_to_utc_text(selected_jd),
        "sun_vector_km": sun_vector,
        "venus_vector_km": venus_vector,
        "distance_m": float(np.linalg.norm(sun_vector)) * 1000.0,
        "separation_arcsec": separation_radians * ARCSEC_PER_RAD,
        "offset_seconds": offset_seconds,
    }


def solar_parallax(radius_m: float, distance_m: float) -> tuple[float, float, float]:
    ratio = radius_m / distance_m
    if not 0.0 < ratio < 1.0:
        raise ValueError("Earth-radius/distance ratio is outside (0, 1).")
    radians = math.asin(ratio)
    return ratio, radians, radians * ARCSEC_PER_RAD


def derive_reduction(geometry: dict[str, object], raw_radius_m: float) -> dict[str, float]:
    exact_iau1976_au_m = C_M_S * TAU_A_S
    _, raw_radians, raw_arcsec = solar_parallax(raw_radius_m, float(geometry["distance_m"]))
    distance_factor = float(geometry["distance_m"]) / exact_iau1976_au_m
    radius_factor = IAU1976_RADIUS_M / raw_radius_m
    total_factor = distance_factor * radius_factor
    linear_arcsec = raw_arcsec * total_factor
    exact_arcsec = math.asin(total_factor * math.sin(raw_radians)) * ARCSEC_PER_RAD
    case_2_arcsec = solar_parallax(IAU1976_RADIUS_M, exact_iau1976_au_m)[2]
    return {
        "exact_iau1976_au_m": exact_iau1976_au_m,
        "raw_arcsec": raw_arcsec,
        "distance_factor": distance_factor,
        "radius_factor": radius_factor,
        "total_factor": total_factor,
        "linear_arcsec": linear_arcsec,
        "exact_arcsec": exact_arcsec,
        "case_2_arcsec": case_2_arcsec,
        "linear_minus_exact_microarcsec": (linear_arcsec - exact_arcsec) * 1_000_000.0,
        "exact_minus_case_2_microarcsec": (exact_arcsec - case_2_arcsec) * 1_000_000.0,
    }


def historical_uncertainty() -> tuple[float, float, float]:
    ratio = IAU1976_RADIUS_M / IAU1976_PUBLISHED_AU_M
    denominator = math.sqrt(1.0 - ratio * ratio)
    radius_component = (
        abs(1.0 / (IAU1976_PUBLISHED_AU_M * denominator))
        * IAU1976_RADIUS_SIGMA_M
        * ARCSEC_PER_RAD
    )
    au_component = (
        abs(
            -IAU1976_RADIUS_M
            / (IAU1976_PUBLISHED_AU_M**2 * denominator)
        )
        * IAU1976_PUBLISHED_AU_SIGMA_M
        * ARCSEC_PER_RAD
    )
    return radius_component, au_component, math.hypot(radius_component, au_component)


def build_summary_table(
    geometry: dict[str, object],
    radius_mode: str,
    raw_radius_m: float,
    reduction: dict[str, float],
) -> pd.DataFrame:
    definitions = [
        ("IAU 1976", "Published AU", IAU1976_RADIUS_M, IAU1976_PUBLISHED_AU_M),
        ("IAU 1976", "Exact cτ_A", IAU1976_RADIUS_M, reduction["exact_iau1976_au_m"]),
        ("IAU 2012", "WGS84", WGS84_RADIUS_M, IAU2012_AU_M),
        ("IERS 2010", "IAU 2012 AU", IERS2010_RADIUS_M, IAU2012_AU_M),
        ("JPL raw", radius_mode, raw_radius_m, float(geometry["distance_m"])),
        ("JPL reduced", "IAU 1976 Case 2", IAU1976_RADIUS_M, reduction["exact_iau1976_au_m"]),
    ]
    rows: list[dict[str, object]] = []
    for case_name, convention, radius_m, distance_m in definitions:
        _, _, parallax_arcsec = solar_parallax(radius_m, distance_m)
        if case_name == "JPL reduced":
            parallax_arcsec = reduction["exact_arcsec"]
        delta_microarcsec = (parallax_arcsec - TARGET_ARCSEC) * 1_000_000.0
        classification = (
            "RAW"
            if case_name == "JPL raw"
            else ("PASS" if abs(delta_microarcsec) <= PASS_TOLERANCE_ARCSEC * 1_000_000.0 else "FAIL")
        )
        rows.append(
            {
                "Case": case_name,
                "Convention": convention,
                "a (m)": radius_m,
                "D or A (m)": distance_m,
                "π⊙ (arcsec)": parallax_arcsec,
                "Δ (µas)": delta_microarcsec,
                "Class": classification,
            }
        )
    return pd.DataFrame(rows)


def save_vector_files(geometry: dict[str, object], output_dir: Path) -> dict[str, Path]:
    frame = geometry["frame"].copy()
    frame.insert(1, "UTC", frame["JD"].map(julian_date_to_utc_text))
    paths = {
        "sun": output_dir / "JPL_1769_GEOCENTER_SUN_VECTORS_V0002.csv",
        "venus": output_dir / "JPL_1769_GEOCENTER_VENUS_VECTORS_V0002.csv",
        "master": output_dir / "JPL_1769_GEOCENTER_MASTER_V0002.csv",
    }
    frame[["JD", "UTC", *VECTOR_COLUMNS[1:4]]].to_csv(
        paths["sun"], index=False, float_format="%.15f"
    )
    frame[["JD", "UTC", *VECTOR_COLUMNS[4:7]]].to_csv(
        paths["venus"], index=False, float_format="%.15f"
    )
    frame[["JD", "UTC", *VECTOR_COLUMNS[1:]]].to_csv(
        paths["master"], index=False, float_format="%.15f"
    )
    return paths


def add_dark_panel(axis, title: str) -> None:
    axis.set_axis_off()
    axis.add_patch(
        FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            transform=axis.transAxes,
            linewidth=0.8,
            edgecolor="#FFFFFF",
            facecolor="#000000",
            clip_on=False,
            zorder=-10,
        )
    )
    axis.text(
        0.025,
        0.94,
        title,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color="#FFFFFF",
    )


def render_publication_summary(
    summary: pd.DataFrame,
    geometry: dict[str, object],
    reduction: dict[str, float],
    uncertainty: tuple[float, float, float],
    source: str,
    vector_paths: dict[str, Path],
    output_png: Path,
    dpi: int,
) -> None:
    plt.close("all")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "figure.facecolor": "#000000",
            "savefig.facecolor": "#000000",
            "text.color": "#FFFFFF",
            "axes.edgecolor": "#FFFFFF",
            "axes.labelcolor": "#FFFFFF",
            "xtick.color": "#FFFFFF",
            "ytick.color": "#FFFFFF",
        }
    )

    figure = plt.figure(figsize=(18.0, 12.0), facecolor="#000000")
    grid = figure.add_gridspec(
        4,
        2,
        height_ratios=(0.18, 0.82, 1.08, 0.92),
        hspace=0.17,
        wspace=0.06,
        left=0.035,
        right=0.965,
        top=0.965,
        bottom=0.05,
    )

    title_axis = figure.add_subplot(grid[0, :])
    title_axis.set_axis_off()
    title_axis.text(
        0.5,
        0.75,
        TITLE,
        ha="center",
        va="center",
        fontsize=19.0,
        fontweight="bold",
        color="#FFFFFF",
    )
    title_axis.text(
        0.5,
        0.24,
        r"1769 Venus Transit  •  exact $c\tau_A$ normalization  •  geocentric JPL Horizons vectors",
        ha="center",
        va="center",
        fontsize=11.0,
        color="#D1D5DB",
    )

    reduction_axis = figure.add_subplot(grid[1, 0])
    vector_axis = figure.add_subplot(grid[1, 1])
    table_axis = figure.add_subplot(grid[2, :])
    audit_axis = figure.add_subplot(grid[3, :])

    add_dark_panel(reduction_axis, "I. HISTORICAL REDUCTION — SUMMARY")
    equations = [
        rf"$\pi_{{\odot,\mathrm{{JPL}}}}={reduction['raw_arcsec']:.12f}^{{\prime\prime}}$",
        rf"$F_D=D_{{\mathrm{{JPL}}}}/A_2={reduction['distance_factor']:.15f}$",
        rf"$F_R=a_{{1976}}/a_{{WGS84}}={reduction['radius_factor']:.15f}$",
        rf"$F_2=F_DF_R={reduction['total_factor']:.15f}$",
        rf"$\pi_{{2,\mathrm{{exact}}}}=\arcsin\!\left[F_2\sin\!\left(\pi_{{\odot,\mathrm{{JPL}}}}\right)\right]$",
        rf"$\pi_{{2,\mathrm{{exact}}}}={reduction['exact_arcsec']:.12f}^{{\prime\prime}}$",
    ]
    y_position = 0.79
    for index, equation in enumerate(equations):
        reduction_axis.text(
            0.05,
            y_position,
            equation,
            transform=reduction_axis.transAxes,
            fontsize=13.2 if index in (0, 4, 5) else 11.8,
            color="#FFFFFF",
        )
        y_position -= 0.125
    reduction_axis.text(
        0.05,
        0.08,
        rf"$\Delta_{{\mathrm{{exact-Case\ 2}}}}={reduction['exact_minus_case_2_microarcsec']:.9f}\ \mu\mathrm{{as}}$",
        transform=reduction_axis.transAxes,
        fontsize=11.0,
        color="#D1D5DB",
    )

    add_dark_panel(vector_axis, "II. JPL VECTOR AUDIT — CLOSEST APPROACH")
    sun_vector = geometry["sun_vector_km"]
    venus_vector = geometry["venus_vector_km"]
    vector_rows = [
        ("Source", source, ""),
        ("Epoch", str(geometry["epoch_utc"]), "UTC"),
        ("Julian date", f"{float(geometry['jd']):.12f}", "JD"),
        (r"$X_\odot$", f"{sun_vector[0]:,.6f}", "km"),
        (r"$Y_\odot$", f"{sun_vector[1]:,.6f}", "km"),
        (r"$Z_\odot$", f"{sun_vector[2]:,.6f}", "km"),
        (r"$\|\mathbf{r}_{E\odot}\|$", f"{float(geometry['distance_m']) / 1000.0:,.6f}", "km"),
        (r"$X_\mathrm{V}$", f"{venus_vector[0]:,.6f}", "km"),
        (r"$Y_\mathrm{V}$", f"{venus_vector[1]:,.6f}", "km"),
        (r"$Z_\mathrm{V}$", f"{venus_vector[2]:,.6f}", "km"),
        (r"$\theta_{\odot\mathrm{V}}$", f"{float(geometry['separation_arcsec']):.9f}", "arcsec"),
    ]
    y_position = 0.80
    for label, value, unit in vector_rows:
        vector_axis.text(0.05, y_position, label, transform=vector_axis.transAxes, fontsize=9.0, color="#FFFFFF")
        vector_axis.text(0.28, y_position, value, transform=vector_axis.transAxes, fontsize=8.7, family="monospace", color="#FFFFFF")
        vector_axis.text(0.86, y_position, unit, transform=vector_axis.transAxes, fontsize=8.5, color="#D1D5DB")
        y_position -= 0.064

    add_dark_panel(table_axis, "III. PARALLAX SUMMARY")
    display_table = summary.copy()
    display_table["a (m)"] = display_table["a (m)"].map(lambda value: f"{float(value):,.3f}")
    display_table["D or A (m)"] = display_table["D or A (m)"].map(lambda value: f"{float(value):,.3f}")
    display_table["π⊙ (arcsec)"] = display_table["π⊙ (arcsec)"].map(lambda value: f"{float(value):.12f}")
    display_table["Δ (µas)"] = display_table["Δ (µas)"].map(lambda value: f"{float(value):+.6f}")
    publication_table = table_axis.table(
        cellText=display_table.values,
        colLabels=display_table.columns,
        cellLoc="left",
        colLoc="center",
        bbox=(0.025, 0.10, 0.95, 0.76),
        colWidths=(0.11, 0.22, 0.14, 0.21, 0.16, 0.10, 0.07),
    )
    publication_table.set_zorder(5)
    publication_table.auto_set_font_size(False)
    publication_table.set_fontsize(8.3)
    for (row, column), cell in publication_table.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor("#FFFFFF")
        cell.set_facecolor("#000000" if row == 0 else ("#111111" if row % 2 else "#1F1F1F"))
        cell.set_text_props(color="#FFFFFF", weight="bold" if row == 0 else "normal")

    add_dark_panel(audit_axis, "IV. UNCERTAINTY, TRACEABILITY, AND GENERATED FILES")
    radius_sigma, au_sigma, combined_sigma = uncertainty
    audit_axis.text(
        0.04,
        0.77,
        rf"$\sigma_{{\pi,a}}={radius_sigma * 1_000_000.0:.6f}\ \mu\mathrm{{as}}$",
        transform=audit_axis.transAxes,
        fontsize=11.5,
        color="#FFFFFF",
    )
    audit_axis.text(
        0.04,
        0.59,
        rf"$\sigma_{{\pi,A}}={au_sigma * 1_000_000.0:.6f}\ \mu\mathrm{{as}}$",
        transform=audit_axis.transAxes,
        fontsize=11.5,
        color="#FFFFFF",
    )
    audit_axis.text(
        0.04,
        0.41,
        rf"$\sigma_\pi={combined_sigma:.12f}^{{\prime\prime}}\rightarrow\pm {PASS_TOLERANCE_ARCSEC:.6f}^{{\prime\prime}}$",
        transform=audit_axis.transAxes,
        fontsize=11.5,
        color="#FFFFFF",
    )
    audit_axis.text(
        0.04,
        0.17,
        "Every displayed result derives from defining constants or minute-by-minute JPL vectors.\nNo manual parallax result enters the calculation chain.",
        transform=audit_axis.transAxes,
        fontsize=9.0,
        color="#D1D5DB",
        linespacing=1.35,
    )
    audit_axis.text(0.53, 0.77, "Generated JPL vector files", transform=audit_axis.transAxes, fontsize=10.0, fontweight="bold", color="#FFFFFF")
    audit_axis.text(0.53, 0.60, vector_paths["sun"].name, transform=audit_axis.transAxes, fontsize=8.7, family="monospace", color="#FFFFFF")
    audit_axis.text(0.53, 0.45, vector_paths["venus"].name, transform=audit_axis.transAxes, fontsize=8.7, family="monospace", color="#FFFFFF")
    audit_axis.text(0.53, 0.30, vector_paths["master"].name, transform=audit_axis.transAxes, fontsize=8.7, family="monospace", color="#FFFFFF")

    figure.text(
        0.5,
        0.017,
        "Figure V0002. Black-and-white summary of the historical solar-parallax reduction and the geocentric JPL vector audit.",
        ha="center",
        fontsize=8.5,
        color="#D1D5DB",
    )
    figure.savefig(
        output_png,
        dpi=max(240, int(dpi)),
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="#000000",
    )
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    radius_mode, raw_radius_m = selected_radius(arguments)
    output_dir = (
        Path(arguments.output_dir).expanduser().resolve()
        if arguments.output_dir
        else OUTPUT_DIR_DEFAULT
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    master_path, jpl_source = locate_jpl_master(arguments.jpl_csv)
    geometry = derive_jpl_geometry(master_path)
    reduction = derive_reduction(geometry, raw_radius_m)
    uncertainty = historical_uncertainty()
    summary = build_summary_table(geometry, radius_mode, raw_radius_m, reduction)
    vector_paths = save_vector_files(geometry, output_dir)

    summary_csv = output_dir / "IERS_REDUCTION_VS_JPL_VECTORS_V0002_SUMMARY.csv"
    reduction_csv = output_dir / "IERS_REDUCTION_VS_JPL_VECTORS_V0002_REDUCTION.csv"
    publication_png = output_dir / "SOLAR_PARALLAX_HISTORICAL_REDUCTION_JPL_AUDIT_V0002.png"
    summary.to_csv(summary_csv, index=False, float_format="%.15f")
    pd.DataFrame([reduction]).to_csv(reduction_csv, index=False, float_format="%.15f")
    render_publication_summary(
        summary,
        geometry,
        reduction,
        uncertainty,
        jpl_source,
        vector_paths,
        publication_png,
        arguments.dpi,
    )

    equation_checks = {
        "JPL vector magnitude": abs(
            float(geometry["distance_m"])
            - float(np.linalg.norm(geometry["sun_vector_km"])) * 1000.0
        ) <= 0.000001,
        "Exact JPL reduction equals Case 2": abs(reduction["exact_minus_case_2_microarcsec"]) <= 0.000001,
        "Case 2 rounds to 8.794148": round(reduction["case_2_arcsec"], 6) == TARGET_ARCSEC,
        "Historical uncertainty rounds to ±0.000007": round(uncertainty[2], 6) == PASS_TOLERANCE_ARCSEC,
    }
    failed_checks = [name for name, passed in equation_checks.items() if not passed]
    if failed_checks:
        raise RuntimeError("Equation checks failed: " + ", ".join(failed_checks))

    try:
        from IPython.display import Image, display
        display(Image(filename=str(publication_png)))
    except Exception:
        print(f"PUBLICATION IMAGE: {publication_png}")

    print(f"CODE OUTPUT: {VERSION}")
    print("CODE INPUTS")
    print(f"Program: {PROGRAM}")
    print(f"JPL source: {jpl_source}")
    print(f"JPL master: {master_path}")
    print("COMMENTS")
    print("Black-and-white Matplotlib publication plate; no AI imagery; no ASCII results table.")
    print("RESULTS")
    print(f"Exact reduced parallax: {reduction['exact_arcsec']:.12f} arcsec")
    print("OUTPUT SUMMARY")
    print(f"Publication image: {publication_png}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Reduction CSV: {reduction_csv}")
    print(f"JPL Sun vectors: {vector_paths['sun']}")
    print(f"JPL Venus vectors: {vector_paths['venus']}")
    print(f"JPL combined master: {vector_paths['master']}")
    print("PAPER COMPARISON")
    print(f"IAU-1976 Case 2: {reduction['case_2_arcsec']:.12f} arcsec")
    print("EQUATION STATUS")
    print("All checks: PASS")
    print(f"LOCAL TIMESTAMP: {datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# V0002
