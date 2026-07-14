# V0110
# Audit reference: 1769 geocentric Venus/Sun fixed-tangent-plane curvature comparison from fresh JPL Horizons geometric vectors.

from __future__ import annotations

import csv
import io
import json
import math
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

VERSION = "V0110"
SCRIPT_NAME = "VENUS_1769_GEOCENTRIC_RHO_EARTH_VS_VENUS_RAW_V0110.py"
PNG_NAME = "VENUS_1769_GEOCENTRIC_RHO_EARTH_VS_VENUS_RAW_V0110.png"
CSV_NAME = "VENUS_1769_GEOCENTRIC_RHO_EARTH_VS_VENUS_RAW_V0110.csv"
HORIZONS_ENDPOINT = "https://ssd.jpl.nasa.gov/api/horizons.api"
ARCSEC_PER_RAD = 206264.80624709636
QUERY_START_UTC = "1769-06-03 21:30"
QUERY_STOP_UTC = "1769-06-03 23:10"
STEP_SIZE = "1 m"
PLOT_WINDOW_MIN = 30.0
FIT_HALF_WIDTH_POINTS = 7
FIT_DEGREE = 6
OUTPUT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


@dataclass(frozen=True)
class HorizonsVectors:
    target_id: str
    target_name: str
    jd_ut: np.ndarray
    calendar_ut: tuple[str, ...]
    position_km: np.ndarray
    source_text: str
    request_url: str


def quoted(value: str) -> str:
    return f"'{value}'"


def build_horizons_url(target_id: str) -> str:
    params = {
        "format": "json",
        "COMMAND": quoted(target_id),
        "OBJ_DATA": quoted("NO"),
        "MAKE_EPHEM": quoted("YES"),
        "EPHEM_TYPE": quoted("VECTORS"),
        "CENTER": quoted("500@399"),
        "START_TIME": quoted(QUERY_START_UTC),
        "STOP_TIME": quoted(QUERY_STOP_UTC),
        "STEP_SIZE": quoted(STEP_SIZE),
        "TIME_TYPE": quoted("UT"),
        "TIME_DIGITS": quoted("FRACSEC"),
        "CAL_TYPE": quoted("GREGORIAN"),
        "REF_PLANE": quoted("FRAME"),
        "REF_SYSTEM": quoted("ICRF"),
        "OUT_UNITS": quoted("KM-S"),
        "VEC_TABLE": quoted("1"),
        "VEC_CORR": quoted("NONE"),
        "CSV_FORMAT": quoted("YES"),
        "VEC_LABELS": quoted("NO"),
    }
    return HORIZONS_ENDPOINT + "?" + urllib.parse.urlencode(params)


def fetch_json(url: str, attempts: int = 3, timeout_s: int = 60) -> dict:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": f"{SCRIPT_NAME} NASA-JPL-Horizons-audit"},
            )
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                if response.status != 200:
                    raise RuntimeError(f"Horizons HTTP status {response.status}")
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # network errors are reported after deterministic retries
            last_error = exc
    raise RuntimeError(f"Unable to retrieve fresh JPL Horizons vectors: {last_error}")


def extract_target_name(result_text: str, fallback: str) -> str:
    for line in result_text.splitlines():
        if line.startswith("Target body name:"):
            return line.split(":", 1)[1].strip()
    return fallback


def parse_horizons_csv(result_text: str, target_id: str, request_url: str) -> HorizonsVectors:
    if "$$SOE" not in result_text or "$$EOE" not in result_text:
        diagnostic = result_text[:1200].replace("\n", " ")
        raise RuntimeError(f"Horizons response has no vector table: {diagnostic}")

    block = result_text.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    rows = csv.reader(io.StringIO(block.strip()))
    jd_values: list[float] = []
    calendar_values: list[str] = []
    positions: list[list[float]] = []

    for row in rows:
        cells = [cell.strip() for cell in row]
        if not cells or not cells[0]:
            continue
        try:
            jd = float(cells[0])
        except ValueError:
            continue
        if len(cells) < 5:
            raise RuntimeError(f"Unexpected Horizons CSV row: {cells}")
        try:
            xyz = [float(cells[2]), float(cells[3]), float(cells[4])]
        except ValueError as exc:
            raise RuntimeError(f"Unable to parse Horizons XYZ vector row: {cells}") from exc
        jd_values.append(jd)
        calendar_values.append(cells[1])
        positions.append(xyz)

    if len(jd_values) < 20:
        raise RuntimeError(f"Insufficient Horizons vectors parsed: {len(jd_values)}")

    return HorizonsVectors(
        target_id=target_id,
        target_name=extract_target_name(result_text, target_id),
        jd_ut=np.asarray(jd_values, dtype=float),
        calendar_ut=tuple(calendar_values),
        position_km=np.asarray(positions, dtype=float),
        source_text=result_text,
        request_url=request_url,
    )


def fetch_vectors(target_id: str) -> HorizonsVectors:
    url = build_horizons_url(target_id)
    payload = fetch_json(url)
    signature = payload.get("signature", {})
    if "result" not in payload:
        raise RuntimeError(f"Horizons API returned no result field: {payload}")
    if "NASA/JPL" not in str(signature.get("source", "")):
        raise RuntimeError(f"Unexpected Horizons API signature: {signature}")
    return parse_horizons_csv(payload["result"], target_id, url)


def assert_matching_epochs(a: HorizonsVectors, b: HorizonsVectors) -> None:
    if a.jd_ut.shape != b.jd_ut.shape:
        raise RuntimeError("Sun and Venus vector tables have different lengths")
    difference_seconds = np.max(np.abs(a.jd_ut - b.jd_ut)) * 86400.0
    if difference_seconds > 1.0e-6:
        raise RuntimeError(f"Sun and Venus epochs differ by {difference_seconds:.9f} s")


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0.0):
        raise RuntimeError("Zero-length JPL vector encountered")
    return vectors / norms[:, None]


def angular_separation_rad(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    cross_norm = np.linalg.norm(np.cross(u, v), axis=1)
    dot = np.einsum("ij,ij->i", u, v)
    return np.arctan2(cross_norm, dot)


def polynomial_minimum(x: np.ndarray, y: np.ndarray) -> tuple[float, np.polynomial.Polynomial]:
    index = int(np.argmin(y))
    lo = max(0, index - FIT_HALF_WIDTH_POINTS)
    hi = min(len(x), index + FIT_HALF_WIDTH_POINTS + 1)
    if hi - lo < FIT_DEGREE + 1:
        raise RuntimeError("Not enough samples around closest approach for interpolation")

    x_center = float(x[index])
    local_x = x[lo:hi] - x_center
    local_y = y[lo:hi]
    fitted = np.polynomial.Polynomial.fit(local_x, local_y, FIT_DEGREE).convert()
    derivative_roots = fitted.deriv().roots()

    candidates = [0.0]
    lower = float(local_x.min())
    upper = float(local_x.max())
    for root in derivative_roots:
        if abs(root.imag) < 1.0e-10 and lower <= root.real <= upper:
            candidates.append(float(root.real))
    candidates.extend([lower, upper])

    best_local = min(candidates, key=lambda value: float(fitted(value)))
    second_derivative = float(fitted.deriv(2)(best_local))
    if second_derivative <= 0.0:
        raise RuntimeError("Interpolated closest-approach stationary point is not a minimum")
    return x_center + best_local, fitted


def interpolate_components(jd: np.ndarray, values: np.ndarray, jd_eval: np.ndarray, degree: int = 5) -> np.ndarray:
    output = np.empty((len(jd_eval), values.shape[1]), dtype=float)
    center = float(np.mean(jd_eval))
    scale = 1.0 / 1440.0
    x = (jd - center) / scale
    x_eval = (jd_eval - center) / scale
    for column in range(values.shape[1]):
        fitted = np.polynomial.Polynomial.fit(x, values[:, column], degree).convert()
        output[:, column] = fitted(x_eval)
    return output


def tangent_basis(reference_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_axis = np.array([0.0, 0.0, 1.0])
    x_axis = np.array([1.0, 0.0, 0.0])
    seed = z_axis if abs(float(np.dot(reference_unit, z_axis))) < 0.95 else x_axis
    east = np.cross(seed, reference_unit)
    east /= np.linalg.norm(east)
    north = np.cross(reference_unit, east)
    north /= np.linalg.norm(north)
    return east, north


def gnomonic_arcsec(unit_vectors: np.ndarray, reference_unit: np.ndarray, east: np.ndarray, north: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    denominator = unit_vectors @ reference_unit
    if np.any(denominator <= 0.0):
        raise RuntimeError("A vector lies outside the forward fixed tangent plane")
    x = (unit_vectors @ east) / denominator * ARCSEC_PER_RAD
    y = (unit_vectors @ north) / denominator * ARCSEC_PER_RAD
    return x, y


def normalize_zero_to_one(values: np.ndarray) -> np.ndarray:
    minimum = float(np.min(values))
    span = float(np.max(values) - minimum)
    if span <= 0.0:
        raise RuntimeError("Cannot normalize a constant curve")
    return (values - minimum) / span


def jd_to_calendar_utc(jd: float) -> str:
    # Proleptic Gregorian conversion; Horizons was explicitly queried with CAL_TYPE=GREGORIAN and TIME_TYPE=UT.
    jd_shifted = jd + 0.5
    z = int(math.floor(jd_shifted))
    f = jd_shifted - z
    alpha = int((z - 1867216.25) / 36524.25)
    a = z + 1 + alpha - alpha // 4
    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)
    day_float = b - d - int(30.6001 * e) + f
    day = int(math.floor(day_float))
    fraction = day_float - day
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    seconds_total = fraction * 86400.0
    hour = int(seconds_total // 3600)
    minute = int((seconds_total - hour * 3600) // 60)
    second = seconds_total - hour * 3600 - minute * 60
    if second >= 59.9995:
        second = 0.0
        minute += 1
    if minute >= 60:
        minute = 0
        hour += 1
    return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:06.3f} UTC"


def save_csv(
    path: Path,
    elapsed_min: np.ndarray,
    jd_ut: np.ndarray,
    venus_x: np.ndarray,
    venus_y: np.ndarray,
    venus_rho: np.ndarray,
    earth_x: np.ndarray,
    earth_y: np.ndarray,
    earth_rho: np.ndarray,
    venus_normalized: np.ndarray,
    earth_normalized: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "elapsed_minutes_from_closest_approach",
            "jd_ut",
            "utc_gregorian",
            "venus_x_arcsec",
            "venus_y_arcsec",
            "venus_rho_arcsec",
            "earth_sun_x_arcsec",
            "earth_sun_y_arcsec",
            "earth_sun_rho_arcsec_raw",
            "venus_normalized_0_1",
            "earth_normalized_0_1",
        ])
        for values in zip(
            elapsed_min,
            jd_ut,
            venus_x,
            venus_y,
            venus_rho,
            earth_x,
            earth_y,
            earth_rho,
            venus_normalized,
            earth_normalized,
        ):
            t, jd, vx, vy, vr, ex, ey, er, vn, en = values
            writer.writerow([
                f"{t:.9f}",
                f"{jd:.12f}",
                jd_to_calendar_utc(float(jd)),
                f"{vx:.12f}",
                f"{vy:.12f}",
                f"{vr:.12f}",
                f"{ex:.12f}",
                f"{ey:.12f}",
                f"{er:.12f}",
                f"{vn:.12f}",
                f"{en:.12f}",
            ])


def make_figure(
    path: Path,
    elapsed_min: np.ndarray,
    venus_rho: np.ndarray,
    earth_rho: np.ndarray,
    venus_normalized: np.ndarray,
    earth_normalized: np.ndarray,
    table_rows: list[tuple[str, str]],
) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.45,
        "ytick.minor.width": 0.45,
    })
    fig, axes = plt.subplots(2, 1, figsize=(8.3, 10.2), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.975, top=0.93, bottom=0.245, hspace=0.24)

    ax1, ax2 = axes
    ax1.plot(elapsed_min, venus_rho, linewidth=0.9, label="Venus")
    ax1.plot(elapsed_min, earth_rho, linewidth=0.9, label="Earth")
    ax1.set_ylabel("Raw tangent-plane distance (arcsec)")
    ax1.set_title("Raw fixed Sun-screen tangent-plane distances")
    ax1.grid(True, linewidth=0.35, alpha=0.45)
    ax1.legend(frameon=False, loc="best")

    ax2.plot(elapsed_min, venus_normalized, linewidth=0.9, label="Venus")
    ax2.plot(elapsed_min, earth_normalized, linewidth=0.9, label="Earth")
    ax2.set_xlabel("Minutes from true geocentric closest approach")
    ax2.set_ylabel("Normalized distance (0–1)")
    ax2.set_title("Normalized comparison only")
    ax2.grid(True, linewidth=0.35, alpha=0.45)
    ax2.legend(frameon=False, loc="best")
    ax2.set_xlim(-PLOT_WINDOW_MIN, PLOT_WINDOW_MIN)

    fig.suptitle("1769 Venus Transit — Geocentric Earth–Sun vs Venus Tangent-Plane Geometry", fontsize=13.0, fontweight="bold")
    table_text = "\n".join(f"{label:<28} {value}" for label, value in table_rows)
    fig.text(
        0.105,
        0.055,
        table_text,
        ha="left",
        va="bottom",
        family="DejaVu Sans Mono",
        fontsize=8.2,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "white", "edgecolor": "0.55", "linewidth": 0.6},
    )
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.show()


def print_section(title: str, lines: list[str]) -> None:
    print(title)
    for line in lines:
        print(line)


def main() -> None:
    print_section("CODE INPUTS", [
        f"Version: {VERSION}",
        f"Query interval: {QUERY_START_UTC} to {QUERY_STOP_UTC} UT",
        f"JPL step: {STEP_SIZE}",
        f"Analysis window: ±{PLOT_WINDOW_MIN:.0f} minutes",
    ])
    print_section("COMMENTS", [
        "Fresh geometric state vectors are requested directly from NASA/JPL Horizons.",
        "The tangent plane is fixed to the geocentric Sun direction at interpolated closest approach.",
    ])

    sun = fetch_vectors("10")
    venus = fetch_vectors("299")
    assert_matching_epochs(sun, venus)

    jd = sun.jd_ut
    sun_unit = normalize_rows(sun.position_km)
    venus_unit = normalize_rows(venus.position_km)
    separation_rad = angular_separation_rad(venus_unit, sun_unit)
    separation_sq = separation_rad * separation_rad

    ca_jd, _ = polynomial_minimum(jd, separation_sq)
    if not (jd[0] < ca_jd < jd[-1]):
        raise RuntimeError("Interpolated closest approach lies outside the JPL query interval")

    output_elapsed_min = np.linspace(-PLOT_WINDOW_MIN, PLOT_WINDOW_MIN, 121)
    output_jd = ca_jd + output_elapsed_min / 1440.0
    interpolation_mask = (jd >= output_jd[0] - 10.0 / 1440.0) & (jd <= output_jd[-1] + 10.0 / 1440.0)
    jd_local = jd[interpolation_mask]
    sun_local = sun.position_km[interpolation_mask]
    venus_local = venus.position_km[interpolation_mask]

    sun_eval = interpolate_components(jd_local, sun_local, output_jd)
    venus_eval = interpolate_components(jd_local, venus_local, output_jd)
    ca_sun = interpolate_components(jd_local, sun_local, np.asarray([ca_jd]))[0]
    reference_unit = ca_sun / np.linalg.norm(ca_sun)
    east, north = tangent_basis(reference_unit)

    sun_eval_unit = normalize_rows(sun_eval)
    venus_eval_unit = normalize_rows(venus_eval)
    sun_x, sun_y = gnomonic_arcsec(sun_eval_unit, reference_unit, east, north)
    venus_absolute_x, venus_absolute_y = gnomonic_arcsec(venus_eval_unit, reference_unit, east, north)

    venus_x = venus_absolute_x - sun_x
    venus_y = venus_absolute_y - sun_y
    venus_rho = np.hypot(venus_x, venus_y)
    earth_x = sun_x
    earth_y = sun_y
    earth_rho = np.hypot(earth_x, earth_y)
    venus_normalized = normalize_zero_to_one(venus_rho)
    earth_normalized = normalize_zero_to_one(earth_rho)

    ca_venus_vector = interpolate_components(jd_local, venus_local, np.asarray([ca_jd]))[0]
    ca_venus_unit = ca_venus_vector / np.linalg.norm(ca_venus_vector)
    ca_sep_arcsec = math.atan2(
        np.linalg.norm(np.cross(ca_venus_unit, reference_unit)),
        float(np.dot(ca_venus_unit, reference_unit)),
    ) * ARCSEC_PER_RAD

    venus_minimum = float(np.min(venus_rho))
    earth_minimum = float(np.min(earth_rho))
    ca_utc = jd_to_calendar_utc(ca_jd)
    projection_description = "Fixed Sun tangent plane at closest approach"
    source_description = "NASA/JPL Horizons geometric vectors (VEC_CORR=NONE)"

    if abs(venus_minimum - ca_sep_arcsec) > 5.0e-5:
        raise RuntimeError(
            f"Equation check failed: tangent-plane minimum {venus_minimum:.9f} arcsec "
            f"vs direct angular separation {ca_sep_arcsec:.9f} arcsec"
        )

    csv_path = OUTPUT_DIR / CSV_NAME
    png_path = OUTPUT_DIR / PNG_NAME
    save_csv(
        csv_path,
        output_elapsed_min,
        output_jd,
        venus_x,
        venus_y,
        venus_rho,
        earth_x,
        earth_y,
        earth_rho,
        venus_normalized,
        earth_normalized,
    )

    table_rows = [
        ("Closest approach UTC", ca_utc),
        ("Closest approach JD", f"{ca_jd:.12f} UT"),
        ("Venus minimum ρ", f"{venus_minimum:.9f} arcsec"),
        ("Earth minimum projected ρ", f"{earth_minimum:.9f} arcsec"),
        ("Window", f"−{PLOT_WINDOW_MIN:.0f} to +{PLOT_WINDOW_MIN:.0f} minutes"),
        ("Projection", projection_description),
        ("JPL source", source_description),
    ]

    print_section("RESULTS", [f"{label}: {value}" for label, value in table_rows])
    make_figure(
        png_path,
        output_elapsed_min,
        venus_rho,
        earth_rho,
        venus_normalized,
        earth_normalized,
        table_rows,
    )
    print_section("OUTPUT SUMMARY", [f"PNG: {png_path}", f"CSV: {csv_path}"])
    print_section("PAPER COMPARISON", ["Not used; this standalone audit derives all plotted quantities from fresh JPL vectors."])
    print_section("EQUATION STATUS", [
        "PASS: closest approach obtained by minimizing an interpolating polynomial of angular separation squared.",
        "PASS: direct angular separation agrees with fixed-plane Venus minimum within 0.000050 arcsec.",
        "PASS: Earth curve is raw fixed-plane Earth–Sun sightline displacement; no normalization in Panel 1.",
    ])
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"{VERSION} COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{VERSION} ERROR: {exc}", file=sys.stderr)
        raise

# V0110
