# V0110
# Audit reference: single-panel second-order curvature residuals from fresh JPL Horizons geometric vectors.

from __future__ import annotations

import csv
import io
import json
import math
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
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
WINDOW_MIN = 30.0
OUTPUT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


@dataclass(frozen=True)
class VectorTable:
    target_id: str
    target_name: str
    jd: np.ndarray
    xyz_km: np.ndarray
    request_url: str


def quoted(value: str) -> str:
    return f"'{value}'"


def horizons_url(target_id: str) -> str:
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


def fetch_json(url: str) -> dict:
    last_error: Exception | None = None
    for _ in range(3):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": f"{SCRIPT_NAME} JPL-Horizons-audit"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 200:
                    raise RuntimeError(f"Horizons HTTP status {response.status}")
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to retrieve fresh JPL Horizons vectors: {last_error}")


def fetch_vectors(target_id: str) -> VectorTable:
    url = horizons_url(target_id)
    payload = fetch_json(url)
    signature = payload.get("signature", {})
    if "NASA/JPL" not in str(signature.get("source", "")):
        raise RuntimeError(f"Unexpected Horizons signature: {signature}")
    result = payload.get("result", "")
    if "$$SOE" not in result or "$$EOE" not in result:
        raise RuntimeError("Horizons response contains no vector table")

    target_name = target_id
    for line in result.splitlines():
        if line.startswith("Target body name:"):
            target_name = line.split(":", 1)[1].strip()
            break

    jd_values: list[float] = []
    xyz_values: list[list[float]] = []
    block = result.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    for row in csv.reader(io.StringIO(block.strip())):
        cells = [cell.strip() for cell in row]
        if len(cells) < 5:
            continue
        try:
            jd_values.append(float(cells[0]))
            xyz_values.append([float(cells[2]), float(cells[3]), float(cells[4])])
        except ValueError:
            continue

    if len(jd_values) < 50:
        raise RuntimeError(f"Insufficient JPL vectors parsed: {len(jd_values)}")

    return VectorTable(
        target_id=target_id,
        target_name=target_name,
        jd=np.asarray(jd_values, dtype=float),
        xyz_km=np.asarray(xyz_values, dtype=float),
        request_url=url,
    )


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0.0):
        raise RuntimeError("Zero-length JPL vector encountered")
    return values / norms[:, None]


def angular_separation(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.arctan2(
        np.linalg.norm(np.cross(u, v), axis=1),
        np.einsum("ij,ij->i", u, v),
    )


def interpolate_minimum(jd: np.ndarray, values: np.ndarray) -> float:
    index = int(np.argmin(values))
    lo = max(0, index - 7)
    hi = min(len(jd), index + 8)
    center = float(jd[index])
    x_minutes = (jd[lo:hi] - center) * 1440.0
    fit = np.polynomial.Polynomial.fit(x_minutes, values[lo:hi], 6).convert()
    candidates = [0.0]
    lower = float(x_minutes.min())
    upper = float(x_minutes.max())
    for root in fit.deriv().roots():
        if abs(root.imag) < 1.0e-10 and lower <= root.real <= upper:
            candidates.append(float(root.real))
    minimum_x = min(candidates, key=lambda value: float(fit(value)))
    if float(fit.deriv(2)(minimum_x)) <= 0.0:
        raise RuntimeError("Interpolated stationary point is not a minimum")
    return center + minimum_x / 1440.0


def interpolate_vectors(jd: np.ndarray, xyz: np.ndarray, jd_eval: np.ndarray) -> np.ndarray:
    center = float(np.mean(jd_eval))
    x = (jd - center) * 1440.0
    x_eval = (jd_eval - center) * 1440.0
    output = np.empty((len(jd_eval), 3), dtype=float)
    for column in range(3):
        fit = np.polynomial.Polynomial.fit(x, xyz[:, column], 5).convert()
        output[:, column] = fit(x_eval)
    return output


def tangent_basis(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, seed))) > 0.95:
        seed = np.array([1.0, 0.0, 0.0])
    east = np.cross(seed, reference)
    east /= np.linalg.norm(east)
    north = np.cross(reference, east)
    north /= np.linalg.norm(north)
    return east, north


def gnomonic(
    unit_vectors: np.ndarray,
    reference: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
) -> np.ndarray:
    denominator = unit_vectors @ reference
    if np.any(denominator <= 0.0):
        raise RuntimeError("Vector outside forward tangent plane")
    x = (unit_vectors @ east) / denominator * ARCSEC_PER_RAD
    y = (unit_vectors @ north) / denominator * ARCSEC_PER_RAD
    return np.column_stack((x, y))


def tangent_linear_residual(elapsed_min: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zero_index = int(np.argmin(np.abs(elapsed_min)))
    if abs(float(elapsed_min[zero_index])) > 1.0e-12:
        raise RuntimeError("Elapsed-time grid does not contain closest approach")

    fit_half_width = 5
    lo = max(0, zero_index - fit_half_width)
    hi = min(len(elapsed_min), zero_index + fit_half_width + 1)
    local_t = elapsed_min[lo:hi]

    velocity = np.empty(2, dtype=float)
    origin = xy[zero_index].copy()
    for axis in range(2):
        fit = np.polynomial.Polynomial.fit(local_t, xy[lo:hi, axis], 3).convert()
        velocity[axis] = float(fit.deriv()(0.0))

    tangent = origin[None, :] + elapsed_min[:, None] * velocity[None, :]
    residual_xy = xy - tangent
    residual_rho = np.linalg.norm(residual_xy, axis=1)
    return residual_xy, residual_rho, velocity


def jd_to_utc(jd: float) -> str:
    shifted = jd + 0.5
    z = int(math.floor(shifted))
    f = shifted - z
    alpha = int((z - 1867216.25) / 36524.25)
    a = z + 1 + alpha - alpha // 4
    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)
    day_float = b - d - int(30.6001 * e) + f
    day = int(day_float)
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    seconds = (day_float - day) * 86400.0
    hour = int(seconds // 3600)
    minute = int((seconds - hour * 3600) // 60)
    second = seconds - hour * 3600 - minute * 60
    return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:06.3f} UTC"


def write_csv(
    path: Path,
    elapsed: np.ndarray,
    jd: np.ndarray,
    venus_xy: np.ndarray,
    earth_xy: np.ndarray,
    venus_residual_xy: np.ndarray,
    earth_residual_xy: np.ndarray,
    venus_curvature: np.ndarray,
    earth_curvature: np.ndarray,
) -> None:
    headers = (
        "elapsed_minutes_from_closest_approach",
        "jd_ut",
        "venus_x_arcsec",
        "venus_y_arcsec",
        "earth_x_arcsec",
        "earth_y_arcsec",
        "venus_curvature_dx_arcsec",
        "venus_curvature_dy_arcsec",
        "earth_curvature_dx_arcsec",
        "earth_curvature_dy_arcsec",
        "venus_curvature_rho_arcsec",
        "earth_curvature_rho_arcsec",
    )
    columns = (
        elapsed,
        jd,
        venus_xy[:, 0],
        venus_xy[:, 1],
        earth_xy[:, 0],
        earth_xy[:, 1],
        venus_residual_xy[:, 0],
        venus_residual_xy[:, 1],
        earth_residual_xy[:, 0],
        earth_residual_xy[:, 1],
        venus_curvature,
        earth_curvature,
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in zip(*columns):
            writer.writerow([f"{float(value):.12f}" for value in row])


def make_figure(
    path: Path,
    elapsed: np.ndarray,
    venus_curvature: np.ndarray,
    earth_curvature: np.ndarray,
) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.linewidth": 0.4,
    })

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.14)
    ax.plot(elapsed, venus_curvature, linewidth=1.0, label="Venus")
    ax.plot(elapsed, earth_curvature, linewidth=1.0, label="Earth")
    ax.set_xlim(-WINDOW_MIN, WINDOW_MIN)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Minutes from true geocentric closest approach")
    ax.set_ylabel("Curvature residual from tangent line (arcsec)")
    ax.set_title("1769 Venus Transit — JPL Tangent-Plane Curvature Residuals", fontweight="bold")
    ax.grid(True, which="major", alpha=0.55)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.20)
    ax.legend(frameon=False, loc="upper center", ncol=2)
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.show()


def section(title: str, lines: list[str]) -> None:
    print(title)
    for line in lines:
        print(line)


def main() -> None:
    section("CODE INPUTS", [
        f"Version: {VERSION}",
        f"Query interval: {QUERY_START_UTC} to {QUERY_STOP_UTC} UT",
        f"JPL step: {STEP_SIZE}",
        f"Analysis window: ±{WINDOW_MIN:.0f} minutes",
    ])
    section("COMMENTS", [
        "Fresh geometric state vectors are requested directly from NASA/JPL Horizons.",
        "The fixed tangent plane is defined by the geocentric Sun direction at closest approach.",
        "Each plotted curve is the magnitude remaining after subtracting its position and tangent velocity at closest approach.",
    ])

    sun = fetch_vectors("10")
    venus = fetch_vectors("299")
    if sun.jd.shape != venus.jd.shape:
        raise RuntimeError("Sun and Venus JPL vector tables have different lengths")
    if np.max(np.abs(sun.jd - venus.jd)) * 86400.0 > 1.0e-6:
        raise RuntimeError("Sun and Venus JPL epochs do not match")

    sun_unit = normalize_rows(sun.xyz_km)
    venus_unit = normalize_rows(venus.xyz_km)
    separation = angular_separation(venus_unit, sun_unit)
    ca_jd = interpolate_minimum(sun.jd, separation * separation)

    elapsed = np.linspace(-WINDOW_MIN, WINDOW_MIN, 121)
    output_jd = ca_jd + elapsed / 1440.0
    mask = (
        (sun.jd >= output_jd[0] - 10.0 / 1440.0)
        & (sun.jd <= output_jd[-1] + 10.0 / 1440.0)
    )
    local_jd = sun.jd[mask]
    sun_eval = interpolate_vectors(local_jd, sun.xyz_km[mask], output_jd)
    venus_eval = interpolate_vectors(local_jd, venus.xyz_km[mask], output_jd)
    ca_sun = interpolate_vectors(local_jd, sun.xyz_km[mask], np.asarray([ca_jd]))[0]

    reference = ca_sun / np.linalg.norm(ca_sun)
    east, north = tangent_basis(reference)
    sun_xy = gnomonic(normalize_rows(sun_eval), reference, east, north)
    venus_absolute_xy = gnomonic(normalize_rows(venus_eval), reference, east, north)
    venus_relative_xy = venus_absolute_xy - sun_xy
    earth_xy = sun_xy

    venus_residual_xy, venus_curvature, venus_velocity = tangent_linear_residual(elapsed, venus_relative_xy)
    earth_residual_xy, earth_curvature, earth_velocity = tangent_linear_residual(elapsed, earth_xy)

    zero_index = int(np.argmin(np.abs(elapsed)))
    if venus_curvature[zero_index] > 1.0e-10 or earth_curvature[zero_index] > 1.0e-10:
        raise RuntimeError("Curvature residual does not vanish at closest approach")

    csv_path = OUTPUT_DIR / CSV_NAME
    png_path = OUTPUT_DIR / PNG_NAME
    write_csv(
        csv_path,
        elapsed,
        output_jd,
        venus_relative_xy,
        earth_xy,
        venus_residual_xy,
        earth_residual_xy,
        venus_curvature,
        earth_curvature,
    )

    section("RESULTS", [
        f"Closest approach UTC: {jd_to_utc(ca_jd)}",
        f"Closest approach JD: {ca_jd:.12f} UT",
        f"Venus tangent velocity: {np.linalg.norm(venus_velocity):.12f} arcsec/min",
        f"Earth tangent velocity: {np.linalg.norm(earth_velocity):.12f} arcsec/min",
        f"Venus curvature at −30 min: {venus_curvature[0]:.12f} arcsec",
        f"Venus curvature at +30 min: {venus_curvature[-1]:.12f} arcsec",
        f"Earth curvature at −30 min: {earth_curvature[0]:.12f} arcsec",
        f"Earth curvature at +30 min: {earth_curvature[-1]:.12f} arcsec",
    ])

    make_figure(png_path, elapsed, venus_curvature, earth_curvature)
    section("OUTPUT SUMMARY", [f"PNG: {png_path}", f"CSV: {csv_path}"])
    section("PAPER COMPARISON", ["Not used; all plotted quantities are derived from fresh JPL vectors."])
    section("EQUATION STATUS", [
        "PASS: closest approach is obtained by interpolated minimization of angular separation squared.",
        "PASS: constant position and first-order tangent motion are removed independently for Venus and Earth.",
        "PASS: plotted residual magnitudes are zero at closest approach and retain only nonlinear tangent-plane motion.",
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