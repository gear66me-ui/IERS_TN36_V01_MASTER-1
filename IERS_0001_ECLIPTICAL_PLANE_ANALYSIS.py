"""
IERS TN36 — Ecliptical Plane Analysis
Part H — North Pole 2012 track-angle sanity check
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

VERSION = "IERS-0001-H"
HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"
ARCSEC_PER_RAD = 206_264.80624709636
SECONDS_PER_DAY = 86_400.0

START_TIME_UTC = "2012-06-05 21:00"
STOP_TIME_UTC = "2012-06-06 07:00"
STEP_SIZE = "1 min"
FIT_HALF_WINDOW_HOURS = 3.25


@dataclass(frozen=True)
class Observer:
    name: str
    longitude_deg: float
    latitude_deg: float
    elevation_km: float
    center_body: int = 399

    @property
    def center(self) -> str:
        return f"coord@{self.center_body}"

    @property
    def site_coord(self) -> str:
        return (
            f"{self.longitude_deg:.8f},"
            f"{self.latitude_deg:.8f},"
            f"{self.elevation_km:.6f}"
        )


NORTH_POLE = Observer(
    name="Geodetic North Pole",
    longitude_deg=0.0,
    latitude_deg=90.0,
    elevation_km=0.0,
)

TARGETS = {"Sun": "10", "Venus": "299"}


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    magnitude = np.linalg.norm(vector)
    if not np.isfinite(magnitude) or magnitude == 0.0:
        raise ValueError("Cannot normalize a zero or non-finite vector.")
    return vector / magnitude


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    magnitudes = np.linalg.norm(vectors, axis=1)
    if np.any(~np.isfinite(magnitudes)) or np.any(magnitudes == 0.0):
        raise ValueError("Cannot normalize zero or non-finite vector rows.")
    return vectors / magnitudes[:, None]


def quoted(value: str) -> str:
    return f"'{value}'"


def build_horizons_parameters(target_id: str, observer: Observer) -> Dict[str, str]:
    return {
        "format": "json",
        "COMMAND": quoted(target_id),
        "OBJ_DATA": quoted("NO"),
        "MAKE_EPHEM": quoted("YES"),
        "EPHEM_TYPE": quoted("VECTORS"),
        "CENTER": quoted(observer.center),
        "COORD_TYPE": quoted("GEODETIC"),
        "SITE_COORD": quoted(observer.site_coord),
        "START_TIME": quoted(START_TIME_UTC),
        "STOP_TIME": quoted(STOP_TIME_UTC),
        "STEP_SIZE": quoted(STEP_SIZE),
        "TIME_TYPE": quoted("UT"),
        "TIME_DIGITS": quoted("SECONDS"),
        "REF_PLANE": quoted("FRAME"),
        "REF_SYSTEM": quoted("ICRF"),
        "OUT_UNITS": quoted("KM-S"),
        "VEC_TABLE": quoted("2"),
        "VEC_CORR": quoted("NONE"),
        "CSV_FORMAT": quoted("YES"),
        "VEC_LABELS": quoted("YES"),
    }


def request_horizons_text(target_id: str, observer: Observer) -> str:
    url = f"{HORIZONS_API}?{urlencode(build_horizons_parameters(target_id, observer))}"
    request = Request(
        url,
        headers={
            "User-Agent": "IERS-TN36-Ecliptical-Plane-Analysis/1.0",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Horizons HTTP {exc.code}: {detail[:500]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Horizons connection failed: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Horizons returned invalid JSON.") from exc

    signature = payload.get("signature", {})
    if "NASA/JPL" not in str(signature.get("source", "")):
        raise RuntimeError("Unexpected Horizons API source.")
    result = payload.get("result")
    if not isinstance(result, str) or not result.strip():
        raise RuntimeError(str(payload.get("error", "No Horizons result returned.")))
    return result


def canonical_header(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def parse_horizons_vectors(result_text: str, label: str) -> dict[str, np.ndarray]:
    lines = result_text.splitlines()
    try:
        soe_index = next(i for i, line in enumerate(lines) if "$$SOE" in line)
        eoe_index = next(
            i for i, line in enumerate(lines) if i > soe_index and "$$EOE" in line
        )
    except StopIteration as exc:
        raise ValueError(f"{label}: missing $$SOE/$$EOE markers.") from exc

    header = None
    for index in range(soe_index - 1, max(-1, soe_index - 20), -1):
        candidate = next(csv.reader([lines[index]], skipinitialspace=True))
        names = [canonical_header(item) for item in candidate]
        if all(axis in names for axis in ("X", "Y", "Z", "VX", "VY", "VZ")):
            header = candidate
            break
    if header is None:
        raise ValueError(f"{label}: vector header not found.")

    names = [canonical_header(item) for item in header]
    jd_index = next(
        (names.index(name) for name in ("JDUT", "JDTDB", "JD") if name in names),
        None,
    )
    if jd_index is None:
        raise ValueError(f"{label}: Julian-date column not found.")

    indices = {
        key: names.index(key.upper())
        for key in ("x", "y", "z", "vx", "vy", "vz")
    }

    epochs: list[float] = []
    positions: list[list[float]] = []
    velocities: list[list[float]] = []

    for row in csv.reader(lines[soe_index + 1:eoe_index], skipinitialspace=True):
        if not row or max([jd_index, *indices.values()]) >= len(row):
            continue
        try:
            epochs.append(float(row[jd_index]))
            positions.append([float(row[indices[key]]) for key in ("x", "y", "z")])
            velocities.append([float(row[indices[key]]) for key in ("vx", "vy", "vz")])
        except ValueError:
            continue

    table = {
        "jd_ut": np.asarray(epochs, dtype=float),
        "position_km": np.asarray(positions, dtype=float),
        "velocity_km_s": np.asarray(velocities, dtype=float),
    }
    validate_vector_table(table, label)
    return table


def validate_vector_table(table: dict[str, np.ndarray], label: str) -> None:
    epochs = table["jd_ut"]
    positions = table["position_km"]
    velocities = table["velocity_km_s"]
    if epochs.ndim != 1 or epochs.size < 3:
        raise ValueError(f"{label}: fewer than three valid epochs.")
    if positions.shape != (epochs.size, 3):
        raise ValueError(f"{label}: invalid position shape {positions.shape}.")
    if velocities.shape != (epochs.size, 3):
        raise ValueError(f"{label}: invalid velocity shape {velocities.shape}.")
    if not np.all(np.isfinite(epochs)):
        raise ValueError(f"{label}: non-finite epochs detected.")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise ValueError(f"{label}: non-finite state vectors detected.")
    if np.any(np.diff(epochs) <= 0.0):
        raise ValueError(f"{label}: epochs are not strictly increasing.")


def fetch_north_pole_vectors() -> dict[str, dict[str, np.ndarray]]:
    tables = {
        label: parse_horizons_vectors(
            request_horizons_text(target_id, NORTH_POLE),
            label,
        )
        for label, target_id in TARGETS.items()
    }
    sun_jd = tables["Sun"]["jd_ut"]
    venus_jd = tables["Venus"]["jd_ut"]
    if sun_jd.shape != venus_jd.shape or not np.allclose(
        sun_jd, venus_jd, atol=1e-12
    ):
        raise ValueError("Sun and Venus epochs are not synchronized.")
    return tables


def angular_separation_rad(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    unit_a = normalize(vector_a)
    unit_b = normalize(vector_b)
    return float(
        np.arctan2(
            np.linalg.norm(np.cross(unit_a, unit_b)),
            np.dot(unit_a, unit_b),
        )
    )


def state_spline(table: dict[str, np.ndarray]) -> tuple[float, CubicSpline]:
    jd0 = float(table["jd_ut"][0])
    seconds = (table["jd_ut"] - jd0) * SECONDS_PER_DAY
    return jd0, CubicSpline(seconds, table["position_km"], axis=0)


def refine_closest_approach(
    sun_table: dict[str, np.ndarray],
    venus_table: dict[str, np.ndarray],
) -> dict[str, object]:
    epochs = sun_table["jd_ut"]
    sampled = np.array(
        [
            angular_separation_rad(sun_vector, venus_vector)
            for sun_vector, venus_vector in zip(
                sun_table["position_km"],
                venus_table["position_km"],
            )
        ]
    )
    index = int(np.argmin(sampled))
    left = max(0, index - 3)
    right = min(epochs.size - 1, index + 3)

    jd0, sun_spline = state_spline(sun_table)
    venus_jd0, venus_spline = state_spline(venus_table)
    if abs(jd0 - venus_jd0) > 1e-12:
        raise ValueError("Interpolation origins do not match.")
    seconds = (epochs - jd0) * SECONDS_PER_DAY

    def objective(epoch_seconds: float) -> float:
        separation = angular_separation_rad(
            sun_spline(epoch_seconds),
            venus_spline(epoch_seconds),
        )
        return separation * separation

    result = minimize_scalar(
        objective,
        bounds=(float(seconds[left]), float(seconds[right])),
        method="bounded",
        options={"xatol": 1e-7},
    )
    if not result.success:
        raise RuntimeError(f"Closest-approach optimization failed: {result.message}")

    ca_seconds = float(result.x)
    return {
        "jd0": jd0,
        "ca_seconds": ca_seconds,
        "ca_jd_ut": jd0 + ca_seconds / SECONDS_PER_DAY,
        "ca_separation_arcsec": np.sqrt(float(result.fun)) * ARCSEC_PER_RAD,
        "sun_spline": sun_spline,
        "venus_spline": venus_spline,
    }


def fixed_x_ray_screen(unit_vectors: np.ndarray) -> np.ndarray:
    unit_vectors = np.asarray(unit_vectors, dtype=float)
    denominator = unit_vectors[:, 0]
    if np.any(np.abs(denominator) < 1e-10):
        raise ValueError("A direction is singular on the fixed +X ray screen.")
    return np.column_stack(
        (unit_vectors[:, 1] / denominator, unit_vectors[:, 2] / denominator)
    )


def fixed_yz_component_screen(unit_vectors: np.ndarray) -> np.ndarray:
    unit_vectors = np.asarray(unit_vectors, dtype=float)
    return unit_vectors[:, 1:3]


def ca_tangent_basis(
    ca_sun_direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sightline = normalize(ca_sun_direction)
    celestial_north = np.array([0.0, 0.0, 1.0])
    east = np.cross(celestial_north, sightline)
    if np.linalg.norm(east) < 1e-12:
        celestial_north = np.array([0.0, 1.0, 0.0])
        east = np.cross(celestial_north, sightline)
    east = normalize(east)
    north = normalize(np.cross(sightline, east))
    return east, north, sightline


def tangent_screen(
    unit_vectors: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    east, north, sightline = basis
    denominator = unit_vectors @ sightline
    if np.any(denominator <= 0.0):
        raise ValueError("A direction lies outside the tangent hemisphere.")
    return np.column_stack(
        ((unit_vectors @ east) / denominator, (unit_vectors @ north) / denominator)
    )


def signed_line_angle_deg(direction: np.ndarray) -> float:
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    while angle > 90.0:
        angle -= 180.0
    while angle <= -90.0:
        angle += 180.0
    return angle


def orthogonal_fit(points: np.ndarray) -> dict[str, object]:
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    direction = vh[0]
    if np.dot(direction, points[-1] - points[0]) < 0.0:
        direction = -direction
    normal = np.array([-direction[1], direction[0]])
    residual = (points - centroid) @ normal
    return {
        "centroid": centroid,
        "direction": direction,
        "normal": normal,
        "angle_deg": signed_line_angle_deg(direction),
        "rms_arcsec": float(np.sqrt(np.mean(residual**2))),
    }


def evaluate_frame(
    name: str,
    projector,
    sun_directions: np.ndarray,
    venus_directions: np.ndarray,
    fit_mask: np.ndarray,
    local_sun: np.ndarray,
    local_venus: np.ndarray,
) -> dict[str, object]:
    relative = (
        projector(venus_directions) - projector(sun_directions)
    ) * ARCSEC_PER_RAD
    fit = orthogonal_fit(relative[fit_mask])
    local_relative = (
        projector(local_venus) - projector(local_sun)
    ) * ARCSEC_PER_RAD
    instantaneous_direction = normalize(local_relative[1] - local_relative[0])
    ca_position = 0.5 * (local_relative[0] + local_relative[1])
    return {
        "name": name,
        "relative_arcsec": relative,
        "ca_position_arcsec": ca_position,
        "instantaneous_angle_deg": signed_line_angle_deg(instantaneous_direction),
        "fit_angle_deg": float(fit["angle_deg"]),
        "rms_arcsec": float(fit["rms_arcsec"]),
        "fit": fit,
    }


def derive_north_pole_frames(
    tables: dict[str, dict[str, np.ndarray]],
) -> dict[str, object]:
    sun_table = tables["Sun"]
    venus_table = tables["Venus"]
    closest = refine_closest_approach(sun_table, venus_table)

    epochs = sun_table["jd_ut"]
    epoch_seconds = (epochs - float(closest["jd0"])) * SECONDS_PER_DAY
    fit_mask = np.abs(
        epoch_seconds - float(closest["ca_seconds"])
    ) <= FIT_HALF_WINDOW_HOURS * 3600.0

    sun_directions = normalize_rows(sun_table["position_km"])
    venus_directions = normalize_rows(venus_table["position_km"])

    derivative_step = 30.0
    sample_seconds = np.array(
        [
            float(closest["ca_seconds"]) - derivative_step,
            float(closest["ca_seconds"]) + derivative_step,
        ]
    )
    local_sun = normalize_rows(closest["sun_spline"](sample_seconds))
    local_venus = normalize_rows(closest["venus_spline"](sample_seconds))

    ca_sun_direction = normalize(
        closest["sun_spline"](float(closest["ca_seconds"]))
    )
    tangent_basis = ca_tangent_basis(ca_sun_direction)

    frames = [
        evaluate_frame(
            "Fixed +X ray screen (Y/X,Z/X)",
            fixed_x_ray_screen,
            sun_directions,
            venus_directions,
            fit_mask,
            local_sun,
            local_venus,
        ),
        evaluate_frame(
            "Fixed ICRF Y-Z components",
            fixed_yz_component_screen,
            sun_directions,
            venus_directions,
            fit_mask,
            local_sun,
            local_venus,
        ),
        evaluate_frame(
            "CA tangent plane (east,north)",
            lambda directions: tangent_screen(directions, tangent_basis),
            sun_directions,
            venus_directions,
            fit_mask,
            local_sun,
            local_venus,
        ),
    ]

    return {
        "jd_ut": epochs,
        "fit_mask": fit_mask,
        "frames": frames,
        **closest,
    }


def jd_to_utc_text(jd_ut: float) -> str:
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    moment = epoch + timedelta(seconds=(jd_ut - 2440587.5) * SECONDS_PER_DAY)
    return moment.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def display_results(
    tables: dict[str, dict[str, np.ndarray]],
    result: dict[str, object],
) -> None:
    epochs = tables["Sun"]["jd_ut"]
    cadence = np.median(np.diff(epochs)) * SECONDS_PER_DAY

    print("IERS TN36 - North Pole Track-Angle Sanity Check")
    print(f"Version : {VERSION}")
    print(
        "Observer: "
        f"longitude={NORTH_POLE.longitude_deg:.6f} deg, "
        f"latitude={NORTH_POLE.latitude_deg:.6f} deg, "
        f"elevation={NORTH_POLE.elevation_km:.6f} km"
    )
    print(
        f"Vectors : {epochs.size:d} synchronized Sun/Venus rows, "
        f"{cadence:.6f} s cadence"
    )

    print("\nCLOSEST APPROACH")
    print("UTC                         JD UT                 SEPARATION arcsec")
    print(
        f"{jd_to_utc_text(float(result['ca_jd_ut'])):<27} "
        f"{float(result['ca_jd_ut']):19.10f} "
        f"{float(result['ca_separation_arcsec']):18.6f}"
    )

    print("\nNORTH POLE 2012 TRACK-ANGLE FRAME CHECK")
    print(
        "FRAME                              "
        "X CA arcsec    Y CA arcsec    "
        "INSTANT deg      FIT deg      RMS arcsec"
    )
    for frame in result["frames"]:
        ca = frame["ca_position_arcsec"]
        print(
            f"{frame['name']:<34} "
            f"{ca[0]:12.6f}  {ca[1]:12.6f}  "
            f"{frame['instantaneous_angle_deg']:11.6f}  "
            f"{frame['fit_angle_deg']:11.6f}  "
            f"{frame['rms_arcsec']:12.6f}"
        )


def plot_fixed_x_screen(result: dict[str, object]) -> None:
    frame = result["frames"][0]
    points = frame["relative_arcsec"]
    mask = result["fit_mask"]
    fit = frame["fit"]
    fitted_points = points[mask]
    along = (fitted_points - fit["centroid"]) @ fit["direction"]
    endpoints = fit["centroid"] + np.array(
        [along.min(), along.max()]
    )[:, None] * fit["direction"]

    fig, axis = plt.subplots(figsize=(8.0, 6.0))
    axis.plot(points[:, 0], points[:, 1], linewidth=1.1, label="North Pole track")
    axis.plot(
        endpoints[:, 0],
        endpoints[:, 1],
        "--",
        linewidth=1.0,
        label="Orthogonal fit ±3.25 h",
    )
    axis.scatter([0.0], [0.0], marker="+", s=32, label="Sun center")
    ca = frame["ca_position_arcsec"]
    axis.scatter([ca[0]], [ca[1]], s=34, label="Closest approach")
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("Fixed +X screen Y/X offset (arcsec)")
    axis.set_ylabel("Fixed +X screen Z/X offset (arcsec)")
    axis.set_title(
        "North Pole — 2012 Venus Transit\n"
        f"Fixed +X fitted track angle {frame['fit_angle_deg']:.6f}°"
    )
    axis.grid(True, linewidth=0.5, alpha=0.35)
    axis.legend()
    plt.show()


def main() -> None:
    tables = fetch_north_pole_vectors()
    result = derive_north_pole_frames(tables)
    display_results(tables, result)
    plot_fixed_x_screen(result)


if __name__ == "__main__":
    main()
