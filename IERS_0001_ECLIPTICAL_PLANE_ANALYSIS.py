"""
IERS TN36 — Ecliptical Plane Analysis
Parts F+G — JPL acquisition and Sun-relative transit track
Bucaramanga observer, 2012 Venus transit
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

VERSION = "IERS-0001-G"
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


BUCARAMANGA = Observer(
    name="Bucaramanga, Colombia",
    longitude_deg=-73.11980000,
    latitude_deg=7.12539000,
    elevation_km=0.959000,
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


def build_horizons_url(target_id: str, observer: Observer) -> str:
    parameters = build_horizons_parameters(target_id, observer)
    return f"{HORIZONS_API}?{urlencode(parameters)}"


def request_horizons_text(target_id: str, observer: Observer) -> tuple[str, str]:
    request = Request(
        build_horizons_url(target_id, observer),
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
    source = str(signature.get("source", ""))
    api_version = str(signature.get("version", ""))
    result = payload.get("result")
    if "NASA/JPL" not in source:
        raise RuntimeError(f"Unexpected Horizons API source: {source!r}")
    if not isinstance(result, str) or not result.strip():
        raise RuntimeError(str(payload.get("error", "No Horizons result text returned.")))
    return result, api_version


def canonical_header(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def find_vector_header(lines: list[str], soe_index: int) -> list[str]:
    for index in range(soe_index - 1, max(-1, soe_index - 20), -1):
        row = next(csv.reader([lines[index]], skipinitialspace=True))
        names = [canonical_header(item) for item in row]
        if all(axis in names for axis in ("X", "Y", "Z", "VX", "VY", "VZ")):
            return row
    raise ValueError("Horizons vector header was not found before $$SOE.")


def parse_horizons_vectors(result_text: str, label: str) -> dict[str, np.ndarray]:
    lines = result_text.splitlines()
    try:
        soe_index = next(i for i, line in enumerate(lines) if "$$SOE" in line)
        eoe_index = next(
            i for i, line in enumerate(lines) if i > soe_index and "$$EOE" in line
        )
    except StopIteration as exc:
        raise ValueError(f"{label}: missing $$SOE/$$EOE markers.") from exc

    header = find_vector_header(lines, soe_index)
    names = [canonical_header(item) for item in header]
    jd_index = next(
        (names.index(name) for name in ("JDUT", "JDTDB", "JD") if name in names),
        None,
    )
    if jd_index is None:
        raise ValueError(f"{label}: no Julian-date column found.")

    indices = {key: names.index(key.upper()) for key in ("x", "y", "z", "vx", "vy", "vz")}
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
        raise ValueError(f"{label}: fewer than three valid vector epochs.")
    if positions.shape != (epochs.size, 3):
        raise ValueError(f"{label}: invalid position shape {positions.shape}.")
    if velocities.shape != (epochs.size, 3):
        raise ValueError(f"{label}: invalid velocity shape {velocities.shape}.")
    if not np.all(np.isfinite(epochs)):
        raise ValueError(f"{label}: non-finite epochs detected.")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise ValueError(f"{label}: non-finite state-vector values detected.")
    if np.any(np.diff(epochs) <= 0.0):
        raise ValueError(f"{label}: epochs are not strictly increasing.")
    if np.any(np.linalg.norm(positions, axis=1) == 0.0):
        raise ValueError(f"{label}: zero-length position vector detected.")


def fetch_target_vectors(label: str, target_id: str) -> dict[str, np.ndarray]:
    result_text, api_version = request_horizons_text(target_id, BUCARAMANGA)
    table = parse_horizons_vectors(result_text, label)
    table["api_version"] = np.asarray([api_version])
    return table


def fetch_bucaramanga_vectors() -> dict[str, dict[str, np.ndarray]]:
    tables = {
        label: fetch_target_vectors(label, target_id)
        for label, target_id in TARGETS.items()
    }
    sun_jd = tables["Sun"]["jd_ut"]
    venus_jd = tables["Venus"]["jd_ut"]
    if sun_jd.shape != venus_jd.shape or not np.allclose(sun_jd, venus_jd, atol=1e-12):
        raise ValueError("Sun and Venus Horizons epochs are not synchronized.")
    return tables


def vector_cadence_seconds(epochs: np.ndarray) -> float:
    return float(np.median(np.diff(epochs)) * SECONDS_PER_DAY)


def display_acquisition_summary(tables: dict[str, dict[str, np.ndarray]]) -> None:
    print("\nJPL HORIZONS VECTOR ACQUISITION")
    print("BODY      ROWS     CADENCE s          JD UT START            JD UT END       R MIN km       R MAX km")
    for label in ("Sun", "Venus"):
        table = tables[label]
        epochs = table["jd_ut"]
        radii = np.linalg.norm(table["position_km"], axis=1)
        print(
            f"{label:<8} {epochs.size:6d}  {vector_cadence_seconds(epochs):12.6f}  "
            f"{epochs[0]:19.10f}  {epochs[-1]:19.10f}  "
            f"{radii.min():13.3f}  {radii.max():13.3f}"
        )


def state_interpolator(table: dict[str, np.ndarray]) -> tuple[float, CubicSpline]:
    jd0 = float(table["jd_ut"][0])
    seconds = (table["jd_ut"] - jd0) * SECONDS_PER_DAY
    spline = CubicSpline(seconds, table["position_km"], axis=0)
    return jd0, spline


def angular_separation_rad(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    unit_a = normalize(vector_a)
    unit_b = normalize(vector_b)
    return float(
        np.arctan2(np.linalg.norm(np.cross(unit_a, unit_b)), np.dot(unit_a, unit_b))
    )


def refine_closest_approach(
    sun_table: dict[str, np.ndarray],
    venus_table: dict[str, np.ndarray],
) -> dict[str, float | np.ndarray | CubicSpline]:
    epochs = sun_table["jd_ut"]
    sampled_separation = np.array(
        [
            angular_separation_rad(sun_vector, venus_vector)
            for sun_vector, venus_vector in zip(
                sun_table["position_km"], venus_table["position_km"]
            )
        ]
    )
    sampled_index = int(np.argmin(sampled_separation))
    left_index = max(0, sampled_index - 3)
    right_index = min(epochs.size - 1, sampled_index + 3)

    sun_jd0, sun_spline = state_interpolator(sun_table)
    venus_jd0, venus_spline = state_interpolator(venus_table)
    if abs(sun_jd0 - venus_jd0) > 1e-12:
        raise ValueError("Sun and Venus interpolation origins do not match.")

    seconds = (epochs - sun_jd0) * SECONDS_PER_DAY

    def objective(epoch_seconds: float) -> float:
        separation = angular_separation_rad(
            sun_spline(epoch_seconds), venus_spline(epoch_seconds)
        )
        return separation * separation

    result = minimize_scalar(
        objective,
        bounds=(float(seconds[left_index]), float(seconds[right_index])),
        method="bounded",
        options={"xatol": 1e-7},
    )
    if not result.success:
        raise RuntimeError(f"Closest-approach optimization failed: {result.message}")

    ca_seconds = float(result.x)
    ca_jd = sun_jd0 + ca_seconds / SECONDS_PER_DAY
    ca_separation = np.sqrt(float(result.fun))
    return {
        "jd0": sun_jd0,
        "ca_seconds": ca_seconds,
        "ca_jd_ut": ca_jd,
        "ca_separation_arcsec": ca_separation * ARCSEC_PER_RAD,
        "sampled_index": sampled_index,
        "sampled_separation_arcsec": sampled_separation * ARCSEC_PER_RAD,
        "sun_spline": sun_spline,
        "venus_spline": venus_spline,
    }


def tangent_basis(reference_direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sightline = normalize(reference_direction)
    north_pole = np.array([0.0, 0.0, 1.0])
    east = np.cross(north_pole, sightline)
    if np.linalg.norm(east) < 1e-12:
        north_pole = np.array([0.0, 1.0, 0.0])
        east = np.cross(north_pole, sightline)
    east = normalize(east)
    north = normalize(np.cross(sightline, east))
    return east, north, sightline


def gnomonic_coordinates(
    directions: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    east, north, sightline = basis
    directions = np.asarray(directions, dtype=float)
    denominator = directions @ sightline
    if np.any(denominator <= 0.0):
        raise ValueError("A direction lies outside the forward tangent hemisphere.")
    return np.column_stack(
        ((directions @ east) / denominator, (directions @ north) / denominator)
    )


def signed_line_angle_deg(direction: np.ndarray) -> float:
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    while angle > 90.0:
        angle -= 180.0
    while angle <= -90.0:
        angle += 180.0
    return angle


def orthogonal_track_fit(points_arcsec: np.ndarray) -> dict[str, np.ndarray | float]:
    centroid = np.mean(points_arcsec, axis=0)
    _, singular_values, vh = np.linalg.svd(points_arcsec - centroid, full_matrices=False)
    direction = vh[0]
    if np.dot(direction, points_arcsec[-1] - points_arcsec[0]) < 0.0:
        direction = -direction
    normal = np.array([-direction[1], direction[0]])
    residual = (points_arcsec - centroid) @ normal
    return {
        "centroid": centroid,
        "direction": direction,
        "normal": normal,
        "angle_deg": signed_line_angle_deg(direction),
        "rms_normal_arcsec": float(np.sqrt(np.mean(residual**2))),
        "singular_values": singular_values,
    }


def derive_sun_relative_track(
    tables: dict[str, dict[str, np.ndarray]],
) -> dict[str, object]:
    sun_table = tables["Sun"]
    venus_table = tables["Venus"]
    closest = refine_closest_approach(sun_table, venus_table)

    epochs = sun_table["jd_ut"]
    epoch_seconds = (epochs - float(closest["jd0"])) * SECONDS_PER_DAY
    sun_vectors = sun_table["position_km"]
    venus_vectors = venus_table["position_km"]
    sun_directions = normalize_rows(sun_vectors)
    venus_directions = normalize_rows(venus_vectors)

    ca_sun_direction = normalize(
        closest["sun_spline"](float(closest["ca_seconds"]))
    )
    basis = tangent_basis(ca_sun_direction)
    sun_xy = gnomonic_coordinates(sun_directions, basis)
    venus_xy = gnomonic_coordinates(venus_directions, basis)
    relative_arcsec = (venus_xy - sun_xy) * ARCSEC_PER_RAD

    fit_half_window_seconds = FIT_HALF_WINDOW_HOURS * 3600.0
    fit_mask = np.abs(epoch_seconds - float(closest["ca_seconds"])) <= fit_half_window_seconds
    fit = orthogonal_track_fit(relative_arcsec[fit_mask])

    derivative_step = 30.0
    before = float(closest["ca_seconds"]) - derivative_step
    after = float(closest["ca_seconds"]) + derivative_step
    sun_local = normalize_rows(
        np.vstack((closest["sun_spline"](before), closest["sun_spline"](after)))
    )
    venus_local = normalize_rows(
        np.vstack((closest["venus_spline"](before), closest["venus_spline"](after)))
    )
    local_relative = (
        gnomonic_coordinates(venus_local, basis)
        - gnomonic_coordinates(sun_local, basis)
    ) * ARCSEC_PER_RAD
    instantaneous_direction = normalize(local_relative[1] - local_relative[0])
    instantaneous_angle = signed_line_angle_deg(instantaneous_direction)

    ca_sun_xy = gnomonic_coordinates(
        normalize_rows(np.atleast_2d(closest["sun_spline"](float(closest["ca_seconds"])))),
        basis,
    )[0]
    ca_venus_xy = gnomonic_coordinates(
        normalize_rows(np.atleast_2d(closest["venus_spline"](float(closest["ca_seconds"])))),
        basis,
    )[0]
    ca_relative_arcsec = (ca_venus_xy - ca_sun_xy) * ARCSEC_PER_RAD

    return {
        "jd_ut": epochs,
        "relative_arcsec": relative_arcsec,
        "fit_mask": fit_mask,
        "fit": fit,
        "instantaneous_angle_deg": instantaneous_angle,
        "ca_relative_arcsec": ca_relative_arcsec,
        **closest,
    }


def jd_to_utc_text(jd_ut: float) -> str:
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    moment = epoch + timedelta(seconds=(jd_ut - 2440587.5) * SECONDS_PER_DAY)
    return moment.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def display_track_summary(track: dict[str, object]) -> None:
    fit = track["fit"]
    ca_position = track["ca_relative_arcsec"]
    print("\nBUCARAMANGA SUN-RELATIVE VENUS TRACK")
    print("CA UTC                      CA JD UT          X CA arcsec     Y CA arcsec     SEP CA arcsec")
    print(
        f"{jd_to_utc_text(float(track['ca_jd_ut'])):<27} "
        f"{float(track['ca_jd_ut']):16.10f}  "
        f"{ca_position[0]:13.6f}  {ca_position[1]:13.6f}  "
        f"{float(track['ca_separation_arcsec']):13.6f}"
    )
    print("\nANGLE DEFINITION                         ANGLE deg       RMS NORMAL arcsec")
    print(
        f"{'Instantaneous tangent at CA':<40} "
        f"{float(track['instantaneous_angle_deg']):11.6f}  {'—':>22}"
    )
    print(
        f"{'Orthogonal fit, ±3.25 h about CA':<40} "
        f"{float(fit['angle_deg']):11.6f}  "
        f"{float(fit['rms_normal_arcsec']):22.6f}"
    )


def plot_sun_relative_track(track: dict[str, object]) -> None:
    relative = track["relative_arcsec"]
    fit_mask = track["fit_mask"]
    ca_position = track["ca_relative_arcsec"]
    fit = track["fit"]

    fitted_points = relative[fit_mask]
    along = (fitted_points - fit["centroid"]) @ fit["direction"]
    endpoints = fit["centroid"] + np.array(
        [along.min(), along.max()]
    )[:, None] * fit["direction"]

    fig, axis = plt.subplots(figsize=(8.0, 6.0))
    axis.plot(relative[:, 0], relative[:, 1], linewidth=1.1, label="JPL Venus track")
    axis.plot(
        endpoints[:, 0],
        endpoints[:, 1],
        "--",
        linewidth=1.0,
        label="Orthogonal fit near transit",
    )
    axis.scatter([0.0], [0.0], s=28, marker="+", label="Sun center")
    axis.scatter(
        [ca_position[0]],
        [ca_position[1]],
        s=34,
        label="Refined closest approach",
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("ICRF tangent-plane east (arcsec)")
    axis.set_ylabel("ICRF tangent-plane north (arcsec)")
    axis.set_title(
        "Bucaramanga 2012 Venus Transit\n"
        f"Instantaneous track angle {float(track['instantaneous_angle_deg']):.6f}°"
    )
    axis.grid(True, linewidth=0.5, alpha=0.35)
    axis.legend()
    plt.show()


def main() -> None:
    print("IERS TN36 - Ecliptical Plane Analysis")
    print(f"Version : {VERSION}")
    print(f"Observer: {BUCARAMANGA.name}")
    print("Transit : 2012-06-06")
    tables = fetch_bucaramanga_vectors()
    display_acquisition_summary(tables)
    track = derive_sun_relative_track(tables)
    display_track_summary(track)
    plot_sun_relative_track(track)


if __name__ == "__main__":
    main()
