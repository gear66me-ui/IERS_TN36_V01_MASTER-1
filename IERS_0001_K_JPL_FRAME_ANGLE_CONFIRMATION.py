"""
IERS TN36 — Ecliptical Plane Analysis
Part K — JPL ecliptic-versus-ICRF angle confirmation
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar

VERSION = "IERS-0001-K"
API = "https://ssd.jpl.nasa.gov/api/horizons.api"
ARCSEC_PER_RAD = 206264.80624709636
SOLAR_RADIUS_KM = 695700.0
VENUS_RADIUS_KM = 6051.8

START_UTC = "2012-06-05 20:00"
STOP_UTC = "2012-06-06 07:30"
STEP_SIZE = "1 min"

REFERENCE_ECLIPTIC_DEG = 8.486205
REFERENCE_ICRF_DEG = 14.646855
REFERENCE_DELTA_DEG = 6.160650
TOLERANCE_DEG = 0.000001


@dataclass(frozen=True)
class Site:
    longitude_deg: float
    latitude_deg: float
    elevation_km: float

    @property
    def coordinate_text(self) -> str:
        return (
            f"{self.longitude_deg:.8f},"
            f"{self.latitude_deg:.8f},"
            f"{self.elevation_km:.6f}"
        )


NORTH_POLE = Site(0.0, 90.0, 0.0)


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    magnitude = np.linalg.norm(vector)
    if not np.isfinite(magnitude) or magnitude == 0.0:
        raise ValueError("Cannot normalize a zero or non-finite vector.")
    return vector / magnitude


def quoted(value: str) -> str:
    return f"'{value}'"


def horizons_parameters(
    target: str,
    center: str,
    ref_plane: str,
    site: Site | None = None,
) -> dict[str, str]:
    parameters = {
        "format": "json",
        "COMMAND": quoted(target),
        "OBJ_DATA": quoted("NO"),
        "MAKE_EPHEM": quoted("YES"),
        "EPHEM_TYPE": quoted("VECTORS"),
        "CENTER": quoted(center),
        "START_TIME": quoted(START_UTC),
        "STOP_TIME": quoted(STOP_UTC),
        "STEP_SIZE": quoted(STEP_SIZE),
        "TIME_TYPE": quoted("UT"),
        "TIME_DIGITS": quoted("SECONDS"),
        "REF_PLANE": quoted(ref_plane),
        "REF_SYSTEM": quoted("ICRF"),
        "OUT_UNITS": quoted("KM-S"),
        "VEC_TABLE": quoted("2"),
        "VEC_CORR": quoted("NONE"),
        "CSV_FORMAT": quoted("YES"),
        "VEC_LABELS": quoted("YES"),
    }
    if site is not None:
        parameters["COORD_TYPE"] = quoted("GEODETIC")
        parameters["SITE_COORD"] = quoted(site.coordinate_text)
    return parameters


def canonical_header(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def parse_vectors(result_text: str, label: str) -> dict[str, np.ndarray]:
    lines = result_text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if "$$SOE" in line)
        stop = next(i for i, line in enumerate(lines) if i > start and "$$EOE" in line)
    except StopIteration as exc:
        raise ValueError(f"{label}: missing Horizons data markers.") from exc

    header = None
    for index in range(start - 1, max(-1, start - 20), -1):
        row = next(csv.reader([lines[index]], skipinitialspace=True))
        names = [canonical_header(item) for item in row]
        if all(axis in names for axis in ("X", "Y", "Z")):
            header = row
            break
    if header is None:
        raise ValueError(f"{label}: vector header not found.")

    names = [canonical_header(item) for item in header]
    jd_index = next(
        names.index(name)
        for name in ("JDUT", "JDTDB", "JD")
        if name in names
    )
    x_index = names.index("X")
    y_index = names.index("Y")
    z_index = names.index("Z")

    epochs: list[float] = []
    vectors: list[list[float]] = []
    for row in csv.reader(lines[start + 1:stop], skipinitialspace=True):
        if not row or max(jd_index, x_index, y_index, z_index) >= len(row):
            continue
        try:
            epochs.append(float(row[jd_index]))
            vectors.append(
                [
                    float(row[x_index]),
                    float(row[y_index]),
                    float(row[z_index]),
                ]
            )
        except ValueError:
            continue

    jd = np.asarray(epochs, dtype=float)
    xyz = np.asarray(vectors, dtype=float)
    if jd.size < 3 or xyz.shape != (jd.size, 3):
        raise ValueError(f"{label}: invalid Horizons vector table.")
    if np.any(np.diff(jd) <= 0.0):
        raise ValueError(f"{label}: epochs are not strictly increasing.")
    return {"jd": jd, "xyz": xyz}


def request_vectors(
    target: str,
    center: str,
    ref_plane: str,
    label: str,
    site: Site | None = None,
) -> dict[str, np.ndarray]:
    query = urlencode(horizons_parameters(target, center, ref_plane, site))
    request = Request(
        f"{API}?{query}",
        headers={
            "User-Agent": "IERS-TN36-Ecliptical-Plane-Analysis/1.0",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=90) as response:
        payload = json.loads(response.read().decode("utf-8"))

    source = str(payload.get("signature", {}).get("source", ""))
    if "NASA/JPL" not in source:
        raise RuntimeError(f"{label}: unexpected Horizons source {source!r}.")

    result = payload.get("result")
    if not isinstance(result, str) or not result.strip():
        raise RuntimeError(str(payload.get("error", f"{label}: no result.")))
    return parse_vectors(result, label)


def fetch_plane(ref_plane: str) -> dict[str, dict[str, np.ndarray]]:
    data = {
        "GS": request_vectors("10", "500@399", ref_plane, f"{ref_plane} geocenter Sun"),
        "GV": request_vectors("299", "500@399", ref_plane, f"{ref_plane} geocenter Venus"),
        "NS": request_vectors(
            "10", "coord@399", ref_plane, f"{ref_plane} North Pole Sun", NORTH_POLE
        ),
        "NV": request_vectors(
            "299", "coord@399", ref_plane, f"{ref_plane} North Pole Venus", NORTH_POLE
        ),
    }

    reference_jd = data["GS"]["jd"]
    for key, table in data.items():
        if table["jd"].shape != reference_jd.shape:
            raise ValueError(f"{ref_plane} {key}: epoch-count mismatch.")
        if not np.allclose(table["jd"], reference_jd, atol=1e-12):
            raise ValueError(f"{ref_plane} {key}: unsynchronized epochs.")
    return data


def build_cache(data: dict[str, dict[str, np.ndarray]]) -> dict[str, object]:
    cache: dict[str, object] = {"jd": data["GS"]["jd"]}
    for key, table in data.items():
        cache[key] = CubicSpline(
            table["jd"],
            table["xyz"],
            axis=0,
            bc_type="natural",
        )
    return cache


def vector(cache: dict[str, object], key: str, jd: float) -> np.ndarray:
    return np.asarray(cache[key](jd), dtype=float)


def angular_separation_arcsec(
    cache: dict[str, object],
    sun_key: str,
    venus_key: str,
    jd: float,
) -> float:
    sun = unit(vector(cache, sun_key, jd))
    venus = unit(vector(cache, venus_key, jd))
    return float(
        np.arctan2(
            np.linalg.norm(np.cross(sun, venus)),
            np.dot(sun, venus),
        )
        * ARCSEC_PER_RAD
    )


def apparent_radii_arcsec(
    cache: dict[str, object],
    jd: float,
) -> tuple[float, float]:
    sun_radius = np.arctan2(
        SOLAR_RADIUS_KM,
        np.linalg.norm(vector(cache, "NS", jd)),
    ) * ARCSEC_PER_RAD
    venus_radius = np.arctan2(
        VENUS_RADIUS_KM,
        np.linalg.norm(vector(cache, "NV", jd)),
    ) * ARCSEC_PER_RAD
    return float(sun_radius), float(venus_radius)


def contact_equation(
    cache: dict[str, object],
    contact_type: str,
    jd: float,
) -> float:
    separation = angular_separation_arcsec(cache, "NS", "NV", jd)
    sun_radius, venus_radius = apparent_radii_arcsec(cache, jd)
    tangent_radius = (
        sun_radius + venus_radius
        if contact_type == "external"
        else sun_radius - venus_radius
    )
    return separation - tangent_radius


def find_contact_roots(
    cache: dict[str, object],
    contact_type: str,
) -> list[float]:
    epochs = np.asarray(cache["jd"], dtype=float)
    values = np.array(
        [contact_equation(cache, contact_type, jd) for jd in epochs],
        dtype=float,
    )
    roots: list[float] = []

    for index in range(epochs.size - 1):
        left_value = values[index]
        right_value = values[index + 1]
        if left_value == 0.0:
            roots.append(float(epochs[index]))
        elif left_value * right_value < 0.0:
            roots.append(
                float(
                    brentq(
                        lambda epoch: contact_equation(
                            cache,
                            contact_type,
                            epoch,
                        ),
                        epochs[index],
                        epochs[index + 1],
                        xtol=1e-13,
                        rtol=1e-13,
                    )
                )
            )
    return roots


def north_pole_contact_interval(cache: dict[str, object]) -> tuple[float, float]:
    external = find_contact_roots(cache, "external")
    internal = find_contact_roots(cache, "internal")
    if len(external) < 2 or len(internal) < 2:
        raise RuntimeError("North Pole C1-C4 contacts were not recovered.")
    return external[0], external[-1]


def geocentric_closest_approach(cache: dict[str, object]) -> float:
    epochs = np.asarray(cache["jd"], dtype=float)
    sampled = np.array(
        [angular_separation_arcsec(cache, "GS", "GV", jd) for jd in epochs],
        dtype=float,
    )
    index = int(np.argmin(sampled))
    left = epochs[max(0, index - 3)]
    right = epochs[min(epochs.size - 1, index + 3)]
    result = minimize_scalar(
        lambda epoch: angular_separation_arcsec(cache, "GS", "GV", epoch),
        bounds=(left, right),
        method="bounded",
        options={"xatol": 1e-12},
    )
    if not result.success:
        raise RuntimeError("Geocentric closest-approach fit failed.")
    return float(result.x)


def solar_screen_basis(
    cache: dict[str, object],
    ca_jd: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    screen_normal = unit(vector(cache, "GS", ca_jd))
    reference_z = np.array([0.0, 0.0, 1.0])
    if np.linalg.norm(np.cross(reference_z, screen_normal)) < 1e-12:
        reference_z = np.array([1.0, 0.0, 0.0])
    screen_x = unit(np.cross(reference_z, screen_normal))
    screen_y = unit(np.cross(screen_normal, screen_x))
    return screen_normal, screen_x, screen_y


def solar_screen_point(
    cache: dict[str, object],
    jd: float,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    screen_normal, screen_x, screen_y = basis
    observer_sun = vector(cache, "NS", jd)
    observer_venus = vector(cache, "NV", jd)

    ray_scale = float(np.dot(observer_sun, screen_normal)) / float(
        np.dot(observer_venus, screen_normal)
    )
    displacement = ray_scale * observer_venus - observer_sun
    earth_sun_distance = np.linalg.norm(vector(cache, "GS", jd))

    return np.array(
        [
            np.arctan2(np.dot(displacement, screen_x), earth_sun_distance)
            * ARCSEC_PER_RAD,
            np.arctan2(np.dot(displacement, screen_y), earth_sun_distance)
            * ARCSEC_PER_RAD,
        ],
        dtype=float,
    )


def pca_track_angle_deg(points: np.ndarray) -> tuple[float, float]:
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    direction = vh[0]
    if direction[0] < 0.0:
        direction = -direction
    direction = unit(direction)
    normal = np.array([-direction[1], direction[0]])
    residuals = (points - centroid) @ normal
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    rms = float(np.sqrt(np.mean(residuals**2)))
    return angle, rms


def derive_angle(
    ref_plane: str,
    c1_jd: float,
    c4_jd: float,
    data: dict[str, dict[str, np.ndarray]] | None = None,
) -> dict[str, float | int]:
    if data is None:
        data = fetch_plane(ref_plane)
    cache = build_cache(data)
    ca_jd = geocentric_closest_approach(cache)
    basis = solar_screen_basis(cache, ca_jd)

    epochs = np.asarray(cache["jd"], dtype=float)
    fit_epochs = epochs[(epochs >= c1_jd) & (epochs <= c4_jd)]
    points = np.array(
        [solar_screen_point(cache, jd, basis) for jd in fit_epochs],
        dtype=float,
    )
    angle, rms = pca_track_angle_deg(points)
    return {
        "angle_deg": angle,
        "rms_arcsec": rms,
        "rows": int(fit_epochs.size),
    }


def pass_text(residual: float) -> str:
    return "PASS" if abs(residual) <= TOLERANCE_DEG else "CHECK"


def main() -> None:
    print("IERS TN36 - JPL Frame-Angle Confirmation")
    print(f"Version : {VERSION}")
    print("Observer: North Pole")
    print("Method  : direct JPL vectors; identical C1-C4 physical solar-screen PCA")

    ecliptic_data = fetch_plane("ECLIPTIC")
    ecliptic_cache = build_cache(ecliptic_data)
    c1_jd, c4_jd = north_pole_contact_interval(ecliptic_cache)

    ecliptic = derive_angle("ECLIPTIC", c1_jd, c4_jd, ecliptic_data)
    icrf = derive_angle("FRAME", c1_jd, c4_jd)
    delta = float(icrf["angle_deg"]) - float(ecliptic["angle_deg"])

    ecliptic_residual = float(ecliptic["angle_deg"]) - REFERENCE_ECLIPTIC_DEG
    icrf_residual = float(icrf["angle_deg"]) - REFERENCE_ICRF_DEG
    delta_residual = delta - REFERENCE_DELTA_DEG

    print("\nDERIVED JPL TRACK ANGLES")
    print("REFERENCE FRAME       ANGLE deg      RMS arcsec    FIT ROWS")
    print(
        f"{'ECLIPTIC':<21}"
        f"{float(ecliptic['angle_deg']):12.6f}"
        f"{float(ecliptic['rms_arcsec']):16.6f}"
        f"{int(ecliptic['rows']):12d}"
    )
    print(
        f"{'ICRF / FRAME':<21}"
        f"{float(icrf['angle_deg']):12.6f}"
        f"{float(icrf['rms_arcsec']):16.6f}"
        f"{int(icrf['rows']):12d}"
    )

    print("\nFRAME DIFFERENCE")
    print("ICRF deg       ECLIPTIC deg       ICRF-ECLIPTIC deg")
    print(
        f"{float(icrf['angle_deg']):12.6f}"
        f"{float(ecliptic['angle_deg']):20.6f}"
        f"{delta:25.6f}"
    )

    print("\nREFERENCE CONFIRMATION")
    print("QUANTITY          DERIVED deg    REFERENCE deg    RESIDUAL deg    STATUS")
    checks = (
        ("Ecliptic", float(ecliptic["angle_deg"]), REFERENCE_ECLIPTIC_DEG, ecliptic_residual),
        ("ICRF", float(icrf["angle_deg"]), REFERENCE_ICRF_DEG, icrf_residual),
        ("Difference", delta, REFERENCE_DELTA_DEG, delta_residual),
    )
    for name, derived, reference, residual in checks:
        print(
            f"{name:<16}"
            f"{derived:12.6f}"
            f"{reference:17.6f}"
            f"{residual:16.6f}"
            f"{pass_text(residual):>10}"
        )

    if any(abs(item[3]) > TOLERANCE_DEG for item in checks):
        raise RuntimeError("One or more six-decimal reference checks failed.")


if __name__ == "__main__":
    main()
