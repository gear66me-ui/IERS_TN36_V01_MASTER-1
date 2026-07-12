"""
IERS TN36 — Ecliptical Plane Analysis
Part J — North Pole ecliptic-versus-ICRF projection audit
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar

VERSION = "IERS-0001-J"
API = "https://ssd.jpl.nasa.gov/api/horizons.api"
ASEC = 206264.80624709636
DAY = 86400.0
RSUN_KM = 695700.0
RVENUS_KM = 6051.8
START = "2012-06-05 20:00"
STOP = "2012-06-06 07:30"
STEP = "1 min"


@dataclass(frozen=True)
class Site:
    label: str
    longitude_deg: float
    latitude_deg: float
    elevation_km: float

    @property
    def coord(self) -> str:
        return (
            f"{self.longitude_deg:.8f},"
            f"{self.latitude_deg:.8f},"
            f"{self.elevation_km:.6f}"
        )


NORTH_POLE = Site("North Pole", 0.0, 90.0, 0.0)
PLANES = {
    "ECLIPTIC": "Ecliptic reference plane",
    "FRAME": "ICRF equatorial frame",
}


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    magnitude = np.linalg.norm(vector)
    if not np.isfinite(magnitude) or magnitude == 0.0:
        raise ValueError("Invalid zero or non-finite vector.")
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
        "START_TIME": quoted(START),
        "STOP_TIME": quoted(STOP),
        "STEP_SIZE": quoted(STEP),
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
        parameters["SITE_COORD"] = quoted(site.coord)
    return parameters


def request_vectors(
    target: str,
    center: str,
    ref_plane: str,
    label: str,
    site: Site | None = None,
) -> dict[str, np.ndarray]:
    url = f"{API}?{urlencode(horizons_parameters(target, center, ref_plane, site))}"
    request = Request(
        url,
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
        raise RuntimeError(str(payload.get("error", f"{label}: no Horizons result.")))

    return parse_vectors(result, label)


def canonical(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def parse_vectors(text: str, label: str) -> dict[str, np.ndarray]:
    lines = text.splitlines()
    try:
        soe = next(i for i, line in enumerate(lines) if "$$SOE" in line)
        eoe = next(i for i, line in enumerate(lines) if i > soe and "$$EOE" in line)
    except StopIteration as exc:
        raise ValueError(f"{label}: missing Horizons data markers.") from exc

    header = None
    for index in range(soe - 1, max(-1, soe - 20), -1):
        row = next(csv.reader([lines[index]], skipinitialspace=True))
        names = [canonical(item) for item in row]
        if all(axis in names for axis in ("X", "Y", "Z")):
            header = row
            break
    if header is None:
        raise ValueError(f"{label}: vector header not found.")

    names = [canonical(item) for item in header]
    jd_index = next(
        names.index(name)
        for name in ("JDUT", "JDTDB", "JD")
        if name in names
    )
    x_index, y_index, z_index = (
        names.index("X"),
        names.index("Y"),
        names.index("Z"),
    )

    epochs: list[float] = []
    vectors: list[list[float]] = []
    for row in csv.reader(lines[soe + 1:eoe], skipinitialspace=True):
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

    epochs_array = np.asarray(epochs, dtype=float)
    vectors_array = np.asarray(vectors, dtype=float)
    if epochs_array.size < 3 or vectors_array.shape != (epochs_array.size, 3):
        raise ValueError(f"{label}: invalid vector table.")

    return {"jd": epochs_array, "xyz": vectors_array}


def fetch_plane(ref_plane: str) -> dict[str, dict[str, np.ndarray]]:
    data = {
        "GS": request_vectors(
            "10",
            "500@399",
            ref_plane,
            f"{ref_plane} geocenter Sun",
        ),
        "GV": request_vectors(
            "299",
            "500@399",
            ref_plane,
            f"{ref_plane} geocenter Venus",
        ),
        "NS": request_vectors(
            "10",
            "coord@399",
            ref_plane,
            f"{ref_plane} North Pole Sun",
            NORTH_POLE,
        ),
        "NV": request_vectors(
            "299",
            "coord@399",
            ref_plane,
            f"{ref_plane} North Pole Venus",
            NORTH_POLE,
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
        * ASEC
    )


def apparent_radii_arcsec(
    cache: dict[str, object],
    sun_key: str,
    venus_key: str,
    jd: float,
) -> tuple[float, float]:
    sun_radius = np.arctan2(
        RSUN_KM,
        np.linalg.norm(vector(cache, sun_key, jd)),
    ) * ASEC
    venus_radius = np.arctan2(
        RVENUS_KM,
        np.linalg.norm(vector(cache, venus_key, jd)),
    ) * ASEC
    return float(sun_radius), float(venus_radius)


def contact_equation(
    cache: dict[str, object],
    contact_type: str,
    jd: float,
) -> float:
    separation = angular_separation_arcsec(cache, "NS", "NV", jd)
    sun_radius, venus_radius = apparent_radii_arcsec(
        cache,
        "NS",
        "NV",
        jd,
    )
    tangent_radius = (
        sun_radius + venus_radius
        if contact_type == "external"
        else sun_radius - venus_radius
    )
    return separation - tangent_radius


def find_roots(
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


def recover_contacts(cache: dict[str, object]) -> dict[str, float]:
    external = find_roots(cache, "external")
    internal = find_roots(cache, "internal")
    if len(external) < 2 or len(internal) < 2:
        raise RuntimeError("North Pole C1-C4 contacts were not recovered.")
    return {
        "C1": external[0],
        "C2": internal[0],
        "C3": internal[-1],
        "C4": external[-1],
    }


def refine_geocentric_ca(cache: dict[str, object]) -> tuple[float, float]:
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
    return float(result.x), float(result.fun)


def solar_screen_basis(
    cache: dict[str, object],
    ca_jd: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = unit(vector(cache, "GS", ca_jd))
    frame_z = np.array([0.0, 0.0, 1.0])
    if np.linalg.norm(np.cross(frame_z, normal)) < 1e-12:
        frame_z = np.array([1.0, 0.0, 0.0])
    screen_x = unit(np.cross(frame_z, normal))
    screen_y = unit(np.cross(normal, screen_x))
    return normal, screen_x, screen_y


def solar_screen_point(
    cache: dict[str, object],
    jd: float,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    normal, screen_x, screen_y = basis
    observer_sun = vector(cache, "NS", jd)
    observer_venus = vector(cache, "NV", jd)

    scale = float(np.dot(observer_sun, normal)) / float(
        np.dot(observer_venus, normal)
    )
    displacement = scale * observer_venus - observer_sun
    earth_sun_distance = np.linalg.norm(vector(cache, "GS", jd))

    return np.array(
        [
            np.arctan2(
                np.dot(displacement, screen_x),
                earth_sun_distance,
            )
            * ASEC,
            np.arctan2(
                np.dot(displacement, screen_y),
                earth_sun_distance,
            )
            * ASEC,
        ],
        dtype=float,
    )


def fit_track(points: np.ndarray) -> dict[str, object]:
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    direction = vh[0]
    if direction[0] < 0.0:
        direction = -direction
    direction = unit(direction)
    normal = np.array([-direction[1], direction[0]])
    residuals = (points - centroid) @ normal

    return {
        "centroid": centroid,
        "direction": direction,
        "angle_deg": float(
            np.degrees(np.arctan2(direction[1], direction[0]))
        ),
        "rms_arcsec": float(np.sqrt(np.mean(residuals**2))),
    }


def evaluate_plane(
    ref_plane: str,
    contacts: dict[str, float],
    data: dict[str, dict[str, np.ndarray]] | None = None,
) -> dict[str, object]:
    if data is None:
        data = fetch_plane(ref_plane)
    cache = build_cache(data)
    ca_jd, ca_sep = refine_geocentric_ca(cache)
    basis = solar_screen_basis(cache, ca_jd)

    epochs = np.asarray(cache["jd"], dtype=float)
    mask = (epochs >= contacts["C1"]) & (epochs <= contacts["C4"])
    fit_epochs = epochs[mask]
    points = np.array(
        [solar_screen_point(cache, jd, basis) for jd in fit_epochs],
        dtype=float,
    )
    fit = fit_track(points)

    return {
        "ref_plane": ref_plane,
        "cache": cache,
        "ca_jd": ca_jd,
        "ca_sep_arcsec": ca_sep,
        "fit_epochs": fit_epochs,
        "points": points,
        "fit": fit,
    }


def utc_text(jd: float) -> str:
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    moment = epoch + timedelta(seconds=(jd - 2440587.5) * DAY)
    return moment.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def display_results(
    contacts: dict[str, float],
    results: dict[str, dict[str, object]],
) -> None:
    ecliptic_angle = float(results["ECLIPTIC"]["fit"]["angle_deg"])
    icrf_angle = float(results["FRAME"]["fit"]["angle_deg"])
    difference = icrf_angle - ecliptic_angle

    print("IERS TN36 - North Pole Projection Audit")
    print(f"Version : {VERSION}")
    print("Observer: North Pole, lon=0.000000°, lat=90.000000°, elev=0.000000 km")
    print("Method  : identical C1-C4 physical solar-screen fit in two JPL reference planes")

    print("\nNORTH POLE CONTACT TIMES UTC")
    print("C1                      C2                      C3                      C4")
    print(
        f"{utc_text(contacts['C1']):<24}"
        f"{utc_text(contacts['C2']):<24}"
        f"{utc_text(contacts['C3']):<24}"
        f"{utc_text(contacts['C4']):<24}"
    )

    print("\nREFERENCE-PLANE TRACK-ANGLE COMPARISON")
    print("REFERENCE PLANE                 TRACK ANGLE deg    RMS NORMAL arcsec    FIT ROWS")
    for key in ("ECLIPTIC", "FRAME"):
        result = results[key]
        print(
            f"{PLANES[key]:<31}"
            f"{float(result['fit']['angle_deg']):16.6f}"
            f"{float(result['fit']['rms_arcsec']):21.6f}"
            f"{len(result['fit_epochs']):12d}"
        )

    print("\nEXACT ANGULAR DISCREPANCY")
    print("ICRF ANGLE deg    ECLIPTIC ANGLE deg    ICRF-ECLIPTIC deg")
    print(f"{icrf_angle:14.6f}{ecliptic_angle:23.6f}{difference:23.6f}")


def plot_comparison(results: dict[str, dict[str, object]]) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 6.0))
    for key in ("ECLIPTIC", "FRAME"):
        result = results[key]
        points = result["points"]
        angle = float(result["fit"]["angle_deg"])
        axis.plot(
            points[:, 0],
            points[:, 1],
            linewidth=1.1,
            label=f"{PLANES[key]}: {angle:.6f}°",
        )

    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("Solar-screen X (arcsec)")
    axis.set_ylabel("Solar-screen Y (arcsec)")
    axis.set_title("North Pole 2012 Venus Track — Ecliptic vs ICRF")
    axis.grid(True, linewidth=0.5, alpha=0.35)
    axis.legend()
    fig.tight_layout()
    plt.show()


def main() -> None:
    ecliptic_data = fetch_plane("ECLIPTIC")
    ecliptic_cache = build_cache(ecliptic_data)
    contacts = recover_contacts(ecliptic_cache)

    results = {
        "ECLIPTIC": evaluate_plane("ECLIPTIC", contacts, ecliptic_data),
        "FRAME": evaluate_plane("FRAME", contacts),
    }
    display_results(contacts, results)
    plot_comparison(results)


if __name__ == "__main__":
    main()
