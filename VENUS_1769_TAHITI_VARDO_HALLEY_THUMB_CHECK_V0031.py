# V0031
# Audit reference: 1769 Venus transit Tahiti–Vardø JPL common-normal Halley thumb-check.
from __future__ import annotations

import math
import subprocess
import sys
import time
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
):
    ensure_package(_import_name, _pip_name)

import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar

VERSION = "V0031"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUTPUT_DIR = ROOT / "VENUS_1769_TAHITI_VARDO_HALLEY_THUMB_CHECK_V0031_OUTPUT"
OUTPUT_CSV = OUTPUT_DIR / "VENUS_1769_TAHITI_VARDO_HALLEY_THUMB_CHECK_V0031.csv"

ARCSEC_PER_RAD = 206_264.80624709636
AU_KM = 149_597_870.000000
EARTH_EQUATORIAL_RADIUS_KM = 6_378.140000
SUN_RADIUS_KM = 695_700.000000
VENUS_RADIUS_KM = 6_051.800000

START = "1769-06-03 12:00"
STOP = "1769-06-04 06:00"
STEP = "1m"
GEOCENTER = "500@399"

SITE_TAHITI = {
    "label": "Tahiti",
    "key": "TAHITI",
    "lat": -17.4956,
    "lon": -149.4939,
    "elevation": 0.0,
    "body": 399,
}
SITE_VARDO = {
    "label": "Vardø",
    "key": "VARDO",
    "lat": 70.3724,
    "lon": 31.1103,
    "elevation": 0.0,
    "body": 399,
}
SITES = (SITE_TAHITI, SITE_VARDO)
TARGETS = (("SUN", "10"), ("VENUS", "299"))
PREFIXES = (
    "GEOCENTER_SUN",
    "GEOCENTER_VENUS",
    "TAHITI_SUN",
    "TAHITI_VENUS",
    "VARDO_SUN",
    "VARDO_VENUS",
)


def norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    magnitude = norm(vector)
    if magnitude == 0.0:
        raise RuntimeError("Zero vector encountered.")
    return vector / magnitude


def site_location(site: dict[str, object]) -> dict[str, object]:
    return {
        "lon": float(site["lon"]),
        "lat": float(site["lat"]),
        "elevation": float(site["elevation"]),
        "body": int(site["body"]),
    }


def download_series(prefix: str, target_id: str, location) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            query = Horizons(
                id=target_id,
                location=location,
                epochs={"start": START, "stop": STOP, "step": STEP},
            )
            table = query.vectors(refplane="ecliptic", aberrations="geometric")
            frame = table.to_pandas()
            result = pd.DataFrame(
                {"JD_TDB": pd.to_numeric(frame["datetime_jd"], errors="coerce")}
            )
            for axis in "xyz":
                result[f"{prefix}_{axis.upper()}_KM"] = (
                    pd.to_numeric(frame[axis], errors="coerce") * AU_KM
                )
            result = (
                result.dropna()
                .drop_duplicates("JD_TDB")
                .sort_values("JD_TDB")
                .reset_index(drop=True)
            )
            if len(result) < 20:
                raise RuntimeError(f"Too few JPL rows for {prefix}.")
            return result
        except Exception as error:
            last_error = error
            if attempt < 3:
                time.sleep(2.0 * attempt)
    raise RuntimeError(f"JPL Horizons download failed for {prefix}: {last_error}")


def build_master() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for target_name, target_id in TARGETS:
        frames.append(download_series(f"GEOCENTER_{target_name}", target_id, GEOCENTER))
    for site in SITES:
        location = site_location(site)
        for target_name, target_id in TARGETS:
            frames.append(
                download_series(f"{site['key']}_{target_name}", target_id, location)
            )

    master = frames[0]
    for frame in frames[1:]:
        master = master.merge(frame, on="JD_TDB", how="inner", validate="one_to_one")
    if len(master) < 20:
        raise RuntimeError("Synchronized JPL master is incomplete.")
    return master.sort_values("JD_TDB").reset_index(drop=True)


def build_cache(master: pd.DataFrame) -> dict[str, object]:
    jds = master["JD_TDB"].to_numpy(dtype=float)
    cache: dict[str, object] = {"JD_TDB": jds}
    for prefix in PREFIXES:
        for axis in "XYZ":
            column = f"{prefix}_{axis}_KM"
            cache[column] = CubicSpline(
                jds,
                master[column].to_numpy(dtype=float),
                bc_type="natural",
            )
    return cache


def vector_at(cache: dict[str, object], prefix: str, jd_tdb: float) -> np.ndarray:
    return np.array(
        [float(cache[f"{prefix}_{axis}_KM"](jd_tdb)) for axis in "XYZ"],
        dtype=float,
    )


def angular_separation(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(unit(first), unit(second)), -1.0, 1.0))
    return math.acos(cosine)


def reference_epoch(cache: dict[str, object]) -> float:
    jds = np.asarray(cache["JD_TDB"], dtype=float)
    values = np.array(
        [
            angular_separation(
                vector_at(cache, "GEOCENTER_SUN", jd),
                vector_at(cache, "GEOCENTER_VENUS", jd),
            )
            for jd in jds
        ],
        dtype=float,
    )
    index = int(np.argmin(values))
    lower = float(jds[max(0, index - 5)])
    upper = float(jds[min(len(jds) - 1, index + 5)])
    result = minimize_scalar(
        lambda jd: angular_separation(
            vector_at(cache, "GEOCENTER_SUN", jd),
            vector_at(cache, "GEOCENTER_VENUS", jd),
        ),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1.0e-13, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError("Geocentric closest-approach solution failed.")
    return float(result.x)


def common_basis(geocentric_sun: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sun_hat = unit(geocentric_sun)
    ecliptic_north = np.array([0.0, 0.0, 1.0], dtype=float)
    xi = unit(np.cross(ecliptic_north, sun_hat))
    eta = unit(np.cross(sun_hat, xi))
    if float(np.dot(eta, ecliptic_north)) < 0.0:
        xi = -xi
        eta = -eta
    return sun_hat, xi, eta


def gnomonic(direction: np.ndarray, center: np.ndarray, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    direction_hat = unit(direction)
    denominator = float(np.dot(direction_hat, center))
    if denominator <= 0.0:
        raise RuntimeError("Direction is outside the common tangent hemisphere.")
    return np.array(
        [
            float(np.dot(direction_hat, xi)) / denominator,
            float(np.dot(direction_hat, eta)) / denominator,
        ],
        dtype=float,
    )


def relative_position_arcsec(
    cache: dict[str, object],
    site_key: str,
    jd_tdb: float,
    center: np.ndarray,
    xi: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    sun = vector_at(cache, f"{site_key}_SUN", jd_tdb)
    venus = vector_at(cache, f"{site_key}_VENUS", jd_tdb)
    return ARCSEC_PER_RAD * (
        gnomonic(venus, center, xi, eta) - gnomonic(sun, center, xi, eta)
    )


def contact_residual(cache: dict[str, object], site_key: str, jd_tdb: float) -> float:
    sun = vector_at(cache, f"{site_key}_SUN", jd_tdb)
    venus = vector_at(cache, f"{site_key}_VENUS", jd_tdb)
    separation = angular_separation(sun, venus)
    solar_radius = math.asin(SUN_RADIUS_KM / norm(sun))
    venus_radius = math.asin(VENUS_RADIUS_KM / norm(venus))
    return separation - (solar_radius + venus_radius)


def external_contacts(cache: dict[str, object], site_key: str) -> tuple[float, float]:
    jds = np.asarray(cache["JD_TDB"], dtype=float)
    values = np.array([contact_residual(cache, site_key, jd) for jd in jds], dtype=float)
    roots: list[float] = []
    for index in range(len(jds) - 1):
        left = float(values[index])
        right = float(values[index + 1])
        if left == 0.0:
            roots.append(float(jds[index]))
        elif left * right < 0.0:
            roots.append(
                float(
                    brentq(
                        lambda jd: contact_residual(cache, site_key, jd),
                        float(jds[index]),
                        float(jds[index + 1]),
                        xtol=1.0e-13,
                        rtol=1.0e-14,
                        maxiter=200,
                    )
                )
            )
    unique: list[float] = []
    for root in roots:
        if not unique or abs(root - unique[-1]) > 0.2 / 86400.0:
            unique.append(root)
    if len(unique) != 2:
        raise RuntimeError(f"Expected two external contacts for {site_key}; found {len(unique)}.")
    return unique[0], unique[1]


def fitted_direction(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    if direction[0] < 0.0:
        direction = -direction
    return unit(direction)


def utc_text(jd_tdb: float) -> str:
    return Time(jd_tdb, format="jd", scale="tdb").utc.datetime.strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )[:-3]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master = build_master()
    cache = build_cache(master)
    jd_reference = reference_epoch(cache)

    geocentric_sun = vector_at(cache, "GEOCENTER_SUN", jd_reference)
    geocentric_venus = vector_at(cache, "GEOCENTER_VENUS", jd_reference)
    venus_to_sun = geocentric_sun - geocentric_venus
    center, xi, eta = common_basis(geocentric_sun)

    contacts = {site["key"]: external_contacts(cache, str(site["key"])) for site in SITES}
    common_start = max(float(contacts["TAHITI"][0]), float(contacts["VARDO"][0]))
    common_stop = min(float(contacts["TAHITI"][1]), float(contacts["VARDO"][1]))
    minute_jds = np.asarray(cache["JD_TDB"], dtype=float)
    selected_jds = minute_jds[(minute_jds >= common_start) & (minute_jds <= common_stop)]
    if len(selected_jds) < 20:
        raise RuntimeError("Too few common in-transit samples.")

    tahiti_track = np.array(
        [relative_position_arcsec(cache, "TAHITI", jd, center, xi, eta) for jd in selected_jds]
    )
    vardo_track = np.array(
        [relative_position_arcsec(cache, "VARDO", jd, center, xi, eta) for jd in selected_jds]
    )
    direction_t = fitted_direction(tahiti_track)
    direction_v = fitted_direction(vardo_track)
    if float(np.dot(direction_t, direction_v)) < 0.0:
        direction_v = -direction_v
    common_direction = unit(direction_t + direction_v)
    common_normal = np.array([-common_direction[1], common_direction[0]], dtype=float)
    common_track_angle_deg = math.degrees(math.atan2(common_direction[1], common_direction[0]))

    tahiti_position = relative_position_arcsec(cache, "TAHITI", jd_reference, center, xi, eta)
    vardo_position = relative_position_arcsec(cache, "VARDO", jd_reference, center, xi, eta)
    aprime_bprime_arcsec = abs(float(np.dot(vardo_position - tahiti_position, common_normal)))

    topocentric_sun_t = vector_at(cache, "TAHITI_SUN", jd_reference)
    topocentric_sun_v = vector_at(cache, "VARDO_SUN", jd_reference)
    observer_t = geocentric_sun - topocentric_sun_t
    observer_v = geocentric_sun - topocentric_sun_v
    baseline = observer_v - observer_t
    baseline_plane = np.array(
        [float(np.dot(baseline, xi)), float(np.dot(baseline, eta))],
        dtype=float,
    )
    ab_direct_km = abs(float(np.dot(baseline_plane, common_normal)))

    earth_sun_km = norm(geocentric_sun)
    earth_venus_km = norm(geocentric_venus)
    venus_sun_km = norm(venus_to_sun)

    aprime_bprime_km = aprime_bprime_arcsec * earth_sun_km / ARCSEC_PER_RAD
    ab_direct_arcsec = ab_direct_km * ARCSEC_PER_RAD / earth_sun_km

    ratio_ev_vs = earth_venus_km / venus_sun_km
    ratio_es_vs = earth_sun_km / venus_sun_km
    ratio_ev_es = earth_venus_km / earth_sun_km
    ratio_geometry = ab_direct_arcsec / aprime_bprime_arcsec

    ab_halley_arcsec = aprime_bprime_arcsec * ratio_ev_vs
    ab_halley_km = ab_halley_arcsec * earth_sun_km / ARCSEC_PER_RAD
    residual_arcsec = ab_direct_arcsec - ab_halley_arcsec
    residual_km = ab_direct_km - ab_halley_km
    residual_percent = 100.0 * residual_arcsec / ab_direct_arcsec

    solar_parallax_arcsec = math.asin(
        EARTH_EQUATORIAL_RADIUS_KM / AU_KM
    ) * ARCSEC_PER_RAD

    rows = [
        ["Reference UTC", utc_text(jd_reference), "UTC"],
        ["Reference JD TDB", jd_reference, "day"],
        ["Common track angle", common_track_angle_deg, "deg"],
        ["Solar horizontal parallax", solar_parallax_arcsec, "arcsec"],
        ["A′B′ direct JPL", aprime_bprime_arcsec, "arcsec"],
        ["A′B′ direct JPL", aprime_bprime_km, "km"],
        ["AB direct JPL", ab_direct_arcsec, "arcsec"],
        ["AB direct JPL", ab_direct_km, "km"],
        ["Earth → Venus", earth_venus_km, "km"],
        ["Venus → Sun", venus_sun_km, "km"],
        ["Earth → Sun", earth_sun_km, "km"],
        ["EV / VS", ratio_ev_vs, "ratio"],
        ["ES / VS", ratio_es_vs, "ratio"],
        ["EV / ES", ratio_ev_es, "ratio"],
        ["AB / A′B′", ratio_geometry, "ratio"],
        ["AB from Halley", ab_halley_arcsec, "arcsec"],
        ["AB from Halley", ab_halley_km, "km"],
        ["Residual direct − Halley", residual_arcsec, "arcsec"],
        ["Residual direct − Halley", residual_km, "km"],
        ["Residual direct − Halley", residual_percent, "%"],
    ]
    result = pd.DataFrame(rows, columns=["Quantity", "Value", "Unit"])
    result.to_csv(OUTPUT_CSV, index=False)

    def format_value(value: object) -> str:
        if isinstance(value, str):
            return value
        return f"{float(value):,.12f}"

    print(
        result.to_string(
            index=False,
            formatters={"Value": format_value},
        )
    )
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0031
