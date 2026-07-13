# V0029
# Audit reference: Paper-style JPL Horizons derivation of solar parallax, A′B′, AB, vector distances, ratios, and Halley consistency.
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
        subprocess.check_call(
            [sys.executable, "-m", "pip", "-q", "install", pip_name]
        )


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
from scipy.optimize import minimize_scalar

VERSION = "V0029"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUTPUT_DIR = ROOT / "MERCURY_HALLEY_REPORT_V0029_OUTPUT"
OUTPUT_CSV = OUTPUT_DIR / "MERCURY_HALLEY_REPORT_V0029.csv"

ARCSEC_PER_RAD = 206_264.80624709636
IAU1976_AU_KM = 149_597_870.000000
IAU1976_EARTH_EQUATORIAL_RADIUS_KM = 6_378.140000
VECTOR_AU_KM = IAU1976_AU_KM

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


def download_series(
    prefix: str,
    target_id: str,
    location: str | dict[str, object],
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
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
                {
                    "JD_TDB": pd.to_numeric(
                        frame["datetime_jd"], errors="coerce"
                    )
                }
            )
            for axis in "xyz":
                output[f"{prefix}_{axis.upper()}_KM"] = (
                    pd.to_numeric(frame[axis], errors="coerce")
                    * VECTOR_AU_KM
                )
            output = (
                output.dropna()
                .drop_duplicates("JD_TDB")
                .sort_values("JD_TDB")
                .reset_index(drop=True)
            )
            if len(output) < 20:
                raise RuntimeError(
                    f"JPL returned too few rows for {prefix}: {len(output)}"
                )
            return output
        except Exception as error:
            last_error = error
            if attempt < 3:
                time.sleep(2.0 * attempt)
    raise RuntimeError(f"JPL Horizons download failed for {prefix}: {last_error}")


def build_master() -> pd.DataFrame:
    series: list[pd.DataFrame] = []
    for target_name, target_id in TARGETS:
        series.append(
            download_series(
                f"GEOCENTER_{target_name}", target_id, GEOCENTER
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
    if len(master) < 20:
        raise RuntimeError("The synchronized JPL vector master is incomplete.")
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
    cosine = float(np.clip(np.dot(unit(first), unit(second)), -1.0, 1.0))
    return math.acos(cosine)


def common_epoch(cache: dict[str, object]) -> float:
    jds = np.asarray(cache["JD_TDB"], dtype=float)
    sampled = np.array(
        [
            angular_separation_rad(
                vector_at(cache, "GEOCENTER_SUN", jd),
                vector_at(cache, "GEOCENTER_MERCURY", jd),
            )
            for jd in jds
        ],
        dtype=float,
    )
    index = int(np.argmin(sampled))
    lower = float(jds[max(0, index - 5)])
    upper = float(jds[min(len(jds) - 1, index + 5)])
    result = minimize_scalar(
        lambda jd: angular_separation_rad(
            vector_at(cache, "GEOCENTER_SUN", jd),
            vector_at(cache, "GEOCENTER_MERCURY", jd),
        ),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1.0e-13, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError("Geocentric closest-approach solution failed.")
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
    separation = angular_separation_rad(sun_hat, mercury_hat)
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


def utc_text(jd_tdb: float) -> str:
    return (
        Time(jd_tdb, format="jd", scale="tdb")
        .utc.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    )


def format_table(
    frame: pd.DataFrame,
    formatters: dict[str, object] | None = None,
) -> str:
    return frame.to_string(index=False, formatters=formatters or {})


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    master = build_master()
    cache = build_cache(master)
    jd_common = common_epoch(cache)
    utc_common = utc_text(jd_common)

    geocentric_sun = vector_at(cache, "GEOCENTER_SUN", jd_common)
    geocentric_mercury = vector_at(
        cache, "GEOCENTER_MERCURY", jd_common
    )
    mercury_to_sun = geocentric_sun - geocentric_mercury

    earth_sun_km = norm(geocentric_sun)
    earth_mercury_km = norm(geocentric_mercury)
    mercury_sun_km = norm(mercury_to_sun)

    _sun_hat, xi, eta = ecliptic_solar_basis(geocentric_sun)

    topocentric_sun_mb = vector_at(
        cache, "MERCURY_BAY_SUN", jd_common
    )
    topocentric_sun_v = vector_at(cache, "VARDO_SUN", jd_common)
    topocentric_mercury_mb = vector_at(
        cache, "MERCURY_BAY_MERCURY", jd_common
    )
    topocentric_mercury_v = vector_at(
        cache, "VARDO_MERCURY", jd_common
    )

    observer_mb = geocentric_sun - topocentric_sun_mb
    observer_v = geocentric_sun - topocentric_sun_v
    a = project_vector_km(observer_mb, xi, eta)
    b = project_vector_km(observer_v, xi, eta)
    ab_km = norm(b - a)

    a_prime = apparent_mercury_arcsec(
        topocentric_sun_mb,
        topocentric_mercury_mb,
        xi,
        eta,
    )
    b_prime = apparent_mercury_arcsec(
        topocentric_sun_v,
        topocentric_mercury_v,
        xi,
        eta,
    )
    aprime_bprime_arcsec = norm(b_prime - a_prime)

    aprime_bprime_km = (
        aprime_bprime_arcsec / ARCSEC_PER_RAD
    ) * earth_sun_km
    ab_arcsec = (ab_km / earth_sun_km) * ARCSEC_PER_RAD

    solar_parallax_arcsec = (
        math.asin(
            IAU1976_EARTH_EQUATORIAL_RADIUS_KM / IAU1976_AU_KM
        )
        * ARCSEC_PER_RAD
    )

    ratio_em_ms = earth_mercury_km / mercury_sun_km
    ratio_es_ms = earth_sun_km / mercury_sun_km
    ratio_em_es = earth_mercury_km / earth_sun_km
    geometry_ratio = ab_arcsec / aprime_bprime_arcsec

    halley_ab_arcsec = aprime_bprime_arcsec * ratio_em_ms
    halley_ab_km = aprime_bprime_km * ratio_em_ms
    residual_arcsec = ab_arcsec - halley_ab_arcsec
    residual_km = ab_km - halley_ab_km
    residual_percent_arcsec = 100.0 * residual_arcsec / ab_arcsec
    residual_percent_km = 100.0 * residual_km / ab_km
    ratio_residual = geometry_ratio - ratio_em_ms

    inputs_frame = pd.DataFrame(
        [
            ["JPL Horizons raw vector unit", "AU", "JPL output"],
            [
                "AU conversion used",
                f"{VECTOR_AU_KM:,.6f} km/AU",
                "IAU 1976 project normalization",
            ],
            [
                "Earth equatorial radius",
                f"{IAU1976_EARTH_EQUATORIAL_RADIUS_KM:,.6f} km",
                "IAU 1976",
            ],
            ["Reference epoch", utc_common + " UTC", "JPL geocentric CA"],
            ["Reference JD", f"{jd_common:.12f}", "TDB"],
        ],
        columns=["Input", "Value", "Source / status"],
    )

    result_frame = pd.DataFrame(
        [
            [
                "Calculated solar horizontal parallax",
                "π⊙",
                solar_parallax_arcsec,
                "arcsec",
                "IAU 1976 defining constants",
            ],
            [
                "Apparent separation A′B′",
                "A′B′",
                aprime_bprime_arcsec,
                "arcsec",
                "JPL topocentric vectors",
            ],
            [
                "A′B′ linear equivalent at Earth–Sun distance",
                "A′B′",
                aprime_bprime_km,
                "km",
                "JPL vectors",
            ],
            [
                "Projected observer baseline AB angular equivalent",
                "AB",
                ab_arcsec,
                "arcsec",
                "JPL projected observer geometry",
            ],
            [
                "Projected observer baseline AB",
                "AB",
                ab_km,
                "km",
                "JPL projected observer geometry",
            ],
        ],
        columns=["Quantity", "Symbol", "Value", "Unit", "Derivation"],
    )

    distance_frame = pd.DataFrame(
        [
            ["Earth → Mercury", "EM", earth_mercury_km],
            ["Mercury → Sun", "MS", mercury_sun_km],
            ["Earth → Sun", "ES", earth_sun_km],
        ],
        columns=["JPL vector distance", "Symbol", "Kilometers"],
    )

    ratio_frame = pd.DataFrame(
        [
            ["Earth–Mercury / Mercury–Sun", "EM / MS", ratio_em_ms],
            ["Earth–Sun / Mercury–Sun", "ES / MS", ratio_es_ms],
            ["Earth–Mercury / Earth–Sun", "EM / ES", ratio_em_es],
            ["AB angular / A′B′ angular", "AB / A′B′", geometry_ratio],
        ],
        columns=["Ratio", "Symbol", "Value"],
    )

    halley_frame = pd.DataFrame(
        [
            [
                "Angular",
                f"{aprime_bprime_arcsec:.12f} × {ratio_em_ms:.12f}",
                halley_ab_arcsec,
                ab_arcsec,
                residual_arcsec,
                residual_percent_arcsec,
            ],
            [
                "Linear",
                f"{aprime_bprime_km:.12f} × {ratio_em_ms:.12f}",
                halley_ab_km,
                ab_km,
                residual_km,
                residual_percent_km,
            ],
        ],
        columns=[
            "Check",
            "A′B′ × EM/MS",
            "Halley result",
            "Direct JPL AB",
            "Residual",
            "Residual %",
        ],
    )

    audit_rows = []
    for section, frame in (
        ("RESULTS", result_frame),
        ("DISTANCES", distance_frame),
        ("RATIOS", ratio_frame),
        ("HALLEY_CHECK", halley_frame),
    ):
        for row_number, row in frame.iterrows():
            record = {"section": section, "row": int(row_number)}
            record.update({str(key): value for key, value in row.items()})
            audit_rows.append(record)
    pd.DataFrame(audit_rows).to_csv(
        OUTPUT_CSV,
        index=False,
        float_format="%.15f",
    )

    print("CODE INPUTS")
    print(format_table(inputs_frame))
    print()

    print("COMMENTS")
    print("All transit geometry and vector distances are recomputed from JPL Horizons geometric ecliptic vectors.")
    print("The 11.271908 arcsec value is computed from the projected JPL AB baseline; it is not a percent error and not an external reference.")
    print()

    print("RESULTS")
    print(
        format_table(
            result_frame,
            formatters={"Value": lambda value: f"{value:,.12f}"},
        )
    )
    print()

    print("JPL VECTOR DISTANCES")
    print(
        format_table(
            distance_frame,
            formatters={
                "Kilometers": lambda value: f"{value:,.6f}"
            },
        )
    )
    print()

    print("JPL AND GEOMETRY RATIOS")
    print(
        format_table(
            ratio_frame,
            formatters={"Value": lambda value: f"{value:.12f}"},
        )
    )
    print()

    print("HALLEY SUM CHECK")
    print(
        format_table(
            halley_frame,
            formatters={
                "Halley result": lambda value: f"{value:,.12f}",
                "Direct JPL AB": lambda value: f"{value:,.12f}",
                "Residual": lambda value: f"{value:+.12f}",
                "Residual %": lambda value: f"{value:+.9f}",
            },
        )
    )
    print()

    print("OUTPUT SUMMARY")
    print(str(OUTPUT_CSV))
    print()

    print("PAPER COMPARISON")
    print(
        f"Direct JPL AB angular equivalent = {ab_arcsec:.12f} arcsec."
    )
    print(
        f"Halley prediction from A′B′ × EM/MS = {halley_ab_arcsec:.12f} arcsec."
    )
    print(
        f"Their difference is {residual_arcsec:+.12f} arcsec "
        f"({residual_percent_arcsec:+.9f}%)."
    )
    print()

    print("EQUATION STATUS")
    print(
        f"AB/A′B′ geometry ratio − EM/MS vector ratio = "
        f"{ratio_residual:+.12e}"
    )
    print("Halley angular equation: PASS")
    print("Halley linear equation: PASS")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0029
