# V0028
# Audit reference: JPL Horizons vector audit of the 1769 Mercury Halley ratio and the A′B′ to AB back-check.
from __future__ import annotations

import math
import subprocess
import sys
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

VERSION = "V0028"
LOCAL_TZ = ZoneInfo("America/Bogota")
OUTPUT_DIR = Path("/content/MERCURY_HALLEY_RATIO_AUDIT_V0028_OUTPUT")
OUTPUT_CSV = OUTPUT_DIR / "MERCURY_HALLEY_RATIO_AUDIT_V0028.csv"

JPL_AU_KM = 149_597_870.7
ARCSEC_PER_RAD = 206_264.80624709636
START = "1769-11-09 12:00"
STOP = "1769-11-10 08:00"
STEP = "1m"
GEOCENTER = "500@399"

SITE_MERCURY_BAY = {
    "label": "Mercury Bay",
    "key": "MERCURY_BAY",
    "lat": -36.783333333333,
    "lon": 175.933333333333,
    "elevation": 0.0,
    "body": 399,
}
SITE_VARDO = {
    "label": "Vardø",
    "key": "VARDO",
    "lat": 70.370600000000,
    "lon": 31.110700000000,
    "elevation": 0.0,
    "body": 399,
}
SITES = (SITE_MERCURY_BAY, SITE_VARDO)
TARGETS = (("SUN", "10"), ("MERCURY", "199"))

A_PRIME_B_PRIME_PROJECT_ARCSEC = 5.215833
AB_PROJECT_ARCSEC = 11.271908


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


def download_vector_series(
    prefix: str,
    target_id: str,
    location: str | dict[str, object],
) -> pd.DataFrame:
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
        {"JD_TDB": pd.to_numeric(frame["datetime_jd"], errors="coerce")}
    )
    for axis in "xyz":
        output[f"{prefix}_{axis.upper()}_KM"] = (
            pd.to_numeric(frame[axis], errors="coerce") * JPL_AU_KM
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
            download_vector_series(
                f"GEOCENTER_{target_name}",
                target_id,
                GEOCENTER,
            )
        )
    for site in SITES:
        location = site_location(site)
        for target_name, target_id in TARGETS:
            series.append(
                download_vector_series(
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
        raise RuntimeError("The synchronized JPL master contains too few rows.")
    return master


def build_cache(master: pd.DataFrame) -> dict[str, object]:
    jds = master["JD_TDB"].to_numpy(dtype=float)
    cache: dict[str, object] = {"JD_TDB": jds}
    for column in master.columns:
        if column.endswith("_KM"):
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


def closest_approach(
    cache: dict[str, object],
    site_key: str,
) -> float:
    jds = np.asarray(cache["JD_TDB"], dtype=float)
    separations = np.array(
        [
            angular_separation_rad(
                vector_at(cache, f"{site_key}_SUN", jd),
                vector_at(cache, f"{site_key}_MERCURY", jd),
            )
            for jd in jds
        ],
        dtype=float,
    )
    index = int(np.argmin(separations))
    lower = float(jds[max(0, index - 4)])
    upper = float(jds[min(len(jds) - 1, index + 4)])
    solution = minimize_scalar(
        lambda jd: angular_separation_rad(
            vector_at(cache, f"{site_key}_SUN", jd),
            vector_at(cache, f"{site_key}_MERCURY", jd),
        ),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1.0e-13, "maxiter": 500},
    )
    if not solution.success:
        raise RuntimeError(f"Closest-approach solution failed for {site_key}.")
    return float(solution.x)


def utc_text(jd_tdb: float) -> str:
    return (
        Time(jd_tdb, format="jd", scale="tdb")
        .utc.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    master = build_master()
    cache = build_cache(master)

    jd_mb = closest_approach(cache, "MERCURY_BAY")
    jd_v = closest_approach(cache, "VARDO")
    jd_reference = 0.5 * (jd_mb + jd_v)

    earth_sun_vector = vector_at(cache, "GEOCENTER_SUN", jd_reference)
    earth_mercury_vector = vector_at(
        cache,
        "GEOCENTER_MERCURY",
        jd_reference,
    )
    mercury_sun_vector = earth_sun_vector - earth_mercury_vector

    earth_mercury_km = norm(earth_mercury_vector)
    mercury_sun_km = norm(mercury_sun_vector)
    earth_sun_km = norm(earth_sun_vector)

    ratio_em_ms = earth_mercury_km / mercury_sun_km
    ratio_es_ms = earth_sun_km / mercury_sun_km
    ratio_em_es = earth_mercury_km / earth_sun_km

    ab_forward_arcsec = (
        A_PRIME_B_PRIME_PROJECT_ARCSEC * ratio_em_ms
    )
    aprime_inverse_arcsec = AB_PROJECT_ARCSEC / ratio_em_ms

    epoch_table = pd.DataFrame(
        [
            ["Mercury Bay closest approach", utc_text(jd_mb), jd_mb],
            ["Vardø closest approach", utc_text(jd_v), jd_v],
            ["Reference epoch", utc_text(jd_reference), jd_reference],
        ],
        columns=["Epoch", "UTC", "JD TDB"],
    )

    distance_table = pd.DataFrame(
        [
            ["Earth → Mercury", "EM", earth_mercury_km],
            ["Mercury → Sun", "MS", mercury_sun_km],
            ["Earth → Sun", "ES", earth_sun_km],
        ],
        columns=["Distance", "Symbol", "Kilometers"],
    )

    ratio_table = pd.DataFrame(
        [
            ["Earth–Mercury / Mercury–Sun", "EM / MS", ratio_em_ms],
            ["Earth–Sun / Mercury–Sun", "ES / MS", ratio_es_ms],
            ["Earth–Mercury / Earth–Sun", "EM / ES", ratio_em_es],
        ],
        columns=["Ratio", "Symbol", "Value"],
    )

    check_table = pd.DataFrame(
        [
            [
                "Forward",
                A_PRIME_B_PRIME_PROJECT_ARCSEC,
                ratio_em_ms,
                ab_forward_arcsec,
                AB_PROJECT_ARCSEC,
                ab_forward_arcsec - AB_PROJECT_ARCSEC,
            ],
            [
                "Inverse",
                aprime_inverse_arcsec,
                ratio_em_ms,
                AB_PROJECT_ARCSEC,
                AB_PROJECT_ARCSEC,
                aprime_inverse_arcsec
                - A_PRIME_B_PRIME_PROJECT_ARCSEC,
            ],
        ],
        columns=[
            "Check",
            "Input arcsec",
            "EM/MS",
            "Calculated arcsec",
            "Project reference arcsec",
            "Residual arcsec",
        ],
    )

    audit_rows = [
        {
            "quantity": "Reference JD TDB",
            "symbol": "JD",
            "value": jd_reference,
            "unit": "day",
        },
        {
            "quantity": "Earth to Mercury",
            "symbol": "EM",
            "value": earth_mercury_km,
            "unit": "km",
        },
        {
            "quantity": "Mercury to Sun",
            "symbol": "MS",
            "value": mercury_sun_km,
            "unit": "km",
        },
        {
            "quantity": "Earth to Sun",
            "symbol": "ES",
            "value": earth_sun_km,
            "unit": "km",
        },
        {
            "quantity": "Halley ratio",
            "symbol": "EM/MS",
            "value": ratio_em_ms,
            "unit": "dimensionless",
        },
        {
            "quantity": "A′B′ project value",
            "symbol": "A′B′",
            "value": A_PRIME_B_PRIME_PROJECT_ARCSEC,
            "unit": "arcsec",
        },
        {
            "quantity": "AB calculated",
            "symbol": "AB",
            "value": ab_forward_arcsec,
            "unit": "arcsec",
        },
        {
            "quantity": "AB project reference",
            "symbol": "AB",
            "value": AB_PROJECT_ARCSEC,
            "unit": "arcsec",
        },
    ]
    pd.DataFrame(audit_rows).to_csv(
        OUTPUT_CSV,
        index=False,
        float_format="%.15f",
    )

    print("JPL EPOCH")
    print(
        epoch_table.to_string(
            index=False,
            formatters={"JD TDB": lambda value: f"{value:.12f}"},
        )
    )
    print()
    print("JPL DISTANCES")
    print(
        distance_table.to_string(
            index=False,
            formatters={
                "Kilometers": lambda value: f"{value:,.6f}",
            },
        )
    )
    print()
    print("JPL RATIOS")
    print(
        ratio_table.to_string(
            index=False,
            formatters={"Value": lambda value: f"{value:.12f}"},
        )
    )
    print()
    print("HALLEY CHECK")
    print(
        check_table.to_string(
            index=False,
            formatters={
                "Input arcsec": lambda value: f"{value:.6f}",
                "EM/MS": lambda value: f"{value:.12f}",
                "Calculated arcsec": lambda value: f"{value:.6f}",
                "Project reference arcsec": lambda value: f"{value:.6f}",
                "Residual arcsec": lambda value: f"{value:+.9f}",
            },
        )
    )
    print()
    print(str(OUTPUT_CSV))
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0028
