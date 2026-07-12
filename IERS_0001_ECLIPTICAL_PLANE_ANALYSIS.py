"""
IERS TN36 — Ecliptical Plane Analysis
Part F — Direct JPL Horizons vector acquisition
Bucaramanga observer, 2012 Venus transit
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from typing import Dict
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

VERSION = "IERS-0001-F"
HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"
HORIZONS_API_VERSION = "1.3"

START_TIME_UTC = "2012-06-05 21:00"
STOP_TIME_UTC = "2012-06-06 07:00"
STEP_SIZE = "1 min"


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

TARGETS = {
    "Sun": "10",
    "Venus": "299",
}


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
    return f"{HORIZONS_API}?{urlencode(build_horizons_parameters(target_id, observer))}"


def request_horizons_text(target_id: str, observer: Observer) -> tuple[str, str]:
    url = build_horizons_url(target_id, observer)
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
    source = str(signature.get("source", ""))
    api_version = str(signature.get("version", ""))
    result = payload.get("result")

    if "NASA/JPL" not in source:
        raise RuntimeError(f"Unexpected Horizons API source: {source!r}")
    if not isinstance(result, str) or not result.strip():
        error_text = payload.get("error", "No Horizons result text returned.")
        raise RuntimeError(str(error_text))

    return result, api_version


def canonical_header(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def find_vector_header(lines: list[str], soe_index: int) -> list[str]:
    for index in range(soe_index - 1, max(-1, soe_index - 20), -1):
        row = next(csv.reader([lines[index]], skipinitialspace=True))
        names = [canonical_header(item) for item in row]
        if all(axis in names for axis in ("X", "Y", "Z")):
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
        diagnostic = result_text[-1000:]
        raise ValueError(f"{label}: missing $$SOE/$$EOE markers.\n{diagnostic}") from exc

    header = find_vector_header(lines, soe_index)
    names = [canonical_header(item) for item in header]

    jd_index = next(
        (names.index(name) for name in ("JDUT", "JDTDB", "JD") if name in names),
        None,
    )
    if jd_index is None:
        raise ValueError(f"{label}: no Julian-date column found in {header!r}.")

    indices = {
        "x": names.index("X"),
        "y": names.index("Y"),
        "z": names.index("Z"),
        "vx": names.index("VX"),
        "vy": names.index("VY"),
        "vz": names.index("VZ"),
    }

    epochs: list[float] = []
    positions: list[list[float]] = []
    velocities: list[list[float]] = []

    for row in csv.reader(lines[soe_index + 1:eoe_index], skipinitialspace=True):
        if not row:
            continue
        needed = [jd_index, *indices.values()]
        if max(needed) >= len(row):
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
    return {
        label: fetch_target_vectors(label, target_id)
        for label, target_id in TARGETS.items()
    }


def vector_cadence_seconds(epochs: np.ndarray) -> float:
    return float(np.median(np.diff(epochs)) * 86400.0)


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

    print("\nCENTER              LONGITUDE deg     LATITUDE deg     ELEVATION km     FRAME     CORRECTION")
    print(
        f"{BUCARAMANGA.center:<19} {BUCARAMANGA.longitude_deg:13.6f}  "
        f"{BUCARAMANGA.latitude_deg:13.6f}  {BUCARAMANGA.elevation_km:13.6f}  "
        f"{'ICRF':<8}  NONE"
    )


def main() -> None:
    print("IERS TN36 - Ecliptical Plane Analysis")
    print(f"Version : {VERSION}")
    print(f"Observer: {BUCARAMANGA.name}")
    print("Transit : 2012-06-06")
    tables = fetch_bucaramanga_vectors()
    display_acquisition_summary(tables)


if __name__ == "__main__":
    main()
