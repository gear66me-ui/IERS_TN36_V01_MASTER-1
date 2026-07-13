# V0035
# Audit reference: Isolate the 1769 Tahiti–Vardø total Halley solar-parallax reduction coefficient at the project φ0 epoch.
from __future__ import annotations

import contextlib
import io
import math
import subprocess
import sys
import time
import urllib.request
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def ensure_package(import_name: str, pip_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pip_name])


ensure_package("pandas", "pandas")
import pandas as pd

VERSION = "V0035"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUTPUT_DIR = ROOT / "VENUS_1769_TAHITI_VARDO_TOTAL_REDUCTION_COEFFICIENT_V0035_OUTPUT"
OUTPUT_CSV = OUTPUT_DIR / "VENUS_1769_TAHITI_VARDO_TOTAL_REDUCTION_COEFFICIENT_V0035.csv"

SOURCE_COMMIT = "b3bee74b7d8e32269b1457875f7ec644000642a3"
SOURCE_NAME = "VENUS_1769_TAHITI_VARDO_PHI0_EPOCH_REDUCTION_V0034.py"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/"
    f"{SOURCE_COMMIT}/{SOURCE_NAME}"
)
SOURCE_PATH = ROOT / SOURCE_NAME
V0034_CSV = (
    ROOT
    / "VENUS_1769_TAHITI_VARDO_PHI0_EPOCH_REDUCTION_V0034_OUTPUT"
    / "VENUS_1769_TAHITI_VARDO_PHI0_EPOCH_REDUCTION_V0034.csv"
)

EARTH_EQUATORIAL_RADIUS_KM = 6_378.140000
ARCSEC_PER_RAD = 206_264.80624709636
PHI0_UTC = "1769-06-03 22:19:15.599 UTC"


def fetch_pinned_source() -> str:
    request = urllib.request.Request(
        f"{SOURCE_URL}?cache={time.time_ns()}",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        source = response.read().decode("utf-8")
    if not source.startswith("# V0034\n") or not source.rstrip().endswith("# V0034"):
        raise RuntimeError("Pinned V0034 source-boundary audit failed.")
    compile(source, str(SOURCE_PATH), "exec")
    SOURCE_PATH.write_text(source, encoding="utf-8")
    return source


def run_v0034_silently(source: str) -> None:
    namespace = {"__name__": "__main__", "__file__": str(SOURCE_PATH)}
    sink_out = io.StringIO()
    sink_err = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(sink_out), contextlib.redirect_stderr(sink_err):
            exec(compile(source, str(SOURCE_PATH), "exec"), namespace)
    if not V0034_CSV.is_file():
        raise RuntimeError("V0034 audit CSV was not produced.")


def require_one(frame: pd.DataFrame, column: str, value: str) -> pd.Series:
    selected = frame[frame[column].astype(str) == value]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one row where {column}={value!r}; found {len(selected)}.")
    return selected.iloc[0]


def read_v0034_values() -> dict[str, float]:
    audit = pd.read_csv(V0034_CSV)

    separations = audit[audit["section"] == "SEPARATIONS"]
    distances = audit[audit["section"] == "DISTANCES"]

    aprime_bprime = require_one(separations, "Separation", "A′B′ direct JPL")
    ab_direct = require_one(separations, "Separation", "AB direct JPL")
    ev = require_one(distances, "Symbol", "EV")
    vs = require_one(distances, "Symbol", "VS")
    es = require_one(distances, "Symbol", "ES")

    return {
        "aprime_bprime_arcsec": float(aprime_bprime["Arcseconds"]),
        "aprime_bprime_km": float(aprime_bprime["Kilometers"]),
        "ab_direct_arcsec": float(ab_direct["Arcseconds"]),
        "ab_direct_km": float(ab_direct["Kilometers"]),
        "ev_km": float(ev["Kilometers"]),
        "vs_km": float(vs["Kilometers"]),
        "es_km": float(es["Kilometers"]),
    }


def format_table(frame: pd.DataFrame) -> str:
    formatters = {
        column: (lambda value: f"{float(value):,.12f}")
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    }
    return frame.to_string(index=False, formatters=formatters)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = fetch_pinned_source()
    run_v0034_silently(source)
    values = read_v0034_values()

    aprime_bprime_arcsec = values["aprime_bprime_arcsec"]
    aprime_bprime_km = values["aprime_bprime_km"]
    ab_direct_arcsec = values["ab_direct_arcsec"]
    ab_direct_km = values["ab_direct_km"]
    ev_km = values["ev_km"]
    vs_km = values["vs_km"]
    es_km = values["es_km"]

    ev_es = ev_km / es_km
    vs_es = vs_km / es_km
    ev_vs = ev_km / vs_km
    earth_radius_over_ab = EARTH_EQUATORIAL_RADIUS_KM / ab_direct_km

    classical_total_coefficient = earth_radius_over_ab * ev_vs
    classical_phi0_arcsec = aprime_bprime_arcsec * classical_total_coefficient

    direct_transfer_factor = ab_direct_arcsec / aprime_bprime_arcsec
    exact_total_coefficient = earth_radius_over_ab * direct_transfer_factor
    exact_phi0_arcsec = aprime_bprime_arcsec * exact_total_coefficient
    epoch_solar_parallax_arcsec = (
        math.asin(EARTH_EQUATORIAL_RADIUS_KM / es_km) * ARCSEC_PER_RAD
    )

    ab_classical_arcsec = aprime_bprime_arcsec * ev_vs
    ab_classical_km = aprime_bprime_km * ev_vs

    coefficient_difference = classical_total_coefficient - exact_total_coefficient
    phi0_difference_arcsec = classical_phi0_arcsec - exact_phi0_arcsec

    inputs = pd.DataFrame(
        [
            ["Reference UTC", PHI0_UTC],
            ["Earth equatorial radius", f"{EARTH_EQUATORIAL_RADIUS_KM:,.6f} km"],
            ["Source audit", str(V0034_CSV)],
        ],
        columns=["Input", "Value"],
    )

    geometry = pd.DataFrame(
        [
            ["A′B′ direct JPL", aprime_bprime_arcsec, aprime_bprime_km],
            ["AB direct JPL", ab_direct_arcsec, ab_direct_km],
        ],
        columns=["JPL geometry", "Arcseconds", "Kilometers"],
    )

    distances = pd.DataFrame(
        [
            ["Earth → Venus", "EV", ev_km],
            ["Venus → Sun", "VS", vs_km],
            ["Earth → Sun", "ES", es_km],
        ],
        columns=["JPL distance", "Symbol", "Kilometers"],
    )

    ratios = pd.DataFrame(
        [
            ["Orbital fraction", "EV / ES", ev_es],
            ["Remaining Sun distance", "VS / ES", vs_es],
            ["Classical Halley transfer", "EV / VS", ev_vs],
            ["Baseline normalization", "R⊕ / AB", earth_radius_over_ab],
            [
                "TOTAL CLASSICAL φ0 COEFFICIENT",
                "(R⊕ / AB) × (EV / VS)",
                classical_total_coefficient,
            ],
            ["Exact JPL transfer", "AB(arcsec) / A′B′", direct_transfer_factor],
            [
                "TOTAL EXACT φ0 COEFFICIENT",
                "(R⊕ / AB) × (AB / A′B′)",
                exact_total_coefficient,
            ],
        ],
        columns=["Ratio / coefficient", "Definition", "Value"],
    )

    reductions = pd.DataFrame(
        [
            [
                "Classical AB angular equivalent",
                f"{aprime_bprime_arcsec:.12f} × {ev_vs:.12f}",
                ab_classical_arcsec,
                "arcsec",
            ],
            [
                "Classical AB linear equivalent",
                f"{aprime_bprime_km:.12f} × {ev_vs:.12f}",
                ab_classical_km,
                "km",
            ],
            [
                "Classical solar parallax φ0",
                f"{aprime_bprime_arcsec:.12f} × {classical_total_coefficient:.12f}",
                classical_phi0_arcsec,
                "arcsec",
            ],
            [
                "Exact JPL solar parallax φ0",
                f"{aprime_bprime_arcsec:.12f} × {exact_total_coefficient:.12f}",
                exact_phi0_arcsec,
                "arcsec",
            ],
            [
                "JPL epoch solar horizontal parallax",
                "asin(R⊕ / ES)",
                epoch_solar_parallax_arcsec,
                "arcsec",
            ],
        ],
        columns=["Result", "Calculation", "Value", "Unit"],
    )

    audit_rows: list[dict[str, object]] = []
    for section, frame in (
        ("GEOMETRY", geometry),
        ("DISTANCES", distances),
        ("RATIOS", ratios),
        ("REDUCTIONS", reductions),
    ):
        for row_number, row in frame.iterrows():
            record: dict[str, object] = {"section": section, "row": int(row_number)}
            record.update({str(key): value for key, value in row.items()})
            audit_rows.append(record)
    pd.DataFrame(audit_rows).to_csv(OUTPUT_CSV, index=False, float_format="%.15f")

    print("CODE INPUTS")
    print(inputs.to_string(index=False))
    print()

    print("COMMENTS")
    print("The 0.284676... value is EV/ES; it is an orbital fraction, not the final multiplier from A′B′ to AB.")
    print("The multiplier from A′B′ to the classical AB angular equivalent is EV/VS = 0.397967...")
    print("The missing 0.2... number is the total solar-parallax coefficient (R⊕/AB) × (EV/VS).")
    print()

    print("RESULTS")
    print("JPL GEOMETRY")
    print(format_table(geometry))
    print()
    print("JPL DISTANCES")
    print(format_table(distances))
    print()
    print("RATIOS AND TOTAL COEFFICIENTS")
    print(format_table(ratios))
    print()
    print("REDUCTIONS")
    print(format_table(reductions))
    print()

    print("OUTPUT SUMMARY")
    print(str(OUTPUT_CSV))
    print()

    print("PAPER COMPARISON")
    comparison = pd.DataFrame(
        [
            ["Classical total coefficient", classical_total_coefficient, "ratio"],
            ["Exact total coefficient", exact_total_coefficient, "ratio"],
            ["Coefficient difference", coefficient_difference, "ratio"],
            ["Classical φ0", classical_phi0_arcsec, "arcsec"],
            ["Exact JPL φ0", exact_phi0_arcsec, "arcsec"],
            ["φ0 difference", phi0_difference_arcsec, "arcsec"],
        ],
        columns=["Quantity", "Value", "Unit"],
    )
    print(format_table(comparison))
    print()

    print("EQUATION STATUS")
    status = pd.DataFrame(
        [
            ["A′B′ × EV/VS = classical AB", ab_direct_arcsec - ab_classical_arcsec],
            ["A′B′ × total classical coefficient = classical φ0", 0.0],
            ["A′B′ × total exact coefficient = JPL epoch φ0", exact_phi0_arcsec - epoch_solar_parallax_arcsec],
        ],
        columns=["Equation", "Residual arcsec"],
    )
    print(format_table(status))
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0035
