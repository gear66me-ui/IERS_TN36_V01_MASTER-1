# V0005
# Audit reference: Re-reduce the preserved IERS-0012N Tahiti–Vardø geometry with exact IAU-1976 Case-2 constants.
from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0005"
PROGRAM = "IERS_REDUCTION_VS_JPL_VECTORS_V0005.py"
LOCAL_TZ = ZoneInfo("America/Bogota")
ARCSEC_PER_RAD = 206264.80624709636

WGS84_RADIUS_KM = 6378.137000
IAU2012_AU_KM = 149597870.700000
IAU1976_RADIUS_KM = 6378.140000
C_KM_S = 299792.458000
TAU_A_S = 499.004782000
IAU1976_EXACT_AU_KM = C_KM_S * TAU_A_S

OUTPUT_ROOT = Path("/content/IERS_TN36_V01_MASTER_OUTPUT")
EXPECTED_SOURCE = OUTPUT_ROOT / "IERS-0012N_VARDO_POINT_VENUS_ENGINEERING_HALF_SUN_EVENTS_AND_GEOMETRY.csv"
AUDIT_CSV = OUTPUT_ROOT / "IERS_0012N_IAU1976_POST_AUDIT_V0005.csv"


def locate_source_csv() -> Path:
    if EXPECTED_SOURCE.is_file():
        return EXPECTED_SOURCE
    candidates = sorted(Path("/content").rglob("IERS-0012N*EVENTS_AND_GEOMETRY.csv"))
    if not candidates:
        raise FileNotFoundError(
            "Run IERS-0012N first. Its events-and-geometry CSV was not found under /content."
        )
    return candidates[0]


def read_geometry(path: Path) -> dict[str, float]:
    required = {
        "Normal Separation ρ",
        "AB Projected Baseline",
        "D EV / D VS",
        "D ES",
        "Computed π⊙",
    }
    values: dict[str, float] = {}
    in_geometry = False
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if row[:4] == ["section", "quantity", "value", "unit"]:
                in_geometry = True
                continue
            if not in_geometry or len(row) < 4:
                continue
            section, quantity, value, _unit = row[:4]
            if (
                section == "PI_SUN_GEOMETRIC_SOLUTION"
                and quantity in required
                and quantity not in values
            ):
                values[quantity] = float(value)
    missing = sorted(required.difference(values))
    if missing:
        raise RuntimeError("Missing IERS-0012N geometry values: " + ", ".join(missing))
    return values


def compute(values: dict[str, float]) -> dict[str, float]:
    rho_arcsec = values["Normal Separation ρ"]
    baseline_km = values["AB Projected Baseline"]
    ev_vs = values["D EV / D VS"]
    des_au2012 = values["D ES"]
    reported_modern = values["Computed π⊙"]
    des_km = des_au2012 * IAU2012_AU_KM

    raw_modern = rho_arcsec * ev_vs * WGS84_RADIUS_KM / baseline_km
    modern_geometry = raw_modern * des_km / IAU2012_AU_KM

    raw_iau1976 = rho_arcsec * ev_vs * IAU1976_RADIUS_KM / baseline_km
    iau1976_geometry = raw_iau1976 * des_km / IAU1976_EXACT_AU_KM

    modern_exact = math.asin(WGS84_RADIUS_KM / IAU2012_AU_KM) * ARCSEC_PER_RAD
    iau1976_exact = math.asin(IAU1976_RADIUS_KM / IAU1976_EXACT_AU_KM) * ARCSEC_PER_RAD
    convention_factor = (
        IAU1976_RADIUS_KM / WGS84_RADIUS_KM
    ) * (
        IAU2012_AU_KM / IAU1976_EXACT_AU_KM
    )

    return {
        "rho_arcsec": rho_arcsec,
        "baseline_km": baseline_km,
        "ev_vs": ev_vs,
        "des_au2012": des_au2012,
        "des_km": des_km,
        "reported_modern": reported_modern,
        "modern_geometry": modern_geometry,
        "modern_exact": modern_exact,
        "iau1976_geometry": iau1976_geometry,
        "iau1976_exact": iau1976_exact,
        "convention_factor": convention_factor,
        "modern_report_difference_microarcsec": (
            modern_geometry - reported_modern
        ) * 1_000_000.0,
        "modern_to_iau1976_change_microarcsec": (
            iau1976_geometry - modern_geometry
        ) * 1_000_000.0,
        "iau1976_geometry_minus_exact_microarcsec": (
            iau1976_geometry - iau1976_exact
        ) * 1_000_000.0,
    }


def write_audit(source: Path, result: dict[str, float]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [
        ("INPUT", "Source CSV", str(source), "path"),
        ("INPUT", "Normal separation rho", result["rho_arcsec"], "arcsec"),
        ("INPUT", "Projected baseline AB", result["baseline_km"], "km"),
        ("INPUT", "D EV / D VS", result["ev_vs"], "ratio"),
        ("INPUT", "JPL D ES", result["des_km"], "km"),
        ("CONSTANT", "WGS84 radius", WGS84_RADIUS_KM, "km"),
        ("CONSTANT", "IAU 2012 AU", IAU2012_AU_KM, "km"),
        ("CONSTANT", "IAU 1976 radius", IAU1976_RADIUS_KM, "km"),
        ("CONSTANT", "IAU 1976 exact c tau_A AU", IAU1976_EXACT_AU_KM, "km"),
        ("RESULT", "12N reported modern pi_sun", result["reported_modern"], "arcsec"),
        ("RESULT", "12N modern geometry recomputation", result["modern_geometry"], "arcsec"),
        ("RESULT", "12N IAU 1976 geometry reduction", result["iau1976_geometry"], "arcsec"),
        ("RESULT", "IAU 1976 exact arcsine", result["iau1976_exact"], "arcsec"),
        (
            "RESULT",
            "IAU 1976 geometry minus exact",
            result["iau1976_geometry_minus_exact_microarcsec"],
            "microarcsec",
        ),
    ]
    with AUDIT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "quantity", "value", "unit"])
        writer.writerows(rows)


def display_table(result: dict[str, float]) -> None:
    try:
        from IPython.display import HTML, display
    except Exception:
        return

    rows = [
        (
            "IERS-0012N reported",
            "WGS84 radius / IAU-2012 AU",
            result["reported_modern"],
            (result["reported_modern"] - result["iau1976_exact"]) * 1_000_000.0,
        ),
        (
            "IERS-0012N recomputed",
            "Same JPL geometry, modern convention",
            result["modern_geometry"],
            (result["modern_geometry"] - result["iau1976_exact"]) * 1_000_000.0,
        ),
        (
            "IERS-0012N re-reduced",
            "Same JPL geometry, IAU-1976 Case 2",
            result["iau1976_geometry"],
            result["iau1976_geometry_minus_exact_microarcsec"],
        ),
        (
            "IAU-1976 exact",
            "Direct arcsine of a1976 / cτA",
            result["iau1976_exact"],
            0.0,
        ),
    ]
    body = "".join(
        "<tr>"
        f"<td>{case}</td>"
        f"<td>{method}</td>"
        f"<td class='num'>{value:.12f}</td>"
        f"<td class='num'>{delta:+.6f}</td>"
        "</tr>"
        for case, method, value, delta in rows
    )
    html = f"""
    <style>
      .v0005-wrap {{
        width: 900px; max-width: 98%; background: #000000; color: #ffffff;
        border: 1px solid #ffffff; padding: 12px; font-family: Georgia, serif;
      }}
      .v0005-title {{
        text-align: center; font-weight: 700; font-size: 16px;
        border-bottom: 1px solid #ffffff; padding-bottom: 7px; margin-bottom: 10px;
      }}
      .v0005-wrap table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
      .v0005-wrap th, .v0005-wrap td {{
        border: 1px solid #ffffff; padding: 7px; background: #000000; color: #ffffff;
      }}
      .v0005-wrap th:nth-child(1) {{ width: 22%; }}
      .v0005-wrap th:nth-child(2) {{ width: 40%; }}
      .v0005-wrap th:nth-child(3) {{ width: 20%; }}
      .v0005-wrap th:nth-child(4) {{ width: 18%; }}
      .v0005-wrap .num {{ text-align: right; font-family: ui-monospace, monospace; }}
      .v0005-note {{ margin-top: 10px; font-size: 12px; line-height: 1.45; }}
    </style>
    <div class='v0005-wrap'>
      <div class='v0005-title'>IERS-0012N — IAU-1976 CASE-2 POST-AUDIT</div>
      <table>
        <thead><tr><th>Reduction</th><th>Convention</th><th>π⊙ (arcsec)</th><th>Δ vs IAU-76 (µas)</th></tr></thead>
        <tbody>{body}</tbody>
      </table>
      <div class='v0005-note'>
        The JPL Tahiti–Vardø geometry is unchanged. Only the adopted Earth-radius and astronomical-unit convention changes.<br>
        Convention factor = {result['convention_factor']:.15f}<br>
        IAU-1976 geometry result = <b>{result['iau1976_geometry']:.12f} arcsec</b><br>
        Exact IAU-1976 Case-2 result = <b>{result['iau1976_exact']:.12f} arcsec</b>
      </div>
    </div>
    """
    display(HTML(html))


def main() -> None:
    source = locate_source_csv()
    values = read_geometry(source)
    result = compute(values)
    write_audit(source, result)
    display_table(result)

    checks = {
        "Modern recomputation matches 12N": abs(
            result["modern_geometry"] - result["reported_modern"]
        ) <= 0.000000005,
        "IAU-1976 geometry rounds to 8.794148": round(
            result["iau1976_geometry"], 6
        ) == 8.794148,
        "IAU-1976 exact rounds to 8.794148": round(
            result["iau1976_exact"], 6
        ) == 8.794148,
        "Audit CSV saved": AUDIT_CSV.is_file(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Audit checks failed: " + ", ".join(failed))

    print(f"CODE OUTPUT: {VERSION}")
    print("CODE INPUTS")
    print(f"Program: {PROGRAM}")
    print(f"Source geometry CSV: {source}")
    print("COMMENTS")
    print("JPL geometry is preserved; only the radius and AU convention is changed.")
    print("RESULTS")
    print(f"IERS-0012N modern result: {result['modern_geometry']:.12f} arcsec")
    print(f"IAU-1976 geometry reduction: {result['iau1976_geometry']:.12f} arcsec")
    print("OUTPUT SUMMARY")
    print(f"Audit CSV: {AUDIT_CSV}")
    print("PAPER COMPARISON")
    print(f"IAU-1976 exact cτA: {result['iau1976_exact']:.12f} arcsec")
    print("EQUATION STATUS")
    print("All checks: PASS")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# V0005
