# V0007
# Audit reference: Standalone IERS-0012N bootstrap and rounded-CSV-safe exact IAU-1976 convention transfer.
from __future__ import annotations

import csv
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

VERSION = "V0007"
PROGRAM = "IERS_REDUCTION_VS_JPL_VECTORS_V0007.py"
LOCAL_TZ = ZoneInfo("America/Bogota")
ARCSEC_PER_RAD = 206264.80624709636

WGS84_RADIUS_KM = 6378.137000
IAU2012_AU_KM = 149597870.700000
IAU1976_RADIUS_KM = 6378.140000
C_KM_S = 299792.458000
TAU_A_S = 499.004782000
IAU1976_EXACT_AU_KM = C_KM_S * TAU_A_S

ROOT = Path("/content")
OUTPUT_DIR = ROOT / "IERS_TN36_V01_MASTER_OUTPUT"
SOURCE_CSV = OUTPUT_DIR / "IERS-0012N_VARDO_POINT_VENUS_ENGINEERING_HALF_SUN_EVENTS_AND_GEOMETRY.csv"
SOURCE_SCRIPT = ROOT / "IERS_0012N_VARDO_POINT_VENUS_ENGINEERING_TRACK_PLOT_PI_SUN.py"
AUDIT_CSV = OUTPUT_DIR / "IERS_0012N_IAU1976_POST_AUDIT_V0007.csv"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "IERS_0012N_VARDO_POINT_VENUS_ENGINEERING_TRACK_PLOT_PI_SUN.py?v=7"
)


def download_file(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": PROGRAM})
    with urlopen(request, timeout=180) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError(f"Downloaded empty file from {url}")
    destination.write_bytes(payload)


def ensure_source_csv() -> bool:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SOURCE_CSV.is_file():
        return False

    download_file(SOURCE_URL, SOURCE_SCRIPT)
    completed = subprocess.run(
        [sys.executable, str(SOURCE_SCRIPT)],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stdout_tail = "\n".join(completed.stdout.splitlines()[-20:])
        stderr_tail = "\n".join(completed.stderr.splitlines()[-20:])
        raise RuntimeError(
            "IERS-0012N failed while generating the geometry CSV.\n"
            f"STDOUT tail:\n{stdout_tail}\n"
            f"STDERR tail:\n{stderr_tail}"
        )
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(
            "IERS-0012N completed but did not create the expected geometry CSV: "
            f"{SOURCE_CSV}"
        )
    return True


def read_geometry(path: Path) -> dict[str, float]:
    required = {
        "Normal Separation ρ",
        "AB Projected Baseline",
        "D EV / D VS",
        "D ES",
        "Computed π⊙",
    }
    values: dict[str, float] = {}
    in_results = False

    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if row[:4] == ["section", "quantity", "value", "unit"]:
                in_results = True
                continue
            if not in_results or len(row) < 4:
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
        raise RuntimeError("Missing IERS-0012N values: " + ", ".join(missing))
    return values


def exact_transfer_arcsec(modern_arcsec: float) -> float:
    convention_factor = (
        IAU1976_RADIUS_KM / WGS84_RADIUS_KM
    ) * (
        IAU2012_AU_KM / IAU1976_EXACT_AU_KM
    )
    modern_radians = modern_arcsec / ARCSEC_PER_RAD
    transferred_radians = math.asin(
        convention_factor * math.sin(modern_radians)
    )
    return transferred_radians * ARCSEC_PER_RAD


def calculate(values: dict[str, float]) -> dict[str, float]:
    rho_arcsec = values["Normal Separation ρ"]
    baseline_km = values["AB Projected Baseline"]
    ev_vs = values["D EV / D VS"]
    des_au = values["D ES"]
    reported_modern = values["Computed π⊙"]

    rounded_geometry_modern = (
        rho_arcsec
        * ev_vs
        * WGS84_RADIUS_KM
        / baseline_km
        * des_au
    )

    convention_factor = (
        IAU1976_RADIUS_KM / WGS84_RADIUS_KM
    ) * (
        IAU2012_AU_KM / IAU1976_EXACT_AU_KM
    )

    iau1976_from_reported = exact_transfer_arcsec(reported_modern)
    iau1976_from_rounded_geometry = exact_transfer_arcsec(
        rounded_geometry_modern
    )
    iau1976_exact_standard = (
        math.asin(IAU1976_RADIUS_KM / IAU1976_EXACT_AU_KM)
        * ARCSEC_PER_RAD
    )
    modern_exact_standard = (
        math.asin(WGS84_RADIUS_KM / IAU2012_AU_KM)
        * ARCSEC_PER_RAD
    )

    return {
        "rho_arcsec": rho_arcsec,
        "baseline_km": baseline_km,
        "ev_vs": ev_vs,
        "des_au": des_au,
        "reported_modern": reported_modern,
        "rounded_geometry_modern": rounded_geometry_modern,
        "modern_exact_standard": modern_exact_standard,
        "convention_factor": convention_factor,
        "iau1976_from_reported": iau1976_from_reported,
        "iau1976_from_rounded_geometry": iau1976_from_rounded_geometry,
        "iau1976_exact_standard": iau1976_exact_standard,
        "csv_rounding_difference_microarcsec": (
            rounded_geometry_modern - reported_modern
        ) * 1_000_000.0,
        "reported_transfer_residual_microarcsec": (
            iau1976_from_reported - iau1976_exact_standard
        ) * 1_000_000.0,
        "rounded_transfer_residual_microarcsec": (
            iau1976_from_rounded_geometry - iau1976_exact_standard
        ) * 1_000_000.0,
    }


def write_audit(result: dict[str, float], generated_source: bool) -> None:
    rows = [
        ("INPUT", "Source CSV", str(SOURCE_CSV), "path"),
        ("INPUT", "Source regenerated in this run", generated_source, "boolean"),
        ("INPUT", "Normal separation rho", result["rho_arcsec"], "arcsec"),
        ("INPUT", "Projected baseline AB", result["baseline_km"], "km"),
        ("INPUT", "D EV / D VS", result["ev_vs"], "ratio"),
        ("INPUT", "D ES", result["des_au"], "AU"),
        ("CONSTANT", "WGS84 radius", WGS84_RADIUS_KM, "km"),
        ("CONSTANT", "IAU 2012 AU", IAU2012_AU_KM, "km"),
        ("CONSTANT", "IAU 1976 radius", IAU1976_RADIUS_KM, "km"),
        ("CONSTANT", "IAU 1976 exact c tau_A AU", IAU1976_EXACT_AU_KM, "km"),
        ("RESULT", "IERS-0012N reported modern pi_sun", result["reported_modern"], "arcsec"),
        ("RESULT", "Rounded CSV modern recomputation", result["rounded_geometry_modern"], "arcsec"),
        ("RESULT", "Exact convention factor", result["convention_factor"], "ratio"),
        ("RESULT", "IAU 1976 from reported 12N result", result["iau1976_from_reported"], "arcsec"),
        ("RESULT", "IAU 1976 from rounded geometry", result["iau1976_from_rounded_geometry"], "arcsec"),
        ("RESULT", "IAU 1976 exact standard", result["iau1976_exact_standard"], "arcsec"),
        ("RESULT", "CSV rounding difference", result["csv_rounding_difference_microarcsec"], "microarcsec"),
        ("RESULT", "Reported transfer residual", result["reported_transfer_residual_microarcsec"], "microarcsec"),
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
            (result["reported_modern"] - result["modern_exact_standard"]) * 1_000_000.0,
        ),
        (
            "Rounded CSV audit",
            "Recomputed from exported geometry fields",
            result["rounded_geometry_modern"],
            result["csv_rounding_difference_microarcsec"],
        ),
        (
            "IAU-1976 transfer",
            "Same 12N geometry; a1976 and exact cτA",
            result["iau1976_from_reported"],
            result["reported_transfer_residual_microarcsec"],
        ),
        (
            "IAU-1976 exact",
            "Direct arcsine standard",
            result["iau1976_exact_standard"],
            0.0,
        ),
    ]
    body = "".join(
        "<tr>"
        f"<td>{label}</td>"
        f"<td>{method}</td>"
        f"<td class='num'>{value:.12f}</td>"
        f"<td class='num'>{delta:+.6f}</td>"
        "</tr>"
        for label, method, value, delta in rows
    )
    html = f"""
    <style>
      .v0007 {{
        width: 920px; max-width: 98%; background: #000; color: #fff;
        border: 1px solid #fff; padding: 12px; font-family: Georgia, serif;
      }}
      .v0007-title {{
        text-align: center; font-size: 16px; font-weight: 700;
        padding-bottom: 7px; margin-bottom: 10px; border-bottom: 1px solid #fff;
      }}
      .v0007 table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
      .v0007 th, .v0007 td {{ border: 1px solid #fff; padding: 7px; background: #000; color: #fff; }}
      .v0007 th:nth-child(1) {{ width: 21%; }}
      .v0007 th:nth-child(2) {{ width: 41%; }}
      .v0007 th:nth-child(3) {{ width: 20%; }}
      .v0007 th:nth-child(4) {{ width: 18%; }}
      .v0007 .num {{ text-align: right; font-family: ui-monospace, monospace; }}
      .v0007-note {{ margin-top: 10px; font-size: 12px; line-height: 1.45; }}
    </style>
    <div class='v0007'>
      <div class='v0007-title'>IERS-0012N — EXACT IAU-1976 CASE-2 CONVENTION AUDIT</div>
      <table>
        <thead><tr><th>Reduction</th><th>Convention</th><th>π⊙ (arcsec)</th><th>Residual (µas)</th></tr></thead>
        <tbody>{body}</tbody>
      </table>
      <div class='v0007-note'>
        The exported geometry fields are rounded, producing {result['csv_rounding_difference_microarcsec']:+.6f} µas in the independent CSV recomputation.<br>
        The authoritative convention transfer therefore starts from the reported IERS-0012N result and gives <b>{result['iau1976_from_reported']:.12f} arcsec</b>.<br>
        Both the transferred result and the direct IAU-1976 standard round to <b>8.794148 arcsec</b>.
      </div>
    </div>
    """
    display(HTML(html))


def main() -> None:
    generated_source = ensure_source_csv()
    values = read_geometry(SOURCE_CSV)
    result = calculate(values)
    write_audit(result, generated_source)
    display_table(result)

    checks = {
        "Source CSV exists": SOURCE_CSV.is_file(),
        "Rounded CSV recomputation within exported precision": abs(
            result["csv_rounding_difference_microarcsec"]
        ) <= 0.100000,
        "Transferred result rounds to 8.794148": round(
            result["iau1976_from_reported"], 6
        ) == 8.794148,
        "Exact standard rounds to 8.794148": round(
            result["iau1976_exact_standard"], 6
        ) == 8.794148,
        "Transferred result within 0.1 microarcsec": abs(
            result["reported_transfer_residual_microarcsec"]
        ) <= 0.100000,
        "Audit CSV saved": AUDIT_CSV.is_file(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Audit checks failed: " + ", ".join(failed))

    print(f"CODE OUTPUT: {VERSION}")
    print("CODE INPUTS")
    print(f"Program: {PROGRAM}")
    print(f"Source geometry CSV: {SOURCE_CSV}")
    print("COMMENTS")
    print("The 12N source is generated automatically when absent. Rounded CSV fields are treated at their actual precision.")
    print("RESULTS")
    print(f"IERS-0012N reported modern result: {result['reported_modern']:.12f} arcsec")
    print(f"IAU-1976 convention transfer: {result['iau1976_from_reported']:.12f} arcsec")
    print(f"IAU-1976 exact standard: {result['iau1976_exact_standard']:.12f} arcsec")
    print("OUTPUT SUMMARY")
    print(f"Audit CSV: {AUDIT_CSV}")
    print("PAPER COMPARISON")
    print(f"Convention-transfer residual: {result['reported_transfer_residual_microarcsec']:+.6f} microarcsec")
    print("EQUATION STATUS")
    print("All checks: PASS")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# V0007
