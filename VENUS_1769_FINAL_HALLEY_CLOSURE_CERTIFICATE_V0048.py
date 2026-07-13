# V0048
# Audit reference: Consolidate physical distances, geometric corrections, AB closure, and normalized solar parallax π.
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from IPython.display import HTML, display

VERSION = "V0048"
TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
SOURCE_V0043 = (
    ROOT
    / "VENUS_1769_FULL_VECTOR_P_REDUCTION_AUDIT_V0043_OUTPUT"
    / "VENUS_1769_FULL_VECTOR_P_REDUCTION_AUDIT_V0043.csv"
)
SOURCE_V0047 = (
    ROOT
    / "VENUS_1769_CORRECTED_HALLEY_COEFFICIENT_AUDIT_V0047_OUTPUT"
    / "VENUS_1769_CORRECTED_HALLEY_COEFFICIENT_AUDIT_V0047.csv"
)
OUT = ROOT / "VENUS_1769_FINAL_HALLEY_CLOSURE_CERTIFICATE_V0048_OUTPUT"
CSV = OUT / "VENUS_1769_FINAL_HALLEY_CLOSURE_CERTIFICATE_V0048.csv"
HTML_FILE = OUT / "VENUS_1769_FINAL_HALLEY_CLOSURE_CERTIFICATE_V0048.html"
NORMAL = "Whole-transit fitted normal"
ARCSEC_PER_RAD = 206_264.80624709636
EARTH_EQUATORIAL_RADIUS_KM = 6_378.140000
AU_KM = 149_597_870.000000


def numeric(value: object) -> float:
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(converted):
        raise RuntimeError(f"Expected numeric value, received {value!r}.")
    return float(converted)


def unique_row(frame: pd.DataFrame, **criteria: str) -> pd.Series:
    selected = frame.copy()
    for column, value in criteria.items():
        if column not in selected.columns:
            raise RuntimeError(f"Missing required column {column!r}.")
        selected = selected[selected[column].astype(str) == value]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one row for {criteria!r}; found {len(selected)}."
        )
    return selected.iloc[0]


def table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    shown = frame.copy()
    for column, pattern in (formats or {}).items():
        if column not in shown.columns:
            continue

        def formatter(value: object) -> str:
            if pd.isna(value):
                return ""
            converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if not pd.isna(converted):
                return pattern.format(float(converted))
            return str(value)

        shown[column] = shown[column].map(formatter)
    return (
        '<div class="wrap">'
        + shown.to_html(index=False, border=0, classes="audit", escape=False)
        + "</div>"
    )


def main() -> None:
    missing = [str(path) for path in (SOURCE_V0043, SOURCE_V0047) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required source CSV files are missing. Run V0043 and V0047 first: "
            + " | ".join(missing)
        )

    OUT.mkdir(parents=True, exist_ok=True)
    source43 = pd.read_csv(SOURCE_V0043)
    source47 = pd.read_csv(SOURCE_V0047)

    epoch = source43[source43["section"] == "EPOCH"].copy()
    baselines = source43[source43["section"] == "BASELINES"].copy()
    residuals = source43[source43["section"] == "RESIDUAL_BUDGET"].copy()
    coefficients = source47[source47["section"] == "COEFFICIENTS"].copy()
    corrections = source47[source47["section"] == "CORRECTIONS"].copy()

    ev = numeric(unique_row(epoch, Quantity="Earth → Venus EV km")["Value"])
    vs = numeric(unique_row(epoch, Quantity="Venus → Sun VS km")["Value"])
    es = numeric(unique_row(epoch, Quantity="Earth → Sun ES km")["Value"])
    km_per_arcsec = numeric(
        unique_row(epoch, Quantity="Earth–Sun scale km/arcsec")["Value"]
    )

    baseline_row = unique_row(
        baselines,
        **{
            "Station model": "JPL topocentric-derived",
            "Normal definition": NORMAL,
        },
    )
    aprime_bprime_arcsec = numeric(baseline_row["Exact synthesized A′B′ arcsec"])
    ab_km = numeric(baseline_row["Direct common-normal AB km"])
    ab_arcsec = ab_km / km_per_arcsec

    classical_factor = numeric(
        unique_row(
            coefficients,
            **{
                "Normal definition": NORMAL,
                "Coefficient": "Classical physical-distance factor",
            },
        )["Value"]
    )
    exact_factor = numeric(
        unique_row(
            coefficients,
            **{
                "Normal definition": NORMAL,
                "Coefficient": "Exact nonlinear screen factor",
            },
        )["Value"]
    )
    reconstructed_factor = numeric(
        unique_row(
            coefficients,
            **{
                "Normal definition": NORMAL,
                "Coefficient": "Reconstructed exact factor",
            },
        )["Value"]
    )
    total_multiplier = numeric(
        unique_row(
            corrections,
            **{
                "Normal definition": NORMAL,
                "Correction": "Total geometric correction",
            },
        )["Multiplier"]
    )

    residual_row = unique_row(residuals, **{"Normal definition": NORMAL})
    contribution_specification = (
        (
            "Finite-baseline nonlinearity",
            "Finite-baseline residual AB km",
        ),
        (
            "Gnomonic-screen Jacobian",
            "Gnomonic-screen correction AB km",
        ),
        (
            "Separate Venus/Sun projectors",
            "Separate-projector correction AB km",
        ),
        (
            "Distance-triangle substitution",
            "Distance-triangle correction AB km",
        ),
    )
    total_residual_km = numeric(residual_row["Exact minus Halley AB km"])
    contribution_rows: list[list[object]] = []
    contribution_sum = 0.0
    for label, column in contribution_specification:
        value = numeric(residual_row[column])
        contribution_sum += value
        contribution_rows.append(
            [
                label,
                value,
                100.0 * value / total_residual_km,
            ]
        )
    contribution_rows.append(
        ["Total exact minus classical", total_residual_km, 100.0]
    )

    classical_ab_arcsec = aprime_bprime_arcsec * classical_factor
    classical_ab_km = classical_ab_arcsec * km_per_arcsec
    exact_ab_arcsec = aprime_bprime_arcsec * exact_factor
    exact_ab_km = exact_ab_arcsec * km_per_arcsec

    pi_event_classical = classical_ab_arcsec * EARTH_EQUATORIAL_RADIUS_KM / ab_km
    pi_event_exact = exact_ab_arcsec * EARTH_EQUATORIAL_RADIUS_KM / ab_km
    normalization = es / AU_KM
    pi_1au_classical = pi_event_classical * normalization
    pi_1au_exact = pi_event_exact * normalization
    pi_event_direct = EARTH_EQUATORIAL_RADIUS_KM / es * ARCSEC_PER_RAD
    pi_1au_direct = EARTH_EQUATORIAL_RADIUS_KM / AU_KM * ARCSEC_PER_RAD

    physical_frame = pd.DataFrame(
        [
            ["Earth → Venus", "EV", ev],
            ["Venus → Sun", "VS", vs],
            ["Earth → Sun", "ES", es],
        ],
        columns=["Physical JPL distance", "Symbol", "Kilometers"],
    )
    geometry_frame = pd.DataFrame(
        [
            ["JPL A′B′", aprime_bprime_arcsec, aprime_bprime_arcsec * km_per_arcsec],
            ["Direct JPL/IERS AB", ab_arcsec, ab_km],
            ["Classical Halley AB", classical_ab_arcsec, classical_ab_km],
            ["Corrected exact AB", exact_ab_arcsec, exact_ab_km],
        ],
        columns=["Quantity", "Arcseconds", "Kilometers"],
    )
    factor_frame = pd.DataFrame(
        [
            ["Classical physical factor", "EV/VS", classical_factor],
            ["Total geometric multiplier", "C_total", total_multiplier],
            ["Corrected reconstructed factor", "(EV/VS) × C_total", reconstructed_factor],
            ["Direct exact screen factor", "AB/A′B′", exact_factor],
            [
                "Classical-to-exact relative reduction",
                "1 − K_exact/K_classical",
                1.0 - exact_factor / classical_factor,
            ],
        ],
        columns=["Factor", "Equation", "Value"],
    )
    contribution_frame = pd.DataFrame(
        contribution_rows,
        columns=["Geometric term", "Equivalent AB km", "Percent of total"],
    )
    pi_frame = pd.DataFrame(
        [
            ["Classical physical-distance reduction", pi_event_classical, pi_1au_classical],
            ["Corrected exact JPL/IERS reduction", pi_event_exact, pi_1au_exact],
            ["Direct distance check", pi_event_direct, pi_1au_direct],
        ],
        columns=["Solar-parallax result", "π event arcsec", "π 1-AU arcsec"],
    )

    status_frame = pd.DataFrame(
        [
            [
                "Physical distance ratio",
                "PASS" if abs(classical_factor - ev / vs) < 1.0e-14 else "FAIL",
                classical_factor - ev / vs,
            ],
            [
                "Correction-chain reconstruction",
                "PASS" if abs(reconstructed_factor - exact_factor) < 1.0e-14 else "FAIL",
                reconstructed_factor - exact_factor,
            ],
            [
                "Residual-budget sum",
                "PASS" if abs(contribution_sum - total_residual_km) < 1.0e-9 else "FAIL",
                contribution_sum - total_residual_km,
            ],
            [
                "Exact AB closure km",
                "PASS" if abs(exact_ab_km - ab_km) < 1.0e-9 else "FAIL",
                exact_ab_km - ab_km,
            ],
            [
                "π event closure",
                "PASS" if abs(pi_event_exact - pi_event_direct) < 1.0e-12 else "FAIL",
                pi_event_exact - pi_event_direct,
            ],
            [
                "π 1-AU closure",
                "PASS" if abs(pi_1au_exact - pi_1au_direct) < 1.0e-12 else "FAIL",
                pi_1au_exact - pi_1au_direct,
            ],
        ],
        columns=["Equation / test", "Status", "Residual"],
    )

    records: list[dict[str, object]] = []
    for section, frame in (
        ("PHYSICAL_DISTANCES", physical_frame),
        ("GEOMETRY", geometry_frame),
        ("FACTORS", factor_frame),
        ("CONTRIBUTIONS", contribution_frame),
        ("PI", pi_frame),
        ("STATUS", status_frame),
    ):
        for row_number, row in frame.iterrows():
            record: dict[str, object] = {"section": section, "row": int(row_number)}
            record.update({str(key): value for key, value in row.items()})
            records.append(record)
    pd.DataFrame(records).to_csv(CSV, index=False, float_format="%.15f")

    css = """
<style>
.r{background:#000;color:#fff;font-family:Arial,Helvetica,sans-serif;padding:16px;border:1px solid #fff;width:100%;box-sizing:border-box}
.r *{background:#000;color:#fff;box-sizing:border-box}
.r h1{font-size:22px;border-bottom:2px solid #fff;padding-bottom:8px}
.r h2{font-size:16px;border-top:1px solid #fff;border-bottom:1px solid #fff;padding:5px 0;margin-top:22px}
.r h3{font-size:14px}.r p{font-size:13px;line-height:1.45}
.wrap{overflow-x:auto}.audit{border-collapse:collapse;min-width:100%;width:max-content;font-size:12px}
.audit th,.audit td{border:1px solid #fff;padding:7px 9px;white-space:nowrap}.audit th{font-weight:700;text-align:center}.audit td{text-align:right}
.audit td:first-child,.audit td:nth-child(2){text-align:left}.note{border:1px solid #fff;padding:9px;font-weight:600}.answer{font-size:18px;border:2px solid #fff;padding:12px;font-weight:700;text-align:center}.path{font-family:monospace;font-size:11px;overflow-wrap:anywhere;white-space:normal}
</style>
"""

    html: list[str] = [css, '<div class="r">']
    html.append("<h1>1769 Venus Transit — Final Halley Closure Certificate</h1>")
    html.append("<h2>CODE INPUTS</h2>")
    html.append(f'<p><b>Full-vector source:</b> {SOURCE_V0043}</p>')
    html.append(f'<p><b>Coefficient source:</b> {SOURCE_V0047}</p>')
    html.append(f'<p><b>Normal definition:</b> {NORMAL}</p>')

    html.append("<h2>COMMENTS</h2>")
    html.append('<p class="note">The physical JPL/TN36 distances are correct. No alternative physical distance set is hidden in A′B′.</p>')
    html.append('<p class="note">The classical factor EV/VS becomes exact only after applying the four full-screen geometric corrections.</p>')

    html.append("<h2>RESULTS</h2>")
    html.append("<h3>Physical distances</h3>")
    html.append(table(physical_frame, {"Kilometers": "{:,.12f}"}))
    html.append("<h3>A′B′ and AB closure</h3>")
    html.append(table(geometry_frame, {"Arcseconds": "{:.12f}", "Kilometers": "{:,.12f}"}))
    html.append("<h3>Classical and corrected coefficients</h3>")
    html.append(table(factor_frame, {"Value": "{:+.15f}"}))
    html.append("<h3>Where the 19.568706 km went</h3>")
    html.append(table(contribution_frame, {"Equivalent AB km": "{:+.12f}", "Percent of total": "{:+.9f}"}))
    html.append("<h3>Solar parallax π</h3>")
    html.append(table(pi_frame, {"π event arcsec": "{:.12f}", "π 1-AU arcsec": "{:.12f}"}))
    html.append(f'<p class="answer">π<sub>1 AU</sub> = {pi_1au_exact:.12f}″ → {pi_1au_exact:.6f}″</p>')

    html.append("<h2>OUTPUT SUMMARY</h2>")
    html.append(f'<p class="path">{CSV}</p>')
    html.append(f'<p class="path">{HTML_FILE}</p>')

    html.append("<h2>PAPER COMPARISON</h2>")
    html.append('<p class="note">The 19.568706 km discrepancy is not a distance or epoch error. It is the exact sum of finite-baseline, gnomonic-screen, separate-projector, and distance-triangle corrections omitted by the scalar Halley approximation.</p>')

    html.append("<h2>EQUATION STATUS</h2>")
    html.append(table(status_frame, {"Residual": "{:+.15e}"}))
    html.append("</div>")

    report = "".join(html)
    HTML_FILE.write_text(
        "<html><head><meta charset='utf-8'></head><body style='margin:0;background:#000;color:#fff'>"
        + report
        + "</body></html>",
        encoding="utf-8",
    )
    display(HTML(report))

    print(datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0048
