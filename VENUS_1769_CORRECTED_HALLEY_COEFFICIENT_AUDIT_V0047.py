# V0047
# Audit reference: Derive the exact screen coefficient as the classical Halley ratio times geometric correction factors.
from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from IPython.display import HTML, display

VERSION = "V0047"
TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
SOURCE_CSV = ROOT / "VENUS_1769_FULL_VECTOR_P_REDUCTION_AUDIT_V0043_OUTPUT" / "VENUS_1769_FULL_VECTOR_P_REDUCTION_AUDIT_V0043.csv"
OUT = ROOT / "VENUS_1769_CORRECTED_HALLEY_COEFFICIENT_AUDIT_V0047_OUTPUT"
CSV = OUT / "VENUS_1769_CORRECTED_HALLEY_COEFFICIENT_AUDIT_V0047.csv"
HTML_FILE = OUT / "VENUS_1769_CORRECTED_HALLEY_COEFFICIENT_AUDIT_V0047.html"


def numeric(value: object) -> float:
    result = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(result):
        raise RuntimeError(f"Expected numeric value, received {value!r}.")
    return float(result)


def row_value(frame: pd.DataFrame, key_column: str, key: str, value_column: str) -> float:
    rows = frame[frame[key_column].astype(str) == key]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row for {key!r}; found {len(rows)}.")
    return numeric(rows.iloc[0][value_column])


def table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    shown = frame.copy()
    for column, pattern in (formats or {}).items():
        if column not in shown.columns:
            continue

        def formatter(value: object) -> str:
            if pd.isna(value):
                return ""
            if isinstance(value, (int, float, np.integer, np.floating)):
                return pattern.format(float(value))
            converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if not pd.isna(converted):
                return pattern.format(float(converted))
            return str(value)

        shown[column] = shown[column].map(formatter)
    return '<div class="wrap">' + shown.to_html(index=False, border=0, classes="audit", escape=False) + "</div>"


def main() -> None:
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(
            "V0043 source CSV is missing. Run V0043 in this Colab session before V0047: "
            + str(SOURCE_CSV)
        )

    OUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE_CSV)

    epoch = source[source["section"] == "EPOCH"].copy()
    ladder = source[source["section"] == "LADDER"].copy()
    baselines = source[source["section"] == "BASELINES"].copy()

    ev = row_value(epoch, "Quantity", "Earth → Venus EV km", "Value")
    vs = row_value(epoch, "Quantity", "Venus → Sun VS km", "Value")
    es = row_value(epoch, "Quantity", "Earth → Sun ES km", "Value")
    km_per_arcsec = row_value(epoch, "Quantity", "Earth–Sun scale km/arcsec", "Value")
    physical_halley = ev / vs

    ladder = ladder[ladder["Station model"] == "JPL topocentric-derived"].copy()
    baselines = baselines[baselines["Station model"] == "JPL topocentric-derived"].copy()
    if len(ladder) != 2 or len(baselines) != 2:
        raise RuntimeError("Expected two JPL normal-definition rows from V0043.")

    coefficient_rows: list[list[object]] = []
    correction_rows: list[list[object]] = []
    effective_rows: list[list[object]] = []
    status_rows: list[list[object]] = []

    for _, ladder_row in ladder.iterrows():
        normal = str(ladder_row["Normal definition"])
        baseline_row = baselines[baselines["Normal definition"] == normal]
        if len(baseline_row) != 1:
            raise RuntimeError(f"Missing baseline row for {normal}.")
        baseline_row = baseline_row.iloc[0]

        ab_km = numeric(baseline_row["Direct common-normal AB km"])
        ab_arcsec = ab_km / km_per_arcsec
        exact = numeric(ladder_row["Exact nonlinear JPL arcsec"])
        gnomonic = numeric(ladder_row["Gnomonic first-order arcsec"])
        separate = numeric(ladder_row["Separate-projector arcsec"])
        common = numeric(ladder_row["Common-projector arcsec"])
        triangle = numeric(ladder_row["Halley-triangle arcsec"])

        k_exact = ab_arcsec / exact
        k_gnomonic = ab_arcsec / gnomonic
        k_separate = ab_arcsec / separate
        k_common = ab_arcsec / common
        k_triangle = ab_arcsec / triangle

        c_finite = gnomonic / exact
        c_gnomonic = separate / gnomonic
        c_projector = common / separate
        c_triangle = triangle / common
        c_total = triangle / exact
        c_product = c_finite * c_gnomonic * c_projector * c_triangle
        k_reconstructed = physical_halley * c_product

        ev_effective = es * k_exact / (1.0 + k_exact)
        vs_effective = es / (1.0 + k_exact)

        coefficient_rows.extend(
            [
                [normal, "Classical physical-distance factor", "EV/VS", physical_halley],
                [normal, "Exact nonlinear screen factor", "AB/A′B′ exact", k_exact],
                [normal, "Gnomonic first-order factor", "AB/A′B′ gnomonic", k_gnomonic],
                [normal, "Separate-projector factor", "AB/A′B′ separate", k_separate],
                [normal, "Common-projector factor", "AB/A′B′ common", k_common],
                [normal, "Halley-triangle factor", "AB/A′B′ triangle", k_triangle],
                [normal, "Reconstructed exact factor", "(EV/VS) × C_total", k_reconstructed],
            ]
        )

        correction_rows.extend(
            [
                [normal, "Finite-baseline correction", "A′B′_gnomonic / A′B′_exact", c_finite],
                [normal, "Gnomonic-screen correction", "A′B′_separate / A′B′_gnomonic", c_gnomonic],
                [normal, "Separate-projector correction", "A′B′_common / A′B′_separate", c_projector],
                [normal, "Distance-triangle correction", "A′B′_triangle / A′B′_common", c_triangle],
                [normal, "Total geometric correction", "A′B′_triangle / A′B′_exact", c_total],
                [normal, "Product of four corrections", "C1 × C2 × C3 × C4", c_product],
            ]
        )

        effective_rows.extend(
            [
                [normal, "Physical Earth → Venus", "EV", ev, "physical pointing distance"],
                [normal, "Physical Venus → Sun", "VS", vs, "physical pointing distance"],
                [normal, "Scalar-equivalent Earth → Venus", "EV_eff", ev_effective, "NOT PHYSICAL — encodes screen projection"],
                [normal, "Scalar-equivalent Venus → Sun", "VS_eff", vs_effective, "NOT PHYSICAL — encodes screen projection"],
                [normal, "Earth → Sun", "ES", es, "common sum constraint"],
            ]
        )

        status_rows.extend(
            [
                [normal, "Halley triangle factor equals EV/VS", "PASS" if abs(k_triangle - physical_halley) < 1e-12 else "FAIL", k_triangle - physical_halley],
                [normal, "Correction-product identity", "PASS" if abs(c_product - c_total) < 1e-14 else "FAIL", c_product - c_total],
                [normal, "Corrected factor equals exact factor", "PASS" if abs(k_reconstructed - k_exact) < 1e-12 else "FAIL", k_reconstructed - k_exact],
                [normal, "Effective-distance sum", "PASS" if abs((ev_effective + vs_effective) - es) < 1e-6 else "FAIL", ev_effective + vs_effective - es],
            ]
        )

    coefficient_frame = pd.DataFrame(
        coefficient_rows,
        columns=["Normal definition", "Coefficient", "Equation", "Value"],
    )
    correction_frame = pd.DataFrame(
        correction_rows,
        columns=["Normal definition", "Correction", "Equation", "Multiplier"],
    )
    effective_frame = pd.DataFrame(
        effective_rows,
        columns=["Normal definition", "Distance", "Symbol", "Kilometers", "Status"],
    )
    status_frame = pd.DataFrame(
        status_rows,
        columns=["Normal definition", "Equation / test", "Status", "Residual"],
    )

    records: list[dict[str, object]] = []
    for section, frame in (
        ("COEFFICIENTS", coefficient_frame),
        ("CORRECTIONS", correction_frame),
        ("DISTANCES", effective_frame),
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
.audit td:first-child,.audit td:nth-child(2){text-align:left}.note{border:1px solid #fff;padding:9px;font-weight:600}.path{font-family:monospace;font-size:11px;overflow-wrap:anywhere;white-space:normal}
</style>
"""

    html: list[str] = [css, '<div class="r">']
    html.append("<h1>1769 Venus Transit — Corrected Halley Coefficient Audit</h1>")
    html.append("<h2>CODE INPUTS</h2>")
    html.append(f'<p><b>Source:</b> {SOURCE_CSV}</p>')
    html.append(f'<p><b>Physical EV:</b> {ev:,.12f} km</p>')
    html.append(f'<p><b>Physical VS:</b> {vs:,.12f} km</p>')
    html.append(f'<p><b>Physical ES:</b> {es:,.12f} km</p>')

    html.append("<h2>COMMENTS</h2>")
    html.append('<p class="note">The pointing reconstruction in V0045 and V0046 recovered the physical JPL distances. Therefore there is no second set of physical TN36 distances hidden in A′B′.</p>')
    html.append('<p class="note">The exact scalar coefficient is obtained by multiplying the classical physical ratio EV/VS by four independently identified geometric correction multipliers.</p>')
    html.append('<p class="note">EV_eff and VS_eff are scalar-equivalent distances only. They encode the full screen geometry and must not be described as physical ranges.</p>')

    html.append("<h2>RESULTS</h2>")
    html.append("<h3>Classical and corrected coefficients</h3>")
    html.append(table(coefficient_frame, {"Value": "{:+.15f}"}))
    html.append("<h3>Multiplicative correction chain</h3>")
    html.append(table(correction_frame, {"Multiplier": "{:+.15f}"}))
    html.append("<h3>Physical versus scalar-equivalent distances</h3>")
    html.append(table(effective_frame, {"Kilometers": "{:,.12f}"}))

    html.append("<h2>OUTPUT SUMMARY</h2>")
    html.append(f'<p class="path">{CSV}</p>')
    html.append(f'<p class="path">{HTML_FILE}</p>')

    html.append("<h2>PAPER COMPARISON</h2>")
    html.append('<p class="note">The classical factor EV/VS is physically correct for the collinear Halley triangle. The exact JPL/IERS factor differs because the observed A′B′ includes finite-baseline, gnomonic, and separate Sun/Venus projector terms.</p>')

    html.append("<h2>EQUATION STATUS</h2>")
    html.append(table(status_frame, {"Residual": "{:+.15e}"}))
    html.append("</div>")

    report = "".join(html)
    HTML_FILE.write_text(
        "<html><head><meta charset='utf-8'></head><body style='margin:0;background:#000;color:#fff'>" + report + "</body></html>",
        encoding="utf-8",
    )
    display(HTML(report))

    print(datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0047
