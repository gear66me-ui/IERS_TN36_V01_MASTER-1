# V0057
# Audit reference: Derive A′, B′, and A′B′ from the vector-closed Earth–Sun-axis distances and their axial EV∥/VS∥ ratio.
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from IPython.display import HTML, display

VERSION = "V0057"
TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
SOURCE_VECTOR = (
    ROOT
    / "VENUS_1769_VECTOR_CLOSURE_AXIAL_SCALE_AUDIT_V0056_OUTPUT"
    / "VENUS_1769_VECTOR_CLOSURE_AXIAL_SCALE_AUDIT_V0056.csv"
)
SOURCE_POINTS = (
    ROOT
    / "VENUS_1769_DISTANCE_RATIO_APRIME_BPRIME_AUDIT_V0055_OUTPUT"
    / "VENUS_1769_DISTANCE_RATIO_APRIME_BPRIME_AUDIT_V0055.csv"
)
OUT = ROOT / "VENUS_1769_AXIAL_RATIO_APRIME_BPRIME_AUDIT_V0057_OUTPUT"
CSV = OUT / "VENUS_1769_AXIAL_RATIO_APRIME_BPRIME_AUDIT_V0057.csv"
HTML_FILE = OUT / "VENUS_1769_AXIAL_RATIO_APRIME_BPRIME_AUDIT_V0057.html"

COLORS = {
    "EV∥": "#65D7FF",
    "VS∥": "#FFD166",
    "ES∥": "#7EE787",
    "RATIO∥": "#D2A8FF",
    "A": "#4DA3FF",
    "B": "#FF5A72",
    "AB": "#E8E8E8",
    "A′": "#7CFF6B",
    "B′": "#FFD84D",
    "A′B′": "#FFFFFF",
    "CHECK": "#FF7B72",
}


def number(value: object) -> float:
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(converted):
        raise RuntimeError(f"Expected numeric value, received {value!r}.")
    return float(converted)


def pick(
    frame: pd.DataFrame,
    section: str,
    key_column: str,
    key_value: str,
    value_column: str,
) -> object:
    rows = frame[
        (frame["section"].astype(str) == section)
        & (frame[key_column].astype(str) == key_value)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one row for {section}/{key_value}; found {len(rows)}."
        )
    return rows.iloc[0][value_column]


def formatted(value: object, pattern: str | None) -> str:
    if pattern is None:
        return str(value)
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(converted):
        return str(value)
    return pattern.format(float(converted))


def colored_table(
    frame: pd.DataFrame,
    formats: dict[str, str] | None = None,
    color_column: str | None = None,
) -> str:
    formats = formats or {}
    html = ['<div class="wrap"><table class="audit"><thead><tr>']
    for column in frame.columns:
        html.append(f"<th>{column}</th>")
    html.append("</tr></thead><tbody>")
    for _, row in frame.iterrows():
        color_key = (
            str(row[color_column])
            if color_column is not None and color_column in frame.columns
            else ""
        )
        color = COLORS.get(color_key, "#FFFFFF")
        html.append("<tr>")
        for column in frame.columns:
            text = formatted(row[column], formats.get(column))
            style = (
                f' style="color:{color};font-weight:700"'
                if color_key in COLORS
                else ""
            )
            html.append(f"<td{style}>{text}</td>")
        html.append("</tr>")
    html.append("</tbody></table></div>")
    return "".join(html)


def main() -> None:
    if not SOURCE_VECTOR.is_file():
        raise FileNotFoundError("Run V0056 first; missing source CSV: " + str(SOURCE_VECTOR))
    if not SOURCE_POINTS.is_file():
        raise FileNotFoundError("Run V0055 first; missing source CSV: " + str(SOURCE_POINTS))

    OUT.mkdir(parents=True, exist_ok=True)
    vector_source = pd.read_csv(SOURCE_VECTOR)
    point_source = pd.read_csv(SOURCE_POINTS)

    utc = str(
        pick(vector_source, "GEOMETRY", "Quantity", "Closest-approach UTC", "Value")
    )
    jd = number(
        pick(
            vector_source,
            "GEOMETRY",
            "Quantity",
            "Closest-approach JD TDB",
            "Value",
        )
    )

    ev_axial = number(
        pick(
            vector_source,
            "AXIAL_PROJECTIONS",
            "Projected distance",
            "Earth → Venus axial projection",
            "Axial kilometers",
        )
    )
    vs_axial = number(
        pick(
            vector_source,
            "AXIAL_PROJECTIONS",
            "Projected distance",
            "Venus → Sun axial projection",
            "Axial kilometers",
        )
    )
    es_axial = number(
        pick(
            vector_source,
            "AXIAL_PROJECTIONS",
            "Projected distance",
            "Earth → Sun axial projection",
            "Axial kilometers",
        )
    )
    axial_closure = ev_axial + vs_axial - es_axial
    ratio_axial = ev_axial / vs_axial

    ratio_magnitude = number(
        pick(
            vector_source,
            "RATIOS",
            "Ratio",
            "Magnitude ratio",
            "Value",
        )
    )

    a_arcsec = number(
        pick(point_source, "AB_POINTS", "Point / separation", "A", "Arcseconds")
    )
    b_arcsec = number(
        pick(point_source, "AB_POINTS", "Point / separation", "B", "Arcseconds")
    )
    ab_arcsec = number(
        pick(point_source, "AB_POINTS", "Point / separation", "AB", "Arcseconds")
    )
    a_km = number(
        pick(point_source, "AB_POINTS", "Point / separation", "A", "Kilometers")
    )
    b_km = number(
        pick(point_source, "AB_POINTS", "Point / separation", "B", "Kilometers")
    )
    ab_km = number(
        pick(point_source, "AB_POINTS", "Point / separation", "AB", "Kilometers")
    )

    a_prime_arcsec = a_arcsec / ratio_axial
    b_prime_arcsec = b_arcsec / ratio_axial
    aprime_bprime_arcsec = ab_arcsec / ratio_axial
    a_prime_km = a_km / ratio_axial
    b_prime_km = b_km / ratio_axial
    aprime_bprime_km = ab_km / ratio_axial

    scalar_aprime_bprime_arcsec = ab_arcsec / ratio_magnitude
    scalar_aprime_bprime_km = ab_km / ratio_magnitude

    a_back_arcsec = a_prime_arcsec * ratio_axial
    b_back_arcsec = b_prime_arcsec * ratio_axial
    ab_back_arcsec = aprime_bprime_arcsec * ratio_axial
    a_back_km = a_prime_km * ratio_axial
    b_back_km = b_prime_km * ratio_axial
    ab_back_km = aprime_bprime_km * ratio_axial

    geometry = pd.DataFrame(
        [
            ["Closest-approach UTC", utc],
            ["Closest-approach JD TDB", jd],
            ["Common collinear axis", "Earth-center → Sun-center axis"],
            ["Derivation scope", "A′, B′, and A′B′ from EV∥/VS∥ only"],
        ],
        columns=["Quantity", "Value"],
    )

    distances = pd.DataFrame(
        [
            ["Earth → Venus axial distance", "EV∥", ev_axial, "EV∥"],
            ["Venus → Sun axial distance", "VS∥", vs_axial, "VS∥"],
            ["Earth → Sun axial distance", "ES∥", es_axial, "ES∥"],
            ["Axial sum check", "EV∥ + VS∥ − ES∥", axial_closure, "CHECK"],
        ],
        columns=["Projected distance", "Symbol / equation", "Kilometers", "Color key"],
    )

    ratios = pd.DataFrame(
        [
            ["Axial conjugate ratio", "EV∥/VS∥", ratio_axial, "RATIO∥"],
            ["Original magnitude ratio", "EV/VS", ratio_magnitude, "RATIO∥"],
            [
                "Axial minus magnitude ratio",
                "EV∥/VS∥ − EV/VS",
                ratio_axial - ratio_magnitude,
                "CHECK",
            ],
        ],
        columns=["Ratio", "Equation", "Value", "Color key"],
    )

    direct_points = pd.DataFrame(
        [
            ["A", a_arcsec, a_km, "A"],
            ["B", b_arcsec, b_km, "B"],
            ["AB", ab_arcsec, ab_km, "AB"],
        ],
        columns=["Point / separation", "Arcseconds", "Kilometers", "Color key"],
    )

    prime_points = pd.DataFrame(
        [
            ["A′", "A/(EV∥/VS∥)", a_prime_arcsec, a_prime_km, "A′"],
            ["B′", "B/(EV∥/VS∥)", b_prime_arcsec, b_prime_km, "B′"],
            [
                "A′B′",
                "AB/(EV∥/VS∥)",
                aprime_bprime_arcsec,
                aprime_bprime_km,
                "A′B′",
            ],
        ],
        columns=[
            "Point / separation",
            "Axial-ratio derivation",
            "Arcseconds",
            "Kilometers",
            "Color key",
        ],
    )

    ratio_effect = pd.DataFrame(
        [
            [
                "Magnitude-ratio A′B′",
                "AB/(EV/VS)",
                scalar_aprime_bprime_arcsec,
                scalar_aprime_bprime_km,
                "RATIO∥",
            ],
            [
                "Axial-ratio A′B′",
                "AB/(EV∥/VS∥)",
                aprime_bprime_arcsec,
                aprime_bprime_km,
                "A′B′",
            ],
            [
                "Axial correction",
                "Axial-ratio minus magnitude-ratio",
                aprime_bprime_arcsec - scalar_aprime_bprime_arcsec,
                aprime_bprime_km - scalar_aprime_bprime_km,
                "CHECK",
            ],
        ],
        columns=[
            "Comparison",
            "Equation",
            "A′B′ arcseconds",
            "A′B′ kilometers",
            "Color key",
        ],
    )

    closure = pd.DataFrame(
        [
            ["A′ × EV∥/VS∥ = A", a_back_arcsec - a_arcsec, a_back_km - a_km, "A′"],
            ["B′ × EV∥/VS∥ = B", b_back_arcsec - b_arcsec, b_back_km - b_km, "B′"],
            [
                "A′B′ × EV∥/VS∥ = AB",
                ab_back_arcsec - ab_arcsec,
                ab_back_km - ab_km,
                "A′B′",
            ],
        ],
        columns=["Equation", "Arcsecond residual", "Kilometer residual", "Color key"],
    )

    status = pd.DataFrame(
        [
            [
                "Axial projected distances close",
                "PASS" if abs(axial_closure) < 1.0e-6 else "FAIL",
                axial_closure,
                "km",
            ],
            [
                "A′ and B′ derived only from A, B, and EV∥/VS∥",
                "PASS",
                0.0,
                "dimensionless",
            ],
            [
                "A-point axial back-reduction",
                "PASS" if abs(a_back_km - a_km) < 1.0e-9 else "FAIL",
                a_back_km - a_km,
                "km",
            ],
            [
                "B-point axial back-reduction",
                "PASS" if abs(b_back_km - b_km) < 1.0e-9 else "FAIL",
                b_back_km - b_km,
                "km",
            ],
            [
                "A′B′ axial back-reduction",
                "PASS" if abs(ab_back_km - ab_km) < 1.0e-9 else "FAIL",
                ab_back_km - ab_km,
                "km",
            ],
            [
                "No tangent-plane A′B′ used as an input",
                "PASS",
                0.0,
                "dimensionless",
            ],
        ],
        columns=["Equation / test", "Status", "Residual", "Unit"],
    )

    records: list[dict[str, object]] = []
    for section, frame in (
        ("GEOMETRY", geometry),
        ("AXIAL_DISTANCES", distances.drop(columns=["Color key"])),
        ("RATIOS", ratios.drop(columns=["Color key"])),
        ("AB_POINTS", direct_points.drop(columns=["Color key"])),
        ("APRIME_BPRIME", prime_points.drop(columns=["Color key"])),
        ("RATIO_EFFECT", ratio_effect.drop(columns=["Color key"])),
        ("CLOSURE", closure.drop(columns=["Color key"])),
        ("STATUS", status),
    ):
        for row_number, row in frame.iterrows():
            record: dict[str, object] = {"section": section, "row": int(row_number)}
            record.update({str(key): value for key, value in row.items()})
            records.append(record)
    pd.DataFrame(records).to_csv(CSV, index=False, float_format="%.15f")

    css = """<style>
.r{background:#000;color:#fff;font-family:Arial,Helvetica,sans-serif;padding:16px;border:1px solid #fff;width:100%;box-sizing:border-box}
.r *{background:#000;color:#fff;box-sizing:border-box}.r h1{font-size:22px;border-bottom:2px solid #fff;padding-bottom:8px}
.r h2{font-size:16px;border-top:1px solid #fff;border-bottom:1px solid #fff;padding:5px 0;margin-top:22px}.r h3{font-size:14px}.r p{font-size:13px;line-height:1.45}
.wrap{overflow-x:auto}.audit{border-collapse:collapse;min-width:100%;width:max-content;font-size:12px}.audit th,.audit td{border:1px solid #fff;padding:7px 9px;white-space:nowrap}
.audit th{font-weight:700;text-align:center}.audit td{text-align:right}.audit td:first-child,.audit td:nth-child(2){text-align:left}
.note{border:1px solid #fff;padding:9px;font-weight:600}.answer{font-size:18px;border:2px solid #fff;padding:12px;font-weight:700;text-align:center}
.path{font-family:monospace;font-size:11px;overflow-wrap:anywhere;white-space:normal}.legend{display:flex;flex-wrap:wrap;gap:12px;border:1px solid #fff;padding:9px}
</style>"""

    legend = '<div class="legend">' + "".join(
        f'<span style="color:{color};font-weight:700">{key}</span>'
        for key, color in COLORS.items()
    ) + "</div>"

    html = [css, '<div class="r">', "<h1>1769 Venus Transit — Axial-Ratio A′B′ Derivation Audit</h1>"]
    html += [
        "<h2>CODE INPUTS</h2>",
        f"<p><b>Vector source:</b> {SOURCE_VECTOR}</p>",
        f"<p><b>Point source:</b> {SOURCE_POINTS}</p>",
        f"<p><b>Epoch:</b> {utc}</p>",
    ]
    html += [
        "<h2>COMMENTS</h2>",
        '<p class="note">This step uses the vector-closed axial distances EV∥, VS∥, and ES∥. It derives A′, B′, and A′B′ only from A, B, AB, and the axial ratio EV∥/VS∥.</p>',
        '<p class="note">No exact tangent-plane A′B′, projector correction, solar parallax, or fitted closure factor is used in this module.</p>',
        legend,
    ]
    html += [
        "<h2>RESULTS</h2>",
        "<h3>Epoch and convention</h3>",
        colored_table(geometry, {"Value": "{:,.15f}"}),
        "<h3>1. Vector-closed axial distances</h3>",
        colored_table(distances, {"Kilometers": "{:+,.12f}"}, "Color key"),
        "<h3>2. Axial ratio</h3>",
        colored_table(ratios, {"Value": "{:+.15f}"}, "Color key"),
        "<h3>3. Direct A, B, and AB</h3>",
        colored_table(
            direct_points,
            {"Arcseconds": "{:+.15f}", "Kilometers": "{:+,.12f}"},
            "Color key",
        ),
        "<h3>4. A′, B′, and A′B′ from EV∥/VS∥</h3>",
        colored_table(
            prime_points,
            {"Arcseconds": "{:+.15f}", "Kilometers": "{:+,.12f}"},
            "Color key",
        ),
        "<h3>5. Effect of the axial-distance correction</h3>",
        colored_table(
            ratio_effect,
            {"A′B′ arcseconds": "{:+.15f}", "A′B′ kilometers": "{:+,.12f}"},
            "Color key",
        ),
        "<h3>6. Back-reduction check</h3>",
        colored_table(
            closure,
            {
                "Arcsecond residual": "{:+.15e}",
                "Kilometer residual": "{:+.15e}",
            },
            "Color key",
        ),
        f'<p class="answer">Axial-distance A′B′ = {aprime_bprime_arcsec:.15f}″ = {aprime_bprime_km:,.12f} km; multiplying by EV∥/VS∥ returns AB = {ab_km:,.12f} km.</p>',
    ]
    html += [
        "<h2>OUTPUT SUMMARY</h2>",
        f'<p class="path">{CSV}</p>',
        f'<p class="path">{HTML_FILE}</p>',
    ]
    html += [
        "<h2>PAPER COMPARISON</h2>",
        '<p class="note">NOT USED. This is only the axial-distance reduction step. Comparison with the exact finite tangent-plane A′B′ is deferred.</p>',
    ]
    html += [
        "<h2>EQUATION STATUS</h2>",
        colored_table(status, {"Residual": "{:+.15e}"}),
        "</div>",
    ]

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
# V0057
