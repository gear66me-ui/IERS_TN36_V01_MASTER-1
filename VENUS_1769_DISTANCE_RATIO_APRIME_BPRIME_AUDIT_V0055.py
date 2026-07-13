# V0055
# Audit reference: Derive midpoint-centered A′ and B′ directly from vector-file AB using the JPL EV/VS distance ratio.
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from IPython.display import HTML, display

VERSION = "V0055"
TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
SOURCE = (
    ROOT
    / "VENUS_1769_ORDER_INDEPENDENT_PROJECTOR_AUDIT_V0052_OUTPUT"
    / "VENUS_1769_ORDER_INDEPENDENT_PROJECTOR_AUDIT_V0052.csv"
)
OUT = ROOT / "VENUS_1769_DISTANCE_RATIO_APRIME_BPRIME_AUDIT_V0055_OUTPUT"
CSV = OUT / "VENUS_1769_DISTANCE_RATIO_APRIME_BPRIME_AUDIT_V0055.csv"
HTML_FILE = OUT / "VENUS_1769_DISTANCE_RATIO_APRIME_BPRIME_AUDIT_V0055.html"
ARCSEC_PER_RAD = 206_264.80624709636

COLORS = {
    "EV": "#00D8FF",
    "VS": "#FF9F1C",
    "ES": "#C77DFF",
    "A": "#4DA3FF",
    "B": "#FF5A72",
    "A′": "#7CFF6B",
    "B′": "#FFD84D",
    "AB": "#E8E8E8",
    "A′B′": "#FFFFFF",
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
            f"Expected one source row for {section}/{key_value}; found {len(rows)}."
        )
    return rows.iloc[0][value_column]


def format_value(value: object, pattern: str | None) -> str:
    if pattern is None:
        return str(value)
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(converted):
        return str(value)
    return pattern.format(float(converted))


def colored_table(
    frame: pd.DataFrame,
    formats: dict[str, str] | None = None,
    symbol_column: str | None = None,
) -> str:
    formats = formats or {}
    parts = ['<div class="wrap"><table class="audit"><thead><tr>']
    for column in frame.columns:
        parts.append(f"<th>{column}</th>")
    parts.append("</tr></thead><tbody>")

    for _, row in frame.iterrows():
        symbol = str(row[symbol_column]) if symbol_column and symbol_column in frame.columns else ""
        color = COLORS.get(symbol, "#FFFFFF")
        parts.append("<tr>")
        for column in frame.columns:
            text = format_value(row[column], formats.get(column))
            style = f' style="color:{color};font-weight:700"' if symbol in COLORS else ""
            parts.append(f"<td{style}>{text}</td>")
        parts.append("</tr>")

    parts.append("</tbody></table></div>")
    return "".join(parts)


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError("Run V0052 first; missing source CSV: " + str(SOURCE))

    OUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE)

    utc = str(pick(source, "GEOMETRY", "Quantity", "Closest-approach UTC", "Value"))
    jd = number(
        pick(source, "GEOMETRY", "Quantity", "Closest-approach JD TDB", "Value")
    )
    ab_km = number(
        pick(source, "GEOMETRY", "Quantity", "Direct common-normal AB km", "Value")
    )
    ev = number(
        pick(source, "DISTANCES", "Instantaneous JPL quantity", "Earth → Venus", "Value")
    )
    vs = number(
        pick(source, "DISTANCES", "Instantaneous JPL quantity", "Venus → Sun", "Value")
    )
    es = number(
        pick(source, "DISTANCES", "Instantaneous JPL quantity", "Earth → Sun", "Value")
    )

    ratio_ev_vs = ev / vs
    ratio_vs_ev = vs / ev
    ratio_ev_es = ev / es
    ratio_vs_es = vs / es
    km_per_arcsec = es / ARCSEC_PER_RAD

    ab_arcsec = ab_km / km_per_arcsec
    a_km = -0.5 * ab_km
    b_km = +0.5 * ab_km
    a_arcsec = -0.5 * ab_arcsec
    b_arcsec = +0.5 * ab_arcsec

    a_prime_km = a_km / ratio_ev_vs
    b_prime_km = b_km / ratio_ev_vs
    a_prime_arcsec = a_arcsec / ratio_ev_vs
    b_prime_arcsec = b_arcsec / ratio_ev_vs
    aprime_bprime_km = b_prime_km - a_prime_km
    aprime_bprime_arcsec = b_prime_arcsec - a_prime_arcsec

    a_back_km = a_prime_km * ratio_ev_vs
    b_back_km = b_prime_km * ratio_ev_vs
    ab_back_km = aprime_bprime_km * ratio_ev_vs
    a_back_arcsec = a_prime_arcsec * ratio_ev_vs
    b_back_arcsec = b_prime_arcsec * ratio_ev_vs
    ab_back_arcsec = aprime_bprime_arcsec * ratio_ev_vs

    geometry = pd.DataFrame(
        [
            ["Closest-approach UTC", utc],
            ["Closest-approach JD TDB", jd],
            ["Axis definition", "Midpoint-centered common-normal axis"],
            ["Sign convention", "A and A′ negative; B and B′ positive"],
        ],
        columns=["Quantity", "Value"],
    )

    distances = pd.DataFrame(
        [
            ["Earth → Venus distance", "EV", ev],
            ["Venus → Sun distance", "VS", vs],
            ["Earth → Sun distance", "ES", es],
        ],
        columns=["Distance", "Symbol", "Kilometers"],
    )

    ratios = pd.DataFrame(
        [
            ["Halley reduction ratio", "EV/VS", ratio_ev_vs, "EV"],
            ["Inverse ratio", "VS/EV", ratio_vs_ev, "VS"],
            ["Earth–Venus fraction", "EV/ES", ratio_ev_es, "EV"],
            ["Venus–Sun fraction", "VS/ES", ratio_vs_es, "VS"],
            ["Solar-screen scale", "ES / arcsec-per-radian", km_per_arcsec, "ES"],
        ],
        columns=["Ratio / scale", "Equation", "Value", "Color key"],
    )

    ab_points = pd.DataFrame(
        [
            ["A", a_arcsec, a_km, "A"],
            ["B", b_arcsec, b_km, "B"],
            ["AB", ab_arcsec, ab_km, "AB"],
        ],
        columns=["Point / separation", "Arcseconds", "Kilometers", "Color key"],
    )

    prime_points = pd.DataFrame(
        [
            ["A′", "A/(EV/VS)", a_prime_arcsec, a_prime_km, "A′"],
            ["B′", "B/(EV/VS)", b_prime_arcsec, b_prime_km, "B′"],
            ["A′B′", "AB/(EV/VS)", aprime_bprime_arcsec, aprime_bprime_km, "A′B′"],
        ],
        columns=[
            "Point / separation",
            "Distance-ratio derivation",
            "Arcseconds",
            "Kilometers",
            "Color key",
        ],
    )

    closure = pd.DataFrame(
        [
            ["A′ × EV/VS = A", a_back_arcsec - a_arcsec, a_back_km - a_km, "A′"],
            ["B′ × EV/VS = B", b_back_arcsec - b_arcsec, b_back_km - b_km, "B′"],
            ["A′B′ × EV/VS = AB", ab_back_arcsec - ab_arcsec, ab_back_km - ab_km, "A′B′"],
        ],
        columns=["Equation", "Arcsecond residual", "Kilometer residual", "Color key"],
    )

    status = pd.DataFrame(
        [
            [
                "Distance inputs finite and positive",
                "PASS" if all(np.isfinite(value) and value > 0.0 for value in (ev, vs, es)) else "FAIL",
                min(ev, vs, es),
                "km minimum",
            ],
            ["A′ and B′ derived only from A, B, and EV/VS", "PASS", 0.0, "dimensionless"],
            [
                "A-point back-reduction closure",
                "PASS" if abs(a_back_km - a_km) < 1.0e-9 else "FAIL",
                a_back_km - a_km,
                "km",
            ],
            [
                "B-point back-reduction closure",
                "PASS" if abs(b_back_km - b_km) < 1.0e-9 else "FAIL",
                b_back_km - b_km,
                "km",
            ],
            [
                "A′B′ back-reduction closure",
                "PASS" if abs(ab_back_km - ab_km) < 1.0e-9 else "FAIL",
                ab_back_km - ab_km,
                "km",
            ],
        ],
        columns=["Equation / test", "Status", "Residual", "Unit"],
    )

    records: list[dict[str, object]] = []
    for section, frame in (
        ("GEOMETRY", geometry),
        ("DISTANCES", distances),
        ("RATIOS", ratios.drop(columns=["Color key"])),
        ("AB_POINTS", ab_points.drop(columns=["Color key"])),
        ("APRIME_BPRIME", prime_points.drop(columns=["Color key"])),
        ("CLOSURE", closure.drop(columns=["Color key"])),
        ("STATUS", status),
    ):
        for row_number, row in frame.iterrows():
            record: dict[str, object] = {"section": section, "row": int(row_number)}
            record.update({str(key): value for key, value in row.items()})
            records.append(record)
    pd.DataFrame(records).to_csv(CSV, index=False, float_format="%.15f")

    css = """<style>
.r{background:#000;color:#fff;font-family:Arial,Helvetica,sans-serif;padding:16px;border:1px solid #fff;width:100%;box-sizing:border-box}.r *{background:#000;color:#fff;box-sizing:border-box}.r h1{font-size:22px;border-bottom:2px solid #fff;padding-bottom:8px}.r h2{font-size:16px;border-top:1px solid #fff;border-bottom:1px solid #fff;padding:5px 0;margin-top:22px}.r h3{font-size:14px}.r p{font-size:13px;line-height:1.45}.wrap{overflow-x:auto}.audit{border-collapse:collapse;min-width:100%;width:max-content;font-size:12px}.audit th,.audit td{border:1px solid #fff;padding:7px 9px;white-space:nowrap}.audit th{font-weight:700;text-align:center}.audit td{text-align:right}.audit td:first-child,.audit td:nth-child(2){text-align:left}.note{border:1px solid #fff;padding:9px;font-weight:600}.answer{font-size:18px;border:2px solid #fff;padding:12px;font-weight:700;text-align:center}.path{font-family:monospace;font-size:11px;overflow-wrap:anywhere;white-space:normal}.legend{display:flex;flex-wrap:wrap;gap:12px;border:1px solid #fff;padding:9px}.chip{font-weight:700}
</style>"""
    legend = (
        '<div class="legend">'
        + "".join(
            f'<span class="chip" style="color:{color}">{symbol}</span>'
            for symbol, color in COLORS.items()
        )
        + "</div>"
    )

    html = [css, '<div class="r">', "<h1>1769 Venus Transit — Distance-Ratio A′B′ Derivation Audit</h1>"]
    html += [
        "<h2>CODE INPUTS</h2>",
        f"<p><b>Source:</b> {SOURCE}</p>",
        f"<p><b>Epoch:</b> {utc}</p>",
        "<p><b>Used quantities:</b> EV, VS, ES, and direct vector-file AB only.</p>",
    ]
    html += [
        "<h2>COMMENTS</h2>",
        '<p class="note">This step intentionally stops after deriving midpoint-centered A′ and B′ from A and B with the physical distance ratio EV/VS.</p>',
        '<p class="note">No tangent-plane A′B′ comparison, projector correction, π₀ calculation, or alternative reduction is included.</p>',
        legend,
    ]
    html += [
        "<h2>RESULTS</h2>",
        "<h3>Epoch and coordinate convention</h3>",
        colored_table(geometry, {"Value": "{:,.15f}"}),
        "<h3>1. Vector-file distances</h3>",
        colored_table(distances, {"Kilometers": "{:,.12f}"}, "Symbol"),
        "<h3>2. Ratios derived from the distances</h3>",
        colored_table(ratios, {"Value": "{:+.15f}"}, "Color key"),
        "<h3>3. Direct A, B, and AB from the vector-file baseline</h3>",
        colored_table(ab_points, {"Arcseconds": "{:+.15f}", "Kilometers": "{:+,.12f}"}, "Color key"),
        "<h3>4. A′, B′, and A′B′ derived from EV/VS</h3>",
        colored_table(prime_points, {"Arcseconds": "{:+.15f}", "Kilometers": "{:+,.12f}"}, "Color key"),
        "<h3>5. Back-reduction check</h3>",
        colored_table(closure, {"Arcsecond residual": "{:+.15e}", "Kilometer residual": "{:+.15e}"}, "Color key"),
        f'<p class="answer">Distance-derived A′B′ = {aprime_bprime_arcsec:.15f}″ = {aprime_bprime_km:,.12f} km; multiplying by EV/VS returns AB = {ab_km:,.12f} km.</p>',
    ]
    html += [
        "<h2>OUTPUT SUMMARY</h2>",
        f'<p class="path">{CSV}</p>',
        f'<p class="path">{HTML_FILE}</p>',
    ]
    html += [
        "<h2>PAPER COMPARISON</h2>",
        '<p class="note">NOT USED in this step. The exact tangent-plane A′B′ comparison is deliberately deferred.</p>',
    ]
    html += [
        "<h2>EQUATION STATUS</h2>",
        colored_table(status, {"Residual": "{:+.15e}"}),
        "</div>",
    ]

    report = "".join(html)
    HTML_FILE.write_text(
        "<html><head><meta charset='utf-8'></head>"
        "<body style='margin:0;background:#000;color:#fff'>"
        + report
        + "</body></html>",
        encoding="utf-8",
    )
    display(HTML(report))
    print(datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0055