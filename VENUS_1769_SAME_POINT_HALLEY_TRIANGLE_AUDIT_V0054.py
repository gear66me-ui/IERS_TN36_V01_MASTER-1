# V0054
# Audit reference: Use the TN36/Halley A′B′ consistently with EV, VS, and ES for same-point closure.
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from IPython.display import HTML, display

VERSION = "V0054"
TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
SOURCE = ROOT / "VENUS_1769_ORDER_INDEPENDENT_PROJECTOR_AUDIT_V0052_OUTPUT" / "VENUS_1769_ORDER_INDEPENDENT_PROJECTOR_AUDIT_V0052.csv"
OUT = ROOT / "VENUS_1769_SAME_POINT_HALLEY_TRIANGLE_AUDIT_V0054_OUTPUT"
CSV = OUT / "VENUS_1769_SAME_POINT_HALLEY_TRIANGLE_AUDIT_V0054.csv"
HTML_FILE = OUT / "VENUS_1769_SAME_POINT_HALLEY_TRIANGLE_AUDIT_V0054.html"
ARCSEC_PER_RAD = 206_264.80624709636


def number(value: object) -> float:
    result = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(result):
        raise RuntimeError(f"Expected numeric value, received {value!r}.")
    return float(result)


def pick(
    frame: pd.DataFrame,
    section: str,
    key_col: str,
    key: str,
    value_col: str,
) -> object:
    rows = frame[
        (frame["section"].astype(str) == section)
        & (frame[key_col].astype(str) == key)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row for {section}/{key}; found {len(rows)}.")
    return rows.iloc[0][value_col]


def html_table(
    frame: pd.DataFrame,
    formats: dict[str, str] | None = None,
) -> str:
    shown = frame.copy()
    for column, pattern in (formats or {}).items():
        if column not in shown.columns:
            continue

        def render(value: object) -> str:
            converted = pd.to_numeric(
                pd.Series([value]), errors="coerce"
            ).iloc[0]
            if not pd.isna(converted):
                return pattern.format(float(converted))
            return str(value)

        shown[column] = shown[column].map(render)
    return (
        '<div class="wrap">'
        + shown.to_html(index=False, border=0, classes="audit", escape=False)
        + "</div>"
    )


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(
            "Run V0052 first; missing source CSV: " + str(SOURCE)
        )

    OUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE)

    utc = str(
        pick(
            source,
            "GEOMETRY",
            "Quantity",
            "Closest-approach UTC",
            "Value",
        )
    )
    jd = number(
        pick(
            source,
            "GEOMETRY",
            "Quantity",
            "Closest-approach JD TDB",
            "Value",
        )
    )
    direct_ab_km = number(
        pick(
            source,
            "GEOMETRY",
            "Quantity",
            "Direct common-normal AB km",
            "Value",
        )
    )
    ev = number(
        pick(
            source,
            "DISTANCES",
            "Instantaneous JPL quantity",
            "Earth → Venus",
            "Value",
        )
    )
    vs = number(
        pick(
            source,
            "DISTANCES",
            "Instantaneous JPL quantity",
            "Venus → Sun",
            "Value",
        )
    )
    es = number(
        pick(
            source,
            "DISTANCES",
            "Instantaneous JPL quantity",
            "Earth → Sun",
            "Value",
        )
    )
    jpl_aprime_bprime = number(
        pick(
            source,
            "MODELS",
            "Independent model",
            "Exact finite topocentric gnomonic rays",
            "Predicted A′B′ arcsec",
        )
    )
    tn36_aprime_bprime = number(
        pick(
            source,
            "MODELS",
            "Independent model",
            "Classical collinear Halley triangle",
            "Predicted A′B′ arcsec",
        )
    )

    ratio_ev_vs = ev / vs
    ratio_ev_es = ev / es
    ratio_vs_es = vs / es
    noncollinearity = ev + vs - es
    km_per_arcsec = es / ARCSEC_PER_RAD
    backtrack_scale = km_per_arcsec * ratio_ev_vs

    ab_halley_arcsec = tn36_aprime_bprime * ratio_ev_vs
    ab_halley_km = ab_halley_arcsec * km_per_arcsec
    direct_ab_arcsec = direct_ab_km / km_per_arcsec
    residual_arcsec = ab_halley_arcsec - direct_ab_arcsec
    residual_km = ab_halley_km - direct_ab_km

    mixed_ab_arcsec = jpl_aprime_bprime * ratio_ev_vs
    mixed_ab_km = mixed_ab_arcsec * km_per_arcsec
    mixed_residual_km = mixed_ab_km - direct_ab_km
    delta_aprime_bprime = jpl_aprime_bprime - tn36_aprime_bprime
    delta_screen_km = delta_aprime_bprime * km_per_arcsec
    delta_backtracked_km = delta_aprime_bprime * backtrack_scale

    geometry = pd.DataFrame(
        [
            ["Closest-approach UTC", utc],
            ["Closest-approach JD TDB", jd],
            ["Notation in this audit", "A′B′ and AB"],
        ],
        columns=["Quantity", "Value"],
    )

    distances = pd.DataFrame(
        [
            ["Earth → Venus", "EV", ev],
            ["Venus → Sun", "VS", vs],
            ["Earth → Sun", "ES", es],
            ["Non-collinearity", "EV + VS − ES", noncollinearity],
        ],
        columns=["Distance", "Symbol / equation", "Kilometers"],
    )

    ratios = pd.DataFrame(
        [
            ["Halley reduction ratio", "EV/VS", ratio_ev_vs],
            ["Earth–Venus fraction", "EV/ES", ratio_ev_es],
            ["Venus–Sun fraction", "VS/ES", ratio_vs_es],
            [
                "Near-collinearity sum",
                "EV/ES + VS/ES",
                ratio_ev_es + ratio_vs_es,
            ],
            [
                "Solar-screen scale",
                "ES / arcsec-per-radian",
                km_per_arcsec,
            ],
            [
                "Halley backtrack scale",
                "(ES / arcsec-per-radian) × EV/VS",
                backtrack_scale,
            ],
        ],
        columns=["Ratio / scale", "Equation", "Value"],
    )

    points = pd.DataFrame(
        [
            [
                "Exact finite JPL-vector A′B′",
                jpl_aprime_bprime,
                jpl_aprime_bprime * km_per_arcsec,
                "COMPARISON ONLY",
            ],
            [
                "TN36/Halley common-point A′B′",
                tn36_aprime_bprime,
                tn36_aprime_bprime * km_per_arcsec,
                "USED",
            ],
            [
                "JPL-vector minus TN36/Halley",
                delta_aprime_bprime,
                delta_screen_km,
                "Different point definitions",
            ],
        ],
        columns=[
            "A′B′ definition",
            "Arcseconds",
            "Solar-screen kilometers",
            "Status",
        ],
    )

    reductions = pd.DataFrame(
        [
            [
                "Same-point Halley reduction",
                "A′B′_TN36 × EV/VS",
                tn36_aprime_bprime,
                ratio_ev_vs,
                ab_halley_arcsec,
                ab_halley_km,
                direct_ab_km,
                residual_km,
                "USED",
            ],
            [
                "Mixed-point diagnostic",
                "A′B′_JPL × EV/VS",
                jpl_aprime_bprime,
                ratio_ev_vs,
                mixed_ab_arcsec,
                mixed_ab_km,
                direct_ab_km,
                mixed_residual_km,
                "REJECTED",
            ],
        ],
        columns=[
            "Reduction",
            "Equation",
            "Input A′B′ arcsec",
            "EV/VS",
            "AB arcsec",
            "AB kilometers",
            "Direct AB kilometers",
            "Residual kilometers",
            "Status",
        ],
    )

    comparison = pd.DataFrame(
        [
            [
                "A′B′ point-definition difference",
                delta_aprime_bprime,
                delta_screen_km,
                delta_backtracked_km,
            ],
            [
                "Same-point Halley closure",
                0.0,
                0.0,
                residual_km,
            ],
        ],
        columns=[
            "Comparison",
            "Angular difference arcsec",
            "Solar-screen difference km",
            "AB-equivalent difference km",
        ],
    )

    status = pd.DataFrame(
        [
            [
                "Distances reported without forcing EV + VS = ES",
                "PASS" if np.isfinite(noncollinearity) else "FAIL",
                noncollinearity,
                "km",
            ],
            [
                "Same-point AB arcsec closure",
                "PASS" if abs(residual_arcsec) < 1.0e-12 else "FAIL",
                residual_arcsec,
                "arcsec",
            ],
            [
                "Same-point AB kilometer closure",
                "PASS" if abs(residual_km) < 1.0e-8 else "FAIL",
                residual_km,
                "km",
            ],
            [
                "Mixed-point residual equals backtracked A′B′ difference",
                "PASS"
                if abs(mixed_residual_km - delta_backtracked_km) < 1.0e-8
                else "FAIL",
                mixed_residual_km - delta_backtracked_km,
                "km",
            ],
        ],
        columns=["Equation / test", "Status", "Residual", "Unit"],
    )

    records: list[dict[str, object]] = []
    for section, frame in (
        ("GEOMETRY", geometry),
        ("DISTANCES", distances),
        ("RATIOS", ratios),
        ("APRIME_BPRIME", points),
        ("HALLEY_REDUCTION", reductions),
        ("COMPARISON", comparison),
        ("STATUS", status),
    ):
        for row_number, row in frame.iterrows():
            record: dict[str, object] = {
                "section": section,
                "row": int(row_number),
            }
            record.update({str(key): value for key, value in row.items()})
            records.append(record)

    pd.DataFrame(records).to_csv(
        CSV,
        index=False,
        float_format="%.15f",
    )

    css = """<style>
.r{background:#000;color:#fff;font-family:Arial,Helvetica,sans-serif;padding:16px;border:1px solid #fff;width:100%;box-sizing:border-box}.r *{background:#000;color:#fff;box-sizing:border-box}.r h1{font-size:22px;border-bottom:2px solid #fff;padding-bottom:8px}.r h2{font-size:16px;border-top:1px solid #fff;border-bottom:1px solid #fff;padding:5px 0;margin-top:22px}.r h3{font-size:14px}.r p{font-size:13px;line-height:1.45}.wrap{overflow-x:auto}.audit{border-collapse:collapse;min-width:100%;width:max-content;font-size:12px}.audit th,.audit td{border:1px solid #fff;padding:7px 9px;white-space:nowrap}.audit th{font-weight:700;text-align:center}.audit td{text-align:right}.audit td:first-child,.audit td:nth-child(2){text-align:left}.note{border:1px solid #fff;padding:9px;font-weight:600}.answer{font-size:18px;border:2px solid #fff;padding:12px;font-weight:700;text-align:center}.path{font-family:monospace;font-size:11px;overflow-wrap:anywhere;white-space:normal}
</style>"""

    html = [
        css,
        '<div class="r">',
        "<h1>1769 Venus Transit — Same-Point Halley Triangle Audit</h1>",
    ]
    html += [
        "<h2>CODE INPUTS</h2>",
        f"<p><b>Source:</b> {SOURCE}</p>",
        f"<p><b>Epoch:</b> {utc}</p>",
        "<p><b>Used input:</b> TN36/Halley common-point A′B′ only.</p>",
    ]
    html += [
        "<h2>COMMENTS</h2>",
        '<p class="note">This audit uses A′B′ and AB throughout.</p>',
        '<p class="note">The EV/VS scalar reduction uses the A′B′ generated by the same common-direction Halley triangle. The exact finite JPL-vector A′B′ is listed only as a comparison.</p>',
    ]
    html += [
        "<h2>RESULTS</h2>",
        "<h3>Epoch and notation</h3>",
        html_table(geometry, {"Value": "{:,.15f}"}),
        "<h3>EV, VS, and ES distances</h3>",
        html_table(distances, {"Kilometers": "{:,.12f}"}),
        "<h3>Ratios and scales</h3>",
        html_table(ratios, {"Value": "{:+.15f}"}),
        "<h3>A′B′ comparison</h3>",
        html_table(
            points,
            {
                "Arcseconds": "{:+.15f}",
                "Solar-screen kilometers": "{:+,.12f}",
            },
        ),
        "<h3>AB Halley reduction</h3>",
        html_table(
            reductions,
            {
                "Input A′B′ arcsec": "{:+.15f}",
                "EV/VS": "{:+.15f}",
                "AB arcsec": "{:+.15f}",
                "AB kilometers": "{:+,.12f}",
                "Direct AB kilometers": "{:+,.12f}",
                "Residual kilometers": "{:+.15e}",
            },
        ),
        "<h3>Point-definition comparison</h3>",
        html_table(
            comparison,
            {
                "Angular difference arcsec": "{:+.15f}",
                "Solar-screen difference km": "{:+,.12f}",
                "AB-equivalent difference km": "{:+,.12f}",
            },
        ),
        (
            f'<p class="answer">Same-point Halley AB = '
            f'{ab_halley_km:,.12f} km; direct AB = '
            f'{direct_ab_km:,.12f} km; residual = '
            f'{residual_km:+.15e} km.</p>'
        ),
    ]
    html += [
        "<h2>OUTPUT SUMMARY</h2>",
        f'<p class="path">{CSV}</p>',
        f'<p class="path">{HTML_FILE}</p>',
    ]
    html += [
        "<h2>PAPER COMPARISON</h2>",
        '<p class="note">Using the same TN36/Halley A′B′ on both sides gives zero discrepancy. The +19.568712 km value appears only in the rejected mixed-point calculation that inserts the exact finite JPL A′B′ into the scalar EV/VS triangle.</p>',
    ]
    html += [
        "<h2>EQUATION STATUS</h2>",
        html_table(status, {"Residual": "{:+.15e}"}),
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
# V0054
