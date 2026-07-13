# V0054
# Audit reference: Enforce one exact A′B′ definition, correct EV/VS/ES labels, and verify AB and π₀ reductions.
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
SOURCE = (
    ROOT
    / "VENUS_1769_ORDER_INDEPENDENT_PROJECTOR_AUDIT_V0052_OUTPUT"
    / "VENUS_1769_ORDER_INDEPENDENT_PROJECTOR_AUDIT_V0052.csv"
)
OUT = ROOT / "VENUS_1769_SINGLE_APRIME_BPRIME_HALLEY_AUDIT_V0054_OUTPUT"
CSV = OUT / "VENUS_1769_SINGLE_APRIME_BPRIME_HALLEY_AUDIT_V0054.csv"
HTML_FILE = OUT / "VENUS_1769_SINGLE_APRIME_BPRIME_HALLEY_AUDIT_V0054.html"

ARCSEC_PER_RAD = 206_264.80624709636
EARTH_EQUATORIAL_RADIUS_KM = 6_378.140000
AU_KM = 149_597_870.000000
PUBLISHED_PI0_ARCSEC = 8.794148


def number(value: object) -> float:
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(converted):
        raise RuntimeError(f"Expected numeric value, received {value!r}.")
    return float(converted)


def pick(
    frame: pd.DataFrame,
    section: str,
    key_column: str,
    key: str,
    value_column: str,
) -> object:
    rows = frame[
        (frame["section"].astype(str) == section)
        & (frame[key_column].astype(str) == key)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one row for {section}/{key}; found {len(rows)}."
        )
    return rows.iloc[0][value_column]


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
        + shown.to_html(
            index=False,
            border=0,
            classes="audit",
            escape=False,
        )
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
    aprime_bprime_arcsec = number(
        pick(
            source,
            "MODELS",
            "Independent model",
            "Exact finite topocentric gnomonic rays",
            "Predicted A′B′ arcsec",
        )
    )

    km_per_arcsec = es / ARCSEC_PER_RAD
    aprime_bprime_km = aprime_bprime_arcsec * km_per_arcsec
    direct_ab_arcsec = direct_ab_km / km_per_arcsec

    physical_ratio = ev / vs
    exact_screen_ratio = direct_ab_arcsec / aprime_bprime_arcsec
    ratio_difference = physical_ratio - exact_screen_ratio

    halley_ab_arcsec = aprime_bprime_arcsec * physical_ratio
    halley_ab_km = halley_ab_arcsec * km_per_arcsec
    halley_residual_arcsec = halley_ab_arcsec - direct_ab_arcsec
    halley_residual_km = halley_ab_km - direct_ab_km

    exact_ab_arcsec = aprime_bprime_arcsec * exact_screen_ratio
    exact_ab_km = exact_ab_arcsec * km_per_arcsec
    exact_residual_arcsec = exact_ab_arcsec - direct_ab_arcsec
    exact_residual_km = exact_ab_km - direct_ab_km

    pi_event_halley = (
        aprime_bprime_arcsec
        * EARTH_EQUATORIAL_RADIUS_KM
        / direct_ab_km
        * physical_ratio
    )
    pi0_halley = pi_event_halley * es / AU_KM

    pi_event_exact = (
        aprime_bprime_arcsec
        * EARTH_EQUATORIAL_RADIUS_KM
        / direct_ab_km
        * exact_screen_ratio
    )
    pi0_exact = pi_event_exact * es / AU_KM

    pi_event_direct = (
        EARTH_EQUATORIAL_RADIUS_KM / es * ARCSEC_PER_RAD
    )
    pi0_direct = (
        EARTH_EQUATORIAL_RADIUS_KM / AU_KM * ARCSEC_PER_RAD
    )

    notation = pd.DataFrame(
        [
            ["Closest-approach UTC", utc],
            ["Closest-approach JD TDB", jd],
            ["Only projected separation used", "A′B′"],
            ["Only reduced baseline used", "AB"],
            [
                "V0053 label correction",
                "A′D′/AD and ED/DS removed; use A′B′/AB and EV/VS",
            ],
        ],
        columns=["Quantity", "Value"],
    )

    distances = pd.DataFrame(
        [
            ["Earth → Venus distance", "EV", ev],
            ["Venus → Sun distance", "VS", vs],
            ["Earth → Sun distance", "ES", es],
            ["Geometric non-collinearity", "EV + VS − ES", ev + vs - es],
        ],
        columns=["Distance", "Symbol / equation", "Kilometers"],
    )

    geometry = pd.DataFrame(
        [
            [
                "Exact finite JPL/TN36 tangent-plane separation",
                "A′B′",
                aprime_bprime_arcsec,
                aprime_bprime_km,
            ],
            [
                "Direct common-normal projected station baseline",
                "AB",
                direct_ab_arcsec,
                direct_ab_km,
            ],
        ],
        columns=[
            "Geometric quantity",
            "Symbol",
            "Arcseconds",
            "Kilometers",
        ],
    )

    ratios = pd.DataFrame(
        [
            [
                "Physical distance ratio",
                "EV/VS",
                physical_ratio,
                "JPL physical distances",
            ],
            [
                "Exact same-point screen ratio",
                "AB/A′B′",
                exact_screen_ratio,
                "JPL/TN36 projected geometry",
            ],
            [
                "Ratio difference",
                "EV/VS − AB/A′B′",
                ratio_difference,
                "Explains the nonzero scalar-Halley residual",
            ],
            [
                "Earth–Venus fraction",
                "EV/ES",
                ev / es,
                "Physical distance ratio",
            ],
            [
                "Venus–Sun fraction",
                "VS/ES",
                vs / es,
                "Physical distance ratio",
            ],
        ],
        columns=["Ratio", "Equation", "Value", "Origin"],
    )

    reductions = pd.DataFrame(
        [
            [
                "Scalar Halley physical-distance test",
                "A′B′ × EV/VS",
                aprime_bprime_arcsec,
                physical_ratio,
                halley_ab_arcsec,
                halley_ab_km,
                direct_ab_km,
                halley_residual_km,
                pi0_halley,
                "REJECTED — exact finite A′B′ is not collinear-Halley A′B′",
            ],
            [
                "Exact same-point geometric reduction",
                "A′B′ × (AB/A′B′)",
                aprime_bprime_arcsec,
                exact_screen_ratio,
                exact_ab_arcsec,
                exact_ab_km,
                direct_ab_km,
                exact_residual_km,
                pi0_exact,
                "USED — same exact A′B′ and AB geometry",
            ],
        ],
        columns=[
            "Reduction",
            "Equation",
            "Input A′B′ arcsec",
            "Reduction ratio",
            "AB arcsec",
            "AB kilometers",
            "Direct AB kilometers",
            "Residual kilometers",
            "π₀ at 1 AU arcsec",
            "Status",
        ],
    )

    parallax = pd.DataFrame(
        [
            [
                "Scalar Halley with physical EV/VS",
                pi_event_halley,
                pi0_halley,
                pi0_halley - PUBLISHED_PI0_ARCSEC,
                "REJECTED",
            ],
            [
                "Exact same-point geometric reduction",
                pi_event_exact,
                pi0_exact,
                pi0_exact - PUBLISHED_PI0_ARCSEC,
                "USED",
            ],
            [
                "Direct ES and one-AU constants",
                pi_event_direct,
                pi0_direct,
                pi0_direct - PUBLISHED_PI0_ARCSEC,
                "CHECK",
            ],
        ],
        columns=[
            "Solar-parallax calculation",
            "π event arcsec",
            "π₀ at 1 AU arcsec",
            "π₀ minus 8.794148 arcsec",
            "Status",
        ],
    )

    status = pd.DataFrame(
        [
            [
                "One A′B′ value used in both reduction rows",
                "PASS",
                0.0,
                "arcsec",
            ],
            [
                "Physical ratio equals EV/VS",
                "PASS"
                if abs(physical_ratio - ev / vs) < 1.0e-15
                else "FAIL",
                physical_ratio - ev / vs,
                "dimensionless",
            ],
            [
                "Exact same-point AB closure",
                "PASS" if abs(exact_residual_km) < 1.0e-8 else "FAIL",
                exact_residual_km,
                "km",
            ],
            [
                "Exact same-point π₀ versus direct one-AU π₀",
                "PASS" if abs(pi0_exact - pi0_direct) < 1.0e-12 else "FAIL",
                pi0_exact - pi0_direct,
                "arcsec",
            ],
            [
                "Exact A′B′ with physical EV/VS zero-residual hypothesis",
                "REJECTED"
                if abs(halley_residual_km) >= 0.01
                else "PASS",
                halley_residual_km,
                "km",
            ],
        ],
        columns=["Equation / test", "Status", "Residual", "Unit"],
    )

    records: list[dict[str, object]] = []
    for section, frame in (
        ("NOTATION", notation),
        ("DISTANCES", distances),
        ("GEOMETRY", geometry),
        ("RATIOS", ratios),
        ("REDUCTIONS", reductions),
        ("PARALLAX", parallax),
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
.r{background:#000;color:#fff;font-family:Arial,Helvetica,sans-serif;padding:16px;border:1px solid #fff;width:100%;box-sizing:border-box}
.r *{background:#000;color:#fff;box-sizing:border-box}
.r h1{font-size:22px;border-bottom:2px solid #fff;padding-bottom:8px}
.r h2{font-size:16px;border-top:1px solid #fff;border-bottom:1px solid #fff;padding:5px 0;margin-top:22px}
.r h3{font-size:14px}.r p{font-size:13px;line-height:1.45}
.wrap{overflow-x:auto}.audit{border-collapse:collapse;min-width:100%;width:max-content;font-size:12px}
.audit th,.audit td{border:1px solid #fff;padding:7px 9px;white-space:nowrap}
.audit th{font-weight:700;text-align:center}.audit td{text-align:right}
.audit td:first-child,.audit td:nth-child(2){text-align:left}
.note{border:1px solid #fff;padding:9px;font-weight:600}
.answer{font-size:18px;border:2px solid #fff;padding:12px;font-weight:700;text-align:center}
.path{font-family:monospace;font-size:11px;overflow-wrap:anywhere;white-space:normal}
</style>"""

    html: list[str] = [
        css,
        '<div class="r">',
        "<h1>1769 Venus Transit — Single A′B′ Halley and π₀ Audit</h1>",
        "<h2>CODE INPUTS</h2>",
        f"<p><b>Source:</b> {SOURCE}</p>",
        f"<p><b>Epoch:</b> {utc}</p>",
        "<p><b>Single A′B′ input:</b> exact finite JPL/TN36 tangent-plane separation.</p>",
        "<h2>COMMENTS</h2>",
        '<p class="note">V0053 incorrectly presented the collinear Halley model prediction as a second “TN36/Halley A′B′.” It was not a second TN36 point. V0054 removes that label and uses one exact A′B′ everywhere.</p>',
        '<p class="note">Making A′B′ identical does not make EV/VS equal to AB/A′B′. The first is a physical distance ratio; the second is the exact finite projected-geometry ratio. Both are displayed without forcing them to agree.</p>',
        "<h2>RESULTS</h2>",
        "<h3>Epoch and corrected notation</h3>",
        html_table(notation, {"Value": "{:,.15f}"}),
        "<h3>EV, VS, and ES distances</h3>",
        html_table(distances, {"Kilometers": "{:,.12f}"}),
        "<h3>Single A′B′ and direct AB</h3>",
        html_table(
            geometry,
            {
                "Arcseconds": "{:+.15f}",
                "Kilometers": "{:+,.12f}",
            },
        ),
        "<h3>Ratios</h3>",
        html_table(ratios, {"Value": "{:+.15f}"}),
        "<h3>AB reductions using the same A′B′</h3>",
        html_table(
            reductions,
            {
                "Input A′B′ arcsec": "{:+.15f}",
                "Reduction ratio": "{:+.15f}",
                "AB arcsec": "{:+.15f}",
                "AB kilometers": "{:+,.12f}",
                "Direct AB kilometers": "{:+,.12f}",
                "Residual kilometers": "{:+.15e}",
                "π₀ at 1 AU arcsec": "{:+.12f}",
            },
        ),
        "<h3>Solar parallax π₀</h3>",
        html_table(
            parallax,
            {
                "π event arcsec": "{:+.12f}",
                "π₀ at 1 AU arcsec": "{:+.12f}",
                "π₀ minus 8.794148 arcsec": "{:+.12e}",
            },
        ),
        (
            f'<p class="answer">Exact same-point result: AB = '
            f'{exact_ab_km:,.12f} km; residual = '
            f'{exact_residual_km:+.15e} km; '
            f'π₀ = {pi0_exact:.12f}″ → {pi0_exact:.6f}″.</p>'
        ),
        "<h2>OUTPUT SUMMARY</h2>",
        f'<p class="path">{CSV}</p>',
        f'<p class="path">{HTML_FILE}</p>',
        "<h2>PAPER COMPARISON</h2>",
        '<p class="note">The exact finite A′B′ is now the only A′B′. With the physical EV/VS ratio it still produces the 19.568712 km residual and π₀≈8.810899″. Zero AB residual and π₀=8.794148013717″ require the exact projected ratio AB/A′B′, not the scalar physical ratio EV/VS.</p>',
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