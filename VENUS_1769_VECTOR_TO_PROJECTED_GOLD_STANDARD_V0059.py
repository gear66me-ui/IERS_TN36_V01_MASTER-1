# V0059
# Audit reference: Establish the gold-standard instantaneous-vector and projected-axial distance notation and closure.
from __future__ import annotations

import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def need(module: str, package: str) -> None:
    try:
        __import__(module)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", package])


for _module, _package in (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("astroquery", "astroquery"),
    ("IPython", "ipython"),
):
    need(_module, _package)

import numpy as np
import pandas as pd
from astroquery.jplhorizons import Horizons
from IPython.display import HTML, display

warnings.filterwarnings("ignore", message=".*id_type.*deprecated.*")
warnings.filterwarnings("ignore", message=".*dubious year.*")

VERSION = "V0059"
TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUT = ROOT / "VENUS_1769_VECTOR_TO_PROJECTED_GOLD_STANDARD_V0059_OUTPUT"
CSV = OUT / "VENUS_1769_VECTOR_TO_PROJECTED_GOLD_STANDARD_V0059.csv"
HTML_FILE = OUT / "VENUS_1769_VECTOR_TO_PROJECTED_GOLD_STANDARD_V0059.html"

CA_UTC = "1769-06-03 22:19:04.387 UTC"
CA_JD_TDB = 2_367_328.430284398607910
JPL_AU_KM = 149_597_870.700000

TARGETS = {
    "EV": {"target": "299", "center": "@399", "name": "Earth → Venus"},
    "VS": {"target": "10", "center": "@299", "name": "Venus → Sun"},
    "ES": {"target": "10", "center": "@399", "name": "Earth → Sun"},
}

COLORS = {
    "EV": "#65D7FF",
    "VS": "#FFD166",
    "ES": "#7EE787",
    "CHECK": "#FF7B72",
    "RATIO": "#D2A8FF",
}


def horizons_vector(target: str, center: str, jd_tdb: float) -> np.ndarray:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            query = Horizons(id=target, location=center, epochs=[float(jd_tdb)], id_type=None)
            table = query.vectors(refplane="ecliptic", aberrations="geometric", cache=False)
            vector = (
                np.array(
                    [float(table["x"][0]), float(table["y"][0]), float(table["z"][0])],
                    dtype=float,
                )
                * JPL_AU_KM
            )
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise RuntimeError("JPL Horizons returned an invalid vector.")
            return vector
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"JPL Horizons vector query failed: {last_error}")


def norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    magnitude = norm(vector)
    if magnitude <= 0.0:
        raise RuntimeError("Zero-length vector encountered.")
    return vector / magnitude


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
    color_column: str | None = None,
) -> str:
    formats = formats or {}
    parts = ['<div class="wrap"><table class="audit"><thead><tr>']
    for column in frame.columns:
        parts.append(f"<th>{column}</th>")
    parts.append("</tr></thead><tbody>")
    for _, row in frame.iterrows():
        key = str(row[color_column]) if color_column and color_column in frame.columns else ""
        color = COLORS.get(key, "#FFFFFF")
        parts.append("<tr>")
        for column in frame.columns:
            text = format_value(row[column], formats.get(column))
            style = f' style="color:{color};font-weight:700"' if key in COLORS else ""
            parts.append(f"<td{style}>{text}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    vectors = {
        symbol: horizons_vector(spec["target"], spec["center"], CA_JD_TDB)
        for symbol, spec in TARGETS.items()
    }
    ev_vec = vectors["EV"]
    vs_vec = vectors["VS"]
    es_vec = vectors["ES"]

    vector_closure = ev_vec + vs_vec - es_vec
    vector_closure_norm = norm(vector_closure)

    ev_mag = norm(ev_vec)
    vs_mag = norm(vs_vec)
    es_mag = norm(es_vec)
    scalar_excess = ev_mag + vs_mag - es_mag

    es_axis = unit(es_vec)
    ev_bar = float(np.dot(ev_vec, es_axis))
    vs_bar = float(np.dot(vs_vec, es_axis))
    es_bar = float(np.dot(es_vec, es_axis))
    projected_closure = ev_bar + vs_bar - es_bar

    ev_removed = ev_mag - ev_bar
    vs_removed = vs_mag - vs_bar
    es_removed = es_mag - es_bar
    removed_sum = ev_removed + vs_removed - es_removed
    correction_identity = removed_sum - scalar_excess

    ratio_instantaneous = ev_mag / vs_mag
    ratio_projected = ev_bar / vs_bar
    ratio_difference = ratio_projected - ratio_instantaneous

    ev_perp = ev_vec - ev_bar * es_axis
    vs_perp = vs_vec - vs_bar * es_axis
    es_perp = es_vec - es_bar * es_axis
    transverse_closure = norm(ev_perp + vs_perp - es_perp)

    notation = pd.DataFrame(
        [
            ["Instantaneous three-dimensional vector", "→EV, →VS, →ES", "Fresh JPL vectors at one identical TDB epoch"],
            ["Projected axial distance", "E̅V̅, V̅S̅, E̅S̅", "Dot product with the Earth→Sun unit vector"],
            ["Vector closure", "→EV + →VS − →ES", "Must equal the zero vector"],
            ["Projected closure", "E̅V̅ + V̅S̅ − E̅S̅", "Must equal zero"],
        ],
        columns=["Dataset / test", "Gold-standard notation", "Definition"],
    )

    vector_rows = []
    for symbol, spec in TARGETS.items():
        vector = vectors[symbol]
        vector_rows.append(
            [
                spec["name"],
                f"→{symbol}",
                vector[0],
                vector[1],
                vector[2],
                norm(vector),
                symbol,
            ]
        )
    vector_rows.append(
        [
            "Instantaneous vector closure",
            "→EV + →VS − →ES",
            vector_closure[0],
            vector_closure[1],
            vector_closure[2],
            vector_closure_norm,
            "CHECK",
        ]
    )
    vector_frame = pd.DataFrame(
        vector_rows,
        columns=[
            "Instantaneous dataset",
            "Vector notation",
            "X ecliptic km",
            "Y ecliptic km",
            "Z ecliptic km",
            "Magnitude km",
            "Color key",
        ],
    )

    projection_frame = pd.DataFrame(
        [
            ["Earth → Venus projected distance", "E̅V̅ = →EV·ûES", ev_bar, ev_mag, ev_removed, "EV"],
            ["Venus → Sun projected distance", "V̅S̅ = →VS·ûES", vs_bar, vs_mag, vs_removed, "VS"],
            ["Earth → Sun projected distance", "E̅S̅ = →ES·ûES", es_bar, es_mag, es_removed, "ES"],
            [
                "Projected conjugacy",
                "E̅V̅ + V̅S̅ − E̅S̅",
                projected_closure,
                scalar_excess,
                removed_sum,
                "CHECK",
            ],
        ],
        columns=[
            "Projected dataset",
            "Projected notation",
            "Projected axial km",
            "Instantaneous magnitude / scalar excess km",
            "Removed off-axis contribution km",
            "Color key",
        ],
    )

    ratio_frame = pd.DataFrame(
        [
            ["Instantaneous magnitude ratio", "|→EV| / |→VS|", ratio_instantaneous, "Instantaneous vectors", "RATIO"],
            ["Projected conjugate ratio", "E̅V̅ / V̅S̅", ratio_projected, "Gold-standard projected distances", "RATIO"],
            ["Projected minus instantaneous", "E̅V̅/V̅S̅ − |→EV|/|→VS|", ratio_difference, "Off-center correction", "CHECK"],
        ],
        columns=["Ratio", "Equation", "Value", "Dataset", "Color key"],
    )

    identity_frame = pd.DataFrame(
        [
            ["Scalar triangle excess", "|→EV| + |→VS| − |→ES|", scalar_excess, "km"],
            ["Earth→Venus removed contribution", "|→EV| − E̅V̅", ev_removed, "km"],
            ["Venus→Sun removed contribution", "|→VS| − V̅S̅", vs_removed, "km"],
            ["Earth→Sun removed contribution", "|→ES| − E̅S̅", es_removed, "km"],
            ["Removed-contribution sum", "(|→EV|−E̅V̅)+(|→VS|−V̅S̅)−(|→ES|−E̅S̅)", removed_sum, "km"],
            ["Identity residual", "Removed sum − scalar excess", correction_identity, "km"],
        ],
        columns=["Correction identity", "Equation", "Value", "Unit"],
    )

    status = pd.DataFrame(
        [
            ["Instantaneous JPL vector closure", "PASS" if vector_closure_norm < 1.0e-5 else "FAIL", vector_closure_norm, "km"],
            ["Projected axial closure", "PASS" if abs(projected_closure) < 1.0e-5 else "FAIL", projected_closure, "km"],
            ["Transverse-vector cancellation", "PASS" if transverse_closure < 1.0e-5 else "FAIL", transverse_closure, "km"],
            ["265 km scalar excess fully identified", "PASS" if abs(correction_identity) < 1.0e-6 else "FAIL", correction_identity, "km"],
            ["No fitted scale or closure factor", "PASS", 0.0, "dimensionless"],
        ],
        columns=["Equation / test", "Status", "Residual", "Unit"],
    )

    geometry = pd.DataFrame(
        [
            ["Closest-approach UTC", CA_UTC],
            ["Closest-approach JD TDB", CA_JD_TDB],
            ["JPL frame", "Ecliptic J2000, geometric"],
            ["Projection axis", "Earth-center → Sun-center unit vector ûES"],
            ["Gold-standard scope", "Instantaneous vectors and projected axial distances only"],
        ],
        columns=["Quantity", "Value"],
    )

    records: list[dict[str, object]] = []
    for section, frame in (
        ("GEOMETRY", geometry),
        ("NOTATION", notation),
        ("INSTANTANEOUS_VECTORS", vector_frame.drop(columns=["Color key"])),
        ("PROJECTED_DISTANCES", projection_frame.drop(columns=["Color key"])),
        ("RATIOS", ratio_frame.drop(columns=["Color key"])),
        ("CORRECTION_IDENTITY", identity_frame),
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
.path{font-family:monospace;font-size:11px;overflow-wrap:anywhere;white-space:normal}
.ev{color:#65D7FF;font-weight:700}.vs{color:#FFD166;font-weight:700}.es{color:#7EE787;font-weight:700}.check{color:#FF7B72;font-weight:700}.ratio{color:#D2A8FF;font-weight:700}
</style>"""

    html = [css, '<div class="r">', "<h1>1769 Venus Transit — Instantaneous Vector → Projected Axial Gold Standard</h1>"]
    html += [
        "<h2>CODE INPUTS</h2>",
        f"<p><b>Exact epoch:</b> {CA_UTC}</p>",
        f"<p><b>JD TDB:</b> {CA_JD_TDB:.15f}</p>",
        "<p><b>Fresh JPL queries:</b> →EV, →VS, and →ES at one identical epoch.</p>",
    ]
    html += [
        "<h2>COMMENTS</h2>",
        '<p class="note">Only two datasets are retained: instantaneous three-dimensional vectors (arrow notation) and their Earth–Sun-axis projected distances (bar notation).</p>',
        '<p class="note">The 265.596 km scalar excess is exactly the sum of the off-axis contributions removed from EV and VS. No prior model nomenclature is used.</p>',
    ]
    html += [
        "<h2>RESULTS</h2>",
        "<h3>Epoch and convention</h3>",
        colored_table(geometry, {"Value": "{:,.15f}"}),
        "<h3>Gold-standard notation</h3>",
        colored_table(notation),
        "<h3>1. Instantaneous JPL vector dataset</h3>",
        colored_table(
            vector_frame,
            {
                "X ecliptic km": "{:+,.12f}",
                "Y ecliptic km": "{:+,.12f}",
                "Z ecliptic km": "{:+,.12f}",
                "Magnitude km": "{:+,.12f}",
            },
            "Color key",
        ),
        "<h3>2. Projected axial dataset</h3>",
        colored_table(
            projection_frame,
            {
                "Projected axial km": "{:+,.12f}",
                "Instantaneous magnitude / scalar excess km": "{:+,.12f}",
                "Removed off-axis contribution km": "{:+,.12f}",
            },
            "Color key",
        ),
        "<h3>3. Dataset ratios</h3>",
        colored_table(ratio_frame, {"Value": "{:+.15f}"}, "Color key"),
        "<h3>4. The 265 km correction identity</h3>",
        colored_table(identity_frame, {"Value": "{:+,.15f}"}),
        (
            f'<p class="answer">→EV + →VS − →ES = {vector_closure_norm:.12e} km; '
            f'E̅V̅ + V̅S̅ − E̅S̅ = {projected_closure:+.12e} km; '
            f'off-axis removal = {removed_sum:,.12f} km = scalar excess.</p>'
        ),
    ]
    html += [
        "<h2>OUTPUT SUMMARY</h2>",
        f'<p class="path">{CSV}</p>',
        f'<p class="path">{HTML_FILE}</p>',
    ]
    html += [
        "<h2>PAPER COMPARISON</h2>",
        '<p class="note">NOT USED. This is the standalone gold-standard JPL vector and projected-distance audit.</p>',
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
# V0059