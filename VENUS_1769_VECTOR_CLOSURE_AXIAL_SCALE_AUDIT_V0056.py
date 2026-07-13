# V0056
# Audit reference: Verify independent JPL EV+VS=ES vector closure and derive conjugate axial distances without forcing scalar magnitudes.
from __future__ import annotations

import math
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
    ("astropy", "astropy"),
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

VERSION = "V0056"
TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
SOURCE_CANDIDATES = (
    ROOT / "VENUS_1769_DISTANCE_RATIO_APRIME_BPRIME_AUDIT_V0055_OUTPUT" / "VENUS_1769_DISTANCE_RATIO_APRIME_BPRIME_AUDIT_V0055.csv",
    ROOT / "VENUS_1769_SINGLE_APRIME_BPRIME_HALLEY_AUDIT_V0054_OUTPUT" / "VENUS_1769_SINGLE_APRIME_BPRIME_HALLEY_AUDIT_V0054.csv",
    ROOT / "VENUS_1769_ORDER_INDEPENDENT_PROJECTOR_AUDIT_V0052_OUTPUT" / "VENUS_1769_ORDER_INDEPENDENT_PROJECTOR_AUDIT_V0052.csv",
)
OUT = ROOT / "VENUS_1769_VECTOR_CLOSURE_AXIAL_SCALE_AUDIT_V0056_OUTPUT"
CSV = OUT / "VENUS_1769_VECTOR_CLOSURE_AXIAL_SCALE_AUDIT_V0056.csv"
HTML_FILE = OUT / "VENUS_1769_VECTOR_CLOSURE_AXIAL_SCALE_AUDIT_V0056.html"
JPL_AU_KM = 149_597_870.700000
TARGETS = {
    "EV": {"target": "299", "center": "@399", "label": "Earth → Venus"},
    "VS": {"target": "10", "center": "@299", "label": "Venus → Sun"},
    "ES": {"target": "10", "center": "@399", "label": "Earth → Sun"},
}


def numeric(value: object) -> float:
    result = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(result):
        raise RuntimeError(f"Expected numeric value, received {value!r}.")
    return float(result)


def load_epoch() -> tuple[float, str, str]:
    for path in SOURCE_CANDIDATES:
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if "section" not in frame.columns:
            continue
        geometry = frame[frame["section"].astype(str) == "GEOMETRY"]
        if geometry.empty:
            continue
        key_columns = [column for column in ("Quantity", "quantity") if column in geometry.columns]
        value_columns = [column for column in ("Value", "value") if column in geometry.columns]
        if not key_columns or not value_columns:
            continue
        key_column = key_columns[0]
        value_column = value_columns[0]
        jd_rows = geometry[geometry[key_column].astype(str) == "Closest-approach JD TDB"]
        utc_rows = geometry[geometry[key_column].astype(str) == "Closest-approach UTC"]
        if len(jd_rows) == 1:
            jd = numeric(jd_rows.iloc[0][value_column])
            utc = str(utc_rows.iloc[0][value_column]) if len(utc_rows) == 1 else "NOT AVAILABLE"
            return jd, utc, str(path)
    raise FileNotFoundError("Run V0052, V0054, or V0055 first; no closest-approach source CSV was found.")


def horizons_vector(target: str, center: str, jd_tdb: float) -> np.ndarray:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            query = Horizons(id=target, location=center, epochs=[float(jd_tdb)], id_type=None)
            result = query.vectors(refplane="ecliptic", aberrations="geometric", cache=False)
            vector = np.array(
                [float(result["x"][0]), float(result["y"][0]), float(result["z"][0])],
                dtype=float,
            ) * JPL_AU_KM
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise RuntimeError("Horizons returned an invalid vector.")
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


def clipped_acos(value: float) -> float:
    return math.acos(max(-1.0, min(1.0, float(value))))


def html_table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    shown = frame.copy()
    for column, pattern in (formats or {}).items():
        if column not in shown.columns:
            continue
        def render(value: object) -> str:
            converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            return pattern.format(float(converted)) if not pd.isna(converted) else str(value)
        shown[column] = shown[column].map(render)
    return '<div class="wrap">' + shown.to_html(index=False, border=0, classes="audit", escape=False) + "</div>"


def main() -> None:
    jd_tdb, utc, source_path = load_epoch()
    OUT.mkdir(parents=True, exist_ok=True)

    vectors = {symbol: horizons_vector(spec["target"], spec["center"], jd_tdb) for symbol, spec in TARGETS.items()}
    ev_vec, vs_vec, es_vec = vectors["EV"], vectors["VS"], vectors["ES"]

    closure_vec = ev_vec + vs_vec - es_vec
    closure_norm = norm(closure_vec)

    ev, vs, es = norm(ev_vec), norm(vs_vec), norm(es_vec)
    scalar_excess = ev + vs - es

    axis = unit(es_vec)
    ev_axial = float(np.dot(ev_vec, axis))
    vs_axial = float(np.dot(vs_vec, axis))
    es_axial = float(np.dot(es_vec, axis))
    axial_closure = ev_axial + vs_axial - es_axial

    ev_perp = ev_vec - ev_axial * axis
    vs_perp = vs_vec - vs_axial * axis
    es_perp = es_vec - es_axial * axis
    transverse_closure = norm(ev_perp + vs_perp - es_perp)

    ev_perp_mag, vs_perp_mag, es_perp_mag = norm(ev_perp), norm(vs_perp), norm(es_perp)
    ev_scale, vs_scale, es_scale = ev_axial / ev, vs_axial / vs, es_axial / es
    ev_angle = clipped_acos(ev_scale)
    vs_angle = clipped_acos(vs_scale)

    ratio_magnitude = ev / vs
    ratio_axial = ev_axial / vs_axial
    ratio_difference = ratio_axial - ratio_magnitude

    vector_rows = []
    for symbol, spec in TARGETS.items():
        vector = vectors[symbol]
        vector_rows.append([spec["label"], symbol, vector[0], vector[1], vector[2], norm(vector), f'<span class="{symbol.lower()}">{symbol}</span>'])
    vector_rows.append(["Vector closure EV + VS − ES", "Δ", closure_vec[0], closure_vec[1], closure_vec[2], closure_norm, '<span class="closure">CLOSURE</span>'])
    vector_frame = pd.DataFrame(vector_rows, columns=["Vector", "Symbol", "X ecliptic km", "Y ecliptic km", "Z ecliptic km", "Magnitude km", "Color key"])

    scalar_frame = pd.DataFrame([
        ["Earth → Venus magnitude", "EV", ev, '<span class="ev">EV</span>'],
        ["Venus → Sun magnitude", "VS", vs, '<span class="vs">VS</span>'],
        ["Earth → Sun magnitude", "ES", es, '<span class="es">ES</span>'],
        ["Scalar triangle excess", "EV + VS − ES", scalar_excess, '<span class="closure">EXPECTED NONZERO</span>'],
    ], columns=["Scalar distance", "Equation", "Kilometers", "Color key"])

    axial_frame = pd.DataFrame([
        ["Earth → Venus axial projection", "EV∥ = EV·ûES", ev_axial, ev_scale, math.degrees(ev_angle), ev_perp_mag, '<span class="ev">EV∥</span>'],
        ["Venus → Sun axial projection", "VS∥ = VS·ûES", vs_axial, vs_scale, math.degrees(vs_angle), vs_perp_mag, '<span class="vs">VS∥</span>'],
        ["Earth → Sun axial projection", "ES∥ = ES·ûES", es_axial, es_scale, 0.0, es_perp_mag, '<span class="es">ES∥</span>'],
        ["Axial conjugacy check", "EV∥ + VS∥ − ES∥", axial_closure, np.nan, np.nan, transverse_closure, '<span class="closure">ZERO CHECK</span>'],
    ], columns=["Projected distance", "Equation", "Axial kilometers", "Projection scale cos θ", "Angular offset deg", "Transverse magnitude / closure km", "Color key"])

    ratio_frame = pd.DataFrame([
        ["Magnitude ratio", "EV/VS", ratio_magnitude, "Original scalar lengths", '<span class="ratio">RATIO</span>'],
        ["Axial conjugate ratio", "EV∥/VS∥", ratio_axial, "Common Earth–Sun axis", '<span class="ratio">RATIO∥</span>'],
        ["Axial minus magnitude ratio", "EV∥/VS∥ − EV/VS", ratio_difference, "Angular-offset effect", '<span class="closure">DIFFERENCE</span>'],
        ["Axial closure ratio", "(EV∥ + VS∥)/ES∥", (ev_axial + vs_axial) / es_axial, "Must equal one", '<span class="closure">UNITY</span>'],
    ], columns=["Ratio", "Equation", "Value", "Meaning", "Color key"])

    geometry_frame = pd.DataFrame([
        ["Closest-approach UTC", utc],
        ["Closest-approach JD TDB", jd_tdb],
        ["JPL frame", "Ecliptic J2000, geometric"],
        ["Common scaling axis", "Earth-center → Sun-center unit vector ûES"],
        ["Next-step scope", "A′B′ derivation deferred"],
    ], columns=["Quantity", "Value"])

    status_frame = pd.DataFrame([
        ["Independent JPL vector closure EV + VS − ES", "PASS" if closure_norm < 0.01 else "FAIL", closure_norm, "km"],
        ["Axial projected-distance closure", "PASS" if abs(axial_closure) < 1.0e-6 else "FAIL", axial_closure, "km"],
        ["Transverse-vector cancellation", "PASS" if transverse_closure < 0.01 else "FAIL", transverse_closure, "km"],
        ["Scalar magnitude excess is not forced to zero", "PASS" if scalar_excess > 0.0 else "FAIL", scalar_excess, "km"],
        ["No fitted correction or closure factor used", "PASS", 0.0, "dimensionless"],
    ], columns=["Equation / test", "Status", "Residual", "Unit"])

    records: list[dict[str, object]] = []
    for section, frame in (("GEOMETRY", geometry_frame), ("VECTORS", vector_frame), ("SCALAR_MAGNITUDES", scalar_frame), ("AXIAL_PROJECTIONS", axial_frame), ("RATIOS", ratio_frame), ("STATUS", status_frame)):
        for row_number, row in frame.iterrows():
            record: dict[str, object] = {"section": section, "row": int(row_number)}
            record.update({str(key): value for key, value in row.items()})
            records.append(record)
    pd.DataFrame(records).to_csv(CSV, index=False, float_format="%.15f")

    css = """
<style>
.r{background:#000;color:#fff;font-family:Arial,Helvetica,sans-serif;padding:16px;border:1px solid #fff;width:100%;box-sizing:border-box}
.r *{background:#000;color:#fff;box-sizing:border-box}.r h1{font-size:22px;border-bottom:2px solid #fff;padding-bottom:8px}
.r h2{font-size:16px;border-top:1px solid #fff;border-bottom:1px solid #fff;padding:5px 0;margin-top:22px}.r h3{font-size:14px}.r p{font-size:13px;line-height:1.45}.wrap{overflow-x:auto}
.audit{border-collapse:collapse;min-width:100%;width:max-content;font-size:12px}.audit th,.audit td{border:1px solid #fff;padding:7px 9px;white-space:nowrap}
.audit th{font-weight:700;text-align:center}.audit td{text-align:right}.audit td:first-child,.audit td:nth-child(2){text-align:left}
.note{border:1px solid #fff;padding:9px;font-weight:600}.answer{font-size:18px;border:2px solid #fff;padding:12px;font-weight:700;text-align:center}
.path{font-family:monospace;font-size:11px;overflow-wrap:anywhere;white-space:normal}.legend{display:flex;flex-wrap:wrap;gap:12px;border:1px solid #fff;padding:9px}
.ev{color:#65d7ff;font-weight:700}.vs{color:#ffd166;font-weight:700}.es{color:#7ee787;font-weight:700}.ratio{color:#d2a8ff;font-weight:700}.closure{color:#ff7b72;font-weight:700}
</style>
"""
    legend = '<div class="legend"><span class="ev">EV — Earth→Venus</span><span class="vs">VS — Venus→Sun</span><span class="es">ES — Earth→Sun</span><span class="ratio">Ratios</span><span class="closure">Closure checks</span></div>'
    html = [css, '<div class="r">', "<h1>1769 Venus Transit — Vector Closure and Axial Distance Audit</h1>"]
    html += ["<h2>CODE INPUTS</h2>", f"<p><b>Epoch source:</b> {source_path}</p>", f"<p><b>Exact epoch:</b> {utc}</p>", "<p><b>JPL queries:</b> EV from Earth center, VS from Venus center, and ES from Earth center at one identical TDB epoch.</p>"]
    html += ["<h2>COMMENTS</h2>", '<p class="note">The independent three-vector identity is tested first. Scalar magnitudes are not altered or forced to add. The only derived “scaled” distances are orthogonal projections onto the common Earth–Sun axis.</p>', '<p class="note">These axial projections are the conjugate collinear distances: EV∥ + VS∥ = ES∥. The next A′B′ calculation is deliberately deferred.</p>', legend]
    html += ["<h2>RESULTS</h2>", "<h3>Epoch and convention</h3>", html_table(geometry_frame, {"Value": "{:,.15f}"})]
    html += ["<h3>1. Independent JPL vectors</h3>", html_table(vector_frame, {"X ecliptic km": "{:+,.12f}", "Y ecliptic km": "{:+,.12f}", "Z ecliptic km": "{:+,.12f}", "Magnitude km": "{:+,.12f}"})]
    html += ["<h3>2. Scalar magnitudes</h3>", html_table(scalar_frame, {"Kilometers": "{:+,.12f}"})]
    html += ["<h3>3. Earth–Sun-axis projected distances</h3>", html_table(axial_frame, {"Axial kilometers": "{:+,.12f}", "Projection scale cos θ": "{:+.15f}", "Angular offset deg": "{:+.12f}", "Transverse magnitude / closure km": "{:+,.12f}"})]
    html += ["<h3>4. Ratios after vector projection</h3>", html_table(ratio_frame, {"Value": "{:+.15f}"})]
    html += ["<p class=\"answer\">Vector closure |EV + VS − ES| = " + f"{closure_norm:.12e} km; axial closure EV∥ + VS∥ − ES∥ = {axial_closure:+.12e} km.</p>"]
    html += ["<h2>OUTPUT SUMMARY</h2>", f'<p class="path">{CSV}</p>', f'<p class="path">{HTML_FILE}</p>']
    html += ["<h2>PAPER COMPARISON</h2>", '<p class="note">NOT USED. This module is a JPL vector-identity and common-axis projection audit only.</p>']
    html += ["<h2>EQUATION STATUS</h2>", html_table(status_frame, {"Residual": "{:+.15e}"}), "</div>"]
    report = "".join(html)
    HTML_FILE.write_text("<html><head><meta charset='utf-8'></head><body style='margin:0;background:#000;color:#fff'>" + report + "</body></html>", encoding="utf-8")
    display(HTML(report))
    print(datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0056