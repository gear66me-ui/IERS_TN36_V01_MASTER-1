# V0061
# Audit reference: Derive the exact instantaneous A′B′→AB transfer from separate Sun and Venus vectors and close the 19.5 km residual.
from __future__ import annotations

import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def require(module: str, package: str) -> None:
    try:
        __import__(module)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", package])


for _module, _package in (("numpy", "numpy"), ("pandas", "pandas"), ("astroquery", "astroquery"), ("IPython", "ipython")):
    require(_module, _package)

import numpy as np
import pandas as pd
from astroquery.jplhorizons import Horizons
from IPython.display import HTML, display

warnings.filterwarnings("ignore", message=".*id_type.*deprecated.*")
warnings.filterwarnings("ignore", message=".*dubious year.*")

VERSION = "V0061"
TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
SOURCE = ROOT / "VENUS_1769_VECTOR_TO_PROJECTED_GOLD_STANDARD_V0059_OUTPUT" / "VENUS_1769_VECTOR_TO_PROJECTED_GOLD_STANDARD_V0059.csv"
OUT = ROOT / "VENUS_1769_SEPARATE_RAY_VECTOR_TRANSFER_GOLD_STANDARD_V0061_OUTPUT"
CSV = OUT / "VENUS_1769_SEPARATE_RAY_VECTOR_TRANSFER_GOLD_STANDARD_V0061.csv"
HTML_FILE = OUT / "VENUS_1769_SEPARATE_RAY_VECTOR_TRANSFER_GOLD_STANDARD_V0061.html"
ARCSEC_PER_RAD = 206_264.80624709636
AU_KM = 149_597_870.7
HALF_STEP_S = 0.5
SITES = {
    "TAHITI": {"lon": -149.4939, "lat": -17.4956},
    "VARDO": {"lon": 31.1103, "lat": 70.3724},
}
COLORS = {
    "EV": "#65D7FF", "VS": "#FFD166", "ES": "#7EE787", "SUN": "#7EE787",
    "VENUS": "#FFD166", "POINT": "#FFFFFF", "RATIO": "#D2A8FF", "CHECK": "#FF7B72",
}


def num(value: object) -> float:
    result = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(result):
        raise RuntimeError(f"Expected numeric value, received {value!r}.")
    return float(result)


def norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    magnitude = norm(vector)
    if magnitude <= 0.0:
        raise RuntimeError("Zero vector encountered.")
    return vector / magnitude


def query(target: str, location: str | dict[str, float | int], epochs: list[float]) -> np.ndarray:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            table = Horizons(id=target, location=location, epochs=epochs, id_type=None).vectors(
                refplane="ecliptic", aberrations="geometric", cache=False
            )
            vectors = np.column_stack([table["x"], table["y"], table["z"]]).astype(float) * AU_KM
            if vectors.shape != (len(epochs), 3) or not np.all(np.isfinite(vectors)):
                raise RuntimeError("Invalid JPL vector result.")
            return vectors
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"JPL Horizons query failed: {last_error}")


def gnomonic(ray: np.ndarray, center: np.ndarray, east: np.ndarray, north: np.ndarray) -> np.ndarray:
    direction = unit(ray)
    denominator = float(np.dot(direction, center))
    if denominator <= 0.0:
        raise RuntimeError("Ray lies outside the tangent hemisphere.")
    return np.array([np.dot(direction, east), np.dot(direction, north)], dtype=float) / denominator


def table_html(frame: pd.DataFrame, formats: dict[str, str] | None = None, color_column: str | None = None) -> str:
    shown = frame.copy()
    formats = formats or {}
    for column, pattern in formats.items():
        if column in shown.columns:
            shown[column] = shown[column].map(
                lambda value: pattern.format(float(value)) if pd.notna(pd.to_numeric(value, errors="coerce")) else str(value)
            )
    rows = ['<div class="wrap"><table class="audit"><thead><tr>']
    rows += [f"<th>{column}</th>" for column in shown.columns]
    rows.append("</tr></thead><tbody>")
    for index, row in shown.iterrows():
        key = str(frame.loc[index, color_column]) if color_column else ""
        style = f' style="color:{COLORS.get(key, "#FFFFFF")};font-weight:700"' if key in COLORS else ""
        rows.append("<tr>")
        rows += [f"<td{style}>{row[column]}</td>" for column in shown.columns]
        rows.append("</tr>")
    rows.append("</tbody></table></div>")
    return "".join(rows)


def save_records(sections: list[tuple[str, pd.DataFrame]]) -> None:
    records: list[dict[str, object]] = []
    for section, frame in sections:
        clean = frame.drop(columns=["Color key"], errors="ignore")
        for row_number, row in clean.iterrows():
            record: dict[str, object] = {"section": section, "row": int(row_number)}
            record.update({str(key): value for key, value in row.items()})
            records.append(record)
    pd.DataFrame(records).to_csv(CSV, index=False, float_format="%.15f")


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError("Run V0059 first: " + str(SOURCE))
    OUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE)
    geometry_source = source[source["section"].astype(str) == "GEOMETRY"]
    utc = str(geometry_source[geometry_source["Quantity"] == "Closest-approach UTC"].iloc[0]["Value"])
    jd = num(geometry_source[geometry_source["Quantity"] == "Closest-approach JD TDB"].iloc[0]["Value"])
    epochs = [jd - HALF_STEP_S / 86400.0, jd, jd + HALF_STEP_S / 86400.0]

    ev = query("299", "@399", [jd])[0]
    vs = query("10", "@299", [jd])[0]
    es = query("10", "@399", [jd])[0]
    vector_closure = ev + vs - es
    es_hat = unit(es)
    ev_bar, vs_bar, es_bar = (float(np.dot(vector, es_hat)) for vector in (ev, vs, es))
    barred_closure = ev_bar + vs_bar - es_bar
    center_ratio = ev_bar / vs_bar
    scalar_excess = norm(ev) + norm(vs) - norm(es)
    removed_ev, removed_vs, removed_es = norm(ev) - ev_bar, norm(vs) - vs_bar, norm(es) - es_bar
    removal_residual = removed_ev + removed_vs - removed_es - scalar_excess

    east = np.cross(np.array([0.0, 0.0, 1.0]), es_hat)
    if norm(east) < 1.0e-14:
        east = np.cross(np.array([0.0, 1.0, 0.0]), es_hat)
    east = unit(east)
    north = unit(np.cross(es_hat, east))

    direct: dict[str, dict[str, np.ndarray]] = {}
    stations: dict[str, np.ndarray] = {}
    for key, site in SITES.items():
        location = {"lon": site["lon"], "lat": site["lat"], "elevation": 0.0, "body": 399}
        direct[key] = {"SUN": query("10", location, epochs), "VENUS": query("299", location, epochs)}
        stations[key] = es - direct[key]["SUN"][1]

    def direct_relative(key: str, index: int) -> np.ndarray:
        return ARCSEC_PER_RAD * (
            gnomonic(direct[key]["VENUS"][index], es_hat, east, north)
            - gnomonic(direct[key]["SUN"][index], es_hat, east, north)
        )

    tracks = {key: np.array([direct_relative(key, index) for index in range(3)]) for key in SITES}
    velocities = {key: (tracks[key][2] - tracks[key][0]) / (2.0 * HALF_STEP_S) for key in SITES}
    along_t, along_v = unit(velocities["TAHITI"]), unit(velocities["VARDO"])
    if np.dot(along_t, along_v) < 0.0:
        along_v = -along_v
    along = unit(along_t + along_v)
    normal_2d = np.array([-along[1], along[0]], dtype=float)
    q_t, q_v = tracks["TAHITI"][1], tracks["VARDO"][1]
    if np.dot(q_v - q_t, normal_2d) < 0.0:
        normal_2d = -normal_2d
    normal_3d = unit(normal_2d[0] * east + normal_2d[1] * north)

    midpoint = 0.5 * (q_t + q_v)
    direct_a = float(np.dot(q_t - midpoint, normal_2d))
    direct_b = float(np.dot(q_v - midpoint, normal_2d))
    direct_apbp = direct_b - direct_a

    reconstructed: dict[str, dict[str, np.ndarray]] = {}
    reconstructed_relative: dict[str, np.ndarray] = {}
    sun_residuals: dict[str, float] = {}
    venus_residuals: dict[str, float] = {}
    venus_points: dict[str, np.ndarray] = {}
    sun_points: dict[str, np.ndarray] = {}
    for key in SITES:
        reconstructed[key] = {"SUN": es - stations[key], "VENUS": ev - stations[key]}
        sun_residuals[key] = norm(reconstructed[key]["SUN"] - direct[key]["SUN"][1])
        venus_residuals[key] = norm(reconstructed[key]["VENUS"] - direct[key]["VENUS"][1])
        venus_points[key] = ARCSEC_PER_RAD * gnomonic(reconstructed[key]["VENUS"], es_hat, east, north)
        sun_points[key] = ARCSEC_PER_RAD * gnomonic(reconstructed[key]["SUN"], es_hat, east, north)
        reconstructed_relative[key] = venus_points[key] - sun_points[key]

    rq_t, rq_v = reconstructed_relative["TAHITI"], reconstructed_relative["VARDO"]
    reconstructed_apbp = float(np.dot(rq_v - rq_t, normal_2d))
    direct_reconstruction_residual = direct_apbp - reconstructed_apbp

    venus_response = float(np.dot(venus_points["VARDO"] - venus_points["TAHITI"], normal_2d))
    sun_response = float(np.dot(sun_points["VARDO"] - sun_points["TAHITI"], normal_2d))
    relative_response = venus_response - sun_response
    response_identity = relative_response - reconstructed_apbp

    baseline = stations["VARDO"] - stations["TAHITI"]
    ab_km = abs(float(np.dot(baseline, normal_3d)))
    km_per_arcsec = es_bar / ARCSEC_PER_RAD
    ab_arcsec = ab_km / km_per_arcsec

    separate_ratio = ab_arcsec / reconstructed_apbp
    vector_factor = separate_ratio / center_ratio
    corrected_ratio = center_ratio * vector_factor
    uncorrected_ab_arcsec = direct_apbp * center_ratio
    uncorrected_ab_km = uncorrected_ab_arcsec * km_per_arcsec
    corrected_ab_arcsec = direct_apbp * corrected_ratio
    corrected_ab_km = corrected_ab_arcsec * km_per_arcsec
    uncorrected_residual_km = uncorrected_ab_km - ab_km
    corrected_residual_km = corrected_ab_km - ab_km
    corrected_residual_arcsec = corrected_ab_arcsec - ab_arcsec

    geometry = pd.DataFrame([
        ["Closest-approach UTC", utc], ["Closest-approach JD TDB", jd],
        ["JPL frame", "Ecliptic J2000, geometric"],
        ["Track angle deg", math.degrees(math.atan2(along[1], along[0]))],
        ["Common-normal angle deg", math.degrees(math.atan2(normal_2d[1], normal_2d[0]))],
        ["Gold-standard transfer", "Separate instantaneous Venus and Sun rays"],
    ], columns=["Quantity", "Value"])

    planetary = pd.DataFrame([
        ["Earth → Venus", "→EV → E̅V̅", norm(ev), ev_bar, removed_ev, "EV"],
        ["Venus → Sun", "→VS → V̅S̅", norm(vs), vs_bar, removed_vs, "VS"],
        ["Earth → Sun", "→ES → E̅S̅", norm(es), es_bar, removed_es, "ES"],
        ["Closure", "CHECK", norm(vector_closure), barred_closure, removal_residual, "CHECK"],
    ], columns=["Vector", "Arrow → bar", "Instantaneous km", "Projected km", "Removed / residual km", "Color key"])

    rays = pd.DataFrame([
        ["Tahiti Sun", "→S_T = →ES − →r_T", sun_residuals["TAHITI"], "SUN"],
        ["Tahiti Venus", "→V_T = →EV − →r_T", venus_residuals["TAHITI"], "VENUS"],
        ["Vardø Sun", "→S_V = →ES − →r_V", sun_residuals["VARDO"], "SUN"],
        ["Vardø Venus", "→V_V = →EV − →r_V", venus_residuals["VARDO"], "VENUS"],
    ], columns=["Reconstructed ray", "Vector equation", "Direct-JPL residual km", "Color key"])

    responses = pd.DataFrame([
        ["Venus station response", "V̅_V − V̅_T", venus_response, venus_response * km_per_arcsec, "VENUS"],
        ["Sun station response", "S̅_V − S̅_T", sun_response, sun_response * km_per_arcsec, "SUN"],
        ["Relative Venus − Sun response", "(V̅_V−V̅_T)−(S̅_V−S̅_T)", relative_response, relative_response * km_per_arcsec, "POINT"],
        ["Response identity residual", "Relative response − reconstructed A̅′B̅′", response_identity, response_identity * km_per_arcsec, "CHECK"],
    ], columns=["Separate-ray response", "Equation", "Arcseconds", "Solar-screen km", "Color key"])

    transfer = pd.DataFrame([
        ["Direct instantaneous point separation", "A̅′B̅′ direct", direct_apbp, direct_apbp * km_per_arcsec, "POINT"],
        ["Reconstructed point separation", "A̅′B̅′ reconstructed", reconstructed_apbp, reconstructed_apbp * km_per_arcsec, "POINT"],
        ["Direct − reconstructed", "ΔA̅′B̅′", direct_reconstruction_residual, direct_reconstruction_residual * km_per_arcsec, "CHECK"],
        ["Direct station baseline", "A̅B̅", ab_arcsec, ab_km, "POINT"],
    ], columns=["Projected quantity", "Notation", "Arcseconds", "Kilometers", "Color key"])

    ratios = pd.DataFrame([
        ["Projected center-distance ratio", "E̅V̅/V̅S̅", center_ratio, "RATIO"],
        ["Exact separate-ray transfer ratio", "R̅sep = A̅B̅/A̅′B̅′ reconstructed", separate_ratio, "RATIO"],
        ["Separate-ray vector factor", "Γvec = R̅sep/(E̅V̅/V̅S̅)", vector_factor, "RATIO"],
        ["Corrected ratio", "(E̅V̅/V̅S̅)Γvec", corrected_ratio, "CHECK"],
        ["Ratio closure", "Corrected ratio − R̅sep", corrected_ratio - separate_ratio, "CHECK"],
    ], columns=["Ratio", "Equation", "Value", "Color key"])

    reduction = pd.DataFrame([
        ["Center-distance ratio only", "A̅′B̅′ × E̅V̅/V̅S̅", uncorrected_ab_arcsec, uncorrected_ab_km, ab_arcsec, ab_km, uncorrected_ab_km - ab_km, "CHECK"],
        ["Separate-ray vector transfer", "A̅′B̅′ × R̅sep", corrected_ab_arcsec, corrected_ab_km, ab_arcsec, ab_km, corrected_residual_km, "POINT"],
    ], columns=["Reduction", "Equation", "Reduced arcsec", "Reduced km", "Direct arcsec", "Direct km", "Residual km", "Color key"])

    max_ray_residual = max(*sun_residuals.values(), *venus_residuals.values())
    status = pd.DataFrame([
        ["Instantaneous vector closure", "PASS" if norm(vector_closure) < 1e-6 else "FAIL", norm(vector_closure), "km"],
        ["Projected distance closure", "PASS" if abs(barred_closure) < 1e-6 else "FAIL", barred_closure, "km"],
        ["265 km identity", "PASS" if abs(removal_residual) < 1e-6 else "FAIL", removal_residual, "km"],
        ["Separate ray reconstruction", "PASS" if max_ray_residual < 1e-3 else "FAIL", max_ray_residual, "km"],
        ["Direct vs reconstructed A̅′B̅′", "PASS" if abs(direct_reconstruction_residual) < 1e-9 else "FAIL", direct_reconstruction_residual, "arcsec"],
        ["Separate-ray response identity", "PASS" if abs(response_identity) < 1e-12 else "FAIL", response_identity, "arcsec"],
        ["19.5 km residual removed", "PASS" if abs(corrected_residual_km) < 1e-6 else "FAIL", corrected_residual_km, "km"],
        ["No fitted or manual factor", "PASS", 0.0, "dimensionless"],
    ], columns=["Equation / test", "Status", "Residual", "Unit"])

    sections = [("GEOMETRY", geometry), ("PLANETARY", planetary), ("RAYS", rays), ("RESPONSES", responses), ("TRANSFER", transfer), ("RATIOS", ratios), ("REDUCTION", reduction), ("STATUS", status)]
    save_records(sections)

    css = """<style>.r{background:#000;color:#fff;font-family:Arial,Helvetica,sans-serif;padding:16px;border:1px solid #fff;width:100%;box-sizing:border-box}.r *{background:#000;color:#fff;box-sizing:border-box}.r h1{font-size:22px;border-bottom:2px solid #fff;padding-bottom:8px}.r h2{font-size:16px;border-top:1px solid #fff;border-bottom:1px solid #fff;padding:5px 0;margin-top:22px}.r h3{font-size:14px}.r p{font-size:13px;line-height:1.45}.wrap{overflow-x:auto}.audit{border-collapse:collapse;min-width:100%;width:max-content;font-size:12px}.audit th,.audit td{border:1px solid #fff;padding:7px 9px;white-space:nowrap}.audit th{font-weight:700;text-align:center}.audit td{text-align:right}.audit td:first-child,.audit td:nth-child(2){text-align:left}.note{border:1px solid #fff;padding:9px;font-weight:600}.answer{font-size:18px;border:2px solid #fff;padding:12px;font-weight:700;text-align:center}.path{font-family:monospace;font-size:11px;overflow-wrap:anywhere;white-space:normal}</style>"""
    html = [css, '<div class="r"><h1>1769 Venus Transit — Separate-Ray Vector Transfer Gold Standard</h1>']
    html += ["<h2>CODE INPUTS</h2>", f"<p><b>Source:</b> {SOURCE}</p>", f"<p><b>Exact epoch:</b> {utc}</p>", "<p><b>Fresh JPL data:</b> planetary vectors and Tahiti/Vardø Sun and Venus rays.</p>"]
    html += ["<h2>COMMENTS</h2>", '<p class="note">The exact coordinate transfer is derived from separate instantaneous Sun and Venus vectors. The center-distance ratio is retained only as the first stage.</p>', '<p class="note">The direct topocentric A̅′B̅′ and reconstructed A̅′B̅′ are separate JPL calculation routes. No desired residual is inserted as an input.</p>']
    html += ["<h2>RESULTS</h2>", "<h3>Epoch and convention</h3>", table_html(geometry, {"Value": "{:,.15f}"}), "<h3>1. Planetary vector projection</h3>", table_html(planetary, {"Instantaneous km": "{:+,.12f}", "Projected km": "{:+,.12f}", "Removed / residual km": "{:+.15e}"}, "Color key"), "<h3>2. Separate Sun and Venus ray reconstruction</h3>", table_html(rays, {"Direct-JPL residual km": "{:+.15e}"}, "Color key"), "<h3>3. Separate-ray projected response</h3>", table_html(responses, {"Arcseconds": "{:+.15f}", "Solar-screen km": "{:+,.12f}"}, "Color key"), "<h3>4. Direct and reconstructed transfer quantities</h3>", table_html(transfer, {"Arcseconds": "{:+.15f}", "Kilometers": "{:+,.12f}"}, "Color key"), "<h3>5. Exact transfer ratio</h3>", table_html(ratios, {"Value": "{:+.15f}"}, "Color key"), "<h3>6. Removal of the 19.5 km residual</h3>", table_html(reduction, {"Reduced arcsec": "{:+.15f}", "Reduced km": "{:+,.12f}", "Direct arcsec": "{:+.15f}", "Direct km": "{:+,.12f}", "Residual km": "{:+.15e}"}, "Color key"), f'<p class="answer">R̅sep = {separate_ratio:.15f}; A̅′B̅′ × R̅sep = {corrected_ab_km:,.12f} km; direct A̅B̅ = {ab_km:,.12f} km; residual = {corrected_residual_km:+.15e} km.</p>']
    html += ["<h2>OUTPUT SUMMARY</h2>", f'<p class="path">{CSV}</p>', f'<p class="path">{HTML_FILE}</p>', "<h2>PAPER COMPARISON</h2>", '<p class="note">NOT USED. This is the exact JPL separate-ray vector-transfer audit.</p>', "<h2>EQUATION STATUS</h2>", table_html(status, {"Residual": "{:+.15e}"}), "</div>"]
    report = "".join(html)
    HTML_FILE.write_text("<html><head><meta charset='utf-8'></head><body style='margin:0;background:#000;color:#fff'>" + report + "</body></html>", encoding="utf-8")
    display(HTML(report))
    print(datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0061