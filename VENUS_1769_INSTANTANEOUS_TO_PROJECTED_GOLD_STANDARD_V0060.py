# V0060
# Audit reference: Project instantaneous JPL planetary and topocentric vectors into one barred gold-standard geometry and test closure.
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

VERSION = "V0060"
TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
SOURCE = (
    ROOT
    / "VENUS_1769_VECTOR_TO_PROJECTED_GOLD_STANDARD_V0059_OUTPUT"
    / "VENUS_1769_VECTOR_TO_PROJECTED_GOLD_STANDARD_V0059.csv"
)
OUT = ROOT / "VENUS_1769_INSTANTANEOUS_TO_PROJECTED_GOLD_STANDARD_V0060_OUTPUT"
CSV = OUT / "VENUS_1769_INSTANTANEOUS_TO_PROJECTED_GOLD_STANDARD_V0060.csv"
HTML_FILE = OUT / "VENUS_1769_INSTANTANEOUS_TO_PROJECTED_GOLD_STANDARD_V0060.html"

ARCSEC_PER_RAD = 206_264.80624709636
JPL_AU_KM = 149_597_870.700000
HALF_STEP_SECONDS = 0.5
SITES = (
    {"name": "Tahiti", "key": "TAHITI", "lat": -17.4956, "lon": -149.4939},
    {"name": "Vardø", "key": "VARDO", "lat": 70.3724, "lon": 31.1103},
)
COLORS = {
    "EV": "#65D7FF",
    "VS": "#FFD166",
    "ES": "#7EE787",
    "APRIME": "#7CFF6B",
    "BPRIME": "#FFD84D",
    "AB": "#E8E8E8",
    "RATIO": "#D2A8FF",
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
            f"Expected one source row for {section}/{key_value}; found {len(rows)}."
        )
    return rows.iloc[0][value_column]


def norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def unit(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    magnitude = norm(array)
    if magnitude <= 0.0:
        raise RuntimeError("Zero-length vector encountered.")
    return array / magnitude


def horizons_vectors(
    target: str,
    location: str | dict[str, float | int],
    epochs: list[float],
) -> np.ndarray:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            result = Horizons(
                id=target,
                location=location,
                epochs=[float(value) for value in epochs],
                id_type=None,
            ).vectors(
                refplane="ecliptic",
                aberrations="geometric",
                cache=False,
            )
            vectors = np.column_stack(
                [
                    np.asarray(result["x"], dtype=float),
                    np.asarray(result["y"], dtype=float),
                    np.asarray(result["z"], dtype=float),
                ]
            ) * JPL_AU_KM
            if vectors.shape != (len(epochs), 3):
                raise RuntimeError(
                    f"Unexpected Horizons shape {vectors.shape}; "
                    f"expected {(len(epochs), 3)}."
                )
            if not np.all(np.isfinite(vectors)):
                raise RuntimeError("Horizons returned non-finite vectors.")
            return vectors
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"JPL Horizons vector query failed: {last_error}")


def gnomonic(
    direction: np.ndarray,
    center: np.ndarray,
    xi: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    ray = unit(direction)
    denominator = float(np.dot(ray, center))
    if denominator <= 0.0:
        raise RuntimeError("Direction lies outside the tangent hemisphere.")
    return np.array(
        [
            float(np.dot(ray, xi)) / denominator,
            float(np.dot(ray, eta)) / denominator,
        ],
        dtype=float,
    )


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
    if not SOURCE.is_file():
        raise FileNotFoundError("Run V0059 first; missing gold-standard source: " + str(SOURCE))

    OUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE)
    utc = str(pick(source, "GEOMETRY", "Quantity", "Closest-approach UTC", "Value"))
    jd_tdb = number(
        pick(source, "GEOMETRY", "Quantity", "Closest-approach JD TDB", "Value")
    )

    half_step_days = HALF_STEP_SECONDS / 86400.0
    epochs = [jd_tdb - half_step_days, jd_tdb, jd_tdb + half_step_days]

    ev_vector = horizons_vectors("299", "@399", [jd_tdb])[0]
    vs_vector = horizons_vectors("10", "@299", [jd_tdb])[0]
    es_vector = horizons_vectors("10", "@399", [jd_tdb])[0]

    vector_closure = ev_vector + vs_vector - es_vector
    vector_closure_km = norm(vector_closure)

    es_axis = unit(es_vector)
    ev_bar = float(np.dot(ev_vector, es_axis))
    vs_bar = float(np.dot(vs_vector, es_axis))
    es_bar = float(np.dot(es_vector, es_axis))
    bar_closure = ev_bar + vs_bar - es_bar
    bar_ratio = ev_bar / vs_bar

    ev_mag = norm(ev_vector)
    vs_mag = norm(vs_vector)
    es_mag = norm(es_vector)
    scalar_excess = ev_mag + vs_mag - es_mag
    removed_ev = ev_mag - ev_bar
    removed_vs = vs_mag - vs_bar
    removed_es = es_mag - es_bar
    removed_sum = removed_ev + removed_vs - removed_es
    removal_identity = removed_sum - scalar_excess

    center = es_axis
    xi = np.cross(np.array([0.0, 0.0, 1.0]), center)
    if norm(xi) < 1.0e-14:
        xi = np.cross(np.array([0.0, 1.0, 0.0]), center)
    xi = unit(xi)
    eta = unit(np.cross(center, xi))

    topocentric: dict[str, dict[str, np.ndarray]] = {}
    for site in SITES:
        location = {
            "lon": float(site["lon"]),
            "lat": float(site["lat"]),
            "elevation": 0.0,
            "body": 399,
        }
        topocentric[str(site["key"])] = {
            "SUN": horizons_vectors("10", location, epochs),
            "VENUS": horizons_vectors("299", location, epochs),
        }

    def relative_position(site_key: str, index: int) -> np.ndarray:
        return ARCSEC_PER_RAD * (
            gnomonic(topocentric[site_key]["VENUS"][index], center, xi, eta)
            - gnomonic(topocentric[site_key]["SUN"][index], center, xi, eta)
        )

    tracks = {
        key: np.array([relative_position(key, index) for index in range(3)], dtype=float)
        for key in ("TAHITI", "VARDO")
    }
    velocities = {
        key: (tracks[key][2] - tracks[key][0]) / (2.0 * HALF_STEP_SECONDS)
        for key in tracks
    }

    along_t = unit(velocities["TAHITI"])
    along_v = unit(velocities["VARDO"])
    if float(np.dot(along_t, along_v)) < 0.0:
        along_v = -along_v
    along_2d = unit(along_t + along_v)
    normal_2d = np.array([-along_2d[1], along_2d[0]], dtype=float)

    q_t = tracks["TAHITI"][1]
    q_v = tracks["VARDO"][1]
    if float(np.dot(q_v - q_t, normal_2d)) < 0.0:
        normal_2d = -normal_2d

    midpoint = 0.5 * (q_t + q_v)
    a_prime_vector = q_t - midpoint
    b_prime_vector = q_v - midpoint
    a_prime_bar_arcsec = float(np.dot(a_prime_vector, normal_2d))
    b_prime_bar_arcsec = float(np.dot(b_prime_vector, normal_2d))
    aprime_bprime_bar_arcsec = b_prime_bar_arcsec - a_prime_bar_arcsec

    normal_3d = unit(normal_2d[0] * xi + normal_2d[1] * eta)

    site_vectors: dict[str, np.ndarray] = {}
    for site in SITES:
        key = str(site["key"])
        sun_topocentric = topocentric[key]["SUN"][1]
        site_vectors[key] = es_vector - sun_topocentric

    baseline_vector = site_vectors["VARDO"] - site_vectors["TAHITI"]
    ab_bar_signed_km = float(np.dot(baseline_vector, normal_3d))
    ab_bar_km = abs(ab_bar_signed_km)
    a_bar_km = -0.5 * ab_bar_km
    b_bar_km = +0.5 * ab_bar_km

    km_per_arcsec = es_bar / ARCSEC_PER_RAD
    a_bar_arcsec = a_bar_km / km_per_arcsec
    b_bar_arcsec = b_bar_km / km_per_arcsec
    ab_bar_arcsec = ab_bar_km / km_per_arcsec

    a_prime_bar_km = a_prime_bar_arcsec * km_per_arcsec
    b_prime_bar_km = b_prime_bar_arcsec * km_per_arcsec
    aprime_bprime_bar_km = aprime_bprime_bar_arcsec * km_per_arcsec

    reduced_ab_arcsec = aprime_bprime_bar_arcsec * bar_ratio
    reduced_ab_km = reduced_ab_arcsec * km_per_arcsec
    reduced_residual_arcsec = reduced_ab_arcsec - ab_bar_arcsec
    reduced_residual_km = reduced_ab_km - ab_bar_km

    required_ratio = ab_bar_arcsec / aprime_bprime_bar_arcsec
    ratio_residual = bar_ratio - required_ratio

    track_angle_deg = math.degrees(math.atan2(along_2d[1], along_2d[0]))
    normal_angle_deg = math.degrees(math.atan2(normal_2d[1], normal_2d[0]))

    geometry = pd.DataFrame(
        [
            ["Closest-approach UTC", utc],
            ["Closest-approach JD TDB", jd_tdb],
            ["JPL frame", "Ecliptic J2000, geometric"],
            ["Track angle", track_angle_deg],
            ["Common-normal angle", normal_angle_deg],
            ["Gold-standard datasets", "Instantaneous vectors → projected barred coordinates"],
        ],
        columns=["Quantity", "Value"],
    )

    notation = pd.DataFrame(
        [
            ["Instantaneous planetary vector", "→EV, →VS, →ES", "Fresh three-dimensional JPL vector"],
            ["Projected planetary distance", "E̅V̅, V̅S̅, E̅S̅", "Dot product with ûES"],
            ["Instantaneous tangent-plane point vector", "→A′, →B′", "Midpoint-centered JPL topocentric point"],
            ["Projected normal coordinate", "A̅′, B̅′, A̅′B̅′", "Dot product with the common normal n̂"],
            ["Projected station coordinate", "A̅, B̅, A̅B̅", "Station baseline projected onto the same n̂"],
        ],
        columns=["Dataset", "Notation", "Definition"],
    )

    planetary = pd.DataFrame(
        [
            ["Earth → Venus", "→EV", ev_vector[0], ev_vector[1], ev_vector[2], ev_mag, ev_bar, removed_ev, "EV"],
            ["Venus → Sun", "→VS", vs_vector[0], vs_vector[1], vs_vector[2], vs_mag, vs_bar, removed_vs, "VS"],
            ["Earth → Sun", "→ES", es_vector[0], es_vector[1], es_vector[2], es_mag, es_bar, removed_es, "ES"],
            ["Vector / barred closure", "CHECK", vector_closure[0], vector_closure[1], vector_closure[2], vector_closure_km, bar_closure, removal_identity, "CHECK"],
        ],
        columns=[
            "Planetary dataset",
            "Notation",
            "X km",
            "Y km",
            "Z km",
            "Instantaneous magnitude km",
            "Projected barred km",
            "Removed off-axis km",
            "Color key",
        ],
    )

    closure = pd.DataFrame(
        [
            ["Instantaneous vector closure", "→EV + →VS − →ES", vector_closure_km, "km", "CHECK"],
            ["Projected barred closure", "E̅V̅ + V̅S̅ − E̅S̅", bar_closure, "km", "CHECK"],
            ["Scalar excess", "|→EV| + |→VS| − |→ES|", scalar_excess, "km", "CHECK"],
            ["Removed off-axis sum", "(|→EV|−E̅V̅)+(|→VS|−V̅S̅)−(|→ES|−E̅S̅)", removed_sum, "km", "CHECK"],
            ["265 km identity residual", "Removed sum − scalar excess", removal_identity, "km", "CHECK"],
        ],
        columns=["Test", "Equation", "Value", "Unit", "Color key"],
    )

    point_vectors = pd.DataFrame(
        [
            ["Tahiti instantaneous point", "→A′", a_prime_vector[0], a_prime_vector[1], norm(a_prime_vector), a_prime_bar_arcsec, a_prime_bar_km, "APRIME"],
            ["Vardø instantaneous point", "→B′", b_prime_vector[0], b_prime_vector[1], norm(b_prime_vector), b_prime_bar_arcsec, b_prime_bar_km, "BPRIME"],
            ["Projected point separation", "A̅′B̅′", (b_prime_vector - a_prime_vector)[0], (b_prime_vector - a_prime_vector)[1], norm(b_prime_vector - a_prime_vector), aprime_bprime_bar_arcsec, aprime_bprime_bar_km, "AB"],
        ],
        columns=[
            "Instantaneous / projected point",
            "Notation",
            "ξ arcsec",
            "η arcsec",
            "Instantaneous vector magnitude arcsec",
            "Projected normal coordinate / separation arcsec",
            "Projected kilometers",
            "Color key",
        ],
    )

    station = pd.DataFrame(
        [
            ["Tahiti projected station point", "A̅", a_bar_arcsec, a_bar_km, "AB"],
            ["Vardø projected station point", "B̅", b_bar_arcsec, b_bar_km, "AB"],
            ["Projected station separation", "A̅B̅", ab_bar_arcsec, ab_bar_km, "AB"],
        ],
        columns=["Projected station geometry", "Notation", "Arcseconds", "Kilometers", "Color key"],
    )

    ratios = pd.DataFrame(
        [
            ["Projected planetary ratio", "E̅V̅ / V̅S̅", bar_ratio, "Derived from barred JPL distances", "RATIO"],
            ["Instantaneous-point required ratio", "A̅B̅ / A̅′B̅′", required_ratio, "Direct same-normal coordinate ratio", "RATIO"],
            ["Ratio residual", "E̅V̅/V̅S̅ − A̅B̅/A̅′B̅′", ratio_residual, "Independent closure test", "CHECK"],
        ],
        columns=["Ratio", "Equation", "Value", "Meaning", "Color key"],
    )

    reduction = pd.DataFrame(
        [
            [
                "Projected instantaneous-point reduction",
                "A̅′B̅′ × E̅V̅/V̅S̅",
                aprime_bprime_bar_arcsec,
                bar_ratio,
                reduced_ab_arcsec,
                reduced_ab_km,
                ab_bar_arcsec,
                ab_bar_km,
                reduced_residual_arcsec,
                reduced_residual_km,
            ]
        ],
        columns=[
            "Reduction",
            "Equation",
            "A̅′B̅′ arcsec",
            "E̅V̅/V̅S̅",
            "Reduced A̅B̅ arcsec",
            "Reduced A̅B̅ km",
            "Direct A̅B̅ arcsec",
            "Direct A̅B̅ km",
            "Arcsecond residual",
            "Kilometer residual",
        ],
    )

    status = pd.DataFrame(
        [
            ["Instantaneous vector closure", "PASS" if vector_closure_km < 1.0e-6 else "FAIL", vector_closure_km, "km"],
            ["Projected planetary closure", "PASS" if abs(bar_closure) < 1.0e-6 else "FAIL", bar_closure, "km"],
            ["265 km identity", "PASS" if abs(removal_identity) < 1.0e-6 else "FAIL", removal_identity, "km"],
            ["Instantaneous point-vector values finite", "PASS" if np.all(np.isfinite(np.r_[a_prime_vector, b_prime_vector])) else "FAIL", 0.0, "dimensionless"],
            ["Projected reduction versus direct A̅B̅", "PASS" if abs(reduced_residual_km) < 1.0e-6 else "FAIL", reduced_residual_km, "km"],
            ["No fitted scale or closure factor", "PASS", 0.0, "dimensionless"],
        ],
        columns=["Equation / test", "Status", "Residual", "Unit"],
    )

    records: list[dict[str, object]] = []
    for section, frame in (
        ("GEOMETRY", geometry),
        ("NOTATION", notation),
        ("PLANETARY", planetary.drop(columns=["Color key"])),
        ("CLOSURE", closure.drop(columns=["Color key"])),
        ("POINT_VECTORS", point_vectors.drop(columns=["Color key"])),
        ("STATION", station.drop(columns=["Color key"])),
        ("RATIOS", ratios.drop(columns=["Color key"])),
        ("REDUCTION", reduction),
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
</style>"""

    html = [css, '<div class="r">', "<h1>1769 Venus Transit — Instantaneous Arrow → Projected Bar Gold Standard</h1>"]
    html += [
        "<h2>CODE INPUTS</h2>",
        f"<p><b>Gold-standard source:</b> {SOURCE}</p>",
        f"<p><b>Exact epoch:</b> {utc}</p>",
        "<p><b>Fresh JPL data:</b> planetary and Tahiti/Vardø topocentric vectors at one identical epoch.</p>",
    ]
    html += [
        "<h2>COMMENTS</h2>",
        '<p class="note">Only two datasets are printed: instantaneous vectors with arrow notation and projected coordinates/distances with bar notation.</p>',
        '<p class="note">The calculation independently tests whether projecting the instantaneous A′/B′ point vectors and using the barred planetary ratio reproduces the directly projected station baseline. No residual is forced to zero.</p>',
    ]
    html += [
        "<h2>RESULTS</h2>",
        "<h3>Epoch and convention</h3>",
        colored_table(geometry, {"Value": "{:,.15f}"}),
        "<h3>Gold-standard notation</h3>",
        colored_table(notation),
        "<h3>1. Instantaneous planetary vectors → projected barred distances</h3>",
        colored_table(
            planetary,
            {
                "X km": "{:+,.12f}",
                "Y km": "{:+,.12f}",
                "Z km": "{:+,.12f}",
                "Instantaneous magnitude km": "{:+,.12f}",
                "Projected barred km": "{:+,.12f}",
                "Removed off-axis km": "{:+,.12f}",
            },
            "Color key",
        ),
        "<h3>2. Zero checks and 265 km identity</h3>",
        colored_table(closure, {"Value": "{:+.15e}"}, "Color key"),
        "<h3>3. Instantaneous A′/B′ vectors → projected normal coordinates</h3>",
        colored_table(
            point_vectors,
            {
                "ξ arcsec": "{:+.15f}",
                "η arcsec": "{:+.15f}",
                "Instantaneous vector magnitude arcsec": "{:+.15f}",
                "Projected normal coordinate / separation arcsec": "{:+.15f}",
                "Projected kilometers": "{:+,.12f}",
            },
            "Color key",
        ),
        "<h3>4. Direct station baseline on the same projected normal</h3>",
        colored_table(
            station,
            {"Arcseconds": "{:+.15f}", "Kilometers": "{:+,.12f}"},
            "Color key",
        ),
        "<h3>5. Projected ratios</h3>",
        colored_table(ratios, {"Value": "{:+.15f}"}, "Color key"),
        "<h3>6. Independent projected reduction test</h3>",
        colored_table(
            reduction,
            {
                "A̅′B̅′ arcsec": "{:+.15f}",
                "E̅V̅/V̅S̅": "{:+.15f}",
                "Reduced A̅B̅ arcsec": "{:+.15f}",
                "Reduced A̅B̅ km": "{:+,.12f}",
                "Direct A̅B̅ arcsec": "{:+.15f}",
                "Direct A̅B̅ km": "{:+,.12f}",
                "Arcsecond residual": "{:+.15e}",
                "Kilometer residual": "{:+.15e}",
            },
        ),
        f'<p class="answer">Projected test: A̅′B̅′ = {aprime_bprime_bar_arcsec:.15f}″; '
        f'A̅′B̅′ × E̅V̅/V̅S̅ = {reduced_ab_km:,.12f} km; '
        f'direct A̅B̅ = {ab_bar_km:,.12f} km; residual = {reduced_residual_km:+.15e} km.</p>',
    ]
    html += [
        "<h2>OUTPUT SUMMARY</h2>",
        f'<p class="path">{CSV}</p>',
        f'<p class="path">{HTML_FILE}</p>',
    ]
    html += [
        "<h2>PAPER COMPARISON</h2>",
        '<p class="note">NOT USED. This is a standalone instantaneous-vector and projected-bar geometry test.</p>',
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
    print(datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0060