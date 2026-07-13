# V0058
# Audit reference: Compare exact instantaneous JPL A′B′ geometry with scalar-distance and vector-closed axial/TN36 reductions at one epoch.
from __future__ import annotations

import math
import subprocess
import sys
import time
import urllib.request
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
    ("matplotlib", "matplotlib"),
    ("IPython", "ipython"),
):
    need(_module, _package)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, Image, display

warnings.filterwarnings("ignore", message=".*id_type.*deprecated.*")
warnings.filterwarnings("ignore", message=".*dubious year.*")

VERSION = "V0058"
TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUT = ROOT / "VENUS_1769_INSTANTANEOUS_VS_AXIAL_COMPARISON_V0058_OUTPUT"
CSV = OUT / "VENUS_1769_INSTANTANEOUS_VS_AXIAL_COMPARISON_V0058.csv"
HTML_FILE = OUT / "VENUS_1769_INSTANTANEOUS_VS_AXIAL_COMPARISON_V0058.html"
FIGURE = OUT / "VENUS_1769_INSTANTANEOUS_VS_AXIAL_COMPARISON_V0058.png"

EPOCH_SOURCE = (
    ROOT
    / "VENUS_1769_AXIAL_RATIO_APRIME_BPRIME_AUDIT_V0057_OUTPUT"
    / "VENUS_1769_AXIAL_RATIO_APRIME_BPRIME_AUDIT_V0057.csv"
)
POINT_SOURCE = (
    ROOT
    / "VENUS_1769_DISTANCE_RATIO_APRIME_BPRIME_AUDIT_V0055_OUTPUT"
    / "VENUS_1769_DISTANCE_RATIO_APRIME_BPRIME_AUDIT_V0055.csv"
)

SOURCE_SHA = "a85520784c131bf505cc7fd490fc1b22082f28ad"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/"
    f"{SOURCE_SHA}/VENUS_1769_APRIME_BPRIME_JPL_DISTANCE_SANITY_V0050.py"
)
SOURCE_PATH = ROOT / "VENUS_1769_APRIME_BPRIME_JPL_DISTANCE_SANITY_V0050_BASE.py"

COLORS = {
    "EV": "#65D7FF",
    "VS": "#FFD166",
    "ES": "#7EE787",
    "A": "#4DA3FF",
    "B": "#FF5A72",
    "AB": "#E8E8E8",
    "A′": "#7CFF6B",
    "B′": "#FFD84D",
    "SCALAR": "#65D7FF",
    "AXIAL": "#D2A8FF",
    "EXACT": "#FFFFFF",
    "CHECK": "#FF7B72",
}


def source_namespace() -> dict[str, object]:
    request = urllib.request.Request(
        f"{SOURCE_URL}?cache={time.time_ns()}",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        source = response.read().decode("utf-8")
    if not source.startswith("# V0050\n") or not source.rstrip().endswith("# V0050"):
        raise RuntimeError("Pinned V0050 source audit failed.")
    compile(source, str(SOURCE_PATH), "exec")
    SOURCE_PATH.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {
        "__name__": "v0050_base",
        "__file__": str(SOURCE_PATH),
    }
    exec(compile(source, str(SOURCE_PATH), "exec"), namespace)
    return namespace


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
    if not EPOCH_SOURCE.is_file():
        raise FileNotFoundError("Run V0057 first; missing source CSV: " + str(EPOCH_SOURCE))
    if not POINT_SOURCE.is_file():
        raise FileNotFoundError("Run V0055 first; missing source CSV: " + str(POINT_SOURCE))

    OUT.mkdir(parents=True, exist_ok=True)
    source = source_namespace()
    epoch_frame = pd.read_csv(EPOCH_SOURCE)
    point_frame = pd.read_csv(POINT_SOURCE)

    jd_tdb = number(
        pick(epoch_frame, "GEOMETRY", "Quantity", "Closest-approach JD TDB", "Value")
    )
    utc = str(
        pick(epoch_frame, "GEOMETRY", "Quantity", "Closest-approach UTC", "Value")
    )
    jd_points = number(
        pick(point_frame, "GEOMETRY", "Quantity", "Closest-approach JD TDB", "Value")
    )
    if abs(jd_tdb - jd_points) * 86400.0 > 1.0e-6:
        raise RuntimeError("V0055 and V0057 epochs do not match at the microsecond level.")

    a_arcsec = number(pick(point_frame, "AB_POINTS", "Point / separation", "A", "Arcseconds"))
    b_arcsec = number(pick(point_frame, "AB_POINTS", "Point / separation", "B", "Arcseconds"))
    ab_arcsec_source = number(
        pick(point_frame, "AB_POINTS", "Point / separation", "AB", "Arcseconds")
    )
    a_km_source = number(pick(point_frame, "AB_POINTS", "Point / separation", "A", "Kilometers"))
    b_km_source = number(pick(point_frame, "AB_POINTS", "Point / separation", "B", "Kilometers"))
    ab_km_source = number(
        pick(point_frame, "AB_POINTS", "Point / separation", "AB", "Kilometers")
    )

    horizons_vectors = source["horizons_vectors"]
    gnomonic = source["gnomonic"]
    norm = source["norm"]
    unit = source["unit"]
    target_ids = dict(source["TARGET_IDS"])
    sites = tuple(source["SITES"])
    arcsec_per_rad = float(source["ARCSEC_PER_RAD"])
    half_step_seconds = float(source["VELOCITY_HALF_STEP_SECONDS"])

    half_step_days = half_step_seconds / 86400.0
    query_epochs = [jd_tdb - half_step_days, jd_tdb, jd_tdb + half_step_days]

    es_vector = horizons_vectors(target_ids["SUN"], "@399", [jd_tdb])[0]
    ev_vector = horizons_vectors(target_ids["VENUS"], "@399", [jd_tdb])[0]
    vs_vector_direct = horizons_vectors(target_ids["SUN"], "@299", [jd_tdb])[0]
    vs_vector_difference = es_vector - ev_vector

    vector_closure = ev_vector + vs_vector_direct - es_vector
    vector_closure_km = norm(vector_closure)
    vs_origin_consistency_km = norm(vs_vector_direct - vs_vector_difference)

    ev = norm(ev_vector)
    vs = norm(vs_vector_direct)
    es = norm(es_vector)
    scalar_excess = ev + vs - es

    axis = unit(es_vector)
    ev_axial = float(np.dot(ev_vector, axis))
    vs_axial = float(np.dot(vs_vector_direct, axis))
    es_axial = float(np.dot(es_vector, axis))
    axial_closure = ev_axial + vs_axial - es_axial

    ev_perp = ev_vector - ev_axial * axis
    vs_perp = vs_vector_direct - vs_axial * axis
    ev_perp_km = norm(ev_perp)
    vs_perp_km = norm(vs_perp)
    transverse_closure_km = norm(ev_perp + vs_perp)

    ratio_scalar = ev / vs
    ratio_axial = ev_axial / vs_axial

    center = unit(es_vector)
    xi = np.cross(np.array([0.0, 0.0, 1.0]), center)
    if norm(xi) < 1.0e-14:
        xi = np.cross(np.array([0.0, 1.0, 0.0]), center)
    xi = unit(xi)
    eta = unit(np.cross(center, xi))

    topocentric: dict[str, dict[str, np.ndarray]] = {}
    for site in sites:
        key = str(site["key"])
        location = {
            "lon": float(site["lon"]),
            "lat": float(site["lat"]),
            "elevation": 0.0,
            "body": 399,
        }
        topocentric[key] = {
            target: horizons_vectors(target_ids[target], location, query_epochs)
            for target in ("SUN", "VENUS")
        }

    def relative_position(key: str, index: int) -> np.ndarray:
        return arcsec_per_rad * (
            gnomonic(topocentric[key]["VENUS"][index], center, xi, eta)
            - gnomonic(topocentric[key]["SUN"][index], center, xi, eta)
        )

    tracks = {
        key: np.array([relative_position(key, index) for index in range(3)], dtype=float)
        for key in ("TAHITI", "VARDO")
    }
    velocities = {
        key: (tracks[key][2] - tracks[key][0]) / (2.0 * half_step_seconds)
        for key in tracks
    }
    direction_t = unit(velocities["TAHITI"])
    direction_v = unit(velocities["VARDO"])
    if float(np.dot(direction_t, direction_v)) < 0.0:
        direction_v = -direction_v
    along = unit(direction_t + direction_v)
    normal_2d = np.array([-along[1], along[0]], dtype=float)

    q_t = tracks["TAHITI"][1]
    q_v = tracks["VARDO"][1]
    exact_delta = q_v - q_t
    if float(np.dot(exact_delta, normal_2d)) < 0.0:
        normal_2d = -normal_2d
    midpoint = 0.5 * (q_t + q_v)
    exact_a_prime_arcsec = float(np.dot(q_t - midpoint, normal_2d))
    exact_b_prime_arcsec = float(np.dot(q_v - midpoint, normal_2d))
    exact_apbp_arcsec = exact_b_prime_arcsec - exact_a_prime_arcsec

    km_per_arcsec = es / arcsec_per_rad
    ab_arcsec = ab_km_source / km_per_arcsec
    source_arcsec_difference = ab_arcsec - ab_arcsec_source
    a_arcsec_direct = -0.5 * ab_arcsec
    b_arcsec_direct = +0.5 * ab_arcsec
    a_km = -0.5 * ab_km_source
    b_km = +0.5 * ab_km_source

    ratio_exact_geometry = ab_arcsec / exact_apbp_arcsec

    scalar_a_prime_arcsec = a_arcsec_direct / ratio_scalar
    scalar_b_prime_arcsec = b_arcsec_direct / ratio_scalar
    scalar_apbp_arcsec = ab_arcsec / ratio_scalar

    axial_a_prime_arcsec = a_arcsec_direct / ratio_axial
    axial_b_prime_arcsec = b_arcsec_direct / ratio_axial
    axial_apbp_arcsec = ab_arcsec / ratio_axial

    models = [
        {
            "label": "Instantaneous JPL scalar magnitudes",
            "short": "SCALAR",
            "ratio": ratio_scalar,
            "a_prime": scalar_a_prime_arcsec,
            "b_prime": scalar_b_prime_arcsec,
            "apbp": scalar_apbp_arcsec,
        },
        {
            "label": "Vector-closed common-axis reduction",
            "short": "AXIAL",
            "ratio": ratio_axial,
            "a_prime": axial_a_prime_arcsec,
            "b_prime": axial_b_prime_arcsec,
            "apbp": axial_apbp_arcsec,
        },
        {
            "label": "Exact instantaneous topocentric JPL geometry",
            "short": "EXACT",
            "ratio": ratio_exact_geometry,
            "a_prime": exact_a_prime_arcsec,
            "b_prime": exact_b_prime_arcsec,
            "apbp": exact_apbp_arcsec,
        },
    ]

    geometry_frame = pd.DataFrame(
        [
            ["Closest-approach UTC", utc],
            ["Closest-approach JD TDB", jd_tdb],
            ["Velocity half-step seconds", half_step_seconds],
            ["Instantaneous track angle deg", math.degrees(math.atan2(along[1], along[0]))],
            ["Instantaneous common-normal angle deg", math.degrees(math.atan2(normal_2d[1], normal_2d[0]))],
            ["Planetary-vector frame", "Ecliptic J2000, geometric"],
            ["Station geometry", "Tahiti and Vardø exact topocentric JPL rays; project AB from TN36/JPL common normal"],
        ],
        columns=["Quantity", "Value"],
    )

    vector_frame = pd.DataFrame(
        [
            ["Earth → Venus", "EV", *ev_vector.tolist(), ev, "EV"],
            ["Venus → Sun", "VS", *vs_vector_direct.tolist(), vs, "VS"],
            ["Earth → Sun", "ES", *es_vector.tolist(), es, "ES"],
            ["Vector closure EV + VS − ES", "Δ", *vector_closure.tolist(), vector_closure_km, "CHECK"],
        ],
        columns=[
            "Vector",
            "Symbol",
            "X ecliptic km",
            "Y ecliptic km",
            "Z ecliptic km",
            "Magnitude km",
            "Color key",
        ],
    )

    distance_frame = pd.DataFrame(
        [
            ["Earth → Venus", "EV", ev, ev_axial, ev_axial - ev, ev_perp_km, "EV"],
            ["Venus → Sun", "VS", vs, vs_axial, vs_axial - vs, vs_perp_km, "VS"],
            ["Earth → Sun", "ES", es, es_axial, es_axial - es, 0.0, "ES"],
            ["Closure / excess", "CHECK", scalar_excess, axial_closure, axial_closure - scalar_excess, transverse_closure_km, "CHECK"],
        ],
        columns=[
            "Distance",
            "Symbol",
            "Instantaneous scalar km",
            "Vector-closed axial km",
            "Axial minus scalar km",
            "Transverse magnitude / closure km",
            "Color key",
        ],
    )

    ratio_rows = []
    for model in models:
        ratio_rows.append(
            [
                model["label"],
                model["short"],
                model["ratio"],
                model["ratio"] - ratio_scalar,
                1.0e6 * (model["ratio"] / ratio_scalar - 1.0),
                model["short"],
            ]
        )
    ratio_frame = pd.DataFrame(
        ratio_rows,
        columns=[
            "Reduction model",
            "Ratio equation",
            "Ratio value",
            "Ratio minus scalar",
            "Difference from scalar ppm",
            "Color key",
        ],
    )

    ab_frame = pd.DataFrame(
        [
            ["A", a_arcsec_direct, a_km, "A"],
            ["B", b_arcsec_direct, b_km, "B"],
            ["AB", ab_arcsec, ab_km_source, "AB"],
        ],
        columns=["Point / separation", "Arcseconds", "Kilometers", "Color key"],
    )

    prime_rows = []
    for model in models:
        prime_rows.extend(
            [
                [model["label"], "A′", model["a_prime"], model["a_prime"] * km_per_arcsec, model["short"]],
                [model["label"], "B′", model["b_prime"], model["b_prime"] * km_per_arcsec, model["short"]],
                [model["label"], "A′B′", model["apbp"], model["apbp"] * km_per_arcsec, model["short"]],
            ]
        )
    prime_frame = pd.DataFrame(
        prime_rows,
        columns=["Reduction model", "Point / separation", "Arcseconds", "Kilometers", "Color key"],
    )

    reduction_rows = []
    for model in models:
        reduced_ab_arcsec = model["apbp"] * model["ratio"]
        reduced_ab_km = reduced_ab_arcsec * km_per_arcsec
        reduction_rows.append(
            [
                model["label"],
                model["ratio"],
                model["apbp"],
                model["apbp"] * km_per_arcsec,
                reduced_ab_arcsec,
                reduced_ab_km,
                reduced_ab_km - ab_km_source,
                model["apbp"] - exact_apbp_arcsec,
                (model["apbp"] - exact_apbp_arcsec) * km_per_arcsec,
                model["short"],
            ]
        )
    reduction_frame = pd.DataFrame(
        reduction_rows,
        columns=[
            "Reduction model",
            "Ratio",
            "A′B′ arcsec",
            "A′B′ km",
            "Back-reduced AB arcsec",
            "Back-reduced AB km",
            "AB residual km",
            "A′B′ minus exact arcsec",
            "A′B′ minus exact km",
            "Color key",
        ],
    )

    status_frame = pd.DataFrame(
        [
            ["Independent JPL vector closure", "PASS" if vector_closure_km < 0.01 else "FAIL", vector_closure_km, "km"],
            ["Direct Venus-center VS agrees with ES − EV", "PASS" if vs_origin_consistency_km < 0.01 else "FAIL", vs_origin_consistency_km, "km"],
            ["Vector-closed axial distance sum", "PASS" if abs(axial_closure) < 1.0e-6 else "FAIL", axial_closure, "km"],
            ["Source AB arcsecond-scale consistency", "PASS" if abs(source_arcsec_difference) < 1.0e-9 else "FAIL", source_arcsec_difference, "arcsec"],
            ["Scalar model algebraic AB closure", "PASS" if abs(reduction_frame.iloc[0]["AB residual km"]) < 1.0e-8 else "FAIL", reduction_frame.iloc[0]["AB residual km"], "km"],
            ["Axial model algebraic AB closure", "PASS" if abs(reduction_frame.iloc[1]["AB residual km"]) < 1.0e-8 else "FAIL", reduction_frame.iloc[1]["AB residual km"], "km"],
            ["Exact instantaneous geometric AB closure", "PASS" if abs(reduction_frame.iloc[2]["AB residual km"]) < 1.0e-8 else "FAIL", reduction_frame.iloc[2]["AB residual km"], "km"],
        ],
        columns=["Equation / test", "Status", "Residual", "Unit"],
    )

    plot_x = np.array([1000.0 * model["ratio"] for model in models], dtype=float)
    plot_y = np.array([model["apbp"] for model in models], dtype=float)
    plot_labels = ["Scalar", "Axial", "Exact"]
    plot_colors = [COLORS["SCALAR"], COLORS["AXIAL"], COLORS["EXACT"]]

    figure, axis_plot = plt.subplots(figsize=(10.0, 6.0))
    figure.patch.set_facecolor("black")
    axis_plot.set_facecolor("black")
    axis_plot.plot(plot_x, plot_y, linewidth=0.8, color="#888888")
    for x_value, y_value, label, color in zip(plot_x, plot_y, plot_labels, plot_colors):
        axis_plot.plot(x_value, y_value, marker="o", markersize=4.0, color=color)
        axis_plot.annotate(
            label,
            (x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
            color=color,
            fontsize=8,
        )
    axis_plot.set_xlabel("Reduction ratio × 1000", color="white")
    axis_plot.set_ylabel("A′B′ separation (arcsec)", color="white")
    axis_plot.set_title("1769 instantaneous ratio versus A′B′ comparison", color="white")
    axis_plot.tick_params(colors="white", width=0.6)
    for spine in axis_plot.spines.values():
        spine.set_color("white")
        spine.set_linewidth(0.6)
    axis_plot.grid(True, linewidth=0.3, alpha=0.25)
    axis_plot.set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=240, bbox_inches="tight", facecolor="black")
    plt.close(figure)

    records: list[dict[str, object]] = []
    for section, frame in (
        ("GEOMETRY", geometry_frame),
        ("VECTORS", vector_frame.drop(columns=["Color key"])),
        ("DISTANCE_COMPARISON", distance_frame.drop(columns=["Color key"])),
        ("RATIOS", ratio_frame.drop(columns=["Color key"])),
        ("AB_POINTS", ab_frame.drop(columns=["Color key"])),
        ("APRIME_BPRIME", prime_frame.drop(columns=["Color key"])),
        ("REDUCTIONS", reduction_frame.drop(columns=["Color key"])),
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
.r *{background:#000;color:#fff;box-sizing:border-box}.r h1{font-size:22px;border-bottom:2px solid #fff;padding-bottom:8px}
.r h2{font-size:16px;border-top:1px solid #fff;border-bottom:1px solid #fff;padding:5px 0;margin-top:22px}.r h3{font-size:14px}.r p{font-size:13px;line-height:1.45}
.wrap{overflow-x:auto}.audit{border-collapse:collapse;min-width:100%;width:max-content;font-size:12px}.audit th,.audit td{border:1px solid #fff;padding:7px 9px;white-space:nowrap}
.audit th{font-weight:700;text-align:center}.audit td{text-align:right}.audit td:first-child,.audit td:nth-child(2){text-align:left}
.note{border:1px solid #fff;padding:9px;font-weight:600}.answer{font-size:18px;border:2px solid #fff;padding:12px;font-weight:700;text-align:center}
.path{font-family:monospace;font-size:11px;overflow-wrap:anywhere;white-space:normal}.legend{display:flex;flex-wrap:wrap;gap:12px;border:1px solid #fff;padding:9px}
</style>
"""
    legend = (
        '<div class="legend">'
        + "".join(
            f'<span style="color:{color};font-weight:700">{key}</span>'
            for key, color in COLORS.items()
        )
        + "</div>"
    )

    html = [css, '<div class="r">', "<h1>1769 Venus Transit — Instantaneous versus Axial/TN36 Comparison</h1>"]
    html += [
        "<h2>CODE INPUTS</h2>",
        f"<p><b>Epoch source:</b> {EPOCH_SOURCE}</p>",
        f"<p><b>Point source:</b> {POINT_SOURCE}</p>",
        f"<p><b>Exact epoch:</b> {utc}</p>",
        "<p><b>Fresh JPL queries:</b> EV from Earth center, VS from Venus center, ES from Earth center, plus Tahiti/Vardø Sun and Venus rays at t−0.5 s, t, and t+0.5 s.</p>",
    ]
    html += [
        "<h2>COMMENTS</h2>",
        '<p class="note">All planetary vectors and both topocentric A′/B′ rays are evaluated at the identical TDB epoch. The TN36/project contribution is the common-normal station baseline AB; the scalar and axial planetary distances are freshly recomputed from JPL.</p>',
        '<p class="note">Three reductions are displayed: instantaneous scalar magnitudes, vector-closed Earth–Sun-axis distances, and the exact instantaneous finite topocentric geometry.</p>',
        legend,
    ]
    html += [
        "<h2>RESULTS</h2>",
        "<h3>Epoch and geometry</h3>",
        colored_table(geometry_frame, {"Value": "{:,.15f}"}),
        "<h3>1. Fresh instantaneous JPL vectors</h3>",
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
        "<h3>2. Instantaneous scalar versus vector-closed axial distances</h3>",
        colored_table(
            distance_frame,
            {
                "Instantaneous scalar km": "{:+,.12f}",
                "Vector-closed axial km": "{:+,.12f}",
                "Axial minus scalar km": "{:+,.12f}",
                "Transverse magnitude / closure km": "{:+,.12f}",
            },
            "Color key",
        ),
        "<h3>3. Ratio comparison</h3>",
        colored_table(
            ratio_frame,
            {
                "Ratio value": "{:+.15f}",
                "Ratio minus scalar": "{:+.15f}",
                "Difference from scalar ppm": "{:+.9f}",
            },
            "Color key",
        ),
        "<h3>4. Direct project A, B, and AB</h3>",
        colored_table(
            ab_frame,
            {"Arcseconds": "{:+.15f}", "Kilometers": "{:+,.12f}"},
            "Color key",
        ),
        "<h3>5. A′, B′, and A′B′ at the identical instantaneous epoch</h3>",
        colored_table(
            prime_frame,
            {"Arcseconds": "{:+.15f}", "Kilometers": "{:+,.12f}"},
            "Color key",
        ),
        "<h3>6. Reduction comparison</h3>",
        colored_table(
            reduction_frame,
            {
                "Ratio": "{:+.15f}",
                "A′B′ arcsec": "{:+.15f}",
                "A′B′ km": "{:+,.12f}",
                "Back-reduced AB arcsec": "{:+.15f}",
                "Back-reduced AB km": "{:+,.12f}",
                "AB residual km": "{:+.15e}",
                "A′B′ minus exact arcsec": "{:+.15f}",
                "A′B′ minus exact km": "{:+,.12f}",
            },
            "Color key",
        ),
        f'<p class="answer">Instantaneous exact A′B′ = {exact_apbp_arcsec:.15f}″ = {exact_apbp_arcsec * km_per_arcsec:,.12f} km; axial-ratio A′B′ = {axial_apbp_arcsec:.15f}″ = {axial_apbp_arcsec * km_per_arcsec:,.12f} km.</p>',
    ]
    html += [
        "<h2>OUTPUT SUMMARY</h2>",
        f'<p class="path">{CSV}</p>',
        f'<p class="path">{HTML_FILE}</p>',
        f'<p class="path">{FIGURE}</p>',
    ]
    html += [
        "<h2>PAPER COMPARISON</h2>",
        '<p class="note">NOT USED. This module compares only fresh instantaneous JPL vectors, vector-closed axial distances, and the project TN36/JPL common-normal A/B geometry.</p>',
    ]
    html += [
        "<h2>EQUATION STATUS</h2>",
        colored_table(status_frame, {"Residual": "{:+.15e}"}),
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
    display(Image(filename=str(FIGURE)))
    print(datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0058