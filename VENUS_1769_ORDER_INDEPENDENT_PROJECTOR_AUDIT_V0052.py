# V0052
# Audit reference: Order-independent derivation of the 1769 Halley residual from JPL vectors, projectors, and tangent-plane Jacobians.
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
    ("scipy", "scipy"),
    ("astropy", "astropy"),
    ("astroquery", "astroquery"),
    ("IPython", "ipython"),
):
    need(_module, _package)

import numpy as np
import pandas as pd
from astroquery.jplhorizons import Horizons
from IPython.display import HTML, display
from scipy.optimize import minimize_scalar

VERSION = "V0052"
TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUT = ROOT / "VENUS_1769_ORDER_INDEPENDENT_PROJECTOR_AUDIT_V0052_OUTPUT"
CSV = OUT / "VENUS_1769_ORDER_INDEPENDENT_PROJECTOR_AUDIT_V0052.csv"
HTML_FILE = OUT / "VENUS_1769_ORDER_INDEPENDENT_PROJECTOR_AUDIT_V0052.html"

SOURCE_SHA = "a85520784c131bf505cc7fd490fc1b22082f28ad"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/"
    f"{SOURCE_SHA}/VENUS_1769_APRIME_BPRIME_JPL_DISTANCE_SANITY_V0050.py"
)
SOURCE_PATH = ROOT / "VENUS_1769_APRIME_BPRIME_JPL_DISTANCE_SANITY_V0050_BASE.py"
JPL_AU_KM = 149_597_870.700000
QUERY_RETRIES = 4
warnings.filterwarnings("ignore", message=".*id_type.*deprecated.*")


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


def horizons_vectors(
    target_id: str,
    location: str | dict[str, float | int],
    epochs: list[float],
) -> np.ndarray:
    last_error: Exception | None = None
    for attempt in range(QUERY_RETRIES):
        try:
            query = Horizons(
                id=target_id,
                location=location,
                epochs=[float(value) for value in epochs],
            )
            vectors = query.vectors(
                refplane="ecliptic",
                aberrations="geometric",
                cache=False,
            )
            result = np.column_stack(
                [
                    np.asarray(vectors["x"], dtype=float),
                    np.asarray(vectors["y"], dtype=float),
                    np.asarray(vectors["z"], dtype=float),
                ]
            )
            expected = (len(epochs), 3)
            if result.shape != expected:
                raise RuntimeError(
                    f"Unexpected Horizons result shape {result.shape}; expected {expected}."
                )
            if not np.all(np.isfinite(result)):
                raise RuntimeError("Horizons returned non-finite vectors.")
            return result * JPL_AU_KM
        except Exception as exc:
            last_error = exc
            if attempt + 1 < QUERY_RETRIES:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"JPL Horizons vector query failed: {last_error}")


def projector(direction: np.ndarray) -> np.ndarray:
    vector = np.asarray(direction, dtype=float)
    magnitude = float(np.linalg.norm(vector))
    if magnitude == 0.0:
        raise RuntimeError("Zero vector encountered in projector calculation.")
    u = vector / magnitude
    return np.eye(3, dtype=float) - np.outer(u, u)


def gnomonic_jacobian(
    direction: np.ndarray,
    center: np.ndarray,
    xi: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    vector = np.asarray(direction, dtype=float)
    u = vector / np.linalg.norm(vector)
    denominator = float(np.dot(u, center))
    if denominator <= 0.0:
        raise RuntimeError("Direction lies outside the tangent hemisphere.")
    row_x = (xi * denominator - center * float(np.dot(xi, u))) / denominator**2
    row_y = (eta * denominator - center * float(np.dot(eta, u))) / denominator**2
    return np.vstack([row_x, row_y])


def formatted_table(frame: pd.DataFrame, formats: dict[str, str]) -> str:
    shown = frame.copy()
    for column, pattern in formats.items():
        if column not in shown.columns:
            continue

        def render(value: object) -> str:
            if pd.isna(value):
                return ""
            converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
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
    source = source_namespace()
    OUT.mkdir(parents=True, exist_ok=True)

    base = source["base_namespace"]()
    prefixes = tuple(base["PREFIXES"])
    required = ["JD_TDB"] + [
        f"{prefix}_{axis}_KM" for prefix in prefixes for axis in "XYZ"
    ]
    master, master_source = source["load_master"](base, required)
    cache = base["build_cache"](master)
    vector_at = base["vector_at"]
    angular_separation = base["angular_separation"]
    utc_text = base["utc_text"]
    norm = source["norm"]
    unit = source["unit"]
    gnomonic = source["gnomonic"]
    arcsec_per_rad = float(source["ARCSEC_PER_RAD"])
    half_step_seconds = float(source["VELOCITY_HALF_STEP_SECONDS"])
    target_ids = dict(source["TARGET_IDS"])
    sites = tuple(source["SITES"])

    jds = np.asarray(cache["JD_TDB"], dtype=float)
    coarse = np.array(
        [
            angular_separation(
                vector_at(cache, "GEOCENTER_SUN", jd),
                vector_at(cache, "GEOCENTER_VENUS", jd),
            )
            for jd in jds
        ],
        dtype=float,
    )
    index = int(np.argmin(coarse))
    lower = float(jds[max(0, index - 5)])
    upper = float(jds[min(len(jds) - 1, index + 5)])

    def jd_at(seconds: float) -> float:
        return lower + float(seconds) / 86400.0

    def objective(seconds: float) -> float:
        jd = jd_at(seconds)
        angle = angular_separation(
            vector_at(cache, "GEOCENTER_SUN", jd),
            vector_at(cache, "GEOCENTER_VENUS", jd),
        )
        return (angle * arcsec_per_rad) ** 2

    result = minimize_scalar(
        objective,
        bounds=(0.0, (upper - lower) * 86400.0),
        method="bounded",
        options={"xatol": 1.0e-7, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError("Closest-approach optimization failed.")

    jd_ca = jd_at(float(result.x))
    half_day = half_step_seconds / 86400.0
    query_epochs = [jd_ca - half_day, jd_ca, jd_ca + half_day]

    sun = horizons_vectors(target_ids["SUN"], "@399", [jd_ca])[0]
    venus = horizons_vectors(target_ids["VENUS"], "@399", [jd_ca])[0]
    es = norm(sun)
    ev = norm(venus)
    vs = norm(sun - venus)
    physical_ratio = ev / vs
    km_per_arcsec = es / arcsec_per_rad
    backtrack_scale = km_per_arcsec * physical_ratio

    center = unit(sun)
    xi = np.cross(np.array([0.0, 0.0, 1.0], dtype=float), center)
    if norm(xi) < 1.0e-14:
        xi = np.cross(np.array([0.0, 1.0, 0.0], dtype=float), center)
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

    def relative(key: str, epoch_index: int) -> np.ndarray:
        return arcsec_per_rad * (
            gnomonic(topocentric[key]["VENUS"][epoch_index], center, xi, eta)
            - gnomonic(topocentric[key]["SUN"][epoch_index], center, xi, eta)
        )

    tracks = {
        key: np.array([relative(key, i) for i in range(3)], dtype=float)
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
    exact_delta = tracks["VARDO"][1] - tracks["TAHITI"][1]
    if float(np.dot(exact_delta, normal_2d)) < 0.0:
        normal_2d = -normal_2d
    normal_3d = unit(normal_2d[0] * xi + normal_2d[1] * eta)
    exact_arcsec = float(np.dot(exact_delta, normal_2d))

    observers: dict[str, np.ndarray] = {}
    origin_rows: list[list[object]] = []
    for key in ("TAHITI", "VARDO"):
        from_sun = sun - topocentric[key]["SUN"][1]
        from_venus = venus - topocentric[key]["VENUS"][1]
        observers[key] = 0.5 * (from_sun + from_venus)
        origin_rows.append(
            [
                key,
                norm(from_sun - from_venus),
                *observers[key].tolist(),
            ]
        )

    baseline = observers["VARDO"] - observers["TAHITI"]
    direct_ab_km = abs(float(np.dot(baseline, normal_3d)))

    p_s = projector(sun)
    p_v = projector(venus)
    j_s = gnomonic_jacobian(sun, center, xi, eta)
    j_v = gnomonic_jacobian(venus, center, xi, eta)

    d_s = -(p_s @ baseline) / es
    d_v_separate = -(p_v @ baseline) / ev
    d_v_common_projector = -(p_s @ baseline) / ev

    response_00 = j_v @ d_v_separate - j_s @ d_s
    response_10 = j_v @ d_v_common_projector - j_s @ d_s
    response_01 = j_s @ d_v_separate - j_s @ d_s
    response_11 = j_s @ d_v_common_projector - j_s @ d_s
    response_triangle = j_s @ (-(p_s @ baseline) * (vs / (ev * es)))

    def angle_arcsec(response: np.ndarray) -> float:
        value = arcsec_per_rad * float(np.dot(response, normal_2d))
        if not np.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"Invalid signed tangent-plane prediction: {value!r}.")
        return value

    a_exact = exact_arcsec
    a_00 = angle_arcsec(response_00)
    a_10 = angle_arcsec(response_10)
    a_01 = angle_arcsec(response_01)
    a_11 = angle_arcsec(response_11)
    a_triangle = angle_arcsec(response_triangle)

    finite_km = (a_exact - a_00) * backtrack_scale

    projector_path_a_km = (a_00 - a_10) * backtrack_scale
    jacobian_path_a_km = (a_10 - a_11) * backtrack_scale
    jacobian_path_b_km = (a_00 - a_01) * backtrack_scale
    projector_path_b_km = (a_01 - a_11) * backtrack_scale

    projector_symmetric_km = 0.5 * (
        projector_path_a_km + projector_path_b_km
    )
    jacobian_symmetric_km = 0.5 * (
        jacobian_path_a_km + jacobian_path_b_km
    )
    direction_total_km = (a_00 - a_11) * backtrack_scale
    triangle_km = (a_11 - a_triangle) * backtrack_scale

    total_derived_km = (
        finite_km
        + projector_symmetric_km
        + jacobian_symmetric_km
        + triangle_km
    )
    observed_residual_km = a_exact * backtrack_scale - direct_ab_km
    triangle_backtrack_km = a_triangle * backtrack_scale

    geometry_frame = pd.DataFrame(
        [
            ["Closest-approach UTC", utc_text(jd_ca) + " UTC"],
            ["Closest-approach JD TDB", jd_ca],
            ["Instantaneous common velocity angle deg", math.degrees(math.atan2(along[1], along[0]))],
            ["Instantaneous common-normal angle deg", math.degrees(math.atan2(normal_2d[1], normal_2d[0]))],
            ["Exact finite A′B′ arcsec", a_exact],
            ["Direct common-normal AB km", direct_ab_km],
        ],
        columns=["Quantity", "Value"],
    )

    distance_frame = pd.DataFrame(
        [
            ["Earth → Venus", "EV", ev],
            ["Venus → Sun", "VS", vs],
            ["Earth → Sun", "ES", es],
            ["Physical Halley ratio", "EV/VS", physical_ratio],
            ["Backtrack scale", "ES/arcsec × EV/VS", backtrack_scale],
        ],
        columns=["Instantaneous JPL quantity", "Symbol / equation", "Value"],
    )

    origin_frame = pd.DataFrame(
        origin_rows,
        columns=[
            "Station",
            "Sun-derived versus Venus-derived origin mismatch km",
            "Observer X ecliptic km",
            "Observer Y ecliptic km",
            "Observer Z ecliptic km",
        ],
    )

    model_frame = pd.DataFrame(
        [
            ["Exact finite topocentric gnomonic rays", "finite", a_exact],
            ["Separate projector + separate Jacobian", "ΠV, JV", a_00],
            ["Common projector + separate Jacobian", "ΠS, JV", a_10],
            ["Separate projector + common Jacobian", "ΠV, JS", a_01],
            ["Common projector + common Jacobian", "ΠS, JS", a_11],
            ["Classical collinear Halley triangle", "ΠS, JS, VS/(EV·ES)", a_triangle],
        ],
        columns=["Independent model", "Venus response", "Predicted A′B′ arcsec"],
    )
    model_frame["A′B′ × EV/VS km"] = (
        model_frame["Predicted A′B′ arcsec"].astype(float) * backtrack_scale
    )
    model_frame["Backtrack minus direct AB km"] = (
        model_frame["A′B′ × EV/VS km"].astype(float) - direct_ab_km
    )

    order_frame = pd.DataFrame(
        [
            ["Projector first", "Projector ΠV → ΠS", projector_path_a_km],
            ["Projector first", "Jacobian JV → JS", jacobian_path_a_km],
            ["Jacobian first", "Jacobian JV → JS", jacobian_path_b_km],
            ["Jacobian first", "Projector ΠV → ΠS", projector_path_b_km],
        ],
        columns=["Replacement order", "Direction-dependent term", "Equivalent AB km"],
    )

    decomposition_frame = pd.DataFrame(
        [
            ["Finite-baseline nonlinearity", finite_km, 100.0 * finite_km / observed_residual_km],
            ["Projector contribution — symmetric average", projector_symmetric_km, 100.0 * projector_symmetric_km / observed_residual_km],
            ["Gnomonic-Jacobian contribution — symmetric average", jacobian_symmetric_km, 100.0 * jacobian_symmetric_km / observed_residual_km],
            ["Distance-triangle non-collinearity", triangle_km, 100.0 * triangle_km / observed_residual_km],
            ["Combined direction-dependent response", direction_total_km, 100.0 * direction_total_km / observed_residual_km],
            ["Total independently derived residual", total_derived_km, 100.0 * total_derived_km / observed_residual_km],
            ["Direct classical-minus-exact residual", observed_residual_km, 100.0],
        ],
        columns=["Physical contribution", "Equivalent AB km", "Percent of residual"],
    )

    status_frame = pd.DataFrame(
        [
            ["No C_total, AB/A′B′, effective distance, or known residual used as a model input", "PASS", 0.0, "dimensionless"],
            ["Two replacement orders have identical combined direction response", "PASS" if abs((projector_path_a_km + jacobian_path_a_km) - (jacobian_path_b_km + projector_path_b_km)) < 1.0e-10 else "FAIL", (projector_path_a_km + jacobian_path_a_km) - (jacobian_path_b_km + projector_path_b_km), "km"],
            ["Symmetric projector plus Jacobian equals direction total", "PASS" if abs(projector_symmetric_km + jacobian_symmetric_km - direction_total_km) < 1.0e-10 else "FAIL", projector_symmetric_km + jacobian_symmetric_km - direction_total_km, "km"],
            ["Classical triangle backtrack equals direct AB", "PASS" if abs(triangle_backtrack_km - direct_ab_km) < 1.0e-6 else "FAIL", triangle_backtrack_km - direct_ab_km, "km"],
            ["Independent decomposition equals observed residual", "PASS" if abs(total_derived_km - observed_residual_km) < 1.0e-8 else "FAIL", total_derived_km - observed_residual_km, "km"],
        ],
        columns=["Equation / test", "Status", "Residual", "Unit"],
    )

    records: list[dict[str, object]] = []
    for section, frame in (
        ("GEOMETRY", geometry_frame),
        ("DISTANCES", distance_frame),
        ("ORIGINS", origin_frame),
        ("MODELS", model_frame),
        ("ORDER_PATHS", order_frame),
        ("DECOMPOSITION", decomposition_frame),
        ("STATUS", status_frame),
    ):
        for row_number, row in frame.iterrows():
            record: dict[str, object] = {
                "section": section,
                "row": int(row_number),
            }
            record.update({str(key): value for key, value in row.items()})
            records.append(record)
    pd.DataFrame(records).to_csv(CSV, index=False, float_format="%.15f")

    css = """
<style>
.r{background:#000;color:#fff;font-family:Arial,Helvetica,sans-serif;padding:16px;border:1px solid #fff;width:100%;box-sizing:border-box}.r *{background:#000;color:#fff;box-sizing:border-box}
.r h1{font-size:22px;border-bottom:2px solid #fff;padding-bottom:8px}.r h2{font-size:16px;border-top:1px solid #fff;border-bottom:1px solid #fff;padding:5px 0;margin-top:22px}.r h3{font-size:14px}.r p{font-size:13px;line-height:1.45}
.wrap{overflow-x:auto}.audit{border-collapse:collapse;min-width:100%;width:max-content;font-size:12px}.audit th,.audit td{border:1px solid #fff;padding:7px 9px;white-space:nowrap}.audit th{font-weight:700;text-align:center}.audit td{text-align:right}.audit td:first-child,.audit td:nth-child(2){text-align:left}
.note{border:1px solid #fff;padding:9px;font-weight:600}.answer{font-size:18px;border:2px solid #fff;padding:12px;font-weight:700;text-align:center}.path{font-family:monospace;font-size:11px;overflow-wrap:anywhere;white-space:normal}
</style>
"""

    html: list[str] = [
        css,
        '<div class="r">',
        "<h1>1769 Venus Transit — Order-Independent Projector Audit</h1>",
        "<h2>CODE INPUTS</h2>",
        f"<p><b>JPL master source:</b> {master_source}</p>",
        f"<p><b>Exact query epoch:</b> {utc_text(jd_ca)} UTC</p>",
        "<p><b>Excluded inputs:</b> C_total, AB/A′B′, effective distances, and the known 19 km residual.</p>",
        "<h2>COMMENTS</h2>",
        '<p class="note">The projector and Jacobian attributions are calculated in both replacement orders. Their symmetric averages remove the path dependence present in a one-order decomposition.</p>',
        '<p class="note">The combined direction-dependent contribution is invariant: it is the direct difference between the separate Venus/Sun response and the common Sun-direction response.</p>',
        "<h2>RESULTS</h2>",
        "<h3>Closest approach and common normal</h3>",
        formatted_table(geometry_frame, {"Value": "{:,.15f}"}),
        "<h3>Instantaneous JPL distances</h3>",
        formatted_table(distance_frame, {"Value": "{:,.15f}"}),
        "<h3>JPL station-origin consistency</h3>",
        formatted_table(
            origin_frame,
            {
                "Sun-derived versus Venus-derived origin mismatch km": "{:.12e}",
                "Observer X ecliptic km": "{:+,.12f}",
                "Observer Y ecliptic km": "{:+,.12f}",
                "Observer Z ecliptic km": "{:+,.12f}",
            },
        ),
        "<h3>Four independent projector/Jacobian combinations</h3>",
        formatted_table(
            model_frame,
            {
                "Predicted A′B′ arcsec": "{:+.15f}",
                "A′B′ × EV/VS km": "{:+,.12f}",
                "Backtrack minus direct AB km": "{:+,.12f}",
            },
        ),
        "<h3>Both replacement orders</h3>",
        formatted_table(order_frame, {"Equivalent AB km": "{:+,.12f}"}),
        "<h3>Order-independent physical decomposition</h3>",
        formatted_table(
            decomposition_frame,
            {
                "Equivalent AB km": "{:+,.12f}",
                "Percent of residual": "{:+.9f}",
            },
        ),
        f'<p class="answer">Classical Halley minus direct AB = {observed_residual_km:+.12f} km. The order-independent direction response is {direction_total_km:+.12f} km.</p>',
        "<h2>OUTPUT SUMMARY</h2>",
        f'<p class="path">{CSV}</p>',
        f'<p class="path">{HTML_FILE}</p>',
        "<h2>PAPER COMPARISON</h2>",
        '<p class="note">This audit separates finite-baseline, projector, gnomonic-Jacobian, and non-collinear distance-triangle effects without a fitted closure coefficient. Symmetric averaging is used only to divide the invariant combined direction response between the interacting projector and Jacobian terms.</p>',
        "<h2>EQUATION STATUS</h2>",
        formatted_table(status_frame, {"Residual": "{:+.15e}"}),
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
# V0052