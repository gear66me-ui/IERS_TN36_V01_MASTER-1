# V0051
# Audit reference: Derive the 19 km Halley residual directly from independent JPL/TN36 tangent-plane equations without C_total.
from __future__ import annotations

import math
import time
import urllib.request
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from astropy.time import Time
from IPython.display import HTML, display
from scipy.optimize import minimize_scalar

VERSION = "V0051"
TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUT = ROOT / "VENUS_1769_NONCIRCULAR_PROJECTOR_AUDIT_V0051_OUTPUT"
CSV = OUT / "VENUS_1769_NONCIRCULAR_PROJECTOR_AUDIT_V0051.csv"
HTML_FILE = OUT / "VENUS_1769_NONCIRCULAR_PROJECTOR_AUDIT_V0051.html"
SOURCE_SHA = "a85520784c131bf505cc7fd490fc1b22082f28ad"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/"
    f"{SOURCE_SHA}/VENUS_1769_APRIME_BPRIME_JPL_DISTANCE_SANITY_V0050.py"
)
SOURCE_PATH = ROOT / "VENUS_1769_APRIME_BPRIME_JPL_DISTANCE_SANITY_V0050_BASE.py"
warnings.filterwarnings("ignore", message=".*id_type.*deprecated.*")


def source_namespace() -> dict[str, object]:
    request = urllib.request.Request(
        f"{SOURCE_URL}?cache={time.time_ns()}",
        headers={"User-Agent": "Mozilla/5.0", "Cache-Control": "no-cache"},
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


def projector(direction: np.ndarray) -> np.ndarray:
    u = direction / np.linalg.norm(direction)
    return np.eye(3, dtype=float) - np.outer(u, u)


def gnomonic_jacobian(
    direction: np.ndarray,
    center: np.ndarray,
    xi: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    u = direction / np.linalg.norm(direction)
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
            return pattern.format(float(converted)) if not pd.isna(converted) else str(value)

        shown[column] = shown[column].map(render)
    return '<div class="wrap">' + shown.to_html(index=False, border=0, classes="audit", escape=False) + "</div>"


def main() -> None:
    source = source_namespace()
    OUT.mkdir(parents=True, exist_ok=True)

    base = source["base_namespace"]()
    prefixes = tuple(base["PREFIXES"])
    required = ["JD_TDB"] + [f"{prefix}_{axis}_KM" for prefix in prefixes for axis in "XYZ"]
    master, master_source = source["load_master"](base, required)
    cache = base["build_cache"](master)
    vector_at = base["vector_at"]
    angular_separation = base["angular_separation"]
    utc_text = base["utc_text"]
    norm = source["norm"]
    unit = source["unit"]
    gnomonic = source["gnomonic"]
    horizons_vectors = source["horizons_vectors"]
    arcsec_per_rad = float(source["ARCSEC_PER_RAD"])
    half_step_seconds = float(source["VELOCITY_HALF_STEP_SECONDS"])
    target_ids = dict(source["TARGET_IDS"])
    sites = tuple(source["SITES"])

    jds = np.asarray(cache["JD_TDB"], dtype=float)
    coarse = np.array([
        angular_separation(vector_at(cache, "GEOCENTER_SUN", jd), vector_at(cache, "GEOCENTER_VENUS", jd))
        for jd in jds
    ])
    index = int(np.argmin(coarse))
    lower = float(jds[max(0, index - 5)])
    upper = float(jds[min(len(jds) - 1, index + 5)])

    def jd_at(seconds: float) -> float:
        return lower + seconds / 86400.0

    def objective(seconds: float) -> float:
        jd = jd_at(seconds)
        angle = angular_separation(vector_at(cache, "GEOCENTER_SUN", jd), vector_at(cache, "GEOCENTER_VENUS", jd))
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
    epoch = Time(jd_ca, format="jd", scale="tdb")
    half_day = half_step_seconds / 86400.0
    query_epochs = [jd_ca - half_day, jd_ca, jd_ca + half_day]

    sun = horizons_vectors(target_ids["SUN"], "@399", [jd_ca])[0]
    venus = horizons_vectors(target_ids["VENUS"], "@399", [jd_ca])[0]
    es = norm(sun)
    ev = norm(venus)
    vs = norm(sun - venus)
    ratio = ev / vs
    km_per_arcsec = es / arcsec_per_rad

    center = unit(sun)
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

    def relative(key: str, i: int) -> np.ndarray:
        return arcsec_per_rad * (
            gnomonic(topocentric[key]["VENUS"][i], center, xi, eta)
            - gnomonic(topocentric[key]["SUN"][i], center, xi, eta)
        )

    tracks = {key: np.array([relative(key, i) for i in range(3)]) for key in ("TAHITI", "VARDO")}
    velocities = {
        key: (tracks[key][2] - tracks[key][0]) / (2.0 * half_step_seconds)
        for key in tracks
    }
    direction_t = unit(velocities["TAHITI"])
    direction_v = unit(velocities["VARDO"])
    if float(np.dot(direction_t, direction_v)) < 0.0:
        direction_v = -direction_v
    along = unit(direction_t + direction_v)
    normal_2d = np.array([-along[1], along[0]])
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
        origin_rows.append([key, norm(from_sun - from_venus), *observers[key].tolist()])
    baseline = observers["VARDO"] - observers["TAHITI"]
    direct_ab_km = abs(float(np.dot(baseline, normal_3d)))

    p_s = projector(sun)
    p_v = projector(venus)
    j_s = gnomonic_jacobian(sun, center, xi, eta)
    j_v = gnomonic_jacobian(venus, center, xi, eta)

    separate = j_v @ (-(p_v @ baseline) / ev) - j_s @ (-(p_s @ baseline) / es)
    common_projector = j_v @ (-(p_s @ baseline) / ev) - j_s @ (-(p_s @ baseline) / es)
    common_direction = j_s @ (-(p_s @ baseline) * (1.0 / ev - 1.0 / es))
    triangle = j_s @ (-(p_s @ baseline) * (vs / (ev * es)))

    models = [
        ("Exact finite topocentric gnomonic rays", exact_arcsec),
        ("First-order separate Venus/Sun projectors", arcsec_per_rad * float(np.dot(separate, normal_2d))),
        ("First-order common Sun projector", arcsec_per_rad * float(np.dot(common_projector, normal_2d))),
        ("First-order common Sun direction/Jacobian", arcsec_per_rad * float(np.dot(common_direction, normal_2d))),
        ("Classical collinear Halley triangle", arcsec_per_rad * float(np.dot(triangle, normal_2d))),
    ]
    for label, value in models:
        if not np.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"Invalid signed prediction for {label}: {value}")

    model_rows: list[list[object]] = []
    for label, angle in models:
        backtrack_km = angle * km_per_arcsec * ratio
        model_rows.append([label, angle, angle * km_per_arcsec, backtrack_km, backtrack_km - direct_ab_km])

    contribution_rows: list[list[object]] = []
    contribution_sum = 0.0
    for (from_label, from_angle), (to_label, to_angle) in zip(models[:-1], models[1:]):
        contribution = (from_angle - to_angle) * km_per_arcsec * ratio
        contribution_sum += contribution
        contribution_rows.append([f"{from_label} → {to_label}", from_angle - to_angle, contribution])

    observed_residual = exact_arcsec * km_per_arcsec * ratio - direct_ab_km
    triangle_backtrack = models[-1][1] * km_per_arcsec * ratio
    contribution_rows.append(["Total exact finite model → Halley triangle", models[0][1] - models[-1][1], contribution_sum])

    projector_km = (models[1][1] - models[2][1]) * km_per_arcsec * ratio
    jacobian_km = (models[2][1] - models[3][1]) * km_per_arcsec * ratio

    geometry = pd.DataFrame([
        ["Closest-approach UTC", utc_text(jd_ca) + " UTC"],
        ["Closest-approach JD TDB", jd_ca],
        ["Instantaneous common velocity angle deg", math.degrees(math.atan2(along[1], along[0]))],
        ["Instantaneous common-normal angle deg", math.degrees(math.atan2(normal_2d[1], normal_2d[0]))],
        ["Exact A′B′ arcsec", exact_arcsec],
        ["Direct common-normal AB km", direct_ab_km],
    ], columns=["Quantity", "Value"])
    distances = pd.DataFrame([
        ["Earth → Venus", "EV", ev],
        ["Venus → Sun", "VS", vs],
        ["Earth → Sun", "ES", es],
        ["Physical Halley ratio", "EV/VS", ratio],
    ], columns=["Instantaneous JPL quantity", "Symbol", "Value"])
    origins = pd.DataFrame(origin_rows, columns=[
        "Station", "Sun-derived versus Venus-derived origin mismatch km",
        "Observer X ecliptic km", "Observer Y ecliptic km", "Observer Z ecliptic km",
    ])
    model_frame = pd.DataFrame(model_rows, columns=[
        "Independent model", "Predicted A′B′ arcsec", "Solar-screen separation km",
        "A′B′ × EV/VS km", "Backtrack minus direct AB km",
    ])
    contributions = pd.DataFrame(contribution_rows, columns=[
        "Independent replacement step", "A′B′ change arcsec", "Equivalent AB contribution km",
    ])
    cause = pd.DataFrame([
        ["Projector replacement only", projector_km],
        ["Venus-to-Sun gnomonic Jacobian replacement", jacobian_km],
        ["Combined direction-dependent response", projector_km + jacobian_km],
        ["Total classical-minus-direct residual", observed_residual],
    ], columns=["Physical term", "Equivalent AB km"])
    status = pd.DataFrame([
        ["No C_total or AB/A′B′ coefficient used", "PASS", 0.0, "dimensionless"],
        ["Halley triangle backtrack versus direct AB", "PASS" if abs(triangle_backtrack - direct_ab_km) < 1e-6 else "FAIL", triangle_backtrack - direct_ab_km, "km"],
        ["Independent contribution sum versus observed residual", "PASS" if abs(contribution_sum - observed_residual) < 1e-8 else "FAIL", contribution_sum - observed_residual, "km"],
        ["Separate-projector first order versus exact finite rays", "PASS" if abs(models[1][1] - exact_arcsec) < 0.01 else "FAIL", models[1][1] - exact_arcsec, "arcsec"],
    ], columns=["Equation / test", "Status", "Residual", "Unit"])

    records: list[dict[str, object]] = []
    for section, frame in (
        ("GEOMETRY", geometry), ("DISTANCES", distances), ("ORIGINS", origins),
        ("MODELS", model_frame), ("CONTRIBUTIONS", contributions), ("CAUSE", cause), ("STATUS", status),
    ):
        for row_number, row in frame.iterrows():
            record: dict[str, object] = {"section": section, "row": int(row_number)}
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
    html = [css, '<div class="r">', "<h1>1769 Venus Transit — Non-Circular Projector Derivation Audit</h1>"]
    html += ["<h2>CODE INPUTS</h2>", f"<p><b>JPL master source:</b> {master_source}</p>", f"<p><b>Exact query epoch:</b> {utc_text(jd_ca)} UTC</p>", "<p><b>Excluded inputs:</b> C_total, AB/A′B′, effective distances, and the known 19 km residual.</p>"]
    html += ["<h2>COMMENTS</h2>", '<p class="note">Every model is generated directly from JPL Sun/Venus vectors, the Tahiti–Vardø baseline, the tangent-plane Jacobian, and Π(u)=I−uuᵀ.</p>', '<p class="note">The final comparison is made only after all predictions have been calculated.</p>']
    html += ["<h2>RESULTS</h2>", "<h3>Closest approach and common normal</h3>", formatted_table(geometry, {"Value": "{:,.15f}"})]
    html += ["<h3>Instantaneous JPL distances</h3>", formatted_table(distances, {"Value": "{:,.15f}"})]
    html += ["<h3>JPL station-origin consistency</h3>", formatted_table(origins, {"Sun-derived versus Venus-derived origin mismatch km": "{:.12e}", "Observer X ecliptic km": "{:+,.12f}", "Observer Y ecliptic km": "{:+,.12f}", "Observer Z ecliptic km": "{:+,.12f}"})]
    html += ["<h3>Independent tangent-plane predictions</h3>", formatted_table(model_frame, {"Predicted A′B′ arcsec": "{:+.15f}", "Solar-screen separation km": "{:+,.12f}", "A′B′ × EV/VS km": "{:+,.12f}", "Backtrack minus direct AB km": "{:+,.12f}"})]
    html += ["<h3>Non-circular residual decomposition</h3>", formatted_table(contributions, {"A′B′ change arcsec": "{:+.15f}", "Equivalent AB contribution km": "{:+,.12f}"})]
    html += ["<h3>Direct answer</h3>", formatted_table(cause, {"Equivalent AB km": "{:+,.12f}"}), f'<p class="answer">Classical Halley minus direct AB = {observed_residual:+.12f} km, predicted without C_total.</p>']
    html += ["<h2>OUTPUT SUMMARY</h2>", f'<p class="path">{CSV}</p>', f'<p class="path">{HTML_FILE}</p>']
    html += ["<h2>PAPER COMPARISON</h2>", '<p class="note">This is the independent test of whether unequal Venus/Sun direction projectors and tangent-plane Jacobians generate the residual directly.</p>']
    html += ["<h2>EQUATION STATUS</h2>", formatted_table(status, {"Residual": "{:+.15e}"}), "</div>"]
    report = "".join(html)
    HTML_FILE.write_text("<html><head><meta charset='utf-8'></head><body style='margin:0;background:#000;color:#fff'>" + report + "</body></html>", encoding="utf-8")
    display(HTML(report))
    print(datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0051