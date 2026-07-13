# V0046
# Audit reference: Compare geometric, astrometric, and apparent TN36 pointing-derived distances and Halley closure.
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

try:
    from erfa import ErfaWarning
    warnings.filterwarnings("ignore", category=ErfaWarning)
except Exception:
    warnings.filterwarnings("ignore", message=".*dubious year.*")

import erfa
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import HTML, display
from scipy.optimize import minimize_scalar

VERSION = "V0046"
TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUT = ROOT / "VENUS_1769_RETARDED_POINTING_DISTANCE_AUDIT_V0046_OUTPUT"
CSV = OUT / "VENUS_1769_RETARDED_POINTING_DISTANCE_AUDIT_V0046.csv"
HTML_FILE = OUT / "VENUS_1769_RETARDED_POINTING_DISTANCE_AUDIT_V0046.html"
MASTER_FILES = (
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0034.csv",
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0033.csv",
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0046.csv",
)
BASE_SHA = "d55bc2274359a3014c19f257e42cc149bc458d57"
BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/"
    f"{BASE_SHA}/VENUS_1769_TAHITI_VARDO_HALLEY_THUMB_CHECK_V0031.py"
)
BASE_PATH = ROOT / "VENUS_1769_TAHITI_VARDO_HALLEY_THUMB_CHECK_V0031_BASE.py"
ARCSEC = 206_264.80624709636
AU_KM = 149_597_870.000000
EARTH_RADIUS_KM = 6_378.140000
SITES = (
    {"name": "Tahiti", "key": "TAHITI", "lat": -17.4956, "lon": -149.4939, "elevation": 0.0, "body": 399},
    {"name": "Vardø", "key": "VARDO", "lat": 70.3724, "lon": 31.1103, "elevation": 0.0, "body": 399},
)
TARGETS = (("SUN", "10"), ("VENUS", "299"))
MODES = ("geometric", "astrometric", "apparent")


def base_namespace() -> dict[str, object]:
    request = urllib.request.Request(
        f"{BASE_URL}?cache={time.time_ns()}",
        headers={"User-Agent": "Mozilla/5.0", "Cache-Control": "no-cache", "Pragma": "no-cache"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        source = response.read().decode("utf-8")
    if not source.startswith("# V0031\n") or not source.rstrip().endswith("# V0031"):
        raise RuntimeError("Pinned V0031 source audit failed.")
    compile(source, str(BASE_PATH), "exec")
    BASE_PATH.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {"__name__": "v0031_base", "__file__": str(BASE_PATH)}
    exec(compile(source, str(BASE_PATH), "exec"), namespace)
    return namespace


def load_master(base: dict[str, object], required_columns: list[str]) -> tuple[pd.DataFrame, str]:
    for path in MASTER_FILES:
        if path.is_file():
            try:
                frame = pd.read_csv(path)
                if all(column in frame.columns for column in required_columns):
                    return frame, str(path)
            except Exception:
                continue
    frame = base["build_master"]()
    frame.to_csv(MASTER_FILES[-1], index=False, float_format="%.15f")
    return frame, "NEW JPL HORIZONS DOWNLOAD"


def norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    magnitude = norm(vector)
    if magnitude == 0.0:
        raise RuntimeError("Zero vector encountered.")
    return vector / magnitude


def split_jd(value: float) -> tuple[float, float]:
    whole = math.floor(value)
    return float(whole), float(value - whole)


def eq_to_ecl_matrix() -> np.ndarray:
    epsilon = float(erfa.obl80(2451545.0, 0.0))
    c, s = math.cos(epsilon), math.sin(epsilon)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]], dtype=float)


def itrs_wgs84(site: dict[str, object]) -> np.ndarray:
    xyz_m = erfa.gd2gc(
        1,
        math.radians(float(site["lon"])),
        math.radians(float(site["lat"])),
        float(site["elevation"]),
    )
    return np.asarray(xyz_m, dtype=float) / 1000.0


def gnomonic(vector: np.ndarray, center: np.ndarray, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    direction = unit(vector)
    denominator = float(np.dot(direction, center))
    if denominator <= 0.0:
        raise RuntimeError("Direction lies outside tangent hemisphere.")
    return np.array(
        [float(np.dot(direction, xi)) / denominator, float(np.dot(direction, eta)) / denominator],
        dtype=float,
    )


def ray_solution(origin_1: np.ndarray, direction_1: np.ndarray, origin_2: np.ndarray, direction_2: np.ndarray) -> dict[str, object]:
    d1 = unit(direction_1)
    d2 = unit(direction_2)
    matrix = np.column_stack((d1, -d2))
    rhs = origin_2 - origin_1
    parameters, _residuals, _rank, singular = np.linalg.lstsq(matrix, rhs, rcond=None)
    point_1 = origin_1 + float(parameters[0]) * d1
    point_2 = origin_2 + float(parameters[1]) * d2
    midpoint = 0.5 * (point_1 + point_2)
    miss = norm(point_1 - point_2)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0.0 else math.inf
    return {
        "position": midpoint,
        "range_1": float(parameters[0]),
        "range_2": float(parameters[1]),
        "miss": miss,
        "condition": condition,
    }


def horizon_vector(target_id: str, location: object, jd_tdb: float, mode: str) -> dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            query = Horizons(id=target_id, location=location, epochs=[jd_tdb])
            table = query.vectors(refplane="ecliptic", aberrations=mode)
            if len(table) != 1:
                raise RuntimeError(f"Expected one Horizons row; received {len(table)}.")
            row = table[0]
            vector = np.array([float(row[axis]) * AU_KM for axis in "xyz"], dtype=float)
            columns = set(table.colnames)
            lighttime_seconds = float(row["lighttime"]) * 86400.0 if "lighttime" in columns else math.nan
            range_km = float(row["range"]) * AU_KM if "range" in columns else norm(vector)
            return {"vector": vector, "lighttime_s": lighttime_seconds, "range_km": range_km}
        except Exception as error:
            last_error = error
            if attempt < 3:
                time.sleep(2.0 * attempt)
    raise RuntimeError(f"Horizons {mode} vector query failed: {last_error}")


def table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    shown = frame.copy()
    for column, pattern in (formats or {}).items():
        if column not in shown.columns:
            continue
        def formatter(value: object) -> str:
            if pd.isna(value):
                return ""
            if isinstance(value, (int, float, np.integer, np.floating)):
                return pattern.format(float(value))
            return str(value)
        shown[column] = shown[column].map(formatter)
    return '<div class="wrap">' + shown.to_html(index=False, border=0, classes="audit", escape=False) + "</div>"


def main() -> None:
    base = base_namespace()
    OUT.mkdir(parents=True, exist_ok=True)

    prefixes = tuple(base["PREFIXES"])
    required_columns = ["JD_TDB"] + [f"{prefix}_{axis}_KM" for prefix in prefixes for axis in "XYZ"]
    master, source = load_master(base, required_columns)
    cache = base["build_cache"](master)
    vector_at = base["vector_at"]
    angular_separation = base["angular_separation"]
    common_basis = base["common_basis"]
    relative_position = base["relative_position_arcsec"]
    external_contacts = base["external_contacts"]
    fitted_direction = base["fitted_direction"]
    utc_text = base["utc_text"]

    jds = np.asarray(cache["JD_TDB"], dtype=float)
    coarse = np.array([
        angular_separation(vector_at(cache, "GEOCENTER_SUN", sample), vector_at(cache, "GEOCENTER_VENUS", sample))
        for sample in jds
    ], dtype=float)
    index = int(np.argmin(coarse))
    lower = float(jds[max(0, index - 5)])
    upper = float(jds[min(len(jds) - 1, index + 5)])
    span_seconds = (upper - lower) * 86400.0

    def jd_at(seconds: float) -> float:
        return lower + float(seconds) / 86400.0

    def ca_objective(seconds: float) -> float:
        sample = jd_at(seconds)
        angle = angular_separation(vector_at(cache, "GEOCENTER_SUN", sample), vector_at(cache, "GEOCENTER_VENUS", sample))
        return (angle * ARCSEC) ** 2

    ca = minimize_scalar(ca_objective, bounds=(0.0, span_seconds), method="bounded", options={"xatol": 1e-7, "maxiter": 500})
    if not ca.success:
        raise RuntimeError("Closest-approach optimization failed.")
    jd = jd_at(float(ca.x))
    epoch = Time(jd, format="jd", scale="tdb")

    geometric_sun = vector_at(cache, "GEOCENTER_SUN", jd)
    center_geo, xi_geo, eta_geo = common_basis(geometric_sun)
    center_geo, xi_geo, eta_geo = unit(center_geo), unit(xi_geo), unit(eta_geo)

    tahiti_contacts = external_contacts(cache, "TAHITI")
    vardo_contacts = external_contacts(cache, "VARDO")
    common_start = max(float(tahiti_contacts[0]), float(vardo_contacts[0]))
    common_stop = min(float(tahiti_contacts[1]), float(vardo_contacts[1]))
    selected = jds[(jds >= common_start) & (jds <= common_stop)]
    tracks = {
        key: np.array([relative_position(cache, key, sample, center_geo, xi_geo, eta_geo) for sample in selected], dtype=float)
        for key in ("TAHITI", "VARDO")
    }
    direction_t = fitted_direction(tracks["TAHITI"])
    direction_v = fitted_direction(tracks["VARDO"])
    if float(np.dot(direction_t, direction_v)) < 0.0:
        direction_v = -direction_v
    along_geo = unit(direction_t + direction_v)
    normal_geo_2d = np.array([-along_geo[1], along_geo[0]], dtype=float)
    normal_geo_3d = unit(normal_geo_2d[0] * xi_geo + normal_geo_2d[1] * eta_geo)

    locations: dict[str, object] = {"GEOCENTER": "500@399"}
    for site in SITES:
        locations[str(site["key"])] = {
            "lon": float(site["lon"]),
            "lat": float(site["lat"]),
            "elevation": float(site["elevation"]),
            "body": int(site["body"]),
        }

    vectors: dict[str, dict[str, dict[str, object]]] = {}
    for mode in MODES:
        vectors[mode] = {}
        for location_key, location in locations.items():
            vectors[mode][location_key] = {}
            for target_name, target_id in TARGETS:
                vectors[mode][location_key][target_name] = horizon_vector(target_id, location, jd, mode)

    geo_observers = {
        key: np.asarray(vectors["geometric"]["GEOCENTER"]["SUN"]["vector"], dtype=float)
        - np.asarray(vectors["geometric"][key]["SUN"]["vector"], dtype=float)
        for key in ("TAHITI", "VARDO")
    }

    tt1, tt2 = split_jd(float(epoch.tt.jd))
    ecl = eq_to_ecl_matrix()
    itrs = {str(site["key"]): itrs_wgs84(site) for site in SITES}

    def iers_observers(dut1_seconds: float) -> dict[str, np.ndarray]:
        ut11, ut12 = split_jd(float(epoch.utc.jd) + float(dut1_seconds) / 86400.0)
        c2t = np.asarray(erfa.c2t06a(tt1, tt2, ut11, ut12, 0.0, 0.0))
        return {key: ecl @ (c2t.T @ itrs[key]) for key in ("TAHITI", "VARDO")}

    def mode_directions(mode: str, target: str) -> tuple[np.ndarray, np.ndarray]:
        return (
            unit(np.asarray(vectors[mode]["TAHITI"][target]["vector"], dtype=float)),
            unit(np.asarray(vectors[mode]["VARDO"][target]["vector"], dtype=float)),
        )

    fitted_dut1: dict[str, float] = {}
    for mode in MODES:
        sun_dirs = mode_directions(mode, "SUN")
        venus_dirs = mode_directions(mode, "VENUS")
        def objective(dut1_seconds: float) -> float:
            observers = iers_observers(dut1_seconds)
            sun_solution = ray_solution(observers["TAHITI"], sun_dirs[0], observers["VARDO"], sun_dirs[1])
            venus_solution = ray_solution(observers["TAHITI"], venus_dirs[0], observers["VARDO"], venus_dirs[1])
            return float(sun_solution["miss"]) ** 2 + float(venus_solution["miss"]) ** 2
        fit = minimize_scalar(objective, bounds=(-300.0, 300.0), method="bounded", options={"xatol": 1e-9, "maxiter": 500})
        if not fit.success:
            raise RuntimeError(f"DUT1 pointing fit failed for {mode}.")
        fitted_dut1[mode] = float(fit.x)

    model_rows: list[list[object]] = []
    distance_rows: list[list[object]] = []
    factor_rows: list[list[object]] = []
    backtrack_rows: list[list[object]] = []
    ray_rows: list[list[object]] = []
    lighttime_rows: list[list[object]] = []

    for mode in MODES:
        geocentric_sun = np.asarray(vectors[mode]["GEOCENTER"]["SUN"]["vector"], dtype=float)
        center, xi, eta = common_basis(geocentric_sun)
        center, xi, eta = unit(center), unit(xi), unit(eta)
        normal_2d = np.array([float(np.dot(normal_geo_3d, xi)), float(np.dot(normal_geo_3d, eta))], dtype=float)
        normal_2d = unit(normal_2d)

        for target_name, _target_id in TARGETS:
            lt = float(vectors[mode]["GEOCENTER"][target_name]["lighttime_s"])
            emission = Time(epoch.tdb.jd - lt / 86400.0, format="jd", scale="tdb") if math.isfinite(lt) else None
            lighttime_rows.append([
                mode,
                target_name,
                lt,
                emission.utc.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC" if emission is not None else "",
                float(vectors[mode]["GEOCENTER"][target_name]["range_km"]),
            ])

        station_models = (
            ("JPL receive-epoch station vectors", geo_observers),
            ("IERS TN36 pointing-fit station vectors", iers_observers(fitted_dut1[mode])),
        )

        for station_label, observers in station_models:
            sun_dirs = mode_directions(mode, "SUN")
            venus_dirs = mode_directions(mode, "VENUS")
            sun_solution = ray_solution(observers["TAHITI"], sun_dirs[0], observers["VARDO"], sun_dirs[1])
            venus_solution = ray_solution(observers["TAHITI"], venus_dirs[0], observers["VARDO"], venus_dirs[1])
            sun_position = np.asarray(sun_solution["position"], dtype=float)
            venus_position = np.asarray(venus_solution["position"], dtype=float)
            ev = norm(venus_position)
            es = norm(sun_position)
            vs = norm(sun_position - venus_position)
            ratio = ev / vs

            relative_t = ARCSEC * (
                gnomonic(np.asarray(vectors[mode]["TAHITI"]["VENUS"]["vector"], dtype=float), center, xi, eta)
                - gnomonic(np.asarray(vectors[mode]["TAHITI"]["SUN"]["vector"], dtype=float), center, xi, eta)
            )
            relative_v = ARCSEC * (
                gnomonic(np.asarray(vectors[mode]["VARDO"]["VENUS"]["vector"], dtype=float), center, xi, eta)
                - gnomonic(np.asarray(vectors[mode]["VARDO"]["SUN"]["vector"], dtype=float), center, xi, eta)
            )
            delta = relative_v - relative_t
            if float(np.dot(delta, normal_2d)) < 0.0:
                normal_2d = -normal_2d
            aprime_bprime_arcsec = float(np.dot(delta, normal_2d))

            baseline = observers["VARDO"] - observers["TAHITI"]
            ab_km = abs(float(np.dot(baseline, normal_geo_3d)))
            scale = es / ARCSEC
            ab_arcsec = ab_km / scale
            exact_factor = ab_arcsec / aprime_bprime_arcsec
            classical_ab_arcsec = aprime_bprime_arcsec * ratio
            classical_ab_km = classical_ab_arcsec * scale
            residual_km = classical_ab_km - ab_km
            pi_event = EARTH_RADIUS_KM / es * ARCSEC
            pi_1au = EARTH_RADIUS_KM / AU_KM * ARCSEC

            model_rows.append([mode, station_label, fitted_dut1[mode] if "IERS" in station_label else math.nan, aprime_bprime_arcsec, ab_arcsec, ab_km])
            distance_rows.extend([
                [mode, station_label, "Earth → Venus", "EV", ev],
                [mode, station_label, "Venus → Sun", "VS", vs],
                [mode, station_label, "Earth → Sun", "ES", es],
            ])
            factor_rows.extend([
                [mode, station_label, "Physical pointing ratio", "EV/VS", ratio],
                [mode, station_label, "Exact screen factor", "AB/A′B′", exact_factor],
                [mode, station_label, "Factor difference", "AB/A′B′ − EV/VS", exact_factor - ratio],
            ])
            backtrack_rows.append([
                mode,
                station_label,
                aprime_bprime_arcsec,
                ratio,
                classical_ab_arcsec,
                classical_ab_km,
                ab_km,
                residual_km,
                pi_event,
                pi_1au,
            ])
            ray_rows.extend([
                [mode, station_label, "Venus", float(venus_solution["miss"]), float(venus_solution["condition"])],
                [mode, station_label, "Sun", float(sun_solution["miss"]), float(sun_solution["condition"])],
            ])

    model_frame = pd.DataFrame(model_rows, columns=["Pointing mode", "Station model", "Fitted DUT1 s", "A′B′ arcsec", "AB arcsec", "AB km"])
    distance_frame = pd.DataFrame(distance_rows, columns=["Pointing mode", "Station model", "Distance", "Symbol", "Pointing-derived km"])
    factor_frame = pd.DataFrame(factor_rows, columns=["Pointing mode", "Station model", "Ratio type", "Definition", "Value"])
    backtrack_frame = pd.DataFrame(backtrack_rows, columns=["Pointing mode", "Station model", "A′B′ arcsec", "EV/VS", "Classical AB arcsec", "Classical AB km", "Direct AB km", "Classical − direct km", "π event arcsec", "π 1-AU arcsec"])
    ray_frame = pd.DataFrame(ray_rows, columns=["Pointing mode", "Station model", "Target", "Ray miss km", "Condition indicator"])
    lighttime_frame = pd.DataFrame(lighttime_rows, columns=["Pointing mode", "Target", "Light time s", "Retarded epoch", "Horizons range km"])

    reference = backtrack_frame[(backtrack_frame["Pointing mode"] == "geometric") & (backtrack_frame["Station model"] == "JPL receive-epoch station vectors")].iloc[0]
    comparison_rows: list[list[object]] = []
    for _, row in backtrack_frame.iterrows():
        comparison_rows.append([
            row["Pointing mode"],
            row["Station model"],
            float(row["Classical − direct km"]),
            float(row["Classical − direct km"]) - float(reference["Classical − direct km"]),
        ])
    comparison_frame = pd.DataFrame(comparison_rows, columns=["Pointing mode", "Station model", "Residual km", "Change from geometric JPL residual km"])

    geometric_distance = distance_frame[(distance_frame["Pointing mode"] == "geometric") & (distance_frame["Station model"] == "JPL receive-epoch station vectors")]
    direct_master = {
        "EV": norm(vector_at(cache, "GEOCENTER_VENUS", jd)),
        "VS": norm(vector_at(cache, "GEOCENTER_SUN", jd) - vector_at(cache, "GEOCENTER_VENUS", jd)),
        "ES": norm(vector_at(cache, "GEOCENTER_SUN", jd)),
    }
    closure_rows = []
    for _, row in geometric_distance.iterrows():
        symbol = str(row["Symbol"])
        closure_rows.append([symbol, float(row["Pointing-derived km"]), direct_master[symbol], float(row["Pointing-derived km"]) - direct_master[symbol]])
    closure_frame = pd.DataFrame(closure_rows, columns=["Symbol", "Pointing-derived km", "Direct master-vector km", "Residual km"])

    status_frame = pd.DataFrame([
        ["Geometric pointing reconstruction versus master distances", "PASS" if float(np.max(np.abs(closure_frame["Residual km"]))) < 0.1 else "FAIL", float(np.max(np.abs(closure_frame["Residual km"])))],
        ["Astrometric residual change explains 19.57 km", "PASS" if abs(float(comparison_frame[(comparison_frame["Pointing mode"] == "astrometric") & (comparison_frame["Station model"] == "JPL receive-epoch station vectors")]["Residual km"].iloc[0])) < 0.1 else "FAIL", float(comparison_frame[(comparison_frame["Pointing mode"] == "astrometric") & (comparison_frame["Station model"] == "JPL receive-epoch station vectors")]["Residual km"].iloc[0])],
        ["Apparent residual change explains 19.57 km", "PASS" if abs(float(comparison_frame[(comparison_frame["Pointing mode"] == "apparent") & (comparison_frame["Station model"] == "JPL receive-epoch station vectors")]["Residual km"].iloc[0])) < 0.1 else "FAIL", float(comparison_frame[(comparison_frame["Pointing mode"] == "apparent") & (comparison_frame["Station model"] == "JPL receive-epoch station vectors")]["Residual km"].iloc[0])],
    ], columns=["Equation / test", "Status", "Residual / diagnostic km"])

    frames = [
        ("LIGHT_TIME", lighttime_frame),
        ("GEOMETRY", model_frame),
        ("DISTANCES", distance_frame),
        ("FACTORS", factor_frame),
        ("BACKTRACK", backtrack_frame),
        ("RAYS", ray_frame),
        ("COMPARISON", comparison_frame),
        ("CLOSURE", closure_frame),
        ("STATUS", status_frame),
    ]
    records: list[dict[str, object]] = []
    for section, frame in frames:
        for row_number, row in frame.iterrows():
            record: dict[str, object] = {"section": section, "row": int(row_number)}
            record.update({str(key): value for key, value in row.items()})
            records.append(record)
    pd.DataFrame(records).to_csv(CSV, index=False, float_format="%.15f")

    css = """
<style>
.r{background:#000;color:#fff;font-family:Arial,Helvetica,sans-serif;padding:16px;border:1px solid #fff;width:100%;box-sizing:border-box}
.r *{background:#000;color:#fff;box-sizing:border-box}.r h1{font-size:22px;border-bottom:2px solid #fff;padding-bottom:8px}
.r h2{font-size:16px;border-top:1px solid #fff;border-bottom:1px solid #fff;padding:5px 0;margin-top:22px}.r h3{font-size:14px}
.r p{font-size:13px;line-height:1.45}.wrap{overflow-x:auto}.audit{border-collapse:collapse;min-width:100%;width:max-content;font-size:12px}
.audit th,.audit td{border:1px solid #fff;padding:7px 9px;white-space:nowrap}.audit th{font-weight:700;text-align:center}.audit td{text-align:right}
.audit td:first-child,.audit td:nth-child(2){text-align:left}.note{border:1px solid #fff;padding:9px;font-weight:600}.path{font-family:monospace;font-size:11px;overflow-wrap:anywhere;white-space:normal}
</style>
"""
    html = [css, '<div class="r">']
    html.append("<h1>1769 Venus Transit — Retarded Pointing-Time Distance Audit</h1>")
    html.append("<h2>CODE INPUTS</h2>")
    html.append(f"<p><b>Geometric master:</b> {source}</p>")
    html.append(f"<p><b>Receive epoch:</b> {utc_text(jd)} UTC</p>")
    html.append("<p><b>Horizons pointing modes:</b> geometric, astrometric, apparent</p>")
    html.append("<h2>COMMENTS</h2>")
    html.append('<p class="note">Each mode discards the target-vector magnitude and triangulates EV and ES from the Tahiti and Vardø unit pointing directions. VS is calculated from the reconstructed target positions.</p>')
    html.append('<p class="note">Astrometric vectors include light-time; apparent vectors include light-time and stellar aberration. The station origins remain receive-epoch TN36/JPL observer vectors.</p>')
    html.append("<h2>RESULTS</h2>")
    html.append("<h3>Horizons light times and retarded epochs</h3>")
    html.append(table(lighttime_frame, {"Light time s": "{:.9f}", "Horizons range km": "{:,.9f}"}))
    html.append("<h3>A′B′ and AB geometry</h3>")
    html.append(table(model_frame, {"Fitted DUT1 s": "{:+.12f}", "A′B′ arcsec": "{:.12f}", "AB arcsec": "{:.12f}", "AB km": "{:,.12f}"}))
    html.append("<h3>Pointing-derived physical distances</h3>")
    html.append(table(distance_frame, {"Pointing-derived km": "{:,.12f}"}))
    html.append("<h3>Physical and exact screen factors</h3>")
    html.append(table(factor_frame, {"Value": "{:+.15f}"}))
    html.append("<h3>Halley backtrack and π</h3>")
    html.append(table(backtrack_frame, {
        "A′B′ arcsec": "{:.12f}", "EV/VS": "{:.15f}", "Classical AB arcsec": "{:.12f}",
        "Classical AB km": "{:,.12f}", "Direct AB km": "{:,.12f}", "Classical − direct km": "{:+.12f}",
        "π event arcsec": "{:.12f}", "π 1-AU arcsec": "{:.12f}",
    }))
    html.append("<h3>Triangulation conditioning</h3>")
    html.append(table(ray_frame, {"Ray miss km": "{:.12e}", "Condition indicator": "{:.12e}"}))
    html.append("<h3>Residual change from geometric pointing</h3>")
    html.append(table(comparison_frame, {"Residual km": "{:+.12f}", "Change from geometric JPL residual km": "{:+.12f}"}))
    html.append("<h3>Geometric reconstruction closure</h3>")
    html.append(table(closure_frame, {"Pointing-derived km": "{:,.12f}", "Direct master-vector km": "{:,.12f}", "Residual km": "{:+.12f}"}))
    html.append("<h2>OUTPUT SUMMARY</h2>")
    html.append(f'<p class="path">{CSV}</p><p class="path">{HTML_FILE}</p>')
    html.append("<h2>PAPER COMPARISON</h2>")
    html.append('<p class="note">The astrometric and apparent rows determine whether retarded pointing times change the physical EV/VS ratio enough to eliminate the classical-minus-direct AB residual.</p>')
    html.append("<h2>EQUATION STATUS</h2>")
    html.append(table(status_frame, {"Residual / diagnostic km": "{:+.12f}"}))
    html.append("</div>")
    report = "".join(html)
    HTML_FILE.write_text("<html><head><meta charset='utf-8'></head><body style='margin:0;background:#000;color:#fff'>" + report + "</body></html>", encoding="utf-8")
    display(HTML(report))

    print(datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0046
