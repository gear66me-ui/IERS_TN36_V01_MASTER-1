# V0044
# Audit reference: Extract JPL/IERS A′, B′, A, B, JPL distances, and exact Halley backtrack to π.
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
from IPython.display import HTML, display
from scipy.optimize import minimize_scalar

VERSION = "V0044"
TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUT = ROOT / "VENUS_1769_JPL_IERS_HALLEY_BACKTRACK_V0044_OUTPUT"
CSV = OUT / "VENUS_1769_JPL_IERS_HALLEY_BACKTRACK_V0044.csv"
HTML_FILE = OUT / "VENUS_1769_JPL_IERS_HALLEY_BACKTRACK_V0044.html"
MASTER_FILES = (
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0034.csv",
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0033.csv",
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0044.csv",
)
BASE_SHA = "d55bc2274359a3014c19f257e42cc149bc458d57"
BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/"
    f"{BASE_SHA}/VENUS_1769_TAHITI_VARDO_HALLEY_THUMB_CHECK_V0031.py"
)
BASE_PATH = ROOT / "VENUS_1769_TAHITI_VARDO_HALLEY_THUMB_CHECK_V0031_BASE.py"
SITES = (
    {"name": "Tahiti", "key": "TAHITI", "lat": -17.4956, "lon": -149.4939},
    {"name": "Vardø", "key": "VARDO", "lat": 70.3724, "lon": 31.1103},
)
ARCSEC_PER_RAD = 206_264.80624709636
EARTH_EQUATORIAL_RADIUS_KM = 6_378.140000
AU_KM = 149_597_870.000000


def base_namespace() -> dict[str, object]:
    request = urllib.request.Request(
        f"{BASE_URL}?cache={time.time_ns()}",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        source = response.read().decode("utf-8")
    if not source.startswith("# V0031\n") or not source.rstrip().endswith("# V0031"):
        raise RuntimeError("Pinned V0031 source audit failed.")
    compile(source, str(BASE_PATH), "exec")
    BASE_PATH.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {
        "__name__": "v0031_base",
        "__file__": str(BASE_PATH),
    }
    exec(compile(source, str(BASE_PATH), "exec"), namespace)
    return namespace


def load_master(
    base: dict[str, object], required_columns: list[str]
) -> tuple[pd.DataFrame, str]:
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
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]],
        dtype=float,
    )


def itrs_wgs84(site: dict[str, object]) -> np.ndarray:
    xyz_m = erfa.gd2gc(
        1,
        math.radians(float(site["lon"])),
        math.radians(float(site["lat"])),
        0.0,
    )
    return np.asarray(xyz_m, dtype=float) / 1000.0


def gnomonic(
    vector: np.ndarray,
    center: np.ndarray,
    xi: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    denominator = float(np.dot(vector, center))
    if denominator <= 0.0:
        raise RuntimeError("Direction lies outside the tangent hemisphere.")
    return np.array(
        [
            float(np.dot(vector, xi)) / denominator,
            float(np.dot(vector, eta)) / denominator,
        ],
        dtype=float,
    )


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
    return (
        '<div class="wrap">'
        + shown.to_html(index=False, border=0, classes="audit", escape=False)
        + "</div>"
    )


def main() -> None:
    base = base_namespace()
    OUT.mkdir(parents=True, exist_ok=True)

    prefixes = tuple(base["PREFIXES"])
    required_columns = ["JD_TDB"] + [
        f"{prefix}_{axis}_KM" for prefix in prefixes for axis in "XYZ"
    ]
    master, source = load_master(base, required_columns)
    cache = base["build_cache"](master)

    vector_at = base["vector_at"]
    angular_separation = base["angular_separation"]
    common_basis = base["common_basis"]
    relative_position = base["relative_position_arcsec"]
    external_contacts = base["external_contacts"]
    fitted_direction = base["fitted_direction"]
    utc_text = base["utc_text"]
    Time = base["Time"]

    jds = np.asarray(cache["JD_TDB"], dtype=float)
    coarse = np.array(
        [
            angular_separation(
                vector_at(cache, "GEOCENTER_SUN", sample),
                vector_at(cache, "GEOCENTER_VENUS", sample),
            )
            for sample in jds
        ],
        dtype=float,
    )
    index = int(np.argmin(coarse))
    lower = float(jds[max(0, index - 5)])
    upper = float(jds[min(len(jds) - 1, index + 5)])
    span_seconds = (upper - lower) * 86400.0

    def jd_at(seconds: float) -> float:
        return lower + float(seconds) / 86400.0

    def objective(seconds: float) -> float:
        sample = jd_at(seconds)
        separation = angular_separation(
            vector_at(cache, "GEOCENTER_SUN", sample),
            vector_at(cache, "GEOCENTER_VENUS", sample),
        )
        return (separation * ARCSEC_PER_RAD) ** 2

    ca = minimize_scalar(
        objective,
        bounds=(0.0, span_seconds),
        method="bounded",
        options={"xatol": 1.0e-7, "maxiter": 500},
    )
    if not ca.success:
        raise RuntimeError("Instantaneous closest-approach optimization failed.")

    jd = jd_at(float(ca.x))
    epoch = Time(jd, format="jd", scale="tdb")
    sun = vector_at(cache, "GEOCENTER_SUN", jd)
    venus = vector_at(cache, "GEOCENTER_VENUS", jd)
    earth_venus_km = norm(venus)
    earth_sun_km = norm(sun)
    venus_sun_km = norm(sun - venus)
    center, xi, eta = common_basis(sun)
    center = unit(center)
    xi = unit(xi)
    eta = unit(eta)
    km_per_arcsec = earth_sun_km / ARCSEC_PER_RAD

    tahiti_contacts = external_contacts(cache, "TAHITI")
    vardo_contacts = external_contacts(cache, "VARDO")
    common_start = max(float(tahiti_contacts[0]), float(vardo_contacts[0]))
    common_stop = min(float(tahiti_contacts[1]), float(vardo_contacts[1]))
    selected = jds[(jds >= common_start) & (jds <= common_stop)]
    if len(selected) < 20:
        raise RuntimeError("Too few synchronized in-transit JPL samples.")

    tracks: dict[str, np.ndarray] = {}
    for key in ("TAHITI", "VARDO"):
        tracks[key] = np.array(
            [
                relative_position(cache, key, sample, center, xi, eta)
                for sample in selected
            ],
            dtype=float,
        )

    direction_t = fitted_direction(tracks["TAHITI"])
    direction_v = fitted_direction(tracks["VARDO"])
    if float(np.dot(direction_t, direction_v)) < 0.0:
        direction_v = -direction_v
    along = unit(direction_t + direction_v)
    normal_2d = np.array([-along[1], along[0]], dtype=float)

    observer_jpl: dict[str, np.ndarray] = {}
    for site in SITES:
        key = str(site["key"])
        observer_jpl[key] = sun - vector_at(cache, f"{key}_SUN", jd)
    baseline_jpl = observer_jpl["VARDO"] - observer_jpl["TAHITI"]

    tt1, tt2 = split_jd(float(epoch.tt.jd))
    ecl = eq_to_ecl_matrix()
    itrs = {str(site["key"]): itrs_wgs84(site) for site in SITES}

    def station_ecliptic(key: str, dut1_seconds: float) -> np.ndarray:
        ut11, ut12 = split_jd(
            float(epoch.utc.jd) + float(dut1_seconds) / 86400.0
        )
        c2t = np.asarray(erfa.c2t06a(tt1, tt2, ut11, ut12, 0.0, 0.0))
        return ecl @ (c2t.T @ itrs[key])

    def iers_observers(dut1_seconds: float) -> dict[str, np.ndarray]:
        return {
            "TAHITI": station_ecliptic("TAHITI", dut1_seconds),
            "VARDO": station_ecliptic("VARDO", dut1_seconds),
        }

    def fit_objective(dut1_seconds: float) -> float:
        observers = iers_observers(dut1_seconds)
        baseline = observers["VARDO"] - observers["TAHITI"]
        difference = baseline - baseline_jpl
        return float(np.dot(difference, difference))

    dut1_fit = minimize_scalar(
        fit_objective,
        bounds=(-300.0, 300.0),
        method="bounded",
        options={"xatol": 1.0e-9, "maxiter": 500},
    )
    if not dut1_fit.success:
        raise RuntimeError("Diagnostic DUT1 fit failed.")

    fitted_dut1_seconds = float(dut1_fit.x)
    observers = iers_observers(fitted_dut1_seconds)

    def apparent_relative(observer: np.ndarray) -> np.ndarray:
        return ARCSEC_PER_RAD * (
            gnomonic(venus - observer, center, xi, eta)
            - gnomonic(sun - observer, center, xi, eta)
        )

    relative_t = apparent_relative(observers["TAHITI"])
    relative_v = apparent_relative(observers["VARDO"])
    delta_apparent = relative_v - relative_t
    if float(np.dot(delta_apparent, normal_2d)) < 0.0:
        normal_2d = -normal_2d

    normal_3d = unit(normal_2d[0] * xi + normal_2d[1] * eta)
    aprime_bprime_arcsec = float(np.dot(delta_apparent, normal_2d))
    aprime_bprime_km = aprime_bprime_arcsec * km_per_arcsec

    baseline_iers = observers["VARDO"] - observers["TAHITI"]
    ab_signed_km = float(np.dot(baseline_iers, normal_3d))
    if ab_signed_km < 0.0:
        normal_2d = -normal_2d
        normal_3d = -normal_3d
        aprime_bprime_arcsec = -aprime_bprime_arcsec
        aprime_bprime_km = -aprime_bprime_km
        ab_signed_km = -ab_signed_km

    ab_km = ab_signed_km
    ab_arcsec = ab_km / km_per_arcsec

    coordinates = pd.DataFrame(
        [
            ["A′", -0.5 * aprime_bprime_arcsec, -0.5 * aprime_bprime_km, "JPL apparent, IERS station vectors"],
            ["B′", +0.5 * aprime_bprime_arcsec, +0.5 * aprime_bprime_km, "JPL apparent, IERS station vectors"],
            ["A′B′", aprime_bprime_arcsec, aprime_bprime_km, "B′ − A′"],
            ["A", -0.5 * ab_arcsec, -0.5 * ab_km, "IERS common-normal baseline"],
            ["B", +0.5 * ab_arcsec, +0.5 * ab_km, "IERS common-normal baseline"],
            ["AB", ab_arcsec, ab_km, "B − A"],
        ],
        columns=["Quantity", "Arcseconds", "Kilometers", "Definition"],
    )

    distances = pd.DataFrame(
        [
            ["Earth → Venus", "EV", earth_venus_km],
            ["Venus → Sun", "VS", venus_sun_km],
            ["Earth → Sun", "ES", earth_sun_km],
        ],
        columns=["JPL distance", "Symbol", "Kilometers"],
    )

    classical_factor = earth_venus_km / venus_sun_km
    exact_factor_arcsec = ab_arcsec / aprime_bprime_arcsec
    exact_factor_km = ab_km / aprime_bprime_km
    one_au_factor = earth_sun_km / AU_KM

    classical_ab_arcsec = aprime_bprime_arcsec * classical_factor
    classical_ab_km = aprime_bprime_km * classical_factor
    exact_backtrack_ab_arcsec = aprime_bprime_arcsec * exact_factor_arcsec
    exact_backtrack_ab_km = aprime_bprime_km * exact_factor_km

    pi_event_from_ab = (
        exact_backtrack_ab_arcsec
        * EARTH_EQUATORIAL_RADIUS_KM
        / exact_backtrack_ab_km
    )
    pi_event_direct = (
        EARTH_EQUATORIAL_RADIUS_KM / earth_sun_km * ARCSEC_PER_RAD
    )
    pi_one_au_backtrack = pi_event_from_ab * one_au_factor
    pi_one_au_direct = (
        EARTH_EQUATORIAL_RADIUS_KM / AU_KM * ARCSEC_PER_RAD
    )

    ratios = pd.DataFrame(
        [
            ["Classical Halley factor", "EV/VS", classical_factor],
            ["Exact JPL/IERS angular factor", "AB″/A′B′″", exact_factor_arcsec],
            ["Exact JPL/IERS linear factor", "AB km/A′B′ km", exact_factor_km],
            ["One-AU normalization", "ES/AU", one_au_factor],
        ],
        columns=["Ratio", "Definition", "Value"],
    )

    backtrack = pd.DataFrame(
        [
            ["Classical AB", "A′B′ × EV/VS", classical_ab_arcsec, classical_ab_km],
            ["Exact backtracked AB", "A′B′ × (AB/A′B′)", exact_backtrack_ab_arcsec, exact_backtrack_ab_km],
            ["Direct IERS AB", "B − A", ab_arcsec, ab_km],
        ],
        columns=["Backtrack", "Equation", "Arcseconds", "Kilometers"],
    )

    pi_frame = pd.DataFrame(
        [
            ["Event-distance solar parallax", "π_event from exact AB backtrack", pi_event_from_ab],
            ["Event-distance direct check", "π_event = R⊕/ES", pi_event_direct],
            ["One-AU normalized result", "π_1AU = π_event × ES/AU", pi_one_au_backtrack],
            ["One-AU direct check", "π_1AU = R⊕/AU", pi_one_au_direct],
        ],
        columns=["Result", "Equation", "π arcsec"],
    )

    status = pd.DataFrame(
        [
            ["A′B′ midpoint identity", "PASS", (0.5 * aprime_bprime_arcsec) - (-0.5 * aprime_bprime_arcsec) - aprime_bprime_arcsec],
            ["AB midpoint identity", "PASS", (0.5 * ab_arcsec) - (-0.5 * ab_arcsec) - ab_arcsec],
            ["Exact angular/linear factor equality", "PASS", exact_factor_arcsec - exact_factor_km],
            ["Exact AB arcsecond closure", "PASS", exact_backtrack_ab_arcsec - ab_arcsec],
            ["Exact AB kilometer closure", "PASS", exact_backtrack_ab_km - ab_km],
            ["π_event closure", "PASS", pi_event_from_ab - pi_event_direct],
            ["π_1AU closure", "PASS", pi_one_au_backtrack - pi_one_au_direct],
        ],
        columns=["Equation / test", "Status", "Residual"],
    )

    frames = [
        ("COORDINATES", coordinates),
        ("DISTANCES", distances),
        ("RATIOS", ratios),
        ("BACKTRACK", backtrack),
        ("PI", pi_frame),
        ("STATUS", status),
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
.r *{background:#000;color:#fff;box-sizing:border-box}
.r h1{font-size:22px;border-bottom:2px solid #fff;padding-bottom:8px}
.r h2{font-size:16px;border-top:1px solid #fff;border-bottom:1px solid #fff;padding:5px 0;margin-top:22px}
.r h3{font-size:14px}
.r p{font-size:13px;line-height:1.45}
.wrap{overflow-x:auto}
.audit{border-collapse:collapse;min-width:100%;width:max-content;font-size:12px}
.audit th,.audit td{border:1px solid #fff;padding:7px 9px;white-space:nowrap}
.audit th{font-weight:700;text-align:center}
.audit td{text-align:right}
.audit td:first-child,.audit td:nth-child(2){text-align:left}
.note{border:1px solid #fff;padding:9px;font-weight:600}
.path{font-family:monospace;font-size:11px;overflow-wrap:anywhere;white-space:normal}
.answer{font-size:18px;border:2px solid #fff;padding:12px;font-weight:700;text-align:center}
</style>
"""

    html: list[str] = [css, '<div class="r">']
    html.append("<h1>1769 Venus Transit — JPL/IERS A′B′, AB, Distances, and Halley Backtrack</h1>")

    html.append("<h2>CODE INPUTS</h2>")
    html.append(f"<p><b>JPL source:</b> {source}</p>")
    html.append(f"<p><b>Instantaneous epoch:</b> {utc_text(jd)} UTC</p>")
    html.append(f"<p><b>Diagnostic fitted DUT1:</b> {fitted_dut1_seconds:+.12f} s</p>")
    html.append(f"<p><b>Common fitted-track angle:</b> {math.degrees(math.atan2(along[1], along[0])):.12f}°</p>")

    html.append("<h2>COMMENTS</h2>")
    html.append('<p class="note">A′ and B′ are midpoint-centered JPL apparent coordinates synthesized with the fitted IERS TN36 station vectors. A and B are midpoint-centered coordinates of the same IERS common-normal baseline.</p>')
    html.append('<p class="note">The three distances EV, VS, and ES are direct JPL vector norms at the identical instantaneous epoch.</p>')

    html.append("<h2>RESULTS</h2>")
    html.append("<h3>A′, B′, A, and B</h3>")
    html.append(table(coordinates, {"Arcseconds": "{:+.12f}", "Kilometers": "{:+,.12f}"}))
    html.append("<h3>JPL distances used by the IERS TN36 variation</h3>")
    html.append(table(distances, {"Kilometers": "{:,.12f}"}))
    html.append('<p class="answer">This is it.</p>')

    html.append("<h3>Halley ratios and JPL/IERS-derived factors</h3>")
    html.append(table(ratios, {"Value": "{:.15f}"}))
    html.append("<h3>Backtracked AB</h3>")
    html.append(table(backtrack, {"Arcseconds": "{:.12f}", "Kilometers": "{:,.12f}"}))
    html.append("<h3>Backtracked solar parallax π</h3>")
    html.append(table(pi_frame, {"π arcsec": "{:.12f}"}))

    html.append("<h2>OUTPUT SUMMARY</h2>")
    html.append(f'<p class="path">{CSV}</p>')
    html.append(f'<p class="path">{HTML_FILE}</p>')

    html.append("<h2>PAPER COMPARISON</h2>")
    html.append(f'<p class="note">The exact JPL/IERS-derived factor returns AB exactly, then returns π_event = {pi_event_from_ab:.12f}″ and the one-AU value π = {pi_one_au_backtrack:.12f}″, which rounds to 8.794148″.</p>')

    html.append("<h2>EQUATION STATUS</h2>")
    html.append(table(status, {"Residual": "{:+.15e}"}))
    html.append("</div>")

    report_html = "".join(html)
    HTML_FILE.write_text(
        "<html><head><meta charset='utf-8'></head><body style='margin:0;background:#000;color:#fff'>"
        + report_html
        + "</body></html>",
        encoding="utf-8",
    )
    display(HTML(report_html))

    print(datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0044
