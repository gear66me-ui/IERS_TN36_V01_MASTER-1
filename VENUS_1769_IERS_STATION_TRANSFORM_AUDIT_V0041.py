# V0041
# Audit reference: Explicit IERS TN36-style ITRS-to-GCRS station transformation versus JPL topocentric observer vectors.
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

VERSION = "V0041"
TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUT = ROOT / "VENUS_1769_IERS_STATION_TRANSFORM_AUDIT_V0041_OUTPUT"
CSV = OUT / "VENUS_1769_IERS_STATION_TRANSFORM_AUDIT_V0041.csv"
HTML_FILE = OUT / "VENUS_1769_IERS_STATION_TRANSFORM_AUDIT_V0041.html"
MASTER_FILES = (
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0034.csv",
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0033.csv",
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0041.csv",
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
ARCSEC = 206264.80624709636


def base_namespace() -> dict[str, object]:
    request = urllib.request.Request(
        f"{BASE_URL}?cache={time.time_ns()}",
        headers={"User-Agent": "Mozilla/5.0", "Cache-Control": "no-cache"},
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


def load_master(base: dict[str, object], columns: list[str]) -> tuple[pd.DataFrame, str]:
    for path in MASTER_FILES:
        if path.is_file():
            try:
                frame = pd.read_csv(path)
                if all(column in frame.columns for column in columns):
                    return frame, str(path)
            except Exception:
                continue
    frame = base["build_master"]()
    frame.to_csv(MASTER_FILES[-1], index=False, float_format="%.15f")
    return frame, "NEW JPL HORIZONS DOWNLOAD"


def split_jd(value: float) -> tuple[float, float]:
    whole = math.floor(value)
    return float(whole), float(value - whole)


def eq_to_ecl_matrix() -> np.ndarray:
    epsilon = float(erfa.obl80(2451545.0, 0.0))
    c, s = math.cos(epsilon), math.sin(epsilon)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def itrs_wgs84(site: dict[str, object]) -> np.ndarray:
    xyz_m = erfa.gd2gc(
        1,
        math.radians(float(site["lon"])),
        math.radians(float(site["lat"])),
        0.0,
    )
    return np.asarray(xyz_m, dtype=float) / 1000.0


def norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    shown = frame.copy()
    for column, pattern in (formats or {}).items():
        if column in shown.columns:
            shown[column] = shown[column].map(
                lambda value: pattern.format(value) if pd.notna(value) else ""
            )
    return '<div class="wrap">' + shown.to_html(
        index=False, border=0, classes="audit", escape=False
    ) + "</div>"


def main() -> None:
    base = base_namespace()
    OUT.mkdir(parents=True, exist_ok=True)
    prefixes = tuple(base["PREFIXES"])
    columns = ["JD_TDB"] + [f"{prefix}_{axis}_KM" for prefix in prefixes for axis in "XYZ"]
    master, source = load_master(base, columns)
    cache = base["build_cache"](master)
    vector_at = base["vector_at"]
    angular_separation = base["angular_separation"]
    common_basis = base["common_basis"]
    relative_position = base["relative_position_arcsec"]
    contacts = base["external_contacts"]
    fitted_direction = base["fitted_direction"]
    unit = base["unit"]
    utc_text = base["utc_text"]
    Time = base["Time"]

    jds = np.asarray(cache["JD_TDB"], dtype=float)
    coarse = np.array([
        angular_separation(vector_at(cache, "GEOCENTER_SUN", jd), vector_at(cache, "GEOCENTER_VENUS", jd))
        for jd in jds
    ])
    index = int(np.argmin(coarse))
    lower = float(jds[max(0, index - 5)])
    upper = float(jds[min(len(jds) - 1, index + 5)])
    span_s = (upper - lower) * 86400.0

    def jd_at(seconds: float) -> float:
        return lower + float(seconds) / 86400.0

    def ca_objective(seconds: float) -> float:
        jd = jd_at(seconds)
        angle = angular_separation(
            vector_at(cache, "GEOCENTER_SUN", jd),
            vector_at(cache, "GEOCENTER_VENUS", jd),
        )
        return (angle * ARCSEC) ** 2

    ca = minimize_scalar(ca_objective, bounds=(0.0, span_s), method="bounded", options={"xatol": 1e-7})
    if not ca.success:
        raise RuntimeError("Closest-approach optimizer failed.")
    jd = jd_at(float(ca.x))
    epoch = Time(jd, format="jd", scale="tdb")

    sun = vector_at(cache, "GEOCENTER_SUN", jd)
    venus = vector_at(cache, "GEOCENTER_VENUS", jd)
    center, xi, eta = common_basis(sun)
    center = unit(center)

    t_contacts, v_contacts = contacts(cache, "TAHITI"), contacts(cache, "VARDO")
    start, stop = max(t_contacts[0], v_contacts[0]), min(t_contacts[1], v_contacts[1])
    selected = jds[(jds >= start) & (jds <= stop)]
    tracks = {}
    for key in ("TAHITI", "VARDO"):
        tracks[key] = np.array([
            relative_position(cache, key, sample, center, xi, eta) for sample in selected
        ])
    direction_t = fitted_direction(tracks["TAHITI"])
    direction_v = fitted_direction(tracks["VARDO"])
    if float(np.dot(direction_t, direction_v)) < 0.0:
        direction_v = -direction_v
    along = unit(direction_t + direction_v)
    normal = np.array([-along[1], along[0]])
    apparent_t = relative_position(cache, "TAHITI", jd, center, xi, eta)
    apparent_v = relative_position(cache, "VARDO", jd, center, xi, eta)
    apparent_delta = apparent_v - apparent_t
    if float(np.dot(apparent_delta, normal)) < 0.0:
        normal = -normal

    observer_jpl: dict[str, np.ndarray] = {}
    closure_rows = []
    for site in SITES:
        key = str(site["key"])
        from_sun = sun - vector_at(cache, f"{key}_SUN", jd)
        from_venus = venus - vector_at(cache, f"{key}_VENUS", jd)
        observer_jpl[key] = from_sun
        closure_rows.append([
            site["name"], norm(from_sun), norm(from_venus), norm(from_sun - from_venus)
        ])
    baseline_jpl = observer_jpl["VARDO"] - observer_jpl["TAHITI"]

    tt1, tt2 = split_jd(float(epoch.tt.jd))
    ecl = eq_to_ecl_matrix()
    itrs = {str(site["key"]): itrs_wgs84(site) for site in SITES}

    def station_ecl(key: str, dut1_s: float) -> np.ndarray:
        ut11, ut12 = split_jd(float(epoch.utc.jd) + float(dut1_s) / 86400.0)
        c2t = np.asarray(erfa.c2t06a(tt1, tt2, ut11, ut12, 0.0, 0.0))
        return ecl @ (c2t.T @ itrs[key])

    def baseline_iers(dut1_s: float) -> np.ndarray:
        return station_ecl("VARDO", dut1_s) - station_ecl("TAHITI", dut1_s)

    def fit_objective(dut1_s: float) -> float:
        delta = baseline_iers(dut1_s) - baseline_jpl
        return float(np.dot(delta, delta))

    fitted = minimize_scalar(fit_objective, bounds=(-300.0, 300.0), method="bounded", options={"xatol": 1e-9})
    if not fitted.success:
        raise RuntimeError("Diagnostic DUT1 fit failed.")
    dut1_fit = float(fitted.x)
    baseline_nominal = baseline_iers(0.0)
    baseline_fitted = baseline_iers(dut1_fit)

    def components(vector: np.ndarray) -> tuple[float, float, float, float]:
        screen = np.array([float(np.dot(vector, xi)), float(np.dot(vector, eta))])
        return screen[0], screen[1], float(np.dot(screen, along)), float(np.dot(screen, normal))

    es, ev, vs = norm(sun), norm(venus), norm(sun - venus)
    scale = es / ARCSEC
    aprime_bprime_as = abs(float(np.dot(apparent_delta, normal)))
    aprime_bprime_km = aprime_bprime_as * scale
    halley_factor = ev / vs
    halley_ab_km = aprime_bprime_km * halley_factor
    jpl_ab_km = abs(components(baseline_jpl)[3])

    scope = pd.DataFrame([
        ["ITRS station coordinates", "ERFA WGS84 geodetic conversion", "USED"],
        ["Polar motion", "xp = yp = 0", "ASSUMED — 1769 EOP unavailable"],
        ["ITRS → GCRS", "ERFA c2t06a transpose = Q·R·W", "USED"],
        ["GCRS → JPL ecliptic", "J2000 mean-obliquity rotation", "USED"],
        ["DUT1=0", "Nominal historical assumption", "USED NOMINAL"],
        ["Fitted DUT1", "Fit to JPL baseline only", "DIAGNOSTIC — NOT INDEPENDENT"],
    ], columns=["Component", "Definition", "Status"])

    epoch_frame = pd.DataFrame([
        ["Instantaneous closest approach UTC", utc_text(jd) + " UTC"],
        ["JD TDB", jd],
        ["Common fitted-track angle deg", math.degrees(math.atan2(along[1], along[0]))],
        ["Earth–Sun scale km/arcsec", scale],
    ], columns=["Quantity", "Value"])

    closure = pd.DataFrame(closure_rows, columns=[
        "Station", "JPL radius from Sun km", "JPL radius from Venus km", "Observer closure km"
    ])

    itrs_frame = pd.DataFrame([
        [site["name"], site["lat"], site["lon"], *itrs[str(site["key"])], norm(itrs[str(site["key"])])]
        for site in SITES
    ], columns=["Station", "Latitude deg", "Longitude deg east", "ITRS X km", "ITRS Y km", "ITRS Z km", "ITRS radius km"])

    baseline_rows = []
    models = (
        ("JPL topocentric-derived", "USED REFERENCE", 0.0, baseline_jpl),
        ("IERS Q·R·W nominal", "DUT1=0", 0.0, baseline_nominal),
        ("IERS Q·R·W fitted", "DIAGNOSTIC", dut1_fit, baseline_fitted),
    )
    for label, status, dut1_s, vector in models:
        sx, sy, parallel, perpendicular = components(vector)
        used_ab = abs(perpendicular)
        baseline_rows.append([
            label, status, dut1_s, norm(vector), sx, sy, parallel, used_ab,
            used_ab / scale, used_ab - jpl_ab_km, used_ab - halley_ab_km,
        ])
    baseline_frame = pd.DataFrame(baseline_rows, columns=[
        "Baseline model", "Status", "DUT1 seconds", "3D magnitude km", "Screen ξ km", "Screen η km",
        "Along-track km", "Common-normal AB km", "AB arcsec", "AB − JPL km", "AB − Halley km"
    ])

    station_rows = []
    for site in SITES:
        key = str(site["key"])
        nominal, matched, reference = station_ecl(key, 0.0), station_ecl(key, dut1_fit), observer_jpl[key]
        station_rows.append([
            site["name"], norm(reference), norm(nominal), norm(matched),
            norm(nominal - reference), norm(matched - reference),
        ])
    station_frame = pd.DataFrame(station_rows, columns=[
        "Station", "JPL observer radius km", "IERS nominal radius km", "IERS fitted radius km",
        "Nominal vector difference km", "Fitted vector difference km"
    ])

    context = pd.DataFrame([
        ["JPL A′B′", aprime_bprime_as, aprime_bprime_km],
        ["JPL EV/VS", halley_factor, np.nan],
        ["Halley AB = JPL A′B′ × EV/VS", halley_ab_km / scale, halley_ab_km],
        ["JPL station-transformed AB", jpl_ab_km / scale, jpl_ab_km],
        ["IERS nominal station-transformed AB", abs(components(baseline_nominal)[3]) / scale, abs(components(baseline_nominal)[3])],
        ["IERS fitted station-transformed AB", abs(components(baseline_fitted)[3]) / scale, abs(components(baseline_fitted)[3])],
    ], columns=["Quantity", "Arcseconds / factor", "Kilometers"])

    ut11, ut12 = split_jd(float(epoch.utc.jd) + dut1_fit / 86400.0)
    c2t = np.asarray(erfa.c2t06a(tt1, tt2, ut11, ut12, 0.0, 0.0))
    orth = float(np.max(np.abs(c2t @ c2t.T - np.identity(3))))
    determinant = float(np.linalg.det(c2t) - 1.0)
    status = pd.DataFrame([
        ["JPL Sun/Venus observer closure", "PASS" if closure["Observer closure km"].max() < 1e-3 else "CHECK", closure["Observer closure km"].max()],
        ["IERS matrix orthogonality", "PASS" if orth < 1e-12 else "CHECK", orth],
        ["IERS matrix determinant", "PASS" if abs(determinant) < 1e-12 else "CHECK", determinant],
        ["IERS nominal AB − JPL AB", "DIAGNOSTIC", abs(components(baseline_nominal)[3]) - jpl_ab_km],
        ["IERS fitted AB − JPL AB", "DIAGNOSTIC", abs(components(baseline_fitted)[3]) - jpl_ab_km],
        ["JPL AB − Halley AB", "UNRESOLVED", jpl_ab_km - halley_ab_km],
    ], columns=["Equation / test", "Status", "Residual / diagnostic"])

    frames = [
        ("IERS transformation scope", scope, {}),
        ("Reference epoch and screen", epoch_frame, {}),
        ("JPL observer-vector closure", closure, {
            "JPL radius from Sun km": "{:,.12f}", "JPL radius from Venus km": "{:,.12f}", "Observer closure km": "{:.12e}"
        }),
        ("ITRS station coordinates", itrs_frame, {
            "Latitude deg": "{:+.8f}", "Longitude deg east": "{:+.8f}", "ITRS X km": "{:+,.9f}",
            "ITRS Y km": "{:+,.9f}", "ITRS Z km": "{:+,.9f}", "ITRS radius km": "{:,.9f}"
        }),
        ("JPL versus IERS Tahiti–Vardø baseline", baseline_frame, {
            "DUT1 seconds": "{:+.12f}", "3D magnitude km": "{:,.12f}", "Screen ξ km": "{:+,.12f}",
            "Screen η km": "{:+,.12f}", "Along-track km": "{:+,.12f}", "Common-normal AB km": "{:,.12f}",
            "AB arcsec": "{:.12f}", "AB − JPL km": "{:+,.12f}", "AB − Halley km": "{:+,.12f}"
        }),
        ("Per-station vector comparison", station_frame, {
            column: "{:,.12f}" for column in station_frame.columns if column != "Station"
        }),
        ("Halley context with station model isolated", context, {
            "Arcseconds / factor": "{:.12f}", "Kilometers": "{:,.12f}"
        }),
        ("Equation status", status, {"Residual / diagnostic": "{:+.12e}"}),
    ]

    records = []
    for section_name, frame, _formats in frames:
        for row_number, row in frame.iterrows():
            record = {"section": section_name, "row": int(row_number)}
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
    body = [css, '<div class="r">', "<h1>1769 Venus Transit — IERS Station Transformation Audit</h1>"]
    body += ["<h2>CODE INPUTS</h2>", f"<p><b>JPL source:</b> {source}</p>", f"<p><b>Instantaneous epoch:</b> {utc_text(jd)} UTC</p>"]
    body += ["<h2>COMMENTS</h2>", '<p class="note">JPL A′B′ and the common fitted-track normal are held fixed. Only the Tahiti–Vardø station transformation is changed.</p>', '<p class="note">The DUT1=0 row is nominal. The fitted-DUT1 row is diagnostic because it is fitted to the JPL baseline and is not an independent historical Earth-orientation solution.</p>', "<h2>RESULTS</h2>"]
    for title, frame, formats in frames[:-1]:
        body += [f"<h3>{title}</h3>", table(frame, formats)]
    body += ["<h2>OUTPUT SUMMARY</h2>", f'<p class="path">{CSV}</p>', f'<p class="path">{HTML_FILE}</p>']
    body += ["<h2>PAPER COMPARISON</h2>", f'<p class="note">Fitted DUT1 diagnostic: {dut1_fit:+.12f} s. Compare the nominal and fitted IERS common-normal AB values with the JPL and Halley rows.</p>']
    body += ["<h2>EQUATION STATUS</h2>", table(status, {"Residual / diagnostic": "{:+.12e}"}), "</div>"]
    report = "".join(body)
    HTML_FILE.write_text("<html><head><meta charset='utf-8'></head><body style='margin:0;background:#000;color:#fff'>" + report + "</body></html>", encoding="utf-8")
    display(HTML(report))
    print(datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0041
