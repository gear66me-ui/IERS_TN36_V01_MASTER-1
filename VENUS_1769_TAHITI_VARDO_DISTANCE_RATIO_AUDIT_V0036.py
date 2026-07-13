# V0036
# Audit reference: Compare JPL vector distances and all distance ratios at the project φ0 epoch and the instantaneous geocentric closest-approach epoch.
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


def ensure_package(import_name: str, pip_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pip_name])


for _import_name, _pip_name in (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("astropy", "astropy"),
    ("astroquery", "astroquery"),
    ("IPython", "ipython"),
):
    ensure_package(_import_name, _pip_name)

try:
    from erfa import ErfaWarning

    warnings.filterwarnings("ignore", category=ErfaWarning)
except Exception:
    warnings.filterwarnings("ignore", message=".*dubious year.*")

import numpy as np
import pandas as pd
from IPython.display import HTML, display

VERSION = "V0036"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUTPUT_DIR = ROOT / "VENUS_1769_TAHITI_VARDO_DISTANCE_RATIO_AUDIT_V0036_OUTPUT"
OUTPUT_CSV = OUTPUT_DIR / "VENUS_1769_TAHITI_VARDO_DISTANCE_RATIO_AUDIT_V0036.csv"
OUTPUT_HTML = OUTPUT_DIR / "VENUS_1769_TAHITI_VARDO_DISTANCE_RATIO_AUDIT_V0036.html"
MASTER_CANDIDATES = (
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0034.csv",
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0033.csv",
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0036.csv",
)

BASE_COMMIT = "d55bc2274359a3014c19f257e42cc149bc458d57"
BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/"
    f"{BASE_COMMIT}/VENUS_1769_TAHITI_VARDO_HALLEY_THUMB_CHECK_V0031.py"
)
BASE_PATH = ROOT / "VENUS_1769_TAHITI_VARDO_HALLEY_THUMB_CHECK_V0031_BASE.py"
PROJECT_PHI0_UTC = "1769-06-03 22:19:15.599"


def load_base_namespace() -> dict[str, object]:
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
        raise RuntimeError("Pinned V0031 source-boundary audit failed.")
    compile(source, str(BASE_PATH), "exec")
    BASE_PATH.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {
        "__name__": "venus_v0031_base",
        "__file__": str(BASE_PATH),
    }
    exec(compile(source, str(BASE_PATH), "exec"), namespace)
    return namespace


def compatible_master(path: Path, required_columns: list[str]) -> bool:
    if not path.is_file():
        return False
    try:
        sample = pd.read_csv(path, nrows=3)
    except Exception:
        return False
    return all(column in sample.columns for column in required_columns)


def load_or_build_master(base: dict[str, object], required_columns: list[str]) -> tuple[pd.DataFrame, str]:
    for candidate in MASTER_CANDIDATES:
        if compatible_master(candidate, required_columns):
            return pd.read_csv(candidate), str(candidate)
    master = base["build_master"]()
    target = MASTER_CANDIDATES[-1]
    master.to_csv(target, index=False, float_format="%.15f")
    return master, "NEW JPL HORIZONS DOWNLOAD"


def distance_set(cache: dict[str, object], vector_at, norm, jd_tdb: float) -> dict[str, float]:
    earth_to_sun = vector_at(cache, "GEOCENTER_SUN", jd_tdb)
    earth_to_venus = vector_at(cache, "GEOCENTER_VENUS", jd_tdb)
    venus_to_sun = earth_to_sun - earth_to_venus
    ev = norm(earth_to_venus)
    vs = norm(venus_to_sun)
    es = norm(earth_to_sun)
    return {
        "EV": ev,
        "VS": vs,
        "ES": es,
        "EV/VS": ev / vs,
        "VS/EV": vs / ev,
        "ES/VS": es / vs,
        "VS/ES": vs / es,
        "EV/ES": ev / es,
        "ES/EV": es / ev,
    }


def html_table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    display_frame = frame.copy()
    for column, fmt in formats.items():
        if column in display_frame.columns:
            display_frame[column] = display_frame[column].map(
                lambda value: fmt.format(value) if pd.notna(value) else ""
            )
    return display_frame.to_html(index=False, border=0, classes="audit-table", escape=False)


def main() -> None:
    base = load_base_namespace()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prefixes = tuple(base["PREFIXES"])
    required_columns = ["JD_TDB"] + [
        f"{prefix}_{axis}_KM" for prefix in prefixes for axis in "XYZ"
    ]
    master, source = load_or_build_master(base, required_columns)

    build_cache = base["build_cache"]
    vector_at = base["vector_at"]
    reference_epoch = base["reference_epoch"]
    common_basis = base["common_basis"]
    external_contacts = base["external_contacts"]
    relative_position_arcsec = base["relative_position_arcsec"]
    fitted_direction = base["fitted_direction"]
    norm = base["norm"]
    unit = base["unit"]
    utc_text = base["utc_text"]
    Time = base["Time"]
    arcsec_per_rad = float(base["ARCSEC_PER_RAD"])

    cache = build_cache(master)
    jd_project = float(Time(PROJECT_PHI0_UTC, format="iso", scale="utc").tdb.jd)
    jd_instant = float(reference_epoch(cache))
    utc_project = str(utc_text(jd_project))
    utc_instant = str(utc_text(jd_instant))
    epoch_delta_seconds = (jd_project - jd_instant) * 86400.0

    project = distance_set(cache, vector_at, norm, jd_project)
    instant = distance_set(cache, vector_at, norm, jd_instant)

    geocentric_sun = vector_at(cache, "GEOCENTER_SUN", jd_project)
    center, xi, eta = common_basis(geocentric_sun)
    center = unit(geocentric_sun)

    contacts_t = external_contacts(cache, "TAHITI")
    contacts_v = external_contacts(cache, "VARDO")
    common_start = max(float(contacts_t[0]), float(contacts_v[0]))
    common_stop = min(float(contacts_t[1]), float(contacts_v[1]))
    minute_jds = np.asarray(cache["JD_TDB"], dtype=float)
    selected_jds = minute_jds[(minute_jds >= common_start) & (minute_jds <= common_stop)]
    if len(selected_jds) < 20:
        raise RuntimeError("Too few synchronized in-transit JPL samples.")

    tahiti_track = np.array(
        [relative_position_arcsec(cache, "TAHITI", jd, center, xi, eta) for jd in selected_jds],
        dtype=float,
    )
    vardo_track = np.array(
        [relative_position_arcsec(cache, "VARDO", jd, center, xi, eta) for jd in selected_jds],
        dtype=float,
    )
    direction_t = fitted_direction(tahiti_track)
    direction_v = fitted_direction(vardo_track)
    if float(np.dot(direction_t, direction_v)) < 0.0:
        direction_v = -direction_v
    common_direction = unit(direction_t + direction_v)
    common_normal = np.array([-common_direction[1], common_direction[0]], dtype=float)
    common_track_angle_deg = math.degrees(math.atan2(common_direction[1], common_direction[0]))

    apparent_t = relative_position_arcsec(cache, "TAHITI", jd_project, center, xi, eta)
    apparent_v = relative_position_arcsec(cache, "VARDO", jd_project, center, xi, eta)
    signed_apparent = float(np.dot(apparent_v - apparent_t, common_normal))
    if signed_apparent < 0.0:
        common_normal = -common_normal
        signed_apparent = -signed_apparent
    aprime_bprime_arcsec = signed_apparent

    topocentric_sun_t = vector_at(cache, "TAHITI_SUN", jd_project)
    topocentric_sun_v = vector_at(cache, "VARDO_SUN", jd_project)
    observer_t = geocentric_sun - topocentric_sun_t
    observer_v = geocentric_sun - topocentric_sun_v
    baseline = observer_v - observer_t
    baseline_plane = np.array(
        [float(np.dot(baseline, xi)), float(np.dot(baseline, eta))], dtype=float
    )
    ab_direct_km = abs(float(np.dot(baseline_plane, common_normal)))

    km_per_arcsec_project = project["ES"] / arcsec_per_rad
    aprime_bprime_km = aprime_bprime_arcsec * km_per_arcsec_project
    ab_direct_arcsec = ab_direct_km / km_per_arcsec_project

    ab_halley_project_arcsec = aprime_bprime_arcsec * project["EV/VS"]
    ab_halley_instant_arcsec = aprime_bprime_arcsec * instant["EV/VS"]
    ab_halley_project_km = aprime_bprime_km * project["EV/VS"]
    ab_halley_instant_km = aprime_bprime_km * instant["EV/VS"]

    factor_epoch_effect_arcsec = ab_halley_project_arcsec - ab_halley_instant_arcsec
    factor_epoch_effect_km = ab_halley_project_km - ab_halley_instant_km
    residual_project_arcsec = ab_direct_arcsec - ab_halley_project_arcsec
    residual_project_km = ab_direct_km - ab_halley_project_km
    residual_instant_arcsec = ab_direct_arcsec - ab_halley_instant_arcsec
    residual_instant_km = ab_direct_km - ab_halley_instant_km

    epoch_frame = pd.DataFrame(
        [
            ["Project φ0 track-reduction epoch", utc_project + " UTC", jd_project],
            ["JPL geocentric instantaneous closest approach", utc_instant + " UTC", jd_instant],
            ["Project minus instantaneous", f"{epoch_delta_seconds:.6f} seconds", np.nan],
        ],
        columns=["Epoch", "UTC", "JD TDB"],
    )

    distance_rows = []
    for label, symbol in (
        ("Earth → Venus", "EV"),
        ("Venus → Sun", "VS"),
        ("Earth → Sun", "ES"),
    ):
        delta = project[symbol] - instant[symbol]
        ppm = 1.0e6 * delta / instant[symbol]
        distance_rows.append([label, symbol, project[symbol], instant[symbol], delta, ppm])
    distance_frame = pd.DataFrame(
        distance_rows,
        columns=[
            "JPL vector distance",
            "Symbol",
            "Project φ0 km",
            "Instantaneous CA km",
            "Difference km",
            "Difference ppm",
        ],
    )

    ratio_labels = (
        ("Earth–Venus / Venus–Sun", "EV/VS"),
        ("Venus–Sun / Earth–Venus", "VS/EV"),
        ("Earth–Sun / Venus–Sun", "ES/VS"),
        ("Venus–Sun / Earth–Sun", "VS/ES"),
        ("Earth–Venus / Earth–Sun", "EV/ES"),
        ("Earth–Sun / Earth–Venus", "ES/EV"),
    )
    ratio_rows = []
    for label, symbol in ratio_labels:
        delta = project[symbol] - instant[symbol]
        ppm = 1.0e6 * delta / instant[symbol]
        ratio_rows.append([label, symbol, project[symbol], instant[symbol], delta, ppm])
    ratio_frame = pd.DataFrame(
        ratio_rows,
        columns=[
            "Ratio",
            "Definition",
            "Project φ0",
            "Instantaneous CA",
            "Difference",
            "Difference ppm",
        ],
    )

    geometry_frame = pd.DataFrame(
        [
            ["A′B′ direct JPL at project φ0", aprime_bprime_arcsec, aprime_bprime_km],
            ["AB direct JPL at project φ0", ab_direct_arcsec, ab_direct_km],
        ],
        columns=["JPL geometry", "Arcseconds", "Kilometers"],
    )

    reduction_frame = pd.DataFrame(
        [
            [
                "Halley using project φ0 distances",
                f"{aprime_bprime_arcsec:.12f} × {project['EV/VS']:.12f}",
                ab_halley_project_arcsec,
                ab_halley_project_km,
            ],
            [
                "Halley using instantaneous-CA distances",
                f"{aprime_bprime_arcsec:.12f} × {instant['EV/VS']:.12f}",
                ab_halley_instant_arcsec,
                ab_halley_instant_km,
            ],
            [
                "Effect of changing only the distance epoch",
                "project-factor result − instantaneous-factor result",
                factor_epoch_effect_arcsec,
                factor_epoch_effect_km,
            ],
            [
                "Direct JPL AB",
                "projected Tahiti–Vardø baseline at project φ0",
                ab_direct_arcsec,
                ab_direct_km,
            ],
            [
                "Residual using project φ0 distances",
                "direct JPL − Halley project-factor result",
                residual_project_arcsec,
                residual_project_km,
            ],
            [
                "Residual using instantaneous-CA distances",
                "direct JPL − Halley instantaneous-factor result",
                residual_instant_arcsec,
                residual_instant_km,
            ],
        ],
        columns=["Calculation", "Arithmetic", "Arcseconds", "Kilometers"],
    )

    conclusion_frame = pd.DataFrame(
        [
            ["Classical Halley factor at project φ0", project["EV/VS"], "EV / VS"],
            ["Classical Halley factor at instantaneous CA", instant["EV/VS"], "EV / VS"],
            ["Change in Halley AB caused only by epoch choice", factor_epoch_effect_km, "km"],
            ["Observed direct-JPL minus classical-Halley residual", residual_project_km, "km"],
        ],
        columns=["Diagnostic", "Value", "Unit / definition"],
    )

    audit_frames = [
        ("EPOCHS", epoch_frame),
        ("DISTANCES", distance_frame),
        ("RATIOS", ratio_frame),
        ("GEOMETRY", geometry_frame),
        ("REDUCTION", reduction_frame),
        ("CONCLUSION", conclusion_frame),
    ]
    audit_rows: list[dict[str, object]] = []
    for section, frame in audit_frames:
        for row_number, row in frame.iterrows():
            record: dict[str, object] = {"section": section, "row": int(row_number)}
            record.update({str(key): value for key, value in row.items()})
            audit_rows.append(record)
    pd.DataFrame(audit_rows).to_csv(OUTPUT_CSV, index=False, float_format="%.15f")

    css = """
<style>
.v0036-report{background:#fff;color:#000;font-family:Arial,Helvetica,sans-serif;padding:18px;border:2px solid #000;max-width:1200px}
.v0036-report h1{font-size:24px;margin:0 0 18px 0;border-bottom:3px solid #000;padding-bottom:8px}
.v0036-report h2{font-size:17px;margin:24px 0 8px 0;background:#000;color:#fff;padding:7px 10px}
.v0036-report h3{font-size:15px;margin:18px 0 6px 0;border-bottom:1px solid #000;padding-bottom:4px}
.v0036-report p{margin:6px 0 10px 0;line-height:1.4}
.audit-table{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0 14px 0}
.audit-table th{background:#000;color:#fff;border:1px solid #000;padding:7px;text-align:center;font-weight:700}
.audit-table td{border:1px solid #000;padding:7px;text-align:right;vertical-align:top}
.audit-table td:first-child,.audit-table td:nth-child(2){text-align:left}
.audit-table tbody tr:nth-child(even) td{background:#f1f1f1}
.note{border:1px solid #000;padding:10px;background:#fff;font-weight:600}
.path{font-family:monospace;font-size:12px;overflow-wrap:anywhere}
</style>
"""

    html = [css, '<div class="v0036-report">']
    html.append("<h1>1769 Venus Transit — Tahiti–Vardø JPL Distance and Ratio Audit</h1>")
    html.append("<h2>CODE INPUTS</h2>")
    html.append(f"<p><b>JPL source:</b> {source}</p>")
    html.append(f"<p><b>Common track angle:</b> {common_track_angle_deg:.12f}°</p>")
    html.append("<h2>COMMENTS</h2>")
    html.append('<p class="note">This audit compares the same JPL vectors at two clearly separated epochs: the project φ0 track-reduction epoch and the independently solved geocentric instantaneous closest-approach epoch.</p>')
    html.append('<p class="note">The classical Halley factor is always EV/VS. No AB/A′B′ ratio is used as a distance factor in this audit.</p>')
    html.append("<h2>RESULTS</h2>")
    html.append("<h3>Epoch identification</h3>")
    html.append(html_table(epoch_frame, {"JD TDB": "{:.12f}"}))
    html.append("<h3>JPL vector distances</h3>")
    html.append(html_table(distance_frame, {
        "Project φ0 km": "{:,.12f}",
        "Instantaneous CA km": "{:,.12f}",
        "Difference km": "{:+,.12f}",
        "Difference ppm": "{:+.9f}",
    }))
    html.append("<h3>All distance-ratio permutations</h3>")
    html.append(html_table(ratio_frame, {
        "Project φ0": "{:.12f}",
        "Instantaneous CA": "{:.12f}",
        "Difference": "{:+.12f}",
        "Difference ppm": "{:+.9f}",
    }))
    html.append("<h3>JPL track-reduction geometry at project φ0</h3>")
    html.append(html_table(geometry_frame, {
        "Arcseconds": "{:.12f}",
        "Kilometers": "{:,.12f}",
    }))
    html.append("<h3>Classical Halley calculation using each JPL distance epoch</h3>")
    html.append(html_table(reduction_frame, {
        "Arcseconds": "{:+.12f}",
        "Kilometers": "{:+,.12f}",
    }))
    html.append("<h2>OUTPUT SUMMARY</h2>")
    html.append(f'<p class="path">{OUTPUT_CSV}</p>')
    html.append(f'<p class="path">{OUTPUT_HTML}</p>')
    html.append("<h2>PAPER COMPARISON</h2>")
    html.append(html_table(conclusion_frame, {"Value": "{:+,.12f}"}))
    html.append("<h2>EQUATION STATUS</h2>")
    html.append('<p class="note">The epoch-input test is complete: compare the “Effect of changing only the distance epoch” with the full direct-JPL minus classical-Halley residual. If the epoch effect is tiny, the ≈19.566 km discrepancy is not caused by using the wrong JPL distance epoch.</p>')
    html.append("</div>")
    report_html = "".join(html)
    OUTPUT_HTML.write_text("<html><head><meta charset='utf-8'></head><body>" + report_html + "</body></html>", encoding="utf-8")
    display(HTML(report_html))

    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0036
