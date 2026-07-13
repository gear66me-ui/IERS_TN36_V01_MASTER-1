# V0037
# Audit reference: Compare track-reduction and instantaneous JPL distance extraction at one identical project φ0 epoch.
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

VERSION = "V0037"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUTPUT_DIR = ROOT / "VENUS_1769_TAHITI_VARDO_IDENTICAL_EPOCH_DISTANCE_AUDIT_V0037_OUTPUT"
OUTPUT_CSV = OUTPUT_DIR / "VENUS_1769_TAHITI_VARDO_IDENTICAL_EPOCH_DISTANCE_AUDIT_V0037.csv"
OUTPUT_HTML = OUTPUT_DIR / "VENUS_1769_TAHITI_VARDO_IDENTICAL_EPOCH_DISTANCE_AUDIT_V0037.html"
MASTER_CANDIDATES = (
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0034.csv",
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0033.csv",
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0037.csv",
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


def load_or_build_master(
    base: dict[str, object], required_columns: list[str]
) -> tuple[pd.DataFrame, str]:
    for candidate in MASTER_CANDIDATES:
        if compatible_master(candidate, required_columns):
            return pd.read_csv(candidate), str(candidate)
    master = base["build_master"]()
    target = MASTER_CANDIDATES[-1]
    master.to_csv(target, index=False, float_format="%.15f")
    return master, "NEW JPL HORIZONS DOWNLOAD"


def distance_set(
    cache: dict[str, object], vector_at, norm, jd_tdb: float
) -> dict[str, float]:
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
    table = display_frame.to_html(
        index=False,
        border=0,
        classes="audit-table",
        escape=False,
    )
    return f'<div class="table-wrap">{table}</div>'


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

    jd_track = float(Time(PROJECT_PHI0_UTC, format="iso", scale="utc").tdb.jd)
    jd_instant = float(Time(PROJECT_PHI0_UTC, format="iso", scale="utc").tdb.jd)
    utc_track = str(utc_text(jd_track))
    utc_instant = str(utc_text(jd_instant))
    epoch_delta_seconds = (jd_track - jd_instant) * 86400.0

    track_values = distance_set(cache, vector_at, norm, jd_track)
    instant_values = distance_set(cache, vector_at, norm, jd_instant)

    geocentric_sun = vector_at(cache, "GEOCENTER_SUN", jd_track)
    center, xi, eta = common_basis(geocentric_sun)
    center = unit(geocentric_sun)

    contacts_t = external_contacts(cache, "TAHITI")
    contacts_v = external_contacts(cache, "VARDO")
    common_start = max(float(contacts_t[0]), float(contacts_v[0]))
    common_stop = min(float(contacts_t[1]), float(contacts_v[1]))
    minute_jds = np.asarray(cache["JD_TDB"], dtype=float)
    selected_jds = minute_jds[
        (minute_jds >= common_start) & (minute_jds <= common_stop)
    ]
    if len(selected_jds) < 20:
        raise RuntimeError("Too few synchronized in-transit JPL samples.")

    tahiti_track = np.array(
        [
            relative_position_arcsec(cache, "TAHITI", jd, center, xi, eta)
            for jd in selected_jds
        ],
        dtype=float,
    )
    vardo_track = np.array(
        [
            relative_position_arcsec(cache, "VARDO", jd, center, xi, eta)
            for jd in selected_jds
        ],
        dtype=float,
    )
    direction_t = fitted_direction(tahiti_track)
    direction_v = fitted_direction(vardo_track)
    if float(np.dot(direction_t, direction_v)) < 0.0:
        direction_v = -direction_v
    common_direction = unit(direction_t + direction_v)
    common_normal = np.array(
        [-common_direction[1], common_direction[0]], dtype=float
    )
    common_track_angle_deg = math.degrees(
        math.atan2(common_direction[1], common_direction[0])
    )

    apparent_t = relative_position_arcsec(
        cache, "TAHITI", jd_track, center, xi, eta
    )
    apparent_v = relative_position_arcsec(
        cache, "VARDO", jd_track, center, xi, eta
    )
    signed_apparent = float(np.dot(apparent_v - apparent_t, common_normal))
    if signed_apparent < 0.0:
        common_normal = -common_normal
        signed_apparent = -signed_apparent
    aprime_bprime_arcsec = signed_apparent

    topocentric_sun_t = vector_at(cache, "TAHITI_SUN", jd_track)
    topocentric_sun_v = vector_at(cache, "VARDO_SUN", jd_track)
    observer_t = geocentric_sun - topocentric_sun_t
    observer_v = geocentric_sun - topocentric_sun_v
    baseline = observer_v - observer_t
    baseline_plane = np.array(
        [float(np.dot(baseline, xi)), float(np.dot(baseline, eta))],
        dtype=float,
    )
    ab_direct_km = abs(float(np.dot(baseline_plane, common_normal)))

    km_per_arcsec = track_values["ES"] / arcsec_per_rad
    aprime_bprime_km = aprime_bprime_arcsec * km_per_arcsec
    ab_direct_arcsec = ab_direct_km / km_per_arcsec
    halley_arcsec = aprime_bprime_arcsec * track_values["EV/VS"]
    halley_km = aprime_bprime_km * track_values["EV/VS"]
    residual_arcsec = ab_direct_arcsec - halley_arcsec
    residual_km = ab_direct_km - halley_km

    epoch_frame = pd.DataFrame(
        [
            [
                "Track-reduction JPL evaluation",
                utc_track + " UTC",
                jd_track,
            ],
            [
                "Instantaneous JPL evaluation",
                utc_instant + " UTC",
                jd_instant,
            ],
            [
                "Difference",
                f"{epoch_delta_seconds:.9f} seconds",
                jd_track - jd_instant,
            ],
        ],
        columns=["Evaluation", "UTC", "JD TDB"],
    )

    distance_rows: list[list[object]] = []
    for label, symbol in (
        ("Earth → Venus", "EV"),
        ("Venus → Sun", "VS"),
        ("Earth → Sun", "ES"),
    ):
        difference = track_values[symbol] - instant_values[symbol]
        distance_rows.append(
            [
                label,
                symbol,
                track_values[symbol],
                instant_values[symbol],
                difference,
            ]
        )
    distance_frame = pd.DataFrame(
        distance_rows,
        columns=[
            "Distance",
            "Symbol",
            "Track reduction km",
            "Instantaneous JPL km",
            "Difference km",
        ],
    )

    ratio_rows: list[list[object]] = []
    for label, symbol in (
        ("Earth–Venus / Venus–Sun", "EV/VS"),
        ("Venus–Sun / Earth–Venus", "VS/EV"),
        ("Earth–Sun / Venus–Sun", "ES/VS"),
        ("Venus–Sun / Earth–Sun", "VS/ES"),
        ("Earth–Venus / Earth–Sun", "EV/ES"),
        ("Earth–Sun / Earth–Venus", "ES/EV"),
    ):
        difference = track_values[symbol] - instant_values[symbol]
        ratio_rows.append(
            [
                label,
                symbol,
                track_values[symbol],
                instant_values[symbol],
                difference,
            ]
        )
    ratio_frame = pd.DataFrame(
        ratio_rows,
        columns=[
            "Ratio",
            "Definition",
            "Track reduction",
            "Instantaneous JPL",
            "Difference",
        ],
    )

    geometry_frame = pd.DataFrame(
        [
            ["A′B′ direct JPL", aprime_bprime_arcsec, aprime_bprime_km],
            ["AB direct JPL", ab_direct_arcsec, ab_direct_km],
            ["AB classical Halley", halley_arcsec, halley_km],
            ["Direct JPL − Halley", residual_arcsec, residual_km],
        ],
        columns=["Quantity", "Arcseconds", "Kilometers"],
    )

    calculation_frame = pd.DataFrame(
        [
            [
                "Classical Halley factor",
                "EV / VS",
                f"{track_values['EV']:,.12f} / {track_values['VS']:,.12f}",
                track_values["EV/VS"],
            ],
            [
                "Classical angular reduction",
                "A′B′ × EV/VS",
                f"{aprime_bprime_arcsec:.12f} × {track_values['EV/VS']:.12f}",
                halley_arcsec,
            ],
            [
                "Classical linear reduction",
                "A′B′ km × EV/VS",
                f"{aprime_bprime_km:.12f} × {track_values['EV/VS']:.12f}",
                halley_km,
            ],
        ],
        columns=["Calculation", "Definition", "Arithmetic", "Result"],
    )

    audit_frames = [
        ("EPOCHS", epoch_frame),
        ("DISTANCES", distance_frame),
        ("RATIOS", ratio_frame),
        ("GEOMETRY", geometry_frame),
        ("HALLEY", calculation_frame),
    ]
    audit_rows: list[dict[str, object]] = []
    for section, frame in audit_frames:
        for row_number, row in frame.iterrows():
            record: dict[str, object] = {
                "section": section,
                "row": int(row_number),
            }
            record.update({str(key): value for key, value in row.items()})
            audit_rows.append(record)
    pd.DataFrame(audit_rows).to_csv(
        OUTPUT_CSV,
        index=False,
        float_format="%.15f",
    )

    css = """
<style>
.v0037-report{background:#ffffff;color:#000000;font-family:Arial,Helvetica,sans-serif;padding:16px;border:1px solid #000000;width:100%;box-sizing:border-box}
.v0037-report *{box-sizing:border-box;background:#ffffff;color:#000000}
.v0037-report h1{font-size:22px;margin:0 0 16px 0;padding:0 0 8px 0;border-bottom:2px solid #000000}
.v0037-report h2{font-size:16px;margin:22px 0 8px 0;padding:5px 0;border-top:1px solid #000000;border-bottom:1px solid #000000}
.v0037-report h3{font-size:14px;margin:16px 0 6px 0;padding:0}
.v0037-report p{font-size:13px;line-height:1.45;margin:5px 0 9px 0}
.table-wrap{width:100%;overflow-x:auto;margin:6px 0 14px 0;border:0;background:#ffffff}
.audit-table{border-collapse:collapse;width:max-content;min-width:100%;font-size:12px;background:#ffffff;color:#000000}
.audit-table th{background:#ffffff;color:#000000;border:1px solid #000000;padding:7px 9px;text-align:center;font-weight:700;white-space:nowrap}
.audit-table td{background:#ffffff;color:#000000;border:1px solid #000000;padding:7px 9px;text-align:right;vertical-align:top;white-space:nowrap}
.audit-table td:first-child,.audit-table td:nth-child(2){text-align:left}
.note{border:1px solid #000000;padding:9px;background:#ffffff;color:#000000;font-weight:600}
.path{font-family:monospace;font-size:11px;overflow-wrap:anywhere;white-space:normal}
</style>
"""

    html: list[str] = [css, '<div class="v0037-report">']
    html.append("<h1>1769 Venus Transit — Tahiti–Vardø Identical-Epoch JPL Audit</h1>")

    html.append("<h2>CODE INPUTS</h2>")
    html.append(f"<p><b>JPL source:</b> {source}</p>")
    html.append(f"<p><b>Project φ0 epoch:</b> {utc_track} UTC</p>")
    html.append(f"<p><b>Common track angle:</b> {common_track_angle_deg:.12f}°</p>")

    html.append("<h2>COMMENTS</h2>")
    html.append(
        '<p class="note">Both columns are intentionally evaluated at the identical project φ0 closest-approach epoch. Therefore, any nonzero distance or ratio difference would indicate a program inconsistency.</p>'
    )
    html.append(
        '<p class="note">The classical Halley factor is EV/VS. AB/A′B′ is not used as a distance factor.</p>'
    )

    html.append("<h2>RESULTS</h2>")
    html.append("<h3>Reference epoch — identical for both evaluations</h3>")
    html.append(html_table(epoch_frame, {"JD TDB": "{:.15f}"}))

    html.append("<h3>JPL vector distances</h3>")
    html.append(
        html_table(
            distance_frame,
            {
                "Track reduction km": "{:,.12f}",
                "Instantaneous JPL km": "{:,.12f}",
                "Difference km": "{:+,.12f}",
            },
        )
    )

    html.append("<h3>All distance-ratio permutations</h3>")
    html.append(
        html_table(
            ratio_frame,
            {
                "Track reduction": "{:.12f}",
                "Instantaneous JPL": "{:.12f}",
                "Difference": "{:+.12f}",
            },
        )
    )

    html.append("<h3>JPL geometry and classical Halley result</h3>")
    html.append(
        html_table(
            geometry_frame,
            {
                "Arcseconds": "{:+.12f}",
                "Kilometers": "{:+,.12f}",
            },
        )
    )

    html.append("<h3>Classical Halley arithmetic</h3>")
    html.append(
        html_table(
            calculation_frame,
            {"Result": "{:.12f}"},
        )
    )

    html.append("<h2>OUTPUT SUMMARY</h2>")
    html.append(f'<p class="path">{OUTPUT_CSV}</p>')
    html.append(f'<p class="path">{OUTPUT_HTML}</p>')

    html.append("<h2>PAPER COMPARISON</h2>")
    html.append(
        '<p class="note">Track-reduction and instantaneous JPL distances are identical because both are evaluated at the same φ0 epoch. The remaining direct-JPL minus classical-Halley difference is therefore not an epoch-input difference.</p>'
    )

    html.append("<h2>EQUATION STATUS</h2>")
    html.append(
        f'<p class="note">Epoch consistency: PASS — Δt = {epoch_delta_seconds:.12f} s. Distance and ratio differences must print as zero. Current AB residual: {residual_arcsec:+.12f} arcsec = {residual_km:+,.12f} km.</p>'
    )
    html.append("</div>")

    report_html = "".join(html)
    OUTPUT_HTML.write_text(
        "<html><head><meta charset='utf-8'></head><body style='margin:0;background:#ffffff;color:#000000'>"
        + report_html
        + "</body></html>",
        encoding="utf-8",
    )
    display(HTML(report_html))

    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0037
