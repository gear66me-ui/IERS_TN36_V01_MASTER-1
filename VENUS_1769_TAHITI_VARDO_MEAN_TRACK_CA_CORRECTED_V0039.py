# V0039
# Audit reference: Correct the mean-track closest-approach solution by optimizing seconds offsets instead of raw Julian dates.
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
        subprocess.check_call(
            [sys.executable, "-m", "pip", "-q", "install", pip_name]
        )


for _import_name, _pip_name in (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("astropy", "astropy"),
    ("astroquery", "astroquery"),
):
    ensure_package(_import_name, _pip_name)

try:
    from erfa import ErfaWarning

    warnings.filterwarnings("ignore", category=ErfaWarning)
except Exception:
    warnings.filterwarnings("ignore", message=".*dubious year.*")

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

VERSION = "V0039"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUTPUT_DIR = ROOT / "VENUS_1769_TAHITI_VARDO_MEAN_TRACK_CA_CORRECTED_V0039_OUTPUT"
OUTPUT_CSV = OUTPUT_DIR / "VENUS_1769_TAHITI_VARDO_MEAN_TRACK_CA_CORRECTED_V0039.csv"
MASTER_CANDIDATES = (
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0034.csv",
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0033.csv",
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0039.csv",
)

BASE_COMMIT = "d55bc2274359a3014c19f257e42cc149bc458d57"
BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/"
    f"{BASE_COMMIT}/VENUS_1769_TAHITI_VARDO_HALLEY_THUMB_CHECK_V0031.py"
)
BASE_PATH = ROOT / "VENUS_1769_TAHITI_VARDO_HALLEY_THUMB_CHECK_V0031_BASE.py"
PROJECT_PHI0_UTC = "1769-06-03 22:19:15.599"
SECONDS_PER_DAY = 86_400.0


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


def table_text(frame: pd.DataFrame, decimals: int = 12) -> str:
    formatters: dict[str, object] = {}
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            formatters[column] = (
                lambda value, digits=decimals: f"{float(value):,.{digits}f}"
            )
    return frame.to_string(index=False, formatters=formatters)


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
    contacts_t = external_contacts(cache, "TAHITI")
    contacts_v = external_contacts(cache, "VARDO")
    common_start = max(float(contacts_t[0]), float(contacts_v[0]))
    common_stop = min(float(contacts_t[1]), float(contacts_v[1]))
    search_span_seconds = (common_stop - common_start) * SECONDS_PER_DAY

    jd_project = float(Time(PROJECT_PHI0_UTC, format="iso", scale="utc").tdb.jd)
    geocentric_sun_anchor = vector_at(cache, "GEOCENTER_SUN", jd_project)
    center, xi, eta = common_basis(geocentric_sun_anchor)
    center = unit(geocentric_sun_anchor)

    def jd_from_seconds(seconds_from_start: float) -> float:
        return common_start + float(seconds_from_start) / SECONDS_PER_DAY

    def site_position(site: str, jd_tdb: float) -> np.ndarray:
        return relative_position_arcsec(
            cache,
            site,
            jd_tdb,
            center,
            xi,
            eta,
        )

    def site_radius_squared(site: str, jd_tdb: float) -> float:
        position = site_position(site, jd_tdb)
        return float(np.dot(position, position))

    def mean_position(jd_tdb: float) -> np.ndarray:
        return 0.5 * (
            site_position("TAHITI", jd_tdb)
            + site_position("VARDO", jd_tdb)
        )

    def mean_radius_squared(jd_tdb: float) -> float:
        position = mean_position(jd_tdb)
        return float(np.dot(position, position))

    def optimize_seconds(objective) -> tuple[float, object]:
        result = minimize_scalar(
            lambda seconds: objective(jd_from_seconds(float(seconds))),
            bounds=(0.0, search_span_seconds),
            method="bounded",
            options={"xatol": 1.0e-6, "maxiter": 1000},
        )
        if not result.success:
            raise RuntimeError("Seconds-offset closest-approach optimization failed.")
        jd_solution = jd_from_seconds(float(result.x))
        return jd_solution, result

    jd_tahiti_ca, tahiti_result = optimize_seconds(
        lambda jd: site_radius_squared("TAHITI", jd)
    )
    jd_vardo_ca, vardo_result = optimize_seconds(
        lambda jd: site_radius_squared("VARDO", jd)
    )
    jd_mean_track_ca, mean_result = optimize_seconds(mean_radius_squared)
    jd_mean_of_individual = 0.5 * (jd_tahiti_ca + jd_vardo_ca)

    minute_jds = np.asarray(cache["JD_TDB"], dtype=float)
    selected_jds = minute_jds[
        (minute_jds >= common_start) & (minute_jds <= common_stop)
    ]
    if len(selected_jds) < 20:
        raise RuntimeError("Too few synchronized in-transit JPL samples.")

    tahiti_track = np.array(
        [site_position("TAHITI", jd) for jd in selected_jds],
        dtype=float,
    )
    vardo_track = np.array(
        [site_position("VARDO", jd) for jd in selected_jds],
        dtype=float,
    )
    direction_t = fitted_direction(tahiti_track)
    direction_v = fitted_direction(vardo_track)
    if float(np.dot(direction_t, direction_v)) < 0.0:
        direction_v = -direction_v
    common_direction = unit(direction_t + direction_v)
    common_normal = np.array(
        [-common_direction[1], common_direction[0]],
        dtype=float,
    )
    common_track_angle_deg = math.degrees(
        math.atan2(common_direction[1], common_direction[0])
    )

    def geometry_at_epoch(jd_tdb: float) -> dict[str, float]:
        tahiti = site_position("TAHITI", jd_tdb)
        vardo = site_position("VARDO", jd_tdb)
        mean = 0.5 * (tahiti + vardo)
        delta = vardo - tahiti

        signed_normal = float(np.dot(delta, common_normal))
        active_normal = common_normal.copy()
        if signed_normal < 0.0:
            active_normal = -active_normal
            signed_normal = -signed_normal

        a_prime = float(np.dot(tahiti - mean, active_normal))
        b_prime = float(np.dot(vardo - mean, active_normal))
        separation_2d = float(np.linalg.norm(delta))

        geocentric_sun = vector_at(cache, "GEOCENTER_SUN", jd_tdb)
        geocentric_venus = vector_at(cache, "GEOCENTER_VENUS", jd_tdb)
        venus_to_sun = geocentric_sun - geocentric_venus

        topocentric_sun_t = vector_at(cache, "TAHITI_SUN", jd_tdb)
        topocentric_sun_v = vector_at(cache, "VARDO_SUN", jd_tdb)
        observer_t = geocentric_sun - topocentric_sun_t
        observer_v = geocentric_sun - topocentric_sun_v
        baseline = observer_v - observer_t
        baseline_plane = np.array(
            [
                float(np.dot(baseline, xi)),
                float(np.dot(baseline, eta)),
            ],
            dtype=float,
        )
        ab_direct_km = abs(float(np.dot(baseline_plane, active_normal)))

        ev = norm(geocentric_venus)
        vs = norm(venus_to_sun)
        es = norm(geocentric_sun)
        km_per_arcsec = es / arcsec_per_rad
        separation_km = signed_normal * km_per_arcsec
        ab_direct_arcsec = ab_direct_km / km_per_arcsec
        halley_factor = ev / vs
        ab_halley_arcsec = signed_normal * halley_factor
        ab_halley_km = separation_km * halley_factor

        return {
            "MEAN_X": float(mean[0]),
            "MEAN_Y": float(mean[1]),
            "MEAN_RADIUS": float(np.linalg.norm(mean)),
            "TAHITI_RADIUS": float(np.linalg.norm(tahiti)),
            "VARDO_RADIUS": float(np.linalg.norm(vardo)),
            "A_PRIME": a_prime,
            "B_PRIME": b_prime,
            "APRIME_BPRIME": signed_normal,
            "APRIME_BPRIME_2D": separation_2d,
            "APRIME_BPRIME_KM": separation_km,
            "EV": ev,
            "VS": vs,
            "ES": es,
            "EV_VS": halley_factor,
            "AB_DIRECT_ARCSEC": ab_direct_arcsec,
            "AB_DIRECT_KM": ab_direct_km,
            "AB_HALLEY_ARCSEC": ab_halley_arcsec,
            "AB_HALLEY_KM": ab_halley_km,
            "RESIDUAL_ARCSEC": ab_direct_arcsec - ab_halley_arcsec,
            "RESIDUAL_KM": ab_direct_km - ab_halley_km,
        }

    epoch_items = [
        ("Tahiti individual track CA", jd_tahiti_ca),
        ("Vardø individual track CA", jd_vardo_ca),
        ("Mean of individual CA epochs", jd_mean_of_individual),
        ("Closest approach of mean track", jd_mean_track_ca),
        ("Project φ0", jd_project),
    ]
    geometry = {label: geometry_at_epoch(jd) for label, jd in epoch_items}

    raw_jd_effective_tolerance_seconds = (
        math.sqrt(np.finfo(float).eps) * jd_project * SECONDS_PER_DAY
    )
    corrected_seconds_effective_tolerance = (
        math.sqrt(np.finfo(float).eps)
        * max(
            abs((jd_mean_track_ca - common_start) * SECONDS_PER_DAY),
            1.0,
        )
    )

    optimizer_frame = pd.DataFrame(
        [
            [
                "V0038 raw-JD optimizer scale",
                jd_project,
                raw_jd_effective_tolerance_seconds,
                "REJECTED",
            ],
            [
                "V0039 seconds-offset optimizer scale",
                (jd_mean_track_ca - common_start) * SECONDS_PER_DAY,
                corrected_seconds_effective_tolerance,
                "USED",
            ],
        ],
        columns=[
            "Optimizer",
            "Independent-variable magnitude",
            "Floating-point scale seconds",
            "Status",
        ],
    )

    epoch_rows: list[list[object]] = []
    for label, jd in epoch_items:
        epoch_rows.append(
            [
                label,
                utc_text(jd) + " UTC",
                jd,
                (jd - jd_project) * SECONDS_PER_DAY,
            ]
        )
    epoch_frame = pd.DataFrame(
        epoch_rows,
        columns=[
            "Epoch definition",
            "UTC",
            "JD TDB",
            "Δ from project φ0 seconds",
        ],
    )

    mean_geometry = geometry["Closest approach of mean track"]
    mean_frame = pd.DataFrame(
        [
            ["Mean-track X", mean_geometry["MEAN_X"], "arcsec"],
            ["Mean-track Y", mean_geometry["MEAN_Y"], "arcsec"],
            ["Mean-track solar-center distance", mean_geometry["MEAN_RADIUS"], "arcsec"],
            ["Tahiti solar-center distance", mean_geometry["TAHITI_RADIUS"], "arcsec"],
            ["Vardø solar-center distance", mean_geometry["VARDO_RADIUS"], "arcsec"],
            ["A′ normal coordinate", mean_geometry["A_PRIME"], "arcsec"],
            ["B′ normal coordinate", mean_geometry["B_PRIME"], "arcsec"],
            ["A′B′ common-normal separation", mean_geometry["APRIME_BPRIME"], "arcsec"],
            ["A′B′ full 2D separation", mean_geometry["APRIME_BPRIME_2D"], "arcsec"],
            ["A′B′ common-normal separation", mean_geometry["APRIME_BPRIME_KM"], "km"],
        ],
        columns=["Quantity", "Value", "Unit"],
    )

    comparison_rows: list[list[object]] = []
    for label in (
        "Project φ0",
        "Mean of individual CA epochs",
        "Closest approach of mean track",
    ):
        values = geometry[label]
        comparison_rows.append(
            [
                label,
                values["APRIME_BPRIME"],
                values["APRIME_BPRIME_KM"],
                values["EV_VS"],
                values["AB_HALLEY_ARCSEC"],
                values["AB_DIRECT_ARCSEC"],
                values["RESIDUAL_KM"],
            ]
        )
    comparison_frame = pd.DataFrame(
        comparison_rows,
        columns=[
            "Evaluation",
            "A′B′ arcsec",
            "A′B′ km",
            "EV/VS",
            "AB Halley arcsec",
            "AB direct JPL arcsec",
            "Direct − Halley km",
        ],
    )

    distance_frame = pd.DataFrame(
        [
            ["Earth → Venus", "EV", mean_geometry["EV"]],
            ["Venus → Sun", "VS", mean_geometry["VS"]],
            ["Earth → Sun", "ES", mean_geometry["ES"]],
            ["Earth–Venus / Venus–Sun", "EV/VS", mean_geometry["EV_VS"]],
        ],
        columns=["JPL quantity at mean-track CA", "Symbol", "Value"],
    )

    audit_frames = [
        ("OPTIMIZER", optimizer_frame),
        ("EPOCHS", epoch_frame),
        ("MEAN_GEOMETRY", mean_frame),
        ("COMPARISON", comparison_frame),
        ("DISTANCES", distance_frame),
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

    local_one_second = 1.0 / SECONDS_PER_DAY
    mean_before = mean_radius_squared(jd_mean_track_ca - local_one_second)
    mean_center = mean_radius_squared(jd_mean_track_ca)
    mean_after = mean_radius_squared(jd_mean_track_ca + local_one_second)
    local_minimum_pass = mean_center <= mean_before and mean_center <= mean_after

    print("CODE INPUTS")
    print(f"JPL source : {source}")
    print(
        "Common search interval : "
        f"{utc_text(common_start)} UTC to {utc_text(common_stop)} UTC"
    )
    print(f"Fixed tangent-plane epoch : {utc_text(jd_project)} UTC")
    print(f"Common fitted-track angle : {common_track_angle_deg:.12f} deg")
    print()

    print("COMMENTS")
    print("V0038 minimized raw Julian dates and is REJECTED because the large JD magnitude produced an approximately 50.8-minute numerical scale.")
    print("V0039 minimizes seconds from the common-contact start and uses one fixed geocentric solar tangent plane.")
    print("Mean track is M(t) = [Tahiti(t) + Vardø(t)] / 2 from synchronized JPL positions.")
    print()

    print("RESULTS")
    print("OPTIMIZER AUDIT")
    print(table_text(optimizer_frame))
    print()
    print("CORRECTED CLOSEST-APPROACH EPOCHS")
    print(table_text(epoch_frame))
    print()
    print("MEAN-TRACK GEOMETRY AT ITS CLOSEST APPROACH")
    print(table_text(mean_frame))
    print()
    print("A′B′ AND HALLEY COMPARISON")
    print(table_text(comparison_frame))
    print()
    print("JPL DISTANCES AT MEAN-TRACK CLOSEST APPROACH")
    print(table_text(distance_frame))
    print()

    print("OUTPUT SUMMARY")
    print(str(OUTPUT_CSV))
    print()

    print("PAPER COMPARISON")
    paper_frame = pd.DataFrame(
        [
            [
                "Project φ0 minus corrected mean-track CA",
                (jd_project - jd_mean_track_ca) * SECONDS_PER_DAY,
                "seconds",
            ],
            [
                "A′B′ at corrected mean-track CA",
                mean_geometry["APRIME_BPRIME"],
                "arcsec",
            ],
            [
                "A′B′ at corrected mean-track CA",
                mean_geometry["APRIME_BPRIME_KM"],
                "km",
            ],
            [
                "AB direct JPL at corrected mean-track CA",
                mean_geometry["AB_DIRECT_KM"],
                "km",
            ],
            [
                "AB Halley at corrected mean-track CA",
                mean_geometry["AB_HALLEY_KM"],
                "km",
            ],
            [
                "Direct JPL − Halley",
                mean_geometry["RESIDUAL_KM"],
                "km",
            ],
        ],
        columns=["Diagnostic", "Value", "Unit"],
    )
    print(table_text(paper_frame))
    print()

    print("EQUATION STATUS")
    status_frame = pd.DataFrame(
        [
            ["Raw-JD V0038 closest approach", "REJECTED", raw_jd_effective_tolerance_seconds],
            ["Seconds-offset optimizer convergence", "PASS", float(mean_result.fun)],
            ["Mean-track local ±1 s minimum", "PASS" if local_minimum_pass else "FAIL", mean_center],
            ["A′B′ common-normal identity B′−A′", "PASS", mean_geometry["B_PRIME"] - mean_geometry["A_PRIME"] - mean_geometry["APRIME_BPRIME"]],
        ],
        columns=["Equation / test", "Status", "Residual / diagnostic"],
    )
    print(table_text(status_frame))
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0039
