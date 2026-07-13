# V0038
# Audit reference: Derive Tahiti–Vardø A′B′ at the closest approach of the mean of the two JPL apparent tracks.
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

VERSION = "V0038"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUTPUT_DIR = ROOT / "VENUS_1769_TAHITI_VARDO_MEAN_TRACK_CA_AUDIT_V0038_OUTPUT"
OUTPUT_CSV = OUTPUT_DIR / "VENUS_1769_TAHITI_VARDO_MEAN_TRACK_CA_AUDIT_V0038.csv"
MASTER_CANDIDATES = (
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0034.csv",
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0033.csv",
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0038.csv",
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


def table_text(frame: pd.DataFrame, decimals: int = 12) -> str:
    formatters: dict[str, object] = {}
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            formatters[column] = lambda value, d=decimals: f"{float(value):,.{d}f}"
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

    def site_angular_radius_arcsec(site: str, jd_tdb: float) -> float:
        sun = vector_at(cache, f"{site}_SUN", jd_tdb)
        venus = vector_at(cache, f"{site}_VENUS", jd_tdb)
        cosine = float(np.clip(np.dot(unit(sun), unit(venus)), -1.0, 1.0))
        return math.acos(cosine) * arcsec_per_rad

    def mean_track_radius_arcsec(jd_tdb: float) -> float:
        geocentric_sun = vector_at(cache, "GEOCENTER_SUN", jd_tdb)
        center, xi, eta = common_basis(geocentric_sun)
        center = unit(geocentric_sun)
        tahiti = relative_position_arcsec(cache, "TAHITI", jd_tdb, center, xi, eta)
        vardo = relative_position_arcsec(cache, "VARDO", jd_tdb, center, xi, eta)
        mean_position = 0.5 * (tahiti + vardo)
        return float(np.linalg.norm(mean_position))

    tahiti_ca = minimize_scalar(
        lambda jd: site_angular_radius_arcsec("TAHITI", float(jd)),
        bounds=(common_start, common_stop),
        method="bounded",
        options={"xatol": 1.0e-13, "maxiter": 500},
    )
    vardo_ca = minimize_scalar(
        lambda jd: site_angular_radius_arcsec("VARDO", float(jd)),
        bounds=(common_start, common_stop),
        method="bounded",
        options={"xatol": 1.0e-13, "maxiter": 500},
    )
    mean_track_ca = minimize_scalar(
        lambda jd: mean_track_radius_arcsec(float(jd)),
        bounds=(common_start, common_stop),
        method="bounded",
        options={"xatol": 1.0e-13, "maxiter": 500},
    )
    if not (tahiti_ca.success and vardo_ca.success and mean_track_ca.success):
        raise RuntimeError("Closest-approach optimization did not converge.")

    jd_tahiti_ca = float(tahiti_ca.x)
    jd_vardo_ca = float(vardo_ca.x)
    jd_mean_of_individual = 0.5 * (jd_tahiti_ca + jd_vardo_ca)
    jd_mean_track_ca = float(mean_track_ca.x)
    jd_project = float(Time(PROJECT_PHI0_UTC, format="iso", scale="utc").tdb.jd)

    minute_jds = np.asarray(cache["JD_TDB"], dtype=float)
    selected_jds = minute_jds[
        (minute_jds >= common_start) & (minute_jds <= common_stop)
    ]
    if len(selected_jds) < 20:
        raise RuntimeError("Too few synchronized in-transit JPL samples.")

    def geometry_at_epoch(jd_tdb: float) -> dict[str, float]:
        geocentric_sun = vector_at(cache, "GEOCENTER_SUN", jd_tdb)
        geocentric_venus = vector_at(cache, "GEOCENTER_VENUS", jd_tdb)
        venus_to_sun = geocentric_sun - geocentric_venus
        center, xi, eta = common_basis(geocentric_sun)
        center = unit(geocentric_sun)

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

        apparent_t = relative_position_arcsec(
            cache, "TAHITI", jd_tdb, center, xi, eta
        )
        apparent_v = relative_position_arcsec(
            cache, "VARDO", jd_tdb, center, xi, eta
        )
        delta = apparent_v - apparent_t
        signed_normal = float(np.dot(delta, common_normal))
        if signed_normal < 0.0:
            common_normal = -common_normal
            signed_normal = -signed_normal
        normal_separation = signed_normal
        two_dimensional_separation = float(np.linalg.norm(delta))

        topocentric_sun_t = vector_at(cache, "TAHITI_SUN", jd_tdb)
        topocentric_sun_v = vector_at(cache, "VARDO_SUN", jd_tdb)
        observer_t = geocentric_sun - topocentric_sun_t
        observer_v = geocentric_sun - topocentric_sun_v
        baseline = observer_v - observer_t
        baseline_plane = np.array(
            [float(np.dot(baseline, xi)), float(np.dot(baseline, eta))],
            dtype=float,
        )
        ab_direct_km = abs(float(np.dot(baseline_plane, common_normal)))

        es = norm(geocentric_sun)
        ev = norm(geocentric_venus)
        vs = norm(venus_to_sun)
        km_per_arcsec = es / arcsec_per_rad
        normal_km = normal_separation * km_per_arcsec
        two_dimensional_km = two_dimensional_separation * km_per_arcsec
        ab_direct_arcsec = ab_direct_km / km_per_arcsec
        halley_factor = ev / vs
        halley_arcsec = normal_separation * halley_factor
        halley_km = normal_km * halley_factor

        midpoint_t = -0.5 * normal_separation
        midpoint_v = +0.5 * normal_separation

        return {
            "TRACK_ANGLE": math.degrees(
                math.atan2(common_direction[1], common_direction[0])
            ),
            "A_PRIME": midpoint_t,
            "B_PRIME": midpoint_v,
            "APRIME_BPRIME_NORMAL": normal_separation,
            "APRIME_BPRIME_2D": two_dimensional_separation,
            "APRIME_BPRIME_NORMAL_KM": normal_km,
            "APRIME_BPRIME_2D_KM": two_dimensional_km,
            "EV": ev,
            "VS": vs,
            "ES": es,
            "EV_VS": halley_factor,
            "AB_DIRECT_ARCSEC": ab_direct_arcsec,
            "AB_DIRECT_KM": ab_direct_km,
            "AB_HALLEY_ARCSEC": halley_arcsec,
            "AB_HALLEY_KM": halley_km,
            "RESIDUAL_ARCSEC": ab_direct_arcsec - halley_arcsec,
            "RESIDUAL_KM": ab_direct_km - halley_km,
        }

    epochs = [
        ("Project φ0", jd_project),
        ("Mean of individual CA epochs", jd_mean_of_individual),
        ("Closest approach of mean track", jd_mean_track_ca),
    ]
    geometry = {label: geometry_at_epoch(jd) for label, jd in epochs}

    epoch_frame = pd.DataFrame(
        [
            ["Tahiti individual track CA", utc_text(jd_tahiti_ca) + " UTC", jd_tahiti_ca],
            ["Vardø individual track CA", utc_text(jd_vardo_ca) + " UTC", jd_vardo_ca],
            [
                "Mean of individual CA epochs",
                utc_text(jd_mean_of_individual) + " UTC",
                jd_mean_of_individual,
            ],
            [
                "Closest approach of mean track M=(T+V)/2",
                utc_text(jd_mean_track_ca) + " UTC",
                jd_mean_track_ca,
            ],
            ["Project φ0", utc_text(jd_project) + " UTC", jd_project],
        ],
        columns=["Epoch definition", "UTC", "JD TDB"],
    )
    epoch_frame["Δ from project φ0 seconds"] = [
        (jd_tahiti_ca - jd_project) * 86400.0,
        (jd_vardo_ca - jd_project) * 86400.0,
        (jd_mean_of_individual - jd_project) * 86400.0,
        (jd_mean_track_ca - jd_project) * 86400.0,
        0.0,
    ]

    aprime_rows: list[list[object]] = []
    for label, jd in epochs:
        values = geometry[label]
        aprime_rows.append(
            [
                label,
                utc_text(jd) + " UTC",
                values["A_PRIME"],
                values["B_PRIME"],
                values["APRIME_BPRIME_NORMAL"],
                values["APRIME_BPRIME_NORMAL_KM"],
                values["APRIME_BPRIME_2D"],
            ]
        )
    aprime_frame = pd.DataFrame(
        aprime_rows,
        columns=[
            "Evaluation",
            "UTC",
            "A′ normal arcsec",
            "B′ normal arcsec",
            "A′B′ normal arcsec",
            "A′B′ normal km",
            "A′B′ 2D arcsec",
        ],
    )

    distance_rows: list[list[object]] = []
    for label, jd in epochs:
        values = geometry[label]
        distance_rows.append(
            [
                label,
                values["EV"],
                values["VS"],
                values["ES"],
                values["EV_VS"],
            ]
        )
    distance_frame = pd.DataFrame(
        distance_rows,
        columns=["Evaluation", "EV km", "VS km", "ES km", "EV/VS"],
    )

    halley_rows: list[list[object]] = []
    for label, jd in epochs:
        values = geometry[label]
        halley_rows.append(
            [
                label,
                values["APRIME_BPRIME_NORMAL"],
                values["EV_VS"],
                values["AB_HALLEY_ARCSEC"],
                values["AB_DIRECT_ARCSEC"],
                values["RESIDUAL_ARCSEC"],
                values["RESIDUAL_KM"],
            ]
        )
    halley_frame = pd.DataFrame(
        halley_rows,
        columns=[
            "Evaluation",
            "A′B′ arcsec",
            "EV/VS",
            "AB Halley arcsec",
            "AB direct JPL arcsec",
            "Direct − Halley arcsec",
            "Direct − Halley km",
        ],
    )

    audit_rows: list[dict[str, object]] = []
    for section, frame in (
        ("EPOCHS", epoch_frame),
        ("APRIME_BPRIME", aprime_frame),
        ("DISTANCES", distance_frame),
        ("HALLEY", halley_frame),
    ):
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

    print("CODE INPUTS")
    print(f"JPL source : {source}")
    print(f"Common search interval : {utc_text(common_start)} UTC to {utc_text(common_stop)} UTC")
    print()

    print("COMMENTS")
    print("Mean track is defined instantaneously as M(t) = [Tahiti(t) + Vardø(t)] / 2 in one geocentric solar tangent plane.")
    print("The closest approach of the mean track is solved independently from the two individual closest-approach epochs.")
    print("A′B′ is the synchronized Tahiti–Vardø separation projected on the common fitted-track normal.")
    print()

    print("RESULTS")
    print("CLOSEST-APPROACH EPOCHS")
    print(table_text(epoch_frame, 12))
    print()
    print("A′B′ AT EACH CANDIDATE EPOCH")
    print(table_text(aprime_frame, 12))
    print()
    print("JPL DISTANCES AT EACH CANDIDATE EPOCH")
    print(table_text(distance_frame, 12))
    print()
    print("HALLEY CHECK AT EACH CANDIDATE EPOCH")
    print(table_text(halley_frame, 12))
    print()

    print("OUTPUT SUMMARY")
    print(str(OUTPUT_CSV))
    print()

    print("PAPER COMPARISON")
    mean_values = geometry["Closest approach of mean track"]
    comparison_frame = pd.DataFrame(
        [
            ["Project φ0 minus mean-track CA", (jd_project - jd_mean_track_ca) * 86400.0, "seconds"],
            ["A′B′ at mean-track CA", mean_values["APRIME_BPRIME_NORMAL"], "arcsec"],
            ["A′B′ at mean-track CA", mean_values["APRIME_BPRIME_NORMAL_KM"], "km"],
            ["AB direct JPL at mean-track CA", mean_values["AB_DIRECT_KM"], "km"],
            ["AB Halley at mean-track CA", mean_values["AB_HALLEY_KM"], "km"],
            ["Direct JPL − Halley", mean_values["RESIDUAL_KM"], "km"],
        ],
        columns=["Diagnostic", "Value", "Unit"],
    )
    print(table_text(comparison_frame, 12))
    print()

    print("EQUATION STATUS")
    print("The mean-track closest-approach calculation is independent of the fixed project φ0 input.")
    print("The residual table shows whether replacing the project epoch with the independently solved mean-track epoch changes the 19.566 km discrepancy.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0038
