# V0045
# Audit reference: Reconstruct Sun/Venus distances from TN36 station vectors and JPL pointing directions without using JPL target ranges.
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
    ("mpmath", "mpmath"),
):
    need(_module, _package)

try:
    from erfa import ErfaWarning
    warnings.filterwarnings("ignore", category=ErfaWarning)
except Exception:
    warnings.filterwarnings("ignore", message=".*dubious year.*")

import erfa
import mpmath as mp
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from scipy.optimize import minimize_scalar

VERSION = "V0045"
TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUT = ROOT / "VENUS_1769_TN36_POINTING_DISTANCE_RECONSTRUCTION_V0045_OUTPUT"
CSV = OUT / "VENUS_1769_TN36_POINTING_DISTANCE_RECONSTRUCTION_V0045.csv"
HTML_FILE = OUT / "VENUS_1769_TN36_POINTING_DISTANCE_RECONSTRUCTION_V0045.html"
MASTER_FILES = (
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0034.csv",
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0033.csv",
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0045.csv",
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
MP_DPS = 60


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
    direction: np.ndarray,
    center: np.ndarray,
    xi: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    direction_hat = unit(direction)
    denominator = float(np.dot(direction_hat, center))
    if denominator <= 0.0:
        raise RuntimeError("Direction lies outside the tangent hemisphere.")
    return np.array(
        [
            float(np.dot(direction_hat, xi)) / denominator,
            float(np.dot(direction_hat, eta)) / denominator,
        ],
        dtype=float,
    )


def mp_vector(vector: np.ndarray) -> list[mp.mpf]:
    return [mp.mpf(repr(float(value))) for value in np.asarray(vector, dtype=float)]


def mp_dot(first: list[mp.mpf], second: list[mp.mpf]) -> mp.mpf:
    return sum(a * b for a, b in zip(first, second))


def mp_norm(vector: list[mp.mpf]) -> mp.mpf:
    return mp.sqrt(mp_dot(vector, vector))


def mp_unit(vector: np.ndarray) -> list[mp.mpf]:
    values = mp_vector(vector)
    magnitude = mp_norm(values)
    if magnitude == 0:
        raise RuntimeError("Zero sight-line vector encountered.")
    return [value / magnitude for value in values]


def triangulate_two_rays(
    observer_1: np.ndarray,
    direction_1: np.ndarray,
    observer_2: np.ndarray,
    direction_2: np.ndarray,
) -> dict[str, object]:
    mp.mp.dps = MP_DPS
    p1 = mp_vector(observer_1)
    p2 = mp_vector(observer_2)
    d1 = mp_unit(direction_1)
    d2 = mp_unit(direction_2)
    w0 = [a - b for a, b in zip(p1, p2)]

    a = mp_dot(d1, d1)
    b = mp_dot(d1, d2)
    c = mp_dot(d2, d2)
    d = mp_dot(d1, w0)
    e = mp_dot(d2, w0)
    denominator = a * c - b * b
    if abs(denominator) < mp.mpf("1e-40"):
        raise RuntimeError("Sight lines are numerically parallel.")

    range_1 = (b * e - c * d) / denominator
    range_2 = (a * e - b * d) / denominator
    point_1 = [p1[i] + range_1 * d1[i] for i in range(3)]
    point_2 = [p2[i] + range_2 * d2[i] for i in range(3)]
    midpoint = [(point_1[i] + point_2[i]) / 2 for i in range(3)]
    miss_vector = [point_1[i] - point_2[i] for i in range(3)]

    return {
        "range_1_km": float(range_1),
        "range_2_km": float(range_2),
        "point_1": np.array([float(value) for value in point_1], dtype=float),
        "point_2": np.array([float(value) for value in point_2], dtype=float),
        "midpoint": np.array([float(value) for value in midpoint], dtype=float),
        "miss_km": float(mp_norm(miss_vector)),
        "denominator": float(denominator),
        "condition_indicator": float(1 / abs(denominator)),
    }


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

    def ca_objective(seconds: float) -> float:
        sample = jd_at(seconds)
        separation = angular_separation(
            vector_at(cache, "GEOCENTER_SUN", sample),
            vector_at(cache, "GEOCENTER_VENUS", sample),
        )
        return (separation * ARCSEC_PER_RAD) ** 2

    ca = minimize_scalar(
        ca_objective,
        bounds=(0.0, span_seconds),
        method="bounded",
        options={"xatol": 1.0e-7, "maxiter": 500},
    )
    if not ca.success:
        raise RuntimeError("Instantaneous closest-approach optimization failed.")

    jd = jd_at(float(ca.x))
    epoch = Time(jd, format="jd", scale="tdb")

    direct_sun = vector_at(cache, "GEOCENTER_SUN", jd)
    direct_venus = vector_at(cache, "GEOCENTER_VENUS", jd)
    direct_es = norm(direct_sun)
    direct_ev = norm(direct_venus)
    direct_vs = norm(direct_sun - direct_venus)

    center, xi, eta = common_basis(direct_sun)
    center = unit(center)
    xi = unit(xi)
    eta = unit(eta)

    topocentric: dict[str, dict[str, np.ndarray]] = {}
    for site in SITES:
        key = str(site["key"])
        topocentric[key] = {
            "SUN": vector_at(cache, f"{key}_SUN", jd),
            "VENUS": vector_at(cache, f"{key}_VENUS", jd),
        }

    tahiti_contacts = external_contacts(cache, "TAHITI")
    vardo_contacts = external_contacts(cache, "VARDO")
    common_start = max(float(tahiti_contacts[0]), float(vardo_contacts[0]))
    common_stop = min(float(tahiti_contacts[1]), float(vardo_contacts[1]))
    selected = jds[(jds >= common_start) & (jds <= common_stop)]
    if len(selected) < 20:
        raise RuntimeError("Too few synchronized in-transit JPL samples.")

    def relative_from_directions(site_key: str, sample_jd: float) -> np.ndarray:
        sun_direction = vector_at(cache, f"{site_key}_SUN", sample_jd)
        venus_direction = vector_at(cache, f"{site_key}_VENUS", sample_jd)
        return ARCSEC_PER_RAD * (
            gnomonic(venus_direction, center, xi, eta)
            - gnomonic(sun_direction, center, xi, eta)
        )

    tracks: dict[str, np.ndarray] = {}
    for key in ("TAHITI", "VARDO"):
        tracks[key] = np.array(
            [relative_from_directions(key, sample) for sample in selected],
            dtype=float,
        )

    direction_t = fitted_direction(tracks["TAHITI"])
    direction_v = fitted_direction(tracks["VARDO"])
    if float(np.dot(direction_t, direction_v)) < 0.0:
        direction_v = -direction_v
    along_2d = unit(direction_t + direction_v)
    normal_2d = np.array([-along_2d[1], along_2d[0]], dtype=float)

    relative_t = relative_from_directions("TAHITI", jd)
    relative_v = relative_from_directions("VARDO", jd)
    delta_relative = relative_v - relative_t
    if float(np.dot(delta_relative, normal_2d)) < 0.0:
        normal_2d = -normal_2d
    normal_3d = unit(normal_2d[0] * xi + normal_2d[1] * eta)

    apparent_separation_arcsec = float(np.dot(delta_relative, normal_2d))
    a_prime_arcsec = -0.5 * apparent_separation_arcsec
    b_prime_arcsec = +0.5 * apparent_separation_arcsec

    observer_jpl = {
        "TAHITI": direct_sun - topocentric["TAHITI"]["SUN"],
        "VARDO": direct_sun - topocentric["VARDO"]["SUN"],
    }
    baseline_jpl = observer_jpl["VARDO"] - observer_jpl["TAHITI"]

    tt1, tt2 = split_jd(float(epoch.tt.jd))
    ecliptic_rotation = eq_to_ecl_matrix()
    itrs = {str(site["key"]): itrs_wgs84(site) for site in SITES}

    def station_ecliptic(key: str, dut1_seconds: float) -> np.ndarray:
        ut11, ut12 = split_jd(
            float(epoch.utc.jd) + float(dut1_seconds) / 86400.0
        )
        c2t = np.asarray(erfa.c2t06a(tt1, tt2, ut11, ut12, 0.0, 0.0))
        return ecliptic_rotation @ (c2t.T @ itrs[key])

    def iers_observers(dut1_seconds: float) -> dict[str, np.ndarray]:
        return {
            "TAHITI": station_ecliptic("TAHITI", dut1_seconds),
            "VARDO": station_ecliptic("VARDO", dut1_seconds),
        }

    def reconstruct_targets(
        observers: dict[str, np.ndarray]
    ) -> dict[str, dict[str, object]]:
        return {
            target: triangulate_two_rays(
                observers["TAHITI"],
                topocentric["TAHITI"][target],
                observers["VARDO"],
                topocentric["VARDO"][target],
            )
            for target in ("SUN", "VENUS")
        }

    def pointing_objective(dut1_seconds: float) -> float:
        reconstruction = reconstruct_targets(iers_observers(dut1_seconds))
        return (
            float(reconstruction["SUN"]["miss_km"]) ** 2
            + float(reconstruction["VENUS"]["miss_km"]) ** 2
        )

    pointing_fit = minimize_scalar(
        pointing_objective,
        bounds=(-300.0, 300.0),
        method="bounded",
        options={"xatol": 1.0e-9, "maxiter": 500},
    )
    if not pointing_fit.success:
        raise RuntimeError("Pointing-only DUT1 fit failed.")
    pointing_dut1 = float(pointing_fit.x)

    def baseline_fit_objective(dut1_seconds: float) -> float:
        observers = iers_observers(dut1_seconds)
        baseline = observers["VARDO"] - observers["TAHITI"]
        difference = baseline - baseline_jpl
        return float(np.dot(difference, difference))

    baseline_fit = minimize_scalar(
        baseline_fit_objective,
        bounds=(-300.0, 300.0),
        method="bounded",
        options={"xatol": 1.0e-9, "maxiter": 500},
    )
    if not baseline_fit.success:
        raise RuntimeError("Baseline diagnostic DUT1 fit failed.")
    baseline_dut1 = float(baseline_fit.x)

    observer_models = {
        "JPL station vectors — reference": observer_jpl,
        "IERS TN36 nominal DUT1=0": iers_observers(0.0),
        "IERS TN36 pointing-fit DUT1": iers_observers(pointing_dut1),
        "IERS TN36 baseline-fit DUT1 — diagnostic": iers_observers(baseline_dut1),
    }

    reconstruction_rows: list[list[object]] = []
    ratio_rows: list[list[object]] = []
    backtrack_rows: list[list[object]] = []
    effective_rows: list[list[object]] = []
    status_rows: list[list[object]] = []
    output_records: list[dict[str, object]] = []

    for model_label, observers in observer_models.items():
        reconstructed = reconstruct_targets(observers)
        reconstructed_sun = np.asarray(reconstructed["SUN"]["midpoint"], dtype=float)
        reconstructed_venus = np.asarray(
            reconstructed["VENUS"]["midpoint"], dtype=float
        )
        es_reconstructed = norm(reconstructed_sun)
        ev_reconstructed = norm(reconstructed_venus)
        vs_reconstructed = norm(reconstructed_sun - reconstructed_venus)
        baseline = observers["VARDO"] - observers["TAHITI"]
        ab_km = abs(float(np.dot(baseline, normal_3d)))
        km_per_arcsec = es_reconstructed / ARCSEC_PER_RAD
        aprime_bprime_km = apparent_separation_arcsec * km_per_arcsec
        a_prime_km = a_prime_arcsec * km_per_arcsec
        b_prime_km = b_prime_arcsec * km_per_arcsec
        ab_arcsec = ab_km / km_per_arcsec
        exact_factor = ab_km / aprime_bprime_km
        physical_factor = ev_reconstructed / vs_reconstructed

        effective_vs = es_reconstructed / (1.0 + exact_factor)
        effective_ev = exact_factor * effective_vs
        effective_es = effective_ev + effective_vs

        classical_ab_arcsec = apparent_separation_arcsec * physical_factor
        classical_ab_km = aprime_bprime_km * physical_factor
        exact_ab_arcsec = apparent_separation_arcsec * exact_factor
        exact_ab_km = aprime_bprime_km * exact_factor

        pi_event_physical = (
            apparent_separation_arcsec
            * EARTH_EQUATORIAL_RADIUS_KM
            / ab_km
            * physical_factor
        )
        pi_event_exact = (
            apparent_separation_arcsec
            * EARTH_EQUATORIAL_RADIUS_KM
            / ab_km
            * exact_factor
        )
        pi_direct_reconstructed = (
            EARTH_EQUATORIAL_RADIUS_KM / es_reconstructed * ARCSEC_PER_RAD
        )
        pi_1au_exact = pi_event_exact * es_reconstructed / AU_KM
        pi_1au_direct = (
            EARTH_EQUATORIAL_RADIUS_KM / AU_KM * ARCSEC_PER_RAD
        )

        reconstruction_rows.extend(
            [
                [
                    model_label,
                    "Earth → Venus",
                    "EV",
                    ev_reconstructed,
                    direct_ev,
                    ev_reconstructed - direct_ev,
                    reconstructed["VENUS"]["miss_km"],
                    reconstructed["VENUS"]["condition_indicator"],
                ],
                [
                    model_label,
                    "Venus → Sun",
                    "VS",
                    vs_reconstructed,
                    direct_vs,
                    vs_reconstructed - direct_vs,
                    "",
                    "",
                ],
                [
                    model_label,
                    "Earth → Sun",
                    "ES",
                    es_reconstructed,
                    direct_es,
                    es_reconstructed - direct_es,
                    reconstructed["SUN"]["miss_km"],
                    reconstructed["SUN"]["condition_indicator"],
                ],
            ]
        )

        ratio_rows.extend(
            [
                [model_label, "Physical sightline ratio", "EV/VS", physical_factor],
                [model_label, "Exact screen factor", "AB/A′B′", exact_factor],
                [
                    model_label,
                    "Factor difference",
                    "AB/A′B′ − EV/VS",
                    exact_factor - physical_factor,
                ],
                [
                    model_label,
                    "Non-collinearity",
                    "EV + VS − ES km",
                    ev_reconstructed + vs_reconstructed - es_reconstructed,
                ],
            ]
        )

        effective_rows.extend(
            [
                [
                    model_label,
                    "Effective Earth → Venus",
                    "EV_eff",
                    effective_ev,
                    ev_reconstructed,
                    effective_ev - ev_reconstructed,
                ],
                [
                    model_label,
                    "Effective Venus → Sun",
                    "VS_eff",
                    effective_vs,
                    vs_reconstructed,
                    effective_vs - vs_reconstructed,
                ],
                [
                    model_label,
                    "Effective Earth → Sun",
                    "ES_eff",
                    effective_es,
                    es_reconstructed,
                    effective_es - es_reconstructed,
                ],
            ]
        )

        backtrack_rows.extend(
            [
                [
                    model_label,
                    "Classical physical-distance backtrack",
                    "A′B′ × EV/VS",
                    classical_ab_arcsec,
                    classical_ab_km,
                    classical_ab_km - ab_km,
                    pi_event_physical,
                    pi_event_physical * es_reconstructed / AU_KM,
                ],
                [
                    model_label,
                    "Exact effective-distance backtrack",
                    "A′B′ × AB/A′B′",
                    exact_ab_arcsec,
                    exact_ab_km,
                    exact_ab_km - ab_km,
                    pi_event_exact,
                    pi_1au_exact,
                ],
                [
                    model_label,
                    "Direct station baseline",
                    "AB",
                    ab_arcsec,
                    ab_km,
                    0.0,
                    pi_direct_reconstructed,
                    pi_1au_direct,
                ],
            ]
        )

        status_rows.extend(
            [
                [
                    model_label,
                    "Exact AB closure km",
                    "PASS" if abs(exact_ab_km - ab_km) < 1.0e-9 else "FAIL",
                    exact_ab_km - ab_km,
                ],
                [
                    model_label,
                    "Exact π event closure arcsec",
                    "PASS"
                    if abs(pi_event_exact - pi_direct_reconstructed) < 1.0e-9
                    else "FAIL",
                    pi_event_exact - pi_direct_reconstructed,
                ],
                [
                    model_label,
                    "Exact π one-AU closure arcsec",
                    "PASS" if abs(pi_1au_exact - pi_1au_direct) < 1.0e-9 else "FAIL",
                    pi_1au_exact - pi_1au_direct,
                ],
            ]
        )

        output_records.append(
            {
                "model": model_label,
                "dut1_pointing_seconds": pointing_dut1,
                "dut1_baseline_seconds": baseline_dut1,
                "jd_tdb": jd,
                "utc": utc_text(jd),
                "a_prime_arcsec": a_prime_arcsec,
                "b_prime_arcsec": b_prime_arcsec,
                "aprime_bprime_arcsec": apparent_separation_arcsec,
                "a_prime_km": a_prime_km,
                "b_prime_km": b_prime_km,
                "aprime_bprime_km": aprime_bprime_km,
                "ab_km": ab_km,
                "ab_arcsec": ab_arcsec,
                "ev_reconstructed_km": ev_reconstructed,
                "vs_reconstructed_km": vs_reconstructed,
                "es_reconstructed_km": es_reconstructed,
                "ev_jpl_km": direct_ev,
                "vs_jpl_km": direct_vs,
                "es_jpl_km": direct_es,
                "physical_factor_ev_vs": physical_factor,
                "exact_factor_ab_aprimebprime": exact_factor,
                "effective_ev_km": effective_ev,
                "effective_vs_km": effective_vs,
                "effective_es_km": effective_es,
                "classical_ab_km": classical_ab_km,
                "exact_ab_km": exact_ab_km,
                "pi_event_classical_arcsec": pi_event_physical,
                "pi_event_exact_arcsec": pi_event_exact,
                "pi_1au_exact_arcsec": pi_1au_exact,
                "sun_ray_miss_km": reconstructed["SUN"]["miss_km"],
                "venus_ray_miss_km": reconstructed["VENUS"]["miss_km"],
            }
        )

    geometry_frame = pd.DataFrame(
        [
            ["Instantaneous pointing epoch", utc_text(jd) + " UTC"],
            ["JD TDB", jd],
            ["Common fitted-track angle deg", math.degrees(math.atan2(along_2d[1], along_2d[0]))],
            ["JPL A′ arcsec", a_prime_arcsec],
            ["JPL B′ arcsec", b_prime_arcsec],
            ["JPL A′B′ arcsec", apparent_separation_arcsec],
            ["IERS pointing-fit DUT1 seconds", pointing_dut1],
            ["IERS baseline-fit DUT1 seconds", baseline_dut1],
            ["Pointing-fit minus baseline-fit DUT1 seconds", pointing_dut1 - baseline_dut1],
        ],
        columns=["Quantity", "Value"],
    )

    reconstruction_frame = pd.DataFrame(
        reconstruction_rows,
        columns=[
            "Station model",
            "Distance",
            "Symbol",
            "Pointing-reconstructed km",
            "Direct JPL km — comparison only",
            "Reconstruction − JPL km",
            "Ray miss km",
            "Condition indicator",
        ],
    )

    ratio_frame = pd.DataFrame(
        ratio_rows,
        columns=["Station model", "Ratio type", "Definition", "Value"],
    )

    effective_frame = pd.DataFrame(
        effective_rows,
        columns=[
            "Station model",
            "Effective distance",
            "Symbol",
            "Equation-derived km",
            "Pointing-reconstructed physical km",
            "Effective − physical km",
        ],
    )

    backtrack_frame = pd.DataFrame(
        backtrack_rows,
        columns=[
            "Station model",
            "Backtrack",
            "Equation",
            "AB arcsec",
            "AB km",
            "AB residual km",
            "π event arcsec",
            "π 1-AU arcsec",
        ],
    )

    status_frame = pd.DataFrame(
        status_rows,
        columns=["Station model", "Equation / test", "Status", "Residual"],
    )

    pd.DataFrame(output_records).to_csv(CSV, index=False, float_format="%.15f")

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
    html.append("<h1>1769 Venus Transit — TN36 Pointing-Derived Distance Reconstruction</h1>")

    html.append("<h2>CODE INPUTS</h2>")
    html.append(f"<p><b>JPL direction source:</b> {source}</p>")
    html.append(f"<p><b>Pointing epoch:</b> {utc_text(jd)} UTC</p>")
    html.append(
        "<p><b>Range policy:</b> JPL target-vector magnitudes are discarded in the reconstruction. "
        "Only their unit pointing directions are used.</p>"
    )

    html.append("<h2>COMMENTS</h2>")
    html.append(
        '<p class="note">A′B′ alone cannot determine three absolute distances. '
        "The full Tahiti and Vardø Sun/Venus pointing directions plus the TN36 station vectors "
        "are used to triangulate physical EV and ES; VS follows from the two reconstructed target positions.</p>"
    )
    html.append(
        '<p class="note">A second set of effective Halley-equivalent distances is solved from '
        "the exact screen factor AB/A′B′ and the reconstructed ES. These are equation-equivalent "
        "distances, not independent physical JPL ranges.</p>"
    )
    html.append(
        '<p class="note">The pointing-fit DUT1 is obtained only by minimizing the Sun and Venus '
        "two-ray intersection misses. It does not use JPL target ranges or the JPL observer baseline.</p>"
    )

    html.append("<h2>RESULTS</h2>")
    html.append("<h3>Pointing epoch and A′B′ geometry</h3>")
    html.append(table(geometry_frame, {"Value": "{:,.15f}"}))

    html.append("<h3>Physical distances reconstructed from pointing rays</h3>")
    html.append(
        table(
            reconstruction_frame,
            {
                "Pointing-reconstructed km": "{:,.12f}",
                "Direct JPL km — comparison only": "{:,.12f}",
                "Reconstruction − JPL km": "{:+,.12f}",
                "Ray miss km": "{:.12e}",
                "Condition indicator": "{:.12e}",
            },
        )
    )

    html.append("<h3>Physical and exact screen factors</h3>")
    html.append(table(ratio_frame, {"Value": "{:+.15f}"}))

    html.append("<h3>Equation-derived effective distances</h3>")
    html.append(
        table(
            effective_frame,
            {
                "Equation-derived km": "{:,.12f}",
                "Pointing-reconstructed physical km": "{:,.12f}",
                "Effective − physical km": "{:+,.12f}",
            },
        )
    )

    html.append("<h3>Halley backtrack and π closure</h3>")
    html.append(
        table(
            backtrack_frame,
            {
                "AB arcsec": "{:+.12f}",
                "AB km": "{:+,.12f}",
                "AB residual km": "{:+,.12f}",
                "π event arcsec": "{:.12f}",
                "π 1-AU arcsec": "{:.12f}",
            },
        )
    )

    html.append("<h2>OUTPUT SUMMARY</h2>")
    html.append(f'<p class="path">{CSV}</p>')
    html.append(f'<p class="path">{HTML_FILE}</p>')

    html.append("<h2>PAPER COMPARISON</h2>")
    html.append(
        '<p class="answer">This audit decides whether the TN36 pointing reconstruction returns '
        "the physical JPL distances, or whether the distances required to close Halley are "
        "effective projection distances.</p>"
    )

    html.append("<h2>EQUATION STATUS</h2>")
    html.append(table(status_frame, {"Residual": "{:+.15e}"}))
    html.append("</div>")

    report_html = "".join(html)
    HTML_FILE.write_text(
        "<html><head><meta charset='utf-8'></head>"
        "<body style='margin:0;background:#000;color:#fff'>"
        + report_html
        + "</body></html>",
        encoding="utf-8",
    )
    display(HTML(report_html))

    print(datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0045
