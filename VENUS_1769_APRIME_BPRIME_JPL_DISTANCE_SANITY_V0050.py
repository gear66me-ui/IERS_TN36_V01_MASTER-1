# V0050
# Audit reference: Reconstruct instantaneous JPL distances from the A′/B′ pointing rays and perform a Halley-ratio sanity check.
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
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import HTML, display
from scipy.optimize import minimize_scalar

VERSION = "V0050"
TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUT = ROOT / "VENUS_1769_APRIME_BPRIME_JPL_DISTANCE_SANITY_V0050_OUTPUT"
CSV = OUT / "VENUS_1769_APRIME_BPRIME_JPL_DISTANCE_SANITY_V0050.csv"
HTML_FILE = OUT / "VENUS_1769_APRIME_BPRIME_JPL_DISTANCE_SANITY_V0050.html"

MASTER_FILES = (
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0034.csv",
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0033.csv",
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0050.csv",
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
TARGET_IDS = {"SUN": "10", "VENUS": "299"}
ARCSEC_PER_RAD = 206_264.80624709636
JPL_AU_KM = 149_597_870.700000
EARTH_EQUATORIAL_RADIUS_KM = 6_378.140000
HISTORICAL_AU_KM = 149_597_870.000000
VELOCITY_HALF_STEP_SECONDS = 0.5
MP_DPS = 70


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
    return frame, "NEW JPL HORIZONS MASTER DOWNLOAD"


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


def horizons_vectors(
    target_id: str,
    location: str | dict[str, float | int],
    epochs: list[float],
) -> np.ndarray:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            query = Horizons(
                id=target_id,
                location=location,
                epochs=[float(value) for value in epochs],
                id_type="majorbody",
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
            if result.shape != (len(epochs), 3):
                raise RuntimeError(
                    f"Unexpected Horizons result shape {result.shape}; "
                    f"expected {(len(epochs), 3)}."
                )
            if not np.all(np.isfinite(result)):
                raise RuntimeError("Horizons returned non-finite vectors.")
            return result * JPL_AU_KM
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"JPL Horizons vector query failed: {last_error}")


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
    miss = [point_1[i] - point_2[i] for i in range(3)]

    return {
        "midpoint": np.array([float(value) for value in midpoint], dtype=float),
        "range_1_km": float(range_1),
        "range_2_km": float(range_2),
        "miss_km": float(mp_norm(miss)),
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
            converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if not pd.isna(converted):
                return pattern.format(float(converted))
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
    master, master_source = load_master(base, required_columns)
    cache = base["build_cache"](master)
    vector_at = base["vector_at"]
    angular_separation = base["angular_separation"]
    utc_text = base["utc_text"]

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
        raise RuntimeError("Closest-approach optimization failed.")

    jd_ca = jd_at(float(ca.x))
    epoch = Time(jd_ca, format="jd", scale="tdb")
    half_step_days = VELOCITY_HALF_STEP_SECONDS / 86400.0
    exact_epochs = [jd_ca - half_step_days, jd_ca, jd_ca + half_step_days]

    geocentric = {
        target: horizons_vectors(
            TARGET_IDS[target],
            "@399",
            [jd_ca],
        )[0]
        for target in ("SUN", "VENUS")
    }
    direct_sun = geocentric["SUN"]
    direct_venus = geocentric["VENUS"]
    direct_es = norm(direct_sun)
    direct_ev = norm(direct_venus)
    direct_vs = norm(direct_sun - direct_venus)

    center = unit(direct_sun)
    pole = np.array([0.0, 0.0, 1.0], dtype=float)
    xi = np.cross(pole, center)
    if norm(xi) < 1.0e-14:
        xi = np.cross(np.array([0.0, 1.0, 0.0]), center)
    xi = unit(xi)
    eta = unit(np.cross(center, xi))

    topocentric: dict[str, dict[str, np.ndarray]] = {}
    for site in SITES:
        key = str(site["key"])
        location = {
            "lon": float(site["lon"]),
            "lat": float(site["lat"]),
            "elevation": 0.0,
            "body": 399,
        }
        topocentric[key] = {
            target: horizons_vectors(
                TARGET_IDS[target],
                location,
                exact_epochs,
            )
            for target in ("SUN", "VENUS")
        }

    def relative_position(site_key: str, epoch_index: int) -> np.ndarray:
        sun_direction = topocentric[site_key]["SUN"][epoch_index]
        venus_direction = topocentric[site_key]["VENUS"][epoch_index]
        return ARCSEC_PER_RAD * (
            gnomonic(venus_direction, center, xi, eta)
            - gnomonic(sun_direction, center, xi, eta)
        )

    relative = {
        key: np.array([relative_position(key, i) for i in range(3)], dtype=float)
        for key in ("TAHITI", "VARDO")
    }
    velocity = {
        key: (
            relative[key][2] - relative[key][0]
        ) / (2.0 * VELOCITY_HALF_STEP_SECONDS)
        for key in ("TAHITI", "VARDO")
    }
    direction_t = unit(velocity["TAHITI"])
    direction_v = unit(velocity["VARDO"])
    if float(np.dot(direction_t, direction_v)) < 0.0:
        direction_v = -direction_v
    along_2d = unit(direction_t + direction_v)
    normal_2d = np.array([-along_2d[1], along_2d[0]], dtype=float)

    q_t = relative["TAHITI"][1]
    q_v = relative["VARDO"][1]
    midpoint_2d = 0.5 * (q_t + q_v)
    a_prime_2d = q_t - midpoint_2d
    b_prime_2d = q_v - midpoint_2d
    if float(np.dot(b_prime_2d - a_prime_2d, normal_2d)) < 0.0:
        normal_2d = -normal_2d

    normal_3d = unit(normal_2d[0] * xi + normal_2d[1] * eta)
    a_prime_normal = float(np.dot(a_prime_2d, normal_2d))
    b_prime_normal = float(np.dot(b_prime_2d, normal_2d))
    aprime_bprime_arcsec = b_prime_normal - a_prime_normal

    tangent_a_abs = direct_es * (
        center
        + (q_t[0] / ARCSEC_PER_RAD) * xi
        + (q_t[1] / ARCSEC_PER_RAD) * eta
    )
    tangent_b_abs = direct_es * (
        center
        + (q_v[0] / ARCSEC_PER_RAD) * xi
        + (q_v[1] / ARCSEC_PER_RAD) * eta
    )

    observer_jpl: dict[str, np.ndarray] = {}
    observer_origin_rows: list[list[object]] = []
    for key in ("TAHITI", "VARDO"):
        observer_from_sun = direct_sun - topocentric[key]["SUN"][1]
        observer_from_venus = direct_venus - topocentric[key]["VENUS"][1]
        observer_jpl[key] = 0.5 * (observer_from_sun + observer_from_venus)
        observer_origin_rows.append(
            [
                key,
                norm(observer_from_sun - observer_from_venus),
                *observer_jpl[key].tolist(),
            ]
        )

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

    rays = {
        key: {
            "SUN": topocentric[key]["SUN"][1],
            "VENUS": topocentric[key]["VENUS"][1],
        }
        for key in ("TAHITI", "VARDO")
    }

    def reconstruct(
        observers: dict[str, np.ndarray],
    ) -> dict[str, dict[str, object]]:
        return {
            target: triangulate_two_rays(
                observers["TAHITI"],
                rays["TAHITI"][target],
                observers["VARDO"],
                rays["VARDO"][target],
            )
            for target in ("SUN", "VENUS")
        }

    def pointing_objective(dut1_seconds: float) -> float:
        result = reconstruct(iers_observers(dut1_seconds))
        return (
            float(result["SUN"]["miss_km"]) ** 2
            + float(result["VENUS"]["miss_km"]) ** 2
        )

    pointing_fit = minimize_scalar(
        pointing_objective,
        bounds=(-300.0, 300.0),
        method="bounded",
        options={"xatol": 1.0e-9, "maxiter": 500},
    )
    if not pointing_fit.success:
        raise RuntimeError("TN36 pointing-only DUT1 fit failed.")
    fitted_dut1 = float(pointing_fit.x)

    observer_models = {
        "JPL station origins inferred from A′/B′ rays": observer_jpl,
        "IERS TN36 pointing-fit station origins": iers_observers(fitted_dut1),
    }

    distance_rows: list[list[object]] = []
    ratio_rows: list[list[object]] = []
    sanity_rows: list[list[object]] = []
    status_rows: list[list[object]] = []
    records: list[dict[str, object]] = []

    for model_label, observers in observer_models.items():
        rebuilt = reconstruct(observers)
        rebuilt_sun = np.asarray(rebuilt["SUN"]["midpoint"], dtype=float)
        rebuilt_venus = np.asarray(rebuilt["VENUS"]["midpoint"], dtype=float)
        es_pointing = norm(rebuilt_sun)
        ev_pointing = norm(rebuilt_venus)
        vs_pointing = norm(rebuilt_sun - rebuilt_venus)

        for distance_label, symbol, derived, direct in (
            ("Earth → Venus", "EV", ev_pointing, direct_ev),
            ("Venus → Sun", "VS", vs_pointing, direct_vs),
            ("Earth → Sun", "ES", es_pointing, direct_es),
        ):
            distance_rows.append(
                [
                    model_label,
                    distance_label,
                    symbol,
                    derived,
                    direct,
                    derived - direct,
                ]
            )

        physical_ratio = ev_pointing / vs_pointing
        direct_ratio = direct_ev / direct_vs
        baseline = observers["VARDO"] - observers["TAHITI"]
        direct_ab_km = abs(float(np.dot(baseline, normal_3d)))
        km_per_arcsec = es_pointing / ARCSEC_PER_RAD
        aprime_bprime_km = aprime_bprime_arcsec * km_per_arcsec
        halley_ab_arcsec = aprime_bprime_arcsec * physical_ratio
        halley_ab_km = aprime_bprime_km * physical_ratio
        direct_ab_arcsec = direct_ab_km / km_per_arcsec
        residual_km = halley_ab_km - direct_ab_km

        ratio_rows.extend(
            [
                [
                    model_label,
                    "Point-derived physical ratio",
                    "EV_point/VS_point",
                    physical_ratio,
                ],
                [
                    model_label,
                    "Instantaneous direct JPL ratio",
                    "EV_JPL/VS_JPL",
                    direct_ratio,
                ],
                [
                    model_label,
                    "Ratio difference",
                    "point-derived − instantaneous",
                    physical_ratio - direct_ratio,
                ],
            ]
        )
        sanity_rows.append(
            [
                model_label,
                aprime_bprime_arcsec,
                aprime_bprime_km,
                physical_ratio,
                halley_ab_arcsec,
                halley_ab_km,
                direct_ab_arcsec,
                direct_ab_km,
                residual_km,
            ]
        )

        distance_max_error = max(
            abs(ev_pointing - direct_ev),
            abs(vs_pointing - direct_vs),
            abs(es_pointing - direct_es),
        )
        distance_tolerance = (
            0.01
            if model_label.startswith("JPL station")
            else 100.0
        )
        status_rows.extend(
            [
                [
                    model_label,
                    "A′/B′ pointing-distance reconstruction",
                    "PASS" if distance_max_error < distance_tolerance else "FAIL",
                    distance_max_error,
                    "km",
                ],
                [
                    model_label,
                    "Point-derived EV/VS versus instantaneous JPL EV/VS",
                    "PASS"
                    if abs(physical_ratio - direct_ratio) < 1.0e-6
                    else "FAIL",
                    physical_ratio - direct_ratio,
                    "dimensionless",
                ],
                [
                    model_label,
                    "Halley-ratio AB versus direct TN36/JPL AB",
                    "PASS" if abs(residual_km) < 0.01 else "FAIL",
                    residual_km,
                    "km",
                ],
            ]
        )

        records.append(
            {
                "model": model_label,
                "jd_tdb": jd_ca,
                "utc": utc_text(jd_ca),
                "fitted_dut1_seconds": fitted_dut1,
                "a_prime_x_arcsec": a_prime_2d[0],
                "a_prime_y_arcsec": a_prime_2d[1],
                "a_prime_normal_arcsec": a_prime_normal,
                "b_prime_x_arcsec": b_prime_2d[0],
                "b_prime_y_arcsec": b_prime_2d[1],
                "b_prime_normal_arcsec": b_prime_normal,
                "aprime_bprime_arcsec": aprime_bprime_arcsec,
                "ev_pointing_km": ev_pointing,
                "vs_pointing_km": vs_pointing,
                "es_pointing_km": es_pointing,
                "ev_direct_jpl_km": direct_ev,
                "vs_direct_jpl_km": direct_vs,
                "es_direct_jpl_km": direct_es,
                "physical_ratio": physical_ratio,
                "direct_jpl_ratio": direct_ratio,
                "halley_ab_km": halley_ab_km,
                "direct_ab_km": direct_ab_km,
                "halley_minus_direct_ab_km": residual_km,
                "sun_ray_miss_km": rebuilt["SUN"]["miss_km"],
                "venus_ray_miss_km": rebuilt["VENUS"]["miss_km"],
            }
        )

    epoch_frame = pd.DataFrame(
        [
            ["Closest-approach UTC", utc_text(jd_ca) + " UTC"],
            ["Closest-approach JD TDB", jd_ca],
            ["Velocity half-step seconds", VELOCITY_HALF_STEP_SECONDS],
            [
                "Instantaneous common velocity angle deg",
                math.degrees(math.atan2(along_2d[1], along_2d[0])),
            ],
            [
                "Instantaneous common-normal angle deg",
                math.degrees(math.atan2(normal_2d[1], normal_2d[0])),
            ],
            ["TN36 pointing-fit DUT1 seconds", fitted_dut1],
        ],
        columns=["Quantity", "Value"],
    )

    point_frame = pd.DataFrame(
        [
            [
                "A′ — Tahiti",
                a_prime_2d[0],
                a_prime_2d[1],
                a_prime_normal,
                *tangent_a_abs.tolist(),
            ],
            [
                "B′ — Vardø",
                b_prime_2d[0],
                b_prime_2d[1],
                b_prime_normal,
                *tangent_b_abs.tolist(),
            ],
            [
                "B′ − A′",
                b_prime_2d[0] - a_prime_2d[0],
                b_prime_2d[1] - a_prime_2d[1],
                aprime_bprime_arcsec,
                *(tangent_b_abs - tangent_a_abs).tolist(),
            ],
        ],
        columns=[
            "Point",
            "Midpoint-centered X arcsec",
            "Midpoint-centered Y arcsec",
            "Common-normal coordinate arcsec",
            "Solar-tangent X ecliptic km",
            "Solar-tangent Y ecliptic km",
            "Solar-tangent Z ecliptic km",
        ],
    )

    instantaneous_frame = pd.DataFrame(
        [
            ["Earth → Venus", "EV", direct_ev],
            ["Venus → Sun", "VS", direct_vs],
            ["Earth → Sun", "ES", direct_es],
            ["Instantaneous ratio", "EV/VS", direct_ev / direct_vs],
        ],
        columns=["JPL instantaneous quantity", "Symbol", "Value"],
    )

    observer_frame = pd.DataFrame(
        observer_origin_rows,
        columns=[
            "Station",
            "Sun-derived versus Venus-derived origin mismatch km",
            "Observer X ecliptic km",
            "Observer Y ecliptic km",
            "Observer Z ecliptic km",
        ],
    )

    distance_frame = pd.DataFrame(
        distance_rows,
        columns=[
            "Station-origin model",
            "Distance",
            "Symbol",
            "A′/B′ pointing-derived km",
            "Instantaneous direct JPL km",
            "Derived − JPL km",
        ],
    )
    ratio_frame = pd.DataFrame(
        ratio_rows,
        columns=["Station-origin model", "Ratio", "Equation", "Value"],
    )
    sanity_frame = pd.DataFrame(
        sanity_rows,
        columns=[
            "Station-origin model",
            "A′B′ arcsec",
            "A′B′ km",
            "EV/VS",
            "Halley AB arcsec",
            "Halley AB km",
            "Direct AB arcsec",
            "Direct AB km",
            "Halley − direct km",
        ],
    )
    status_frame = pd.DataFrame(
        status_rows,
        columns=[
            "Station-origin model",
            "Equation / test",
            "Status",
            "Residual",
            "Unit",
        ],
    )

    output_frames = (
        ("EPOCH", epoch_frame),
        ("APRIME_BPRIME_POINTS", point_frame),
        ("INSTANTANEOUS_JPL", instantaneous_frame),
        ("JPL_OBSERVER_ORIGINS", observer_frame),
        ("POINTING_DISTANCES", distance_frame),
        ("RATIOS", ratio_frame),
        ("AB_SANITY", sanity_frame),
        ("STATUS", status_frame),
    )
    csv_records: list[dict[str, object]] = []
    for section, frame in output_frames:
        for row_number, row in frame.iterrows():
            record: dict[str, object] = {
                "section": section,
                "row": int(row_number),
            }
            record.update({str(key): value for key, value in row.items()})
            csv_records.append(record)
    pd.DataFrame(csv_records).to_csv(CSV, index=False, float_format="%.15f")

    css = """
<style>
.r{background:#000;color:#fff;font-family:Arial,Helvetica,sans-serif;padding:16px;border:1px solid #fff;width:100%;box-sizing:border-box}
.r *{background:#000;color:#fff;box-sizing:border-box}
.r h1{font-size:22px;border-bottom:2px solid #fff;padding-bottom:8px}
.r h2{font-size:16px;border-top:1px solid #fff;border-bottom:1px solid #fff;padding:5px 0;margin-top:22px}
.r h3{font-size:14px}.r p{font-size:13px;line-height:1.45}
.wrap{overflow-x:auto}.audit{border-collapse:collapse;min-width:100%;width:max-content;font-size:12px}
.audit th,.audit td{border:1px solid #fff;padding:7px 9px;white-space:nowrap}.audit th{font-weight:700;text-align:center}.audit td{text-align:right}
.audit td:first-child,.audit td:nth-child(2){text-align:left}.note{border:1px solid #fff;padding:9px;font-weight:600}
.answer{font-size:18px;border:2px solid #fff;padding:12px;font-weight:700;text-align:center}
.path{font-family:monospace;font-size:11px;overflow-wrap:anywhere;white-space:normal}
</style>
"""

    html: list[str] = [css, '<div class="r">']
    html.append(
        "<h1>1769 Venus Transit — A′/B′ Instantaneous JPL Distance Sanity Audit</h1>"
    )
    html.append("<h2>CODE INPUTS</h2>")
    html.append(f"<p><b>JPL master source:</b> {master_source}</p>")
    html.append(
        f"<p><b>Exact JPL query epoch:</b> {utc_text(jd_ca)} UTC "
        f"(JD TDB {jd_ca:.15f})</p>"
    )
    html.append(
        "<p><b>JPL exact queries:</b> geometric Sun and Venus vectors from "
        "Earth center, Tahiti, and Vardø at the closest-approach millisecond.</p>"
    )

    html.append("<h2>COMMENTS</h2>")
    html.append(
        '<p class="note">A′ and B′ are not independent Horizons target bodies. '
        "They are the two solar-tangent-plane points generated by the Tahiti and "
        "Vardø Venus-minus-Sun sight lines at the same closest-approach epoch. "
        "The audit therefore queries the exact JPL rays that define A′ and B′, "
        "then derives EV, VS, and ES by two-ray triangulation.</p>"
    )
    html.append(
        '<p class="note">No C_total, AB/A′B′ closure factor, effective distance, '
        "or manually entered correction is used in the distance reconstruction "
        "or in the Halley-ratio sanity check.</p>"
    )

    html.append("<h2>RESULTS</h2>")
    html.append("<h3>Closest-approach epoch and instantaneous velocity normal</h3>")
    html.append(table(epoch_frame, {"Value": "{:,.15f}"}))
    html.append("<h3>A′ and B′ solar-tangent-plane points</h3>")
    html.append(
        table(
            point_frame,
            {
                "Midpoint-centered X arcsec": "{:+.12f}",
                "Midpoint-centered Y arcsec": "{:+.12f}",
                "Common-normal coordinate arcsec": "{:+.12f}",
                "Solar-tangent X ecliptic km": "{:+,.6f}",
                "Solar-tangent Y ecliptic km": "{:+,.6f}",
                "Solar-tangent Z ecliptic km": "{:+,.6f}",
            },
        )
    )
    html.append("<h3>Instantaneous direct JPL distances</h3>")
    html.append(table(instantaneous_frame, {"Value": "{:,.12f}"}))
    html.append("<h3>JPL station-origin consistency</h3>")
    html.append(
        table(
            observer_frame,
            {
                "Sun-derived versus Venus-derived origin mismatch km": "{:.12e}",
                "Observer X ecliptic km": "{:+,.12f}",
                "Observer Y ecliptic km": "{:+,.12f}",
                "Observer Z ecliptic km": "{:+,.12f}",
            },
        )
    )
    html.append("<h3>Distances derived from the A′/B′ pointing rays</h3>")
    html.append(
        table(
            distance_frame,
            {
                "A′/B′ pointing-derived km": "{:,.12f}",
                "Instantaneous direct JPL km": "{:,.12f}",
                "Derived − JPL km": "{:+,.12f}",
            },
        )
    )
    html.append("<h3>Instantaneous and point-derived ratios</h3>")
    html.append(table(ratio_frame, {"Value": "{:+.15f}"}))
    html.append("<h3>A′B′ × EV/VS → AB sanity check</h3>")
    html.append(
        table(
            sanity_frame,
            {
                "A′B′ arcsec": "{:.12f}",
                "A′B′ km": "{:,.12f}",
                "EV/VS": "{:.15f}",
                "Halley AB arcsec": "{:.12f}",
                "Halley AB km": "{:,.12f}",
                "Direct AB arcsec": "{:.12f}",
                "Direct AB km": "{:,.12f}",
                "Halley − direct km": "{:+,.12f}",
            },
        )
    )
    html.append('<p class="answer">This is it.</p>')

    html.append("<h2>OUTPUT SUMMARY</h2>")
    html.append(f'<p class="path">{CSV}</p>')
    html.append(f'<p class="path">{HTML_FILE}</p>')

    html.append("<h2>PAPER COMPARISON</h2>")
    html.append(
        '<p class="note">The JPL-origin reconstruction is the direct check that '
        "the two A′/B′ rays encode the same instantaneous EV, VS, and ES queried "
        "from Horizons. The TN36-origin reconstruction shows the effect of the "
        "historical station-orientation realization. The final row tests whether "
        "the independently reconstructed physical ratio EV/VS alone predicts the "
        "direct common-normal baseline AB.</p>"
    )

    html.append("<h2>EQUATION STATUS</h2>")
    html.append(table(status_frame, {"Residual": "{:+.15e}"}))
    html.append("</div>")

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
# V0050
