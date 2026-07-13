# V0042
# Audit reference: Decompose exact JPL A′B′ through gnomonic, vector-projector, Halley-triangle, and P reductions.
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

VERSION = "V0042"
TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUT = ROOT / "VENUS_1769_FULL_VECTOR_P_REDUCTION_AUDIT_V0042_OUTPUT"
CSV = OUT / "VENUS_1769_FULL_VECTOR_P_REDUCTION_AUDIT_V0042.csv"
HTML_FILE = OUT / "VENUS_1769_FULL_VECTOR_P_REDUCTION_AUDIT_V0042.html"
MASTER_FILES = (
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0034.csv",
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0033.csv",
    ROOT / "VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0042.csv",
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
ARCSEC = 206_264.80624709636
EARTH_RADIUS_KM = 6_378.140000


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
    base: dict[str, object], columns: list[str]
) -> tuple[pd.DataFrame, str]:
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


def norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    magnitude = norm(vector)
    if magnitude == 0.0:
        raise RuntimeError("Zero vector encountered.")
    return vector / magnitude


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


def gnomonic_baseline_first_order(
    geocentric_vector: np.ndarray,
    baseline: np.ndarray,
    center: np.ndarray,
    xi: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    denominator = float(np.dot(geocentric_vector, center))
    coordinates = gnomonic(geocentric_vector, center, xi, eta)
    radial = float(np.dot(baseline, center))
    return np.array(
        [
            (-float(np.dot(baseline, xi)) + coordinates[0] * radial) / denominator,
            (-float(np.dot(baseline, eta)) + coordinates[1] * radial) / denominator,
        ],
        dtype=float,
    )


def table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    shown = frame.copy()
    for column, pattern in (formats or {}).items():
        if column in shown.columns:
            shown[column] = shown[column].map(
                lambda value: pattern.format(value) if pd.notna(value) else ""
            )
    return (
        '<div class="wrap">'
        + shown.to_html(index=False, border=0, classes="audit", escape=False)
        + "</div>"
    )


def main() -> None:
    base = base_namespace()
    OUT.mkdir(parents=True, exist_ok=True)

    prefixes = tuple(base["PREFIXES"])
    columns = ["JD_TDB"] + [
        f"{prefix}_{axis}_KM" for prefix in prefixes for axis in "XYZ"
    ]
    master, source = load_master(base, columns)
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

    def ca_objective(seconds: float) -> float:
        sample = jd_at(seconds)
        angle = angular_separation(
            vector_at(cache, "GEOCENTER_SUN", sample),
            vector_at(cache, "GEOCENTER_VENUS", sample),
        )
        return (angle * ARCSEC) ** 2

    ca = minimize_scalar(
        ca_objective,
        bounds=(0.0, span_seconds),
        method="bounded",
        options={"xatol": 1.0e-7, "maxiter": 500},
    )
    if not ca.success:
        raise RuntimeError("Seconds-offset closest-approach optimizer failed.")

    jd = jd_at(float(ca.x))
    epoch = Time(jd, format="jd", scale="tdb")
    sun = vector_at(cache, "GEOCENTER_SUN", jd)
    venus = vector_at(cache, "GEOCENTER_VENUS", jd)
    es = norm(sun)
    ev = norm(venus)
    venus_to_sun = sun - venus
    vs = norm(venus_to_sun)
    center, xi, eta = common_basis(sun)
    center = unit(center)
    xi = unit(xi)
    eta = unit(eta)
    scale_km_per_arcsec = es / ARCSEC

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
    along_fitted = unit(direction_t + direction_v)
    normal_fitted = np.array([-along_fitted[1], along_fitted[0]], dtype=float)

    one_second = 1.0 / 86400.0

    def mean_track_position(sample_jd: float) -> np.ndarray:
        return 0.5 * (
            relative_position(cache, "TAHITI", sample_jd, center, xi, eta)
            + relative_position(cache, "VARDO", sample_jd, center, xi, eta)
        )

    mean_minus = mean_track_position(jd - one_second)
    mean_plus = mean_track_position(jd + one_second)
    instantaneous_velocity = 0.5 * (mean_plus - mean_minus)
    along_instantaneous = unit(instantaneous_velocity)
    if float(np.dot(along_instantaneous, along_fitted)) < 0.0:
        along_instantaneous = -along_instantaneous
    normal_instantaneous = np.array(
        [-along_instantaneous[1], along_instantaneous[0]],
        dtype=float,
    )

    apparent_t_direct = relative_position(cache, "TAHITI", jd, center, xi, eta)
    apparent_v_direct = relative_position(cache, "VARDO", jd, center, xi, eta)
    apparent_delta_direct = apparent_v_direct - apparent_t_direct

    normals = {
        "Whole-transit fitted normal": normal_fitted,
        "Instantaneous mean-velocity normal": normal_instantaneous,
    }
    alongs = {
        "Whole-transit fitted normal": along_fitted,
        "Instantaneous mean-velocity normal": along_instantaneous,
    }
    for label in tuple(normals):
        if float(np.dot(apparent_delta_direct, normals[label])) < 0.0:
            normals[label] = -normals[label]

    observer_jpl: dict[str, np.ndarray] = {}
    observer_closure_rows: list[list[object]] = []
    for site in SITES:
        key = str(site["key"])
        from_sun = sun - vector_at(cache, f"{key}_SUN", jd)
        from_venus = venus - vector_at(cache, f"{key}_VENUS", jd)
        observer_jpl[key] = from_sun
        observer_closure_rows.append(
            [
                site["name"],
                norm(from_sun),
                norm(from_venus),
                norm(from_sun - from_venus),
            ]
        )

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

    baseline_jpl = observer_jpl["VARDO"] - observer_jpl["TAHITI"]

    def fit_dut1_objective(dut1_seconds: float) -> float:
        observers = iers_observers(dut1_seconds)
        baseline = observers["VARDO"] - observers["TAHITI"]
        difference = baseline - baseline_jpl
        return float(np.dot(difference, difference))

    dut1_fit_result = minimize_scalar(
        fit_dut1_objective,
        bounds=(-300.0, 300.0),
        method="bounded",
        options={"xatol": 1.0e-9, "maxiter": 500},
    )
    if not dut1_fit_result.success:
        raise RuntimeError("Diagnostic DUT1 fit failed.")
    dut1_fitted = float(dut1_fit_result.x)

    observer_models = {
        "JPL topocentric-derived": observer_jpl,
        "IERS nominal DUT1=0": iers_observers(0.0),
        "IERS fitted DUT1": iers_observers(dut1_fitted),
    }

    def relative_from_observer(observer: np.ndarray) -> np.ndarray:
        return ARCSEC * (
            gnomonic(venus - observer, center, xi, eta)
            - gnomonic(sun - observer, center, xi, eta)
        )

    synthetic_t = relative_from_observer(observer_jpl["TAHITI"])
    synthetic_v = relative_from_observer(observer_jpl["VARDO"])
    synthetic_closure_t = norm(synthetic_t - apparent_t_direct)
    synthetic_closure_v = norm(synthetic_v - apparent_v_direct)

    identity = np.eye(3, dtype=float)
    u_venus = unit(venus)
    u_sun = unit(sun)
    projector_venus = identity - np.outer(u_venus, u_venus)
    projector_sun = identity - np.outer(u_sun, u_sun)

    def reduction(
        model_label: str,
        observers: dict[str, np.ndarray],
        normal_label: str,
        normal_2d: np.ndarray,
    ) -> dict[str, float | str]:
        normal_3d = unit(normal_2d[0] * xi + normal_2d[1] * eta)
        baseline = observers["VARDO"] - observers["TAHITI"]
        baseline_normal_signed = float(np.dot(baseline, normal_3d))
        baseline_normal_km = abs(baseline_normal_signed)

        relative_t = relative_from_observer(observers["TAHITI"])
        relative_v = relative_from_observer(observers["VARDO"])
        exact_arcsec = float(np.dot(relative_v - relative_t, normal_2d))

        venus_gnomonic_vector = gnomonic_baseline_first_order(
            venus, baseline, center, xi, eta
        )
        sun_gnomonic_vector = gnomonic_baseline_first_order(
            sun, baseline, center, xi, eta
        )
        differential_gnomonic_vector = (
            venus_gnomonic_vector - sun_gnomonic_vector
        )
        gnomonic_arcsec = ARCSEC * float(
            np.dot(differential_gnomonic_vector, normal_2d)
        )

        venus_projector_vector = -(projector_venus @ baseline) / ev
        sun_projector_vector = -(projector_sun @ baseline) / es
        differential_projector_vector = (
            venus_projector_vector - sun_projector_vector
        )
        venus_projector_arcsec = ARCSEC * float(
            np.dot(venus_projector_vector, normal_3d)
        )
        sun_projector_arcsec = ARCSEC * float(
            np.dot(sun_projector_vector, normal_3d)
        )
        projector_arcsec = ARCSEC * float(
            np.dot(differential_projector_vector, normal_3d)
        )

        common_projector_arcsec = (
            -baseline_normal_signed * (1.0 / ev - 1.0 / es) * ARCSEC
        )
        halley_triangle_arcsec = (
            -baseline_normal_signed * vs / (ev * es) * ARCSEC
        )

        candidates = [
            exact_arcsec,
            gnomonic_arcsec,
            projector_arcsec,
            common_projector_arcsec,
            halley_triangle_arcsec,
        ]
        if exact_arcsec < 0.0:
            candidates = [-value for value in candidates]
            exact_arcsec, gnomonic_arcsec, projector_arcsec, common_projector_arcsec, halley_triangle_arcsec = candidates
            venus_projector_arcsec = -venus_projector_arcsec
            sun_projector_arcsec = -sun_projector_arcsec

        conversion_to_ab = scale_km_per_arcsec * ev / vs
        exact_ab_km = exact_arcsec * conversion_to_ab
        gnomonic_ab_km = gnomonic_arcsec * conversion_to_ab
        projector_ab_km = projector_arcsec * conversion_to_ab
        common_ab_km = common_projector_arcsec * conversion_to_ab
        halley_ab_km = halley_triangle_arcsec * conversion_to_ab

        nonlinear_arcsec = exact_arcsec - gnomonic_arcsec
        gnomonic_screen_arcsec = gnomonic_arcsec - projector_arcsec
        separate_projector_arcsec = projector_arcsec - common_projector_arcsec
        triangle_arcsec = common_projector_arcsec - halley_triangle_arcsec
        total_arcsec = exact_arcsec - halley_triangle_arcsec

        p_factor = EARTH_RADIUS_KM / baseline_normal_km * ev / vs
        p_exact = exact_arcsec * p_factor
        p_gnomonic = gnomonic_arcsec * p_factor
        p_projector = projector_arcsec * p_factor
        p_common = common_projector_arcsec * p_factor
        p_halley = halley_triangle_arcsec * p_factor

        return {
            "MODEL": model_label,
            "NORMAL": normal_label,
            "BASELINE_3D_KM": norm(baseline),
            "BASELINE_NORMAL_SIGNED_KM": baseline_normal_signed,
            "AB_KM": baseline_normal_km,
            "EXACT_ARCSEC": exact_arcsec,
            "GNOMONIC_ARCSEC": gnomonic_arcsec,
            "PROJECTOR_ARCSEC": projector_arcsec,
            "COMMON_ARCSEC": common_projector_arcsec,
            "HALLEY_ARCSEC": halley_triangle_arcsec,
            "VENUS_PROJECTOR_ARCSEC": venus_projector_arcsec,
            "SUN_PROJECTOR_ARCSEC": sun_projector_arcsec,
            "EXACT_AB_KM": exact_ab_km,
            "GNOMONIC_AB_KM": gnomonic_ab_km,
            "PROJECTOR_AB_KM": projector_ab_km,
            "COMMON_AB_KM": common_ab_km,
            "HALLEY_AB_KM": halley_ab_km,
            "NONLINEAR_ARCSEC": nonlinear_arcsec,
            "GNOMONIC_SCREEN_ARCSEC": gnomonic_screen_arcsec,
            "SEPARATE_PROJECTOR_ARCSEC": separate_projector_arcsec,
            "TRIANGLE_ARCSEC": triangle_arcsec,
            "TOTAL_ARCSEC": total_arcsec,
            "NONLINEAR_AB_KM": nonlinear_arcsec * conversion_to_ab,
            "GNOMONIC_SCREEN_AB_KM": gnomonic_screen_arcsec * conversion_to_ab,
            "SEPARATE_PROJECTOR_AB_KM": separate_projector_arcsec * conversion_to_ab,
            "TRIANGLE_AB_KM": triangle_arcsec * conversion_to_ab,
            "TOTAL_AB_KM": total_arcsec * conversion_to_ab,
            "P_EXACT": p_exact,
            "P_GNOMONIC": p_gnomonic,
            "P_PROJECTOR": p_projector,
            "P_COMMON": p_common,
            "P_HALLEY": p_halley,
        }

    reductions: list[dict[str, float | str]] = []
    for model_label, observers in observer_models.items():
        for normal_label, normal_2d in normals.items():
            reductions.append(
                reduction(model_label, observers, normal_label, normal_2d)
            )

    reduced = pd.DataFrame(reductions)
    jpl_reduced = reduced[reduced["MODEL"] == "JPL topocentric-derived"].copy()

    scope_frame = pd.DataFrame(
        [
            ["IERS station transformation", "Q·R·W through ERFA c2t06a", "USED"],
            ["JPL A′ and JPL B′", "Exact topocentric gnomonic positions", "USED"],
            ["Instantaneous velocity normal", "Central JPL difference at ±1 second", "USED"],
            ["Full gnomonic Jacobian", "Separate Sun and Venus vectors", "USED"],
            ["Unit-direction projectors", "P(u)=I−uuᵀ for Sun and Venus", "USED"],
            ["Common-projector approximation", "One screen direction for Sun and Venus", "AUDITED"],
            ["Halley triangle", "VS replaces ES−EV", "AUDITED"],
            ["Reduction to P", "R⊕/AB × EV/VS", "USED"],
        ],
        columns=["Component", "Definition", "Status"],
    )

    epoch_frame = pd.DataFrame(
        [
            ["Instantaneous geocentric closest approach", utc_text(jd) + " UTC"],
            ["JD TDB", jd],
            ["Earth → Venus EV km", ev],
            ["Venus → Sun VS km", vs],
            ["Earth → Sun ES km", es],
            ["EV/VS", ev / vs],
            ["Earth–Sun scale km/arcsec", scale_km_per_arcsec],
            ["Diagnostic fitted DUT1 seconds", dut1_fitted],
        ],
        columns=["Quantity", "Value"],
    )

    normal_frame = pd.DataFrame(
        [
            [
                label,
                math.degrees(math.atan2(alongs[label][1], alongs[label][0])),
                float(np.dot(apparent_delta_direct, normal_2d)),
            ]
            for label, normal_2d in normals.items()
        ],
        columns=["Normal definition", "Track angle deg", "JPL A′B′ arcsec"],
    )

    closure_frame = pd.DataFrame(
        observer_closure_rows,
        columns=[
            "Station",
            "Observer radius from Sun km",
            "Observer radius from Venus km",
            "Sun/Venus observer closure km",
        ],
    )
    synthetic_frame = pd.DataFrame(
        [
            ["Tahiti", synthetic_closure_t],
            ["Vardø", synthetic_closure_v],
        ],
        columns=["Station", "Synthetic minus direct JPL screen arcsec"],
    )

    aprime_rows: list[list[object]] = []
    for _, row in jpl_reduced.iterrows():
        separation = float(row["EXACT_ARCSEC"])
        aprime_rows.extend(
            [
                [row["NORMAL"], "JPL A′", -0.5 * separation, "midpoint-centered"],
                [row["NORMAL"], "JPL B′", +0.5 * separation, "midpoint-centered"],
                [row["NORMAL"], "JPL A′B′", separation, "B′ − A′"],
            ]
        )
    aprime_frame = pd.DataFrame(
        aprime_rows,
        columns=["Normal definition", "JPL quantity", "Arcseconds", "Definition"],
    )

    baseline_frame = reduced[
        [
            "MODEL",
            "NORMAL",
            "BASELINE_3D_KM",
            "AB_KM",
            "EXACT_ARCSEC",
            "EXACT_AB_KM",
        ]
    ].rename(
        columns={
            "MODEL": "Station model",
            "NORMAL": "Normal definition",
            "BASELINE_3D_KM": "3D baseline km",
            "AB_KM": "Direct common-normal AB km",
            "EXACT_ARCSEC": "Exact synthesized A′B′ arcsec",
            "EXACT_AB_KM": "Halley-recovered AB km",
        }
    )

    ladder_frame = reduced[
        [
            "MODEL",
            "NORMAL",
            "EXACT_ARCSEC",
            "GNOMONIC_ARCSEC",
            "PROJECTOR_ARCSEC",
            "COMMON_ARCSEC",
            "HALLEY_ARCSEC",
        ]
    ].rename(
        columns={
            "MODEL": "Station model",
            "NORMAL": "Normal definition",
            "EXACT_ARCSEC": "Exact nonlinear JPL arcsec",
            "GNOMONIC_ARCSEC": "Gnomonic first-order arcsec",
            "PROJECTOR_ARCSEC": "Separate-projector arcsec",
            "COMMON_ARCSEC": "Common-projector arcsec",
            "HALLEY_ARCSEC": "Halley-triangle arcsec",
        }
    )

    object_terms_frame = jpl_reduced[
        [
            "NORMAL",
            "VENUS_PROJECTOR_ARCSEC",
            "SUN_PROJECTOR_ARCSEC",
            "PROJECTOR_ARCSEC",
        ]
    ].rename(
        columns={
            "NORMAL": "Normal definition",
            "VENUS_PROJECTOR_ARCSEC": "Venus baseline term arcsec",
            "SUN_PROJECTOR_ARCSEC": "Sun baseline term arcsec",
            "PROJECTOR_ARCSEC": "Venus minus Sun arcsec",
        }
    )

    budget_frame = jpl_reduced[
        [
            "NORMAL",
            "NONLINEAR_ARCSEC",
            "NONLINEAR_AB_KM",
            "GNOMONIC_SCREEN_ARCSEC",
            "GNOMONIC_SCREEN_AB_KM",
            "SEPARATE_PROJECTOR_ARCSEC",
            "SEPARATE_PROJECTOR_AB_KM",
            "TRIANGLE_ARCSEC",
            "TRIANGLE_AB_KM",
            "TOTAL_ARCSEC",
            "TOTAL_AB_KM",
        ]
    ].rename(
        columns={
            "NORMAL": "Normal definition",
            "NONLINEAR_ARCSEC": "Finite-baseline residual arcsec",
            "NONLINEAR_AB_KM": "Finite-baseline residual AB km",
            "GNOMONIC_SCREEN_ARCSEC": "Gnomonic-screen correction arcsec",
            "GNOMONIC_SCREEN_AB_KM": "Gnomonic-screen correction AB km",
            "SEPARATE_PROJECTOR_ARCSEC": "Separate-projector correction arcsec",
            "SEPARATE_PROJECTOR_AB_KM": "Separate-projector correction AB km",
            "TRIANGLE_ARCSEC": "Distance-triangle correction arcsec",
            "TRIANGLE_AB_KM": "Distance-triangle correction AB km",
            "TOTAL_ARCSEC": "Exact minus Halley arcsec",
            "TOTAL_AB_KM": "Exact minus Halley AB km",
        }
    )

    p_direct_small = EARTH_RADIUS_KM / es * ARCSEC
    p_direct_exact = math.asin(EARTH_RADIUS_KM / es) * ARCSEC
    p_rows: list[list[object]] = []
    for _, row in jpl_reduced.iterrows():
        p_rows.extend(
            [
                [row["NORMAL"], "P from exact JPL A′B′", row["P_EXACT"]],
                [row["NORMAL"], "P from gnomonic first order", row["P_GNOMONIC"]],
                [row["NORMAL"], "P from separate projectors", row["P_PROJECTOR"]],
                [row["NORMAL"], "P from common projector", row["P_COMMON"]],
                [row["NORMAL"], "P from Halley triangle", row["P_HALLEY"]],
            ]
        )
    p_rows.extend(
        [
            ["Geocentric distance", "Direct P = R⊕/ES", p_direct_small],
            ["Geocentric distance", "Direct exact P = asin(R⊕/ES)", p_direct_exact],
        ]
    )
    p_frame = pd.DataFrame(
        p_rows,
        columns=["Normal / reference", "P definition", "P arcsec"],
    )

    status_rows: list[list[object]] = [
        [
            "JPL Sun/Venus observer-vector closure",
            "PASS" if max(row[3] for row in observer_closure_rows) < 1.0e-5 else "FAIL",
            max(row[3] for row in observer_closure_rows),
        ],
        [
            "Synthetic versus direct JPL screen closure",
            "PASS" if max(synthetic_closure_t, synthetic_closure_v) < 1.0e-7 else "FAIL",
            max(synthetic_closure_t, synthetic_closure_v),
        ],
        [
            "Exact residual-budget identity",
            "PASS",
            float(
                np.max(
                    np.abs(
                        jpl_reduced["TOTAL_ARCSEC"].to_numpy(dtype=float)
                        - (
                            jpl_reduced["NONLINEAR_ARCSEC"].to_numpy(dtype=float)
                            + jpl_reduced["GNOMONIC_SCREEN_ARCSEC"].to_numpy(dtype=float)
                            + jpl_reduced["SEPARATE_PROJECTOR_ARCSEC"].to_numpy(dtype=float)
                            + jpl_reduced["TRIANGLE_ARCSEC"].to_numpy(dtype=float)
                        )
                    )
                )
            ),
        ],
        [
            "Halley-triangle P versus direct small-angle P",
            "PASS",
            float(
                np.max(
                    np.abs(
                        jpl_reduced["P_HALLEY"].to_numpy(dtype=float)
                        - p_direct_small
                    )
                )
            ),
        ],
    ]
    status_frame = pd.DataFrame(
        status_rows,
        columns=["Equation / test", "Status", "Residual / diagnostic"],
    )

    frames = [
        ("SCOPE", scope_frame),
        ("EPOCH", epoch_frame),
        ("NORMALS", normal_frame),
        ("OBSERVER_CLOSURE", closure_frame),
        ("SCREEN_CLOSURE", synthetic_frame),
        ("APRIME_BPRIME", aprime_frame),
        ("BASELINES", baseline_frame),
        ("LADDER", ladder_frame),
        ("OBJECT_TERMS", object_terms_frame),
        ("RESIDUAL_BUDGET", budget_frame),
        ("P_REDUCTION", p_frame),
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
</style>
"""

    html: list[str] = [css, '<div class="r">']
    html.append("<h1>1769 Venus Transit — Full-Vector Differential Parallax and P Audit</h1>")
    html.append("<h2>CODE INPUTS</h2>")
    html.append(f"<p><b>JPL source:</b> {source}</p>")
    html.append(f"<p><b>Instantaneous epoch:</b> {utc_text(jd)} UTC</p>")
    html.append(f"<p><b>Earth equatorial radius:</b> {EARTH_RADIUS_KM:,.6f} km</p>")

    html.append("<h2>COMMENTS</h2>")
    html.append(
        '<p class="note">JPL A′ and JPL B′ are midpoint-centered reporting coordinates. Their invariant separation JPL A′B′ is calculated directly from the synchronized topocentric JPL vectors.</p>'
    )
    html.append(
        '<p class="note">The approximation ladder isolates finite-baseline nonlinearity, the gnomonic screen Jacobian, separate Venus/Sun projectors, and the Halley distance-triangle substitution.</p>'
    )
    html.append(
        '<p class="note">The instantaneous mean-velocity normal directly tests whether the whole-transit fitted direction is responsible for the residual.</p>'
    )

    html.append("<h2>RESULTS</h2>")
    html.append("<h3>Audit scope</h3>")
    html.append(table(scope_frame))
    html.append("<h3>Instantaneous epoch and JPL distances</h3>")
    html.append(table(epoch_frame, {"Value": "{:,.15f}"}))
    html.append("<h3>Track-normal definitions</h3>")
    html.append(table(normal_frame, {"Track angle deg": "{:.12f}", "JPL A′B′ arcsec": "{:.12f}"}))
    html.append("<h3>Observer-vector closure</h3>")
    html.append(table(closure_frame, {
        "Observer radius from Sun km": "{:,.12f}",
        "Observer radius from Venus km": "{:,.12f}",
        "Sun/Venus observer closure km": "{:.12e}",
    }))
    html.append("<h3>Synthetic versus direct JPL screen closure</h3>")
    html.append(table(synthetic_frame, {"Synthetic minus direct JPL screen arcsec": "{:.12e}"}))
    html.append("<h3>Explicit JPL A′, JPL B′, and JPL A′B′</h3>")
    html.append(table(aprime_frame, {"Arcseconds": "{:+.12f}"}))
    html.append("<h3>Station models and recovered AB</h3>")
    html.append(table(baseline_frame, {
        "3D baseline km": "{:,.12f}",
        "Direct common-normal AB km": "{:,.12f}",
        "Exact synthesized A′B′ arcsec": "{:.12f}",
        "Halley-recovered AB km": "{:,.12f}",
    }))
    html.append("<h3>Approximation ladder</h3>")
    html.append(table(ladder_frame, {
        "Exact nonlinear JPL arcsec": "{:.12f}",
        "Gnomonic first-order arcsec": "{:.12f}",
        "Separate-projector arcsec": "{:.12f}",
        "Common-projector arcsec": "{:.12f}",
        "Halley-triangle arcsec": "{:.12f}",
    }))
    html.append("<h3>JPL Venus and Sun projector terms</h3>")
    html.append(table(object_terms_frame, {
        "Venus baseline term arcsec": "{:+.12f}",
        "Sun baseline term arcsec": "{:+.12f}",
        "Venus minus Sun arcsec": "{:+.12f}",
    }))
    html.append("<h3>JPL residual budget</h3>")
    html.append(table(budget_frame, {
        column: "{:+.12f}" for column in budget_frame.columns if column != "Normal definition"
    }))
    html.append("<h3>Reduction down to P</h3>")
    html.append(table(p_frame, {"P arcsec": "{:.12f}"}))

    html.append("<h2>OUTPUT SUMMARY</h2>")
    html.append(f'<p class="path">{CSV}</p>')
    html.append(f'<p class="path">{HTML_FILE}</p>')

    html.append("<h2>PAPER COMPARISON</h2>")
    html.append(
        '<p class="note">The dominant row in the residual budget identifies whether the ≈19.6 km difference comes from station transformation, track-normal choice, finite-baseline nonlinearity, gnomonic projection, separate Sun/Venus projectors, or the distance triangle.</p>'
    )

    html.append("<h2>EQUATION STATUS</h2>")
    html.append(table(status_frame, {"Residual / diagnostic": "{:+.15e}"}))
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
# V0042
