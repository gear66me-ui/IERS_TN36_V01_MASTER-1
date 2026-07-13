# V0032
# Audit reference: Exact JPL topocentric vector inversion for the 1769 Tahiti–Vardø Venus Halley reduction.
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
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
):
    ensure_package(_import_name, _pip_name)

import numpy as np
import pandas as pd
from scipy.optimize import brentq

VERSION = "V0032"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUTPUT_DIR = ROOT / "VENUS_1769_TAHITI_VARDO_EXACT_VECTOR_REDUCTION_V0032_OUTPUT"
OUTPUT_CSV = OUTPUT_DIR / "VENUS_1769_TAHITI_VARDO_EXACT_VECTOR_REDUCTION_V0032.csv"
BASE_COMMIT = "d55bc2274359a3014c19f257e42cc149bc458d57"
BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/"
    f"{BASE_COMMIT}/VENUS_1769_TAHITI_VARDO_HALLEY_THUMB_CHECK_V0031.py"
)
BASE_PATH = ROOT / "VENUS_1769_TAHITI_VARDO_HALLEY_THUMB_CHECK_V0031_BASE.py"


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


def main() -> None:
    base = load_base_namespace()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    build_master = base["build_master"]
    build_cache = base["build_cache"]
    vector_at = base["vector_at"]
    reference_epoch = base["reference_epoch"]
    common_basis = base["common_basis"]
    external_contacts = base["external_contacts"]
    relative_position_arcsec = base["relative_position_arcsec"]
    fitted_direction = base["fitted_direction"]
    gnomonic = base["gnomonic"]
    norm = base["norm"]
    unit = base["unit"]
    utc_text = base["utc_text"]

    arcsec_per_rad = float(base["ARCSEC_PER_RAD"])

    master = build_master()
    cache = build_cache(master)
    jd_reference = float(reference_epoch(cache))

    geocentric_sun = vector_at(cache, "GEOCENTER_SUN", jd_reference)
    geocentric_venus = vector_at(cache, "GEOCENTER_VENUS", jd_reference)
    venus_to_sun = geocentric_sun - geocentric_venus
    center, xi, eta = common_basis(geocentric_sun)

    contacts_t = external_contacts(cache, "TAHITI")
    contacts_v = external_contacts(cache, "VARDO")
    common_start = max(float(contacts_t[0]), float(contacts_v[0]))
    common_stop = min(float(contacts_t[1]), float(contacts_v[1]))
    minute_jds = np.asarray(cache["JD_TDB"], dtype=float)
    selected_jds = minute_jds[(minute_jds >= common_start) & (minute_jds <= common_stop)]
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
        [-common_direction[1], common_direction[0]],
        dtype=float,
    )

    apparent_t = relative_position_arcsec(
        cache, "TAHITI", jd_reference, center, xi, eta
    )
    apparent_v = relative_position_arcsec(
        cache, "VARDO", jd_reference, center, xi, eta
    )
    observed_signed_arcsec = float(
        np.dot(apparent_v - apparent_t, common_normal)
    )
    aprime_bprime_arcsec = abs(observed_signed_arcsec)

    topocentric_sun_t = vector_at(cache, "TAHITI_SUN", jd_reference)
    topocentric_sun_v = vector_at(cache, "VARDO_SUN", jd_reference)
    observer_t = geocentric_sun - topocentric_sun_t
    observer_v = geocentric_sun - topocentric_sun_v
    observer_midpoint = 0.5 * (observer_t + observer_v)
    observer_baseline = observer_v - observer_t

    baseline_plane = np.array(
        [
            float(np.dot(observer_baseline, xi)),
            float(np.dot(observer_baseline, eta)),
        ],
        dtype=float,
    )
    direct_signed_ab_km = float(np.dot(baseline_plane, common_normal))
    ab_direct_km = abs(direct_signed_ab_km)

    earth_sun_km = norm(geocentric_sun)
    earth_venus_km = norm(geocentric_venus)
    venus_sun_km = norm(venus_to_sun)

    aprime_bprime_km = (
        aprime_bprime_arcsec * earth_sun_km / arcsec_per_rad
    )
    ab_direct_arcsec = ab_direct_km * arcsec_per_rad / earth_sun_km

    classical_factor = earth_venus_km / venus_sun_km
    ab_classical_arcsec = aprime_bprime_arcsec * classical_factor
    ab_classical_km = aprime_bprime_km * classical_factor
    classical_residual_arcsec = ab_direct_arcsec - ab_classical_arcsec
    classical_residual_km = ab_direct_km - ab_classical_km
    classical_residual_percent = (
        100.0 * classical_residual_arcsec / ab_direct_arcsec
    )

    def relative_from_observer(observer: np.ndarray) -> np.ndarray:
        topocentric_sun = geocentric_sun - observer
        topocentric_venus = geocentric_venus - observer
        return arcsec_per_rad * (
            gnomonic(topocentric_venus, center, xi, eta)
            - gnomonic(topocentric_sun, center, xi, eta)
        )

    def modeled_signed_separation(scale: float) -> float:
        trial_t = observer_midpoint - 0.5 * scale * observer_baseline
        trial_v = observer_midpoint + 0.5 * scale * observer_baseline
        relative_t = relative_from_observer(trial_t)
        relative_v = relative_from_observer(trial_v)
        return float(np.dot(relative_v - relative_t, common_normal))

    def root_function(scale: float) -> float:
        return modeled_signed_separation(scale) - observed_signed_arcsec

    lower = 0.0
    upper = 2.0
    f_lower = root_function(lower)
    f_upper = root_function(upper)
    while f_lower * f_upper > 0.0 and upper < 64.0:
        upper *= 2.0
        f_upper = root_function(upper)
    if f_lower * f_upper > 0.0:
        raise RuntimeError("Exact JPL vector inversion failed to bracket the baseline scale.")

    recovered_scale = float(
        brentq(
            root_function,
            lower,
            upper,
            xtol=1.0e-14,
            rtol=1.0e-14,
            maxiter=300,
        )
    )

    ab_exact_km = abs(recovered_scale * direct_signed_ab_km)
    ab_exact_arcsec = ab_exact_km * arcsec_per_rad / earth_sun_km
    exact_transfer_factor = ab_exact_km / aprime_bprime_km

    exact_residual_arcsec = ab_direct_arcsec - ab_exact_arcsec
    exact_residual_km = ab_direct_km - ab_exact_km
    exact_residual_percent = 100.0 * exact_residual_arcsec / ab_direct_arcsec

    rows = [
        ["Reference UTC", utc_text(jd_reference), "UTC"],
        ["A′B′ direct JPL", aprime_bprime_arcsec, "arcsec"],
        ["A′B′ direct JPL", aprime_bprime_km, "km"],
        ["Earth → Venus", earth_venus_km, "km"],
        ["Venus → Sun", venus_sun_km, "km"],
        ["Earth → Sun", earth_sun_km, "km"],
        ["Classical Halley factor EV/VS", classical_factor, "ratio"],
        ["AB classical Halley", ab_classical_arcsec, "arcsec"],
        ["AB classical Halley", ab_classical_km, "km"],
        ["AB direct JPL", ab_direct_arcsec, "arcsec"],
        ["AB direct JPL", ab_direct_km, "km"],
        ["Classical residual", classical_residual_arcsec, "arcsec"],
        ["Classical residual", classical_residual_km, "km"],
        ["Classical residual", classical_residual_percent, "%"],
        ["Exact JPL vector scale", recovered_scale, "ratio"],
        ["Exact JPL transfer factor", exact_transfer_factor, "ratio"],
        ["AB exact vector reduction", ab_exact_arcsec, "arcsec"],
        ["AB exact vector reduction", ab_exact_km, "km"],
        ["Exact closure residual", exact_residual_arcsec, "arcsec"],
        ["Exact closure residual", exact_residual_km, "km"],
        ["Exact closure residual", exact_residual_percent, "%"],
    ]
    result = pd.DataFrame(rows, columns=["Quantity", "Value", "Unit"])
    result.to_csv(OUTPUT_CSV, index=False, float_format="%.15f")

    def format_value(value: object) -> str:
        if isinstance(value, str):
            return value
        return f"{float(value):,.12f}"

    print(
        result.to_string(
            index=False,
            formatters={"Value": format_value},
        )
    )
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0032
