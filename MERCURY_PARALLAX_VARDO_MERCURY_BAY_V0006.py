# V0006
# Audit reference: GitHub-native correction of the Mercury Bay–Vardø JPL parallax audit with separated inputs and outputs.
from __future__ import annotations

import math
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "MERCURY_PARALLAX_VARDO_MERCURY_BAY_V0005.py"
)
VERSION = "V0006"
PROGRAM = "MERCURY_PARALLAX_VARDO_MERCURY_BAY_V0006.py"
LOCAL_TZ = ZoneInfo("America/Bogota")


def load_base_module() -> dict[str, object]:
    with urllib.request.urlopen(BASE_URL, timeout=60) as response:
        source = response.read().decode("utf-8")
    namespace: dict[str, object] = {
        "__name__": "mercury_v0005_base",
        "__file__": BASE_URL,
    }
    exec(compile(source, BASE_URL, "exec"), namespace)
    return namespace


base = load_base_module()
np = base["np"]
pd = base["pd"]
brentq = base["brentq"]
tabulate = __import__("tabulate").tabulate

AU_KM = float(base["AU_KM"])
ARCSEC_PER_RAD = float(base["ARCSEC_PER_RAD"])
EARTH_RADIUS_WGS84_KM = float(base["EARTH_RADIUS_WGS84_KM"])
REFERENCE_PI_ARCSEC = math.asin(EARTH_RADIUS_WGS84_KM / AU_KM) * ARCSEC_PER_RAD
SITE_A = base["SITE_A"]
SITE_B = base["SITE_B"]
OUTPUT_DEFAULT = Path("/content/MERCURY_PARALLAX_VARDO_MERCURY_BAY_V0006_OUTPUT")


def observer_position(cache, site_key: str, target: str, jd: float):
    return base["vector_at"](cache, f"GEOCENTER_{target}", jd) - base["vector_at"](
        cache, f"{site_key}_{target}", jd
    )


def common_normal_audit(cache, closest_jd: float, basis) -> dict[str, object]:
    tangent_a = base["local_tangent"](cache, "MERCURY_BAY", closest_jd, basis)
    tangent_b = base["local_tangent"](cache, "VARDO", closest_jd, basis)
    tangent = base["unit"](tangent_a + tangent_b)
    if tangent[0] < 0.0:
        tangent = -tangent
    normal_2d = np.array([-tangent[1], tangent[0]])

    point_a = base["apparent_point"](cache, "MERCURY_BAY", closest_jd, basis)
    point_b = base["apparent_point"](cache, "VARDO", closest_jd, basis)
    observed_rho = abs(float(np.dot(point_b - point_a, normal_2d)))

    _screen_normal, xhat, yhat = basis
    baseline_3d = base["observer_baseline"](cache, closest_jd)
    baseline_screen = np.array([
        float(np.dot(baseline_3d, xhat)),
        float(np.dot(baseline_3d, yhat)),
    ])
    baseline_normal = abs(float(np.dot(baseline_screen, normal_2d)))

    sun = base["vector_at"](cache, "GEOCENTER_SUN", closest_jd)
    mercury = base["vector_at"](cache, "GEOCENTER_MERCURY", closest_jd)
    d_es = base["norm"](sun)
    d_em = base["norm"](mercury)
    d_ms = base["norm"](sun - mercury)

    linear_rho = baseline_normal * d_ms / (d_em * d_es) * ARCSEC_PER_RAD
    rejected_event = observed_rho * (d_em / d_ms) * (EARTH_RADIUS_WGS84_KM / baseline_normal)
    rejected_normalized = rejected_event * d_es / AU_KM

    return {
        "normal_2d": normal_2d,
        "observed_rho_arcsec": observed_rho,
        "linear_rho_arcsec": linear_rho,
        "rho_excess_arcsec": observed_rho - linear_rho,
        "perspective_factor": observed_rho / linear_rho,
        "baseline_normal_km": baseline_normal,
        "d_es_km": d_es,
        "d_em_km": d_em,
        "d_ms_km": d_ms,
        "rejected_normalized_arcsec": rejected_normalized,
    }


def point_from_vectors(sun, mercury, basis):
    _normal, xhat, yhat = basis
    difference = base["unit"](mercury) - base["unit"](sun)
    return np.array([
        float(np.dot(difference, xhat)) * ARCSEC_PER_RAD,
        float(np.dot(difference, yhat)) * ARCSEC_PER_RAD,
    ])


def exact_nonlinear_recovery(cache, closest_jd: float, basis, normal_2d, observed_rho: float) -> dict[str, float]:
    geo_sun = base["vector_at"](cache, "GEOCENTER_SUN", closest_jd)
    geo_mercury = base["vector_at"](cache, "GEOCENTER_MERCURY", closest_jd)
    observer_vectors = {}
    consistency = []

    for site_key in ("MERCURY_BAY", "VARDO"):
        observer_sun = observer_position(cache, site_key, "SUN", closest_jd)
        observer_mercury = observer_position(cache, site_key, "MERCURY", closest_jd)
        observer_vectors[site_key] = (observer_sun, observer_mercury)
        consistency.append(base["norm"](observer_sun - observer_mercury))

    def model_point(site_key: str, scale: float):
        observer_sun, observer_mercury = observer_vectors[site_key]
        return point_from_vectors(
            geo_sun - scale * observer_sun,
            geo_mercury - scale * observer_mercury,
            basis,
        )

    def model_rho(scale: float) -> float:
        difference = model_point("VARDO", scale) - model_point("MERCURY_BAY", scale)
        return abs(float(np.dot(difference, normal_2d)))

    upper = 2.0
    while model_rho(upper) < observed_rho and upper < 16.0:
        upper *= 2.0
    if model_rho(upper) < observed_rho:
        raise RuntimeError("Exact nonlinear recovery could not bracket the JPL separation.")

    recovered_scale = float(
        brentq(
            lambda scale: model_rho(scale) - observed_rho,
            0.0,
            upper,
            xtol=1.0e-14,
            rtol=1.0e-14,
        )
    )
    event_parallax = math.asin(
        recovered_scale * EARTH_RADIUS_WGS84_KM / base["norm"](geo_sun)
    ) * ARCSEC_PER_RAD
    normalized = math.asin(
        recovered_scale * EARTH_RADIUS_WGS84_KM / AU_KM
    ) * ARCSEC_PER_RAD

    return {
        "recovered_earth_scale": recovered_scale,
        "modeled_rho_arcsec": model_rho(recovered_scale),
        "rho_closure_microarcsec": (model_rho(recovered_scale) - observed_rho) * 1_000_000.0,
        "observer_consistency_max_km": max(consistency),
        "event_parallax_arcsec": event_parallax,
        "normalized_parallax_arcsec": normalized,
        "residual_vs_standard_arcsec": normalized - REFERENCE_PI_ARCSEC,
    }


def print_table(frame) -> None:
    print(tabulate(frame, headers="keys", tablefmt="rounded_grid", showindex=False))


def main() -> None:
    output_dir = OUTPUT_DEFAULT
    output_dir.mkdir(parents=True, exist_ok=True)

    master, master_path, source = base["locate_or_build_master"]("")
    cache = base["build_cache"](master)
    closest_jd = base["find_geocenter_closest"](cache)
    closest_utc = base["utc_at"](closest_jd)
    basis = base["fixed_screen_basis"](cache, closest_jd)

    contacts_a = base["contacts"](cache, "MERCURY_BAY")
    contacts_b = base["contacts"](cache, "VARDO")
    contact_table = pd.concat(
        [
            base["build_contact_table"](cache, SITE_A, contacts_a, closest_jd),
            base["build_contact_table"](cache, SITE_B, contacts_b, closest_jd),
        ],
        ignore_index=True,
    )

    geometry = common_normal_audit(cache, closest_jd, basis)
    exact = exact_nonlinear_recovery(
        cache,
        closest_jd,
        basis,
        geometry["normal_2d"],
        float(geometry["observed_rho_arcsec"]),
    )

    rejected_error = float(geometry["rejected_normalized_arcsec"]) - float(
        exact["normalized_parallax_arcsec"]
    )
    rejected_percent = rejected_error / float(exact["normalized_parallax_arcsec"]) * 100.0

    mercury_bay_visible = bool(
        contact_table.loc[
            contact_table["site"] == SITE_A["label"],
            "observable_sun_above_horizon",
        ].all()
    )
    vardo_visible = bool(
        contact_table.loc[
            contact_table["site"] == SITE_B["label"],
            "observable_sun_above_horizon",
        ].all()
    )

    input_table = pd.DataFrame(
        [
            ["JPL target series", "Sun + Mercury", "6 vector series"],
            ["JPL cadence", "1 minute", "1769-11-09/10"],
            ["Mercury Bay latitude", f"{SITE_A['lat_deg']:.12f}", "deg"],
            ["Mercury Bay longitude", f"{SITE_A['lon_deg_east']:.12f}", "deg east"],
            ["Vardø latitude", f"{SITE_B['lat_deg']:.12f}", "deg"],
            ["Vardø longitude", f"{SITE_B['lon_deg_east']:.12f}", "deg east"],
            ["Earth equatorial radius a", f"{EARTH_RADIUS_WGS84_KM:.6f}", "km"],
            ["Astronomical unit AU", f"{AU_KM:.6f}", "km"],
        ],
        columns=["Calculation input", "Value", "Unit / range"],
    )

    output_table = pd.DataFrame(
        [
            ["Geocentric closest approach", closest_utc, "", "DERIVED"],
            ["Exact JPL normal separation ρN", f"{geometry['observed_rho_arcsec']:.12f}", "arcsec", "DERIVED"],
            ["First-order predicted separation", f"{geometry['linear_rho_arcsec']:.12f}", "arcsec", "DERIVED"],
            ["Perspective/projection excess", f"{geometry['rho_excess_arcsec']:+.12f}", "arcsec", "ERROR SOURCE"],
            ["Perspective correction factor", f"{geometry['perspective_factor']:.12f}", "", "DERIVED"],
            ["Independent normal baseline BN", f"{geometry['baseline_normal_km']:.6f}", "km", "DERIVED"],
            ["First-order π⊙", f"{geometry['rejected_normalized_arcsec']:.12f}", "arcsec", "REJECTED"],
            ["Exact nonlinear recovered π⊙", f"{exact['normalized_parallax_arcsec']:.12f}", "arcsec", "ACCEPTED"],
            ["WGS84 / IAU 2012 reference", f"{REFERENCE_PI_ARCSEC:.12f}", "arcsec", "COMPARISON ONLY"],
            ["Rejected-formula error", f"{rejected_error:+.12f}", "arcsec", "RESIDUAL"],
            ["Rejected-formula error", f"{rejected_percent:+.9f}", "%", "RESIDUAL"],
            ["Exact recovery residual", f"{exact['residual_vs_standard_arcsec']:+.12f}", "arcsec", "RESIDUAL"],
        ],
        columns=["Derived output", "Value", "Unit", "Status"],
    )

    visibility_table = pd.DataFrame(
        [
            ["Mercury Bay", "YES" if mercury_bay_visible else "NO"],
            ["Vardø", "YES" if vardo_visible else "NO"],
        ],
        columns=["Station", "Sun above horizon for full transit"],
    )

    results_csv = output_dir / "MERCURY_1769_PARALLAX_RESULTS_V0006.csv"
    geometry_csv = output_dir / "MERCURY_1769_GEOMETRY_AUDIT_V0006.csv"
    contacts_csv = output_dir / "MERCURY_1769_CONTACTS_VISIBILITY_V0006.csv"
    output_table.to_csv(results_csv, index=False)
    pd.DataFrame(
        [
            {
                **{key: value for key, value in geometry.items() if key != "normal_2d"},
                **exact,
                "rejected_error_arcsec": rejected_error,
                "rejected_error_percent": rejected_percent,
            }
        ]
    ).to_csv(geometry_csv, index=False, float_format="%.15f")
    contact_table.to_csv(contacts_csv, index=False, float_format="%.15f")

    checks = {
        "exact rho closure": abs(float(exact["rho_closure_microarcsec"])) < 0.01,
        "exact standard recovery": abs(float(exact["residual_vs_standard_arcsec"])) < 1.0e-8,
        "first-order result rejected": rejected_error > 0.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Audit failure: " + ", ".join(failed))

    print(f"CODE OUTPUT: {VERSION}")
    print("CODE INPUTS")
    print_table(input_table)
    print("COMMENTS")
    print("8.794143838847 arcsec is a calculated comparison standard, not a solver input.")
    print("8.861702747591 arcsec is the rejected first-order scalar approximation.")
    print("RESULTS")
    print_table(output_table)
    print("VISIBILITY STATUS")
    print_table(visibility_table)
    print("OUTPUT SUMMARY")
    print(f"JPL source: {source}")
    print(f"JPL master: {master_path}")
    print(f"Results CSV: {results_csv}")
    print(f"Geometry CSV: {geometry_csv}")
    print(f"Contacts CSV: {contacts_csv}")
    print("Plot: NOT GENERATED IN V0006 — calculation audit first.")
    print("PAPER COMPARISON")
    print("Mercury Bay is observable; Vardø is a virtual JPL station because the Sun is below the horizon.")
    print("EQUATION STATUS")
    print("Exact nonlinear JPL ray inversion: PASS")
    print("First-order common-normal formula: REJECTED / NOT USED")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# V0006
