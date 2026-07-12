# V0016
# Audit reference: Exact IERS-0012N plotting configuration with IAU-1976 Earth radius and c tau_A solar distance applied to all plotted scales and parallax values.
from __future__ import annotations

import py_compile
import runpy
from pathlib import Path
from urllib.request import Request, urlopen

VERSION = "V0016"
PROGRAM = "IERS_0012N_EXACT_PLOT_IAU1976_V0016.py"
ROOT = Path("/content")
ORIGINAL = ROOT / "IERS_0012N_ORIGINAL_SOURCE_V0016.py"
PATCHED = ROOT / "IERS_0012N_EXACT_PLOT_IAU1976_ENGINE_V0016.py"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "IERS_0012N_VARDO_POINT_VENUS_ENGINEERING_TRACK_PLOT_PI_SUN.py?v=16"
)


def fetch_source() -> str:
    request = Request(SOURCE_URL, headers={"User-Agent": PROGRAM})
    with urlopen(request, timeout=180) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError("IERS-0012N source download was empty.")
    ORIGINAL.write_bytes(payload)
    return payload.decode("utf-8")


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"V0016 patch failed for {label}: expected one match, found {count}."
        )
    return text.replace(old, new, 1)


def build_engine(source: str) -> str:
    text = source

    text = replace_exact(
        text,
        "# IERS-0012N\n# Audit reference: GitHubDelivery@IERS-0012N; 1769 Vardo-Point Venus engineering half-Sun plot using JPL Horizons SITE_COORD vectors.",
        "# V0016\n# Audit reference: Exact IERS-0012N plot configuration with IAU-1976 distance normalization.",
        "header",
    )
    text = replace_exact(text, 'VERSION = "IERS-0012N"', 'VERSION = "V0016"', "version")
    text = replace_exact(
        text,
        'PROGRAM_NAME = "IERS_0012N_VARDO_POINT_VENUS_ENGINEERING_TRACK_PLOT_PI_SUN.py"',
        'PROGRAM_NAME = "IERS_0012N_EXACT_PLOT_IAU1976_ENGINE_V0016.py"',
        "program name",
    )

    text = replace_exact(
        text,
        """AU_KM = 149_597_870.7
ARCSEC_PER_RAD = 206_264.80624709636
EARTH_RADIUS_KM = 6_378.137
SUN_RADIUS_KM = 695_700.0
VENUS_RADIUS_KM = 6_051.8
PI_SUN_REFERENCE_ARCSEC = 8.794148""",
        """AU_KM = 149_597_870.7
ARCSEC_PER_RAD = 206_264.80624709636
EARTH_RADIUS_KM = 6_378.140
C_KM_S = 299_792.458
TAU_A_S = 499.004782
IAU1976_AU_KM = C_KM_S * TAU_A_S
SUN_RADIUS_KM = 695_700.0
VENUS_RADIUS_KM = 6_051.8
PI_SUN_REFERENCE_ARCSEC = math.asin(EARTH_RADIUS_KM / IAU1976_AU_KM) * ARCSEC_PER_RAD""",
        "IAU-1976 constants",
    )

    text = replace_exact(
        text,
        """def ray_screen_point_arcsec_sitecoord(geo_cache, topo_cache, site, jd_tdb, basis):
    n, xhat, yhat = basis
    sun_geo = vec_at(geo_cache, "GEOCENTER_SUN", jd_tdb)
    sun_topo = site_sun_vector(topo_cache, site, jd_tdb)
    venus_topo = site_venus_vector(topo_cache, site, jd_tdb)
    obs_geo = sun_geo - sun_topo
    ray = venus_topo
    denom = float(np.dot(ray, n))
    if abs(denom) < 1e-14:
        raise RuntimeError("SITE_COORD ray nearly parallel to solar screen.")
    tau = float(np.dot(sun_geo - obs_geo, n) / denom)
    hit = obs_geo + tau * ray
    screen_vec = hit - sun_geo
    es = norm(sun_geo)
    x = math.atan2(float(np.dot(screen_vec, xhat)), es) * ARCSEC_PER_RAD
    y = math.atan2(float(np.dot(screen_vec, yhat)), es) * ARCSEC_PER_RAD
    return np.array([x, y], dtype=float)""",
        """def ray_screen_point_arcsec_sitecoord(geo_cache, topo_cache, site, jd_tdb, basis):
    n, xhat, yhat = basis
    sun_geo = vec_at(geo_cache, "GEOCENTER_SUN", jd_tdb)
    sun_topo = site_sun_vector(topo_cache, site, jd_tdb)
    venus_topo = site_venus_vector(topo_cache, site, jd_tdb)
    obs_geo = sun_geo - sun_topo
    ray = venus_topo
    denom = float(np.dot(ray, n))
    if abs(denom) < 1e-14:
        raise RuntimeError("SITE_COORD ray nearly parallel to solar screen.")
    tau = float(np.dot(sun_geo - obs_geo, n) / denom)
    hit = obs_geo + tau * ray
    screen_vec = hit - sun_geo
    x = math.atan2(float(np.dot(screen_vec, xhat)), IAU1976_AU_KM) * ARCSEC_PER_RAD
    y = math.atan2(float(np.dot(screen_vec, yhat)), IAU1976_AU_KM) * ARCSEC_PER_RAD
    return np.array([x, y], dtype=float)""",
        "solar-screen normalization",
    )

    text = replace_exact(
        text,
        """def sun_radius_arcsec(geo_cache, jd_tdb):
    es = norm(vec_at(geo_cache, "GEOCENTER_SUN", jd_tdb))
    return math.atan2(SUN_RADIUS_KM, es) * ARCSEC_PER_RAD""",
        """def sun_radius_arcsec(geo_cache, jd_tdb):
    return math.atan2(SUN_RADIUS_KM, IAU1976_AU_KM) * ARCSEC_PER_RAD""",
        "solar-limb normalization",
    )

    text = replace_exact(
        text,
        """def site_track(geo_cache, topo_cache, site, contacts, closest_jd, basis):
    jds = topo_cache["jd_tdb"]
    mask = (jds >= contacts["C1"]) & (jds <= contacts["C4"])
    use_jds = sorted(set([contacts["C1"], contacts["C2"], closest_jd, contacts["C3"], contacts["C4"]] + list(jds[mask])))
    pts = np.array([ray_screen_point_arcsec_sitecoord(geo_cache, topo_cache, site, jd, basis) for jd in use_jds], dtype=float)
    mu, direction = pca_direction(pts)
    event_jds = {"C1": contacts["C1"], "C2": contacts["C2"], "CA": closest_jd, "C3": contacts["C3"], "C4": contacts["C4"]}
    event_pts = {name: ray_screen_point_arcsec_sitecoord(geo_cache, topo_cache, site, jd, basis) for name, jd in event_jds.items()}
    event_radii = {name: angular_radii_arcsec(topo_cache, site, jd)[1] for name, jd in event_jds.items()}
    return {"site": site, "jds": np.array(use_jds, dtype=float), "pts": pts, "mu": mu, "direction": direction, "event_jds": event_jds, "event_pts": event_pts, "event_radii": event_radii, "closest_jd": closest_jd, "closest_utc": utc_at(closest_jd), "track_angle_deg": math.degrees(math.atan2(direction[1], direction[0]))}""",
        """def site_track(geo_cache, topo_cache, site, contacts, closest_jd, basis):
    jds = topo_cache["jd_tdb"]
    mask = (jds >= contacts["C1"]) & (jds <= contacts["C4"])
    use_jds = sorted(set([contacts["C1"], contacts["C2"], closest_jd, contacts["C3"], contacts["C4"]] + list(jds[mask])))
    pts = np.array([ray_screen_point_arcsec_sitecoord(geo_cache, topo_cache, site, jd, basis) for jd in use_jds], dtype=float)
    mu, direction = pca_direction(pts)
    event_jds = {"C1": contacts["C1"], "C2": contacts["C2"], "CA": closest_jd, "C3": contacts["C3"], "C4": contacts["C4"]}
    event_pts = {name: ray_screen_point_arcsec_sitecoord(geo_cache, topo_cache, site, jd, basis) for name, jd in event_jds.items()}
    event_radii = {}
    for name, jd in event_jds.items():
        actual_earth_sun_km = norm(vec_at(geo_cache, "GEOCENTER_SUN", jd))
        venus_radius_actual = angular_radii_arcsec(topo_cache, site, jd)[1]
        event_radii[name] = venus_radius_actual * actual_earth_sun_km / IAU1976_AU_KM
    return {"site": site, "jds": np.array(use_jds, dtype=float), "pts": pts, "mu": mu, "direction": direction, "event_jds": event_jds, "event_pts": event_pts, "event_radii": event_radii, "closest_jd": closest_jd, "closest_utc": utc_at(closest_jd), "track_angle_deg": math.degrees(math.atan2(direction[1], direction[0]))}""",
        "Venus-disk normalization",
    )

    text = replace_exact(
        text,
        """    es, ev, vs = distances_at(geo_cache, screen_jd)
    abp_km = math.tan(abp_arcsec / ARCSEC_PER_RAD) * es
    ab_km = abp_km * ev / vs
    ab_arcsec = math.atan2(ab_km, es) * ARCSEC_PER_RAD
    halley_ratio = abp_km / ab_km
    raw_phi = rho_arcsec * (ev / vs) * (EARTH_RADIUS_KM / ab_km)
    pi_sun = raw_phi * (es / AU_KM)
    rho_scaled = rho_arcsec * EARTH_RADIUS_KM / ab_km
    return {"aprime": aprime, "bprime": bprime, "A_prime_B_prime_arcsec": abp_arcsec, "A_prime_B_prime_km": abp_km, "rho_arcsec": rho_arcsec, "rho_scaled_arcsec": rho_scaled, "AB_arcsec": ab_arcsec, "AB_km": ab_km, "halley_ratio": halley_ratio, "raw_phi_arcsec": raw_phi, "pi_sun_arcsec": pi_sun, "pi_sun_residual_arcsec": pi_sun - PI_SUN_REFERENCE_ARCSEC, "pi_sun_residual_percent": 100.0 * (pi_sun - PI_SUN_REFERENCE_ARCSEC) / PI_SUN_REFERENCE_ARCSEC, "pi_sun_reference_arcsec": PI_SUN_REFERENCE_ARCSEC, "D_ES_AU": es / AU_KM, "D_EV_D_VS": ev / vs, "D_VS_D_EV": vs / ev, "D_ES_source": "|GEOCENTER_SUN| / AU_KM"}""",
        """    _jpl_es, ev, vs = distances_at(geo_cache, screen_jd)
    theta_rad = abp_arcsec / ARCSEC_PER_RAD
    abp_km = math.tan(theta_rad) * IAU1976_AU_KM
    ab_km = abp_km * ev / vs
    ab_arcsec = math.atan2(ab_km, IAU1976_AU_KM) * ARCSEC_PER_RAD
    halley_ratio = abp_km / ab_km
    raw_phi = rho_arcsec * (ev / vs) * (EARTH_RADIUS_KM / ab_km)
    chord_normal_factor = abp_arcsec / rho_arcsec
    tangent_factor = math.tan(theta_rad) / theta_rad
    horizontal_ratio = EARTH_RADIUS_KM / IAU1976_AU_KM
    exact_arcsine_factor = math.asin(horizontal_ratio) / horizontal_ratio
    pi_sun = raw_phi * chord_normal_factor * tangent_factor * exact_arcsine_factor
    rho_scaled = rho_arcsec * EARTH_RADIUS_KM / ab_km
    return {"aprime": aprime, "bprime": bprime, "A_prime_B_prime_arcsec": abp_arcsec, "A_prime_B_prime_km": abp_km, "rho_arcsec": rho_arcsec, "rho_scaled_arcsec": rho_scaled, "AB_arcsec": ab_arcsec, "AB_km": ab_km, "halley_ratio": halley_ratio, "raw_phi_arcsec": raw_phi, "pi_sun_arcsec": pi_sun, "pi_sun_residual_arcsec": pi_sun - PI_SUN_REFERENCE_ARCSEC, "pi_sun_residual_percent": 100.0 * (pi_sun - PI_SUN_REFERENCE_ARCSEC) / PI_SUN_REFERENCE_ARCSEC, "pi_sun_reference_arcsec": PI_SUN_REFERENCE_ARCSEC, "D_ES_AU": 1.0, "D_EV_D_VS": ev / vs, "D_VS_D_EV": vs / ev, "D_ES_source": "IAU 1976 c tau_A / c tau_A"}""",
        "IAU-1976 parallax geometry",
    )

    text = replace_exact(
        text,
        '    if quantity in ["Computed π⊙", "Reference π⊙", "Residual π⊙", "Raw φ"]:\n        return 9',
        '    if quantity in ["Computed π⊙", "Reference π⊙", "Residual π⊙", "Raw φ"]:\n        return 10',
        "ten-decimal display",
    )

    text = replace_exact(
        text,
        '        dy = 15.0 if site_label.startswith("Vardo") else -15.0\n        add_label(ax, ca, f"{track[\'site\'][\'short\']} CA", 18.0, dy, color)',
        '        dy = 44.0 if site_label.startswith("Vardo") else -44.0\n        add_label(ax, ca, f"{track[\'site\'][\'short\']} CA", 18.0, dy, color)',
        "closest-approach labels",
    )

    text = replace_exact(
        text,
        'A′B′ = solar-screen chord; AB = projected baseline; D ES is JPL |Sun|/AU.',
        'A′B′ = solar-screen chord; AB = projected baseline; D ES = IAU 1976 cτA.',
        "plot table note",
    )

    text = replace_exact(
        text,
        'print("SOURCE DATA: JPL Horizons geocenter vectors and JPL Horizons SITE_COORD topocentric vectors")',
        'print("SOURCE DATA: JPL vector directions; IAU 1976 R_E and c tau_A used for all plotted scales and parallax distances")',
        "source statement",
    )

    text = replace_exact(text, "# IERS-0012N", "# V0016", "final marker")
    return text


def main() -> None:
    source = fetch_source()
    engine = build_engine(source)
    PATCHED.write_text(engine, encoding="utf-8")
    py_compile.compile(str(PATCHED), doraise=True)
    runpy.run_path(str(PATCHED), run_name="__main__")


if __name__ == "__main__":
    main()
# V0016
