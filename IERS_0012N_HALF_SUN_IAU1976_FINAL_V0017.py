# V0017
# Audit reference: Corrected IERS-0012N half-Sun track plot with IAU-1976 Earth radius and c tau_A used for every plotted angular scale and parallax distance.
from __future__ import annotations

import py_compile
import re
import runpy
from pathlib import Path
from urllib.request import Request, urlopen

VERSION = "V0017"
PROGRAM = "IERS_0012N_HALF_SUN_IAU1976_FINAL_V0017.py"
ROOT = Path("/content")
SOURCE_PATH = ROOT / "IERS_0012N_SOURCE_V0017.py"
ENGINE_PATH = ROOT / "IERS_0012N_HALF_SUN_IAU1976_ENGINE_V0017.py"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "IERS_0012N_VARDO_POINT_VENUS_ENGINEERING_TRACK_PLOT_PI_SUN.py?v=17"
)


def download_source() -> str:
    request = Request(SOURCE_URL, headers={"User-Agent": PROGRAM})
    with urlopen(request, timeout=180) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError("The IERS-0012N source download was empty.")
    SOURCE_PATH.write_bytes(payload)
    return payload.decode("utf-8")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"V0017 patch failed for {label}: expected one match, found {count}."
        )
    return text.replace(old, new, 1)


def replace_regex_once(
    text: str,
    pattern: str,
    replacement: str,
    label: str,
) -> str:
    updated, count = re.subn(
        pattern,
        replacement,
        text,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )
    if count != 1:
        raise RuntimeError(
            f"V0017 regex patch failed for {label}: expected one match, found {count}."
        )
    return updated


def build_engine(source: str) -> str:
    text = source

    text = replace_once(
        text,
        "# IERS-0012N\n# Audit reference: GitHubDelivery@IERS-0012N; 1769 Vardo-Point Venus engineering half-Sun plot using JPL Horizons SITE_COORD vectors.",
        "# V0017\n# Audit reference: Exact 12N half-Sun track plot with all angular scales normalized to IAU-1976 c tau_A.",
        "header",
    )
    text = replace_once(
        text,
        'VERSION = "IERS-0012N"',
        'VERSION = "V0017"',
        "version",
    )
    text = replace_once(
        text,
        'PROGRAM_NAME = "IERS_0012N_VARDO_POINT_VENUS_ENGINEERING_TRACK_PLOT_PI_SUN.py"',
        'PROGRAM_NAME = "IERS_0012N_HALF_SUN_IAU1976_ENGINE_V0017.py"',
        "program name",
    )

    text = replace_regex_once(
        text,
        r"AU_KM = 149_597_870\.7\nARCSEC_PER_RAD = 206_264\.80624709636\nEARTH_RADIUS_KM = 6_378\.137\nSUN_RADIUS_KM = 695_700\.0\nVENUS_RADIUS_KM = 6_051\.8\nPI_SUN_REFERENCE_ARCSEC = 8\.794148",
        "AU_KM = 149_597_870.7\n"
        "ARCSEC_PER_RAD = 206_264.80624709636\n"
        "EARTH_RADIUS_KM = 6_378.140\n"
        "C_KM_S = 299_792.458\n"
        "TAU_A_S = 499.004782\n"
        "IAU1976_AU_KM = C_KM_S * TAU_A_S\n"
        "SUN_RADIUS_KM = 695_700.0\n"
        "VENUS_RADIUS_KM = 6_051.8\n"
        "PI_SUN_REFERENCE_ARCSEC = math.asin(EARTH_RADIUS_KM / IAU1976_AU_KM) * ARCSEC_PER_RAD",
        "IAU-1976 constants",
    )

    text = replace_regex_once(
        text,
        r"def ray_screen_point_arcsec_sitecoord\(geo_cache, topo_cache, site, jd_tdb, basis\):\n.*?\n    return np\.array\(\[x, y\], dtype=float\)\n",
        '''def ray_screen_point_arcsec_sitecoord(geo_cache, topo_cache, site, jd_tdb, basis):
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
    return np.array([x, y], dtype=float)
''',
        "solar-screen coordinates",
    )

    text = replace_regex_once(
        text,
        r"def sun_radius_arcsec\(geo_cache, jd_tdb\):\n.*?\n    return math\.atan2\(SUN_RADIUS_KM, es\) \* ARCSEC_PER_RAD\n",
        '''def sun_radius_arcsec(geo_cache, jd_tdb):
    return math.atan2(SUN_RADIUS_KM, IAU1976_AU_KM) * ARCSEC_PER_RAD
''',
        "half-Sun limb",
    )

    text = replace_regex_once(
        text,
        r"def site_track\(geo_cache, topo_cache, site, contacts, closest_jd, basis\):\n.*?\n    return \{\"site\": site, .*?\}\n",
        '''def site_track(geo_cache, topo_cache, site, contacts, closest_jd, basis):
    jds = topo_cache["jd_tdb"]
    mask = (jds >= contacts["C1"]) & (jds <= contacts["C4"])
    use_jds = sorted(set([contacts["C1"], contacts["C2"], closest_jd, contacts["C3"], contacts["C4"]] + list(jds[mask])))
    pts = np.array([ray_screen_point_arcsec_sitecoord(geo_cache, topo_cache, site, jd, basis) for jd in use_jds], dtype=float)
    mu, direction = pca_direction(pts)
    event_jds = {"C1": contacts["C1"], "C2": contacts["C2"], "CA": closest_jd, "C3": contacts["C3"], "C4": contacts["C4"]}
    event_pts = {name: ray_screen_point_arcsec_sitecoord(geo_cache, topo_cache, site, jd, basis) for name, jd in event_jds.items()}
    event_radii = {}
    for name, jd in event_jds.items():
        actual_es_km = norm(vec_at(geo_cache, "GEOCENTER_SUN", jd))
        actual_venus_radius = angular_radii_arcsec(topo_cache, site, jd)[1]
        event_radii[name] = actual_venus_radius * actual_es_km / IAU1976_AU_KM
    return {"site": site, "jds": np.array(use_jds, dtype=float), "pts": pts, "mu": mu, "direction": direction, "event_jds": event_jds, "event_pts": event_pts, "event_radii": event_radii, "closest_jd": closest_jd, "closest_utc": utc_at(closest_jd), "track_angle_deg": math.degrees(math.atan2(direction[1], direction[0]))}
''',
        "track and Venus-disk scaling",
    )

    text = replace_regex_once(
        text,
        r"def compute_parallax_geometry\(geo_cache, track_a, track_b, screen_jd\):\n.*?\n    return \{\"aprime\": aprime, .*?\}\n",
        '''def compute_parallax_geometry(geo_cache, track_a, track_b, screen_jd):
    tangent = unit(track_a["direction"] + track_b["direction"])
    if tangent[0] < 0:
        tangent = -tangent
    normal = np.array([-tangent[1], tangent[0]])
    mid = 0.5 * (track_a["mu"] + track_b["mu"])
    aprime = line_intersection(track_a["mu"], track_a["direction"], mid, normal)
    bprime = line_intersection(track_b["mu"], track_b["direction"], mid, normal)
    abp_vec = bprime - aprime
    abp_arcsec = float(np.sqrt(np.sum(abp_vec * abp_vec)))
    rho_arcsec = abs(float(np.dot(abp_vec, normal)))
    _jpl_es, ev, vs = distances_at(geo_cache, screen_jd)
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
    return {"aprime": aprime, "bprime": bprime, "A_prime_B_prime_arcsec": abp_arcsec, "A_prime_B_prime_km": abp_km, "rho_arcsec": rho_arcsec, "rho_scaled_arcsec": rho_scaled, "AB_arcsec": ab_arcsec, "AB_km": ab_km, "halley_ratio": halley_ratio, "raw_phi_arcsec": raw_phi, "pi_sun_arcsec": pi_sun, "pi_sun_residual_arcsec": pi_sun - PI_SUN_REFERENCE_ARCSEC, "pi_sun_residual_percent": 100.0 * (pi_sun - PI_SUN_REFERENCE_ARCSEC) / PI_SUN_REFERENCE_ARCSEC, "pi_sun_reference_arcsec": PI_SUN_REFERENCE_ARCSEC, "D_ES_AU": 1.0, "D_EV_D_VS": ev / vs, "D_VS_D_EV": vs / ev, "D_ES_source": "IAU 1976 c tau_A / c tau_A"}
''',
        "parallax geometry",
    )

    text = replace_once(
        text,
        '    if quantity in ["Computed π⊙", "Reference π⊙", "Residual π⊙", "Raw φ"]:\n        return 9',
        '    if quantity in ["Computed π⊙", "Reference π⊙", "Residual π⊙", "Raw φ"]:\n        return 10',
        "ten-decimal output",
    )

    text = replace_once(
        text,
        '        dy = 15.0 if site_label.startswith("Vardo") else -15.0',
        '        dy = 44.0 if site_label.startswith("Vardo") else -44.0',
        "closest-approach label offsets",
    )

    text = replace_once(
        text,
        "A′B′ = solar-screen chord; AB = projected baseline; D ES is JPL |Sun|/AU.",
        "A′B′ = solar-screen chord; AB = projected baseline; D ES = IAU 1976 cτA.",
        "plot table note",
    )

    text = replace_once(
        text,
        'print("SOURCE DATA: JPL Horizons geocenter vectors and JPL Horizons SITE_COORD topocentric vectors")',
        'print("SOURCE DATA: JPL-derived track directions; IAU 1976 R_E and c tau_A used for all plotted distances and normalization")',
        "source statement",
    )

    text = replace_once(
        text,
        "# IERS-0012N",
        "# V0017",
        "final marker",
    )

    return text


def main() -> None:
    source = download_source()
    engine = build_engine(source)
    ENGINE_PATH.write_text(engine, encoding="utf-8")
    py_compile.compile(str(ENGINE_PATH), doraise=True)
    runpy.run_path(str(ENGINE_PATH), run_name="__main__")


if __name__ == "__main__":
    main()
# V0017
