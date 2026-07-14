# V0072
# Audit reference: Compact Halley-ratio audit table with reduced/raw closures and IAU-normalized Halley parallax; no plot.
from __future__ import annotations

import math
import subprocess
import sys
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

VERSION = "V0072"
LOCAL_TZ = ZoneInfo("America/Bogota")
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "VENUS_1769_V0027_FORMAT_STANDALONE_V0067.py"
)


def require(import_name: str, package_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", package_name])


for _import_name, _package_name in (("pandas", "pandas"), ("IPython", "ipython")):
    require(_import_name, _package_name)

import pandas as pd
from IPython.display import HTML, display


def load_base_namespace() -> dict[str, object]:
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": VERSION})
    with urllib.request.urlopen(request, timeout=60) as response:
        source = response.read().decode("utf-8")
    if "# V0067" not in source or "def separate_ray_geometry(" not in source:
        raise RuntimeError("Verified V0067 source was not loaded correctly.")
    namespace = {
        "__name__": "venus_v0067_library",
        "__file__": "VENUS_1769_V0027_FORMAT_STANDALONE_V0067.py",
    }
    exec(compile(source, "VENUS_1769_V0027_FORMAT_STANDALONE_V0067.py", "exec"), namespace, namespace)
    required = [
        "build_master",
        "build_cache",
        "separate_ray_geometry",
        "vector_at",
        "norm",
        "EARTH_RADIUS_KM",
        "IAU1976_AU_KM",
        "ARCSEC_PER_RAD",
    ]
    missing = [name for name in required if name not in namespace]
    if missing:
        raise RuntimeError(f"V0067 source is missing required definitions: {missing}")
    return namespace


def fmt(value: float, decimals: int = 6) -> str:
    return f"{float(value):,.{decimals}f}"


def make_table_html(frame: pd.DataFrame) -> str:
    colors = {
        "Earth": "#1d4ed8",
        "Venus": "#7c3aed",
        "Sun": "#b45309",
        "A′B′": "#0f766e",
        "AB": "#92400e",
        "Ratio": "#374151",
        "Closure": "#065f46",
        "Parallax": "#4c1d95",
    }
    title = (
        "<div style='font-family:DejaVu Sans,Arial,sans-serif;max-width:1220px;'>"
        "<div style='font-size:20px;font-weight:800;margin:8px 0 10px 0;color:#f8fafc;"
        "background:#020617;padding:10px 12px;border-radius:8px;'>"
        "1769 Venus Transit — Halley Ratio / IAU-Normalized Audit Table — V0072</div>"
    )
    table = [
        "<table style='border-collapse:collapse;width:100%;background:#020617;color:#e5e7eb;"
        "font-family:DejaVu Sans,Arial,sans-serif;font-size:13px;'>",
        "<thead><tr>"
        "<th style='background:#1f2937;color:white;padding:8px;border:1px solid #64748b;text-align:left;'>Group</th>"
        "<th style='background:#1f2937;color:white;padding:8px;border:1px solid #64748b;text-align:left;'>Quantity</th>"
        "<th style='background:#1f2937;color:white;padding:8px;border:1px solid #64748b;text-align:left;'>Symbol</th>"
        "<th style='background:#1f2937;color:white;padding:8px;border:1px solid #64748b;text-align:right;'>Value</th>"
        "<th style='background:#1f2937;color:white;padding:8px;border:1px solid #64748b;text-align:left;'>Unit</th>"
        "<th style='background:#1f2937;color:white;padding:8px;border:1px solid #64748b;text-align:left;'>Trace</th>"
        "</tr></thead><tbody>",
    ]
    for _, row in frame.iterrows():
        color = colors[str(row["group"])]
        table.append(
            f"<tr style='background:{color}33;font-weight:650;'>"
            f"<td style='background:{color};color:white;padding:7px 8px;border:1px solid #475569;font-weight:850;'>{row['group']}</td>"
            f"<td style='padding:7px 8px;border:1px solid #475569;'>{row['quantity']}</td>"
            f"<td style='padding:7px 8px;border:1px solid #475569;font-weight:850;'>{row['symbol']}</td>"
            f"<td style='padding:7px 8px;border:1px solid #475569;text-align:right;font-variant-numeric:tabular-nums;'>{row['value']}</td>"
            f"<td style='padding:7px 8px;border:1px solid #475569;'>{row['unit']}</td>"
            f"<td style='padding:7px 8px;border:1px solid #475569;'>{row['trace']}</td>"
            "</tr>"
        )
    table.append("</tbody></table>")
    table.append(
        "<div style='font-family:DejaVu Sans,Arial,sans-serif;font-size:11px;color:#cbd5e1;"
        "background:#020617;padding:8px 12px;border-radius:0 0 8px 8px;'>"
        "Fresh JPL Horizons vectors through verified V0067 geometry. Reduced closure uses ES-axis projections. "
        "Raw closure uses full 3D magnitudes. No plot is generated.</div></div>"
    )
    return title + "\n".join(table)


def main() -> None:
    ns = load_base_namespace()
    master = ns["build_master"]()
    cache = ns["build_cache"](master)
    geometry = ns["separate_ray_geometry"](cache)

    arcsec_per_rad = float(ns["ARCSEC_PER_RAD"])
    earth_radius_km = float(ns["EARTH_RADIUS_KM"])
    iau1976_au_km = float(ns["IAU1976_AU_KM"])
    vector_at = ns["vector_at"]
    norm = ns["norm"]

    jd = float(geometry["jd_tdb"])
    rho_arcsec = float(geometry["A_prime_B_prime_arcsec"])
    rho_km = float(geometry["A_prime_B_prime_km"])
    beta_arcsec = float(geometry["AB_arcsec"])
    beta_km = float(geometry["AB_km"])
    ev_bar_km = float(geometry["EV_bar_km"])
    vs_bar_km = float(geometry["VS_bar_km"])
    es_bar_km = float(geometry["ES_bar_km"])

    geo_sun = vector_at(cache, "GEOCENTER_SUN", jd)
    geo_venus = vector_at(cache, "GEOCENTER_VENUS", jd)
    ev_raw_km = float(norm(geo_venus))
    vs_raw_km = float(norm(geo_sun - geo_venus))
    es_raw_km = float(norm(geo_sun))

    delta_parallel_km = ev_bar_km + vs_bar_km - es_bar_km
    delta_3d_km = ev_raw_km + vs_raw_km - es_raw_km

    kappa = ev_bar_km / vs_bar_km
    lambda_jpl = beta_arcsec / rho_arcsec
    beta_halley_arcsec = rho_arcsec * kappa
    es_halley_km = beta_km / (beta_halley_arcsec / arcsec_per_rad)
    n_halley_to_iau76 = es_halley_km / iau1976_au_km

    pi0_ab_raw = beta_arcsec * earth_radius_km / beta_km
    pi0_halley_raw = beta_halley_arcsec * earth_radius_km / beta_km
    pi0_halley_iau76 = math.asin(earth_radius_km / iau1976_au_km) * arcsec_per_rad
    pi0_jpl_es = math.asin(earth_radius_km / es_bar_km) * arcsec_per_rad
    pi0_iau76 = math.asin(earth_radius_km / iau1976_au_km) * arcsec_per_rad
    halley_minus_iau = pi0_halley_iau76 - pi0_iau76

    rows = [
        ["Earth", "Earth equatorial radius", "R⊕", fmt(earth_radius_km), "km", "IAU 1976 reduction constant"],
        ["Venus", "Reduced Earth–Venus distance", "EV̄", fmt(ev_bar_km), "km", "JPL vector projected on ES axis"],
        ["Sun", "Reduced Venus–Sun distance", "VS̄", fmt(vs_bar_km), "km", "JPL vector projected on ES axis"],
        ["Sun", "Reduced Earth–Sun distance", "ES̄", fmt(es_bar_km), "km", "JPL vector projected on ES axis"],
        ["Closure", "Reduced distance closure", "Δ∥ = EV̄ + VS̄ − ES̄", fmt(delta_parallel_km, 9), "km", "must close by same-axis construction"],
        ["Venus", "Raw Earth–Venus distance", "|EV|", fmt(ev_raw_km), "km", "full 3D JPL vector magnitude"],
        ["Sun", "Raw Venus–Sun distance", "|VS|", fmt(vs_raw_km), "km", "full 3D JPL vector magnitude"],
        ["Sun", "Raw Earth–Sun distance", "|ES|", fmt(es_raw_km), "km", "full 3D JPL vector magnitude"],
        ["Closure", "Raw 3D non-collinearity", "Δ₃D = |EV| + |VS| − |ES|", fmt(delta_3d_km), "km", "expected hundreds-of-km geometric offset"],
        ["A′B′", "A′ normal coordinate", "A′", fmt(float(geometry["A_prime_bar_arcsec"])), "arcsec", "common-normal tangent-plane coordinate"],
        ["A′B′", "B′ normal coordinate", "B′", fmt(float(geometry["B_prime_bar_arcsec"])), "arcsec", "common-normal tangent-plane coordinate"],
        ["A′B′", "Venus-track separation", "ρ = B′ − A′", fmt(rho_arcsec), "arcsec", "same epoch, same normal; sign retained"],
        ["A′B′", "Venus-track separation", "ρ", fmt(rho_km), "km", "ρ × ES̄ / 206264.806"],
        ["AB", "Projected observer baseline", "β", fmt(beta_arcsec), "arcsec", "JPL projected baseline angle"],
        ["AB", "Projected observer baseline", "AB", fmt(beta_km), "km", "station baseline projected on normal"],
        ["Ratio", "Distance ratio", "κ = EV̄ / VS̄", fmt(kappa, 12), "dimensionless", "Halley reduced-distance ratio"],
        ["Ratio", "JPL angular ratio", "λ = β / ρ", fmt(lambda_jpl, 12), "dimensionless", "direct JPL angle ratio"],
        ["Ratio", "Halley angular baseline", "βH = ρ κ", fmt(beta_halley_arcsec), "arcsec", "baseline angle from Halley ratio path"],
        ["Ratio", "Halley implied Earth–Sun distance", "ESH", fmt(es_halley_km), "km", "AB / (βH / 206264.806)"],
        ["Ratio", "Halley-to-IAU normalization", "NH→IAU76 = ESH / AU₁₉₇₆", fmt(n_halley_to_iau76, 12), "dimensionless", "normalizes ratio-derived parallax to IAU AU"],
        ["Parallax", "AB/JPL parallax", "π₀,AB", fmt(pi0_ab_raw, 12), "arcsec", "β × R⊕ / AB"],
        ["Parallax", "Halley ratio parallax, raw", "π₀,H", fmt(pi0_halley_raw, 12), "arcsec", "βH × R⊕ / AB"],
        ["Parallax", "Halley ratio parallax, IAU-normalized", "π₀,H→IAU76", fmt(pi0_halley_iau76, 12), "arcsec", "π₀,H normalized to AU₁₉₇₆"],
        ["Parallax", "JPL Earth–Sun parallax", "π₀,JPL", fmt(pi0_jpl_es, 12), "arcsec", "asin(R⊕ / ES̄)"],
        ["Parallax", "IAU 1976 parallax", "π₀,IAU76", fmt(pi0_iau76, 12), "arcsec", "asin(R⊕ / AU₁₉₇₆)"],
        ["Closure", "Normalized Halley residual", "π₀,H→IAU76 − π₀,IAU76", fmt(halley_minus_iau, 12), "arcsec", "should be 0 after IAU normalization"],
    ]
    frame = pd.DataFrame(rows, columns=["group", "quantity", "symbol", "value", "unit", "trace"])
    display(HTML(make_table_html(frame)))
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0072
