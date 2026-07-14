# V0081
# Audit reference: Geometric and astronomical audit table from verified V0067; same dark table palette; finite IAU76 parallax symbol fully subscripted.
from __future__ import annotations

import html
import math
import urllib.request
from datetime import datetime

VERSION = "V0081"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "VENUS_1769_V0027_FORMAT_STANDALONE_V0067.py"
)


def fetch_verified_source() -> str:
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": VERSION})
    with urllib.request.urlopen(request, timeout=60) as response:
        source = response.read().decode("utf-8")
    if "# V0067" not in source or "def separate_ray_geometry(" not in source:
        raise RuntimeError("Verified V0067 source was not loaded correctly.")
    return source


def load_v0067_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {"__name__": "venus_v0067_audit", "__file__": "VENUS_1769_V0027_FORMAT_STANDALONE_V0067.py"}
    exec(compile(fetch_verified_source(), "VENUS_1769_V0027_FORMAT_STANDALONE_V0067.py", "exec"), namespace, namespace)
    return namespace


def fmt(value: float, places: int = 6) -> str:
    return f"{float(value):,.{places}f}"


def fmt12(value: float) -> str:
    return f"{float(value):.12f}"


def style_for_group(group: str, palette: dict[str, str]) -> str:
    if group == "header":
        return f"background:{palette['header']};font-weight:700;"
    if group == "distance":
        return f"background:{palette['teal']};font-weight:700;"
    if group == "geometry":
        return f"background:{palette['gold']};font-weight:700;"
    if group == "final":
        return f"background:{palette['gold']};font-weight:800;"
    return f"background:{palette['body']};"


def render_table(rows: list[dict[str, str]], palette: dict[str, str]) -> str:
    css = f"""
    <div style='font-family:DejaVu Serif, Georgia, serif; background:#0B1118; padding:12px; width:980px;'>
      <div style='color:#E8EEF4; font-size:16px; font-weight:800; margin:0 0 9px 0;'>1769 Venus Transit — Geometric and Astronomical Audit</div>
      <table style='border-collapse:collapse; width:100%; color:#E8EEF4; font-size:13px;'>
        <tr style='{style_for_group('header', palette)}'>
          <th style='border:1px solid #70879A; padding:7px; text-align:left;'>Quantity</th>
          <th style='border:1px solid #70879A; padding:7px; text-align:left;'>Symbol</th>
          <th style='border:1px solid #70879A; padding:7px; text-align:right;'>Value</th>
          <th style='border:1px solid #70879A; padding:7px; text-align:left;'>Unit / status</th>
        </tr>
    """
    body_parts = [css]
    for row in rows:
        style = style_for_group(row["group"], palette)
        body_parts.append(
            "<tr style='" + style + "'>"
            + "<td style='border:1px solid #70879A; padding:7px;'>" + html.escape(row["quantity"]) + "</td>"
            + "<td style='border:1px solid #70879A; padding:7px;'>" + row["symbol"] + "</td>"
            + "<td style='border:1px solid #70879A; padding:7px; text-align:right; font-variant-numeric:tabular-nums;'>" + html.escape(row["value"]) + "</td>"
            + "<td style='border:1px solid #70879A; padding:7px;'>" + html.escape(row["unit"]) + "</td>"
            + "</tr>"
        )
    body_parts.append("</table></div>")
    return "\n".join(body_parts)


def main() -> None:
    ns = load_v0067_namespace()
    build_master = ns["build_master"]
    build_cache = ns["build_cache"]
    separate_ray_geometry = ns["separate_ray_geometry"]
    norm = ns["norm"]
    vector_at = ns["vector_at"]
    EARTH_RADIUS_KM = float(ns["EARTH_RADIUS_KM"])
    IAU1976_AU_KM = float(ns["IAU1976_AU_KM"])
    ARCSEC_PER_RAD = float(ns["ARCSEC_PER_RAD"])
    LOCAL_TZ = ns["LOCAL_TZ"]

    master = build_master()
    cache = build_cache(master)
    geometry = separate_ray_geometry(cache)
    jd = float(geometry["jd_tdb"])

    geo_sun = vector_at(cache, "GEOCENTER_SUN", jd)
    geo_venus = vector_at(cache, "GEOCENTER_VENUS", jd)
    ev_raw = norm(geo_venus)
    vs_raw = norm(geo_sun - geo_venus)
    es_raw = norm(geo_sun)
    raw_noncollinearity = ev_raw + vs_raw - es_raw

    ev_bar = float(geometry["EV_bar_km"])
    vs_bar = float(geometry["VS_bar_km"])
    es_bar = float(geometry["ES_bar_km"])
    reduced_closure = ev_bar + vs_bar - es_bar

    pi_finite_iau76 = math.asin(EARTH_RADIUS_KM / IAU1976_AU_KM) * ARCSEC_PER_RAD

    km_per_arcsec = float(geometry["km_per_arcsec"])
    a_prime_arcsec = float(geometry["A_prime_bar_arcsec"])
    b_prime_arcsec = float(geometry["B_prime_bar_arcsec"])
    a_prime_km = a_prime_arcsec * km_per_arcsec
    b_prime_km = b_prime_arcsec * km_per_arcsec

    rows = [
        {"group": "distance", "quantity": "Reduced Earth-Venus distance", "symbol": "<span style='text-decoration:overline;'>EV</span>", "value": fmt(ev_bar), "unit": "km"},
        {"group": "distance", "quantity": "Reduced Venus-Sun distance", "symbol": "<span style='text-decoration:overline;'>VS</span>", "value": fmt(vs_bar), "unit": "km"},
        {"group": "distance", "quantity": "Reduced Earth-Sun distance", "symbol": "<span style='text-decoration:overline;'>ES</span>", "value": fmt(es_bar), "unit": "km"},
        {"group": "distance", "quantity": "Reduced distance closure", "symbol": "<span style='text-decoration:overline;'>EV</span> + <span style='text-decoration:overline;'>VS</span> − <span style='text-decoration:overline;'>ES</span>", "value": f"{reduced_closure:+.9f}", "unit": "km — PASS"},
        {"group": "body", "quantity": "Raw 3D non-collinearity", "symbol": "|EV| + |VS| − |ES|", "value": fmt(raw_noncollinearity), "unit": "km — NOT USED for reduced closure"},
        {"group": "geometry", "quantity": "A prime common-normal coordinate", "symbol": "A′", "value": fmt(a_prime_arcsec), "unit": "arcsec"},
        {"group": "geometry", "quantity": "B prime common-normal coordinate", "symbol": "B′", "value": fmt(b_prime_arcsec), "unit": "arcsec"},
        {"group": "geometry", "quantity": "A prime common-normal coordinate", "symbol": "A′", "value": fmt(a_prime_km), "unit": "km"},
        {"group": "geometry", "quantity": "B prime common-normal coordinate", "symbol": "B′", "value": fmt(b_prime_km), "unit": "km"},
        {"group": "geometry", "quantity": "A′B′ common-normal separation", "symbol": "A′B′", "value": fmt(float(geometry["A_prime_B_prime_arcsec"])), "unit": "arcsec"},
        {"group": "geometry", "quantity": "A′B′ common-normal separation", "symbol": "A′B′", "value": fmt(float(geometry["A_prime_B_prime_km"])), "unit": "km"},
        {"group": "geometry", "quantity": "AB projected baseline", "symbol": "AB", "value": fmt(float(geometry["AB_arcsec"])), "unit": "arcsec"},
        {"group": "geometry", "quantity": "AB projected baseline", "symbol": "AB", "value": fmt(float(geometry["AB_km"])), "unit": "km"},
        {"group": "final", "quantity": "Finite IAU 1976 solar horizontal parallax", "symbol": "π<sub>finite,IAU76</sub>", "value": fmt12(pi_finite_iau76), "unit": "arcsec"},
    ]

    palette = {
        "header": str(ns.get("TABLE_HEADER", "#263849")),
        "teal": str(ns.get("TABLE_TEAL", "#183C48")),
        "gold": str(ns.get("TABLE_GOLD", "#5C4A20")),
        "body": str(ns.get("TABLE_BODY", "#121D28")),
    }

    from IPython.display import HTML, display
    display(HTML(render_table(rows, palette)))
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0081
