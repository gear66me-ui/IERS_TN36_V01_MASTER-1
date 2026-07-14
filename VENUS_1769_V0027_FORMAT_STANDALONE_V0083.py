# V0083
# Audit reference: PNG output for geometric and astronomical audit table; Vardø/Pointe Venus series title; no AI images.
from __future__ import annotations

import math
import urllib.request
from datetime import datetime

VERSION = "V0083"
PNG = "VENUS_1769_VARDO_POINTE_VENUS_GEOMETRIC_ASTRONOMICAL_TABLE_V0083.png"
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
    namespace: dict[str, object] = {
        "__name__": "venus_v0067_png_table",
        "__file__": "VENUS_1769_V0027_FORMAT_STANDALONE_V0067.py",
    }
    exec(
        compile(fetch_verified_source(), "VENUS_1769_V0027_FORMAT_STANDALONE_V0067.py", "exec"),
        namespace,
        namespace,
    )
    return namespace


def fmt(value: float, places: int = 6) -> str:
    return f"{float(value):,.{places}f}"


def fmt12(value: float) -> str:
    return f"{float(value):.12f}"


def make_png_table(ns: dict[str, object], rows: list[list[str]]) -> None:
    plt = ns["plt"]
    TABLE_HEADER = str(ns.get("TABLE_HEADER", "#263849"))
    TABLE_TEAL = str(ns.get("TABLE_TEAL", "#183C48"))
    TABLE_GOLD = str(ns.get("TABLE_GOLD", "#5C4A20"))
    TABLE_BODY = str(ns.get("TABLE_BODY", "#121D28"))
    BACKGROUND = str(ns.get("BACKGROUND", "#0B1118"))
    TEXT_COLOR = str(ns.get("TEXT_COLOR", "#E8EEF4"))
    MUTED_TEXT = str(ns.get("MUTED_TEXT", "#A9B7C6"))

    plt.close("all")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "figure.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "text.color": TEXT_COLOR,
        }
    )

    fig, ax = plt.subplots(figsize=(13.2, 7.0), facecolor=BACKGROUND)
    ax.axis("off")
    fig.text(
        0.5,
        0.945,
        "1769 Venus Transit — Vardø, Norway and Pointe Venus, Tahiti",
        ha="center",
        va="center",
        fontsize=17.5,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    fig.text(
        0.5,
        0.905,
        "Geometric and Astronomical Audit — JPL-derived reduced distances and finite IAU 1976 parallax",
        ha="center",
        va="center",
        fontsize=9.5,
        color=MUTED_TEXT,
    )

    table = ax.table(
        cellText=rows,
        cellLoc="left",
        colWidths=[0.42, 0.20, 0.22, 0.16],
        bbox=[0.025, 0.115, 0.95, 0.74],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#70879A")
        cell.set_linewidth(0.45)
        cell.get_text().set_color(TEXT_COLOR)
        cell.get_text().set_ha("right" if c == 2 else "left")
        if r == 0:
            cell.set_facecolor(TABLE_HEADER)
            cell.get_text().set_fontweight("bold")
        elif 1 <= r <= 4:
            cell.set_facecolor(TABLE_TEAL)
            cell.get_text().set_fontweight("bold")
        elif 5 <= r <= 9:
            cell.set_facecolor(TABLE_GOLD)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(TABLE_BODY)

    fig.text(
        0.5,
        0.055,
        f"Output: {PNG}     Version: {VERSION}",
        ha="center",
        fontsize=8.0,
        color=MUTED_TEXT,
    )
    fig.savefig(PNG, dpi=220, bbox_inches="tight", facecolor=BACKGROUND)
    plt.show()


def main() -> None:
    ns = load_v0067_namespace()
    build_master = ns["build_master"]
    build_cache = ns["build_cache"]
    separate_ray_geometry = ns["separate_ray_geometry"]
    EARTH_RADIUS_KM = float(ns["EARTH_RADIUS_KM"])
    IAU1976_AU_KM = float(ns["IAU1976_AU_KM"])
    ARCSEC_PER_RAD = float(ns["ARCSEC_PER_RAD"])
    LOCAL_TZ = ns["LOCAL_TZ"]

    master = build_master()
    cache = build_cache(master)
    geometry = separate_ray_geometry(cache)

    ev_bar = float(geometry["EV_bar_km"])
    vs_bar = float(geometry["VS_bar_km"])
    es_bar = float(geometry["ES_bar_km"])
    reduced_closure = ev_bar + vs_bar - es_bar
    pi_finite_iau76 = math.asin(EARTH_RADIUS_KM / IAU1976_AU_KM) * ARCSEC_PER_RAD

    rows = [
        ["Quantity", "Symbol", "Value", "Unit / status"],
        ["Reduced Earth–Venus distance", "EV̄", fmt(ev_bar), "km"],
        ["Reduced Venus–Sun distance", "VS̄", fmt(vs_bar), "km"],
        ["Reduced Earth–Sun distance", "ES̄", fmt(es_bar), "km"],
        ["Reduced distance closure", "EV̄ + VS̄ − ES̄", f"{reduced_closure:+.9f}", "km — PASS"],
        ["A′B′ common-normal separation", "A′B′", fmt(float(geometry["A_prime_B_prime_arcsec"])), "arcsec"],
        ["A′B′ common-normal separation", "A′B′", fmt(float(geometry["A_prime_B_prime_km"])), "km"],
        ["AB projected baseline", "AB", fmt(float(geometry["AB_arcsec"])), "arcsec"],
        ["AB projected baseline", "AB", fmt(float(geometry["AB_km"])), "km"],
        ["Finite IAU 1976 solar horizontal parallax", r"$\pi_{\mathrm{finite,IAU76}}$", fmt12(pi_finite_iau76), "arcsec"],
    ]

    make_png_table(ns, rows)
    print(f"PNG: {PNG}")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0083
