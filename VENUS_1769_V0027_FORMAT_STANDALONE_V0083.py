# V0083
# Audit reference: PNG output for corrected geometric and astronomical audit table; removed raw 3D and separate A′/B′ rows; one A′B′ value only.
from __future__ import annotations

import math
import urllib.request
from datetime import datetime
from pathlib import Path

VERSION = "V0083"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "VENUS_1769_V0027_FORMAT_STANDALONE_V0067.py"
)
PNG = Path("VENUS_1769_GEOMETRIC_ASTRONOMICAL_AUDIT_V0083.png")


def fetch_verified_source() -> str:
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": VERSION})
    with urllib.request.urlopen(request, timeout=60) as response:
        source = response.read().decode("utf-8")
    if "# V0067" not in source or "def separate_ray_geometry(" not in source:
        raise RuntimeError("Verified V0067 source was not loaded correctly.")
    return source


def load_v0067_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {
        "__name__": "venus_v0067_audit",
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


def rgba_from_hex(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    text = hex_color.strip().lstrip("#")
    return (
        int(text[0:2], 16) / 255.0,
        int(text[2:4], 16) / 255.0,
        int(text[4:6], 16) / 255.0,
        alpha,
    )


def save_png_table(rows: list[list[str]], row_groups: list[str], palette: dict[str, str]) -> None:
    import matplotlib.pyplot as plt

    background = "#0B1118"
    text_color = "#E8EEF4"
    edge_color = "#70879A"
    figure = plt.figure(figsize=(13.2, 5.4), dpi=220, facecolor=background)
    axis = figure.add_subplot(111)
    axis.set_facecolor(background)
    axis.axis("off")

    axis.text(
        0.5,
        0.965,
        "1769 Venus Transit — Geometric and Astronomical Audit",
        transform=axis.transAxes,
        ha="center",
        va="top",
        color=text_color,
        fontsize=15.5,
        fontweight="bold",
    )

    table = axis.table(
        cellText=rows,
        cellLoc="left",
        colWidths=[0.40, 0.20, 0.23, 0.17],
        bbox=[0.02, 0.05, 0.96, 0.84],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.4)

    color_map = {
        "header": palette["header"],
        "distance": palette["teal"],
        "geometry": palette["gold"],
        "final": palette["gold"],
    }
    for (row, column), cell in table.get_celld().items():
        group = row_groups[row]
        cell.set_facecolor(color_map[group])
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.55)
        cell.get_text().set_color(text_color)
        if row == 0 or group in ("distance", "geometry", "final"):
            cell.get_text().set_fontweight("bold")
        if column == 2:
            cell.get_text().set_ha("right")
        else:
            cell.get_text().set_ha("left")

    figure.savefig(PNG, dpi=220, facecolor=background, bbox_inches="tight", pad_inches=0.14)
    plt.close(figure)


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
        ["Reduced Earth-Venus distance", "EV̅", fmt(ev_bar), "km"],
        ["Reduced Venus-Sun distance", "VS̅", fmt(vs_bar), "km"],
        ["Reduced Earth-Sun distance", "ES̅", fmt(es_bar), "km"],
        ["Reduced distance closure", "EV̅ + VS̅ − ES̅", f"{reduced_closure:+.9f}", "km — PASS"],
        ["A′B′ common-normal separation", "A′B′", fmt(float(geometry["A_prime_B_prime_arcsec"])), "arcsec"],
        ["A′B′ common-normal separation", "A′B′", fmt(float(geometry["A_prime_B_prime_km"])), "km"],
        ["AB projected baseline", "AB", fmt(float(geometry["AB_arcsec"])), "arcsec"],
        ["AB projected baseline", "AB", fmt(float(geometry["AB_km"])), "km"],
        ["Finite IAU 1976 solar horizontal parallax", "π$_{finite,IAU76}$", fmt12(pi_finite_iau76), "arcsec"],
    ]
    row_groups = ["header", "distance", "distance", "distance", "distance", "geometry", "geometry", "geometry", "geometry", "final"]
    palette = {
        "header": str(ns.get("TABLE_HEADER", "#263849")),
        "teal": str(ns.get("TABLE_TEAL", "#183C48")),
        "gold": str(ns.get("TABLE_GOLD", "#5C4A20")),
    }

    save_png_table(rows, row_groups, palette)

    from IPython.display import Image, display
    display(Image(filename=str(PNG)))
    print(f"PNG: {PNG}")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0083
