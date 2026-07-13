# V0025
# Audit reference: Clean-process correction of the V0013 Mercury publication table and title while preserving all plot geometry and right-side tables.
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

VERSION = "V0025"
SOURCE_COMMIT = "388aae990157cd88db8f1314d174d48f9dd2f4c2"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/"
    f"{SOURCE_COMMIT}/MERCURY_PARALLAX_PUBLICATION_V0023.py"
)
ISOLATED_PATH = Path("/content/MERCURY_PARALLAX_PUBLICATION_V0025_ISOLATED.py")

NEW_CENTER_FUNCTION = r'''def _v0023_replace_center_table(figure, frame_locals):
    candidate_axes = []
    for axis in figure.axes:
        position = axis.get_position()
        midpoint_x = 0.5 * (position.x0 + position.x1)
        if (
            _v0023_tables(axis)
            and position.y0 < 0.35
            and 0.18 < midpoint_x < 0.58
        ):
            candidate_axes.append(axis)

    if candidate_axes:
        track_axis = max(
            candidate_axes,
            key=lambda axis: axis.get_position().width,
        )
    else:
        track_axis = _v0023_axis_by_title(figure, "TRACK GEOMETRY")

    if track_axis is None:
        raise RuntimeError("The V0013 center geometry axis was not found.")

    aprime_bprime_arcsec = _v0023_extract_aprime_bprime(track_axis)

    site_results = frame_locals["site_results"]
    cache = frame_locals["cache"]
    vector_function = frame_locals.get("vector_at", globals()["vector_at"])
    arcsec_per_rad = float(
        frame_locals.get("ARCSEC_PER_RAD", globals()["ARCSEC_PER_RAD"])
    )

    jd_reference = 0.5 * (
        float(site_results["MERCURY_BAY"]["maximum"])
        + float(site_results["VARDO"]["maximum"])
    )
    sun_vector = vector_function(cache, "GEOCENTER_SUN", jd_reference)
    mercury_vector = vector_function(cache, "GEOCENTER_MERCURY", jd_reference)

    earth_sun_km = float(
        np.linalg.norm(np.asarray(sun_vector, dtype=float))
    )
    earth_mercury_km = float(
        np.linalg.norm(np.asarray(mercury_vector, dtype=float))
    )
    mercury_sun_km = float(
        np.linalg.norm(
            np.asarray(sun_vector, dtype=float)
            - np.asarray(mercury_vector, dtype=float)
        )
    )

    halley_ratio = earth_mercury_km / mercury_sun_km
    aprime_bprime_km = (
        aprime_bprime_arcsec / arcsec_per_rad
    ) * earth_sun_km
    ab_arcsec = aprime_bprime_arcsec * halley_ratio
    ab_km = aprime_bprime_km * halley_ratio

    angle_mb = float(
        site_results["MERCURY_BAY"]["fit"]["angle_deg"]
    )
    angle_v = float(
        site_results["VARDO"]["fit"]["angle_deg"]
    )
    angle_average = 0.5 * (angle_mb + angle_v)

    geometry_axes = []
    for axis in figure.axes:
        position = axis.get_position()
        midpoint_x = 0.5 * (position.x0 + position.x1)
        if position.y0 < 0.35 and 0.18 < midpoint_x < 0.58:
            geometry_axes.append(axis)

    for axis in geometry_axes:
        axis.set_title("")
        axis.title.set_visible(False)
        for table in list(_v0023_tables(axis)):
            table.remove()
        for text_object in axis.texts:
            text_value = text_object.get_text().upper()
            if (
                "TRACK GEOMETRY" in text_value
                or "HALLEY DERIVATION" in text_value
                or "A′B′ TO AB" in text_value
            ):
                text_object.set_visible(False)

    for text_object in figure.texts:
        text_value = text_object.get_text().upper()
        if (
            "TRACK GEOMETRY" in text_value
            or "HALLEY DERIVATION" in text_value
            or "A′B′ TO AB" in text_value
        ):
            text_object.set_visible(False)

    track_axis.axis("off")
    track_axis.text(
        0.5,
        0.985,
        "A′B′ AND AB DERIVATION",
        transform=track_axis.transAxes,
        ha="center",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color="white",
        zorder=20,
    )

    rows = [
        ["Quantity", "Definition", "Arcseconds", "Kilometers"],
        [
            "A′B′",
            "JPL derived",
            f"{aprime_bprime_arcsec:.6f}",
            f"{aprime_bprime_km:,.6f}",
        ],
        [
            "AB",
            "JPL derived",
            f"{ab_arcsec:.6f}",
            f"{ab_km:,.6f}",
        ],
        [
            "α MB",
            "Mercury Bay track angle (degrees)",
            f"{angle_mb:.6f}°",
            "",
        ],
        [
            "α V",
            "Vardø track angle (degrees)",
            f"{angle_v:.6f}°",
            "",
        ],
        [
            "ᾱ",
            "Average track angle (degrees)",
            f"{angle_average:.6f}°",
            "",
        ],
    ]

    table = track_axis.table(
        cellText=rows,
        cellLoc="left",
        colWidths=[0.15, 0.43, 0.19, 0.23],
        bbox=[0.00, 0.00, 1.00, 0.84],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)

    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        cell.set_linewidth(0.375)
        cell.get_text().set_color("white")
        if row == 0:
            cell.set_facecolor("#1E3A5F")
            cell.get_text().set_fontweight("bold")
        elif row == 1:
            cell.set_facecolor("#123B48")
            cell.get_text().set_fontweight("bold")
        elif row == 2:
            cell.set_facecolor("#4A3510")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#0F172A")
'''

NEW_PATCH_FUNCTION = r'''def _v0023_patch_figure(figure, frame_locals):
    ingress_axis = _v0023_axis_by_title(figure, "INGRESS ZOOM")
    egress_axis = _v0023_axis_by_title(figure, "EGRESS ZOOM")
    if ingress_axis is None or egress_axis is None:
        raise RuntimeError("The V0013 ingress/egress zoom axes were not found.")

    ingress_position = ingress_axis.get_position().frozen()
    egress_position = egress_axis.get_position().frozen()
    ingress_axis.set_position(egress_position)
    egress_axis.set_position(ingress_position)

    _v0023_shift_label(ingress_axis, "MB C2", -0.12)
    _v0023_shift_label(egress_axis, "V C4", +0.10)
    _v0023_shift_label(egress_axis, "MB C3", +0.10)
    _v0023_shift_label(egress_axis, "V C3", -0.10)

    _v0023_replace_center_table(figure, frame_locals)
    _v0023_apply_line_weights(figure)
'''


def fetch_source() -> str:
    request = urllib.request.Request(
        f"{SOURCE_URL}?cache={time.time_ns()}",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read().decode("utf-8")


def replace_function(
    text: str,
    function_name: str,
    next_function_name: str,
    replacement: str,
) -> str:
    start_marker = f"def {function_name}"
    end_marker = f"\ndef {next_function_name}"
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"Function not found: {function_name}")
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError(
            f"Following function not found: {next_function_name}"
        )
    return text[:start] + replacement.rstrip() + "\n\n" + text[end + 1 :]


def build_isolated_source() -> str:
    source = fetch_source()
    source = replace_function(
        source,
        "_v0023_replace_center_table(figure, frame_locals):",
        "_v0023_apply_line_weights(figure):",
        NEW_CENTER_FUNCTION,
    )
    source = replace_function(
        source,
        "_v0023_patch_figure(figure, frame_locals):",
        "_v0023_savefig(figure, *args, **kwargs):",
        NEW_PATCH_FUNCTION,
    )
    source = source.replace("V0023", VERSION)
    if not source.startswith(f"# {VERSION}\n"):
        raise RuntimeError("V0025 source boundary check failed.")
    compile(source, str(ISOLATED_PATH), "exec")
    return source


def main() -> None:
    source = build_isolated_source()
    ISOLATED_PATH.write_text(source, encoding="utf-8")
    subprocess.run(
        [sys.executable, str(ISOLATED_PATH)],
        check=True,
    )

    candidates = sorted(
        Path("/content").rglob("*V0025*.png"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("V0025 completed but no PNG output was found.")

    from IPython.display import Image, display
    display(Image(filename=str(candidates[0])))


if __name__ == "__main__":
    main()
# V0025
