# V0062
# Audit reference: Preserve the approved V0027 publication figure exactly while replacing only the scientific inputs with the 1769 Venus transit and V0061 separate-ray transfer.
from __future__ import annotations

import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0062"
TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
TEMPLATE_COMMIT = "e0ed886b732ab0dd7e4b48b8294073294bd06479"
TEMPLATE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/"
    f"{TEMPLATE_COMMIT}/MERCURY_PARALLAX_PUBLICATION_V0027.py"
)
GOLD_COMMIT = "b0a2df3cb9d5ea2f7fa36af79c39bd60e4986f60"
GOLD_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/"
    f"{GOLD_COMMIT}/VENUS_1769_SEPARATE_RAY_VECTOR_TRANSFER_GOLD_STANDARD_V0061.py"
)
GOLD_SCRIPT = ROOT / "VENUS_1769_SEPARATE_RAY_VECTOR_TRANSFER_GOLD_STANDARD_V0061.py"
GOLD_CSV = (
    ROOT
    / "VENUS_1769_SEPARATE_RAY_VECTOR_TRANSFER_GOLD_STANDARD_V0061_OUTPUT"
    / "VENUS_1769_SEPARATE_RAY_VECTOR_TRANSFER_GOLD_STANDARD_V0061.csv"
)
FULL_SOURCE = ROOT / "VENUS_PARALLAX_PUBLICATION_V0062_FULL.py"


def fetch_text(url: str) -> str:
    request = urllib.request.Request(
        f"{url}?cache={time.time_ns()}",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8")


def execute_builder(source: str, name: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "__name__": name,
        "__file__": f"<{name}>",
    }
    exec(compile(source, f"<{name}>", "exec"), namespace)
    return namespace


def materialize_exact_v0027_source() -> str:
    stage_27 = fetch_text(TEMPLATE_URL)
    if not stage_27.startswith("# V0027\n"):
        raise RuntimeError("Pinned V0027 opening boundary was not found.")

    namespace_27 = execute_builder(stage_27, "v0062_v0027_builder")
    stage_26 = namespace_27["build_isolated_source"]()

    namespace_26 = execute_builder(stage_26, "v0062_v0026_builder")
    stage_25 = namespace_26["build_isolated_source"]()

    namespace_25 = execute_builder(stage_25, "v0062_v0025_builder")
    stage_23 = namespace_25["build_isolated_source"]()

    namespace_23 = execute_builder(stage_23, "v0062_v0023_builder")
    full_source = namespace_23["build_source"]()
    if not full_source.startswith("# V0027\n"):
        raise RuntimeError("Expanded V0027 source boundary was not preserved.")
    return str(full_source)


def ensure_gold_standard() -> None:
    if GOLD_CSV.is_file() and GOLD_CSV.stat().st_size > 0:
        return

    GOLD_SCRIPT.write_text(fetch_text(GOLD_URL), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(GOLD_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "V0061 gold-standard generation failed.\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )
    if not GOLD_CSV.is_file() or GOLD_CSV.stat().st_size == 0:
        raise RuntimeError("V0061 completed without producing its CSV output.")


def replace_required(text: str, old: str, new: str, minimum: int = 1) -> str:
    count = text.count(old)
    if count < minimum:
        raise RuntimeError(f"Required template token was not found: {old!r}")
    return text.replace(old, new)


def replace_function(
    text: str,
    function_signature: str,
    next_signature: str,
    replacement: str,
) -> str:
    start = text.find(f"def {function_signature}")
    if start < 0:
        raise RuntimeError(f"Template function was not found: {function_signature}")
    end = text.find(f"\ndef {next_signature}", start)
    if end < 0:
        raise RuntimeError(f"Following template function was not found: {next_signature}")
    return text[:start] + replacement.rstrip() + "\n\n" + text[end + 1 :]


GOLD_INJECTION = r'''
V0062_GOLD_CSV = (
    Path("/content")
    / "VENUS_1769_SEPARATE_RAY_VECTOR_TRANSFER_GOLD_STANDARD_V0061_OUTPUT"
    / "VENUS_1769_SEPARATE_RAY_VECTOR_TRANSFER_GOLD_STANDARD_V0061.csv"
)


def _v0062_gold_value(section, key_column, key_value, value_column):
    frame = pd.read_csv(V0062_GOLD_CSV)
    rows = frame[
        (frame["section"].astype(str) == section)
        & (frame[key_column].astype(str) == key_value)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one V0061 row for {section}/{key_value}; found {len(rows)}."
        )
    value = pd.to_numeric(rows.iloc[0][value_column], errors="coerce")
    if pd.isna(value):
        raise RuntimeError(
            f"V0061 value is not numeric: {section}/{key_value}/{value_column}."
        )
    return float(value)


V0062_APRIME_BPRIME_ARCSEC = _v0062_gold_value(
    "TRANSFER",
    "Projected quantity",
    "Direct instantaneous point separation",
    "Arcseconds",
)
V0062_APRIME_BPRIME_KM = _v0062_gold_value(
    "TRANSFER",
    "Projected quantity",
    "Direct instantaneous point separation",
    "Kilometers",
)
V0062_AB_ARCSEC = _v0062_gold_value(
    "TRANSFER",
    "Projected quantity",
    "Direct station baseline",
    "Arcseconds",
)
V0062_AB_KM = _v0062_gold_value(
    "TRANSFER",
    "Projected quantity",
    "Direct station baseline",
    "Kilometers",
)
V0062_SEPARATE_RAY_RATIO = _v0062_gold_value(
    "RATIOS",
    "Ratio",
    "Exact separate-ray transfer ratio",
    "Value",
)
'''


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
        raise RuntimeError("The V0027 center geometry axis was not found.")

    site_results = frame_locals["site_results"]
    aprime_bprime_arcsec = V0062_APRIME_BPRIME_ARCSEC
    aprime_bprime_km = V0062_APRIME_BPRIME_KM
    ab_arcsec = V0062_AB_ARCSEC
    ab_km = V0062_AB_KM

    angle_pv = float(
        site_results["POINT_VENUS"]["fit"]["angle_deg"]
    )
    angle_v = float(
        site_results["VARDO"]["fit"]["angle_deg"]
    )
    angle_average = 0.5 * (angle_pv + angle_v)

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
            "α PV",
            "Point Venus, Tahiti track angle (degrees)",
            f"{angle_pv:.6f}°",
            "",
        ],
        [
            "α V",
            "Vardo, Norway track angle (degrees)",
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


def convert_to_venus(source: str) -> str:
    source = replace_required(source, "V0027", VERSION)

    source = replace_required(source, 'START = "1769-11-09 12:00"', 'START = "1769-06-03 18:00"')
    source = replace_required(source, 'STOP = "1769-11-10 08:00"', 'STOP = "1769-06-04 05:00"')
    source = replace_required(source, '"lat": -36.783333333333,', '"lat": -17.495600000000,')
    source = replace_required(source, '"lon": 175.933333333333,', '"lon": -149.493900000000,')
    source = replace_required(source, '"lat": 70.370600000000,', '"lat": 70.372400000000,')
    source = replace_required(source, '"lon": 31.110700000000,', '"lon": 31.110300000000,')

    source = source.replace("MERCURY BAY", "POINT VENUS, TAHITI")
    source = source.replace("Mercury Bay", "Point Venus, Tahiti")
    source = source.replace("MERCURY_BAY", "POINT_VENUS")
    source = source.replace('"MB"', '"PV"')
    source = source.replace("MB C", "PV C")
    source = source.replace("α MB", "α PV")
    source = source.replace("VARDØ", "VARDO, NORWAY")
    source = source.replace("Vardø", "Vardo, Norway")

    source = source.replace("MERCURY", "VENUS")
    source = source.replace("Mercury", "Venus")
    source = source.replace("mercury", "venus")
    source = replace_required(source, '"199"', '"299"')
    source = replace_required(
        source,
        "VENUS_RADIUS_KM = 2_439.700000",
        "VENUS_RADIUS_KM = 6_051.800000",
    )

    source = source.replace(
        "MERCURY_PARALLAX_PUBLICATION",
        "VENUS_PARALLAX_PUBLICATION",
    )
    source = source.replace("VENUS_BAY", "POINT_VENUS")

    import_marker = "import pandas as pd\n"
    if source.count(import_marker) != 1:
        raise RuntimeError("The exact pandas import marker was not unique.")
    source = source.replace(import_marker, import_marker + GOLD_INJECTION + "\n", 1)

    source = replace_function(
        source,
        "_v0023_replace_center_table(figure, frame_locals):",
        "_v0023_apply_line_weights(figure):",
        NEW_CENTER_FUNCTION,
    )

    ratio_pattern = re.compile(r"(?m)^(\s*)halley_ratio\s*=.*$")
    source, ratio_count = ratio_pattern.subn(
        r"\1halley_ratio = V0062_SEPARATE_RAY_RATIO",
        source,
    )
    if ratio_count < 1:
        raise RuntimeError("No Halley-ratio assignment was found in the V0027 template.")

    source = source.replace("PV C2", "PV C2")
    source = source.replace("V C1", "V C1")

    if "Mercury" in source or "MERCURY" in source or "mercury" in source:
        raise RuntimeError("A Mercury scientific token remained after conversion.")
    if "Mercury Bay" in source or "Vardø" in source:
        raise RuntimeError("An old station label remained after conversion.")
    if "Point Venus, Tahiti" not in source or "Vardo, Norway" not in source:
        raise RuntimeError("The requested station labels were not installed.")
    if not source.startswith(f"# {VERSION}\n"):
        raise RuntimeError("V0062 opening boundary audit failed.")
    if not source.rstrip().endswith(f"# {VERSION}"):
        raise RuntimeError("V0062 closing boundary audit failed.")

    compile(source, str(FULL_SOURCE), "exec")
    return source


def main() -> None:
    ensure_gold_standard()
    exact_v0027 = materialize_exact_v0027_source()
    venus_source = convert_to_venus(exact_v0027)
    FULL_SOURCE.write_text(venus_source, encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(FULL_SOURCE)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "The V0062 Venus publication process failed.\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )

    candidates = sorted(
        ROOT.rglob("*V0062*.png"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("V0062 completed but no PNG output was found.")
    output_png = candidates[0]

    from IPython.display import Image, display

    display(Image(filename=str(output_png)))

    print("CODE INPUTS")
    print("Exact approved V0027 figure source; 1769 Venus JPL vectors; V0061 separate-ray transfer data.")
    print("COMMENTS")
    print("Figure layout, colors, axes, zoom panels, tables, line weights, markers, and typography are preserved from V0027.")
    print("RESULTS")
    print("Transit pair: Point Venus, Tahiti and Vardo, Norway")
    print("OUTPUT SUMMARY")
    print(f"PNG: {output_png}")
    print(f"PY: {FULL_SOURCE}")
    print("PAPER COMPARISON")
    print("NOT USED. This is the updated JPL Venus publication figure.")
    print("EQUATION STATUS")
    print("V0027 format preservation and V0061 separate-ray transfer integration: PASS")
    print(datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0062