# V0023
# Audit reference: Exact V0013 publication layout with egress-left/ingress-right zooms, corrected zoom labels, and Halley A′B′→AB derivation table.
from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import time
import urllib.request
from pathlib import Path

VERSION = "V0023"
SOURCE_COMMIT = "83d03041feeb6ebd5b95e9f0edbf68c4b8b99f44"
SOURCE_NAME = "MERCURY_PARALLAX_PUBLICATION_V0013.py"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/"
    f"{SOURCE_COMMIT}/{SOURCE_NAME}"
)
EXPANDED_PATH = Path("/content/MERCURY_PARALLAX_PUBLICATION_V0023_FULL.py")


def fetch_exact_v0013_wrapper() -> str:
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


def literal_assignment(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise RuntimeError(f"Required V0013 assignment not found: {name}")


PATCH = r'''
import inspect as _v0023_inspect
import re as _v0023_re
from matplotlib.figure import Figure as _V0023Figure
from matplotlib.table import Table as _V0023Table

_v0023_original_savefig = _V0023Figure.savefig


def _v0023_axis_by_title(figure, token):
    token = token.upper()
    for axis in figure.axes:
        if token in axis.get_title().upper():
            return axis
    return None


def _v0023_tables(axis):
    found = []
    for table in getattr(axis, "tables", []):
        if table not in found:
            found.append(table)
    for child in axis.get_children():
        if isinstance(child, _V0023Table) and child not in found:
            found.append(child)
    return found


def _v0023_shift_label(axis, label, fraction):
    wanted = " ".join(label.upper().split())
    span = abs(float(axis.get_xlim()[1] - axis.get_xlim()[0]))
    for text_object in axis.texts:
        current = " ".join(text_object.get_text().upper().split())
        if current == wanted:
            x_value, y_value = text_object.get_position()
            text_object.set_position((x_value + fraction * span, y_value))


def _v0023_extract_aprime_bprime(axis):
    number_pattern = _v0023_re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
    for table in _v0023_tables(axis):
        cells = table.get_celld()
        row_numbers = sorted({row for row, _column in cells})
        column_numbers = sorted({_column for _row, _column in cells})
        for row in row_numbers:
            row_text = [
                cells[(row, column)].get_text().get_text().strip()
                if (row, column) in cells
                else ""
                for column in column_numbers
            ]
            symbol = row_text[0].replace(" ", "")
            if symbol in {"A′B′", "A'B'", "A’B’"}:
                for candidate in reversed(row_text):
                    match = number_pattern.search(candidate)
                    if match:
                        return float(match.group(0).replace(",", ""))
    raise RuntimeError("A′B′ value was not found in the original V0013 center table.")


def _v0023_replace_center_table(figure, frame_locals):
    track_axis = _v0023_axis_by_title(figure, "TRACK GEOMETRY")
    if track_axis is None:
        candidates = []
        for axis in figure.axes:
            tables = _v0023_tables(axis)
            position = axis.get_position()
            if tables and position.y0 < 0.35 and position.x0 < 0.70:
                candidates.append((abs((position.x0 + position.x1) / 2.0 - 0.43), axis))
        if candidates:
            track_axis = sorted(candidates, key=lambda item: item[0])[0][1]
    if track_axis is None:
        raise RuntimeError("The V0013 center geometry axis was not found.")

    aprime_bprime_arcsec = _v0023_extract_aprime_bprime(track_axis)

    site_results = frame_locals["site_results"]
    cache = frame_locals["cache"]
    vector_function = frame_locals.get("vector_at", globals()["vector_at"])
    arcsec_per_rad = float(frame_locals.get("ARCSEC_PER_RAD", globals()["ARCSEC_PER_RAD"]))

    jd_reference = 0.5 * (
        float(site_results["MERCURY_BAY"]["maximum"])
        + float(site_results["VARDO"]["maximum"])
    )
    sun_vector = vector_function(cache, "GEOCENTER_SUN", jd_reference)
    mercury_vector = vector_function(cache, "GEOCENTER_MERCURY", jd_reference)

    es_km = float(np.linalg.norm(np.asarray(sun_vector, dtype=float)))
    em_km = float(np.linalg.norm(np.asarray(mercury_vector, dtype=float)))
    ms_km = float(
        np.linalg.norm(
            np.asarray(sun_vector, dtype=float)
            - np.asarray(mercury_vector, dtype=float)
        )
    )
    halley_ratio = em_km / ms_km
    aprime_bprime_km = (
        aprime_bprime_arcsec / arcsec_per_rad
    ) * es_km
    ab_km = aprime_bprime_km * halley_ratio

    angle_mb = float(site_results["MERCURY_BAY"]["fit"]["angle_deg"])
    angle_v = float(site_results["VARDO"]["fit"]["angle_deg"])
    angle_average = 0.5 * (angle_mb + angle_v)

    for table in _v0023_tables(track_axis):
        table.remove()

    track_axis.set_title(
        "HALLEY DERIVATION — A′B′ TO AB",
        fontsize=10,
        fontweight="bold",
        pad=6,
    )
    track_axis.axis("off")

    rows = [
        ["Quantity", "Derivation", "Angular", "Physical / ratio", "Unit"],
        [
            "A′B′",
            "Common-normal apparent separation",
            f"{aprime_bprime_arcsec:.6f}",
            "",
            "arcsec",
        ],
        [
            "A′B′",
            "(A′B′ / 206264.806247) × ES",
            "",
            f"{aprime_bprime_km:,.6f}",
            "km",
        ],
        [
            "Halley ratio",
            "EM / MS",
            "",
            f"{halley_ratio:.12f}",
            "dimensionless",
        ],
        [
            "AB",
            "A′B′(km) × EM / MS",
            "",
            f"{ab_km:,.6f}",
            "km",
        ],
        [
            "α MB",
            "Mercury Bay fitted-track angle",
            f"{angle_mb:.6f}",
            "",
            "deg",
        ],
        [
            "α V",
            "Vardø fitted-track angle",
            f"{angle_v:.6f}",
            "",
            "deg",
        ],
        [
            "ᾱ",
            "Average fitted-track angle",
            f"{angle_average:.6f}",
            "",
            "deg",
        ],
    ]

    table = track_axis.table(
        cellText=rows,
        cellLoc="left",
        colWidths=[0.16, 0.42, 0.14, 0.18, 0.10],
        bbox=[0.00, 0.00, 1.00, 0.92],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)

    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        cell.set_linewidth(0.375)
        cell.get_text().set_color("white")
        if row == 0:
            cell.set_facecolor("#1E3A5F")
            cell.get_text().set_fontweight("bold")
        elif row in (1, 2, 4):
            cell.set_facecolor("#123B48")
            if row in (1, 4):
                cell.get_text().set_fontweight("bold")
        elif row == 3:
            cell.set_facecolor("#4A3510")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#0F172A")


def _v0023_apply_line_weights(figure):
    for axis in figure.axes:
        for line in axis.lines:
            color = str(line.get_color()).lower()
            x_data = np.asarray(line.get_xdata())
            y_data = np.asarray(line.get_ydata())
            solar_curve = (
                len(x_data) > 500
                and len(y_data) > 500
                and color in {
                    "white",
                    "#ffffff",
                    "#f8fafc",
                    "#e2e8f0",
                }
            )
            line.set_linewidth(0.500 if solar_curve else 0.375)
        for patch in axis.patches:
            if isinstance(patch, Circle):
                patch.set_linewidth(0.375)


def _v0023_patch_figure(figure, frame_locals):
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


def _v0023_savefig(figure, *args, **kwargs):
    caller = _v0023_inspect.currentframe().f_back
    _v0023_patch_figure(figure, caller.f_locals)
    return _v0023_original_savefig(figure, *args, **kwargs)


_V0023Figure.savefig = _v0023_savefig
'''


def build_source() -> str:
    wrapper_text = fetch_exact_v0013_wrapper()
    wrapper_tree = ast.parse(wrapper_text, filename=SOURCE_NAME)
    payload = literal_assignment(wrapper_tree, "PAYLOAD")
    expected_sha256 = literal_assignment(wrapper_tree, "EXPECTED_SHA256")

    source_bytes = gzip.decompress(base64.b64decode(payload))
    actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Exact V0013 source verification failed: "
            f"expected {expected_sha256}, received {actual_sha256}"
        )

    source_text = source_bytes.decode("utf-8")
    source_text = source_text.replace("V0013", VERSION)

    original_tail = (
        'if __name__ == "__main__":\n'
        "    main()\n"
        f"# {VERSION}"
    )
    revised_tail = (
        f"{PATCH}\n\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
        f"# {VERSION}"
    )
    if original_tail not in source_text:
        raise RuntimeError("V0013 execution tail was not found.")
    revised_source = source_text.replace(original_tail, revised_tail, 1)
    compile(revised_source, str(EXPANDED_PATH), "exec")
    return revised_source


def main() -> None:
    source = build_source()
    EXPANDED_PATH.write_text(source, encoding="utf-8")
    exec(
        compile(source, str(EXPANDED_PATH), "exec"),
        {"__name__": "__main__", "__file__": str(EXPANDED_PATH)},
    )


if __name__ == "__main__":
    main()
# V0023
