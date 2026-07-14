# V0085
# Audit reference: GitHub widget only; expands verified IERS V0079/V0080 lineage, preserves reference styling, applies plot-only X-data mirror about Y-axis; no AI images.
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

VERSION = "V0085"
REPO_RAW = "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main"
RAW_V0079 = f"{REPO_RAW}/VENUS_1769_V0027_FORMAT_STANDALONE_V0079.py"
EXPANDED_OUT = Path("/content/VENUS_1769_V0027_FORMAT_STANDALONE_V0085_EXPANDED.py")


def fetch_url(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": VERSION})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read().decode("utf-8")


def build_reference_v0080_expanded_source() -> str:
    v0079_wrapper = fetch_url(RAW_V0079)
    if "# V0079" not in v0079_wrapper:
        raise RuntimeError("Reference lineage failure: V0079 wrapper was not loaded.")
    if "IERS_TN36_V01_MASTER-1" not in v0079_wrapper:
        raise RuntimeError("Reference lineage failure: V0079 is not from IERS repository.")
    if "GitHub_Sandbox" in v0079_wrapper:
        raise RuntimeError("Rejected wrong lineage: GitHub_Sandbox source detected.")

    namespace = {"__name__": "v0079_reference_builder"}
    exec(compile(v0079_wrapper, "V0079_reference_builder.py", "exec"), namespace, namespace)
    expanded = namespace["build_v0079_source"]()

    if "# V0079" not in expanded or "def plot_publication(" not in expanded:
        raise RuntimeError("Expanded V0079 source was not built correctly.")
    if "SUN_FILL_COLOR = \"#D95A1B\"" not in expanded:
        raise RuntimeError("Reference styling failure: V0079 solar fill is missing.")
    if "VENUS_PAINT_LINE_FACTOR" not in expanded:
        raise RuntimeError("Reference styling failure: Venus paint outlines are missing.")

    label_replacements = {
        "y_shift = 28.0 if above else -28.0": "y_shift = 36.0 if above else -36.0",
        "y_shift = 15.5 if above else -15.5": "y_shift = 21.0 if above else -21.0",
    }
    for old, new in label_replacements.items():
        if old not in expanded:
            raise RuntimeError(f"V0080 label-offset anchor not found: {old}")
        expanded = expanded.replace(old, new, 1)

    expanded = expanded.replace("V0079", VERSION)
    expanded = expanded.replace(
        "# Audit reference: Plot-only correction from verified V0067; redder stronger solar fill, track-colored Venus paint outlines, corrected label offsets, delta track-angle row, geometry unchanged.",
        "# Audit reference: Expanded from verified IERS V0079/V0080 reference; reference styling preserved; plot-only X-data mirror about Y-axis.",
    )
    expanded = expanded.replace(
        "VENUS_1769_V0027_FORMAT_STANDALONE_V0079.py",
        "VENUS_1769_V0027_FORMAT_STANDALONE_V0085_WIDGET.py",
    )
    return expanded


def inject_plot_only_y_axis_mirror(source: str) -> str:
    anchor = '''def write_outputs(
    point_result: dict[str, object],
    vardo_result: dict[str, object],
    geometry: dict[str, object],
) -> None:
'''
    mirror_function = '''def mirrored_plot_result_about_y_axis(result: dict[str, object]) -> dict[str, object]:
    # V0085 display-only mirror: x -> -x for plotted tracks and event centers.
    # Timings, local closest approach, contact roots, radii, geometry, tables, colors, and label offsets are preserved.
    mirrored = dict(result)
    mirrored["points"] = np.asarray(result["points"], dtype=float).copy()
    mirrored["points"][:, 0] *= -1.0
    mirrored["event_points"] = {
        key: np.asarray(value, dtype=float).copy()
        for key, value in result["event_points"].items()
    }
    for key in mirrored["event_points"]:
        mirrored["event_points"][key][0] *= -1.0
    return mirrored


'''
    if anchor not in source:
        raise RuntimeError("Mirror injection anchor not found: write_outputs block.")
    source = source.replace(anchor, mirror_function + anchor, 1)

    old = '''    write_outputs(point_result, vardo_result, geometry)
    plot_publication(
        point_result,
        vardo_result,
        geometry,
        max_contact_residual,
    )
'''
    new = '''    write_outputs(point_result, vardo_result, geometry)
    point_plot_result = mirrored_plot_result_about_y_axis(point_result)
    vardo_plot_result = mirrored_plot_result_about_y_axis(vardo_result)
    plot_publication(
        point_plot_result,
        vardo_plot_result,
        geometry,
        max_contact_residual,
    )
'''
    if old not in source:
        raise RuntimeError("Mirror call anchor not found: plot_publication call block.")
    return source.replace(old, new, 1)


def audit_source(source: str) -> None:
    checks = {
        "version_v0085": "# V0085" in source and 'VERSION = "V0085"' in source,
        "iers_reference_lineage": "IERS_TN36_V01_MASTER-1" in RAW_V0079,
        "no_sandbox_lineage": "GitHub_Sandbox" not in source,
        "main_solar_fill_reference": 'SUN_FILL_COLOR = "#D95A1B"' in source and "SUN_FILL_ALPHA = 0.260" in source,
        "solar_limb_reference": 'SUN_COLOR = "#FFD34A"' in source,
        "venus_paint_reference": "VENUS_PAINT_LINE_FACTOR" in source,
        "main_label_offsets_v0080": "y_shift = 36.0 if above else -36.0" in source,
        "zoom_label_offsets_v0080": "y_shift = 21.0 if above else -21.0" in source,
        "x_data_mirror_tracks": 'mirrored["points"][:, 0] *= -1.0' in source,
        "x_data_mirror_events": 'mirrored["event_points"][key][0] *= -1.0' in source,
        "not_fake_axis_reverse": "invert_xaxis" not in source and "set_xlim(1.07" not in source,
        "plot_uses_mirrored_copy": "point_plot_result = mirrored_plot_result_about_y_axis(point_result)" in source,
        "csv_uses_unmirrored_geometry": "write_outputs(point_result, vardo_result, geometry)" in source,
        "no_ai_images": "image_gen" not in source.lower(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    print("V0085 SOURCE AUDIT")
    for name, ok in checks.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if failed:
        raise RuntimeError(f"V0085 audit failed: {failed}")
    print("V0085 audit complete: PASS")


def main() -> None:
    source = build_reference_v0080_expanded_source()
    source = inject_plot_only_y_axis_mirror(source)
    audit_source(source)
    EXPANDED_OUT.write_text(source, encoding="utf-8")
    namespace = {"__name__": "__main__", "__file__": str(EXPANDED_OUT)}
    exec(compile(source, str(EXPANDED_OUT), "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
# V0085
