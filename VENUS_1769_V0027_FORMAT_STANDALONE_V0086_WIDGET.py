# V0086
# Audit reference: GitHub widget only; expands verified V0085 plot, preserves all styling/math, swaps zoom window positions only; no AI images.
from __future__ import annotations

import urllib.request
from pathlib import Path

VERSION = "V0086"
RAW_V0085 = "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/VENUS_1769_V0027_FORMAT_STANDALONE_V0085_WIDGET.py"
EXPANDED_OUT = Path("/content/VENUS_1769_V0027_FORMAT_STANDALONE_V0086_EXPANDED.py")


def fetch_url(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": VERSION})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read().decode("utf-8")


def build_v0085_expanded_source() -> str:
    wrapper = fetch_url(RAW_V0085)
    if "# V0085" not in wrapper:
        raise RuntimeError("Reference lineage failure: V0085 wrapper was not loaded.")
    if "image_gen" in wrapper.lower():
        raise RuntimeError("Rejected: AI image call detected in wrapper.")
    namespace = {"__name__": "v0085_reference_builder"}
    exec(compile(wrapper, "V0085_reference_builder.py", "exec"), namespace, namespace)
    source = namespace["build_reference_v0080_expanded_source"]()
    source = namespace["inject_plot_only_y_axis_mirror"](source)
    if "# V0085" not in source or "def plot_publication(" not in source:
        raise RuntimeError("Expanded V0085 source was not built correctly.")
    return source


def swap_zoom_windows_only(source: str) -> str:
    old = '''    egress_axis = figure.add_subplot(lower[0, 0])
    derivation_axis = figure.add_subplot(lower[0, 1])
    ingress_axis = figure.add_subplot(lower[0, 2])
'''
    new = '''    ingress_axis = figure.add_subplot(lower[0, 0])
    derivation_axis = figure.add_subplot(lower[0, 1])
    egress_axis = figure.add_subplot(lower[0, 2])
'''
    if old not in source:
        raise RuntimeError("Zoom-axis assignment anchor not found; no unsafe partial patch applied.")
    return source.replace(old, new, 1)


def promote_to_v0086(source: str) -> str:
    source = source.replace("V0085", VERSION)
    source = source.replace(
        "# Audit reference: Expanded from verified IERS V0079/V0080 reference; reference styling preserved; plot-only X-data mirror about Y-axis.",
        "# Audit reference: Expanded from verified IERS V0085 reference; reference styling preserved; plot-only X-data mirror retained; zoom windows swapped only.",
    )
    source = source.replace(
        "VENUS_1769_V0027_FORMAT_STANDALONE_V0085_WIDGET.py",
        "VENUS_1769_V0027_FORMAT_STANDALONE_V0086_WIDGET.py",
    )
    return source


def audit_source(source: str) -> None:
    checks = {
        "version_v0086": "# V0086" in source and 'VERSION = "V0086"' in source,
        "iers_reference_lineage": "IERS_TN36_V01_MASTER-1" not in source or "GitHub_Sandbox" not in source,
        "no_sandbox_lineage": "GitHub_Sandbox" not in source,
        "main_solar_fill_reference": 'SUN_FILL_COLOR = "#D95A1B"' in source and "SUN_FILL_ALPHA = 0.260" in source,
        "solar_limb_reference": 'SUN_COLOR = "#FFD34A"' in source,
        "venus_paint_reference": "VENUS_PAINT_LINE_FACTOR" in source,
        "main_label_offsets_v0080": "y_shift = 36.0 if above else -36.0" in source,
        "zoom_label_offsets_v0080": "y_shift = 21.0 if above else -21.0" in source,
        "x_data_mirror_tracks": 'mirrored["points"][:, 0] *= -1.0' in source,
        "x_data_mirror_events": 'mirrored["event_points"][key][0] *= -1.0' in source,
        "ingress_left": "ingress_axis = figure.add_subplot(lower[0, 0])" in source,
        "egress_right": "egress_axis = figure.add_subplot(lower[0, 2])" in source,
        "derivation_middle": "derivation_axis = figure.add_subplot(lower[0, 1])" in source,
        "old_zoom_order_removed": "egress_axis = figure.add_subplot(lower[0, 0])" not in source and "ingress_axis = figure.add_subplot(lower[0, 2])" not in source,
        "not_fake_axis_reverse": "invert_xaxis" not in source and "set_xlim(1.07" not in source,
        "plot_uses_mirrored_copy": "point_plot_result = mirrored_plot_result_about_y_axis(point_result)" in source,
        "csv_uses_unmirrored_geometry": "write_outputs(point_result, vardo_result, geometry)" in source,
        "no_ai_images": "image_gen" not in source.lower(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    print("V0086 SOURCE AUDIT")
    for name, ok in checks.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if failed:
        raise RuntimeError(f"V0086 audit failed: {failed}")
    print("V0086 audit complete: PASS")


def main() -> None:
    source = build_v0085_expanded_source()
    source = swap_zoom_windows_only(source)
    source = promote_to_v0086(source)
    audit_source(source)
    EXPANDED_OUT.write_text(source, encoding="utf-8")
    namespace = {"__name__": "__main__", "__file__": str(EXPANDED_OUT)}
    exec(compile(source, str(EXPANDED_OUT), "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
# V0086
