# V0088
# Audit reference: GitHub widget only; expands verified V0087 plot; adjusts zoom labels and derivation-table wording/alignment only; no AI images.
from __future__ import annotations

from pathlib import Path
import urllib.request

VERSION = "V0088"
RAW_V0087 = "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/VENUS_1769_V0027_FORMAT_STANDALONE_V0087_WIDGET.py"
EXPANDED_OUT = Path("/content/VENUS_1769_V0027_FORMAT_STANDALONE_V0088_EXPANDED.py")


def fetch_url(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": VERSION})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read().decode("utf-8")


def build_v0087_expanded_source() -> str:
    wrapper = fetch_url(RAW_V0087)
    if "# V0087" not in wrapper:
        raise RuntimeError("Reference lineage failure: V0087 wrapper was not loaded.")
    if "image_gen." in wrapper.lower() or "image_gen(" in wrapper.lower():
        raise RuntimeError("Rejected: executable AI image call detected in wrapper.")
    namespace = {"__name__": "v0087_reference_builder"}
    exec(compile(wrapper, "V0087_reference_builder.py", "exec"), namespace, namespace)
    source = namespace["build_v0085_expanded_source"]()
    source = namespace["swap_zoom_windows_only"](source)
    source = namespace["promote_to_v0087"](source)
    if "# V0087" not in source or "def plot_publication(" not in source:
        raise RuntimeError("Expanded V0087 source was not built correctly.")
    return source


def patch_zoom_labels(source: str) -> str:
    start = source.find("def draw_events(")
    end = source.find("\ndef zoom_limits(", start)
    if start < 0 or end < 0:
        raise RuntimeError("draw_events block not found; no unsafe partial patch applied.")
    new_block = '''def draw_events(
    axis: plt.Axes,
    result: dict[str, object],
    events: tuple[str, ...],
    main: bool,
) -> None:
    site = result["site"]
    color = str(site["color"])
    short = str(site["short"])
    x_shifts = {
        "C1": -13.0,
        "C2": -4.0,
        "CA": 0.0,
        "C3": 4.0,
        "C4": 13.0,
    }
    zoom_offsets = {
        ("PV", "C1"): (0.0, 0.0, True),
        ("PV", "C2"): (0.0, 0.0, True),
        ("V", "C1"): (10.0, 10.0, True),
        ("V", "C2"): (10.0, 10.0, True),
        ("PV", "C3"): (0.0, -18.0, False),
        ("PV", "C4"): (-18.0, -18.0, False),
        ("V", "C3"): (-14.0, -14.0, False),
        ("V", "C4"): (-16.0, 10.0, True),
    }
    for event in events:
        center = np.asarray(result["event_points"][event], dtype=float)
        radius = float(result["event_radii"][event][1])
        add_venus_disk(axis, center, radius, color)
        marker = "X" if event == "CA" else "o"
        axis.scatter(
            [center[0]],
            [center[1]],
            s=16.0 if event == "CA" else 7.5,
            marker=marker,
            color=color,
            edgecolors=BACKGROUND,
            linewidths=MARKER_EDGE_WIDTH,
            zorder=8,
        )
        if main:
            above = short == "PV"
            y_shift = 36.0 if above else -36.0
            x_shift = x_shifts[event]
        else:
            x_shift, y_shift, above = zoom_offsets.get(
                (short, event),
                (x_shifts[event], 21.0 if short == "PV" else -21.0, short == "PV"),
            )
        annotate_event(
            axis,
            center,
            f"{short} {event}",
            color,
            above,
            x_shift,
            y_shift,
        )

'''
    return source[:start] + new_block + source[end + 1:]


def patch_derivation_table_text(source: str) -> str:
    replacements = {
        '"Point Venus, Tahiti track angle (degrees)"': '"Point Venus, Tahiti track angle"',
        '"Vardo, Norway track angle (degrees)"': '"Vardo, Norway track angle"',
        '"Vardø, Norway track angle (degrees)"': '"Vardø, Norway track angle"',
        '"Delta track angle, |αV − αPV| (degrees)"': '"Delta track angle, |αV − αPV|"',
        '"Average track angle (degrees)"': '"Average track angle"',
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    source = source.replace(
        '    style_table(\n        table,\n        teal_rows=(1,),\n        gold_rows=(2, 3, 4, 5),\n        font_size=5.75,\n    )',
        '    style_table(\n        table,\n        teal_rows=(1,),\n        gold_rows=(2, 3, 4, 5),\n        font_size=5.75,\n    )\n    for row_index in range(1, len(rows)):\n        table[(row_index, 2)].get_text().set_ha("center")'
    )
    source = source.replace(
        '    style_table(\n        table,\n        teal_rows=(1,),\n        gold_rows=(2,),\n        font_size=6.4,\n    )',
        '    style_table(\n        table,\n        teal_rows=(1,),\n        gold_rows=(2,),\n        font_size=6.4,\n    )\n    for row_index in range(1, len(rows)):\n        table[(row_index, 2)].get_text().set_ha("center")'
    )
    return source


def promote_to_v0088(source: str) -> str:
    source = source.replace("V0087", VERSION)
    source = source.replace(
        "# Audit reference: Expanded from verified IERS V0085 reference; reference styling preserved; plot-only X-data mirror retained; zoom windows swapped only.",
        "# Audit reference: Expanded from verified IERS V0087 reference; zoom labels adjusted for legibility; derivation-table degree wording removed; math/style preserved.",
    )
    source = source.replace(
        "VENUS_1769_V0027_FORMAT_STANDALONE_V0087_WIDGET.py",
        "VENUS_1769_V0027_FORMAT_STANDALONE_V0088_WIDGET.py",
    )
    return source


def audit_source(source: str) -> None:
    checks = {
        "version_v0088": "# V0088" in source and 'VERSION = "V0088"' in source,
        "no_sandbox_lineage": "GitHub_Sandbox" not in source,
        "no_executable_ai_images": "image_gen." not in source.lower() and "image_gen(" not in source.lower(),
        "ingress_left": "ingress_axis = figure.add_subplot(lower[0, 0])" in source,
        "egress_right": "egress_axis = figure.add_subplot(lower[0, 2])" in source,
        "zoom_label_offsets_custom": "zoom_offsets =" in source and '("PV", "C3"): (0.0, -18.0, False)' in source,
        "main_label_offsets_preserved": "y_shift = 36.0 if above else -36.0" in source,
        "table_degree_words_removed": "(degrees)" not in source,
        "arcsec_column_centered": 'table[(row_index, 2)].get_text().set_ha("center")' in source,
        "mirror_preserved": 'mirrored["points"][:, 0] *= -1.0' in source,
        "reference_style_preserved": 'SUN_FILL_COLOR = "#D95A1B"' in source and 'SUN_COLOR = "#FFD34A"' in source,
    }
    print("V0088 SOURCE AUDIT")
    for name, ok in checks.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"V0088 audit failed: {failed}")
    print("V0088 audit complete: PASS")


def main() -> None:
    source = build_v0087_expanded_source()
    source = patch_zoom_labels(source)
    source = patch_derivation_table_text(source)
    source = promote_to_v0088(source)
    audit_source(source)
    EXPANDED_OUT.write_text(source, encoding="utf-8")
    namespace = {"__name__": "__main__", "__file__": str(EXPANDED_OUT)}
    exec(compile(source, str(EXPANDED_OUT), "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
# V0088
