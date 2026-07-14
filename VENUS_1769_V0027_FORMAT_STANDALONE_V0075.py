# V0075
# Audit reference: Plot-only correction from verified V0067; warm yellow limb, orange translucent solar fill, thicker limb, zoom limbs restored, center label removed, geometry unchanged.
from __future__ import annotations

import re
import urllib.request

VERSION = "V0075"
SOLAR_LIMB_COLOR = "#FFD34A"
SOLAR_FILL_COLOR = "#FF9A1F"
SOLAR_FILL_ALPHA = 0.140
LIMB_WIDTH_FACTOR = 1.875
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "VENUS_1769_V0027_FORMAT_STANDALONE_V0067.py"
)


def fetch_source() -> str:
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": VERSION})
    with urllib.request.urlopen(request, timeout=60) as response:
        source = response.read().decode("utf-8")
    if "# V0067" not in source or "def plot_publication(" not in source:
        raise RuntimeError("Verified V0067 source was not loaded correctly.")
    return source


def remove_floating_track_geometry_label(source: str) -> str:
    pattern = re.compile(
        r"\n\s*main_axis\.text\(\n"
        r"\s*-0\.42 \* reference_solar_radius,\n"
        r"\s*0\.21 \* reference_solar_radius,\n"
        r"\s*\"TRACK GEOMETRY — A, B, A′, B′\",\n"
        r"\s*fontsize=7\.3,\n"
        r"\s*fontweight=\"bold\",\n"
        r"\s*color=TEXT_COLOR,\n"
        r"\s*\)\n",
        re.MULTILINE,
    )
    return pattern.sub("\n", source)


def patch_solar_constants(source: str) -> str:
    source = re.sub(
        r'SUN_COLOR\s*=\s*"#[0-9A-Fa-f]{6}"',
        f'SUN_COLOR = "{SOLAR_LIMB_COLOR}"\nSUN_FILL_COLOR = "{SOLAR_FILL_COLOR}"\nSUN_FILL_ALPHA = {SOLAR_FILL_ALPHA:.3f}',
        source,
        count=1,
    )
    width_match = re.search(r"SUN_LINE_WIDTH\s*=\s*([0-9]+(?:\.[0-9]+)?)", source)
    if width_match is None:
        raise RuntimeError("SUN_LINE_WIDTH was not found; no unsafe partial patch applied.")
    old_width = float(width_match.group(1))
    new_width = old_width * LIMB_WIDTH_FACTOR
    source = re.sub(
        r"SUN_LINE_WIDTH\s*=\s*[0-9]+(?:\.[0-9]+)?",
        f"SUN_LINE_WIDTH = {new_width:.6f}",
        source,
        count=1,
    )
    return source


def patch_main_solar_limb(source: str) -> str:
    old = '''    main_axis.plot(
        reference_solar_radius * np.cos(theta),
        reference_solar_radius * np.sin(theta),
        color=SUN_COLOR,
        linewidth=SUN_LINE_WIDTH,
        zorder=1,
    )
'''
    new = '''    main_axis.add_patch(
        Circle(
            (0.0, 0.0),
            reference_solar_radius,
            facecolor=SUN_FILL_COLOR,
            edgecolor="none",
            alpha=SUN_FILL_ALPHA,
            zorder=0,
        )
    )
    main_axis.plot(
        reference_solar_radius * np.cos(theta),
        reference_solar_radius * np.sin(theta),
        color=SUN_COLOR,
        linewidth=SUN_LINE_WIDTH,
        zorder=1,
    )
'''
    if old not in source:
        if "SUN_FILL_COLOR" in source and "reference_solar_radius * np.cos(theta)" in source:
            return source
        raise RuntimeError("Main solar limb plotting block was not found; no unsafe partial patch applied.")
    return source.replace(old, new, 1)


def patch_zoom_solar_limb(source: str) -> str:
    old = '''    for zoom_axis, events, title in (
        (egress_axis, ("C3", "C4"), "EGRESS ZOOM — C3 / C4 TANGENCY"),
        (ingress_axis, ("C1", "C2"), "INGRESS ZOOM — C1 / C2 TANGENCY"),
    ):
        for result in (point_result, vardo_result):
            draw_track(zoom_axis, result, main=False)
            draw_events(zoom_axis, result, events, main=False)
        x_limits, y_limits = zoom_limits((point_result, vardo_result), events)
        zoom_axis.set_xlim(*x_limits)
        zoom_axis.set_ylim(*y_limits)
        zoom_axis.set_aspect("equal", adjustable="box")
        zoom_axis.set_title(title, fontsize=6.4, pad=3)
'''
    new = '''    for zoom_axis, events, title in (
        (egress_axis, ("C3", "C4"), "EGRESS ZOOM — C3 / C4 TANGENCY"),
        (ingress_axis, ("C1", "C2"), "INGRESS ZOOM — C1 / C2 TANGENCY"),
    ):
        zoom_solar_radius = float(
            np.mean(
                [
                    float(result["event_radii"][event][0])
                    for result in (point_result, vardo_result)
                    for event in events
                ]
            )
        )
        zoom_axis.add_patch(
            Circle(
                (0.0, 0.0),
                zoom_solar_radius,
                facecolor=SUN_FILL_COLOR,
                edgecolor="none",
                alpha=SUN_FILL_ALPHA,
                zorder=0,
            )
        )
        zoom_axis.plot(
            zoom_solar_radius * np.cos(theta),
            zoom_solar_radius * np.sin(theta),
            color=SUN_COLOR,
            linewidth=SUN_LINE_WIDTH,
            zorder=2,
        )
        for result in (point_result, vardo_result):
            draw_track(zoom_axis, result, main=False)
            draw_events(zoom_axis, result, events, main=False)
        x_limits, y_limits = zoom_limits((point_result, vardo_result), events)
        zoom_axis.set_xlim(*x_limits)
        zoom_axis.set_ylim(*y_limits)
        zoom_axis.set_aspect("equal", adjustable="box")
        zoom_axis.set_title(title, fontsize=6.4, pad=3)
'''
    if old not in source:
        if "zoom_solar_radius" in source:
            return source
        raise RuntimeError("Zoom plotting block was not found; no unsafe partial patch applied.")
    return source.replace(old, new, 1)


def build_v0075_source() -> str:
    source = fetch_source()
    source = patch_solar_constants(source)
    source = remove_floating_track_geometry_label(source)
    source = patch_main_solar_limb(source)
    source = patch_zoom_solar_limb(source)
    source = source.replace("V0067", VERSION)
    source = source.replace(
        "# Audit reference: Correct both closest-approach solutions in seconds space, lower the derivation table, and match paired station-row colors.",
        "# Audit reference: Plot-only correction from verified V0067; warm yellow limb, orange translucent solar fill, thicker limb, zoom limbs restored, center label removed, geometry unchanged.",
    )
    return source


def main() -> None:
    source = build_v0075_source()
    namespace = {"__name__": "__main__", "__file__": "VENUS_1769_V0027_FORMAT_STANDALONE_V0075.py"}
    exec(compile(source, "VENUS_1769_V0027_FORMAT_STANDALONE_V0075.py", "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
# V0075
