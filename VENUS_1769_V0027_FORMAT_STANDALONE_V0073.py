# V0073
# Audit reference: Plot-only correction from verified V0067; red-orange solar limb, zoom limbs restored, center label removed, geometry unchanged.
from __future__ import annotations

import re
import urllib.request

VERSION = "V0073"
HOT_SOLAR_LIMB = "#FF4B1F"
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


def set_hot_solar_limb(source: str) -> str:
    source = re.sub(
        r'SUN_COLOR\s*=\s*"#[0-9A-Fa-f]{6}"',
        f'SUN_COLOR = "{HOT_SOLAR_LIMB}"',
        source,
        count=1,
    )
    return source


def add_zoom_solar_limb(source: str) -> str:
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


def build_v0073_source() -> str:
    source = fetch_source()
    source = set_hot_solar_limb(source)
    source = remove_floating_track_geometry_label(source)
    source = add_zoom_solar_limb(source)
    source = source.replace("V0067", VERSION)
    source = source.replace(
        "# Audit reference: Correct both closest-approach solutions in seconds space, lower the derivation table, and match paired station-row colors.",
        "# Audit reference: Plot-only correction from verified V0067; red-orange solar limb, zoom limbs restored, center label removed, geometry unchanged.",
    )
    return source


def main() -> None:
    source = build_v0073_source()
    namespace = {"__name__": "__main__", "__file__": "VENUS_1769_V0027_FORMAT_STANDALONE_V0073.py"}
    exec(compile(source, "VENUS_1769_V0027_FORMAT_STANDALONE_V0073.py", "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
# V0073
