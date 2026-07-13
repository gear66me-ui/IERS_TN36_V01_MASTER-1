# V0068
# Audit reference: Add solar-limb arcs to both contact zooms, remove the floating track-geometry title, and preserve the verified V0067 JPL calculation.
from __future__ import annotations

import urllib.request

VERSION = "V0068"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "VENUS_1769_V0027_FORMAT_STANDALONE_V0067.py"
)


def fetch_source() -> str:
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "VENUS-TRANSIT-V0068"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        source = response.read().decode("utf-8")
    if "# V0067" not in source or "def plot_publication(" not in source:
        raise RuntimeError("The verified V0067 source could not be validated.")
    return source


def build_v0068_source(source: str) -> str:
    floating_title = '''    main_axis.text(
        -0.42 * reference_solar_radius,
        0.21 * reference_solar_radius,
        "TRACK GEOMETRY — A, B, A′, B′",
        fontsize=7.3,
        fontweight="bold",
        color=TEXT_COLOR,
    )
'''
    if source.count(floating_title) != 1:
        raise RuntimeError("Floating track-geometry title block was not found exactly once.")
    source = source.replace(floating_title, "", 1)

    zoom_anchor = '''    for zoom_axis, events, title in (
        (egress_axis, ("C3", "C4"), "EGRESS ZOOM — C3 / C4 TANGENCY"),
        (ingress_axis, ("C1", "C2"), "INGRESS ZOOM — C1 / C2 TANGENCY"),
    ):
        for result in (point_result, vardo_result):
            draw_track(zoom_axis, result, main=False)
            draw_events(zoom_axis, result, events, main=False)
'''
    zoom_replacement = '''    for zoom_axis, events, title in (
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
'''
    if source.count(zoom_anchor) != 1:
        raise RuntimeError("Zoom plotting block was not found exactly once.")
    source = source.replace(zoom_anchor, zoom_replacement, 1)

    source = source.replace("V0067", VERSION)
    source = source.replace(
        "# Audit reference: Correct both closest-approach solutions in seconds space, lower the derivation table, and match paired station-row colors.",
        "# Audit reference: Preserve corrected closest approaches; add solar-limb arcs to ingress and egress zooms; remove the floating track-geometry title.",
        1,
    )
    return source


def main() -> None:
    corrected_source = build_v0068_source(fetch_source())
    namespace = {
        "__name__": "__main__",
        "__file__": "VENUS_1769_V0027_FORMAT_STANDALONE_V0068.py",
    }
    exec(
        compile(
            corrected_source,
            "VENUS_1769_V0027_FORMAT_STANDALONE_V0068.py",
            "exec",
        ),
        namespace,
        namespace,
    )


if __name__ == "__main__":
    main()
# V0068
