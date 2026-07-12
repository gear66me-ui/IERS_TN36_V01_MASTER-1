"""
IERS TN36 — Ecliptical Plane Analysis
Part N — Fixed black JPL equation slide
Imports the numerical derivation from Part M but replaces its broken renderer.
No AI images. Matplotlib only.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyBboxPatch
import numpy as np

import IERS_0001_M_JPL_EQUATION_SLIDE as source

VERSION = "IERS-0001-N"
OUTPUT_PNG = "IERS_0001_N_JPL_EQUATION_SLIDE_FIXED.png"


def vector_text(vector: np.ndarray, precision: int = 9) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in vector) + "]"


def panel(
    figure: plt.Figure,
    x: float,
    y: float,
    width: float,
    height: float,
    edge: str,
    face: str,
) -> None:
    figure.patches.append(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.006,rounding_size=0.010",
            linewidth=1.1,
            edgecolor=edge,
            facecolor=face,
            transform=figure.transFigure,
            clip_on=False,
        )
    )


def add_ray(
    axis: plt.Axes,
    angle_deg: float,
    length: float,
    color: str,
    linewidth: float,
) -> np.ndarray:
    angle_rad = np.radians(angle_deg)
    endpoint = length * np.array([np.cos(angle_rad), np.sin(angle_rad)])
    axis.annotate(
        "",
        xy=endpoint,
        xytext=(0.0, 0.0),
        arrowprops={
            "arrowstyle": "-|>",
            "linewidth": linewidth,
            "color": color,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    return endpoint


def render_slide(result: dict[str, object]) -> None:
    background = "#05070a"
    panel_face = "#090d13"
    white = "#f5f7fa"
    blue = "#36b7ff"
    green = "#7cf58a"
    gold = "#ffd24d"
    gray = "#aeb8c4"

    figure = plt.figure(figsize=(16, 9), facecolor=background)
    figure.patch.set_facecolor(background)

    figure.text(
        0.030,
        0.955,
        "ECLIPTICAL PLANE ANALYSIS",
        color=white,
        fontsize=24,
        fontweight="bold",
        va="top",
    )
    figure.text(
        0.030,
        0.915,
        "Exact JPL derivation of the local ECLIPTIC-to-ICRF screen rotation",
        color=blue,
        fontsize=14,
        va="top",
    )

    # Panel 1: obliquity and rotation matrix.
    panel(figure, 0.025, 0.720, 0.565, 0.160, blue, panel_face)
    figure.text(
        0.043,
        0.852,
        "1  J2000 obliquity and 3-D frame rotation",
        color=blue,
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.055,
        0.805,
        rf"$\varepsilon=\frac{{84381.448^{{\prime\prime}}}}{{3600}}={source.OBLIQUITY_DEG:.9f}^\circ$",
        color=white,
        fontsize=20,
    )
    figure.text(0.055, 0.757, r"$R_x(\varepsilon)=$", color=white, fontsize=18)
    figure.text(
        0.190,
        0.741,
        "⎡ 1          0              0       ⎤\n"
        "⎢ 0       cos ε         −sin ε     ⎥\n"
        "⎣ 0       sin ε          cos ε     ⎦",
        color=white,
        fontsize=14,
        family="monospace",
        linespacing=1.12,
    )

    # Panel 2: actual JPL Sun vectors and normalization divisions.
    panel(figure, 0.025, 0.465, 0.565, 0.230, blue, panel_face)
    figure.text(
        0.043,
        0.667,
        "2  JPL Sun vectors and explicit normalization",
        color=blue,
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.050,
        0.620,
        r"$\hat{\mathbf{n}}_E=\mathbf{S}_E/\|\mathbf{S}_E\|$",
        color=white,
        fontsize=18,
    )
    figure.text(
        0.050,
        0.584,
        f"S_E = {vector_text(result['sun_e'], 3)} km",
        color=gray,
        fontsize=10.5,
        family="monospace",
    )
    figure.text(
        0.050,
        0.550,
        f"||S_E|| = {result['norm_sun_e']:.6f} km",
        color=gray,
        fontsize=10.5,
        family="monospace",
    )
    figure.text(
        0.050,
        0.516,
        f"n_E = S_E / ||S_E|| = {vector_text(result['n_e'], 9)}",
        color=green,
        fontsize=10.5,
        family="monospace",
    )

    figure.text(
        0.322,
        0.620,
        r"$\hat{\mathbf{n}}_F=\mathbf{S}_F/\|\mathbf{S}_F\|$",
        color=white,
        fontsize=18,
    )
    figure.text(
        0.322,
        0.584,
        f"S_F = {vector_text(result['sun_f'], 3)} km",
        color=gray,
        fontsize=10.5,
        family="monospace",
    )
    figure.text(
        0.322,
        0.550,
        f"||S_F|| = {result['norm_sun_f']:.6f} km",
        color=gray,
        fontsize=10.5,
        family="monospace",
    )
    figure.text(
        0.322,
        0.516,
        f"n_F = S_F / ||S_F|| = {vector_text(result['n_f'], 9)}",
        color=gold,
        fontsize=10.5,
        family="monospace",
    )

    # Panel 3: axis construction and atan2 substitution.
    panel(figure, 0.025, 0.205, 0.565, 0.235, blue, panel_face)
    figure.text(
        0.043,
        0.412,
        "3  Solar-screen axes and signed local rotation",
        color=blue,
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.050,
        0.365,
        r"$\hat{\mathbf{x}}_E=\frac{\mathbf{k}\times\hat{\mathbf{n}}_E}{\|\mathbf{k}\times\hat{\mathbf{n}}_E\|},\qquad"
        r"\hat{\mathbf{x}}_F=\frac{\mathbf{k}\times\hat{\mathbf{n}}_F}{\|\mathbf{k}\times\hat{\mathbf{n}}_F\|}$",
        color=white,
        fontsize=16,
    )
    figure.text(
        0.050,
        0.313,
        r"$\phi_{\mathrm{axes}}=\mathrm{atan2}\left[\hat{\mathbf{n}}_F\cdot((R_x\hat{\mathbf{x}}_E)\times\hat{\mathbf{x}}_F),"
        r"\,(R_x\hat{\mathbf{x}}_E)\cdot\hat{\mathbf{x}}_F\right]$",
        color=white,
        fontsize=14,
    )
    figure.text(
        0.050,
        0.260,
        f"atan2({result['numerator']:.12f}, {result['denominator']:.12f})",
        color=gray,
        fontsize=12,
        family="monospace",
    )
    figure.text(
        0.350,
        0.255,
        rf"$\phi_{{\mathrm{{axes}}}}={result['phi_axes_deg']:.6f}^\circ$",
        color=gold,
        fontsize=20,
        fontweight="bold",
    )

    # Panel 4: track-angle transformation and zero closure.
    panel(figure, 0.025, 0.040, 0.565, 0.140, blue, panel_face)
    figure.text(
        0.043,
        0.151,
        "4  Coordinate-angle transformation",
        color=blue,
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.050,
        0.105,
        rf"$\Delta\theta=-\phi_{{\mathrm{{axes}}}}={result['coordinate_offset_deg']:.6f}^\circ$",
        color=green,
        fontsize=18,
    )
    figure.text(
        0.050,
        0.066,
        rf"$\theta_F=\theta_E+\Delta\theta={result['theta_e_deg']:.6f}^\circ+"
        rf"{result['coordinate_offset_deg']:.6f}^\circ={result['theta_f_from_rotation_deg']:.6f}^\circ$",
        color=white,
        fontsize=16,
    )
    figure.text(
        0.430,
        0.108,
        f"closure = {result['closure_arcsec']:.9f} arcsec",
        color=green,
        fontsize=11,
        family="monospace",
    )

    # Right-side angle diagram with labels intentionally separated.
    axis = figure.add_axes([0.625, 0.135, 0.345, 0.700], facecolor=background)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-0.18, 1.40)
    axis.set_ylim(-0.50, 0.72)
    axis.axis("off")
    axis.set_title(
        "One physical Venus track, two reference axes",
        color=white,
        fontsize=16,
        pad=20,
    )

    phi_axes = float(result["phi_axes_deg"])
    theta_e = float(result["theta_e_deg"])
    theta_f = float(result["theta_f_deg"])

    ecliptic_endpoint = add_ray(axis, 0.0, 1.12, blue, 2.6)
    icrf_endpoint = add_ray(axis, phi_axes, 1.03, gold, 2.6)
    track_endpoint = add_ray(axis, theta_e, 1.23, green, 3.2)

    axis.add_patch(
        Arc((0.0, 0.0), 0.48, 0.48, theta1=phi_axes, theta2=0.0, color=gold, linewidth=1.6)
    )
    axis.add_patch(
        Arc((0.0, 0.0), 0.86, 0.86, theta1=0.0, theta2=theta_e, color=blue, linewidth=1.6)
    )
    axis.add_patch(
        Arc((0.0, 0.0), 1.30, 1.30, theta1=phi_axes, theta2=theta_e, color=green, linewidth=1.6)
    )

    axis.text(
        ecliptic_endpoint[0] - 0.02,
        ecliptic_endpoint[1] - 0.10,
        "Ecliptic +X",
        color=blue,
        fontsize=13,
        ha="right",
    )
    axis.text(
        icrf_endpoint[0] - 0.01,
        icrf_endpoint[1] - 0.13,
        "ICRF +X",
        color=gold,
        fontsize=13,
        ha="right",
    )
    axis.text(
        track_endpoint[0] + 0.02,
        track_endpoint[1] + 0.03,
        "Venus track",
        color=green,
        fontsize=13,
        ha="left",
    )

    axis.annotate(
        rf"$\Delta\theta={result['coordinate_offset_deg']:.6f}^\circ$",
        xy=(0.25, -0.012),
        xytext=(-0.02, -0.36),
        color=gold,
        fontsize=13,
        arrowprops={"arrowstyle": "->", "color": gold, "linewidth": 1.2},
    )
    axis.annotate(
        rf"$\theta_E={theta_e:.6f}^\circ$",
        xy=(0.42, 0.035),
        xytext=(0.08, 0.50),
        color=blue,
        fontsize=13,
        arrowprops={"arrowstyle": "->", "color": blue, "linewidth": 1.2},
    )
    axis.annotate(
        rf"$\theta_F={theta_f:.6f}^\circ$",
        xy=(0.63, 0.025),
        xytext=(0.82, 0.56),
        color=green,
        fontsize=13,
        arrowprops={"arrowstyle": "->", "color": green, "linewidth": 1.2},
    )

    axis.scatter([0.0], [0.0], s=38, color=white, zorder=5)
    axis.text(
        0.11,
        -0.47,
        rf"$\theta_F-\theta_E={result['frame_difference_deg']:.6f}^\circ$\n"
        rf"$\Delta\theta={result['coordinate_offset_deg']:.6f}^\circ$\n"
        rf"$\mathrm{{closure}}={result['closure_arcsec']:.9f}\;\mathrm{{arcsec}}$",
        color=white,
        fontsize=13,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": panel_face,
            "edgecolor": blue,
        },
    )

    figure.savefig(
        OUTPUT_PNG,
        dpi=180,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
    )
    plt.show()
    print(f"\nPRESENTATION SLIDE: {OUTPUT_PNG}")


def main() -> None:
    result = source.derive_jpl_geometry()
    source.print_results(result)
    print(f"Renderer version             {VERSION}")
    render_slide(result)


if __name__ == "__main__":
    main()
