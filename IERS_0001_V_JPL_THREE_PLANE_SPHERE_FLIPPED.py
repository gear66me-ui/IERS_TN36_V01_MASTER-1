"""
IERS TN36 — Ecliptical Plane Analysis
Part V — Flipped fine-line three-plane sphere geometry
No AI images. Matplotlib only.

Requires:
    IERS_0001_K_JPL_FRAME_ANGLE_CONFIRMATION.py
    IERS_0001_P_JPL_BLACK_VECTOR_EQUATIONS.py

All angles are derived at runtime from JPL ECLIPTIC and FRAME vectors.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import IERS_0001_P_JPL_BLACK_VECTOR_EQUATIONS as source

VERSION = "IERS-0001-V"
OUTPUT_PNG = "IERS_0001_V_JPL_THREE_PLANE_SPHERE_FLIPPED.png"
OUTPUT_PDF = "IERS_0001_V_JPL_THREE_PLANE_SPHERE_FLIPPED.pdf"

COL_ECLIPTIC = "#4EA3FF"
COL_CELESTIAL = "#FFB347"
COL_VENUS = "#4DFF88"
COL_TEXT = "white"
COL_SPHERE = "#D8D8D8"


def polar_xy(angle_deg: float, radius: float) -> tuple[float, float]:
    angle_rad = np.radians(angle_deg)
    return radius * np.cos(angle_rad), radius * np.sin(angle_rad)


def draw_diameter(
    axis: plt.Axes,
    angle_deg: float,
    radius: float,
    color: str,
    linewidth: float,
    linestyle: str,
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.asarray(polar_xy(angle_deg, radius))
    negative = -positive

    axis.plot(
        [negative[0], positive[0]],
        [negative[1], positive[1]],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        solid_capstyle="round",
        zorder=3,
    )

    axis.annotate(
        "",
        xy=positive,
        xytext=positive * 0.91,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": linewidth,
            "mutation_scale": 8,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=5,
    )
    return negative, positive


def draw_rotation_arc(
    axis: plt.Axes,
    start_deg: float,
    end_deg: float,
    radius: float,
    color: str,
    label: str,
    label_xy: tuple[float, float],
) -> None:
    samples = np.linspace(start_deg, end_deg, 180)
    points = np.array([polar_xy(angle, radius) for angle in samples])

    axis.plot(points[:, 0], points[:, 1], color=color, linewidth=0.85, zorder=6)
    axis.annotate(
        "",
        xy=points[-1],
        xytext=points[-12],
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": 0.85,
            "mutation_scale": 8,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=7,
    )

    axis.annotate(
        label,
        xy=points[len(points) // 2],
        xytext=label_xy,
        color=color,
        fontsize=13,
        ha="center",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "color": color,
            "linewidth": 0.55,
        },
        zorder=8,
    )


def render(result: dict[str, object]) -> None:
    theta_e = float(result["theta_e_deg"])
    theta_f = float(result["theta_f_deg"])
    delta_theta = float(result["delta_theta_deg"])
    closure_arcsec = float(result["closure_arcsec"])

    # Mirror the earlier diagram vertically so the physical transit direction is
    # left/up to right/down. The ICRF plane is mirrored with it, preserving all
    # angular magnitudes exactly.
    ecliptic_angle = 0.0
    celestial_angle = delta_theta
    venus_angle = -theta_e

    plt.close("all")
    figure = plt.figure(figsize=(14, 9), facecolor="black")
    axis = figure.add_axes([0.05, 0.08, 0.90, 0.82], facecolor="black")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-1.56, 1.60)
    axis.set_ylim(-1.30, 1.28)
    axis.axis("off")

    figure.text(
        0.05,
        0.965,
        "JPL THREE-PLANE SPHERE GEOMETRY",
        color=COL_TEXT,
        fontsize=24,
        fontweight="bold",
        ha="left",
        va="top",
    )
    figure.text(
        0.05,
        0.925,
        "Three colored diameters through one projected celestial sphere",
        color=COL_TEXT,
        fontsize=13,
        ha="left",
        va="top",
    )

    axis.add_patch(
        Circle(
            (0.0, 0.0),
            1.0,
            edgecolor=COL_SPHERE,
            facecolor="none",
            linewidth=0.95,
            zorder=2,
        )
    )

    draw_diameter(axis, ecliptic_angle, 1.12, COL_ECLIPTIC, 1.05, "-")
    draw_diameter(axis, celestial_angle, 1.12, COL_CELESTIAL, 0.95, "--")
    draw_diameter(axis, venus_angle, 1.12, COL_VENUS, 1.20, "-")

    axis.scatter([0.0], [0.0], s=18, color=COL_TEXT, zorder=9)

    # Left-side labels are deliberately separated and aligned with their planes.
    axis.text(
        -1.10,
        -0.075,
        "ECLIPTIC PLANE",
        color=COL_ECLIPTIC,
        fontsize=13.5,
        ha="right",
        va="top",
        bbox={"facecolor": "black", "edgecolor": "none", "pad": 1.0},
    )

    celestial_anchor = np.asarray(polar_xy(celestial_angle, -1.17))
    axis.text(
        celestial_anchor[0] - 0.02,
        celestial_anchor[1] - 0.015,
        "ICRF / CELESTIAL PLANE",
        color=COL_CELESTIAL,
        fontsize=13.5,
        ha="right",
        va="center",
        rotation=celestial_angle,
        rotation_mode="anchor",
        bbox={"facecolor": "black", "edgecolor": "none", "pad": 1.0},
    )

    venus_anchor = np.asarray(polar_xy(venus_angle, 1.12))
    axis.text(
        venus_anchor[0] + 0.03,
        venus_anchor[1] - 0.035,
        "VENUS TRANSIT PLANE",
        color=COL_VENUS,
        fontsize=13.5,
        ha="left",
        va="top",
        rotation=venus_angle,
        rotation_mode="anchor",
    )

    draw_rotation_arc(
        axis,
        ecliptic_angle,
        celestial_angle,
        0.30,
        COL_CELESTIAL,
        rf"$\Delta\theta={delta_theta:.6f}^\circ$",
        (-0.08, 0.45),
    )
    draw_rotation_arc(
        axis,
        ecliptic_angle,
        venus_angle,
        0.56,
        COL_ECLIPTIC,
        rf"$\theta_E={theta_e:.6f}^\circ$",
        (0.20, -0.54),
    )
    draw_rotation_arc(
        axis,
        celestial_angle,
        venus_angle,
        0.84,
        COL_VENUS,
        rf"$\theta_F={theta_f:.6f}^\circ$",
        (1.08, 0.10),
    )

    summary = (
        rf"$\theta_F=\theta_E+\Delta\theta$" "\n"
        rf"${theta_f:.6f}^\circ={theta_e:.6f}^\circ+{delta_theta:.6f}^\circ$" "\n"
        rf"$\mathrm{{closure}}={closure_arcsec:.9f}\;\mathrm{{arcsec}}$"
    )
    axis.text(
        0.34,
        -0.97,
        summary,
        color=COL_TEXT,
        fontsize=15.0,
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.48",
            "facecolor": "black",
            "edgecolor": COL_SPHERE,
            "linewidth": 0.85,
        },
    )

    legend = (
        "Blue solid diameter    : Ecliptic plane\n"
        "Orange dashed diameter : ICRF / celestial plane\n"
        "Green solid diameter   : Venus transit plane"
    )
    axis.text(
        -1.34,
        -0.97,
        legend,
        color=COL_TEXT,
        fontsize=11.5,
        ha="left",
        va="top",
    )

    figure.savefig(OUTPUT_PNG, dpi=220, facecolor="black", bbox_inches="tight")
    figure.savefig(OUTPUT_PDF, facecolor="black", bbox_inches="tight")
    plt.show()

    print(f"Version             {VERSION}")
    print(f"Ecliptic angle      {theta_e:.6f} deg")
    print(f"ICRF angle          {theta_f:.6f} deg")
    print(f"Frame offset        {delta_theta:.6f} deg")
    print(f"Closure             {closure_arcsec:.9f} arcsec")
    print(f"PNG                 {OUTPUT_PNG}")
    print(f"PDF                 {OUTPUT_PDF}")


def main() -> None:
    result = source.derive()
    render(result)


if __name__ == "__main__":
    main()
