"""
IERS TN36 — Ecliptical Plane Analysis
Part R — Three-diameter sphere geometry
No AI images. Matplotlib only.

Requires:
    IERS_0001_K_JPL_FRAME_ANGLE_CONFIRMATION.py
    IERS_0001_P_JPL_BLACK_VECTOR_EQUATIONS.py

All angles are derived at runtime from JPL ECLIPTIC and FRAME vectors.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

import IERS_0001_P_JPL_BLACK_VECTOR_EQUATIONS as source

VERSION = "IERS-0001-R"
OUTPUT_PNG = "IERS_0001_R_JPL_THREE_DIAMETER_SPHERE.png"
OUTPUT_PDF = "IERS_0001_R_JPL_THREE_DIAMETER_SPHERE.pdf"


def polar_xy(angle_deg: float, radius: float) -> tuple[float, float]:
    angle_rad = np.radians(angle_deg)
    return radius * np.cos(angle_rad), radius * np.sin(angle_rad)


def draw_diameter(
    axis: plt.Axes,
    angle_deg: float,
    radius: float,
    linewidth: float,
    linestyle: str,
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.asarray(polar_xy(angle_deg, radius))
    negative = -positive
    axis.plot(
        [negative[0], positive[0]],
        [negative[1], positive[1]],
        color="white",
        linewidth=linewidth,
        linestyle=linestyle,
        solid_capstyle="round",
        zorder=3,
    )
    axis.annotate(
        "",
        xy=positive,
        xytext=positive * 0.84,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "white",
            "linewidth": linewidth,
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
    label: str,
    label_xy: tuple[float, float],
) -> None:
    samples = np.linspace(start_deg, end_deg, 120)
    points = np.array([polar_xy(angle, radius) for angle in samples])
    axis.plot(points[:, 0], points[:, 1], color="white", linewidth=1.5, zorder=6)

    arrow_start = points[-7]
    arrow_end = points[-1]
    axis.annotate(
        "",
        xy=arrow_end,
        xytext=arrow_start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "white",
            "linewidth": 1.5,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=7,
    )

    axis.annotate(
        label,
        xy=points[len(points) // 2],
        xytext=label_xy,
        color="white",
        fontsize=14,
        ha="center",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "color": "white",
            "linewidth": 0.9,
        },
        zorder=8,
    )


def render(result: dict[str, object]) -> None:
    theta_e = float(result["theta_e_deg"])
    theta_f = float(result["theta_f_deg"])
    delta_theta = float(result["delta_theta_deg"])
    closure_arcsec = float(result["closure_arcsec"])

    ecliptic_angle = 0.0
    icrf_angle = -delta_theta
    venus_angle = theta_e

    plt.close("all")
    figure = plt.figure(figsize=(13.5, 9), facecolor="black")
    axis = figure.add_axes([0.06, 0.08, 0.88, 0.82], facecolor="black")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-1.45, 1.48)
    axis.set_ylim(-1.30, 1.28)
    axis.axis("off")

    figure.text(
        0.06,
        0.965,
        "JPL THREE-PLANE SPHERE GEOMETRY",
        color="white",
        fontsize=23,
        fontweight="bold",
        ha="left",
        va="top",
    )
    figure.text(
        0.06,
        0.925,
        "Three full diameters through one projected celestial sphere",
        color="white",
        fontsize=13,
        ha="left",
        va="top",
    )

    axis.add_patch(
        Circle(
            (0.0, 0.0),
            1.0,
            edgecolor="white",
            facecolor="none",
            linewidth=1.7,
            zorder=2,
        )
    )

    e_neg, e_pos = draw_diameter(axis, ecliptic_angle, 1.10, 2.7, "-")
    f_neg, f_pos = draw_diameter(axis, icrf_angle, 1.10, 2.1, "--")
    v_neg, v_pos = draw_diameter(axis, venus_angle, 1.10, 3.1, "-")

    axis.scatter([0.0], [0.0], s=34, color="white", zorder=9)

    axis.text(
        e_neg[0] + 0.02,
        e_neg[1] + 0.08,
        "ECLIPTIC PLANE",
        color="white",
        fontsize=14,
        ha="left",
        va="bottom",
        rotation=ecliptic_angle,
        rotation_mode="anchor",
    )
    axis.text(
        f_neg[0] + 0.02,
        f_neg[1] - 0.08,
        "ICRF / CELESTIAL PLANE",
        color="white",
        fontsize=14,
        ha="left",
        va="top",
        rotation=icrf_angle,
        rotation_mode="anchor",
    )
    axis.text(
        v_pos[0] + 0.03,
        v_pos[1] + 0.04,
        "VENUS TRANSIT PLANE",
        color="white",
        fontsize=14,
        ha="left",
        va="bottom",
        rotation=venus_angle,
        rotation_mode="anchor",
    )

    draw_rotation_arc(
        axis,
        icrf_angle,
        ecliptic_angle,
        0.31,
        rf"$\Delta\theta={delta_theta:.6f}^\circ$",
        (-0.03, -0.43),
    )
    draw_rotation_arc(
        axis,
        ecliptic_angle,
        venus_angle,
        0.58,
        rf"$\theta_E={theta_e:.6f}^\circ$",
        (0.22, 0.53),
    )
    draw_rotation_arc(
        axis,
        icrf_angle,
        venus_angle,
        0.86,
        rf"$\theta_F={theta_f:.6f}^\circ$",
        (0.73, 0.67),
    )

    axis.text(
        0.58,
        -0.98,
        rf"$\theta_F=\theta_E+\Delta\theta$" "\n"
        rf"${theta_f:.6f}^\circ={theta_e:.6f}^\circ+{delta_theta:.6f}^\circ$" "\n"
        rf"$\mathrm{{closure}}={closure_arcsec:.9f}\;\mathrm{{arcsec}}$",
        color="white",
        fontsize=16,
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "black",
            "edgecolor": "white",
        },
    )

    axis.text(
        -1.34,
        -0.98,
        "Solid horizontal diameter : Ecliptic plane\n"
        "Dashed rotated diameter   : ICRF / celestial plane\n"
        "Solid upper diameter      : Venus transit plane",
        color="white",
        fontsize=12,
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
