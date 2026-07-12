"""
IERS TN36 — Ecliptical Plane Analysis
Part W — Ecliptic-label position correction
No AI images. Matplotlib only.

Requires:
    IERS_0001_K_JPL_FRAME_ANGLE_CONFIRMATION.py
    IERS_0001_P_JPL_BLACK_VECTOR_EQUATIONS.py
    IERS_0001_V_JPL_THREE_PLANE_SPHERE_FINAL.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import IERS_0001_V_JPL_THREE_PLANE_SPHERE_FINAL as base

VERSION = "IERS-0001-W"
OUTPUT_PNG = "IERS_0001_W_JPL_THREE_PLANE_LABEL_FIX.png"
OUTPUT_PDF = "IERS_0001_W_JPL_THREE_PLANE_LABEL_FIX.pdf"


def render(result: dict[str, object]) -> None:
    theta_e = float(result["theta_e_deg"])
    theta_f = float(result["theta_f_deg"])
    delta_theta = float(result["delta_theta_deg"])
    closure_arcsec = float(result["closure_arcsec"])

    ecliptic_angle = theta_e - theta_e
    celestial_angle = delta_theta
    venus_angle = -theta_e

    plt.close("all")
    figure = plt.figure(figsize=(14, 9), facecolor="black")
    axis = figure.add_axes([0.045, 0.215, 0.91, 0.695], facecolor="black")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-1.48, 1.52)
    axis.set_ylim(-1.10, 1.16)
    axis.axis("off")

    figure.text(
        0.045,
        0.965,
        "JPL THREE-PLANE SPHERE GEOMETRY",
        color=base.COL_TEXT,
        fontsize=23,
        fontweight="bold",
        ha="left",
        va="top",
    )
    figure.text(
        0.045,
        0.927,
        "All plane inclinations derived directly from JPL ECLIPTIC and FRAME vectors",
        color="#D6D6D6",
        fontsize=12.5,
        ha="left",
        va="top",
    )

    axis.add_patch(
        Circle(
            (0.0, 0.0),
            1.0,
            edgecolor=base.COL_GRID,
            facecolor="none",
            linewidth=0.80,
            zorder=2,
        )
    )

    base.draw_diameter(axis, ecliptic_angle, 1.12, base.COL_ECLIPTIC, 0.95, "-")
    base.draw_diameter(axis, celestial_angle, 1.12, base.COL_CELESTIAL, 0.88, "--")
    base.draw_diameter(axis, venus_angle, 1.12, base.COL_VENUS, 1.05, "-")

    axis.scatter([0.0], [0.0], s=15, color=base.COL_TEXT, zorder=9)

    # Moved farther left and upward so it clears the celestial-plane label.
    axis.text(
        -1.18,
        0.145,
        "ECLIPTIC PLANE",
        color=base.COL_ECLIPTIC,
        fontsize=13.5,
        ha="right",
        va="bottom",
        bbox={"facecolor": "black", "edgecolor": "none", "pad": 1.0},
    )

    celestial_anchor = np.asarray(base.polar_xy(celestial_angle, -1.09))
    axis.text(
        celestial_anchor[0] - 0.025,
        celestial_anchor[1] - 0.015,
        "ICRF / CELESTIAL PLANE",
        color=base.COL_CELESTIAL,
        fontsize=13.5,
        ha="right",
        va="center",
        rotation=celestial_angle,
        rotation_mode="anchor",
        bbox={"facecolor": "black", "edgecolor": "none", "pad": 1.0},
    )

    venus_anchor = np.asarray(base.polar_xy(venus_angle, 1.12))
    axis.text(
        venus_anchor[0] + 0.035,
        venus_anchor[1] - 0.025,
        "VENUS TRANSIT PLANE",
        color=base.COL_VENUS,
        fontsize=13.5,
        ha="left",
        va="top",
        rotation=venus_angle,
        rotation_mode="anchor",
    )

    delta_arc = base.draw_rotation_arc(
        axis, ecliptic_angle, celestial_angle, 0.31, base.COL_CELESTIAL
    )
    theta_e_arc = base.draw_rotation_arc(
        axis, venus_angle, ecliptic_angle, 0.56, base.COL_ECLIPTIC
    )
    theta_f_arc = base.draw_rotation_arc(
        axis, venus_angle, celestial_angle, 0.84, base.COL_VENUS
    )

    axis.annotate(
        rf"$\Delta\theta={delta_theta:.6f}^\circ$",
        xy=delta_arc[len(delta_arc) // 2],
        xytext=(0.03, 0.43),
        color=base.COL_CELESTIAL,
        fontsize=13,
        ha="center",
        arrowprops={"arrowstyle": "-", "color": base.COL_CELESTIAL, "linewidth": 0.55},
    )
    axis.annotate(
        rf"$\theta_E={theta_e:.6f}^\circ$",
        xy=theta_e_arc[len(theta_e_arc) // 2],
        xytext=(0.16, -0.50),
        color=base.COL_ECLIPTIC,
        fontsize=13,
        ha="center",
        arrowprops={"arrowstyle": "-", "color": base.COL_ECLIPTIC, "linewidth": 0.55},
    )
    axis.annotate(
        rf"$\theta_F={theta_f:.6f}^\circ$",
        xy=theta_f_arc[len(theta_f_arc) // 2],
        xytext=(1.18, 0.18),
        color=base.COL_VENUS,
        fontsize=13.5,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": base.COL_VENUS, "linewidth": 0.60},
    )

    base.add_bottom_table(figure, theta_e, theta_f, delta_theta, closure_arcsec)

    figure.savefig(OUTPUT_PNG, dpi=220, facecolor="black", bbox_inches="tight")
    figure.savefig(OUTPUT_PDF, facecolor="black", bbox_inches="tight")
    plt.show()

    print(f"Version                    {VERSION}")
    print(f"Ecliptic reference         {ecliptic_angle:.6f} deg")
    print(f"Celestial plane            {celestial_angle:.6f} deg")
    print(f"Venus transit plane        {venus_angle:.6f} deg")
    print(f"Venus-ICRF separation      {theta_f:.6f} deg")
    print(f"Frame rotation             {delta_theta:.6f} deg")
    print(f"Closure                    {closure_arcsec:.9f} arcsec")
    print(f"PNG                        {OUTPUT_PNG}")
    print(f"PDF                        {OUTPUT_PDF}")


def main() -> None:
    result = base.source.derive()
    render(result)


if __name__ == "__main__":
    main()
