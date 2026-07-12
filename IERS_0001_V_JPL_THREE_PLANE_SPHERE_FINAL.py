"""
IERS TN36 — Ecliptical Plane Analysis
Part V — Final three-plane JPL sphere geometry
No AI images. Matplotlib only.

Requires:
    IERS_0001_K_JPL_FRAME_ANGLE_CONFIRMATION.py
    IERS_0001_P_JPL_BLACK_VECTOR_EQUATIONS.py

All reported angles are derived at runtime from JPL ECLIPTIC and FRAME vectors.
No target angles or correction factors are supplied to the calculation.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import IERS_0001_P_JPL_BLACK_VECTOR_EQUATIONS as source

VERSION = "IERS-0001-V"
OUTPUT_PNG = "IERS_0001_V_JPL_THREE_PLANE_SPHERE_FINAL.png"
OUTPUT_PDF = "IERS_0001_V_JPL_THREE_PLANE_SPHERE_FINAL.pdf"

COL_ECLIPTIC = "#4EA3FF"
COL_CELESTIAL = "#FFB347"
COL_VENUS = "#4DFF88"
COL_TEXT = "#F4F4F4"
COL_GRID = "#BFC5CC"
COL_PANEL = "#080A0D"


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
        xytext=positive * 0.925,
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
) -> np.ndarray:
    samples = np.linspace(start_deg, end_deg, 180)
    points = np.array([polar_xy(angle, radius) for angle in samples])

    axis.plot(points[:, 0], points[:, 1], color=color, linewidth=0.75, zorder=6)
    axis.annotate(
        "",
        xy=points[-1],
        xytext=points[-13],
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": 0.75,
            "mutation_scale": 8,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=7,
    )
    return points


def add_bottom_table(
    figure: plt.Figure,
    theta_e: float,
    theta_f: float,
    delta_theta: float,
    closure_arcsec: float,
) -> None:
    table_axis = figure.add_axes([0.31, 0.035, 0.64, 0.175], facecolor="black")
    table_axis.axis("off")

    rows = [
        ["Ecliptic reference plane", "0.000000°", "reference datum"],
        ["ICRF / celestial plane", f"+{delta_theta:.6f}°", "JPL projected frame rotation"],
        ["Venus transit plane", f"−{theta_e:.6f}°", "JPL ECLIPTIC track fit"],
        ["Venus–ICRF separation", f"{theta_f:.6f}°", "JPL FRAME track fit"],
        [
            "Closure",
            f"{theta_f:.6f}° − {theta_e:.6f}° = {delta_theta:.6f}°",
            f"{closure_arcsec:.9f} arcsec",
        ],
    ]

    table = table_axis.table(
        cellText=rows,
        colLabels=["PLANE / RELATION", "JPL-DERIVED ANGLE", "DERIVATION"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.31, 0.37, 0.32],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.32)

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor(COL_PANEL)
        cell.set_edgecolor("#5C626A")
        cell.set_linewidth(0.55)
        cell.get_text().set_color(COL_TEXT)
        if row == 0:
            cell.get_text().set_weight("bold")
            cell.set_facecolor("#11151A")

    table[(1, 0)].get_text().set_color(COL_ECLIPTIC)
    table[(2, 0)].get_text().set_color(COL_CELESTIAL)
    table[(3, 0)].get_text().set_color(COL_VENUS)
    table[(5, 1)].get_text().set_weight("bold")
    table[(5, 2)].get_text().set_weight("bold")


def render(result: dict[str, object]) -> None:
    theta_e = float(result["theta_e_deg"])
    theta_f = float(result["theta_f_deg"])
    delta_theta = float(result["delta_theta_deg"])
    closure_arcsec = float(result["closure_arcsec"])

    # Plot orientation only: no numerical angle is supplied manually.
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
        color=COL_TEXT,
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
            edgecolor=COL_GRID,
            facecolor="none",
            linewidth=0.80,
            zorder=2,
        )
    )

    draw_diameter(axis, ecliptic_angle, 1.12, COL_ECLIPTIC, 0.95, "-")
    draw_diameter(axis, celestial_angle, 1.12, COL_CELESTIAL, 0.88, "--")
    draw_diameter(axis, venus_angle, 1.12, COL_VENUS, 1.05, "-")

    axis.scatter([0.0], [0.0], s=15, color=COL_TEXT, zorder=9)

    # Labels repositioned to avoid all line overlap.
    axis.text(
        -1.05,
        -0.075,
        "ECLIPTIC PLANE",
        color=COL_ECLIPTIC,
        fontsize=13.5,
        ha="right",
        va="top",
        bbox={"facecolor": "black", "edgecolor": "none", "pad": 1.0},
    )

    celestial_anchor = np.asarray(polar_xy(celestial_angle, -1.09))
    axis.text(
        celestial_anchor[0] - 0.025,
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
        venus_anchor[0] + 0.035,
        venus_anchor[1] - 0.025,
        "VENUS TRANSIT PLANE",
        color=COL_VENUS,
        fontsize=13.5,
        ha="left",
        va="top",
        rotation=venus_angle,
        rotation_mode="anchor",
    )

    delta_arc = draw_rotation_arc(
        axis, ecliptic_angle, celestial_angle, 0.31, COL_CELESTIAL
    )
    theta_e_arc = draw_rotation_arc(
        axis, venus_angle, ecliptic_angle, 0.56, COL_ECLIPTIC
    )
    theta_f_arc = draw_rotation_arc(
        axis, venus_angle, celestial_angle, 0.84, COL_VENUS
    )

    axis.annotate(
        rf"$\Delta\theta={delta_theta:.6f}^\circ$",
        xy=delta_arc[len(delta_arc) // 2],
        xytext=(0.03, 0.43),
        color=COL_CELESTIAL,
        fontsize=13,
        ha="center",
        arrowprops={"arrowstyle": "-", "color": COL_CELESTIAL, "linewidth": 0.55},
    )

    axis.annotate(
        rf"$\theta_E={theta_e:.6f}^\circ$",
        xy=theta_e_arc[len(theta_e_arc) // 2],
        xytext=(0.16, -0.50),
        color=COL_ECLIPTIC,
        fontsize=13,
        ha="center",
        arrowprops={"arrowstyle": "-", "color": COL_ECLIPTIC, "linewidth": 0.55},
    )

    # The full 14-degree ICRF angle is outside the circle, above the descending Venus line.
    axis.annotate(
        rf"$\theta_F={theta_f:.6f}^\circ$",
        xy=theta_f_arc[len(theta_f_arc) // 2],
        xytext=(1.18, 0.18),
        color=COL_VENUS,
        fontsize=13.5,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": COL_VENUS, "linewidth": 0.60},
    )

    add_bottom_table(figure, theta_e, theta_f, delta_theta, closure_arcsec)

    figure.savefig(OUTPUT_PNG, dpi=220, facecolor="black", bbox_inches="tight")
    figure.savefig(OUTPUT_PDF, facecolor="black", bbox_inches="tight")
    plt.show()

    print(f"Version                    {VERSION}")
    print(f"Ecliptic reference         {ecliptic_angle:.6f} deg")
    print(f"Celestial plane            {celestial_angle:.6f} deg")
    print(f"Venus transit plane        {venus_angle:.6f} deg")
    print(f"Venus-ICRF separation      {theta_f:.6f} deg")
    print(f"14.646192 - 8.485527       {theta_f - theta_e:.6f} deg")
    print(f"Frame rotation             {delta_theta:.6f} deg")
    print(f"Closure                    {closure_arcsec:.9f} arcsec")
    print(f"PNG                        {OUTPUT_PNG}")
    print(f"PDF                        {OUTPUT_PDF}")


def main() -> None:
    result = source.derive()
    render(result)


if __name__ == "__main__":
    main()
