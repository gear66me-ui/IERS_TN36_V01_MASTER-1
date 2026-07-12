"""
IERS TN36 — Ecliptical Plane Analysis
Part Q — One-circle JPL frame-angle geometry
No AI images. Matplotlib only.

Requires:
    IERS_0001_K_JPL_FRAME_ANGLE_CONFIRMATION.py
    IERS_0001_P_JPL_BLACK_VECTOR_EQUATIONS.py

All angles are derived at runtime from the JPL ECLIPTIC and FRAME vectors.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
import numpy as np

import IERS_0001_P_JPL_BLACK_VECTOR_EQUATIONS as source

VERSION = "IERS-0001-Q"
OUTPUT_PNG = "IERS_0001_Q_JPL_ONE_CIRCLE_GEOMETRY.png"
OUTPUT_PDF = "IERS_0001_Q_JPL_ONE_CIRCLE_GEOMETRY.pdf"


def polar_xy(angle_deg: float, radius: float) -> tuple[float, float]:
    angle_rad = np.radians(angle_deg)
    return radius * np.cos(angle_rad), radius * np.sin(angle_rad)


def draw_ray(
    axis: plt.Axes,
    angle_deg: float,
    radius: float,
    linewidth: float,
    linestyle: str = "-",
) -> tuple[float, float]:
    endpoint = polar_xy(angle_deg, radius)
    axis.annotate(
        "",
        xy=endpoint,
        xytext=(0.0, 0.0),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "white",
            "linewidth": linewidth,
            "linestyle": linestyle,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=5,
    )
    return endpoint


def label_arc(
    axis: plt.Axes,
    start_deg: float,
    end_deg: float,
    radius: float,
    text: str,
    text_angle_deg: float,
    text_radius: float,
    text_offset: tuple[float, float],
) -> None:
    theta1, theta2 = sorted((start_deg, end_deg))
    axis.add_patch(
        Arc(
            (0.0, 0.0),
            2.0 * radius,
            2.0 * radius,
            theta1=theta1,
            theta2=theta2,
            color="white",
            linewidth=1.4,
        )
    )
    x_text, y_text = polar_xy(text_angle_deg, text_radius)
    axis.text(
        x_text + text_offset[0],
        y_text + text_offset[1],
        text,
        color="white",
        fontsize=14,
        ha="center",
        va="center",
    )


def render(result: dict[str, object]) -> None:
    theta_e = float(result["theta_e_deg"])
    theta_f = float(result["theta_f_deg"])
    delta_theta = float(result["delta_theta_deg"])
    closure_arcsec = float(result["closure_arcsec"])

    ecliptic_axis = 0.0
    icrf_axis = -delta_theta
    venus_track = theta_e

    plt.close("all")
    figure = plt.figure(figsize=(13, 9), facecolor="black")
    axis = figure.add_axes([0.06, 0.09, 0.88, 0.82], facecolor="black")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-1.42, 1.52)
    axis.set_ylim(-1.22, 1.25)
    axis.axis("off")

    figure.text(
        0.06,
        0.965,
        "JPL SOLAR-SCREEN ANGLE GEOMETRY",
        color="white",
        fontsize=23,
        fontweight="bold",
        ha="left",
        va="top",
    )
    figure.text(
        0.06,
        0.925,
        "One Venus transit track measured from the ECLIPTIC and ICRF reference axes",
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
            linewidth=1.5,
        )
    )

    e_end = draw_ray(axis, ecliptic_axis, 1.14, 2.6, "-")
    f_end = draw_ray(axis, icrf_axis, 1.05, 2.1, "--")
    v_end = draw_ray(axis, venus_track, 1.22, 3.2, "-")

    axis.scatter([0.0], [0.0], s=34, color="white", zorder=8)

    axis.text(
        e_end[0] - 0.02,
        e_end[1] - 0.10,
        "Ecliptic reference axis",
        color="white",
        fontsize=14,
        ha="right",
        va="top",
    )
    axis.text(
        f_end[0] - 0.02,
        f_end[1] - 0.14,
        "ICRF / celestial reference axis",
        color="white",
        fontsize=14,
        ha="right",
        va="top",
    )
    axis.text(
        v_end[0] + 0.03,
        v_end[1] + 0.04,
        "Venus transit track",
        color="white",
        fontsize=15,
        ha="left",
        va="bottom",
    )

    label_arc(
        axis,
        icrf_axis,
        ecliptic_axis,
        0.28,
        rf"$\Delta\theta={delta_theta:.6f}^\circ$",
        -0.5 * delta_theta,
        0.43,
        (-0.05, -0.12),
    )
    label_arc(
        axis,
        ecliptic_axis,
        venus_track,
        0.55,
        rf"$\theta_E={theta_e:.6f}^\circ$",
        0.5 * theta_e,
        0.73,
        (-0.03, 0.10),
    )
    label_arc(
        axis,
        icrf_axis,
        venus_track,
        0.82,
        rf"$\theta_F={theta_f:.6f}^\circ$",
        0.5 * (icrf_axis + venus_track),
        1.01,
        (0.15, 0.13),
    )

    summary = (
        rf"$\theta_F=\theta_E+\Delta\theta$" "\n"
        rf"${theta_f:.6f}^\circ={theta_e:.6f}^\circ+{delta_theta:.6f}^\circ$" "\n"
        rf"$\mathrm{{closure}}={closure_arcsec:.9f}\;\mathrm{{arcsec}}$"
    )
    axis.text(
        0.53,
        -0.88,
        summary,
        color="white",
        fontsize=16,
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "black",
            "edgecolor": "white",
        },
    )

    legend = (
        "Solid horizontal ray : Ecliptic reference\n"
        "Dashed lower ray     : ICRF reference\n"
        "Solid upper ray      : Venus transit track"
    )
    axis.text(
        -1.32,
        -0.88,
        legend,
        color="white",
        fontsize=12,
        ha="left",
        va="top",
    )

    figure.savefig(OUTPUT_PNG, dpi=220, facecolor="black", bbox_inches="tight")
    figure.savefig(OUTPUT_PDF, facecolor="black", bbox_inches="tight")
    plt.show()

    print(f"Version : {VERSION}")
    print(f"Ecliptic angle     {theta_e:.6f} deg")
    print(f"ICRF angle         {theta_f:.6f} deg")
    print(f"Frame offset       {delta_theta:.6f} deg")
    print(f"Closure            {closure_arcsec:.9f} arcsec")
    print(f"PNG                {OUTPUT_PNG}")
    print(f"PDF                {OUTPUT_PDF}")


def main() -> None:
    result = source.derive()
    render(result)


if __name__ == "__main__":
    main()
