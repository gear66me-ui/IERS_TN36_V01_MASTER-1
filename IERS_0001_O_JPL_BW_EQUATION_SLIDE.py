"""
IERS TN36 — Ecliptical Plane Analysis
Part O — Black-and-white JPL equation derivation slide
No AI images. Matplotlib only.
Requires IERS_0001_K_JPL_FRAME_ANGLE_CONFIRMATION.py in the same folder.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyBboxPatch
import numpy as np

import IERS_0001_K_JPL_FRAME_ANGLE_CONFIRMATION as base

VERSION = "IERS-0001-O"
OBLIQUITY_ARCSEC = 84381.448
OBLIQUITY_DEG = OBLIQUITY_ARCSEC / 3600.0
OUTPUT_PNG = "IERS_0001_O_JPL_BW_EQUATION_SLIDE.png"
OUTPUT_PDF = "IERS_0001_O_JPL_BW_EQUATION_SLIDE.pdf"


def rotation_ecliptic_to_icrf() -> np.ndarray:
    epsilon = np.radians(OBLIQUITY_DEG)
    cosine = np.cos(epsilon)
    sine = np.sin(epsilon)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ],
        dtype=float,
    )


def physical_screen_displacement(
    observer_sun: np.ndarray,
    observer_venus: np.ndarray,
    screen_normal: np.ndarray,
) -> np.ndarray:
    scale = float(np.dot(observer_sun, screen_normal)) / float(
        np.dot(observer_venus, screen_normal)
    )
    displacement = scale * observer_venus - observer_sun
    return displacement - screen_normal * np.dot(displacement, screen_normal)


def pca_angle_deg(points: np.ndarray) -> float:
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    direction = vh[0]
    if direction[0] < 0.0:
        direction = -direction
    direction = base.unit(direction)
    return float(np.degrees(np.arctan2(direction[1], direction[0])))


def derive_jpl_geometry() -> dict[str, object]:
    ecliptic_data = base.fetch_plane("ECLIPTIC")
    icrf_data = base.fetch_plane("FRAME")
    ecliptic_cache = base.build_cache(ecliptic_data)
    icrf_cache = base.build_cache(icrf_data)

    c1_jd, c4_jd = base.north_pole_contact_interval(ecliptic_cache)
    ca_jd = base.geocentric_closest_approach(ecliptic_cache)
    rotation = rotation_ecliptic_to_icrf()

    sun_e = base.vector(ecliptic_cache, "GS", ca_jd)
    sun_f = base.vector(icrf_cache, "GS", ca_jd)
    norm_e = float(np.linalg.norm(sun_e))
    norm_f = float(np.linalg.norm(sun_f))
    n_e = base.unit(sun_e)
    n_f = base.unit(sun_f)

    k = np.array([0.0, 0.0, 1.0])
    x_e = base.unit(np.cross(k, n_e))
    y_e = base.unit(np.cross(n_e, x_e))
    x_f = base.unit(np.cross(k, n_f))
    y_f = base.unit(np.cross(n_f, x_f))

    x_e_in_f = base.unit(rotation @ x_e)
    y_e_in_f = base.unit(rotation @ y_e)
    n_e_in_f = base.unit(rotation @ n_e)

    numerator = float(np.dot(n_f, np.cross(x_e_in_f, x_f)))
    denominator = float(np.dot(x_e_in_f, x_f))
    phi_axes_deg = float(np.degrees(np.arctan2(numerator, denominator)))
    delta_theta_deg = -phi_axes_deg

    basis_matrix = np.array(
        [
            [np.dot(x_f, x_e_in_f), np.dot(x_f, y_e_in_f)],
            [np.dot(y_f, x_e_in_f), np.dot(y_f, y_e_in_f)],
        ],
        dtype=float,
    )

    epochs = np.asarray(ecliptic_cache["jd"], dtype=float)
    fit_epochs = epochs[(epochs >= c1_jd) & (epochs <= c4_jd)]
    points_e: list[np.ndarray] = []
    points_f: list[np.ndarray] = []

    for jd in fit_epochs:
        displacement_e = physical_screen_displacement(
            base.vector(ecliptic_cache, "NS", jd),
            base.vector(ecliptic_cache, "NV", jd),
            n_e,
        )
        displacement_f = physical_screen_displacement(
            base.vector(icrf_cache, "NS", jd),
            base.vector(icrf_cache, "NV", jd),
            n_f,
        )

        points_e.append(
            np.array(
                [np.dot(displacement_e, x_e), np.dot(displacement_e, y_e)],
                dtype=float,
            )
        )
        points_f.append(
            np.array(
                [np.dot(displacement_f, x_f), np.dot(displacement_f, y_f)],
                dtype=float,
            )
        )

    points_e_array = np.asarray(points_e)
    points_f_array = np.asarray(points_f)
    theta_e_deg = pca_angle_deg(points_e_array)
    theta_f_deg = pca_angle_deg(points_f_array)
    theta_f_derived_deg = theta_e_deg + delta_theta_deg
    closure_arcsec = (theta_f_deg - theta_f_derived_deg) * 3600.0

    cosine = float(np.cos(np.radians(OBLIQUITY_DEG)))
    sine = float(np.sin(np.radians(OBLIQUITY_DEG)))

    return {
        "ca_jd": ca_jd,
        "fit_rows": int(fit_epochs.size),
        "rotation": rotation,
        "cosine": cosine,
        "sine": sine,
        "sun_e": sun_e,
        "sun_f": sun_f,
        "norm_e": norm_e,
        "norm_f": norm_f,
        "n_e": n_e,
        "n_f": n_f,
        "n_e_in_f": n_e_in_f,
        "x_e": x_e,
        "x_f": x_f,
        "x_e_in_f": x_e_in_f,
        "numerator": numerator,
        "denominator": denominator,
        "phi_axes_deg": phi_axes_deg,
        "delta_theta_deg": delta_theta_deg,
        "basis_matrix": basis_matrix,
        "theta_e_deg": theta_e_deg,
        "theta_f_deg": theta_f_deg,
        "theta_f_derived_deg": theta_f_derived_deg,
        "frame_difference_deg": theta_f_deg - theta_e_deg,
        "closure_arcsec": closure_arcsec,
        "sun_transform_error": float(np.linalg.norm(n_e_in_f - n_f)),
    }


def vector_text(vector: np.ndarray, precision: int) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in vector) + "]"


def panel(
    figure: plt.Figure,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    figure.patches.append(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            linewidth=1.0,
            edgecolor="black",
            facecolor="white",
            transform=figure.transFigure,
            clip_on=False,
        )
    )


def add_ray(
    axis: plt.Axes,
    angle_deg: float,
    length: float,
    linewidth: float,
    linestyle: str,
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
            "color": "black",
            "linestyle": linestyle,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    return endpoint


def render_slide(result: dict[str, object]) -> None:
    plt.close("all")
    figure = plt.figure(figsize=(16, 9), facecolor="white")
    figure.patch.set_facecolor("white")

    figure.text(
        0.035,
        0.958,
        "JPL SOLAR-SCREEN FRAME ROTATION",
        fontsize=23,
        fontweight="bold",
        color="black",
        va="top",
    )
    figure.text(
        0.035,
        0.918,
        "Exact derivation of the 6.160665° ECLIPTIC-to-ICRF angle offset",
        fontsize=14,
        color="black",
        va="top",
    )

    panel(figure, 0.030, 0.735, 0.575, 0.150)
    figure.text(0.050, 0.855, "1. J2000 obliquity", fontsize=14, fontweight="bold")
    figure.text(
        0.065,
        0.805,
        rf"$\varepsilon=\frac{{84381.448^{{\prime\prime}}}}{{3600}}={OBLIQUITY_DEG:.9f}^\circ$",
        fontsize=20,
        color="black",
    )
    figure.text(
        0.065,
        0.758,
        rf"$\cos\varepsilon={result['cosine']:.12f},\qquad \sin\varepsilon={result['sine']:.12f}$",
        fontsize=16,
        color="black",
    )

    panel(figure, 0.030, 0.515, 0.575, 0.195)
    figure.text(0.050, 0.680, "2. Rotation matrix and JPL Sun vectors", fontsize=14, fontweight="bold")
    figure.text(0.065, 0.632, r"$R_x(\varepsilon)=$", fontsize=17)
    figure.text(
        0.185,
        0.603,
        f"[[1.000000000000, 0.000000000000, 0.000000000000],\n"
        f" [0.000000000000, {result['cosine']:.12f}, {-result['sine']:.12f}],\n"
        f" [0.000000000000, {result['sine']:.12f}, {result['cosine']:.12f}]]",
        fontsize=10.5,
        family="monospace",
        color="black",
        linespacing=1.3,
    )
    figure.text(
        0.065,
        0.550,
        f"S_E = {vector_text(result['sun_e'], 3)} km",
        fontsize=10.5,
        family="monospace",
    )
    figure.text(
        0.065,
        0.525,
        f"S_F = {vector_text(result['sun_f'], 3)} km",
        fontsize=10.5,
        family="monospace",
    )

    panel(figure, 0.030, 0.285, 0.575, 0.205)
    figure.text(0.050, 0.458, "3. Normalize the Sun directions", fontsize=14, fontweight="bold")
    figure.text(
        0.065,
        0.414,
        rf"$\hat{{n}}_E=\frac{{S_E}}{{|S_E|}},\qquad |S_E|={result['norm_e']:.6f}\;\mathrm{{km}}$",
        fontsize=16,
    )
    figure.text(
        0.065,
        0.377,
        f"n_E = {vector_text(result['n_e'], 9)}",
        fontsize=10.5,
        family="monospace",
    )
    figure.text(
        0.065,
        0.337,
        rf"$\hat{{n}}_F=\frac{{S_F}}{{|S_F|}},\qquad |S_F|={result['norm_f']:.6f}\;\mathrm{{km}}$",
        fontsize=16,
    )
    figure.text(
        0.065,
        0.300,
        f"n_F = {vector_text(result['n_f'], 9)}",
        fontsize=10.5,
        family="monospace",
    )

    panel(figure, 0.030, 0.045, 0.575, 0.215)
    figure.text(0.050, 0.228, "4. Local solar-screen rotation", fontsize=14, fontweight="bold")
    figure.text(
        0.065,
        0.188,
        r"$x_E=\frac{k\times n_E}{|k\times n_E|},\qquad x_F=\frac{k\times n_F}{|k\times n_F|}$",
        fontsize=15,
    )
    figure.text(
        0.065,
        0.146,
        r"$\phi_{axes}=\mathrm{atan2}\left[n_F\cdot((R_xx_E)\times x_F),(R_xx_E)\cdot x_F\right]$",
        fontsize=13.5,
    )
    figure.text(
        0.065,
        0.108,
        f"= atan2({result['numerator']:.12f}, {result['denominator']:.12f})",
        fontsize=11.5,
        family="monospace",
    )
    figure.text(
        0.065,
        0.068,
        rf"$\phi_{{axes}}={result['phi_axes_deg']:.6f}^\circ,\qquad "
        rf"\Delta\theta=-\phi_{{axes}}={result['delta_theta_deg']:.6f}^\circ$",
        fontsize=17,
        fontweight="bold",
    )

    axis = figure.add_axes([0.635, 0.350, 0.325, 0.475], facecolor="white")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-0.15, 1.35)
    axis.set_ylim(-0.38, 0.52)
    axis.axis("off")
    axis.set_title("Local solar-screen geometry", fontsize=15, pad=14)

    phi_axes = float(result["phi_axes_deg"])
    theta_e = float(result["theta_e_deg"])

    ecliptic_end = add_ray(axis, 0.0, 1.15, 2.2, "-")
    icrf_end = add_ray(axis, phi_axes, 1.05, 1.8, "--")
    track_end = add_ray(axis, theta_e, 1.22, 2.8, "-")

    axis.add_patch(
        Arc(
            (0.0, 0.0),
            0.56,
            0.56,
            theta1=phi_axes,
            theta2=0.0,
            linewidth=1.3,
            color="black",
        )
    )

    axis.text(ecliptic_end[0] - 0.02, -0.08, "Ecliptic +X", fontsize=12, ha="right")
    axis.text(icrf_end[0] - 0.01, icrf_end[1] - 0.12, "ICRF +X", fontsize=12, ha="right")
    axis.text(track_end[0] + 0.02, track_end[1] + 0.02, "Venus track", fontsize=12, ha="left")
    axis.annotate(
        rf"$\Delta\theta={result['delta_theta_deg']:.6f}^\circ$",
        xy=(0.27, -0.015),
        xytext=(0.02, -0.31),
        fontsize=13,
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.0},
    )
    axis.scatter([0.0], [0.0], s=26, color="black")

    panel(figure, 0.630, 0.080, 0.330, 0.225)
    figure.text(0.650, 0.272, "Numerical angle closure", fontsize=15, fontweight="bold")
    figure.text(
        0.655,
        0.225,
        f"Ecliptic track angle      {result['theta_e_deg']:12.6f}°",
        fontsize=13,
        family="monospace",
    )
    figure.text(
        0.655,
        0.187,
        f"Frame-axis offset         {result['delta_theta_deg']:12.6f}°",
        fontsize=13,
        family="monospace",
    )
    figure.text(
        0.655,
        0.149,
        f"Derived ICRF angle        {result['theta_f_derived_deg']:12.6f}°",
        fontsize=13,
        family="monospace",
    )
    figure.text(
        0.655,
        0.111,
        f"Direct ICRF angle         {result['theta_f_deg']:12.6f}°",
        fontsize=13,
        family="monospace",
    )
    figure.text(
        0.655,
        0.073,
        f"Closure                   {result['closure_arcsec']:12.9f} arcsec",
        fontsize=13,
        family="monospace",
        fontweight="bold",
    )

    figure.savefig(OUTPUT_PNG, dpi=200, facecolor="white", bbox_inches="tight")
    figure.savefig(OUTPUT_PDF, facecolor="white", bbox_inches="tight")
    plt.show()
    print(f"\nPRESENTATION PNG: {OUTPUT_PNG}")
    print(f"PRESENTATION PDF: {OUTPUT_PDF}")


def print_results(result: dict[str, object]) -> None:
    print("IERS TN36 - JPL Black-and-White Equation Slide")
    print(f"Version : {VERSION}")
    print(f"Obliquity                    {OBLIQUITY_DEG:.9f} deg")
    print(f"atan2 numerator              {result['numerator']:.12f}")
    print(f"atan2 denominator            {result['denominator']:.12f}")
    print(f"Physical axis rotation       {result['phi_axes_deg']:.6f} deg")
    print(f"Coordinate offset            {result['delta_theta_deg']:.6f} deg")
    print(f"Ecliptic track angle         {result['theta_e_deg']:.6f} deg")
    print(f"Derived ICRF angle           {result['theta_f_derived_deg']:.6f} deg")
    print(f"Direct ICRF angle            {result['theta_f_deg']:.6f} deg")
    print(f"Exact closure                {result['closure_arcsec']:.9f} arcsec")


def main() -> None:
    plt.close("all")
    result = derive_jpl_geometry()
    print_results(result)
    render_slide(result)
    plt.close("all")


if __name__ == "__main__":
    main()
