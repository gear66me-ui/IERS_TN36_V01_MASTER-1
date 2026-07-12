"""
IERS TN36 — Ecliptical Plane Analysis
Part M — Exact JPL equation derivation and black-background presentation slide
Requires IERS_0001_K_JPL_FRAME_ANGLE_CONFIRMATION.py in the same folder.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyBboxPatch
import numpy as np

import IERS_0001_K_JPL_FRAME_ANGLE_CONFIRMATION as base

VERSION = "IERS-0001-M"
OBLIQUITY_ARCSEC = 84381.448
OBLIQUITY_DEG = OBLIQUITY_ARCSEC / 3600.0
OUTPUT_PNG = "IERS_0001_M_JPL_EQUATION_DERIVATION.png"


def rotation_ecliptic_to_icrf() -> np.ndarray:
    epsilon = np.radians(OBLIQUITY_DEG)
    c = np.cos(epsilon)
    s = np.sin(epsilon)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
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
    center = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
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
    norm_sun_e = float(np.linalg.norm(sun_e))
    norm_sun_f = float(np.linalg.norm(sun_f))
    n_e = base.unit(sun_e)
    n_f = base.unit(sun_f)
    n_e_in_f = base.unit(rotation @ n_e)

    k = np.array([0.0, 0.0, 1.0])
    x_e = base.unit(np.cross(k, n_e))
    y_e = base.unit(np.cross(n_e, x_e))
    x_f = base.unit(np.cross(k, n_f))
    y_f = base.unit(np.cross(n_f, x_f))
    x_e_in_f = base.unit(rotation @ x_e)
    y_e_in_f = base.unit(rotation @ y_e)

    numerator = float(np.dot(n_f, np.cross(x_e_in_f, x_f)))
    denominator = float(np.dot(x_e_in_f, x_f))
    phi_axes_deg = float(np.degrees(np.arctan2(numerator, denominator)))
    coordinate_offset_deg = -phi_axes_deg

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
    transformed_points_f: list[np.ndarray] = []

    for jd in fit_epochs:
        d_e = physical_screen_displacement(
            base.vector(ecliptic_cache, "NS", jd),
            base.vector(ecliptic_cache, "NV", jd),
            n_e,
        )
        d_f = physical_screen_displacement(
            base.vector(icrf_cache, "NS", jd),
            base.vector(icrf_cache, "NV", jd),
            n_f,
        )

        p_e = np.array([np.dot(d_e, x_e), np.dot(d_e, y_e)], dtype=float)
        p_f = np.array([np.dot(d_f, x_f), np.dot(d_f, y_f)], dtype=float)
        points_e.append(p_e)
        points_f.append(p_f)
        transformed_points_f.append(basis_matrix @ p_e)

    points_e_array = np.asarray(points_e)
    points_f_array = np.asarray(points_f)
    transformed_points_f_array = np.asarray(transformed_points_f)

    theta_e_deg = pca_angle_deg(points_e_array)
    theta_f_deg = pca_angle_deg(points_f_array)
    theta_f_from_rotation_deg = theta_e_deg + coordinate_offset_deg
    frame_difference_deg = theta_f_deg - theta_e_deg
    closure_deg = theta_f_deg - theta_f_from_rotation_deg

    return {
        "ca_jd": ca_jd,
        "fit_rows": int(fit_epochs.size),
        "rotation": rotation,
        "sun_e": sun_e,
        "sun_f": sun_f,
        "norm_sun_e": norm_sun_e,
        "norm_sun_f": norm_sun_f,
        "n_e": n_e,
        "n_f": n_f,
        "n_e_in_f": n_e_in_f,
        "x_e": x_e,
        "x_f": x_f,
        "x_e_in_f": x_e_in_f,
        "numerator": numerator,
        "denominator": denominator,
        "phi_axes_deg": phi_axes_deg,
        "coordinate_offset_deg": coordinate_offset_deg,
        "basis_matrix": basis_matrix,
        "theta_e_deg": theta_e_deg,
        "theta_f_deg": theta_f_deg,
        "theta_f_from_rotation_deg": theta_f_from_rotation_deg,
        "frame_difference_deg": frame_difference_deg,
        "closure_arcsec": closure_deg * 3600.0,
        "sun_transform_error": float(np.linalg.norm(n_e_in_f - n_f)),
        "point_transform_error_km": float(
            np.max(np.linalg.norm(points_f_array - transformed_points_f_array, axis=1))
        ),
    }


def format_vector(vector: np.ndarray, precision: int = 9) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in vector) + "]"


def print_results(result: dict[str, object]) -> None:
    print("IERS TN36 - Exact JPL Equation Derivation")
    print(f"Version : {VERSION}")
    print("Observer: North Pole")
    print("Reference planes: JPL ECLIPTIC and JPL FRAME")
    print()
    print(f"J2000 obliquity               {OBLIQUITY_ARCSEC:.3f} / 3600")
    print(f"J2000 obliquity               {OBLIQUITY_DEG:.9f} deg")
    print(f"Ecliptic Sun norm             {result['norm_sun_e']:.6f} km")
    print(f"ICRF Sun norm                 {result['norm_sun_f']:.6f} km")
    print(f"atan2 numerator               {result['numerator']:.12f}")
    print(f"atan2 denominator             {result['denominator']:.12f}")
    print(f"Physical axis rotation        {result['phi_axes_deg']:.6f} deg")
    print(f"Coordinate angle offset       {result['coordinate_offset_deg']:.6f} deg")
    print(f"Ecliptic track angle          {result['theta_e_deg']:.6f} deg")
    print(f"ICRF track angle              {result['theta_f_deg']:.6f} deg")
    print(f"Derived ICRF angle            {result['theta_f_from_rotation_deg']:.6f} deg")
    print(f"Frame difference              {result['frame_difference_deg']:.6f} deg")
    print(f"Exact closure                 {result['closure_arcsec']:.9f} arcsec")
    print(f"Sun-vector transform error    {result['sun_transform_error']:.6e}")
    print(f"Screen-point transform error  {result['point_transform_error_km']:.6e} km")


def add_box(figure: plt.Figure, x: float, y: float, width: float, height: float) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=1.0,
        edgecolor="#2aa8ff",
        facecolor="#080b10",
        transform=figure.transFigure,
        clip_on=False,
    )
    figure.patches.append(box)


def add_equation_slide(result: dict[str, object]) -> None:
    figure = plt.figure(figsize=(16, 9), facecolor="#05070a")
    figure.patch.set_facecolor("#05070a")

    white = "#f5f7fa"
    blue = "#35b5ff"
    green = "#7ef58a"
    gold = "#ffd34d"
    gray = "#aeb7c2"

    figure.text(
        0.035,
        0.945,
        "ECLIPTICAL PLANE ANALYSIS",
        color=white,
        fontsize=24,
        fontweight="bold",
        va="top",
    )
    figure.text(
        0.035,
        0.905,
        "Exact JPL solar-screen basis derivation — North Pole, 2012 Venus transit",
        color=blue,
        fontsize=14,
        va="top",
    )

    add_box(figure, 0.025, 0.695, 0.56, 0.185)
    figure.text(0.045, 0.852, "1  J2000 obliquity and frame rotation", color=blue, fontsize=15, fontweight="bold")
    figure.text(
        0.055,
        0.805,
        rf"$\varepsilon=\dfrac{{84381.448^{{\prime\prime}}}}{{3600}}={OBLIQUITY_DEG:.9f}^\circ$",
        color=white,
        fontsize=20,
    )
    figure.text(0.055, 0.748, r"$R_x(\varepsilon)=$", color=white, fontsize=18)
    figure.text(
        0.190,
        0.726,
        "⎡ 1          0              0       ⎤\n"
        "⎢ 0       cos ε         −sin ε     ⎥\n"
        "⎣ 0       sin ε          cos ε     ⎦",
        color=white,
        fontsize=15,
        family="monospace",
        linespacing=1.15,
    )

    add_box(figure, 0.025, 0.455, 0.56, 0.215)
    figure.text(0.045, 0.642, "2  JPL Sun vectors and normalization", color=blue, fontsize=15, fontweight="bold")
    figure.text(0.055, 0.595, r"$\hat{\mathbf n}_E=\mathbf S_E/\lVert\mathbf S_E\rVert$", color=white, fontsize=18)
    figure.text(0.055, 0.557, f"S_E = {format_vector(result['sun_e'], 3)} km", color=gray, fontsize=11, family="monospace")
    figure.text(0.055, 0.526, f"||S_E|| = {result['norm_sun_e']:.6f} km", color=gray, fontsize=11, family="monospace")
    figure.text(0.055, 0.495, f"n_E = {format_vector(result['n_e'], 9)}", color=green, fontsize=11, family="monospace")
    figure.text(0.315, 0.595, r"$\hat{\mathbf n}_F=\mathbf S_F/\lVert\mathbf S_F\rVert$", color=white, fontsize=18)
    figure.text(0.315, 0.557, f"S_F = {format_vector(result['sun_f'], 3)} km", color=gray, fontsize=11, family="monospace")
    figure.text(0.315, 0.526, f"||S_F|| = {result['norm_sun_f']:.6f} km", color=gray, fontsize=11, family="monospace")
    figure.text(0.315, 0.495, f"n_F = {format_vector(result['n_f'], 9)}", color=gold, fontsize=11, family="monospace")

    add_box(figure, 0.025, 0.215, 0.56, 0.215)
    figure.text(0.045, 0.402, "3  Signed rotation of the solar-screen axes", color=blue, fontsize=15, fontweight="bold")
    figure.text(
        0.055,
        0.350,
        r"$\hat{\mathbf x}_E=\dfrac{\mathbf k\times\hat{\mathbf n}_E}{\lVert\mathbf k\times\hat{\mathbf n}_E\rVert},\qquad"
        r"\hat{\mathbf x}_F=\dfrac{\mathbf k\times\hat{\mathbf n}_F}{\lVert\mathbf k\times\hat{\mathbf n}_F\rVert}$",
        color=white,
        fontsize=17,
    )
    figure.text(
        0.055,
        0.295,
        r"$\phi_{\rm axes}=\operatorname{atan2}\!\left[\hat{\mathbf n}_F\!\cdot\!((R_x\hat{\mathbf x}_E)\times\hat{\mathbf x}_F),"
        r"\,(R_x\hat{\mathbf x}_E)\!\cdot\!\hat{\mathbf x}_F\right]$",
        color=white,
        fontsize=15,
    )
    figure.text(
        0.055,
        0.245,
        f"numerator = {result['numerator']:.12f}    denominator = {result['denominator']:.12f}",
        color=gray,
        fontsize=11,
        family="monospace",
    )
    figure.text(
        0.355,
        0.245,
        rf"$\phi_{{\rm axes}}={result['phi_axes_deg']:.6f}^\circ$",
        color=gold,
        fontsize=20,
        fontweight="bold",
    )

    add_box(figure, 0.025, 0.045, 0.56, 0.145)
    figure.text(0.045, 0.162, "4  Coordinate-angle transformation and zero closure", color=blue, fontsize=15, fontweight="bold")
    figure.text(
        0.055,
        0.112,
        rf"$\Delta\theta=-\phi_{{\rm axes}}={result['coordinate_offset_deg']:.6f}^\circ$",
        color=green,
        fontsize=19,
    )
    figure.text(
        0.055,
        0.072,
        rf"$\theta_F=\theta_E+\Delta\theta={result['theta_e_deg']:.6f}^\circ+{result['coordinate_offset_deg']:.6f}^\circ={result['theta_f_from_rotation_deg']:.6f}^\circ$",
        color=white,
        fontsize=17,
    )
    figure.text(
        0.410,
        0.108,
        f"closure = {result['closure_arcsec']:.9f} arcsec",
        color=green,
        fontsize=12,
        family="monospace",
    )

    axis = figure.add_axes([0.62, 0.12, 0.35, 0.73], facecolor="#05070a")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-0.15, 1.35)
    axis.set_ylim(-0.45, 0.65)
    axis.axis("off")
    axis.set_title("One physical Venus track, two reference axes", color=white, fontsize=16, pad=18)

    phi = float(result["phi_axes_deg"])
    theta_e = float(result["theta_e_deg"])
    theta_f = float(result["theta_f_deg"])

    def ray(angle_deg: float, length: float, color: str, width: float) -> np.ndarray:
        angle_rad = np.radians(angle_deg)
        endpoint = length * np.array([np.cos(angle_rad), np.sin(angle_rad)])
        axis.annotate(
            "",
            xy=endpoint,
            xytext=(0.0, 0.0),
            arrowprops={"arrowstyle": "-|>", "linewidth": width, "color": color},
        )
        return endpoint

    ecliptic_end = ray(0.0, 1.12, blue, 2.5)
    icrf_end = ray(phi, 1.04, gold, 2.5)
    track_end = ray(theta_e, 1.22, green, 3.0)

    axis.add_patch(Arc((0, 0), 0.52, 0.52, theta1=phi, theta2=0, color=gold, linewidth=1.5))
    axis.add_patch(Arc((0, 0), 0.86, 0.86, theta1=0, theta2=theta_e, color=blue, linewidth=1.5))
    axis.add_patch(Arc((0, 0), 1.22, 1.22, theta1=phi, theta2=theta_e, color=green, linewidth=1.5))

    axis.text(ecliptic_end[0] - 0.03, ecliptic_end[1] - 0.10, "Ecliptic +X", color=blue, fontsize=13, ha="right")
    axis.text(icrf_end[0] - 0.02, icrf_end[1] - 0.12, "ICRF +X", color=gold, fontsize=13, ha="right")
    axis.text(track_end[0] + 0.02, track_end[1] + 0.02, "Venus track", color=green, fontsize=13, ha="left")

    axis.annotate(
        rf"$\Delta\theta={result['coordinate_offset_deg']:.6f}^\circ$",
        xy=(0.33, -0.018),
        xytext=(0.18, -0.30),
        color=gold,
        fontsize=13,
        arrowprops={"arrowstyle": "->", "color": gold},
    )
    axis.annotate(
        rf"$\theta_E={theta_e:.6f}^\circ$",
        xy=(0.44, 0.035),
        xytext=(0.18, 0.34),
        color=blue,
        fontsize=13,
        arrowprops={"arrowstyle": "->", "color": blue},
    )
    axis.annotate(
        rf"$\theta_F={theta_f:.6f}^\circ$",
        xy=(0.61, 0.015),
        xytext=(0.73, 0.43),
        color=green,
        fontsize=13,
        arrowprops={"arrowstyle": "->", "color": green},
    )

    axis.scatter([0.0], [0.0], s=38, color=white, zorder=5)
    axis.text(
        0.05,
        -0.40,
        rf"$\theta_F-\theta_E={result['frame_difference_deg']:.6f}^\circ$\n"
        rf"$\Delta\theta={result['coordinate_offset_deg']:.6f}^\circ$\n"
        rf"$\mathrm{{closure}}={result['closure_arcsec']:.9f}\;\mathrm{{arcsec}}$",
        color=white,
        fontsize=13,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#080b10", "edgecolor": blue},
    )

    figure.savefig(OUTPUT_PNG, dpi=180, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.show()
    print(f"\nPRESENTATION SLIDE: {OUTPUT_PNG}")


def main() -> None:
    result = derive_jpl_geometry()
    print_results(result)
    add_equation_slide(result)


if __name__ == "__main__":
    main()
