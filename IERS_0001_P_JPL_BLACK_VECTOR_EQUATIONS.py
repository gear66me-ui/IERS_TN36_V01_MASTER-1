"""
IERS TN36 — Ecliptical Plane Analysis
Part P — Black-background JPL vector-equation derivation slides
No AI images. Matplotlib only.
Requires IERS_0001_K_JPL_FRAME_ANGLE_CONFIRMATION.py in the same folder.
"""

from __future__ import annotations

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Arc
import matplotlib.pyplot as plt
import numpy as np

import IERS_0001_K_JPL_FRAME_ANGLE_CONFIRMATION as base

VERSION = "IERS-0001-P"
OBLIQUITY_ARCSEC = 84381.448
OBLIQUITY_DEG = OBLIQUITY_ARCSEC / 3600.0
DERIVATION_PNG = "IERS_0001_P_JPL_VECTOR_DERIVATION.png"
GEOMETRY_PNG = "IERS_0001_P_JPL_ANGLE_GEOMETRY.png"
DECK_PDF = "IERS_0001_P_JPL_EQUATION_DECK.pdf"


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
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    direction = vh[0]
    if direction[0] < 0.0:
        direction = -direction
    direction = base.unit(direction)
    return float(np.degrees(np.arctan2(direction[1], direction[0])))


def derive() -> dict[str, object]:
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
        points_e.append(np.array([np.dot(d_e, x_e), np.dot(d_e, y_e)]))
        points_f.append(np.array([np.dot(d_f, x_f), np.dot(d_f, y_f)]))

    theta_e_deg = pca_angle_deg(np.asarray(points_e))
    theta_f_deg = pca_angle_deg(np.asarray(points_f))
    theta_f_derived_deg = theta_e_deg + delta_theta_deg
    closure_arcsec = (theta_f_deg - theta_f_derived_deg) * 3600.0

    return {
        "rotation": rotation,
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
        "basis_matrix": basis_matrix,
        "numerator": numerator,
        "denominator": denominator,
        "phi_axes_deg": phi_axes_deg,
        "delta_theta_deg": delta_theta_deg,
        "theta_e_deg": theta_e_deg,
        "theta_f_deg": theta_f_deg,
        "theta_f_derived_deg": theta_f_derived_deg,
        "closure_arcsec": closure_arcsec,
        "fit_rows": int(fit_epochs.size),
        "sun_transform_error": float(np.linalg.norm(n_e_in_f - n_f)),
    }


def vtext(vector: np.ndarray, precision: int = 9) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in vector) + "]"


def matrix_text(matrix: np.ndarray, precision: int = 12) -> str:
    rows = []
    for row in matrix:
        rows.append("  ".join(f"{value: .{precision}f}" for value in row))
    return "⎡ " + rows[0] + " ⎤\n⎢ " + rows[1] + " ⎥\n⎣ " + rows[2] + " ⎦"


def render_derivation(result: dict[str, object]) -> plt.Figure:
    plt.close("all")
    fig = plt.figure(figsize=(16, 9), facecolor="black")
    fig.patch.set_facecolor("black")
    white = "white"
    gray = "#d0d0d0"

    fig.text(0.04, 0.95, "JPL SOLAR-SCREEN BASIS DERIVATION", color=white,
             fontsize=25, fontweight="bold", va="top")
    fig.text(0.04, 0.905,
             "ECLIPTIC → ICRF local frame rotation at the 2012 Venus transit",
             color=gray, fontsize=14, va="top")

    fig.text(0.05, 0.845, "1. J2000 obliquity", color=white,
             fontsize=16, fontweight="bold")
    fig.text(0.08, 0.795,
             rf"$\varepsilon=\frac{{84381.448^{{\prime\prime}}}}{{3600}}={OBLIQUITY_DEG:.9f}^\circ$",
             color=white, fontsize=22)
    fig.text(0.08, 0.735, r"$R_x(\varepsilon)=$", color=white, fontsize=20)
    fig.text(0.245, 0.700, matrix_text(result["rotation"]), color=white,
             fontsize=13.5, family="monospace", linespacing=1.25)

    fig.text(0.05, 0.620, "2. JPL Sun-direction vectors", color=white,
             fontsize=16, fontweight="bold")
    fig.text(0.08, 0.565,
             r"$\hat{\mathbf{n}}_E=\frac{\mathbf{S}_E}{\|\mathbf{S}_E\|}$",
             color=white, fontsize=22)
    fig.text(0.08, 0.525,
             f"S_E = {vtext(result['sun_e'], 3)} km", color=gray,
             fontsize=11.5, family="monospace")
    fig.text(0.08, 0.492,
             f"||S_E|| = {result['norm_e']:.6f} km", color=gray,
             fontsize=11.5, family="monospace")
    fig.text(0.08, 0.459,
             f"n_E = {vtext(result['n_e'], 9)}", color=white,
             fontsize=11.5, family="monospace")

    fig.text(0.54, 0.565,
             r"$\hat{\mathbf{n}}_F=\frac{\mathbf{S}_F}{\|\mathbf{S}_F\|}$",
             color=white, fontsize=22)
    fig.text(0.54, 0.525,
             f"S_F = {vtext(result['sun_f'], 3)} km", color=gray,
             fontsize=11.5, family="monospace")
    fig.text(0.54, 0.492,
             f"||S_F|| = {result['norm_f']:.6f} km", color=gray,
             fontsize=11.5, family="monospace")
    fig.text(0.54, 0.459,
             f"n_F = {vtext(result['n_f'], 9)}", color=white,
             fontsize=11.5, family="monospace")

    fig.text(0.05, 0.385, "3. Solar-screen axes", color=white,
             fontsize=16, fontweight="bold")
    fig.text(0.08, 0.330,
             r"$\mathbf{k}=(0,0,1)^{T}$",
             color=white, fontsize=20)
    fig.text(0.08, 0.275,
             r"$\hat{\mathbf{x}}_E=\frac{\mathbf{k}\times\hat{\mathbf{n}}_E}{\|\mathbf{k}\times\hat{\mathbf{n}}_E\|},\qquad"
             r"\hat{\mathbf{x}}_F=\frac{\mathbf{k}\times\hat{\mathbf{n}}_F}{\|\mathbf{k}\times\hat{\mathbf{n}}_F\|}$",
             color=white, fontsize=19)
    fig.text(0.08, 0.225,
             r"$\hat{\mathbf{x}}_{E\rightarrow F}=R_x(\varepsilon)\hat{\mathbf{x}}_E$",
             color=white, fontsize=19)

    fig.text(0.05, 0.155, "4. Signed local frame rotation", color=white,
             fontsize=16, fontweight="bold")
    fig.text(0.08, 0.102,
             r"$\phi_{\mathrm{axes}}=\mathrm{atan2}\left[\hat{\mathbf{n}}_F\cdot\left(\hat{\mathbf{x}}_{E\rightarrow F}\times\hat{\mathbf{x}}_F\right),\;"
             r"\hat{\mathbf{x}}_{E\rightarrow F}\cdot\hat{\mathbf{x}}_F\right]$",
             color=white, fontsize=17)
    fig.text(0.08, 0.055,
             f"= atan2({result['numerator']:.12f}, {result['denominator']:.12f})"
             f" = {result['phi_axes_deg']:.6f}°",
             color=white, fontsize=13, family="monospace")

    return fig


def add_ray(axis: plt.Axes, angle_deg: float, length: float,
            linewidth: float, linestyle: str = "-") -> np.ndarray:
    angle = np.radians(angle_deg)
    endpoint = length * np.array([np.cos(angle), np.sin(angle)])
    axis.annotate("", xy=endpoint, xytext=(0.0, 0.0),
                  arrowprops={"arrowstyle": "-|>", "color": "white",
                              "linewidth": linewidth, "linestyle": linestyle})
    return endpoint


def render_geometry(result: dict[str, object]) -> plt.Figure:
    fig = plt.figure(figsize=(16, 9), facecolor="black")
    fig.patch.set_facecolor("black")
    white = "white"
    gray = "#d0d0d0"

    fig.text(0.04, 0.95, "LOCAL ANGLE TRANSFORMATION", color=white,
             fontsize=25, fontweight="bold", va="top")
    fig.text(0.04, 0.905,
             "One physical Venus track measured from two different solar-screen axes",
             color=gray, fontsize=14, va="top")

    axis = fig.add_axes([0.08, 0.14, 0.55, 0.70], facecolor="black")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-0.15, 1.35)
    axis.set_ylim(-0.48, 0.68)
    axis.axis("off")

    phi = float(result["phi_axes_deg"])
    theta_e = float(result["theta_e_deg"])
    theta_f = float(result["theta_f_deg"])

    e_end = add_ray(axis, 0.0, 1.12, 2.5)
    f_end = add_ray(axis, phi, 1.03, 2.0, "--")
    t_end = add_ray(axis, theta_e, 1.22, 3.0)

    axis.add_patch(Arc((0, 0), 0.54, 0.54, theta1=phi, theta2=0,
                       color=white, linewidth=1.4))

    axis.text(e_end[0] - 0.02, -0.10, "Ecliptic +X", color=white,
              fontsize=14, ha="right")
    axis.text(f_end[0] - 0.01, f_end[1] - 0.14, "ICRF +X", color=white,
              fontsize=14, ha="right")
    axis.text(t_end[0] + 0.03, t_end[1] + 0.03, "Venus track", color=white,
              fontsize=14, ha="left")
    axis.annotate(rf"$\Delta\theta={result['delta_theta_deg']:.6f}^\circ$",
                  xy=(0.27, -0.01), xytext=(-0.02, -0.34),
                  color=white, fontsize=15,
                  arrowprops={"arrowstyle": "->", "color": white})
    axis.scatter([0.0], [0.0], color=white, s=28)

    fig.text(0.68, 0.74,
             rf"$\Delta\theta=-\phi_{{\mathrm{{axes}}}}={result['delta_theta_deg']:.6f}^\circ$",
             color=white, fontsize=22)
    fig.text(0.68, 0.61,
             rf"$\theta_F=\theta_E+\Delta\theta$",
             color=white, fontsize=24)
    fig.text(0.68, 0.49,
             rf"$\theta_F={result['theta_e_deg']:.6f}^\circ+{result['delta_theta_deg']:.6f}^\circ$",
             color=white, fontsize=21)
    fig.text(0.68, 0.39,
             rf"$\theta_F={result['theta_f_derived_deg']:.6f}^\circ$",
             color=white, fontsize=25)
    fig.text(0.68, 0.25,
             f"Direct JPL ICRF fit     {result['theta_f_deg']:.6f}°\n"
             f"Derived ICRF angle      {result['theta_f_derived_deg']:.6f}°\n"
             f"Closure                 {result['closure_arcsec']:.9f} arcsec",
             color=white, fontsize=14, family="monospace", linespacing=1.7)

    return fig


def print_results(result: dict[str, object]) -> None:
    print("IERS TN36 - JPL Black Vector Equation Slides")
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
    result = derive()
    print_results(result)

    derivation = render_derivation(result)
    derivation.savefig(DERIVATION_PNG, dpi=200, facecolor="black",
                        bbox_inches="tight")
    geometry = render_geometry(result)
    geometry.savefig(GEOMETRY_PNG, dpi=200, facecolor="black",
                      bbox_inches="tight")

    with PdfPages(DECK_PDF) as pdf:
        pdf.savefig(derivation, facecolor="black", bbox_inches="tight")
        pdf.savefig(geometry, facecolor="black", bbox_inches="tight")

    plt.show()
    print(f"\nDERIVATION SLIDE: {DERIVATION_PNG}")
    print(f"GEOMETRY SLIDE:   {GEOMETRY_PNG}")
    print(f"TWO-PAGE PDF:     {DECK_PDF}")


if __name__ == "__main__":
    main()
