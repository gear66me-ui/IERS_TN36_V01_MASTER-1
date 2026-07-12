"""
IERS TN36 — Ecliptical Plane Analysis
Part L — Exact JPL frame-rotation closure and presentation plot
Requires IERS_0001_K_JPL_FRAME_ANGLE_CONFIRMATION.py in the same folder.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np

import IERS_0001_K_JPL_FRAME_ANGLE_CONFIRMATION as base

VERSION = "IERS-0001-L"
JPL_OBLIQUITY_ARCSEC = 84381.448
JPL_OBLIQUITY_DEG = JPL_OBLIQUITY_ARCSEC / 3600.0
PLOT_FILE = "IERS_0001_L_FRAME_ROTATION_PRESENTATION.png"


def rotation_ecliptic_to_icrf() -> np.ndarray:
    epsilon = np.radians(JPL_OBLIQUITY_DEG)
    cosine, sine = np.cos(epsilon), np.sin(epsilon)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=float,
    )


def signed_angle_deg(first: np.ndarray, second: np.ndarray, normal: np.ndarray) -> float:
    return float(
        np.degrees(
            np.arctan2(
                np.dot(normal, np.cross(first, second)),
                np.dot(first, second),
            )
        )
    )


def physical_displacement(
    observer_sun: np.ndarray,
    observer_venus: np.ndarray,
    screen_normal: np.ndarray,
) -> np.ndarray:
    ray_scale = float(np.dot(observer_sun, screen_normal)) / float(
        np.dot(observer_venus, screen_normal)
    )
    displacement = ray_scale * observer_venus - observer_sun
    return displacement - screen_normal * np.dot(displacement, screen_normal)


def fit_angle(points: np.ndarray) -> tuple[float, float]:
    angle, rms = base.pca_track_angle_deg(points)
    return float(angle), float(rms)


def derive() -> dict[str, object]:
    ecliptic_data = base.fetch_plane("ECLIPTIC")
    icrf_data = base.fetch_plane("FRAME")
    ecliptic_cache = base.build_cache(ecliptic_data)
    icrf_cache = base.build_cache(icrf_data)

    c1_jd, c4_jd = base.north_pole_contact_interval(ecliptic_cache)
    ca_jd = base.geocentric_closest_approach(ecliptic_cache)
    rotation = rotation_ecliptic_to_icrf()

    normal_e, x_e, y_e = base.solar_screen_basis(ecliptic_cache, ca_jd)
    normal_f, x_f, y_f = base.solar_screen_basis(icrf_cache, ca_jd)
    normal_e_in_f = base.unit(rotation @ normal_e)
    x_e_in_f = base.unit(rotation @ x_e)
    y_e_in_f = base.unit(rotation @ y_e)

    vector_transform_error = float(np.linalg.norm(normal_e_in_f - normal_f))
    physical_axis_rotation_deg = signed_angle_deg(x_e_in_f, x_f, normal_f)
    coordinate_offset_deg = -physical_axis_rotation_deg

    basis_matrix = np.array(
        [
            [np.dot(x_f, x_e_in_f), np.dot(x_f, y_e_in_f)],
            [np.dot(y_f, x_e_in_f), np.dot(y_f, y_e_in_f)],
        ],
        dtype=float,
    )

    epochs = np.asarray(ecliptic_cache["jd"], dtype=float)
    fit_epochs = epochs[(epochs >= c1_jd) & (epochs <= c4_jd)]

    linear_e, linear_f, rotated_f = [], [], []
    nonlinear_e, nonlinear_f = [], []
    displacement_errors = []

    for jd in fit_epochs:
        d_e = physical_displacement(
            base.vector(ecliptic_cache, "NS", jd),
            base.vector(ecliptic_cache, "NV", jd),
            normal_e,
        )
        d_f = physical_displacement(
            base.vector(icrf_cache, "NS", jd),
            base.vector(icrf_cache, "NV", jd),
            normal_f,
        )
        d_e_in_f = rotation @ d_e
        displacement_errors.append(np.linalg.norm(d_e_in_f - d_f))

        p_e = np.array([np.dot(d_e, x_e), np.dot(d_e, y_e)], dtype=float)
        p_f = np.array([np.dot(d_f, x_f), np.dot(d_f, y_f)], dtype=float)
        p_f_rotated = basis_matrix @ p_e

        distance_e = np.linalg.norm(base.vector(ecliptic_cache, "GS", jd))
        distance_f = np.linalg.norm(base.vector(icrf_cache, "GS", jd))

        linear_e.append(p_e)
        linear_f.append(p_f)
        rotated_f.append(p_f_rotated)
        nonlinear_e.append(np.arctan2(p_e, distance_e) * base.ARCSEC_PER_RAD)
        nonlinear_f.append(np.arctan2(p_f, distance_f) * base.ARCSEC_PER_RAD)

    linear_e = np.asarray(linear_e)
    linear_f = np.asarray(linear_f)
    rotated_f = np.asarray(rotated_f)
    nonlinear_e = np.asarray(nonlinear_e)
    nonlinear_f = np.asarray(nonlinear_f)

    theta_e_linear, rms_e_km = fit_angle(linear_e)
    theta_f_linear, rms_f_km = fit_angle(linear_f)
    theta_f_rotated, _ = fit_angle(rotated_f)
    theta_e_nonlinear, rms_e_arcsec = fit_angle(nonlinear_e)
    theta_f_nonlinear, rms_f_arcsec = fit_angle(nonlinear_f)

    delta_linear = theta_f_linear - theta_e_linear
    delta_rotated = theta_f_rotated - theta_e_linear
    delta_nonlinear = theta_f_nonlinear - theta_e_nonlinear

    return {
        "fit_rows": int(fit_epochs.size),
        "basis_matrix": basis_matrix,
        "vector_transform_error": vector_transform_error,
        "displacement_transform_error_km": float(max(displacement_errors)),
        "point_rotation_error_km": float(
            np.max(np.linalg.norm(linear_f - rotated_f, axis=1))
        ),
        "physical_axis_rotation_deg": physical_axis_rotation_deg,
        "coordinate_offset_deg": coordinate_offset_deg,
        "theta_e_linear": theta_e_linear,
        "theta_f_linear": theta_f_linear,
        "delta_linear": delta_linear,
        "closure_linear_arcsec": (delta_linear - coordinate_offset_deg) * 3600.0,
        "theta_f_rotated": theta_f_rotated,
        "delta_rotated": delta_rotated,
        "closure_rotated_arcsec": (delta_rotated - coordinate_offset_deg) * 3600.0,
        "theta_e_nonlinear": theta_e_nonlinear,
        "theta_f_nonlinear": theta_f_nonlinear,
        "delta_nonlinear": delta_nonlinear,
        "nonlinear_mismatch_arcsec": (delta_nonlinear - coordinate_offset_deg) * 3600.0,
        "rms_e_km": rms_e_km,
        "rms_f_km": rms_f_km,
        "rms_e_arcsec": rms_e_arcsec,
        "rms_f_arcsec": rms_f_arcsec,
    }


def display(result: dict[str, object]) -> None:
    matrix = np.asarray(result["basis_matrix"])

    print("IERS TN36 - Exact JPL Frame-Rotation Closure")
    print(f"Version : {VERSION}")
    print("Observer: North Pole")
    print("Method  : one fixed physical screen; exact 3-D rotation; linear 2-D fit")

    print("\nJPL ECLIPTIC-TO-ICRF BASIS ROTATION")
    print("OBLIQUITY deg    PHYSICAL AXIS deg    COORDINATE OFFSET deg")
    print(
        f"{JPL_OBLIQUITY_DEG:14.9f}"
        f"{float(result['physical_axis_rotation_deg']):21.6f}"
        f"{float(result['coordinate_offset_deg']):25.6f}"
    )

    print("\nEXACT 2-D BASIS MATRIX")
    print(
        f"[[{matrix[0, 0]: .12f}, {matrix[0, 1]: .12f}],\n"
        f" [{matrix[1, 0]: .12f}, {matrix[1, 1]: .12f}]]"
    )

    print("\nEXACT LINEAR PHYSICAL-SCREEN CLOSURE")
    print("ECLIPTIC deg    ICRF deg    DIFFERENCE deg    OFFSET deg    CLOSURE arcsec")
    print(
        f"{float(result['theta_e_linear']):12.6f}"
        f"{float(result['theta_f_linear']):12.6f}"
        f"{float(result['delta_linear']):18.6f}"
        f"{float(result['coordinate_offset_deg']):14.6f}"
        f"{float(result['closure_linear_arcsec']):18.9f}"
    )

    print("\nROTATED-POINT NUMERICAL CLOSURE")
    print("ICRF ROTATED deg    DIFFERENCE deg    CLOSURE arcsec")
    print(
        f"{float(result['theta_f_rotated']):16.6f}"
        f"{float(result['delta_rotated']):18.6f}"
        f"{float(result['closure_rotated_arcsec']):18.9f}"
    )

    print("\nLEGACY COMPONENT-WISE atan2 AUDIT")
    print("ECLIPTIC deg    ICRF deg    DIFFERENCE deg    OFFSET-DIFFERENCE arcsec")
    print(
        f"{float(result['theta_e_nonlinear']):12.6f}"
        f"{float(result['theta_f_nonlinear']):12.6f}"
        f"{float(result['delta_nonlinear']):18.6f}"
        f"{float(result['nonlinear_mismatch_arcsec']):28.9f}"
    )

    print("\nTRANSFORMATION RESIDUALS")
    print("SUN UNIT VECTOR       DISPLACEMENT km       ROTATED POINT km       FIT ROWS")
    print(
        f"{float(result['vector_transform_error']):15.6e}"
        f"{float(result['displacement_transform_error_km']):22.6e}"
        f"{float(result['point_rotation_error_km']):23.6e}"
        f"{int(result['fit_rows']):15d}"
    )

    print("\nSOURCE OF THE 0.054 ARCSEC")
    print(
        "The residual is created by applying atan2 separately to the X and Y "
        "screen components before PCA."
    )
    print(
        "That nonlinear component mapping is not a rigid 2-D rotation. The "
        "linear physical screen closes at numerical zero."
    )


def draw_ray(axis, angle_deg: float, length: float, label: str, width: float) -> None:
    angle = np.radians(angle_deg)
    endpoint = np.array([np.cos(angle), np.sin(angle)]) * length
    axis.annotate(
        "",
        xy=endpoint,
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "-|>", "linewidth": width},
    )
    axis.text(*(endpoint * 1.08), label, ha="center", va="center")


def presentation_plot(result: dict[str, object]) -> None:
    theta_e = float(result["theta_e_linear"])
    theta_f = float(result["theta_f_linear"])
    axis_rotation = float(result["physical_axis_rotation_deg"])
    offset = float(result["coordinate_offset_deg"])
    closure = float(result["closure_linear_arcsec"])
    legacy = float(result["nonlinear_mismatch_arcsec"])

    figure, (diagram, summary) = plt.subplots(1, 2, figsize=(13.0, 6.3))

    draw_ray(diagram, 0.0, 1.0, "Ecliptic +X", 1.8)
    draw_ray(diagram, axis_rotation, 1.0, "ICRF +X", 1.8)
    draw_ray(diagram, theta_e, 1.15, "Venus track", 2.4)

    diagram.add_patch(Arc((0, 0), 0.72, 0.72, theta1=axis_rotation, theta2=0, linewidth=1.2))
    diagram.add_patch(Arc((0, 0), 1.05, 1.05, theta1=0, theta2=theta_e, linewidth=1.2))
    diagram.add_patch(Arc((0, 0), 1.42, 1.42, theta1=axis_rotation, theta2=theta_e, linewidth=1.2))

    diagram.text(0.35, -0.035, f"Δθ = {offset:.6f}°", ha="center", va="top")
    diagram.text(0.50, 0.085, f"θₑ = {theta_e:.6f}°", ha="center", va="bottom")
    diagram.text(0.68, 0.015, f"θᶠ = {theta_f:.6f}°", ha="center", va="bottom")
    diagram.scatter([0.0], [0.0], s=28)
    diagram.set_aspect("equal", adjustable="box")
    diagram.set_xlim(-0.15, 1.35)
    diagram.set_ylim(-0.35, 0.55)
    diagram.axis("off")
    diagram.set_title("One physical track, two reference-plane axes")

    summary.axis("off")
    summary.set_title("Exact frame-rotation closure", pad=18)
    summary.text(
        0.04,
        0.87,
        r"$\theta_{\mathrm{ICRF}}=\theta_{\mathrm{ECLIPTIC}}+\Delta\theta$",
        fontsize=17,
    )
    summary.text(0.04, 0.73, f"Ecliptic track angle     {theta_e:12.6f}°", fontsize=13)
    summary.text(0.04, 0.64, f"Frame-axis offset        {offset:12.6f}°", fontsize=13)
    summary.text(0.04, 0.55, f"ICRF track angle         {theta_f:12.6f}°", fontsize=13)
    summary.text(0.04, 0.42, f"Exact closure             {closure:12.9f} arcsec", fontsize=13)
    summary.text(0.04, 0.29, f"Legacy atan2 residual     {legacy:12.9f} arcsec", fontsize=13)
    summary.text(
        0.04,
        0.12,
        "The old residual is a nonlinear coordinate-mapping artifact,\n"
        "not an astronomical discrepancy.",
        fontsize=12,
    )

    figure.suptitle(
        "2012 Venus Transit — Ecliptic and ICRF Solar-Screen Angles",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(PLOT_FILE, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"\nPRESENTATION PLOT: {PLOT_FILE}")


def main() -> None:
    result = derive()
    display(result)
    presentation_plot(result)


if __name__ == "__main__":
    main()
