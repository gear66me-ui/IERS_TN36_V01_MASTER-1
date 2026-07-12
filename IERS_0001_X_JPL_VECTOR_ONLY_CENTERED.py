"""JPL-vector-only three-plane audit and centered plot. Matplotlib only."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

import IERS_0001_K_JPL_FRAME_ANGLE_CONFIRMATION as jpl

VERSION = "IERS-0001-X"
OUTPUT_PNG = "IERS_0001_X_JPL_VECTOR_ONLY_CENTERED.png"
OUTPUT_PDF = "IERS_0001_X_JPL_VECTOR_ONLY_CENTERED.pdf"

BLUE = "#4EA3FF"
ORANGE = "#FFB347"
GREEN = "#4DFF88"
WHITE = "#F4F4F4"
GRID = "#BFC5CC"
PANEL = "#080A0D"


def physical_displacement(sun: np.ndarray, venus: np.ndarray, normal: np.ndarray) -> np.ndarray:
    scale = float(np.dot(sun, normal)) / float(np.dot(venus, normal))
    displacement = scale * venus - sun
    return displacement - normal * np.dot(displacement, normal)


def pca_angle(points: np.ndarray) -> tuple[float, float]:
    center = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    direction = vh[0]
    if direction[0] < 0.0:
        direction = -direction
    direction = jpl.unit(direction)
    normal = np.array([-direction[1], direction[0]])
    residuals = (points - center) @ normal
    return (
        float(np.degrees(np.arctan2(direction[1], direction[0]))),
        float(np.sqrt(np.mean(residuals**2))),
    )


def solve_jpl_rotation(e_data: dict, f_data: dict) -> tuple[np.ndarray, float]:
    e_rows, f_rows = [], []
    for key in ("GS", "GV", "NS", "NV"):
        e_rows.extend(jpl.unit(row) for row in e_data[key]["xyz"])
        f_rows.extend(jpl.unit(row) for row in f_data[key]["xyz"])
    a = np.asarray(e_rows)
    b = np.asarray(f_rows)
    u, _, vt = np.linalg.svd(a.T @ b)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    transformed = (rotation @ a.T).T
    return rotation, float(np.max(np.linalg.norm(transformed - b, axis=1)))


def signed_angle(first: np.ndarray, second: np.ndarray, normal: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(
        np.dot(normal, np.cross(first, second)),
        np.dot(first, second),
    )))


def derive() -> dict[str, float | int]:
    e_data = jpl.fetch_plane("ECLIPTIC")
    f_data = jpl.fetch_plane("FRAME")
    e_cache = jpl.build_cache(e_data)
    f_cache = jpl.build_cache(f_data)

    c1_jd, c4_jd = jpl.north_pole_contact_interval(e_cache)
    ca_jd = jpl.geocentric_closest_approach(e_cache)
    rotation, rotation_residual = solve_jpl_rotation(e_data, f_data)

    n_e, x_e, y_e = jpl.solar_screen_basis(e_cache, ca_jd)
    n_f, x_f, y_f = jpl.solar_screen_basis(f_cache, ca_jd)
    x_e_in_f = jpl.unit(rotation @ x_e)
    basis_separation = -signed_angle(x_e_in_f, x_f, n_f)

    epochs = np.asarray(e_cache["jd"], dtype=float)
    fit_epochs = epochs[(epochs >= c1_jd) & (epochs <= c4_jd)]
    e_points, f_points = [], []
    for jd in fit_epochs:
        d_e = physical_displacement(
            jpl.vector(e_cache, "NS", jd), jpl.vector(e_cache, "NV", jd), n_e
        )
        d_f = physical_displacement(
            jpl.vector(f_cache, "NS", jd), jpl.vector(f_cache, "NV", jd), n_f
        )
        e_points.append([np.dot(d_e, x_e), np.dot(d_e, y_e)])
        f_points.append([np.dot(d_f, x_f), np.dot(d_f, y_f)])

    theta_e, rms_e = pca_angle(np.asarray(e_points))
    theta_f, rms_f = pca_angle(np.asarray(f_points))
    difference = theta_f - theta_e
    closure = (difference - basis_separation) * 3600.0

    return {
        "theta_e": theta_e,
        "theta_f": theta_f,
        "basis": basis_separation,
        "difference": difference,
        "closure": closure,
        "rotation_residual": rotation_residual,
        "rms_e": rms_e,
        "rms_f": rms_f,
        "rows": int(fit_epochs.size),
    }


def polar(angle: float, radius: float) -> np.ndarray:
    radians = np.radians(angle)
    return radius * np.array([np.cos(radians), np.sin(radians)])


def diameter(ax, angle, color, linewidth, linestyle):
    end = polar(angle, 1.12)
    ax.plot([-end[0], end[0]], [-end[1], end[1]], color=color,
            linewidth=linewidth, linestyle=linestyle, solid_capstyle="round")
    ax.annotate("", xy=end, xytext=end * 0.925,
                arrowprops={"arrowstyle": "-|>", "color": color,
                            "linewidth": linewidth, "mutation_scale": 8})


def angle_arc(ax, start, stop, radius, color) -> np.ndarray:
    angles = np.linspace(start, stop, 180)
    points = np.asarray([polar(angle, radius) for angle in angles])
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=0.75)
    ax.annotate("", xy=points[-1], xytext=points[-13],
                arrowprops={"arrowstyle": "-|>", "color": color,
                            "linewidth": 0.75, "mutation_scale": 8})
    return points


def centered_table(fig, result):
    theta_e = float(result["theta_e"])
    theta_f = float(result["theta_f"])
    basis = float(result["basis"])
    closure = float(result["closure"])
    rows = [
        ["Venus track vs ECLIPTIC", f"{theta_e:.6f}°", "JPL ECLIPTIC vectors + PCA"],
        ["Venus track vs ICRF", f"{theta_f:.6f}°", "JPL FRAME vectors + PCA"],
        ["ICRF–ECLIPTIC basis separation", f"{basis:.6f}°", "paired JPL vectors + SVD"],
        ["Independent difference",
         f"{theta_f:.6f}° − {theta_e:.6f}° = {theta_f-theta_e:.6f}°",
         f"closure {closure:.9f} arcsec"],
    ]
    table_ax = fig.add_axes([0.17, 0.025, 0.66, 0.185], facecolor="black")
    table_ax.axis("off")
    table = table_ax.table(cellText=rows,
        colLabels=["JPL-DERIVED QUANTITY", "ANGLE", "METHOD"],
        cellLoc="left", colLoc="left", colWidths=[0.34, 0.36, 0.30], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.6)
    table.scale(1.0, 1.40)
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor(PANEL)
        cell.set_edgecolor("#5C626A")
        cell.set_linewidth(0.55)
        cell.get_text().set_color(WHITE)
        if row == 0:
            cell.set_facecolor("#11151A")
            cell.get_text().set_weight("bold")
    table[(1, 0)].get_text().set_color(BLUE)
    table[(2, 0)].get_text().set_color(GREEN)
    table[(3, 0)].get_text().set_color(ORANGE)
    table[(4, 1)].get_text().set_weight("bold")
    table[(4, 2)].get_text().set_weight("bold")


def render(result):
    theta_e = float(result["theta_e"])
    theta_f = float(result["theta_f"])
    basis = float(result["basis"])

    ecliptic = theta_e - theta_e
    celestial = basis
    venus = -theta_e

    plt.close("all")
    fig = plt.figure(figsize=(14, 9), facecolor="black")
    ax = fig.add_axes([0.045, 0.220, 0.91, 0.680], facecolor="black")
    ax.set_aspect("equal")
    ax.set_xlim(-1.48, 1.52)
    ax.set_ylim(-1.08, 1.14)
    ax.axis("off")

    fig.text(0.50, 0.965, "JPL THREE-PLANE SPHERE GEOMETRY",
             color=WHITE, fontsize=23, fontweight="bold", ha="center", va="top")
    fig.text(0.50, 0.927,
             "Every displayed angle is independently derived from paired JPL Horizons vectors",
             color="#D6D6D6", fontsize=12.5, ha="center", va="top")

    ax.add_patch(Circle((0, 0), 1.0, edgecolor=GRID, facecolor="none", linewidth=0.80))
    diameter(ax, ecliptic, BLUE, 0.95, "-")
    diameter(ax, celestial, ORANGE, 0.88, "--")
    diameter(ax, venus, GREEN, 1.05, "-")
    ax.scatter([0], [0], s=15, color=WHITE, zorder=9)

    ax.text(-1.07, -0.045, "ECLIPTIC PLANE", color=BLUE, fontsize=13.5,
            ha="right", va="top",
            bbox={"facecolor": "black", "edgecolor": "none", "pad": 1.0})

    c_anchor = polar(celestial, -1.09)
    ax.text(c_anchor[0] - 0.025, c_anchor[1] - 0.012,
            "ICRF / CELESTIAL PLANE", color=ORANGE, fontsize=13.5,
            ha="right", va="center", rotation=celestial, rotation_mode="anchor",
            bbox={"facecolor": "black", "edgecolor": "none", "pad": 1.0})

    v_anchor = polar(venus, 1.12)
    ax.text(v_anchor[0] + 0.035, v_anchor[1] - 0.025, "VENUS TRANSIT PLANE",
            color=GREEN, fontsize=13.5, ha="left", va="top",
            rotation=venus, rotation_mode="anchor")

    b_arc = angle_arc(ax, ecliptic, celestial, 0.31, ORANGE)
    e_arc = angle_arc(ax, venus, ecliptic, 0.56, BLUE)
    f_arc = angle_arc(ax, venus, celestial, 0.84, GREEN)

    ax.annotate(rf"$\Delta_{{EF}}={basis:.6f}^\circ$",
                xy=b_arc[len(b_arc)//2], xytext=(0.03, 0.43), color=ORANGE,
                fontsize=13, ha="center",
                arrowprops={"arrowstyle": "-", "color": ORANGE, "linewidth": 0.55})
    ax.annotate(rf"$\theta_E={theta_e:.6f}^\circ$",
                xy=e_arc[len(e_arc)//2], xytext=(0.16, -0.50), color=BLUE,
                fontsize=13, ha="center",
                arrowprops={"arrowstyle": "-", "color": BLUE, "linewidth": 0.55})
    ax.annotate(rf"$\theta_F={theta_f:.6f}^\circ$",
                xy=f_arc[len(f_arc)//2], xytext=(1.18, 0.18), color=GREEN,
                fontsize=13.5, ha="left", va="center",
                arrowprops={"arrowstyle": "-", "color": GREEN, "linewidth": 0.60})

    centered_table(fig, result)
    fig.savefig(OUTPUT_PNG, dpi=220, facecolor="black", bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, facecolor="black", bbox_inches="tight")
    plt.show()


def audit(result):
    print("IERS TN36 - JPL Vector-Only Audit")
    print(f"Version                         {VERSION}")
    print("Target/reference angles entered NONE")
    print("Frame-offset constant entered   NONE")
    print("Empirical multiplier entered    NONE")
    print("Frame transform source          paired JPL vectors / SVD")
    print(f"JPL transform residual           {result['rotation_residual']:.6e}")
    print(f"Ecliptic track angle             {result['theta_e']:.6f} deg")
    print(f"ICRF track angle                 {result['theta_f']:.6f} deg")
    print(f"JPL basis separation             {result['basis']:.6f} deg")
    print(f"Independent subtraction          {result['difference']:.6f} deg")
    print(f"Closure                          {result['closure']:.9f} arcsec")
    print(f"Fit rows                         {result['rows']}")
    print(f"PNG                              {OUTPUT_PNG}")
    print(f"PDF                              {OUTPUT_PDF}")


def main():
    result = derive()
    audit(result)
    render(result)


if __name__ == "__main__":
    main()
