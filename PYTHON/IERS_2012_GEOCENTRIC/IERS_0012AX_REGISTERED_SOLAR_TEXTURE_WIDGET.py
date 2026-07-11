# IERS-0012AX
# Audit reference: Limb-fitted and remapped user solar texture with verified Cape Town geometry and complete comparison widget.
from __future__ import annotations

import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def ensure_package(import_name: str, pip_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pip_name])


for _import_name, _pip_name in (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("cv2", "opencv-python-headless"),
):
    ensure_package(_import_name, _pip_name)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from IPython.display import Image as IPythonImage, display

import IERS_0012AU_CAPE_TOWN_COMPLETE_WIDGET as au

VERSION = "IERS-0012AX"
LOCAL_TZ = ZoneInfo("America/Bogota")
SITE_NAME = "Cape Town, South Africa"
EXPECTED_IMAGE_NAME = "1000008451.jpg"
TEXTURE_SIZE = 1400

DRIVE_ROOT = Path("/content/drive/MyDrive/Colab Notebooks")
PROJECT_ROOT = (
    DRIVE_ROOT / "IERS_TN36_OUTPUT"
    if DRIVE_ROOT.exists()
    else Path("/content/IERS_TN36_OUTPUT")
)
OUTPUT_PNG_DIR = PROJECT_ROOT / "OUTPUT_PNG"
OUTPUT_CSV_DIR = PROJECT_ROOT / "OUTPUT_CSV"
OUTPUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)

PLOT_PNG = OUTPUT_PNG_DIR / f"{VERSION}_REGISTERED_SOLAR_TEXTURE_WIDGET.png"
OUTPUT_CSV = OUTPUT_CSV_DIR / f"{VERSION}_REGISTERED_SOLAR_TEXTURE_WIDGET.csv"

FIGURE_BG = "#1c0500"
AXES_BG = "#7a2507"
GRID = "#6e4024"
TEXT = "#fff7e6"
MUTED = "#e6c6a2"
JPL_COLOR = "#61ffd5"
VIDEO_COLOR = "#ff72d7"
VENUS_FILL = "#080808"
VENUS_EDGE = "#ffd76c"
CONTACT_EDGE = "#ffffff"
TABLE_BG = "#170b08"
TABLE_HEADER = "#4b1b0d"
TABLE_EDGE = "#b97a3c"
JPL_VALUE = "#7dffdc"
VIDEO_VALUE = "#ff91e2"
DELTA_VALUE = "#ffe47b"


def locate_image() -> Path:
    candidates = [
        Path.cwd() / EXPECTED_IMAGE_NAME,
        Path("/content") / EXPECTED_IMAGE_NAME,
        DRIVE_ROOT / EXPECTED_IMAGE_NAME,
        DRIVE_ROOT / "IERS_TN36_OUTPUT" / EXPECTED_IMAGE_NAME,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    try:
        from google.colab import files
    except ImportError as exc:
        raise FileNotFoundError(
            f"Place {EXPECTED_IMAGE_NAME} beside the Python file."
        ) from exc

    print(f"Upload the second solar image as {EXPECTED_IMAGE_NAME}.")
    uploaded = files.upload()
    if not uploaded:
        raise RuntimeError("No image was uploaded.")

    preferred = next(
        (name for name in uploaded if Path(name).name == EXPECTED_IMAGE_NAME),
        None,
    )
    if preferred is None:
        preferred = next(
            (
                name
                for name in uploaded
                if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ),
            None,
        )
    if preferred is None:
        raise RuntimeError("No supported image was uploaded.")
    return Path(preferred)


def largest_solar_contour(rgb_u8: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3.0)

    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _hierarchy = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise RuntimeError("No solar-disk contour was detected.")

    contour = max(contours, key=cv2.contourArea)
    image_area = float(rgb_u8.shape[0] * rgb_u8.shape[1])
    if cv2.contourArea(contour) < 0.12 * image_area:
        raise RuntimeError("Detected contour is too small to be the solar disk.")
    return contour


def fit_circle_least_squares(points_xy: np.ndarray) -> tuple[float, float, float, float]:
    points = np.asarray(points_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 100:
        raise RuntimeError("Insufficient limb points for circle fitting.")

    x = points[:, 0]
    y = points[:, 1]
    design = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    rhs = x * x + y * y
    solution, _residuals, _rank, _singular = np.linalg.lstsq(design, rhs, rcond=None)
    center_x, center_y, constant = solution
    radius_sq = constant + center_x * center_x + center_y * center_y
    if radius_sq <= 0.0:
        raise RuntimeError("Circle fit returned a non-positive radius.")
    radius = math.sqrt(float(radius_sq))

    radial = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    residual_rms = float(np.sqrt(np.mean((radial - radius) ** 2)))
    return float(center_x), float(center_y), float(radius), residual_rms


def robust_circle_fit(contour: np.ndarray) -> tuple[float, float, float, float]:
    points = contour.reshape(-1, 2).astype(float)
    center_x, center_y, radius, _rms = fit_circle_least_squares(points)

    for _iteration in range(4):
        radial = np.sqrt(
            (points[:, 0] - center_x) ** 2
            + (points[:, 1] - center_y) ** 2
        )
        residual = radial - radius
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        sigma = max(1.4826 * mad, 0.75)
        keep = np.abs(residual - median) <= 3.5 * sigma
        if np.count_nonzero(keep) < 100:
            break
        center_x, center_y, radius, _rms = fit_circle_least_squares(points[keep])

    radial = np.sqrt(
        (points[:, 0] - center_x) ** 2
        + (points[:, 1] - center_y) ** 2
    )
    final_rms = float(np.sqrt(np.mean((radial - radius) ** 2)))
    return center_x, center_y, radius, final_rms


def remap_solar_texture(
    rgb_u8: np.ndarray,
    center_x: float,
    center_y: float,
    radius_px: float,
) -> np.ndarray:
    size = TEXTURE_SIZE
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    x_unit, y_unit = np.meshgrid(axis, axis)

    map_x = (center_x + radius_px * x_unit).astype(np.float32)
    map_y = (center_y - radius_px * y_unit).astype(np.float32)

    remapped = cv2.remap(
        rgb_u8,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    ).astype(np.float32) / 255.0

    radial = np.sqrt(x_unit * x_unit + y_unit * y_unit)
    disk_mask = radial <= 1.0

    # Darken the real image while retaining its orange/yellow structure.
    luminance = (
        0.2126 * remapped[:, :, 0]
        + 0.7152 * remapped[:, :, 1]
        + 0.0722 * remapped[:, :, 2]
    )
    p_low, p_high = np.percentile(luminance[disk_mask], [2.0, 98.0])
    normalized = np.clip((luminance - p_low) / max(p_high - p_low, 1.0e-6), 0.0, 1.0)
    normalized = np.power(normalized, 0.85)

    graded = np.zeros_like(remapped)
    graded[:, :, 0] = 0.18 + 0.58 * normalized
    graded[:, :, 1] = 0.07 + 0.32 * normalized
    graded[:, :, 2] = 0.025 + 0.085 * normalized

    # Blend some original chromatic texture back in.
    graded = 0.76 * graded + 0.24 * np.clip(remapped * 0.62, 0.0, 1.0)

    alpha = np.zeros((size, size, 1), dtype=np.float32)
    feather = np.clip((1.0 - radial) / 0.006, 0.0, 1.0)
    alpha[:, :, 0] = feather
    return np.concatenate((np.clip(graded, 0.0, 1.0), alpha), axis=2)


def event_point(event, solar_radius: float) -> np.ndarray:
    return au.reflect_x_point(event.point_arcsec / solar_radius)


def render() -> None:
    image_path = locate_image()
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    rgb_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    contour = largest_solar_contour(rgb_u8)
    center_x, center_y, radius_px, limb_rms_px = robust_circle_fit(contour)
    texture_rgba = remap_solar_texture(rgb_u8, center_x, center_y, radius_px)

    geometry = au.base.build_geometry()
    events = geometry["events"]
    solar_radius = events["CA"].sun_radius_arcsec

    original_track = (
        np.asarray(geometry["fit_points_arcsec"], dtype=float) / solar_radius
    )
    track_rsun = au.reflect_x(original_track)

    cape_angle_deg = au.reduced_horizontal_angle_deg(track_rsun)
    cape_cubic_r_squared = au.cubic_regression_r_squared(track_rsun)
    cape_rms_rsun = au.perpendicular_rms(track_rsun)

    video_points, video_status = au.load_clean_video_points()
    video_cubic_r_squared = au.cubic_regression_r_squared(video_points)
    video_rms_rsun = au.perpendicular_rms(video_points)
    video_angle_deg = au.VIDEO_ANGLE_DEG

    delta_beta_deg = cape_angle_deg - video_angle_deg
    delta_m = (
        math.tan(math.radians(cape_angle_deg))
        - math.tan(math.radians(video_angle_deg))
    )
    if abs(delta_m) < 1.0e-15:
        raise RuntimeError("Delta-m is too small for the requested ratio.")
    delta_beta_over_delta_m_deg = delta_beta_deg / delta_m

    closest_rsun = event_point(events["CA"], solar_radius)
    video_line_rsun = au.make_video_reference_line(track_rsun, closest_rsun)

    c1_rsun = event_point(events["C1"], solar_radius)
    c4_rsun = event_point(events["C4"], solar_radius)
    if not (
        c1_rsun[0] < c4_rsun[0]
        and c1_rsun[1] > c4_rsun[1]
        and c1_rsun[1] > 0.0
        and c4_rsun[1] > 0.0
    ):
        raise RuntimeError(
            "Orientation verification failed. "
            f"C1=({c1_rsun[0]:.6f},{c1_rsun[1]:.6f}), "
            f"C4=({c4_rsun[0]:.6f},{c4_rsun[1]:.6f})"
        )

    metrics: dict[str, float | str] = {
        "cape_angle_deg": cape_angle_deg,
        "video_angle_deg": video_angle_deg,
        "delta_beta_deg": delta_beta_deg,
        "delta_m": delta_m,
        "delta_beta_over_delta_m_deg": delta_beta_over_delta_m_deg,
        "cape_cubic_r_squared": cape_cubic_r_squared,
        "video_cubic_r_squared": video_cubic_r_squared,
        "cape_rms_rsun": cape_rms_rsun,
        "video_rms_rsun": video_rms_rsun,
        "video_status": video_status,
    }
    au.OUTPUT_CSV = OUTPUT_CSV
    au.save_csv(geometry, track_rsun, video_line_rsun, metrics)

    figure = plt.figure(figsize=(12.2, 8.5), dpi=190, facecolor=FIGURE_BG)
    ax = figure.add_axes([0.045, 0.065, 0.84, 0.86])
    ax.set_facecolor(AXES_BG)

    ax.imshow(
        texture_rgba,
        extent=[-1.0, 1.0, -1.0, 1.0],
        origin="upper",
        interpolation="lanczos",
        zorder=0,
    )

    theta = np.linspace(0.0, math.pi, 1600)
    ax.plot(np.cos(theta), np.sin(theta), color="#fff1bc", linewidth=0.72, zorder=2)
    ax.plot([-1.0, 1.0], [0.0, 0.0], color="#ffc45f", linewidth=0.30, alpha=0.85, zorder=2)

    ax.plot(
        track_rsun[:, 0],
        track_rsun[:, 1],
        color=JPL_COLOR,
        linewidth=0.95,
        label="Cape Town JPL SITE_COORD",
        zorder=5,
    )
    ax.scatter(
        track_rsun[::8, 0],
        track_rsun[::8, 1],
        s=2.1,
        color=JPL_COLOR,
        linewidths=0,
        zorder=6,
    )
    ax.plot(
        video_line_rsun[:, 0],
        video_line_rsun[:, 1],
        color=VIDEO_COLOR,
        linewidth=1.05,
        linestyle=(0, (7, 5)),
        label="SDO video TLS reference",
        zorder=6,
    )

    label_offsets = {
        "C1": (-0.075, 0.030),
        "C2": (-0.064, 0.023),
        "CA": (-0.025, 0.037),
        "C3": (0.022, -0.038),
        "C4": (0.030, -0.048),
    }
    for event_name in ("C1", "C2", "CA", "C3", "C4"):
        event = events[event_name]
        point = event_point(event, solar_radius)
        radius = event.venus_radius_arcsec / solar_radius
        edge_color = CONTACT_EDGE if event_name in {"C1", "C4"} else VENUS_EDGE
        edge_width = 1.05 if event_name in {"C1", "C4"} else 0.70
        ax.add_patch(
            Circle(
                (point[0], point[1]),
                radius,
                facecolor=VENUS_FILL,
                edgecolor=edge_color,
                linewidth=edge_width,
                zorder=9,
            )
        )
        dx, dy = label_offsets[event_name]
        ax.text(
            point[0] + dx,
            point[1] + dy,
            event_name,
            color="#ffffff",
            fontsize=7.3,
            fontweight="bold",
            zorder=10,
        )

    statistics_rows = [
        ["Track angle β", f"{cape_angle_deg:.6f}", f"{video_angle_deg:.6f}", "deg"],
        ["Δβ", f"{delta_beta_deg:+.6f}", "", "deg"],
        ["Δm", f"{delta_m:+.9f}", "", ""],
        ["Δβ / Δm", f"{delta_beta_over_delta_m_deg:+.6f}", "", "deg"],
        ["Cubic regression R²", f"{cape_cubic_r_squared:.9f}", f"{video_cubic_r_squared:.9f}", ""],
        ["RMS perpendicular", f"{cape_rms_rsun:.9f}", f"{video_rms_rsun:.9f}", "R_sun"],
    ]
    statistics_table = ax.table(
        cellText=statistics_rows,
        colLabels=["Statistic", "Cape Town JPL", "SDO video", "Unit"],
        colWidths=[0.35, 0.24, 0.24, 0.11],
        bbox=[0.31, 0.045, 0.51, 0.235],
        cellLoc="center",
        colLoc="center",
        zorder=20,
    )
    statistics_table.auto_set_font_size(False)
    statistics_table.set_fontsize(6.8)
    for (row, column), cell in statistics_table.get_celld().items():
        cell.set_edgecolor(TABLE_EDGE)
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor(TABLE_HEADER)
            cell.get_text().set_color(TEXT)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(TABLE_BG)
            if row in (2, 3, 4) and column == 1:
                cell.get_text().set_color(DELTA_VALUE)
                cell.get_text().set_weight("bold")
            elif column == 1:
                cell.get_text().set_color(JPL_VALUE)
                cell.get_text().set_weight("bold")
            elif column == 2:
                cell.get_text().set_color(VIDEO_VALUE)
                cell.get_text().set_weight("bold")
            else:
                cell.get_text().set_color(MUTED)

    event_rows = [
        ["Site", SITE_NAME],
        ["C1", au.format_event_utc(events["C1"].utc)],
        ["C2", au.format_event_utc(events["C2"].utc)],
        ["C3", au.format_event_utc(events["C3"].utc)],
        ["C4", au.format_event_utc(events["C4"].utc)],
    ]
    event_table = ax.table(
        cellText=event_rows,
        colLabels=["Observation", "UTC / Location"],
        colWidths=[0.26, 0.70],
        bbox=[0.035, 0.045, 0.255, 0.235],
        cellLoc="left",
        colLoc="left",
        zorder=20,
    )
    event_table.auto_set_font_size(False)
    event_table.set_fontsize(6.45)
    for (row, column), cell in event_table.get_celld().items():
        cell.set_edgecolor(TABLE_EDGE)
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor(TABLE_HEADER)
            cell.get_text().set_color(TEXT)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(TABLE_BG)
            cell.get_text().set_color(TEXT if column == 0 else MUTED)
            if column == 0:
                cell.get_text().set_weight("bold")

    ax.text(
        0.565,
        0.292,
        "TRACK COMPARISON",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=TEXT,
        fontsize=8.4,
        fontweight="bold",
        zorder=21,
    )

    ax.text(
        0.985,
        0.015,
        f"Solar limb fit: center=({center_x:.2f}, {center_y:.2f}) px, "
        f"radius={radius_px:.2f} px, RMS={limb_rms_px:.3f} px",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=5.4,
        zorder=21,
    )

    ax.set_xlim(-1.025, 1.025)
    ax.set_ylim(-0.02, 1.025)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=GRID, linewidth=0.18, alpha=0.34)
    ax.tick_params(colors=MUTED, labelsize=7, width=0.25, length=2.0)
    for spine in ax.spines.values():
        spine.set_color(TABLE_EDGE)
        spine.set_linewidth(0.38)

    ax.set_xlabel("Solar-screen X (R_sun)", color=MUTED, fontsize=8)
    ax.set_ylabel("Solar-screen Y (R_sun)", color=MUTED, fontsize=8)
    ax.set_title(
        "2012 Venus Transit — Cape Town SITE_COORD vs SDO Video\n"
        "Limb-Fitted User Solar Texture — Verified Upper Half-Sun",
        color=TEXT,
        fontsize=10.5,
        pad=10,
    )

    legend = ax.legend(
        loc="upper right",
        fontsize=6.8,
        frameon=True,
        handlelength=3.0,
    )
    legend.get_frame().set_facecolor(TABLE_BG)
    legend.get_frame().set_edgecolor(TABLE_EDGE)
    legend.get_frame().set_linewidth(0.35)
    for item in legend.get_texts():
        item.set_color(TEXT)

    figure.savefig(
        PLOT_PNG,
        dpi=320,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
    )
    plt.close(figure)
    display(IPythonImage(filename=str(PLOT_PNG)))


def main() -> int:
    render()
    print("CODE INPUTS")
    print("User second solar image, Cape Town JPL SITE_COORD, and V0007 SDO fit")
    print("COMMENTS")
    print("No AI images; solar limb fitted and remapped to the unit disk before plotting")
    print("RESULTS")
    print(f"PNG: {PLOT_PNG}")
    print("OUTPUT SUMMARY")
    print(f"CSV: {OUTPUT_CSV}")
    print("PAPER COMPARISON")
    print("NOT USED")
    print("EQUATION STATUS")
    print("Limb fit, track angles, delta metrics, cubic R-squared, RMS, site, and contacts rendered")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")
    return 0


if __name__ == "__main__":
    main()
# IERS-0012AX
