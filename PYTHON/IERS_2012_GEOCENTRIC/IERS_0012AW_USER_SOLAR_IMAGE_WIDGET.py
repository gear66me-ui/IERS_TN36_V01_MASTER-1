# IERS-0012AW
# Audit reference: User-supplied solar image texture with verified Cape Town geometry and complete comparison widget.
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from IPython.display import Image as IPythonImage, display

import IERS_0012AU_CAPE_TOWN_COMPLETE_WIDGET as au

VERSION = "IERS-0012AW"
LOCAL_TZ = ZoneInfo("America/Bogota")
SITE_NAME = "Cape Town, South Africa"
EXPECTED_IMAGE_NAME = "1000008451.jpg"

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

PLOT_PNG = OUTPUT_PNG_DIR / f"{VERSION}_USER_SOLAR_IMAGE_WIDGET.png"
OUTPUT_CSV = OUTPUT_CSV_DIR / f"{VERSION}_USER_SOLAR_IMAGE_WIDGET.csv"

FIGURE_BG = "#210500"
AXES_BG = "#6e2107"
GRID = "#7f4b25"
TEXT = "#fff8df"
MUTED = "#e6c9a5"
JPL_COLOR = "#5fffd0"
VIDEO_COLOR = "#ff68d5"
VENUS_FILL = "#101010"
VENUS_EDGE = "#ffd66b"
CONTACT_EDGE = "#ffffff"
TABLE_BG = "#160b08"
TABLE_HEADER = "#4b1b0c"
TABLE_EDGE = "#b87537"
JPL_VALUE = "#76ffd8"
VIDEO_VALUE = "#ff86de"
DELTA_VALUE = "#ffe278"


def locate_user_image() -> Path:
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
            f"Solar image not found. Place {EXPECTED_IMAGE_NAME} beside the Python file."
        ) from exc

    print(f"Upload the second solar image as {EXPECTED_IMAGE_NAME}.")
    uploaded = files.upload()
    if not uploaded:
        raise RuntimeError("No solar image was uploaded.")

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
        raise RuntimeError("The uploaded file is not a supported image.")

    return Path(preferred)


def otsu_threshold(gray: np.ndarray) -> float:
    values = np.clip(np.asarray(gray, dtype=float), 0.0, 1.0)
    histogram, edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    probabilities = histogram.astype(float) / max(float(histogram.sum()), 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cumulative_probability = np.cumsum(probabilities)
    cumulative_mean = np.cumsum(probabilities * centers)
    total_mean = cumulative_mean[-1]
    denominator = cumulative_probability * (1.0 - cumulative_probability)
    numerator = (total_mean * cumulative_probability - cumulative_mean) ** 2
    score = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1.0e-15,
    )
    return float(centers[int(np.argmax(score))])


def crop_solar_disk(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image, dtype=float) / 255.0
    gray = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    threshold = max(otsu_threshold(gray), 0.18)
    mask = gray > threshold

    y_indices, x_indices = np.nonzero(mask)
    if len(x_indices) < 1000:
        raise RuntimeError("Solar disk detection failed in the uploaded image.")

    center_x = 0.5 * (float(x_indices.min()) + float(x_indices.max()))
    center_y = 0.5 * (float(y_indices.min()) + float(y_indices.max()))
    radius_x = 0.5 * (float(x_indices.max()) - float(x_indices.min()))
    radius_y = 0.5 * (float(y_indices.max()) - float(y_indices.min()))
    radius = 1.015 * max(radius_x, radius_y)

    left = max(0, int(round(center_x - radius)))
    right = min(rgb.shape[1], int(round(center_x + radius)))
    top = max(0, int(round(center_y - radius)))
    bottom = min(rgb.shape[0], int(round(center_y + radius)))
    crop = rgb[top:bottom, left:right]

    side = max(crop.shape[0], crop.shape[1])
    square = np.zeros((side, side, 3), dtype=float)
    square[:, :, 0] = 0.25
    square[:, :, 1] = 0.055
    square[:, :, 2] = 0.010
    y_offset = (side - crop.shape[0]) // 2
    x_offset = (side - crop.shape[1]) // 2
    square[
        y_offset : y_offset + crop.shape[0],
        x_offset : x_offset + crop.shape[1],
    ] = crop
    return square


def event_point(event, solar_radius: float) -> np.ndarray:
    return au.reflect_x_point(event.point_arcsec / solar_radius)


def render() -> None:
    image_path = locate_user_image()
    solar_texture = crop_solar_disk(image_path)

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
            "Orientation verification failed: expected C1 left/high and C4 "
            "right/lower in the upper half-Sun. "
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

    figure = plt.figure(figsize=(12.2, 8.4), dpi=190, facecolor=FIGURE_BG)
    ax = figure.add_axes([0.045, 0.065, 0.84, 0.86])
    ax.set_facecolor(AXES_BG)

    image_artist = ax.imshow(
        solar_texture,
        extent=[-1.0, 1.0, -1.0, 1.0],
        origin="upper",
        interpolation="lanczos",
        zorder=0,
    )
    image_artist.set_clip_path(Circle((0.0, 0.0), 1.0, transform=ax.transData))

    theta = np.linspace(0.0, math.pi, 1600)
    ax.plot(np.cos(theta), np.sin(theta), color="#fff0bb", linewidth=0.72, zorder=2)
    ax.plot([-1.0, 1.0], [0.0, 0.0], color="#ffbd52", linewidth=0.30, alpha=0.85, zorder=2)

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
            path_effects=[],
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

    ax.set_xlim(-1.025, 1.025)
    ax.set_ylim(-0.02, 1.025)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=GRID, linewidth=0.18, alpha=0.38)
    ax.tick_params(colors=MUTED, labelsize=7, width=0.25, length=2.0)
    for spine in ax.spines.values():
        spine.set_color(TABLE_EDGE)
        spine.set_linewidth(0.38)

    ax.set_xlabel("Solar-screen X (R_sun)", color=MUTED, fontsize=8)
    ax.set_ylabel("Solar-screen Y (R_sun)", color=MUTED, fontsize=8)
    ax.set_title(
        "2012 Venus Transit — Cape Town SITE_COORD vs SDO Video\n"
        "User Solar Image Texture — Verified Upper Half-Sun",
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
    print("User-supplied second solar image, Cape Town JPL SITE_COORD, and V0007 SDO fit")
    print("COMMENTS")
    print("No AI images; Python/Pillow/Matplotlib only; C1 and C4 use white outlines")
    print("RESULTS")
    print(f"PNG: {PLOT_PNG}")
    print("OUTPUT SUMMARY")
    print(f"CSV: {OUTPUT_CSV}")
    print("PAPER COMPARISON")
    print("NOT USED")
    print("EQUATION STATUS")
    print("Angles, delta metrics, cubic R-squared, RMS, site, and C1-C4 UTC rendered")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")
    return 0


if __name__ == "__main__":
    main()
# IERS-0012AW
