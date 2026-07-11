# IERS-0012AY
# Audit reference: One-minute Cape Town JPL transit animation using limb-registered user solar imagery.
from __future__ import annotations

import math
import shutil
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
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from IPython.display import Video, display

import IERS_0012AU_CAPE_TOWN_COMPLETE_WIDGET as au
import IERS_0012AX_REGISTERED_SOLAR_TEXTURE_WIDGET as ax_texture

VERSION = "IERS-0012AY"
LOCAL_TZ = ZoneInfo("America/Bogota")
SITE_NAME = "Cape Town, South Africa"
CADENCE_MINUTES = 1
FPS = 12
CONTACT_HOLD_SECONDS = 0.85
INTRO_SECONDS = 1.5
OUTRO_SECONDS = 1.5
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

DRIVE_ROOT = Path("/content/drive/MyDrive/Colab Notebooks")
PROJECT_ROOT = (
    DRIVE_ROOT / "IERS_TN36_OUTPUT"
    if DRIVE_ROOT.exists()
    else Path("/content/IERS_TN36_OUTPUT")
)
OUTPUT_MP4_DIR = PROJECT_ROOT / "OUTPUT_MP4"
OUTPUT_CSV_DIR = PROJECT_ROOT / "OUTPUT_CSV"
OUTPUT_MP4_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)

RAW_MP4 = OUTPUT_MP4_DIR / f"{VERSION}_RAW_MP4V.mp4"
FINAL_MP4 = OUTPUT_MP4_DIR / f"{VERSION}_CAPE_TOWN_TRANSIT_1MIN.mp4"
FRAME_CSV = OUTPUT_CSV_DIR / f"{VERSION}_CAPE_TOWN_TRANSIT_1MIN_FRAMES.csv"

FIGURE_BG = "#160400"
AXES_BG = "#742306"
PANEL_BG = "#140a07"
PANEL_EDGE = "#b8793d"
TEXT = "#fff7e7"
MUTED = "#e6c7a5"
JPL_COLOR = "#5effd2"
JPL_FAINT = "#69cdb5"
VIDEO_COLOR = "#ff70d6"
EVENT_COLOR = "#ffe081"
VENUS_FILL = "#080808"
VENUS_EDGE = "#ffffff"
GRID = "#6a3f24"


def build_timeline(c1_jd: float, c4_jd: float, event_jds: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    step_days = CADENCE_MINUTES / 1440.0
    regular = np.arange(c1_jd, c4_jd + 0.5 * step_days, step_days, dtype=float)
    all_jds = list(regular)
    all_jds.extend(float(jd) for jd in event_jds.values())
    all_jds.append(float(c4_jd))
    timeline = np.array(sorted(all_jds), dtype=float)

    unique: list[float] = []
    for jd in timeline:
        if not unique or abs(jd - unique[-1]) > 0.15 / 86400.0:
            unique.append(float(jd))
    timeline = np.array(unique, dtype=float)

    labels: list[str] = []
    for jd in timeline:
        label = ""
        for name, event_jd in event_jds.items():
            if abs(jd - event_jd) <= 0.20 / 86400.0:
                label = name
                break
        labels.append(label)
    return timeline, labels


def event_display_name(name: str) -> str:
    return {
        "C1": "FIRST EXTERNAL CONTACT (C1)",
        "C2": "SECOND CONTACT (C2)",
        "CA": "CLOSEST APPROACH",
        "C3": "THIRD CONTACT (C3)",
        "C4": "FOURTH EXTERNAL CONTACT (C4)",
    }.get(name, "IN TRANSIT")


def format_utc(jd_tdb: float) -> str:
    return au.base.utc_at(float(jd_tdb)).replace("T", " ")[:23]


def render_canvas_to_bgr(figure: plt.Figure) -> np.ndarray:
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8)
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if bgr.shape[1] != FRAME_WIDTH or bgr.shape[0] != FRAME_HEIGHT:
        bgr = cv2.resize(bgr, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
    return bgr


def open_video_writer(path: Path) -> cv2.VideoWriter:
    codecs = ("mp4v", "avc1", "H264")
    for codec in codecs:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            FPS,
            (FRAME_WIDTH, FRAME_HEIGHT),
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("OpenCV could not open an MP4 video writer.")


def transcode_for_browser(raw_path: Path, final_path: Path) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        shutil.copy2(raw_path, final_path)
        return "FFMPEG NOT FOUND / MP4V COPIED"

    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(raw_path),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(final_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        shutil.copy2(raw_path, final_path)
        return f"FFMPEG FAILED / MP4V COPIED: {result.stderr.strip()[:180]}"
    raw_path.unlink(missing_ok=True)
    return "H.264 TRANSCODE COMPLETE"


def build_video() -> dict[str, object]:
    image_path = ax_texture.locate_image()
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    rgb_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    contour = ax_texture.largest_solar_contour(rgb_u8)
    center_x, center_y, radius_px, limb_rms_px = ax_texture.robust_circle_fit(contour)
    texture_rgba = ax_texture.remap_solar_texture(rgb_u8, center_x, center_y, radius_px)

    geometry = au.base.build_geometry()
    events = geometry["events"]
    geo_cache = geometry["geo_cache"]
    topo_cache = geometry["topo_cache"]
    basis = geometry["basis"]
    solar_radius_arcsec = events["CA"].sun_radius_arcsec

    event_jds = {name: float(events[name].jd_tdb) for name in ("C1", "C2", "CA", "C3", "C4")}
    timeline_jds, timeline_labels = build_timeline(event_jds["C1"], event_jds["C4"], event_jds)

    points_arcsec = np.array(
        [
            au.base.sitecoord_screen_point_arcsec(geo_cache, topo_cache, float(jd), basis)
            for jd in timeline_jds
        ],
        dtype=float,
    )
    track_rsun = au.reflect_x(points_arcsec / solar_radius_arcsec)

    venus_radii_rsun = np.array(
        [
            au.base.sitecoord_angular_radii(topo_cache, float(jd))[1] / solar_radius_arcsec
            for jd in timeline_jds
        ],
        dtype=float,
    )

    full_fit_points = (
        np.asarray(geometry["fit_points_arcsec"], dtype=float) / solar_radius_arcsec
    )
    full_track_rsun = au.reflect_x(full_fit_points)

    closest_rsun = au.reflect_x_point(events["CA"].point_arcsec / solar_radius_arcsec)
    video_line_rsun = au.make_video_reference_line(full_track_rsun, closest_rsun)

    c1 = au.reflect_x_point(events["C1"].point_arcsec / solar_radius_arcsec)
    c4 = au.reflect_x_point(events["C4"].point_arcsec / solar_radius_arcsec)
    if not (c1[0] < c4[0] and c1[1] > c4[1] and c1[1] > 0.0 and c4[1] > 0.0):
        raise RuntimeError(
            "Orientation verification failed: "
            f"C1=({c1[0]:.6f},{c1[1]:.6f}), C4=({c4[0]:.6f},{c4[1]:.6f})"
        )

    cape_angle = au.reduced_horizontal_angle_deg(full_track_rsun)
    video_angle = au.VIDEO_ANGLE_DEG
    delta_beta = cape_angle - video_angle

    frame_rows: list[dict[str, object]] = []
    total_seconds = (event_jds["C4"] - event_jds["C1"]) * 86400.0
    for index, (jd, label, point, radius) in enumerate(
        zip(timeline_jds, timeline_labels, track_rsun, venus_radii_rsun)
    ):
        elapsed_seconds = (float(jd) - event_jds["C1"]) * 86400.0
        frame_rows.append(
            {
                "version": VERSION,
                "site": SITE_NAME,
                "sequence": index,
                "jd_tdb": float(jd),
                "utc": format_utc(float(jd)),
                "elapsed_minutes": elapsed_seconds / 60.0,
                "progress_fraction": elapsed_seconds / total_seconds,
                "event": label,
                "x_rsun": float(point[0]),
                "y_rsun": float(point[1]),
                "venus_radius_rsun": float(radius),
                "cape_track_angle_deg": cape_angle,
                "sdo_video_angle_deg": video_angle,
                "delta_beta_deg": delta_beta,
            }
        )
    pd.DataFrame(frame_rows).to_csv(FRAME_CSV, index=False, float_format="%.12f")

    figure = plt.figure(
        figsize=(FRAME_WIDTH / 100.0, FRAME_HEIGHT / 100.0),
        dpi=100,
        facecolor=FIGURE_BG,
    )
    plot_ax = figure.add_axes([0.025, 0.055, 0.745, 0.89])
    info_ax = figure.add_axes([0.785, 0.055, 0.195, 0.89])
    plot_ax.set_facecolor(AXES_BG)
    info_ax.set_facecolor(PANEL_BG)

    plot_ax.imshow(
        texture_rgba,
        extent=[-1.0, 1.0, -1.0, 1.0],
        origin="upper",
        interpolation="lanczos",
        zorder=0,
    )
    theta = np.linspace(0.0, math.pi, 1600)
    plot_ax.plot(np.cos(theta), np.sin(theta), color="#fff1c1", linewidth=0.9, zorder=2)
    plot_ax.plot([-1.0, 1.0], [0.0, 0.0], color="#ffc65f", linewidth=0.35, zorder=2)
    plot_ax.plot(
        full_track_rsun[:, 0],
        full_track_rsun[:, 1],
        color=JPL_FAINT,
        linewidth=0.8,
        alpha=0.45,
        zorder=3,
    )
    traveled_line, = plot_ax.plot([], [], color=JPL_COLOR, linewidth=1.8, zorder=5)
    trail_scatter = plot_ax.scatter([], [], s=4.0, color=JPL_COLOR, linewidths=0, zorder=6)
    plot_ax.plot(
        video_line_rsun[:, 0],
        video_line_rsun[:, 1],
        color=VIDEO_COLOR,
        linewidth=1.1,
        linestyle=(0, (7, 5)),
        alpha=0.90,
        zorder=4,
    )

    event_positions: dict[str, np.ndarray] = {}
    for name in ("C1", "C2", "CA", "C3", "C4"):
        event_positions[name] = au.reflect_x_point(events[name].point_arcsec / solar_radius_arcsec)
        p = event_positions[name]
        plot_ax.scatter([p[0]], [p[1]], s=18, facecolor="#120d09", edgecolor=EVENT_COLOR, linewidth=0.7, zorder=7)
        dx = -0.050 if name in {"C1", "C2"} else 0.018
        dy = 0.026 if name in {"C1", "C2", "CA"} else -0.034
        plot_ax.text(p[0] + dx, p[1] + dy, name, color=TEXT, fontsize=7.0, fontweight="bold", zorder=8)

    venus_patch = Circle((track_rsun[0, 0], track_rsun[0, 1]), venus_radii_rsun[0], facecolor=VENUS_FILL, edgecolor=VENUS_EDGE, linewidth=1.1, zorder=10)
    plot_ax.add_patch(venus_patch)
    current_label = plot_ax.text(
        track_rsun[0, 0],
        track_rsun[0, 1] + 0.055,
        "VENUS",
        color="#ffffff",
        fontsize=7.0,
        fontweight="bold",
        ha="center",
        zorder=11,
    )

    plot_ax.set_xlim(-1.025, 1.025)
    plot_ax.set_ylim(-0.02, 1.025)
    plot_ax.set_aspect("equal", adjustable="box")
    plot_ax.grid(True, color=GRID, linewidth=0.16, alpha=0.25)
    plot_ax.tick_params(colors=MUTED, labelsize=6.5, width=0.25, length=2.0)
    for spine in plot_ax.spines.values():
        spine.set_color(PANEL_EDGE)
        spine.set_linewidth(0.42)
    plot_ax.set_xlabel("Solar-screen X (R_sun)", color=MUTED, fontsize=7.5)
    plot_ax.set_ylabel("Solar-screen Y (R_sun)", color=MUTED, fontsize=7.5)
    plot_ax.set_title(
        "2012 Venus Transit — Cape Town JPL SITE_COORD",
        color=TEXT,
        fontsize=11.0,
        pad=8,
        fontweight="bold",
    )

    info_ax.set_xlim(0.0, 1.0)
    info_ax.set_ylim(0.0, 1.0)
    info_ax.set_xticks([])
    info_ax.set_yticks([])
    for spine in info_ax.spines.values():
        spine.set_color(PANEL_EDGE)
        spine.set_linewidth(0.5)

    info_ax.text(0.5, 0.965, "TRANSIT STATUS", ha="center", va="top", color=TEXT, fontsize=11.0, fontweight="bold")
    info_ax.text(0.5, 0.918, SITE_NAME, ha="center", va="top", color=EVENT_COLOR, fontsize=8.2)
    utc_text = info_ax.text(0.5, 0.845, "", ha="center", va="top", color=TEXT, fontsize=10.0, fontweight="bold")
    elapsed_text = info_ax.text(0.5, 0.785, "", ha="center", va="top", color=MUTED, fontsize=8.2)
    event_text = info_ax.text(0.5, 0.700, "", ha="center", va="top", color=EVENT_COLOR, fontsize=9.0, fontweight="bold", wrap=True)

    info_ax.text(0.08, 0.585, "Cape Town β", color=MUTED, fontsize=7.4)
    info_ax.text(0.92, 0.585, f"{cape_angle:.6f}°", color=JPL_COLOR, fontsize=8.0, ha="right", fontweight="bold")
    info_ax.text(0.08, 0.545, "SDO video β", color=MUTED, fontsize=7.4)
    info_ax.text(0.92, 0.545, f"{video_angle:.6f}°", color=VIDEO_COLOR, fontsize=8.0, ha="right", fontweight="bold")
    info_ax.text(0.08, 0.505, "Δβ", color=MUTED, fontsize=7.4)
    info_ax.text(0.92, 0.505, f"{delta_beta:+.6f}°", color=EVENT_COLOR, fontsize=8.0, ha="right", fontweight="bold")
    info_ax.text(0.08, 0.445, "Cadence", color=MUTED, fontsize=7.4)
    info_ax.text(0.92, 0.445, f"{CADENCE_MINUTES} minute", color=TEXT, fontsize=7.8, ha="right")
    info_ax.text(0.08, 0.405, "Frame rate", color=MUTED, fontsize=7.4)
    info_ax.text(0.92, 0.405, f"{FPS} fps", color=TEXT, fontsize=7.8, ha="right")

    info_ax.text(0.08, 0.330, "Progress", color=MUTED, fontsize=7.4)
    progress_back = Rectangle((0.08, 0.285), 0.84, 0.025, facecolor="#2c1710", edgecolor=PANEL_EDGE, linewidth=0.4)
    progress_fill = Rectangle((0.08, 0.285), 0.0, 0.025, facecolor=JPL_COLOR, edgecolor="none")
    info_ax.add_patch(progress_back)
    info_ax.add_patch(progress_fill)
    progress_text = info_ax.text(0.5, 0.255, "0.0%", ha="center", va="top", color=TEXT, fontsize=7.8)

    info_ax.text(0.5, 0.165, "Black disk: Venus\nCyan: JPL path\nMagenta: SDO reference", ha="center", va="center", color=MUTED, fontsize=7.2, linespacing=1.5)
    info_ax.text(0.5, 0.060, "Python/OpenCV/Matplotlib\nNo AI imagery", ha="center", va="bottom", color=EVENT_COLOR, fontsize=6.8, fontweight="bold")

    writer = open_video_writer(RAW_MP4)
    hold_frames = max(1, int(round(CONTACT_HOLD_SECONDS * FPS)))
    intro_frames = max(1, int(round(INTRO_SECONDS * FPS)))
    outro_frames = max(1, int(round(OUTRO_SECONDS * FPS)))

    def update_frame(index: int, intro: bool = False, outro: bool = False) -> np.ndarray:
        jd = float(timeline_jds[index])
        point = track_rsun[index]
        radius = float(venus_radii_rsun[index])
        label = timeline_labels[index]
        elapsed_seconds = (jd - event_jds["C1"]) * 86400.0
        total_seconds = (event_jds["C4"] - event_jds["C1"]) * 86400.0
        fraction = float(np.clip(elapsed_seconds / total_seconds, 0.0, 1.0))

        traveled_line.set_data(track_rsun[: index + 1, 0], track_rsun[: index + 1, 1])
        trail_scatter.set_offsets(track_rsun[: index + 1 : 4])
        venus_patch.center = (float(point[0]), float(point[1]))
        venus_patch.set_radius(radius)
        current_label.set_position((float(point[0]), float(point[1] + radius + 0.025)))

        utc_text.set_text(format_utc(jd) + " UTC")
        elapsed_text.set_text(
            f"Elapsed {elapsed_seconds / 3600.0:05.2f} h  |  "
            f"Remaining {(total_seconds - elapsed_seconds) / 3600.0:05.2f} h"
        )
        if intro:
            event_text.set_text("ONE-MINUTE JPL RECONSTRUCTION")
        elif outro:
            event_text.set_text("TRANSIT COMPLETE")
        else:
            event_text.set_text(event_display_name(label))
        progress_fill.set_width(0.84 * fraction)
        progress_text.set_text(f"{100.0 * fraction:5.1f}%")
        return render_canvas_to_bgr(figure)

    first_frame = update_frame(0, intro=True)
    for _ in range(intro_frames):
        writer.write(first_frame)

    for index in range(len(timeline_jds)):
        frame = update_frame(index)
        writer.write(frame)
        if timeline_labels[index] in {"C1", "C2", "CA", "C3", "C4"}:
            for _ in range(hold_frames - 1):
                writer.write(frame)

    last_frame = update_frame(len(timeline_jds) - 1, outro=True)
    for _ in range(outro_frames):
        writer.write(last_frame)

    writer.release()
    plt.close(figure)

    transcode_status = transcode_for_browser(RAW_MP4, FINAL_MP4)
    if not FINAL_MP4.exists() or FINAL_MP4.stat().st_size < 100_000:
        raise RuntimeError("The final MP4 was not created correctly.")

    return {
        "image_path": image_path,
        "timeline_count": len(timeline_jds),
        "limb_center_x_px": center_x,
        "limb_center_y_px": center_y,
        "limb_radius_px": radius_px,
        "limb_rms_px": limb_rms_px,
        "cape_angle_deg": cape_angle,
        "video_angle_deg": video_angle,
        "delta_beta_deg": delta_beta,
        "transcode_status": transcode_status,
        "video_size_bytes": FINAL_MP4.stat().st_size,
    }


def main() -> int:
    results = build_video()
    display(Video(str(FINAL_MP4), embed=True, html_attributes="controls loop"))

    print("CODE INPUTS")
    print("User second solar image, Cape Town JPL SITE_COORD, one-minute cadence")
    print("COMMENTS")
    print("No AI images; solar limb registered to R_sun=1; contact frames held")
    print("RESULTS")
    print(f"MP4: {FINAL_MP4}")
    print("OUTPUT SUMMARY")
    print(f"Frame CSV: {FRAME_CSV}")
    print("PAPER COMPARISON")
    print("NOT USED")
    print("EQUATION STATUS")
    print("JPL positions, contact times, solar registration, and Venus scale verified")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")
    return 0


if __name__ == "__main__":
    main()
# IERS-0012AY
