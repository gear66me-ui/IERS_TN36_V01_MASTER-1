# IERS-0012AU
# Audit reference: Verified Cape Town upper-half widget with angles, delta metrics, cubic regression R-squared, RMS, city, and contact times.
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from IPython.display import Image as IPythonImage, display

import IERS_0012AJ_CAPE_TOWN_SITE_COORD_VS_VIDEO as base

VERSION = "IERS-0012AU"
LOCAL_TZ = ZoneInfo("America/Bogota")
VIDEO_ANGLE_DEG = 8.4340601435
SITE_NAME = "Cape Town, South Africa"

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

PLOT_PNG = OUTPUT_PNG_DIR / f"{VERSION}_CAPE_TOWN_COMPLETE_WIDGET.png"
OUTPUT_CSV = OUTPUT_CSV_DIR / f"{VERSION}_CAPE_TOWN_COMPLETE_WIDGET.csv"

VIDEO_CSV = (
    DRIVE_ROOT
    / "ESO Parallax 2012 Venus Transit"
    / "NASA_SDO_2012_OUTPUT"
    / "V0007"
    / "NASA_SDO_2012_TRACK.csv"
)

BG = "#02070b"
PANEL = "#071116"
GRID = "#10252f"
SPINE = "#245d70"
TEXT = "#dff8ff"
MUTED = "#8fb4c1"
SOLAR = "#50b9d6"
JPL_COLOR = "#59d987"
VIDEO_COLOR = "#ff5a5f"
VENUS_COLOR = "#d8ffe4"
HEADER_BG = "#0b2330"
JPL_VALUE = "#7df0a6"
VIDEO_VALUE = "#ff8b8f"
DELTA_VALUE = "#ffd166"


def reflect_x(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    return np.column_stack((-array[:, 0], array[:, 1]))


def reflect_x_point(point: np.ndarray) -> np.ndarray:
    vector = np.asarray(point, dtype=float)
    return np.array([-vector[0], vector[1]], dtype=float)


def reduced_horizontal_angle_deg(points: np.ndarray) -> float:
    array = np.asarray(points, dtype=float)
    center = np.mean(array, axis=0)
    centered = array - center
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    direction = direction / np.linalg.norm(direction)
    signed_angle = math.degrees(math.atan2(direction[1], direction[0]))
    return abs((signed_angle + 90.0) % 180.0 - 90.0)


def perpendicular_rms(points: np.ndarray) -> float:
    array = np.asarray(points, dtype=float)
    center = np.mean(array, axis=0)
    centered = array - center
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    direction = direction / np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    residual = centered @ normal
    return float(np.sqrt(np.mean(residual**2)))


def cubic_regression_r_squared(points: np.ndarray) -> float:
    array = np.asarray(points, dtype=float)
    x = array[:, 0] - float(np.mean(array[:, 0]))
    y = array[:, 1]
    coefficients = np.polyfit(x, y, 3)
    fitted = np.polyval(coefficients, x)
    ss_residual = float(np.sum((y - fitted) ** 2))
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_residual / ss_total if ss_total > 0.0 else float("nan")


def load_clean_video_points() -> tuple[np.ndarray, str]:
    if not VIDEO_CSV.exists():
        raise FileNotFoundError(f"V0007 video CSV not found: {VIDEO_CSV}")

    table = pd.read_csv(VIDEO_CSV)
    required = {"x_norm", "y_norm"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise RuntimeError(f"V0007 video CSV missing required columns: {missing}")

    finite = (
        np.isfinite(table["x_norm"].to_numpy(dtype=float))
        & np.isfinite(table["y_norm"].to_numpy(dtype=float))
    )
    if "venus_track_inlier" in table.columns:
        selected = table["venus_track_inlier"].astype(bool).to_numpy() & finite
    elif "venus_detected" in table.columns:
        selected = table["venus_detected"].astype(bool).to_numpy() & finite
    else:
        selected = finite

    points = table.loc[selected, ["x_norm", "y_norm"]].to_numpy(dtype=float)
    if len(points) < 8:
        raise RuntimeError(f"Too few accepted video points: {len(points)}")

    angle_rad = math.radians(VIDEO_ANGLE_DEG)
    direction = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=float)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    cross_track = points @ normal
    cross_center = float(np.median(cross_track))
    mad = float(np.median(np.abs(cross_track - cross_center)))
    sigma = 1.4826 * mad
    tolerance = max(4.0 * sigma, 0.0030)
    keep = np.abs(cross_track - cross_center) <= tolerance
    cleaned = points[keep]

    if len(cleaned) < 8:
        return points, f"ORIGINAL INLIERS USED {len(points)}"
    return cleaned, f"ROBUST CLIP USED {len(cleaned)}/{len(points)}"


def make_video_reference_line(
    track_rsun: np.ndarray,
    closest_rsun: np.ndarray,
) -> np.ndarray:
    display_angle_rad = math.radians(-VIDEO_ANGLE_DEG)
    direction = np.array(
        [math.cos(display_angle_rad), math.sin(display_angle_rad)],
        dtype=float,
    )

    track_direction = track_rsun[-1] - track_rsun[0]
    track_direction = track_direction / np.linalg.norm(track_direction)
    along = (track_rsun - closest_rsun) @ track_direction

    return np.vstack(
        [
            closest_rsun + float(np.min(along)) * direction,
            closest_rsun + float(np.max(along)) * direction,
        ]
    )


def format_event_utc(event_utc: str) -> str:
    return str(event_utc).replace("T", " ")[:23]


def save_csv(
    geometry: dict[str, object],
    track_rsun: np.ndarray,
    video_line_rsun: np.ndarray,
    metrics: dict[str, float | str],
) -> None:
    rows: list[dict[str, object]] = []
    fit_jds = np.asarray(geometry["fit_jds"], dtype=float)

    for index, (jd_tdb, point) in enumerate(zip(fit_jds, track_rsun)):
        rows.append(
            {
                "version": VERSION,
                "site": SITE_NAME,
                "record_type": "CAPE_TOWN_SITE_COORD_TRACK_X_REFLECTED",
                "sequence": index,
                "event": "",
                "jd_tdb": jd_tdb,
                "utc": base.utc_at(float(jd_tdb)),
                "x_rsun": point[0],
                "y_rsun": point[1],
                "cape_angle_deg": metrics["cape_angle_deg"],
                "video_angle_deg": metrics["video_angle_deg"],
                "delta_beta_deg": metrics["delta_beta_deg"],
                "delta_m": metrics["delta_m"],
                "delta_beta_over_delta_m_deg": metrics[
                    "delta_beta_over_delta_m_deg"
                ],
                "cape_cubic_r_squared": metrics["cape_cubic_r_squared"],
                "video_cubic_r_squared": metrics["video_cubic_r_squared"],
                "cape_rms_rsun": metrics["cape_rms_rsun"],
                "video_rms_rsun": metrics["video_rms_rsun"],
                "status": "JPL SITE_COORD / X DISPLAY REFLECTION ONLY",
            }
        )

    solar_radius = geometry["events"]["CA"].sun_radius_arcsec
    for event_name, event in geometry["events"].items():
        point = reflect_x_point(event.point_arcsec / solar_radius)
        rows.append(
            {
                "version": VERSION,
                "site": SITE_NAME,
                "record_type": "CAPE_TOWN_EVENT_X_REFLECTED",
                "sequence": np.nan,
                "event": event_name,
                "jd_tdb": event.jd_tdb,
                "utc": event.utc,
                "x_rsun": point[0],
                "y_rsun": point[1],
                "cape_angle_deg": metrics["cape_angle_deg"],
                "video_angle_deg": metrics["video_angle_deg"],
                "delta_beta_deg": metrics["delta_beta_deg"],
                "delta_m": metrics["delta_m"],
                "delta_beta_over_delta_m_deg": metrics[
                    "delta_beta_over_delta_m_deg"
                ],
                "cape_cubic_r_squared": metrics["cape_cubic_r_squared"],
                "video_cubic_r_squared": metrics["video_cubic_r_squared"],
                "cape_rms_rsun": metrics["cape_rms_rsun"],
                "video_rms_rsun": metrics["video_rms_rsun"],
                "status": "JPL SITE_COORD EVENT",
            }
        )

    for index, point in enumerate(video_line_rsun):
        rows.append(
            {
                "version": VERSION,
                "site": SITE_NAME,
                "record_type": "SDO_VIDEO_REFERENCE_X_REFLECTED",
                "sequence": index,
                "event": "",
                "jd_tdb": np.nan,
                "utc": "",
                "x_rsun": point[0],
                "y_rsun": point[1],
                "cape_angle_deg": metrics["cape_angle_deg"],
                "video_angle_deg": metrics["video_angle_deg"],
                "delta_beta_deg": metrics["delta_beta_deg"],
                "delta_m": metrics["delta_m"],
                "delta_beta_over_delta_m_deg": metrics[
                    "delta_beta_over_delta_m_deg"
                ],
                "cape_cubic_r_squared": metrics["cape_cubic_r_squared"],
                "video_cubic_r_squared": metrics["video_cubic_r_squared"],
                "cape_rms_rsun": metrics["cape_rms_rsun"],
                "video_rms_rsun": metrics["video_rms_rsun"],
                "status": metrics["video_status"],
            }
        )

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, float_format="%.12f")


def render() -> None:
    geometry = base.build_geometry()
    events = geometry["events"]
    solar_radius = events["CA"].sun_radius_arcsec

    original_track = (
        np.asarray(geometry["fit_points_arcsec"], dtype=float) / solar_radius
    )
    track_rsun = reflect_x(original_track)

    cape_angle_deg = reduced_horizontal_angle_deg(track_rsun)
    cape_cubic_r_squared = cubic_regression_r_squared(track_rsun)
    cape_rms_rsun = perpendicular_rms(track_rsun)

    video_points, video_status = load_clean_video_points()
    video_cubic_r_squared = cubic_regression_r_squared(video_points)
    video_rms_rsun = perpendicular_rms(video_points)

    video_angle_deg = VIDEO_ANGLE_DEG
    delta_beta_deg = cape_angle_deg - video_angle_deg
    delta_m = (
        math.tan(math.radians(cape_angle_deg))
        - math.tan(math.radians(video_angle_deg))
    )
    if abs(delta_m) < 1.0e-15:
        raise RuntimeError("Delta-m is too small for the requested ratio.")
    delta_beta_over_delta_m_deg = delta_beta_deg / delta_m

    closest_rsun = reflect_x_point(events["CA"].point_arcsec / solar_radius)
    video_line_rsun = make_video_reference_line(track_rsun, closest_rsun)

    c1_rsun = reflect_x_point(events["C1"].point_arcsec / solar_radius)
    c4_rsun = reflect_x_point(events["C4"].point_arcsec / solar_radius)
    orientation_ok = (
        c1_rsun[0] < c4_rsun[0]
        and c1_rsun[1] > c4_rsun[1]
        and c1_rsun[1] > 0.0
        and c4_rsun[1] > 0.0
    )
    if not orientation_ok:
        raise RuntimeError(
            "Orientation verification failed: expected C1 left/high and C4 "
            "right/lower in upper half-Sun. "
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
    save_csv(geometry, track_rsun, video_line_rsun, metrics)

    figure = plt.figure(figsize=(11.2, 8.0), dpi=190, facecolor=BG)
    ax = figure.add_axes([0.045, 0.065, 0.82, 0.86])
    ax.set_facecolor(BG)

    theta = np.linspace(0.0, math.pi, 1400)
    ax.plot(np.cos(theta), np.sin(theta), color=SOLAR, linewidth=0.38, zorder=1)
    ax.plot(
        [-1.0, 1.0],
        [0.0, 0.0],
        color=SOLAR,
        linewidth=0.24,
        alpha=0.72,
        zorder=1,
    )
    ax.axvline(0.0, color=SPINE, linewidth=0.18, alpha=0.50, zorder=0)

    ax.plot(
        track_rsun[:, 0],
        track_rsun[:, 1],
        color=JPL_COLOR,
        linewidth=0.74,
        label="Cape Town JPL SITE_COORD",
        zorder=4,
    )
    ax.scatter(
        track_rsun[::8, 0],
        track_rsun[::8, 1],
        s=1.6,
        color=JPL_COLOR,
        linewidths=0,
        alpha=0.78,
        zorder=5,
    )
    ax.plot(
        video_line_rsun[:, 0],
        video_line_rsun[:, 1],
        color=VIDEO_COLOR,
        linewidth=0.94,
        linestyle=(0, (7, 5)),
        label="SDO video TLS reference",
        zorder=6,
    )

    label_offsets = {
        "C1": (-0.074, 0.024),
        "C2": (-0.063, 0.018),
        "CA": (-0.026, 0.032),
        "C3": (0.021, -0.032),
        "C4": (0.029, -0.041),
    }
    for event_name in ("C1", "C2", "CA", "C3", "C4"):
        event = events[event_name]
        point = reflect_x_point(event.point_arcsec / solar_radius)
        radius = event.venus_radius_arcsec / solar_radius
        ax.add_patch(
            Circle(
                (point[0], point[1]),
                radius,
                fill=False,
                edgecolor=VENUS_COLOR,
                linewidth=0.28,
                zorder=7,
            )
        )
        ax.scatter(
            [point[0]],
            [point[1]],
            s=4.2,
            color=JPL_COLOR,
            edgecolor=BG,
            linewidth=0.18,
            zorder=8,
        )
        dx, dy = label_offsets[event_name]
        ax.text(
            point[0] + dx,
            point[1] + dy,
            event_name,
            color=TEXT,
            fontsize=6.8,
            zorder=9,
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
        bbox=[0.30, 0.045, 0.52, 0.235],
        cellLoc="center",
        colLoc="center",
        zorder=20,
    )
    statistics_table.auto_set_font_size(False)
    statistics_table.set_fontsize(6.8)
    for (row, column), cell in statistics_table.get_celld().items():
        cell.set_edgecolor(SPINE)
        cell.set_linewidth(0.25)
        if row == 0:
            cell.set_facecolor(HEADER_BG)
            cell.get_text().set_color(TEXT)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(PANEL)
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
        ["C1", format_event_utc(events["C1"].utc)],
        ["C2", format_event_utc(events["C2"].utc)],
        ["C3", format_event_utc(events["C3"].utc)],
        ["C4", format_event_utc(events["C4"].utc)],
    ]
    event_table = ax.table(
        cellText=event_rows,
        colLabels=["Observation", "UTC / Location"],
        colWidths=[0.26, 0.70],
        bbox=[0.035, 0.045, 0.245, 0.235],
        cellLoc="left",
        colLoc="left",
        zorder=20,
    )
    event_table.auto_set_font_size(False)
    event_table.set_fontsize(6.45)
    for (row, column), cell in event_table.get_celld().items():
        cell.set_edgecolor(SPINE)
        cell.set_linewidth(0.25)
        if row == 0:
            cell.set_facecolor(HEADER_BG)
            cell.get_text().set_color(TEXT)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(PANEL)
            cell.get_text().set_color(TEXT if column == 0 else MUTED)
            if column == 0:
                cell.get_text().set_weight("bold")

    ax.text(
        0.56,
        0.291,
        "TRACK COMPARISON",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=TEXT,
        fontsize=8.1,
        fontweight="bold",
        zorder=21,
    )

    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=GRID, linewidth=0.16, alpha=0.60)
    ax.tick_params(colors=MUTED, labelsize=7, width=0.22, length=2.0)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.24)

    ax.set_xlabel("Solar-screen X (R_sun)", color=MUTED, fontsize=8)
    ax.set_ylabel("Solar-screen Y (R_sun)", color=MUTED, fontsize=8)
    ax.set_title(
        "2012 Venus Transit — Cape Town SITE_COORD vs SDO Video\n"
        "Upper Half-Sun — Verified X-axis Display Reflection",
        color=TEXT,
        fontsize=10.2,
        pad=9,
    )

    legend = ax.legend(
        loc="upper right",
        fontsize=6.7,
        frameon=True,
        handlelength=3.0,
    )
    legend.get_frame().set_facecolor(PANEL)
    legend.get_frame().set_edgecolor(SPINE)
    legend.get_frame().set_linewidth(0.28)
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
    print("Cape Town JPL Horizons SITE_COORD and V0007 SDO video fit")
    print("COMMENTS")
    print("No AI images; verified x-axis reflection; C1 left/high and C4 right/lower")
    print("RESULTS")
    print(f"PNG: {PLOT_PNG}")
    print("OUTPUT SUMMARY")
    print(f"CSV: {OUTPUT_CSV}")
    print("PAPER COMPARISON")
    print("NOT USED")
    print("EQUATION STATUS")
    print("Track angles, delta-beta, delta-m, ratio, cubic R-squared, RMS, site, and contacts rendered")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# IERS-0012AU
