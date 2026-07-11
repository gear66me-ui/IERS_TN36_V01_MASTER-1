# IERS-0012AP
# Audit reference: Cape Town SITE_COORD upper-half display using y-only reflection; cubic R-squared, RMS, delta-beta, and delta-slope only.
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

VERSION = "IERS-0012AP"
LOCAL_TZ = ZoneInfo("America/Bogota")
VIDEO_TLS_ANGLE_DEG = 8.4340601435

DRIVE_ROOT = Path("/content/drive/MyDrive/Colab Notebooks")
PROJECT_ROOT = DRIVE_ROOT / "IERS_TN36_OUTPUT" if DRIVE_ROOT.exists() else Path("/content/IERS_TN36_OUTPUT")
OUTPUT_PNG_DIR = PROJECT_ROOT / "OUTPUT_PNG"
OUTPUT_CSV_DIR = PROJECT_ROOT / "OUTPUT_CSV"
OUTPUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)

PLOT_PNG = OUTPUT_PNG_DIR / f"{VERSION}_CAPE_TOWN_UPPER_HALF_CUBIC_R2_RMS.png"
OUTPUT_CSV = OUTPUT_CSV_DIR / f"{VERSION}_CAPE_TOWN_UPPER_HALF_CUBIC_R2_RMS.csv"

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


def flip_vertical(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    return np.column_stack((array[:, 0], -array[:, 1]))


def flip_vertical_point(point: np.ndarray) -> np.ndarray:
    vector = np.asarray(point, dtype=float)
    return np.array([vector[0], -vector[1]], dtype=float)


def fit_metrics(points: np.ndarray) -> dict[str, float]:
    array = np.asarray(points, dtype=float)
    finite = np.isfinite(array).all(axis=1)
    array = array[finite]
    if len(array) < 4:
        raise RuntimeError("At least four finite points are required.")

    center = np.mean(array, axis=0)
    centered = array - center
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    direction = direction / np.linalg.norm(direction)
    if direction[0] < 0.0:
        direction = -direction
    normal = np.array([-direction[1], direction[0]], dtype=float)
    perpendicular = centered @ normal
    rms = float(np.sqrt(np.mean(perpendicular**2)))

    angle_deg = math.degrees(math.atan2(direction[1], direction[0]))
    angle_from_horizontal_deg = abs((angle_deg + 90.0) % 180.0 - 90.0)
    slope_signed = math.tan(math.radians(angle_deg))
    slope_magnitude = abs(slope_signed)

    x = array[:, 0] - float(np.mean(array[:, 0]))
    y = array[:, 1]
    coefficients = np.polyfit(x, y, 3)
    fitted = np.polyval(coefficients, x)
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    cubic_r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    return {
        "angle_from_horizontal_deg": angle_from_horizontal_deg,
        "signed_angle_deg": angle_deg,
        "slope_magnitude": slope_magnitude,
        "signed_slope": slope_signed,
        "cubic_r_squared": cubic_r_squared,
        "rms": rms,
        "count": float(len(array)),
    }


def load_clean_video_points() -> tuple[np.ndarray, str]:
    if not VIDEO_CSV.exists():
        raise FileNotFoundError(f"V0007 video CSV not found: {VIDEO_CSV}")

    table = pd.read_csv(VIDEO_CSV)
    required = {"x_norm", "y_norm"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise RuntimeError(f"Video CSV missing required columns: {missing}")

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
        raise RuntimeError(f"Too few accepted video detections: {len(points)}")

    angle_rad = math.radians(VIDEO_TLS_ANGLE_DEG)
    direction = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=float)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    cross = points @ normal
    median_cross = float(np.median(cross))
    mad = float(np.median(np.abs(cross - median_cross)))
    sigma = 1.4826 * mad
    tolerance = max(4.0 * sigma, 0.0030)
    keep = np.abs(cross - median_cross) <= tolerance
    cleaned = points[keep]

    if len(cleaned) < 8:
        return points, f"ORIGINAL INLIERS USED {len(points)}"
    return cleaned, f"ROBUST CLIP USED {len(cleaned)}/{len(points)}"


def video_reference_line(track_rsun: np.ndarray, ca_rsun: np.ndarray) -> np.ndarray:
    displayed_angle_rad = math.radians(-VIDEO_TLS_ANGLE_DEG)
    direction = np.array(
        [math.cos(displayed_angle_rad), math.sin(displayed_angle_rad)],
        dtype=float,
    )

    track_direction = track_rsun[-1] - track_rsun[0]
    track_direction = track_direction / np.linalg.norm(track_direction)
    along = (track_rsun - ca_rsun) @ track_direction

    return np.vstack(
        [
            ca_rsun + float(np.min(along)) * direction,
            ca_rsun + float(np.max(along)) * direction,
        ]
    )


def write_csv(
    geometry: dict[str, object],
    track_rsun: np.ndarray,
    video_line_rsun: np.ndarray,
    jpl_metrics: dict[str, float],
    video_metrics: dict[str, float],
    delta_beta_deg: float,
    delta_m: float,
    video_status: str,
) -> None:
    rows: list[dict[str, object]] = []
    fit_jds = np.asarray(geometry["fit_jds"], dtype=float)

    for index, (jd, point) in enumerate(zip(fit_jds, track_rsun)):
        rows.append(
            {
                "version": VERSION,
                "record_type": "CAPE_TOWN_SITE_COORD_TRACK_Y_FLIPPED",
                "sequence": index,
                "event": "",
                "jd_tdb": jd,
                "utc": base.utc_at(float(jd)),
                "x_rsun": point[0],
                "y_rsun": point[1],
                "cubic_r_squared": jpl_metrics["cubic_r_squared"],
                "rms_perpendicular_rsun": jpl_metrics["rms"],
                "delta_beta_deg": delta_beta_deg,
                "delta_m": delta_m,
                "status": "JPL SITE_COORD / Y DISPLAY FLIP ONLY",
            }
        )

    solar_radius = geometry["events"]["CA"].sun_radius_arcsec
    for name, event in geometry["events"].items():
        point = flip_vertical_point(event.point_arcsec / solar_radius)
        rows.append(
            {
                "version": VERSION,
                "record_type": "CAPE_TOWN_EVENT_Y_FLIPPED",
                "sequence": np.nan,
                "event": name,
                "jd_tdb": event.jd_tdb,
                "utc": event.utc,
                "x_rsun": point[0],
                "y_rsun": point[1],
                "cubic_r_squared": jpl_metrics["cubic_r_squared"],
                "rms_perpendicular_rsun": jpl_metrics["rms"],
                "delta_beta_deg": delta_beta_deg,
                "delta_m": delta_m,
                "status": "JPL SITE_COORD EVENT",
            }
        )

    for index, point in enumerate(video_line_rsun):
        rows.append(
            {
                "version": VERSION,
                "record_type": "SDO_VIDEO_TLS_REFERENCE_Y_FLIPPED",
                "sequence": index,
                "event": "",
                "jd_tdb": np.nan,
                "utc": "",
                "x_rsun": point[0],
                "y_rsun": point[1],
                "cubic_r_squared": video_metrics["cubic_r_squared"],
                "rms_perpendicular_rsun": video_metrics["rms"],
                "delta_beta_deg": delta_beta_deg,
                "delta_m": delta_m,
                "status": video_status,
            }
        )

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, float_format="%.12f")


def render() -> None:
    geometry = base.build_geometry()
    events = geometry["events"]
    solar_radius = events["CA"].sun_radius_arcsec

    original_track = np.asarray(geometry["fit_points_arcsec"], dtype=float) / solar_radius
    track_rsun = flip_vertical(original_track)
    jpl_metrics = fit_metrics(track_rsun)

    raw_video_points, video_status = load_clean_video_points()
    video_points_display = flip_vertical(raw_video_points)
    video_metrics = fit_metrics(video_points_display)

    ca_original = events["CA"].point_arcsec / solar_radius
    ca_rsun = flip_vertical_point(ca_original)
    video_line_rsun = video_reference_line(track_rsun, ca_rsun)

    delta_beta_deg = (
        jpl_metrics["angle_from_horizontal_deg"]
        - VIDEO_TLS_ANGLE_DEG
    )
    delta_m = (
        jpl_metrics["slope_magnitude"]
        - abs(math.tan(math.radians(VIDEO_TLS_ANGLE_DEG)))
    )

    c1 = flip_vertical_point(events["C1"].point_arcsec / solar_radius)
    c4 = flip_vertical_point(events["C4"].point_arcsec / solar_radius)
    if not (c1[0] < c4[0] and c1[1] > c4[1] and c1[1] > 0.0 and c4[1] > 0.0):
        raise RuntimeError(
            "Display orientation verification failed: expected C1 left/high and C4 right/lower in upper half-Sun. "
            f"C1=({c1[0]:.6f},{c1[1]:.6f}), C4=({c4[0]:.6f},{c4[1]:.6f})"
        )

    write_csv(
        geometry,
        track_rsun,
        video_line_rsun,
        jpl_metrics,
        video_metrics,
        delta_beta_deg,
        delta_m,
        video_status,
    )

    figure = plt.figure(figsize=(10.8, 7.6), dpi=190, facecolor=BG)
    ax = figure.add_axes([0.055, 0.075, 0.80, 0.84])
    ax.set_facecolor(BG)

    theta = np.linspace(0.0, math.pi, 1200)
    ax.plot(np.cos(theta), np.sin(theta), color=SOLAR, linewidth=0.38, zorder=1)
    ax.plot([-1.0, 1.0], [0.0, 0.0], color=SOLAR, linewidth=0.24, alpha=0.72, zorder=1)
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
    for name in ("C1", "C2", "CA", "C3", "C4"):
        event = events[name]
        point = flip_vertical_point(event.point_arcsec / solar_radius)
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
        dx, dy = label_offsets[name]
        ax.text(point[0] + dx, point[1] + dy, name, color=TEXT, fontsize=6.8, zorder=9)

    table_rows = [
        ["R² cubic", f"{jpl_metrics['cubic_r_squared']:.9f}", f"{video_metrics['cubic_r_squared']:.9f}"],
        ["RMS perpendicular", f"{jpl_metrics['rms']:.9f}", f"{video_metrics['rms']:.9f}"],
        ["Δβ", f"{delta_beta_deg:+.6f}", "deg"],
        ["Δm", f"{delta_m:+.9f}", ""],
    ]
    table = ax.table(
        cellText=table_rows,
        colLabels=["Statistic", "Value A", "Value B / Unit"],
        colWidths=[0.36, 0.27, 0.27],
        bbox=[0.34, 0.045, 0.44, 0.165],
        cellLoc="center",
        colLoc="center",
        zorder=20,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(SPINE)
        cell.set_linewidth(0.25)
        if row == 0:
            cell.set_facecolor(HEADER_BG)
            cell.get_text().set_color(TEXT)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(PANEL)
            if row in (1, 2) and column == 1:
                cell.get_text().set_color(JPL_VALUE)
                cell.get_text().set_weight("bold")
            elif row in (1, 2) and column == 2:
                cell.get_text().set_color(VIDEO_VALUE)
                cell.get_text().set_weight("bold")
            elif row in (3, 4) and column == 1:
                cell.get_text().set_color(DELTA_VALUE)
                cell.get_text().set_weight("bold")
            else:
                cell.get_text().set_color(MUTED)

    ax.text(
        0.56,
        0.220,
        "CUBIC R², RMS, Δβ, AND Δm",
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
        "Upper Half-Sun — Y-axis Display Flip Only",
        color=TEXT,
        fontsize=10.2,
        pad=9,
    )

    legend = ax.legend(loc="upper right", fontsize=6.7, frameon=True, handlelength=3.0)
    legend.get_frame().set_facecolor(PANEL)
    legend.get_frame().set_edgecolor(SPINE)
    legend.get_frame().set_linewidth(0.28)
    for item in legend.get_texts():
        item.set_color(TEXT)

    figure.savefig(PLOT_PNG, dpi=320, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    display(IPythonImage(filename=str(PLOT_PNG)))


def main() -> int:
    render()
    print("CODE INPUTS")
    print("Cape Town JPL Horizons SITE_COORD and V0007 SDO video fit")
    print("COMMENTS")
    print("No AI images; y-axis display flip only; C1 left/high and C4 right/lower")
    print("RESULTS")
    print(f"PNG: {PLOT_PNG}")
    print("OUTPUT SUMMARY")
    print(f"CSV: {OUTPUT_CSV}")
    print("PAPER COMPARISON")
    print("NOT USED")
    print("EQUATION STATUS")
    print("Only cubic R-squared, perpendicular RMS, delta-beta, and delta-m are rendered")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# IERS-0012AP
