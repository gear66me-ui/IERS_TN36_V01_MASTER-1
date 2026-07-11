# IERS-0012AN
# Audit reference: Upper half-Sun Cape Town SITE_COORD versus SDO video with only R-squared and RMS statistics.
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

VERSION = "IERS-0012AN"
LOCAL_TZ = ZoneInfo("America/Bogota")
VIDEO_TLS_ANGLE_DEG = 8.4340601435

DRIVE_ROOT = Path("/content/drive/MyDrive/Colab Notebooks")
PROJECT_ROOT = DRIVE_ROOT / "IERS_TN36_OUTPUT" if DRIVE_ROOT.exists() else Path("/content/IERS_TN36_OUTPUT")
OUTPUT_PNG_DIR = PROJECT_ROOT / "OUTPUT_PNG"
OUTPUT_CSV_DIR = PROJECT_ROOT / "OUTPUT_CSV"
OUTPUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)

PLOT_PNG = OUTPUT_PNG_DIR / f"{VERSION}_CAPE_TOWN_UPPER_HALF_R2_RMS.png"
OUTPUT_CSV = OUTPUT_CSV_DIR / f"{VERSION}_CAPE_TOWN_UPPER_HALF_R2_RMS.csv"

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


def transform_upper_half(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    return np.column_stack((-array[:, 0], array[:, 1]))


def transform_upper_point(point: np.ndarray) -> np.ndarray:
    vector = np.asarray(point, dtype=float)
    return np.array([-vector[0], vector[1]], dtype=float)


def fit_metrics(points: np.ndarray) -> dict[str, float]:
    array = np.asarray(points, dtype=float)
    finite = np.isfinite(array).all(axis=1)
    array = array[finite]
    if len(array) < 3:
        raise RuntimeError("At least three finite points are required.")

    center = np.mean(array, axis=0)
    centered = array - center
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    direction = direction / np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    perpendicular = centered @ normal
    rms = float(np.sqrt(np.mean(perpendicular**2)))

    x = array[:, 0] - float(np.mean(array[:, 0]))
    y = array[:, 1]
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    return {
        "r_squared": r_squared,
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
    center = float(np.median(cross))
    mad = float(np.median(np.abs(cross - center)))
    sigma = 1.4826 * mad
    tolerance = max(4.0 * sigma, 0.0030)
    keep = np.abs(cross - center) <= tolerance
    cleaned = points[keep]

    if len(cleaned) < 8:
        return points, f"ORIGINAL INLIERS USED {len(points)}"
    return cleaned, f"ROBUST CLIP USED {len(cleaned)}/{len(points)}"


def video_reference_line(track_rsun: np.ndarray, ca_rsun: np.ndarray) -> np.ndarray:
    angle_rad = math.radians(-VIDEO_TLS_ANGLE_DEG)
    direction = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=float)

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
    video_status: str,
) -> None:
    rows: list[dict[str, object]] = []
    fit_jds = np.asarray(geometry["fit_jds"], dtype=float)

    for index, (jd, point) in enumerate(zip(fit_jds, track_rsun)):
        rows.append(
            {
                "version": VERSION,
                "record_type": "CAPE_TOWN_SITE_COORD_TRACK_UPPER_HALF",
                "sequence": index,
                "event": "",
                "jd_tdb": jd,
                "utc": base.utc_at(float(jd)),
                "x_rsun": point[0],
                "y_rsun": point[1],
                "r_squared_linear": jpl_metrics["r_squared"],
                "rms_perpendicular_rsun": jpl_metrics["rms"],
                "status": "JPL SITE_COORD / X DISPLAY REFLECTION ONLY",
            }
        )

    solar_radius = geometry["events"]["CA"].sun_radius_arcsec
    for name, event in geometry["events"].items():
        point = transform_upper_point(event.point_arcsec / solar_radius)
        rows.append(
            {
                "version": VERSION,
                "record_type": "CAPE_TOWN_EVENT_UPPER_HALF",
                "sequence": np.nan,
                "event": name,
                "jd_tdb": event.jd_tdb,
                "utc": event.utc,
                "x_rsun": point[0],
                "y_rsun": point[1],
                "r_squared_linear": jpl_metrics["r_squared"],
                "rms_perpendicular_rsun": jpl_metrics["rms"],
                "status": "JPL SITE_COORD EVENT",
            }
        )

    for index, point in enumerate(video_line_rsun):
        rows.append(
            {
                "version": VERSION,
                "record_type": "SDO_VIDEO_TLS_REFERENCE_UPPER_HALF",
                "sequence": index,
                "event": "",
                "jd_tdb": np.nan,
                "utc": "",
                "x_rsun": point[0],
                "y_rsun": point[1],
                "r_squared_linear": video_metrics["r_squared"],
                "rms_perpendicular_rsun": video_metrics["rms"],
                "status": video_status,
            }
        )

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, float_format="%.12f")


def render() -> None:
    geometry = base.build_geometry()
    events = geometry["events"]
    solar_radius = events["CA"].sun_radius_arcsec

    original_track = np.asarray(geometry["fit_points_arcsec"], dtype=float) / solar_radius
    track_rsun = transform_upper_half(original_track)
    jpl_metrics = fit_metrics(track_rsun)

    video_points, video_status = load_clean_video_points()
    video_metrics = fit_metrics(video_points)

    ca_original = events["CA"].point_arcsec / solar_radius
    ca_rsun = transform_upper_point(ca_original)
    video_line_rsun = video_reference_line(track_rsun, ca_rsun)

    write_csv(
        geometry,
        track_rsun,
        video_line_rsun,
        jpl_metrics,
        video_metrics,
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
        point = transform_upper_point(event.point_arcsec / solar_radius)
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
        ["R² linear", f"{jpl_metrics['r_squared']:.9f}", f"{video_metrics['r_squared']:.9f}"],
        ["RMS perpendicular", f"{jpl_metrics['rms']:.9f}", f"{video_metrics['rms']:.9f}"],
    ]
    table = ax.table(
        cellText=table_rows,
        colLabels=["Statistic", "Cape Town JPL", "SDO video"],
        colWidths=[0.34, 0.27, 0.27],
        bbox=[0.34, 0.055, 0.44, 0.115],
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
            if column == 1:
                cell.get_text().set_color(JPL_VALUE)
                cell.get_text().set_weight("bold")
            elif column == 2:
                cell.get_text().set_color(VIDEO_VALUE)
                cell.get_text().set_weight("bold")
            else:
                cell.get_text().set_color(MUTED)

    ax.text(
        0.56,
        0.180,
        "R² AND RMS",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=TEXT,
        fontsize=8.2,
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
        "True Upper Half-Sun Engineering Reconstruction",
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
    print("No AI images; true upper half-Sun; only R-squared and RMS shown")
    print("RESULTS")
    print(f"PNG: {PLOT_PNG}")
    print("OUTPUT SUMMARY")
    print(f"CSV: {OUTPUT_CSV}")
    print("PAPER COMPARISON")
    print("NOT USED")
    print("EQUATION STATUS")
    print("R-squared and RMS are rendered inside the PNG")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# IERS-0012AN
