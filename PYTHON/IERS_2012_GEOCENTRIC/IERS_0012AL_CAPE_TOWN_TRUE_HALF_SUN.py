# IERS-0012AL
# Audit reference: True lower half-Sun Cape Town SITE_COORD versus SDO video, compact IERS-0012AA in-plot table.
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from IPython.display import Image as IPythonImage, display

import IERS_0012AK_CAPE_TOWN_FLIPPED_AA_FORMAT as previous

VERSION = "IERS-0012AL"
LOCAL_TZ = ZoneInfo("America/Bogota")

DRIVE_ROOT = Path("/content/drive/MyDrive/Colab Notebooks")
PROJECT_ROOT = DRIVE_ROOT / "IERS_TN36_OUTPUT" if DRIVE_ROOT.exists() else Path("/content/IERS_TN36_OUTPUT")
OUTPUT_PNG_DIR = PROJECT_ROOT / "OUTPUT_PNG"
OUTPUT_CSV_DIR = PROJECT_ROOT / "OUTPUT_CSV"
OUTPUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)

PLOT_PNG = OUTPUT_PNG_DIR / f"{VERSION}_CAPE_TOWN_TRUE_HALF_SUN.png"
OUTPUT_CSV = OUTPUT_CSV_DIR / f"{VERSION}_CAPE_TOWN_TRUE_HALF_SUN.csv"

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
CAPE_VALUE = "#7df0a6"
VIDEO_VALUE = "#ff8b8f"
DELTA_VALUE = "#ffd166"


def render() -> None:
    geometry = previous.base.build_geometry()
    events = geometry["events"]
    solar_radius_arcsec = events["CA"].sun_radius_arcsec

    original_track_rsun = np.asarray(geometry["fit_points_arcsec"], dtype=float) / solar_radius_arcsec
    track_rsun = -original_track_rsun
    cape_metrics = previous.pca_metrics(track_rsun)

    original_ca_rsun = events["CA"].point_arcsec / solar_radius_arcsec
    original_video_line = previous.make_video_reference_line(original_track_rsun, original_ca_rsun)
    video_line_rsun = -original_video_line

    _video_points, video_metrics, video_status = previous.load_clean_video_points()
    delta_beta = cape_metrics["angle_deg"] - previous.VIDEO_TLS_ANGLE_DEG
    delta_slope = cape_metrics["slope"] - video_metrics["slope"]

    previous.OUTPUT_CSV = OUTPUT_CSV
    previous.save_csv(
        geometry,
        track_rsun,
        video_line_rsun,
        cape_metrics,
        video_metrics,
        video_status,
    )

    figure = plt.figure(figsize=(10.8, 7.6), dpi=190, facecolor=BG)
    ax = figure.add_axes([0.055, 0.075, 0.78, 0.84])
    ax.set_facecolor(BG)

    theta = np.linspace(math.pi, 2.0 * math.pi, 1000)
    ax.plot(np.cos(theta), np.sin(theta), color=SOLAR, linewidth=0.38, zorder=1)
    ax.plot([-1.0, 1.0], [0.0, 0.0], color=SOLAR, linewidth=0.24, alpha=0.72, zorder=1)
    ax.axvline(0.0, color=SPINE, linewidth=0.18, alpha=0.50, zorder=0)

    ax.plot(
        track_rsun[:, 0],
        track_rsun[:, 1],
        color=JPL_COLOR,
        linewidth=0.72,
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
        linewidth=0.92,
        linestyle=(0, (7, 5)),
        label="SDO video TLS reference",
        zorder=6,
    )

    label_offsets = {
        "C1": (-0.072, 0.022),
        "C2": (-0.062, 0.018),
        "CA": (-0.025, 0.030),
        "C3": (0.020, -0.033),
        "C4": (0.028, -0.042),
    }
    for name in ("C1", "C2", "CA", "C3", "C4"):
        event = events[name]
        point = -(event.point_arcsec / solar_radius_arcsec)
        radius = event.venus_radius_arcsec / solar_radius_arcsec
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
        ["β horizontal", f"{cape_metrics['angle_deg']:.6f}", f"{previous.VIDEO_TLS_ANGLE_DEG:.6f}", "deg"],
        ["m = tan(β)", f"{cape_metrics['slope']:.9f}", f"{video_metrics['slope']:.9f}", ""],
        ["R² linear", f"{cape_metrics['r2_linear']:.9f}", f"{video_metrics['r2_linear']:.9f}", ""],
        ["R² quadratic", f"{cape_metrics['r2_quadratic']:.9f}", f"{video_metrics['r2_quadratic']:.9f}", ""],
        ["R² cubic", f"{cape_metrics['r2_cubic']:.9f}", f"{video_metrics['r2_cubic']:.9f}", ""],
        ["RMS ⟂", f"{cape_metrics['rms']:.9f}", f"{video_metrics['rms']:.9f}", "R_sun"],
        ["Δβ / Δm", f"{delta_beta:+.6f}°", f"{delta_slope:+.9f}", ""],
    ]
    table = ax.table(
        cellText=table_rows,
        colLabels=["Quantity", "Cape Town JPL", "SDO video", "Unit"],
        colWidths=[0.28, 0.25, 0.25, 0.12],
        bbox=[0.32, 0.045, 0.50, 0.255],
        cellLoc="center",
        colLoc="center",
        zorder=20,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.6)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(SPINE)
        cell.set_linewidth(0.24)
        if row == 0:
            cell.set_facecolor(HEADER_BG)
            cell.get_text().set_color(TEXT)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(PANEL)
            if row == len(table_rows) and column in (1, 2):
                cell.get_text().set_color(DELTA_VALUE)
                cell.get_text().set_weight("bold")
            elif column == 1:
                cell.get_text().set_color(CAPE_VALUE)
                cell.get_text().set_weight("bold")
            elif column == 2:
                cell.get_text().set_color(VIDEO_VALUE)
                cell.get_text().set_weight("bold")
            else:
                cell.get_text().set_color(MUTED)

    ax.text(
        0.57,
        0.315,
        "TRACK GEOMETRY AND TRIGONOMETRY",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=TEXT,
        fontsize=8.0,
        fontweight="bold",
        zorder=21,
    )
    ax.text(
        0.57,
        0.022,
        f"180° display rotation only. Video statistics: {video_status}.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=MUTED,
        fontsize=5.6,
        zorder=21,
    )

    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 0.02)
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
        "True Lower Half-Sun Engineering Reconstruction — IERS-0012AA format",
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
    print("No AI images; true lower half-Sun only; plotting area shifted left")
    print("RESULTS")
    print(f"PNG: {PLOT_PNG}")
    print("OUTPUT SUMMARY")
    print(f"CSV: {OUTPUT_CSV}")
    print("PAPER COMPARISON")
    print("NOT USED")
    print("EQUATION STATUS")
    print("Angles, slopes, R2 values, and RMS are rendered inside the PNG")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# IERS-0012AL
