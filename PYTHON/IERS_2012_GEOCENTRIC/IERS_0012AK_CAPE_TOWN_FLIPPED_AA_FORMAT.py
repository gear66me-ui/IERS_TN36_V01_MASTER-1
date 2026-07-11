# IERS-0012AK
# Audit reference: Cape Town SITE_COORD half-Sun plot, 180-degree display rotation, IERS-0012AA table format.
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
):
    ensure_package(_import_name, _pip_name)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from IPython.display import Image as IPythonImage, display

import IERS_0012AJ_CAPE_TOWN_SITE_COORD_VS_VIDEO as base


VERSION = "IERS-0012AK"
LOCAL_TZ = ZoneInfo("America/Bogota")
VIDEO_TLS_ANGLE_DEG = 8.4340601435
DISPLAY_ROTATION_DEG = 180.0

DRIVE_ROOT = Path("/content/drive/MyDrive/Colab Notebooks")
PROJECT_ROOT = DRIVE_ROOT / "IERS_TN36_OUTPUT" if DRIVE_ROOT.exists() else Path("/content/IERS_TN36_OUTPUT")
OUTPUT_PNG_DIR = PROJECT_ROOT / "OUTPUT_PNG"
OUTPUT_CSV_DIR = PROJECT_ROOT / "OUTPUT_CSV"
OUTPUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)

PLOT_PNG = OUTPUT_PNG_DIR / f"{VERSION}_CAPE_TOWN_FLIPPED_AA_FORMAT.png"
OUTPUT_CSV = OUTPUT_CSV_DIR / f"{VERSION}_CAPE_TOWN_FLIPPED_AA_FORMAT.csv"

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
CAPE_VALUE = "#7df0a6"
VIDEO_VALUE = "#ff8b8f"
DELTA_VALUE = "#ffd166"


def wrap_horizontal(angle_deg: float) -> float:
    return (float(angle_deg) + 90.0) % 180.0 - 90.0


def fit_r2(points: np.ndarray, degree: int) -> float:
    array = np.asarray(points, dtype=float)
    x = array[:, 0] - float(np.mean(array[:, 0]))
    y = array[:, 1]
    coeff = np.polyfit(x, y, degree)
    fitted = np.polyval(coeff, x)
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")


def pca_metrics(points: np.ndarray) -> dict[str, float]:
    array = np.asarray(points, dtype=float)
    center = np.mean(array, axis=0)
    centered = array - center
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    direction = direction / np.linalg.norm(direction)
    if direction[0] < 0.0:
        direction = -direction
    normal = np.array([-direction[1], direction[0]], dtype=float)
    cross = centered @ normal
    angle = wrap_horizontal(math.degrees(math.atan2(direction[1], direction[0])))
    return {
        "angle_deg": angle,
        "slope": math.tan(math.radians(angle)),
        "rms": float(np.sqrt(np.mean(cross**2))),
        "r2_linear": fit_r2(array, 1),
        "r2_quadratic": fit_r2(array, 2),
        "r2_cubic": fit_r2(array, 3),
        "count": float(len(array)),
    }


def load_clean_video_points() -> tuple[np.ndarray, dict[str, float], str]:
    if not VIDEO_CSV.exists():
        return np.empty((0, 2), dtype=float), {
            "angle_deg": VIDEO_TLS_ANGLE_DEG,
            "slope": math.tan(math.radians(VIDEO_TLS_ANGLE_DEG)),
            "rms": float("nan"),
            "r2_linear": float("nan"),
            "r2_quadratic": float("nan"),
            "r2_cubic": float("nan"),
            "count": 0.0,
        }, "VIDEO CSV NOT FOUND"

    table = pd.read_csv(VIDEO_CSV)
    if not {"x_norm", "y_norm"}.issubset(table.columns):
        raise RuntimeError("V0007 video CSV is missing x_norm/y_norm columns.")

    finite = np.isfinite(table["x_norm"].to_numpy(float)) & np.isfinite(table["y_norm"].to_numpy(float))
    if "venus_track_inlier" in table.columns:
        selected = table["venus_track_inlier"].astype(bool).to_numpy() & finite
    elif "venus_detected" in table.columns:
        selected = table["venus_detected"].astype(bool).to_numpy() & finite
    else:
        selected = finite

    points = table.loc[selected, ["x_norm", "y_norm"]].to_numpy(dtype=float)
    if len(points) < 8:
        raise RuntimeError(f"Too few accepted V0007 video points: {len(points)}")

    angle_rad = math.radians(VIDEO_TLS_ANGLE_DEG)
    direction = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=float)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    cross = points @ normal
    cross_center = float(np.median(cross))
    mad = float(np.median(np.abs(cross - cross_center)))
    sigma = 1.4826 * mad
    tolerance = max(4.0 * sigma, 0.0030)
    keep = np.abs(cross - cross_center) <= tolerance
    cleaned = points[keep]
    if len(cleaned) < 8:
        cleaned = points
        status = "ROBUST CLIP REJECTED / ORIGINAL INLIERS USED"
    else:
        status = f"ROBUST CLIP USED {len(cleaned)}/{len(points)}"

    metrics = pca_metrics(cleaned)
    metrics["angle_deg"] = VIDEO_TLS_ANGLE_DEG
    metrics["slope"] = math.tan(math.radians(VIDEO_TLS_ANGLE_DEG))
    return cleaned, metrics, status


def make_video_reference_line(track_rsun: np.ndarray, ca_rsun: np.ndarray) -> np.ndarray:
    angle_rad = math.radians(VIDEO_TLS_ANGLE_DEG)
    direction = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=float)
    track_direction = track_rsun[-1] - track_rsun[0]
    track_direction = track_direction / np.linalg.norm(track_direction)
    along = (track_rsun - ca_rsun) @ track_direction
    return np.vstack([
        ca_rsun + float(np.min(along)) * direction,
        ca_rsun + float(np.max(along)) * direction,
    ])


def save_csv(
    geometry: dict[str, object],
    track_rsun: np.ndarray,
    video_line_rsun: np.ndarray,
    cape_metrics: dict[str, float],
    video_metrics: dict[str, float],
    video_status: str,
) -> None:
    rows: list[dict[str, object]] = []
    for index, (jd, point) in enumerate(zip(geometry["fit_jds"], track_rsun)):
        rows.append({
            "version": VERSION,
            "record_type": "CAPE_TOWN_SITE_COORD_TRACK_ROTATED_180",
            "sequence": index,
            "event": "",
            "jd_tdb": float(jd),
            "utc": base.utc_at(float(jd)),
            "x_rsun": float(point[0]),
            "y_rsun": float(point[1]),
            "beta_from_horizontal_deg": cape_metrics["angle_deg"],
            "slope_tan_beta": cape_metrics["slope"],
            "r2_linear": cape_metrics["r2_linear"],
            "r2_quadratic": cape_metrics["r2_quadratic"],
            "r2_cubic": cape_metrics["r2_cubic"],
            "status": "JPL SITE_COORD / DISPLAY ROTATION ONLY",
        })

    for name, event in geometry["events"].items():
        solar_radius = geometry["events"]["CA"].sun_radius_arcsec
        point = -(event.point_arcsec / solar_radius)
        rows.append({
            "version": VERSION,
            "record_type": "CAPE_TOWN_EVENT_ROTATED_180",
            "sequence": np.nan,
            "event": name,
            "jd_tdb": event.jd_tdb,
            "utc": event.utc,
            "x_rsun": float(point[0]),
            "y_rsun": float(point[1]),
            "beta_from_horizontal_deg": cape_metrics["angle_deg"],
            "slope_tan_beta": cape_metrics["slope"],
            "r2_linear": cape_metrics["r2_linear"],
            "r2_quadratic": cape_metrics["r2_quadratic"],
            "r2_cubic": cape_metrics["r2_cubic"],
            "status": "JPL SITE_COORD EVENT",
        })

    for index, point in enumerate(video_line_rsun):
        rows.append({
            "version": VERSION,
            "record_type": "SDO_VIDEO_TLS_REFERENCE_ROTATED_180",
            "sequence": index,
            "event": "",
            "jd_tdb": np.nan,
            "utc": "",
            "x_rsun": float(point[0]),
            "y_rsun": float(point[1]),
            "beta_from_horizontal_deg": video_metrics["angle_deg"],
            "slope_tan_beta": video_metrics["slope"],
            "r2_linear": video_metrics["r2_linear"],
            "r2_quadratic": video_metrics["r2_quadratic"],
            "r2_cubic": video_metrics["r2_cubic"],
            "status": video_status,
        })

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, float_format="%.12f")


def render() -> None:
    geometry = base.build_geometry()
    events = geometry["events"]
    solar_radius_arcsec = events["CA"].sun_radius_arcsec

    original_track_rsun = np.asarray(geometry["fit_points_arcsec"], dtype=float) / solar_radius_arcsec
    track_rsun = -original_track_rsun
    cape_metrics = pca_metrics(track_rsun)

    original_ca_rsun = events["CA"].point_arcsec / solar_radius_arcsec
    original_video_line = make_video_reference_line(original_track_rsun, original_ca_rsun)
    video_line_rsun = -original_video_line

    _video_points, video_metrics, video_status = load_clean_video_points()
    delta_beta = cape_metrics["angle_deg"] - VIDEO_TLS_ANGLE_DEG
    delta_slope = cape_metrics["slope"] - video_metrics["slope"]

    save_csv(
        geometry,
        track_rsun,
        video_line_rsun,
        cape_metrics,
        video_metrics,
        video_status,
    )

    fig, ax = plt.subplots(figsize=(12.4, 8.1), dpi=190)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    theta = np.linspace(0.0, 2.0 * np.pi, 1800)
    ax.plot(np.cos(theta), np.sin(theta), color=SOLAR, lw=0.36, zorder=1)
    ax.axhline(0.0, color=SPINE, lw=0.18, alpha=0.55, zorder=0)
    ax.axvline(0.0, color=SPINE, lw=0.18, alpha=0.55, zorder=0)

    ax.plot(
        track_rsun[:, 0],
        track_rsun[:, 1],
        color=JPL_COLOR,
        lw=0.72,
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
        lw=0.90,
        linestyle=(0, (7, 5)),
        label="SDO video TLS reference",
        zorder=6,
    )

    label_offsets = {
        "C1": (-0.075, 0.024),
        "C2": (-0.067, 0.018),
        "CA": (-0.028, 0.034),
        "C3": (0.024, -0.034),
        "C4": (0.030, -0.044),
    }
    for name in ("C1", "C2", "CA", "C3", "C4"):
        event = events[name]
        point = -(event.point_arcsec / solar_radius_arcsec)
        radius = event.venus_radius_arcsec / solar_radius_arcsec
        ax.add_patch(Circle(
            (point[0], point[1]),
            radius,
            fill=False,
            ec=VENUS_COLOR,
            lw=0.28,
            zorder=7,
        ))
        ax.scatter([point[0]], [point[1]], s=4.2, color=JPL_COLOR, edgecolor=BG, lw=0.18, zorder=8)
        dx, dy = label_offsets[name]
        ax.text(point[0] + dx, point[1] + dy, name, color=TEXT, fontsize=6.8, zorder=9)

    table_rows = [
        ["β from horizontal", f"{cape_metrics['angle_deg']:.6f}", f"{VIDEO_TLS_ANGLE_DEG:.6f}", "deg"],
        ["slope m = tan(β)", f"{cape_metrics['slope']:.9f}", f"{video_metrics['slope']:.9f}", ""],
        ["R² linear", f"{cape_metrics['r2_linear']:.9f}", f"{video_metrics['r2_linear']:.9f}", ""],
        ["R² quadratic", f"{cape_metrics['r2_quadratic']:.9f}", f"{video_metrics['r2_quadratic']:.9f}", ""],
        ["R² cubic", f"{cape_metrics['r2_cubic']:.9f}", f"{video_metrics['r2_cubic']:.9f}", ""],
        ["RMS perpendicular", f"{cape_metrics['rms']:.9f}", f"{video_metrics['rms']:.9f}", "R_sun"],
        ["samples", f"{int(cape_metrics['count'])}", f"{int(video_metrics['count'])}", ""],
        ["Δβ / Δm", f"{delta_beta:+.6f}°", f"{delta_slope:+.9f}", ""],
    ]
    table = ax.table(
        cellText=table_rows,
        colLabels=["Quantity", "Cape Town JPL", "SDO video", "Unit"],
        colWidths=[0.30, 0.24, 0.24, 0.12],
        bbox=[0.31, 0.075, 0.49, 0.30],
        cellLoc="center",
        colLoc="center",
        zorder=20,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.6)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(SPINE)
        cell.set_linewidth(0.24)
        if row == 0:
            cell.set_facecolor(HEADER_BG)
            cell.get_text().set_color(TEXT)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(PANEL)
            if row == len(table_rows) and col in (1, 2):
                cell.get_text().set_color(DELTA_VALUE)
                cell.get_text().set_weight("bold")
            elif col == 1:
                cell.get_text().set_color(CAPE_VALUE)
                cell.get_text().set_weight("bold")
            elif col == 2:
                cell.get_text().set_color(VIDEO_VALUE)
                cell.get_text().set_weight("bold")
            else:
                cell.get_text().set_color(MUTED)

    ax.text(
        0.555,
        0.392,
        "TRACK GEOMETRY AND TRIGONOMETRY",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=TEXT,
        fontsize=8.2,
        fontweight="bold",
        zorder=21,
    )
    ax.text(
        0.555,
        0.052,
        f"Display rotation: 180° (x→−x, y→−y); slopes from horizontal are invariant modulo 180°.  Video stats: {video_status}.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=MUTED,
        fontsize=5.8,
        zorder=21,
    )

    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-1.06, 0.08)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=GRID, lw=0.16, alpha=0.62)
    ax.tick_params(colors=MUTED, labelsize=7, width=0.22, length=2.0)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.24)

    ax.set_xlabel("Solar-screen X (R_sun)", color=MUTED, fontsize=8)
    ax.set_ylabel("Solar-screen Y (R_sun)", color=MUTED, fontsize=8)
    ax.set_title(
        "2012 Venus Transit — Cape Town SITE_COORD vs SDO Video\n"
        "Engineering Half-Sun Track Reconstruction — IERS-0012AA format",
        color=TEXT,
        fontsize=10.2,
        pad=9,
    )

    legend = ax.legend(loc="lower right", fontsize=6.8, frameon=True, handlelength=3.0)
    legend.get_frame().set_facecolor(PANEL)
    legend.get_frame().set_edgecolor(SPINE)
    legend.get_frame().set_linewidth(0.28)
    for item in legend.get_texts():
        item.set_color(TEXT)

    fig.savefig(PLOT_PNG, dpi=320, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    display(IPythonImage(filename=str(PLOT_PNG)))


def main() -> int:
    render()
    print("CODE INPUTS")
    print("Cape Town JPL Horizons SITE_COORD and V0007 SDO video fit")
    print("COMMENTS")
    print("No AI images; 180-degree display rotation only; C1 left and C4 lower-right")
    print("RESULTS")
    print(f"PNG: {PLOT_PNG}")
    print("OUTPUT SUMMARY")
    print(f"CSV: {OUTPUT_CSV}")
    print("PAPER COMPARISON")
    print("NOT USED")
    print("EQUATION STATUS")
    print("Angles, tan(beta), R2 values, and RMS are rendered inside the PNG")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# IERS-0012AK
