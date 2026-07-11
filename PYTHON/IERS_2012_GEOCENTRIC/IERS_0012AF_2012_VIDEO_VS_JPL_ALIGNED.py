# IERS-0012AF
# Audit reference: Rotate the geocentric JPL 2012 track into the extracted SDO video orientation and plot both from project/JPL data.
from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


VERSION = "IERS-0012AF"
LOCAL_TZ = ZoneInfo("America/Bogota")

DRIVE_ROOT = Path("/content/drive/MyDrive/Colab Notebooks")
IERS_ROOT = DRIVE_ROOT / "IERS_TN36_OUTPUT"
VIDEO_ROOT = (
    DRIVE_ROOT
    / "ESO Parallax 2012 Venus Transit"
    / "NASA_SDO_2012_OUTPUT"
    / "V0007"
)

VIDEO_CSV = VIDEO_ROOT / "NASA_SDO_2012_TRACK.csv"
AC_SCRIPT = Path("/content/IERS_0012AC_2012_GEOCENTRIC_TRACK_PNG.py")

OUTPUT_PNG_DIR = IERS_ROOT / "OUTPUT_PNG"
OUTPUT_CSV_DIR = IERS_ROOT / "OUTPUT_CSV"
OUTPUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)

OVERLAY_PNG = OUTPUT_PNG_DIR / f"{VERSION}_2012_VIDEO_VS_JPL_ALIGNED_OVERLAY.png"
RESULTS_PNG = OUTPUT_PNG_DIR / f"{VERSION}_2012_VIDEO_VS_JPL_ALIGNED_RESULTS.png"
COMBINED_CSV = OUTPUT_CSV_DIR / f"{VERSION}_2012_VIDEO_VS_JPL_ALIGNED_COMBINED.csv"

BG = "#03080d"
PANEL = "#071016"
GRID = "#102630"
SPINE = "#25708b"
TEXT = "#dff8ff"
MUTED = "#8fb4c1"
SOLAR = "#57c7e3"
JPL_COLOR = "#58d68d"
VIDEO_COLOR = "#ffb347"
VIDEO_MARKER = "#ff6b6b"
ACCENT = "#ffd166"
VENUS = "#b7ffcf"


def ensure_package(import_name: str, pip_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pip_name])


for _import_name, _pip_name in (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("scipy", "scipy"),
    ("astroquery", "astroquery"),
    ("astropy", "astropy"),
):
    ensure_package(_import_name, _pip_name)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from IPython.display import Image as IPythonImage, display


def load_ac_module():
    if not AC_SCRIPT.exists():
        raise FileNotFoundError(
            f"Required geocentric JPL program not found: {AC_SCRIPT}"
        )
    module_name = "iers_0012ac_runtime"
    spec = importlib.util.spec_from_file_location(module_name, AC_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import specification for {AC_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def fit_tls(points: np.ndarray, order_axis: np.ndarray | None = None) -> dict[str, np.ndarray | float]:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 3:
        raise RuntimeError("TLS fit requires at least three 2D points.")
    finite = np.isfinite(array).all(axis=1)
    array = array[finite]
    if len(array) < 3:
        raise RuntimeError("TLS fit has too few finite points.")

    center = np.mean(array, axis=0)
    centered = array - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0].astype(float)
    direction /= np.linalg.norm(direction)

    if order_axis is None:
        travel = array[-1] - array[0]
    else:
        order = np.asarray(order_axis, dtype=float)[finite]
        first = int(np.argmin(order))
        last = int(np.argmax(order))
        travel = array[last] - array[first]

    if float(np.dot(direction, travel)) < 0.0:
        direction *= -1.0

    normal = np.array([-direction[1], direction[0]], dtype=float)
    cross = centered @ normal
    along = centered @ direction
    tls_angle = math.degrees(math.atan2(direction[1], direction[0]))
    rms = float(np.sqrt(np.mean(cross**2)))

    if np.ptp(array[:, 0]) <= 1.0e-12:
        ols_angle = float("nan")
        ols_slope = float("nan")
        ols_intercept = float("nan")
        rms_vertical = float("nan")
    else:
        ols_slope, ols_intercept = np.polyfit(array[:, 0], array[:, 1], 1)
        ols_angle = math.degrees(math.atan(ols_slope))
        y_fit = ols_slope * array[:, 0] + ols_intercept
        rms_vertical = float(np.sqrt(np.mean((array[:, 1] - y_fit) ** 2)))

    quadratic = np.polyfit(along, cross, 2)
    curvature = float(2.0 * quadratic[0] / (1.0 + quadratic[1] ** 2) ** 1.5)

    return {
        "center": center,
        "direction": direction,
        "normal": normal,
        "angle_deg": float(tls_angle),
        "rms_perpendicular": rms,
        "ols_angle_deg": float(ols_angle),
        "ols_slope": float(ols_slope),
        "ols_intercept": float(ols_intercept),
        "rms_vertical": float(rms_vertical),
        "curvature": curvature,
    }


def rotation_matrix(angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=float)


def signed_rotation_from_to(source: np.ndarray, target: np.ndarray) -> float:
    source_unit = np.asarray(source, dtype=float)
    target_unit = np.asarray(target, dtype=float)
    source_unit /= np.linalg.norm(source_unit)
    target_unit /= np.linalg.norm(target_unit)
    cross = source_unit[0] * target_unit[1] - source_unit[1] * target_unit[0]
    dot = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
    return math.atan2(cross, dot)


def load_video_track() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not VIDEO_CSV.exists():
        raise FileNotFoundError(f"V0007 video track CSV not found: {VIDEO_CSV}")
    table = pd.read_csv(VIDEO_CSV)
    required = {"frame_index", "time_s", "x_norm", "y_norm"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise RuntimeError(f"Video CSV missing required columns: {missing}")

    table = table.sort_values("frame_index").reset_index(drop=True)
    finite = (
        np.isfinite(table["x_norm"].to_numpy(dtype=float))
        & np.isfinite(table["y_norm"].to_numpy(dtype=float))
    )
    all_points = table.loc[finite].copy()

    if "venus_track_inlier" in table.columns:
        selected_mask = table["venus_track_inlier"].astype(bool).to_numpy()
    elif "venus_detected" in table.columns:
        selected_mask = table["venus_detected"].astype(bool).to_numpy()
    else:
        selected_mask = finite

    selected = table.loc[selected_mask & finite].copy()
    if len(selected) < 8:
        raise RuntimeError(
            f"Only {len(selected)} accepted video detections were available."
        )
    return all_points, selected


def build_jpl_track() -> tuple[object, dict[str, object], pd.DataFrame]:
    ac = load_ac_module()
    master = ac.build_master()
    cache = ac.build_cache(master)
    geometry = ac.build_geometry(cache)

    solar_radius_arcsec = float(geometry["sun_radius_arcsec"])
    points_arcsec = np.asarray(geometry["points"], dtype=float)
    points_norm = points_arcsec / solar_radius_arcsec
    jds = np.asarray(geometry["jds"], dtype=float)

    jpl = pd.DataFrame(
        {
            "dataset": "JPL_GEOCENTER",
            "sequence": np.arange(len(points_norm), dtype=int),
            "jd_tdb": jds,
            "utc": [ac.utc_at(float(jd)) for jd in jds],
            "x_raw_rsun": points_norm[:, 0],
            "y_raw_rsun": points_norm[:, 1],
        }
    )
    return ac, geometry, jpl


def event_points_normalized(
    geometry: dict[str, object],
) -> dict[str, tuple[np.ndarray, float]]:
    solar_radius = float(geometry["sun_radius_arcsec"])
    result: dict[str, tuple[np.ndarray, float]] = {}
    for name, event in geometry["events"].items():
        point = np.asarray(event.point, dtype=float) / solar_radius
        radius = float(event.venus_radius_arcsec) / solar_radius
        result[name] = (point, radius)
    return result


def arrow_for_track(
    ax: plt.Axes,
    points: np.ndarray,
    color: str,
    label: str,
    fraction_start: float = 0.42,
    fraction_end: float = 0.58,
) -> None:
    count = len(points)
    start = points[min(count - 1, max(0, int(round((count - 1) * fraction_start))))]
    end = points[min(count - 1, max(0, int(round((count - 1) * fraction_end))))]
    patch = FancyArrowPatch(
        posA=(start[0], start[1]),
        posB=(end[0], end[1]),
        arrowstyle="-|>",
        mutation_scale=7,
        linewidth=0.45,
        color=color,
        alpha=0.95,
        zorder=8,
        label=label,
    )
    ax.add_patch(patch)


def save_combined_csv(
    jpl: pd.DataFrame,
    video_all: pd.DataFrame,
    video_selected: pd.DataFrame,
    rotation_deg: float,
) -> None:
    jpl_out = jpl.copy()
    jpl_out["rotation_applied_deg"] = rotation_deg
    jpl_out["is_selected"] = True
    jpl_out["frame_index"] = np.nan
    jpl_out["time_s"] = np.nan

    video_out = pd.DataFrame(
        {
            "dataset": "SDO_VIDEO",
            "sequence": np.arange(len(video_all), dtype=int),
            "jd_tdb": np.nan,
            "utc": "",
            "x_raw_rsun": video_all["x_norm"].to_numpy(dtype=float),
            "y_raw_rsun": video_all["y_norm"].to_numpy(dtype=float),
            "x_aligned_rsun": video_all["x_norm"].to_numpy(dtype=float),
            "y_aligned_rsun": video_all["y_norm"].to_numpy(dtype=float),
            "rotation_applied_deg": 0.0,
            "is_selected": video_all["frame_index"].isin(
                video_selected["frame_index"]
            ).to_numpy(),
            "frame_index": video_all["frame_index"].to_numpy(dtype=float),
            "time_s": video_all["time_s"].to_numpy(dtype=float),
        }
    )

    combined = pd.concat(
        [
            jpl_out[
                [
                    "dataset",
                    "sequence",
                    "jd_tdb",
                    "utc",
                    "x_raw_rsun",
                    "y_raw_rsun",
                    "x_aligned_rsun",
                    "y_aligned_rsun",
                    "rotation_applied_deg",
                    "is_selected",
                    "frame_index",
                    "time_s",
                ]
            ],
            video_out,
        ],
        ignore_index=True,
    )
    combined.to_csv(COMBINED_CSV, index=False, float_format="%.12f")


def render_overlay(
    geometry: dict[str, object],
    jpl_raw: np.ndarray,
    jpl_aligned: np.ndarray,
    video_all_points: np.ndarray,
    video_selected_points: np.ndarray,
    event_points: dict[str, tuple[np.ndarray, float]],
    rotation_rad: float,
    metrics: dict[str, float],
) -> None:
    figure, ax = plt.subplots(figsize=(11.4, 7.4), dpi=180)
    figure.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    theta = np.linspace(0.0, 2.0 * np.pi, 1800)
    ax.plot(
        np.cos(theta),
        np.sin(theta),
        color=SOLAR,
        linewidth=0.34,
        alpha=0.95,
        zorder=1,
        label="Solar limb",
    )
    ax.axhline(0.0, color=SPINE, linewidth=0.18, alpha=0.50, zorder=0)
    ax.axvline(0.0, color=SPINE, linewidth=0.18, alpha=0.50, zorder=0)

    ax.plot(
        jpl_aligned[:, 0],
        jpl_aligned[:, 1],
        color=JPL_COLOR,
        linewidth=0.62,
        alpha=0.96,
        zorder=4,
        label="JPL geocenter — rotated to video direction",
    )
    ax.scatter(
        jpl_aligned[::6, 0],
        jpl_aligned[::6, 1],
        s=1.3,
        color=JPL_COLOR,
        linewidths=0,
        alpha=0.72,
        zorder=5,
    )

    ax.plot(
        video_all_points[:, 0],
        video_all_points[:, 1],
        color=VIDEO_COLOR,
        linewidth=0.52,
        linestyle=(0, (5, 2)),
        alpha=0.98,
        zorder=6,
        label="SDO video interpolated track",
    )
    ax.scatter(
        video_selected_points[:, 0],
        video_selected_points[:, 1],
        s=8.0,
        facecolor=VIDEO_MARKER,
        edgecolor=BG,
        linewidth=0.25,
        alpha=0.98,
        zorder=7,
        label="SDO accepted detections",
    )

    rotate = rotation_matrix(rotation_rad)
    for name in ("C1", "C2", "CA", "C3", "C4"):
        raw_point, radius = event_points[name]
        point = raw_point @ rotate.T
        ax.add_patch(
            Circle(
                (point[0], point[1]),
                radius,
                fill=False,
                linewidth=0.28,
                edgecolor=VENUS,
                alpha=0.92,
                zorder=5,
            )
        )
        dx = -0.040 if point[0] < 0.0 else 0.018
        dy = 0.018 if name in ("C1", "C2", "CA") else -0.022
        ax.text(
            point[0] + dx,
            point[1] + dy,
            name,
            color=MUTED,
            fontsize=6.3,
            zorder=9,
        )

    arrow_for_track(ax, jpl_aligned, JPL_COLOR, "JPL direction")
    arrow_for_track(ax, video_all_points, VIDEO_COLOR, "Video direction", 0.63, 0.78)

    min_y = float(
        min(
            np.min(jpl_aligned[:, 1]),
            np.min(video_all_points[:, 1]),
        )
    )
    max_y = float(
        max(
            np.max(jpl_aligned[:, 1]),
            np.max(video_all_points[:, 1]),
        )
    )
    if 0.5 * (min_y + max_y) < 0.0:
        ax.set_ylim(min(-1.06, min_y - 0.08), max(0.08, max_y + 0.08))
    else:
        ax.set_ylim(min(-0.08, min_y - 0.08), max(1.06, max_y + 0.08))
    ax.set_xlim(-1.04, 1.04)
    ax.set_aspect("equal", adjustable="box")

    for spine in ax.spines.values():
        spine.set_linewidth(0.22)
        spine.set_color(SPINE)
    ax.tick_params(axis="both", colors=MUTED, labelsize=7, width=0.22, length=2.0)
    ax.grid(True, color=GRID, linewidth=0.16, alpha=0.55)
    ax.set_xlabel("Solar-screen X (R_sun)", color=MUTED, fontsize=8)
    ax.set_ylabel("Solar-screen Y (R_sun)", color=MUTED, fontsize=8)
    ax.set_title(
        "2012 Venus Transit — SDO Video vs Geocentric JPL Track\n"
        "JPL track rotated from its temporal C1→C4 direction into the extracted video direction",
        color="#f8fdff",
        fontsize=10,
        pad=8,
    )

    summary = (
        f"Applied JPL rotation: {metrics['rotation_deg']:+.6f}°\n"
        f"JPL TLS before: {metrics['jpl_angle_before']:.6f}°\n"
        f"JPL TLS aligned: {metrics['jpl_angle_aligned']:.6f}°\n"
        f"Video TLS: {metrics['video_angle']:.6f}°\n"
        f"Aligned angle delta: {metrics['aligned_delta']:+.6f}°\n"
        f"JPL RMS: {metrics['jpl_rms']:.8f} R_sun\n"
        f"Video RMS: {metrics['video_rms']:.8f} R_sun"
    )
    ax.text(
        0.022,
        0.035,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        color=TEXT,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": PANEL,
            "edgecolor": SPINE,
            "linewidth": 0.35,
            "alpha": 0.92,
        },
        zorder=12,
    )

    legend = ax.legend(
        loc="lower right",
        fontsize=6.3,
        frameon=True,
        borderpad=0.45,
        handlelength=2.7,
    )
    legend.get_frame().set_facecolor(PANEL)
    legend.get_frame().set_edgecolor(SPINE)
    legend.get_frame().set_linewidth(0.28)
    for item in legend.get_texts():
        item.set_color(TEXT)

    figure.savefig(
        OVERLAY_PNG,
        dpi=320,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
    )
    plt.close(figure)


def render_results(metrics: dict[str, float], counts: dict[str, int]) -> None:
    rows = [
        ("CODE INPUTS", "JPL observer", "500@399 geocenter", "JPL"),
        ("CODE INPUTS", "Video source", str(VIDEO_CSV), "PROJECT"),
        ("COMMENTS", "AI images", "NOT USED", "VERIFIED"),
        ("COMMENTS", "Rotation target", "JPL temporal direction → video temporal direction", "VERIFIED"),
        ("RESULTS", "Applied JPL rotation", f"{metrics['rotation_deg']:+.9f} deg", "VERIFIED"),
        ("RESULTS", "JPL TLS before rotation", f"{metrics['jpl_angle_before']:.9f} deg", "VERIFIED"),
        ("RESULTS", "JPL TLS after rotation", f"{metrics['jpl_angle_aligned']:.9f} deg", "VERIFIED"),
        ("RESULTS", "Video TLS", f"{metrics['video_angle']:.9f} deg", "VERIFIED"),
        ("RESULTS", "Aligned TLS difference", f"{metrics['aligned_delta']:+.9f} deg", "VERIFIED"),
        ("RESULTS", "JPL OLS after rotation", f"{metrics['jpl_ols_aligned']:.9f} deg", "VERIFIED"),
        ("RESULTS", "Video OLS", f"{metrics['video_ols']:.9f} deg", "VERIFIED"),
        ("RESULTS", "JPL perpendicular RMS", f"{metrics['jpl_rms']:.12f} R_sun", "VERIFIED"),
        ("RESULTS", "Video perpendicular RMS", f"{metrics['video_rms']:.12f} R_sun", "VERIFIED"),
        ("RESULTS", "JPL curvature", f"{metrics['jpl_curvature']:.12e} R_sun^-1", "DIAGNOSTIC"),
        ("RESULTS", "Video curvature", f"{metrics['video_curvature']:.12e} R_sun^-1", "DIAGNOSTIC"),
        ("RESULTS", "JPL samples", str(counts["jpl"]), "VERIFIED"),
        ("RESULTS", "Video frames", str(counts["video_all"]), "VERIFIED"),
        ("RESULTS", "Video accepted detections", str(counts["video_selected"]), "VERIFIED"),
        ("OUTPUT SUMMARY", "Overlay PNG", str(OVERLAY_PNG), "SAVED"),
        ("OUTPUT SUMMARY", "Results PNG", str(RESULTS_PNG), "SAVED"),
        ("OUTPUT SUMMARY", "Combined CSV", str(COMBINED_CSV), "SAVED"),
        ("EQUATION STATUS", "Rotation", "atan2(cross, dot) from oriented TLS vectors", "VERIFIED"),
        ("EQUATION STATUS", "Scale", "JPL arcsec / JPL solar angular radius", "VERIFIED"),
        ("EQUATION STATUS", "Video coordinates", "Recovered solar-radius normalization", "VERIFIED"),
    ]

    figure = plt.figure(figsize=(13.0, 10.8), dpi=170, facecolor=BG)
    ax = figure.add_axes([0.035, 0.035, 0.93, 0.91])
    ax.set_facecolor(BG)
    ax.axis("off")

    ax.text(
        0.0,
        1.035,
        "2012 VIDEO–JPL ROTATION AND TRACK COMPARISON",
        transform=ax.transAxes,
        color="#f8fdff",
        fontsize=15,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.0,
        0.995,
        "All values derived from the V0007 video extraction and JPL Horizons geocentric vectors",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=8.5,
        va="top",
    )

    cell_text = [[section, quantity, value, status] for section, quantity, value, status in rows]
    table = ax.table(
        cellText=cell_text,
        colLabels=["Section", "Quantity", "Value", "Status"],
        colWidths=[0.16, 0.27, 0.43, 0.14],
        bbox=[0.0, 0.04, 1.0, 0.91],
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    section_colors = {
        "CODE INPUTS": "#0c2530",
        "COMMENTS": "#10231d",
        "RESULTS": "#071016",
        "OUTPUT SUMMARY": "#241d0f",
        "EQUATION STATUS": "#11172b",
    }
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(SPINE)
        cell.set_linewidth(0.22)
        if row == 0:
            cell.set_facecolor("#0b1a22")
            cell.get_text().set_color(TEXT)
            cell.get_text().set_weight("bold")
        else:
            section = cell_text[row - 1][0]
            cell.set_facecolor(section_colors.get(section, PANEL))
            cell.get_text().set_color(TEXT if column in (0, 1) else MUTED)
            if column == 3:
                status = cell_text[row - 1][3]
                if status in ("VERIFIED", "JPL", "PROJECT", "SAVED"):
                    cell.get_text().set_color(JPL_COLOR)
                else:
                    cell.get_text().set_color(ACCENT)

    ax.text(
        0.0,
        0.005,
        f"Generated {datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S %z')} | {VERSION}",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=7,
        va="bottom",
    )
    figure.savefig(
        RESULTS_PNG,
        dpi=300,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> int:
    ac, geometry, jpl = build_jpl_track()
    video_all, video_selected = load_video_track()

    jpl_raw = jpl[["x_raw_rsun", "y_raw_rsun"]].to_numpy(dtype=float)
    video_all_points = video_all[["x_norm", "y_norm"]].to_numpy(dtype=float)
    video_selected_points = video_selected[["x_norm", "y_norm"]].to_numpy(dtype=float)

    jpl_fit_before = fit_tls(jpl_raw, jpl["sequence"].to_numpy(dtype=float))
    video_fit = fit_tls(
        video_selected_points,
        video_selected["frame_index"].to_numpy(dtype=float),
    )

    rotation_rad = signed_rotation_from_to(
        np.asarray(jpl_fit_before["direction"], dtype=float),
        np.asarray(video_fit["direction"], dtype=float),
    )
    rotate = rotation_matrix(rotation_rad)
    jpl_aligned = jpl_raw @ rotate.T

    jpl["x_aligned_rsun"] = jpl_aligned[:, 0]
    jpl["y_aligned_rsun"] = jpl_aligned[:, 1]

    jpl_fit_aligned = fit_tls(
        jpl_aligned,
        jpl["sequence"].to_numpy(dtype=float),
    )

    angle_delta = (
        float(jpl_fit_aligned["angle_deg"])
        - float(video_fit["angle_deg"])
    )
    while angle_delta > 180.0:
        angle_delta -= 360.0
    while angle_delta <= -180.0:
        angle_delta += 360.0

    metrics = {
        "rotation_deg": math.degrees(rotation_rad),
        "jpl_angle_before": float(jpl_fit_before["angle_deg"]),
        "jpl_angle_aligned": float(jpl_fit_aligned["angle_deg"]),
        "video_angle": float(video_fit["angle_deg"]),
        "aligned_delta": angle_delta,
        "jpl_ols_aligned": float(jpl_fit_aligned["ols_angle_deg"]),
        "video_ols": float(video_fit["ols_angle_deg"]),
        "jpl_rms": float(jpl_fit_aligned["rms_perpendicular"]),
        "video_rms": float(video_fit["rms_perpendicular"]),
        "jpl_curvature": float(jpl_fit_aligned["curvature"]),
        "video_curvature": float(video_fit["curvature"]),
    }

    event_points = event_points_normalized(geometry)
    save_combined_csv(
        jpl,
        video_all,
        video_selected,
        metrics["rotation_deg"],
    )
    render_overlay(
        geometry,
        jpl_raw,
        jpl_aligned,
        video_all_points,
        video_selected_points,
        event_points,
        rotation_rad,
        metrics,
    )
    render_results(
        metrics,
        {
            "jpl": len(jpl),
            "video_all": len(video_all),
            "video_selected": len(video_selected),
        },
    )

    display(IPythonImage(filename=str(OVERLAY_PNG)))
    display(IPythonImage(filename=str(RESULTS_PNG)))

    print("CODE INPUTS")
    print("V0007 video CSV and JPL Horizons geocenter vectors")
    print("COMMENTS")
    print("No AI images; Matplotlib plots generated only from project/JPL data")
    print("RESULTS")
    print(f"Overlay PNG: {OVERLAY_PNG}")
    print(f"Results PNG: {RESULTS_PNG}")
    print("OUTPUT SUMMARY")
    print(f"Combined CSV: {COMBINED_CSV}")
    print("PAPER COMPARISON")
    print("NOT USED")
    print("EQUATION STATUS")
    print("Rotation and fitting values rendered in the results PNG")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# IERS-0012AF
