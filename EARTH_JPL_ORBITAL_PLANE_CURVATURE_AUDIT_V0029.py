# V0029
# Audit reference: Earth JPL orbital-plane cross-track curvature and 2-D tangent-plane path-curvature analysis, A.D. 3000-4500.
from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def need(module: str, package: str) -> None:
    if importlib.util.find_spec(module) is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", package],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


for module_name, package_name in [
    ("astroquery", "astroquery"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("IPython", "ipython"),
]:
    need(module_name, package_name)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display

VERSION = "V0029"
LOCAL_TZ = ZoneInfo("America/Bogota")
EARTH_ID = "399"
CENTER = "@10"
REFPLANE = "ecliptic"
ABERRATIONS = "geometric"
AU_KM = 149597870.700
START_YEAR = 3000
STOP_YEAR = 4500
SMOOTH_WINDOWS = (101, 201, 301)
PRIMARY_WINDOW = 201
MAX_RETRIES = 4
RETRY_DELAY_SECONDS = 3.0

OUT = Path("/content/EARTH_JPL_ORBITAL_PLANE_CURVATURE_AUDIT_V0029_OUTPUT")
CACHE_CSV = OUT / "EARTH_JPL_ANNUAL_STATES_3000_4500_V0029.csv"
DATA_CSV = OUT / "EARTH_JPL_CURVATURE_RESULTS_V0029.csv"
SUMMARY_CSV = OUT / "EARTH_JPL_CURVATURE_SUMMARY_V0029.csv"
PNG_CROSS = OUT / "EARTH_JPL_CROSS_TRACK_CURVATURE_2X_V0029.png"
PNG_PATH = OUT / "EARTH_JPL_TANGENT_PLANE_PATH_CURVATURE_V0029.png"
PNG_DERIV = OUT / "EARTH_JPL_CURVATURE_DERIVATIVES_V0029.png"
PNG_TABLE = OUT / "EARTH_JPL_CURVATURE_SUMMARY_TABLE_V0029.png"


def normalize(vector: np.ndarray) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    if not np.isfinite(magnitude) or magnitude <= 0.0:
        raise RuntimeError("REJECTED invalid vector magnitude")
    return vector / magnitude


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    magnitudes = np.linalg.norm(vectors, axis=1)
    if np.any(~np.isfinite(magnitudes)) or np.any(magnitudes <= 0.0):
        raise RuntimeError("REJECTED invalid vector-row magnitude")
    return vectors / magnitudes[:, None]


def query_states() -> pd.DataFrame:
    if CACHE_CSV.exists():
        cached = pd.read_csv(CACHE_CSV)
        required = {"astronomical_year", "datetime_jd", "x", "y", "z", "vx", "vy", "vz"}
        if required.issubset(cached.columns) and len(cached) >= 1490:
            return cached.sort_values("astronomical_year").reset_index(drop=True)

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                table = Horizons(
                    id=EARTH_ID,
                    id_type="majorbody",
                    location=CENTER,
                    epochs={
                        "start": f"{START_YEAR}-01-01",
                        "stop": f"{STOP_YEAR}-01-01",
                        "step": "1y",
                    },
                ).vectors(refplane=REFPLANE, aberrations=ABERRATIONS)
            frame = table.to_pandas()
            required = ["datetime_jd", "x", "y", "z", "vx", "vy", "vz"]
            if any(column not in frame.columns for column in required):
                raise RuntimeError("REJECTED missing JPL vector columns")
            frame = frame[required].sort_values("datetime_jd").reset_index(drop=True)
            if len(frame) < 1490:
                raise RuntimeError(f"REJECTED insufficient annual samples: {len(frame)}")
            frame.insert(0, "astronomical_year", START_YEAR + np.arange(len(frame), dtype=float))
            frame.to_csv(CACHE_CSV, index=False, float_format="%.12f")
            return frame
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * attempt)
    raise RuntimeError(f"JPL annual query failed after {MAX_RETRIES} attempts: {last_error}")


def local_polynomial_series(years: np.ndarray, values: np.ndarray, window: int, degree: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if window % 2 == 0:
        window += 1
    half = window // 2
    smooth = np.full_like(values, np.nan, dtype=float)
    first = np.full_like(values, np.nan, dtype=float)
    second = np.full_like(values, np.nan, dtype=float)
    for index in range(half, len(years) - half):
        local_x = years[index - half:index + half + 1] - years[index]
        local_y = values[index - half:index + half + 1]
        coefficients = np.polyfit(local_x, local_y, degree)
        polynomial = np.poly1d(coefficients)
        smooth[index] = float(polynomial(0.0))
        first[index] = float(np.polyder(polynomial, 1)(0.0))
        second[index] = float(np.polyder(polynomial, 2)(0.0))
    return smooth, first, second


def derive_geometry(states: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    years = states["astronomical_year"].to_numpy(dtype=float)
    r = states[["x", "y", "z"]].to_numpy(dtype=float) * AU_KM
    v = states[["vx", "vy", "vz"]].to_numpy(dtype=float) * AU_KM
    normals = normalize_rows(np.cross(r, v))
    normals[normals[:, 2] < 0.0] *= -1.0

    n0 = normalize(normals[0])
    n1 = normalize(normals[-1])
    u = normalize(n1 - np.dot(n1, n0) * n0)
    w = normalize(np.cross(n0, u))

    along = np.degrees(np.arctan2(normals @ u, normals @ n0))
    cross = np.degrees(np.arcsin(np.clip(normals @ w, -1.0, 1.0)))

    result = pd.DataFrame({
        "astronomical_year": years,
        "signed_along_track_deg": along,
        "cross_track_deg": cross,
    })

    for window in SMOOTH_WINDOWS:
        cross_smooth, cross_first, cross_second = local_polynomial_series(years, cross, window, degree=3)
        along_smooth, along_first, along_second = local_polynomial_series(years, along, window, degree=3)
        result[f"cross_smooth_{window}_deg"] = cross_smooth
        result[f"cross_first_{window}_arcsec_per_century"] = cross_first * 360000.0
        result[f"cross_second_{window}_arcsec_per_century2"] = cross_second * 36000000.0
        result[f"along_smooth_{window}_deg"] = along_smooth
        result[f"along_first_{window}_arcsec_per_century"] = along_first * 360000.0
        result[f"along_second_{window}_arcsec_per_century2"] = along_second * 36000000.0

        speed = np.sqrt(along_first ** 2 + cross_first ** 2)
        numerator = np.abs(along_first * cross_second - cross_first * along_second)
        curvature = np.divide(numerator, speed ** 3, out=np.full_like(speed, np.nan), where=speed > 0.0)
        signed_curvature = np.divide(
            along_first * cross_second - cross_first * along_second,
            speed ** 3,
            out=np.full_like(speed, np.nan),
            where=speed > 0.0,
        )
        result[f"path_curvature_{window}_per_deg"] = curvature
        result[f"signed_path_curvature_{window}_per_deg"] = signed_curvature
        result[f"radius_of_curvature_{window}_deg"] = np.divide(
            1.0,
            curvature,
            out=np.full_like(curvature, np.nan),
            where=curvature > 0.0,
        )

    primary_cross = result[f"cross_smooth_{PRIMARY_WINDOW}_deg"].to_numpy(dtype=float)
    valid = np.isfinite(primary_cross)
    valid_indices = np.where(valid)[0]
    peak_index = int(valid_indices[np.argmax(primary_cross[valid])])

    x_centered = years[valid] - np.mean(years[valid])
    linear_coeff = np.polyfit(x_centered, primary_cross[valid], 1)
    quadratic_coeff = np.polyfit(x_centered, primary_cross[valid], 2)
    linear_fit = np.polyval(linear_coeff, x_centered)
    quadratic_fit = np.polyval(quadratic_coeff, x_centered)

    def r2(y: np.ndarray, yhat: np.ndarray) -> float:
        return 1.0 - float(np.sum((y - yhat) ** 2)) / float(np.sum((y - np.mean(y)) ** 2))

    a, b, _ = quadratic_coeff
    vertex_centered = -b / (2.0 * a) if abs(a) > 0.0 else float("nan")
    vertex_year = float(vertex_centered + np.mean(years[valid]))

    primary_curvature = result[f"path_curvature_{PRIMARY_WINDOW}_per_deg"].to_numpy(dtype=float)
    curvature_valid = np.isfinite(primary_curvature)
    curvature_indices = np.where(curvature_valid)[0]
    curvature_max_index = int(curvature_indices[np.argmax(primary_curvature[curvature_valid])])

    summary = {
        "cross_peak_year": float(years[peak_index]),
        "cross_peak_deg": float(primary_cross[peak_index]),
        "cross_peak_2x_deg": float(2.0 * primary_cross[peak_index]),
        "linear_slope_arcsec_per_century": float(linear_coeff[0] * 360000.0),
        "linear_r2": r2(primary_cross[valid], linear_fit),
        "quadratic_r2": r2(primary_cross[valid], quadratic_fit),
        "quadratic_vertex_year": vertex_year,
        "quadratic_curvature_deg_per_year2": float(2.0 * a),
        "maximum_path_curvature_year": float(years[curvature_max_index]),
        "maximum_path_curvature_per_deg": float(primary_curvature[curvature_max_index]),
        "minimum_radius_of_curvature_deg": float(result[f"radius_of_curvature_{PRIMARY_WINDOW}_deg"].iloc[curvature_max_index]),
        "reference_normal": n0,
        "direction_u": u,
        "cross_w": w,
    }
    return result, summary


def style_axis(ax) -> None:
    ax.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.5)
    ax.grid(True, linewidth=0.35, alpha=0.22)


def plot_cross(data: pd.DataFrame, summary: dict) -> None:
    years = data["astronomical_year"].to_numpy(dtype=float)
    raw = data["cross_track_deg"].to_numpy(dtype=float)
    smooth = data[f"cross_smooth_{PRIMARY_WINDOW}_deg"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(17, 8.5), dpi=140)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(years, 2.0 * raw, linewidth=0.28, alpha=0.28, label="Raw cross-track ×2")
    ax.plot(years, 2.0 * smooth, linewidth=1.25, label=f"{PRIMARY_WINDOW}-year local-polynomial cross-track ×2")
    ax.axvline(summary["quadratic_vertex_year"], linewidth=0.6, linestyle=":", alpha=0.9)
    ax.scatter([summary["cross_peak_year"]], [summary["cross_peak_2x_deg"]], s=22, zorder=5)
    ax.set_title(
        "Earth JPL Orbital-Plane Cross-Track Curvature — Physical 2× Exaggeration\n"
        f"Peak year {summary['cross_peak_year']:.0f}   |   Quadratic vertex {summary['quadratic_vertex_year']:.2f}",
        color="white", fontsize=14, weight="bold", pad=16,
    )
    ax.set_xlabel("Astronomical year", color="white")
    ax.set_ylabel("Cross-track displacement ×2 (degrees)", color="white")
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()
    fig.savefig(PNG_CROSS, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_path(data: pd.DataFrame, summary: dict) -> None:
    along = data[f"along_smooth_{PRIMARY_WINDOW}_deg"].to_numpy(dtype=float)
    cross = data[f"cross_smooth_{PRIMARY_WINDOW}_deg"].to_numpy(dtype=float)
    years = data["astronomical_year"].to_numpy(dtype=float)
    valid = np.isfinite(along) & np.isfinite(cross)
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=150)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(along[valid], cross[valid], linewidth=1.1, label="Smoothed JPL orbital-normal path")
    for year in range(3200, 4500, 200):
        index = int(np.argmin(np.abs(years - year)))
        if np.isfinite(along[index]) and np.isfinite(cross[index]):
            ax.scatter([along[index]], [cross[index]], s=14)
            ax.text(along[index], cross[index], f" {year}", color="white", fontsize=7)
    ax.set_title("Earth Orbital-Normal Path in the Local Tangent Plane", color="white", fontsize=14, weight="bold", pad=14)
    ax.set_xlabel("Signed along-track displacement (degrees)", color="white")
    ax.set_ylabel("Cross-track displacement (degrees)", color="white")
    ax.set_aspect("equal", adjustable="datalim")
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_derivatives(data: pd.DataFrame) -> None:
    years = data["astronomical_year"].to_numpy(dtype=float)
    first = data[f"cross_first_{PRIMARY_WINDOW}_arcsec_per_century"].to_numpy(dtype=float)
    second = data[f"cross_second_{PRIMARY_WINDOW}_arcsec_per_century2"].to_numpy(dtype=float)
    curvature = data[f"signed_path_curvature_{PRIMARY_WINDOW}_per_deg"].to_numpy(dtype=float)
    finite_curvature = curvature[np.isfinite(curvature)]
    scale = np.nanmax(np.abs(first)) / max(np.nanmax(np.abs(finite_curvature)), np.finfo(float).eps)

    fig, ax = plt.subplots(figsize=(17, 8.5), dpi=140)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(years, first, linewidth=0.9, label="Cross-track first derivative")
    ax.plot(years, second, linewidth=0.75, label="Cross-track second derivative")
    ax.plot(years, curvature * scale, linewidth=0.85, linestyle="--", label="Signed path curvature, display-scaled")
    ax.axhline(0.0, linewidth=0.5, alpha=0.7)
    ax.set_title(
        "Earth JPL Cross-Track Derivatives and Signed Path Curvature\n"
        f"{PRIMARY_WINDOW}-year local-polynomial extraction",
        color="white", fontsize=14, weight="bold", pad=16,
    )
    ax.set_xlabel("Astronomical year", color="white")
    ax.set_ylabel("Derivative scale / display-scaled curvature", color="white")
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()
    fig.savefig(PNG_DERIV, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_summary_table(summary_frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(15, 5.6), dpi=160)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.axis("off")
    table = ax.table(cellText=summary_frame.values, colLabels=summary_frame.columns, cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#777777")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#17324d")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#1b1b1b" if row % 2 else "#252525")
            cell.get_text().set_color("white")
    ax.set_title("Earth JPL Orbital-Plane Curvature Audit", color="white", fontsize=15, weight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(PNG_TABLE, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"OUTPUT VERSION {VERSION}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    states = query_states()
    data, summary = derive_geometry(states)
    data.to_csv(DATA_CSV, index=False, float_format="%.12f")

    rows = [
        ("Cross-track peak year", f"{summary['cross_peak_year']:.0f}"),
        ("Cross-track peak", f"{summary['cross_peak_deg']:.9f} deg"),
        ("Cross-track peak at 2×", f"{summary['cross_peak_2x_deg']:.9f} deg"),
        ("Global linear slope", f"{summary['linear_slope_arcsec_per_century']:+.9f} arcsec/century"),
        ("Global linear R²", f"{summary['linear_r2']:.12f}"),
        ("Quadratic R²", f"{summary['quadratic_r2']:.12f}"),
        ("Quadratic vertex year", f"{summary['quadratic_vertex_year']:.6f}"),
        ("Quadratic second derivative", f"{summary['quadratic_curvature_deg_per_year2']:+.12e} deg/year²"),
        ("Maximum path-curvature year", f"{summary['maximum_path_curvature_year']:.0f}"),
        ("Maximum path curvature", f"{summary['maximum_path_curvature_per_deg']:.12e} per degree"),
        ("Minimum radius of curvature", f"{summary['minimum_radius_of_curvature_deg']:.9f} deg"),
    ]
    summary_frame = pd.DataFrame(rows, columns=["Quantity", "Result"])
    summary_frame.to_csv(SUMMARY_CSV, index=False)

    plot_cross(data, summary)
    plot_path(data, summary)
    plot_derivatives(data)
    plot_summary_table(summary_frame)

    display(Image(filename=str(PNG_CROSS)))
    display(Image(filename=str(PNG_PATH)))
    display(Image(filename=str(PNG_DERIV)))
    display(Image(filename=str(PNG_TABLE)))
    print(summary_frame.to_string(index=False))
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0029