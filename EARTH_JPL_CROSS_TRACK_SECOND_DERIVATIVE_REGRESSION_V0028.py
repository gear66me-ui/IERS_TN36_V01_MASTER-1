# V0028
# Audit reference: JPL Earth orbital-plane cross-track and second-derivative linear-regression analysis with normalized 2x display.
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

VERSION = "V0028"
LOCAL_TZ = ZoneInfo("America/Bogota")
EARTH_ID = "399"
CENTER = "@10"
REFPLANE = "ecliptic"
ABERRATIONS = "geometric"
AU_KM = 149597870.700
START_YEAR = 3000
STOP_YEAR = 4500
DERIVATIVE_WINDOW = 101
DERIVATIVE_DEGREE = 3
DISPLAY_EXAGGERATION = 2.0
MAX_RETRIES = 4
RETRY_DELAY_SECONDS = 3.0

OUT = Path("/content/EARTH_JPL_CROSS_TRACK_SECOND_DERIVATIVE_REGRESSION_V0028_OUTPUT")
DATA_CSV = OUT / "EARTH_JPL_CROSS_TRACK_SECOND_DERIVATIVE_DATA_V0028.csv"
FIT_CSV = OUT / "EARTH_JPL_CROSS_TRACK_SECOND_DERIVATIVE_LINEAR_REGRESSION_V0028.csv"
PNG_RAW = OUT / "EARTH_JPL_CROSS_TRACK_SECOND_DERIVATIVE_RAW_V0028.png"
PNG_NORMALIZED = OUT / "EARTH_JPL_CROSS_TRACK_SECOND_DERIVATIVE_NORMALIZED_2X_V0028.png"
PNG_REGRESSION = OUT / "EARTH_JPL_CROSS_TRACK_SECOND_DERIVATIVE_LINEAR_FITS_V0028.png"
PNG_RESIDUALS = OUT / "EARTH_JPL_CROSS_TRACK_SECOND_DERIVATIVE_RESIDUALS_V0028.png"
PNG_TABLE = OUT / "EARTH_JPL_CROSS_TRACK_SECOND_DERIVATIVE_RESULTS_TABLE_V0028.png"
CACHE_CSV = OUT / "EARTH_JPL_ANNUAL_STATES_3000_4500_V0028.csv"


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("REJECTED invalid vector norm")
    return vector / norm


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise RuntimeError("REJECTED invalid vector-row norm")
    return vectors / norms[:, None]


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
                raise RuntimeError(f"REJECTED insufficient annual rows: {len(frame)}")
            frame.insert(0, "astronomical_year", START_YEAR + np.arange(len(frame), dtype=float))
            frame.to_csv(CACHE_CSV, index=False, float_format="%.12f")
            return frame
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * attempt)
    raise RuntimeError(f"JPL annual query failed after {MAX_RETRIES} attempts: {last_error}")


def local_derivatives(years: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = np.full_like(values, np.nan, dtype=float)
    second = np.full_like(values, np.nan, dtype=float)
    half = DERIVATIVE_WINDOW // 2
    for index in range(half, len(values) - half):
        local_x = years[index - half:index + half + 1] - years[index]
        local_y = values[index - half:index + half + 1]
        coefficients = np.polyfit(local_x, local_y, DERIVATIVE_DEGREE)
        polynomial = np.poly1d(coefficients)
        first[index] = float(np.polyder(polynomial, 1)(0.0))
        second[index] = float(np.polyder(polynomial, 2)(0.0))
    return first, second


def derive_signals(states: pd.DataFrame) -> pd.DataFrame:
    r = states[["x", "y", "z"]].to_numpy(dtype=float) * AU_KM
    v = states[["vx", "vy", "vz"]].to_numpy(dtype=float) * AU_KM
    normals = normalize_rows(np.cross(r, v))
    normals[normals[:, 2] < 0.0] *= -1.0

    years = states["astronomical_year"].to_numpy(dtype=float)
    n0 = normalize(normals[0])
    n1 = normalize(normals[-1])
    u = normalize(n1 - np.dot(n1, n0) * n0)
    w = normalize(np.cross(n0, u))

    along_deg = np.degrees(np.arctan2(normals @ u, normals @ n0))
    cross_deg = np.degrees(np.arcsin(np.clip(normals @ w, -1.0, 1.0)))
    first_deg_per_year, second_deg_per_year2 = local_derivatives(years, along_deg)

    result = pd.DataFrame({
        "astronomical_year": years,
        "signed_along_track_deg": along_deg,
        "cross_track_deg": cross_deg,
        "first_derivative_arcsec_per_century": first_deg_per_year * 360000.0,
        "second_derivative_arcsec_per_century2": second_deg_per_year2 * 36000000.0,
    })

    for source, target in [
        ("cross_track_deg", "cross_track_normalized_2x"),
        ("second_derivative_arcsec_per_century2", "second_derivative_normalized_2x"),
    ]:
        values = result[source].to_numpy(dtype=float)
        finite = np.isfinite(values)
        centered = values - np.nanmean(values)
        peak = float(np.nanmax(np.abs(centered[finite])))
        if not np.isfinite(peak) or peak <= 0.0:
            raise RuntimeError(f"REJECTED normalization peak for {source}")
        result[target] = DISPLAY_EXAGGERATION * centered / peak

    return result


def linear_fit(years: np.ndarray, values: np.ndarray, signal_name: str, units: str) -> tuple[dict, np.ndarray]:
    finite = np.isfinite(years) & np.isfinite(values)
    x = years[finite]
    y = values[finite]
    x0 = float(np.mean(x))
    xs = x - x0
    slope, intercept_centered = np.polyfit(xs, y, 1)
    prediction = slope * xs + intercept_centered
    residual = y - prediction
    rss = float(np.sum(residual ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0.0 else float("nan")
    rmse = math.sqrt(rss / len(y))
    mae = float(np.mean(np.abs(residual)))
    correlation = float(np.corrcoef(x, y)[0, 1])
    full_prediction = np.full_like(values, np.nan, dtype=float)
    full_prediction[finite] = prediction
    intercept_absolute = intercept_centered - slope * x0
    row = {
        "signal": signal_name,
        "units": units,
        "sample_count": int(len(y)),
        "slope_per_year": float(slope),
        "slope_per_century": float(slope * 100.0),
        "intercept_absolute": float(intercept_absolute),
        "r2": float(r2),
        "correlation": correlation,
        "rmse": rmse,
        "mae": mae,
        "max_abs_residual": float(np.max(np.abs(residual))),
    }
    return row, full_prediction


def style_axis(ax) -> None:
    ax.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.5)
    ax.grid(True, linewidth=0.35, alpha=0.22)


def finish_legend(ax, columns: int = 1) -> None:
    legend = ax.legend(frameon=False, fontsize=9, ncol=columns)
    for text in legend.get_texts():
        text.set_color("white")


def plot_raw(data: pd.DataFrame) -> None:
    years = data["astronomical_year"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(17, 8.5), dpi=140)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(years, data["cross_track_deg"], linewidth=0.75, label="Cross-track displacement (deg)")
    ax.plot(years, data["second_derivative_arcsec_per_century2"], linewidth=0.55, label="Second derivative (arcsec/century²)")
    ax.axhline(0.0, linewidth=0.45, alpha=0.7)
    ax.set_title("Earth Orbital-Plane Cross-Track and Second-Derivative Signals\nJPL annual vectors, A.D. 3000–4500", color="white", fontsize=14, weight="bold", pad=16)
    ax.set_xlabel("Astronomical year", color="white")
    ax.set_ylabel("Native signal units", color="white")
    finish_legend(ax)
    fig.tight_layout()
    fig.savefig(PNG_RAW, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_normalized(data: pd.DataFrame) -> None:
    years = data["astronomical_year"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(17, 8.5), dpi=140)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(years, data["cross_track_normalized_2x"], linewidth=0.85, label="Cross-track normalized ×2")
    ax.plot(years, data["second_derivative_normalized_2x"], linewidth=0.65, label="Second derivative normalized ×2")
    ax.axhline(0.0, linewidth=0.45, alpha=0.7)
    ax.set_title("Normalized and 2× Exaggerated JPL Orbital-Plane Signals\nZero mean; each signal scaled to ±2 peak display amplitude", color="white", fontsize=14, weight="bold", pad=16)
    ax.set_xlabel("Astronomical year", color="white")
    ax.set_ylabel("Normalized display amplitude", color="white")
    finish_legend(ax)
    fig.tight_layout()
    fig.savefig(PNG_NORMALIZED, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_regressions(data: pd.DataFrame, predictions: dict[str, np.ndarray]) -> None:
    years = data["astronomical_year"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(17, 8.5), dpi=140)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(years, data["cross_track_normalized_2x"], linewidth=0.75, label="Cross-track normalized ×2")
    ax.plot(years, predictions["Cross-track displacement normalized ×2"], linewidth=1.05, linestyle="--", label="Cross-track linear fit")
    ax.plot(years, data["second_derivative_normalized_2x"], linewidth=0.55, label="Second derivative normalized ×2")
    ax.plot(years, predictions["Second derivative normalized ×2"], linewidth=1.05, linestyle="--", label="Second-derivative linear fit")
    ax.axhline(0.0, linewidth=0.45, alpha=0.7)
    ax.set_title("Linear Regression of Normalized 2× JPL Signals", color="white", fontsize=14, weight="bold", pad=16)
    ax.set_xlabel("Astronomical year", color="white")
    ax.set_ylabel("Normalized display amplitude", color="white")
    finish_legend(ax, columns=2)
    fig.tight_layout()
    fig.savefig(PNG_REGRESSION, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_residuals(data: pd.DataFrame, predictions: dict[str, np.ndarray]) -> None:
    years = data["astronomical_year"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(17, 8.5), dpi=140)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    cross_residual = data["cross_track_normalized_2x"].to_numpy(dtype=float) - predictions["Cross-track displacement normalized ×2"]
    second_residual = data["second_derivative_normalized_2x"].to_numpy(dtype=float) - predictions["Second derivative normalized ×2"]
    ax.plot(years, cross_residual, linewidth=0.75, label="Cross-track residual")
    ax.plot(years, second_residual, linewidth=0.55, label="Second-derivative residual")
    ax.axhline(0.0, linewidth=0.45, alpha=0.7)
    ax.set_title("Residual Structure After Linear Regression", color="white", fontsize=14, weight="bold", pad=16)
    ax.set_xlabel("Astronomical year", color="white")
    ax.set_ylabel("Normalized residual", color="white")
    finish_legend(ax)
    fig.tight_layout()
    fig.savefig(PNG_RESIDUALS, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_table(table: pd.DataFrame) -> None:
    shown = table[["signal", "slope_per_century", "r2", "correlation", "rmse", "mae", "max_abs_residual"]].copy()
    for column in ["slope_per_century", "r2", "correlation", "rmse", "mae", "max_abs_residual"]:
        shown[column] = shown[column].map(lambda value: f"{value:.9f}")
    fig, ax = plt.subplots(figsize=(18, 4.7), dpi=160)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.axis("off")
    table_artist = ax.table(cellText=shown.values, colLabels=shown.columns, cellLoc="center", loc="center")
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(8.8)
    table_artist.scale(1.0, 1.7)
    for (row, column), cell in table_artist.get_celld().items():
        cell.set_edgecolor("#777777")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#17324d")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#1b1b1b" if row % 2 else "#252525")
            cell.get_text().set_color("white")
    ax.set_title("Earth JPL Cross-Track and Second-Derivative Linear Regression Results", color="white", fontsize=15, weight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(PNG_TABLE, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"OUTPUT VERSION {VERSION}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    states = query_states()
    data = derive_signals(states)

    fits: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    definitions = [
        ("Cross-track displacement", "cross_track_deg", "degrees"),
        ("Second derivative", "second_derivative_arcsec_per_century2", "arcsec/century^2"),
        ("Cross-track displacement normalized ×2", "cross_track_normalized_2x", "normalized"),
        ("Second derivative normalized ×2", "second_derivative_normalized_2x", "normalized"),
    ]
    years = data["astronomical_year"].to_numpy(dtype=float)
    for name, column, units in definitions:
        row, prediction = linear_fit(years, data[column].to_numpy(dtype=float), name, units)
        fits.append(row)
        predictions[name] = prediction

    fit_table = pd.DataFrame(fits)
    data.to_csv(DATA_CSV, index=False, float_format="%.12f")
    fit_table.to_csv(FIT_CSV, index=False, float_format="%.12f")

    plot_raw(data)
    plot_normalized(data)
    plot_regressions(data, predictions)
    plot_residuals(data, predictions)
    plot_table(fit_table)

    display(Image(filename=str(PNG_RAW)))
    display(Image(filename=str(PNG_NORMALIZED)))
    display(Image(filename=str(PNG_REGRESSION)))
    display(Image(filename=str(PNG_RESIDUALS)))
    display(Image(filename=str(PNG_TABLE)))
    print(fit_table.to_string(index=False))
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0028