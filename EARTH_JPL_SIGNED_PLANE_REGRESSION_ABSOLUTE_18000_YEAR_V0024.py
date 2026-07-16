# V0024
# Audit reference: BCE-safe explicit-JD JPL Earth orbital-plane regression from astronomical year -9000 through +9000.
from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
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
    ("astropy", "astropy"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("IPython", "ipython"),
]:
    need(module_name, package_name)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display

VERSION = "V0024"
LOCAL_TZ = ZoneInfo("America/Bogota")
EARTH_ID = "399"
CENTER = "@10"
REFPLANE = "ecliptic"
ABERRATIONS = "geometric"
AU_KM = 149597870.700
J2000_JD = 2451545.0
DAYS_PER_JULIAN_YEAR = 365.25
START_YEAR = -9000
STOP_YEAR = 9000
YEAR_STEP = 5
QUERY_BATCH_SIZE = 64
MAX_RETRIES = 4
RETRY_DELAY_SECONDS = 4.0

OUT = Path("/content/EARTH_JPL_SIGNED_PLANE_REGRESSION_ABSOLUTE_18000_YEAR_V0024_OUTPUT")
DATA_CSV = OUT / "EARTH_JPL_SIGNED_PLANE_5YEAR_DATA_V0024.csv"
FIT_CSV = OUT / "EARTH_JPL_SIGNED_PLANE_REGRESSION_TABLE_V0024.csv"
PNG_FITS = OUT / "EARTH_JPL_SIGNED_PLANE_REGRESSION_FITS_V0024.png"
PNG_RESIDUALS = OUT / "EARTH_JPL_SIGNED_PLANE_REGRESSION_RESIDUALS_V0024.png"
PNG_TABLE = OUT / "EARTH_JPL_SIGNED_PLANE_REGRESSION_TABLE_V0024.png"
CACHE_CSV = OUT / "EARTH_JPL_EXPLICIT_JD_STATES_V0024.csv"


def normalize(vector: np.ndarray) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    if not np.isfinite(magnitude) or magnitude <= 0.0:
        raise RuntimeError("REJECTED invalid vector magnitude")
    return vector / magnitude


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    magnitudes = np.linalg.norm(vectors, axis=1)
    if np.any(~np.isfinite(magnitudes)) or np.any(magnitudes <= 0.0):
        raise RuntimeError("REJECTED invalid vector row magnitude")
    return vectors / magnitudes[:, None]


def astronomical_year_to_jd(year: float) -> float:
    return J2000_JD + (float(year) - 2000.0) * DAYS_PER_JULIAN_YEAR


def query_batch(jds: np.ndarray) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                table = Horizons(
                    id=EARTH_ID,
                    id_type="majorbody",
                    location=CENTER,
                    epochs=jds.tolist(),
                ).vectors(refplane=REFPLANE, aberrations=ABERRATIONS)
            frame = table.to_pandas()
            required = ["datetime_jd", "x", "y", "z", "vx", "vy", "vz"]
            if any(column not in frame.columns for column in required):
                raise RuntimeError("REJECTED missing JPL vector columns")
            return frame[required].copy()
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * attempt)
    raise RuntimeError(f"JPL explicit-epoch query failed after {MAX_RETRIES} attempts: {last_error}")


def load_or_download_states() -> pd.DataFrame:
    years = np.arange(START_YEAR, STOP_YEAR + YEAR_STEP, YEAR_STEP, dtype=float)
    present_year = datetime.now(timezone.utc).year + (datetime.now(timezone.utc).timetuple().tm_yday - 1) / 365.25
    years = np.unique(np.concatenate([years, np.array([present_year], dtype=float)]))
    jds = np.array([astronomical_year_to_jd(year) for year in years], dtype=float)

    if CACHE_CSV.exists():
        cached = pd.read_csv(CACHE_CSV)
        required = {"astronomical_year", "datetime_jd", "x", "y", "z", "vx", "vy", "vz"}
        if required.issubset(cached.columns) and len(cached) == len(years):
            return cached.sort_values("astronomical_year").reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    for start in range(0, len(jds), QUERY_BATCH_SIZE):
        batch_jds = jds[start:start + QUERY_BATCH_SIZE]
        frames.append(query_batch(batch_jds))

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates(subset=["datetime_jd"]).sort_values("datetime_jd").reset_index(drop=True)
    if len(frame) != len(years):
        raise RuntimeError(f"REJECTED explicit-JD sample count: expected {len(years)}, received {len(frame)}")
    frame.insert(0, "astronomical_year", years)
    frame.to_csv(CACHE_CSV, index=False, float_format="%.12f")
    return frame


def derive_signed_displacement(states: pd.DataFrame) -> pd.DataFrame:
    r = states[["x", "y", "z"]].to_numpy(dtype=float) * AU_KM
    v = states[["vx", "vy", "vz"]].to_numpy(dtype=float) * AU_KM
    normals = normalize_rows(np.cross(r, v))
    normals[normals[:, 2] < 0.0] *= -1.0

    years = states["astronomical_year"].to_numpy(dtype=float)
    start_index = int(np.argmin(np.abs(years - START_YEAR)))
    stop_index = int(np.argmin(np.abs(years - STOP_YEAR)))
    n0 = normalize(normals[start_index])
    n1 = normalize(normals[stop_index])
    endpoint_tangent = n1 - np.dot(n1, n0) * n0
    u = normalize(endpoint_tangent)
    w = normalize(np.cross(n0, u))

    along = np.degrees(np.arctan2(normals @ u, normals @ n0))
    cross = np.degrees(np.arcsin(np.clip(normals @ w, -1.0, 1.0)))
    total = np.degrees(np.arccos(np.clip(normals @ n0, -1.0, 1.0)))

    result = pd.DataFrame({
        "astronomical_year": years,
        "jd_tdb": states["datetime_jd"].to_numpy(dtype=float),
        "nx": normals[:, 0],
        "ny": normals[:, 1],
        "nz": normals[:, 2],
        "signed_along_track_deg": along,
        "cross_track_deg": cross,
        "total_angle_from_start_deg": total,
    })
    return result.sort_values("astronomical_year").reset_index(drop=True)


def regression_metrics(y: np.ndarray, yhat: np.ndarray, parameter_count: int) -> dict:
    residual = y - yhat
    n = len(y)
    rss = float(np.sum(residual ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / tss
    adjusted_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - parameter_count - 1, 1)
    rmse = math.sqrt(rss / n)
    mae = float(np.mean(np.abs(residual)))
    max_abs = float(np.max(np.abs(residual)))
    sigma2 = max(rss / n, np.finfo(float).tiny)
    aic = n * math.log(sigma2) + 2.0 * parameter_count
    bic = n * math.log(sigma2) + parameter_count * math.log(n)
    return {
        "r2": r2,
        "adjusted_r2": adjusted_r2,
        "rmse_deg": rmse,
        "mae_deg": mae,
        "max_abs_residual_deg": max_abs,
        "aic": aic,
        "bic": bic,
    }


def fit_models(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    x = data["astronomical_year"].to_numpy(dtype=float)
    y = data["signed_along_track_deg"].to_numpy(dtype=float)
    x_scaled = x / 10000.0
    predictions: dict[str, np.ndarray] = {}
    rows: list[dict] = []

    model_specs = [("Linear", 1), ("Quadratic / parabolic", 2), ("Cubic", 3)]
    for name, degree in model_specs:
        coefficients = np.polyfit(x_scaled, y, degree)
        yhat = np.polyval(coefficients, x_scaled)
        predictions[name] = yhat
        row = {"model": name, "parameters": degree + 1, "equation_coefficients": np.array2string(coefficients, precision=12, separator=",")}
        row.update(regression_metrics(y, yhat, degree + 1))
        rows.append(row)

    shift = 1.0 - float(np.min(x_scaled))
    log_x = np.log(x_scaled + shift)
    design = np.column_stack([np.ones_like(log_x), log_x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    yhat = design @ coefficients
    predictions["Shifted logarithmic"] = yhat
    row = {"model": "Shifted logarithmic", "parameters": 2, "equation_coefficients": np.array2string(coefficients, precision=12, separator=",") + f"; shift={shift:.12f}"}
    row.update(regression_metrics(y, yhat, 2))
    rows.append(row)

    table = pd.DataFrame(rows).sort_values(["bic", "rmse_deg"], ascending=[True, True]).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table, predictions


def style_axis(ax) -> None:
    ax.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.5)
    ax.grid(True, linewidth=0.35, alpha=0.22)


def plot_fits(data: pd.DataFrame, table: pd.DataFrame, predictions: dict[str, np.ndarray], present_year: float) -> None:
    x = data["astronomical_year"].to_numpy(dtype=float)
    y = data["signed_along_track_deg"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(17, 9), dpi=140)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(x, y, linewidth=0.95, label="JPL signed plane displacement")
    for name in table["model"]:
        ax.plot(x, predictions[name], linewidth=0.7, linestyle="--", label=name)
    ax.axvline(present_year, linewidth=0.55, linestyle=":", alpha=0.9)
    best = str(table.iloc[0]["model"])
    ax.set_title(
        "Earth Orbital-Plane Signed Displacement — JPL Astronomical Years −9000 to +9000\n"
        f"Fixed reference: year −9000 plane   |   Best BIC model: {best}",
        color="white", fontsize=14, weight="bold", pad=16,
    )
    ax.set_xlabel("Astronomical year", color="white")
    ax.set_ylabel("Signed along-track plane displacement (degrees)", color="white")
    legend = ax.legend(frameon=False, fontsize=8.5, ncol=2)
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()
    fig.savefig(PNG_FITS, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_residuals(data: pd.DataFrame, table: pd.DataFrame, predictions: dict[str, np.ndarray], present_year: float) -> None:
    x = data["astronomical_year"].to_numpy(dtype=float)
    y = data["signed_along_track_deg"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(17, 9), dpi=140)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    for name in table["model"]:
        ax.plot(x, (y - predictions[name]) * 3600.0, linewidth=0.65, label=name)
    ax.axhline(0.0, linewidth=0.5, alpha=0.7)
    ax.axvline(present_year, linewidth=0.55, linestyle=":", alpha=0.9)
    ax.set_title("Regression Residuals for Earth JPL Signed Orbital-Plane Motion", color="white", fontsize=14, weight="bold", pad=16)
    ax.set_xlabel("Astronomical year", color="white")
    ax.set_ylabel("Residual (arcseconds)", color="white")
    legend = ax.legend(frameon=False, fontsize=8.5, ncol=2)
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()
    fig.savefig(PNG_RESIDUALS, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_table(table: pd.DataFrame) -> None:
    display_table = table[["rank", "model", "r2", "adjusted_r2", "rmse_deg", "mae_deg", "max_abs_residual_deg", "aic", "bic"]].copy()
    for column in ["r2", "adjusted_r2"]:
        display_table[column] = display_table[column].map(lambda value: f"{value:.12f}")
    for column in ["rmse_deg", "mae_deg", "max_abs_residual_deg"]:
        display_table[column] = display_table[column].map(lambda value: f"{value:.9e}")
    for column in ["aic", "bic"]:
        display_table[column] = display_table[column].map(lambda value: f"{value:.3f}")

    fig, ax = plt.subplots(figsize=(18, 4.8), dpi=160)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.axis("off")
    tbl = ax.table(cellText=display_table.values, colLabels=display_table.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.65)
    for (row, column), cell in tbl.get_celld().items():
        cell.set_edgecolor("#777777")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#17324d")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#1b1b1b" if row % 2 else "#252525")
            cell.get_text().set_color("white")
            if row == 1:
                cell.get_text().set_weight("bold")
    ax.set_title("Earth JPL Signed Orbital-Plane Regression Ranking", color="white", fontsize=15, weight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(PNG_TABLE, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"OUTPUT VERSION {VERSION}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    states = load_or_download_states()
    data = derive_signed_displacement(states)
    table, predictions = fit_models(data)
    data.to_csv(DATA_CSV, index=False, float_format="%.12f")
    table.to_csv(FIT_CSV, index=False, float_format="%.12f")
    present_year = float(data.iloc[np.argmin(np.abs(data["astronomical_year"].to_numpy(dtype=float) - datetime.now(timezone.utc).year))]["astronomical_year"])
    plot_fits(data, table, predictions, present_year)
    plot_residuals(data, table, predictions, present_year)
    plot_table(table)
    display(Image(filename=str(PNG_FITS)))
    display(Image(filename=str(PNG_RESIDUALS)))
    display(Image(filename=str(PNG_TABLE)))
    print(table[["rank", "model", "r2", "adjusted_r2", "rmse_deg", "mae_deg", "max_abs_residual_deg", "aic", "bic"]].to_string(index=False))
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0024