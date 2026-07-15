# V0023
# Audit reference: JPL Earth orbital-plane signed displacement regression over astronomical years -9000 through +9000.
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
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


for module_name, package_name in [
    ("astroquery", "astroquery"), ("astropy", "astropy"), ("numpy", "numpy"),
    ("pandas", "pandas"), ("matplotlib", "matplotlib"), ("IPython", "ipython")]:
    need(module_name, package_name)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display

VERSION = "V0023"
LOCAL_TZ = ZoneInfo("America/Bogota")
EARTH_ID = "399"
CENTER = "@10"
REFPLANE = "ecliptic"
ABERRATIONS = "geometric"
AU_KM = 149597870.700
DAYS_PER_JULIAN_YEAR = 365.25
HALF_SPAN_YEARS = 9000
CHUNK_YEARS = 100
STEP = "30d"
MAX_RETRIES = 4
RETRY_DELAY_SECONDS = 4.0
OUT = Path("/content/EARTH_JPL_SIGNED_PLANE_REGRESSION_18000_YEAR_V0023_OUTPUT")
RAW_CSV = OUT / "EARTH_JPL_MONTHLY_NORMALS_18000_YEAR_V0023.csv"
ANNUAL_CSV = OUT / "EARTH_JPL_ANNUAL_SIGNED_DISPLACEMENT_V0023.csv"
FIT_CSV = OUT / "EARTH_JPL_REGRESSION_COMPARISON_V0023.csv"
PNG_FITS = OUT / "EARTH_JPL_SIGNED_DISPLACEMENT_REGRESSIONS_V0023.png"
PNG_RESIDUALS = OUT / "EARTH_JPL_REGRESSION_RESIDUALS_V0023.png"
PNG_TABLE = OUT / "EARTH_JPL_REGRESSION_RESULTS_TABLE_V0023.png"


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


def query_chunk(start_jd: float, stop_jd: float) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                table = Horizons(
                    id=EARTH_ID, id_type="majorbody", location=CENTER,
                    epochs={"start": f"JD{start_jd:.9f}", "stop": f"JD{stop_jd:.9f}", "step": STEP},
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
    raise RuntimeError(f"JPL query failed after {MAX_RETRIES} attempts: {last_error}")


def download_monthly(center_jd: float) -> pd.DataFrame:
    start_jd = center_jd - HALF_SPAN_YEARS * DAYS_PER_JULIAN_YEAR
    stop_jd = center_jd + HALF_SPAN_YEARS * DAYS_PER_JULIAN_YEAR
    frames: list[pd.DataFrame] = []
    left = start_jd
    chunk_days = CHUNK_YEARS * DAYS_PER_JULIAN_YEAR
    while left < stop_jd:
        right = min(left + chunk_days, stop_jd)
        frames.append(query_chunk(left, right))
        left = right + 1.0e-6
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates(subset=["datetime_jd"]).sort_values("datetime_jd").reset_index(drop=True)
    if len(frame) < 210000:
        raise RuntimeError(f"REJECTED insufficient JPL samples: {len(frame)}")
    return frame


def load_or_download(center_jd: float) -> pd.DataFrame:
    if RAW_CSV.exists():
        cached = pd.read_csv(RAW_CSV)
        required = {"jd_tdb", "relative_year", "annual_index", "nx", "ny", "nz"}
        if required.issubset(cached.columns) and len(cached) >= 210000:
            return cached
    raw = download_monthly(center_jd)
    r = raw[["x", "y", "z"]].to_numpy(float) * AU_KM
    v = raw[["vx", "vy", "vz"]].to_numpy(float) * AU_KM
    normals = normalize_rows(np.cross(r, v))
    normals[normals[:, 2] < 0.0] *= -1.0
    jd = raw["datetime_jd"].to_numpy(float)
    relative_year = (jd - center_jd) / DAYS_PER_JULIAN_YEAR
    monthly = pd.DataFrame({
        "jd_tdb": jd, "relative_year": relative_year,
        "annual_index": np.rint(relative_year).astype(int),
        "nx": normals[:, 0], "ny": normals[:, 1], "nz": normals[:, 2],
    })
    monthly.to_csv(RAW_CSV, index=False, float_format="%.12f")
    return monthly


def annualize(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for year, group in monthly.groupby("annual_index", sort=True):
        if -HALF_SPAN_YEARS <= year <= HALF_SPAN_YEARS:
            normal = normalize(group[["nx", "ny", "nz"]].to_numpy(float).mean(axis=0))
            if normal[2] < 0.0:
                normal *= -1.0
            rows.append({"relative_year": int(year), "sample_count": int(len(group)),
                         "nx": normal[0], "ny": normal[1], "nz": normal[2]})
    annual = pd.DataFrame(rows).sort_values("relative_year").reset_index(drop=True)
    if len(annual) < 17800:
        raise RuntimeError(f"REJECTED insufficient annual planes: {len(annual)}")
    return annual


def signed_displacement(annual: pd.DataFrame) -> pd.DataFrame:
    years = annual["relative_year"].to_numpy(float)
    normals = annual[["nx", "ny", "nz"]].to_numpy(float)
    i0 = int(np.argmin(np.abs(years + HALF_SPAN_YEARS)))
    i1 = int(np.argmin(np.abs(years - HALF_SPAN_YEARS)))
    n0 = normalize(normals[i0])
    n1 = normalize(normals[i1])
    u = normalize(n1 - np.dot(n1, n0) * n0)
    v = normalize(np.cross(n0, u))
    along = np.degrees(np.arctan2(normals @ u, normals @ n0))
    cross = np.degrees(np.arcsin(np.clip(normals @ v, -1.0, 1.0)))
    result = annual.copy()
    result["signed_along_track_deg"] = along
    result["cross_track_deg"] = cross
    return result


def metrics(y: np.ndarray, pred: np.ndarray, k: int) -> dict:
    residual = y - pred
    n = len(y)
    sse = float(np.sum(residual ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst
    adjusted = 1.0 - (1.0 - r2) * (n - 1) / max(1, n - k - 1)
    rmse = math.sqrt(sse / n)
    mae = float(np.mean(np.abs(residual)))
    maximum = float(np.max(np.abs(residual)))
    safe_sse = max(sse, np.finfo(float).tiny)
    aic = n * math.log(safe_sse / n) + 2.0 * k
    bic = n * math.log(safe_sse / n) + k * math.log(n)
    return {"r2": r2, "adjusted_r2": adjusted, "rmse_deg": rmse, "mae_deg": mae,
            "max_abs_residual_deg": maximum, "aic": aic, "bic": bic, "sse": sse}


def fit_models(years: np.ndarray, y: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    x = years / 1000.0
    predictions: dict[str, np.ndarray] = {}
    equations: dict[str, str] = {}
    parameter_counts: dict[str, int] = {}

    for name, degree in [("Linear", 1), ("Quadratic / parabolic", 2), ("Cubic", 3)]:
        coeff = np.polyfit(x, y, degree)
        predictions[name] = np.polyval(coeff, x)
        equations[name] = " + ".join(f"{c:+.12e}x^{degree-i}" for i, c in enumerate(coeff))
        parameter_counts[name] = degree + 1

    shifted = years - np.min(years) + 1.0
    log_x = np.log(shifted)
    design = np.column_stack([np.ones_like(log_x), log_x])
    coeff_log, *_ = np.linalg.lstsq(design, y, rcond=None)
    predictions["Shifted logarithmic"] = design @ coeff_log
    equations["Shifted logarithmic"] = f"{coeff_log[0]:+.12e} {coeff_log[1]:+.12e}*ln(year-year_min+1)"
    parameter_counts["Shifted logarithmic"] = 2

    rows: list[dict] = []
    for name, pred in predictions.items():
        row = {"model": name, "parameters": parameter_counts[name], "equation": equations[name]}
        row.update(metrics(y, pred, parameter_counts[name]))
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["rmse_deg", "bic"]).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table, predictions


def style_axis(ax) -> None:
    ax.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.5)
    ax.grid(True, linewidth=0.35, alpha=0.22)


def plot_fits(years: np.ndarray, y: np.ndarray, predictions: dict[str, np.ndarray], fit_table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(17, 9), dpi=140)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(years, y, linewidth=0.65, label="Annual JPL signed displacement")
    for name in fit_table["model"]:
        ax.plot(years, predictions[name], linewidth=0.9, label=name)
    ax.axvline(0.0, linewidth=0.5, linestyle=":")
    ax.set_title(f"Earth Orbital-Plane Signed Displacement: −9000 to +9000 Years\nBest regression: {fit_table.iloc[0]['model']}",
                 color="white", fontsize=15, weight="bold", pad=16)
    ax.set_xlabel("Years from present JPL epoch", color="white")
    ax.set_ylabel("Signed angular displacement from −9000-year plane (degrees)", color="white")
    legend = ax.legend(frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()
    fig.savefig(PNG_FITS, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_residuals(years: np.ndarray, y: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(17, 9), dpi=140)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    for name, pred in predictions.items():
        ax.plot(years, (y - pred) * 3600.0, linewidth=0.65, label=name)
    ax.axhline(0.0, linewidth=0.5, linestyle="--")
    ax.set_title("Regression Residuals — JPL Signed Orbital-Plane Displacement", color="white", fontsize=15, weight="bold", pad=16)
    ax.set_xlabel("Years from present JPL epoch", color="white")
    ax.set_ylabel("Residual (arcseconds)", color="white")
    legend = ax.legend(frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()
    fig.savefig(PNG_RESIDUALS, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_table(fit_table: pd.DataFrame) -> None:
    shown = fit_table[["rank", "model", "r2", "adjusted_r2", "rmse_deg", "mae_deg", "max_abs_residual_deg", "aic", "bic"]].copy()
    for col in ["r2", "adjusted_r2"]:
        shown[col] = shown[col].map(lambda value: f"{value:.12f}")
    for col in ["rmse_deg", "mae_deg", "max_abs_residual_deg"]:
        shown[col] = shown[col].map(lambda value: f"{value:.9e}")
    for col in ["aic", "bic"]:
        shown[col] = shown[col].map(lambda value: f"{value:.3f}")
    fig, ax = plt.subplots(figsize=(18, 5.5), dpi=150)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.axis("off")
    table = ax.table(cellText=shown.values, colLabels=shown.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.3)
    table.scale(1.0, 1.75)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#777777")
        cell.set_linewidth(0.45)
        cell.set_text_props(color="white", weight="bold" if row == 0 or row == 1 else "normal")
        cell.set_facecolor("#202020" if row == 0 else "#173A3A" if row == 1 else "#101010")
    ax.set_title("Earth JPL Signed Orbital-Plane Regression Comparison", color="white", fontsize=15, weight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(PNG_TABLE, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"OUTPUT VERSION {VERSION}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    center_jd = float(Time(datetime.now(timezone.utc), scale="utc").tdb.jd)
    monthly = load_or_download(center_jd)
    annual = signed_displacement(annualize(monthly))
    annual.to_csv(ANNUAL_CSV, index=False, float_format="%.12f")
    years = annual["relative_year"].to_numpy(float)
    y = annual["signed_along_track_deg"].to_numpy(float)
    fit_table, predictions = fit_models(years, y)
    fit_table.to_csv(FIT_CSV, index=False, float_format="%.12f")
    plot_fits(years, y, predictions, fit_table)
    plot_residuals(years, y, predictions)
    plot_table(fit_table)
    display(Image(filename=str(PNG_FITS)))
    display(Image(filename=str(PNG_RESIDUALS)))
    display(Image(filename=str(PNG_TABLE)))
    print("REGRESSION RESULTS")
    print(fit_table[["rank", "model", "r2", "adjusted_r2", "rmse_deg", "mae_deg", "max_abs_residual_deg", "aic", "bic"]].to_string(index=False))
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0023