# V0020
# Audit reference: Earth JPL Horizons monthly state-vector orbital-plane derivative over a centered 6000-year interval.
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

VERSION = "V0020"
LOCAL_TZ = ZoneInfo("America/Bogota")
EARTH_ID = "399"
CENTER = "@10"
REFPLANE = "ecliptic"
ABERRATIONS = "geometric"
AU_KM = 149597870.700
DAYS_PER_JULIAN_YEAR = 365.25
HALF_SPAN_YEARS = 3000
CHUNK_YEARS = 100
STEP = "30d"
LOCAL_FIT_HALF_WIDTH_YEARS = 250
LOCAL_FIT_DEGREE = 3
DERIVATIVE_SMOOTH_YEARS = 101
MAX_RETRIES = 4
RETRY_DELAY_SECONDS = 4.0

OUT = Path("/content/EARTH_JPL_ORBITAL_PLANE_NEUTRAL_AXIS_DERIVATIVE_6000_YEAR_V0020_OUTPUT")
RAW_CSV = OUT / "EARTH_JPL_MONTHLY_ORBITAL_NORMALS_V0020.csv"
ANNUAL_CSV = OUT / "EARTH_JPL_ANNUAL_ORBITAL_PLANE_DERIVATIVE_V0020.csv"
PNG_TILT = OUT / "EARTH_JPL_NEUTRAL_PLANE_TILT_6000_YEAR_V0020.png"
PNG_RATE = OUT / "EARTH_JPL_NEUTRAL_PLANE_DERIVATIVE_6000_YEAR_V0020.png"


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise RuntimeError("REJECTED invalid vector norm")
    return vectors / norms[:, None]


def query_chunk(start_jd: float, stop_jd: float) -> pd.DataFrame:
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
                        "start": f"JD{start_jd:.9f}",
                        "stop": f"JD{stop_jd:.9f}",
                        "step": STEP,
                    },
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


def download_monthly_states(center_jd: float) -> pd.DataFrame:
    start_jd = center_jd - HALF_SPAN_YEARS * DAYS_PER_JULIAN_YEAR
    stop_jd = center_jd + HALF_SPAN_YEARS * DAYS_PER_JULIAN_YEAR
    chunk_days = CHUNK_YEARS * DAYS_PER_JULIAN_YEAR
    frames: list[pd.DataFrame] = []
    left = start_jd
    while left < stop_jd:
        right = min(left + chunk_days, stop_jd)
        frames.append(query_chunk(left, right))
        left = right + 1.0e-6
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates(subset=["datetime_jd"]).sort_values("datetime_jd").reset_index(drop=True)
    if len(frame) < 70000:
        raise RuntimeError(f"REJECTED insufficient monthly JPL samples: {len(frame)}")
    return frame


def build_monthly_normals(frame: pd.DataFrame, center_jd: float) -> pd.DataFrame:
    r = frame[["x", "y", "z"]].to_numpy(dtype=float) * AU_KM
    v = frame[["vx", "vy", "vz"]].to_numpy(dtype=float) * AU_KM
    h = np.cross(r, v)
    normals = normalize_rows(h)
    flip = normals[:, 2] < 0.0
    normals[flip] *= -1.0
    jd = frame["datetime_jd"].to_numpy(dtype=float)
    relative_year = (jd - center_jd) / DAYS_PER_JULIAN_YEAR
    annual_index = np.rint(relative_year).astype(int)
    result = pd.DataFrame({
        "jd_tdb": jd,
        "relative_year": relative_year,
        "annual_index": annual_index,
        "nx": normals[:, 0],
        "ny": normals[:, 1],
        "nz": normals[:, 2],
    })
    return result


def annualize(monthly: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    grouped = monthly.groupby("annual_index", sort=True)
    rows: list[dict] = []
    normals: list[np.ndarray] = []
    for year, group in grouped:
        if year < -HALF_SPAN_YEARS or year > HALF_SPAN_YEARS:
            continue
        mean_vector = group[["nx", "ny", "nz"]].to_numpy(dtype=float).mean(axis=0)
        mean_vector /= np.linalg.norm(mean_vector)
        if mean_vector[2] < 0.0:
            mean_vector *= -1.0
        rows.append({
            "relative_year": int(year),
            "sample_count": int(len(group)),
            "nx": float(mean_vector[0]),
            "ny": float(mean_vector[1]),
            "nz": float(mean_vector[2]),
        })
        normals.append(mean_vector)
    annual = pd.DataFrame(rows).sort_values("relative_year").reset_index(drop=True)
    normal_array = np.vstack(normals)
    if len(annual) < 5900:
        raise RuntimeError(f"REJECTED insufficient annual planes: {len(annual)}")
    mean_normal = normal_array.mean(axis=0)
    mean_normal /= np.linalg.norm(mean_normal)
    if mean_normal[2] < 0.0:
        mean_normal *= -1.0
    return annual, mean_normal


def make_plane_basis(mean_normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([1.0, 0.0, 0.0])
    e1 = reference - np.dot(reference, mean_normal) * mean_normal
    if np.linalg.norm(e1) < 1.0e-8:
        reference = np.array([0.0, 1.0, 0.0])
        e1 = reference - np.dot(reference, mean_normal) * mean_normal
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(mean_normal, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def centered_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def derive_angles_and_rates(annual: pd.DataFrame, mean_normal: np.ndarray) -> tuple[pd.DataFrame, dict]:
    normals = annual[["nx", "ny", "nz"]].to_numpy(dtype=float)
    years = annual["relative_year"].to_numpy(dtype=float)
    e1, e2 = make_plane_basis(mean_normal)

    dot_mean = np.clip(normals @ mean_normal, -1.0, 1.0)
    tilt_deg = np.degrees(np.arccos(dot_mean))
    p = normals @ e1
    q = normals @ e2
    node_deg = np.degrees(np.unwrap(np.arctan2(q, p)))

    raw_rate = np.gradient(tilt_deg, years)
    smooth_rate = centered_moving_average(raw_rate, DERIVATIVE_SMOOTH_YEARS)
    node_rate = centered_moving_average(np.gradient(node_deg, years), DERIVATIVE_SMOOTH_YEARS)

    local_mask = np.abs(years) <= LOCAL_FIT_HALF_WIDTH_YEARS
    local_years = years[local_mask]
    local_tilt = tilt_deg[local_mask]
    local_node = node_deg[local_mask]
    tilt_coeff = np.polyfit(local_years, local_tilt, LOCAL_FIT_DEGREE)
    node_coeff = np.polyfit(local_years, local_node, LOCAL_FIT_DEGREE)
    tilt_poly = np.poly1d(tilt_coeff)
    node_poly = np.poly1d(node_coeff)
    tilt_derivative = np.polyder(tilt_poly)
    node_derivative = np.polyder(node_poly)

    current_tilt = float(tilt_poly(0.0))
    current_tilt_rate = float(tilt_derivative(0.0))
    current_node = float(node_poly(0.0))
    current_node_rate = float(node_derivative(0.0))
    state = "UPSWING" if current_tilt_rate > 0.0 else "DOWNSWING" if current_tilt_rate < 0.0 else "TURNING POINT"

    annual = annual.copy()
    annual["tilt_from_interval_mean_deg"] = tilt_deg
    annual["node_angle_unwrapped_deg"] = node_deg
    annual["tilt_rate_raw_deg_per_year"] = raw_rate
    annual["tilt_rate_smoothed_deg_per_year"] = smooth_rate
    annual["node_rate_smoothed_deg_per_year"] = node_rate

    result = {
        "current_tilt_deg": current_tilt,
        "current_tilt_rate_deg_per_year": current_tilt_rate,
        "current_tilt_rate_arcsec_per_century": current_tilt_rate * 3600.0 * 100.0,
        "current_node_deg": current_node,
        "current_node_rate_deg_per_year": current_node_rate,
        "current_node_rate_arcsec_per_century": current_node_rate * 3600.0 * 100.0,
        "state": state,
        "tilt_poly": tilt_poly,
        "mean_normal": mean_normal,
    }
    return annual, result


def style_axis(ax) -> None:
    ax.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.5)
    ax.grid(True, linewidth=0.35, alpha=0.22)


def plot_tilt(annual: pd.DataFrame, result: dict) -> None:
    years = annual["relative_year"].to_numpy(dtype=float)
    tilt = annual["tilt_from_interval_mean_deg"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(16, 8), dpi=140)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(years, tilt, linewidth=0.55, label="Annual JPL orbital-plane tilt")
    ax.axvline(0.0, linewidth=0.55, linestyle="--", alpha=0.8)
    ax.axhline(0.0, linewidth=0.45, alpha=0.6)

    local_x = np.linspace(-LOCAL_FIT_HALF_WIDTH_YEARS, LOCAL_FIT_HALF_WIDTH_YEARS, 501)
    local_y = result["tilt_poly"](local_x)
    ax.plot(local_x, local_y, linewidth=1.1, label=f"Local degree-{LOCAL_FIT_DEGREE} fit")
    ax.scatter([0.0], [result["current_tilt_deg"]], s=24.0, zorder=5)

    ax.set_title(
        "Earth Orbital-Plane Oscillation Relative to the JPL 6000-Year Mean Plane\n"
        f"Current classification: {result['state']}   |   "
        f"di/dt = {result['current_tilt_rate_arcsec_per_century']:+.6f} arcsec/century",
        color="white",
        fontsize=14,
        weight="bold",
        pad=16,
    )
    ax.set_xlabel("Years from present JPL epoch", color="white")
    ax.set_ylabel("Angular distance from interval-mean plane (degrees)", color="white")
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()
    fig.savefig(PNG_TILT, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_rate(annual: pd.DataFrame, result: dict) -> None:
    years = annual["relative_year"].to_numpy(dtype=float)
    rate = annual["tilt_rate_smoothed_deg_per_year"].to_numpy(dtype=float) * 3600.0 * 100.0
    fig, ax = plt.subplots(figsize=(16, 8), dpi=140)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(years, rate, linewidth=0.65, label=f"{DERIVATIVE_SMOOTH_YEARS}-year smoothed derivative")
    ax.axhline(0.0, linewidth=0.55, linestyle="--", alpha=0.8)
    ax.axvline(0.0, linewidth=0.55, linestyle="--", alpha=0.8)
    ax.scatter([0.0], [result["current_tilt_rate_arcsec_per_century"]], s=24.0, zorder=5)
    ax.set_title(
        "Earth Orbital-Plane Inclination Derivative\n"
        f"Present result: {result['state']}   |   "
        f"di/dt = {result['current_tilt_rate_arcsec_per_century']:+.6f} arcsec/century",
        color="white",
        fontsize=14,
        weight="bold",
        pad=16,
    )
    ax.set_xlabel("Years from present JPL epoch", color="white")
    ax.set_ylabel("Tilt derivative (arcsec/century)", color="white")
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()
    fig.savefig(PNG_RATE, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"OUTPUT VERSION {VERSION}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    center_time = Time(datetime.now(timezone.utc), scale="utc").tdb
    center_jd = float(center_time.jd)

    raw = download_monthly_states(center_jd)
    monthly = build_monthly_normals(raw, center_jd)
    monthly.to_csv(RAW_CSV, index=False, float_format="%.12f")

    annual, mean_normal = annualize(monthly)
    annual, result = derive_angles_and_rates(annual, mean_normal)
    annual["current_state"] = result["state"]
    annual["current_tilt_fit_deg"] = result["current_tilt_deg"]
    annual["current_tilt_rate_arcsec_per_century"] = result["current_tilt_rate_arcsec_per_century"]
    annual["current_node_rate_arcsec_per_century"] = result["current_node_rate_arcsec_per_century"]
    annual["neutral_plane_nx"] = mean_normal[0]
    annual["neutral_plane_ny"] = mean_normal[1]
    annual["neutral_plane_nz"] = mean_normal[2]
    annual.to_csv(ANNUAL_CSV, index=False, float_format="%.12f")

    plot_tilt(annual, result)
    plot_rate(annual, result)
    display(Image(filename=str(PNG_TILT)))
    display(Image(filename=str(PNG_RATE)))

    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0020