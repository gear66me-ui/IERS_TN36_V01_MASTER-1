# V0032
# Audit reference: JPL Earth-Moon barycenter orbital-plane path and precession-speed audit, A.D. 3000-4500.
from __future__ import annotations

import base64
import contextlib
import importlib.util
import io
import json
import math
import os
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
    ("requests", "requests"),
    ("IPython", "ipython"),
]:
    need(module_name, package_name)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display

VERSION = "V0032"
LOCAL_TZ = ZoneInfo("America/Bogota")
TARGET_ID = "3"
TARGET_NAME = "Earth-Moon barycenter"
CENTER = "@10"
REFPLANE = "ecliptic"
ABERRATIONS = "geometric"
AU_KM = 149597870.700
START_YEAR = 3000
STOP_YEAR = 4500
STEP = "1y"
SMOOTH_WINDOW_YEARS = 201
SMOOTH_DEGREE = 3
DISPLAY_STEP_YEARS = 10
EVENT_SEARCH_START = 3400
EVENT_SEARCH_STOP = 4100
MAX_RETRIES = 4
RETRY_DELAY_SECONDS = 3.0
REPO = "gear66me-ui/IERS_TN36_V01_MASTER-1"
BRANCH = "main"
REMOTE_DATA_PATH = "results/EARTH_MOON_BARYCENTER_PRECESSION_SPEED_CURVATURE_V0032.csv"
REMOTE_SUMMARY_PATH = "results/EARTH_MOON_BARYCENTER_PRECESSION_SHIFT_SUMMARY_V0032.csv"

OUT = Path("/content/EARTH_MOON_BARYCENTER_PRECESSION_SPEED_AUDIT_V0032_OUTPUT")
DATA_CSV = OUT / "EARTH_MOON_BARYCENTER_PRECESSION_SPEED_CURVATURE_V0032.csv"
SUMMARY_CSV = OUT / "EARTH_MOON_BARYCENTER_PRECESSION_SHIFT_SUMMARY_V0032.csv"
CACHE_CSV = OUT / "EARTH_MOON_BARYCENTER_ANNUAL_STATES_3000_4500_V0032.csv"
PNG_PATH = OUT / "EARTH_MOON_BARYCENTER_ORBITAL_NORMAL_PATH_V0032.png"
PNG_SPEED = OUT / "EARTH_MOON_BARYCENTER_PRECESSION_SPEED_COMPONENTS_V0032.png"
PNG_TABLE = OUT / "EARTH_MOON_BARYCENTER_PRECESSION_RESULTS_V0032.png"


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
        required = {"astronomical_year", "x", "y", "z", "vx", "vy", "vz"}
        if required.issubset(cached.columns) and len(cached) >= 1490:
            return cached.sort_values("astronomical_year").reset_index(drop=True)
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                table = Horizons(
                    id=TARGET_ID,
                    id_type="majorbody",
                    location=CENTER,
                    epochs={"start": f"{START_YEAR}-01-01", "stop": f"{STOP_YEAR}-01-01", "step": STEP},
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
    raise RuntimeError(f"JPL query failed after {MAX_RETRIES} attempts: {last_error}")


def local_polynomial_series(years: np.ndarray, values: np.ndarray, window: int, degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def safe_unwrapped_tangent(cross_rate: np.ndarray, along_rate: np.ndarray) -> np.ndarray:
    result = np.full_like(cross_rate, np.nan, dtype=float)
    finite = np.isfinite(cross_rate) & np.isfinite(along_rate)
    if finite.any():
        radians = np.arctan2(cross_rate[finite], along_rate[finite])
        result[finite] = np.degrees(np.unwrap(radians))
    return result


def derive_geometry(states: pd.DataFrame) -> pd.DataFrame:
    years = states["astronomical_year"].to_numpy(dtype=float)
    r = states[["x", "y", "z"]].to_numpy(dtype=float) * AU_KM
    v = states[["vx", "vy", "vz"]].to_numpy(dtype=float) * AU_KM
    normals = normalize_rows(np.cross(r, v))
    normals[normals[:, 2] < 0.0] *= -1.0
    n0 = normalize(normals[0])
    n1 = normalize(normals[-1])
    along_axis = normalize(n1 - np.dot(n1, n0) * n0)
    cross_axis = normalize(np.cross(n0, along_axis))
    along = np.degrees(np.arctan2(normals @ along_axis, normals @ n0))
    cross = np.degrees(np.arcsin(np.clip(normals @ cross_axis, -1.0, 1.0)))
    along_s, along_1, along_2 = local_polynomial_series(years, along, SMOOTH_WINDOW_YEARS, SMOOTH_DEGREE)
    cross_s, cross_1, cross_2 = local_polynomial_series(years, cross, SMOOTH_WINDOW_YEARS, SMOOTH_DEGREE)
    total_speed = np.sqrt(along_1 ** 2 + cross_1 ** 2)
    total_speed_s, total_speed_1, _ = local_polynomial_series(years, total_speed, 101, 3)
    tangent_direction = safe_unwrapped_tangent(cross_1, along_1)
    numerator = along_1 * cross_2 - cross_1 * along_2
    denominator = np.power(along_1 ** 2 + cross_1 ** 2, 1.5)
    signed_curvature = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0.0)
    radius = np.divide(1.0, np.abs(signed_curvature), out=np.full_like(signed_curvature, np.nan), where=np.abs(signed_curvature) > 0.0)
    return pd.DataFrame({
        "astronomical_year": years,
        "normal_x": normals[:, 0],
        "normal_y": normals[:, 1],
        "normal_z": normals[:, 2],
        "along_raw_deg": along,
        "cross_raw_deg": cross,
        "along_smooth_deg": along_s,
        "cross_smooth_deg": cross_s,
        "along_speed_arcsec_per_century": along_1 * 360000.0,
        "cross_speed_arcsec_per_century": cross_1 * 360000.0,
        "total_speed_arcsec_per_century": total_speed * 360000.0,
        "total_speed_smooth_arcsec_per_century": total_speed_s * 360000.0,
        "speed_acceleration_arcsec_per_century_per_century": total_speed_1 * 36000000.0,
        "tangent_direction_deg": tangent_direction,
        "along_second_deg_per_year2": along_2,
        "cross_second_deg_per_year2": cross_2,
        "signed_curvature_per_deg": signed_curvature,
        "radius_of_curvature_deg": radius,
    })


def interpolate_zero(years: np.ndarray, values: np.ndarray, index: int) -> float:
    x0, x1 = years[index], years[index + 1]
    y0, y1 = values[index], values[index + 1]
    if y1 == y0:
        return float(x0)
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def summarize(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    years = data["astronomical_year"].to_numpy(dtype=float)
    cross_speed = data["cross_speed_arcsec_per_century"].to_numpy(dtype=float)
    curvature = data["signed_curvature_per_deg"].to_numpy(dtype=float)
    total_speed = data["total_speed_smooth_arcsec_per_century"].to_numpy(dtype=float)
    valid = np.isfinite(total_speed)
    search = valid & (years >= EVENT_SEARCH_START) & (years <= EVENT_SEARCH_STOP)
    indices = np.where(search)[0]
    if len(indices) == 0:
        raise RuntimeError("REJECTED no valid event-search samples")
    cross_candidates: list[float] = []
    curvature_candidates: list[float] = []
    for i in indices[:-1]:
        if np.isfinite(cross_speed[i:i + 2]).all() and cross_speed[i] * cross_speed[i + 1] <= 0.0:
            cross_candidates.append(interpolate_zero(years, cross_speed, i))
        if np.isfinite(curvature[i:i + 2]).all() and curvature[i] * curvature[i + 1] < 0.0:
            curvature_candidates.append(interpolate_zero(years, curvature, i))
    cross_turn = min(cross_candidates, key=lambda value: abs(value - 3760.0)) if cross_candidates else float("nan")
    curvature_flip = min(curvature_candidates, key=lambda value: abs(value - 3760.0)) if curvature_candidates else float("nan")
    min_i = indices[int(np.nanargmin(total_speed[indices]))]
    max_i = indices[int(np.nanargmax(total_speed[indices]))]
    near_i = int(np.nanargmin(np.abs(years - 3760.0)))
    speed_range_percent = (total_speed[max_i] - total_speed[min_i]) / total_speed[max_i] * 100.0
    verdict = "DIRECTION SHIFT WITHOUT MAJOR SLOWDOWN" if np.isfinite(cross_turn) and speed_range_percent < 5.0 else "RATE CHANGE REQUIRES REVIEW"
    summary = {
        "cross_track_velocity_zero_year": cross_turn,
        "signed_curvature_zero_year": curvature_flip,
        "minimum_total_speed_year": float(years[min_i]),
        "minimum_total_speed_arcsec_per_century": float(total_speed[min_i]),
        "maximum_total_speed_year": float(years[max_i]),
        "maximum_total_speed_arcsec_per_century": float(total_speed[max_i]),
        "speed_variation_percent": float(speed_range_percent),
        "speed_at_3760_arcsec_per_century": float(total_speed[near_i]),
        "tangent_direction_at_3760_deg": float(data.iloc[near_i]["tangent_direction_deg"]),
        "verdict": verdict,
    }
    rows = [
        {"quantity": "JPL target", "value": f"{TARGET_ID} | {TARGET_NAME}"},
        {"quantity": "Cross-track velocity zero", "value": f"{cross_turn:.3f} yr" if np.isfinite(cross_turn) else "NOT FOUND"},
        {"quantity": "Signed-curvature zero", "value": f"{curvature_flip:.3f} yr" if np.isfinite(curvature_flip) else "NOT FOUND"},
        {"quantity": "Minimum total speed", "value": f"{total_speed[min_i]:.6f} arcsec/century at {years[min_i]:.0f}"},
        {"quantity": "Maximum total speed", "value": f"{total_speed[max_i]:.6f} arcsec/century at {years[max_i]:.0f}"},
        {"quantity": "Speed variation", "value": f"{speed_range_percent:.6f} %"},
        {"quantity": "Speed at year 3760", "value": f"{total_speed[near_i]:.6f} arcsec/century"},
        {"quantity": "Tangent direction at 3760", "value": f"{summary['tangent_direction_at_3760_deg']:.9f} deg"},
        {"quantity": "Verdict", "value": verdict},
    ]
    return pd.DataFrame(rows), summary


def style_axis(ax) -> None:
    ax.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.5)
    ax.grid(True, linewidth=0.35, alpha=0.22)


def plot_path(data: pd.DataFrame, summary: dict) -> None:
    valid = np.isfinite(data["along_smooth_deg"]) & np.isfinite(data["cross_smooth_deg"])
    sample = data.loc[valid].iloc[::DISPLAY_STEP_YEARS]
    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(sample["along_smooth_deg"], sample["cross_smooth_deg"] * 2.0, linewidth=0.9)
    for target in [3100, 3400, 3760, 4100, 4400]:
        idx = int(np.argmin(np.abs(sample["astronomical_year"].to_numpy(dtype=float) - target)))
        row = sample.iloc[idx]
        ax.scatter([row["along_smooth_deg"]], [row["cross_smooth_deg"] * 2.0], s=18)
        ax.text(row["along_smooth_deg"], row["cross_smooth_deg"] * 2.0, f" {int(row['astronomical_year'])}", color="white", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Earth-Moon Barycenter JPL Orbital-Plane Normal Path\nCross-track physically exaggerated ×2", color="white", fontsize=14, weight="bold", pad=14)
    ax.set_xlabel("Signed along-track displacement (degrees)", color="white")
    ax.set_ylabel("Cross-track displacement ×2 (degrees)", color="white")
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_speed(data: pd.DataFrame, summary: dict) -> None:
    sample = data.iloc[::DISPLAY_STEP_YEARS]
    fig, ax = plt.subplots(figsize=(17, 8.5), dpi=145)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(sample["astronomical_year"], sample["along_speed_arcsec_per_century"], linewidth=0.75, label="Along-track speed")
    ax.plot(sample["astronomical_year"], sample["cross_speed_arcsec_per_century"], linewidth=0.75, label="Cross-track speed")
    ax.plot(sample["astronomical_year"], sample["total_speed_smooth_arcsec_per_century"], linewidth=1.05, label="Total plane-normal speed")
    if np.isfinite(summary["cross_track_velocity_zero_year"]):
        ax.axvline(summary["cross_track_velocity_zero_year"], linewidth=0.7, linestyle="--", alpha=0.85)
    ax.set_title(
        "Earth-Moon Barycenter Orbital-Plane Precession Speed Components\n"
        f"Cross-track turning year {summary['cross_track_velocity_zero_year']:.3f} | Total-speed range {summary['speed_variation_percent']:.6f}%",
        color="white", fontsize=14, weight="bold", pad=14,
    )
    ax.set_xlabel("Astronomical year", color="white")
    ax.set_ylabel("Angular rate (arcsec/century)", color="white")
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()
    fig.savefig(PNG_SPEED, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_table(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(15, 6.0), dpi=160)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.axis("off")
    rendered = ax.table(cellText=table.values, colLabels=table.columns, cellLoc="left", loc="center", colWidths=[0.35, 0.65])
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(10)
    rendered.scale(1.0, 1.7)
    for (row, column), cell in rendered.get_celld().items():
        cell.set_edgecolor("#777777")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#17324d")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#1b1b1b" if row % 2 else "#252525")
            cell.get_text().set_color("white")
            if row == len(table):
                cell.get_text().set_weight("bold")
    ax.set_title("Earth-Moon Barycenter Precession Audit Results", color="white", fontsize=15, weight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(PNG_TABLE, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        return token
    try:
        from google.colab import userdata
        token = (userdata.get("GITHUB_TOKEN") or "").strip()
    except Exception:
        token = ""
    return token


def upload_file(local_path: Path, remote_path: str, token: str) -> tuple[str, str]:
    api_url = f"https://api.github.com/repos/{REPO}/contents/{remote_path}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    existing_sha = None
    get_response = requests.get(api_url, headers=headers, params={"ref": BRANCH}, timeout=60)
    if get_response.status_code == 200:
        existing_sha = get_response.json().get("sha")
    elif get_response.status_code not in (404,):
        raise RuntimeError(f"GitHub lookup failed {get_response.status_code}: {get_response.text[:300]}")
    payload = {
        "message": f"Publish {local_path.name} from {VERSION}",
        "content": base64.b64encode(local_path.read_bytes()).decode("ascii"),
        "branch": BRANCH,
    }
    if existing_sha:
        payload["sha"] = existing_sha
    put_response = requests.put(api_url, headers=headers, data=json.dumps(payload), timeout=120)
    if put_response.status_code not in (200, 201):
        raise RuntimeError(f"GitHub upload failed {put_response.status_code}: {put_response.text[:500]}")
    raw_url = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{remote_path}"
    github_url = f"https://github.com/{REPO}/blob/{BRANCH}/{remote_path}"
    return raw_url, github_url


def main() -> None:
    print(f"OUTPUT VERSION {VERSION}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    states = query_states()
    data = derive_geometry(states)
    table, summary = summarize(data)
    data.to_csv(DATA_CSV, index=False, float_format="%.12f")
    table.to_csv(SUMMARY_CSV, index=False)
    plot_path(data, summary)
    plot_speed(data, summary)
    plot_table(table)
    display(Image(filename=str(PNG_PATH)))
    display(Image(filename=str(PNG_SPEED)))
    display(Image(filename=str(PNG_TABLE)))
    print(table.to_string(index=False))
    token = github_token()
    if token:
        data_raw, data_github = upload_file(DATA_CSV, REMOTE_DATA_PATH, token)
        summary_raw, summary_github = upload_file(SUMMARY_CSV, REMOTE_SUMMARY_PATH, token)
        print("FULL CSV RAW HTTPS:")
        print(data_raw)
        print("FULL CSV GITHUB:")
        print(data_github)
        print("SUMMARY CSV RAW HTTPS:")
        print(summary_raw)
        print("SUMMARY CSV GITHUB:")
        print(summary_github)
    else:
        print("REJECTED GitHub upload: GITHUB_TOKEN not available")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0032