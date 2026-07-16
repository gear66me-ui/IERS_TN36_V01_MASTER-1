# V0031
# Audit reference: JPL Earth orbital-plane precession audit with automatic GitHub CSV publication and raw HTTPS output.
from __future__ import annotations

import base64
import contextlib
import importlib.util
import io
import json
import math
import subprocess
import sys
import time
import urllib.error
import urllib.request
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

VERSION = "V0031"
LOCAL_TZ = ZoneInfo("America/Bogota")
EARTH_ID = "399"
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
GITHUB_OWNER = "gear66me-ui"
GITHUB_REPO = "IERS_TN36_V01_MASTER-1"
GITHUB_BRANCH = "main"
GITHUB_RESULTS_DIR = "results"

OUT = Path("/content/EARTH_JPL_PRECESSION_SHIFT_SPEED_CURVATURE_HTTP_CSV_V0031_OUTPUT")
DATA_CSV = OUT / "EARTH_JPL_PRECESSION_SPEED_CURVATURE_V0031.csv"
SUMMARY_CSV = OUT / "EARTH_JPL_PRECESSION_SHIFT_SUMMARY_V0031.csv"
PNG_PATH = OUT / "EARTH_JPL_ORBITAL_NORMAL_PATH_V0031.png"
PNG_SPEED = OUT / "EARTH_JPL_PRECESSION_SPEED_COMPONENTS_V0031.png"
PNG_DIRECTION = OUT / "EARTH_JPL_TANGENT_DIRECTION_CURVATURE_V0031.png"
PNG_TABLE = OUT / "EARTH_JPL_PRECESSION_SHIFT_TABLE_V0031.png"
CACHE_CSV = OUT / "EARTH_JPL_ANNUAL_STATES_3000_4500_V0031.csv"
V0030_CACHE = Path("/content/EARTH_JPL_PRECESSION_SHIFT_SPEED_CURVATURE_AUDIT_V0030_OUTPUT/EARTH_JPL_ANNUAL_STATES_3000_4500_V0030.csv")
V0027_FOCUS = Path("/content/EARTH_JPL_INFLECTION_LOCAL_INTEGRITY_AUDIT_V0027_OUTPUT/EARTH_JPL_FOCUS_ANNUAL_3000_4500_V0027.csv")


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
    required = {"astronomical_year", "x", "y", "z", "vx", "vy", "vz"}
    for candidate in [CACHE_CSV, V0030_CACHE, V0027_FOCUS]:
        if candidate.exists():
            cached = pd.read_csv(candidate)
            if required.issubset(cached.columns) and len(cached) >= 1490:
                cached = cached.sort_values("astronomical_year").reset_index(drop=True)
                cached.to_csv(CACHE_CSV, index=False, float_format="%.12f")
                return cached

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                table = Horizons(
                    id=EARTH_ID,
                    id_type="majorbody",
                    location=CENTER,
                    epochs={"start": f"{START_YEAR}-01-01", "stop": f"{STOP_YEAR}-01-01", "step": STEP},
                ).vectors(refplane=REFPLANE, aberrations=ABERRATIONS)
            frame = table.to_pandas()
            columns = ["datetime_jd", "x", "y", "z", "vx", "vy", "vz"]
            if any(column not in frame.columns for column in columns):
                raise RuntimeError("REJECTED missing JPL vector columns")
            frame = frame[columns].sort_values("datetime_jd").reset_index(drop=True)
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


def local_polynomial_series(
    years: np.ndarray,
    values: np.ndarray,
    window: int,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    speed_deg_per_year = np.sqrt(along_1 ** 2 + cross_1 ** 2)
    tangent_angle_deg = np.degrees(np.unwrap(np.arctan2(cross_1, along_1)))
    speed_s, speed_1, _ = local_polynomial_series(years, speed_deg_per_year, 101, 3)
    numerator = along_1 * cross_2 - cross_1 * along_2
    denominator = np.power(along_1 ** 2 + cross_1 ** 2, 1.5)
    signed_curvature = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0.0,
    )
    radius_deg = np.divide(
        1.0,
        np.abs(signed_curvature),
        out=np.full_like(signed_curvature, np.nan),
        where=np.abs(signed_curvature) > 0.0,
    )

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
        "total_speed_arcsec_per_century": speed_deg_per_year * 360000.0,
        "total_speed_smooth_arcsec_per_century": speed_s * 360000.0,
        "speed_acceleration_arcsec_per_century_per_century": speed_1 * 36000000.0,
        "tangent_direction_deg": tangent_angle_deg,
        "along_second_deg_per_year2": along_2,
        "cross_second_deg_per_year2": cross_2,
        "signed_curvature_per_deg": signed_curvature,
        "radius_of_curvature_deg": radius_deg,
    })


def interpolate_zero(years: np.ndarray, values: np.ndarray, index: int) -> float:
    x0, x1 = years[index], years[index + 1]
    y0, y1 = values[index], values[index + 1]
    if y1 == y0:
        return float(x0)
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def find_event_summary(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    years = data["astronomical_year"].to_numpy(dtype=float)
    total_speed = data["total_speed_smooth_arcsec_per_century"].to_numpy(dtype=float)
    valid = np.isfinite(total_speed)
    search = valid & (years >= EVENT_SEARCH_START) & (years <= EVENT_SEARCH_STOP)
    indices = np.where(search)[0]
    if len(indices) == 0:
        raise RuntimeError("REJECTED no valid event-search samples")

    cross_speed = data["cross_speed_arcsec_per_century"].to_numpy(dtype=float)
    curvature = data["signed_curvature_per_deg"].to_numpy(dtype=float)
    tangent = data["tangent_direction_deg"].to_numpy(dtype=float)
    cross_candidates: list[float] = []
    curvature_candidates: list[float] = []
    for index in indices[:-1]:
        if np.isfinite(cross_speed[index:index + 2]).all() and cross_speed[index] * cross_speed[index + 1] <= 0.0:
            cross_candidates.append(interpolate_zero(years, cross_speed, index))
        if np.isfinite(curvature[index:index + 2]).all() and curvature[index] * curvature[index + 1] < 0.0:
            curvature_candidates.append(interpolate_zero(years, curvature, index))

    cross_turn_year = min(cross_candidates, key=lambda value: abs(value - 3760.0)) if cross_candidates else float("nan")
    curvature_flip_year = min(curvature_candidates, key=lambda value: abs(value - 3760.0)) if curvature_candidates else float("nan")
    min_speed_index = indices[int(np.nanargmin(total_speed[indices]))]
    max_speed_index = indices[int(np.nanargmax(total_speed[indices]))]
    event_index = int(np.argmin(np.abs(years - 3760.0)))

    speed_range_percent = (
        (float(total_speed[max_speed_index]) - float(total_speed[min_speed_index]))
        / float(total_speed[max_speed_index])
        * 100.0
    )
    verdict = (
        "DIRECTION SHIFT WITHOUT MAJOR SLOWDOWN"
        if np.isfinite(cross_turn_year) and speed_range_percent < 5.0
        else "RATE CHANGE REQUIRES REVIEW"
    )
    summary = {
        "cross_track_turn_year": cross_turn_year,
        "curvature_sign_change_year": curvature_flip_year,
        "minimum_total_speed_year": float(years[min_speed_index]),
        "minimum_total_speed_arcsec_per_century": float(total_speed[min_speed_index]),
        "maximum_total_speed_year": float(years[max_speed_index]),
        "maximum_total_speed_arcsec_per_century": float(total_speed[max_speed_index]),
        "speed_range_percent": speed_range_percent,
        "speed_at_3760_arcsec_per_century": float(total_speed[event_index]),
        "tangent_direction_at_3760_deg": float(tangent[event_index]),
        "signed_curvature_at_3760_per_deg": float(curvature[event_index]),
        "verdict": verdict,
    }
    rows = [
        {"quantity": "Cross-track velocity zero", "value": f"{cross_turn_year:.3f} yr" if np.isfinite(cross_turn_year) else "NOT FOUND"},
        {"quantity": "Signed-curvature zero", "value": f"{curvature_flip_year:.3f} yr" if np.isfinite(curvature_flip_year) else "NOT FOUND"},
        {"quantity": "Minimum total speed", "value": f"{summary['minimum_total_speed_arcsec_per_century']:.6f} arcsec/century at {summary['minimum_total_speed_year']:.0f}"},
        {"quantity": "Maximum total speed", "value": f"{summary['maximum_total_speed_arcsec_per_century']:.6f} arcsec/century at {summary['maximum_total_speed_year']:.0f}"},
        {"quantity": "Speed variation", "value": f"{speed_range_percent:.6f} %"},
        {"quantity": "Speed at year 3760", "value": f"{summary['speed_at_3760_arcsec_per_century']:.6f} arcsec/century"},
        {"quantity": "Tangent direction at 3760", "value": f"{summary['tangent_direction_at_3760_deg']:.9f} deg"},
        {"quantity": "Signed curvature at 3760", "value": f"{summary['signed_curvature_at_3760_per_deg']:.12e} 1/deg"},
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


def plot_path(data: pd.DataFrame) -> None:
    valid = np.isfinite(data["along_smooth_deg"]) & np.isfinite(data["cross_smooth_deg"])
    sample = data.loc[valid].iloc[::DISPLAY_STEP_YEARS]
    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(sample["along_smooth_deg"], sample["cross_smooth_deg"] * 2.0, linewidth=0.9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Earth JPL Orbital-Plane Normal Path\nCross-track physically exaggerated ×2", color="white", fontsize=14, weight="bold", pad=14)
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
    if np.isfinite(summary["cross_track_turn_year"]):
        ax.axvline(summary["cross_track_turn_year"], linewidth=0.7, linestyle="--", alpha=0.85)
    ax.set_title(
        "Earth Orbital-Plane Precession Speed Components\n"
        f"Cross-track turning year {summary['cross_track_turn_year']:.3f} | Total-speed range {summary['speed_range_percent']:.6f}%",
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


def plot_direction(data: pd.DataFrame, summary: dict) -> None:
    sample = data.iloc[::DISPLAY_STEP_YEARS]
    fig, ax = plt.subplots(figsize=(17, 8.5), dpi=145)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(sample["astronomical_year"], sample["tangent_direction_deg"], linewidth=0.85, label="Tangent-direction angle")
    ax2 = ax.twinx()
    ax2.tick_params(colors="white", labelsize=9)
    ax2.plot(sample["astronomical_year"], sample["signed_curvature_per_deg"], linewidth=0.65, linestyle="--", label="Signed curvature")
    if np.isfinite(summary["cross_track_turn_year"]):
        ax.axvline(summary["cross_track_turn_year"], linewidth=0.7, linestyle="--", alpha=0.85)
    ax.set_title("Earth Orbital-Plane Direction and Signed Curvature", color="white", fontsize=14, weight="bold", pad=14)
    ax.set_xlabel("Astronomical year", color="white")
    ax.set_ylabel("Tangent direction (degrees)", color="white")
    ax2.set_ylabel("Signed curvature (1/degree)", color="white")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    legend = ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()
    fig.savefig(PNG_DIRECTION, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_table(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(15, 6.2), dpi=160)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.axis("off")
    rendered = ax.table(cellText=table.values, colLabels=table.columns, cellLoc="left", loc="center", colWidths=[0.35, 0.65])
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(10)
    rendered.scale(1.0, 1.65)
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
    ax.set_title("Earth JPL Precession-Shift Audit Results", color="white", fontsize=15, weight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(PNG_TABLE, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def get_github_token() -> str:
    try:
        from google.colab import userdata
    except Exception as exc:
        raise RuntimeError("REJECTED Google Colab userdata is unavailable") from exc
    for secret_name in ("GITHUB_TOKEN", "GH_TOKEN"):
        try:
            token = userdata.get(secret_name)
        except Exception:
            token = None
        if token:
            return str(token).strip()
    raise RuntimeError("REJECTED missing Colab secret GITHUB_TOKEN or GH_TOKEN")


def github_api_request(url: str, token: str, method: str = "GET", payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": f"{VERSION}-Colab",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API HTTP {exc.code}: {detail}") from exc


def publish_csv(local_path: Path, repository_path: str, token: str) -> tuple[str, str]:
    api_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{repository_path}"
    existing_sha = None
    try:
        existing = github_api_request(f"{api_url}?ref={GITHUB_BRANCH}", token)
        existing_sha = existing.get("sha")
    except RuntimeError as exc:
        if "HTTP 404" not in str(exc):
            raise

    payload = {
        "message": f"Publish {VERSION} result {local_path.name}",
        "content": base64.b64encode(local_path.read_bytes()).decode("ascii"),
        "branch": GITHUB_BRANCH,
    }
    if existing_sha:
        payload["sha"] = existing_sha
    github_api_request(api_url, token, method="PUT", payload=payload)

    raw_url = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{repository_path}"
    github_url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{repository_path}"
    return raw_url, github_url


def main() -> None:
    print(f"OUTPUT VERSION {VERSION}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    states = query_states()
    data = derive_geometry(states)
    table, summary = find_event_summary(data)
    data.to_csv(DATA_CSV, index=False, float_format="%.12f")
    table.to_csv(SUMMARY_CSV, index=False)

    plot_path(data)
    plot_speed(data, summary)
    plot_direction(data, summary)
    plot_table(table)
    display(Image(filename=str(PNG_PATH)))
    display(Image(filename=str(PNG_SPEED)))
    display(Image(filename=str(PNG_DIRECTION)))
    display(Image(filename=str(PNG_TABLE)))
    print(table.to_string(index=False))

    token = get_github_token()
    data_repo_path = f"{GITHUB_RESULTS_DIR}/{DATA_CSV.name}"
    summary_repo_path = f"{GITHUB_RESULTS_DIR}/{SUMMARY_CSV.name}"
    data_raw, data_github = publish_csv(DATA_CSV, data_repo_path, token)
    summary_raw, summary_github = publish_csv(SUMMARY_CSV, summary_repo_path, token)

    print("CSV DATA RAW HTTPS:")
    print(data_raw)
    print("CSV DATA GITHUB:")
    print(data_github)
    print("CSV SUMMARY RAW HTTPS:")
    print(summary_raw)
    print("CSV SUMMARY GITHUB:")
    print(summary_github)
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0031