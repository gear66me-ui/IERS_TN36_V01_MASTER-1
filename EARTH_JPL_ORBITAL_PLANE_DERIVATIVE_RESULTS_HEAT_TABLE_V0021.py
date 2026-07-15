# V0021
# Audit reference: JPL Horizons Earth orbital-plane derivative audit with shareable result tables and 6000-year heat table.
from __future__ import annotations

import contextlib
import importlib.util
import io
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

VERSION = "V0021"
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
HEAT_BIN_YEARS = 250
MAX_RETRIES = 4
RETRY_DELAY_SECONDS = 4.0

OUT = Path("/content/EARTH_JPL_ORBITAL_PLANE_DERIVATIVE_RESULTS_HEAT_TABLE_V0021_OUTPUT")
RAW_CSV = OUT / "EARTH_JPL_MONTHLY_ORBITAL_NORMALS_V0021.csv"
ANNUAL_CSV = OUT / "EARTH_JPL_ANNUAL_ORBITAL_PLANE_DERIVATIVE_V0021.csv"
SUMMARY_CSV = OUT / "EARTH_JPL_PRESENT_RESULT_SUMMARY_V0021.csv"
SELECTED_CSV = OUT / "EARTH_JPL_SELECTED_EPOCH_RESULTS_V0021.csv"
PNG_TILT = OUT / "EARTH_JPL_NEUTRAL_PLANE_TILT_6000_YEAR_V0021.png"
PNG_RATE = OUT / "EARTH_JPL_NEUTRAL_PLANE_DERIVATIVE_6000_YEAR_V0021.png"
PNG_SUMMARY = OUT / "EARTH_JPL_PRESENT_RESULT_TABLE_V0021.png"
PNG_SELECTED = OUT / "EARTH_JPL_SELECTED_EPOCH_TABLE_V0021.png"
PNG_HEAT = OUT / "EARTH_JPL_ORBITAL_PLANE_HEAT_TABLE_V0021.png"


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
    normals = normalize_rows(np.cross(r, v))
    normals[normals[:, 2] < 0.0] *= -1.0
    jd = frame["datetime_jd"].to_numpy(dtype=float)
    relative_year = (jd - center_jd) / DAYS_PER_JULIAN_YEAR
    return pd.DataFrame({
        "jd_tdb": jd,
        "relative_year": relative_year,
        "annual_index": np.rint(relative_year).astype(int),
        "nx": normals[:, 0],
        "ny": normals[:, 1],
        "nz": normals[:, 2],
    })


def annualize(monthly: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict] = []
    normals: list[np.ndarray] = []
    for year, group in monthly.groupby("annual_index", sort=True):
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
    if len(annual) < 5900:
        raise RuntimeError(f"REJECTED insufficient annual planes: {len(annual)}")
    mean_normal = np.vstack(normals).mean(axis=0)
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
    tilt_deg = np.degrees(np.arccos(np.clip(normals @ mean_normal, -1.0, 1.0)))
    p = normals @ e1
    q = normals @ e2
    node_deg = np.degrees(np.unwrap(np.arctan2(q, p)))
    raw_rate = np.gradient(tilt_deg, years)
    smooth_rate = centered_moving_average(raw_rate, DERIVATIVE_SMOOTH_YEARS)
    node_rate = centered_moving_average(np.gradient(node_deg, years), DERIVATIVE_SMOOTH_YEARS)

    local_mask = np.abs(years) <= LOCAL_FIT_HALF_WIDTH_YEARS
    tilt_poly = np.poly1d(np.polyfit(years[local_mask], tilt_deg[local_mask], LOCAL_FIT_DEGREE))
    node_poly = np.poly1d(np.polyfit(years[local_mask], node_deg[local_mask], LOCAL_FIT_DEGREE))
    current_tilt = float(tilt_poly(0.0))
    current_tilt_rate = float(np.polyder(tilt_poly)(0.0))
    current_node = float(node_poly(0.0))
    current_node_rate = float(np.polyder(node_poly)(0.0))
    state = "UPSWING" if current_tilt_rate > 0.0 else "DOWNSWING" if current_tilt_rate < 0.0 else "TURNING POINT"

    annual = annual.copy()
    annual["tilt_from_interval_mean_deg"] = tilt_deg
    annual["node_angle_unwrapped_deg"] = node_deg
    annual["tilt_rate_raw_deg_per_year"] = raw_rate
    annual["tilt_rate_smoothed_deg_per_year"] = smooth_rate
    annual["tilt_rate_smoothed_arcsec_per_century"] = smooth_rate * 3600.0 * 100.0
    annual["node_rate_smoothed_deg_per_year"] = node_rate
    annual["node_rate_smoothed_arcsec_per_century"] = node_rate * 3600.0 * 100.0
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
    local_x = np.linspace(-LOCAL_FIT_HALF_WIDTH_YEARS, LOCAL_FIT_HALF_WIDTH_YEARS, 501)
    ax.plot(local_x, result["tilt_poly"](local_x), linewidth=1.1, label=f"Local degree-{LOCAL_FIT_DEGREE} fit")
    ax.scatter([0.0], [result["current_tilt_deg"]], s=24.0, zorder=5)
    ax.set_title(
        "Earth Orbital-Plane Oscillation Relative to the JPL 6000-Year Mean Plane\n"
        f"Present classification: {result['state']}   |   di/dt = {result['current_tilt_rate_arcsec_per_century']:+.6f} arcsec/century",
        color="white", fontsize=14, weight="bold", pad=16,
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
    rate = annual["tilt_rate_smoothed_arcsec_per_century"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(16, 8), dpi=140)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(years, rate, linewidth=0.65, label=f"{DERIVATIVE_SMOOTH_YEARS}-year smoothed derivative")
    ax.axhline(0.0, linewidth=0.55, linestyle="--", alpha=0.8)
    ax.axvline(0.0, linewidth=0.55, linestyle="--", alpha=0.8)
    ax.scatter([0.0], [result["current_tilt_rate_arcsec_per_century"]], s=24.0, zorder=5)
    ax.set_title(
        "Earth Orbital-Plane Inclination Derivative\n"
        f"Present result: {result['state']}   |   di/dt = {result['current_tilt_rate_arcsec_per_century']:+.6f} arcsec/century",
        color="white", fontsize=14, weight="bold", pad=16,
    )
    ax.set_xlabel("Years from present JPL epoch", color="white")
    ax.set_ylabel("Tilt derivative (arcsec/century)", color="white")
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()
    fig.savefig(PNG_RATE, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def save_table_png(frame: pd.DataFrame, title: str, path: Path, font_size: float = 9.0) -> None:
    rows, cols = frame.shape
    fig_width = max(12.0, cols * 2.15)
    fig_height = max(3.2, 1.7 + rows * 0.56)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.axis("off")
    table = ax.table(
        cellText=frame.values,
        colLabels=frame.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#6f6f6f")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#20344f")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#111820" if row % 2 else "#182431")
            cell.get_text().set_color("white")
    ax.set_title(title, color="white", fontsize=15, weight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def build_summary_table(result: dict, mean_normal: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([
        ["Classification", result["state"], "sign of local fitted di/dt"],
        ["Present tilt", f"{result['current_tilt_deg']:.9f}", "deg from 6000-year mean plane"],
        ["Present tilt derivative", f"{result['current_tilt_rate_arcsec_per_century']:+.9f}", "arcsec/century"],
        ["Present node angle", f"{result['current_node_deg']:.9f}", "deg in mean-plane basis"],
        ["Present node derivative", f"{result['current_node_rate_arcsec_per_century']:+.9f}", "arcsec/century"],
        ["Neutral-plane normal X", f"{mean_normal[0]:+.12f}", "unit vector"],
        ["Neutral-plane normal Y", f"{mean_normal[1]:+.12f}", "unit vector"],
        ["Neutral-plane normal Z", f"{mean_normal[2]:+.12f}", "unit vector"],
        ["Local fit", f"degree {LOCAL_FIT_DEGREE}, ±{LOCAL_FIT_HALF_WIDTH_YEARS} yr", "JPL annual means"],
        ["Derivative smoothing", f"{DERIVATIVE_SMOOTH_YEARS} years", "centered moving average"],
    ], columns=["Quantity", "Result", "Definition / unit"])


def build_selected_table(annual: pd.DataFrame) -> pd.DataFrame:
    selected_years = [-3000, -2500, -2000, -1500, -1000, -500, -250, -100, -50, 0, 50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000]
    rows = []
    for target in selected_years:
        index = int(np.argmin(np.abs(annual["relative_year"].to_numpy(dtype=float) - target)))
        row = annual.iloc[index]
        rate = float(row["tilt_rate_smoothed_arcsec_per_century"])
        state = "UP" if rate > 0.0 else "DOWN" if rate < 0.0 else "TURN"
        rows.append([
            f"{int(row['relative_year']):+d}",
            f"{row['tilt_from_interval_mean_deg']:.8f}",
            f"{rate:+.8f}",
            f"{row['node_angle_unwrapped_deg']:.6f}",
            f"{row['node_rate_smoothed_arcsec_per_century']:+.8f}",
            state,
        ])
    return pd.DataFrame(rows, columns=[
        "Year from present", "Tilt (deg)", "di/dt (arcsec/cy)",
        "Node angle (deg)", "dNode/dt (arcsec/cy)", "Tilt state",
    ])


def plot_heat_table(annual: pd.DataFrame) -> pd.DataFrame:
    edges = np.arange(-HALF_SPAN_YEARS, HALF_SPAN_YEARS + HEAT_BIN_YEARS, HEAT_BIN_YEARS)
    labels = ((edges[:-1] + edges[1:]) / 2.0).astype(int)
    bins = pd.cut(annual["relative_year"], bins=edges, labels=labels, include_lowest=True, right=False)
    heat = annual.assign(heat_bin=bins).groupby("heat_bin", observed=False).agg(
        tilt_deg=("tilt_from_interval_mean_deg", "mean"),
        tilt_rate_arcsec_cy=("tilt_rate_smoothed_arcsec_per_century", "mean"),
        node_rate_arcsec_cy=("node_rate_smoothed_arcsec_per_century", "mean"),
    ).reset_index()
    heat["heat_bin"] = heat["heat_bin"].astype(float)

    raw = heat[["tilt_deg", "tilt_rate_arcsec_cy", "node_rate_arcsec_cy"]].to_numpy(dtype=float).T
    normalized = np.zeros_like(raw)
    for i in range(raw.shape[0]):
        scale = np.nanmax(np.abs(raw[i]))
        normalized[i] = raw[i] / scale if scale > 0.0 else raw[i]

    fig, ax = plt.subplots(figsize=(19, 5.8), dpi=150)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    image = ax.imshow(normalized, aspect="auto", interpolation="nearest", vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Mean tilt", "Tilt derivative", "Node derivative"], color="white", fontsize=10)
    tick_indices = np.arange(0, len(heat), 2)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([f"{int(heat.iloc[i]['heat_bin']):+d}" for i in tick_indices], rotation=45, ha="right", color="white", fontsize=8)
    present_index = int(np.argmin(np.abs(heat["heat_bin"].to_numpy(dtype=float))))
    ax.axvline(present_index, linewidth=1.0, linestyle="--", color="white")
    for row in range(raw.shape[0]):
        for col in range(raw.shape[1]):
            if row == 0:
                text = f"{raw[row, col]:.3f}°"
            else:
                text = f"{raw[row, col]:+.1f}"
            ax.text(col, row, text, ha="center", va="center", fontsize=6.1, color="black" if abs(normalized[row, col]) < 0.45 else "white")
    ax.set_title(
        "Earth JPL Orbital-Plane Heat Table — 250-Year Means\n"
        "Tilt in degrees; derivative rows in arcsec/century; dashed line marks the present bin",
        color="white", fontsize=14, weight="bold", pad=14,
    )
    ax.set_xlabel("Bin-center year from present", color="white")
    ax.tick_params(colors="white")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.ax.tick_params(colors="white", labelsize=8)
    colorbar.set_label("Row-normalized signed magnitude", color="white")
    fig.tight_layout()
    fig.savefig(PNG_HEAT, dpi=320, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    return heat


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
    annual["neutral_plane_nx"] = mean_normal[0]
    annual["neutral_plane_ny"] = mean_normal[1]
    annual["neutral_plane_nz"] = mean_normal[2]
    annual.to_csv(ANNUAL_CSV, index=False, float_format="%.12f")

    summary = build_summary_table(result, mean_normal)
    selected = build_selected_table(annual)
    summary.to_csv(SUMMARY_CSV, index=False)
    selected.to_csv(SELECTED_CSV, index=False)

    plot_tilt(annual, result)
    plot_rate(annual, result)
    save_table_png(summary, "Earth Orbital-Plane Derivative — Present Result Summary", PNG_SUMMARY, 9.5)
    save_table_png(selected, "Earth Orbital-Plane Derivative — Selected Epochs", PNG_SELECTED, 8.2)
    plot_heat_table(annual)

    display(Image(filename=str(PNG_SUMMARY)))
    display(Image(filename=str(PNG_SELECTED)))
    display(Image(filename=str(PNG_TILT)))
    display(Image(filename=str(PNG_RATE)))
    display(Image(filename=str(PNG_HEAT)))

    print("SHAREABLE RESULT")
    print(f"State                         {result['state']}")
    print(f"Present tilt                  {result['current_tilt_deg']:.9f} deg")
    print(f"Present tilt derivative       {result['current_tilt_rate_arcsec_per_century']:+.9f} arcsec/century")
    print(f"Present node angle            {result['current_node_deg']:.9f} deg")
    print(f"Present node derivative       {result['current_node_rate_arcsec_per_century']:+.9f} arcsec/century")
    print(f"Neutral-plane normal          [{mean_normal[0]:+.12f}, {mean_normal[1]:+.12f}, {mean_normal[2]:+.12f}]")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0021