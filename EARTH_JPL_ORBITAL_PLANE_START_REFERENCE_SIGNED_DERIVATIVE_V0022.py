# V0022
# Audit reference: JPL Earth orbital-plane motion using the -3000-year plane as a fixed zero and signed tangent-plane displacement.
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

VERSION = "V0022"
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
SMOOTH_YEARS = 101
LOCAL_FIT_HALF_WIDTH_YEARS = 250
LOCAL_FIT_DEGREE = 3
HEAT_BIN_YEARS = 250
MAX_RETRIES = 4
RETRY_DELAY_SECONDS = 4.0

OUT = Path("/content/EARTH_JPL_ORBITAL_PLANE_START_REFERENCE_SIGNED_DERIVATIVE_V0022_OUTPUT")
RAW_CSV = OUT / "EARTH_JPL_MONTHLY_ORBITAL_NORMALS_V0022.csv"
ANNUAL_CSV = OUT / "EARTH_JPL_ANNUAL_SIGNED_PLANE_MOTION_V0022.csv"
SUMMARY_CSV = OUT / "EARTH_JPL_PRESENT_SIGNED_RESULT_V0022.csv"
SELECTED_CSV = OUT / "EARTH_JPL_SELECTED_SIGNED_EPOCHS_V0022.csv"
PNG_MOTION = OUT / "EARTH_JPL_SIGNED_PLANE_MOTION_V0022.png"
PNG_RATE = OUT / "EARTH_JPL_SIGNED_PLANE_RATE_V0022.png"
PNG_SUMMARY = OUT / "EARTH_JPL_SIGNED_RESULT_TABLE_V0022.png"
PNG_SELECTED = OUT / "EARTH_JPL_SELECTED_SIGNED_EPOCH_TABLE_V0022.png"
PNG_HEAT = OUT / "EARTH_JPL_SIGNED_PLANE_HEAT_TABLE_V0022.png"
V0021_RAW = Path("/content/EARTH_JPL_ORBITAL_PLANE_DERIVATIVE_RESULTS_HEAT_TABLE_V0021_OUTPUT/EARTH_JPL_MONTHLY_ORBITAL_NORMALS_V0021.csv")


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
                    id=EARTH_ID,
                    id_type="majorbody",
                    location=CENTER,
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


def load_or_download_monthly(center_jd: float) -> pd.DataFrame:
    if V0021_RAW.exists():
        cached = pd.read_csv(V0021_RAW)
        required = {"jd_tdb", "relative_year", "annual_index", "nx", "ny", "nz"}
        if required.issubset(cached.columns) and len(cached) >= 70000:
            return cached[list(required)].copy()
    raw = download_monthly_states(center_jd)
    return build_monthly_normals(raw, center_jd)


def annualize(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for year, group in monthly.groupby("annual_index", sort=True):
        if year < -HALF_SPAN_YEARS or year > HALF_SPAN_YEARS:
            continue
        normal = normalize(group[["nx", "ny", "nz"]].to_numpy(dtype=float).mean(axis=0))
        if normal[2] < 0.0:
            normal *= -1.0
        rows.append({
            "relative_year": int(year),
            "sample_count": int(len(group)),
            "nx": float(normal[0]),
            "ny": float(normal[1]),
            "nz": float(normal[2]),
        })
    annual = pd.DataFrame(rows).sort_values("relative_year").reset_index(drop=True)
    if len(annual) < 5900:
        raise RuntimeError(f"REJECTED insufficient annual planes: {len(annual)}")
    return annual


def centered_average(values: np.ndarray, window: int) -> np.ndarray:
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, (window // 2, window // 2), mode="reflect")
    result = np.convolve(padded, kernel, mode="valid")
    edge = window // 2
    result[:edge] = np.nan
    result[-edge:] = np.nan
    return result


def derive_signed_motion(annual: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    years = annual["relative_year"].to_numpy(dtype=float)
    normals = annual[["nx", "ny", "nz"]].to_numpy(dtype=float)
    start_index = int(np.argmin(np.abs(years + HALF_SPAN_YEARS)))
    end_index = int(np.argmin(np.abs(years - HALF_SPAN_YEARS)))
    present_index = int(np.argmin(np.abs(years)))
    n0 = normalize(normals[start_index])
    n_end = normalize(normals[end_index])

    endpoint_tangent = n_end - np.dot(n_end, n0) * n0
    u = normalize(endpoint_tangent)
    v = normalize(np.cross(n0, u))

    along = np.degrees(np.arctan2(normals @ u, normals @ n0))
    cross = np.degrees(np.arcsin(np.clip(normals @ v, -1.0, 1.0)))
    total = np.degrees(np.arccos(np.clip(normals @ n0, -1.0, 1.0)))

    along_rate_raw = np.gradient(along, years)
    cross_rate_raw = np.gradient(cross, years)
    along_rate_smooth = centered_average(along_rate_raw, SMOOTH_YEARS)
    cross_rate_smooth = centered_average(cross_rate_raw, SMOOTH_YEARS)

    local = np.abs(years) <= LOCAL_FIT_HALF_WIDTH_YEARS
    along_poly = np.poly1d(np.polyfit(years[local], along[local], LOCAL_FIT_DEGREE))
    cross_poly = np.poly1d(np.polyfit(years[local], cross[local], LOCAL_FIT_DEGREE))
    present_along = float(along_poly(0.0))
    present_cross = float(cross_poly(0.0))
    present_along_rate = float(np.polyder(along_poly)(0.0))
    present_cross_rate = float(np.polyder(cross_poly)(0.0))
    state = "POSITIVE DIRECTION" if present_along_rate > 0.0 else "NEGATIVE DIRECTION" if present_along_rate < 0.0 else "STATIONARY"

    annual = annual.copy()
    annual["signed_along_track_deg"] = along
    annual["cross_track_deg"] = cross
    annual["total_angle_from_start_deg"] = total
    annual["along_rate_raw_deg_per_year"] = along_rate_raw
    annual["along_rate_smoothed_deg_per_year"] = along_rate_smooth
    annual["along_rate_smoothed_arcsec_per_century"] = along_rate_smooth * 360000.0
    annual["cross_rate_smoothed_arcsec_per_century"] = cross_rate_smooth * 360000.0

    result = {
        "state": state,
        "present_year": float(years[present_index]),
        "present_along_deg": present_along,
        "present_cross_deg": present_cross,
        "present_total_deg": float(total[present_index]),
        "present_along_rate_arcsec_per_century": present_along_rate * 360000.0,
        "present_cross_rate_arcsec_per_century": present_cross_rate * 360000.0,
        "start_normal": n0,
        "direction_u": u,
        "cross_v": v,
        "along_poly": along_poly,
        "cross_poly": cross_poly,
    }
    return annual, result


def style_axis(ax) -> None:
    ax.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.5)
    ax.grid(True, linewidth=0.35, alpha=0.22)


def plot_motion(annual: pd.DataFrame, result: dict) -> None:
    years = annual["relative_year"].to_numpy(dtype=float)
    along = annual["signed_along_track_deg"].to_numpy(dtype=float)
    cross = annual["cross_track_deg"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(16, 8), dpi=140)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(years, along, linewidth=0.75, label="Signed along-track plane displacement")
    ax.plot(years, cross, linewidth=0.55, linestyle="--", label="Cross-track departure")
    ax.axvline(0.0, linewidth=0.55, linestyle=":", alpha=0.8)
    ax.axhline(0.0, linewidth=0.45, alpha=0.7)
    ax.scatter([-HALF_SPAN_YEARS, 0, HALF_SPAN_YEARS], np.interp([-HALF_SPAN_YEARS, 0, HALF_SPAN_YEARS], years, along), s=18)
    ax.set_title(
        "Earth Orbital-Plane Motion from a Fixed −3000-Year JPL Reference Plane\n"
        f"Present signed rate = {result['present_along_rate_arcsec_per_century']:+.6f} arcsec/century",
        color="white", fontsize=14, weight="bold", pad=16,
    )
    ax.set_xlabel("Years from present JPL epoch", color="white")
    ax.set_ylabel("Signed angular displacement (degrees)", color="white")
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()
    fig.savefig(PNG_MOTION, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_rate(annual: pd.DataFrame, result: dict) -> None:
    years = annual["relative_year"].to_numpy(dtype=float)
    rate = annual["along_rate_smoothed_arcsec_per_century"].to_numpy(dtype=float)
    cross_rate = annual["cross_rate_smoothed_arcsec_per_century"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(16, 8), dpi=140)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(years, rate, linewidth=0.75, label=f"Along-track rate, {SMOOTH_YEARS}-year centered smoothing")
    ax.plot(years, cross_rate, linewidth=0.50, linestyle="--", label="Cross-track rate")
    ax.axvline(0.0, linewidth=0.55, linestyle=":", alpha=0.8)
    ax.axhline(0.0, linewidth=0.55, alpha=0.7)
    ax.scatter([0.0], [result["present_along_rate_arcsec_per_century"]], s=22, zorder=5)
    ax.set_xlim(-HALF_SPAN_YEARS + SMOOTH_YEARS, HALF_SPAN_YEARS - SMOOTH_YEARS)
    ax.set_title(
        "Earth Orbital-Plane Signed Derivative — Fixed Start Reference\n"
        f"No interval-mean cusp | Present direction: {result['state']}",
        color="white", fontsize=14, weight="bold", pad=16,
    )
    ax.set_xlabel("Years from present JPL epoch", color="white")
    ax.set_ylabel("Angular rate (arcsec/century)", color="white")
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()
    fig.savefig(PNG_RATE, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def save_table_png(frame: pd.DataFrame, path: Path, title: str, figsize: tuple[float, float], font_size: float = 9.0) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.axis("off")
    table = ax.table(cellText=frame.values, colLabels=frame.columns, cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#5f6b73")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#17324d")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#101820" if row % 2 else "#18242e")
            cell.set_text_props(color="white")
    ax.set_title(title, color="white", fontsize=14, weight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(path, dpi=320, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def make_tables(annual: pd.DataFrame, result: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.DataFrame([
        ["Reference plane", "Earth annual JPL plane at −3000 years"],
        ["Present direction", result["state"]],
        ["Present signed displacement", f"{result['present_along_deg']:+.9f} deg"],
        ["Present cross-track departure", f"{result['present_cross_deg']:+.9f} deg"],
        ["Present total angle from start", f"{result['present_total_deg']:.9f} deg"],
        ["Present signed derivative", f"{result['present_along_rate_arcsec_per_century']:+.9f} arcsec/century"],
        ["Present cross-track derivative", f"{result['present_cross_rate_arcsec_per_century']:+.9f} arcsec/century"],
        ["Start-plane normal", np.array2string(result["start_normal"], precision=12, separator=", ")],
    ], columns=["Quantity", "JPL-derived result"])

    selected_years = [-3000, -2000, -1000, 0, 1000, 2000, 3000]
    rows = []
    for target in selected_years:
        row = annual.iloc[int(np.argmin(np.abs(annual["relative_year"].to_numpy() - target)))]
        rows.append({
            "Year": int(row["relative_year"]),
            "Signed deg": f"{row['signed_along_track_deg']:+.8f}",
            "Cross deg": f"{row['cross_track_deg']:+.8f}",
            "Total deg": f"{row['total_angle_from_start_deg']:.8f}",
            "Rate arcsec/cy": f"{row['along_rate_smoothed_arcsec_per_century']:+.6f}" if np.isfinite(row["along_rate_smoothed_arcsec_per_century"]) else "EDGE",
        })
    selected = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    selected.to_csv(SELECTED_CSV, index=False)
    save_table_png(summary, PNG_SUMMARY, "Earth Orbital-Plane Fixed-Start Result Summary", (14, 6.5), 9.3)
    save_table_png(selected, PNG_SELECTED, "Selected Epochs — Signed Motion from the −3000-Year Plane", (13, 5.4), 9.2)
    return summary, selected


def make_heat_table(annual: pd.DataFrame) -> None:
    starts = np.arange(-HALF_SPAN_YEARS, HALF_SPAN_YEARS, HEAT_BIN_YEARS)
    centers = starts + HEAT_BIN_YEARS / 2.0
    values = []
    for column in ["signed_along_track_deg", "cross_track_deg", "along_rate_smoothed_arcsec_per_century"]:
        row = []
        for start in starts:
            mask = (annual["relative_year"] >= start) & (annual["relative_year"] < start + HEAT_BIN_YEARS)
            row.append(float(np.nanmean(annual.loc[mask, column].to_numpy(dtype=float))))
        values.append(row)
    matrix = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(18, 5.8), dpi=150)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="coolwarm")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Signed displacement (deg)", "Cross-track (deg)", "Rate (arcsec/cy)"], color="white")
    tick_positions = np.arange(0, len(centers), 2)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{int(centers[index]):+d}" for index in tick_positions], rotation=45, ha="right", color="white")
    present_bin = int(np.argmin(np.abs(centers)))
    ax.axvline(present_bin, color="white", linewidth=0.9, linestyle="--")
    ax.set_title("Earth Orbital-Plane Signed Motion Heat Table — 250-Year JPL Bins", color="white", fontsize=14, weight="bold", pad=15)
    ax.set_xlabel("Bin-center year relative to present", color="white")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#777777")
    fig.tight_layout()
    fig.savefig(PNG_HEAT, dpi=320, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"OUTPUT VERSION {VERSION}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    center_jd = float(Time(datetime.now(timezone.utc), scale="utc").tdb.jd)
    monthly = load_or_download_monthly(center_jd)
    monthly.to_csv(RAW_CSV, index=False, float_format="%.12f")
    annual = annualize(monthly)
    annual, result = derive_signed_motion(annual)
    annual.to_csv(ANNUAL_CSV, index=False, float_format="%.12f")
    plot_motion(annual, result)
    plot_rate(annual, result)
    make_tables(annual, result)
    make_heat_table(annual)
    for path in [PNG_SUMMARY, PNG_SELECTED, PNG_MOTION, PNG_RATE, PNG_HEAT]:
        display(Image(filename=str(path)))
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0022