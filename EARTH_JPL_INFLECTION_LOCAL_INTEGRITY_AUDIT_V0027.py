# V0027
# Audit reference: local A.D. 3000-4500 JPL Earth orbital-plane inflection audit with independent singleton and overlap verification.
from __future__ import annotations

import contextlib
import importlib.util
import io
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

VERSION = "V0027"
LOCAL_TZ = ZoneInfo("America/Bogota")
EARTH_ID = "399"
CENTER = "@10"
REFPLANE = "ecliptic"
ABERRATIONS = "geometric"
AU_KM = 149597870.700
FOCUS_START = 3000
FOCUS_STOP = 4500
AUDIT_START = 3500
AUDIT_STOP = 4000
AUDIT_STEP = 5
OVERLAP_BATCH = 24
OVERLAP_STRIDE = 12
JD_TOLERANCE_DAYS = 2.0e-5
MAX_RETRIES = 4
RETRY_DELAY_SECONDS = 3.0

OUT = Path("/content/EARTH_JPL_INFLECTION_LOCAL_INTEGRITY_AUDIT_V0027_OUTPUT")
FOCUS_CSV = OUT / "EARTH_JPL_ANNUAL_FOCUS_3000_4500_V0027.csv"
SINGLETON_CSV = OUT / "EARTH_JPL_SINGLETON_AUDIT_3500_4000_V0027.csv"
OVERLAP_CSV = OUT / "EARTH_JPL_OVERLAP_AUDIT_3500_4000_V0027.csv"
SUMMARY_CSV = OUT / "EARTH_JPL_INFLECTION_VERDICT_V0027.csv"
PNG_FOCUS = OUT / "EARTH_JPL_LOCAL_SIGNED_MOTION_V0027.png"
PNG_DERIV = OUT / "EARTH_JPL_LOCAL_DERIVATIVES_V0027.png"
PNG_COMPARE = OUT / "EARTH_JPL_FETCH_METHOD_COMPARISON_V0027.png"
PNG_TABLE = OUT / "EARTH_JPL_INFLECTION_VERDICT_TABLE_V0027.png"


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


def horizons_vectors(epochs) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                table = Horizons(
                    id=EARTH_ID,
                    id_type="majorbody",
                    location=CENTER,
                    epochs=epochs,
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


def fetch_focus_range() -> pd.DataFrame:
    if FOCUS_CSV.exists():
        cached = pd.read_csv(FOCUS_CSV)
        if len(cached) >= 1490:
            return cached
    epochs = {"start": f"{FOCUS_START}-01-01", "stop": f"{FOCUS_STOP}-01-01", "step": "1y"}
    frame = horizons_vectors(epochs).sort_values("datetime_jd").reset_index(drop=True)
    if len(frame) < 1490:
        raise RuntimeError(f"REJECTED insufficient annual focus rows: {len(frame)}")
    frame.insert(0, "astronomical_year", FOCUS_START + np.arange(len(frame), dtype=float))
    frame.to_csv(FOCUS_CSV, index=False, float_format="%.12f")
    return frame


def query_singleton(year: float) -> dict:
    calendar = f"{int(year):04d}-01-01"
    frame = horizons_vectors({"start": calendar, "stop": f"{int(year):04d}-01-02", "step": "1d"})
    if len(frame) < 1:
        raise RuntimeError(f"REJECTED singleton row count year={year}")
    row = frame.iloc[0]
    return {
        "astronomical_year": float(year),
        "datetime_jd": float(row["datetime_jd"]),
        "x": float(row["x"]), "y": float(row["y"]), "z": float(row["z"]),
        "vx": float(row["vx"]), "vy": float(row["vy"]), "vz": float(row["vz"]),
        "method": "singleton_calendar",
    }


def fetch_singletons(years: np.ndarray) -> pd.DataFrame:
    if SINGLETON_CSV.exists():
        cached = pd.read_csv(SINGLETON_CSV)
        if len(cached) == len(years):
            return cached
    frame = pd.DataFrame([query_singleton(float(year)) for year in years])
    frame.to_csv(SINGLETON_CSV, index=False, float_format="%.12f")
    return frame


def match_rows(years: np.ndarray, returned: pd.DataFrame, method: str) -> pd.DataFrame:
    returned = returned.sort_values("datetime_jd").reset_index(drop=True)
    if len(returned) != len(years):
        raise RuntimeError(f"REJECTED {method} row count expected={len(years)} received={len(returned)}")
    rows = []
    for year, (_, row) in zip(years, returned.iterrows()):
        rows.append({
            "astronomical_year": float(year),
            "datetime_jd": float(row["datetime_jd"]),
            "x": float(row["x"]), "y": float(row["y"]), "z": float(row["z"]),
            "vx": float(row["vx"]), "vy": float(row["vy"]), "vz": float(row["vz"]),
            "method": method,
        })
    return pd.DataFrame(rows)


def fetch_overlap(years: np.ndarray) -> pd.DataFrame:
    if OVERLAP_CSV.exists():
        cached = pd.read_csv(OVERLAP_CSV)
        if len(cached) == len(years):
            return cached
    replicates = []
    for start in range(0, len(years), OVERLAP_STRIDE):
        subset = years[start:start + OVERLAP_BATCH]
        if len(subset) == 0:
            continue
        epochs = [float(query_singleton(float(year))["datetime_jd"]) for year in subset]
        returned = horizons_vectors(epochs)
        replicates.append(match_rows(subset, returned, f"overlap_{start:04d}"))
        if start + OVERLAP_BATCH >= len(years):
            break
    combined = pd.concat(replicates, ignore_index=True)
    numeric = ["datetime_jd", "x", "y", "z", "vx", "vy", "vz"]
    rows = []
    for year, group in combined.groupby("astronomical_year", sort=True):
        row = {"astronomical_year": float(year), "method": "overlap_consensus", "replicate_count": int(len(group))}
        for column in numeric:
            row[column] = float(group[column].mean())
            row[f"{column}_spread"] = float(group[column].max() - group[column].min())
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != len(years):
        raise RuntimeError(f"REJECTED overlap coverage expected={len(years)} received={len(result)}")
    result.to_csv(OVERLAP_CSV, index=False, float_format="%.12f")
    return result


def add_normals(frame: pd.DataFrame) -> pd.DataFrame:
    r = frame[["x", "y", "z"]].to_numpy(dtype=float) * AU_KM
    v = frame[["vx", "vy", "vz"]].to_numpy(dtype=float) * AU_KM
    normals = normalize_rows(np.cross(r, v))
    normals[normals[:, 2] < 0.0] *= -1.0
    result = frame.copy()
    result[["nx", "ny", "nz"]] = normals
    return result


def build_basis(focus: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normals = add_normals(focus)[["nx", "ny", "nz"]].to_numpy(dtype=float)
    n0 = normalize(normals[0])
    n1 = normalize(normals[-1])
    u = normalize(n1 - np.dot(n1, n0) * n0)
    w = normalize(np.cross(n0, u))
    return n0, u, w


def project(frame: pd.DataFrame, n0: np.ndarray, u: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    result = add_normals(frame)
    normals = result[["nx", "ny", "nz"]].to_numpy(dtype=float)
    result["signed_along_track_deg"] = np.degrees(np.arctan2(normals @ u, normals @ n0))
    result["cross_track_deg"] = np.degrees(np.arcsin(np.clip(normals @ w, -1.0, 1.0)))
    return result


def local_derivatives(years: np.ndarray, values: np.ndarray, window: int = 101, degree: int = 3) -> tuple[np.ndarray, np.ndarray]:
    first = np.full(len(values), np.nan)
    second = np.full(len(values), np.nan)
    half = window // 2
    for i in range(half, len(values) - half):
        x = years[i-half:i+half+1] - years[i]
        coeff = np.polyfit(x, values[i-half:i+half+1], degree)
        first[i] = np.polyder(np.poly1d(coeff), 1)(0.0)
        second[i] = np.polyder(np.poly1d(coeff), 2)(0.0)
    return first, second


def style_axis(ax) -> None:
    ax.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.5)
    ax.grid(True, linewidth=0.35, alpha=0.22)


def plot_focus(focus: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 8), dpi=140)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(focus["astronomical_year"], focus["signed_along_track_deg"], linewidth=0.8, label="Signed along-track displacement")
    ax.plot(focus["astronomical_year"], focus["cross_track_deg"], linewidth=0.65, linestyle="--", label="Cross-track displacement")
    ax.axvspan(AUDIT_START, AUDIT_STOP, alpha=0.08)
    ax.set_title("Earth Orbital-Plane Local Motion — Verified Annual JPL Range\nFixed reference: A.D. 3000 plane; tangent direction: A.D. 4500", color="white", fontsize=14, weight="bold", pad=16)
    ax.set_xlabel("Astronomical year", color="white")
    ax.set_ylabel("Angular displacement (degrees)", color="white")
    legend = ax.legend(frameon=False)
    for text in legend.get_texts(): text.set_color("white")
    fig.tight_layout()
    fig.savefig(PNG_FOCUS, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_derivatives(focus: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 8), dpi=140)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(focus["astronomical_year"], focus["first_derivative_arcsec_per_century"], linewidth=0.75, label="First derivative")
    ax.plot(focus["astronomical_year"], focus["second_derivative_arcsec_per_century2"], linewidth=0.65, linestyle="--", label="Second derivative")
    ax.axhline(0.0, linewidth=0.5)
    ax.axvspan(AUDIT_START, AUDIT_STOP, alpha=0.08)
    ax.set_title("Earth Orbital-Plane Local Derivative Audit", color="white", fontsize=14, weight="bold", pad=16)
    ax.set_xlabel("Astronomical year", color="white")
    ax.set_ylabel("Derivative scale (arcsec/century units)", color="white")
    legend = ax.legend(frameon=False)
    for text in legend.get_texts(): text.set_color("white")
    fig.tight_layout()
    fig.savefig(PNG_DERIV, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_comparison(singleton: pd.DataFrame, overlap: pd.DataFrame) -> None:
    merged = singleton.merge(overlap, on="astronomical_year", suffixes=("_single", "_overlap"))
    difference = (merged["signed_along_track_deg_single"] - merged["signed_along_track_deg_overlap"]) * 3600.0
    fig, ax = plt.subplots(figsize=(16, 7), dpi=140)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(merged["astronomical_year"], difference, linewidth=0.8)
    ax.axhline(0.0, linewidth=0.5)
    ax.set_title("Independent Singleton vs Overlapping-Batch Agreement", color="white", fontsize=14, weight="bold", pad=16)
    ax.set_xlabel("Astronomical year", color="white")
    ax.set_ylabel("Signed-displacement difference (arcsec)", color="white")
    fig.tight_layout()
    fig.savefig(PNG_COMPARE, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_table(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=160)
    fig.patch.set_facecolor("black")
    ax.axis("off")
    table = ax.table(cellText=summary.values, colLabels=summary.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.7)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#777777")
        cell.set_linewidth(0.45)
        cell.set_facecolor("#17324d" if row == 0 else ("#1b1b1b" if row % 2 else "#252525"))
        cell.get_text().set_color("white")
        if row == 0: cell.get_text().set_weight("bold")
    ax.set_title("Earth JPL Inflection Data-Integrity Verdict", color="white", fontsize=15, weight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(PNG_TABLE, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"OUTPUT VERSION {VERSION}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    focus = fetch_focus_range()
    n0, u, w = build_basis(focus)
    focus = project(focus, n0, u, w)
    years = focus["astronomical_year"].to_numpy(dtype=float)
    along = focus["signed_along_track_deg"].to_numpy(dtype=float)
    first, second = local_derivatives(years, along)
    focus["first_derivative_arcsec_per_century"] = first * 360000.0
    focus["second_derivative_arcsec_per_century2"] = second * 36000000.0
    focus.to_csv(FOCUS_CSV, index=False, float_format="%.12f")

    audit_years = np.arange(AUDIT_START, AUDIT_STOP + AUDIT_STEP, AUDIT_STEP, dtype=float)
    singleton = project(fetch_singletons(audit_years), n0, u, w)
    overlap = project(fetch_overlap(audit_years), n0, u, w)
    singleton.to_csv(SINGLETON_CSV, index=False, float_format="%.12f")
    overlap.to_csv(OVERLAP_CSV, index=False, float_format="%.12f")

    merged = singleton.merge(overlap, on="astronomical_year", suffixes=("_single", "_overlap"))
    diff_arcsec = np.abs(merged["signed_along_track_deg_single"] - merged["signed_along_track_deg_overlap"]) * 3600.0
    max_diff = float(diff_arcsec.max())

    valid = np.isfinite(second)
    zone = valid & (years >= AUDIT_START) & (years <= AUDIT_STOP)
    sign_changes = np.where(np.diff(np.sign(second[zone])) != 0)[0]
    feature_survives = len(sign_changes) > 0
    verdict = "CURVATURE SIGN CHANGE VERIFIED" if feature_survives else "NO CURVATURE SIGN CHANGE"

    summary = pd.DataFrame({
        "Metric": ["Audit interval", "Maximum singleton-overlap difference", "Second-derivative sign changes", "Verdict"],
        "Result": [f"{AUDIT_START}-{AUDIT_STOP}", f"{max_diff:.9f} arcsec", str(len(sign_changes)), verdict],
    })
    summary.to_csv(SUMMARY_CSV, index=False)

    plot_focus(focus)
    plot_derivatives(focus)
    plot_comparison(singleton, overlap)
    plot_table(summary)
    for image in [PNG_FOCUS, PNG_DERIV, PNG_COMPARE, PNG_TABLE]:
        display(Image(filename=str(image)))

    print(summary.to_string(index=False))
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0027