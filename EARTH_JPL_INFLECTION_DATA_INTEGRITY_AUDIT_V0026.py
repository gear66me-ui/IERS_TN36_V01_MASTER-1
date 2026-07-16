# V0026
# Audit reference: robust JPL Earth orbital-plane inflection audit using singleton context epochs, a continuous annual focus range, and JD-matched overlap checks.
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
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display

VERSION = "V0026"
LOCAL_TZ = ZoneInfo("America/Bogota")
EARTH_ID = "399"
CENTER = "@10"
REFPLANE = "ecliptic"
ABERRATIONS = "geometric"
AU_KM = 149597870.700
J2000_JD = 2451545.0
DAYS_PER_JULIAN_YEAR = 365.25
CONTEXT_START = -9000
CONTEXT_STOP = 9000
CONTEXT_STEP = 100
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

OUT = Path("/content/EARTH_JPL_INFLECTION_DATA_INTEGRITY_AUDIT_V0026_OUTPUT")
CONTEXT_CSV = OUT / "EARTH_JPL_CONTEXT_100YEAR_SINGLETONS_V0026.csv"
FOCUS_CSV = OUT / "EARTH_JPL_FOCUS_ANNUAL_3000_4500_V0026.csv"
AUDIT_CSV = OUT / "EARTH_JPL_INFLECTION_AUDIT_COMPARISON_V0026.csv"
SUMMARY_CSV = OUT / "EARTH_JPL_INFLECTION_AUDIT_SUMMARY_V0026.csv"
PNG_CONTEXT = OUT / "EARTH_JPL_SIGNED_PLANE_CONTEXT_100YEAR_V0026.png"
PNG_FOCUS = OUT / "EARTH_JPL_INFLECTION_FOCUS_3000_4500_V0026.png"
PNG_DERIV = OUT / "EARTH_JPL_INFLECTION_DERIVATIVES_V0026.png"
PNG_TABLE = OUT / "EARTH_JPL_INFLECTION_AUDIT_TABLE_V0026.png"


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("REJECTED invalid vector magnitude")
    return vector / norm


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise RuntimeError("REJECTED invalid vector-row magnitude")
    return vectors / norms[:, None]


def year_to_jd(year: float) -> float:
    return J2000_JD + (float(year) - 2000.0) * DAYS_PER_JULIAN_YEAR


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


def query_singleton(year: float, method: str) -> dict:
    requested_jd = year_to_jd(year)
    frame = horizons_vectors([requested_jd])
    if len(frame) != 1:
        raise RuntimeError(f"REJECTED singleton row count method={method} year={year}: {len(frame)}")
    row = frame.iloc[0]
    jd_error = abs(float(row["datetime_jd"]) - requested_jd)
    if jd_error > JD_TOLERANCE_DAYS:
        raise RuntimeError(f"REJECTED singleton JD mismatch method={method} year={year} error_days={jd_error}")
    return {
        "astronomical_year": float(year),
        "requested_jd": requested_jd,
        "returned_jd": float(row["datetime_jd"]),
        "jd_error_days": jd_error,
        "x": float(row["x"]),
        "y": float(row["y"]),
        "z": float(row["z"]),
        "vx": float(row["vx"]),
        "vy": float(row["vy"]),
        "vz": float(row["vz"]),
        "method": method,
    }


def fetch_singletons(years: np.ndarray, method: str) -> pd.DataFrame:
    rows = [query_singleton(float(year), method) for year in years]
    return pd.DataFrame(rows).sort_values("astronomical_year").reset_index(drop=True)


def fetch_context() -> pd.DataFrame:
    if CONTEXT_CSV.exists():
        cached = pd.read_csv(CONTEXT_CSV)
        if len(cached) == len(np.arange(CONTEXT_START, CONTEXT_STOP + CONTEXT_STEP, CONTEXT_STEP)):
            return cached
    years = np.arange(CONTEXT_START, CONTEXT_STOP + CONTEXT_STEP, CONTEXT_STEP, dtype=float)
    frame = fetch_singletons(years, "context_singleton")
    frame.to_csv(CONTEXT_CSV, index=False, float_format="%.12f")
    return frame


def fetch_focus_range() -> pd.DataFrame:
    if FOCUS_CSV.exists():
        cached = pd.read_csv(FOCUS_CSV)
        if len(cached) >= 1490:
            return cached
    epochs = {
        "start": f"{FOCUS_START}-01-01",
        "stop": f"{FOCUS_STOP}-01-01",
        "step": "1y",
    }
    frame = horizons_vectors(epochs).sort_values("datetime_jd").reset_index(drop=True)
    if len(frame) < 1490:
        raise RuntimeError(f"REJECTED insufficient annual focus rows: {len(frame)}")
    frame.insert(0, "astronomical_year", FOCUS_START + np.arange(len(frame), dtype=float))
    frame["requested_jd"] = frame["astronomical_year"].map(year_to_jd)
    frame["returned_jd"] = frame["datetime_jd"].astype(float)
    frame["jd_error_days"] = np.abs(frame["returned_jd"] - frame["requested_jd"])
    frame["method"] = "annual_range"
    frame.to_csv(FOCUS_CSV, index=False, float_format="%.12f")
    return frame


def match_returned_rows(requested_years: np.ndarray, returned: pd.DataFrame, method: str) -> pd.DataFrame:
    returned = returned.sort_values("datetime_jd").reset_index(drop=True)
    returned_jds = returned["datetime_jd"].to_numpy(dtype=float)
    used: set[int] = set()
    rows: list[dict] = []
    for year in requested_years:
        requested_jd = year_to_jd(float(year))
        idx = int(np.argmin(np.abs(returned_jds - requested_jd)))
        error = float(abs(returned_jds[idx] - requested_jd))
        if error > JD_TOLERANCE_DAYS:
            raise RuntimeError(f"REJECTED JD mismatch method={method} year={year} error_days={error}")
        if idx in used:
            raise RuntimeError(f"REJECTED duplicate matched row method={method} year={year}")
        used.add(idx)
        row = returned.iloc[idx]
        rows.append({
            "astronomical_year": float(year),
            "requested_jd": requested_jd,
            "returned_jd": float(row["datetime_jd"]),
            "jd_error_days": error,
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z": float(row["z"]),
            "vx": float(row["vx"]),
            "vy": float(row["vy"]),
            "vz": float(row["vz"]),
            "method": method,
        })
    return pd.DataFrame(rows)


def fetch_overlap_batches(years: np.ndarray) -> pd.DataFrame:
    accumulated: list[pd.DataFrame] = []
    for start in range(0, len(years), OVERLAP_STRIDE):
        subset = years[start:start + OVERLAP_BATCH]
        if len(subset) == 0:
            continue
        jds = np.array([year_to_jd(year) for year in subset], dtype=float)
        returned = horizons_vectors(jds.tolist())
        accumulated.append(match_returned_rows(subset, returned, f"overlap_{start:04d}"))
        if start + OVERLAP_BATCH >= len(years):
            break
    combined = pd.concat(accumulated, ignore_index=True)
    numeric = ["requested_jd", "returned_jd", "jd_error_days", "x", "y", "z", "vx", "vy", "vz"]
    grouped = combined.groupby("astronomical_year", sort=True)
    rows: list[dict] = []
    for year, group in grouped:
        row = {"astronomical_year": float(year), "method": "overlap_consensus", "replicate_count": int(len(group))}
        for column in numeric:
            row[column] = float(group[column].mean())
            row[f"{column}_spread"] = float(group[column].max() - group[column].min())
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("astronomical_year").reset_index(drop=True)
    if len(result) != len(years):
        raise RuntimeError(f"REJECTED overlap coverage: expected {len(years)}, received {len(result)}")
    return result


def add_normals(frame: pd.DataFrame) -> pd.DataFrame:
    r = frame[["x", "y", "z"]].to_numpy(dtype=float) * AU_KM
    v = frame[["vx", "vy", "vz"]].to_numpy(dtype=float) * AU_KM
    normals = normalize_rows(np.cross(r, v))
    normals[normals[:, 2] < 0.0] *= -1.0
    result = frame.copy()
    result["nx"] = normals[:, 0]
    result["ny"] = normals[:, 1]
    result["nz"] = normals[:, 2]
    return result


def build_basis(context: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    context = add_normals(context)
    years = context["astronomical_year"].to_numpy(dtype=float)
    normals = context[["nx", "ny", "nz"]].to_numpy(dtype=float)
    n0 = normalize(normals[int(np.argmin(np.abs(years - CONTEXT_START)))])
    n1 = normalize(normals[int(np.argmin(np.abs(years - CONTEXT_STOP)))])
    u = normalize(n1 - np.dot(n1, n0) * n0)
    w = normalize(np.cross(n0, u))
    return n0, u, w


def project(frame: pd.DataFrame, n0: np.ndarray, u: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    result = add_normals(frame)
    normals = result[["nx", "ny", "nz"]].to_numpy(dtype=float)
    result["signed_along_track_deg"] = np.degrees(np.arctan2(normals @ u, normals @ n0))
    result["cross_track_deg"] = np.degrees(np.arcsin(np.clip(normals @ w, -1.0, 1.0)))
    return result


def local_polynomial_derivatives(years: np.ndarray, values: np.ndarray, window: int = 101, degree: int = 3) -> tuple[np.ndarray, np.ndarray]:
    first = np.full_like(values, np.nan, dtype=float)
    second = np.full_like(values, np.nan, dtype=float)
    half = window // 2
    for i in range(half, len(years) - half):
        x = years[i - half:i + half + 1] - years[i]
        y = values[i - half:i + half + 1]
        coeff = np.polyfit(x, y, degree)
        poly = np.poly1d(coeff)
        first[i] = float(np.polyder(poly, 1)(0.0))
        second[i] = float(np.polyder(poly, 2)(0.0))
    return first, second


def prepare_focus(focus: pd.DataFrame, n0: np.ndarray, u: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    focus = project(focus, n0, u, w)
    years = focus["astronomical_year"].to_numpy(dtype=float)
    along = focus["signed_along_track_deg"].to_numpy(dtype=float)
    first, second = local_polynomial_derivatives(years, along, window=101, degree=3)
    focus["first_derivative_arcsec_per_century"] = first * 360000.0
    focus["second_derivative_arcsec_per_century2"] = second * 36000000.0
    return focus


def compare_methods(singleton: pd.DataFrame, overlap: pd.DataFrame, n0: np.ndarray, u: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    s = project(singleton, n0, u, w)
    o = project(overlap, n0, u, w)
    merged = s.merge(o, on="astronomical_year", suffixes=("_singleton", "_overlap"), validate="one_to_one")
    dot = np.clip(
        merged[["nx_singleton", "ny_singleton", "nz_singleton"]].to_numpy(dtype=float)
        * merged[["nx_overlap", "ny_overlap", "nz_overlap"]].to_numpy(dtype=float),
        -1.0,
        1.0,
    ).sum(axis=1)
    merged["normal_difference_milliarcsec"] = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))) * 3600.0 * 1000.0
    merged["along_difference_milliarcsec"] = (
        merged["signed_along_track_deg_singleton"] - merged["signed_along_track_deg_overlap"]
    ) * 3600.0 * 1000.0
    merged["cross_difference_milliarcsec"] = (
        merged["cross_track_deg_singleton"] - merged["cross_track_deg_overlap"]
    ) * 3600.0 * 1000.0
    return merged


def detect_inflection(focus: pd.DataFrame) -> dict:
    valid = focus.dropna(subset=["second_derivative_arcsec_per_century2"]).copy()
    zone = valid[(valid["astronomical_year"] >= 3300) & (valid["astronomical_year"] <= 4200)].copy()
    years = zone["astronomical_year"].to_numpy(dtype=float)
    second = zone["second_derivative_arcsec_per_century2"].to_numpy(dtype=float)
    signs = np.sign(second)
    candidates: list[float] = []
    for i in range(1, len(signs)):
        if signs[i] == 0.0:
            candidates.append(float(years[i]))
        elif signs[i - 1] != 0.0 and signs[i] != signs[i - 1]:
            x0, x1 = years[i - 1], years[i]
            y0, y1 = second[i - 1], second[i]
            root = x0 - y0 * (x1 - x0) / (y1 - y0)
            candidates.append(float(root))
    near = [value for value in candidates if 3500.0 <= value <= 4000.0]
    return {
        "candidate_count_3300_4200": len(candidates),
        "candidate_count_3500_4000": len(near),
        "candidate_years_3500_4000": ", ".join(f"{value:.2f}" for value in near) if near else "NONE",
    }


def style_axis(ax) -> None:
    ax.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.5)
    ax.grid(True, linewidth=0.35, alpha=0.22)


def plot_context(context: pd.DataFrame, n0: np.ndarray, u: np.ndarray, w: np.ndarray) -> None:
    data = project(context, n0, u, w)
    fig, ax = plt.subplots(figsize=(17, 8), dpi=150)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(data["astronomical_year"], data["signed_along_track_deg"], linewidth=0.8, marker="o", markersize=1.8, label="Signed along-track displacement")
    ax.plot(data["astronomical_year"], data["cross_track_deg"], linewidth=0.65, marker="o", markersize=1.5, label="Cross-track departure")
    ax.axvline(FOCUS_START, linewidth=0.5, linestyle=":", alpha=0.8)
    ax.axvline(FOCUS_STOP, linewidth=0.5, linestyle=":", alpha=0.8)
    ax.set_title("Earth Orbital-Plane Signed Motion — Verified 100-Year Singleton JPL Context\nFixed reference plane: astronomical year −9000", color="white", fontsize=14, weight="bold", pad=15)
    ax.set_xlabel("Astronomical year", color="white")
    ax.set_ylabel("Signed angular displacement (degrees)", color="white")
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()
    fig.savefig(PNG_CONTEXT, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_focus(focus: pd.DataFrame, comparison: pd.DataFrame, inflection: dict) -> None:
    fig, ax = plt.subplots(figsize=(17, 8), dpi=150)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(focus["astronomical_year"], focus["signed_along_track_deg"], linewidth=0.85, label="Annual continuous-range JPL solution")
    ax.scatter(comparison["astronomical_year"], comparison["signed_along_track_deg_singleton"], s=7, label="Independent 5-year singleton checks")
    ax.scatter(comparison["astronomical_year"], comparison["signed_along_track_deg_overlap"], s=5, marker="x", label="Overlapping explicit-JD batch checks")
    for value in [float(v.strip()) for v in inflection["candidate_years_3500_4000"].split(",") if v.strip() not in {"", "NONE"}]:
        ax.axvline(value, linewidth=0.6, linestyle="--", alpha=0.8)
    ax.set_title("Earth Orbital-Plane Signed Motion — A.D. 3000 to 4500 Data-Integrity Audit\nAnnual range plus independent singleton and overlapping-batch verification", color="white", fontsize=14, weight="bold", pad=15)
    ax.set_xlabel("Astronomical year", color="white")
    ax.set_ylabel("Signed along-track displacement (degrees)", color="white")
    legend = ax.legend(frameon=False, fontsize=8.5)
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()
    fig.savefig(PNG_FOCUS, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_derivatives(focus: pd.DataFrame) -> None:
    valid = focus.dropna(subset=["first_derivative_arcsec_per_century", "second_derivative_arcsec_per_century2"])
    fig, ax = plt.subplots(figsize=(17, 8), dpi=150)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(valid["astronomical_year"], valid["first_derivative_arcsec_per_century"], linewidth=0.8, label="First derivative (arcsec/century)")
    ax.plot(valid["astronomical_year"], valid["second_derivative_arcsec_per_century2"], linewidth=0.7, label="Second derivative (arcsec/century²)")
    ax.axhline(0.0, linewidth=0.5, alpha=0.7)
    ax.axvspan(AUDIT_START, AUDIT_STOP, alpha=0.08)
    ax.set_title("Earth Orbital-Plane Signed Motion Derivatives\nLocal cubic fits over 101-year windows", color="white", fontsize=14, weight="bold", pad=15)
    ax.set_xlabel("Astronomical year", color="white")
    ax.set_ylabel("Derivative value", color="white")
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()
    fig.savefig(PNG_DERIV, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_table(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(17, 5.2), dpi=160)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.axis("off")
    table = ax.table(cellText=summary.values, colLabels=summary.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#777777")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#17324d")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#1c1c1c" if row % 2 else "#282828")
            cell.get_text().set_color("white")
            if column == 1:
                cell.get_text().set_weight("bold")
    ax.set_title("Earth JPL Apparent-Inflection Data-Integrity Verdict", color="white", fontsize=15, weight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(PNG_TABLE, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"OUTPUT VERSION {VERSION}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    context = fetch_context()
    n0, u, w = build_basis(context)

    focus = fetch_focus_range()
    focus = prepare_focus(focus, n0, u, w)
    focus.to_csv(FOCUS_CSV, index=False, float_format="%.12f")

    audit_years = np.arange(AUDIT_START, AUDIT_STOP + AUDIT_STEP, AUDIT_STEP, dtype=float)
    singleton = fetch_singletons(audit_years, "audit_singleton")
    overlap = fetch_overlap_batches(audit_years)
    comparison = compare_methods(singleton, overlap, n0, u, w)
    comparison.to_csv(AUDIT_CSV, index=False, float_format="%.12f")

    inflection = detect_inflection(focus)
    max_normal_mas = float(np.nanmax(comparison["normal_difference_milliarcsec"]))
    max_along_mas = float(np.nanmax(np.abs(comparison["along_difference_milliarcsec"])))
    max_jd_error = float(max(singleton["jd_error_days"].max(), overlap["jd_error_days"].max()))
    overlap_spread = float(overlap.filter(regex="_spread$").to_numpy(dtype=float).max())

    verified = max_normal_mas < 1.0 and max_along_mas < 1.0 and overlap_spread < 1.0e-10
    verdict = "FETCH METHODS AGREE" if verified else "FETCH DISAGREEMENT DETECTED"
    feature = "CURVATURE SIGN CHANGE PRESENT" if inflection["candidate_count_3500_4000"] > 0 else "NO VERIFIED INFLECTION IN 3500–4000"

    summary = pd.DataFrame([
        ["Fetch integrity", verdict],
        ["Maximum normal difference", f"{max_normal_mas:.6f} milliarcsec"],
        ["Maximum along-track difference", f"{max_along_mas:.6f} milliarcsec"],
        ["Maximum JD match error", f"{max_jd_error:.12e} days"],
        ["Maximum overlap replicate spread", f"{overlap_spread:.12e}"],
        ["Inflection audit", feature],
        ["Candidate year(s)", inflection["candidate_years_3500_4000"]],
        ["Reference plane", "Verified singleton Earth plane at astronomical year −9000"],
    ], columns=["Audit item", "Result"])
    summary.to_csv(SUMMARY_CSV, index=False)

    plot_context(context, n0, u, w)
    plot_focus(focus, comparison, inflection)
    plot_derivatives(focus)
    plot_table(summary)

    display(Image(filename=str(PNG_CONTEXT)))
    display(Image(filename=str(PNG_FOCUS)))
    display(Image(filename=str(PNG_DERIV)))
    display(Image(filename=str(PNG_TABLE)))
    print(summary.to_string(index=False))
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0026