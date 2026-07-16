# V0025
# Audit reference: Earth JPL orbital-plane apparent-inflection test using JD-matched overlapping batches and singleton validation.
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
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


for module_name, package_name in [
    ("astroquery", "astroquery"), ("numpy", "numpy"), ("pandas", "pandas"),
    ("matplotlib", "matplotlib"), ("IPython", "ipython")
]:
    need(module_name, package_name)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display

VERSION = "V0025"
LOCAL_TZ = ZoneInfo("America/Bogota")
EARTH_ID = "399"
CENTER = "@10"
REFPLANE = "ecliptic"
ABERRATIONS = "geometric"
AU_KM = 149597870.700
J2000_JD = 2451545.0
DAYS_PER_JULIAN_YEAR = 365.25
FOCUS_START = 3000
FOCUS_STOP = 4500
FOCUS_STEP = 1
CONTEXT_START = -9000
CONTEXT_STOP = 9000
CONTEXT_STEP = 100
BATCH_SIZE = 64
OVERLAP = 8
SINGLETON_STEP = 25
MAX_RETRIES = 4
RETRY_DELAY_SECONDS = 3.0
JD_MATCH_TOLERANCE_DAYS = 1.0e-5

OUT = Path("/content/EARTH_JPL_INFLECTION_DATA_INTEGRITY_AUDIT_V0025_OUTPUT")
FOCUS_CSV = OUT / "EARTH_JPL_FOCUS_3000_4500_AUDIT_V0025.csv"
CONTEXT_CSV = OUT / "EARTH_JPL_CONTEXT_100YEAR_V0025.csv"
SUMMARY_CSV = OUT / "EARTH_JPL_INFLECTION_AUDIT_SUMMARY_V0025.csv"
PNG_FOCUS = OUT / "EARTH_JPL_INFLECTION_FOCUS_AUDIT_V0025.png"
PNG_DERIV = OUT / "EARTH_JPL_INFLECTION_DERIVATIVES_V0025.png"
PNG_CONTEXT = OUT / "EARTH_JPL_SIGNED_MOTION_100YEAR_CONTEXT_V0025.png"
PNG_TABLE = OUT / "EARTH_JPL_INFLECTION_AUDIT_TABLE_V0025.png"


def astronomical_year_to_jd(year: float) -> float:
    return J2000_JD + (float(year) - 2000.0) * DAYS_PER_JULIAN_YEAR


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("REJECTED invalid vector norm")
    return vector / norm


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise RuntimeError("REJECTED invalid vector rows")
    return vectors / norms[:, None]


def horizons_query(jds: np.ndarray) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                table = Horizons(id=EARTH_ID, id_type="majorbody", location=CENTER,
                                 epochs=[float(x) for x in jds]).vectors(
                                     refplane=REFPLANE, aberrations=ABERRATIONS)
            frame = table.to_pandas()
            required = ["datetime_jd", "x", "y", "z", "vx", "vy", "vz"]
            if any(c not in frame.columns for c in required):
                raise RuntimeError("REJECTED missing JPL columns")
            return frame[required].copy()
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * attempt)
    raise RuntimeError(f"JPL query failed after {MAX_RETRIES} attempts: {last_error}")


def match_rows(requested_years: np.ndarray, returned: pd.DataFrame, method: str) -> pd.DataFrame:
    requested_jds = np.array([astronomical_year_to_jd(y) for y in requested_years], dtype=float)
    ret = returned.sort_values("datetime_jd").drop_duplicates("datetime_jd").reset_index(drop=True)
    ret_jds = ret["datetime_jd"].to_numpy(dtype=float)
    rows: list[dict] = []
    used: set[int] = set()
    for year, jd in zip(requested_years, requested_jds):
        idx = int(np.argmin(np.abs(ret_jds - jd)))
        error = float(abs(ret_jds[idx] - jd))
        if error > JD_MATCH_TOLERANCE_DAYS:
            raise RuntimeError(f"REJECTED JD mismatch {method} year={year} error_days={error}")
        if idx in used:
            raise RuntimeError(f"REJECTED duplicate returned row {method} year={year}")
        used.add(idx)
        row = ret.iloc[idx]
        rows.append({"astronomical_year": float(year), "requested_jd": jd,
                     "returned_jd": float(row["datetime_jd"]), "jd_error_days": error,
                     "x": float(row["x"]), "y": float(row["y"]), "z": float(row["z"]),
                     "vx": float(row["vx"]), "vy": float(row["vy"]), "vz": float(row["vz"]),
                     "method": method})
    return pd.DataFrame(rows)


def fetch_batched(years: np.ndarray, offset: int, method: str) -> pd.DataFrame:
    all_rows: list[pd.DataFrame] = []
    start = max(0, offset)
    if start > 0:
        first = years[:start]
        if len(first):
            all_rows.append(match_rows(first, horizons_query(np.array([astronomical_year_to_jd(y) for y in first])), method))
    stride = BATCH_SIZE - OVERLAP
    i = start
    while i < len(years):
        subset = years[i:min(i + BATCH_SIZE, len(years))]
        jds = np.array([astronomical_year_to_jd(y) for y in subset], dtype=float)
        all_rows.append(match_rows(subset, horizons_query(jds), method))
        if i + BATCH_SIZE >= len(years):
            break
        i += stride
    frame = pd.concat(all_rows, ignore_index=True)
    numeric_cols = ["requested_jd", "returned_jd", "jd_error_days", "x", "y", "z", "vx", "vy", "vz"]
    frame = frame.groupby("astronomical_year", as_index=False)[numeric_cols].mean()
    frame["method"] = method
    expected = len(years)
    if len(frame) != expected:
        raise RuntimeError(f"REJECTED {method} sample count expected={expected} received={len(frame)}")
    return frame.sort_values("astronomical_year").reset_index(drop=True)


def fetch_singletons(years: np.ndarray) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for year in years:
        jd = np.array([astronomical_year_to_jd(float(year))], dtype=float)
        rows.append(match_rows(np.array([year], dtype=float), horizons_query(jd), "singleton"))
    return pd.concat(rows, ignore_index=True).sort_values("astronomical_year").reset_index(drop=True)


def normals_from_states(frame: pd.DataFrame) -> np.ndarray:
    r = frame[["x", "y", "z"]].to_numpy(dtype=float) * AU_KM
    v = frame[["vx", "vy", "vz"]].to_numpy(dtype=float) * AU_KM
    normals = normalize_rows(np.cross(r, v))
    normals[normals[:, 2] < 0.0] *= -1.0
    return normals


def build_basis(reference_states: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normals = normals_from_states(reference_states)
    n0 = normalize(normals[0])
    n1 = normalize(normals[-1])
    u = normalize(n1 - np.dot(n1, n0) * n0)
    w = normalize(np.cross(n0, u))
    return n0, u, w


def project(frame: pd.DataFrame, n0: np.ndarray, u: np.ndarray, w: np.ndarray, prefix: str) -> pd.DataFrame:
    normals = normals_from_states(frame)
    years = frame["astronomical_year"].to_numpy(dtype=float)
    along = np.degrees(np.arctan2(normals @ u, normals @ n0))
    cross = np.degrees(np.arcsin(np.clip(normals @ w, -1.0, 1.0)))
    result = pd.DataFrame({"astronomical_year": years,
                           f"along_{prefix}_deg": along,
                           f"cross_{prefix}_deg": cross,
                           f"jd_error_{prefix}_days": frame["jd_error_days"].to_numpy(dtype=float)})
    return result


def style_axis(ax) -> None:
    ax.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.5)
    ax.grid(True, linewidth=0.35, alpha=0.22)


def plot_focus(data: pd.DataFrame, verified_years: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(17, 8.5), dpi=150)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(data["astronomical_year"], data["along_a_deg"], linewidth=0.8, label="Overlapping batches A")
    ax.plot(data["astronomical_year"], data["along_b_deg"], linewidth=0.55, linestyle="--", label="Shifted overlapping batches B")
    sparse = data[np.isfinite(data["along_singleton_deg"])]
    ax.scatter(sparse["astronomical_year"], sparse["along_singleton_deg"], s=7, label="Independent singleton validation")
    for year in verified_years:
        ax.axvline(year, linewidth=0.5, linestyle=":", alpha=0.8)
    ax.set_title("Earth JPL Orbital-Plane Motion — A.D. 3000 to 4500 Data-Integrity Audit\nJD-matched overlapping batches and independent singleton validation",
                 color="white", fontsize=14, weight="bold", pad=16)
    ax.set_xlabel("Astronomical year", color="white")
    ax.set_ylabel("Signed along-track plane displacement (degrees)", color="white")
    legend = ax.legend(frameon=False, fontsize=8.5, ncol=3)
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()
    fig.savefig(PNG_FOCUS, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_derivatives(data: pd.DataFrame, verified_years: list[float]) -> None:
    x = data["astronomical_year"].to_numpy(dtype=float)
    y = data["along_consensus_deg"].to_numpy(dtype=float)
    first = np.gradient(y, x) * 360000.0
    second = np.gradient(np.gradient(y, x), x) * 360000.0
    fig, ax = plt.subplots(figsize=(17, 8.5), dpi=150)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(x, first, linewidth=0.75, label="First derivative (arcsec/century)")
    ax.plot(x, second, linewidth=0.65, label="Second derivative scaled consistently")
    ax.axhline(0.0, linewidth=0.45, alpha=0.7)
    for year in verified_years:
        ax.axvline(year, linewidth=0.5, linestyle=":", alpha=0.8)
    ax.set_title("Earth JPL Signed Plane Motion — First and Second Derivative Audit",
                 color="white", fontsize=14, weight="bold", pad=16)
    ax.set_xlabel("Astronomical year", color="white")
    ax.set_ylabel("Derivative diagnostic", color="white")
    legend = ax.legend(frameon=False, fontsize=8.5)
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()
    fig.savefig(PNG_DERIV, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_context(context: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(17, 8.5), dpi=150)
    fig.patch.set_facecolor("black")
    style_axis(ax)
    ax.plot(context["astronomical_year"], context["along_context_deg"], linewidth=0.8, marker="o", markersize=1.8,
            label="Signed along-track displacement — 100-year cadence")
    ax.plot(context["astronomical_year"], context["cross_context_deg"], linewidth=0.65, marker="o", markersize=1.5,
            label="Cross-track departure — 100-year cadence")
    ax.axvspan(FOCUS_START, FOCUS_STOP, alpha=0.08)
    ax.set_title("Earth Orbital-Plane Signed Motion — Fixed −9000-Year Start Reference\nJPL explicit-JD data displayed every 100 years; no interval-mean construction",
                 color="white", fontsize=14, weight="bold", pad=16)
    ax.set_xlabel("Astronomical year", color="white")
    ax.set_ylabel("Signed angular displacement (degrees)", color="white")
    legend = ax.legend(frameon=False, fontsize=8.5)
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()
    fig.savefig(PNG_CONTEXT, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_table(summary: pd.DataFrame) -> None:
    display_table = summary.copy()
    fig, ax = plt.subplots(figsize=(17, 4.8), dpi=160)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.axis("off")
    tbl = ax.table(cellText=display_table.values, colLabels=display_table.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.65)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#777777")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#17324d")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#1b1b1b" if row % 2 else "#252525")
            cell.get_text().set_color("white")
    ax.set_title("Earth JPL Apparent-Inflection Data-Integrity Audit", color="white", fontsize=15, weight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(PNG_TABLE, dpi=360, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"OUTPUT VERSION {VERSION}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    focus_years = np.arange(FOCUS_START, FOCUS_STOP + FOCUS_STEP, FOCUS_STEP, dtype=float)
    context_years = np.arange(CONTEXT_START, CONTEXT_STOP + CONTEXT_STEP, CONTEXT_STEP, dtype=float)
    singleton_years = np.arange(FOCUS_START, FOCUS_STOP + SINGLETON_STEP, SINGLETON_STEP, dtype=float)

    focus_a = fetch_batched(focus_years, 0, "batch_a")
    focus_b = fetch_batched(focus_years, 17, "batch_b_shifted")
    singleton = fetch_singletons(singleton_years)
    context_states = fetch_batched(context_years, 0, "context")

    n0, u, w = build_basis(context_states)
    pa = project(focus_a, n0, u, w, "a")
    pb = project(focus_b, n0, u, w, "b")
    ps = project(singleton, n0, u, w, "singleton")
    pc = project(context_states, n0, u, w, "context")

    focus = pa.merge(pb, on="astronomical_year", how="inner").merge(ps, on="astronomical_year", how="left")
    focus["along_consensus_deg"] = 0.5 * (focus["along_a_deg"] + focus["along_b_deg"])
    focus["cross_consensus_deg"] = 0.5 * (focus["cross_a_deg"] + focus["cross_b_deg"])
    focus["batch_difference_arcsec"] = np.abs(focus["along_a_deg"] - focus["along_b_deg"]) * 3600.0
    focus["singleton_difference_arcsec"] = np.abs(focus["along_a_deg"] - focus["along_singleton_deg"]) * 3600.0

    x = focus["astronomical_year"].to_numpy(dtype=float)
    y = focus["along_consensus_deg"].to_numpy(dtype=float)
    first = np.gradient(y, x)
    second = np.gradient(first, x)
    focus["first_derivative_arcsec_per_century"] = first * 360000.0
    focus["second_derivative_arcsec_per_century2"] = second * 360000.0

    sign = np.sign(second)
    candidates = np.where(sign[1:] * sign[:-1] < 0.0)[0]
    verified_years = [float((x[i] + x[i + 1]) / 2.0) for i in candidates if 3400 <= x[i] <= 4100]

    max_batch = float(np.nanmax(focus["batch_difference_arcsec"]))
    max_single = float(np.nanmax(focus["singleton_difference_arcsec"]))
    max_jd = float(max(focus_a["jd_error_days"].max(), focus_b["jd_error_days"].max(), singleton["jd_error_days"].max()))
    status = "PHYSICAL CURVATURE CANDIDATE" if verified_years and max_batch < 1.0e-3 and max_single < 1.0e-3 else "BATCHING ARTIFACT OR UNVERIFIED"

    summary = pd.DataFrame([
        {"Metric": "Audit status", "Result": status},
        {"Metric": "Verified curvature sign-change years", "Result": ", ".join(f"{v:.1f}" for v in verified_years) if verified_years else "NONE"},
        {"Metric": "Maximum A-vs-B difference", "Result": f"{max_batch:.9f} arcsec"},
        {"Metric": "Maximum singleton difference", "Result": f"{max_single:.9f} arcsec"},
        {"Metric": "Maximum JD match error", "Result": f"{max_jd:.12e} days"},
        {"Metric": "Focus sampling", "Result": "Annual, A.D. 3000–4500"},
        {"Metric": "Context display sampling", "Result": "100 years, −9000 to +9000"},
    ])

    focus.to_csv(FOCUS_CSV, index=False, float_format="%.12f")
    pc.to_csv(CONTEXT_CSV, index=False, float_format="%.12f")
    summary.to_csv(SUMMARY_CSV, index=False)

    plot_focus(focus, verified_years)
    plot_derivatives(focus, verified_years)
    plot_context(pc)
    plot_table(summary)

    display(Image(filename=str(PNG_FOCUS)))
    display(Image(filename=str(PNG_DERIV)))
    display(Image(filename=str(PNG_CONTEXT)))
    display(Image(filename=str(PNG_TABLE)))
    print(summary.to_string(index=False))
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0025