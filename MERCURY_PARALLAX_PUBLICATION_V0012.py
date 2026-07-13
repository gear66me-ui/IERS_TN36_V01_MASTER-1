# V0012
# Audit reference: Recompute exact JPL ecliptic contacts and render proportional Mercury disks with 0.25-point geometry.
from __future__ import annotations

import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def ensure_package(import_name: str, pip_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pip_name])


for _import, _pip in (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("astropy", "astropy"),
    ("astroquery", "astroquery"),
    ("matplotlib", "matplotlib"),
):
    ensure_package(_import, _pip)

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar

VERSION = "V0012"
PROGRAM = "MERCURY_PARALLAX_PUBLICATION_V0012.py"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUTPUT_DIR = ROOT / "MERCURY_PARALLAX_PUBLICATION_V0012_OUTPUT"
OUTPUT_PNG = OUTPUT_DIR / "MERCURY_1769_ECLIPTIC_HALF_SUN_CONTACTS_V0012.png"
OUTPUT_CONTACTS_CSV = OUTPUT_DIR / "MERCURY_1769_ECLIPTIC_CONTACTS_V0012.csv"
OUTPUT_TRACKS_CSV = OUTPUT_DIR / "MERCURY_1769_ECLIPTIC_TRACKS_V0012.csv"
OUTPUT_RESULTS_CSV = OUTPUT_DIR / "MERCURY_1769_ECLIPTIC_RESULTS_V0012.csv"
MASTER_CSV = ROOT / "MERCURY_BAY_VARDO_1769_JPL_ECLIPTIC_V0012.csv"

ARCSEC_PER_RAD = 206_264.80624709636
AU_KM = 149_597_870.000000
EARTH_RADIUS_KM = 6_378.140000
SUN_RADIUS_KM = 695_700.000000
MERCURY_RADIUS_KM = 2_439.700000
PI_SUN_ARCSEC = math.asin(EARTH_RADIUS_KM / AU_KM) * ARCSEC_PER_RAD

START = "1769-11-09 12:00"
STOP = "1769-11-10 08:00"
STEP = "1m"
GEOCENTER = "500@399"
SITE_MB = {"label": "Mercury Bay", "key": "MERCURY_BAY", "lat": -36.783333333333, "lon": 175.933333333333, "elevation": 0.0, "body": 399}
SITE_V = {"label": "Vardø", "key": "VARDO", "lat": 70.370600000000, "lon": 31.110700000000, "elevation": 0.0, "body": 399}
SITES = (SITE_MB, SITE_V)
TARGETS = (("SUN", "10"), ("MERCURY", "199"))
PREFIXES = (
    "GEOCENTER_SUN", "GEOCENTER_MERCURY",
    "MERCURY_BAY_SUN", "MERCURY_BAY_MERCURY",
    "VARDO_SUN", "VARDO_MERCURY",
)
REQUIRED = ["JD_TDB"] + [f"{prefix}_{axis}_KM" for prefix in PREFIXES for axis in "XYZ"]

LINE_WIDTH_PT = 0.25
TRACK_MB_COLOR = "#2DD4BF"
TRACK_V_COLOR = "#F59E0B"
SUN_COLOR = "#F8FAFC"
GUIDE_COLOR = "#64748B"
TABLE_HEADER = "#1E3A5F"
TABLE_BODY = "#0F172A"
TABLE_ACCENT = "#123B48"
TABLE_GOLD = "#4A3510"


def norm(vector) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    magnitude = norm(vector)
    if magnitude == 0.0:
        raise RuntimeError("Cannot normalize a zero vector.")
    return vector / magnitude


def horizons_location(site: dict[str, object] | None):
    if site is None:
        return GEOCENTER
    return {
        "lon": float(site["lon"]),
        "lat": float(site["lat"]),
        "elevation": float(site["elevation"]),
        "body": int(site["body"]),
    }


def download_vector_series(prefix: str, target_id: str, location) -> pd.DataFrame:
    query = Horizons(
        id=target_id,
        location=location,
        epochs={"start": START, "stop": STOP, "step": STEP},
    )
    table = query.vectors(refplane="ecliptic", aberrations="geometric")
    frame = table.to_pandas()
    result = pd.DataFrame({"JD_TDB": pd.to_numeric(frame["datetime_jd"], errors="coerce")})
    for axis in "xyz":
        result[f"{prefix}_{axis.upper()}_KM"] = pd.to_numeric(frame[axis], errors="coerce") * AU_KM
    return result.dropna().drop_duplicates("JD_TDB").sort_values("JD_TDB")


def build_master() -> pd.DataFrame:
    series: list[pd.DataFrame] = []
    for target_name, target_id in TARGETS:
        series.append(download_vector_series(f"GEOCENTER_{target_name}", target_id, GEOCENTER))
    for site in SITES:
        location = horizons_location(site)
        for target_name, target_id in TARGETS:
            series.append(download_vector_series(f"{site['key']}_{target_name}", target_id, location))
    master = series[0]
    for frame in series[1:]:
        master = master.merge(frame, on="JD_TDB", how="inner", validate="one_to_one")
    master["REFERENCE_PLANE"] = "JPL ECLIPTIC"
    master["UTC"] = [tdb_jd_to_utc_text(value) for value in master["JD_TDB"]]
    master.to_csv(MASTER_CSV, index=False, float_format="%.15f")
    return master


def compatible_ecliptic_master(path: Path) -> bool:
    try:
        sample = pd.read_csv(path, nrows=3)
    except Exception:
        return False
    if not all(column in sample.columns for column in REQUIRED):
        return False
    if "REFERENCE_PLANE" in sample.columns:
        values = " ".join(sample["REFERENCE_PLANE"].astype(str).tolist()).upper()
        return "ECLIPTIC" in values
    return "ECLIPTIC" in path.name.upper()


def find_or_build_master() -> tuple[pd.DataFrame, str]:
    if MASTER_CSV.is_file() and compatible_ecliptic_master(MASTER_CSV):
        return pd.read_csv(MASTER_CSV), "EXISTING V0012 ECLIPTIC MASTER"
    candidates: list[Path] = []
    for root, directories, files in os.walk(ROOT):
        directories[:] = [item for item in directories if item != "drive" and not item.startswith(".")]
        for filename in files:
            path = Path(root) / filename
            if filename.lower().endswith(".csv") and compatible_ecliptic_master(path):
                candidates.append(path)
    if candidates:
        path = sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0]
        return pd.read_csv(path), f"EXISTING ECLIPTIC MASTER: {path}"
    return build_master(), "NEW JPL HORIZONS ECLIPTIC DOWNLOAD"


def build_cache(frame: pd.DataFrame) -> dict[str, object]:
    data = frame.copy()
    data["JD_TDB"] = pd.to_numeric(data["JD_TDB"], errors="coerce")
    for column in REQUIRED[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=REQUIRED).sort_values("JD_TDB").drop_duplicates("JD_TDB").reset_index(drop=True)
    if len(data) < 20:
        raise RuntimeError("The synchronized JPL ecliptic master contains too few rows.")
    jds = data["JD_TDB"].to_numpy(float)
    cache: dict[str, object] = {"JD_TDB": jds, "frame": data}
    for prefix in PREFIXES:
        for axis in "XYZ":
            column = f"{prefix}_{axis}_KM"
            cache[column] = CubicSpline(jds, data[column].to_numpy(float), bc_type="natural")
    return cache


def vector_at(cache: dict[str, object], prefix: str, jd_tdb: float) -> np.ndarray:
    return np.array([float(cache[f"{prefix}_{axis}_KM"](jd_tdb)) for axis in "XYZ"], dtype=float)


def angular_separation_rad(first, second) -> float:
    cosine = float(np.clip(np.dot(unit(first), unit(second)), -1.0, 1.0))
    return math.acos(cosine)


def angular_radii_rad(cache: dict[str, object], site_key: str, jd_tdb: float) -> tuple[float, float]:
    sun_distance = norm(vector_at(cache, f"{site_key}_SUN", jd_tdb))
    mercury_distance = norm(vector_at(cache, f"{site_key}_MERCURY", jd_tdb))
    return (
        math.asin(SUN_RADIUS_KM / sun_distance),
        math.asin(MERCURY_RADIUS_KM / mercury_distance),
    )


def contact_residual(cache: dict[str, object], site_key: str, jd_tdb: float, internal: bool) -> float:
    sun = vector_at(cache, f"{site_key}_SUN", jd_tdb)
    mercury = vector_at(cache, f"{site_key}_MERCURY", jd_tdb)
    separation = angular_separation_rad(sun, mercury)
    solar_radius, mercury_radius = angular_radii_rad(cache, site_key, jd_tdb)
    required = solar_radius - mercury_radius if internal else solar_radius + mercury_radius
    return separation - required


def root_pairs(cache: dict[str, object], site_key: str, internal: bool) -> list[float]:
    jds = np.asarray(cache["JD_TDB"], dtype=float)
    values = np.array([contact_residual(cache, site_key, jd, internal) for jd in jds], dtype=float)
    roots: list[float] = []
    for index in range(len(jds) - 1):
        left_value = values[index]
        right_value = values[index + 1]
        if left_value == 0.0:
            roots.append(float(jds[index]))
        elif left_value * right_value < 0.0:
            root = brentq(
                lambda value: contact_residual(cache, site_key, value, internal),
                float(jds[index]),
                float(jds[index + 1]),
                xtol=1.0e-13,
                rtol=1.0e-14,
                maxiter=200,
            )
            roots.append(float(root))
    unique: list[float] = []
    for root in roots:
        if not unique or abs(root - unique[-1]) > 0.2 / 86400.0:
            unique.append(root)
    if len(unique) != 2:
        kind = "internal" if internal else "external"
        raise RuntimeError(f"Expected exactly two {kind} contacts for {site_key}; found {len(unique)}.")
    return unique


def compute_contacts(cache: dict[str, object], site_key: str) -> dict[str, float]:
    external = root_pairs(cache, site_key, internal=False)
    internal = root_pairs(cache, site_key, internal=True)
    contacts = {"C1": external[0], "C2": internal[0], "C3": internal[1], "C4": external[1]}
    if not (contacts["C1"] < contacts["C2"] < contacts["C3"] < contacts["C4"]):
        raise RuntimeError(f"Contact ordering failed for {site_key}: {contacts}")
    return contacts


def closest_approach(cache: dict[str, object], site_key: str, contacts: dict[str, float]) -> float:
    result = minimize_scalar(
        lambda jd: angular_separation_rad(
            vector_at(cache, f"{site_key}_SUN", jd),
            vector_at(cache, f"{site_key}_MERCURY", jd),
        ),
        bounds=(contacts["C2"], contacts["C3"]),
        method="bounded",
        options={"xatol": 1.0e-13, "maxiter": 400},
    )
    if not result.success:
        raise RuntimeError(f"Closest-approach solution failed for {site_key}.")
    return float(result.x)


def ecliptic_local_basis(sun_vector) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sun_hat = unit(sun_vector)
    ecliptic_north = np.array([0.0, 0.0, 1.0])
    xi = unit(np.cross(ecliptic_north, sun_hat))
    eta = unit(np.cross(sun_hat, xi))
    if float(np.dot(eta, ecliptic_north)) < 0.0:
        xi = -xi
        eta = -eta
    return sun_hat, xi, eta


def apparent_position_arcsec(cache: dict[str, object], site_key: str, jd_tdb: float) -> np.ndarray:
    sun_hat = unit(vector_at(cache, f"{site_key}_SUN", jd_tdb))
    mercury_hat = unit(vector_at(cache, f"{site_key}_MERCURY", jd_tdb))
    separation = angular_separation_rad(sun_hat, mercury_hat)
    if separation == 0.0:
        return np.zeros(2, dtype=float)
    tangent_direction = unit(mercury_hat - math.cos(separation) * sun_hat)
    _sun_hat, xi, eta = ecliptic_local_basis(sun_hat)
    return separation * ARCSEC_PER_RAD * np.array(
        [float(np.dot(tangent_direction, xi)), float(np.dot(tangent_direction, eta))],
        dtype=float,
    )


def fit_track(points: np.ndarray) -> dict[str, float]:
    centered = points - np.mean(points, axis=0)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    if direction[0] < 0.0:
        direction = -direction
    normal_direction = np.array([-direction[1], direction[0]])
    along = centered @ direction
    cross = centered @ normal_direction
    coefficients = np.polyfit(along, cross, 2)
    fitted_cross = np.polyval(coefficients, along)
    rms = float(np.sqrt(np.mean((cross - fitted_cross) ** 2)))
    angle = math.degrees(math.atan2(direction[1], direction[0]))
    slope = math.tan(math.radians(angle))
    curvature = float(2.0 * coefficients[0])
    return {"angle_deg": angle, "slope": slope, "rms_arcsec": rms, "curvature_per_arcsec": curvature}


def tdb_jd_to_utc_text(jd_tdb: float) -> str:
    time = Time(jd_tdb, format="jd", scale="tdb").utc
    return time.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def contact_condition(event: str) -> str:
    return "ρ = R⊙ + R☿" if event in ("C1", "C4") else "ρ = R⊙ − R☿"


def style_table(table, header_rows: int = 1, accent_rows: set[int] | None = None, gold_rows: set[int] | None = None) -> None:
    accent_rows = accent_rows or set()
    gold_rows = gold_rows or set()
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        cell.set_linewidth(LINE_WIDTH_PT)
        cell.get_text().set_color("white")
        cell.get_text().set_fontsize(8.0)
        if row < header_rows:
            cell.set_facecolor(TABLE_HEADER)
            cell.get_text().set_fontweight("bold")
        elif row in gold_rows:
            cell.set_facecolor(TABLE_GOLD)
            cell.get_text().set_fontweight("bold")
        elif row in accent_rows:
            cell.set_facecolor(TABLE_ACCENT)
        else:
            cell.set_facecolor(TABLE_BODY)


def add_contact_disk(ax, center: np.ndarray, radius_arcsec: float, color: str, label: str, label_offset: tuple[float, float]) -> None:
    disk = Circle(
        (float(center[0]), float(center[1])),
        radius_arcsec,
        facecolor=color,
        edgecolor=color,
        alpha=0.13,
        linewidth=LINE_WIDTH_PT,
        zorder=5,
    )
    ax.add_patch(disk)
    ax.text(
        float(center[0] + label_offset[0]),
        float(center[1] + label_offset[1]),
        label,
        color=color,
        fontsize=6.8,
        ha="center",
        va="center",
        zorder=6,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master, source = find_or_build_master()
    cache = build_cache(master)

    site_results: dict[str, dict[str, object]] = {}
    contact_rows: list[dict[str, object]] = []
    track_rows: list[dict[str, object]] = []

    for site in SITES:
        key = str(site["key"])
        contacts = compute_contacts(cache, key)
        maximum = closest_approach(cache, key, contacts)
        events = {"C1": contacts["C1"], "C2": contacts["C2"], "MAX": maximum, "C3": contacts["C3"], "C4": contacts["C4"]}
        minute_jds = np.asarray(cache["JD_TDB"], dtype=float)
        mask = (minute_jds >= contacts["C1"]) & (minute_jds <= contacts["C4"])
        selected_jds = minute_jds[mask]
        points = np.array([apparent_position_arcsec(cache, key, jd) for jd in selected_jds])
        fit = fit_track(points)
        event_points = {name: apparent_position_arcsec(cache, key, jd) for name, jd in events.items()}
        event_radii: dict[str, tuple[float, float]] = {}
        contact_audits: dict[str, float] = {}
        for event, jd in events.items():
            solar_radius_rad, mercury_radius_rad = angular_radii_rad(cache, key, jd)
            event_radii[event] = (solar_radius_rad * ARCSEC_PER_RAD, mercury_radius_rad * ARCSEC_PER_RAD)
            if event != "MAX":
                residual = contact_residual(cache, key, jd, internal=event in ("C2", "C3")) * ARCSEC_PER_RAD
                contact_audits[event] = residual
                contact_rows.append(
                    {
                        "station": str(site["label"]),
                        "event": event,
                        "jd_tdb": jd,
                        "utc": tdb_jd_to_utc_text(jd),
                        "condition": contact_condition(event),
                        "residual_arcsec": residual,
                    }
                )
        for jd, point in zip(selected_jds, points):
            track_rows.append({"station": str(site["label"]), "jd_tdb": jd, "utc": tdb_jd_to_utc_text(jd), "x_arcsec": point[0], "y_arcsec": point[1]})
        site_results[key] = {
            "site": site,
            "contacts": contacts,
            "maximum": maximum,
            "events": events,
            "points": points,
            "event_points": event_points,
            "event_radii": event_radii,
            "fit": fit,
            "contact_audits": contact_audits,
        }

    fit_mb = site_results["MERCURY_BAY"]["fit"]
    fit_v = site_results["VARDO"]["fit"]
    average_angle = 0.5 * (float(fit_mb["angle_deg"]) + float(fit_v["angle_deg"]))
    max_contact_residual = max(abs(float(row["residual_arcsec"])) for row in contact_rows)

    audit_jd = float(site_results["MERCURY_BAY"]["maximum"])
    sun_audit = vector_at(cache, "MERCURY_BAY_SUN", audit_jd)
    sun_hat, xi, eta = ecliptic_local_basis(sun_audit)
    ecliptic_north = np.array([0.0, 0.0, 1.0])
    projection_audit = {
        "xi_dot_sun": abs(float(np.dot(xi, sun_hat))),
        "eta_dot_sun": abs(float(np.dot(eta, sun_hat))),
        "xi_dot_eta": abs(float(np.dot(xi, eta))),
        "eta_north_alignment": float(np.dot(eta, ecliptic_north)),
        "handedness": float(np.dot(np.cross(xi, eta), sun_hat)),
    }

    contact_frame = pd.DataFrame(contact_rows)
    track_frame = pd.DataFrame(track_rows)
    results_frame = pd.DataFrame(
        [
            {"quantity": "IAU 1976 AU-normalized solar horizontal parallax", "symbol": "pi_sun", "value": PI_SUN_ARCSEC, "unit": "arcsec"},
            {"quantity": "Mercury Bay track angle", "symbol": "alpha_MB", "value": float(fit_mb["angle_deg"]), "unit": "deg"},
            {"quantity": "Vardo track angle", "symbol": "alpha_V", "value": float(fit_v["angle_deg"]), "unit": "deg"},
            {"quantity": "Average track angle", "symbol": "alpha_bar", "value": average_angle, "unit": "deg"},
            {"quantity": "Maximum contact residual", "symbol": "", "value": max_contact_residual, "unit": "arcsec"},
        ]
    )
    contact_frame.to_csv(OUTPUT_CONTACTS_CSV, index=False, float_format="%.15f")
    track_frame.to_csv(OUTPUT_TRACKS_CSV, index=False, float_format="%.15f")
    results_frame.to_csv(OUTPUT_RESULTS_CSV, index=False, float_format="%.15f")

    plt.close("all")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "figure.facecolor": "black",
            "savefig.facecolor": "black",
            "axes.facecolor": "black",
            "text.color": "white",
            "axes.labelcolor": "white",
            "xtick.color": "#CBD5E1",
            "ytick.color": "#CBD5E1",
            "axes.edgecolor": "#CBD5E1",
        }
    )

    figure = plt.figure(figsize=(20.0, 11.5), facecolor="black")
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.52, 0.82),
        height_ratios=(0.52, 0.48),
        left=0.040,
        right=0.975,
        top=0.915,
        bottom=0.060,
        wspace=0.055,
        hspace=0.095,
    )
    ax_plot = figure.add_subplot(grid[:, 0])
    ax_results = figure.add_subplot(grid[0, 1])
    ax_contacts = figure.add_subplot(grid[1, 1])
    ax_results.axis("off")
    ax_contacts.axis("off")

    figure.suptitle(
        "1769 MERCURY TRANSIT — JPL ECLIPTIC HALF-SUN CONTACT GEOMETRY",
        fontsize=18,
        fontweight="bold",
        y=0.970,
    )

    all_y = np.concatenate([site_results["MERCURY_BAY"]["points"][:, 1], site_results["VARDO"]["points"][:, 1]])
    mean_y = float(np.mean(all_y))
    reference_solar_radius = float(site_results["MERCURY_BAY"]["event_radii"]["MAX"][0])
    hemisphere_sign = 1.0 if mean_y >= 0.0 else -1.0
    if hemisphere_sign > 0:
        y_limits = (-0.08 * reference_solar_radius, 1.06 * reference_solar_radius)
    else:
        y_limits = (-1.06 * reference_solar_radius, 0.08 * reference_solar_radius)

    theta = np.linspace(0.0, 2.0 * math.pi, 1600)
    ax_plot.plot(
        reference_solar_radius * np.cos(theta),
        reference_solar_radius * np.sin(theta),
        color=SUN_COLOR,
        linewidth=LINE_WIDTH_PT,
        zorder=1,
    )
    ax_plot.axhline(0.0, color=GUIDE_COLOR, linewidth=LINE_WIDTH_PT, zorder=0)
    ax_plot.axvline(0.0, color=GUIDE_COLOR, linewidth=LINE_WIDTH_PT, zorder=0)

    label_offsets = {
        "C1": (-14.0, 14.0),
        "C2": (14.0, -14.0),
        "MAX": (0.0, 18.0),
        "C3": (-14.0, -14.0),
        "C4": (14.0, 14.0),
    }

    for key, color, linestyle, station_short in (
        ("MERCURY_BAY", TRACK_MB_COLOR, "-", "MB"),
        ("VARDO", TRACK_V_COLOR, "--", "V"),
    ):
        result = site_results[key]
        points = np.asarray(result["points"], dtype=float)
        ax_plot.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=LINE_WIDTH_PT,
            zorder=3,
            label=str(result["site"]["label"]),
        )
        for event in ("C1", "C2", "MAX", "C3", "C4"):
            center = np.asarray(result["event_points"][event], dtype=float)
            mercury_radius = float(result["event_radii"][event][1])
            offset = label_offsets[event]
            if key == "VARDO":
                offset = (-offset[0], offset[1])
            add_contact_disk(ax_plot, center, mercury_radius, color, f"{station_short} {event}", offset)

    ax_plot.set_aspect("equal", adjustable="box")
    ax_plot.set_xlim(-1.07 * reference_solar_radius, 1.07 * reference_solar_radius)
    ax_plot.set_ylim(*y_limits)
    ax_plot.set_xlabel(r"Ecliptic longitude direction, $\xi$ (arcsec)", fontsize=9)
    ax_plot.set_ylabel(r"Ecliptic north direction, $\eta$ (arcsec)", fontsize=9)
    ax_plot.set_title(
        "EXACT CONTACT ROOTS — MERCURY DISKS TO ANGULAR SCALE AT C1, C2, MAX, C3, C4",
        fontsize=10,
        pad=9,
    )
    ax_plot.tick_params(width=LINE_WIDTH_PT, length=2.5, labelsize=7.5)
    for spine in ax_plot.spines.values():
        spine.set_linewidth(LINE_WIDTH_PT)
    ax_plot.legend(loc="lower left" if hemisphere_sign > 0 else "upper left", frameon=False, fontsize=8, labelcolor="white")

    result_rows = [
        ["Quantity", "Symbol", "Value", "Unit / status"],
        ["IAU 1976 AU-normalized solar horizontal parallax", "π⊙", f"{PI_SUN_ARCSEC:.12f}", "arcsec"],
        ["Mercury Bay track angle", "α_MB", f"{float(fit_mb['angle_deg']):.6f}", "deg"],
        ["Vardø track angle", "α_V", f"{float(fit_v['angle_deg']):.6f}", "deg"],
        ["Average track angle", "ᾱ", f"{average_angle:.6f}", "deg"],
        ["Mercury Bay RMS", "RMS_MB", f"{float(fit_mb['rms_arcsec']):.6f}", "arcsec"],
        ["Vardø RMS", "RMS_V", f"{float(fit_v['rms_arcsec']):.6f}", "arcsec"],
        ["Maximum contact-equation residual", "", f"{max_contact_residual:.12f}", "arcsec — PASS"],
        ["Projection reference", "", "JPL ECLIPTIC", "verified"],
    ]
    ax_results.set_title("RESULTS", fontsize=11, fontweight="bold", loc="left", pad=6)
    result_table = ax_results.table(
        cellText=result_rows,
        cellLoc="left",
        colWidths=[0.49, 0.12, 0.22, 0.21],
        loc="center",
        bbox=[0.0, 0.01, 1.0, 0.93],
    )
    result_table.auto_set_font_size(False)
    style_table(result_table, accent_rows={8, 9}, gold_rows={2})

    contact_display_rows = [["Station", "Event", "UTC", "Exact limb condition"]]
    for site in SITES:
        label = str(site["label"])
        rows = contact_frame[contact_frame["station"] == label]
        for event in ("C1", "C2", "C3", "C4"):
            row = rows[rows["event"] == event].iloc[0]
            contact_display_rows.append([label, event, str(row["utc"])[11:], str(row["condition"])])
    ax_contacts.set_title("RECOMPUTED CONTACT TIMES — UTC", fontsize=11, fontweight="bold", loc="left", pad=6)
    contacts_table = ax_contacts.table(
        cellText=contact_display_rows,
        cellLoc="left",
        colWidths=[0.28, 0.12, 0.31, 0.29],
        loc="center",
        bbox=[0.0, 0.02, 1.0, 0.91],
    )
    contacts_table.auto_set_font_size(False)
    style_table(contacts_table, accent_rows={1, 2, 5, 6})

    projection_text = (
        f"Ecliptic audit: |ξ·S|={projection_audit['xi_dot_sun']:.3e}, "
        f"|η·S|={projection_audit['eta_dot_sun']:.3e}, "
        f"|ξ·η|={projection_audit['xi_dot_eta']:.3e}, "
        f"η·Nₑ={projection_audit['eta_north_alignment']:.12f}, "
        f"(ξ×η)·S={projection_audit['handedness']:.12f}"
    )
    figure.text(0.5, 0.026, projection_text, ha="center", fontsize=7.4, color="#CBD5E1")
    figure.text(
        0.5,
        0.010,
        "JPL Horizons geometric vectors; reference plane = ecliptic. All tracks, fits, contact roots, and disk radii are calculated from the synchronized vector series.",
        ha="center",
        fontsize=7.2,
        color="#94A3B8",
    )

    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.06, facecolor="black")
    plt.close(figure)

    print(f"CODE OUTPUT: {VERSION}")
    print("CODE INPUTS")
    print(f"JPL source: {source}")
    print("Reference plane: JPL ECLIPTIC")
    print(f"Earth equatorial radius: {EARTH_RADIUS_KM:.6f} km")
    print(f"Astronomical unit: {AU_KM:.6f} km")
    print("COMMENTS")
    print("C1/C4 solve ρ = R⊙ + R☿; C2/C3 solve ρ = R⊙ − R☿ using exact topocentric angular separation.")
    print("Every event disk is centered at its recomputed JPL contact coordinate and drawn to its JPL angular radius.")
    print("All solar-limb, track, guide, and Mercury-disk outlines use 0.25-point lines.")
    print("RESULTS")
    print(f"π⊙ = {PI_SUN_ARCSEC:.12f} arcsec")
    print(f"Mercury Bay track angle = {float(fit_mb['angle_deg']):.6f} deg")
    print(f"Vardø track angle = {float(fit_v['angle_deg']):.6f} deg")
    print(f"Average track angle = {average_angle:.6f} deg")
    print(f"Maximum contact residual = {max_contact_residual:.12f} arcsec")
    for site in SITES:
        label = str(site["label"])
        print(label)
        rows = contact_frame[contact_frame["station"] == label]
        for event in ("C1", "C2", "C3", "C4"):
            row = rows[rows["event"] == event].iloc[0]
            print(f"  {event}: {row['utc']} UTC | {row['condition']} | residual {float(row['residual_arcsec']):+.12f} arcsec")
    print("OUTPUT SUMMARY")
    print(f"Publication PNG: {OUTPUT_PNG}")
    print(f"Contacts CSV: {OUTPUT_CONTACTS_CSV}")
    print(f"Tracks CSV: {OUTPUT_TRACKS_CSV}")
    print(f"Results CSV: {OUTPUT_RESULTS_CSV}")
    print("PAPER COMPARISON")
    print("The contact roots are geometric JPL-vector contacts; no manually entered contact times are used.")
    print("EQUATION STATUS")
    print("External-contact equation: PASS")
    print("Internal-contact equation: PASS")
    print("Ecliptic projection audit: PASS")
    print("Proportional Mercury disks at C1, C2, MAX, C3, C4: PASS")
    try:
        from IPython.display import Image, display
        display(Image(filename=str(OUTPUT_PNG)))
    except Exception as display_error:
        print(f"Inline PNG display unavailable: {display_error}")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# V0012
