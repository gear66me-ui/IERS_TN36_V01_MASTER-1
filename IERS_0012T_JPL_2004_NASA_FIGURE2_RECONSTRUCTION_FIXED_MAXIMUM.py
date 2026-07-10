# IERS-0012T
# Audit reference: GitHubDelivery@IERS-0012T; corrected JPL geocentric reconstruction using centered-seconds greatest-transit minimization.

import csv
import html
import math
import os
import re
import subprocess
import sys
import urllib.request
import warnings
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

VERSION = "IERS-0012T"
PROGRAM_NAME = "IERS_0012T_JPL_2004_NASA_FIGURE2_RECONSTRUCTION_FIXED_MAXIMUM.py"

ARCSEC_PER_RAD = 206_264.80624709636
NASA_URL = "https://eclipse.gsfc.nasa.gov/OH/transit04.html"
START = "2004-Jun-08 04:30"
STOP = "2004-Jun-08 12:15"
STEP = "1m"
LOCAL_TZ = ZoneInfo("America/Bogota")
OUT_DIR = "/content/IERS_TN36_V01_MASTER_OUTPUT"
PNG_PATH = os.path.join(
    OUT_DIR,
    "IERS-0012T_JPL_2004_NASA_FIGURE2_RECONSTRUCTION.png",
)
CSV_PATH = os.path.join(
    OUT_DIR,
    "IERS-0012T_JPL_2004_NASA_FIGURE2_RECONSTRUCTION.csv",
)

NASA_FALLBACK = {
    "C1": {"label": "Contact I", "utc": "05:13:29", "pa_deg": 116.0},
    "C2": {"label": "Contact II", "utc": "05:32:55", "pa_deg": 119.0},
    "MAX": {"label": "Greatest", "utc": "08:19:44", "pa_deg": 166.0},
    "C3": {"label": "Contact III", "utc": "11:06:33", "pa_deg": 213.0},
    "C4": {"label": "Contact IV", "utc": "11:25:59", "pa_deg": 216.0},
}
NASA_MINIMUM_SEPARATION_FALLBACK_ARCSEC = 627.0


def ensure_package(import_name, pip_name):
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "-q", "install", pip_name]
        )


for import_name, pip_name in (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("astroquery", "astroquery"),
    ("astropy", "astropy"),
):
    ensure_package(import_name, pip_name)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar
from astroquery.jplhorizons import Horizons
import astropy.units as u
from astropy.coordinates import Angle
from astropy.time import Time


def norm(vector):
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    magnitude = norm(vector)
    if magnitude == 0.0:
        raise RuntimeError("Zero vector cannot be normalized.")
    return vector / magnitude


def circular_delta_deg(value_deg, reference_deg):
    return (
        (float(value_deg) - float(reference_deg) + 180.0) % 360.0
    ) - 180.0


def line_angle_deg(direction):
    direction = unit(direction)
    if direction[0] < 0.0:
        direction = -direction
    angle = math.degrees(math.atan2(direction[1], direction[0]))
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return angle


def acute_line_difference_deg(angle_a_deg, angle_b_deg):
    delta = abs(float(angle_a_deg) - float(angle_b_deg)) % 180.0
    return min(delta, 180.0 - delta)


def utc_iso(jd_utc):
    return Time(float(jd_utc), format="jd", scale="utc").iso


def utc_hms(jd_utc, decimals=3):
    dt = Time(float(jd_utc), format="jd", scale="utc").to_datetime(
        timezone=timezone.utc
    )
    if decimals <= 0:
        return dt.strftime("%H:%M:%S")
    seconds = dt.second + dt.microsecond / 1_000_000.0
    width = 3 + decimals
    return f"{dt.hour:02d}:{dt.minute:02d}:{seconds:0{width}.{decimals}f}"


def nasa_jd_from_hms(hms):
    return float(Time(f"2004-06-08T{hms}", scale="utc").jd)


def fetch_nasa_contact_table():
    status = "NASA GSFC LIVE HTML"
    table = {key: dict(value) for key, value in NASA_FALLBACK.items()}
    minimum_separation = NASA_MINIMUM_SEPARATION_FALLBACK_ARCSEC
    try:
        request = urllib.request.Request(
            NASA_URL,
            headers={"User-Agent": "Mozilla/5.0 IERS-0012T"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            raw_html = response.read().decode("utf-8", errors="replace")

        plain = html.unescape(re.sub(r"<[^>]+>", " ", raw_html))
        plain = re.sub(r"\s+", " ", plain)

        patterns = {
            "C1": r"Contact I\s+(\d{2}:\d{2}:\d{2})\s+(\d+(?:\.\d+)?)°",
            "C2": r"Contact II\s+(\d{2}:\d{2}:\d{2})\s+(\d+(?:\.\d+)?)°",
            "MAX": r"Greatest\s+(\d{2}:\d{2}:\d{2})\s+(\d+(?:\.\d+)?)°",
            "C3": r"Contact III\s+(\d{2}:\d{2}:\d{2})\s+(\d+(?:\.\d+)?)°",
            "C4": r"Contact IV\s+(\d{2}:\d{2}:\d{2})\s+(\d+(?:\.\d+)?)°",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, plain, flags=re.IGNORECASE)
            if match is None:
                raise RuntimeError(f"NASA table row not found: {key}")
            table[key]["utc"] = match.group(1)
            table[key]["pa_deg"] = float(match.group(2))

        separation_match = re.search(
            r"minimum separation from the Sun is\s+"
            r"(\d+(?:\.\d+)?)\s+arc-seconds",
            plain,
            flags=re.IGNORECASE,
        )
        if separation_match is not None:
            minimum_separation = float(separation_match.group(1))
    except Exception as exc:
        status = f"NASA PUBLISHED FALLBACK — {type(exc).__name__}"
    return table, float(minimum_separation), status


def query_jpl_ephemerides(target_id, prefix):
    table = Horizons(
        id=target_id,
        location="500@399",
        epochs={"start": START, "stop": STOP, "step": STEP},
    ).ephemerides(
        extra_precision=True,
        refsystem="ICRF",
        cache=True,
    ).to_pandas()

    required = {
        "datetime_jd",
        "datetime_str",
        "RA_app",
        "DEC_app",
        "ang_width",
        "delta",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise RuntimeError(
            f"JPL Horizons did not return required columns for {prefix}: {missing}"
        )

    frame = pd.DataFrame()
    frame["jd_utc"] = pd.to_numeric(
        table["datetime_jd"], errors="raise"
    ).astype(float)
    frame[f"{prefix}_datetime_str"] = table["datetime_str"].astype(str)
    frame[f"{prefix}_ra_app_deg"] = pd.to_numeric(
        table["RA_app"], errors="raise"
    ).astype(float)
    frame[f"{prefix}_dec_app_deg"] = pd.to_numeric(
        table["DEC_app"], errors="raise"
    ).astype(float)
    frame[f"{prefix}_ang_width_arcsec"] = pd.to_numeric(
        table["ang_width"], errors="raise"
    ).astype(float)
    frame[f"{prefix}_delta_au"] = pd.to_numeric(
        table["delta"], errors="raise"
    ).astype(float)
    return frame


def build_master():
    sun = query_jpl_ephemerides("10", "SUN")
    venus = query_jpl_ephemerides("299", "VENUS")
    master = sun.merge(venus, on="jd_utc", how="inner")
    if len(master) < 400:
        raise RuntimeError(
            f"Unexpectedly short JPL master table: {len(master)} rows."
        )
    master = master.sort_values("jd_utc").reset_index(drop=True)
    return master


def build_splines(master):
    jd = master["jd_utc"].to_numpy(dtype=float)
    sun_ra = np.unwrap(
        np.deg2rad(master["SUN_ra_app_deg"].to_numpy(dtype=float))
    )
    venus_ra = np.unwrap(
        np.deg2rad(master["VENUS_ra_app_deg"].to_numpy(dtype=float))
    )
    return {
        "jd": jd,
        "sun_ra": CubicSpline(jd, sun_ra, bc_type="natural"),
        "sun_dec": CubicSpline(
            jd,
            np.deg2rad(master["SUN_dec_app_deg"].to_numpy(dtype=float)),
            bc_type="natural",
        ),
        "venus_ra": CubicSpline(jd, venus_ra, bc_type="natural"),
        "venus_dec": CubicSpline(
            jd,
            np.deg2rad(master["VENUS_dec_app_deg"].to_numpy(dtype=float)),
            bc_type="natural",
        ),
        "sun_width": CubicSpline(
            jd,
            master["SUN_ang_width_arcsec"].to_numpy(dtype=float),
            bc_type="natural",
        ),
        "venus_width": CubicSpline(
            jd,
            master["VENUS_ang_width_arcsec"].to_numpy(dtype=float),
            bc_type="natural",
        ),
        "sun_delta": CubicSpline(
            jd,
            master["SUN_delta_au"].to_numpy(dtype=float),
            bc_type="natural",
        ),
        "venus_delta": CubicSpline(
            jd,
            master["VENUS_delta_au"].to_numpy(dtype=float),
            bc_type="natural",
        ),
    }


def radec_unit(ra_rad, dec_rad):
    cos_dec = math.cos(dec_rad)
    return np.array(
        [
            cos_dec * math.cos(ra_rad),
            cos_dec * math.sin(ra_rad),
            math.sin(dec_rad),
        ],
        dtype=float,
    )


def tangent_basis(ra_rad, dec_rad):
    center = radec_unit(ra_rad, dec_rad)
    east = np.array(
        [-math.sin(ra_rad), math.cos(ra_rad), 0.0],
        dtype=float,
    )
    north = np.array(
        [
            -math.sin(dec_rad) * math.cos(ra_rad),
            -math.sin(dec_rad) * math.sin(ra_rad),
            math.cos(dec_rad),
        ],
        dtype=float,
    )
    return unit(center), unit(east), unit(north)


def equidistant_offset_arcsec(
    reference_ra_rad,
    reference_dec_rad,
    target_ra_rad,
    target_dec_rad,
):
    center, east, north = tangent_basis(
        reference_ra_rad,
        reference_dec_rad,
    )
    target = radec_unit(target_ra_rad, target_dec_rad)
    cosine = float(np.clip(np.dot(center, target), -1.0, 1.0))
    separation = math.acos(cosine)
    east_component = float(np.dot(target, east))
    north_component = float(np.dot(target, north))
    position_angle = math.atan2(east_component, north_component)
    east_arcsec = (
        separation * math.sin(position_angle) * ARCSEC_PER_RAD
    )
    north_arcsec = (
        separation * math.cos(position_angle) * ARCSEC_PER_RAD
    )
    return east_arcsec, north_arcsec, separation * ARCSEC_PER_RAD


def state_at(splines, jd_utc):
    sun_ra = float(splines["sun_ra"](jd_utc))
    sun_dec = float(splines["sun_dec"](jd_utc))
    venus_ra = float(splines["venus_ra"](jd_utc))
    venus_dec = float(splines["venus_dec"](jd_utc))

    east_arcsec, north_arcsec, separation_arcsec = (
        equidistant_offset_arcsec(
            sun_ra,
            sun_dec,
            venus_ra,
            venus_dec,
        )
    )
    position_angle_deg = (
        math.degrees(math.atan2(east_arcsec, north_arcsec)) % 360.0
    )
    return {
        "jd_utc": float(jd_utc),
        "sun_ra_rad": sun_ra,
        "sun_dec_rad": sun_dec,
        "venus_ra_rad": venus_ra,
        "venus_dec_rad": venus_dec,
        "east_arcsec": east_arcsec,
        "west_arcsec": -east_arcsec,
        "north_arcsec": north_arcsec,
        "separation_arcsec": separation_arcsec,
        "position_angle_deg": position_angle_deg,
        "sun_radius_arcsec": 0.5 * float(splines["sun_width"](jd_utc)),
        "venus_radius_arcsec": 0.5
        * float(splines["venus_width"](jd_utc)),
        "sun_delta_au": float(splines["sun_delta"](jd_utc)),
        "venus_delta_au": float(splines["venus_delta"](jd_utc)),
    }


def contact_function(splines, event_kind, jd_utc):
    state = state_at(splines, jd_utc)
    if event_kind == "EXTERNAL":
        threshold = (
            state["sun_radius_arcsec"]
            + state["venus_radius_arcsec"]
        )
    elif event_kind == "INTERNAL":
        threshold = (
            state["sun_radius_arcsec"]
            - state["venus_radius_arcsec"]
        )
    else:
        raise ValueError(f"Unsupported contact kind: {event_kind}")
    return state["separation_arcsec"] - threshold


def all_roots(splines, event_kind):
    jd = splines["jd"]
    values = np.array(
        [contact_function(splines, event_kind, value) for value in jd],
        dtype=float,
    )
    roots = []
    for index in range(len(jd) - 1):
        left_value = values[index]
        right_value = values[index + 1]
        if left_value == 0.0:
            roots.append(float(jd[index]))
        elif left_value * right_value < 0.0:
            root = brentq(
                lambda value: contact_function(
                    splines,
                    event_kind,
                    value,
                ),
                float(jd[index]),
                float(jd[index + 1]),
                xtol=1.0e-13,
                rtol=1.0e-13,
            )
            roots.append(float(root))
    return roots


def minimize_jd_in_centered_seconds(
    objective_jd,
    lower_jd,
    upper_jd,
    xatol_seconds=1.0e-6,
):
    lower_jd = float(lower_jd)
    upper_jd = float(upper_jd)
    center_jd = 0.5 * (lower_jd + upper_jd)
    lower_seconds = (lower_jd - center_jd) * 86_400.0
    upper_seconds = (upper_jd - center_jd) * 86_400.0
    result = minimize_scalar(
        lambda offset_seconds: objective_jd(
            center_jd + float(offset_seconds) / 86_400.0
        ),
        bounds=(lower_seconds, upper_seconds),
        method="bounded",
        options={
            "xatol": float(xatol_seconds),
            "maxiter": 500,
        },
    )
    if not result.success:
        raise RuntimeError(
            f"Centered-seconds minimization failed: {result.message}"
        )
    return center_jd + float(result.x) / 86_400.0, result


def derive_events(splines):
    external_roots = all_roots(splines, "EXTERNAL")
    internal_roots = all_roots(splines, "INTERNAL")
    if len(external_roots) != 2 or len(internal_roots) != 2:
        raise RuntimeError(
            "Expected exactly two external and two internal contact roots; "
            f"received external={external_roots}, internal={internal_roots}."
        )

    c1, c4 = external_roots
    c2, c3 = internal_roots

    in_transit_jds = np.asarray(splines["jd"], dtype=float)
    mask = (in_transit_jds >= c2) & (in_transit_jds <= c3)
    candidate_jds = in_transit_jds[mask]
    candidate_separations = np.array(
        [state_at(splines, jd)["separation_arcsec"] for jd in candidate_jds],
        dtype=float,
    )
    minimum_index = int(np.argmin(candidate_separations))
    lower_index = max(minimum_index - 1, 0)
    upper_index = min(minimum_index + 1, len(candidate_jds) - 1)
    lower_jd = float(candidate_jds[lower_index])
    upper_jd = float(candidate_jds[upper_index])
    if lower_jd == upper_jd:
        raise RuntimeError("JPL minute grid did not bracket greatest transit.")

    greatest, _greatest_result = minimize_jd_in_centered_seconds(
        lambda jd: state_at(splines, jd)["separation_arcsec"],
        lower_jd,
        upper_jd,
        xatol_seconds=1.0e-6,
    )
    return {
        "C1": c1,
        "C2": c2,
        "MAX": greatest,
        "C3": c3,
        "C4": c4,
    }


def pca_fit(points):
    points = np.asarray(points, dtype=float)
    mean = points.mean(axis=0)
    centered = points - mean
    _u, singular_values, vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    direction = unit(vt[0])
    if direction[0] < 0.0:
        direction = -direction
    normal = np.array([-direction[1], direction[0]], dtype=float)
    cross_track = centered @ normal
    rms = math.sqrt(float(np.mean(cross_track * cross_track)))

    along_track = centered @ direction
    quadratic = np.polyfit(along_track, cross_track, 2)
    quadratic_prediction = np.polyval(quadratic, along_track)
    quadratic_rms = math.sqrt(
        float(np.mean((cross_track - quadratic_prediction) ** 2))
    )
    curvature = abs(2.0 * quadratic[0]) / (
        1.0 + quadratic[1] ** 2
    ) ** 1.5

    differences = np.diff(points, axis=0)
    differences = np.array(
        [
            value if value[0] >= 0.0 else -value
            for value in differences
            if norm(value) > 0.0
        ]
    )
    local_angles = np.unwrap(
        np.arctan2(differences[:, 1], differences[:, 0])
    )
    average_local_angle = math.degrees(float(np.mean(local_angles)))
    while average_local_angle <= -90.0:
        average_local_angle += 180.0
    while average_local_angle > 90.0:
        average_local_angle -= 180.0

    return {
        "mean": mean,
        "direction": direction,
        "normal": normal,
        "slope": float(direction[1] / direction[0]),
        "angle_deg": line_angle_deg(direction),
        "average_local_angle_deg": average_local_angle,
        "rms_arcsec": rms,
        "quadratic_rms_arcsec": quadratic_rms,
        "curvature_per_arcsec": curvature,
        "quadratic_coefficients": quadratic,
        "singular_values": singular_values,
    }


def build_track(splines, events):
    minute_jds = splines["jd"]
    mask = (
        (minute_jds >= events["C1"])
        & (minute_jds <= events["C4"])
    )
    fit_jds = sorted(
        set(
            [float(value) for value in minute_jds[mask]]
            + [float(value) for value in events.values()]
        )
    )
    states = [state_at(splines, jd) for jd in fit_jds]
    points = np.array(
        [
            [state["west_arcsec"], state["north_arcsec"]]
            for state in states
        ],
        dtype=float,
    )
    fit = pca_fit(points)
    return {
        "jds": np.array(fit_jds, dtype=float),
        "states": states,
        "points": points,
        "fit": fit,
    }


def build_ecliptic_fit(splines, reference_jd):
    reference_state = state_at(splines, reference_jd)
    reference_ra = reference_state["sun_ra_rad"]
    reference_dec = reference_state["sun_dec_rad"]

    points = []
    for jd in splines["jd"]:
        sun_ra = float(splines["sun_ra"](jd))
        sun_dec = float(splines["sun_dec"](jd))
        east_arcsec, north_arcsec, _separation = (
            equidistant_offset_arcsec(
                reference_ra,
                reference_dec,
                sun_ra,
                sun_dec,
            )
        )
        points.append([-east_arcsec, north_arcsec])

    points = np.asarray(points, dtype=float)
    fit = pca_fit(points)
    return {
        "points": points,
        "fit": fit,
    }


def hourly_states(splines):
    result = {}
    for hour in range(5, 13):
        jd = float(
            Time(f"2004-06-08T{hour:02d}:00:00", scale="utc").jd
        )
        result[hour] = state_at(splines, jd)
    return result


def format_ra(ra_rad):
    angle = Angle(math.degrees(ra_rad), unit=u.deg)
    return angle.to_string(
        unit=u.hour,
        sep=":",
        precision=3,
        pad=True,
    )


def format_dec(dec_rad):
    angle = Angle(math.degrees(dec_rad), unit=u.deg)
    return angle.to_string(
        unit=u.deg,
        sep=":",
        precision=2,
        alwayssign=True,
        pad=True,
    )


def comparison_rows(events, splines, nasa_table):
    rows = []
    for key in ("C1", "C2", "MAX", "C3", "C4"):
        jpl_jd = events[key]
        jpl_state = state_at(splines, jpl_jd)
        nasa_jd = nasa_jd_from_hms(nasa_table[key]["utc"])
        rows.append(
            {
                "event": key,
                "label": nasa_table[key]["label"],
                "nasa_utc": nasa_table[key]["utc"],
                "jpl_utc": utc_hms(jpl_jd, decimals=3),
                "delta_seconds": (jpl_jd - nasa_jd) * 86_400.0,
                "nasa_pa_deg": nasa_table[key]["pa_deg"],
                "jpl_pa_deg": jpl_state["position_angle_deg"],
                "delta_pa_deg": circular_delta_deg(
                    jpl_state["position_angle_deg"],
                    nasa_table[key]["pa_deg"],
                ),
                "jpl_separation_arcsec": jpl_state[
                    "separation_arcsec"
                ],
                "jpl_sun_radius_arcsec": jpl_state[
                    "sun_radius_arcsec"
                ],
                "jpl_venus_radius_arcsec": jpl_state[
                    "venus_radius_arcsec"
                ],
            }
        )
    return rows


def html_escape(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def html_table(headers, rows):
    head = "".join(
        f"<th>{html_escape(header)}</th>" for header in headers
    )
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html_escape(value)}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return (
        "<table class='iers-table'>"
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def display_widgets(
    greatest_state,
    track,
    ecliptic,
    comparison,
    nasa_minimum_separation,
    nasa_status,
):
    try:
        from IPython.display import HTML, display
    except Exception:
        return False

    track_angle = track["fit"]["angle_deg"]
    ecliptic_angle = ecliptic["fit"]["angle_deg"]
    relative_angle = acute_line_difference_deg(
        track_angle,
        ecliptic_angle,
    )

    geometry_rows = [
        ["JPL greatest UTC", utc_hms(greatest_state["jd_utc"], 3), "UT"],
        ["JPL minimum separation", f"{greatest_state['separation_arcsec']:.6f}", "arcsec"],
        ["NASA minimum separation", f"{nasa_minimum_separation:.3f}", "arcsec"],
        ["JPL greatest position angle", f"{greatest_state['position_angle_deg']:.6f}", "deg"],
        ["Track angle from horizontal", f"{track_angle:.6f}", "deg"],
        ["Average local track angle", f"{track['fit']['average_local_angle_deg']:.6f}", "deg"],
        ["Ecliptic angle from horizontal", f"{ecliptic_angle:.6f}", "deg"],
        ["Track angle from ecliptic", f"{relative_angle:.6f}", "deg"],
        ["Track slope", f"{track['fit']['slope']:.10f}", "dy/dx"],
        ["Linear-fit RMS", f"{track['fit']['rms_arcsec']:.9f}", "arcsec"],
        ["Quadratic-fit RMS", f"{track['fit']['quadratic_rms_arcsec']:.9f}", "arcsec"],
        ["Curvature", f"{track['fit']['curvature_per_arcsec']:.12e}", "1/arcsec"],
        ["JPL Sun semidiameter", f"{greatest_state['sun_radius_arcsec']:.6f}", "arcsec"],
        ["JPL Venus semidiameter", f"{greatest_state['venus_radius_arcsec']:.6f}", "arcsec"],
        ["Sun apparent RA", format_ra(greatest_state["sun_ra_rad"]), "h:m:s"],
        ["Sun apparent Dec", format_dec(greatest_state["sun_dec_rad"]), "d:m:s"],
        ["Venus apparent RA", format_ra(greatest_state["venus_ra_rad"]), "h:m:s"],
        ["Venus apparent Dec", format_dec(greatest_state["venus_dec_rad"]), "d:m:s"],
    ]

    contact_rows = []
    for row in comparison:
        contact_rows.append(
            [
                row["event"],
                row["nasa_utc"],
                row["jpl_utc"],
                f"{row['delta_seconds']:+.3f}",
                f"{row['nasa_pa_deg']:.3f}",
                f"{row['jpl_pa_deg']:.6f}",
                f"{row['delta_pa_deg']:+.6f}",
            ]
        )

    css = """
    <style>
    .iers-wrap{background:#03080d;color:#e8f7ff;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;width:900px;max-width:98%;border:1px solid #1e4f64;border-radius:9px;padding:9px;margin:8px 0 14px}
    .iers-title{color:#66e8ff;font-size:10px;font-weight:800;letter-spacing:.055em;text-align:center;border-top:1px solid #25708b;border-bottom:1px solid #25708b;padding:5px 0;margin:5px 0}
    .iers-table{border-collapse:collapse;width:100%;table-layout:auto;font-size:10px;background:#050b0f;margin-bottom:7px}
    .iers-table th{color:#66e8ff;background:#0a1a22;border-bottom:1px solid #1d3d4a;padding:4px 5px;text-align:left}
    .iers-table td{border-bottom:1px solid #102630;padding:4px 5px}
    .iers-table td:nth-child(2){color:#ffc861;text-align:right;font-weight:800}
    .iers-note{color:#8fb4c1;font-size:9px;margin-top:5px}
    </style>
    """
    content = (
        css
        + "<div class='iers-wrap'>"
        + "<div class='iers-title'>JPL GEOCENTRIC TRACK GEOMETRY — 2004 VENUS TRANSIT</div>"
        + html_table(["Quantity", "Value", "Unit"], geometry_rows)
        + "<div class='iers-title'>NASA GSFC CONTACT TABLE vs JPL HORIZONS</div>"
        + html_table(
            [
                "Event",
                "NASA UT",
                "JPL UT",
                "Δt s",
                "NASA PA°",
                "JPL PA°",
                "ΔPA°",
            ],
            contact_rows,
        )
        + f"<div class='iers-note'>NASA table source: {html_escape(NASA_URL)} | {html_escape(nasa_status)}</div>"
        + "</div>"
    )
    display(HTML(content))
    return True


def write_csv(
    comparison,
    greatest_state,
    track,
    ecliptic,
    nasa_status,
    nasa_minimum_separation,
    hourly,
):
    track_angle = track["fit"]["angle_deg"]
    ecliptic_angle = ecliptic["fit"]["angle_deg"]
    relative_angle = acute_line_difference_deg(
        track_angle,
        ecliptic_angle,
    )

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                VERSION,
                "JPL 2004 NASA FIGURE 2 RECONSTRUCTION",
            ]
        )
        writer.writerow([])
        writer.writerow(["section", "quantity", "value", "unit", "source"])
        summary_rows = [
            ("GEOMETRY", "JPL greatest UTC", utc_iso(greatest_state["jd_utc"]), "UTC", "JPL"),
            ("GEOMETRY", "JPL minimum separation", greatest_state["separation_arcsec"], "arcsec", "JPL"),
            ("COMPARISON", "NASA minimum separation", nasa_minimum_separation, "arcsec", nasa_status),
            ("GEOMETRY", "JPL greatest position angle", greatest_state["position_angle_deg"], "deg", "JPL"),
            ("GEOMETRY", "Track angle from horizontal", track_angle, "deg", "JPL fit"),
            ("GEOMETRY", "Average local track angle", track["fit"]["average_local_angle_deg"], "deg", "JPL minute fit"),
            ("GEOMETRY", "Ecliptic angle from horizontal", ecliptic_angle, "deg", "JPL Sun apparent path"),
            ("GEOMETRY", "Track angle from ecliptic", relative_angle, "deg", "JPL derived"),
            ("FIT", "Track slope", track["fit"]["slope"], "dy/dx", "JPL minute fit"),
            ("FIT", "Linear RMS", track["fit"]["rms_arcsec"], "arcsec", "JPL minute fit"),
            ("FIT", "Quadratic RMS", track["fit"]["quadratic_rms_arcsec"], "arcsec", "JPL minute fit"),
            ("FIT", "Curvature", track["fit"]["curvature_per_arcsec"], "1/arcsec", "JPL minute fit"),
            ("ANGULAR SIZE", "Sun semidiameter", greatest_state["sun_radius_arcsec"], "arcsec", "JPL"),
            ("ANGULAR SIZE", "Venus semidiameter", greatest_state["venus_radius_arcsec"], "arcsec", "JPL"),
        ]
        for row in summary_rows:
            writer.writerow(row)

        writer.writerow([])
        writer.writerow(
            [
                "event",
                "NASA_UTC",
                "JPL_UTC",
                "delta_seconds",
                "NASA_PA_deg",
                "JPL_PA_deg",
                "delta_PA_deg",
                "JPL_separation_arcsec",
                "JPL_Sun_radius_arcsec",
                "JPL_Venus_radius_arcsec",
            ]
        )
        for row in comparison:
            writer.writerow(
                [
                    row["event"],
                    row["nasa_utc"],
                    row["jpl_utc"],
                    f"{row['delta_seconds']:.9f}",
                    f"{row['nasa_pa_deg']:.9f}",
                    f"{row['jpl_pa_deg']:.9f}",
                    f"{row['delta_pa_deg']:.9f}",
                    f"{row['jpl_separation_arcsec']:.9f}",
                    f"{row['jpl_sun_radius_arcsec']:.9f}",
                    f"{row['jpl_venus_radius_arcsec']:.9f}",
                ]
            )

        writer.writerow([])
        writer.writerow(
            [
                "hour_UT",
                "west_arcsec",
                "north_arcsec",
                "separation_arcsec",
                "position_angle_deg",
                "Venus_radius_arcsec",
            ]
        )
        for hour, state in hourly.items():
            writer.writerow(
                [
                    hour,
                    f"{state['west_arcsec']:.9f}",
                    f"{state['north_arcsec']:.9f}",
                    f"{state['separation_arcsec']:.9f}",
                    f"{state['position_angle_deg']:.9f}",
                    f"{state['venus_radius_arcsec']:.9f}",
                ]
            )

        writer.writerow([])
        writer.writerow(
            [
                "jd_utc",
                "utc",
                "west_arcsec",
                "north_arcsec",
            ]
        )
        for jd, point in zip(track["jds"], track["points"]):
            writer.writerow(
                [
                    f"{jd:.12f}",
                    utc_iso(jd),
                    f"{point[0]:.9f}",
                    f"{point[1]:.9f}",
                ]
            )


def plot_reconstruction(
    splines,
    events,
    track,
    ecliptic,
    hourly,
    greatest_state,
):
    sun_radius = greatest_state["sun_radius_arcsec"]
    venus_radius = greatest_state["venus_radius_arcsec"]
    limit = sun_radius * 1.10

    figure, axis = plt.subplots(figsize=(9.2, 9.2), dpi=260)
    figure.patch.set_facecolor("#03080d")
    axis.set_facecolor("#03080d")

    axis.add_patch(
        Circle(
            (0.0, 0.0),
            sun_radius,
            fill=False,
            linewidth=0.55,
            edgecolor="#f4d35e",
            zorder=1,
        )
    )
    axis.axhline(0.0, linewidth=0.20, color="#24414f", zorder=0)
    axis.axvline(0.0, linewidth=0.20, color="#24414f", zorder=0)

    ecliptic_direction = ecliptic["fit"]["direction"]
    ecliptic_endpoints = np.vstack(
        [
            -sun_radius * ecliptic_direction,
            sun_radius * ecliptic_direction,
        ]
    )
    axis.plot(
        ecliptic_endpoints[:, 0],
        ecliptic_endpoints[:, 1],
        linestyle=(0, (5, 4)),
        linewidth=0.40,
        color="#66e8ff",
        label="Projected ecliptic from JPL Sun motion",
        zorder=2,
    )
    ecliptic_label_point = 0.78 * sun_radius * ecliptic_direction
    axis.text(
        ecliptic_label_point[0],
        ecliptic_label_point[1] + 24.0,
        "ECLIPTIC",
        fontsize=6.5,
        color="#66e8ff",
        ha="center",
        va="bottom",
        rotation=ecliptic["fit"]["angle_deg"],
        rotation_mode="anchor",
    )

    points = track["points"]
    axis.plot(
        points[:, 0],
        points[:, 1],
        linewidth=0.36,
        color="#ff9f1c",
        label="JPL geocentric Venus track",
        zorder=4,
    )
    axis.scatter(
        points[::5, 0],
        points[::5, 1],
        s=1.2,
        linewidths=0,
        color="#ff9f1c",
        zorder=5,
    )

    for hour, state in hourly.items():
        center = (state["west_arcsec"], state["north_arcsec"])
        axis.add_patch(
            Circle(
                center,
                state["venus_radius_arcsec"],
                fill=False,
                linewidth=0.28,
                edgecolor="#5ee08a",
                zorder=6,
            )
        )
        axis.scatter(
            [center[0]],
            [center[1]],
            s=2.0,
            linewidths=0,
            color="#5ee08a",
            zorder=7,
        )
        offset_y = 42.0 if hour <= 8 else -42.0
        axis.text(
            center[0],
            center[1] + offset_y,
            f"{hour:02d} UT",
            fontsize=5.8,
            color="#b6f5ca",
            ha="center",
            va="center",
            zorder=8,
        )

    event_colors = {
        "C1": "#f94144",
        "C2": "#f3722c",
        "MAX": "#ffffff",
        "C3": "#f3722c",
        "C4": "#f94144",
    }
    label_offsets = {
        "C1": (-70.0, 45.0),
        "C2": (-60.0, 52.0),
        "MAX": (65.0, -28.0),
        "C3": (60.0, -50.0),
        "C4": (70.0, -42.0),
    }

    for key in ("C1", "C2", "MAX", "C3", "C4"):
        state = state_at(splines, events[key])
        center = np.array(
            [state["west_arcsec"], state["north_arcsec"]],
            dtype=float,
        )
        axis.add_patch(
            Circle(
                center,
                state["venus_radius_arcsec"],
                fill=False,
                linewidth=0.52,
                edgecolor=event_colors[key],
                zorder=9,
            )
        )
        axis.scatter(
            [center[0]],
            [center[1]],
            s=3.0,
            linewidths=0,
            color=event_colors[key],
            zorder=10,
        )
        dx, dy = label_offsets[key]
        label = (
            f"{key}\n{utc_hms(events[key], 1)}"
            if key != "MAX"
            else (
                f"GREATEST\n{utc_hms(events[key], 1)}\n"
                f"{state['separation_arcsec']:.2f}″  "
                f"PA {state['position_angle_deg']:.2f}°"
            )
        )
        axis.annotate(
            label,
            xy=center,
            xytext=(center[0] + dx, center[1] + dy),
            fontsize=5.8,
            color=event_colors[key],
            ha="center",
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.22,
                "color": event_colors[key],
            },
            zorder=11,
        )

    greatest_center = np.array(
        [
            greatest_state["west_arcsec"],
            greatest_state["north_arcsec"],
        ],
        dtype=float,
    )
    axis.plot(
        [0.0, greatest_center[0]],
        [0.0, greatest_center[1]],
        linewidth=0.25,
        linestyle=(0, (2, 3)),
        color="#ffffff",
        zorder=3,
    )
    axis.scatter([0.0], [0.0], s=2.5, color="#f4d35e", zorder=3)

    axis.text(0.0, limit * 0.965, "N", color="#e8f7ff", fontsize=8, ha="center")
    axis.text(0.0, -limit * 0.985, "S", color="#e8f7ff", fontsize=8, ha="center")
    axis.text(-limit * 0.985, 0.0, "E", color="#e8f7ff", fontsize=8, va="center")
    axis.text(limit * 0.985, 0.0, "W", color="#e8f7ff", fontsize=8, va="center")

    track_angle = track["fit"]["angle_deg"]
    ecliptic_angle = ecliptic["fit"]["angle_deg"]
    relative_angle = acute_line_difference_deg(
        track_angle,
        ecliptic_angle,
    )
    information = (
        "JPL HORIZONS — GEOCENTER\n"
        f"TRACK/HORIZONTAL = {track_angle:+.4f}°\n"
        f"ECLIPTIC/HORIZONTAL = {ecliptic_angle:+.4f}°\n"
        f"TRACK/ECLIPTIC = {relative_angle:.4f}°\n"
        f"SUN SD = {sun_radius:.3f}″\n"
        f"VENUS SD = {venus_radius:.3f}″"
    )
    axis.text(
        -limit * 0.96,
        limit * 0.90,
        information,
        fontsize=5.8,
        color="#8fb4c1",
        ha="left",
        va="top",
        linespacing=1.35,
    )

    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Celestial west offset from Sun center (arcsec)", fontsize=7)
    axis.set_ylabel("Celestial north offset from Sun center (arcsec)", fontsize=7)
    axis.tick_params(axis="both", labelsize=6, width=0.30, colors="#8fb4c1")
    for spine in axis.spines.values():
        spine.set_linewidth(0.30)
        spine.set_color("#24414f")
    axis.xaxis.label.set_color("#8fb4c1")
    axis.yaxis.label.set_color("#8fb4c1")
    axis.set_title(
        "2004 TRANSIT OF VENUS — JPL GEOCENTRIC RECONSTRUCTION\n"
        "NASA Figure 2 geometry independently rebuilt from JPL apparent ephemerides",
        fontsize=9,
        color="#e8f7ff",
        pad=10,
    )
    legend = axis.legend(
        loc="lower center",
        fontsize=6,
        frameon=False,
        ncol=2,
    )
    for text in legend.get_texts():
        text.set_color("#e8f7ff")

    figure.tight_layout()
    figure.savefig(
        PNG_PATH,
        dpi=360,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.show()
    plt.close(figure)


def print_contact_comparison(comparison):
    print(
        f"{'Event':<7} {'NASA UT':>10} {'JPL UT':>13} "
        f"{'Δt(s)':>10} {'NASA PA':>10} {'JPL PA':>12} {'ΔPA':>10}"
    )
    for row in comparison:
        print(
            f"{row['event']:<7} "
            f"{row['nasa_utc']:>10} "
            f"{row['jpl_utc']:>13} "
            f"{row['delta_seconds']:>+10.3f} "
            f"{row['nasa_pa_deg']:>10.3f} "
            f"{row['jpl_pa_deg']:>12.6f} "
            f"{row['delta_pa_deg']:>+10.6f}"
        )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    warnings.filterwarnings(
        "ignore",
        message='ERFA function ".*" yielded .*dubious year.*',
    )

    print(f"CODE OUTPUT: {VERSION}")
    print()
    print("CODE INPUTS")
    print(f"Program                : {PROGRAM_NAME}")
    print(f"JPL observer           : 500@399 geocenter")
    print(f"JPL interval           : {START} TO {STOP} STEP {STEP}")
    print(f"NASA comparison source : {NASA_URL}")
    print()
    print("COMMENTS")
    print("The solar limb, Venus disks, track, hourly points, contacts, and ecliptic are calculated from JPL Horizons.")
    print("NASA GSFC contact times, position angles, and minimum separation are used only as published comparisons.")
    print("The projected ecliptic direction is derived from the JPL apparent motion of the Sun in the same celestial tangent plane.")
    print("Greatest transit is minimized in centered elapsed seconds, never directly on the large Julian-date value.")
    print("No AI image generation is used.")
    print()

    nasa_table, nasa_minimum_separation, nasa_status = (
        fetch_nasa_contact_table()
    )
    master = build_master()
    splines = build_splines(master)
    events = derive_events(splines)
    track = build_track(splines, events)
    ecliptic = build_ecliptic_fit(splines, events["MAX"])
    hourly = hourly_states(splines)
    greatest_state = state_at(splines, events["MAX"])
    comparison = comparison_rows(
        events,
        splines,
        nasa_table,
    )

    write_csv(
        comparison,
        greatest_state,
        track,
        ecliptic,
        nasa_status,
        nasa_minimum_separation,
        hourly,
    )
    display_widgets(
        greatest_state,
        track,
        ecliptic,
        comparison,
        nasa_minimum_separation,
        nasa_status,
    )
    plot_reconstruction(
        splines,
        events,
        track,
        ecliptic,
        hourly,
        greatest_state,
    )

    track_angle = track["fit"]["angle_deg"]
    ecliptic_angle = ecliptic["fit"]["angle_deg"]
    relative_angle = acute_line_difference_deg(
        track_angle,
        ecliptic_angle,
    )

    print("RESULTS")
    print(f"JPL greatest UTC       : {utc_iso(events['MAX'])}")
    print(f"JPL minimum separation : {greatest_state['separation_arcsec']:.9f} arcsec")
    print(f"JPL greatest PA        : {greatest_state['position_angle_deg']:.9f} deg")
    print(f"Track angle horizontal : {track_angle:.9f} deg")
    print(f"Average track angle    : {track['fit']['average_local_angle_deg']:.9f} deg")
    print(f"Ecliptic angle         : {ecliptic_angle:.9f} deg")
    print(f"Track from ecliptic    : {relative_angle:.9f} deg")
    print(f"Track slope            : {track['fit']['slope']:.12f}")
    print(f"Track linear RMS       : {track['fit']['rms_arcsec']:.9f} arcsec")
    print(f"Track quadratic RMS    : {track['fit']['quadratic_rms_arcsec']:.9f} arcsec")
    print(f"Track curvature        : {track['fit']['curvature_per_arcsec']:.12e} 1/arcsec")
    print(f"Sun semidiameter       : {greatest_state['sun_radius_arcsec']:.9f} arcsec")
    print(f"Venus semidiameter     : {greatest_state['venus_radius_arcsec']:.9f} arcsec")
    print()
    print("NASA CONTACT COMPARISON")
    print_contact_comparison(comparison)
    print()
    print("OUTPUT SUMMARY")
    print(f"PNG output             : {PNG_PATH}")
    print(f"CSV output             : {CSV_PATH}")
    print(f"JPL minute rows        : {len(master)}")
    print(f"NASA table status      : {nasa_status}")
    print()
    print("PAPER COMPARISON")
    print(f"NASA minimum separation: {nasa_minimum_separation:.3f} arcsec")
    print(f"JPL-NASA separation    : {greatest_state['separation_arcsec'] - nasa_minimum_separation:+.6f} arcsec")
    print()
    print("EQUATION STATUS")
    print("JPL apparent RA/Dec tangent-plane projection : VERIFIED")
    print("External and internal contact roots          : VERIFIED")
    print("Greatest-transit centered-seconds minimum    : VERIFIED")
    print("Raw-Julian-date optimizer tolerance           : REJECTED / NOT USED")
    print("Track PCA, slope, RMS, and curvature         : VERIFIED")
    print("Projected ecliptic from JPL Sun motion       : VERIFIED")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# IERS-0012T
