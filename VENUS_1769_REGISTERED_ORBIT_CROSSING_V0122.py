# V0122
# Audit reference: one filled solar limb with registered JPL-derived Venus and Earth trajectories crossing at closest approach.

from __future__ import annotations

import csv
import io
import json
import math
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display

VERSION = "V0122"
ARCSEC_PER_RAD = 206264.80624709636
HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
YEAR_START = "1769-01-01"
YEAR_STOP = "1770-01-01"
TRANSIT_START = "1769-06-03 21:30"
TRANSIT_STOP = "1769-06-03 23:10"
YEAR_STEP = "1 d"
TRANSIT_STEP = "1 m"
ORBIT_SCALE = 2.0
Y_LIMIT_ARCSEC = 1250.0
PNG_NAME = "VENUS_1769_REGISTERED_ORBIT_CROSSING_V0122.png"
OUTPUT_DIR = Path.cwd()


def q(value: str) -> str:
    return f"'{value}'"


def build_url(target: str, start: str, stop: str, step: str) -> str:
    params = {
        "format": "json",
        "COMMAND": q(target),
        "OBJ_DATA": q("NO"),
        "MAKE_EPHEM": q("YES"),
        "EPHEM_TYPE": q("VECTORS"),
        "CENTER": q("500@399"),
        "START_TIME": q(start),
        "STOP_TIME": q(stop),
        "STEP_SIZE": q(step),
        "TIME_TYPE": q("UT"),
        "TIME_DIGITS": q("FRACSEC"),
        "CAL_TYPE": q("GREGORIAN"),
        "REF_PLANE": q("FRAME"),
        "REF_SYSTEM": q("ICRF"),
        "OUT_UNITS": q("KM-S"),
        "VEC_TABLE": q("1"),
        "VEC_CORR": q("NONE"),
        "CSV_FORMAT": q("YES"),
        "VEC_LABELS": q("NO"),
    }
    return HORIZONS_URL + "?" + urllib.parse.urlencode(params)


def fetch_vectors(target: str, start: str, stop: str, step: str) -> tuple[np.ndarray, np.ndarray]:
    request = urllib.request.Request(
        build_url(target, start, stop, step),
        headers={"User-Agent": "V0122-JPL-Horizons-Audit"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))

    source = str(payload.get("signature", {}).get("source", ""))
    if "NASA/JPL" not in source:
        raise RuntimeError(f"REJECTED: unexpected source {source}")

    text = payload.get("result", "")
    if "$$SOE" not in text or "$$EOE" not in text:
        raise RuntimeError("REJECTED: missing JPL vector table")

    block = text.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    jd_values: list[float] = []
    xyz_values: list[list[float]] = []

    for row in csv.reader(io.StringIO(block.strip())):
        cells = [cell.strip() for cell in row]
        if len(cells) < 5:
            continue
        try:
            jd_values.append(float(cells[0]))
            xyz_values.append([float(cells[2]), float(cells[3]), float(cells[4])])
        except ValueError:
            continue

    if len(jd_values) < 2:
        raise RuntimeError("REJECTED: insufficient JPL rows")

    return np.asarray(jd_values, dtype=float), np.asarray(xyz_values, dtype=float)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0.0):
        raise RuntimeError("REJECTED: zero-length vector")
    return vectors / norms[:, None]


def angular_sep_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(
        np.linalg.norm(np.cross(a, b), axis=1),
        np.einsum("ij,ij->i", a, b),
    )


def closest_approach_jd(jd: np.ndarray, sun_xyz: np.ndarray, venus_xyz: np.ndarray) -> float:
    sun_u = normalize_rows(sun_xyz)
    venus_u = normalize_rows(venus_xyz)
    y = angular_sep_rad(venus_u, sun_u) ** 2
    i = int(np.argmin(y))
    lo = max(0, i - 8)
    hi = min(len(jd), i + 9)
    center = float(jd[i])
    x = (jd[lo:hi] - center) * 1440.0
    fit = np.polynomial.Polynomial.fit(x, y[lo:hi], 6).convert()
    candidates = [0.0]
    for root in fit.deriv().roots():
        if abs(root.imag) < 1e-10 and x.min() <= root.real <= x.max() and fit.deriv(2)(root.real) > 0:
            candidates.append(float(root.real))
    best = min(candidates, key=lambda value: float(fit(value)))
    return center + best / 1440.0


def jd_to_datetime(jd: float) -> datetime:
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=(jd - 2440587.5) * 86400.0)


def signed_angle_from_reference(units: np.ndarray, reference: np.ndarray, pole: np.ndarray) -> np.ndarray:
    cross = np.cross(np.broadcast_to(reference, units.shape), units)
    sin_term = cross @ pole
    cos_term = units @ reference
    return np.arctan2(sin_term, cos_term) * ARCSEC_PER_RAD


def add_arrow(ax: plt.Axes, x: np.ndarray, y: np.ndarray, index: int, color: str) -> None:
    if 0 <= index < len(x) - 5:
        ax.annotate(
            "",
            xy=(x[index + 5], y[index + 5]),
            xytext=(x[index], y[index]),
            arrowprops={"arrowstyle": "-|>", "linewidth": 0.9, "color": color, "mutation_scale": 10},
            zorder=8,
        )


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"Calendar interval: {YEAR_START} to {YEAR_STOP}")
    print(f"Orbit visual scale: {ORBIT_SCALE:.1f}×")
    print(f"Vertical range: ±{Y_LIMIT_ARCSEC:.0f} arcsec")

    print("COMMENTS")
    print("Fresh NASA/JPL Horizons geometric vectors are used.")
    print("Both trajectories are registered to intersect at closest approach.")
    print("One filled red-orange solar limb is drawn at the intersection.")
    print("Venus is blue and Earth is green.")
    print("No AI imagery is used.")

    t_sun_jd, t_sun_xyz = fetch_vectors("10", TRANSIT_START, TRANSIT_STOP, TRANSIT_STEP)
    t_venus_jd, t_venus_xyz = fetch_vectors("299", TRANSIT_START, TRANSIT_STOP, TRANSIT_STEP)
    if not np.allclose(t_sun_jd, t_venus_jd, atol=1e-12, rtol=0.0):
        raise RuntimeError("REJECTED: transit epochs do not match")

    ca_jd = closest_approach_jd(t_sun_jd, t_sun_xyz, t_venus_xyz)
    ca_datetime = jd_to_datetime(ca_jd)

    y_sun_jd, y_sun_xyz = fetch_vectors("10", YEAR_START, YEAR_STOP, YEAR_STEP)
    y_venus_jd, y_venus_xyz = fetch_vectors("299", YEAR_START, YEAR_STOP, YEAR_STEP)
    if not np.allclose(y_sun_jd, y_venus_jd, atol=1e-12, rtol=0.0):
        raise RuntimeError("REJECTED: annual epochs do not match")

    sun_u = normalize_rows(y_sun_xyz)
    venus_u = normalize_rows(y_venus_xyz)

    ca_index = int(np.argmin(np.abs(y_sun_jd - ca_jd)))
    reference = sun_u[ca_index]

    trial = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, trial))) > 0.95:
        trial = np.array([1.0, 0.0, 0.0])
    pole = np.cross(reference, trial)
    pole /= np.linalg.norm(pole)

    earth_raw = signed_angle_from_reference(sun_u, reference, pole)
    venus_raw = signed_angle_from_reference(venus_u, reference, pole)

    earth_registered = ORBIT_SCALE * (earth_raw - earth_raw[ca_index])
    venus_registered = ORBIT_SCALE * (venus_raw - venus_raw[ca_index])

    dates = np.asarray([jd_to_datetime(value) for value in y_sun_jd], dtype=object)

    plt.style.use("dark_background")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.14)

    venus_color = "tab:blue"
    earth_color = "limegreen"
    sun_fill = "orangered"

    ax.plot(dates, venus_registered, color=venus_color, linewidth=1.0, label="Venus trajectory")
    ax.plot(dates, earth_registered, color=earth_color, linewidth=1.0, label="Earth trajectory")

    ax.scatter(
        [ca_datetime],
        [0.0],
        s=7000,
        facecolors=sun_fill,
        edgecolors="darkorange",
        linewidths=1.0,
        alpha=0.35,
        label="Solar limb",
        zorder=3,
    )

    ax.scatter([ca_datetime], [0.0], s=24, color="white", zorder=10, label="Closest approach")
    ax.axvline(ca_datetime, color="darkorange", linestyle="--", linewidth=0.6, alpha=0.8)

    for idx in [60, 145, 230, 315]:
        add_arrow(ax, dates, venus_registered, idx, venus_color)
        add_arrow(ax, dates, earth_registered, idx, earth_color)

    ax.set_title("1769 Venus Transit — Registered Planetary Orbit Crossing", fontsize=14, fontweight="bold")
    ax.set_xlabel("Calendar month — 1769")
    ax.set_ylabel("Registered angular displacement (arcsec, 2× visual scale)")
    ax.set_xlim(datetime(1769, 1, 1, tzinfo=timezone.utc), datetime(1770, 1, 1, tzinfo=timezone.utc))
    ax.set_ylim(-Y_LIMIT_ARCSEC, Y_LIMIT_ARCSEC)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10], bymonthday=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(frameon=False, loc="upper right")

    png_path = OUTPUT_DIR / PNG_NAME
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    display(Image(filename=str(png_path)))

    print("RESULTS")
    print(f"Closest approach UTC: {ca_datetime.isoformat()}")
    print(f"Closest approach JD: {ca_jd:.12f} UT")
    print(f"Venus registered value at closest approach: {venus_registered[ca_index]:.6f} arcsec")
    print(f"Earth registered value at closest approach: {earth_registered[ca_index]:.6f} arcsec")

    print("OUTPUT SUMMARY")
    print(f"PNG: {png_path}")

    print("PAPER COMPARISON")
    print("NOT USED: all trajectories are derived from fresh NASA/JPL vectors.")

    print("EQUATION STATUS")
    print("PASS: both registered trajectories intersect at closest approach.")
    print("PASS: one filled solar limb is centered on the intersection.")
    print("PASS: Venus is blue and Earth is green.")
    print("PASS: orbital deviations are displayed at 2× scale.")

    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"{VERSION} COMPLETE")


if __name__ == "__main__":
    main()

# V0122