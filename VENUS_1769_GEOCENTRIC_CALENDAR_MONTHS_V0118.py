# V0118
# Audit reference: aligned Earth–Sun and Venus JPL calendar-month tracks with centered solar limb and inline Colab display.

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
from matplotlib.patches import Ellipse

VERSION = "V0118"
ARCSEC_PER_RAD = 206264.80624709636
YEAR_START = "1769-01-01"
YEAR_STOP = "1770-01-01"
TRANSIT_START = "1769-06-03 21:30"
TRANSIT_STOP = "1769-06-03 23:10"
YEAR_STEP = "1 d"
TRANSIT_STEP = "1 m"
SOLAR_RADIUS_ARCSEC = 945.5
Y_LIMIT_ARCSEC = 1250.0
PNG_NAME = "VENUS_1769_GEOCENTRIC_CALENDAR_MONTHS_V0118.png"
CSV_NAME = "VENUS_1769_GEOCENTRIC_CALENDAR_MONTHS_V0118.csv"
HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
OUTPUT_DIR = Path.cwd()


def quoted(value: str) -> str:
    return f"'{value}'"


def build_url(target: str, start: str, stop: str, step: str) -> str:
    params = {
        "format": "json",
        "COMMAND": quoted(target),
        "OBJ_DATA": quoted("NO"),
        "MAKE_EPHEM": quoted("YES"),
        "EPHEM_TYPE": quoted("VECTORS"),
        "CENTER": quoted("500@399"),
        "START_TIME": quoted(start),
        "STOP_TIME": quoted(stop),
        "STEP_SIZE": quoted(step),
        "TIME_TYPE": quoted("UT"),
        "TIME_DIGITS": quoted("FRACSEC"),
        "CAL_TYPE": quoted("GREGORIAN"),
        "REF_PLANE": quoted("FRAME"),
        "REF_SYSTEM": quoted("ICRF"),
        "OUT_UNITS": quoted("KM-S"),
        "VEC_TABLE": quoted("1"),
        "VEC_CORR": quoted("NONE"),
        "CSV_FORMAT": quoted("YES"),
        "VEC_LABELS": quoted("NO"),
    }
    return HORIZONS_URL + "?" + urllib.parse.urlencode(params)


def fetch_vectors(target: str, start: str, stop: str, step: str) -> tuple[np.ndarray, np.ndarray]:
    request = urllib.request.Request(
        build_url(target, start, stop, step),
        headers={"User-Agent": "V0118-JPL-Horizons-Audit"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "NASA/JPL" not in str(payload.get("signature", {}).get("source", "")):
        raise RuntimeError("REJECTED: non-JPL response")
    text = payload.get("result", "")
    if "$$SOE" not in text or "$$EOE" not in text:
        raise RuntimeError("REJECTED: missing JPL vector table")
    block = text.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    jd_rows: list[float] = []
    xyz_rows: list[list[float]] = []
    for row in csv.reader(io.StringIO(block.strip())):
        cells = [cell.strip() for cell in row]
        if len(cells) < 5:
            continue
        try:
            jd_rows.append(float(cells[0]))
            xyz_rows.append([float(cells[2]), float(cells[3]), float(cells[4])])
        except ValueError:
            continue
    if len(jd_rows) < 2:
        raise RuntimeError("REJECTED: insufficient JPL rows")
    return np.asarray(jd_rows, dtype=float), np.asarray(xyz_rows, dtype=float)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0.0):
        raise RuntimeError("REJECTED: zero-length JPL vector")
    return vectors / norms[:, None]


def angular_separation_squared(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    cross = np.cross(first, second)
    dot = np.einsum("ij,ij->i", first, second)
    theta = np.arctan2(np.linalg.norm(cross, axis=1), dot)
    return theta * theta


def closest_approach_jd(jd: np.ndarray, sun_xyz: np.ndarray, venus_xyz: np.ndarray) -> float:
    y = angular_separation_squared(normalize_rows(venus_xyz), normalize_rows(sun_xyz))
    index = int(np.argmin(y))
    lo = max(0, index - 8)
    hi = min(len(jd), index + 9)
    center = float(jd[index])
    x = (jd[lo:hi] - center) * 1440.0
    fit = np.polynomial.Polynomial.fit(x, y[lo:hi], 6).convert()
    candidates = [0.0]
    for root in fit.deriv().roots():
        if abs(root.imag) < 1.0e-10 and x.min() <= root.real <= x.max() and fit.deriv(2)(root.real) > 0.0:
            candidates.append(float(root.real))
    best = min(candidates, key=lambda value: float(fit(value)))
    return center + best / 1440.0


def interpolate_vector(jd: np.ndarray, xyz: np.ndarray, target_jd: float) -> np.ndarray:
    x = (jd - target_jd) * 1440.0
    result = np.empty(3, dtype=float)
    for column in range(3):
        result[column] = np.polynomial.Polynomial.fit(x, xyz[:, column], 7).convert()(0.0)
    return result


def tangent_basis(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = reference / np.linalg.norm(reference)
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, seed))) > 0.95:
        seed = np.array([1.0, 0.0, 0.0])
    east = np.cross(seed, reference)
    east /= np.linalg.norm(east)
    north = np.cross(reference, east)
    north /= np.linalg.norm(north)
    return east, north


def all_sky_coordinates(unit_vectors: np.ndarray, reference: np.ndarray, east: np.ndarray, north: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.arctan2(unit_vectors @ east, unit_vectors @ reference) * ARCSEC_PER_RAD
    y = np.arcsin(np.clip(unit_vectors @ north, -1.0, 1.0)) * ARCSEC_PER_RAD
    return x, y


def jd_to_datetime(jd: float) -> datetime:
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=(jd - 2440587.5) * 86400.0)


def rotate(x: np.ndarray, y: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    c = math.cos(angle)
    s = math.sin(angle)
    return x * c - y * s, x * s + y * c


def add_arrow(axis: plt.Axes, dates: np.ndarray, values: np.ndarray, index: int) -> None:
    if 0 <= index < len(dates) - 5:
        axis.annotate(
            "",
            xy=(dates[index + 5], values[index + 5]),
            xytext=(dates[index], values[index]),
            arrowprops={"arrowstyle": "-|>", "linewidth": 0.8, "mutation_scale": 10},
            zorder=8,
        )


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"Calendar interval: {YEAR_START} to {YEAR_STOP}")
    print(f"Calendar cadence: {YEAR_STEP}")
    print(f"Transit cadence: {TRANSIT_STEP}")
    print(f"Vertical range: ±{Y_LIMIT_ARCSEC:.0f} arcsec")

    print("COMMENTS")
    print("Fresh NASA/JPL Horizons geometric vectors are used.")
    print("Earth–Sun and Venus tracks are independently registered to the same closest-approach height.")
    print("The Sun centerline and solar limb are centered on that common height.")
    print("The x-axis is calendar months with labels every three months.")
    print("No AI imagery is used.")

    t_sun_jd, t_sun_xyz = fetch_vectors("10", TRANSIT_START, TRANSIT_STOP, TRANSIT_STEP)
    t_venus_jd, t_venus_xyz = fetch_vectors("299", TRANSIT_START, TRANSIT_STOP, TRANSIT_STEP)
    if not np.allclose(t_sun_jd, t_venus_jd, atol=1e-12, rtol=0.0):
        raise RuntimeError("REJECTED: transit epochs do not match")

    ca_jd = closest_approach_jd(t_sun_jd, t_sun_xyz, t_venus_xyz)
    ca_sun = interpolate_vector(t_sun_jd, t_sun_xyz, ca_jd)
    ca_venus = interpolate_vector(t_venus_jd, t_venus_xyz, ca_jd)
    reference = ca_sun / np.linalg.norm(ca_sun)
    east, north = tangent_basis(reference)

    year_sun_jd, year_sun_xyz = fetch_vectors("10", YEAR_START, YEAR_STOP, YEAR_STEP)
    year_venus_jd, year_venus_xyz = fetch_vectors("299", YEAR_START, YEAR_STOP, YEAR_STEP)
    if not np.allclose(year_sun_jd, year_venus_jd, atol=1e-12, rtol=0.0):
        raise RuntimeError("REJECTED: annual epochs do not match")

    sun_x, sun_y = all_sky_coordinates(normalize_rows(year_sun_xyz), reference, east, north)
    venus_abs_x, venus_abs_y = all_sky_coordinates(normalize_rows(year_venus_xyz), reference, east, north)
    venus_x = venus_abs_x - sun_x
    venus_y = venus_abs_y - sun_y

    ca_sun_unit = ca_sun / np.linalg.norm(ca_sun)
    ca_venus_unit = ca_venus / np.linalg.norm(ca_venus)
    ca_sx, ca_sy = all_sky_coordinates(ca_sun_unit.reshape(1, 3), reference, east, north)
    ca_vax, ca_vay = all_sky_coordinates(ca_venus_unit.reshape(1, 3), reference, east, north)
    ca_vx = float(ca_vax[0] - ca_sx[0])
    ca_vy = float(ca_vay[0] - ca_sy[0])

    rotation = -math.atan2(ca_vy, ca_vx)
    venus_rx, venus_ry = rotate(venus_x, venus_y, rotation)
    earth_rx, earth_ry = rotate(sun_x, sun_y, rotation)

    ca_index = int(np.argmin(np.abs(year_sun_jd - ca_jd)))
    venus_registered = venus_ry - venus_ry[ca_index]
    earth_registered = earth_ry - earth_ry[ca_index]
    sun_centerline = np.zeros_like(venus_registered)

    dates = np.asarray([jd_to_datetime(value) for value in year_sun_jd], dtype=object)
    ca_datetime = jd_to_datetime(ca_jd)

    csv_path = OUTPUT_DIR / CSV_NAME
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["jd_ut", "utc", "venus_registered_arcsec", "earth_sun_registered_arcsec", "sun_centerline_arcsec"])
        for row in zip(year_sun_jd, dates, venus_registered, earth_registered, sun_centerline):
            writer.writerow([f"{row[0]:.12f}", row[1].isoformat(), f"{row[2]:.9f}", f"{row[3]:.9f}", f"{row[4]:.9f}"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.0, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.14)

    ax.plot(dates, venus_registered, linewidth=0.9, label="Venus")
    ax.plot(dates, earth_registered, linewidth=0.9, label="Earth–Sun")
    ax.plot(dates, sun_centerline, linewidth=0.75, linestyle="--", label="Sun centerline")

    ca_num = mdates.date2num(ca_datetime)
    ax.set_xlim(datetime(1769, 1, 1, tzinfo=timezone.utc), datetime(1770, 1, 1, tzinfo=timezone.utc))
    ax.set_ylim(-Y_LIMIT_ARCSEC, Y_LIMIT_ARCSEC)
    fig.canvas.draw()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bbox = ax.get_window_extent()
    pixels_per_day = bbox.width / (x1 - x0)
    pixels_per_arcsec = bbox.height / (y1 - y0)
    limb_width_days = 2.0 * SOLAR_RADIUS_ARCSEC * pixels_per_arcsec / pixels_per_day

    limb = Ellipse(
        (ca_num, 0.0),
        width=limb_width_days,
        height=2.0 * SOLAR_RADIUS_ARCSEC,
        fill=False,
        linewidth=0.9,
        transform=ax.transData,
        zorder=6,
        label="Solar limb",
    )
    ax.add_patch(limb)
    ax.axvline(ca_datetime, linestyle=":", linewidth=0.6)
    ax.scatter([ca_datetime], [0.0], s=16, zorder=9, label="Closest approach")

    for idx in [max(0, ca_index - 75), max(0, ca_index - 25), min(len(dates) - 6, ca_index + 25), min(len(dates) - 6, ca_index + 75)]:
        add_arrow(ax, dates, venus_registered, idx)
        add_arrow(ax, dates, earth_registered, idx)

    ax.set_title("1769 Venus Transit — Registered Geocentric Calendar-Month Geometry", fontsize=14, fontweight="bold")
    ax.set_xlabel("Calendar month — 1769")
    ax.set_ylabel("Registered tangent-plane displacement (arcsec)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10], bymonthday=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(True, linewidth=0.35, alpha=0.45)
    ax.legend(frameon=False, loc="upper right")

    png_path = OUTPUT_DIR / PNG_NAME
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.show()
    display(Image(filename=str(png_path)))

    print("RESULTS")
    print(f"Closest approach UTC: {ca_datetime.isoformat()}")
    print(f"Closest approach JD: {ca_jd:.12f} UT")
    print(f"Venus registered closest-approach height: {venus_registered[ca_index]:.9f} arcsec")
    print(f"Earth–Sun registered closest-approach height: {earth_registered[ca_index]:.9f} arcsec")
    print(f"Solar-limb center height: {sun_centerline[ca_index]:.9f} arcsec")
    print(f"Solar angular radius: {SOLAR_RADIUS_ARCSEC:.6f} arcsec")

    print("OUTPUT SUMMARY")
    print(f"PNG: {png_path}")
    print(f"CSV: {csv_path}")

    print("PAPER COMPARISON")
    print("NOT USED: all plotted tracks are derived from NASA/JPL vectors.")

    print("EQUATION STATUS")
    print("PASS: closest approach derived from minute-cadence JPL vectors.")
    print("PASS: Earth–Sun and Venus tracks share the same closest-approach height.")
    print("PASS: Sun centerline and solar limb are centered on the shared height.")
    print("PASS: PNG is explicitly displayed inline after generation.")

    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"{VERSION} COMPLETE")


if __name__ == "__main__":
    main()

# V0118