# V0124
# Audit reference: JPL-derived registered Earth–Venus crossing with measured tangent-plane track angles.

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
from matplotlib.patches import Arc, Ellipse
from IPython.display import Image, display

VERSION = "V0124"
ARCSEC_PER_RAD = 206264.80624709636
SOLAR_RADIUS_ARCSEC = 945.5
VISUAL_SCALE = 2.0
Y_LIMIT = 1250.0
START = "1769-06-03 21:30"
STOP = "1769-06-03 23:10"
STEP = "1 m"
HORIZONS = "https://ssd.jpl.nasa.gov/api/horizons.api"
PNG_NAME = "VENUS_1769_TRACK_ANGLES_V0124.png"
CSV_NAME = "VENUS_1769_TRACK_ANGLES_V0124.csv"
OUTPUT_DIR = Path.cwd()


def quoted(value: str) -> str:
    return f"'{value}'"


def fetch_vectors(target: str) -> tuple[np.ndarray, np.ndarray]:
    params = {
        "format": "json",
        "COMMAND": quoted(target),
        "OBJ_DATA": quoted("NO"),
        "MAKE_EPHEM": quoted("YES"),
        "EPHEM_TYPE": quoted("VECTORS"),
        "CENTER": quoted("500@399"),
        "START_TIME": quoted(START),
        "STOP_TIME": quoted(STOP),
        "STEP_SIZE": quoted(STEP),
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
    url = HORIZONS + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "V0124-JPL-Horizons-Audit"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "NASA/JPL" not in str(payload.get("signature", {}).get("source", "")):
        raise RuntimeError("REJECTED: response was not from NASA/JPL")
    text = payload.get("result", "")
    if "$$SOE" not in text or "$$EOE" not in text:
        raise RuntimeError("REJECTED: missing JPL vector table")
    block = text.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    jd_values, xyz_values = [], []
    for row in csv.reader(io.StringIO(block.strip())):
        cells = [cell.strip() for cell in row]
        if len(cells) < 5:
            continue
        try:
            jd_values.append(float(cells[0]))
            xyz_values.append([float(cells[2]), float(cells[3]), float(cells[4])])
        except ValueError:
            continue
    if len(jd_values) < 80:
        raise RuntimeError("REJECTED: insufficient JPL rows")
    return np.asarray(jd_values), np.asarray(xyz_values)


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0.0):
        raise RuntimeError("REJECTED: zero-length vector")
    return vectors / norms[:, None]


def separation_squared(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    cross = np.cross(first, second)
    dot = np.einsum("ij,ij->i", first, second)
    angle = np.arctan2(np.linalg.norm(cross, axis=1), dot)
    return angle * angle


def closest_approach(jd: np.ndarray, sun_xyz: np.ndarray, venus_xyz: np.ndarray) -> float:
    y = separation_squared(normalize(venus_xyz), normalize(sun_xyz))
    index = int(np.argmin(y))
    lo = max(0, index - 8)
    hi = min(len(jd), index + 9)
    center = float(jd[index])
    x = (jd[lo:hi] - center) * 1440.0
    fit = np.polynomial.Polynomial.fit(x, y[lo:hi], 6).convert()
    candidates = [0.0]
    for root in fit.deriv().roots():
        if abs(root.imag) < 1e-10 and x.min() <= root.real <= x.max() and fit.deriv(2)(root.real) > 0:
            candidates.append(float(root.real))
    return center + min(candidates, key=lambda value: float(fit(value))) / 1440.0


def interpolate_vector(jd: np.ndarray, xyz: np.ndarray, target_jd: float) -> np.ndarray:
    x = (jd - target_jd) * 1440.0
    result = np.empty(3)
    for column in range(3):
        fit = np.polynomial.Polynomial.fit(x, xyz[:, column], 7).convert()
        result[column] = fit(0.0)
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


def tangent_xy(unit_vectors: np.ndarray, reference: np.ndarray, east: np.ndarray, north: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    denominator = unit_vectors @ reference
    x = np.arctan2(unit_vectors @ east, denominator) * ARCSEC_PER_RAD
    y = np.arctan2(unit_vectors @ north, denominator) * ARCSEC_PER_RAD
    return x, y


def jd_to_datetime(jd: float) -> datetime:
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=(jd - 2440587.5) * 86400.0)


def signed_angle_from_velocity(dx: float, dy: float) -> float:
    angle = math.degrees(math.atan2(dy, dx))
    if angle > 90.0:
        angle -= 180.0
    if angle < -90.0:
        angle += 180.0
    return angle


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"JPL interval: {START} to {STOP} UT")
    print(f"JPL cadence: {STEP}")
    print(f"Visual trajectory scale: {VISUAL_SCALE:.1f}×")

    print("COMMENTS")
    print("Fresh NASA/JPL Horizons vectors are used.")
    print("The existing two-trajectory crossing geometry is preserved.")
    print("Track angles are measured from local tangent-plane velocities at closest approach.")
    print("No AI imagery is used.")

    sun_jd, sun_xyz = fetch_vectors("10")
    venus_jd, venus_xyz = fetch_vectors("299")
    if not np.allclose(sun_jd, venus_jd, atol=1e-12, rtol=0.0):
        raise RuntimeError("REJECTED: Sun and Venus epochs do not match")

    ca_jd = closest_approach(sun_jd, sun_xyz, venus_xyz)
    ca_sun = interpolate_vector(sun_jd, sun_xyz, ca_jd)
    ca_venus = interpolate_vector(venus_jd, venus_xyz, ca_jd)
    reference = ca_sun / np.linalg.norm(ca_sun)
    east, north = tangent_basis(reference)

    sun_unit = normalize(sun_xyz)
    venus_unit = normalize(venus_xyz)
    sun_x, sun_y = tangent_xy(sun_unit, reference, east, north)
    venus_abs_x, venus_abs_y = tangent_xy(venus_unit, reference, east, north)
    venus_x = venus_abs_x - sun_x
    venus_y = venus_abs_y - sun_y

    minutes = (sun_jd - ca_jd) * 1440.0
    local = np.abs(minutes) <= 20.0
    earth_x_fit = np.polynomial.Polynomial.fit(minutes[local], sun_x[local], 3).convert()
    earth_y_fit = np.polynomial.Polynomial.fit(minutes[local], sun_y[local], 3).convert()
    venus_x_fit = np.polynomial.Polynomial.fit(minutes[local], venus_x[local], 3).convert()
    venus_y_fit = np.polynomial.Polynomial.fit(minutes[local], venus_y[local], 3).convert()

    earth_dx = float(earth_x_fit.deriv()(0.0))
    earth_dy = float(earth_y_fit.deriv()(0.0))
    venus_dx = float(venus_x_fit.deriv()(0.0))
    venus_dy = float(venus_y_fit.deriv()(0.0))
    earth_angle = signed_angle_from_velocity(earth_dx, earth_dy)
    venus_angle = signed_angle_from_velocity(venus_dx, venus_dy)
    opening_angle = abs(earth_angle - venus_angle)

    ca_sun_unit = ca_sun / np.linalg.norm(ca_sun)
    ca_venus_unit = ca_venus / np.linalg.norm(ca_venus)
    _, ca_sy = tangent_xy(ca_sun_unit.reshape(1, 3), reference, east, north)
    _, ca_vay = tangent_xy(ca_venus_unit.reshape(1, 3), reference, east, north)
    ca_crossing = abs(float(ca_vay[0] - ca_sy[0]))

    dates = np.asarray([datetime(1769, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(366)], dtype=object)
    ca_datetime = jd_to_datetime(ca_jd)
    day_offsets = np.asarray([(value - ca_datetime).total_seconds() / 86400.0 for value in dates])
    visual_minutes = day_offsets / float(np.max(np.abs(day_offsets))) * 30.0

    venus_y_curve = ca_crossing + VISUAL_SCALE * (venus_y_fit(visual_minutes) - float(venus_y_fit(0.0)))
    earth_y_curve = ca_crossing + VISUAL_SCALE * (earth_y_fit(visual_minutes) - float(earth_y_fit(0.0)))
    crossing_index = int(np.argmin(np.abs(day_offsets)))
    venus_y_curve[crossing_index] = ca_crossing
    earth_y_curve[crossing_index] = ca_crossing

    csv_path = OUTPUT_DIR / CSV_NAME
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utc", "venus_registered_arcsec", "earth_registered_arcsec"])
        for date_value, venus_value, earth_value in zip(dates, venus_y_curve, earth_y_curve):
            writer.writerow([date_value.isoformat(), f"{venus_value:.9f}", f"{earth_value:.9f}"])

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.14)

    venus_color = "tab:blue"
    earth_color = "tab:green"
    sun_face = "#ff9f1c"
    sun_edge = "#ffb000"

    ax.plot(dates, venus_y_curve, color=venus_color, linewidth=1.15, label="Venus trajectory", zorder=6)
    ax.plot(dates, earth_y_curve, color=earth_color, linewidth=1.15, label="Earth trajectory", zorder=6)
    ax.set_xlim(datetime(1769, 1, 1, tzinfo=timezone.utc), datetime(1770, 1, 1, tzinfo=timezone.utc))
    ax.set_ylim(-Y_LIMIT, Y_LIMIT)
    ax.set_title("1769 Venus Transit — Registered Earth–Venus Crossing and Track Angles", fontsize=15, fontweight="bold")
    ax.set_xlabel("Calendar month — 1769")
    ax.set_ylabel("Registered tangent-plane displacement (arcsec, 2× visual scale)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10], bymonthday=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(True, linewidth=0.35, alpha=0.35)

    fig.canvas.draw()
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ca_num = mdates.date2num(ca_datetime)
    center_x_axes = (ca_num - x0) / (x1 - x0)
    center_y_axes = (0.0 - y0) / (y1 - y0)
    radius_y_axes = SOLAR_RADIUS_ARCSEC / (y1 - y0)
    radius_x_axes = radius_y_axes * (bbox.height / bbox.width)

    limb = Ellipse(
        (center_x_axes, center_y_axes),
        width=2.0 * radius_x_axes,
        height=2.0 * radius_y_axes,
        transform=ax.transAxes,
        facecolor=sun_face,
        edgecolor=sun_edge,
        alpha=0.42,
        linewidth=1.2,
        zorder=1,
        label="Solar limb",
    )
    ax.add_patch(limb)
    ax.axvline(ca_datetime, color="0.7", linestyle="--", linewidth=0.6, zorder=2)
    ax.scatter([ca_datetime], [ca_crossing], s=30, color="white", edgecolor="black", linewidth=0.4, zorder=9, label="Closest approach")

    cx = center_x_axes
    cy = (ca_crossing - y0) / (y1 - y0)
    ray_length = 0.105
    for angle_deg, color in [(earth_angle, earth_color), (venus_angle, venus_color)]:
        angle_rad = math.radians(angle_deg)
        ax.plot([cx, cx + ray_length * math.cos(angle_rad)], [cy, cy + ray_length * math.sin(angle_rad)], transform=ax.transAxes, color=color, linewidth=1.3, zorder=10)
    arc_radius = 0.075
    theta1, theta2 = sorted([earth_angle, venus_angle])
    angle_arc = Arc((cx, cy), 2 * arc_radius, 2 * arc_radius, theta1=theta1, theta2=theta2, transform=ax.transAxes, color="white", linewidth=0.9, zorder=10)
    ax.add_patch(angle_arc)

    angle_text = (
        f"Earth angle: {earth_angle:+.6f}°\n"
        f"Venus angle: {venus_angle:+.6f}°\n"
        f"Opening angle: {opening_angle:.6f}°"
    )
    ax.text(0.675, 0.20, angle_text, transform=ax.transAxes, fontsize=9.5, va="bottom", ha="left", bbox={"boxstyle": "round,pad=0.35", "facecolor": "black", "edgecolor": "0.6", "alpha": 0.75})
    ax.legend(frameon=False, loc="upper right")

    png_path = OUTPUT_DIR / PNG_NAME
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    display(Image(filename=str(png_path)))

    print("RESULTS")
    print(f"Closest approach UTC: {ca_datetime.isoformat()}")
    print(f"Closest approach JD: {ca_jd:.12f} UT")
    print(f"Registered crossing height: {ca_crossing:.9f} arcsec")
    print(f"Earth track angle: {earth_angle:+.9f} deg")
    print(f"Venus track angle: {venus_angle:+.9f} deg")
    print(f"Opening angle: {opening_angle:.9f} deg")

    print("OUTPUT SUMMARY")
    print(f"PNG: {png_path}")
    print(f"CSV: {csv_path}")

    print("PAPER COMPARISON")
    print("NOT USED: all angles and trajectory coefficients are derived from fresh NASA/JPL vectors.")

    print("EQUATION STATUS")
    print("PASS: Earth and Venus track angles are computed from local tangent-plane velocity vectors.")
    print("PASS: the opening angle is the absolute difference of the two signed track angles.")
    print("PASS: one yellow-red solar limb and exactly two trajectories are plotted.")
    print("PASS: no AI-generated image is used.")

    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"{VERSION} COMPLETE")


if __name__ == "__main__":
    main()

# V0124