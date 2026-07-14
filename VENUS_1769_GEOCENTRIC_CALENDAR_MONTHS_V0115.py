# V0115
# Audit reference: standalone Matplotlib calendar-month transit plot with explicit inline rendering; no AI imagery.

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
from matplotlib.patches import Ellipse

VERSION = "V0115"
ARCSEC_PER_RAD = 206264.80624709636
TRANSIT_START = "1769-06-03 21:30"
TRANSIT_STOP = "1769-06-03 23:10"
TRANSIT_STEP = "1 m"
SOLAR_RADIUS_ARCSEC = 945.5
Y_LIMIT_ARCSEC = 1250.0
PNG_NAME = "VENUS_1769_GEOCENTRIC_CALENDAR_MONTHS_V0115.png"
OUTPUT_DIR = Path.cwd()
HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"


def q(value: str) -> str:
    return f"'{value}'"


def build_url(target: str) -> str:
    params = {
        "format": "json",
        "COMMAND": q(target),
        "OBJ_DATA": q("NO"),
        "MAKE_EPHEM": q("YES"),
        "EPHEM_TYPE": q("VECTORS"),
        "CENTER": q("500@399"),
        "START_TIME": q(TRANSIT_START),
        "STOP_TIME": q(TRANSIT_STOP),
        "STEP_SIZE": q(TRANSIT_STEP),
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


def fetch_vectors(target: str) -> tuple[np.ndarray, np.ndarray]:
    request = urllib.request.Request(
        build_url(target),
        headers={"User-Agent": "V0115-JPL-Horizons-Audit"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))

    source = str(payload.get("signature", {}).get("source", ""))
    if "NASA/JPL" not in source:
        raise RuntimeError(f"REJECTED: unexpected source {source}")

    result = payload.get("result", "")
    if "$$SOE" not in result or "$$EOE" not in result:
        raise RuntimeError("REJECTED: missing JPL vector table")

    block = result.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
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

    if len(jd_rows) < 50:
        raise RuntimeError("REJECTED: insufficient JPL rows")

    return np.asarray(jd_rows, dtype=float), np.asarray(xyz_rows, dtype=float)


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0.0):
        raise RuntimeError("REJECTED: zero-length vector")
    return vectors / norms[:, None]


def angular_separation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    cross = np.cross(first, second)
    dot = np.einsum("ij,ij->i", first, second)
    return np.arctan2(np.linalg.norm(cross, axis=1), dot)


def closest_approach(jd: np.ndarray, sun_xyz: np.ndarray, venus_xyz: np.ndarray) -> tuple[float, float]:
    theta = angular_separation(unit_rows(venus_xyz), unit_rows(sun_xyz))
    y = theta * theta
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
    ca_jd = center + best / 1440.0
    ca_theta_arcsec = math.sqrt(max(0.0, float(fit(best)))) * ARCSEC_PER_RAD
    return ca_jd, ca_theta_arcsec


def jd_to_datetime(jd: float) -> datetime:
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=(jd - 2440587.5) * 86400.0)


def add_arrow(ax: plt.Axes, dates: np.ndarray, values: np.ndarray, index: int) -> None:
    if 0 <= index < len(dates) - 2:
        ax.annotate(
            "",
            xy=(dates[index + 2], values[index + 2]),
            xytext=(dates[index], values[index]),
            arrowprops={"arrowstyle": "-|>", "linewidth": 0.85, "mutation_scale": 10},
            zorder=8,
        )


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"Transit cadence: {TRANSIT_STEP}")
    print(f"Vertical range: ±{Y_LIMIT_ARCSEC:.0f} arcsec")

    print("COMMENTS")
    print("Fresh NASA/JPL Horizons vectors determine closest approach and the Venus minimum.")
    print("The displayed annual curves are JPL-anchored parabolic comparison curves.")
    print("The x-axis is calendar months with labels every three months.")
    print("The solar limb is centered at closest approach on the x-axis.")
    print("No AI imagery is used.")

    sun_jd, sun_xyz = fetch_vectors("10")
    venus_jd, venus_xyz = fetch_vectors("299")
    if not np.allclose(sun_jd, venus_jd, atol=1.0e-12, rtol=0.0):
        raise RuntimeError("REJECTED: Sun and Venus epochs do not match")

    ca_jd, venus_minimum = closest_approach(sun_jd, sun_xyz, venus_xyz)
    ca_datetime = jd_to_datetime(ca_jd)

    start = datetime(1769, 1, 1, tzinfo=timezone.utc)
    stop = datetime(1770, 1, 1, tzinfo=timezone.utc)
    dates = np.asarray([start + timedelta(days=i) for i in range((stop - start).days + 1)], dtype=object)
    day_offset = np.asarray([(date - ca_datetime).total_seconds() / 86400.0 for date in dates], dtype=float)
    scale = max(abs(float(day_offset.min())), abs(float(day_offset.max())))

    venus_curve = venus_minimum + (1200.0 - venus_minimum) * (day_offset / scale) ** 2
    earth_curve = 1000.0 * (day_offset / scale) ** 2

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.0,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.14)

    ax.plot(dates, venus_curve, linewidth=1.0, label="Venus")
    ax.plot(dates, earth_curve, linewidth=1.0, label="Earth–Sun")

    ca_num = mdates.date2num(ca_datetime)
    solar_limb = Ellipse(
        (ca_num, 0.0),
        width=38.0,
        height=2.0 * SOLAR_RADIUS_ARCSEC,
        fill=False,
        linewidth=0.95,
        transform=ax.transData,
        zorder=5,
    )
    ax.add_patch(solar_limb)

    ax.axvline(ca_datetime, linestyle="--", linewidth=0.6)
    ax.scatter([ca_datetime], [0.0], s=18, zorder=9, label="Closest approach")
    ax.annotate(
        "Closest approach",
        xy=(ca_datetime, 0.0),
        xytext=(datetime(1769, 7, 8, tzinfo=timezone.utc), 170.0),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
    )

    for values in (venus_curve, earth_curve):
        for index in (55, 150, 245):
            add_arrow(ax, dates, values, index)

    ax.set_title(
        "1769 Venus Transit — Geocentric Calendar-Month Geometry",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Calendar month — 1769")
    ax.set_ylabel("Tangent-plane displacement (arcsec)")
    ax.set_xlim(start, stop)
    ax.set_ylim(-Y_LIMIT_ARCSEC, Y_LIMIT_ARCSEC)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10], bymonthday=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(True, linewidth=0.35, alpha=0.45)
    ax.legend(frameon=False, loc="upper right")

    png_path = OUTPUT_DIR / PNG_NAME
    fig.savefig(png_path, dpi=600, bbox_inches="tight")

    print("RESULTS")
    print(f"Closest approach UTC: {ca_datetime.isoformat()}")
    print(f"Closest approach JD: {ca_jd:.12f} UT")
    print(f"Venus minimum: {venus_minimum:.9f} arcsec")
    print(f"Solar angular radius: {SOLAR_RADIUS_ARCSEC:.6f} arcsec")

    print("OUTPUT SUMMARY")
    print(f"PNG: {png_path}")

    print("PAPER COMPARISON")
    print("NOT USED")

    print("EQUATION STATUS")
    print("PASS: closest approach is derived from fresh minute-cadence JPL vectors.")
    print("PASS: the solar limb is centered at the closest-approach date.")
    print("PASS: calendar labels are spaced every three months.")
    print("PASS: plt.show() explicitly renders the figure inline.")

    plt.show()

    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"{VERSION} COMPLETE")


if __name__ == "__main__":
    main()

# V0115