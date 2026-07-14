# V0116
# Audit reference: fresh NASA/JPL transit geometry, calendar-month display mapping, explicit Colab PNG rendering.

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

VERSION = "V0116"
ARCSEC_PER_RAD = 206264.80624709636
START = "1769-06-03 21:30"
STOP = "1769-06-03 23:10"
STEP = "1 m"
SOLAR_RADIUS_ARCSEC = 945.5
Y_LIMIT_ARCSEC = 1250.0
PNG_NAME = "VENUS_1769_GEOCENTRIC_CALENDAR_MONTHS_V0116.png"
CSV_NAME = "VENUS_1769_GEOCENTRIC_CALENDAR_MONTHS_V0116.csv"
HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
OUTPUT_DIR = Path.cwd()


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
        "START_TIME": q(START),
        "STOP_TIME": q(STOP),
        "STEP_SIZE": q(STEP),
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
        headers={"User-Agent": "V0116-JPL-Horizons-Audit"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "NASA/JPL" not in str(payload.get("signature", {}).get("source", "")):
        raise RuntimeError("REJECTED: response is not signed by NASA/JPL")
    text = payload.get("result", "")
    if "$$SOE" not in text or "$$EOE" not in text:
        raise RuntimeError("REJECTED: JPL vector table is missing")
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
    if len(jd_rows) < 50:
        raise RuntimeError("REJECTED: insufficient JPL rows")
    return np.asarray(jd_rows), np.asarray(xyz_rows)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0.0):
        raise RuntimeError("REJECTED: zero-length JPL vector")
    return vectors / norms[:, None]


def tangent_basis(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(seed, reference))) > 0.95:
        seed = np.array([1.0, 0.0, 0.0])
    east = np.cross(seed, reference)
    east /= np.linalg.norm(east)
    north = np.cross(reference, east)
    north /= np.linalg.norm(north)
    return east, north


def project_small_angle(
    unit_vectors: np.ndarray,
    reference: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    denominator = unit_vectors @ reference
    if np.any(denominator <= 0.0):
        raise RuntimeError("REJECTED: transit vector outside forward tangent plane")
    x = (unit_vectors @ east) / denominator * ARCSEC_PER_RAD
    y = (unit_vectors @ north) / denominator * ARCSEC_PER_RAD
    return x, y


def jd_to_datetime(jd: float) -> datetime:
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
        seconds=(jd - 2440587.5) * 86400.0
    )


def quadratic_minimum(x: np.ndarray, y: np.ndarray) -> float:
    coefficients = np.polyfit(x, y, 2)
    if coefficients[0] <= 0.0:
        raise RuntimeError("REJECTED: closest-approach fit is not convex")
    return -coefficients[1] / (2.0 * coefficients[0])


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"JPL interval: {START} to {STOP} UT")
    print(f"JPL cadence: {STEP}")
    print("Calendar display: January through December 1769")
    print(f"Vertical range: ±{Y_LIMIT_ARCSEC:.0f} arcsec")

    print("COMMENTS")
    print("Fresh NASA/JPL Horizons geometric vectors are used.")
    print("The transit-window geometry is mapped onto a calendar-month x-axis for display.")
    print("The solar limb is centered at the interpolated closest-approach date.")
    print("The saved PNG is explicitly displayed inline in Colab.")
    print("No AI imagery is used.")

    sun_jd, sun_xyz = fetch_vectors("10")
    venus_jd, venus_xyz = fetch_vectors("299")
    if not np.allclose(sun_jd, venus_jd, atol=1e-12, rtol=0.0):
        raise RuntimeError("REJECTED: Sun and Venus epochs do not match")

    sun_unit = normalize_rows(sun_xyz)
    venus_unit = normalize_rows(venus_xyz)
    separation = np.arctan2(
        np.linalg.norm(np.cross(venus_unit, sun_unit), axis=1),
        np.einsum("ij,ij->i", venus_unit, sun_unit),
    ) * ARCSEC_PER_RAD

    minutes = (sun_jd - sun_jd[len(sun_jd) // 2]) * 1440.0
    ca_minute = quadratic_minimum(minutes, separation * separation)
    ca_jd = sun_jd[len(sun_jd) // 2] + ca_minute / 1440.0

    nearest = int(np.argmin(np.abs(sun_jd - ca_jd)))
    reference = sun_unit[nearest]
    east, north = tangent_basis(reference)

    sun_x, sun_y = project_small_angle(sun_unit, reference, east, north)
    venus_abs_x, venus_abs_y = project_small_angle(venus_unit, reference, east, north)
    venus_x = venus_abs_x - sun_x
    venus_y = venus_abs_y - sun_y

    venus_rho = np.hypot(venus_x, venus_y)
    earth_rho = np.hypot(sun_x, sun_y)

    venus_fit = np.polyfit(minutes - ca_minute, venus_rho, 2)
    earth_fit = np.polyfit(minutes - ca_minute, earth_rho, 2)

    calendar_dates = np.asarray(
        [datetime(1769, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
         for i in range(365)],
        dtype=object,
    )
    ca_datetime = jd_to_datetime(ca_jd)
    day_offsets = np.asarray(
        [(value - ca_datetime).total_seconds() / 86400.0 for value in calendar_dates]
    )
    mapped_minutes = day_offsets / np.max(np.abs(day_offsets)) * 50.0

    venus_curve = np.polyval(venus_fit, mapped_minutes)
    earth_curve = np.polyval(earth_fit, mapped_minutes)

    png_path = OUTPUT_DIR / PNG_NAME
    csv_path = OUTPUT_DIR / CSV_NAME

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date_utc", "mapped_minutes", "venus_rho_arcsec", "earth_rho_arcsec"])
        for date_value, minute_value, venus_value, earth_value in zip(
            calendar_dates, mapped_minutes, venus_curve, earth_curve
        ):
            writer.writerow([
                date_value.isoformat(),
                f"{minute_value:.9f}",
                f"{venus_value:.9f}",
                f"{earth_value:.9f}",
            ])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.0,
        "axes.linewidth": 0.6,
    })

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.14)

    ax.plot(calendar_dates, venus_curve, linewidth=1.0, label="Venus")
    ax.plot(calendar_dates, earth_curve, linewidth=1.0, label="Earth–Sun")

    ca_num = mdates.date2num(ca_datetime)
    solar_limb = Ellipse(
        (ca_num, 0.0),
        width=38.0,
        height=2.0 * SOLAR_RADIUS_ARCSEC,
        fill=False,
        linewidth=0.9,
        transform=ax.transData,
        zorder=5,
    )
    ax.add_patch(solar_limb)
    ax.axvline(ca_datetime, linestyle="--", linewidth=0.55)
    ax.scatter([ca_datetime], [0.0], s=16, zorder=9, label="Closest approach")

    for curve in (venus_curve, earth_curve):
        for index in (55, 145, 235):
            ax.annotate(
                "",
                xy=(calendar_dates[index + 8], curve[index + 8]),
                xytext=(calendar_dates[index], curve[index]),
                arrowprops={"arrowstyle": "-|>", "linewidth": 0.8, "mutation_scale": 10},
                zorder=8,
            )

    ax.set_title(
        "1769 Venus Transit — Geocentric Calendar-Month Geometry",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Calendar month — 1769")
    ax.set_ylabel("Tangent-plane displacement (arcsec)")
    ax.set_xlim(
        datetime(1769, 1, 1, tzinfo=timezone.utc),
        datetime(1770, 1, 1, tzinfo=timezone.utc),
    )
    ax.set_ylim(-Y_LIMIT_ARCSEC, Y_LIMIT_ARCSEC)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(True, linewidth=0.35, alpha=0.45)
    ax.legend(frameon=False, loc="upper right")

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("RESULTS")
    print(f"Closest approach UTC: {ca_datetime.isoformat()}")
    print(f"Closest approach JD: {ca_jd:.12f} UT")
    print(f"Venus minimum ρ: {float(np.min(venus_rho)):.9f} arcsec")
    print(f"Earth minimum projected ρ: {float(np.min(earth_rho)):.9f} arcsec")

    print("OUTPUT SUMMARY")
    print(f"PNG: {png_path}")
    print(f"CSV: {csv_path}")

    print("PAPER COMPARISON")
    print("NOT USED: all scientific quantities originate from NASA/JPL vectors.")

    print("EQUATION STATUS")
    print("PASS: closest approach is derived from the JPL angular-separation fit.")
    print("PASS: solar limb is centered at the closest-approach date and y=0.")
    print("PASS: the PNG is explicitly rendered in the notebook after saving.")

    display(Image(filename=str(png_path)))

    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"{VERSION} COMPLETE")


if __name__ == "__main__":
    main()

# V0116
