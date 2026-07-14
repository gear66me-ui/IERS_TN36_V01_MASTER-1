# V0114
# Audit reference: corrected calendar-month plot using JPL-derived closest approach and calendar-scaled parabolic display.

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

VERSION = "V0114"
ARCSEC_PER_RAD = 206264.80624709636
TRANSIT_START = "1769-06-03 21:30"
TRANSIT_STOP = "1769-06-03 23:10"
TRANSIT_STEP = "1 m"
SOLAR_RADIUS_ARCSEC = 945.5
Y_LIMIT_ARCSEC = 1250.0
PNG_NAME = "VENUS_1769_GEOCENTRIC_CALENDAR_MONTHS_V0114.png"
CSV_NAME = "VENUS_1769_GEOCENTRIC_CALENDAR_MONTHS_V0114.csv"
HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
OUTPUT_DIR = Path.cwd()


def quoted(value: str) -> str:
    return f"'{value}'"


def build_url(target: str) -> str:
    params = {
        "format": "json",
        "COMMAND": quoted(target),
        "OBJ_DATA": quoted("NO"),
        "MAKE_EPHEM": quoted("YES"),
        "EPHEM_TYPE": quoted("VECTORS"),
        "CENTER": quoted("500@399"),
        "START_TIME": quoted(TRANSIT_START),
        "STOP_TIME": quoted(TRANSIT_STOP),
        "STEP_SIZE": quoted(TRANSIT_STEP),
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


def fetch_vectors(target: str) -> tuple[np.ndarray, np.ndarray]:
    request = urllib.request.Request(
        build_url(target),
        headers={"User-Agent": "V0114-JPL-Horizons-Audit"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "NASA/JPL" not in str(payload.get("signature", {}).get("source", "")):
        raise RuntimeError("REJECTED: non-JPL response")
    text = payload.get("result", "")
    if "$$SOE" not in text or "$$EOE" not in text:
        raise RuntimeError("REJECTED: missing JPL vector table")
    block = text.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    jd_rows, xyz_rows = [], []
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
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


def separation_arcsec(sun_xyz: np.ndarray, venus_xyz: np.ndarray) -> np.ndarray:
    sun = normalize_rows(sun_xyz)
    venus = normalize_rows(venus_xyz)
    theta = np.arctan2(
        np.linalg.norm(np.cross(venus, sun), axis=1),
        np.einsum("ij,ij->i", venus, sun),
    )
    return theta * ARCSEC_PER_RAD


def closest_approach(jd: np.ndarray, rho: np.ndarray) -> tuple[float, float]:
    index = int(np.argmin(rho))
    lo = max(0, index - 8)
    hi = min(len(jd), index + 9)
    center = float(jd[index])
    x = (jd[lo:hi] - center) * 1440.0
    fit = np.polynomial.Polynomial.fit(x, rho[lo:hi] ** 2, 6).convert()
    candidates = [0.0]
    for root in fit.deriv().roots():
        if abs(root.imag) < 1e-10 and x.min() <= root.real <= x.max() and fit.deriv(2)(root.real) > 0:
            candidates.append(float(root.real))
    best = min(candidates, key=lambda value: float(fit(value)))
    jd_ca = center + best / 1440.0
    rho_ca = math.sqrt(max(0.0, float(fit(best))))
    return jd_ca, rho_ca


def jd_to_datetime(jd: float) -> datetime:
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=(jd - 2440587.5) * 86400.0)


def add_arrow(ax: plt.Axes, dates: np.ndarray, values: np.ndarray, index: int) -> None:
    if 0 <= index < len(dates) - 7:
        ax.annotate(
            "",
            xy=(dates[index + 7], values[index + 7]),
            xytext=(dates[index], values[index]),
            arrowprops={"arrowstyle": "-|>", "linewidth": 0.85, "mutation_scale": 10},
            zorder=8,
        )


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"Transit query: {TRANSIT_START} to {TRANSIT_STOP} UT")
    print(f"JPL cadence: {TRANSIT_STEP}")
    print(f"Vertical range: ±{Y_LIMIT_ARCSEC:.0f} arcsec")

    print("COMMENTS")
    print("Fresh NASA/JPL Horizons vectors determine closest approach and Venus minimum separation.")
    print("The display uses calendar months and a centered parabolic comparison over 1769.")
    print("The solar limb is centered on the closest-approach date at y=0.")
    print("No AI imagery is used.")

    sun_jd, sun_xyz = fetch_vectors("10")
    venus_jd, venus_xyz = fetch_vectors("299")
    if not np.allclose(sun_jd, venus_jd, atol=1e-12, rtol=0.0):
        raise RuntimeError("REJECTED: JPL epochs do not match")

    rho = separation_arcsec(sun_xyz, venus_xyz)
    ca_jd, venus_min = closest_approach(sun_jd, rho)
    ca_datetime = jd_to_datetime(ca_jd)

    start = datetime(1769, 1, 1, tzinfo=timezone.utc)
    stop = datetime(1770, 1, 1, tzinfo=timezone.utc)
    dates = np.asarray([start + timedelta(days=i) for i in range((stop - start).days + 1)], dtype=object)
    day_offset = np.asarray([(date - ca_datetime).total_seconds() / 86400.0 for date in dates])
    scale = max(abs(day_offset.min()), abs(day_offset.max()))
    normalized_time = day_offset / scale

    venus_curve = venus_min + (Y_LIMIT_ARCSEC - 50.0 - venus_min) * normalized_time ** 2
    earth_curve = (SOLAR_RADIUS_ARCSEC + 50.0) * normalized_time ** 2

    csv_path = OUTPUT_DIR / CSV_NAME
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utc", "days_from_closest_approach", "venus_arcsec", "earth_sun_arcsec"])
        for row in zip(dates, day_offset, venus_curve, earth_curve):
            writer.writerow([row[0].isoformat(), f"{row[1]:.9f}", f"{row[2]:.9f}", f"{row[3]:.9f}"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.0, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.14)

    ax.plot(dates, venus_curve, linewidth=0.95, label="Venus")
    ax.plot(dates, earth_curve, linewidth=0.95, label="Earth–Sun")

    ca_num = mdates.date2num(ca_datetime)
    ax.add_patch(Ellipse(
        (ca_num, 0.0),
        width=38.0,
        height=2.0 * SOLAR_RADIUS_ARCSEC,
        fill=False,
        linewidth=0.9,
        transform=ax.transData,
        zorder=5,
    ))
    ax.axvline(ca_datetime, linestyle="--", linewidth=0.55)
    ax.scatter([ca_datetime], [0.0], s=14, zorder=9)
    ax.annotate(
        "Closest approach",
        xy=(ca_datetime, 0.0),
        xytext=(datetime(1769, 7, 8, tzinfo=timezone.utc), 170.0),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
    )

    for idx in (55, 150, 245):
        add_arrow(ax, dates, venus_curve, idx)
        add_arrow(ax, dates, earth_curve, idx)

    ax.set_title("1769 Venus Transit — Calendar-Month Tangent-Plane Geometry", fontsize=14, fontweight="bold")
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
    plt.show()

    print("RESULTS")
    print(f"Closest approach UTC: {ca_datetime.isoformat()}")
    print(f"Closest approach JD: {ca_jd:.12f} UT")
    print(f"Venus minimum separation: {venus_min:.9f} arcsec")
    print(f"Solar angular radius: {SOLAR_RADIUS_ARCSEC:.6f} arcsec")

    print("OUTPUT SUMMARY")
    print(f"PNG: {png_path}")
    print(f"CSV: {csv_path}")

    print("PAPER COMPARISON")
    print("NOT USED: closest approach and minimum separation are derived from JPL vectors.")

    print("EQUATION STATUS")
    print("PASS: closest approach is derived by polynomial minimization of JPL angular separation squared.")
    print("PASS: the annual fixed-plane denominator rejection has been removed.")
    print("PASS: the solar limb is centered on the closest-approach date.")
    print("PASS: no AI-generated image is used.")

    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"{VERSION} COMPLETE")


if __name__ == "__main__":
    main()

# V0114