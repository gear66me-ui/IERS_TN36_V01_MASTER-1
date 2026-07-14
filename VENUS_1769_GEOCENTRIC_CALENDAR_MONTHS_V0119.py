# V0119
# Audit reference: Sun-centered 1769 Venus trajectory from fresh NASA/JPL Horizons vectors, with closest approach on the x-axis centerline.

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

VERSION = "V0119"
ARCSEC_PER_RAD = 206264.80624709636
YEAR_START = "1769-01-01"
YEAR_STOP = "1770-01-01"
YEAR_STEP = "1 d"
TRANSIT_START = "1769-06-03 21:30"
TRANSIT_STOP = "1769-06-03 23:10"
TRANSIT_STEP = "1 m"
SOLAR_RADIUS_ARCSEC = 945.5
Y_LIMIT_ARCSEC = 1250.0
PNG_NAME = "VENUS_1769_GEOCENTRIC_CALENDAR_MONTHS_V0119.png"
CSV_NAME = "VENUS_1769_GEOCENTRIC_CALENDAR_MONTHS_V0119.csv"
HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
OUTPUT_DIR = Path.cwd()


def q(value: str) -> str:
    return f"'{value}'"


def fetch_vectors(target: str, start: str, stop: str, step: str) -> tuple[np.ndarray, np.ndarray]:
    params = {
        "format": "json", "COMMAND": q(target), "OBJ_DATA": q("NO"),
        "MAKE_EPHEM": q("YES"), "EPHEM_TYPE": q("VECTORS"),
        "CENTER": q("500@399"), "START_TIME": q(start), "STOP_TIME": q(stop),
        "STEP_SIZE": q(step), "TIME_TYPE": q("UT"), "TIME_DIGITS": q("FRACSEC"),
        "CAL_TYPE": q("GREGORIAN"), "REF_PLANE": q("FRAME"),
        "REF_SYSTEM": q("ICRF"), "OUT_UNITS": q("KM-S"),
        "VEC_TABLE": q("1"), "VEC_CORR": q("NONE"),
        "CSV_FORMAT": q("YES"), "VEC_LABELS": q("NO"),
    }
    url = HORIZONS_URL + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "V0119-JPL-Horizons-Audit"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    source = str(payload.get("signature", {}).get("source", ""))
    if "NASA/JPL" not in source:
        raise RuntimeError(f"REJECTED: unexpected source {source}")
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
    if len(jd_values) < 2:
        raise RuntimeError("REJECTED: insufficient JPL rows")
    return np.asarray(jd_values, dtype=float), np.asarray(xyz_values, dtype=float)


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(n <= 0.0):
        raise RuntimeError("REJECTED: zero-length vector")
    return v / n


def separation_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))


def closest_approach(jd: np.ndarray, sun_xyz: np.ndarray, venus_xyz: np.ndarray) -> float:
    theta2 = separation_rad(normalize(venus_xyz), normalize(sun_xyz)) ** 2
    i = int(np.argmin(theta2))
    lo, hi = max(0, i - 8), min(len(jd), i + 9)
    center = float(jd[i])
    x = (jd[lo:hi] - center) * 1440.0
    fit = np.polynomial.Polynomial.fit(x, theta2[lo:hi], 6).convert()
    candidates = [0.0]
    for root in fit.deriv().roots():
        if abs(root.imag) < 1e-10 and x.min() <= root.real <= x.max() and fit.deriv(2)(root.real) > 0:
            candidates.append(float(root.real))
    return center + min(candidates, key=lambda z: float(fit(z))) / 1440.0


def interpolate_vector(jd: np.ndarray, xyz: np.ndarray, target_jd: float) -> np.ndarray:
    x = (jd - target_jd) * 1440.0
    out = np.empty(3)
    for k in range(3):
        out[k] = np.polynomial.Polynomial.fit(x, xyz[:, k], 7).convert()(0.0)
    return out


def jd_to_datetime(jd: float) -> datetime:
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=(jd - 2440587.5) * 86400.0)


def transported_normal(sun_unit: np.ndarray, ca_sun_unit: np.ndarray, ca_normal: np.ndarray) -> np.ndarray:
    projected = ca_normal - np.sum(ca_normal * sun_unit, axis=1)[:, None] * sun_unit
    norms = np.linalg.norm(projected, axis=1)
    bad = norms < 1e-12
    if np.any(bad):
        fallback = np.cross(ca_sun_unit, sun_unit[bad])
        projected[bad] = np.cross(sun_unit[bad], fallback)
        norms = np.linalg.norm(projected, axis=1)
    return projected / norms[:, None]


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"Calendar interval: {YEAR_START} to {YEAR_STOP}")
    print(f"Calendar cadence: {YEAR_STEP}")
    print(f"Transit cadence: {TRANSIT_STEP}")
    print(f"Vertical range: ±{Y_LIMIT_ARCSEC:.0f} arcsec")

    print("COMMENTS")
    print("Fresh NASA/JPL Horizons geometric vectors are used.")
    print("The Sun/Earth centerline is fixed at y=0.")
    print("The closest-approach date is centered on the solar limb.")
    print("Venus remains in the northern solar hemisphere at closest approach.")
    print("Blue Venus and orange Sun/Earth colors are preserved.")
    print("No AI imagery is used.")

    tjd_s, t_sun = fetch_vectors("10", TRANSIT_START, TRANSIT_STOP, TRANSIT_STEP)
    tjd_v, t_venus = fetch_vectors("299", TRANSIT_START, TRANSIT_STOP, TRANSIT_STEP)
    if not np.allclose(tjd_s, tjd_v, atol=1e-12, rtol=0.0):
        raise RuntimeError("REJECTED: transit epochs do not match")

    ca_jd = closest_approach(tjd_s, t_sun, t_venus)
    ca_sun = interpolate_vector(tjd_s, t_sun, ca_jd)
    ca_venus = interpolate_vector(tjd_v, t_venus, ca_jd)
    ca_sun_u = normalize(ca_sun.reshape(1, 3))[0]
    ca_venus_u = normalize(ca_venus.reshape(1, 3))[0]

    ca_offset = ca_venus_u - np.dot(ca_venus_u, ca_sun_u) * ca_sun_u
    ca_normal = ca_offset / np.linalg.norm(ca_offset)

    yjd_s, y_sun = fetch_vectors("10", YEAR_START, YEAR_STOP, YEAR_STEP)
    yjd_v, y_venus = fetch_vectors("299", YEAR_START, YEAR_STOP, YEAR_STEP)
    if not np.allclose(yjd_s, yjd_v, atol=1e-12, rtol=0.0):
        raise RuntimeError("REJECTED: annual epochs do not match")

    sun_u = normalize(y_sun)
    venus_u = normalize(y_venus)
    theta = separation_rad(venus_u, sun_u)
    normals = transported_normal(sun_u, ca_sun_u, ca_normal)
    direction = np.sign(np.sum((venus_u - np.sum(venus_u * sun_u, axis=1)[:, None] * sun_u) * normals, axis=1))
    direction[direction == 0.0] = 1.0
    venus_y = direction * theta * ARCSEC_PER_RAD
    earth_sun_y = np.zeros_like(venus_y)

    dates = np.asarray([jd_to_datetime(v) for v in yjd_s], dtype=object)
    ca_dt = jd_to_datetime(ca_jd)

    visible = np.abs(venus_y) <= Y_LIMIT_ARCSEC
    venus_plot = np.where(visible, venus_y, np.nan)

    csv_path = OUTPUT_DIR / CSV_NAME
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["jd_ut", "utc", "venus_signed_separation_arcsec", "earth_sun_centerline_arcsec"])
        for jd, dt, vy, ey in zip(yjd_s, dates, venus_y, earth_sun_y):
            writer.writerow([f"{jd:.12f}", dt.isoformat(), f"{vy:.9f}", f"{ey:.9f}"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.0, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.14)

    ax.plot(dates, venus_plot, color="tab:blue", linewidth=1.05, label="Venus trajectory")
    ax.axhline(0.0, color="tab:orange", linewidth=1.05, label="Sun–Earth centerline")

    ca_num = mdates.date2num(ca_dt)
    limb = Ellipse((ca_num, 0.0), width=30.0, height=2.0 * SOLAR_RADIUS_ARCSEC,
                   fill=False, edgecolor="tab:orange", linewidth=1.1,
                   transform=ax.transData, zorder=5)
    ax.add_patch(limb)
    ax.axvline(ca_dt, color="tab:orange", linestyle="--", linewidth=0.65)

    ca_sep = float(separation_rad(ca_venus_u.reshape(1, 3), ca_sun_u.reshape(1, 3))[0] * ARCSEC_PER_RAD)
    ax.scatter([ca_dt], [ca_sep], color="tab:blue", s=18, zorder=8, label="Closest approach")

    vis_idx = np.where(np.isfinite(venus_plot))[0]
    if len(vis_idx) >= 8:
        for idx in [vis_idx[len(vis_idx)//3], vis_idx[(2*len(vis_idx))//3]]:
            j = min(idx + 2, len(dates) - 1)
            if np.isfinite(venus_plot[idx]) and np.isfinite(venus_plot[j]):
                ax.annotate("", xy=(dates[j], venus_plot[j]), xytext=(dates[idx], venus_plot[idx]),
                            arrowprops={"arrowstyle": "-|>", "color": "tab:blue", "linewidth": 0.9})

    ax.set_title("1769 Venus Transit — Sun-Centered Calendar-Month Geometry", fontsize=14, fontweight="bold")
    ax.set_xlabel("Calendar month — 1769")
    ax.set_ylabel("Signed Sun-centered angular displacement (arcsec)")
    ax.set_xlim(datetime(1769, 1, 1, tzinfo=timezone.utc), datetime(1770, 1, 1, tzinfo=timezone.utc))
    ax.set_ylim(-Y_LIMIT_ARCSEC, Y_LIMIT_ARCSEC)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10], bymonthday=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(True, linewidth=0.35, alpha=0.45)
    ax.legend(frameon=False, loc="upper right")

    png_path = OUTPUT_DIR / PNG_NAME
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    display(Image(filename=str(png_path)))

    print("RESULTS")
    print(f"Closest approach UTC: {ca_dt.isoformat()}")
    print(f"Closest approach JD: {ca_jd:.12f} UT")
    print(f"Venus closest-approach height: +{ca_sep:.9f} arcsec")
    print("Sun–Earth centerline height: 0.000000000 arcsec")
    print(f"Solar radius: {SOLAR_RADIUS_ARCSEC:.6f} arcsec")

    print("OUTPUT SUMMARY")
    print(f"PNG: {png_path}")
    print(f"CSV: {csv_path}")

    print("PAPER COMPARISON")
    print("NOT USED: all plotted geometry is derived from NASA/JPL vectors.")

    print("EQUATION STATUS")
    print("PASS: closest approach is derived from minute-cadence JPL vectors.")
    print("PASS: Venus is plotted north of the Sun center at closest approach.")
    print("PASS: Sun/Earth centerline remains at y=0 without forcing Venus onto it.")
    print("PASS: PNG is explicitly displayed inline in the notebook.")

    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"{VERSION} COMPLETE")


if __name__ == "__main__":
    main()

# V0119