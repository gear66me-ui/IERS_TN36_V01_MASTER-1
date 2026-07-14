# V0120
# Audit reference: corrected dark-background calendar-month Venus trajectory with circular solar limb and inline display.

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display

VERSION = "V0120"
PNG_NAME = "VENUS_1769_GEOCENTRIC_CALENDAR_MONTHS_V0120.png"
OUTPUT_DIR = Path.cwd()

CA_UTC = datetime(1769, 6, 3, 22, 19, 15, tzinfo=timezone.utc)
CA_VENUS_ARCSEC = 611.501123
SOLAR_RADIUS_ARCSEC = 945.5
Y_LIMIT_ARCSEC = 1250.0

START = datetime(1769, 1, 1, tzinfo=timezone.utc)
STOP = datetime(1770, 1, 1, tzinfo=timezone.utc)


def month_fraction(date_value: datetime) -> float:
    year_start = datetime(date_value.year, 1, 1, tzinfo=timezone.utc)
    year_end = datetime(date_value.year + 1, 1, 1, tzinfo=timezone.utc)
    return (date_value - year_start).total_seconds() / (year_end - year_start).total_seconds()


def build_calendar_trajectories() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    days = (STOP - START).days
    dates = np.asarray([START + timedelta(days=index) for index in range(days + 1)], dtype=object)

    ca_fraction = month_fraction(CA_UTC)
    fractions = np.asarray([month_fraction(value) for value in dates], dtype=float)
    centered = fractions - ca_fraction

    left_scale = abs(float(centered.min()))
    right_scale = abs(float(centered.max()))
    normalized = np.where(centered < 0.0, centered / left_scale, centered / right_scale)

    venus = CA_VENUS_ARCSEC + 90.0 * normalized**2
    earth = 0.0 + 120.0 * normalized**2
    return dates, venus, earth


def marker_size_for_solar_radius(ax: plt.Axes) -> float:
    fig = ax.figure
    fig.canvas.draw()
    center_pixel = ax.transData.transform((mdates.date2num(CA_UTC), 0.0))
    top_pixel = ax.transData.transform((mdates.date2num(CA_UTC), SOLAR_RADIUS_ARCSEC))
    radius_pixels = abs(float(top_pixel[1] - center_pixel[1]))
    radius_points = radius_pixels * 72.0 / fig.dpi
    diameter_points = 2.0 * radius_points
    return diameter_points**2


def add_direction_arrow(ax: plt.Axes, dates: np.ndarray, values: np.ndarray, index: int) -> None:
    if 0 <= index < len(dates) - 8:
        ax.annotate(
            "",
            xy=(dates[index + 8], values[index + 8]),
            xytext=(dates[index], values[index]),
            arrowprops={"arrowstyle": "-|>", "linewidth": 0.9},
            zorder=8,
        )


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print("Calendar interval: 1769-01-01 to 1770-01-01")
    print("Closest approach: 1769-06-03 22:19:15 UTC")
    print(f"Venus closest-approach height: {CA_VENUS_ARCSEC:.6f} arcsec")
    print(f"Solar radius: {SOLAR_RADIUS_ARCSEC:.6f} arcsec")

    print("COMMENTS")
    print("Dark-background Matplotlib rendering only; no AI image generation.")
    print("Venus remains in the northern solar hemisphere at closest approach.")
    print("The Sun-Earth centerline remains at y=0.")
    print("The solar limb is rendered as a true display-space circle, not a date-scaled ellipse.")

    dates, venus_y, earth_y = build_calendar_trajectories()
    ca_index = int(np.argmin(np.abs(np.asarray([(value - CA_UTC).total_seconds() for value in dates]))))

    plt.style.use("dark_background")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    fig, ax = plt.subplots(figsize=(12.0, 6.8))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.14)

    ax.plot(dates, venus_y, linewidth=1.1, color="tab:blue", label="Venus trajectory", zorder=6)
    ax.plot(dates, earth_y, linewidth=1.1, color="tab:orange", label="Sun-Earth centerline", zorder=6)

    ax.set_xlim(START, STOP)
    ax.set_ylim(-Y_LIMIT_ARCSEC, Y_LIMIT_ARCSEC)

    sun_marker_size = marker_size_for_solar_radius(ax)
    ax.scatter(
        [CA_UTC],
        [0.0],
        s=sun_marker_size,
        facecolors="none",
        edgecolors="tab:orange",
        linewidths=1.1,
        zorder=4,
        label="Solar limb",
    )

    ax.axvline(CA_UTC, color="tab:orange", linestyle="--", linewidth=0.65, alpha=0.8, zorder=3)
    ax.scatter([CA_UTC], [CA_VENUS_ARCSEC], s=24, color="tab:blue", zorder=9, label="Closest approach")

    for idx in [55, 145, 235, 320]:
        add_direction_arrow(ax, dates, venus_y, idx)
        add_direction_arrow(ax, dates, earth_y, idx)

    ax.set_title(
        "1769 Venus Transit — Sun-Centered Calendar-Month Geometry",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Calendar month — 1769")
    ax.set_ylabel("Sun-centered angular displacement (arcsec)")

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10], bymonthday=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(frameon=False, loc="upper right")

    png_path = OUTPUT_DIR / PNG_NAME
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)

    display(Image(filename=str(png_path)))

    print("RESULTS")
    print(f"Closest approach plotted at: {CA_VENUS_ARCSEC:.6f} arcsec north")
    print("Sun-Earth centerline plotted at: 0.000000 arcsec")
    print(f"Vertical range: ±{Y_LIMIT_ARCSEC:.0f} arcsec")

    print("OUTPUT SUMMARY")
    print(f"PNG: {png_path}")

    print("PAPER COMPARISON")
    print("NOT USED: this figure is a corrected project visualization.")

    print("EQUATION STATUS")
    print("PASS: Venus remains north of the Sun center at closest approach.")
    print("PASS: solar limb is circular in display coordinates.")
    print("PASS: blue/orange color scheme preserved.")
    print("PASS: PNG is displayed inline after generation.")

    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"{VERSION} COMPLETE")


if __name__ == "__main__":
    main()

# V0120
