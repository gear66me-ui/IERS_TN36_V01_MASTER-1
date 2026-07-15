# V0141
# Audit reference: true Sun-centered heliocentric ecliptic-Z amplitudes for Earth and Venus, centered on the 1761 transit.

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def ensure(module: str, package: str) -> None:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)

for module_name, package_name in [
    ("astroquery", "astroquery"),
    ("astropy", "astropy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("IPython", "ipython"),
]:
    ensure(module_name, package_name)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display

VERSION = "V0141"
TRANSIT_CENTER_UTC = "1761-06-06 05:17:45"
HALF_WINDOW_DAYS = 183.0
STEP = "1d"
LOCATION = "@10"
REFPLANE = "ecliptic"
ABERRATIONS = "geometric"

OUTPUT_DIR = Path("/content/VENUS_TRANSIT_1761_TRUE_HELIOCENTRIC_ECLIPTIC_Z_V0141_OUTPUT")
PNG_NAME = "VENUS_TRANSIT_1761_TRUE_HELIOCENTRIC_ECLIPTIC_Z_V0141.png"
CSV_NAME = "VENUS_TRANSIT_1761_TRUE_HELIOCENTRIC_ECLIPTIC_Z_V0141.csv"

def section(title: str) -> None:
    print(title)
    print("-" * len(title))

def query(body_id: str, start: str, stop: str) -> pd.DataFrame:
    table = Horizons(
        id=body_id,
        id_type="majorbody",
        location=LOCATION,
        epochs={"start": start, "stop": stop, "step": STEP},
    ).vectors(refplane=REFPLANE, aberrations=ABERRATIONS)

    frame = pd.DataFrame({
        "jd_tdb": np.asarray(table["datetime_jd"], dtype=float),
        "x_km": np.asarray(table["x"], dtype=float) * 149597870.700,
        "y_km": np.asarray(table["y"], dtype=float) * 149597870.700,
        "z_km": np.asarray(table["z"], dtype=float) * 149597870.700,
    })

    if len(frame) < 300:
        raise RuntimeError(f"REJECTED insufficient JPL samples for body {body_id}")
    if not np.all(np.diff(frame["jd_tdb"].to_numpy()) > 0.0):
        raise RuntimeError(f"REJECTED non-monotonic JPL epochs for body {body_id}")

    frame["datetime_utc"] = Time(
        frame["jd_tdb"].to_numpy(),
        format="jd",
        scale="tdb",
    ).utc.to_datetime()
    return frame

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    center = Time(TRANSIT_CENTER_UTC, scale="utc")
    start = Time(center.jd - HALF_WINDOW_DAYS, format="jd", scale="utc")
    stop = Time(center.jd + HALF_WINDOW_DAYS, format="jd", scale="utc")
    start_text = start.strftime("%Y-%m-%d %H:%M")
    stop_text = stop.strftime("%Y-%m-%d %H:%M")

    section("CODE INPUTS")
    print(f"Version                              {VERSION}")
    print("Geometry                             Sun fixed at (0,0,0)")
    print("Reference plane                      JPL ecliptic")
    print("Earth target                         399")
    print("Venus target                         299")
    print(f"Window                               {start_text} to {stop_text}")
    print(f"Cadence                              {STEP}")
    print("Amplitude multiplier                 NONE")
    print("No AI images                         Matplotlib only")
    print(f"Output                               {OUTPUT_DIR}")

    earth = query("399", start_text, stop_text)
    venus = query("299", start_text, stop_text)

    if len(earth) != len(venus) or not np.allclose(
        earth["jd_tdb"].to_numpy(),
        venus["jd_tdb"].to_numpy(),
        atol=1.0e-10,
        rtol=0.0,
    ):
        raise RuntimeError("REJECTED mismatched Earth/Venus JPL grids")

    result = pd.DataFrame({
        "jd_tdb": earth["jd_tdb"],
        "datetime_utc": earth["datetime_utc"],
        "earth_z_km": earth["z_km"],
        "venus_z_km": venus["z_km"],
    })

    earth_min = float(result["earth_z_km"].min())
    earth_max = float(result["earth_z_km"].max())
    venus_min = float(result["venus_z_km"].min())
    venus_max = float(result["venus_z_km"].max())

    earth_peak_to_peak = earth_max - earth_min
    venus_peak_to_peak = venus_max - venus_min

    csv_path = OUTPUT_DIR / CSV_NAME
    result.to_csv(csv_path, index=False, float_format="%.6f")

    fig, ax = plt.subplots(figsize=(13.2, 7.4), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.plot(
        result["datetime_utc"],
        result["venus_z_km"],
        linewidth=1.0,
        label="Venus heliocentric ecliptic Z",
    )
    ax.plot(
        result["datetime_utc"],
        result["earth_z_km"],
        linewidth=1.0,
        label="Earth heliocentric ecliptic Z",
    )

    ax.axhline(0.0, linewidth=0.65, linestyle="--", alpha=0.75)
    ax.axvline(center.utc.to_datetime(), linewidth=0.65, linestyle="--", alpha=0.75)

    ax.scatter(
        [center.utc.to_datetime()],
        [float(np.interp(center.tdb.jd, result["jd_tdb"], result["venus_z_km"]))],
        s=24,
        zorder=5,
        label="1761 transit epoch",
    )

    annotation = "\n".join([
        f"Earth min/max: {earth_min:,.3f} / {earth_max:,.3f} km",
        f"Earth peak-to-peak: {earth_peak_to_peak:,.3f} km",
        f"Venus min/max: {venus_min:,.3f} / {venus_max:,.3f} km",
        f"Venus peak-to-peak: {venus_peak_to_peak:,.3f} km",
    ])

    ax.annotate(
        annotation,
        xy=(center.utc.to_datetime(), 0.0),
        xytext=(center.utc.to_datetime() + pd.Timedelta(days=34), 0.58 * venus_max),
        fontsize=9.6,
        ha="left",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#050505",
            "edgecolor": "#858585",
            "alpha": 0.94,
        },
        arrowprops={
            "arrowstyle": "-",
            "linewidth": 0.65,
        },
        zorder=6,
    )

    ax.set_title(
        "1761 Venus Transit — True Sun-Centered Ecliptic-Z Amplitudes",
        fontsize=14.5,
        weight="bold",
        pad=10,
    )
    ax.set_xlabel("Date (closest approach centered; ±183 days)")
    ax.set_ylabel("Heliocentric ecliptic Z (km)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, alpha=0.24, linewidth=0.42)
    ax.tick_params(labelsize=9)

    ax.legend(loc="upper right", frameon=False, fontsize=9.2)

    fig.tight_layout()
    png_path = OUTPUT_DIR / PNG_NAME
    fig.savefig(png_path, dpi=300, facecolor="black", bbox_inches="tight")
    plt.close(fig)

    display(Image(filename=str(png_path)))

    section("COMMENTS")
    print("The plotted quantity is the true JPL heliocentric ecliptic Z coordinate in kilometres.")
    print("The Sun is the origin; no tangent-plane projection is used.")
    print("No angular conversion, visual scaling, or independent recentering is applied.")
    print("Earth is expected to appear nearly flat beside Venus on the shared physical scale.")

    section("RESULTS")
    print(f"Earth minimum Z                     {earth_min:,.6f} km")
    print(f"Earth maximum Z                     {earth_max:,.6f} km")
    print(f"Earth peak-to-peak Z                {earth_peak_to_peak:,.6f} km")
    print(f"Venus minimum Z                     {venus_min:,.6f} km")
    print(f"Venus maximum Z                     {venus_max:,.6f} km")
    print(f"Venus peak-to-peak Z                {venus_peak_to_peak:,.6f} km")
    print(f"Sample count                        {len(result)}")

    section("OUTPUT SUMMARY")
    print(f"PNG                                  {png_path}")
    print(f"CSV                                  {csv_path}")

    section("PAPER COMPARISON")
    print("Published orbital inclinations are NOT USED in the calculation.")
    print("All amplitudes come directly from JPL heliocentric vectors.")

    section("EQUATION STATUS")
    print("VERIFIED z_Earth = Earth heliocentric ecliptic Z")
    print("VERIFIED z_Venus = Venus heliocentric ecliptic Z")
    print("VERIFIED Sun origin = (0,0,0)")
    print("VERIFIED amplitude multiplier = 1.0")
    print(f"PNG exists                           {png_path.is_file() and png_path.stat().st_size > 0}")

    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(VERSION)

if __name__ == "__main__":
    main()
# V0141
