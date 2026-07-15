# V0001
# Audit reference: standalone JPL Horizons 10-year heliocentric ecliptic X-coordinate sine-wave and radial-distance variation audit for Earth and Venus.
from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def need(module: str, package: str) -> None:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)


for module_name, package_name in [
    ("astroquery", "astroquery"),
    ("astropy", "astropy"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("IPython", "ipython"),
]:
    need(module_name, package_name)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display

VERSION = "V0001"
LOCAL_TZ = ZoneInfo("America/Bogota")
START_UTC = "2026-01-01 00:00"
STOP_UTC = "2036-01-01 00:00"
STEP = "1d"
LOCATION = "@10"
REFPLANE = "ecliptic"
ABERRATIONS = "geometric"
AU_KM = 149597870.700
OUT = Path("/content/EARTH_VENUS_10_YEAR_HELIOCENTRIC_ECLIPTIC_X_V0001_OUTPUT")
PNG = OUT / "EARTH_VENUS_10_YEAR_HELIOCENTRIC_ECLIPTIC_X_V0001.png"
CSV = OUT / "EARTH_VENUS_10_YEAR_HELIOCENTRIC_ECLIPTIC_X_V0001.csv"

PUBLISHED = {
    "Earth": {
        "perihelion_km": 147_100_000.0,
        "aphelion_km": 152_100_000.0,
        "range_km": 5_000_000.0,
    },
    "Venus": {
        "perihelion_km": 107_480_000.0,
        "aphelion_km": 108_940_000.0,
        "range_km": 1_460_000.0,
    },
}


def section(name: str) -> None:
    print(name)
    print("-" * len(name))


def query(body: str) -> tuple[np.ndarray, np.ndarray]:
    table = Horizons(
        id=body,
        id_type="majorbody",
        location=LOCATION,
        epochs={"start": START_UTC, "stop": STOP_UTC, "step": STEP},
    ).vectors(refplane=REFPLANE, aberrations=ABERRATIONS)
    jd = np.asarray(table["datetime_jd"], dtype=float)
    xyz = np.column_stack([
        np.asarray(table["x"], dtype=float),
        np.asarray(table["y"], dtype=float),
        np.asarray(table["z"], dtype=float),
    ]) * AU_KM
    if len(jd) < 3000 or not np.all(np.diff(jd) > 0.0):
        raise RuntimeError(f"REJECTED JPL vector grid for body {body}")
    return jd, xyz


def extrema_summary(name: str, jd: np.ndarray, xyz: np.ndarray) -> dict:
    radius = np.linalg.norm(xyz, axis=1)
    i_min = int(np.argmin(radius))
    i_max = int(np.argmax(radius))
    minimum = float(radius[i_min])
    maximum = float(radius[i_max])
    return {
        "planet": name,
        "minimum_km": minimum,
        "maximum_km": maximum,
        "range_km": maximum - minimum,
        "minimum_utc": Time(float(jd[i_min]), format="jd", scale="tdb").utc.strftime("%Y-%m-%d %H:%M:%S"),
        "maximum_utc": Time(float(jd[i_max]), format="jd", scale="tdb").utc.strftime("%Y-%m-%d %H:%M:%S"),
        "mean_km": float(np.mean(radius)),
        "x_min_km": float(np.min(xyz[:, 0])),
        "x_max_km": float(np.max(xyz[:, 0])),
        "z_min_km": float(np.min(xyz[:, 2])),
        "z_max_km": float(np.max(xyz[:, 2])),
    }


def make_plot(dates: np.ndarray, earth_xyz: np.ndarray, venus_xyz: np.ndarray, summaries: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(18.0, 8.5), dpi=110)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    earth_x_mkm = earth_xyz[:, 0] / 1.0e6
    venus_x_mkm = venus_xyz[:, 0] / 1.0e6

    ax.plot(dates, earth_x_mkm, linewidth=0.75, label="Earth heliocentric ecliptic X")
    ax.plot(dates, venus_x_mkm, linewidth=0.75, label="Venus heliocentric ecliptic X")
    ax.axhline(0.0, linewidth=0.60, linestyle="--", label="Sun center: X = 0")

    ax.set_title(
        "Earth and Venus — 10-Year Heliocentric Ecliptic X-Coordinate Sine Waves",
        color="#F0F0F0",
        fontsize=15,
        weight="bold",
        pad=10,
    )
    ax.set_xlabel("UTC date", color="#E0E0E0", fontsize=10.5)
    ax.set_ylabel("Heliocentric ecliptic X (million km)", color="#E0E0E0", fontsize=10.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(colors="#D7D7D7", labelsize=9, width=0.48)
    ax.grid(True, alpha=0.28, linewidth=0.42)
    for spine in ax.spines.values():
        spine.set_color("#8E8E8E")
        spine.set_linewidth(0.55)

    text = "\n".join([
        f"Earth Sun-distance range: {summaries['Earth']['range_km']:,.0f} km",
        f"Earth min/max: {summaries['Earth']['minimum_km']:,.0f} / {summaries['Earth']['maximum_km']:,.0f} km",
        f"Venus Sun-distance range: {summaries['Venus']['range_km']:,.0f} km",
        f"Venus min/max: {summaries['Venus']['minimum_km']:,.0f} / {summaries['Venus']['maximum_km']:,.0f} km",
        "Reference frame: Sun-centered ecliptic; Sun = (0, 0, 0)",
    ])
    ax.text(
        0.015,
        0.025,
        text,
        transform=ax.transAxes,
        color="#ECECEC",
        fontsize=9.2,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.38", "facecolor": "#050505", "edgecolor": "#858585", "alpha": 0.94},
    )

    legend = ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    for label in legend.get_texts():
        label.set_color("#DFDFDF")

    fig.tight_layout()
    fig.savefig(PNG, dpi=300, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    display(Image(filename=str(PNG)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    section("CODE INPUTS")
    print(f"Version                              {VERSION}")
    print(f"UTC interval                         {START_UTC} to {STOP_UTC}")
    print(f"JPL cadence                          {STEP}")
    print(f"Reference center                     Sun, {LOCATION}")
    print(f"Reference plane                      {REFPLANE}")
    print(f"Aberrations                          {ABERRATIONS}")
    print(f"Output                               {OUT}")

    section("COMMENTS")
    print("The sine-wave plot uses heliocentric ecliptic X(t), because an X-Z or Y-Z orbit-plane trace does not display orbital phase as clearly over ten years.")
    print("All Sun-distance extrema and ranges are derived from the full JPL X, Y, Z vectors before any plotting scale is applied.")
    print("The Sun is the JPL vector origin at X=0, Y=0, Z=0.")

    earth_jd, earth_xyz = query("399")
    venus_jd, venus_xyz = query("299")
    if len(earth_jd) != len(venus_jd) or not np.allclose(earth_jd, venus_jd, atol=1.0e-11, rtol=0.0):
        raise RuntimeError("REJECTED mismatched Earth/Venus JPL grids")

    dates = Time(earth_jd, format="jd", scale="tdb").utc.to_datetime()
    summaries = {
        "Earth": extrema_summary("Earth", earth_jd, earth_xyz),
        "Venus": extrema_summary("Venus", venus_jd, venus_xyz),
    }

    frame = pd.DataFrame({
        "utc": [value.strftime("%Y-%m-%d %H:%M:%S") for value in dates],
        "earth_x_km": earth_xyz[:, 0],
        "earth_y_km": earth_xyz[:, 1],
        "earth_z_km": earth_xyz[:, 2],
        "earth_radius_km": np.linalg.norm(earth_xyz, axis=1),
        "venus_x_km": venus_xyz[:, 0],
        "venus_y_km": venus_xyz[:, 1],
        "venus_z_km": venus_xyz[:, 2],
        "venus_radius_km": np.linalg.norm(venus_xyz, axis=1),
    })
    frame.to_csv(CSV, index=False, float_format="%.6f")
    make_plot(dates, earth_xyz, venus_xyz, summaries)

    section("RESULTS")
    for name in ("Earth", "Venus"):
        row = summaries[name]
        print(f"{name} minimum Sun distance            {row['minimum_km']:.6f} km")
        print(f"{name} maximum Sun distance            {row['maximum_km']:.6f} km")
        print(f"{name} total distance variation        {row['range_km']:.6f} km")
        print(f"{name} minimum-distance UTC            {row['minimum_utc']}")
        print(f"{name} maximum-distance UTC            {row['maximum_utc']}")
        print(f"{name} ecliptic X range                {row['x_min_km']:.6f} to {row['x_max_km']:.6f} km")
        print(f"{name} ecliptic Z range                {row['z_min_km']:.6f} to {row['z_max_km']:.6f} km")

    section("OUTPUT SUMMARY")
    print(f"PNG                                  {PNG}")
    print(f"CSV                                  {CSV}")
    print(f"JPL samples                          {len(earth_jd)}")

    section("PAPER COMPARISON")
    for name in ("Earth", "Venus"):
        derived = summaries[name]
        published = PUBLISHED[name]
        print(f"{name} published perihelion            {published['perihelion_km']:.0f} km")
        print(f"{name} published aphelion              {published['aphelion_km']:.0f} km")
        print(f"{name} published range                 {published['range_km']:.0f} km")
        print(f"{name} JPL-minus-published range       {derived['range_km'] - published['range_km']:.6f} km")

    section("EQUATION STATUS")
    print("VERIFIED radius = sqrt(X^2 + Y^2 + Z^2) from JPL heliocentric ecliptic vectors")
    print("VERIFIED variation = maximum radius - minimum radius over the full ten-year interval")
    print("VERIFIED plotted quantity = JPL heliocentric ecliptic X coordinate; Sun remains at zero")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0001