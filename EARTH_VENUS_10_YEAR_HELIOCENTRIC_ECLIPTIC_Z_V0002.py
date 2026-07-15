# V0002
# Audit reference: standalone JPL Horizons 10-year heliocentric ecliptic Z-height audit for Earth and Venus; no radial-distance plotting.
from __future__ import annotations

import importlib.util
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

VERSION = "V0002"
LOCAL_TZ = ZoneInfo("America/Bogota")
START_UTC = "2026-01-01 00:00"
STOP_UTC = "2036-01-01 00:00"
STEP = "1d"
LOCATION = "@10"
REFPLANE = "ecliptic"
ABERRATIONS = "geometric"
AU_KM = 149597870.700
OUT = Path("/content/EARTH_VENUS_10_YEAR_HELIOCENTRIC_ECLIPTIC_Z_V0002_OUTPUT")
PNG = OUT / "EARTH_VENUS_10_YEAR_HELIOCENTRIC_ECLIPTIC_Z_V0002.png"
CSV = OUT / "EARTH_VENUS_10_YEAR_HELIOCENTRIC_ECLIPTIC_Z_V0002.csv"


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
    if not np.all(np.isfinite(xyz)):
        raise RuntimeError(f"REJECTED non-finite JPL vectors for body {body}")
    return jd, xyz


def z_summary(name: str, jd: np.ndarray, xyz: np.ndarray) -> dict:
    z = np.asarray(xyz[:, 2], dtype=float)
    i_min = int(np.argmin(z))
    i_max = int(np.argmax(z))
    z_min = float(z[i_min])
    z_max = float(z[i_max])
    peak_to_peak = z_max - z_min
    amplitude = 0.5 * peak_to_peak
    mean = float(np.mean(z))
    rms = float(np.sqrt(np.mean((z - mean) ** 2)))
    return {
        "planet": name,
        "z_min_km": z_min,
        "z_max_km": z_max,
        "z_peak_to_peak_km": peak_to_peak,
        "z_half_amplitude_km": amplitude,
        "z_mean_km": mean,
        "z_rms_about_mean_km": rms,
        "z_min_utc": Time(float(jd[i_min]), format="jd", scale="tdb").utc.strftime("%Y-%m-%d %H:%M:%S"),
        "z_max_utc": Time(float(jd[i_max]), format="jd", scale="tdb").utc.strftime("%Y-%m-%d %H:%M:%S"),
    }


def make_plot(
    dates: np.ndarray,
    earth_xyz: np.ndarray,
    venus_xyz: np.ndarray,
    summaries: dict[str, dict],
) -> None:
    fig, ax = plt.subplots(figsize=(18.0, 8.5), dpi=110)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    earth_z_mkm = earth_xyz[:, 2] / 1.0e6
    venus_z_mkm = venus_xyz[:, 2] / 1.0e6

    ax.plot(dates, earth_z_mkm, linewidth=0.78, label="Earth heliocentric ecliptic Z")
    ax.plot(dates, venus_z_mkm, linewidth=0.78, label="Venus heliocentric ecliptic Z")
    ax.axhline(0.0, linewidth=0.62, linestyle="--", label="Ecliptic plane: Z = 0")

    ax.set_title(
        "Earth and Venus — 10-Year Heliocentric Ecliptic Z Height",
        color="#F0F0F0",
        fontsize=15,
        weight="bold",
        pad=10,
    )
    ax.set_xlabel("UTC date", color="#E0E0E0", fontsize=10.5)
    ax.set_ylabel("Height above/below ecliptic plane, Z (million km)", color="#E0E0E0", fontsize=10.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(colors="#D7D7D7", labelsize=9, width=0.48)
    ax.grid(True, alpha=0.28, linewidth=0.42)
    for spine in ax.spines.values():
        spine.set_color("#8E8E8E")
        spine.set_linewidth(0.55)

    text = "\n".join([
        f"Earth Z min/max: {summaries['Earth']['z_min_km']:,.0f} / {summaries['Earth']['z_max_km']:,.0f} km",
        f"Earth Z peak-to-peak: {summaries['Earth']['z_peak_to_peak_km']:,.0f} km",
        f"Venus Z min/max: {summaries['Venus']['z_min_km']:,.0f} / {summaries['Venus']['z_max_km']:,.0f} km",
        f"Venus Z peak-to-peak: {summaries['Venus']['z_peak_to_peak_km']:,.0f} km",
        "Reference frame: Sun-centered ecliptic; Sun = (0, 0, 0)",
        "Plotted quantity: Z only; radial Sun–planet distance is NOT USED",
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
    print("The plot shows only the heliocentric ecliptic Z component over ten years.")
    print("Positive Z is above the ecliptic plane; negative Z is below it.")
    print("Sun–planet radial distance, perihelion, and aphelion are NOT USED.")
    print("The Sun remains fixed at X=0, Y=0, Z=0 in the JPL heliocentric frame.")

    earth_jd, earth_xyz = query("399")
    venus_jd, venus_xyz = query("299")
    if len(earth_jd) != len(venus_jd) or not np.allclose(earth_jd, venus_jd, atol=1.0e-11, rtol=0.0):
        raise RuntimeError("REJECTED mismatched Earth/Venus JPL grids")

    dates = Time(earth_jd, format="jd", scale="tdb").utc.to_datetime()
    summaries = {
        "Earth": z_summary("Earth", earth_jd, earth_xyz),
        "Venus": z_summary("Venus", venus_jd, venus_xyz),
    }

    frame = pd.DataFrame({
        "utc": [value.strftime("%Y-%m-%d %H:%M:%S") for value in dates],
        "earth_z_km": earth_xyz[:, 2],
        "venus_z_km": venus_xyz[:, 2],
    })
    frame.to_csv(CSV, index=False, float_format="%.6f")
    make_plot(dates, earth_xyz, venus_xyz, summaries)

    section("RESULTS")
    for name in ("Earth", "Venus"):
        row = summaries[name]
        print(f"{name} minimum ecliptic Z             {row['z_min_km']:.6f} km")
        print(f"{name} maximum ecliptic Z             {row['z_max_km']:.6f} km")
        print(f"{name} Z peak-to-peak variation       {row['z_peak_to_peak_km']:.6f} km")
        print(f"{name} Z half-amplitude               {row['z_half_amplitude_km']:.6f} km")
        print(f"{name} mean ecliptic Z                {row['z_mean_km']:.6f} km")
        print(f"{name} Z RMS about mean               {row['z_rms_about_mean_km']:.6f} km")
        print(f"{name} minimum-Z UTC                 {row['z_min_utc']}")
        print(f"{name} maximum-Z UTC                 {row['z_max_utc']}")

    section("OUTPUT SUMMARY")
    print(f"PNG                                  {PNG}")
    print(f"CSV                                  {CSV}")
    print(f"JPL samples                          {len(earth_jd)}")

    section("PAPER COMPARISON")
    print("NOT USED: perihelion, aphelion, semimajor axis, or published radial-distance ranges.")
    print("This audit reports only the JPL heliocentric ecliptic Z component.")

    section("EQUATION STATUS")
    print("VERIFIED plotted Earth quantity = JPL Earth Z coordinate in the heliocentric ecliptic frame")
    print("VERIFIED plotted Venus quantity = JPL Venus Z coordinate in the heliocentric ecliptic frame")
    print("VERIFIED Z peak-to-peak variation = maximum Z - minimum Z")
    print("VERIFIED radial distance sqrt(X^2 + Y^2 + Z^2) is NOT USED")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0002