# V0003
# Audit reference: standalone JPL Horizons five-year heliocentric ecliptic Z-height audit for Earth, Venus, and Mercury; no radial-distance plotting.
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

VERSION = "V0003"
LOCAL_TZ = ZoneInfo("America/Bogota")
START_UTC = "2026-01-01 00:00"
STOP_UTC = "2031-01-01 00:00"
STEP = "1d"
LOCATION = "@10"
REFPLANE = "ecliptic"
ABERRATIONS = "geometric"
AU_KM = 149597870.700
BODIES = {"Mercury": "199", "Venus": "299", "Earth": "399"}
OUT = Path("/content/EARTH_VENUS_MERCURY_5_YEAR_HELIOCENTRIC_ECLIPTIC_Z_V0003_OUTPUT")
PNG = OUT / "EARTH_VENUS_MERCURY_5_YEAR_HELIOCENTRIC_ECLIPTIC_Z_V0003.png"
CSV = OUT / "EARTH_VENUS_MERCURY_5_YEAR_HELIOCENTRIC_ECLIPTIC_Z_V0003.csv"


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
    if len(jd) < 1800 or not np.all(np.diff(jd) > 0.0):
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
    return {
        "planet": name,
        "z_min_km": z_min,
        "z_max_km": z_max,
        "z_peak_to_peak_km": z_max - z_min,
        "z_max_abs_km": float(np.max(np.abs(z))),
        "z_mean_km": float(np.mean(z)),
        "z_rms_about_mean_km": float(np.sqrt(np.mean((z - np.mean(z)) ** 2))),
        "z_min_utc": Time(float(jd[i_min]), format="jd", scale="tdb").utc.strftime("%Y-%m-%d %H:%M:%S"),
        "z_max_utc": Time(float(jd[i_max]), format="jd", scale="tdb").utc.strftime("%Y-%m-%d %H:%M:%S"),
    }


def make_plot(dates: np.ndarray, vectors: dict[str, np.ndarray], summaries: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(18.0, 8.5), dpi=110)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    for name in ("Earth", "Venus", "Mercury"):
        ax.plot(dates, vectors[name][:, 2] / 1.0e6, linewidth=0.78, label=f"{name} heliocentric ecliptic Z")
    ax.axhline(0.0, linewidth=0.62, linestyle="--", label="Ecliptic plane: Z = 0")

    ax.set_title(
        "Earth, Venus, and Mercury — 5-Year Heliocentric Ecliptic Z Height",
        color="#F0F0F0",
        fontsize=15,
        weight="bold",
        pad=10,
    )
    ax.set_xlabel("UTC date", color="#E0E0E0", fontsize=10.5)
    ax.set_ylabel("Height above/below ecliptic plane, Z (million km)", color="#E0E0E0", fontsize=10.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(colors="#D7D7D7", labelsize=9, width=0.48, rotation=45)
    ax.grid(True, alpha=0.28, linewidth=0.42)
    for spine in ax.spines.values():
        spine.set_color("#8E8E8E")
        spine.set_linewidth(0.55)

    text_lines = [
        f"Earth max |Z|: {summaries['Earth']['z_max_abs_km']:,.0f} km",
        f"Venus max |Z|: {summaries['Venus']['z_max_abs_km']:,.0f} km",
        f"Mercury max |Z|: {summaries['Mercury']['z_max_abs_km']:,.0f} km",
        "Reference frame: Sun-centered ecliptic; Sun = (0, 0, 0)",
        "Plotted quantity: Z only; radial Sun–planet distance is NOT USED",
    ]
    ax.text(
        0.015,
        0.025,
        "\n".join(text_lines),
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
    print("The plot shows only the heliocentric ecliptic Z component over five years.")
    print("Positive Z is above the ecliptic plane; negative Z is below it.")
    print("Earth, Venus, and Mercury are all derived directly from JPL Horizons vectors.")
    print("Sun–planet radial distance, perihelion, and aphelion are NOT USED.")

    jds: dict[str, np.ndarray] = {}
    vectors: dict[str, np.ndarray] = {}
    for name, body in BODIES.items():
        jd, xyz = query(body)
        jds[name] = jd
        vectors[name] = xyz

    reference_jd = jds["Earth"]
    for name in ("Venus", "Mercury"):
        if len(jds[name]) != len(reference_jd) or not np.allclose(jds[name], reference_jd, atol=1.0e-11, rtol=0.0):
            raise RuntimeError(f"REJECTED mismatched {name}/Earth JPL grids")

    dates = Time(reference_jd, format="jd", scale="tdb").utc.to_datetime()
    summaries = {name: z_summary(name, jds[name], vectors[name]) for name in BODIES}

    frame = pd.DataFrame({
        "utc": [value.strftime("%Y-%m-%d %H:%M:%S") for value in dates],
        "earth_z_km": vectors["Earth"][:, 2],
        "venus_z_km": vectors["Venus"][:, 2],
        "mercury_z_km": vectors["Mercury"][:, 2],
    })
    frame.to_csv(CSV, index=False, float_format="%.6f")
    make_plot(dates, vectors, summaries)

    section("RESULTS")
    for name in ("Earth", "Venus", "Mercury"):
        row = summaries[name]
        print(f"{name} minimum ecliptic Z             {row['z_min_km']:.6f} km")
        print(f"{name} maximum ecliptic Z             {row['z_max_km']:.6f} km")
        print(f"{name} Z peak-to-peak variation       {row['z_peak_to_peak_km']:.6f} km")
        print(f"{name} maximum absolute Z             {row['z_max_abs_km']:.6f} km")
        print(f"{name} mean ecliptic Z                {row['z_mean_km']:.6f} km")
        print(f"{name} Z RMS about mean               {row['z_rms_about_mean_km']:.6f} km")
        print(f"{name} minimum-Z UTC                 {row['z_min_utc']}")
        print(f"{name} maximum-Z UTC                 {row['z_max_utc']}")

    section("OUTPUT SUMMARY")
    print(f"PNG                                  {PNG}")
    print(f"CSV                                  {CSV}")
    print(f"JPL samples                          {len(reference_jd)}")

    section("PAPER COMPARISON")
    print("NOT USED: published inclinations, perihelion, aphelion, or radial-distance ranges.")
    print("This audit reports only the JPL heliocentric ecliptic Z component.")

    section("EQUATION STATUS")
    print("VERIFIED plotted quantity = JPL heliocentric ecliptic Z coordinate for Earth, Venus, and Mercury")
    print("VERIFIED Z peak-to-peak variation = maximum Z - minimum Z")
    print("VERIFIED maximum absolute Z = max(abs(Z))")
    print("VERIFIED radial distance sqrt(X^2 + Y^2 + Z^2) is NOT USED")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0003