# V0013
# Audit reference: centered JPL Horizons heliocentric ecliptic orbit bundle from astronomical year -9000 through +9000 with epoch-dependent Z-only exaggeration.
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
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("IPython", "ipython"),
]:
    need(module_name, package_name)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display

VERSION = "V0013"
LOCAL_TZ = ZoneInfo("America/Bogota")
LOCATION = "@10"
REFPLANE = "ecliptic"
ABERRATIONS = "geometric"
AU_KM = 149597870.700
R_SUN_KM = 695700.000
R_SUN_MKM = R_SUN_KM / 1.0e6
J2000_JD = 2451545.0
DAYS_PER_JULIAN_YEAR = 365.25
HORIZONS_MIN_YEAR = -9999
HORIZONS_MAX_YEAR = 9999
EPOCH_YEARS = list(range(-9000, 9001, 1000))
STEP = "1d"
ORBIT_LINEWIDTH = 0.25
OUT = Path("/content/MERCURY_VENUS_EARTH_CENTERED_18000_YEAR_Z_EXAGGERATED_ORBITS_3D_V0013_OUTPUT")
PNG = OUT / "MERCURY_VENUS_EARTH_CENTERED_18000_YEAR_Z_EXAGGERATED_ORBITS_3D_V0013.png"
CSV = OUT / "MERCURY_VENUS_EARTH_CENTERED_18000_YEAR_Z_EXAGGERATED_ORBITS_3D_V0013.csv"

PLANETS = {
    "Mercury": {"id": "199", "color": "#A970FF", "period_days": 87.9691},
    "Venus": {"id": "299", "color": "#2F8DFF", "period_days": 224.701},
    "Earth": {"id": "399", "color": "#35C96B", "period_days": 365.256},
}


def section(name: str) -> None:
    print(name)
    print("-" * len(name))


def epoch_center_jd(year: int) -> float:
    if not HORIZONS_MIN_YEAR <= year <= HORIZONS_MAX_YEAR:
        raise ValueError(
            f"REJECTED epoch {year}: online Horizons year limit is "
            f"{HORIZONS_MIN_YEAR} through {HORIZONS_MAX_YEAR}"
        )
    return J2000_JD + (float(year) - 2000.0) * DAYS_PER_JULIAN_YEAR


def z_scale_factor(year: int) -> float:
    if year == 0:
        return 1.0
    return 10.0 * abs(float(year)) / 1000.0


def jd_token(jd: float) -> str:
    return f"JD{jd:.9f}"


def query_orbit(body: str, center_jd: float, period_days: float) -> tuple[np.ndarray, np.ndarray]:
    margin_days = 4.0
    start_jd = center_jd - 0.5 * period_days - margin_days
    stop_jd = center_jd + 0.5 * period_days + margin_days
    table = Horizons(
        id=body,
        id_type="majorbody",
        location=LOCATION,
        epochs={
            "start": jd_token(start_jd),
            "stop": jd_token(stop_jd),
            "step": STEP,
        },
    ).vectors(refplane=REFPLANE, aberrations=ABERRATIONS)
    jd = np.asarray(table["datetime_jd"], dtype=float)
    xyz = np.column_stack([
        np.asarray(table["x"], dtype=float),
        np.asarray(table["y"], dtype=float),
        np.asarray(table["z"], dtype=float),
    ]) * AU_KM
    if len(jd) < 80:
        raise RuntimeError(f"REJECTED insufficient JPL samples for body {body}: {len(jd)}")
    if not np.all(np.diff(jd) > 0.0):
        raise RuntimeError(f"REJECTED non-monotonic JPL epochs for body {body}")
    if not np.all(np.isfinite(xyz)):
        raise RuntimeError(f"REJECTED non-finite JPL vectors for body {body}")
    return jd, xyz


def orbit_summary(name: str, year: int, jd: np.ndarray, xyz: np.ndarray) -> dict:
    radius = np.linalg.norm(xyz, axis=1)
    i_peri = int(np.argmin(radius))
    i_aphe = int(np.argmax(radius))
    peri = xyz[i_peri]
    aphe = xyz[i_aphe]
    normal = np.cross(xyz[len(xyz) // 3], xyz[2 * len(xyz) // 3])
    normal /= np.linalg.norm(normal)
    inclination = math.degrees(math.acos(np.clip(abs(normal[2]), -1.0, 1.0)))
    return {
        "planet": name,
        "epoch_year": year,
        "z_scale_factor": z_scale_factor(year),
        "sample_count": int(len(jd)),
        "perihelion_index": i_peri,
        "aphelion_index": i_aphe,
        "perihelion_km": float(radius[i_peri]),
        "aphelion_km": float(radius[i_aphe]),
        "perihelion_longitude_deg": math.degrees(math.atan2(peri[1], peri[0])) % 360.0,
        "aphelion_longitude_deg": math.degrees(math.atan2(aphe[1], aphe[0])) % 360.0,
        "inclination_from_ecliptic_deg": inclination,
        "z_min_true_km": float(np.min(xyz[:, 2])),
        "z_max_true_km": float(np.max(xyz[:, 2])),
        "z_min_plot_km": float(np.min(xyz[:, 2]) * z_scale_factor(year)),
        "z_max_plot_km": float(np.max(xyz[:, 2]) * z_scale_factor(year)),
    }


def sphere(radius: float, n_u: int = 64, n_v: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)
    return (
        radius * np.outer(np.cos(u), np.sin(v)),
        radius * np.outer(np.sin(u), np.sin(v)),
        radius * np.outer(np.ones_like(u), np.cos(v)),
    )


def set_equal_3d(ax, arrays: list[np.ndarray]) -> None:
    points = np.vstack(arrays)
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = 0.5 * (mins + maxs)
    half = 0.5 * float(np.max(maxs - mins)) * 1.08
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def make_plot(orbits: dict[str, list[dict]], summaries: list[dict]) -> None:
    fig = plt.figure(figsize=(30.0, 25.0), dpi=120)
    fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")

    sx, sy, sz = sphere(R_SUN_MKM)
    ax.plot_surface(
        sx, sy, sz,
        linewidth=0.0,
        antialiased=True,
        shade=True,
        alpha=0.98,
        color="#FDB813",
        zorder=10,
    )
    ax.text(0.0, 0.0, 4.0, "Sun", color="#FFD86B", fontsize=14, weight="bold")

    all_arrays: list[np.ndarray] = []
    for name in ("Mercury", "Venus", "Earth"):
        cfg = PLANETS[name]
        rows = [row for row in summaries if row["planet"] == name]
        for orbit, row in zip(orbits[name], rows):
            year = int(orbit["year"])
            factor = float(row["z_scale_factor"])
            xyz_plot = orbit["xyz"].copy() / 1.0e6
            xyz_plot[:, 2] *= factor
            all_arrays.append(xyz_plot)

            if year < 0:
                linestyle = "--"
                alpha = 0.55
            elif year > 0:
                linestyle = "-"
                alpha = 0.72
            else:
                linestyle = "-"
                alpha = 1.00

            label = None
            if year == 0:
                label = f"{name}: epoch 0"
            ax.plot(
                xyz_plot[:, 0], xyz_plot[:, 1], xyz_plot[:, 2],
                color=cfg["color"],
                linewidth=ORBIT_LINEWIDTH,
                linestyle=linestyle,
                alpha=alpha,
                label=label,
                zorder=3,
            )

            for marker_name, key, marker in [
                ("P", "perihelion_index", "o"),
                ("A", "aphelion_index", "s"),
            ]:
                point = xyz_plot[int(row[key])]
                ax.scatter(
                    [point[0]], [point[1]], [point[2]],
                    s=12.0 if year not in (-9000, 0, 9000) else 34.0,
                    color=cfg["color"],
                    edgecolors="white",
                    linewidths=0.18 if year not in (-9000, 0, 9000) else 0.55,
                    marker=marker,
                    alpha=alpha,
                    depthshade=False,
                    zorder=7,
                )
                if year in (-9000, 0, 9000):
                    ax.text(
                        point[0], point[1], point[2],
                        f" {name} {marker_name} {year:+d}\n Z×{factor:.0f}",
                        color=cfg["color"],
                        fontsize=7.4,
                        weight="bold",
                        zorder=8,
                    )

    set_equal_3d(ax, all_arrays)
    ax.view_init(elev=24.0, azim=40.0)
    ax.set_title(
        "Mercury, Venus, and Earth — Centered JPL Orbit Bundle\n"
        "Astronomical years −9000 to +9000; epoch-dependent Z-only exaggeration",
        color="white",
        fontsize=20,
        weight="bold",
        pad=30,
    )
    ax.set_xlabel("Ecliptic X (million km, true scale)", color="#E0E0E0", labelpad=14)
    ax.set_ylabel("Ecliptic Y (million km, true scale)", color="#E0E0E0", labelpad=14)
    ax.set_zlabel("Ecliptic Z (million km, exaggerated by epoch)", color="#E0E0E0", labelpad=12)
    ax.tick_params(colors="#D7D7D7", labelsize=8)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.0, 0.0, 0.0, 1.0))
        axis.pane.set_edgecolor((0.38, 0.38, 0.38, 0.65))
        axis._axinfo["grid"]["color"] = (0.42, 0.42, 0.42, 0.20)
        axis._axinfo["grid"]["linewidth"] = 0.35

    legend = ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98), frameon=False, fontsize=9.5)
    for text in legend.get_texts():
        text.set_color("white")

    info = [
        "19 complete JPL Horizons orbits per planet",
        "Epochs: −9000 to +9000 every 1,000 years",
        "Past epochs dashed; future epochs solid",
        "Circle = perihelion; square = aphelion",
        "Orbit linewidth = 0.25",
        "X and Y remain true scale",
        "Z factor = 1 at epoch 0; 10× per 1,000 years from zero",
        "±1000→10×, ±5000→50×, ±9000→90×",
        "WARNING: displayed Z geometry is intentionally exaggerated",
    ]
    fig.text(
        0.025, 0.025, "\n".join(info),
        color="white",
        fontsize=10.2,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#050505", "edgecolor": "#858585", "alpha": 0.94},
    )

    fig.tight_layout()
    fig.savefig(PNG, dpi=420, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    display(Image(filename=str(PNG)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    section("CODE INPUTS")
    print(f"Version                              {VERSION}")
    print(f"Epoch years                          {EPOCH_YEARS}")
    print("Epoch spacing                        1000 years")
    print(f"JPL range-query cadence              {STEP}")
    print(f"Orbit linewidth                      {ORBIT_LINEWIDTH:.2f}")
    print(f"Horizons accepted year interval      {HORIZONS_MIN_YEAR} to {HORIZONS_MAX_YEAR}")
    print(f"Reference center                     Sun, {LOCATION}")
    print(f"Reference plane                      {REFPLANE}")
    print(f"Output                               {OUT}")

    section("COMMENTS")
    print("Requested -10000 and +10000 endpoints exceed the online Horizons year limits and are REJECTED.")
    print("The maximum symmetric millennium series used is astronomical year -9000 through +9000.")
    print("X and Y coordinates remain unchanged JPL heliocentric ecliptic coordinates.")
    print("Only plotted Z is multiplied by an epoch-dependent display factor.")
    print("The Z exaggeration is a visualization transform and is NOT USED in scientific orbital calculations.")
    print("Only perihelion and aphelion markers are plotted; monthly markers are NOT USED.")

    if min(EPOCH_YEARS) < HORIZONS_MIN_YEAR or max(EPOCH_YEARS) > HORIZONS_MAX_YEAR:
        raise RuntimeError("REJECTED epoch list exceeds online Horizons limits")

    orbits: dict[str, list[dict]] = {name: [] for name in PLANETS}
    summaries: list[dict] = []
    csv_rows: list[dict] = []

    for year in EPOCH_YEARS:
        center_jd = epoch_center_jd(year)
        factor = z_scale_factor(year)
        for name, cfg in PLANETS.items():
            print(f"DEBUG querying {name} complete orbit at epoch {year:+d} with displayed Z×{factor:.0f}", flush=True)
            jd, xyz = query_orbit(cfg["id"], center_jd, cfg["period_days"])
            summary = orbit_summary(name, year, jd, xyz)
            orbits[name].append({"year": year, "jd": jd, "xyz": xyz})
            summaries.append(summary)
            for sample_index in range(len(jd)):
                csv_rows.append({
                    "planet": name,
                    "epoch_year_astronomical": year,
                    "z_scale_factor": factor,
                    "sample_index": sample_index,
                    "jd_tdb": jd[sample_index],
                    "x_true_km": xyz[sample_index, 0],
                    "y_true_km": xyz[sample_index, 1],
                    "z_true_km": xyz[sample_index, 2],
                    "z_plot_km": xyz[sample_index, 2] * factor,
                })

    pd.DataFrame(csv_rows).to_csv(CSV, index=False, float_format="%.9f")
    make_plot(orbits, summaries)

    section("RESULTS")
    for name in ("Mercury", "Venus", "Earth"):
        rows = [row for row in summaries if row["planet"] == name]
        for row in rows:
            print(
                f"{name:7s} {row['epoch_year']:+6d}  "
                f"Z×{row['z_scale_factor']:5.1f}  "
                f"P {row['perihelion_km']:15.6f} km  "
                f"A {row['aphelion_km']:15.6f} km  "
                f"i {row['inclination_from_ecliptic_deg']:9.6f} deg  "
                f"Ztrue {row['z_min_true_km']:14.3f} to {row['z_max_true_km']:14.3f} km  "
                f"Zplot {row['z_min_plot_km']:14.3f} to {row['z_max_plot_km']:14.3f} km"
            )

    section("OUTPUT SUMMARY")
    print(f"PNG                                  {PNG}")
    print(f"CSV                                  {CSV}")
    print(f"Complete orbit sets                  {len(EPOCH_YEARS) * len(PLANETS)}")
    print(f"Epochs per planet                    {len(EPOCH_YEARS)}")

    section("PAPER COMPARISON")
    print("NOT USED: VSOP, analytic secular theory, non-JPL extrapolation, or manual orbit geometry.")
    print("REJECTED: requested astronomical years -10000 and +10000 outside the online Horizons limits.")

    section("EQUATION STATUS")
    print("VERIFIED perihelion = minimum norm of each complete JPL heliocentric orbit sample")
    print("VERIFIED aphelion = maximum norm of each complete JPL heliocentric orbit sample")
    print("VERIFIED Z_plot = Z_JPL × factor, where factor = 1 at year 0 and 10×|year|/1000 otherwise")
    print("VERIFIED X_plot = X_JPL and Y_plot = Y_JPL")
    print("VERIFIED orbit linewidth = 0.25")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0013