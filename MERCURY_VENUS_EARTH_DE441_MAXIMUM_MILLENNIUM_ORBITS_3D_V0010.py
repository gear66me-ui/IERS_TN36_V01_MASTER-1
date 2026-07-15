# V0010
# Audit reference: corrected standalone JPL DE441 millennium 3D orbit bundle using compact start-stop range queries instead of oversized epoch lists.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display

VERSION = "V0010"
LOCAL_TZ = ZoneInfo("America/Bogota")
LOCATION = "@10"
REFPLANE = "ecliptic"
ABERRATIONS = "geometric"
AU_KM = 149597870.700
R_SUN_KM = 695700.000
R_SUN_MKM = R_SUN_KM / 1.0e6
EPOCH_YEARS = list(range(2026, 17027, 1000))
QUERY_STEP = "1d"
OUT = Path("/content/MERCURY_VENUS_EARTH_DE441_MAXIMUM_MILLENNIUM_ORBITS_3D_V0010_OUTPUT")
PNG = OUT / "MERCURY_VENUS_EARTH_DE441_MAXIMUM_MILLENNIUM_ORBITS_3D_V0010.png"
CSV = OUT / "MERCURY_VENUS_EARTH_DE441_MAXIMUM_MILLENNIUM_ORBITS_3D_V0010.csv"

PLANETS = {
    "Mercury": {"id": "199", "color": "#A970FF", "period_days": 87.9691},
    "Venus": {"id": "299", "color": "#2F8DFF", "period_days": 224.701},
    "Earth": {"id": "399", "color": "#35C96B", "period_days": 365.256},
}


def section(name: str) -> None:
    print(name)
    print("-" * len(name))


def epoch_jd(year: int) -> float:
    return float(Time(f"J{year}.0", format="jyear_str", scale="tdb").jd)


def query_orbit(body: str, center_jd: float, period_days: float) -> tuple[np.ndarray, np.ndarray]:
    margin_days = 2.0
    start_jd = center_jd - 0.5 * period_days - margin_days
    stop_jd = center_jd + 0.5 * period_days + margin_days
    epochs = {
        "start": f"JD{start_jd:.9f}",
        "stop": f"JD{stop_jd:.9f}",
        "step": QUERY_STEP,
    }
    table = Horizons(
        id=body,
        id_type="majorbody",
        location=LOCATION,
        epochs=epochs,
    ).vectors(refplane=REFPLANE, aberrations=ABERRATIONS)
    jd = np.asarray(table["datetime_jd"], dtype=float)
    xyz = np.column_stack([
        np.asarray(table["x"], dtype=float),
        np.asarray(table["y"], dtype=float),
        np.asarray(table["z"], dtype=float),
    ]) * AU_KM
    if len(jd) < 80:
        raise RuntimeError(f"REJECTED insufficient JPL samples for body {body}: {len(jd)}")
    if not np.all(np.diff(jd) > 0.0) or not np.all(np.isfinite(xyz)):
        raise RuntimeError(f"REJECTED invalid JPL orbit for body {body}")
    return jd, xyz


def orbit_summary(name: str, year: int, jd: np.ndarray, xyz: np.ndarray) -> dict:
    radius = np.linalg.norm(xyz, axis=1)
    i_peri = int(np.argmin(radius))
    i_aphe = int(np.argmax(radius))
    peri = xyz[i_peri]
    aphe = xyz[i_aphe]
    peri_lon = math.degrees(math.atan2(peri[1], peri[0])) % 360.0
    aphe_lon = math.degrees(math.atan2(aphe[1], aphe[0])) % 360.0

    centered = xyz - np.mean(xyz, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0.0:
        normal = -normal
    inclination = math.degrees(math.acos(np.clip(normal[2], -1.0, 1.0)))

    return {
        "planet": name,
        "epoch_year": year,
        "sample_count": len(jd),
        "perihelion_index": i_peri,
        "aphelion_index": i_aphe,
        "perihelion_km": float(radius[i_peri]),
        "aphelion_km": float(radius[i_aphe]),
        "perihelion_longitude_deg": peri_lon,
        "aphelion_longitude_deg": aphe_lon,
        "inclination_from_ecliptic_deg": inclination,
        "perihelion_x_km": float(peri[0]),
        "perihelion_y_km": float(peri[1]),
        "perihelion_z_km": float(peri[2]),
        "aphelion_x_km": float(aphe[0]),
        "aphelion_y_km": float(aphe[1]),
        "aphelion_z_km": float(aphe[2]),
    }


def unwrap_relative(values_deg: list[float]) -> np.ndarray:
    radians = np.unwrap(np.radians(np.asarray(values_deg, dtype=float)))
    degrees = np.degrees(radians)
    return degrees - degrees[0]


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
    fig = plt.figure(figsize=(26.0, 22.0), dpi=120)
    fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")

    sx, sy, sz = sphere(R_SUN_MKM)
    ax.plot_surface(sx, sy, sz, linewidth=0.0, antialiased=True, shade=True,
                    alpha=0.98, color="#FDB813", zorder=10)
    ax.text(0.0, 0.0, 4.0, "Sun", color="#FFD86B", fontsize=14, weight="bold")

    all_arrays: list[np.ndarray] = []
    epoch_count = len(EPOCH_YEARS)
    for name in ("Mercury", "Venus", "Earth"):
        cfg = PLANETS[name]
        planet_rows = [row for row in summaries if row["planet"] == name]
        rotations = unwrap_relative([row["perihelion_longitude_deg"] for row in planet_rows])
        for index, orbit in enumerate(orbits[name]):
            xyz_mkm = orbit["xyz"] / 1.0e6
            all_arrays.append(xyz_mkm)
            alpha = 0.20 + 0.75 * index / max(1, epoch_count - 1)
            width = 0.50 + 0.80 * index / max(1, epoch_count - 1)
            label = f"{name}: 2026–17026" if index == epoch_count - 1 else None
            ax.plot(xyz_mkm[:, 0], xyz_mkm[:, 1], xyz_mkm[:, 2],
                    color=cfg["color"], linewidth=width, alpha=alpha,
                    label=label, zorder=3)

            row = planet_rows[index]
            for marker_name, key, marker in [
                ("P", "perihelion_index", "o"),
                ("A", "aphelion_index", "s"),
            ]:
                point = xyz_mkm[int(row[key])]
                ax.scatter([point[0]], [point[1]], [point[2]],
                           s=20.0 if index < epoch_count - 1 else 52.0,
                           color=cfg["color"], edgecolors="white",
                           linewidths=0.25 if index < epoch_count - 1 else 0.75,
                           marker=marker, alpha=alpha, depthshade=False, zorder=7)
                if index in (0, epoch_count - 1):
                    ax.text(point[0], point[1], point[2],
                            f" {name} {marker_name} {orbit['year']}\n Δϖ={rotations[index]:+.2f}°",
                            color=cfg["color"], fontsize=7.5, weight="bold", zorder=8)

    set_equal_3d(ax, all_arrays)
    ax.view_init(elev=26.0, azim=40.0)
    ax.set_title(
        "Mercury, Venus, and Earth — JPL DE441 Millennium Orbit Bundle\n"
        "Complete daily-sampled orbits every 1,000 years from 2026 through 17026",
        color="white", fontsize=20, weight="bold", pad=28,
    )
    ax.set_xlabel("Ecliptic X (million km)", color="#E0E0E0", labelpad=14)
    ax.set_ylabel("Ecliptic Y (million km)", color="#E0E0E0", labelpad=14)
    ax.set_zlabel("Ecliptic Z (million km)", color="#E0E0E0", labelpad=12)
    ax.tick_params(colors="#D7D7D7", labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.0, 0.0, 0.0, 1.0))
        axis.pane.set_edgecolor((0.38, 0.38, 0.38, 0.65))
        axis._axinfo["grid"]["color"] = (0.42, 0.42, 0.42, 0.20)
        axis._axinfo["grid"]["linewidth"] = 0.45

    legend = ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98), frameon=False, fontsize=10.0)
    for text in legend.get_texts():
        text.set_color("white")

    info = [
        "16 complete JPL orbits per planet",
        "Epoch spacing: 1,000 years",
        "Range-query cadence: 1 day",
        "Circle = perihelion; square = aphelion",
        "Δϖ = perihelion-direction rotation from 2026",
        "Sun-centered ecliptic frame; Sun radius true scale",
    ]
    for name in ("Mercury", "Venus", "Earth"):
        rows = [row for row in summaries if row["planet"] == name]
        rotation = unwrap_relative([row["perihelion_longitude_deg"] for row in rows])[-1]
        info.append(f"{name}: total Δϖ 2026→17026 = {rotation:+.3f}°")
    fig.text(0.025, 0.025, "\n".join(info), color="white", fontsize=10.2,
             ha="left", va="bottom",
             bbox={"boxstyle": "round,pad=0.45", "facecolor": "#050505", "edgecolor": "#858585", "alpha": 0.94})

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
    print(f"JPL range-query cadence              {QUERY_STEP}")
    print(f"Reference center                     Sun, {LOCATION}")
    print(f"Reference plane                      {REFPLANE}")
    print(f"Output                               {OUT}")

    section("COMMENTS")
    print("A 100,000-year plot is outside the JPL DE441 interval and is REJECTED by the JPL-only project rule.")
    print("This widget uses the maximum practical future interval inside DE441: 2026 through 17026.")
    print("Each epoch contains one complete JPL orbit queried with compact START, STOP, and STEP parameters.")
    print("The oversized TLIST method from V0009 is REJECTED and NOT USED.")
    print("Only perihelion and aphelion markers are plotted; monthly markers are NOT USED.")

    orbits: dict[str, list[dict]] = {name: [] for name in PLANETS}
    summaries: list[dict] = []
    csv_rows: list[dict] = []

    for year in EPOCH_YEARS:
        center = epoch_jd(year)
        for name, cfg in PLANETS.items():
            print(f"DEBUG querying {name} complete orbit at epoch {year}", flush=True)
            jd, xyz = query_orbit(cfg["id"], center, cfg["period_days"])
            summary = orbit_summary(name, year, jd, xyz)
            orbits[name].append({"year": year, "jd": jd, "xyz": xyz})
            summaries.append(summary)
            for sample_index in range(len(jd)):
                csv_rows.append({
                    "planet": name,
                    "epoch_year": year,
                    "sample_index": sample_index,
                    "jd_tdb": jd[sample_index],
                    "x_km": xyz[sample_index, 0],
                    "y_km": xyz[sample_index, 1],
                    "z_km": xyz[sample_index, 2],
                    "radius_km": float(np.linalg.norm(xyz[sample_index])),
                })

    pd.DataFrame(csv_rows).to_csv(CSV, index=False, float_format="%.9f")
    make_plot(orbits, summaries)

    section("RESULTS")
    for name in ("Mercury", "Venus", "Earth"):
        rows = [row for row in summaries if row["planet"] == name]
        rotations = unwrap_relative([row["perihelion_longitude_deg"] for row in rows])
        for row, rotation in zip(rows, rotations):
            print(
                f"{name:8s} {row['epoch_year']:5d}  samples {row['sample_count']:4d}  "
                f"peri {row['perihelion_km']:.3f} km  aphe {row['aphelion_km']:.3f} km  "
                f"ϖ {row['perihelion_longitude_deg']:.6f}°  Δϖ {rotation:+.6f}°  "
                f"i {row['inclination_from_ecliptic_deg']:.6f}°"
            )

    section("OUTPUT SUMMARY")
    print(f"PNG                                  {PNG}")
    print(f"CSV                                  {CSV}")
    print(f"Complete orbits plotted              {len(EPOCH_YEARS) * len(PLANETS)}")
    print(f"Total JPL vector samples             {len(csv_rows)}")

    section("PAPER COMPARISON")
    print("NOT USED: analytical secular-orbit models or non-JPL extrapolations beyond DE441.")
    print("The 100,000-year request remains outside the JPL-only interval and is not plotted.")

    section("EQUATION STATUS")
    print("VERIFIED heliocentric radius = sqrt(X^2 + Y^2 + Z^2)")
    print("VERIFIED perihelion = minimum sampled heliocentric radius in each complete orbit")
    print("VERIFIED aphelion = maximum sampled heliocentric radius in each complete orbit")
    print("VERIFIED perihelion longitude = atan2(Y_peri, X_peri) in the JPL ecliptic frame")
    print("VERIFIED V0009 oversized TLIST URL is replaced by compact START/STOP/STEP range queries")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0010