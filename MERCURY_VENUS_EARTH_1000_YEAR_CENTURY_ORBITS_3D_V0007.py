# V0007
# Audit reference: standalone JPL Horizons century-sampled 1000-year heliocentric ecliptic 3D orbit/precession plot for Mercury, Venus, and Earth.
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

VERSION = "V0007"
LOCAL_TZ = ZoneInfo("America/Bogota")
BASE_YEAR = 2026
CENTURY_YEARS = list(range(BASE_YEAR, BASE_YEAR + 1001, 100))
LOCATION = "@10"
REFPLANE = "ecliptic"
ABERRATIONS = "geometric"
AU_KM = 149597870.700
R_SUN_KM = 695700.000
R_SUN_MKM = R_SUN_KM / 1.0e6
OUT = Path("/content/MERCURY_VENUS_EARTH_1000_YEAR_CENTURY_ORBITS_3D_V0007_OUTPUT")
PNG = OUT / "MERCURY_VENUS_EARTH_1000_YEAR_CENTURY_ORBITS_3D_V0007.png"
CSV = OUT / "MERCURY_VENUS_EARTH_1000_YEAR_CENTURY_ORBITS_3D_V0007.csv"

PLANETS = {
    "Mercury": {
        "id": "199",
        "color": "#A970FF",
        "duration_days": 105.0,
        "step": "6h",
        "line_width": 0.72,
    },
    "Venus": {
        "id": "299",
        "color": "#2F8DFF",
        "duration_days": 250.0,
        "step": "12h",
        "line_width": 0.76,
    },
    "Earth": {
        "id": "399",
        "color": "#35C96B",
        "duration_days": 400.0,
        "step": "1d",
        "line_width": 0.82,
    },
}


def section(name: str) -> None:
    print(name)
    print("-" * len(name))


def wrap180(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def query_orbit(body: str, year: int, duration_days: float, step: str) -> tuple[np.ndarray, np.ndarray]:
    start = Time(f"{year}-01-01 00:00", scale="utc")
    stop = Time(start.jd + duration_days, format="jd", scale="utc")
    table = Horizons(
        id=body,
        id_type="majorbody",
        location=LOCATION,
        epochs={
            "start": start.strftime("%Y-%m-%d %H:%M"),
            "stop": stop.strftime("%Y-%m-%d %H:%M"),
            "step": step,
        },
    ).vectors(refplane=REFPLANE, aberrations=ABERRATIONS)
    jd = np.asarray(table["datetime_jd"], dtype=float)
    xyz = np.column_stack([
        np.asarray(table["x"], dtype=float),
        np.asarray(table["y"], dtype=float),
        np.asarray(table["z"], dtype=float),
    ]) * AU_KM
    if len(jd) < 80 or not np.all(np.diff(jd) > 0.0):
        raise RuntimeError(f"REJECTED JPL grid for body {body}, year {year}")
    if not np.all(np.isfinite(xyz)):
        raise RuntimeError(f"REJECTED non-finite JPL vectors for body {body}, year {year}")
    return jd, xyz


def orbital_plane(xyz: np.ndarray) -> tuple[np.ndarray, float, float]:
    matrix = np.asarray(xyz, dtype=float)
    _, _, vh = np.linalg.svd(matrix, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if normal[2] < 0.0:
        normal = -normal
    inclination = math.degrees(math.acos(float(np.clip(normal[2], -1.0, 1.0))))
    node = np.cross(np.array([0.0, 0.0, 1.0]), normal)
    if np.linalg.norm(node) <= 1.0e-15:
        node_longitude = 0.0
    else:
        node /= np.linalg.norm(node)
        node_longitude = math.degrees(math.atan2(node[1], node[0])) % 360.0
    return normal, inclination, node_longitude


def summarize_orbit(name: str, year: int, jd: np.ndarray, xyz: np.ndarray) -> dict:
    radius = np.linalg.norm(xyz, axis=1)
    i_peri = int(np.argmin(radius))
    i_aphe = int(np.argmax(radius))
    peri = xyz[i_peri]
    aphe = xyz[i_aphe]
    normal, inclination, node_longitude = orbital_plane(xyz)
    peri_longitude = math.degrees(math.atan2(peri[1], peri[0])) % 360.0
    aphe_longitude = math.degrees(math.atan2(aphe[1], aphe[0])) % 360.0
    return {
        "planet": name,
        "epoch_year": year,
        "perihelion_index": i_peri,
        "aphelion_index": i_aphe,
        "perihelion_km": float(radius[i_peri]),
        "aphelion_km": float(radius[i_aphe]),
        "perihelion_utc": Time(float(jd[i_peri]), format="jd", scale="tdb").utc.strftime("%Y-%m-%d %H:%M:%S"),
        "aphelion_utc": Time(float(jd[i_aphe]), format="jd", scale="tdb").utc.strftime("%Y-%m-%d %H:%M:%S"),
        "perihelion_longitude_deg": peri_longitude,
        "aphelion_longitude_deg": aphe_longitude,
        "inclination_deg": inclination,
        "ascending_node_longitude_deg": node_longitude,
        "normal_x": float(normal[0]),
        "normal_y": float(normal[1]),
        "normal_z": float(normal[2]),
        "perihelion_x_km": float(peri[0]),
        "perihelion_y_km": float(peri[1]),
        "perihelion_z_km": float(peri[2]),
        "aphelion_x_km": float(aphe[0]),
        "aphelion_y_km": float(aphe[1]),
        "aphelion_z_km": float(aphe[2]),
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
    half = 0.5 * float(np.max(maxs - mins)) * 1.10
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def make_plot(orbits: dict[str, dict[int, np.ndarray]], summaries: dict[str, dict[int, dict]]) -> None:
    fig = plt.figure(figsize=(26.0, 22.0), dpi=120)
    fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")

    sx, sy, sz = sphere(R_SUN_MKM)
    ax.plot_surface(
        sx, sy, sz,
        rstride=2,
        cstride=2,
        linewidth=0.0,
        antialiased=True,
        shade=True,
        alpha=0.98,
        color="#FDB813",
        zorder=10,
    )
    ax.text(0.0, 0.0, 3.0, "Sun", color="#FFD86B", fontsize=14, weight="bold")

    all_arrays: list[np.ndarray] = []
    n_epochs = len(CENTURY_YEARS)
    for name in ("Mercury", "Venus", "Earth"):
        cfg = PLANETS[name]
        for epoch_index, year in enumerate(CENTURY_YEARS):
            xyz_mkm = orbits[name][year] / 1.0e6
            all_arrays.append(xyz_mkm)
            alpha = 0.24 + 0.70 * epoch_index / max(1, n_epochs - 1)
            label = f"{name}: {CENTURY_YEARS[0]}–{CENTURY_YEARS[-1]}" if epoch_index == n_epochs - 1 else None
            ax.plot(
                xyz_mkm[:, 0],
                xyz_mkm[:, 1],
                xyz_mkm[:, 2],
                color=cfg["color"],
                linewidth=cfg["line_width"],
                alpha=alpha,
                label=label,
                zorder=3,
            )

            row = summaries[name][year]
            for marker_name, index_key, marker_shape in [
                ("P", "perihelion_index", "o"),
                ("A", "aphelion_index", "s"),
            ]:
                point = xyz_mkm[int(row[index_key])]
                ax.scatter(
                    [point[0]], [point[1]], [point[2]],
                    s=18.0 if year != CENTURY_YEARS[-1] else 52.0,
                    color=cfg["color"],
                    edgecolors="white",
                    linewidths=0.25 if year != CENTURY_YEARS[-1] else 0.75,
                    marker=marker_shape,
                    alpha=alpha,
                    depthshade=False,
                    zorder=7,
                )
                if year in (CENTURY_YEARS[0], CENTURY_YEARS[-1]):
                    ax.text(
                        point[0], point[1], point[2],
                        f" {name} {marker_name} {year}",
                        color=cfg["color"],
                        fontsize=8.0,
                        weight="bold",
                        zorder=9,
                    )

    set_equal_3d(ax, all_arrays)
    ax.view_init(elev=27.0, azim=42.0)
    ax.set_title(
        "Mercury, Venus, and Earth — Century-Spaced Orbits Across 1,000 Years",
        color="white",
        fontsize=20,
        weight="bold",
        pad=28,
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
        "JPL heliocentric ecliptic vectors",
        "One complete orbit every 100 years",
        "Circle = perihelion; square = aphelion",
        "Orbit opacity increases from 2026 to 3026",
        "Sun radius shown at true orbital scale",
    ]
    for name in ("Mercury", "Venus", "Earth"):
        first = summaries[name][CENTURY_YEARS[0]]
        last = summaries[name][CENTURY_YEARS[-1]]
        apsidal_rotation = wrap180(last["perihelion_longitude_deg"] - first["perihelion_longitude_deg"])
        node_rotation = wrap180(last["ascending_node_longitude_deg"] - first["ascending_node_longitude_deg"])
        info.append(
            f"{name}: perihelion longitude shift {apsidal_rotation:+.3f}°, "
            f"node shift {node_rotation:+.3f}°"
        )
    fig.text(
        0.025,
        0.025,
        "\n".join(info),
        color="white",
        fontsize=10.0,
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
    print(f"Century epochs                       {CENTURY_YEARS}")
    print(f"Reference center                     Sun, {LOCATION}")
    print(f"Reference plane                      {REFPLANE}")
    print(f"Aberrations                          {ABERRATIONS}")
    print(f"Sun radius                           {R_SUN_KM:.3f} km")
    print(f"Output                               {OUT}")

    section("COMMENTS")
    print("One complete JPL orbit is queried at each 100-year epoch from 2026 through 3026.")
    print("Only perihelion and aphelion markers are plotted; monthly markers are NOT USED.")
    print("The orbit bundles reveal long-term apsidal and nodal precession when it is large enough to resolve.")
    print("All orbital coordinates retain the original JPL heliocentric ecliptic geometry.")

    orbits: dict[str, dict[int, np.ndarray]] = {name: {} for name in PLANETS}
    summaries: dict[str, dict[int, dict]] = {name: {} for name in PLANETS}
    csv_rows: list[dict] = []

    for name, cfg in PLANETS.items():
        for year in CENTURY_YEARS:
            print(f"DEBUG querying {name} epoch {year}", flush=True)
            jd, xyz = query_orbit(cfg["id"], year, cfg["duration_days"], cfg["step"])
            row = summarize_orbit(name, year, jd, xyz)
            orbits[name][year] = xyz
            summaries[name][year] = row
            csv_rows.append({key: value for key, value in row.items() if not key.endswith("_index")})

    pd.DataFrame(csv_rows).to_csv(CSV, index=False, float_format="%.12f")
    make_plot(orbits, summaries)

    section("RESULTS")
    for name in ("Mercury", "Venus", "Earth"):
        first = summaries[name][CENTURY_YEARS[0]]
        for year in CENTURY_YEARS:
            row = summaries[name][year]
            peri_shift = wrap180(row["perihelion_longitude_deg"] - first["perihelion_longitude_deg"])
            node_shift = wrap180(row["ascending_node_longitude_deg"] - first["ascending_node_longitude_deg"])
            print(
                f"{name:8s} {year}  peri {row['perihelion_km']:.3f} km  "
                f"aphe {row['aphelion_km']:.3f} km  "
                f"peri-longitude shift {peri_shift:+.6f} deg  "
                f"node shift {node_shift:+.6f} deg  "
                f"inclination {row['inclination_deg']:.6f} deg"
            )

    section("OUTPUT SUMMARY")
    print(f"PNG                                  {PNG}")
    print(f"CSV                                  {CSV}")
    print(f"Epochs per planet                    {len(CENTURY_YEARS)}")
    print(f"Total JPL orbit windows              {len(CENTURY_YEARS) * len(PLANETS)}")

    section("PAPER COMPARISON")
    print("NOT USED: fixed published orbital elements or manually rotated ellipses.")
    print("All perihelia, aphelia, plane normals, inclinations, and angular shifts are derived from JPL vectors.")

    section("EQUATION STATUS")
    print("VERIFIED perihelion = minimum heliocentric vector magnitude within each complete orbit window")
    print("VERIFIED aphelion = maximum heliocentric vector magnitude within each complete orbit window")
    print("VERIFIED orbital-plane normal = smallest-singular-value direction of the JPL position matrix")
    print("VERIFIED apsidal rotation = wrapped change in ecliptic longitude of the perihelion vector")
    print("VERIFIED nodal rotation = wrapped change in ascending-node longitude")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0007