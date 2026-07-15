# V0018
# Audit reference: standalone BCE-safe JPL Horizons centered 3D orbit bundle with uniform Z x 1.10 scaling and minimal notebook output.
from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def need(module: str, package: str) -> None:
    if importlib.util.find_spec(module) is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", package],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


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

VERSION = "V0018"
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
SAMPLES_PER_ORBIT = 64
QUERY_BATCH_SIZE = 32
ORBIT_LINEWIDTH = 0.25
Z_SCALE = 1.10
OUT = Path("/content/MERCURY_VENUS_EARTH_CENTERED_18000_YEAR_Z_1P10X_ORBITS_3D_V0018_OUTPUT")
PNG = OUT / "MERCURY_VENUS_EARTH_CENTERED_18000_YEAR_Z_1P10X_ORBITS_3D_V0018.png"
CSV = OUT / "MERCURY_VENUS_EARTH_CENTERED_18000_YEAR_Z_1P10X_ORBITS_3D_V0018.csv"

PLANETS = {
    "Mercury": {"id": "199", "color": "#A970FF", "period_days": 87.9691},
    "Venus": {"id": "299", "color": "#2F8DFF", "period_days": 224.701},
    "Earth": {"id": "399", "color": "#35C96B", "period_days": 365.256},
}


def epoch_center_jd(year: int) -> float:
    if not HORIZONS_MIN_YEAR <= year <= HORIZONS_MAX_YEAR:
        raise ValueError(f"REJECTED epoch {year}: Horizons limit is {HORIZONS_MIN_YEAR} to {HORIZONS_MAX_YEAR}")
    return J2000_JD + (float(year) - 2000.0) * DAYS_PER_JULIAN_YEAR


def query_orbit(body: str, center_jd: float, period_days: float) -> tuple[np.ndarray, np.ndarray]:
    epochs = np.linspace(
        center_jd - 0.5 * period_days,
        center_jd + 0.5 * period_days,
        SAMPLES_PER_ORBIT,
        endpoint=True,
    )
    jd_parts: list[np.ndarray] = []
    xyz_parts: list[np.ndarray] = []
    for start in range(0, len(epochs), QUERY_BATCH_SIZE):
        batch = epochs[start:start + QUERY_BATCH_SIZE]
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            table = Horizons(
                id=body,
                id_type="majorbody",
                location=LOCATION,
                epochs=batch.tolist(),
            ).vectors(refplane=REFPLANE, aberrations=ABERRATIONS)
        jd_parts.append(np.asarray(table["datetime_jd"], dtype=float))
        xyz_parts.append(
            np.column_stack([
                np.asarray(table["x"], dtype=float),
                np.asarray(table["y"], dtype=float),
                np.asarray(table["z"], dtype=float),
            ]) * AU_KM
        )
    jd = np.concatenate(jd_parts)
    xyz = np.vstack(xyz_parts)
    order = np.argsort(jd)
    jd = jd[order]
    xyz = xyz[order]
    if len(jd) != SAMPLES_PER_ORBIT:
        raise RuntimeError(f"REJECTED sample count for body {body}: {len(jd)}")
    if not np.all(np.diff(jd) > 0.0) or not np.all(np.isfinite(xyz)):
        raise RuntimeError(f"REJECTED invalid JPL vectors for body {body}")
    return jd, xyz


def orbit_summary(name: str, year: int, jd: np.ndarray, xyz: np.ndarray) -> dict:
    radius = np.linalg.norm(xyz, axis=1)
    i_peri = int(np.argmin(radius))
    i_aphe = int(np.argmax(radius))
    p1 = xyz[len(xyz) // 3]
    p2 = xyz[2 * len(xyz) // 3]
    normal = np.cross(p1, p2)
    normal /= np.linalg.norm(normal)
    inclination = math.degrees(math.acos(np.clip(abs(normal[2]), -1.0, 1.0)))
    return {
        "planet": name,
        "epoch_year": year,
        "z_scale_factor": Z_SCALE,
        "sample_count": len(jd),
        "perihelion_index": i_peri,
        "aphelion_index": i_aphe,
        "perihelion_km": float(radius[i_peri]),
        "aphelion_km": float(radius[i_aphe]),
        "inclination_deg": inclination,
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
    ax.plot_surface(sx, sy, sz, linewidth=0.0, antialiased=True, shade=True,
                    alpha=0.98, color="#FDB813", zorder=10)
    ax.text(0.0, 0.0, 4.0, "Sun", color="#FFD86B", fontsize=14, weight="bold")

    all_arrays: list[np.ndarray] = []
    for name in ("Mercury", "Venus", "Earth"):
        cfg = PLANETS[name]
        rows = [row for row in summaries if row["planet"] == name]
        for orbit, row in zip(orbits[name], rows):
            year = int(orbit["year"])
            xyz_plot = orbit["xyz"].copy() / 1.0e6
            xyz_plot[:, 2] *= Z_SCALE
            all_arrays.append(xyz_plot)
            linestyle = "--" if year < 0 else "-"
            alpha = 1.0 if year == 0 else 0.68
            label = f"{name}: epoch 0" if year == 0 else None
            ax.plot(
                xyz_plot[:, 0], xyz_plot[:, 1], xyz_plot[:, 2],
                color=cfg["color"], linewidth=ORBIT_LINEWIDTH,
                linestyle=linestyle, alpha=alpha, label=label, zorder=3,
            )
            for marker_name, key, marker in [
                ("P", "perihelion_index", "o"),
                ("A", "aphelion_index", "s"),
            ]:
                point = xyz_plot[int(row[key])]
                emphasized = year in (-9000, 0, 9000)
                ax.scatter(
                    [point[0]], [point[1]], [point[2]],
                    s=34.0 if emphasized else 10.0,
                    color=cfg["color"], edgecolors="white",
                    linewidths=0.55 if emphasized else 0.15,
                    marker=marker, alpha=alpha, depthshade=False, zorder=7,
                )
                if emphasized:
                    ax.text(
                        point[0], point[1], point[2],
                        f" {name} {marker_name} {year:+d}\n Z×1.10",
                        color=cfg["color"], fontsize=7.4, weight="bold", zorder=8,
                    )

    set_equal_3d(ax, all_arrays)
    ax.view_init(elev=24.0, azim=40.0)
    ax.set_title(
        "Mercury, Venus, and Earth — Centered JPL Orbit Bundle\n"
        "Astronomical years −9000 to +9000; uniform Z × 1.10",
        color="white", fontsize=20, weight="bold", pad=30,
    )
    ax.set_xlabel("Ecliptic X (million km, true scale)", color="#E0E0E0", labelpad=14)
    ax.set_ylabel("Ecliptic Y (million km, true scale)", color="#E0E0E0", labelpad=14)
    ax.set_zlabel("Ecliptic Z (million km, uniformly ×1.10)", color="#E0E0E0", labelpad=12)
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
        "Z factor = 1.10 for every epoch",
    ]
    fig.text(
        0.025, 0.025, "\n".join(info), color="white", fontsize=10.2,
        ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#050505", "edgecolor": "#858585", "alpha": 0.94},
    )
    fig.tight_layout()
    fig.savefig(PNG, dpi=420, facecolor="black", bbox_inches="tight")
    display(Image(filename=str(PNG)))
    plt.close(fig)


def main() -> None:
    print(f"OUTPUT VERSION {VERSION}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    orbits: dict[str, list[dict]] = {name: [] for name in PLANETS}
    summaries: list[dict] = []
    csv_rows: list[dict] = []

    for year in EPOCH_YEARS:
        center = epoch_center_jd(year)
        for name, cfg in PLANETS.items():
            jd, xyz = query_orbit(cfg["id"], center, cfg["period_days"])
            summary = orbit_summary(name, year, jd, xyz)
            orbits[name].append({"year": year, "jd": jd, "xyz": xyz})
            summaries.append(summary)
            for index in range(len(jd)):
                csv_rows.append({
                    "planet": name,
                    "epoch_year": year,
                    "z_scale_factor": Z_SCALE,
                    "sample_index": index,
                    "jd_tdb": jd[index],
                    "x_km": xyz[index, 0],
                    "y_km": xyz[index, 1],
                    "z_true_km": xyz[index, 2],
                    "z_plot_km": xyz[index, 2] * Z_SCALE,
                })

    pd.DataFrame(csv_rows).to_csv(CSV, index=False, float_format="%.9f")
    make_plot(orbits, summaries)
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0018