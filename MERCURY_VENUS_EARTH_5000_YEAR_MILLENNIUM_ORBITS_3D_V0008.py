# V0008
# Audit reference: standalone JPL Horizons heliocentric ecliptic 3D orbit comparison at 1000-year epochs from 2026 through 7026.
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
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display

VERSION = "V0008"
LOCAL_TZ = ZoneInfo("America/Bogota")
LOCATION = "@10"
REFPLANE = "ecliptic"
ABERRATIONS = "geometric"
AU_KM = 149597870.700
R_SUN_KM = 695700.000
R_SUN_MKM = R_SUN_KM / 1.0e6
EPOCH_YEARS = [2026, 3026, 4026, 5026, 6026, 7026]
STEP = "1d"
OUT = Path("/content/MERCURY_VENUS_EARTH_5000_YEAR_MILLENNIUM_ORBITS_3D_V0008_OUTPUT")
PNG = OUT / "MERCURY_VENUS_EARTH_5000_YEAR_MILLENNIUM_ORBITS_3D_V0008.png"
CSV = OUT / "MERCURY_VENUS_EARTH_5000_YEAR_MILLENNIUM_ORBITS_3D_V0008.csv"

PLANETS = {
    "Mercury": {"id": "199", "color": "#A970FF", "days": 100},
    "Venus": {"id": "299", "color": "#2F8DFF", "days": 240},
    "Earth": {"id": "399", "color": "#35C96B", "days": 380},
}


def section(name: str) -> None:
    print(name)
    print("-" * len(name))


def query_orbit(body: str, year: int, span_days: int) -> tuple[np.ndarray, np.ndarray]:
    start = f"{year:04d}-01-01 00:00"
    stop = f"{year:04d}-01-{1 + span_days:02d} 00:00" if span_days < 28 else None
    if stop is None:
        from astropy.time import Time, TimeDelta
        t0 = Time(start, scale="utc")
        stop = (t0 + TimeDelta(span_days, format="jd")).strftime("%Y-%m-%d %H:%M")
    table = Horizons(
        id=body,
        id_type="majorbody",
        location=LOCATION,
        epochs={"start": start, "stop": stop, "step": STEP},
    ).vectors(refplane=REFPLANE, aberrations=ABERRATIONS)
    jd = np.asarray(table["datetime_jd"], dtype=float)
    xyz = np.column_stack([
        np.asarray(table["x"], dtype=float),
        np.asarray(table["y"], dtype=float),
        np.asarray(table["z"], dtype=float),
    ]) * AU_KM
    if len(jd) < 60 or not np.all(np.diff(jd) > 0.0):
        raise RuntimeError(f"REJECTED JPL grid body={body} year={year}")
    if not np.all(np.isfinite(xyz)):
        raise RuntimeError(f"REJECTED non-finite JPL vectors body={body} year={year}")
    return jd, xyz


def summarize(name: str, year: int, jd: np.ndarray, xyz: np.ndarray) -> dict:
    radius = np.linalg.norm(xyz, axis=1)
    i_peri = int(np.argmin(radius))
    i_aphe = int(np.argmax(radius))
    peri = xyz[i_peri]
    aphe = xyz[i_aphe]
    peri_lon = math.degrees(math.atan2(float(peri[1]), float(peri[0]))) % 360.0
    aphe_lon = math.degrees(math.atan2(float(aphe[1]), float(aphe[0]))) % 360.0
    return {
        "planet": name,
        "epoch_year": year,
        "perihelion_index": i_peri,
        "aphelion_index": i_aphe,
        "perihelion_km": float(radius[i_peri]),
        "aphelion_km": float(radius[i_aphe]),
        "perihelion_longitude_deg": peri_lon,
        "aphelion_longitude_deg": aphe_lon,
        "peri_x_km": float(peri[0]),
        "peri_y_km": float(peri[1]),
        "peri_z_km": float(peri[2]),
        "aphe_x_km": float(aphe[0]),
        "aphe_y_km": float(aphe[1]),
        "aphe_z_km": float(aphe[2]),
    }


def unwrap_degrees(values: list[float]) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(np.asarray(values, dtype=float))))


def sphere(radius: float, n_u: int = 72, n_v: int = 36) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def make_plot(orbits: dict[str, dict[int, np.ndarray]], summaries: list[dict]) -> None:
    fig = plt.figure(figsize=(26.0, 22.0), dpi=120)
    fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")

    sx, sy, sz = sphere(R_SUN_MKM)
    ax.plot_surface(sx, sy, sz, linewidth=0.0, antialiased=True, shade=True,
                    alpha=0.98, color="#FDB813", zorder=10)
    ax.text(0.0, 0.0, 3.5, "Sun", color="#FFD86B", fontsize=14, weight="bold")

    all_arrays: list[np.ndarray] = []
    n_epochs = len(EPOCH_YEARS)
    for name in ("Mercury", "Venus", "Earth"):
        cfg = PLANETS[name]
        planet_rows = [row for row in summaries if row["planet"] == name]
        for epoch_index, year in enumerate(EPOCH_YEARS):
            xyz_mkm = orbits[name][year] / 1.0e6
            all_arrays.append(xyz_mkm)
            alpha = 0.30 + 0.70 * epoch_index / max(1, n_epochs - 1)
            width = 0.55 + 0.55 * epoch_index / max(1, n_epochs - 1)
            label = f"{name} {year}" if epoch_index in (0, n_epochs - 1) else None
            ax.plot(xyz_mkm[:, 0], xyz_mkm[:, 1], xyz_mkm[:, 2],
                    color=cfg["color"], linewidth=width, alpha=alpha,
                    label=label, zorder=3)

            row = next(r for r in planet_rows if r["epoch_year"] == year)
            for key, marker in (("perihelion_index", "o"), ("aphelion_index", "s")):
                i = int(row[key])
                p = xyz_mkm[i]
                ax.scatter([p[0]], [p[1]], [p[2]], s=34.0,
                           color=cfg["color"], edgecolors="white",
                           linewidths=0.45, marker=marker,
                           depthshade=False, alpha=alpha, zorder=7)

            if epoch_index in (0, n_epochs - 1):
                peri_i = int(row["perihelion_index"])
                aphe_i = int(row["aphelion_index"])
                p_peri = xyz_mkm[peri_i]
                p_aphe = xyz_mkm[aphe_i]
                ax.text(p_peri[0], p_peri[1], p_peri[2],
                        f" {name} peri {year}", color=cfg["color"], fontsize=7.8)
                ax.text(p_aphe[0], p_aphe[1], p_aphe[2],
                        f" {name} aphe {year}", color=cfg["color"], fontsize=7.8)

    set_equal_3d(ax, all_arrays)
    ax.view_init(elev=27.0, azim=42.0)
    ax.set_title("Mercury, Venus, and Earth — Millennium-Spaced Heliocentric Ecliptic Orbits",
                 color="white", fontsize=20, weight="bold", pad=28)
    ax.set_xlabel("Ecliptic X (million km)", color="#E0E0E0", labelpad=14)
    ax.set_ylabel("Ecliptic Y (million km)", color="#E0E0E0", labelpad=14)
    ax.set_zlabel("Ecliptic Z (million km)", color="#E0E0E0", labelpad=12)
    ax.tick_params(colors="#D7D7D7", labelsize=8)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.0, 0.0, 0.0, 1.0))
        axis.pane.set_edgecolor((0.38, 0.38, 0.38, 0.65))
        axis._axinfo["grid"]["color"] = (0.42, 0.42, 0.42, 0.18)
        axis._axinfo["grid"]["linewidth"] = 0.42

    legend = ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98),
                       frameon=False, fontsize=8.8)
    for text in legend.get_texts():
        text.set_color("white")

    fig.text(
        0.025, 0.025,
        "Epochs: 2026, 3026, 4026, 5026, 6026, 7026\n"
        "Circle = perihelion; square = aphelion\n"
        "Thin-to-bright traces progress forward in time\n"
        "JPL heliocentric ecliptic vectors; Sun at (0,0,0)",
        color="white", fontsize=10.0, ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#050505",
              "edgecolor": "#858585", "alpha": 0.94},
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
    print(f"Epoch spacing                        1000 years")
    print(f"Reference center                     Sun, {LOCATION}")
    print(f"Reference plane                      {REFPLANE}")
    print(f"JPL cadence                          {STEP}")
    print(f"Output                               {OUT}")

    section("COMMENTS")
    print("One complete JPL orbit is queried for each planet at each millennium epoch.")
    print("Only orbit traces, perihelion markers, and aphelion markers are plotted.")
    print("No monthly markers and no radial rescaling are used.")
    print("The purpose is to reveal long-term apsidal and orbital-plane rotation.")

    orbits: dict[str, dict[int, np.ndarray]] = {name: {} for name in PLANETS}
    rows: list[dict] = []
    for name, cfg in PLANETS.items():
        for year in EPOCH_YEARS:
            print(f"DEBUG querying {name} epoch {year}", flush=True)
            jd, xyz = query_orbit(cfg["id"], year, cfg["days"])
            orbits[name][year] = xyz
            rows.append(summarize(name, year, jd, xyz))

    frame = pd.DataFrame(rows)
    for name in ("Mercury", "Venus", "Earth"):
        mask = frame["planet"] == name
        unwrapped = unwrap_degrees(frame.loc[mask, "perihelion_longitude_deg"].tolist())
        frame.loc[mask, "perihelion_rotation_from_2026_deg"] = unwrapped - unwrapped[0]
    frame.to_csv(CSV, index=False, float_format="%.9f")

    make_plot(orbits, rows)

    section("RESULTS")
    for name in ("Mercury", "Venus", "Earth"):
        subset = frame[frame["planet"] == name].copy()
        for _, row in subset.iterrows():
            print(
                f"{name:8s} {int(row['epoch_year'])}  "
                f"peri {row['perihelion_km']:.3f} km  "
                f"aphe {row['aphelion_km']:.3f} km  "
                f"peri longitude {row['perihelion_longitude_deg']:.6f} deg  "
                f"rotation {row['perihelion_rotation_from_2026_deg']:.6f} deg"
            )

    section("OUTPUT SUMMARY")
    print(f"PNG                                  {PNG}")
    print(f"CSV                                  {CSV}")
    print(f"Planet-epoch orbit sets              {len(rows)}")

    section("PAPER COMPARISON")
    print("NOT USED: published orbital elements or manually rotated ellipses.")
    print("All orbit traces and apsidal directions are derived from JPL vectors.")

    section("EQUATION STATUS")
    print("VERIFIED perihelion = minimum heliocentric vector magnitude within each complete orbit")
    print("VERIFIED aphelion = maximum heliocentric vector magnitude within each complete orbit")
    print("VERIFIED perihelion longitude = atan2(Yperi, Xperi) in the ecliptic plane")
    print("VERIFIED rotation values are unwrapped relative to the 2026 perihelion direction")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0008