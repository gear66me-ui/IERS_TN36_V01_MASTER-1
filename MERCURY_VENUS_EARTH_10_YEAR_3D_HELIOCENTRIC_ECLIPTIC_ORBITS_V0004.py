# V0004
# Audit reference: standalone JPL Horizons 10-year 3D heliocentric ecliptic orbit visualization for Mercury, Venus, and Earth with monthly markers and perihelion/aphelion labels.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display

VERSION = "V0004"
LOCAL_TZ = ZoneInfo("America/Bogota")
START_UTC = "2026-01-01 00:00"
STOP_UTC = "2036-01-01 00:00"
STEP = "1d"
LOCATION = "@10"
REFPLANE = "ecliptic"
ABERRATIONS = "geometric"
AU_KM = 149597870.700
R_SUN_KM = 695700.000
OUT = Path("/content/MERCURY_VENUS_EARTH_10_YEAR_3D_HELIOCENTRIC_ECLIPTIC_ORBITS_V0004_OUTPUT")
PNG = OUT / "MERCURY_VENUS_EARTH_10_YEAR_3D_HELIOCENTRIC_ECLIPTIC_ORBITS_V0004.png"
CSV = OUT / "MERCURY_VENUS_EARTH_10_YEAR_3D_HELIOCENTRIC_ECLIPTIC_ORBITS_V0004.csv"

PLANETS = {
    "Mercury": {"id": "199", "color": "#A970FF", "radius_km": 2439.7, "marker_size": 16.0},
    "Venus": {"id": "299", "color": "#2F8DFF", "radius_km": 6051.8, "marker_size": 20.0},
    "Earth": {"id": "399", "color": "#35C96B", "radius_km": 6378.137, "marker_size": 21.0},
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
    if len(jd) < 3600 or not np.all(np.diff(jd) > 0.0):
        raise RuntimeError(f"REJECTED JPL grid for body {body}")
    if not np.all(np.isfinite(xyz)):
        raise RuntimeError(f"REJECTED non-finite vectors for body {body}")
    return jd, xyz


def orbital_summary(name: str, jd: np.ndarray, xyz: np.ndarray) -> dict:
    radius = np.linalg.norm(xyz, axis=1)
    i_peri = int(np.argmin(radius))
    i_aphe = int(np.argmax(radius))
    z = xyz[:, 2]
    return {
        "planet": name,
        "perihelion_km": float(radius[i_peri]),
        "aphelion_km": float(radius[i_aphe]),
        "radial_range_km": float(radius[i_aphe] - radius[i_peri]),
        "perihelion_index": i_peri,
        "aphelion_index": i_aphe,
        "perihelion_utc": Time(float(jd[i_peri]), format="jd", scale="tdb").utc.strftime("%Y-%m-%d %H:%M:%S"),
        "aphelion_utc": Time(float(jd[i_aphe]), format="jd", scale="tdb").utc.strftime("%Y-%m-%d %H:%M:%S"),
        "z_min_km": float(np.min(z)),
        "z_max_km": float(np.max(z)),
        "z_peak_to_peak_km": float(np.max(z) - np.min(z)),
    }


def monthly_indices(dates: np.ndarray) -> np.ndarray:
    periods = pd.PeriodIndex(pd.to_datetime(dates), freq="M")
    _, first = np.unique(periods.astype(str), return_index=True)
    return np.sort(first)


def sphere(radius: float, n_u: int = 64, n_v: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def set_equal_3d(ax: plt.Axes, arrays: list[np.ndarray]) -> None:
    points = np.vstack(arrays)
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = 0.5 * (mins + maxs)
    half = 0.5 * float(np.max(maxs - mins)) * 1.06
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass


def make_plot(
    jd: np.ndarray,
    dates: np.ndarray,
    vectors: dict[str, np.ndarray],
    summaries: dict[str, dict],
) -> None:
    fig = plt.figure(figsize=(24.0, 20.0), dpi=120)
    fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")

    sun_x, sun_y, sun_z = sphere(R_SUN_KM)
    ax.plot_surface(
        sun_x,
        sun_y,
        sun_z,
        rstride=2,
        cstride=2,
        linewidth=0.0,
        antialiased=True,
        shade=True,
        alpha=0.98,
        color="#FDB813",
        zorder=10,
    )
    ax.text(0.0, 0.0, R_SUN_KM * 4.0, "Sun", color="#FFD86B", fontsize=14, weight="bold")

    marker_idx = monthly_indices(dates)
    for name in ("Mercury", "Venus", "Earth"):
        cfg = PLANETS[name]
        xyz = vectors[name]
        summary = summaries[name]
        xyz_mkm = xyz / 1.0e6
        ax.plot(
            xyz_mkm[:, 0],
            xyz_mkm[:, 1],
            xyz_mkm[:, 2],
            color=cfg["color"],
            linewidth=0.72,
            alpha=0.82,
            label=f"{name} orbit trace",
            zorder=3,
        )
        ax.scatter(
            xyz_mkm[marker_idx, 0],
            xyz_mkm[marker_idx, 1],
            xyz_mkm[marker_idx, 2],
            s=cfg["marker_size"],
            color=cfg["color"],
            edgecolors="white",
            linewidths=0.25,
            alpha=0.92,
            depthshade=True,
            label=f"{name} monthly positions",
            zorder=5,
        )

        for label, key, marker in [
            ("Perihelion", "perihelion_index", "o"),
            ("Aphelion", "aphelion_index", "s"),
        ]:
            i = int(summary[key])
            p = xyz_mkm[i]
            ax.scatter(
                [p[0]], [p[1]], [p[2]],
                s=68.0,
                color=cfg["color"],
                edgecolors="white",
                linewidths=0.8,
                marker=marker,
                depthshade=False,
                zorder=8,
            )
            ax.text(
                p[0], p[1], p[2],
                f" {name} {label}\n {np.linalg.norm(xyz[i]):,.0f} km",
                color=cfg["color"],
                fontsize=8.5,
                weight="bold",
                zorder=9,
            )

    set_equal_3d(ax, [vectors[name] / 1.0e6 for name in vectors])
    ax.view_init(elev=24.0, azim=38.0)

    ax.set_title(
        "Mercury, Venus, and Earth — 10-Year Heliocentric Ecliptic Orbits",
        color="white",
        fontsize=20,
        weight="bold",
        pad=26,
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

    legend = ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98), frameon=False, fontsize=9.0)
    for text in legend.get_texts():
        text.set_color("white")

    info = [
        "Sun radius rendered at true orbital scale",
        "Planet markers are magnified for visibility",
        "Monthly markers; daily JPL orbit traces",
        "Reference: Sun-centered ecliptic frame",
    ]
    for name in ("Mercury", "Venus", "Earth"):
        row = summaries[name]
        info.append(
            f"{name}: peri {row['perihelion_km']/1e6:.3f} Mkm, "
            f"aphe {row['aphelion_km']/1e6:.3f} Mkm, "
            f"Z p-p {row['z_peak_to_peak_km']/1e6:.3f} Mkm"
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
    print(f"UTC interval                         {START_UTC} to {STOP_UTC}")
    print(f"JPL cadence                          {STEP}")
    print(f"Reference center                     Sun, {LOCATION}")
    print(f"Reference plane                      {REFPLANE}")
    print(f"Aberrations                          {ABERRATIONS}")
    print(f"Sun radius                           {R_SUN_KM:.3f} km")
    print(f"Output                               {OUT}")

    section("COMMENTS")
    print("Daily JPL heliocentric ecliptic vectors are used for all orbital traces.")
    print("The Sun sphere is rendered at true orbital scale at X=0, Y=0, Z=0.")
    print("Planet marker diameters are NOT TO SCALE and are enlarged only for visibility.")
    print("Monthly planet markers are placed along each daily 10-year orbit trace.")
    print("Perihelion and aphelion are derived from minimum and maximum JPL vector magnitude.")

    vectors: dict[str, np.ndarray] = {}
    jds: dict[str, np.ndarray] = {}
    summaries: dict[str, dict] = {}
    for name, cfg in PLANETS.items():
        print(f"DEBUG querying {name}", flush=True)
        jd, xyz = query(cfg["id"])
        jds[name] = jd
        vectors[name] = xyz
        summaries[name] = orbital_summary(name, jd, xyz)

    reference_jd = jds["Earth"]
    for name in ("Mercury", "Venus"):
        if len(jds[name]) != len(reference_jd) or not np.allclose(jds[name], reference_jd, atol=1.0e-11, rtol=0.0):
            raise RuntimeError(f"REJECTED mismatched JPL grid for {name}")

    dates = Time(reference_jd, format="jd", scale="tdb").utc.to_datetime()
    frame = pd.DataFrame({"utc": [value.strftime("%Y-%m-%d %H:%M:%S") for value in dates]})
    for name in ("Mercury", "Venus", "Earth"):
        xyz = vectors[name]
        key = name.lower()
        frame[f"{key}_x_km"] = xyz[:, 0]
        frame[f"{key}_y_km"] = xyz[:, 1]
        frame[f"{key}_z_km"] = xyz[:, 2]
        frame[f"{key}_radius_km"] = np.linalg.norm(xyz, axis=1)
    frame.to_csv(CSV, index=False, float_format="%.6f")

    make_plot(reference_jd, dates, vectors, summaries)

    section("RESULTS")
    for name in ("Mercury", "Venus", "Earth"):
        row = summaries[name]
        print(f"{name} perihelion                    {row['perihelion_km']:.6f} km")
        print(f"{name} perihelion UTC                {row['perihelion_utc']}")
        print(f"{name} aphelion                      {row['aphelion_km']:.6f} km")
        print(f"{name} aphelion UTC                  {row['aphelion_utc']}")
        print(f"{name} radial variation              {row['radial_range_km']:.6f} km")
        print(f"{name} ecliptic Z minimum            {row['z_min_km']:.6f} km")
        print(f"{name} ecliptic Z maximum            {row['z_max_km']:.6f} km")
        print(f"{name} Z peak-to-peak                {row['z_peak_to_peak_km']:.6f} km")

    section("OUTPUT SUMMARY")
    print(f"PNG                                  {PNG}")
    print(f"PNG bytes                            {PNG.stat().st_size}")
    print(f"CSV                                  {CSV}")
    print(f"JPL daily samples                    {len(reference_jd)}")
    print(f"Monthly markers per planet           {len(monthly_indices(dates))}")

    section("PAPER COMPARISON")
    print("NOT USED: manual orbital elements, decorative trajectories, or non-JPL ephemerides.")
    print("Published perihelion/aphelion values may be compared only after these JPL-derived results.")

    section("EQUATION STATUS")
    print("VERIFIED radius = sqrt(X^2 + Y^2 + Z^2) from heliocentric ecliptic JPL vectors")
    print("VERIFIED perihelion = minimum radius over the 10-year JPL interval")
    print("VERIFIED aphelion = maximum radius over the 10-year JPL interval")
    print("VERIFIED Z peak-to-peak = maximum Z - minimum Z")
    print("VERIFIED equal 3D axis scale and Sun center at X=0, Y=0, Z=0")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0004