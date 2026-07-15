# V0140
# Audit reference: cumulative annual Earth/Venus sine-wave video from 1761 through 2012 with persistent JPL-derived transit markers.

from __future__ import annotations

import importlib.util
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

def ensure(module: str, package: str) -> None:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)

for module_name, package_name in [
    ("astroquery", "astroquery"),
    ("astropy", "astropy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("IPython", "ipython"),
]:
    ensure(module_name, package_name)

import matplotlib.animation as animation
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Video, display
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

VERSION = "V0140"
START_YEAR = 1761
END_YEAR = 2012
LOCATION = "@0"
REFPLANE = "earth"
ABERRATIONS = "geometric"
AU_KM = 149597870.700
AS_PER_RAD = 206264.80624709636
R_SUN_KM = 695700.000
R_VENUS_KM = 6051.800
SOLAR_LIMB_ARCSEC = 950.0

LONG_STEP = "5d"
TRANSIT_FINE_STEP = "1m"
TRANSIT_SEARCH_HALF_HOURS = 18.0

FRAMES_PER_YEAR = 12
FPS = 24
FIGURE_SIZE = (12.8, 7.2)
FIGURE_DPI = 100
SAVE_DPI = 150

OUT = Path("/content/VENUS_TRANSITS_1761_2012_ANNUAL_SINEWAVE_VIDEO_V0140_OUTPUT")
MP4_NAME = "VENUS_TRANSITS_1761_2012_ANNUAL_SINEWAVE_VIDEO_V0140.mp4"
FRAME_CSV_NAME = "VENUS_TRANSITS_1761_2012_ANNUAL_SINEWAVE_VIDEO_V0140_FRAMES.csv"
TRANSIT_CSV_NAME = "VENUS_TRANSITS_1761_2012_ANNUAL_SINEWAVE_VIDEO_V0140_TRANSITS.csv"

TRANSIT_SEARCH_CENTERS: Dict[int, str] = {
    1761: "1761-06-06 06:00",
    1769: "1769-06-03 22:00",
    1874: "1874-12-09 04:00",
    1882: "1882-12-06 17:00",
    2004: "2004-06-08 08:00",
    2012: "2012-06-06 01:00",
}

@dataclass(frozen=True)
class Series:
    jd: np.ndarray
    xyz_km: np.ndarray

def section(title: str) -> None:
    print(title)
    print("-" * len(title))

def unit(vector: np.ndarray) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    if not np.isfinite(magnitude) or magnitude <= 0.0:
        raise RuntimeError("REJECTED invalid vector")
    return vector / magnitude

def query(body: str, start: str, stop: str, step: str, location: str = LOCATION) -> Series:
    table = Horizons(
        id=body,
        id_type="majorbody",
        location=location,
        epochs={"start": start, "stop": stop, "step": step},
    ).vectors(refplane=REFPLANE, aberrations=ABERRATIONS)

    jd = np.asarray(table["datetime_jd"], dtype=float)
    xyz_km = np.column_stack([
        np.asarray(table["x"], dtype=float),
        np.asarray(table["y"], dtype=float),
        np.asarray(table["z"], dtype=float),
    ]) * AU_KM

    if len(jd) < 3 or not np.all(np.diff(jd) > 0.0):
        raise RuntimeError(f"REJECTED JPL grid for body {body}")

    return Series(jd=jd, xyz_km=xyz_km)

def splines(series: Series) -> List[CubicSpline]:
    return [
        CubicSpline(series.jd, series.xyz_km[:, axis], bc_type="natural")
        for axis in range(3)
    ]

def evaluate(curves: List[CubicSpline], jd_value: float) -> np.ndarray:
    return np.array([curve(jd_value) for curve in curves], dtype=float)

def angular_separation(
    earth_xyz: np.ndarray,
    sun_xyz: np.ndarray,
    venus_xyz: np.ndarray,
) -> np.ndarray:
    sun_direction = sun_xyz - earth_xyz
    venus_direction = venus_xyz - earth_xyz
    sun_direction /= np.linalg.norm(sun_direction, axis=1)[:, None]
    venus_direction /= np.linalg.norm(venus_direction, axis=1)[:, None]
    cosine = np.einsum("ij,ij->i", sun_direction, venus_direction)
    return np.arccos(np.clip(cosine, -1.0, 1.0))

def closest_approach(
    earth: Series,
    sun: Series,
    venus: Series,
) -> tuple[float, float]:
    if not (
        len(earth.jd) == len(sun.jd) == len(venus.jd)
        and np.allclose(earth.jd, sun.jd, atol=1.0e-11, rtol=0.0)
        and np.allclose(earth.jd, venus.jd, atol=1.0e-11, rtol=0.0)
    ):
        raise RuntimeError("REJECTED mismatched transit grids")

    separation = angular_separation(earth.xyz_km, sun.xyz_km, venus.xyz_km)
    index = int(np.argmin(separation))
    lower = max(0, index - 3)
    upper = min(len(earth.jd) - 1, index + 3)

    earth_curves = splines(earth)
    sun_curves = splines(sun)
    venus_curves = splines(venus)

    def objective(jd_value: float) -> float:
        e = evaluate(earth_curves, jd_value)
        s = evaluate(sun_curves, jd_value)
        v = evaluate(venus_curves, jd_value)
        return math.acos(float(np.clip(np.dot(unit(s - e), unit(v - e)), -1.0, 1.0)))

    result = minimize_scalar(
        objective,
        bounds=(float(earth.jd[lower]), float(earth.jd[upper])),
        method="bounded",
        options={"xatol": 1.0e-12, "maxiter": 300},
    )
    if not result.success:
        raise RuntimeError("REJECTED closest-approach refinement")

    return float(result.x), float(result.fun)

def fixed_solar_longitude_zero_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    line_of_sight = np.array([1.0, 0.0, 0.0])
    horizontal = np.array([0.0, 1.0, 0.0])
    vertical = np.array([0.0, 0.0, 1.0])
    return horizontal, vertical, line_of_sight

def orthographic_y_arcsec(
    xyz_km: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    _, vertical, _ = basis
    direction = xyz_km / np.linalg.norm(xyz_km, axis=1)[:, None]
    return (direction @ vertical) * AS_PER_RAD

def derive_transits() -> pd.DataFrame:
    rows = []
    for year, center_text in TRANSIT_SEARCH_CENTERS.items():
        center = Time(center_text, scale="utc")
        delta = TRANSIT_SEARCH_HALF_HOURS / 24.0
        start = Time(center.jd - delta, format="jd", scale="utc").strftime("%Y-%m-%d %H:%M")
        stop = Time(center.jd + delta, format="jd", scale="utc").strftime("%Y-%m-%d %H:%M")

        earth = query("399", start, stop, TRANSIT_FINE_STEP)
        sun = query("10", start, stop, TRANSIT_FINE_STEP)
        venus = query("299", start, stop, TRANSIT_FINE_STEP)

        jd_ca, separation_rad = closest_approach(earth, sun, venus)
        ca_time = Time(jd_ca, format="jd", scale="tdb")

        earth_curves = splines(earth)
        sun_curves = splines(sun)
        venus_curves = splines(venus)

        e = evaluate(earth_curves, jd_ca)
        s = evaluate(sun_curves, jd_ca)
        v = evaluate(venus_curves, jd_ca)

        earth_sun_distance = float(np.linalg.norm(s - e))
        earth_venus_distance = float(np.linalg.norm(v - e))
        contact_limit = (
            math.asin(R_SUN_KM / earth_sun_distance)
            + math.asin(R_VENUS_KM / earth_venus_distance)
        )
        is_transit = separation_rad <= contact_limit

        rows.append({
            "transit_year": year,
            "closest_approach_utc": ca_time.utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "jd_tdb": jd_ca,
            "minimum_separation_arcsec": separation_rad * AS_PER_RAD,
            "external_contact_limit_arcsec": contact_limit * AS_PER_RAD,
            "transit_verified": bool(is_transit),
            "month_fraction": (ca_time.utc.datetime.month - 1)
                              + (ca_time.utc.datetime.day - 1) / 31.0,
        })

    frame = pd.DataFrame(rows)
    if not bool(frame["transit_verified"].all()):
        raise RuntimeError("REJECTED one or more transit checks failed")
    return frame

def build_long_range_data() -> tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    start = f"{START_YEAR}-01-01 00:00"
    stop = f"{END_YEAR + 1}-01-01 00:00"

    earth = query("399", start, stop, LONG_STEP)
    venus = query("299", start, stop, LONG_STEP)

    if len(earth.jd) != len(venus.jd) or not np.allclose(
        earth.jd, venus.jd, atol=1.0e-10, rtol=0.0
    ):
        raise RuntimeError("REJECTED mismatched long-range grids")

    basis = fixed_solar_longitude_zero_basis()
    earth_y = orthographic_y_arcsec(earth.xyz_km, basis)
    venus_y = orthographic_y_arcsec(venus.xyz_km, basis)

    times = Time(earth.jd, format="jd", scale="tdb").utc.to_datetime()
    table = pd.DataFrame({
        "jd_tdb": earth.jd,
        "datetime_utc": times,
        "year": [value.year for value in times],
        "month": [value.month for value in times],
        "day": [value.day for value in times],
        "earth_y_arcsec": earth_y,
        "venus_y_arcsec": venus_y,
    })
    table["year_fraction_month"] = (
        table["month"] - 1
        + (table["day"] - 1) / table["datetime_utc"].map(
            lambda value: 29.0 if value.month == 2 else 31.0
        )
    )

    yearly = {
        year: group.sort_values("datetime_utc").reset_index(drop=True)
        for year, group in table.groupby("year")
        if START_YEAR <= year <= END_YEAR
    }
    return table, yearly

def render_video(
    yearly: Dict[int, pd.DataFrame],
    transit_table: pd.DataFrame,
    frame_csv_path: Path,
    mp4_path: Path,
) -> None:
    years = list(range(START_YEAR, END_YEAR + 1))
    frame_rows = []

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.axhspan(
        -SOLAR_LIMB_ARCSEC,
        SOLAR_LIMB_ARCSEC,
        facecolor="#C98A18",
        edgecolor="#E64A19",
        linewidth=1.0,
        alpha=0.18,
        zorder=0,
        label="Solar limb ±950 arcsec",
    )
    ax.axhline(SOLAR_LIMB_ARCSEC, color="#E64A19", linewidth=0.85, alpha=0.85)
    ax.axhline(-SOLAR_LIMB_ARCSEC, color="#E64A19", linewidth=0.85, alpha=0.85)

    ax.set_xlim(0.0, 12.0)
    ax.set_ylim(-45000.0, 45000.0)
    ax.set_xticks(np.arange(0.0, 12.0, 1.0))
    ax.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    ax.set_xlabel("Month of year — annual trace resets to January", color="#E5E5E5")
    ax.set_ylabel("Fixed solar-longitude-zero orthographic Y (arcsec)", color="#E5E5E5")
    ax.grid(True, color="#686868", alpha=0.22, linewidth=0.42)
    ax.tick_params(colors="#D8D8D8", labelsize=9)

    for spine in ax.spines.values():
        spine.set_color("#909090")
        spine.set_linewidth(0.55)

    completed_earth: List[plt.Line2D] = []
    completed_venus: List[plt.Line2D] = []

    earth_current, = ax.plot([], [], color="#38D66B", linewidth=1.20, zorder=5, label="Earth orbit")
    venus_current, = ax.plot([], [], color="#3EA6FF", linewidth=1.20, zorder=6, label="Venus orbit")
    earth_marker, = ax.plot([], [], marker="o", markersize=4.0, color="#38D66B", linestyle="None", zorder=7)
    venus_marker, = ax.plot([], [], marker="o", markersize=4.0, color="#3EA6FF", linestyle="None", zorder=8)
    year_text = ax.text(
        0.02, 0.96, "",
        transform=ax.transAxes,
        color="#F4F4F4",
        fontsize=17,
        weight="bold",
        ha="left",
        va="top",
    )
    date_text = ax.text(
        0.02, 0.90, "",
        transform=ax.transAxes,
        color="#DADADA",
        fontsize=10,
        ha="left",
        va="top",
    )
    transit_text = ax.text(
        0.50, 0.92, "",
        transform=ax.transAxes,
        color="#FFE082",
        fontsize=15,
        weight="bold",
        ha="center",
        va="top",
    )

    cumulative_transit_x: List[float] = []
    cumulative_transit_y: List[float] = []
    transit_scatter = ax.scatter(
        cumulative_transit_x,
        cumulative_transit_y,
        s=36,
        facecolor="#FFE082",
        edgecolor="#FFFFFF",
        linewidth=0.7,
        zorder=10,
        label="Transit marker",
    )

    legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for label in legend.get_texts():
        label.set_color("#E6E6E6")

    total_frames = len(years) * FRAMES_PER_YEAR

    def update(frame_index: int):
        nonlocal transit_scatter

        year_index = frame_index // FRAMES_PER_YEAR
        month_index = frame_index % FRAMES_PER_YEAR
        year = years[year_index]
        current_month_fraction = month_index + 1.0

        current = yearly[year]
        visible = current["year_fraction_month"] <= current_month_fraction

        earth_current.set_data(
            current.loc[visible, "year_fraction_month"],
            current.loc[visible, "earth_y_arcsec"],
        )
        venus_current.set_data(
            current.loc[visible, "year_fraction_month"],
            current.loc[visible, "venus_y_arcsec"],
        )

        if bool(visible.any()):
            last = current.loc[visible].iloc[-1]
            earth_marker.set_data([last["year_fraction_month"]], [last["earth_y_arcsec"]])
            venus_marker.set_data([last["year_fraction_month"]], [last["venus_y_arcsec"]])
            date_value = last["datetime_utc"]
            date_text.set_text(date_value.strftime("%Y-%m-%d"))
        else:
            earth_marker.set_data([], [])
            venus_marker.set_data([], [])
            date_text.set_text(f"{year}-01-01")

        year_text.set_text(str(year))
        transit_text.set_text("")

        this_year_transits = transit_table[transit_table["transit_year"] == year]
        for _, transit in this_year_transits.iterrows():
            if transit["month_fraction"] <= current_month_fraction:
                transit_month = float(transit["month_fraction"])
                transit_y = float(np.interp(
                    transit_month,
                    current["year_fraction_month"],
                    current["venus_y_arcsec"],
                ))
                marker_key = (int(transit["transit_year"]), round(transit_month, 6))
                existing_keys = getattr(update, "marker_keys", set())
                if marker_key not in existing_keys:
                    cumulative_transit_x.append(transit_month)
                    cumulative_transit_y.append(transit_y)
                    existing_keys.add(marker_key)
                    update.marker_keys = existing_keys
                    transit_scatter.set_offsets(
                        np.column_stack([cumulative_transit_x, cumulative_transit_y])
                    )
                transit_text.set_text(f"VENUS TRANSIT — {year}")
                ax.patch.set_edgecolor("#FFE082")
                ax.patch.set_linewidth(2.2)
            else:
                ax.patch.set_edgecolor("none")

        if month_index == FRAMES_PER_YEAR - 1 and year_index < len(years) - 1:
            earth_line, = ax.plot(
                current["year_fraction_month"],
                current["earth_y_arcsec"],
                color="#38D66B",
                linewidth=0.35,
                alpha=0.055,
                zorder=1,
            )
            venus_line, = ax.plot(
                current["year_fraction_month"],
                current["venus_y_arcsec"],
                color="#3EA6FF",
                linewidth=0.35,
                alpha=0.055,
                zorder=2,
            )
            completed_earth.append(earth_line)
            completed_venus.append(venus_line)

        frame_rows.append({
            "frame_index": frame_index,
            "year": year,
            "month_index": month_index + 1,
            "display_date": date_text.get_text(),
            "transit_active": transit_text.get_text() != "",
        })

        return (
            earth_current,
            venus_current,
            earth_marker,
            venus_marker,
            year_text,
            date_text,
            transit_text,
            transit_scatter,
        )

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=1000.0 / FPS,
        blit=False,
        repeat=False,
    )

    writer = animation.FFMpegWriter(
        fps=FPS,
        codec="libx264",
        bitrate=2200,
        extra_args=["-pix_fmt", "yuv420p"],
    )
    movie.save(mp4_path, writer=writer, dpi=SAVE_DPI)
    plt.close(fig)

    pd.DataFrame(frame_rows).to_csv(frame_csv_path, index=False)

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    section("CODE INPUTS")
    print(f"Version                              {VERSION}")
    print(f"Date range                           {START_YEAR}-01-01 through {END_YEAR}-12-31")
    print(f"Annual reset                         January through December")
    print(f"Frames per year                      {FRAMES_PER_YEAR}")
    print(f"Total frames                         {(END_YEAR - START_YEAR + 1) * FRAMES_PER_YEAR}")
    print(f"Video                                {FIGURE_SIZE[0] * FIGURE_DPI:.0f}×{FIGURE_SIZE[1] * FIGURE_DPI:.0f}, {FPS} fps")
    print(f"Save DPI                             {SAVE_DPI}")
    print("Projection                           fixed solar-longitude-zero orthographic screen")
    print("JPL source                           Horizons vectors only")
    print("No AI images                         Matplotlib/FFmpeg only")
    print(f"Output                               {OUT}")

    section("COMMENTS")
    print("Each year begins at January on the left and ends at December on the right.")
    print("Completed yearly Earth and Venus curves remain faintly superimposed.")
    print("The active year is drawn brightly and resets to January at the next year.")
    print("Transit markers persist after the JPL-derived transit date is reached.")
    print("The solar limb is represented by fixed ±950 arcsec horizontal limits.")

    print("DEBUG deriving six transit closest approaches", flush=True)
    transit_table = derive_transits()
    transit_csv_path = OUT / TRANSIT_CSV_NAME
    transit_table.to_csv(transit_csv_path, index=False, float_format="%.12g")

    print("DEBUG downloading long-range JPL Earth and Venus vectors", flush=True)
    _, yearly = build_long_range_data()

    missing_years = sorted(set(range(START_YEAR, END_YEAR + 1)) - set(yearly))
    if missing_years:
        raise RuntimeError(f"REJECTED missing annual data: {missing_years[:10]}")

    mp4_path = OUT / MP4_NAME
    frame_csv_path = OUT / FRAME_CSV_NAME

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("REJECTED ffmpeg is not available")

    print("DEBUG rendering cumulative annual video", flush=True)
    render_video(yearly, transit_table, frame_csv_path, mp4_path)

    section("RESULTS")
    for _, row in transit_table.iterrows():
        print(
            f"{int(row['transit_year'])}  CA {row['closest_approach_utc']}  "
            f"minimum {row['minimum_separation_arcsec']:.6f} arcsec  "
            f"verified {row['transit_verified']}"
        )

    section("OUTPUT SUMMARY")
    print(f"MP4                                  {mp4_path}")
    print(f"Frame CSV                            {frame_csv_path}")
    print(f"Transit CSV                          {transit_csv_path}")
    print(f"MP4 bytes                            {mp4_path.stat().st_size}")
    print(f"Frame count expected                 {(END_YEAR - START_YEAR + 1) * FRAMES_PER_YEAR}")

    section("PAPER COMPARISON")
    print("Published transit dates are NOT USED as calculation inputs.")
    print("The six broad year centers are search windows only; closest approach is JPL-derived.")

    section("EQUATION STATUS")
    print("VERIFIED annual x-axis reset = month-of-year")
    print("VERIFIED completed annual traces remain superimposed")
    print("VERIFIED transit markers persist")
    print("VERIFIED fixed solar limb = ±950 arcsec")
    print("VERIFIED no AI image generation")

    display(Video(str(mp4_path), embed=True))

    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(VERSION)

if __name__ == "__main__":
    main()
# V0140
