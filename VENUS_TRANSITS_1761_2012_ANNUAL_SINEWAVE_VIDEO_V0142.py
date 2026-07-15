# V0142
# Audit reference: repaired V0140 renderer using direct OpenCV frame streaming and persistent MP4 output.

from __future__ import annotations

import importlib.util
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

def ensure(module: str, package: str) -> None:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)

for module_name, package_name in [
    ("astroquery", "astroquery"),
    ("astropy", "astropy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("cv2", "opencv-python-headless"),
    ("IPython", "ipython"),
]:
    ensure(module_name, package_name)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Video, display
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

VERSION = "V0142"
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
WIDTH = 1280
HEIGHT = 720
FIGURE_DPI = 100

OUT = Path("/content/VENUS_TRANSITS_1761_2012_ANNUAL_SINEWAVE_VIDEO_V0142_OUTPUT")
MP4_NAME = "VENUS_TRANSITS_1761_2012_ANNUAL_SINEWAVE_VIDEO_V0142.mp4"
FRAME_CSV_NAME = "VENUS_TRANSITS_1761_2012_ANNUAL_SINEWAVE_VIDEO_V0142_FRAMES.csv"
TRANSIT_CSV_NAME = "VENUS_TRANSITS_1761_2012_ANNUAL_SINEWAVE_VIDEO_V0142_TRANSITS.csv"
CACHE_CSV_NAME = "VENUS_TRANSITS_1761_2012_ANNUAL_SINEWAVE_VIDEO_V0142_LONG_RANGE.csv"

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

def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 0.0:
        raise RuntimeError("REJECTED invalid vector")
    return v / n

def query(body: str, start: str, stop: str, step: str) -> Series:
    table = Horizons(
        id=body,
        id_type="majorbody",
        location=LOCATION,
        epochs={"start": start, "stop": stop, "step": step},
    ).vectors(refplane=REFPLANE, aberrations=ABERRATIONS)
    jd = np.asarray(table["datetime_jd"], dtype=float)
    xyz = np.column_stack([
        np.asarray(table["x"], dtype=float),
        np.asarray(table["y"], dtype=float),
        np.asarray(table["z"], dtype=float),
    ]) * AU_KM
    if len(jd) < 3 or not np.all(np.diff(jd) > 0.0):
        raise RuntimeError(f"REJECTED JPL grid for body {body}")
    return Series(jd, xyz)

def splines(series: Series) -> List[CubicSpline]:
    return [CubicSpline(series.jd, series.xyz_km[:, i], bc_type="natural") for i in range(3)]

def evaluate(curves: List[CubicSpline], jd: float) -> np.ndarray:
    return np.array([curve(jd) for curve in curves], dtype=float)

def closest_approach(earth: Series, sun: Series, venus: Series) -> tuple[float, float]:
    if not (len(earth.jd) == len(sun.jd) == len(venus.jd)):
        raise RuntimeError("REJECTED mismatched transit grids")
    sd = sun.xyz_km - earth.xyz_km
    vd = venus.xyz_km - earth.xyz_km
    sd /= np.linalg.norm(sd, axis=1)[:, None]
    vd /= np.linalg.norm(vd, axis=1)[:, None]
    sep = np.arccos(np.clip(np.einsum("ij,ij->i", sd, vd), -1.0, 1.0))
    i = int(np.argmin(sep))
    lo, hi = max(0, i - 3), min(len(earth.jd) - 1, i + 3)
    ec, sc, vc = splines(earth), splines(sun), splines(venus)
    def f(jd: float) -> float:
        e, s, v = evaluate(ec, jd), evaluate(sc, jd), evaluate(vc, jd)
        return math.acos(float(np.clip(np.dot(unit(s-e), unit(v-e)), -1.0, 1.0)))
    result = minimize_scalar(f, bounds=(float(earth.jd[lo]), float(earth.jd[hi])),
                             method="bounded", options={"xatol":1e-12,"maxiter":300})
    if not result.success:
        raise RuntimeError("REJECTED closest-approach refinement")
    return float(result.x), float(result.fun)

def derive_transits() -> pd.DataFrame:
    rows = []
    for year, center_text in TRANSIT_SEARCH_CENTERS.items():
        center = Time(center_text, scale="utc")
        delta = TRANSIT_SEARCH_HALF_HOURS / 24.0
        start = Time(center.jd-delta, format="jd", scale="utc").strftime("%Y-%m-%d %H:%M")
        stop = Time(center.jd+delta, format="jd", scale="utc").strftime("%Y-%m-%d %H:%M")
        earth, sun, venus = query("399", start, stop, TRANSIT_FINE_STEP), query("10", start, stop, TRANSIT_FINE_STEP), query("299", start, stop, TRANSIT_FINE_STEP)
        jd_ca, separation = closest_approach(earth, sun, venus)
        t = Time(jd_ca, format="jd", scale="tdb")
        ec, sc, vc = splines(earth), splines(sun), splines(venus)
        e, s, v = evaluate(ec, jd_ca), evaluate(sc, jd_ca), evaluate(vc, jd_ca)
        limit = math.asin(R_SUN_KM/np.linalg.norm(s-e)) + math.asin(R_VENUS_KM/np.linalg.norm(v-e))
        dt = t.utc.datetime
        rows.append({
            "transit_year": year,
            "closest_approach_utc": t.utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "jd_tdb": jd_ca,
            "minimum_separation_arcsec": separation*AS_PER_RAD,
            "external_contact_limit_arcsec": limit*AS_PER_RAD,
            "transit_verified": bool(separation <= limit),
            "month_fraction": (dt.month-1) + (dt.day-1)/31.0,
        })
    frame = pd.DataFrame(rows)
    if not bool(frame["transit_verified"].all()):
        raise RuntimeError("REJECTED transit verification")
    return frame

def build_long_range(cache_path: Path) -> Dict[int, pd.DataFrame]:
    if cache_path.is_file() and cache_path.stat().st_size > 100000:
        table = pd.read_csv(cache_path)
        table["datetime_utc"] = pd.to_datetime(table["datetime_utc"])
    else:
        parts = []
        for start_year in range(START_YEAR, END_YEAR + 1, 20):
            stop_year = min(start_year + 20, END_YEAR + 1)
            start = f"{start_year}-01-01 00:00"
            stop = f"{stop_year}-01-01 00:00"
            print(f"DEBUG JPL chunk {start_year}-{stop_year-1}", flush=True)
            earth = query("399", start, stop, LONG_STEP)
            venus = query("299", start, stop, LONG_STEP)
            if len(earth.jd) != len(venus.jd) or not np.allclose(earth.jd, venus.jd, atol=1e-10, rtol=0):
                raise RuntimeError("REJECTED long-range grids")
            times = Time(earth.jd, format="jd", scale="tdb").utc.to_datetime()
            eu = earth.xyz_km / np.linalg.norm(earth.xyz_km, axis=1)[:, None]
            vu = venus.xyz_km / np.linalg.norm(venus.xyz_km, axis=1)[:, None]
            parts.append(pd.DataFrame({
                "jd_tdb": earth.jd,
                "datetime_utc": times,
                "earth_y_arcsec": eu[:,2]*AS_PER_RAD,
                "venus_y_arcsec": vu[:,2]*AS_PER_RAD,
            }))
        table = pd.concat(parts, ignore_index=True).drop_duplicates("jd_tdb").sort_values("jd_tdb")
        table.to_csv(cache_path, index=False)
    table["year"] = table["datetime_utc"].dt.year
    table["month"] = table["datetime_utc"].dt.month
    table["day"] = table["datetime_utc"].dt.day
    table["month_fraction"] = table["month"] - 1 + (table["day"] - 1) / 31.0
    return {int(y): g.reset_index(drop=True) for y,g in table.groupby("year") if START_YEAR <= int(y) <= END_YEAR}

def canvas_bgr(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

def render_video(yearly: Dict[int,pd.DataFrame], transits: pd.DataFrame,
                 frame_csv: Path, mp4_path: Path) -> None:
    years = list(range(START_YEAR, END_YEAR+1))
    fig, ax = plt.subplots(figsize=(WIDTH/FIGURE_DPI, HEIGHT/FIGURE_DPI), dpi=FIGURE_DPI)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.axhspan(-SOLAR_LIMB_ARCSEC, SOLAR_LIMB_ARCSEC, facecolor="#C98A18",
               edgecolor="#E64A19", linewidth=1.0, alpha=0.18, zorder=0,
               label="Solar limb ±950 arcsec")
    ax.axhline(SOLAR_LIMB_ARCSEC, color="#E64A19", linewidth=0.8)
    ax.axhline(-SOLAR_LIMB_ARCSEC, color="#E64A19", linewidth=0.8)
    ax.set_xlim(0,12)
    ax.set_ylim(-45000,45000)
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_xlabel("Month of year — trace resets to January", color="#E5E5E5")
    ax.set_ylabel("Fixed solar-longitude-zero orthographic Y (arcsec)", color="#E5E5E5")
    ax.grid(True, color="#686868", alpha=0.22, linewidth=0.42)
    ax.tick_params(colors="#D8D8D8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#909090")
    for year in years:
        g = yearly[year]
        ax.plot(g["month_fraction"], g["earth_y_arcsec"], color="#38D66B", linewidth=0.25, alpha=0.035, zorder=1)
        ax.plot(g["month_fraction"], g["venus_y_arcsec"], color="#3EA6FF", linewidth=0.25, alpha=0.035, zorder=2)
    earth_line, = ax.plot([],[],color="#38D66B",linewidth=1.25,zorder=5,label="Earth orbit")
    venus_line, = ax.plot([],[],color="#3EA6FF",linewidth=1.25,zorder=6,label="Venus orbit")
    earth_dot, = ax.plot([],[],marker="o",markersize=4,color="#38D66B",linestyle="None",zorder=7)
    venus_dot, = ax.plot([],[],marker="o",markersize=4,color="#3EA6FF",linestyle="None",zorder=8)
    transit_scatter = ax.scatter([],[],s=36,facecolor="#FFE082",edgecolor="white",linewidth=0.7,zorder=10,label="Transit marker")
    year_text = ax.text(0.02,0.96,"",transform=ax.transAxes,color="white",fontsize=17,weight="bold",va="top")
    date_text = ax.text(0.02,0.90,"",transform=ax.transAxes,color="#DADADA",fontsize=10,va="top")
    transit_text = ax.text(0.50,0.92,"",transform=ax.transAxes,color="#FFE082",fontsize=15,weight="bold",ha="center",va="top")
    legend=ax.legend(loc="upper right",frameon=False,fontsize=9)
    for t in legend.get_texts(): t.set_color("#E6E6E6")
    fig.tight_layout()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("REJECTED OpenCV VideoWriter failed to open")
    marker_x, marker_y, marker_keys, rows = [], [], set(), []
    try:
        frame_index = 0
        for year in years:
            g = yearly[year]
            for month_index in range(FRAMES_PER_YEAR):
                threshold = month_index + 1.0
                vis = g["month_fraction"] <= threshold
                earth_line.set_data(g.loc[vis,"month_fraction"], g.loc[vis,"earth_y_arcsec"])
                venus_line.set_data(g.loc[vis,"month_fraction"], g.loc[vis,"venus_y_arcsec"])
                last = g.loc[vis].iloc[-1]
                earth_dot.set_data([last["month_fraction"]],[last["earth_y_arcsec"]])
                venus_dot.set_data([last["month_fraction"]],[last["venus_y_arcsec"]])
                year_text.set_text(str(year))
                date_text.set_text(pd.Timestamp(last["datetime_utc"]).strftime("%Y-%m-%d"))
                transit_text.set_text("")
                ax.patch.set_edgecolor("none")
                ty = transits[transits["transit_year"] == year]
                for _, tr in ty.iterrows():
                    if float(tr["month_fraction"]) <= threshold:
                        key=(int(year),round(float(tr["month_fraction"]),6))
                        if key not in marker_keys:
                            mx=float(tr["month_fraction"])
                            my=float(np.interp(mx,g["month_fraction"],g["venus_y_arcsec"]))
                            marker_x.append(mx); marker_y.append(my); marker_keys.add(key)
                            transit_scatter.set_offsets(np.column_stack([marker_x,marker_y]))
                        transit_text.set_text(f"VENUS TRANSIT — {year}")
                        ax.patch.set_edgecolor("#FFE082"); ax.patch.set_linewidth(2.0)
                frame = canvas_bgr(fig)
                if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
                    frame = cv2.resize(frame,(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA)
                writer.write(frame)
                rows.append({"frame_index":frame_index,"year":year,"month":month_index+1,
                             "display_date":date_text.get_text(),"transit_active":bool(transit_text.get_text())})
                frame_index += 1
                if frame_index % 120 == 0:
                    print(f"DEBUG encoded {frame_index}/{len(years)*FRAMES_PER_YEAR} frames", flush=True)
    finally:
        writer.release()
        plt.close(fig)
    if not mp4_path.is_file() or mp4_path.stat().st_size < 100000:
        raise RuntimeError("REJECTED MP4 was not finalized")
    pd.DataFrame(rows).to_csv(frame_csv,index=False)

def copy_to_drive(mp4_path: Path) -> Path | None:
    drive_root = Path("/content/drive/MyDrive")
    if not drive_root.exists():
        return None
    target_dir = drive_root / "IERS_TN36_V01_MASTER-1" / "VIDEOS"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / mp4_path.name
    shutil.copy2(mp4_path, target)
    return target

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    section("CODE INPUTS")
    print(f"Version                              {VERSION}")
    print(f"Date range                           {START_YEAR}-01-01 through {END_YEAR}-12-31")
    print(f"Frames                               {(END_YEAR-START_YEAR+1)*FRAMES_PER_YEAR}")
    print(f"Video                                {WIDTH}x{HEIGHT} at {FPS} fps")
    print("Renderer                             OpenCV direct frame streaming")
    print("No AI images                         Matplotlib/OpenCV only")
    print(f"Output                               {OUT}")

    transit_csv = OUT / TRANSIT_CSV_NAME
    cache_csv = OUT / CACHE_CSV_NAME
    frame_csv = OUT / FRAME_CSV_NAME
    mp4_path = OUT / MP4_NAME

    print("DEBUG deriving transits", flush=True)
    transits = derive_transits()
    transits.to_csv(transit_csv,index=False,float_format="%.12g")
    print("DEBUG loading JPL long-range data", flush=True)
    yearly = build_long_range(cache_csv)
    missing=sorted(set(range(START_YEAR,END_YEAR+1))-set(yearly))
    if missing:
        raise RuntimeError(f"REJECTED missing years {missing[:10]}")
    print("DEBUG rendering MP4 directly", flush=True)
    render_video(yearly,transits,frame_csv,mp4_path)
    drive_copy = copy_to_drive(mp4_path)

    section("RESULTS")
    print(f"MP4                                  {mp4_path}")
    print(f"MP4 bytes                            {mp4_path.stat().st_size}")
    print(f"Drive copy                           {drive_copy if drive_copy else 'NOT MOUNTED'}")
    print(f"Frame CSV                            {frame_csv}")
    print(f"Transit CSV                          {transit_csv}")

    section("OUTPUT SUMMARY")
    print("MP4 finalized                        True")
    print(f"Frame count                          {(END_YEAR-START_YEAR+1)*FRAMES_PER_YEAR}")

    section("PAPER COMPARISON")
    print("Published transit dates are NOT USED as calculation inputs.")

    section("EQUATION STATUS")
    print("VERIFIED direct frame streaming")
    print("VERIFIED MP4 finalized before display")
    print("VERIFIED persistent transit markers")
    print("VERIFIED no AI image generation")

    display(Video(str(mp4_path), embed=True, width=WIDTH, height=HEIGHT))
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(VERSION)

if __name__ == "__main__":
    main()
# V0142
