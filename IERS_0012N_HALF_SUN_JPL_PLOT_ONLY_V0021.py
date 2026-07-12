# V0021
# Audit reference: Fresh JPL Horizons derivation of Vardo and Point Venus tracks with C1, C2, CA, C3, C4 and an IAU-1976-normalized half-Sun plot only.
from __future__ import annotations

import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0021"
LOCAL_TZ = ZoneInfo("America/Bogota")
ARCSEC_PER_RAD = 206_264.80624709636
JPL_AU_KM = 149_597_870.7
C_KM_S = 299_792.458
TAU_A_S = 499.004782
IAU1976_AU_KM = C_KM_S * TAU_A_S
SUN_RADIUS_KM = 695_700.0
VENUS_RADIUS_KM = 6_051.8
START = "1769-Jun-03 18:00"
STOP = "1769-Jun-04 04:00"
STEP = "1m"
EVENTS = ("C1", "C2", "CA", "C3", "C4")

VARDO = {
    "key": "VARDO",
    "label": "Vardo, Norway",
    "short": "Vardo",
    "lon": 31.1107,
    "lat": 70.3706,
    "elev_km": 0.0,
    "color": "#ffc861",
}
TAHITI = {
    "key": "TAHITI",
    "label": "Point Venus, Tahiti",
    "short": "Tahiti",
    "lon": -149.4947,
    "lat": -17.4958,
    "elev_km": 0.0,
    "color": "#5ee08a",
}
SITES = (VARDO, TAHITI)
OUTPUT_DIR = Path("/content/IERS_TN36_V01_MASTER_OUTPUT/V0021_HALF_SUN_PLOT_ONLY")
PNG = OUTPUT_DIR / "V0021_VARDO_TAHITI_IAU1976_HALF_SUN.png"


def ensure_package(import_name: str, pip_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pip_name])


for _name, _pip in (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("astroquery", "astroquery"),
    ("astropy", "astropy"),
):
    ensure_package(_name, _pip)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar
from astroquery.jplhorizons import Horizons
import astropy.units as u
from astropy.time import Time
from IPython.display import Image, display


def norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def unit(vector: np.ndarray) -> np.ndarray:
    magnitude = norm(vector)
    if magnitude == 0.0:
        raise RuntimeError("Zero-length vector.")
    return np.asarray(vector, dtype=float) / magnitude


def site_location(site: dict[str, object]) -> dict[str, u.Quantity]:
    return {
        "lon": float(site["lon"]) * u.deg,
        "lat": float(site["lat"]) * u.deg,
        "elevation": float(site["elev_km"]) * u.km,
    }


def get_vectors(target: str, location: object, prefix: str) -> pd.DataFrame:
    query = Horizons(
        id=target,
        location=location,
        epochs={"start": START, "stop": STOP, "step": STEP},
    )
    vectors = query.vectors().to_pandas()
    return pd.DataFrame(
        {
            "jd": vectors["datetime_jd"].astype(float),
            f"{prefix}_x": vectors["x"].astype(float) * JPL_AU_KM,
            f"{prefix}_y": vectors["y"].astype(float) * JPL_AU_KM,
            f"{prefix}_z": vectors["z"].astype(float) * JPL_AU_KM,
        }
    )


def merge_vectors() -> tuple[pd.DataFrame, pd.DataFrame]:
    geocenter = get_vectors("10", "500@399", "GEO_SUN")
    topocentric: pd.DataFrame | None = None
    for site in SITES:
        location = site_location(site)
        for target, suffix in (("10", "SUN"), ("299", "VENUS")):
            frame = get_vectors(target, location, f"{site['key']}_{suffix}")
            topocentric = frame if topocentric is None else topocentric.merge(frame, on="jd", how="inner")
    if topocentric is None or len(geocenter) < 500 or len(topocentric) < 500:
        raise RuntimeError("Incomplete JPL vector retrieval.")
    return geocenter, topocentric


def cache(frame: pd.DataFrame) -> dict[str, object]:
    jd = frame["jd"].to_numpy(dtype=float)
    result: dict[str, object] = {"jd": jd}
    for column in frame.columns:
        if column != "jd":
            result[column] = CubicSpline(jd, frame[column].to_numpy(dtype=float), bc_type="natural")
    return result


def vector_at(data: dict[str, object], prefix: str, jd: float) -> np.ndarray:
    return np.array(
        [
            float(data[f"{prefix}_x"](jd)),
            float(data[f"{prefix}_y"](jd)),
            float(data[f"{prefix}_z"](jd)),
        ],
        dtype=float,
    )


def angular_separation_arcsec(a: np.ndarray, b: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0))
    return math.acos(cosine) * ARCSEC_PER_RAD


def site_vector(data: dict[str, object], site: dict[str, object], body: str, jd: float) -> np.ndarray:
    return vector_at(data, f"{site['key']}_{body}", jd)


def apparent_radii(data: dict[str, object], site: dict[str, object], jd: float) -> tuple[float, float]:
    sun = site_vector(data, site, "SUN", jd)
    venus = site_vector(data, site, "VENUS", jd)
    return (
        math.atan2(SUN_RADIUS_KM, norm(sun)) * ARCSEC_PER_RAD,
        math.atan2(VENUS_RADIUS_KM, norm(venus)) * ARCSEC_PER_RAD,
    )


def contact_function(data: dict[str, object], site: dict[str, object], internal: bool, jd: float) -> float:
    sun = site_vector(data, site, "SUN", jd)
    venus = site_vector(data, site, "VENUS", jd)
    separation = angular_separation_arcsec(sun, venus)
    solar_radius, venus_radius = apparent_radii(data, site, jd)
    threshold = solar_radius - venus_radius if internal else solar_radius + venus_radius
    return separation - threshold


def roots(data: dict[str, object], site: dict[str, object], internal: bool) -> list[float]:
    jd = np.asarray(data["jd"], dtype=float)
    values = np.array([contact_function(data, site, internal, value) for value in jd], dtype=float)
    found: list[float] = []
    for index in range(len(jd) - 1):
        left = values[index]
        right = values[index + 1]
        if left == 0.0:
            found.append(float(jd[index]))
        elif np.isfinite(left) and np.isfinite(right) and left * right < 0.0:
            found.append(
                float(
                    brentq(
                        lambda trial: contact_function(data, site, internal, trial),
                        jd[index],
                        jd[index + 1],
                        xtol=1e-13,
                        rtol=1e-13,
                        maxiter=200,
                    )
                )
            )
    return sorted(found)


def closest_approach(data: dict[str, object], site: dict[str, object]) -> float:
    jd = np.asarray(data["jd"], dtype=float)
    separations = np.array(
        [
            angular_separation_arcsec(
                site_vector(data, site, "SUN", value),
                site_vector(data, site, "VENUS", value),
            )
            for value in jd
        ],
        dtype=float,
    )
    index = int(np.argmin(separations))
    left = jd[max(0, index - 3)]
    right = jd[min(len(jd) - 1, index + 3)]
    solution = minimize_scalar(
        lambda trial: angular_separation_arcsec(
            site_vector(data, site, "SUN", trial),
            site_vector(data, site, "VENUS", trial),
        ),
        bounds=(left, right),
        method="bounded",
        options={"xatol": 1e-13},
    )
    return float(solution.x)


def compute_events(data: dict[str, object], site: dict[str, object]) -> dict[str, float]:
    outer = roots(data, site, internal=False)
    inner = roots(data, site, internal=True)
    if len(outer) < 2 or len(inner) < 2:
        raise RuntimeError(f"Could not derive four contacts for {site['label']}.")
    events = {
        "C1": outer[0],
        "C2": inner[0],
        "CA": closest_approach(data, site),
        "C3": inner[-1],
        "C4": outer[-1],
    }
    ordered = [events[name] for name in EVENTS]
    if ordered != sorted(ordered):
        raise RuntimeError(f"Event chronology failed for {site['label']}.")
    return events


def basis(geo: dict[str, object], jd: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = unit(vector_at(geo, "GEO_SUN", jd))
    reference = np.array([0.0, 0.0, 1.0])
    if norm(np.cross(reference, normal)) < 1e-12:
        reference = np.array([1.0, 0.0, 0.0])
    xhat = unit(np.cross(reference, normal))
    yhat = unit(np.cross(normal, xhat))
    return normal, xhat, yhat


def screen_point(
    geo: dict[str, object],
    topo: dict[str, object],
    site: dict[str, object],
    jd: float,
    frame: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    normal, xhat, yhat = frame
    sun_geo = vector_at(geo, "GEO_SUN", jd)
    sun_topo = site_vector(topo, site, "SUN", jd)
    venus_topo = site_vector(topo, site, "VENUS", jd)
    observer_geo = sun_geo - sun_topo
    scale = float(np.dot(sun_geo - observer_geo, normal) / np.dot(venus_topo, normal))
    hit = observer_geo + scale * venus_topo
    offset = hit - sun_geo
    return np.array(
        [
            math.atan2(float(np.dot(offset, xhat)), IAU1976_AU_KM) * ARCSEC_PER_RAD,
            math.atan2(float(np.dot(offset, yhat)), IAU1976_AU_KM) * ARCSEC_PER_RAD,
        ],
        dtype=float,
    )


def normalized_venus_radius(
    geo: dict[str, object],
    topo: dict[str, object],
    site: dict[str, object],
    jd: float,
) -> float:
    actual_radius = apparent_radii(topo, site, jd)[1]
    actual_es = norm(vector_at(geo, "GEO_SUN", jd))
    return math.atan(math.tan(actual_radius / ARCSEC_PER_RAD) * actual_es / IAU1976_AU_KM) * ARCSEC_PER_RAD


def build_track(
    geo: dict[str, object],
    topo: dict[str, object],
    site: dict[str, object],
    events: dict[str, float],
    frame: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, object]:
    minute_jd = np.asarray(topo["jd"], dtype=float)
    selected = minute_jd[(minute_jd >= events["C1"]) & (minute_jd <= events["C4"])]
    all_jd = np.array(sorted(set(selected.tolist() + list(events.values()))), dtype=float)
    points = np.array([screen_point(geo, topo, site, value, frame) for value in all_jd], dtype=float)
    event_points = {name: screen_point(geo, topo, site, value, frame) for name, value in events.items()}
    event_radii = {name: normalized_venus_radius(geo, topo, site, value) for name, value in events.items()}
    return {
        "site": site,
        "jd": all_jd,
        "points": points,
        "events": events,
        "event_points": event_points,
        "event_radii": event_radii,
    }


def label_event(axes, point: np.ndarray, text: str, color: str, dx: float, dy: float) -> None:
    axes.annotate(
        text,
        xy=point,
        xytext=(point[0] + dx, point[1] + dy),
        textcoords="data",
        fontsize=5.5,
        color=color,
        arrowprops=dict(arrowstyle="-", lw=0.20, color=color),
    )


def make_plot(track_a: dict[str, object], track_b: dict[str, object]) -> None:
    solar_radius_arcsec = math.atan2(SUN_RADIUS_KM, IAU1976_AU_KM) * ARCSEC_PER_RAD
    figure, axes = plt.subplots(figsize=(9.6, 5.8), dpi=240)
    figure.patch.set_facecolor("#03080d")
    axes.set_facecolor("#03080d")

    solar_limb = Circle(
        (0.0, 0.0),
        solar_radius_arcsec,
        fill=False,
        lw=0.36,
        ec="#66e8ff",
        alpha=0.95,
    )
    axes.add_patch(solar_limb)
    axes.axhline(0.0, lw=0.18, color="#1d3d4a")
    axes.axvline(0.0, lw=0.18, color="#1d3d4a")

    plotted_events = 0
    for track in (track_a, track_b):
        site = track["site"]
        color = str(site["color"])
        points = np.asarray(track["points"], dtype=float)
        axes.plot(points[:, 0], points[:, 1], lw=0.30, color=color, label=str(site["label"]), zorder=3)
        axes.scatter(points[::6, 0], points[::6, 1], s=0.8, color=color, linewidths=0, zorder=4)

        for event in EVENTS:
            point = np.asarray(track["event_points"][event], dtype=float)
            radius = float(track["event_radii"][event])
            axes.add_patch(
                Circle(
                    (point[0], point[1]),
                    radius,
                    fill=False,
                    lw=0.28 if event == "CA" else 0.20,
                    ec=color,
                    alpha=0.95,
                    zorder=5,
                )
            )
            axes.scatter([point[0]], [point[1]], s=3.5, color=color, linewidths=0, zorder=6)
            plotted_events += 1

        if site["key"] == "VARDO":
            label_event(axes, track["event_points"]["C1"], "V C1", color, -55, 12)
            label_event(axes, track["event_points"]["C2"], "V C2", color, -45, 10)
            label_event(axes, track["event_points"]["CA"], "Vardo CA", color, 18, 44)
            label_event(axes, track["event_points"]["C3"], "V C3", color, 18, -10)
            label_event(axes, track["event_points"]["C4"], "V C4", color, 28, -13)
        else:
            label_event(axes, track["event_points"]["C1"], "T C1", color, -55, -16)
            label_event(axes, track["event_points"]["C2"], "T C2", color, -45, -13)
            label_event(axes, track["event_points"]["CA"], "Tahiti CA", color, 18, -44)
            label_event(axes, track["event_points"]["C3"], "T C3", color, 18, 12)
            label_event(axes, track["event_points"]["C4"], "T C4", color, 28, 15)

    if plotted_events != 10:
        raise RuntimeError(f"Expected 10 plotted event disks; found {plotted_events}.")

    combined = np.vstack((track_a["points"], track_b["points"]))
    median_y = float(np.median(combined[:, 1]))
    axes.set_xlim(-1.04 * solar_radius_arcsec, 1.04 * solar_radius_arcsec)
    if median_y >= 0.0:
        axes.set_ylim(-0.06 * solar_radius_arcsec, 1.06 * solar_radius_arcsec)
    else:
        axes.set_ylim(-1.06 * solar_radius_arcsec, 0.06 * solar_radius_arcsec)

    axes.set_aspect("equal", adjustable="box")
    axes.grid(True, color="#102630", linewidth=0.16, alpha=0.55)
    axes.tick_params(colors="#8fb4c1", labelsize=6.5, width=0.22, length=2)
    axes.set_xlabel("IAU-1976-normalized solar-screen X offset (arcsec)", color="#8fb4c1", fontsize=7.5)
    axes.set_ylabel("IAU-1976-normalized solar-screen Y offset (arcsec)", color="#8fb4c1", fontsize=7.5)
    axes.set_title(
        "1769 Venus Transit — IAU-1976 Half-Sun Track Plot\nVardo, Norway and Point Venus, Tahiti",
        color="#f8fdff",
        fontsize=9,
        pad=8,
    )
    legend = axes.legend(loc="lower right", fontsize=6.3, frameon=True)
    legend.get_frame().set_facecolor("#071016")
    legend.get_frame().set_edgecolor("#1e4f64")
    for text in legend.get_texts():
        text.set_color("#dff8ff")

    figure.text(
        0.5,
        0.016,
        f"Fresh JPL Horizons geometry; C1, C2, CA, C3, C4 plotted for both tracks; cτA = {IAU1976_AU_KM:.6f} km.",
        ha="center",
        fontsize=6.2,
        color="#8fb4c1",
    )
    figure.savefig(PNG, dpi=460, facecolor=figure.get_facecolor(), bbox_inches="tight", pad_inches=0.055)
    plt.show()
    plt.close(figure)
    if not PNG.is_file() or PNG.stat().st_size == 0:
        raise RuntimeError("PNG plot was not generated.")
    display(Image(filename=str(PNG)))


def utc(jd: float) -> str:
    return Time(jd, format="jd", scale="tdb").utc.iso


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    geocenter_frame, topocentric_frame = merge_vectors()
    geo = cache(geocenter_frame)
    topo = cache(topocentric_frame)

    events_vardo = compute_events(topo, VARDO)
    events_tahiti = compute_events(topo, TAHITI)
    reference_jd = 0.5 * (events_vardo["CA"] + events_tahiti["CA"])
    frame = basis(geo, reference_jd)

    track_vardo = build_track(geo, topo, VARDO, events_vardo, frame)
    track_tahiti = build_track(geo, topo, TAHITI, events_tahiti, frame)
    make_plot(track_vardo, track_tahiti)

    print("RESULTS")
    for site, events in ((VARDO, events_vardo), (TAHITI, events_tahiti)):
        print(str(site["short"]) + " | " + " | ".join(f"{name} {utc(events[name])}" for name in EVENTS))
    print(f"PNG | {PNG}")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# V0021
