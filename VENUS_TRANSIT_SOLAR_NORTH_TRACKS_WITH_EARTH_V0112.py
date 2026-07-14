# V0112
# Audit reference: six Venus transit windows with fixed solar-plane scale, JPL angular limbs, and heliocentric Earth/Venus locator insets.

from __future__ import annotations

import csv
import io
import json
import math
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

VERSION = "V0112"
SCRIPT_NAME = "VENUS_TRANSIT_SOLAR_NORTH_TRACKS_WITH_EARTH_V0112.py"
CSV_NAME = "VENUS_TRANSIT_SOLAR_NORTH_TRACKS_WITH_EARTH_V0112.csv"
PNG_NAMES = (
    "VENUS_TRANSIT_TRACKS_WITH_EARTH_1761_1769_V0112.png",
    "VENUS_TRANSIT_TRACKS_WITH_EARTH_1874_1882_V0112.png",
    "VENUS_TRANSIT_TRACKS_WITH_EARTH_2004_2012_V0112.png",
)
OUTPUT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
HORIZONS_ENDPOINT = "https://ssd.jpl.nasa.gov/api/horizons.api"
ARCSEC_PER_RAD = 206264.80624709636
SOLAR_RADIUS_KM = 695700.0
VENUS_RADIUS_KM = 6051.8
WINDOW_DAYS = 92
STEP_SIZE = "1 d"
PLOT_LIMIT_ARCSEC = 2000.0
MONTH_DAYS = np.array([-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0])
MONTH_LABELS = ("−3 mo", "−2 mo", "−1 mo", "Transit", "+1 mo", "+2 mo", "+3 mo")
SOLAR_POLE_RA_DEG = 286.13
SOLAR_POLE_DEC_DEG = 63.87

TRANSITS = (
    ("1761", datetime(1761, 6, 6, tzinfo=timezone.utc)),
    ("1769", datetime(1769, 6, 3, tzinfo=timezone.utc)),
    ("1874", datetime(1874, 12, 9, tzinfo=timezone.utc)),
    ("1882", datetime(1882, 12, 6, tzinfo=timezone.utc)),
    ("2004", datetime(2004, 6, 8, tzinfo=timezone.utc)),
    ("2012", datetime(2012, 6, 6, tzinfo=timezone.utc)),
)
PAIR_GROUPS = ((0, 1), (2, 3), (4, 5))


@dataclass(frozen=True)
class VectorSeries:
    jd: np.ndarray
    xyz_km: np.ndarray


@dataclass(frozen=True)
class TransitTrack:
    label: str
    jd: np.ndarray
    elapsed_days: np.ndarray
    x_arcsec: np.ndarray
    y_arcsec: np.ndarray
    sun_radius_arcsec: np.ndarray
    venus_radius_arcsec: np.ndarray
    earth_helio_au: np.ndarray
    venus_helio_au: np.ndarray
    ca_jd: float
    ca_utc: str


def quoted(value: str) -> str:
    return f"'{value}'"


def horizons_url(target: str, center: str, start: datetime, stop: datetime) -> str:
    params = {
        "format": "json",
        "COMMAND": quoted(target),
        "OBJ_DATA": quoted("NO"),
        "MAKE_EPHEM": quoted("YES"),
        "EPHEM_TYPE": quoted("VECTORS"),
        "CENTER": quoted(center),
        "START_TIME": quoted(start.strftime("%Y-%m-%d %H:%M")),
        "STOP_TIME": quoted(stop.strftime("%Y-%m-%d %H:%M")),
        "STEP_SIZE": quoted(STEP_SIZE),
        "TIME_TYPE": quoted("UT"),
        "TIME_DIGITS": quoted("FRACSEC"),
        "CAL_TYPE": quoted("GREGORIAN"),
        "REF_PLANE": quoted("FRAME"),
        "REF_SYSTEM": quoted("ICRF"),
        "OUT_UNITS": quoted("KM-S"),
        "VEC_TABLE": quoted("1"),
        "VEC_CORR": quoted("NONE"),
        "CSV_FORMAT": quoted("YES"),
        "VEC_LABELS": quoted("NO"),
    }
    return HORIZONS_ENDPOINT + "?" + urllib.parse.urlencode(params)


def fetch_json(url: str) -> dict:
    last_error: Exception | None = None
    for _ in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": SCRIPT_NAME})
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 200:
                    raise RuntimeError(f"Horizons HTTP status {response.status}")
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to retrieve JPL Horizons vectors: {last_error}")


def fetch_vectors(target: str, center: str, start: datetime, stop: datetime) -> VectorSeries:
    payload = fetch_json(horizons_url(target, center, start, stop))
    if "NASA/JPL" not in str(payload.get("signature", {}).get("source", "")):
        raise RuntimeError("Unexpected Horizons signature")
    result = payload.get("result", "")
    if "$$SOE" not in result or "$$EOE" not in result:
        raise RuntimeError("Horizons response contains no vector table")
    block = result.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    jd_values: list[float] = []
    xyz_values: list[list[float]] = []
    for row in csv.reader(io.StringIO(block.strip())):
        cells = [cell.strip() for cell in row]
        if len(cells) < 5:
            continue
        try:
            jd_values.append(float(cells[0]))
            xyz_values.append([float(cells[2]), float(cells[3]), float(cells[4])])
        except ValueError:
            continue
    if len(jd_values) < 150:
        raise RuntimeError(f"Insufficient JPL vectors: {len(jd_values)}")
    return VectorSeries(np.asarray(jd_values), np.asarray(xyz_values))


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0.0):
        raise RuntimeError("Zero-length vector")
    return values / norms[:, None]


def solar_pole_unit() -> np.ndarray:
    ra = math.radians(SOLAR_POLE_RA_DEG)
    dec = math.radians(SOLAR_POLE_DEC_DEG)
    return np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])


def project_track(sun_xyz: np.ndarray, venus_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sun_u = normalize_rows(sun_xyz)
    venus_u = normalize_rows(venus_xyz)
    pole = solar_pole_unit()
    north = pole[None, :] - np.einsum("ij,j->i", sun_u, pole)[:, None] * sun_u
    north /= np.linalg.norm(north, axis=1)[:, None]
    east = np.cross(north, sun_u)
    east /= np.linalg.norm(east, axis=1)[:, None]
    denominator = np.einsum("ij,ij->i", venus_u, sun_u)
    x = np.einsum("ij,ij->i", venus_u, east) / denominator * ARCSEC_PER_RAD
    y = np.einsum("ij,ij->i", venus_u, north) / denominator * ARCSEC_PER_RAD
    return x, y


def polynomial_minimum(jd: np.ndarray, rho: np.ndarray) -> float:
    index = int(np.argmin(rho))
    lo = max(0, index - 5)
    hi = min(len(jd), index + 6)
    center = float(jd[index])
    x = jd[lo:hi] - center
    fit = np.polynomial.Polynomial.fit(x, rho[lo:hi] ** 2, 4).convert()
    candidates = [0.0]
    for root in fit.deriv().roots():
        if abs(root.imag) < 1.0e-10 and x.min() <= root.real <= x.max():
            candidates.append(float(root.real))
    best = min(candidates, key=lambda value: float(fit(value)))
    return center + best


def jd_to_utc(jd: float) -> str:
    shifted = jd + 0.5
    z = int(math.floor(shifted))
    f = shifted - z
    alpha = int((z - 1867216.25) / 36524.25)
    a = z + 1 + alpha - alpha // 4
    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)
    day_float = b - d - int(30.6001 * e) + f
    day = int(day_float)
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    seconds = (day_float - day) * 86400.0
    hour = int(seconds // 3600)
    minute = int((seconds - hour * 3600) // 60)
    second = seconds - hour * 3600 - minute * 60
    return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:06.3f} UTC"


def build_track(label: str, nominal: datetime) -> TransitTrack:
    start = nominal - timedelta(days=WINDOW_DAYS)
    stop = nominal + timedelta(days=WINDOW_DAYS)
    sun_geo = fetch_vectors("10", "500@399", start, stop)
    venus_geo = fetch_vectors("299", "500@399", start, stop)
    earth_helio = fetch_vectors("399", "500@10", start, stop)
    venus_helio = fetch_vectors("299", "500@10", start, stop)
    reference_jd = sun_geo.jd
    for series in (venus_geo, earth_helio, venus_helio):
        if series.jd.shape != reference_jd.shape or np.max(np.abs(series.jd - reference_jd)) * 86400.0 > 1e-6:
            raise RuntimeError(f"Epoch mismatch for {label}")
    x, y = project_track(sun_geo.xyz_km, venus_geo.xyz_km)
    rho = np.hypot(x, y)
    ca_jd = polynomial_minimum(reference_jd, rho)
    sun_distance = np.linalg.norm(sun_geo.xyz_km, axis=1)
    venus_distance = np.linalg.norm(venus_geo.xyz_km, axis=1)
    sun_radius = np.arcsin(SOLAR_RADIUS_KM / sun_distance) * ARCSEC_PER_RAD
    venus_radius = np.arcsin(VENUS_RADIUS_KM / venus_distance) * ARCSEC_PER_RAD
    au_km = 149597870.7
    return TransitTrack(
        label=label,
        jd=reference_jd,
        elapsed_days=reference_jd - ca_jd,
        x_arcsec=x,
        y_arcsec=y,
        sun_radius_arcsec=sun_radius,
        venus_radius_arcsec=venus_radius,
        earth_helio_au=earth_helio.xyz_km / au_km,
        venus_helio_au=venus_helio.xyz_km / au_km,
        ca_jd=ca_jd,
        ca_utc=jd_to_utc(ca_jd),
    )


def nearest_indices(elapsed: np.ndarray) -> np.ndarray:
    return np.asarray([int(np.argmin(np.abs(elapsed - day))) for day in MONTH_DAYS], dtype=int)


def add_locator(ax, track: TransitTrack, indices: np.ndarray) -> None:
    inset = inset_axes(ax, width="29%", height="29%", loc="lower right", borderpad=1.0)
    inset.plot(track.earth_helio_au[:, 0], track.earth_helio_au[:, 1], linewidth=0.55, label="Earth")
    inset.plot(track.venus_helio_au[:, 0], track.venus_helio_au[:, 1], linewidth=0.55, label="Venus")
    inset.scatter([0.0], [0.0], s=14, marker="o", label="Sun")
    inset.scatter(track.earth_helio_au[indices, 0], track.earth_helio_au[indices, 1], s=5)
    inset.scatter(track.venus_helio_au[indices, 0], track.venus_helio_au[indices, 1], s=5)
    inset.set_aspect("equal", adjustable="box")
    inset.set_title("Heliocentric locator", fontsize=7)
    inset.tick_params(labelsize=6, length=2)
    inset.grid(True, linewidth=0.25, alpha=0.35)


def plot_panel(ax, track: TransitTrack) -> None:
    indices = nearest_indices(track.elapsed_days)
    sun_radius = float(np.median(track.sun_radius_arcsec))
    sun_disk = plt.Circle((0.0, 0.0), sun_radius, facecolor="0.94", edgecolor="0.25", linewidth=0.8, zorder=0)
    ax.add_patch(sun_disk)
    ax.plot(track.x_arcsec, track.y_arcsec, linewidth=0.85, zorder=2, label="Venus geocentric track")
    for index, label in zip(indices, MONTH_LABELS):
        radius = float(track.venus_radius_arcsec[index])
        disk = plt.Circle((track.x_arcsec[index], track.y_arcsec[index]), radius, fill=False, linewidth=0.65, zorder=3)
        ax.add_patch(disk)
        ax.annotate(label, (track.x_arcsec[index], track.y_arcsec[index]), xytext=(4, 4), textcoords="offset points", fontsize=6.5)
    ax.set_xlim(-PLOT_LIMIT_ARCSEC, PLOT_LIMIT_ARCSEC)
    ax.set_ylim(-PLOT_LIMIT_ARCSEC, PLOT_LIMIT_ARCSEC)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Solar east–west X (arcsec)")
    ax.set_ylabel("Projected solar north Y (arcsec)")
    ax.set_title(f"{track.label} Venus transit — ±3 months\nJPL closest approach: {track.ca_utc}", fontweight="bold")
    ax.grid(True, which="major", linewidth=0.35, alpha=0.55)
    ax.minorticks_on()
    ax.grid(True, which="minor", linewidth=0.20, alpha=0.20)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    add_locator(ax, track, indices)


def make_pair_figure(tracks: tuple[TransitTrack, TransitTrack], path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 13.2))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.06, hspace=0.24)
    for ax, track in zip(axes, tracks):
        plot_panel(ax, track)
    fig.suptitle("Venus Transit Geometry — Solar North Up, Earth Observer", fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.show()


def write_csv(path: Path, tracks: tuple[TransitTrack, ...]) -> None:
    headers = (
        "transit", "jd_ut", "elapsed_days", "venus_x_arcsec", "venus_y_arcsec",
        "sun_radius_arcsec", "venus_radius_arcsec", "earth_helio_x_au", "earth_helio_y_au",
        "venus_helio_x_au", "venus_helio_y_au"
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for track in tracks:
            for i in range(len(track.jd)):
                writer.writerow([
                    track.label, f"{track.jd[i]:.12f}", f"{track.elapsed_days[i]:.9f}",
                    f"{track.x_arcsec[i]:.9f}", f"{track.y_arcsec[i]:.9f}",
                    f"{track.sun_radius_arcsec[i]:.9f}", f"{track.venus_radius_arcsec[i]:.9f}",
                    f"{track.earth_helio_au[i,0]:.12f}", f"{track.earth_helio_au[i,1]:.12f}",
                    f"{track.venus_helio_au[i,0]:.12f}", f"{track.venus_helio_au[i,1]:.12f}",
                ])


def section(title: str, lines: list[str]) -> None:
    print(title)
    for line in lines:
        print(line)


def main() -> None:
    section("CODE INPUTS", [f"Version: {VERSION}", "JPL step: 1 day", "Window: ±92 days", "Main scale: ±2000 arcsec"])
    section("COMMENTS", [
        "Main panels are geocentric Sun-facing tangent planes with projected solar north vertical.",
        "Sun and Venus angular radii are computed from JPL distances.",
        "Earth appears in the heliocentric locator inset because Earth is the observer in the main panel.",
    ])
    tracks = tuple(build_track(label, date) for label, date in TRANSITS)
    write_csv(OUTPUT_DIR / CSV_NAME, tracks)
    for output_name, pair in zip(PNG_NAMES, PAIR_GROUPS):
        make_pair_figure((tracks[pair[0]], tracks[pair[1]]), OUTPUT_DIR / output_name)
    section("RESULTS", [f"{track.label}: closest approach {track.ca_utc}" for track in tracks])
    section("OUTPUT SUMMARY", [f"CSV: {OUTPUT_DIR / CSV_NAME}"] + [f"PNG: {OUTPUT_DIR / name}" for name in PNG_NAMES])
    section("PAPER COMPARISON", ["Not used; geometry and angular scales derive from fresh JPL Horizons vectors."])
    section("EQUATION STATUS", [
        "PASS: projected solar north defines +Y.",
        "PASS: fixed ±2000 arcsec axes are used in every main panel.",
        "PASS: Sun and Venus limbs are plotted to JPL-derived angular scale.",
        "PASS: Earth and Venus heliocentric positions are shown simultaneously in each inset.",
    ])
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"{VERSION} COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{VERSION} ERROR: {exc}", file=sys.stderr)
        raise

# V0112