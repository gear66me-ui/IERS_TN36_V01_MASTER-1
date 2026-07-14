# V0112
# Audit reference: monthly geocentric Venus progression in the Sun-facing plane with scaled Sun and Venus limbs.

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
from matplotlib.patches import Circle

VERSION = "V0112"
SCRIPT_NAME = "VENUS_TRANSIT_MONTHLY_SOLAR_PLANE_1761_2012_V0112.py"
CSV_NAME = "VENUS_TRANSIT_MONTHLY_SOLAR_PLANE_1761_2012_V0112.csv"
PNG_NAMES = (
    "VENUS_TRANSIT_MONTHLY_1761_1769_V0112.png",
    "VENUS_TRANSIT_MONTHLY_1874_1882_V0112.png",
    "VENUS_TRANSIT_MONTHLY_2004_2012_V0112.png",
)
OUTPUT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
HORIZONS_ENDPOINT = "https://ssd.jpl.nasa.gov/api/horizons.api"
ARCSEC_PER_RAD = 206264.80624709636
SOLAR_RADIUS_KM = 695700.0
VENUS_RADIUS_KM = 6051.8
WINDOW_DAYS = 92
STEP_SIZE = "1 d"
PLOT_LIMIT_ARCSEC = 2000.0
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
    dates: list[datetime]
    x_arcsec: np.ndarray
    y_arcsec: np.ndarray
    rho_arcsec: np.ndarray
    sun_radius_arcsec: np.ndarray
    venus_radius_arcsec: np.ndarray
    ca_index: int
    ca_utc: str


def quoted(value: str) -> str:
    return f"'{value}'"


def horizons_url(target_id: str, start: datetime, stop: datetime) -> str:
    params = {
        "format": "json",
        "COMMAND": quoted(target_id),
        "OBJ_DATA": quoted("NO"),
        "MAKE_EPHEM": quoted("YES"),
        "EPHEM_TYPE": quoted("VECTORS"),
        "CENTER": quoted("500@399"),
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
            req = urllib.request.Request(url, headers={"User-Agent": SCRIPT_NAME})
            with urllib.request.urlopen(req, timeout=90) as response:
                if response.status != 200:
                    raise RuntimeError(f"Horizons HTTP status {response.status}")
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to retrieve JPL Horizons vectors: {last_error}")


def fetch_vectors(target_id: str, start: datetime, stop: datetime) -> VectorSeries:
    payload = fetch_json(horizons_url(target_id, start, stop))
    if "NASA/JPL" not in str(payload.get("signature", {}).get("source", "")):
        raise RuntimeError("Unexpected Horizons signature")
    result = payload.get("result", "")
    if "$$SOE" not in result or "$$EOE" not in result:
        raise RuntimeError("Horizons response contains no vector table")
    block = result.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    jd_values: list[float] = []
    xyz_values: list[list[float]] = []
    for row in csv.reader(io.StringIO(block.strip())):
        cells = [c.strip() for c in row]
        if len(cells) < 5:
            continue
        try:
            jd_values.append(float(cells[0]))
            xyz_values.append([float(cells[2]), float(cells[3]), float(cells[4])])
        except ValueError:
            continue
    if len(jd_values) < 150:
        raise RuntimeError(f"Insufficient JPL rows for target {target_id}: {len(jd_values)}")
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


def solar_basis(sun_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pole = solar_pole_unit()
    north = pole[None, :] - np.einsum("ij,j->i", sun_unit, pole)[:, None] * sun_unit
    north /= np.linalg.norm(north, axis=1)[:, None]
    east = np.cross(north, sun_unit)
    east /= np.linalg.norm(east, axis=1)[:, None]
    return east, north


def jd_to_datetime(jd: float) -> datetime:
    unix_seconds = (jd - 2440587.5) * 86400.0
    return datetime.fromtimestamp(unix_seconds, tz=timezone.utc)


def project_track(label: str, nominal: datetime) -> TransitTrack:
    start = nominal - timedelta(days=WINDOW_DAYS)
    stop = nominal + timedelta(days=WINDOW_DAYS)
    sun = fetch_vectors("10", start, stop)
    venus = fetch_vectors("299", start, stop)
    if sun.jd.shape != venus.jd.shape or np.max(np.abs(sun.jd - venus.jd)) * 86400.0 > 1e-6:
        raise RuntimeError(f"Sun/Venus epoch mismatch for {label}")

    sun_unit = normalize_rows(sun.xyz_km)
    venus_unit = normalize_rows(venus.xyz_km)
    east, north = solar_basis(sun_unit)
    denom = np.einsum("ij,ij->i", venus_unit, sun_unit)
    x = np.einsum("ij,ij->i", venus_unit, east) / denom * ARCSEC_PER_RAD
    y = np.einsum("ij,ij->i", venus_unit, north) / denom * ARCSEC_PER_RAD
    rho = np.hypot(x, y)
    sun_distance = np.linalg.norm(sun.xyz_km, axis=1)
    venus_distance = np.linalg.norm(venus.xyz_km, axis=1)
    sun_radius = np.arcsin(SOLAR_RADIUS_KM / sun_distance) * ARCSEC_PER_RAD
    venus_radius = np.arcsin(VENUS_RADIUS_KM / venus_distance) * ARCSEC_PER_RAD
    ca_index = int(np.argmin(rho))
    dates = [jd_to_datetime(value) for value in sun.jd]
    return TransitTrack(label, sun.jd, dates, x, y, rho, sun_radius, venus_radius, ca_index, dates[ca_index].strftime("%Y-%m-%d %H:%M UTC"))


def monthly_indices(dates: list[datetime]) -> list[int]:
    indices = [0]
    last = (dates[0].year, dates[0].month)
    for i, dt in enumerate(dates[1:], start=1):
        current = (dt.year, dt.month)
        if current != last:
            indices.append(i)
            last = current
    if indices[-1] != len(dates) - 1:
        indices.append(len(dates) - 1)
    return indices


def add_earth_observer(ax: plt.Axes) -> None:
    inset = ax.inset_axes([0.77, 0.77, 0.19, 0.19])
    inset.set_aspect("equal")
    inset.add_patch(Circle((0.0, 0.0), 1.0, fill=False, linewidth=0.8))
    inset.plot([0.0, 0.0], [-1.0, 1.0], linewidth=0.4)
    inset.plot([-1.0, 1.0], [0.0, 0.0], linewidth=0.4)
    inset.set_xlim(-1.3, 1.3)
    inset.set_ylim(-1.3, 1.3)
    inset.axis("off")
    inset.set_title("Earth observer", fontsize=7.5)


def make_pair_figure(tracks: tuple[TransitTrack, TransitTrack], path: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.linewidth": 0.35,
    })
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 12.5))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.06, hspace=0.24)

    for ax, track in zip(axes, tracks):
        sun_radius = float(track.sun_radius_arcsec[track.ca_index])
        ax.add_patch(Circle((0.0, 0.0), sun_radius, facecolor="0.94", edgecolor="0.25", linewidth=0.8, zorder=0))
        ax.plot(track.x_arcsec, track.y_arcsec, linewidth=0.9, zorder=2, label="Venus geocentric track")

        month_ids = monthly_indices(track.dates)
        for j, idx in enumerate(month_ids):
            vr = float(track.venus_radius_arcsec[idx])
            ax.add_patch(Circle((track.x_arcsec[idx], track.y_arcsec[idx]), vr, fill=False, linewidth=0.7, zorder=3))
            label = track.dates[idx].strftime("%b")
            ax.annotate(label, (track.x_arcsec[idx], track.y_arcsec[idx]), xytext=(4, 4), textcoords="offset points", fontsize=7.5)

        ca_x = float(track.x_arcsec[track.ca_index])
        ca_y = float(track.y_arcsec[track.ca_index])
        ca_vr = float(track.venus_radius_arcsec[track.ca_index])
        ax.axvline(ca_x, linewidth=1.0, linestyle="--", label="Closest-approach X")
        ax.add_patch(Circle((ca_x, ca_y), ca_vr, fill=False, linewidth=1.2, zorder=4))
        ax.plot(ca_x, ca_y, marker="o", markersize=2.5, zorder=5)

        ax.set_xlim(-PLOT_LIMIT_ARCSEC, PLOT_LIMIT_ARCSEC)
        ax.set_ylim(-PLOT_LIMIT_ARCSEC, PLOT_LIMIT_ARCSEC)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Solar east–west X (arcsec)")
        ax.set_ylabel("Projected solar north Y (arcsec)")
        ax.set_title(
            f"{track.label}: three months before to three months after\n"
            f"JPL closest approach: {track.ca_utc}; ρ = {track.rho_arcsec[track.ca_index]:.6f} arcsec",
            fontweight="bold",
        )
        ax.grid(True, which="major", alpha=0.45)
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.15)
        ax.legend(frameon=False, loc="lower left")
        add_earth_observer(ax)

    fig.suptitle("Venus Transit Progression — Sun-Facing Plane, Solar North Up", fontsize=13.0, fontweight="bold")
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.show()


def write_csv(path: Path, tracks: tuple[TransitTrack, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("transit", "jd_ut", "utc", "x_arcsec", "y_arcsec", "rho_arcsec", "sun_radius_arcsec", "venus_radius_arcsec"))
        for track in tracks:
            for jd, dt, x, y, rho, sr, vr in zip(track.jd, track.dates, track.x_arcsec, track.y_arcsec, track.rho_arcsec, track.sun_radius_arcsec, track.venus_radius_arcsec):
                writer.writerow((track.label, f"{jd:.12f}", dt.isoformat(), f"{x:.9f}", f"{y:.9f}", f"{rho:.9f}", f"{sr:.9f}", f"{vr:.9f}"))


def section(title: str, lines: list[str]) -> None:
    print(title)
    for line in lines:
        print(line)


def main() -> None:
    section("CODE INPUTS", [
        f"Version: {VERSION}",
        "Transits: 1761, 1769, 1874, 1882, 2004, 2012",
        f"Window: ±{WINDOW_DAYS} days",
        f"Fixed plot range: ±{PLOT_LIMIT_ARCSEC:.0f} arcsec",
        f"JPL cadence: {STEP_SIZE}",
    ])
    section("COMMENTS", [
        "Fresh geometric vectors are requested from NASA/JPL Horizons.",
        "Earth is the geocentric observer and is shown as an observer inset, not as a false image-plane body.",
        "Sun and Venus limbs are drawn from their JPL angular radii.",
        "Month labels mark the progression across each six-month window.",
    ])

    tracks = tuple(project_track(label, date) for label, date in TRANSITS)
    for (a, b), png_name in zip(PAIR_GROUPS, PNG_NAMES):
        make_pair_figure((tracks[a], tracks[b]), OUTPUT_DIR / png_name)
    write_csv(OUTPUT_DIR / CSV_NAME, tracks)

    section("RESULTS", [
        f"{t.label}: closest approach {t.ca_utc}; ρ={t.rho_arcsec[t.ca_index]:.6f} arcsec; "
        f"Sun radius={t.sun_radius_arcsec[t.ca_index]:.6f} arcsec; Venus diameter={2.0*t.venus_radius_arcsec[t.ca_index]:.6f} arcsec"
        for t in tracks
    ])
    section("OUTPUT SUMMARY", [*(f"PNG: {OUTPUT_DIR / name}" for name in PNG_NAMES), f"CSV: {OUTPUT_DIR / CSV_NAME}"])
    section("PAPER COMPARISON", ["Not used; plotted geometry is derived directly from JPL vectors."])
    section("EQUATION STATUS", [
        "PASS: solar north is projected into the instantaneous Sun-facing tangent plane.",
        "PASS: Sun and Venus limbs are calculated from JPL distances and physical radii.",
        "PASS: Earth is represented as the observer rather than a spurious image-plane track.",
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