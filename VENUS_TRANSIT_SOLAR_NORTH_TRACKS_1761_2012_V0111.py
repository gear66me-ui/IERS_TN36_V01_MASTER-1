# V0111
# Audit reference: geocentric Venus tracks in the Sun-facing tangent plane with projected solar north vertical.

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

VERSION = "V0111"
SCRIPT_NAME = "VENUS_TRANSIT_SOLAR_NORTH_TRACKS_1761_2012_V0111.py"
CSV_NAME = "VENUS_TRANSIT_SOLAR_NORTH_TRACKS_1761_2012_V0111.csv"
PNG_NAMES = (
    "VENUS_TRANSIT_SOLAR_NORTH_TRACKS_1761_1769_V0111.png",
    "VENUS_TRANSIT_SOLAR_NORTH_TRACKS_1874_1882_V0111.png",
    "VENUS_TRANSIT_SOLAR_NORTH_TRACKS_2004_2012_V0111.png",
)
OUTPUT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
HORIZONS_ENDPOINT = "https://ssd.jpl.nasa.gov/api/horizons.api"
ARCSEC_PER_RAD = 206264.80624709636
SOLAR_RADIUS_KM = 695700.0
WINDOW_DAYS = 92
STEP_SIZE = "1 d"

# IAU solar north pole direction in ICRF/J2000.
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
    target_id: str
    jd: np.ndarray
    xyz_km: np.ndarray
    request_url: str


@dataclass(frozen=True)
class TransitTrack:
    label: str
    nominal_date: datetime
    jd: np.ndarray
    elapsed_days: np.ndarray
    x_arcsec: np.ndarray
    y_arcsec: np.ndarray
    rho_arcsec: np.ndarray
    sun_radius_arcsec: np.ndarray
    ca_jd: float
    ca_utc: str
    ca_rho_arcsec: float


def quoted(value: str) -> str:
    return f"'{value}'"


def datetime_to_horizons(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M")


def horizons_url(target_id: str, start: datetime, stop: datetime) -> str:
    params = {
        "format": "json",
        "COMMAND": quoted(target_id),
        "OBJ_DATA": quoted("NO"),
        "MAKE_EPHEM": quoted("YES"),
        "EPHEM_TYPE": quoted("VECTORS"),
        "CENTER": quoted("500@399"),
        "START_TIME": quoted(datetime_to_horizons(start)),
        "STOP_TIME": quoted(datetime_to_horizons(stop)),
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
            request = urllib.request.Request(
                url,
                headers={"User-Agent": f"{SCRIPT_NAME} JPL-Horizons-audit"},
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 200:
                    raise RuntimeError(f"Horizons HTTP status {response.status}")
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to retrieve JPL Horizons vectors: {last_error}")


def fetch_vectors(target_id: str, start: datetime, stop: datetime) -> VectorSeries:
    url = horizons_url(target_id, start, stop)
    payload = fetch_json(url)
    signature = payload.get("signature", {})
    if "NASA/JPL" not in str(signature.get("source", "")):
        raise RuntimeError(f"Unexpected Horizons signature: {signature}")

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

    if len(jd_values) < 120:
        raise RuntimeError(f"Insufficient JPL vectors parsed for target {target_id}: {len(jd_values)}")

    return VectorSeries(
        target_id=target_id,
        jd=np.asarray(jd_values, dtype=float),
        xyz_km=np.asarray(xyz_values, dtype=float),
        request_url=url,
    )


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0.0):
        raise RuntimeError("Zero-length JPL vector encountered")
    return values / norms[:, None]


def solar_pole_unit() -> np.ndarray:
    ra = math.radians(SOLAR_POLE_RA_DEG)
    dec = math.radians(SOLAR_POLE_DEC_DEG)
    return np.array([
        math.cos(dec) * math.cos(ra),
        math.cos(dec) * math.sin(ra),
        math.sin(dec),
    ])


def solar_north_basis(sun_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pole = solar_pole_unit()
    north = pole[None, :] - np.einsum("ij,j->i", sun_unit, pole)[:, None] * sun_unit
    north_norm = np.linalg.norm(north, axis=1)
    if np.any(north_norm <= 1.0e-12):
        raise RuntimeError("Solar north projection became singular")
    north /= north_norm[:, None]
    east = np.cross(north, sun_unit)
    east /= np.linalg.norm(east, axis=1)[:, None]
    return east, north


def project_venus(
    sun_xyz: np.ndarray,
    venus_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sun_unit = normalize_rows(sun_xyz)
    venus_unit = normalize_rows(venus_xyz)
    east, north = solar_north_basis(sun_unit)
    denominator = np.einsum("ij,ij->i", venus_unit, sun_unit)
    if np.any(denominator <= 0.0):
        raise RuntimeError("Venus lies outside the forward Sun-facing tangent plane")

    x = np.einsum("ij,ij->i", venus_unit, east) / denominator * ARCSEC_PER_RAD
    y = np.einsum("ij,ij->i", venus_unit, north) / denominator * ARCSEC_PER_RAD
    rho = np.hypot(x, y)
    sun_distance = np.linalg.norm(sun_xyz, axis=1)
    sun_radius = np.arcsin(SOLAR_RADIUS_KM / sun_distance) * ARCSEC_PER_RAD
    return x, y, rho, sun_radius


def polynomial_minimum(jd: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    index = int(np.argmin(values))
    lo = max(0, index - 5)
    hi = min(len(jd), index + 6)
    center = float(jd[index])
    x_days = jd[lo:hi] - center
    fit = np.polynomial.Polynomial.fit(x_days, values[lo:hi] ** 2, 4).convert()
    candidates = [0.0]
    lower = float(x_days.min())
    upper = float(x_days.max())
    for root in fit.deriv().roots():
        if abs(root.imag) < 1.0e-10 and lower <= root.real <= upper:
            candidates.append(float(root.real))
    best = min(candidates, key=lambda value: float(fit(value)))
    if float(fit.deriv(2)(best)) <= 0.0:
        raise RuntimeError("Interpolated transit minimum is not a minimum")
    return center + best, math.sqrt(max(0.0, float(fit(best))))


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


def build_track(label: str, nominal_date: datetime) -> TransitTrack:
    start = nominal_date - timedelta(days=WINDOW_DAYS)
    stop = nominal_date + timedelta(days=WINDOW_DAYS)
    sun = fetch_vectors("10", start, stop)
    venus = fetch_vectors("299", start, stop)

    if sun.jd.shape != venus.jd.shape:
        raise RuntimeError(f"Sun/Venus vector length mismatch for {label}")
    if np.max(np.abs(sun.jd - venus.jd)) * 86400.0 > 1.0e-6:
        raise RuntimeError(f"Sun/Venus epoch mismatch for {label}")

    x, y, rho, sun_radius = project_venus(sun.xyz_km, venus.xyz_km)
    ca_jd, ca_rho = polynomial_minimum(sun.jd, rho)
    elapsed = sun.jd - ca_jd

    return TransitTrack(
        label=label,
        nominal_date=nominal_date,
        jd=sun.jd,
        elapsed_days=elapsed,
        x_arcsec=x,
        y_arcsec=y,
        rho_arcsec=rho,
        sun_radius_arcsec=sun_radius,
        ca_jd=ca_jd,
        ca_utc=jd_to_utc(ca_jd),
        ca_rho_arcsec=ca_rho,
    )


def track_extent(track: TransitTrack) -> float:
    values = np.concatenate((np.abs(track.x_arcsec), np.abs(track.y_arcsec)))
    return float(np.max(values)) * 1.04


def make_pair_figure(
    tracks: tuple[TransitTrack, TransitTrack],
    output_path: Path,
) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.linewidth": 0.35,
    })

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 13.0))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.94, bottom=0.07, hspace=0.24)

    for ax, track in zip(axes, tracks):
        ax.plot(track.x_arcsec, track.y_arcsec, linewidth=0.8, label="Venus geocentric track")
        radius = float(np.median(track.sun_radius_arcsec))
        limb = plt.Circle((0.0, 0.0), radius, fill=False, linewidth=0.7, label="Sun limb")
        ax.add_patch(limb)
        limit = max(track_extent(track), radius * 1.25)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Solar east–west, X (arcsec)")
        ax.set_ylabel("Projected solar north, Y (arcsec)")
        ax.set_title(
            f"{track.label} Venus transit window: −3 to +3 months\n"
            f"JPL closest approach {track.ca_utc}; ρ = {track.ca_rho_arcsec:.6f} arcsec",
            fontweight="bold",
        )
        ax.grid(True, which="major", alpha=0.50)
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.18)
        ax.legend(frameon=False, loc="best")

    fig.suptitle(
        "Geocentric Venus Tracks in the Sun-Facing Plane — Solar North Up",
        fontsize=13.0,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.show()


def write_csv(path: Path, tracks: tuple[TransitTrack, ...]) -> None:
    headers = (
        "transit_label",
        "jd_ut",
        "elapsed_days_from_jpl_closest_approach",
        "venus_x_arcsec_solar_east_west",
        "venus_y_arcsec_solar_north",
        "venus_rho_arcsec",
        "sun_radius_arcsec",
        "closest_approach_jd_ut",
        "closest_approach_utc",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for track in tracks:
            for jd, elapsed, x, y, rho, radius in zip(
                track.jd,
                track.elapsed_days,
                track.x_arcsec,
                track.y_arcsec,
                track.rho_arcsec,
                track.sun_radius_arcsec,
            ):
                writer.writerow([
                    track.label,
                    f"{jd:.12f}",
                    f"{elapsed:.9f}",
                    f"{x:.9f}",
                    f"{y:.9f}",
                    f"{rho:.9f}",
                    f"{radius:.9f}",
                    f"{track.ca_jd:.12f}",
                    track.ca_utc,
                ])


def section(title: str, lines: list[str]) -> None:
    print(title)
    for line in lines:
        print(line)


def main() -> None:
    section("CODE INPUTS", [
        f"Version: {VERSION}",
        "Transit set: 1761, 1769, 1874, 1882, 2004, 2012",
        f"Window: ±{WINDOW_DAYS} days around each transit",
        f"JPL cadence: {STEP_SIZE}",
        "Observer: geocenter",
    ])
    section("COMMENTS", [
        "Fresh geometric vectors are requested directly from NASA/JPL Horizons.",
        "The viewing direction follows the geocentric Sun direction at every epoch.",
        "Projected solar north defines +Y; solar east–west defines +X.",
        "Earth is the observer and therefore defines the origin rather than a separate apparent track.",
    ])

    tracks = tuple(build_track(label, date) for label, date in TRANSITS)
    csv_path = OUTPUT_DIR / CSV_NAME
    write_csv(csv_path, tracks)

    for pair, png_name in zip(PAIR_GROUPS, PNG_NAMES):
        pair_tracks = (tracks[pair[0]], tracks[pair[1]])
        make_pair_figure(pair_tracks, OUTPUT_DIR / png_name)

    result_lines = []
    for track in tracks:
        result_lines.append(
            f"{track.label}: CA {track.ca_utc}; JD {track.ca_jd:.12f}; "
            f"minimum ρ {track.ca_rho_arcsec:.6f} arcsec"
        )
    section("RESULTS", result_lines)
    section("OUTPUT SUMMARY", [
        f"CSV: {csv_path}",
        *[f"PNG: {OUTPUT_DIR / name}" for name in PNG_NAMES],
    ])
    section("PAPER COMPARISON", [
        "Transit years are used only to define search windows; plotted coordinates and closest approaches come from JPL vectors."
    ])
    section("EQUATION STATUS", [
        "PASS: geocentric Sun and Venus vectors share identical JPL epochs.",
        "PASS: each tangent plane follows the Sun and is rotated so projected solar north is vertical.",
        "PASS: all panels preserve equal X/Y scale and include the solar limb.",
    ])
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"{VERSION} COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{VERSION} ERROR: {exc}", file=sys.stderr)
        raise

# V0111