# V0120
# Audit reference: JPL heliocentric Earth/Venus orbital time extrusion across the six 1761-2012 Venus-transit epochs.

from __future__ import annotations

import csv
import io
import json
import math
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

VERSION = "V0120"
SCRIPT_NAME = "VENUS_TRANSITS_1761_2012_HELIOCENTRIC_TIME_ORBITS_V0120.py"
PNG_NAME = "VENUS_TRANSITS_1761_2012_HELIOCENTRIC_TIME_ORBITS_V0120.png"
CSV_NAME = "VENUS_TRANSITS_1761_2012_HELIOCENTRIC_TIME_ORBITS_V0120.csv"
HORIZONS_ENDPOINT = "https://ssd.jpl.nasa.gov/api/horizons.api"
AU_KM = 149_597_870.7
START_UTC = "1761-01-01 00:00"
STOP_UTC = "2012-12-31 00:00"
STEP_SIZE = "1 mo"
OUTPUT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

TRANSIT_SEARCH_WINDOWS = (
    ("1761 transit", "1761-05-20 00:00", "1761-06-20 00:00"),
    ("1769 transit", "1769-05-20 00:00", "1769-06-20 00:00"),
    ("1874 transit", "1874-11-20 00:00", "1874-12-20 00:00"),
    ("1882 transit", "1882-11-20 00:00", "1882-12-20 00:00"),
    ("2004 transit", "2004-05-20 00:00", "2004-06-20 00:00"),
    ("2012 transit", "2012-05-20 00:00", "2012-06-20 00:00"),
)


@dataclass(frozen=True)
class VectorTable:
    target_id: str
    target_name: str
    jd: np.ndarray
    xyz_km: np.ndarray
    request_url: str


@dataclass(frozen=True)
class TransitEpoch:
    label: str
    jd: float
    utc: str
    earth_xyz_au: np.ndarray
    venus_xyz_au: np.ndarray
    elongation_arcsec: float


def quoted(value: str) -> str:
    return f"'{value}'"


def horizons_url(target_id: str, center: str, start: str, stop: str, step: str) -> str:
    params = {
        "format": "json",
        "COMMAND": quoted(target_id),
        "OBJ_DATA": quoted("NO"),
        "MAKE_EPHEM": quoted("YES"),
        "EPHEM_TYPE": quoted("VECTORS"),
        "CENTER": quoted(center),
        "START_TIME": quoted(start),
        "STOP_TIME": quoted(stop),
        "STEP_SIZE": quoted(step),
        "TIME_TYPE": quoted("UT"),
        "TIME_DIGITS": quoted("FRACSEC"),
        "CAL_TYPE": quoted("GREGORIAN"),
        "REF_PLANE": quoted("ECLIPTIC"),
        "REF_SYSTEM": quoted("ICRF"),
        "OUT_UNITS": quoted("KM-S"),
        "VEC_TABLE": quoted("1"),
        "VEC_CORR": quoted("NONE"),
        "CSV_FORMAT": quoted("YES"),
        "VEC_LABELS": quoted("NO"),
    }
    return HORIZONS_ENDPOINT + "?" + urllib.parse.urlencode(params)


def fetch_json(url: str, attempts: int = 3, timeout_s: int = 120) -> dict:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": f"{SCRIPT_NAME} NASA-JPL-Horizons-audit"},
            )
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                if response.status != 200:
                    raise RuntimeError(f"Horizons HTTP status {response.status}")
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to retrieve fresh JPL Horizons vectors: {last_error}")


def parse_vectors(result_text: str, target_id: str, request_url: str) -> VectorTable:
    if "$$SOE" not in result_text or "$$EOE" not in result_text:
        diagnostic = result_text[:1000].replace("\n", " ")
        raise RuntimeError(f"Horizons response has no vector table: {diagnostic}")

    target_name = target_id
    for line in result_text.splitlines():
        if line.startswith("Target body name:"):
            target_name = line.split(":", 1)[1].strip()
            break

    block = result_text.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
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

    if len(jd_values) < 10:
        raise RuntimeError(f"Insufficient JPL vectors parsed for {target_id}: {len(jd_values)}")

    return VectorTable(
        target_id=target_id,
        target_name=target_name,
        jd=np.asarray(jd_values, dtype=float),
        xyz_km=np.asarray(xyz_values, dtype=float),
        request_url=request_url,
    )


def fetch_vectors(target_id: str, center: str, start: str, stop: str, step: str) -> VectorTable:
    url = horizons_url(target_id, center, start, stop, step)
    payload = fetch_json(url)
    signature = payload.get("signature", {})
    if "NASA/JPL" not in str(signature.get("source", "")):
        raise RuntimeError(f"Unexpected Horizons signature: {signature}")
    result = payload.get("result", "")
    return parse_vectors(result, target_id, url)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0.0):
        raise RuntimeError("Zero-length vector encountered")
    return values / norms[:, None]


def angular_separation(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.arctan2(
        np.linalg.norm(np.cross(u, v), axis=1),
        np.einsum("ij,ij->i", u, v),
    )


def polynomial_minimum(jd: np.ndarray, values: np.ndarray) -> float:
    index = int(np.argmin(values))
    lo = max(0, index - 10)
    hi = min(len(jd), index + 11)
    center = float(jd[index])
    x_hours = (jd[lo:hi] - center) * 24.0
    fit = np.polynomial.Polynomial.fit(x_hours, values[lo:hi], 6).convert()
    candidates = [0.0]
    lower = float(x_hours.min())
    upper = float(x_hours.max())
    for root in fit.deriv().roots():
        if abs(root.imag) < 1.0e-10 and lower <= root.real <= upper:
            candidates.append(float(root.real))
    best = min(candidates, key=lambda value: float(fit(value)))
    if float(fit.deriv(2)(best)) <= 0.0:
        raise RuntimeError("Transit search stationary point is not a minimum")
    return center + best / 24.0


def interpolate_xyz(jd: np.ndarray, xyz: np.ndarray, jd_eval: float) -> np.ndarray:
    index = int(np.argmin(np.abs(jd - jd_eval)))
    lo = max(0, index - 8)
    hi = min(len(jd), index + 9)
    center = float(jd[index])
    x_hours = (jd[lo:hi] - center) * 24.0
    x_eval = (jd_eval - center) * 24.0
    result = np.empty(3, dtype=float)
    for axis in range(3):
        fit = np.polynomial.Polynomial.fit(x_hours, xyz[lo:hi, axis], 6).convert()
        result[axis] = float(fit(x_eval))
    return result


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


def jd_to_elapsed_months(jd: np.ndarray, jd_start: float) -> np.ndarray:
    return (jd - jd_start) / 365.2425 * 12.0


def refine_transit(label: str, start: str, stop: str) -> TransitEpoch:
    earth = fetch_vectors("399", "500@10", start, stop, "1 h")
    venus = fetch_vectors("299", "500@10", start, stop, "1 h")
    if earth.jd.shape != venus.jd.shape or np.max(np.abs(earth.jd - venus.jd)) * 86400.0 > 1.0e-6:
        raise RuntimeError(f"JPL epoch mismatch in {label}")

    earth_to_sun = -earth.xyz_km
    earth_to_venus = venus.xyz_km - earth.xyz_km
    elongation = angular_separation(normalize_rows(earth_to_venus), normalize_rows(earth_to_sun))
    transit_jd = polynomial_minimum(earth.jd, elongation * elongation)
    earth_xyz = interpolate_xyz(earth.jd, earth.xyz_km, transit_jd) / AU_KM
    venus_xyz = interpolate_xyz(venus.jd, venus.xyz_km, transit_jd) / AU_KM

    earth_to_sun_eval = -earth_xyz
    earth_to_venus_eval = venus_xyz - earth_xyz
    theta = math.atan2(
        np.linalg.norm(np.cross(earth_to_venus_eval, earth_to_sun_eval)),
        float(np.dot(earth_to_venus_eval, earth_to_sun_eval)),
    )

    return TransitEpoch(
        label=label,
        jd=transit_jd,
        utc=jd_to_utc(transit_jd),
        earth_xyz_au=earth_xyz,
        venus_xyz_au=venus_xyz,
        elongation_arcsec=theta * 206264.80624709636,
    )


def write_csv(
    path: Path,
    earth: VectorTable,
    venus: VectorTable,
    elapsed_months: np.ndarray,
    transits: tuple[TransitEpoch, ...],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "record_type",
            "label",
            "jd_ut",
            "elapsed_months_from_1761_01_01",
            "earth_x_au",
            "earth_y_au",
            "earth_z_au",
            "venus_x_au",
            "venus_y_au",
            "venus_z_au",
            "earth_seen_venus_sun_elongation_arcsec",
        ])
        for index, jd in enumerate(earth.jd):
            writer.writerow([
                "monthly_orbit",
                "",
                f"{jd:.12f}",
                f"{elapsed_months[index]:.9f}",
                f"{earth.xyz_km[index, 0] / AU_KM:.12f}",
                f"{earth.xyz_km[index, 1] / AU_KM:.12f}",
                f"{earth.xyz_km[index, 2] / AU_KM:.12f}",
                f"{venus.xyz_km[index, 0] / AU_KM:.12f}",
                f"{venus.xyz_km[index, 1] / AU_KM:.12f}",
                f"{venus.xyz_km[index, 2] / AU_KM:.12f}",
                "",
            ])
        for transit in transits:
            transit_month = float(jd_to_elapsed_months(np.asarray([transit.jd]), earth.jd[0])[0])
            writer.writerow([
                "transit",
                transit.label,
                f"{transit.jd:.12f}",
                f"{transit_month:.9f}",
                f"{transit.earth_xyz_au[0]:.12f}",
                f"{transit.earth_xyz_au[1]:.12f}",
                f"{transit.earth_xyz_au[2]:.12f}",
                f"{transit.venus_xyz_au[0]:.12f}",
                f"{transit.venus_xyz_au[1]:.12f}",
                f"{transit.venus_xyz_au[2]:.12f}",
                f"{transit.elongation_arcsec:.9f}",
            ])


def make_figure(
    path: Path,
    earth_xyz_au: np.ndarray,
    venus_xyz_au: np.ndarray,
    elapsed_months: np.ndarray,
    transits: tuple[TransitEpoch, ...],
    jd_start: float,
) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    })

    fig = plt.figure(figsize=(10.0, 18.0))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.965, bottom=0.035)

    ax.plot(earth_xyz_au[:, 0], elapsed_months, earth_xyz_au[:, 1], linewidth=0.45, label="Earth orbit")
    ax.plot(venus_xyz_au[:, 0], elapsed_months, venus_xyz_au[:, 1], linewidth=0.45, label="Venus orbit")
    ax.plot(np.zeros_like(elapsed_months), elapsed_months, np.zeros_like(elapsed_months), linewidth=0.55, label="Sun centerline")

    for transit in transits:
        month = float(jd_to_elapsed_months(np.asarray([transit.jd]), jd_start)[0])
        ax.scatter([transit.earth_xyz_au[0]], [month], [transit.earth_xyz_au[1]], s=9.0, depthshade=False)
        ax.scatter([transit.venus_xyz_au[0]], [month], [transit.venus_xyz_au[1]], s=9.0, depthshade=False)
        ax.text(0.0, month, 0.0, f"  {transit.label}", fontsize=7.2)

    tick_years = np.asarray([1761, 1769, 1800, 1850, 1874, 1882, 1900, 1950, 2000, 2004, 2012])
    tick_months = (tick_years - 1761) * 12.0
    ax.set_yticks(tick_months)
    ax.set_yticklabels([str(year) for year in tick_years])

    ax.set_xlabel("Heliocentric ecliptic X (AU)", labelpad=8)
    ax.set_ylabel("Calendar time — elapsed months from 1761-01-01", labelpad=14)
    ax.set_zlabel("Heliocentric ecliptic Y (AU)", labelpad=8)
    ax.set_title(
        "Earth and Venus Heliocentric Orbits Through the Six Venus Transits, 1761–2012\n"
        "Fresh NASA/JPL Horizons geometric vectors; orbital planes stacked along calendar time",
        fontsize=11.5,
        fontweight="bold",
        pad=18,
    )

    ax.set_xlim(-1.08, 1.08)
    ax.set_zlim(-1.08, 1.08)
    ax.set_ylim(float(elapsed_months.min()), float(elapsed_months.max()))
    ax.set_box_aspect((2.16, 8.0, 2.16))
    ax.view_init(elev=16.0, azim=-58.0)
    ax.grid(True, linewidth=0.25, alpha=0.35)
    ax.legend(loc="upper left", frameon=False)

    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.show()


def section(title: str, lines: list[str]) -> None:
    print(title)
    for line in lines:
        print(line)


def main() -> None:
    section("CODE INPUTS", [
        f"Version: {VERSION}",
        f"Interval: {START_UTC} to {STOP_UTC} UT",
        f"Orbit cadence: {STEP_SIZE}",
        "Reference frame: Sun-centered ecliptic J2000",
        "Targets: Earth (399), Venus (299)",
    ])
    section("COMMENTS", [
        "Fresh geometric vectors are requested directly from NASA/JPL Horizons.",
        "The long vertical axis is elapsed calendar months; each orbital slice remains parallel to the heliocentric ecliptic X-Y plane.",
        "Transit epochs are refined independently from hourly JPL vectors by minimizing Earth-observed Venus-Sun elongation.",
        "This first version plots orbital lines and transit markers only; planet and solar limbs are not drawn.",
    ])

    earth = fetch_vectors("399", "500@10", START_UTC, STOP_UTC, STEP_SIZE)
    venus = fetch_vectors("299", "500@10", START_UTC, STOP_UTC, STEP_SIZE)
    if earth.jd.shape != venus.jd.shape:
        raise RuntimeError("Earth and Venus monthly vector tables have different lengths")
    if np.max(np.abs(earth.jd - venus.jd)) * 86400.0 > 1.0e-6:
        raise RuntimeError("Earth and Venus monthly JPL epochs do not match")

    transits = tuple(refine_transit(*window) for window in TRANSIT_SEARCH_WINDOWS)
    elapsed_months = jd_to_elapsed_months(earth.jd, earth.jd[0])
    earth_xyz_au = earth.xyz_km / AU_KM
    venus_xyz_au = venus.xyz_km / AU_KM

    csv_path = OUTPUT_DIR / CSV_NAME
    png_path = OUTPUT_DIR / PNG_NAME
    write_csv(csv_path, earth, venus, elapsed_months, transits)

    result_lines = [
        f"Monthly samples: {len(earth.jd)}",
        f"Time span: {elapsed_months[-1]:.3f} months",
        f"Earth radial range: {np.min(np.linalg.norm(earth_xyz_au, axis=1)):.9f} to {np.max(np.linalg.norm(earth_xyz_au, axis=1)):.9f} AU",
        f"Venus radial range: {np.min(np.linalg.norm(venus_xyz_au, axis=1)):.9f} to {np.max(np.linalg.norm(venus_xyz_au, axis=1)):.9f} AU",
    ]
    result_lines.extend(
        f"{transit.label}: {transit.utc}; minimum elongation {transit.elongation_arcsec:.6f} arcsec"
        for transit in transits
    )
    section("RESULTS", result_lines)

    make_figure(png_path, earth_xyz_au, venus_xyz_au, elapsed_months, transits, earth.jd[0])
    section("OUTPUT SUMMARY", [f"PNG: {png_path}", f"CSV: {csv_path}"])
    section("PAPER COMPARISON", [
        "Published transit years are used only to define broad search windows; plotted positions and refined epochs come from fresh JPL vectors."
    ])
    section("EQUATION STATUS", [
        "PASS: all orbit coordinates are heliocentric NASA/JPL Horizons geometric vectors.",
        "PASS: calendar height is derived from JPL Julian dates.",
        "PASS: each transit epoch minimizes Earth-observed Venus-Sun angular separation within its search window.",
        "PASS: no AI-generated imagery is used; output is Matplotlib only.",
    ])
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"{VERSION} COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{VERSION} ERROR: {exc}", file=sys.stderr)
        raise

# V0120
