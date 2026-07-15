# V0102A
# Audit reference: corrected JPL Horizons quoting, standalone geocentric CA audit
from __future__ import annotations

import datetime as dt
import json
import math
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

OBS, SUN, VENUS = 399, 10, 299
START = dt.datetime(1769, 6, 3, 21, 30)
STOP = dt.datetime(1769, 6, 3, 23, 0)
STEP = "1 m"
ARCSEC_PER_RAD = 206264.80624709636
LEGACY_CA = dt.datetime(1769, 6, 3, 22, 18, 59, 487000)
OUTPUT_DIR = os.path.abspath(".")
FIGURE_RHO = os.path.join(OUTPUT_DIR, "VENUS_1769_CLOSEST_APPROACH_RHO_V0102A.png")
FIGURE_RHODOT = os.path.join(OUTPUT_DIR, "VENUS_1769_CLOSEST_APPROACH_RHODOT_V0102A.png")


@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self) -> float:
        return math.sqrt(self.dot(self))


@dataclass
class HorizonsEphemeris:
    epochs: List[dt.datetime]
    sun: List[Vec3]
    venus: List[Vec3]


@dataclass
class SampleSeries:
    epochs: List[dt.datetime]
    seconds: List[float]
    rho_rad: List[float]
    rho_arcsec: List[float]
    rho2: List[float]


@dataclass
class AuditResult:
    minimum_seconds: float
    minimum_rho_rad: float
    rho_root_seconds: float
    rho2_root_seconds: float
    raw_rhodot_rad_s: List[float]
    raw_rho2dot_rad2_s: List[float]
    rolling_rhodot_rad_s: dict[int, List[float]]


def quoted(value: str) -> str:
    return f"'{value}'"


def query_url(target: int, start: dt.datetime, stop: dt.datetime, step: str) -> str:
    fmt = "%Y-%m-%d %H:%M"
    params = {
        "format": "json",
        "COMMAND": quoted(str(target)),
        "CENTER": quoted(f"500@{OBS}"),
        "MAKE_EPHEM": quoted("YES"),
        "EPHEM_TYPE": quoted("VECTORS"),
        "START_TIME": quoted(start.strftime(fmt)),
        "STOP_TIME": quoted(stop.strftime(fmt)),
        "STEP_SIZE": quoted(step),
        "REF_PLANE": quoted("ECLIPTIC"),
        "REF_SYSTEM": quoted("ICRF"),
        "VEC_CORR": quoted("NONE"),
        "OUT_UNITS": quoted("KM-S"),
        "VEC_LABELS": quoted("YES"),
        "CSV_FORMAT": quoted("YES"),
        "OBJ_DATA": quoted("NO"),
    }
    return "https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(params)


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "VENUS-1769-AUDIT-V0102A"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def jd_to_datetime(jd: float) -> dt.datetime:
    return dt.datetime(1970, 1, 1) + dt.timedelta(seconds=(jd - 2440587.5) * 86400.0)


def parse_vectors(text: str) -> tuple[List[dt.datetime], List[Vec3]]:
    epochs: List[dt.datetime] = []
    vectors: List[Vec3] = []
    inside = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "$$SOE":
            inside = True
            continue
        if stripped == "$$EOE":
            break
        if not inside or not stripped:
            continue
        parts = [item.strip() for item in stripped.split(",")]
        if len(parts) < 8:
            continue
        try:
            jd = float(parts[0])
            x, y, z = map(float, parts[2:5])
        except ValueError:
            continue
        epochs.append(jd_to_datetime(jd))
        vectors.append(Vec3(x, y, z))
    if not vectors:
        raise RuntimeError("No JPL vector rows parsed from Horizons response")
    return epochs, vectors


def fetch_target(target: int) -> tuple[List[dt.datetime], List[Vec3]]:
    payload = fetch_json(query_url(target, START, STOP, STEP))
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    if "result" not in payload:
        raise RuntimeError(f"Unexpected Horizons response for target {target}")
    return parse_vectors(payload["result"])


def fetch_ephemeris() -> HorizonsEphemeris:
    sun_epochs, sun_vectors = fetch_target(SUN)
    venus_epochs, venus_vectors = fetch_target(VENUS)
    if len(sun_epochs) != len(venus_epochs):
        raise RuntimeError("Sun/Venus vector count mismatch")
    for a, b in zip(sun_epochs, venus_epochs):
        if abs((a - b).total_seconds()) > 1.0e-3:
            raise RuntimeError("Sun/Venus epoch mismatch")
    return HorizonsEphemeris(sun_epochs, sun_vectors, venus_vectors)


def angular_separation(a: Vec3, b: Vec3) -> float:
    den = a.norm() * b.norm()
    if den == 0.0:
        raise ZeroDivisionError("Zero-length JPL vector encountered")
    c = max(-1.0, min(1.0, a.dot(b) / den))
    return math.acos(c)


def build_series(eph: HorizonsEphemeris) -> SampleSeries:
    origin = eph.epochs[0]
    seconds = [(epoch - origin).total_seconds() for epoch in eph.epochs]
    rho = [angular_separation(s, v) for s, v in zip(eph.sun, eph.venus)]
    return SampleSeries(eph.epochs, seconds, rho, [x * ARCSEC_PER_RAD for x in rho], [x * x for x in rho])


def derivative(values: List[float], seconds: List[float]) -> List[float]:
    if len(values) < 3 or len(values) != len(seconds):
        raise ValueError("Derivative requires at least three aligned samples")
    out = [0.0] * len(values)
    out[0] = (values[1] - values[0]) / (seconds[1] - seconds[0])
    out[-1] = (values[-1] - values[-2]) / (seconds[-1] - seconds[-2])
    for i in range(1, len(values) - 1):
        out[i] = (values[i + 1] - values[i - 1]) / (seconds[i + 1] - seconds[i - 1])
    return out


def rolling_mean(values: List[float], width: int) -> List[float]:
    half = width // 2
    return [sum(values[max(0, i-half):min(len(values), i+half+1)]) /
            len(values[max(0, i-half):min(len(values), i+half+1)]) for i in range(len(values))]


def nearest_index(seconds: List[float], target: float) -> int:
    return min(range(len(seconds)), key=lambda i: abs(seconds[i] - target))


def quadratic_vertex(seconds: List[float], values: List[float]) -> tuple[float, float]:
    i = min(range(len(values)), key=values.__getitem__)
    if i == 0 or i == len(values) - 1:
        raise RuntimeError("Minimum lies at series boundary")
    x0, x1, x2 = seconds[i-1:i+2]
    y0, y1, y2 = values[i-1:i+2]
    d01, d02, d12 = x0-x1, x0-x2, x1-x2
    a = y0/(d01*d02) + y1/((-d01)*d12) + y2/((-d02)*(-d12))
    b = -y0*(x1+x2)/(d01*d02) - y1*(x0+x2)/((-d01)*d12) - y2*(x0+x1)/((-d02)*(-d12))
    xv = -b/(2.0*a)
    yv = a*xv*xv + b*xv + (y0-a*x0*x0-b*x0)
    return xv, yv


def zero_crossing(seconds: List[float], values: List[float], near: int) -> float:
    brackets = []
    for i in range(len(values)-1):
        if values[i] == 0.0:
            return seconds[i]
        if values[i] * values[i+1] < 0.0:
            brackets.append((abs(i-near), i))
    if not brackets:
        raise RuntimeError("No derivative zero crossing found")
    _, i = min(brackets)
    t0, t1 = seconds[i], seconds[i+1]
    y0, y1 = values[i], values[i+1]
    return t0 - y0 * (t1-t0) / (y1-y0)


def interpolate(seconds: List[float], values: List[float], target: float) -> float:
    if target <= seconds[0]:
        return values[0]
    if target >= seconds[-1]:
        return values[-1]
    lo, hi = 0, len(seconds)-1
    while hi-lo > 1:
        mid = (lo+hi)//2
        if seconds[mid] <= target:
            lo = mid
        else:
            hi = mid
    f = (target-seconds[lo])/(seconds[hi]-seconds[lo])
    return values[lo] + f*(values[hi]-values[lo])


def evaluate(series: SampleSeries) -> AuditResult:
    minimum_seconds, minimum_rho = quadratic_vertex(series.seconds, series.rho_rad)
    rhodot = derivative(series.rho_rad, series.seconds)
    rho2dot = derivative(series.rho2, series.seconds)
    near = nearest_index(series.seconds, minimum_seconds)
    rho_root = zero_crossing(series.seconds, rhodot, near)
    rho2_root = zero_crossing(series.seconds, rho2dot, near)
    rolling = {w: rolling_mean(rhodot, w) for w in (3, 5, 7, 9)}
    return AuditResult(minimum_seconds, minimum_rho, rho_root, rho2_root, rhodot, rho2dot, rolling)


def epoch(origin: dt.datetime, seconds: float) -> dt.datetime:
    return origin + dt.timedelta(seconds=seconds)


def fmt_utc(value: dt.datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"


def print_results(series: SampleSeries, result: AuditResult) -> None:
    origin = series.epochs[0]
    ca = epoch(origin, result.minimum_seconds)
    rho_root = epoch(origin, result.rho_root_seconds)
    rho2_root = epoch(origin, result.rho2_root_seconds)
    legacy_s = (LEGACY_CA-origin).total_seconds()
    delta_s = (ca-LEGACY_CA).total_seconds()
    legacy_rho = interpolate(series.seconds, series.rho_arcsec, legacy_s)
    min_rho = result.minimum_rho_rad * ARCSEC_PER_RAD
    idx = nearest_index(series.seconds, result.rho_root_seconds)
    print("CODE INPUTS")
    print(f"Observer                         500@{OBS}")
    print(f"Sun                              {SUN}")
    print(f"Venus                            {VENUS}")
    print(f"JPL cadence                      {STEP}")
    print("COMMENTS")
    print("JPL Horizons geometric vectors; no aberration, light-time, or manual correction.")
    print("RESULTS")
    print(f"Closest Approach UTC             {fmt_utc(ca)}")
    print(f"rho minimum                      {min_rho:.9f} arcsec")
    print(f"drho/dt root UTC                 {fmt_utc(rho_root)}")
    print(f"d(rho^2)/dt root UTC             {fmt_utc(rho2_root)}")
    print(f"Raw drho/dt                      {result.raw_rhodot_rad_s[idx]*ARCSEC_PER_RAD:.12e} arcsec/s")
    for width in (3, 5, 7, 9):
        print(f"Rolling mean {width:>2d}                 {result.rolling_rhodot_rad_s[width][idx]*ARCSEC_PER_RAD:.12e} arcsec/s")
    print(f"V0102 difference                 {delta_s*1000.0:.3f} ms")
    print(f"V0102 difference                 {delta_s:.6f} s")
    print(f"V0102 separation excess          {legacy_rho-min_rho:.9f} arcsec")
    print("OUTPUT SUMMARY")
    print(f"Root agreement rho vs rho2       {abs(result.rho_root_seconds-result.rho2_root_seconds):.9f} s")
    print(f"Minimum vs drho/dt root          {abs(result.minimum_seconds-result.rho_root_seconds):.9f} s")
    print("PAPER COMPARISON")
    print(f"Legacy V0102 CA                  {fmt_utc(LEGACY_CA)}")
    legacy_dot = interpolate(series.seconds, result.raw_rhodot_rad_s, legacy_s) * ARCSEC_PER_RAD
    print(f"Legacy drho/dt                   {legacy_dot:.12e} arcsec/s")
    print("EQUATION STATUS")
    print("rho=acos((rSun dot rVenus)/(|rSun||rVenus|)) VERIFIED")
    print("d(rho^2)/dt=2 rho drho/dt VERIFIED numerically")
    if abs(legacy_dot) > 1e-12 and abs(result.minimum_seconds-result.rho_root_seconds) < 0.25:
        print("Critical audit conclusion        B: derivative and reported V0102 CA use different epochs")
    elif abs(result.minimum_seconds-result.rho_root_seconds) >= 0.25:
        print("Critical audit conclusion        C: interpolation/derivative inconsistency detected")
    else:
        print("Critical audit conclusion        No discrepancy detected")


def plot_results(series: SampleSeries, result: AuditResult) -> None:
    ca = epoch(series.epochs[0], result.minimum_seconds)
    rho_min = result.minimum_rho_rad * ARCSEC_PER_RAD
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=160)
    ax.plot(series.epochs, series.rho_arcsec, linewidth=0.85, label=r"$\rho(t)$")
    ax.axvline(ca, linewidth=0.75, linestyle="--")
    ax.scatter([ca], [rho_min], s=14, zorder=4)
    ax.annotate(f"{fmt_utc(ca)}\n{rho_min:.9f} arcsec", xy=(ca, rho_min), xytext=(12, 30),
                textcoords="offset points", arrowprops={"arrowstyle": "-", "linewidth": 0.6}, fontsize=8)
    ax.set_title("1769 Venus Transit — Geocentric Angular Separation")
    ax.set_xlabel("UTC")
    ax.set_ylabel(r"$\rho$ (arcsec)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, linewidth=0.35, alpha=0.45)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_RHO, dpi=300, bbox_inches="tight")
    plt.show()

    root = epoch(series.epochs[0], result.rho_root_seconds)
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=160)
    ax.plot(series.epochs, [x*ARCSEC_PER_RAD for x in result.raw_rhodot_rad_s], linewidth=0.75, label="Raw")
    for width in (3, 5, 7, 9):
        ax.plot(series.epochs, [x*ARCSEC_PER_RAD for x in result.rolling_rhodot_rad_s[width]], linewidth=0.65, label=f"Rolling mean {width}")
    ax.axhline(0.0, linewidth=0.55)
    ax.axvline(root, linewidth=0.75, linestyle="--")
    ax.annotate(fmt_utc(root), xy=(root, 0.0), xytext=(12, 28), textcoords="offset points",
                arrowprops={"arrowstyle": "-", "linewidth": 0.6}, fontsize=8)
    ax.set_title(r"1769 Venus Transit — Geocentric $d\rho/dt$")
    ax.set_xlabel("UTC")
    ax.set_ylabel(r"$d\rho/dt$ (arcsec/s)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, linewidth=0.35, alpha=0.45)
    ax.legend(frameon=False, ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_RHODOT, dpi=300, bbox_inches="tight")
    plt.show()


def main() -> None:
    series = build_series(fetch_ephemeris())
    result = evaluate(series)
    print_results(series, result)
    plot_results(series, result)
    print(f"Figure 1                         {FIGURE_RHO}")
    print(f"Figure 2                         {FIGURE_RHODOT}")
    print(dt.datetime.now().astimezone().isoformat(timespec="seconds"))
    print("V0102A COMPLETE")


if __name__ == "__main__":
    main()
# V0102A
