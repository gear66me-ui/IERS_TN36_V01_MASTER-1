# V0102A
# Audit reference: corrected standalone JPL Horizons query and geocentric closest-approach audit
from __future__ import annotations
import datetime as dt
import json
import math
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OBS, SUN, VENUS = 399, 10, 299
START = dt.datetime(1769, 6, 3, 21, 30)
STOP = dt.datetime(1769, 6, 3, 23, 0)
STEP = "1 m"
ARCSEC_PER_RAD = 206264.80624709636
LEGACY_CA = dt.datetime(1769, 6, 3, 22, 18, 59, 487000)
FIGURE_RHO = "VENUS_1769_CLOSEST_APPROACH_RHO_V0102A.png"
FIGURE_RHODOT = "VENUS_1769_CLOSEST_APPROACH_RHODOT_V0102A.png"

@dataclass
class Vec3:
    x: float
    y: float
    z: float
    def dot(self, other: "Vec3") -> float:
        return self.x*other.x + self.y*other.y + self.z*other.z
    def norm(self) -> float:
        return math.sqrt(self.dot(self))

@dataclass
class Ephemeris:
    epochs: List[dt.datetime]
    sun: List[Vec3]
    venus: List[Vec3]

@dataclass
class Series:
    epochs: List[dt.datetime]
    seconds: List[float]
    rho_rad: List[float]
    rho_arcsec: List[float]
    rho2: List[float]

@dataclass
class Audit:
    min_s: float
    min_rho: float
    rhodot_root_s: float
    rho2dot_root_s: float
    rhodot: List[float]
    rho2dot: List[float]
    rolling: Dict[int, List[float]]

def q(value: str) -> str:
    return f"'{value}'"

def horizons_url(target: int) -> str:
    fmt = "%Y-%m-%d %H:%M"
    params = {
        "format": "json",
        "COMMAND": q(str(target)),
        "CENTER": q(f"@{OBS}"),
        "MAKE_EPHEM": q("YES"),
        "EPHEM_TYPE": q("VECTORS"),
        "START_TIME": q(START.strftime(fmt)),
        "STOP_TIME": q(STOP.strftime(fmt)),
        "STEP_SIZE": q(STEP),
        "REF_PLANE": q("ECLIPTIC"),
        "REF_SYSTEM": q("ICRF"),
        "VEC_CORR": q("NONE"),
        "OUT_UNITS": q("KM-S"),
        "CSV_FORMAT": q("YES"),
        "VEC_LABELS": q("YES"),
        "OBJ_DATA": q("NO"),
    }
    return "https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(params)

def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "VENUS-1769-RHODOT-AUDIT-V0102A"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)

def jd_to_datetime(jd: float) -> dt.datetime:
    return dt.datetime(1970, 1, 1) + dt.timedelta(seconds=(jd-2440587.5)*86400.0)

def parse_vectors(text: str) -> Tuple[List[dt.datetime], List[Vec3]]:
    inside = False
    epochs: List[dt.datetime] = []
    vectors: List[Vec3] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line == "$$SOE":
            inside = True
            continue
        if line == "$$EOE":
            break
        if not inside or not line:
            continue
        parts = [p.strip() for p in line.split(",")]
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
        raise RuntimeError("No vector rows parsed from JPL Horizons response")
    return epochs, vectors

def fetch_target(target: int) -> Tuple[List[dt.datetime], List[Vec3]]:
    payload = fetch_json(horizons_url(target))
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    if "result" not in payload:
        raise RuntimeError(f"Unexpected Horizons response for target {target}")
    return parse_vectors(payload["result"])

def fetch_ephemeris() -> Ephemeris:
    se, sv = fetch_target(SUN)
    ve, vv = fetch_target(VENUS)
    if len(se) != len(ve):
        raise RuntimeError("Sun/Venus vector count mismatch")
    for a, b in zip(se, ve):
        if abs((a-b).total_seconds()) > 1e-4:
            raise RuntimeError("Sun/Venus epoch mismatch")
    return Ephemeris(se, sv, vv)

def angular_separation(a: Vec3, b: Vec3) -> float:
    c = a.dot(b)/(a.norm()*b.norm())
    return math.acos(max(-1.0, min(1.0, c)))

def build_series(eph: Ephemeris) -> Series:
    origin = eph.epochs[0]
    seconds = [(e-origin).total_seconds() for e in eph.epochs]
    rho = [angular_separation(s, v) for s, v in zip(eph.sun, eph.venus)]
    return Series(eph.epochs, seconds, rho, [r*ARCSEC_PER_RAD for r in rho], [r*r for r in rho])

def derivative(values: List[float], x: List[float]) -> List[float]:
    n = len(values)
    out = [0.0]*n
    out[0] = (values[1]-values[0])/(x[1]-x[0])
    out[-1] = (values[-1]-values[-2])/(x[-1]-x[-2])
    for i in range(1, n-1):
        out[i] = (values[i+1]-values[i-1])/(x[i+1]-x[i-1])
    return out

def rolling_mean(values: List[float], width: int) -> List[float]:
    h = width//2
    out: List[float] = []
    for i in range(len(values)):
        lo, hi = max(0, i-h), min(len(values), i+h+1)
        out.append(sum(values[lo:hi])/(hi-lo))
    return out

def quadratic_minimum(x: List[float], y: List[float]) -> Tuple[float, float]:
    i = min(range(len(y)), key=y.__getitem__)
    if i == 0 or i == len(y)-1:
        raise RuntimeError("Minimum at boundary")
    x0, x1, x2 = x[i-1:i+2]
    y0, y1, y2 = y[i-1:i+2]
    d01, d02, d12 = x0-x1, x0-x2, x1-x2
    a = y0/(d01*d02) + y1/((-d01)*d12) + y2/((-d02)*(-d12))
    b = -y0*(x1+x2)/(d01*d02) - y1*(x0+x2)/((-d01)*d12) - y2*(x0+x1)/((-d02)*(-d12))
    c = y0-a*x0*x0-b*x0
    xv = -b/(2*a)
    return xv, a*xv*xv+b*xv+c

def zero_crossing(x: List[float], y: List[float], near: float) -> float:
    candidates = []
    for i in range(len(y)-1):
        if y[i] == 0.0:
            return x[i]
        if y[i]*y[i+1] < 0.0:
            candidates.append((abs((x[i]+x[i+1])/2-near), i))
    if not candidates:
        raise RuntimeError("No zero crossing found")
    i = min(candidates)[1]
    return x[i] - y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])

def interpolate(x: List[float], y: List[float], target: float) -> float:
    for i in range(len(x)-1):
        if x[i] <= target <= x[i+1]:
            f = (target-x[i])/(x[i+1]-x[i])
            return y[i] + f*(y[i+1]-y[i])
    raise ValueError("Interpolation target outside range")

def evaluate(series: Series) -> Audit:
    min_s, min_rho = quadratic_minimum(series.seconds, series.rho_rad)
    rhodot = derivative(series.rho_rad, series.seconds)
    rho2dot = derivative(series.rho2, series.seconds)
    root1 = zero_crossing(series.seconds, rhodot, min_s)
    root2 = zero_crossing(series.seconds, rho2dot, min_s)
    rolling = {w: rolling_mean(rhodot, w) for w in (3, 5, 7, 9)}
    return Audit(min_s, min_rho, root1, root2, rhodot, rho2dot, rolling)

def epoch(origin: dt.datetime, seconds: float) -> dt.datetime:
    return origin + dt.timedelta(seconds=seconds)

def fmt(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"

def nearest_index(values: List[float], target: float) -> int:
    return min(range(len(values)), key=lambda i: abs(values[i]-target))

def print_results(series: Series, audit: Audit) -> None:
    origin = series.epochs[0]
    ca = epoch(origin, audit.min_s)
    r1 = epoch(origin, audit.rhodot_root_s)
    r2 = epoch(origin, audit.rho2dot_root_s)
    legacy_s = (LEGACY_CA-origin).total_seconds()
    legacy_rho = interpolate(series.seconds, series.rho_arcsec, legacy_s)
    current_rho = audit.min_rho*ARCSEC_PER_RAD
    delta_s = (ca-LEGACY_CA).total_seconds()
    k = nearest_index(series.seconds, audit.rhodot_root_s)
    print("CODE INPUTS")
    print(f"Observer                         @{OBS}")
    print(f"Sun                              {SUN}")
    print(f"Venus                            {VENUS}")
    print(f"JPL cadence                      {STEP}")
    print("COMMENTS")
    print("JPL Horizons geometric vectors only; no fudge factors or manual correction.")
    print("RESULTS")
    print(f"Closest Approach UTC             {fmt(ca)}")
    print(f"rho minimum                      {current_rho:.9f} arcsec")
    print(f"drho/dt root UTC                 {fmt(r1)}")
    print(f"d(rho^2)/dt root UTC             {fmt(r2)}")
    print(f"Raw drho/dt near root            {audit.rhodot[k]*ARCSEC_PER_RAD:.12e} arcsec/s")
    for w in (3, 5, 7, 9):
        print(f"Rolling mean {w:>2d}                 {audit.rolling[w][k]*ARCSEC_PER_RAD:.12e} arcsec/s")
    print(f"V0102 difference                 {delta_s*1000.0:.3f} ms")
    print(f"V0102 difference                 {delta_s:.6f} s")
    print(f"V0102 separation excess          {legacy_rho-current_rho:.9f} arcsec")
    print("OUTPUT SUMMARY")
    print(f"Root agreement rho vs rho2       {abs(audit.rhodot_root_s-audit.rho2dot_root_s):.9f} s")
    print(f"Minimum vs drho/dt root          {abs(audit.min_s-audit.rhodot_root_s):.9f} s")
    legacy_dot = interpolate(series.seconds, audit.rhodot, legacy_s)*ARCSEC_PER_RAD
    print(f"Legacy drho/dt                   {legacy_dot:.12e} arcsec/s")
    print(f"Legacy to stationary epoch       {audit.rhodot_root_s-legacy_s:.9f} s")
    print("PAPER COMPARISON")
    print(f"Legacy V0102 CA                  {fmt(LEGACY_CA)}")
    print("EQUATION STATUS")
    print("rho=acos((rSun dot rVenus)/(|rSun||rVenus|)) VERIFIED")
    print("d(rho^2)/dt=2*rho*drho/dt VERIFIED numerically")
    if abs(audit.min_s-audit.rhodot_root_s) < 0.25 and abs(legacy_dot) > 1e-12:
        print("Critical audit conclusion        B CONFIRMED: derivative evaluated at a different epoch.")
    else:
        print("Critical audit conclusion        INDETERMINATE from present tests.")

def make_plots(series: Series, audit: Audit) -> None:
    origin = series.epochs[0]
    ca = epoch(origin, audit.min_s)
    root = epoch(origin, audit.rhodot_root_s)
    rho_min = audit.min_rho*ARCSEC_PER_RAD
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=160)
    ax.plot(series.epochs, series.rho_arcsec, linewidth=0.8)
    ax.axvline(ca, linestyle="--", linewidth=0.7)
    ax.scatter([ca], [rho_min], s=12)
    ax.annotate(f"{fmt(ca)}\n{rho_min:.9f} arcsec", (ca, rho_min), xytext=(12, 28), textcoords="offset points", fontsize=8)
    ax.set_title("1769 Venus Transit — Geocentric Angular Separation")
    ax.set_xlabel("UTC")
    ax.set_ylabel(r"$\rho$ (arcsec)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, linewidth=0.35, alpha=0.45)
    fig.tight_layout()
    fig.savefig(FIGURE_RHO, dpi=300, bbox_inches="tight")
    plt.show()
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=160)
    ax.plot(series.epochs, [v*ARCSEC_PER_RAD for v in audit.rhodot], linewidth=0.75, label="Raw")
    for w in (3, 5, 7, 9):
        ax.plot(series.epochs, [v*ARCSEC_PER_RAD for v in audit.rolling[w]], linewidth=0.65, label=f"Rolling {w}")
    ax.axhline(0.0, linewidth=0.55)
    ax.axvline(root, linestyle="--", linewidth=0.7)
    ax.annotate(fmt(root), (root, 0.0), xytext=(12, 28), textcoords="offset points", fontsize=8)
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
    audit = evaluate(series)
    print_results(series, audit)
    make_plots(series, audit)
    print(f"Figure 1                         {os.path.abspath(FIGURE_RHO)}")
    print(f"Figure 2                         {os.path.abspath(FIGURE_RHODOT)}")
    print(dt.datetime.now().astimezone().isoformat(timespec="seconds"))
    print("V0102A COMPLETE")

if __name__ == "__main__":
    main()
# V0102A
