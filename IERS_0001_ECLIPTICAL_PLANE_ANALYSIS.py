"""
IERS TN36 — Ecliptical Plane Analysis
Part I — Restored 2012 North/South Pole common solar-screen sanity check
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar

VERSION = "IERS-0001-I"
API = "https://ssd.jpl.nasa.gov/api/horizons.api"
ASEC = 206264.80624709636
DAY = 86400.0
RSUN = 695700.0
RVENUS = 6051.8
START = "2012-06-05 20:00"
STOP = "2012-06-06 07:30"
STEP = "1 min"


@dataclass(frozen=True)
class Site:
    key: str
    label: str
    lon: float
    lat: float
    elev: float

    @property
    def coord(self) -> str:
        return f"{self.lon:.8f},{self.lat:.8f},{self.elev:.6f}"


NP = Site("NP", "North Pole", 0.0, 90.0, 0.0)
SP = Site("SP", "South Pole", 0.0, -90.0, 0.0)
SITES = (NP, SP)


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0.0:
        raise ValueError("Invalid zero/non-finite vector.")
    return v / n


def q(value):
    return f"'{value}'"


def request_vectors(target, center, label, site=None):
    p = {
        "format": "json",
        "COMMAND": q(target),
        "OBJ_DATA": q("NO"),
        "MAKE_EPHEM": q("YES"),
        "EPHEM_TYPE": q("VECTORS"),
        "CENTER": q(center),
        "START_TIME": q(START),
        "STOP_TIME": q(STOP),
        "STEP_SIZE": q(STEP),
        "TIME_TYPE": q("UT"),
        "TIME_DIGITS": q("SECONDS"),
        "REF_PLANE": q("FRAME"),
        "REF_SYSTEM": q("ICRF"),
        "OUT_UNITS": q("KM-S"),
        "VEC_TABLE": q("2"),
        "VEC_CORR": q("NONE"),
        "CSV_FORMAT": q("YES"),
        "VEC_LABELS": q("YES"),
    }
    if site is not None:
        p["COORD_TYPE"] = q("GEODETIC")
        p["SITE_COORD"] = q(site.coord)

    req = Request(
        f"{API}?{urlencode(p)}",
        headers={"User-Agent": "IERS-TN36/1.0", "Accept": "application/json"},
    )
    with urlopen(req, timeout=90) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "NASA/JPL" not in str(payload.get("signature", {}).get("source", "")):
        raise RuntimeError("Unexpected Horizons source.")
    text = payload.get("result")
    if not isinstance(text, str):
        raise RuntimeError(str(payload.get("error", "No Horizons result.")))
    return parse_vectors(text, label)


def canon(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def parse_vectors(text, label):
    lines = text.splitlines()
    try:
        soe = next(i for i, line in enumerate(lines) if "$$SOE" in line)
        eoe = next(i for i, line in enumerate(lines) if i > soe and "$$EOE" in line)
    except StopIteration as exc:
        raise ValueError(f"{label}: missing Horizons data markers.") from exc

    header = None
    for i in range(soe - 1, max(-1, soe - 20), -1):
        row = next(csv.reader([lines[i]], skipinitialspace=True))
        names = [canon(x) for x in row]
        if all(x in names for x in ("X", "Y", "Z")):
            header = row
            break
    if header is None:
        raise ValueError(f"{label}: vector header not found.")

    names = [canon(x) for x in header]
    ji = next(names.index(x) for x in ("JDUT", "JDTDB", "JD") if x in names)
    xi, yi, zi = names.index("X"), names.index("Y"), names.index("Z")
    jd, xyz = [], []
    for row in csv.reader(lines[soe + 1:eoe], skipinitialspace=True):
        if not row or max(ji, xi, yi, zi) >= len(row):
            continue
        try:
            jd.append(float(row[ji]))
            xyz.append([float(row[xi]), float(row[yi]), float(row[zi])])
        except ValueError:
            pass

    jd = np.asarray(jd, float)
    xyz = np.asarray(xyz, float)
    if jd.size < 3 or xyz.shape != (jd.size, 3):
        raise ValueError(f"{label}: invalid vector table.")
    return {"jd": jd, "xyz": xyz}


def fetch_all():
    data = {
        "GS": request_vectors("10", "500@399", "Geocenter Sun"),
        "GV": request_vectors("299", "500@399", "Geocenter Venus"),
    }
    for site in SITES:
        data[f"{site.key}S"] = request_vectors(
            "10", "coord@399", f"{site.label} Sun", site
        )
        data[f"{site.key}V"] = request_vectors(
            "299", "coord@399", f"{site.label} Venus", site
        )

    jd = data["GS"]["jd"]
    for key, table in data.items():
        if table["jd"].shape != jd.shape or not np.allclose(table["jd"], jd, atol=1e-12):
            raise ValueError(f"{key}: unsynchronized epochs.")
    return data


def build_cache(data):
    cache = {"jd": data["GS"]["jd"]}
    for key, table in data.items():
        cache[key] = CubicSpline(
            table["jd"], table["xyz"], axis=0, bc_type="natural"
        )
    return cache


def vec(cache, key, jd):
    return np.asarray(cache[key](jd), float)


def separation(cache, sun_key, venus_key, jd):
    a, b = unit(vec(cache, sun_key, jd)), unit(vec(cache, venus_key, jd))
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)) * ASEC)


def radii(cache, site, jd):
    sun = vec(cache, f"{site.key}S", jd)
    venus = vec(cache, f"{site.key}V", jd)
    return (
        float(np.arctan2(RSUN, np.linalg.norm(sun)) * ASEC),
        float(np.arctan2(RVENUS, np.linalg.norm(venus)) * ASEC),
    )


def contact_value(cache, site, kind, jd):
    sep = separation(cache, f"{site.key}S", f"{site.key}V", jd)
    rs, rv = radii(cache, site, jd)
    return sep - (rs + rv if kind == "external" else rs - rv)


def roots(cache, site, kind):
    jd = cache["jd"]
    values = np.array([contact_value(cache, site, kind, t) for t in jd])
    found = []
    for i in range(len(jd) - 1):
        if values[i] == 0.0:
            found.append(float(jd[i]))
        elif values[i] * values[i + 1] < 0.0:
            found.append(
                float(
                    brentq(
                        lambda t: contact_value(cache, site, kind, t),
                        jd[i],
                        jd[i + 1],
                        xtol=1e-13,
                        rtol=1e-13,
                    )
                )
            )
    return found


def contacts(cache):
    result = {}
    for site in SITES:
        ext = roots(cache, site, "external")
        inte = roots(cache, site, "internal")
        if len(ext) < 2 or len(inte) < 2:
            raise RuntimeError(f"{site.label}: contacts not recovered.")
        result[site.key] = {
            "C1": ext[0],
            "C2": inte[0],
            "C3": inte[-1],
            "C4": ext[-1],
        }
    return result


def geocentric_ca(cache):
    jd = cache["jd"]
    sampled = np.array([separation(cache, "GS", "GV", t) for t in jd])
    i = int(np.argmin(sampled))
    result = minimize_scalar(
        lambda t: separation(cache, "GS", "GV", t),
        bounds=(jd[max(0, i - 3)], jd[min(len(jd) - 1, i + 3)]),
        method="bounded",
        options={"xatol": 1e-12},
    )
    if not result.success:
        raise RuntimeError("Closest-approach fit failed.")
    return float(result.x), float(result.fun)


def screen_basis(cache, ca_jd):
    n = unit(vec(cache, "GS", ca_jd))
    k = np.array([0.0, 0.0, 1.0])
    if np.linalg.norm(np.cross(k, n)) < 1e-12:
        k = np.array([1.0, 0.0, 0.0])
    xhat = unit(np.cross(k, n))
    yhat = unit(np.cross(n, xhat))
    return n, xhat, yhat


def screen_point(cache, site, jd, basis):
    n, xhat, yhat = basis
    osun = vec(cache, f"{site.key}S", jd)
    ovenus = vec(cache, f"{site.key}V", jd)
    scale = float(np.dot(osun, n)) / float(np.dot(ovenus, n))
    qvec = scale * ovenus - osun
    es = np.linalg.norm(vec(cache, "GS", jd))
    return np.array(
        [
            np.arctan2(np.dot(qvec, xhat), es) * ASEC,
            np.arctan2(np.dot(qvec, yhat), es) * ASEC,
        ],
        float,
    )


def pca(points):
    center = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    direction = vh[0]
    if direction[0] < 0.0:
        direction = -direction
    direction = unit(direction)
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    return center, direction, angle


def track_geometry(cache, event_times, ca_jd):
    basis = screen_basis(cache, ca_jd)
    all_contacts = [
        t for site_events in event_times.values() for t in site_events.values()
    ]
    jd = cache["jd"]
    fit_jd = jd[(jd >= min(all_contacts)) & (jd <= max(all_contacts))]

    tracks, directions, angles = {}, {}, {}
    for site in SITES:
        points = np.array([screen_point(cache, site, t, basis) for t in fit_jd])
        _, direction, angle = pca(points)
        tracks[site.key] = points
        directions[site.key] = direction
        angles[site.key] = angle

    common = unit(directions[NP.key] + directions[SP.key])
    if common[0] < 0.0:
        common = -common
    common_angle = float(np.degrees(np.arctan2(common[1], common[0])))
    return {
        "tracks": tracks,
        "angles": angles,
        "common_angle": common_angle,
        "fit_rows": len(fit_jd),
    }


def utc_text(jd):
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    dt = epoch + timedelta(seconds=(jd - 2440587.5) * DAY)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def display(event_times, ca_jd, ca_sep, geometry):
    print("IERS TN36 - Restored Common Solar-Screen Sanity Check")
    print(f"Version : {VERSION}")
    print("Method  : physical Venus-ray intersection with fixed geocentric solar screen")

    print("\nCONTACT TIMES UTC")
    print("SITE          C1                      C2                      C3                      C4")
    for site in SITES:
        e = event_times[site.key]
        print(
            f"{site.label:<13}"
            f"{utc_text(e['C1']):<24}"
            f"{utc_text(e['C2']):<24}"
            f"{utc_text(e['C3']):<24}"
            f"{utc_text(e['C4']):<24}"
        )

    print("\nGEOCENTRIC CLOSEST APPROACH")
    print("UTC                         JD UT                 SEPARATION arcsec")
    print(f"{utc_text(ca_jd):<27}{ca_jd:19.10f}{ca_sep:18.6f}")

    print("\nRESTORED C1-C4 PCA TRACK ANGLES")
    print("TRACK                 ANGLE deg")
    print(f"{'North Pole':<22}{geometry['angles'][NP.key]:12.6f}")
    print(f"{'South Pole':<22}{geometry['angles'][SP.key]:12.6f}")
    print(f"{'Common tangent':<22}{geometry['common_angle']:12.6f}")
    print(f"{'Fit rows':<22}{geometry['fit_rows']:12d}")


def plot_tracks(geometry):
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    for site in SITES:
        points = geometry["tracks"][site.key]
        ax.plot(
            points[:, 0],
            points[:, 1],
            linewidth=1.1,
            label=f"{site.label} {geometry['angles'][site.key]:.6f}°",
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Common solar-screen X (arcsec)")
    ax.set_ylabel("Common solar-screen Y (arcsec)")
    ax.set_title("2012 Venus Transit — Restored Solar-Screen Geometry")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    plt.show()


def main():
    data = fetch_all()
    cache = build_cache(data)
    event_times = contacts(cache)
    ca_jd, ca_sep = geocentric_ca(cache)
    geometry = track_geometry(cache, event_times, ca_jd)
    display(event_times, ca_jd, ca_sep, geometry)
    plot_tracks(geometry)


if __name__ == "__main__":
    main()
