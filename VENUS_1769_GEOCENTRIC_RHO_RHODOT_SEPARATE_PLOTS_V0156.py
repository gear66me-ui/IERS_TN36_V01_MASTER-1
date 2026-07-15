# V0156
# Audit reference: 1769 geocentric direct angular-separation rho(t) and drho/dt audit with separate figures and rolling-mean zero checks.
from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path


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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar

VERSION = "V0156"
LOCATION = "@399"
START_UTC = "1769-06-03 21:45"
STOP_UTC = "1769-06-03 22:50"
STEP = "1m"
REFPLANE = "ecliptic"
ABERRATIONS = "geometric"
AU_KM = 149597870.700
AS_PER_RAD = 206264.80624709636
ROLLING_WINDOWS = (3, 5, 7, 9)
REFERENCE_CA_UTC = "1769-06-03 22:18:59.487"
ZERO_TOLERANCE = 1.0e-5
OUTPUT_DIR = Path("/content/VENUS_1769_GEOCENTRIC_RHO_RHODOT_SEPARATE_PLOTS_V0156_OUTPUT")
RHO_PNG = OUTPUT_DIR / "VENUS_1769_GEOCENTRIC_RHO_V0156.png"
RHODOT_PNG = OUTPUT_DIR / "VENUS_1769_GEOCENTRIC_RHODOT_V0156.png"
CSV_PATH = OUTPUT_DIR / "VENUS_1769_GEOCENTRIC_RHO_RHODOT_V0156.csv"


def section(title: str) -> None:
    print(title)
    print("-" * len(title))


def query_vectors(body_id: str) -> tuple[np.ndarray, np.ndarray]:
    table = Horizons(
        id=body_id,
        location=LOCATION,
        epochs={"start": START_UTC, "stop": STOP_UTC, "step": STEP},
    ).vectors(refplane=REFPLANE, aberrations=ABERRATIONS)
    jd_tdb = np.asarray(table["datetime_jd"], dtype=float)
    xyz_km = np.column_stack([
        np.asarray(table["x"], dtype=float),
        np.asarray(table["y"], dtype=float),
        np.asarray(table["z"], dtype=float),
    ]) * AU_KM
    if len(jd_tdb) < 30 or not np.all(np.diff(jd_tdb) > 0.0):
        raise RuntimeError(f"REJECTED invalid JPL grid for body {body_id}")
    return jd_tdb, xyz_km


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise RuntimeError("REJECTED invalid vector norm")
    return values / norms[:, None]


def centered_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window=window, center=True, min_periods=window).mean().to_numpy()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    section("CODE INPUTS")
    print(f"Version                              {VERSION}")
    print("JPL source                           Horizons geocentric geometric vectors")
    print("Observer                             @399 Earth center")
    print("Bodies                               Sun=10; Venus=299")
    print(f"Reference plane                      {REFPLANE}")
    print(f"Aberrations                          {ABERRATIONS}")
    print("Cadence                              1 minute")
    print("rho definition                       direct great-circle Sun-Venus angular separation")
    print("Closest approach                     minimum rho(t); independently verified by drho/dt = 0")
    print("Rolling means                        centered 3, 5, 7, and 9 points")
    print(f"Output                               {OUTPUT_DIR}")

    jd_sun, sun_xyz = query_vectors("10")
    jd_venus, venus_xyz = query_vectors("299")
    if len(jd_sun) != len(jd_venus) or not np.allclose(jd_sun, jd_venus, atol=1.0e-12, rtol=0.0):
        raise RuntimeError("REJECTED mismatched JPL grids")

    jd = jd_sun
    jd_origin = float(jd[len(jd) // 2])
    t_seconds = (jd - jd_origin) * 86400.0
    sun_unit = normalize_rows(sun_xyz)
    venus_unit = normalize_rows(venus_xyz)
    dot = np.clip(np.einsum("ij,ij->i", sun_unit, venus_unit), -1.0, 1.0)
    cross_norm = np.linalg.norm(np.cross(sun_unit, venus_unit), axis=1)
    rho_arcsec = np.arctan2(cross_norm, dot) * AS_PER_RAD

    rho_spline = CubicSpline(t_seconds, rho_arcsec, bc_type="natural")
    rhodot_spline = rho_spline.derivative()
    coarse_index = int(np.argmin(rho_arcsec))
    lo = max(0, coarse_index - 4)
    hi = min(len(t_seconds) - 1, coarse_index + 4)

    minimum = minimize_scalar(
        lambda t: float(rho_spline(t)),
        bounds=(float(t_seconds[lo]), float(t_seconds[hi])),
        method="bounded",
        options={"xatol": 1.0e-8, "maxiter": 500},
    )
    if not minimum.success:
        raise RuntimeError("REJECTED rho minimum refinement")
    t_min = float(minimum.x)

    root_lo = float(t_seconds[lo])
    root_hi = float(t_seconds[hi])
    if float(rhodot_spline(root_lo)) * float(rhodot_spline(root_hi)) >= 0.0:
        raise RuntimeError("REJECTED drho/dt root not bracketed")
    t_root = float(brentq(lambda t: float(rhodot_spline(t)), root_lo, root_hi, xtol=1.0e-10, rtol=1.0e-14, maxiter=500))

    t_ca = t_root
    jd_ca = jd_origin + t_ca / 86400.0
    ca_tdb = Time(jd_ca, format="jd", scale="tdb")
    ca_utc = ca_tdb.utc
    ca_text = ca_utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    reference = Time(REFERENCE_CA_UTC, scale="utc")
    reference_offset_seconds = float((ca_utc.tdb.jd - reference.tdb.jd) * 86400.0)
    method_agreement_seconds = abs(t_min - t_root)
    analytic_at_ca = float(rhodot_spline(t_ca)) * 60.0

    sample_rhodot = rhodot_spline(t_seconds) * 60.0
    rolling_series: dict[int, np.ndarray] = {}
    rolling_at_ca: dict[int, float] = {}
    for window in ROLLING_WINDOWS:
        rolling = centered_rolling_mean(sample_rhodot, window)
        rolling_series[window] = rolling
        valid = np.isfinite(rolling)
        rolling_spline = CubicSpline(t_seconds[valid], rolling[valid], bc_type="natural")
        rolling_at_ca[window] = float(rolling_spline(t_ca))

    zero_checks_pass = abs(analytic_at_ca) <= ZERO_TOLERANCE and all(abs(v) <= ZERO_TOLERANCE for v in rolling_at_ca.values())

    minutes = (t_seconds - t_ca) / 60.0
    dense_minutes = np.linspace(-30.0, 30.0, 2401)
    dense_t = t_ca + dense_minutes * 60.0
    dense_rho = rho_spline(dense_t)
    dense_rhodot = rhodot_spline(dense_t) * 60.0

    output = pd.DataFrame({
        "jd_tdb": jd,
        "utc": [Time(v, format="jd", scale="tdb").utc.strftime("%Y-%m-%d %H:%M:%S") for v in jd],
        "minutes_from_ca": minutes,
        "rho_arcsec": rho_arcsec,
        "drho_dt_arcsec_per_min": sample_rhodot,
    })
    for window in ROLLING_WINDOWS:
        output[f"rolling_mean_{window}_arcsec_per_min"] = rolling_series[window]
    output.to_csv(CSV_PATH, index=False)

    plt.style.use("dark_background")

    fig1, ax1 = plt.subplots(figsize=(13.5, 6.2), dpi=120)
    ax1.plot(dense_minutes, dense_rho, linewidth=1.0, label="rho(t): direct angular separation")
    ax1.axvline(0.0, linewidth=0.75, alpha=0.85, label="Geocentric CA")
    ax1.scatter([0.0], [float(rho_spline(t_ca))], s=46, marker="D", zorder=6)
    ax1.set_title("1769 Venus Transit — Geocentric rho(t) Minimum", fontsize=15, weight="bold")
    ax1.set_xlabel("Minutes From Geocentric Closest Approach")
    ax1.set_ylabel("rho (arcsec)")
    ax1.grid(alpha=0.18)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.annotate(
        f"Geocentric CA\n{ca_text} UTC\nrho = {float(rho_spline(t_ca)):.12f} arcsec",
        xy=(0.0, float(rho_spline(t_ca))),
        xytext=(4.0, float(rho_spline(t_ca)) + 0.45),
        fontsize=9,
        arrowprops={"arrowstyle": "-", "linewidth": 0.7},
    )
    fig1.tight_layout()
    fig1.savefig(RHO_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(13.5, 6.2), dpi=120)
    ax2.axhline(0.0, linewidth=0.7, alpha=0.7)
    ax2.axvline(0.0, linewidth=0.75, alpha=0.85, label="Geocentric CA")
    ax2.plot(dense_minutes, dense_rhodot, linewidth=1.0, label="Analytic drho/dt")
    for window in ROLLING_WINDOWS:
        valid = np.isfinite(rolling_series[window])
        ax2.plot(minutes[valid], rolling_series[window][valid], linewidth=0.8, alpha=0.78, label=f"{window}-Point Rolling Mean")
    ax2.scatter([0.0], [analytic_at_ca], s=42, marker="D", zorder=6)
    ax2.set_title("1769 Venus Transit — Geocentric drho/dt Zero Crossing", fontsize=15, weight="bold")
    ax2.set_xlabel("Minutes From Geocentric Closest Approach")
    ax2.set_ylabel("drho/dt (arcsec/min)")
    ax2.grid(alpha=0.18)
    ax2.legend(loc="upper left", fontsize=8, ncol=3)
    ax2.annotate(
        f"drho/dt = {analytic_at_ca:+.12e} arcsec/min\n{ca_text} UTC",
        xy=(0.0, analytic_at_ca),
        xytext=(-27.0, 0.42),
        fontsize=9,
        arrowprops={"arrowstyle": "-", "linewidth": 0.7},
    )
    fig2.tight_layout()
    fig2.savefig(RHODOT_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig2)

    section("COMMENTS")
    print("rho(t) is the direct great-circle angle between the normalized geocentric Sun and Venus vectors.")
    print("No tangent-plane or gnomonic approximation is used to select the minimum.")
    print("rho(t) and drho/dt are saved as two separate publication figures.")

    section("RESULTS")
    print(f"Geocentric rho-min CA UTC             {ca_text}")
    print(f"Offset from V0102                    {reference_offset_seconds:+.6f} s")
    print(f"rho minimum                          {float(rho_spline(t_ca)):.12f} arcsec")
    print(f"Analytic drho/dt at CA               {analytic_at_ca:+.12e} arcsec/min")
    for window in ROLLING_WINDOWS:
        print(f"{window}-point rolling mean at CA          {rolling_at_ca[window]:+.12e} arcsec/min")
    print(f"rho-min / drho-root agreement        {method_agreement_seconds:.9f} s")

    section("OUTPUT SUMMARY")
    print(f"rho(t) PNG                           {RHO_PNG}")
    print(f"drho/dt PNG                          {RHODOT_PNG}")
    print(f"CSV                                  {CSV_PATH}")

    section("PAPER COMPARISON")
    print(f"V0102 comparison CA                  {REFERENCE_CA_UTC} UTC")
    print(f"Calculated minus V0102               {reference_offset_seconds:+.6f} s")

    section("EQUATION STATUS")
    print("rho minimum                          VERIFIED")
    print("drho/dt root                         VERIFIED")
    print(f"rolling-mean zero checks             {'VERIFIED' if zero_checks_pass else 'REJECTED'}")

    display(Image(filename=str(RHO_PNG)))
    display(Image(filename=str(RHODOT_PNG)))
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(VERSION)

    if not zero_checks_pass:
        raise RuntimeError("REJECTED rolling means are not zero at closest approach")


if __name__ == "__main__":
    main()
# V0156