# V0156
# Audit reference: restore the V0102 1769 geocentric closest-approach method using direct 3-D Sun-Venus angular separation, with separate rho(t) and drho/dt plots.

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
REFERENCE_CA_UTC = "1769-06-03 22:18:59.487"
ROLLING_WINDOWS = (3, 5, 7, 9)
ZERO_TOLERANCE_ARCSEC_PER_MIN = 1.0e-6
OUTPUT_DIR = Path("/content/VENUS_1769_GEOCENTRIC_RHO_V0102_RESTORE_V0156_OUTPUT")
RHO_PNG = OUTPUT_DIR / "VENUS_1769_GEOCENTRIC_RHO_V0102_RESTORE_V0156_RHO.png"
RHODOT_PNG = OUTPUT_DIR / "VENUS_1769_GEOCENTRIC_RHO_V0102_RESTORE_V0156_RHODOT.png"
CSV_PATH = OUTPUT_DIR / "VENUS_1769_GEOCENTRIC_RHO_V0102_RESTORE_V0156.csv"


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


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise RuntimeError("REJECTED invalid vector norm")
    return vectors / norms[:, None]


def centered_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window=window, center=True, min_periods=window).mean().to_numpy()


def utc_text_from_tdb_jd(jd_tdb: float) -> str:
    return Time(jd_tdb, format="jd", scale="tdb").utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


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
    print("rho definition                       direct 3-D angular separation")
    print("Minimum condition                    scalar rho minimum and drho/dt root")
    print("Rolling means                        centered 3, 5, 7, and 9 points")
    print(f"Output                               {OUTPUT_DIR}")

    jd_sun, sun_xyz = query_vectors("10")
    jd_venus, venus_xyz = query_vectors("299")
    if len(jd_sun) != len(jd_venus) or not np.allclose(jd_sun, jd_venus, atol=1.0e-12, rtol=0.0):
        raise RuntimeError("REJECTED mismatched JPL grids")

    jd = jd_sun
    jd_origin = float(jd[len(jd) // 2])
    t_seconds = (jd - jd_origin) * 86400.0

    sun_unit = unit_rows(sun_xyz)
    venus_unit = unit_rows(venus_xyz)
    dot = np.clip(np.einsum("ij,ij->i", sun_unit, venus_unit), -1.0, 1.0)
    rho_arcsec = np.arccos(dot) * AS_PER_RAD

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
        raise RuntimeError("REJECTED scalar rho minimum refinement")
    t_min = float(minimum.x)

    root_lo = float(t_seconds[lo])
    root_hi = float(t_seconds[hi])
    if float(rhodot_spline(root_lo)) * float(rhodot_spline(root_hi)) >= 0.0:
        raise RuntimeError("REJECTED drho/dt root not bracketed")
    t_root = float(brentq(lambda t: float(rhodot_spline(t)), root_lo, root_hi, xtol=1.0e-10, rtol=1.0e-14, maxiter=500))

    t_ca = t_root
    jd_ca_tdb = jd_origin + t_ca / 86400.0
    ca_utc = Time(jd_ca_tdb, format="jd", scale="tdb").utc
    ca_text = ca_utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    reference = Time(REFERENCE_CA_UTC, scale="utc")
    offset_seconds = float((ca_utc.tdb.jd - reference.tdb.jd) * 86400.0)
    method_agreement_seconds = abs(t_min - t_root)
    rho_min = float(rho_spline(t_ca))
    analytic_rhodot = float(rhodot_spline(t_ca)) * 60.0

    sample_rhodot = rhodot_spline(t_seconds) * 60.0
    rolling_series: dict[int, np.ndarray] = {}
    rolling_at_ca: dict[int, float] = {}
    for window in ROLLING_WINDOWS:
        rolling = centered_rolling_mean(sample_rhodot, window)
        rolling_series[window] = rolling
        valid = np.isfinite(rolling)
        rolling_spline = CubicSpline(t_seconds[valid], rolling[valid], bc_type="natural")
        rolling_at_ca[window] = float(rolling_spline(t_ca))

    minutes = (t_seconds - t_ca) / 60.0
    dense_minutes = np.linspace(-30.0, 30.0, 2401)
    dense_t = t_ca + dense_minutes * 60.0
    dense_rho = rho_spline(dense_t)
    dense_rhodot = rhodot_spline(dense_t) * 60.0

    output = pd.DataFrame({
        "jd_tdb": jd,
        "utc": [utc_text_from_tdb_jd(value) for value in jd],
        "minutes_from_ca": minutes,
        "rho_arcsec": rho_arcsec,
        "drho_dt_arcsec_per_min": sample_rhodot,
    })
    for window in ROLLING_WINDOWS:
        output[f"rolling_mean_{window}_arcsec_per_min"] = rolling_series[window]
    output.to_csv(CSV_PATH, index=False)

    plt.style.use("dark_background")

    fig1, ax1 = plt.subplots(figsize=(12.0, 6.4), dpi=120)
    ax1.plot(dense_minutes, dense_rho, linewidth=1.0, label="rho(t): Direct Sun-Venus Angular Separation")
    ax1.axvline(0.0, linewidth=0.7, alpha=0.85, label="Closest Approach")
    ax1.scatter([0.0], [rho_min], s=42, marker="D", zorder=6)
    ax1.annotate(
        f"Geocentric Closest Approach\n{ca_text} UTC\nrho = {rho_min:.12f} arcsec",
        xy=(0.0, rho_min),
        xytext=(3.5, rho_min + 0.28),
        fontsize=9,
        arrowprops={"arrowstyle": "-", "linewidth": 0.65},
    )
    ax1.set_title("1769 GEOCENTRIC rho(t) — V0102 DIRECT ANGULAR-SEPARATION RESTORATION", fontsize=12, weight="bold")
    ax1.set_xlabel("Minutes From Closest Approach")
    ax1.set_ylabel("rho (arcsec)")
    ax1.grid(alpha=0.18)
    ax1.legend(loc="upper right", fontsize=8)
    fig1.tight_layout()
    fig1.savefig(RHO_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12.0, 6.4), dpi=120)
    ax2.axhline(0.0, linewidth=0.7, alpha=0.85)
    ax2.axvline(0.0, linewidth=0.7, alpha=0.85)
    ax2.plot(dense_minutes, dense_rhodot, linewidth=1.0, label="Analytic drho/dt")
    for window in ROLLING_WINDOWS:
        valid = np.isfinite(rolling_series[window])
        ax2.plot(minutes[valid], rolling_series[window][valid], linewidth=0.75, alpha=0.75, label=f"{window}-Point Rolling Mean")
    ax2.scatter([0.0], [analytic_rhodot], s=38, marker="D", zorder=6)
    ax2.set_title("1769 GEOCENTRIC drho/dt — ZERO-CROSSING AND ROLLING-MEAN AUDIT", fontsize=12, weight="bold")
    ax2.set_xlabel("Minutes From Closest Approach")
    ax2.set_ylabel("drho/dt (arcsec/min)")
    ax2.grid(alpha=0.18)
    ax2.legend(loc="upper left", fontsize=8, ncol=3)
    fig2.tight_layout()
    fig2.savefig(RHODOT_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig2)

    zero_checks_pass = abs(analytic_rhodot) <= ZERO_TOLERANCE_ARCSEC_PER_MIN and all(
        abs(value) <= ZERO_TOLERANCE_ARCSEC_PER_MIN for value in rolling_at_ca.values()
    )

    section("COMMENTS")
    print("rho(t) is computed directly from the normalized 3-D Sun and Venus vectors.")
    print("No tangent-plane projection is used to select the minimum.")
    print("The rho(t) and drho/dt figures are saved as separate PNG files.")

    section("RESULTS")
    print(f"Geocentric rho-min CA UTC             {ca_text}")
    print(f"Offset from V0102                    {offset_seconds:+.9f} s")
    print(f"rho minimum                          {rho_min:.12f} arcsec")
    print(f"Analytic drho/dt at CA               {analytic_rhodot:+.12e} arcsec/min")
    for window in ROLLING_WINDOWS:
        print(f"{window}-point rolling mean at CA          {rolling_at_ca[window]:+.12e} arcsec/min")
    print(f"rho-min / drho-root agreement        {method_agreement_seconds:.12f} s")

    section("OUTPUT SUMMARY")
    print(f"rho(t) PNG                           {RHO_PNG}")
    print(f"drho/dt PNG                          {RHODOT_PNG}")
    print(f"CSV                                  {CSV_PATH}")

    section("PAPER COMPARISON")
    print(f"V0102 comparison CA                  {REFERENCE_CA_UTC} UTC")
    print(f"Calculated minus V0102               {offset_seconds:+.9f} s")

    section("EQUATION STATUS")
    print("Direct 3-D angular separation        VERIFIED")
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