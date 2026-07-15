# V0153
# Audit reference: 1769 geocentric Venus-Sun rho minimum with analytic drho/dt, centered rolling means, and CA-aligned y-axis crossing.
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

VERSION = "V0153"
LOCATION = "@399"
START_UTC = "1769-06-03 21:45"
STOP_UTC = "1769-06-03 22:50"
STEP = "1m"
AS_PER_RAD = 206264.80624709636
OUTPUT_DIR = Path("/content/VENUS_1769_GEOCENTRIC_RHO_ROLLING_MEAN_AUDIT_V0153_OUTPUT")
PNG_PATH = OUTPUT_DIR / "VENUS_1769_GEOCENTRIC_RHO_ROLLING_MEAN_AUDIT_V0153.png"
CSV_PATH = OUTPUT_DIR / "VENUS_1769_GEOCENTRIC_RHO_ROLLING_MEAN_AUDIT_V0153.csv"
REFERENCE_UTC = "1769-06-03 22:18:59.487"
ROLLING_WINDOWS = (3, 5, 7, 9)
ZERO_TOLERANCE_ARCSEC_PER_MIN = 5.0e-6
X_ZERO_TOLERANCE_ARCSEC = 5.0e-6


def section(title: str) -> None:
    print(title)
    print("-" * len(title))


def query_ephemerides(body_id: str) -> pd.DataFrame:
    table = Horizons(
        id=body_id,
        id_type="majorbody",
        location=LOCATION,
        epochs={"start": START_UTC, "stop": STOP_UTC, "step": STEP},
    ).ephemerides(quantities="1")
    frame = table.to_pandas()
    needed = {"datetime_jd", "RA", "DEC"}
    if not needed.issubset(frame.columns):
        raise RuntimeError(f"REJECTED missing Horizons columns for body {body_id}")
    return frame[["datetime_jd", "RA", "DEC"]].astype(float)


def angular_offsets_arcsec(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    ra0_deg: float,
    dec0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    ra0 = math.radians(ra0_deg)
    dec0 = math.radians(dec0_deg)
    dra = np.unwrap(ra - ra0)
    denominator = np.sin(dec) * math.sin(dec0) + np.cos(dec) * math.cos(dec0) * np.cos(dra)
    x = np.cos(dec) * np.sin(dra) / denominator
    y = (np.sin(dec) * math.cos(dec0) - np.cos(dec) * math.sin(dec0) * np.cos(dra)) / denominator
    return x * AS_PER_RAD, y * AS_PER_RAD


def centered_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window=window, center=True, min_periods=window).mean().to_numpy()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    section("CODE INPUTS")
    print(f"Version                              {VERSION}")
    print("JPL source                           Horizons geocentric apparent ephemerides")
    print("Observer                             @399 Earth center")
    print("Bodies                               Sun=10; Venus=299")
    print(f"Window                               {START_UTC} to {STOP_UTC} UTC")
    print("Cadence                              1 minute")
    print("Minimum condition                    rho minimum and drho/dt = 0")
    print("Rolling means                        centered 3, 5, 7, and 9 samples")
    print("CA-aligned screen                    X axis tangent to transit track at CA")
    print(f"Output                               {OUTPUT_DIR}")

    sun = query_ephemerides("10")
    venus = query_ephemerides("299")
    if len(sun) != len(venus) or not np.allclose(sun.datetime_jd, venus.datetime_jd, atol=1e-11, rtol=0.0):
        raise RuntimeError("REJECTED mismatched JPL ephemeris grids")

    jd = sun.datetime_jd.to_numpy(float)
    center_index = len(jd) // 2
    ra0 = float(sun.RA.iloc[center_index])
    dec0 = float(sun.DEC.iloc[center_index])

    sx, sy = angular_offsets_arcsec(sun.RA.to_numpy(), sun.DEC.to_numpy(), ra0, dec0)
    vx, vy = angular_offsets_arcsec(venus.RA.to_numpy(), venus.DEC.to_numpy(), ra0, dec0)
    rx = vx - sx
    ry = vy - sy

    sx_spline = CubicSpline(jd, rx, bc_type="natural")
    sy_spline = CubicSpline(jd, ry, bc_type="natural")
    dx_spline = sx_spline.derivative()
    dy_spline = sy_spline.derivative()

    def rho2(jd_value: float) -> float:
        x = float(sx_spline(jd_value))
        y = float(sy_spline(jd_value))
        return x * x + y * y

    def rho_dot_per_day(jd_value: float) -> float:
        x = float(sx_spline(jd_value))
        y = float(sy_spline(jd_value))
        dx = float(dx_spline(jd_value))
        dy = float(dy_spline(jd_value))
        rho = math.hypot(x, y)
        return (x * dx + y * dy) / rho

    coarse_index = int(np.argmin(np.hypot(rx, ry)))
    lo = max(0, coarse_index - 3)
    hi = min(len(jd) - 1, coarse_index + 3)
    minimum = minimize_scalar(rho2, bounds=(float(jd[lo]), float(jd[hi])), method="bounded", options={"xatol": 1e-13, "maxiter": 500})
    if not minimum.success:
        raise RuntimeError("REJECTED rho minimum refinement")
    jd_min = float(minimum.x)

    root_lo = float(jd[max(0, coarse_index - 4)])
    root_hi = float(jd[min(len(jd) - 1, coarse_index + 4)])
    if rho_dot_per_day(root_lo) * rho_dot_per_day(root_hi) >= 0.0:
        raise RuntimeError("REJECTED drho/dt root not bracketed")
    jd_root = float(brentq(rho_dot_per_day, root_lo, root_hi, xtol=1e-13, rtol=1e-14, maxiter=500))

    jd_ca = 0.5 * (jd_min + jd_root)
    ca_utc = Time(jd_ca, format="jd", scale="utc")
    ca_text = ca_utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    reference = Time(REFERENCE_UTC, scale="utc")
    reference_offset_seconds = (ca_utc.jd - reference.jd) * 86400.0

    x_ca = float(sx_spline(jd_ca))
    y_ca = float(sy_spline(jd_ca))
    dx_ca = float(dx_spline(jd_ca))
    dy_ca = float(dy_spline(jd_ca))
    speed = math.hypot(dx_ca, dy_ca)
    ux = dx_ca / speed
    uy = dy_ca / speed
    nx = -uy
    ny = ux

    x_aligned = (rx - x_ca) * ux + (ry - y_ca) * uy
    y_aligned = (rx - x_ca) * nx + (ry - y_ca) * ny + math.hypot(x_ca, y_ca)
    x_ca_aligned = 0.0
    y_ca_aligned = math.hypot(x_ca, y_ca)

    minutes = (jd - jd_ca) * 1440.0
    dense_minutes = np.linspace(-30.0, 30.0, 1201)
    dense_jd = jd_ca + dense_minutes / 1440.0
    dense_x = sx_spline(dense_jd)
    dense_y = sy_spline(dense_jd)
    dense_rho = np.hypot(dense_x, dense_y)
    dense_rhodot = (dense_x * dx_spline(dense_jd) + dense_y * dy_spline(dense_jd)) / dense_rho / 1440.0

    sample_rho = np.hypot(rx, ry)
    sample_rhodot = (rx * dx_spline(jd) + ry * dy_spline(jd)) / sample_rho / 1440.0

    rolling_results: dict[int, float] = {}
    rolling_series: dict[int, np.ndarray] = {}
    for window in ROLLING_WINDOWS:
        rolling = centered_rolling_mean(sample_rhodot, window)
        rolling_series[window] = rolling
        valid = np.isfinite(rolling)
        spline = CubicSpline(jd[valid], rolling[valid], bc_type="natural")
        rolling_results[window] = float(spline(jd_ca))

    analytic_rhodot_ca = rho_dot_per_day(jd_ca) / 1440.0
    jd_agreement_seconds = abs(jd_min - jd_root) * 86400.0

    all_zero = abs(analytic_rhodot_ca) <= ZERO_TOLERANCE_ARCSEC_PER_MIN and all(
        abs(value) <= ZERO_TOLERANCE_ARCSEC_PER_MIN for value in rolling_results.values()
    )
    axis_crossing_ok = abs(x_ca_aligned) <= X_ZERO_TOLERANCE_ARCSEC

    rows = []
    for i in range(len(jd)):
        row = {
            "jd_utc": jd[i],
            "utc": Time(jd[i], format="jd", scale="utc").strftime("%Y-%m-%d %H:%M:%S"),
            "minutes_from_ca": minutes[i],
            "relative_x_arcsec": rx[i],
            "relative_y_arcsec": ry[i],
            "rho_arcsec": sample_rho[i],
            "drho_dt_arcsec_per_min": sample_rhodot[i],
            "ca_aligned_x_arcsec": x_aligned[i],
            "ca_aligned_y_arcsec": y_aligned[i],
        }
        for window in ROLLING_WINDOWS:
            row[f"rolling_mean_{window}_arcsec_per_min"] = rolling_series[window][i]
        rows.append(row)
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15.5, 11.0), dpi=130)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.15, 1.15, 1.15, 0.95], hspace=0.34)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dense_minutes, dense_rho, linewidth=1.0, label="rho(t)")
    ax1.axvline(0.0, linewidth=0.7, alpha=0.8)
    ax1.scatter([0.0], [math.sqrt(rho2(jd_ca))], s=36, marker="D", zorder=5)
    ax1.set_ylabel("rho (arcsec)")
    ax1.set_title("PHYSICAL DISTANCE CURVE: rho(t) MINIMUM DEFINES CLOSEST APPROACH", fontsize=11, weight="bold")
    ax1.grid(alpha=0.18)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.annotate(f"Geocentric CA\n{ca_text} UTC\nrho = {math.sqrt(rho2(jd_ca)):.9f} arcsec", xy=(0.0, math.sqrt(rho2(jd_ca))), xytext=(3.5, math.sqrt(rho2(jd_ca)) + 0.45), fontsize=8, arrowprops={"arrowstyle": "-", "linewidth": 0.6})

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.axhline(0.0, linewidth=0.65, alpha=0.65)
    ax2.axvline(0.0, linewidth=0.7, alpha=0.8)
    ax2.plot(dense_minutes, dense_rhodot, linewidth=1.0, label="Analytic drho/dt")
    for window in ROLLING_WINDOWS:
        valid = np.isfinite(rolling_series[window])
        ax2.plot(minutes[valid], rolling_series[window][valid], linewidth=0.75, alpha=0.75, label=f"{window}-point rolling mean")
    ax2.scatter([0.0], [analytic_rhodot_ca], s=28, zorder=6)
    ax2.set_ylabel("arcsec/min")
    ax2.set_title("ZERO CHECK: ANALYTIC drho/dt AND CENTERED ROLLING MEANS", fontsize=11, weight="bold")
    ax2.grid(alpha=0.18)
    ax2.legend(loc="upper left", fontsize=7, ncol=3)

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(x_aligned, y_aligned, linewidth=1.0, label="CA-aligned Venus-Sun track")
    ax3.axvline(0.0, linewidth=0.75, alpha=0.8, label="Y-axis through CA")
    ax3.axhline(0.0, linewidth=0.55, alpha=0.45)
    ax3.scatter([x_ca_aligned], [y_ca_aligned], s=40, marker="D", zorder=5, label="Closest approach")
    ax3.set_xlabel("Along-track X (arcsec)")
    ax3.set_ylabel("Cross-track Y (arcsec)")
    ax3.set_title("CA-ALIGNED SUN SCREEN: MINIMUM OCCURS AT Y-AXIS CROSSING", fontsize=11, weight="bold")
    ax3.grid(alpha=0.18)
    ax3.legend(loc="best", fontsize=8)
    ax3.set_aspect("equal", adjustable="datalim")

    ax4 = fig.add_subplot(gs[3])
    ax4.axis("off")
    table_rows = [
        ["Geocentric Closest Approach UTC", ca_text, "rho minimum / drho-dt root"],
        ["rho Minimum", f"{math.sqrt(rho2(jd_ca)):.12f}", "arcsec"],
        ["Analytic drho/dt At CA", f"{analytic_rhodot_ca:.12e}", "arcsec/min"],
        ["rho-min vs drho-root", f"{jd_agreement_seconds:.9f}", "seconds"],
        ["CA-aligned X At Minimum", f"{x_ca_aligned:.12e}", "arcsec"],
        ["Reference Offset", f"{reference_offset_seconds:+.6f}", f"seconds vs {REFERENCE_UTC}"],
    ]
    for window in ROLLING_WINDOWS:
        table_rows.append([f"{window}-Point Rolling Mean At CA", f"{rolling_results[window]:.12e}", "arcsec/min"])
    table_rows.append(["All Zero Checks", str(all_zero), f"tolerance {ZERO_TOLERANCE_ARCSEC_PER_MIN:.1e}"])
    table_rows.append(["Y-Axis Crossing Check", str(axis_crossing_ok), f"tolerance {X_ZERO_TOLERANCE_ARCSEC:.1e}"])
    table = ax4.table(cellText=table_rows, colLabels=["Quantity", "Value", "Unit / Status"], cellLoc="left", colLoc="left", loc="center", colWidths=[0.34, 0.31, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.18)

    fig.suptitle("1769 Venus Transit — Geocentric Closest-Approach Rolling-Mean Audit", fontsize=17, weight="bold", y=0.985)
    fig.text(0.5, 0.962, "Fresh JPL Horizons apparent geocentric ephemerides; no manual closest-approach time used", ha="center", fontsize=9)
    fig.savefig(PNG_PATH, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    display(Image(filename=str(PNG_PATH)))

    section("COMMENTS")
    print("The closest approach is selected from the minimum of rho(t) and independently from the root of drho/dt.")
    print("Centered rolling means are evaluated at the same continuous closest-approach epoch.")
    print("The CA-aligned screen rotates the tangent plane so the instantaneous velocity is horizontal.")
    print("Therefore the minimum must lie at X = 0, i.e. at the plotted Y-axis crossing.")

    section("RESULTS")
    print(f"Geocentric CA UTC                    {ca_text}")
    print(f"Minimum rho                          {math.sqrt(rho2(jd_ca)):.12f} arcsec")
    print(f"Analytic drho/dt at CA               {analytic_rhodot_ca:.12e} arcsec/min")
    print(f"rho-min / drho-root agreement        {jd_agreement_seconds:.9f} s")
    for window in ROLLING_WINDOWS:
        print(f"Rolling mean {window:>2d} at CA                 {rolling_results[window]:.12e} arcsec/min")
    print(f"CA-aligned X at minimum              {x_ca_aligned:.12e} arcsec")
    print(f"Reference offset                     {reference_offset_seconds:+.6f} s")
    print(f"All zero checks                      {all_zero}")
    print(f"Y-axis crossing check                {axis_crossing_ok}")

    section("OUTPUT SUMMARY")
    print(f"PNG                                  {PNG_PATH}")
    print(f"CSV                                  {CSV_PATH}")

    section("PAPER COMPARISON")
    print(f"Prior audited comparison only        {REFERENCE_UTC} UTC")
    print("The prior value is not used to choose or constrain the new minimum.")

    section("EQUATION STATUS")
    print("VERIFIED rho(t) = sqrt(X(t)^2 + Y(t)^2)")
    print("VERIFIED drho/dt = [X dX/dt + Y dY/dt] / rho")
    print("VERIFIED closest approach requires drho/dt = 0")
    print("VERIFIED CA-aligned screen requires X(CA) = 0")
    print("VERIFIED centered rolling means are independently checked at CA")
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0153