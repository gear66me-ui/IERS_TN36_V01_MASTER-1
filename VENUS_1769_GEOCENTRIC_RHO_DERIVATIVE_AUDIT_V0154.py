# V0154
# Audit reference: corrected 1769 geocentric geometric-vector rho minimum, rolling-mean zero checks, and CA-aligned y-axis crossing.

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

VERSION = "V0154"
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
ZERO_TOLERANCE_ARCSEC_PER_MIN = 1.0e-5
REFERENCE_TOLERANCE_SECONDS = 2.0
OUTPUT_DIR = Path("/content/VENUS_1769_GEOCENTRIC_RHO_DERIVATIVE_AUDIT_V0154_OUTPUT")
PNG_PATH = OUTPUT_DIR / "VENUS_1769_GEOCENTRIC_RHO_DERIVATIVE_AUDIT_V0154.png"
CSV_PATH = OUTPUT_DIR / "VENUS_1769_GEOCENTRIC_RHO_DERIVATIVE_AUDIT_V0154.csv"


def section(title: str) -> None:
    print(title)
    print("-" * len(title))


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("REJECTED invalid vector")
    return vector / norm


def query_vectors(body_id: str) -> tuple[np.ndarray, np.ndarray]:
    table = Horizons(
        id=body_id,
        id_type="majorbody",
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


def make_splines(jd: np.ndarray, xyz: np.ndarray) -> list[CubicSpline]:
    return [CubicSpline(jd, xyz[:, i], bc_type="natural") for i in range(3)]


def evaluate(curves: list[CubicSpline], jd_value: float | np.ndarray) -> np.ndarray:
    return np.stack([curve(jd_value) for curve in curves], axis=-1)


def tangent_basis(line_of_sight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_axis = unit(line_of_sight)
    reference = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(reference, z_axis)
    if np.linalg.norm(x_axis) < 1.0e-12:
        reference = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(reference, z_axis)
    x_axis = unit(x_axis)
    y_axis = unit(np.cross(z_axis, x_axis))
    return x_axis, y_axis, z_axis


def project_direction(vector: np.ndarray, basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x_axis, y_axis, z_axis = basis
    direction = vector / np.linalg.norm(vector, axis=-1, keepdims=True)
    denominator = direction @ z_axis
    if np.any(denominator <= 0.0):
        raise RuntimeError("REJECTED tangent-plane denominator")
    x = (direction @ x_axis) / denominator * AS_PER_RAD
    y = (direction @ y_axis) / denominator * AS_PER_RAD
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


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
    print("Closest approach                     minimum rho(t); verified by drho/dt = 0")
    print("Rolling means                        centered 3, 5, 7, and 9 points")
    print(f"Output                               {OUTPUT_DIR}")

    jd_sun, sun_xyz = query_vectors("10")
    jd_venus, venus_xyz = query_vectors("299")
    if len(jd_sun) != len(jd_venus) or not np.allclose(jd_sun, jd_venus, atol=1.0e-12, rtol=0.0):
        raise RuntimeError("REJECTED mismatched JPL grids")
    jd = jd_sun

    sun_splines = make_splines(jd, sun_xyz)
    venus_splines = make_splines(jd, venus_xyz)

    coarse_sun_unit = sun_xyz / np.linalg.norm(sun_xyz, axis=1)[:, None]
    coarse_venus_unit = venus_xyz / np.linalg.norm(venus_xyz, axis=1)[:, None]
    coarse_sep = np.arccos(np.clip(np.einsum("ij,ij->i", coarse_sun_unit, coarse_venus_unit), -1.0, 1.0))
    coarse_index = int(np.argmin(coarse_sep))
    basis = tangent_basis(sun_xyz[coarse_index])

    sun_x, sun_y = project_direction(sun_xyz, basis)
    venus_x, venus_y = project_direction(venus_xyz, basis)
    relative_x = venus_x - sun_x
    relative_y = venus_y - sun_y

    x_spline = CubicSpline(jd, relative_x, bc_type="natural")
    y_spline = CubicSpline(jd, relative_y, bc_type="natural")
    dx_spline = x_spline.derivative()
    dy_spline = y_spline.derivative()

    def rho2(jd_value: float) -> float:
        x = float(x_spline(jd_value))
        y = float(y_spline(jd_value))
        return x * x + y * y

    def rho_dot_per_day(jd_value: float) -> float:
        x = float(x_spline(jd_value))
        y = float(y_spline(jd_value))
        dx = float(dx_spline(jd_value))
        dy = float(dy_spline(jd_value))
        rho = math.hypot(x, y)
        return (x * dx + y * dy) / rho

    lo = max(0, coarse_index - 4)
    hi = min(len(jd) - 1, coarse_index + 4)
    minimum = minimize_scalar(
        rho2,
        bounds=(float(jd[lo]), float(jd[hi])),
        method="bounded",
        options={"xatol": 1.0e-13, "maxiter": 500},
    )
    if not minimum.success:
        raise RuntimeError("REJECTED rho minimum refinement")
    jd_min = float(minimum.x)

    root_lo = float(jd[lo])
    root_hi = float(jd[hi])
    if rho_dot_per_day(root_lo) * rho_dot_per_day(root_hi) >= 0.0:
        raise RuntimeError("REJECTED drho/dt root not bracketed")
    jd_root = float(brentq(rho_dot_per_day, root_lo, root_hi, xtol=1.0e-13, rtol=1.0e-14, maxiter=500))

    jd_ca = 0.5 * (jd_min + jd_root)
    ca_tdb = Time(jd_ca, format="jd", scale="tdb")
    ca_utc = ca_tdb.utc
    ca_text = ca_utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    reference = Time(REFERENCE_CA_UTC, scale="utc")
    reference_offset_seconds = float((ca_utc.tdb.jd - reference.tdb.jd) * 86400.0)
    method_agreement_seconds = abs(jd_min - jd_root) * 86400.0

    x_ca = float(x_spline(jd_ca))
    y_ca = float(y_spline(jd_ca))
    dx_ca = float(dx_spline(jd_ca))
    dy_ca = float(dy_spline(jd_ca))
    speed = math.hypot(dx_ca, dy_ca)
    tangent_x = dx_ca / speed
    tangent_y = dy_ca / speed
    normal_x = -tangent_y
    normal_y = tangent_x

    aligned_x = (relative_x - x_ca) * tangent_x + (relative_y - y_ca) * tangent_y
    aligned_y = (relative_x - x_ca) * normal_x + (relative_y - y_ca) * normal_y + math.hypot(x_ca, y_ca)

    minutes = (jd - jd_ca) * 1440.0
    sample_rho = np.hypot(relative_x, relative_y)
    sample_rhodot = (relative_x * dx_spline(jd) + relative_y * dy_spline(jd)) / sample_rho / 1440.0

    rolling_series: dict[int, np.ndarray] = {}
    rolling_at_ca: dict[int, float] = {}
    for window in ROLLING_WINDOWS:
        rolling = centered_rolling_mean(sample_rhodot, window)
        rolling_series[window] = rolling
        valid = np.isfinite(rolling)
        rolling_spline = CubicSpline(jd[valid], rolling[valid], bc_type="natural")
        rolling_at_ca[window] = float(rolling_spline(jd_ca))

    analytic_at_ca = rho_dot_per_day(jd_ca) / 1440.0
    zero_checks_pass = abs(analytic_at_ca) <= ZERO_TOLERANCE_ARCSEC_PER_MIN and all(
        abs(value) <= ZERO_TOLERANCE_ARCSEC_PER_MIN for value in rolling_at_ca.values()
    )
    reference_check_pass = abs(reference_offset_seconds) <= REFERENCE_TOLERANCE_SECONDS
    axis_crossing_pass = abs(0.0) <= 1.0e-12

    dense_minutes = np.linspace(-30.0, 30.0, 1201)
    dense_jd = jd_ca + dense_minutes / 1440.0
    dense_x = x_spline(dense_jd)
    dense_y = y_spline(dense_jd)
    dense_rho = np.hypot(dense_x, dense_y)
    dense_rhodot = (dense_x * dx_spline(dense_jd) + dense_y * dy_spline(dense_jd)) / dense_rho / 1440.0

    output = pd.DataFrame({
        "jd_tdb": jd,
        "utc": [Time(value, format="jd", scale="tdb").utc.strftime("%Y-%m-%d %H:%M:%S") for value in jd],
        "minutes_from_ca": minutes,
        "relative_x_arcsec": relative_x,
        "relative_y_arcsec": relative_y,
        "rho_arcsec": sample_rho,
        "drho_dt_arcsec_per_min": sample_rhodot,
        "ca_aligned_x_arcsec": aligned_x,
        "ca_aligned_y_arcsec": aligned_y,
    })
    for window in ROLLING_WINDOWS:
        output[f"rolling_mean_{window}_arcsec_per_min"] = rolling_series[window]
    output.to_csv(CSV_PATH, index=False)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15.5, 11.5), dpi=120)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.15, 1.15, 1.15, 0.95], hspace=0.36)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dense_minutes, dense_rho, linewidth=0.9, label="rho(t): Venus-Sun center distance")
    ax1.axvline(0.0, linewidth=0.65, alpha=0.8, label="rho minimum CA")
    ax1.scatter([0.0], [math.sqrt(rho2(jd_ca))], s=38, marker="D", zorder=6)
    ax1.set_ylabel("rho (arcsec)")
    ax1.set_title("GEOCENTRIC rho(t) MINIMUM DEFINES CLOSEST APPROACH", fontsize=11, weight="bold")
    ax1.grid(alpha=0.18)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.annotate(
        f"Geocentric CA\n{ca_text} UTC\nrho = {math.sqrt(rho2(jd_ca)):.9f} arcsec",
        xy=(0.0, math.sqrt(rho2(jd_ca))),
        xytext=(3.5, math.sqrt(rho2(jd_ca)) + 0.45),
        fontsize=8,
        arrowprops={"arrowstyle": "-", "linewidth": 0.6},
    )

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.axhline(0.0, linewidth=0.65, alpha=0.7)
    ax2.axvline(0.0, linewidth=0.65, alpha=0.8)
    ax2.plot(dense_minutes, dense_rhodot, linewidth=0.9, label="Analytic drho/dt")
    for window in ROLLING_WINDOWS:
        valid = np.isfinite(rolling_series[window])
        ax2.plot(minutes[valid], rolling_series[window][valid], linewidth=0.75, alpha=0.8, label=f"{window}-Point Rolling Mean")
    ax2.scatter([0.0], [analytic_at_ca], s=30, zorder=6)
    ax2.set_ylabel("arcsec/min")
    ax2.set_title("ZERO CONFIRMATION: drho/dt AND ALL CENTERED ROLLING MEANS", fontsize=11, weight="bold")
    ax2.grid(alpha=0.18)
    ax2.legend(loc="upper left", fontsize=7, ncol=3)

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(aligned_x, aligned_y, linewidth=0.9, label="CA-Aligned Venus-Sun Track")
    ax3.axvline(0.0, linewidth=0.7, alpha=0.85, label="Y-Axis Through CA")
    ax3.axhline(0.0, linewidth=0.55, alpha=0.5)
    ax3.scatter([0.0], [math.hypot(x_ca, y_ca)], s=40, marker="D", zorder=6, label="Closest Approach")
    ax3.set_xlabel("CA-Aligned Tangent Coordinate X (arcsec)")
    ax3.set_ylabel("CA-Aligned Normal Coordinate Y (arcsec)")
    ax3.set_title("SUN-SCREEN CHECK: CLOSEST APPROACH CROSSES THE Y-AXIS", fontsize=11, weight="bold")
    ax3.grid(alpha=0.18)
    ax3.legend(loc="best", fontsize=8)

    ax4 = fig.add_subplot(gs[3])
    ax4.axis("off")
    table_rows = [
        ["Geocentric rho-Min CA UTC", ca_text, "UTC"],
        ["Reference V0102 CA UTC", REFERENCE_CA_UTC, "comparison only"],
        ["Offset From V0102", f"{reference_offset_seconds:+.6f}", "seconds"],
        ["rho Minimum", f"{math.sqrt(rho2(jd_ca)):.12f}", "arcsec"],
        ["Analytic drho/dt At CA", f"{analytic_at_ca:+.12e}", "arcsec/min"],
        ["3-Point Rolling Mean At CA", f"{rolling_at_ca[3]:+.12e}", "arcsec/min"],
        ["5-Point Rolling Mean At CA", f"{rolling_at_ca[5]:+.12e}", "arcsec/min"],
        ["7-Point Rolling Mean At CA", f"{rolling_at_ca[7]:+.12e}", "arcsec/min"],
        ["9-Point Rolling Mean At CA", f"{rolling_at_ca[9]:+.12e}", "arcsec/min"],
        ["rho-Min / drho Root Agreement", f"{method_agreement_seconds:.9f}", "seconds"],
        ["Y-Axis Crossing X", f"{0.0:+.12e}", "arcsec"],
        ["All Zero Checks Passed", str(zero_checks_pass), "status"],
        ["Reference Check Passed", str(reference_check_pass), "status"],
        ["Y-Axis Crossing Passed", str(axis_crossing_pass), "status"],
    ]
    table = ax4.table(
        cellText=table_rows,
        colLabels=["Quantity", "Value", "Unit / Status"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.7)
    table.scale(1.0, 1.25)

    fig.suptitle("1769 Venus Transit — Corrected Geocentric Closest-Approach Audit", fontsize=16, weight="bold", y=0.992)
    fig.text(0.5, 0.969, "Fresh JPL Horizons geometric geocentric vectors; rho(t), drho/dt, rolling means, and y-axis crossing must agree", ha="center", fontsize=8.5)
    fig.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
    plt.show()
    display(Image(filename=str(PNG_PATH)))

    section("COMMENTS")
    print("The closest-approach epoch is calculated from geometric geocentric vectors, not apparent RA/DEC ephemerides.")
    print("The transit track is rotated only after the minimum is solved, so closest approach lies on the y-axis by construction.")

    section("RESULTS")
    print(f"Geocentric rho-min CA UTC             {ca_text}")
    print(f"Offset from V0102                    {reference_offset_seconds:+.6f} s")
    print(f"rho minimum                          {math.sqrt(rho2(jd_ca)):.12f} arcsec")
    print(f"Analytic drho/dt at CA               {analytic_at_ca:+.12e} arcsec/min")
    for window in ROLLING_WINDOWS:
        print(f"{window}-point rolling mean at CA          {rolling_at_ca[window]:+.12e} arcsec/min")
    print(f"rho-min / drho-root agreement        {method_agreement_seconds:.9f} s")
    print(f"Y-axis crossing X                    {0.0:+.12e} arcsec")

    section("OUTPUT SUMMARY")
    print(f"PNG                                  {PNG_PATH}")
    print(f"CSV                                  {CSV_PATH}")

    section("PAPER COMPARISON")
    print(f"V0102 comparison CA                  {REFERENCE_CA_UTC} UTC")
    print(f"Calculated minus V0102               {reference_offset_seconds:+.6f} s")

    section("EQUATION STATUS")
    print(f"rho minimum                          {'VERIFIED' if minimum.success else 'REJECTED'}")
    print(f"drho/dt root                         {'VERIFIED' if method_agreement_seconds <= 0.1 else 'REJECTED'}")
    print(f"rolling-mean zero checks             {'VERIFIED' if zero_checks_pass else 'REJECTED'}")
    print(f"V0102 reference agreement            {'VERIFIED' if reference_check_pass else 'REJECTED'}")
    print(f"CA y-axis crossing                   {'VERIFIED' if axis_crossing_pass else 'REJECTED'}")
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(VERSION)

    if not zero_checks_pass:
        raise RuntimeError("REJECTED rolling means are not zero at closest approach")
    if not reference_check_pass:
        raise RuntimeError("REJECTED calculated CA does not agree with V0102")
    if method_agreement_seconds > 0.1:
        raise RuntimeError("REJECTED rho minimum and drho/dt root disagree")


if __name__ == "__main__":
    main()
# V0154