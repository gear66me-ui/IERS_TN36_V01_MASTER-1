# V0001
# Audit reference: Standalone IAU-1976, modern-IERS, and JPL-vector solar horizontal parallax audit.
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VERSION = "V0001"
PROGRAM = "SOLAR_PARALLAX_AUDIT_V01.py"
PROJECT = "1769 VENUS TRANSIT – SOLAR PARALLAX AUDIT V01"
LOCAL_TZ = ZoneInfo("America/Bogota")
ARCSEC_PER_RAD = 206264.80624709636
TARGET_ARCSEC = 8.794148
TOLERANCE_ARCSEC = 0.000007

C_M_S = 299792458.000000
TAU_A_S = 499.004782000
IAU1976_RADIUS_M = 6378140.000000
IAU1976_AU_M = 149597870000.000000
WGS84_RADIUS_M = 6378137.000000
IERS2010_RADIUS_M = 6378136.600000
IAU2012_AU_M = 149597870700.000000

DEFAULT_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/JPL - 1769 VENUS TRANSIT")
DEFAULT_MASTER = DEFAULT_ROOT / "O6 MASTER" / "O6_TAHITI_VARDO_1769_1MIN_MASTER.csv"
DEFAULT_OUTPUT = "SOLAR_PARALLAX_AUDIT_V01_OUTPUT"
VECTOR_COLUMNS = [
    "JD",
    "GEOCENTER_SUN_X_KM",
    "GEOCENTER_SUN_Y_KM",
    "GEOCENTER_SUN_Z_KM",
    "GEOCENTER_VENUS_X_KM",
    "GEOCENTER_VENUS_Y_KM",
    "GEOCENTER_VENUS_Z_KM",
]
RADIUS_MODES = {
    "IAU1976": IAU1976_RADIUS_M,
    "WGS84": WGS84_RADIUS_M,
    "IERS2010": IERS2010_RADIUS_M,
}


@dataclass(frozen=True)
class Case:
    case_id: str
    label: str
    radius_m: float
    distance_m: float
    distance_definition: str
    source: str
    epoch_utc: str = ""


@dataclass(frozen=True)
class JPLResult:
    source_file: Path
    jd: float
    epoch_utc: str
    sun_vector_km: np.ndarray
    distance_m: float
    sun_venus_separation_arcsec: float
    nearest_index: int
    offset_seconds: float
    fit_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=PROJECT)
    parser.add_argument("--jpl-csv", default=str(DEFAULT_MASTER))
    parser.add_argument(
        "--earth-radius-mode",
        choices=("IAU1976", "WGS84", "IERS2010", "CUSTOM"),
        default=os.environ.get("SOLAR_PARALLAX_EARTH_RADIUS_MODE", "WGS84").upper(),
    )
    parser.add_argument("--earth-radius-m", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dpi", type=int, default=420)
    return parser.parse_args()


def positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def radius_from_args(args: argparse.Namespace) -> tuple[str, float]:
    mode = args.earth_radius_mode.upper()
    if mode == "CUSTOM":
        if args.earth_radius_m is None:
            raise ValueError("CUSTOM mode requires --earth-radius-m.")
        return mode, positive("Custom Earth radius", args.earth_radius_m)
    return mode, RADIUS_MODES[mode]


def project_root() -> Path:
    if DEFAULT_ROOT.exists():
        return DEFAULT_ROOT
    return Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def output_dir(root: Path, requested: str | None) -> Path:
    path = Path(requested).expanduser().resolve() if requested else root / DEFAULT_OUTPUT
    path.mkdir(parents=True, exist_ok=True)
    return path


def valid_master(path: Path) -> bool:
    try:
        columns = pd.read_csv(path, nrows=0).columns
    except Exception:
        return False
    return all(name in columns for name in VECTOR_COLUMNS)


def locate_master(requested: str, root: Path) -> Path:
    direct = [Path(requested).expanduser(), DEFAULT_MASTER, root / "O6 MASTER" / DEFAULT_MASTER.name]
    for path in direct:
        if path.is_file() and valid_master(path):
            return path.resolve()
    search_roots = []
    for candidate in (root, Path("/content"), Path.cwd()):
        if candidate.exists():
            resolved = candidate.resolve()
            if resolved not in search_roots:
                search_roots.append(resolved)
    for search_root in search_roots:
        for pattern in ("O6_TAHITI_VARDO_1769_1MIN_MASTER.csv", "*1769*MASTER*.csv"):
            for path in sorted(search_root.rglob(pattern)):
                if path.is_file() and valid_master(path):
                    return path.resolve()
    attempted = "\n".join(f"  - {path}" for path in direct)
    raise FileNotFoundError(
        "No compatible 1769 JPL master CSV was found.\n"
        f"Direct paths checked:\n{attempted}\n"
        "Supply the project master with --jpl-csv PATH."
    )


def norm_rows(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.asarray(values, dtype=float) ** 2, axis=1))


def angle_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(norm_rows(np.cross(a, b)), np.sum(a * b, axis=1))


def angle_one(a: np.ndarray, b: np.ndarray) -> float:
    return math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b)))


def polynomial_vector(seconds: np.ndarray, vectors: np.ndarray, degree: int) -> list[np.poly1d]:
    return [np.poly1d(np.polyfit(seconds, vectors[:, axis], degree)) for axis in range(3)]


def evaluate(polynomials: list[np.poly1d], seconds: float) -> np.ndarray:
    return np.array([float(polynomial(seconds)) for polynomial in polynomials], dtype=float)


def golden_minimum(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    tolerance: float = 0.000001,
    iterations: int = 240,
) -> tuple[float, float]:
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    c = upper - ratio * (upper - lower)
    d = lower + ratio * (upper - lower)
    fc = function(c)
    fd = function(d)
    for _ in range(iterations):
        if upper - lower <= tolerance:
            break
        if fc < fd:
            upper, d, fd = d, c, fc
            c = upper - ratio * (upper - lower)
            fc = function(c)
        else:
            lower, c, fc = c, d, fd
            d = lower + ratio * (upper - lower)
            fd = function(d)
    x = 0.5 * (lower + upper)
    return x, function(x)


def jd_to_utc(julian_date: float) -> datetime:
    shifted = float(julian_date) + 0.5
    z = int(math.floor(shifted))
    fraction = shifted - z
    if z >= 2299161:
        alpha = int((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - int(alpha / 4)
    else:
        a = z
    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)
    day_decimal = b - d - int(30.6001 * e) + fraction
    day = int(math.floor(day_decimal))
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    microseconds = int(round((day_decimal - day) * 86400.0 * 1_000_000.0))
    return datetime(year, month, day, tzinfo=timezone.utc) + timedelta(microseconds=microseconds)


def utc_text(julian_date: float) -> str:
    return jd_to_utc(julian_date).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"


def derive_jpl(master: Path) -> JPLResult:
    frame = pd.read_csv(master)
    missing = [name for name in VECTOR_COLUMNS if name not in frame.columns]
    if missing:
        raise KeyError("Missing JPL columns: " + ", ".join(missing))
    numeric = frame[VECTOR_COLUMNS].apply(pd.to_numeric, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)]
    numeric = numeric.sort_values("JD").drop_duplicates("JD").reset_index(drop=True)
    if len(numeric) < 7:
        raise RuntimeError("At least seven finite JPL rows are required.")
    jd = numeric["JD"].to_numpy(dtype=float)
    if np.any(np.diff(jd) <= 0.0):
        raise RuntimeError("JPL epochs must be strictly increasing.")
    sun = numeric[["GEOCENTER_SUN_X_KM", "GEOCENTER_SUN_Y_KM", "GEOCENTER_SUN_Z_KM"]].to_numpy(dtype=float)
    venus = numeric[["GEOCENTER_VENUS_X_KM", "GEOCENTER_VENUS_Y_KM", "GEOCENTER_VENUS_Z_KM"]].to_numpy(dtype=float)
    nearest = int(np.argmin(angle_rows(sun, venus)))
    if nearest == 0 or nearest == len(numeric) - 1:
        raise RuntimeError("Transit closest approach lies at a JPL data boundary.")
    lo = max(0, nearest - 3)
    hi = min(len(numeric), nearest + 4)
    center_jd = jd[nearest]
    seconds = (jd[lo:hi] - center_jd) * 86400.0
    degree = min(3, len(seconds) - 1)
    sun_fit = polynomial_vector(seconds, sun[lo:hi], degree)
    venus_fit = polynomial_vector(seconds, venus[lo:hi], degree)

    def separation(offset: float) -> float:
        return angle_one(evaluate(sun_fit, offset), evaluate(venus_fit, offset))

    lower = (jd[nearest - 1] - center_jd) * 86400.0
    upper = (jd[nearest + 1] - center_jd) * 86400.0
    offset, separation_rad = golden_minimum(separation, lower, upper)
    sun_vector = evaluate(sun_fit, offset)
    selected_jd = center_jd + offset / 86400.0
    return JPLResult(
        source_file=master,
        jd=float(selected_jd),
        epoch_utc=utc_text(selected_jd),
        sun_vector_km=sun_vector,
        distance_m=float(np.linalg.norm(sun_vector)) * 1000.0,
        sun_venus_separation_arcsec=float(separation_rad) * ARCSEC_PER_RAD,
        nearest_index=nearest,
        offset_seconds=float(offset),
        fit_rows=len(seconds),
    )


def parallax(radius_m: float, distance_m: float) -> tuple[float, float]:
    ratio = positive("Earth radius", radius_m) / positive("Earth-Sun distance", distance_m)
    if not 0.0 < ratio < 1.0:
        raise ValueError("a/D must be between zero and one.")
    return ratio, math.asin(ratio)


def calculate(case: Case) -> dict[str, object]:
    ratio, radians = parallax(case.radius_m, case.distance_m)
    arcseconds = radians * ARCSEC_PER_RAD
    difference = arcseconds - TARGET_ARCSEC
    return {
        "case_id": case.case_id,
        "case": case.label,
        "source": case.source,
        "earth_radius_m": case.radius_m,
        "au_or_distance_m": case.distance_m,
        "distance_definition": case.distance_definition,
        "ratio_a_over_distance": ratio,
        "solar_parallax_rad": radians,
        "solar_parallax_deg": math.degrees(radians),
        "solar_parallax_arcmin": math.degrees(radians) * 60.0,
        "solar_parallax_arcsec": arcseconds,
        "difference_arcsec": difference,
        "difference_microarcsec": difference * 1_000_000.0,
        "percent_difference": difference / TARGET_ARCSEC * 100.0,
        "tolerance_arcsec": TOLERANCE_ARCSEC,
        "pass_fail": "PASS" if abs(difference) <= TOLERANCE_ARCSEC else "FAIL",
        "rounded_6dp_arcsec": round(arcseconds, 6),
        "epoch_utc": case.epoch_utc,
    }


def investigation(cases: pd.DataFrame, exact_au: float) -> pd.DataFrame:
    by_id = cases.set_index("case_id")
    p1 = float(by_id.loc["CASE_1", "solar_parallax_arcsec"])
    p2 = float(by_id.loc["CASE_2", "solar_parallax_arcsec"])
    p3 = float(by_id.loc["CASE_3", "solar_parallax_arcsec"])
    piers = float(by_id.loc["CASE_IERS", "solar_parallax_arcsec"])
    p_wgs84_old_au = parallax(WGS84_RADIUS_M, IAU1976_AU_M)[1] * ARCSEC_PER_RAD
    p_1976_new_au = parallax(IAU1976_RADIUS_M, IAU2012_AU_M)[1] * ARCSEC_PER_RAD
    target_rad = TARGET_ARCSEC / ARCSEC_PER_RAD
    implied_au_1976_radius = IAU1976_RADIUS_M / math.sin(target_rad)
    lower_round = TARGET_ARCSEC - 0.0000005
    upper_round = TARGET_ARCSEC + 0.0000005
    rows = [
        ("Exact c × τA product", exact_au, "m", "Direct multiplication of supplied defining constants."),
        ("Published AU minus exact c × τA", IAU1976_AU_M - exact_au, "m", "Whole-kilometre published AU rounding offset."),
        ("AU-rounding parallax contribution", (p1 - p2) * 1_000_000.0, "microarcsec", "Only AU changes; IAU-1976 radius is fixed."),
        ("IAU-1976 to WGS84 radius contribution", (p_wgs84_old_au - p1) * 1_000_000.0, "microarcsec", "Only radius changes; published 1976 AU is fixed."),
        ("Published 1976 AU to exact IAU-2012 AU contribution", (p_1976_new_au - p1) * 1_000_000.0, "microarcsec", "Only AU changes; IAU-1976 radius is fixed."),
        ("Combined IAU-1976 to IAU-2012/WGS84 shift", (p3 - p1) * 1_000_000.0, "microarcsec", "Net radius and AU convention change."),
        ("WGS84 to IERS-2010 radius shift", (piers - p3) * 1_000_000.0, "microarcsec", "0.4 m radius convention difference at exact IAU-2012 AU."),
        ("AU implied by exactly 8.794148 arcsec with IAU-1976 radius", implied_au_1976_radius, "m", "Inverse relation AU = a/sin(π)."),
        ("Six-decimal rounding interval lower edge", lower_round, "arcsec", "Inclusive lower half-unit boundary for ordinary rounding."),
        ("Six-decimal rounding interval upper edge", upper_round, "arcsec", "Exclusive upper half-unit boundary for ordinary rounding."),
        ("Published case rounded to six decimals", round(p1, 6), "arcsec", "Digit-for-digit historical result."),
        ("Exact c × τA case rounded to six decimals", round(p2, 6), "arcsec", "Unrounded product reaches the same printed result."),
    ]
    return pd.DataFrame(rows, columns=["investigation", "value", "unit", "interpretation"])


def jpl_audit(result: JPLResult, mode: str, radius_m: float) -> pd.DataFrame:
    vector = result.sun_vector_km
    rows = [
        ("JPL source file", str(result.source_file), ""),
        ("Closest-approach epoch", result.epoch_utc, "UTC"),
        ("Closest-approach Julian date", result.jd, "JD"),
        ("Nearest one-minute row", result.nearest_index, "zero-based index"),
        ("Sub-minute offset", result.offset_seconds, "s"),
        ("Polynomial-fit rows", result.fit_rows, "row"),
        ("Earth-Sun X", vector[0], "km"),
        ("Earth-Sun Y", vector[1], "km"),
        ("Earth-Sun Z", vector[2], "km"),
        ("Earth-Sun vector magnitude", result.distance_m / 1000.0, "km"),
        ("Earth-Sun vector magnitude", result.distance_m, "m"),
        ("Geocenter Sun-Venus separation", result.sun_venus_separation_arcsec, "arcsec"),
        ("Selected radius mode", mode, ""),
        ("Selected Earth equatorial radius", radius_m, "m"),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value", "unit"])


def equation_checks(cases: pd.DataFrame, exact_au: float, inv: pd.DataFrame, jpl: JPLResult) -> pd.DataFrame:
    indexed = cases.set_index("case_id")
    p1 = float(indexed.loc["CASE_1", "solar_parallax_arcsec"])
    p2 = float(indexed.loc["CASE_2", "solar_parallax_arcsec"])
    radius_shift = float(inv.loc[inv["investigation"] == "IAU-1976 to WGS84 radius contribution", "value"].iloc[0])
    au_shift = float(inv.loc[inv["investigation"] == "Published 1976 AU to exact IAU-2012 AU contribution", "value"].iloc[0])
    combined = float(inv.loc[inv["investigation"] == "Combined IAU-1976 to IAU-2012/WGS84 shift", "value"].iloc[0])
    vector_residual = jpl.distance_m - float(np.linalg.norm(jpl.sun_vector_km)) * 1000.0
    checks = [
        ("π⊙ = asin(a/D) domain", bool(((cases["ratio_a_over_distance"] > 0.0) & (cases["ratio_a_over_distance"] < 1.0)).all()), "Every case has 0 < a/D < 1."),
        ("AU = c × τA", abs(exact_au - C_M_S * TAU_A_S) <= 0.000001, f"Residual {exact_au - C_M_S * TAU_A_S:.12f} m."),
        ("Published case rounds to 8.794148 arcsec", round(p1, 6) == TARGET_ARCSEC, f"Unrounded {p1:.12f} arcsec."),
        ("Exact c × τA case rounds to 8.794148 arcsec", round(p2, 6) == TARGET_ARCSEC, f"Unrounded {p2:.12f} arcsec."),
        ("Modern-shift decomposition closes", abs(combined - radius_shift - au_shift) <= 0.000001, f"Residual {combined - radius_shift - au_shift:.9f} microarcsec."),
        ("JPL distance equals vector magnitude", abs(vector_residual) <= 0.000001, f"Residual {vector_residual:.12f} m."),
        ("JPL solution lies within adjacent minute rows", abs(jpl.offset_seconds) <= 61.0, f"Offset {jpl.offset_seconds:.9f} s."),
    ]
    return pd.DataFrame(
        [{"equation_or_check": name, "status": "PASS" if passed else "FAIL", "audit_detail": detail} for name, passed, detail in checks]
    )


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.15f")


def engineering_png(cases: pd.DataFrame, inv: pd.DataFrame, checks: pd.DataFrame, path: Path, dpi: int) -> None:
    case_display = cases[["case", "earth_radius_m", "au_or_distance_m", "solar_parallax_arcsec", "difference_microarcsec", "pass_fail"]].copy()
    formats = {
        "earth_radius_m": 3,
        "au_or_distance_m": 3,
        "solar_parallax_arcsec": 12,
        "difference_microarcsec": 6,
    }
    for column, decimals in formats.items():
        case_display[column] = case_display[column].map(lambda value, d=decimals: f"{float(value):.{d}f}")
    case_display.columns = ["Case", "a (m)", "AU / |rES| (m)", "π⊙ (arcsec)", "Δ (µas)", "±7 µas"]
    names = [
        "Exact c × τA product",
        "Published AU minus exact c × τA",
        "AU-rounding parallax contribution",
        "IAU-1976 to WGS84 radius contribution",
        "Published 1976 AU to exact IAU-2012 AU contribution",
        "Combined IAU-1976 to IAU-2012/WGS84 shift",
        "AU implied by exactly 8.794148 arcsec with IAU-1976 radius",
    ]
    inv_display = inv[inv["investigation"].isin(names)][["investigation", "value", "unit"]].copy()
    inv_display["value"] = inv_display["value"].map(lambda value: f"{float(value):.12f}")
    inv_display.columns = ["Investigation", "Value", "Unit"]
    check_display = checks[["equation_or_check", "status"]].copy()
    check_display.columns = ["Equation / check", "Status"]
    figure = plt.figure(figsize=(17, 12), constrained_layout=True)
    grid = figure.add_gridspec(3, 1, height_ratios=(1.25, 1.5, 1.0))
    axes = [figure.add_subplot(grid[i, 0]) for i in range(3)]
    for axis in axes:
        axis.axis("off")
    figure.suptitle(
        "1769 VENUS TRANSIT — SOLAR HORIZONTAL PARALLAX AUDIT\n"
        "IAU 1976 • exact cτA • IAU 2012/WGS84 • IERS 2010 • JPL dynamic",
        fontsize=15,
        fontweight="bold",
    )
    specifications = [
        (axes[0], case_display, "PRIMARY DERIVATION TABLE", [0.27, 0.13, 0.19, 0.16, 0.13, 0.08]),
        (axes[1], inv_display, "SOURCE-OF-DIFFERENCE AUDIT", [0.60, 0.22, 0.12]),
        (axes[2], check_display, "EQUATION STATUS", [0.76, 0.16]),
    ]
    for axis, frame, title, widths in specifications:
        axis.set_title(title, fontsize=11, fontweight="bold", pad=8)
        table = axis.table(cellText=frame.values, colLabels=frame.columns, cellLoc="left", colLoc="center", loc="center", colWidths=widths)
        table.auto_set_font_size(False)
        table.set_fontsize(8.0)
        table.scale(1.0, 1.55)
        for (row, _), cell in table.get_celld().items():
            cell.set_linewidth(0.35)
            if row == 0:
                cell.set_text_props(weight="bold")
    figure.savefig(path, dpi=max(180, int(dpi)), bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def value_text(value: object, decimals: int = 12) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{decimals}f}"
    return str(value)


def print_table(frame: pd.DataFrame, columns: list[tuple[str, str, int, int]]) -> None:
    print(" | ".join(header.ljust(width) for _, header, width, _ in columns))
    print("-+-".join("-" * width for _, _, width, _ in columns))
    for _, row in frame.iterrows():
        cells = []
        for column, _, width, decimals in columns:
            text = value_text(row[column], decimals)
            numeric = isinstance(row[column], (int, float, np.integer, np.floating))
            cells.append(text.rjust(width) if numeric else text[:width].ljust(width))
        print(" | ".join(cells))


def main() -> None:
    args = parse_args()
    mode, selected_radius = radius_from_args(args)
    root = project_root()
    out = output_dir(root, args.output_dir)
    master = locate_master(args.jpl_csv, root)
    jpl = derive_jpl(master)
    exact_au = C_M_S * TAU_A_S
    definitions = [
        Case("CASE_1", "CASE 1 — IAU 1976 (Published)", IAU1976_RADIUS_M, IAU1976_AU_M, "Published rounded AU", "IAU-1976 defining constants"),
        Case("CASE_2", "CASE 2 — IAU 1976 (Exact c × τA)", IAU1976_RADIUS_M, exact_au, "c × τA", "Direct product of supplied c and τA"),
        Case("CASE_3", "CASE 3 — IAU 2012 / WGS84", WGS84_RADIUS_M, IAU2012_AU_M, "Exact IAU-2012 defining AU", "IAU-2012 AU and WGS84 equatorial radius"),
        Case("CASE_IERS", "IERS 2010 / IAU 2012", IERS2010_RADIUS_M, IAU2012_AU_M, "Exact IAU-2012 defining AU", "IERS-2010 equatorial radius and IAU-2012 AU"),
        Case("CASE_4", f"CASE 4 — JPL Dynamic / {mode}", selected_radius, jpl.distance_m, "Magnitude of interpolated GEOCENTER_SUN vector", "JPL Horizons project vectors", jpl.epoch_utc),
    ]
    cases = pd.DataFrame([calculate(case) for case in definitions])
    inv = investigation(cases, exact_au)
    vectors = jpl_audit(jpl, mode, selected_radius)
    checks = equation_checks(cases, exact_au, inv, jpl)
    paths = {
        "Cases CSV": out / "SOLAR_PARALLAX_AUDIT_V01_CASES.csv",
        "Investigation CSV": out / "SOLAR_PARALLAX_AUDIT_V01_INVESTIGATION.csv",
        "JPL vector CSV": out / "SOLAR_PARALLAX_AUDIT_V01_JPL_VECTOR.csv",
        "Equation-status CSV": out / "SOLAR_PARALLAX_AUDIT_V01_EQUATION_STATUS.csv",
        "Engineering tables PNG": out / "SOLAR_PARALLAX_AUDIT_V01_ENGINEERING_TABLES.png",
    }
    save_csv(cases, paths["Cases CSV"])
    save_csv(inv, paths["Investigation CSV"])
    save_csv(vectors, paths["JPL vector CSV"])
    save_csv(checks, paths["Equation-status CSV"])
    engineering_png(cases, inv, checks, paths["Engineering tables PNG"], args.dpi)

    print(f"CODE OUTPUT: {VERSION}")
    print("\nCODE INPUTS")
    print(f"Program                         : {PROGRAM}")
    print(f"Project                         : {PROJECT}")
    print(f"JPL master CSV                  : {master}")
    print(f"Selected JPL Earth-radius mode  : {mode}")
    print(f"Selected JPL Earth radius       : {selected_radius:.6f} m")
    print(f"Output directory                : {out}")
    print("\nCOMMENTS")
    print("Solar horizontal parallax is calculated as π⊙ = asin(a / D).")
    print("The target 8.794148 arcsec is used only for comparison and pass/fail classification.")
    print("The JPL dynamic distance is the GEOCENTER_SUN vector magnitude at calculated closest approach.")
    print("No calculated parallax result is entered manually.")
    print("\nRESULTS")
    print_table(cases, [
        ("case_id", "ID", 9, 0),
        ("case", "CASE", 39, 0),
        ("earth_radius_m", "a (m)", 17, 6),
        ("au_or_distance_m", "AU / D (m)", 23, 6),
        ("ratio_a_over_distance", "a/D", 20, 15),
        ("solar_parallax_arcsec", "π⊙ (arcsec)", 18, 12),
        ("difference_microarcsec", "Δ (µas)", 14, 6),
        ("pass_fail", "STATUS", 7, 0),
    ])
    print("\nOUTPUT SUMMARY")
    for label, path in paths.items():
        print(f"{label:<31}: {path}")
    print("\nPAPER COMPARISON")
    p1 = float(cases.loc[cases["case_id"] == "CASE_1", "solar_parallax_arcsec"].iloc[0])
    p2 = float(cases.loc[cases["case_id"] == "CASE_2", "solar_parallax_arcsec"].iloc[0])
    print(f"Historical comparison target   : {TARGET_ARCSEC:.6f} arcsec")
    print(f"IAU-1976 published derivation  : {p1:.12f} arcsec")
    print(f"IAU-1976 exact c × τA          : {p2:.12f} arcsec")
    print(f"Published derivation, 6 dp     : {round(p1, 6):.6f} arcsec")
    print(f"Exact c × τA, 6 dp             : {round(p2, 6):.6f} arcsec")
    print("Conclusion                     : Both historical AU forms round to 8.794148 arcsec; AU rounding is sub-microarcsecond, while the Earth-radius convention controls most of the modern shift.")
    print("\nEQUATION STATUS")
    print_table(checks, [
        ("equation_or_check", "EQUATION / CHECK", 55, 0),
        ("status", "STATUS", 8, 0),
        ("audit_detail", "AUDIT DETAIL", 70, 0),
    ])
    failures = checks[checks["status"] != "PASS"]
    if not failures.empty:
        raise RuntimeError("Equation checks failed: " + ", ".join(failures["equation_or_check"]))
    print(f"LOCAL TIMESTAMP: {datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# V0001
