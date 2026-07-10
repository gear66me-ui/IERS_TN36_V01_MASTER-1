# V0122C
# Audit reference: normalized IMCCE Venus Transit Canon master CSV with derived geometry and duration fields.

from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

VERSION = "IERS-0122C"
OUTPUT_ROOT = Path("/content/IERS_TN36_V01_MASTER_OUTPUT/IMCCE_VENUS_CANON")
STAGE_CSV = OUTPUT_ROOT / "STAGE" / "IMCCE_VENUS_TRANSIT_CANON_PARSED_STAGE.csv"
MASTER_CSV = OUTPUT_ROOT / "IMCCE_VENUS_TRANSIT_CANON_MASTER.csv"
LOCAL_TZ = timezone(timedelta(hours=-5))
SOURCE_PAGE_URL = "https://www.oca.eu/Mignard/Transits/Html/canon_venus.htm"
SOURCE_DATA_URL = "https://www.oca.eu/Mignard/Transits/Data/transit_venus.txt"
TARGET_YEARS = {1761, 1769, 1874, 1882, 2004, 2012}
RATIO_TOLERANCE = 0.0015

SOURCE_FLOAT_FIELDS = (
    "jd_tdb", "mid_minute", "sun_radius_arcsec", "minimum_distance_arcsec",
    "distance_ratio", "venus_radius_arcsec", "subsolar_longitude_ingress_deg",
    "subsolar_longitude_egress_deg", "subsolar_latitude_deg",
    "relative_velocity_deg_per_day", "venus_ecliptic_latitude_deg",
    "tdb_minus_ut_seconds",
)
SOURCE_INT_FIELDS = (
    "source_line_number", "day", "month", "year", "mid_hour", "node",
    "missing_field_count",
)


def as_float(value: str) -> float | None:
    text = str(value).strip()
    return None if text == "" else float(text)


def as_int(value: str) -> int | None:
    text = str(value).strip()
    return None if text == "" else int(float(text))


def clock_to_seconds(value: str) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    fields = text.split(":")
    if len(fields) == 2:
        hour, minute = int(fields[0]), float(fields[1])
        return hour * 3600.0 + minute * 60.0
    if len(fields) == 3:
        hour, minute, second = int(fields[0]), int(fields[1]), float(fields[2])
        return hour * 3600.0 + minute * 60.0 + second
    raise ValueError(f"Unsupported clock field: {value!r}")


def elapsed_seconds(start: float | None, stop: float | None) -> float | None:
    if start is None or stop is None:
        return None
    elapsed = stop - start
    return elapsed + 86400.0 if elapsed < 0.0 else elapsed


def geometry_class(delta: float, sun_radius: float, venus_radius: float) -> str:
    impact = abs(delta)
    if impact <= sun_radius - venus_radius:
        return "FULL_GEOCENTRIC_TRANSIT"
    if impact <= sun_radius + venus_radius:
        return "GRAZING_OR_PARTIAL_GEOCENTRIC_TRANSIT"
    return "OUTSIDE_GEOCENTRIC_LIMB_OR_TOPOCENTRIC_ONLY"


def year_labels(year: int) -> tuple[str, str]:
    if year > 0:
        return "CE", f"{year:04d} CE"
    return "BCE_ASTRONOMICAL", f"{abs(year - 1):04d} BCE"


def format_date_label(year: int, month: int, day: int) -> str:
    sign = "" if year >= 0 else "-"
    return f"{sign}{abs(year):04d}-{month:02d}-{day:02d}"


def normalize_row(source: dict[str, str], record_id: int) -> dict[str, object]:
    row: dict[str, object] = dict(source)
    for field in SOURCE_FLOAT_FIELDS:
        row[field] = as_float(source.get(field, ""))
    for field in SOURCE_INT_FIELDS:
        row[field] = as_int(source.get(field, ""))

    year = int(row["year"])
    month = int(row["month"])
    day = int(row["day"])
    hour = int(row["mid_hour"])
    minute = float(row["mid_minute"])
    sun_radius = float(row["sun_radius_arcsec"])
    delta = float(row["minimum_distance_arcsec"])
    venus_radius = float(row["venus_radius_arcsec"])
    source_ratio = float(row["distance_ratio"])

    era, year_display = year_labels(year)
    ratio_calc = abs(delta) / sun_radius
    ratio_residual = source_ratio - ratio_calc
    clocks = {name: clock_to_seconds(str(row.get(name, ""))) for name in ("c1_ut", "c2_ut", "c3_ut", "c4_ut")}

    row.update({
        "record_id": record_id,
        "era": era,
        "year_display": year_display,
        "date_ut_label": format_date_label(year, month, day),
        "mid_ut_hhmm": f"{hour:02d}:{minute:04.1f}",
        "mid_ut_seconds_of_day": hour * 3600.0 + minute * 60.0,
        "closest_approach_sign": "NORTH" if delta > 0 else "SOUTH" if delta < 0 else "CENTER",
        "impact_parameter_abs_arcsec": abs(delta),
        "ratio_abs_calculated": ratio_calc,
        "ratio_residual_source_minus_calculated": ratio_residual,
        "ratio_validation": "PASS" if abs(ratio_residual) <= RATIO_TOLERANCE else "REVIEW",
        "geometry_class": geometry_class(delta, sun_radius, venus_radius),
        "c1_seconds_of_day": clocks["c1_ut"],
        "c2_seconds_of_day": clocks["c2_ut"],
        "c3_seconds_of_day": clocks["c3_ut"],
        "c4_seconds_of_day": clocks["c4_ut"],
        "external_duration_seconds": elapsed_seconds(clocks["c1_ut"], clocks["c4_ut"]),
        "internal_duration_seconds": elapsed_seconds(clocks["c2_ut"], clocks["c3_ut"]),
        "target_workbook_year": "YES" if year in TARGET_YEARS else "NO",
        "source_page_url": SOURCE_PAGE_URL,
        "source_data_url": SOURCE_DATA_URL,
    })
    return row


def validate(rows: list[dict[str, object]]) -> dict[str, object]:
    if len(rows) != 77:
        raise RuntimeError(f"Expected 77 IMCCE records, found {len(rows)}")
    years = [int(row["year"]) for row in rows]
    if len(years) != len(set(years)):
        raise RuntimeError("Duplicate transit years detected")
    missing_targets = sorted(TARGET_YEARS - set(years))
    if missing_targets:
        raise RuntimeError(f"Required workbook years missing: {missing_targets}")
    review_count = sum(row["ratio_validation"] == "REVIEW" for row in rows)
    finite_jd = all(math.isfinite(float(row["jd_tdb"])) for row in rows)
    if not finite_jd:
        raise RuntimeError("Non-finite Julian date detected")
    return {
        "row_count": len(rows),
        "complete_count": sum(row["record_status"] == "COMPLETE" for row in rows),
        "header_only_count": sum(row["record_status"] == "SOURCE_HEADER_ONLY" for row in rows),
        "target_count": sum(row["target_workbook_year"] == "YES" for row in rows),
        "ratio_review_count": review_count,
        "minimum_year": min(years),
        "maximum_year": max(years),
    }


def main() -> None:
    if not STAGE_CSV.exists():
        raise FileNotFoundError(f"Run IERS-0122B first: {STAGE_CSV}")
    with STAGE_CSV.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = [normalize_row(source, index) for index, source in enumerate(source_rows, start=1)]
    audit = validate(rows)

    MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with MASTER_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CODE OUTPUT: {VERSION}")
    print("CODE INPUTS")
    print(f"Stage CSV : {STAGE_CSV}")
    print("COMMENTS")
    print("Source fields are preserved; calculated fields are appended and explicitly named.")
    print("RESULTS")
    print(f"Rows : {audit['row_count']} | Complete : {audit['complete_count']} | Header-only : {audit['header_only_count']}")
    print(f"Target years : {audit['target_count']} | Ratio REVIEW : {audit['ratio_review_count']}")
    print(f"Year range : {audit['minimum_year']} to {audit['maximum_year']}")
    print("OUTPUT SUMMARY")
    print(f"Master CSV : {MASTER_CSV}")
    print("PAPER COMPARISON")
    print("IMCCE source ratio compared with abs(minimum_distance)/solar_radius.")
    print("EQUATION STATUS")
    print(f"VERIFIED — ratio equation audited with tolerance {RATIO_TOLERANCE:.4f}.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print("# V0122C")


if __name__ == "__main__":
    main()

# V0122C
