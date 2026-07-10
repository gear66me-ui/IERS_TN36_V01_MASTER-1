# V0122B
# Audit reference: strict field parser for the IMCCE Venus Transit Canon records; contact clocks support HH:MM:SS.

from __future__ import annotations

import csv
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

VERSION = "IERS-0122B"
SOURCE_URL = "https://www.oca.eu/Mignard/Transits/Data/transit_venus.txt"
OUTPUT_ROOT = Path("/content/IERS_TN36_V01_MASTER_OUTPUT/IMCCE_VENUS_CANON")
SOURCE_TEXT = OUTPUT_ROOT / "SOURCE" / "IMCCE_VENUS_TRANSIT_CANON_SOURCE.txt"
STAGE_DIR = OUTPUT_ROOT / "STAGE"
STAGE_CSV = STAGE_DIR / "IMCCE_VENUS_TRANSIT_CANON_PARSED_STAGE.csv"
LOCAL_TZ = timezone(timedelta(hours=-5))
USER_AGENT = "Mozilla/5.0 (compatible; IERS-TN36-IMCCE-Parser/1.0)"
REQUIRED_YEARS = {1761, 1769, 1874, 1882, 2004, 2012}
NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
CLOCK = rf"(?:\d{{1,2}}\s*:\s*\d{{1,2}}\s*:\s*{NUM}|\d{{1,2}}\s*:\s*{NUM}|[-.]+(?:\s*:\s*[-.]+){{0,2}})"

RECORD_RE = re.compile(
    rf"^\s*(?P<jd_tdb>\d{{7}}\.\d{{3}})\s+"
    rf"(?P<day>\d{{1,2}})\s*/\s*(?P<month>\d{{1,2}})\s*/\s*(?P<year>-?\d{{1,4}})\s+"
    rf"(?P<mid_hour>\d{{1,2}})\s*:\s*(?P<mid_minute>\d{{1,2}}(?:\.\d+)?)\s+"
    rf"(?P<sun_radius_arcsec>{NUM})\s+(?P<minimum_distance_arcsec>{NUM})\s+"
    rf"(?P<distance_ratio>{NUM})\s+(?P<venus_radius_arcsec>{NUM})\s+"
    rf"(?P<c1>{CLOCK})\s+(?P<c2>{CLOCK})\s+(?P<c3>{CLOCK})\s+(?P<c4>{CLOCK})\s+"
    rf"(?P<subsolar_longitude_ingress_deg>{NUM})\s+"
    rf"(?P<subsolar_longitude_egress_deg>{NUM})\s+"
    rf"(?P<subsolar_latitude_deg>{NUM})\s+"
    rf"(?P<relative_velocity_deg_per_day>{NUM})\s+"
    rf"(?P<venus_ecliptic_latitude_deg>{NUM})\s+"
    rf"(?P<node>[+-]?\d+)\s+(?P<tdb_minus_ut_seconds>{NUM})\s*$"
)

FLOAT_FIELDS = (
    "jd_tdb", "mid_minute", "sun_radius_arcsec", "minimum_distance_arcsec",
    "distance_ratio", "venus_radius_arcsec", "subsolar_longitude_ingress_deg",
    "subsolar_longitude_egress_deg", "subsolar_latitude_deg",
    "relative_velocity_deg_per_day", "venus_ecliptic_latitude_deg",
    "tdb_minus_ut_seconds",
)
INT_FIELDS = ("source_line_number", "day", "month", "year", "mid_hour", "node")


def download_source() -> str:
    error = None
    for attempt in range(4):
        try:
            request = Request(SOURCE_URL, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=45) as response:
                payload = response.read()
            for encoding in ("utf-8-sig", "cp1252", "iso-8859-1"):
                try:
                    return payload.decode(encoding)
                except UnicodeDecodeError:
                    continue
        except (HTTPError, URLError, TimeoutError) as exc:
            error = exc
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Unable to obtain IMCCE source: {error}")


def load_source() -> str:
    if SOURCE_TEXT.exists():
        return SOURCE_TEXT.read_text(encoding="utf-8")
    SOURCE_TEXT.parent.mkdir(parents=True, exist_ok=True)
    text = download_source().replace("\r\n", "\n").replace("\r", "\n")
    SOURCE_TEXT.write_text(text, encoding="utf-8", newline="\n")
    return text


def clean_clock(value: str) -> str:
    compact = re.sub(r"\s+", "", value)
    return "" if not re.search(r"\d", compact) else compact


def parse_record(line: str, line_number: int) -> dict[str, object]:
    match = RECORD_RE.fullmatch(line)
    if match is None:
        raise ValueError(f"Line {line_number} does not match the canonical record layout: {line!r}")
    row: dict[str, object] = {"source_line_number": line_number, "raw_record": line.rstrip()}
    row.update(match.groupdict())
    for field in FLOAT_FIELDS:
        row[field] = float(str(row[field]))
    for field in INT_FIELDS:
        row[field] = int(row[field])
    for field in ("c1", "c2", "c3", "c4"):
        row[f"{field}_ut"] = clean_clock(str(row.pop(field)))
    row["node_label"] = "ASCENDING" if row["node"] == 1 else "DESCENDING" if row["node"] == -1 else "UNKNOWN"
    return row


def main() -> None:
    text = load_source()
    candidates = [(number, line) for number, line in enumerate(text.splitlines(), 1)
                  if re.match(r"^\s*\d{7}\.\d{3}\b", line)]
    if not candidates:
        raise RuntimeError("No IMCCE transit records were detected")
    rows = [parse_record(line, number) for number, line in candidates]
    years = {int(row["year"]) for row in rows}
    missing = sorted(REQUIRED_YEARS - years)
    if missing:
        raise RuntimeError(f"Required transit years missing after parsing: {missing}")
    if len(rows) != len(candidates):
        raise RuntimeError("Parsed record count does not equal candidate record count")

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with STAGE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CODE OUTPUT: {VERSION}")
    print("CODE INPUTS")
    print(f"Source : {SOURCE_TEXT}")
    print("COMMENTS")
    print("Every candidate source record was parsed with one strict canonical schema.")
    print("RESULTS")
    print(f"Candidate records : {len(candidates)} | Parsed records : {len(rows)}")
    print(f"Year range : {min(years)} to {max(years)} | Required years verified : {len(REQUIRED_YEARS)}")
    print("OUTPUT SUMMARY")
    print(f"Stage CSV : {STAGE_CSV}")
    print("PAPER COMPARISON")
    print("NOT USED — structural parsing stage.")
    print("EQUATION STATUS")
    print("VERIFIED — source candidate count equals parsed record count.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print("# V0122B")


if __name__ == "__main__":
    main()

# V0122B
