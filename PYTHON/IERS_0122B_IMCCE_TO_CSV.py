# V0122B
# Audit reference: robust IMCCE Venus Transit Canon parser with right-anchored numeric tail and four-clock extraction.

from __future__ import annotations

import csv
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

VERSION = "IERS-0122B"
REVISION = "R2"
SOURCE_URL = "https://www.oca.eu/Mignard/Transits/Data/transit_venus.txt"
OUTPUT_ROOT = Path("/content/IERS_TN36_V01_MASTER_OUTPUT/IMCCE_VENUS_CANON")
SOURCE_TEXT = OUTPUT_ROOT / "SOURCE" / "IMCCE_VENUS_TRANSIT_CANON_SOURCE.txt"
STAGE_DIR = OUTPUT_ROOT / "STAGE"
STAGE_CSV = STAGE_DIR / "IMCCE_VENUS_TRANSIT_CANON_PARSED_STAGE.csv"
LOCAL_TZ = timezone(timedelta(hours=-5))
USER_AGENT = "Mozilla/5.0 (compatible; IERS-TN36-IMCCE-Parser/1.0)"
REQUIRED_YEARS = {1761, 1769, 1874, 1882, 2004, 2012}
NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"

HEADER_RE = re.compile(
    rf"^\s*(?P<jd_tdb>\d{{7}}\.\d{{3}})\s+"
    rf"(?P<day>\d{{1,2}})\s*/\s*(?P<month>\d{{1,2}})\s*/\s*(?P<year>-?\d{{1,4}})\s+"
    rf"(?P<mid_hour>\d{{1,2}})\s*:\s*(?P<mid_minute>\d{{1,2}}(?:\.\d+)?)\s+"
    rf"(?P<sun_radius_arcsec>{NUM})\s+(?P<minimum_distance_arcsec>{NUM})\s+"
    rf"(?P<distance_ratio>{NUM})\s+(?P<venus_radius_arcsec>{NUM})\s+"
    rf"(?P<tail>.+?)\s*$"
)

CLOCK_RE = re.compile(
    rf"(?:\d{{1,2}}\s*:\s*\d{{1,2}}\s*:\s*{NUM}|"
    rf"\d{{1,2}}\s*:\s*{NUM}|"
    rf"[-.]+(?:\s*:\s*[-.]+){{0,2}})"
)

FLOAT_FIELDS = (
    "jd_tdb", "mid_minute", "sun_radius_arcsec", "minimum_distance_arcsec",
    "distance_ratio", "venus_radius_arcsec", "subsolar_longitude_ingress_deg",
    "subsolar_longitude_egress_deg", "subsolar_latitude_deg",
    "relative_velocity_deg_per_day", "venus_ecliptic_latitude_deg",
    "tdb_minus_ut_seconds",
)
INT_FIELDS = ("source_line_number", "day", "month", "year", "mid_hour", "node")

SELF_TEST_LINE = (
    "   1030146.899  21/ 5/-1892  19:42   942.6  -619.0     0.657   28.84  "
    "16:32:38  16:51:41  22:33:26  22:52:28   285   199    16    1.59   -0.17 -1 49917.8"
)


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


def normalize_line(line: str) -> str:
    return re.sub(r"[\u00a0\u1680\u2000-\u200b\u202f\u205f\u3000]", " ", line)


def clean_clock(value: str) -> str:
    compact = re.sub(r"\s+", "", value)
    return "" if not re.search(r"\d", compact) else compact


def parse_record(line: str, line_number: int) -> dict[str, object]:
    normalized = normalize_line(line)
    header = HEADER_RE.fullmatch(normalized)
    if header is None:
        raise ValueError(f"Line {line_number} header parse failed: {line!r}")

    tail_parts = header.group("tail").rsplit(None, 7)
    if len(tail_parts) != 8:
        raise ValueError(f"Line {line_number} numeric tail parse failed: {line!r}")

    clock_blob = tail_parts[0]
    clock_matches = list(CLOCK_RE.finditer(clock_blob))
    residue = CLOCK_RE.sub("", clock_blob).strip()
    if len(clock_matches) != 4 or residue:
        raise ValueError(
            f"Line {line_number} contact parse failed: clocks={len(clock_matches)}, residue={residue!r}, line={line!r}"
        )

    tail_names = (
        "subsolar_longitude_ingress_deg", "subsolar_longitude_egress_deg",
        "subsolar_latitude_deg", "relative_velocity_deg_per_day",
        "venus_ecliptic_latitude_deg", "node", "tdb_minus_ut_seconds",
    )
    row: dict[str, object] = {"source_line_number": line_number, "raw_record": line.rstrip()}
    row.update({key: value for key, value in header.groupdict().items() if key != "tail"})
    row.update(dict(zip(tail_names, tail_parts[1:])))
    for index, match in enumerate(clock_matches, start=1):
        row[f"c{index}_ut"] = clean_clock(match.group())
    for field in FLOAT_FIELDS:
        row[field] = float(str(row[field]))
    for field in INT_FIELDS:
        row[field] = int(row[field])
    row["node_label"] = "ASCENDING" if row["node"] == 1 else "DESCENDING" if row["node"] == -1 else "UNKNOWN"
    return row


def run_self_test() -> None:
    row = parse_record(SELF_TEST_LINE, 17)
    expected = ("16:32:38", "16:51:41", "22:33:26", "22:52:28")
    actual = tuple(row[f"c{i}_ut"] for i in range(1, 5))
    if actual != expected or row["year"] != -1892 or row["node"] != -1:
        raise RuntimeError(f"Built-in parser self-test failed: {row}")


def main() -> None:
    run_self_test()
    text = load_source()
    candidates = [
        (number, line) for number, line in enumerate(text.splitlines(), 1)
        if re.match(r"^\s*\d{7}\.\d{3}\b", normalize_line(line))
    ]
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
    with STAGE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"CODE OUTPUT: {VERSION} {REVISION}")
    print("CODE INPUTS")
    print(f"Source : {SOURCE_TEXT}")
    print("COMMENTS")
    print("Parser uses a fixed header, four independent contact clocks, and a seven-field right-anchored numeric tail.")
    print("RESULTS")
    print(f"Self-test : PASS | Candidate records : {len(candidates)} | Parsed records : {len(rows)}")
    print(f"Year range : {min(years)} to {max(years)} | Required years verified : {len(REQUIRED_YEARS)}")
    print("OUTPUT SUMMARY")
    print(f"Stage CSV : {STAGE_CSV}")
    print("PAPER COMPARISON")
    print("NOT USED — structural parsing stage.")
    print("EQUATION STATUS")
    print("VERIFIED — self-test passed and source candidate count equals parsed record count.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print("# V0122B")


if __name__ == "__main__":
    main()

# V0122B
