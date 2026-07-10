# V0122B
# Audit reference: robust IMCCE Venus Transit Canon parser with logical-record assembly and regression tests.

from __future__ import annotations

import csv
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

VERSION = "IERS-0122B"
REVISION = "R3"
SOURCE_URL = "https://www.oca.eu/Mignard/Transits/Data/transit_venus.txt"
OUTPUT_ROOT = Path("/content/IERS_TN36_V01_MASTER_OUTPUT/IMCCE_VENUS_CANON")
SOURCE_TEXT = OUTPUT_ROOT / "SOURCE" / "IMCCE_VENUS_TRANSIT_CANON_SOURCE.txt"
STAGE_DIR = OUTPUT_ROOT / "STAGE"
STAGE_CSV = STAGE_DIR / "IMCCE_VENUS_TRANSIT_CANON_PARSED_STAGE.csv"
LOCAL_TZ = timezone(timedelta(hours=-5))
USER_AGENT = "Mozilla/5.0 (compatible; IERS-TN36-IMCCE-Parser/1.0)"
REQUIRED_YEARS = {1761, 1769, 1874, 1882, 2004, 2012}
NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
CANDIDATE_RE = re.compile(r"^\s*\d{7}\.\d{3}\b")

HEADER_RE = re.compile(
    rf"^\s*(?P<jd_tdb>\d{{7}}\.\d{{3}})\s+"
    rf"(?P<day>\d{{1,2}})\s*/\s*(?P<month>\d{{1,2}})\s*/\s*(?P<year>-?\s*\d{{1,4}})\s+"
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

SELF_TEST_FULL = (
    "1030146.899 21/5/-1892 19:42 942.6 -619.0 0.657 28.84 "
    "16:32:38 16:51:41 22:33:26 22:52:28 285 199 16 1.59 -0.17 -1 49917.8"
)
SELF_TEST_SPLIT = (
    "1385171.464 23/5/ -920 16:5 942.8 -943.3 1.001 28.79\n"
    "12:00:00 --:--:-- --:--:-- 18:00:00 120 210 20 1.60 -0.17 -1 21000.0"
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
    line = re.sub(r"[\u00a0\u1680\u2000-\u200b\u202f\u205f\u3000]", " ", line)
    return re.sub(r"\s+", " ", line).strip()


def clean_clock(value: str) -> str:
    compact = re.sub(r"\s+", "", value)
    return "" if not re.search(r"\d", compact) else compact


def parse_record(record: str, line_number: int) -> dict[str, object]:
    normalized = normalize_line(record)
    header = HEADER_RE.fullmatch(normalized)
    if header is None:
        raise ValueError(f"Line {line_number} header parse failed: {record!r}")

    tail_parts = header.group("tail").rsplit(None, 7)
    if len(tail_parts) != 8:
        raise ValueError(f"Line {line_number} numeric tail parse failed: {record!r}")

    clock_blob = tail_parts[0]
    clock_matches = list(CLOCK_RE.finditer(clock_blob))
    residue = CLOCK_RE.sub("", clock_blob).strip()
    if len(clock_matches) != 4 or residue:
        raise ValueError(
            f"Line {line_number} contact parse failed: clocks={len(clock_matches)}, "
            f"residue={residue!r}, record={record!r}"
        )

    tail_names = (
        "subsolar_longitude_ingress_deg", "subsolar_longitude_egress_deg",
        "subsolar_latitude_deg", "relative_velocity_deg_per_day",
        "venus_ecliptic_latitude_deg", "node", "tdb_minus_ut_seconds",
    )
    row: dict[str, object] = {"source_line_number": line_number, "raw_record": normalized}
    row.update({key: value for key, value in header.groupdict().items() if key != "tail"})
    row.update(dict(zip(tail_names, tail_parts[1:])))
    for index, match in enumerate(clock_matches, start=1):
        row[f"c{index}_ut"] = clean_clock(match.group())
    row["year"] = str(row["year"]).replace(" ", "")
    for field in FLOAT_FIELDS:
        row[field] = float(str(row[field]))
    for field in INT_FIELDS:
        row[field] = int(row[field])
    row["node_label"] = "ASCENDING" if row["node"] == 1 else "DESCENDING" if row["node"] == -1 else "UNKNOWN"
    return row


def assemble_and_parse(text: str) -> tuple[list[dict[str, object]], int]:
    lines = text.splitlines()
    starts = [index for index, line in enumerate(lines) if CANDIDATE_RE.match(normalize_line(line))]
    if not starts:
        raise RuntimeError("No IMCCE transit records were detected")

    rows: list[dict[str, object]] = []
    continuation_count = 0
    for position, start in enumerate(starts):
        stop = starts[position + 1] if position + 1 < len(starts) else len(lines)
        pieces = [normalize_line(lines[start])]
        parsed = None
        last_error: Exception | None = None
        for index in range(start, stop):
            if index > start:
                piece = normalize_line(lines[index])
                if not piece:
                    continue
                pieces.append(piece)
                continuation_count += 1
            try:
                parsed = parse_record(" ".join(pieces), start + 1)
                break
            except ValueError as exc:
                last_error = exc
        if parsed is None:
            raise ValueError(
                f"Unable to assemble logical record beginning at line {start + 1}: "
                f"{' | '.join(pieces)!r}. Last error: {last_error}"
            )
        rows.append(parsed)
    return rows, continuation_count


def run_self_tests() -> None:
    full = parse_record(SELF_TEST_FULL, 17)
    split_rows, split_count = assemble_and_parse(SELF_TEST_SPLIT)
    expected = ("16:32:38", "16:51:41", "22:33:26", "22:52:28")
    actual = tuple(full[f"c{i}_ut"] for i in range(1, 5))
    if actual != expected or full["year"] != -1892 or full["node"] != -1:
        raise RuntimeError(f"Full-record regression test failed: {full}")
    if len(split_rows) != 1 or split_rows[0]["year"] != -920 or split_count != 1:
        raise RuntimeError(f"Split-record regression test failed: {split_rows}")


def main() -> None:
    run_self_tests()
    rows, continuation_count = assemble_and_parse(load_source())
    years = {int(row["year"]) for row in rows}
    missing = sorted(REQUIRED_YEARS - years)
    if missing:
        raise RuntimeError(f"Required transit years missing after parsing: {missing}")

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    with STAGE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"CODE OUTPUT: {VERSION} {REVISION}")
    print("CODE INPUTS")
    print(f"Source : {SOURCE_TEXT}")
    print("COMMENTS")
    print("Physical continuation lines are assembled into one logical transit record before parsing.")
    print("RESULTS")
    print(f"Self-tests : PASS | Parsed records : {len(rows)} | Continuation lines used : {continuation_count}")
    print(f"Year range : {min(years)} to {max(years)} | Required years verified : {len(REQUIRED_YEARS)}")
    print("OUTPUT SUMMARY")
    print(f"Stage CSV : {STAGE_CSV}")
    print("PAPER COMPARISON")
    print("NOT USED — structural parsing stage.")
    print("EQUATION STATUS")
    print("VERIFIED — full-record and split-record regression tests passed.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print("# V0122B")


if __name__ == "__main__":
    main()

# V0122B
