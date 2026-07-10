# V0122A
# Audit reference: IMCCE Venus Transit Canon source acquisition and integrity manifest.

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

VERSION = "IERS-0122A"
SOURCE_PAGE_URL = "https://www.oca.eu/Mignard/Transits/Html/canon_venus.htm"
SOURCE_DATA_URL = "https://www.oca.eu/Mignard/Transits/Data/transit_venus.txt"
OUTPUT_ROOT = Path("/content/IERS_TN36_V01_MASTER_OUTPUT/IMCCE_VENUS_CANON")
SOURCE_DIR = OUTPUT_ROOT / "SOURCE"
SOURCE_TEXT = SOURCE_DIR / "IMCCE_VENUS_TRANSIT_CANON_SOURCE.txt"
MANIFEST_JSON = SOURCE_DIR / "IERS_0122A_IMCCE_SOURCE_MANIFEST.json"
LINE_INDEX_CSV = SOURCE_DIR / "IERS_0122A_IMCCE_SOURCE_LINE_INDEX.csv"
LOCAL_TZ = timezone(timedelta(hours=-5))
KNOWN_YEARS = (1761, 1769, 1874, 1882, 2004, 2012)
USER_AGENT = "Mozilla/5.0 (compatible; IERS-TN36-IMCCE-Parser/1.0)"


def fetch_bytes(url: str, attempts: int = 4) -> bytes:
    error = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/plain,*/*"})
            with urlopen(request, timeout=45) as response:
                payload = response.read()
            if not payload:
                raise RuntimeError("Empty source response")
            return payload
        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"Unable to download IMCCE canon: {error}")


def decode_payload(payload: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "cp1252", "iso-8859-1"):
        try:
            text = payload.decode(encoding)
        except UnicodeDecodeError:
            continue
        if re.search(r"\d{7}\.\d{3}", text):
            return text, encoding
    return payload.decode("iso-8859-1", errors="replace"), "iso-8859-1-replace"


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "").replace("\xa0", " ").replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip() + "\n"


def classify_line(line: str) -> str:
    value = line.strip()
    if not value:
        return "BLANK"
    if re.match(r"^\d{7}\.\d{3}\b", value):
        return "TRANSIT_RECORD_CANDIDATE"
    if value.startswith(("#", "!", ";")):
        return "COMMENT"
    if "JD" in value and "date" in value.lower():
        return "HEADER"
    return "TEXT"


def build_index(text: str) -> list[dict[str, object]]:
    return [
        {
            "line_number": number,
            "line_type": classify_line(line),
            "character_count": len(line),
            "text": line,
        }
        for number, line in enumerate(text.splitlines(), start=1)
    ]


def validate(text: str, rows: list[dict[str, object]]) -> dict[str, object]:
    years = sorted({int(value) for value in re.findall(r"\b\d{1,2}/\s*\d{1,2}/\s*(-?\d{1,4})\b", text)})
    missing = [year for year in KNOWN_YEARS if year not in years]
    candidates = sum(row["line_type"] == "TRANSIT_RECORD_CANDIDATE" for row in rows)
    if missing:
        raise RuntimeError(f"Missing required historical transit years: {missing}")
    if candidates < 20:
        raise RuntimeError(f"Only {candidates} candidate transit records detected")
    return {
        "line_count": len(rows),
        "nonblank_line_count": sum(row["line_type"] != "BLANK" for row in rows),
        "candidate_record_count": candidates,
        "minimum_year": min(years),
        "maximum_year": max(years),
        "known_years_verified": list(KNOWN_YEARS),
    }


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    payload = fetch_bytes(SOURCE_DATA_URL)
    decoded, encoding = decode_payload(payload)
    text = normalize_text(decoded)
    rows = build_index(text)
    audit = validate(text, rows)
    SOURCE_TEXT.write_text(text, encoding="utf-8", newline="\n")

    with LINE_INDEX_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "version": VERSION,
        "source_page_url": SOURCE_PAGE_URL,
        "source_data_url": SOURCE_DATA_URL,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_encoding": encoding,
        "source_byte_count": len(payload),
        "normalized_character_count": len(text),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "normalized_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "validation": audit,
        "source_text_path": str(SOURCE_TEXT),
        "line_index_csv_path": str(LINE_INDEX_CSV),
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"CODE OUTPUT: {VERSION}")
    print("CODE INPUTS")
    print(f"Source URL : {SOURCE_DATA_URL}")
    print("COMMENTS")
    print("IMCCE source downloaded, decoded, normalized, indexed, and validated.")
    print("RESULTS")
    print(f"Encoding : {encoding} | Bytes : {len(payload)} | Lines : {audit['line_count']}")
    print(f"Candidates : {audit['candidate_record_count']} | Years : {audit['minimum_year']} to {audit['maximum_year']}")
    print("OUTPUT SUMMARY")
    print(f"Source : {SOURCE_TEXT}\nIndex : {LINE_INDEX_CSV}\nManifest : {MANIFEST_JSON}")
    print("PAPER COMPARISON")
    print("NOT USED — source acquisition stage.")
    print("EQUATION STATUS")
    print("NOT USED — no scientific equations evaluated.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print("# V0122A")


if __name__ == "__main__":
    main()

# V0122A
