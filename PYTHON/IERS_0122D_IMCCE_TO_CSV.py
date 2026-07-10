# V0122D
# Audit reference: publication workbook builder for the normalized IMCCE Venus Transit Canon master CSV.

from __future__ import annotations

import csv
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from artifact_tool import SpreadsheetFile, Workbook
except ImportError as exc:
    raise RuntimeError("artifact_tool is required to build the XLSX workbook.") from exc

VERSION = "IERS-0122D"
OUTPUT_ROOT = Path("/content/IERS_TN36_V01_MASTER_OUTPUT/IMCCE_VENUS_CANON")
MASTER_CSV = OUTPUT_ROOT / "IMCCE_VENUS_TRANSIT_CANON_MASTER.csv"
MASTER_XLSX = OUTPUT_ROOT / "IMCCE_VENUS_TRANSIT_CANON_MASTER.xlsx"
LOCAL_TZ = timezone(timedelta(hours=-5))
TARGET_YEARS = (1761, 1769, 1874, 1882, 2004, 2012)
SHEET_ORDER = ("README", "MASTER", "1761", "1769", "1874", "1882", "2004", "2012")
SOURCE_PAGE_URL = "https://www.oca.eu/Mignard/Transits/Html/canon_venus.htm"
SOURCE_DATA_URL = "https://www.oca.eu/Mignard/Transits/Data/transit_venus.txt"

INT_COLUMNS = {
    "source_line_number", "day", "month", "year", "mid_hour", "node",
    "missing_field_count", "record_id",
}
FLOAT_COLUMNS = {
    "jd_tdb", "mid_minute", "sun_radius_arcsec", "minimum_distance_arcsec",
    "distance_ratio", "venus_radius_arcsec", "subsolar_longitude_ingress_deg",
    "subsolar_longitude_egress_deg", "subsolar_latitude_deg",
    "relative_velocity_deg_per_day", "venus_ecliptic_latitude_deg",
    "tdb_minus_ut_seconds", "mid_ut_seconds_of_day", "impact_parameter_abs_arcsec",
    "ratio_abs_calculated", "ratio_residual_source_minus_calculated",
    "c1_seconds_of_day", "c2_seconds_of_day", "c3_seconds_of_day", "c4_seconds_of_day",
    "external_duration_seconds", "internal_duration_seconds",
}
WIDE_TEXT_COLUMNS = {"raw_record", "missing_fields", "source_page_url", "source_data_url"}

TITLE_FORMAT = {
    "fill": "#0B5D4B",
    "font": {"bold": True, "color": "#FFFFFF", "size": 16},
    "horizontal_alignment": "left",
    "vertical_alignment": "center",
}
SUBTITLE_FORMAT = {
    "fill": "#DDEFE9",
    "font": {"italic": True, "color": "#234E42", "size": 10},
    "horizontal_alignment": "left",
    "vertical_alignment": "center",
}
HEADER_FORMAT = {
    "fill": "#1F7A63",
    "font": {"bold": True, "color": "#FFFFFF", "size": 9},
    "horizontal_alignment": "center",
    "vertical_alignment": "center",
    "wrap_text": True,
}
SECTION_FORMAT = {
    "fill": "#B7DDD1",
    "font": {"bold": True, "color": "#163D33"},
}


def column_letter(number: int) -> str:
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def typed_value(header: str, value: str) -> object:
    text = value.strip()
    if text == "":
        return None
    if header in INT_COLUMNS:
        return int(float(text))
    if header in FLOAT_COLUMNS:
        return float(text)
    return text


def load_master() -> tuple[list[str], list[list[object]]]:
    if not MASTER_CSV.exists():
        raise FileNotFoundError(f"Run IERS-0122C first: {MASTER_CSV}")
    with MASTER_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        rows = [[typed_value(header, record.get(header, "")) for header in headers] for record in reader]
    if len(rows) != 77:
        raise RuntimeError(f"Expected 77 master rows, found {len(rows)}")
    return headers, rows


def apply_column_formats(sheet, headers: list[str], first_data_row: int, last_data_row: int) -> None:
    for index, header in enumerate(headers, start=1):
        letter = column_letter(index)
        column = sheet.get_range(f"{letter}4:{letter}{max(last_data_row, 5)}")
        if header in WIDE_TEXT_COLUMNS:
            column.format.column_width = 38
            column.format.wrap_text = True
        elif header.endswith("_url"):
            column.format.column_width = 34
            column.format.wrap_text = True
        elif header in {"geometry_class", "record_status", "node_label", "year_display"}:
            column.format.column_width = 25
            column.format.wrap_text = True
        elif header.endswith("_ut") or header in {"date_ut_label", "mid_ut_hhmm"}:
            column.format.column_width = 16
        else:
            column.format.column_width = 14

        if last_data_row >= first_data_row:
            data_range = sheet.get_range(f"{letter}{first_data_row}:{letter}{last_data_row}")
            if header in INT_COLUMNS:
                data_range.format.number_format = "0"
            elif header == "jd_tdb":
                data_range.format.number_format = "0.000"
            elif "ratio" in header:
                data_range.format.number_format = "0.000000"
            elif header in FLOAT_COLUMNS:
                data_range.format.number_format = "0.000"


def add_data_sheet(workbook, name: str, headers: list[str], rows: list[list[object]], table_name: str) -> None:
    sheet = workbook.worksheets.add(name)
    last_column = column_letter(len(headers))
    last_row = 4 + max(len(rows), 1)
    sheet.merge_cells(f"A1:{last_column}1")
    sheet.get_range("A1").values = [["IMCCE VENUS TRANSIT CANON"]]
    sheet.get_range(f"A1:{last_column}1").format = TITLE_FORMAT
    sheet.get_range(f"A1:{last_column}1").format.row_height = 28
    sheet.merge_cells(f"A2:{last_column}2")
    subtitle = "All 77 canon records" if name == "MASTER" else f"Canonical record for the {name} Venus transit"
    sheet.get_range("A2").values = [[subtitle]]
    sheet.get_range(f"A2:{last_column}2").format = SUBTITLE_FORMAT
    sheet.get_range("A4").write([headers] + (rows if rows else [[None] * len(headers)]))
    sheet.get_range(f"A4:{last_column}4").format = HEADER_FORMAT
    sheet.get_range(f"A4:{last_column}4").format.row_height = 34
    sheet.freeze_panes.freeze_rows(4)
    sheet.tables.add(f"A4:{last_column}{last_row}", True, table_name)
    apply_column_formats(sheet, headers, 5, last_row)

    if "record_status" in headers and rows:
        status_letter = column_letter(headers.index("record_status") + 1)
        sheet.get_range(f"A5:{last_column}{last_row}").conditional_formats.add_custom(
            f'=${status_letter}5="SOURCE_HEADER_ONLY"', {"fill": "#FFF2CC"}
        )
    if "ratio_validation" in headers and rows:
        ratio_letter = column_letter(headers.index("ratio_validation") + 1)
        sheet.get_range(f"{ratio_letter}5:{ratio_letter}{last_row}").conditional_formats.add_custom(
            f'={ratio_letter}5="REVIEW"', {"fill": "#F4CCCC", "font": {"bold": True}}
        )


def add_readme(workbook, row_count: int, complete_count: int, header_only_count: int) -> None:
    sheet = workbook.worksheets.add("README")
    sheet.merge_cells("A1:D1")
    sheet.get_range("A1").values = [["IMCCE VENUS TRANSIT CANON — WORKBOOK README"]]
    sheet.get_range("A1:D1").format = TITLE_FORMAT
    sheet.get_range("A1:D1").format.row_height = 30
    content = [
        ["Workbook", "IMCCE_VENUS_TRANSIT_CANON_MASTER.xlsx", "Version", VERSION],
        ["Source page", SOURCE_PAGE_URL, "Source data", SOURCE_DATA_URL],
        ["Master records", row_count, "Complete records", complete_count],
        ["Header-only records", header_only_count, "Target-year sheets", len(TARGET_YEARS)],
        ["", "", "", ""],
        ["SHEET GUIDE", "", "", ""],
        ["MASTER", "Normalized 77-record canon with source and calculated fields.", "", ""],
        ["1761–2012", "One filtered record per requested historical transit year.", "", ""],
        ["", "", "", ""],
        ["TRACEABILITY", "", "", ""],
        ["Source fields", "Preserved from the IMCCE text canon.", "", ""],
        ["Calculated fields", "Appended by IERS-0122C and explicitly named.", "", ""],
        ["SOURCE_HEADER_ONLY", "The IMCCE source omitted contact and trailing geometry fields.", "", ""],
        ["REVIEW", "A validation flag requiring inspection; no ratio reviews existed at build time.", "", ""],
    ]
    sheet.get_range("A3").write(content)
    sheet.get_range("A8:D8").format = SECTION_FORMAT
    sheet.get_range("A12:D12").format = SECTION_FORMAT
    sheet.get_range("A3:D16").format.wrap_text = True
    sheet.get_range("A:A").format.column_width = 24
    sheet.get_range("B:B").format.column_width = 48
    sheet.get_range("C:C").format.column_width = 22
    sheet.get_range("D:D").format.column_width = 42
    sheet.freeze_panes.freeze_rows(2)


def main() -> None:
    headers, rows = load_master()
    year_index = headers.index("year")
    status_index = headers.index("record_status")
    complete_count = sum(row[status_index] == "COMPLETE" for row in rows)
    header_only_count = sum(row[status_index] == "SOURCE_HEADER_ONLY" for row in rows)

    workbook = Workbook.create()
    add_readme(workbook, len(rows), complete_count, header_only_count)
    add_data_sheet(workbook, "MASTER", headers, rows, "IMCCE_Master_Table")
    for year in TARGET_YEARS:
        selected = [row for row in rows if row[year_index] == year]
        if len(selected) != 1:
            raise RuntimeError(f"Expected exactly one record for {year}, found {len(selected)}")
        add_data_sheet(workbook, str(year), headers, selected, f"IMCCE_{year}_Table")

    inspection = workbook.inspect({"kind": "sheet", "include": "id,name"}).ndjson
    missing_sheets = [name for name in SHEET_ORDER if name not in inspection]
    if missing_sheets:
        raise RuntimeError(f"Workbook sheet verification failed: {missing_sheets}")
    errors = workbook.inspect({
        "kind": "match",
        "search_term": "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
        "options": {"use_regex": True, "max_results": 100},
        "summary": "formula error scan",
    }).ndjson
    if re.search(r"#REF!|#DIV/0!|#VALUE!|#NAME\?|#N/A", errors):
        raise RuntimeError(f"Workbook formula error scan failed: {errors}")

    MASTER_XLSX.parent.mkdir(parents=True, exist_ok=True)
    SpreadsheetFile.export_xlsx(workbook).save(str(MASTER_XLSX))

    print(f"CODE OUTPUT: {VERSION}")
    print("CODE INPUTS")
    print(f"Master CSV : {MASTER_CSV}")
    print("COMMENTS")
    print("Workbook created with artifact_tool, styled tables, frozen headers, and traceability notes.")
    print("RESULTS")
    print(f"Rows : {len(rows)} | Sheets : {len(SHEET_ORDER)} | Complete : {complete_count} | Header-only : {header_only_count}")
    print("OUTPUT SUMMARY")
    print(f"Workbook : {MASTER_XLSX}")
    print("PAPER COMPARISON")
    print("NOT USED — workbook publication stage.")
    print("EQUATION STATUS")
    print("VERIFIED — required sheets and formula-error scan passed.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print("# V0122D")


if __name__ == "__main__":
    main()

# V0122D
