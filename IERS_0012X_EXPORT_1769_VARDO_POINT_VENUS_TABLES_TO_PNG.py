# IERS-0012X
# Audit reference: GitHubDelivery@IERS-0012X; export the IERS-0012N trigonometry and solar-parallax tables as presentation-ready PNG files.

import csv
import glob
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt

VERSION = "IERS-0012X"
PROGRAM_NAME = "IERS_0012X_EXPORT_1769_VARDO_POINT_VENUS_TABLES_TO_PNG.py"
LOCAL_TZ = ZoneInfo("America/Bogota")
OUTPUT_DIR = "/content/IERS_TN36_V01_MASTER_OUTPUT"
EXACT_SOURCE_CSV = os.path.join(
    OUTPUT_DIR,
    "IERS-0012N_VARDO_POINT_VENUS_ENGINEERING_HALF_SUN_EVENTS_AND_GEOMETRY.csv",
)
TRIGONOMETRY_PNG = os.path.join(
    OUTPUT_DIR,
    "IERS-0012X_1769_VARDO_POINT_VENUS_TRIGONOMETRY_TABLE.png",
)
PI_SUN_PNG = os.path.join(
    OUTPUT_DIR,
    "IERS-0012X_1769_VARDO_POINT_VENUS_PI_SUN_GEOMETRIC_SOLUTION_TABLE.png",
)

SECTION_TRIGONOMETRY = "TRIGONOMETRY"
SECTION_PI_SUN = "PI_SUN_GEOMETRIC_SOLUTION"

BACKGROUND = "#03080d"
TABLE_BACKGROUND = "#050b0f"
HEADER_BACKGROUND = "#0a1a22"
BORDER = "#16333f"
TITLE = "#66e8ff"
TEXT = "#dff8ff"
VALUE = "#ffc861"
UNIT = "#5ee08a"
NOTE = "#8fb4c1"


def locate_source_csv():
    override = os.environ.get("IERS_0012N_CSV", "").strip()
    candidates = []
    if override:
        candidates.append(override)
    candidates.append(EXACT_SOURCE_CSV)
    candidates.extend(
        sorted(
            glob.glob(
                os.path.join(
                    OUTPUT_DIR,
                    "IERS-0012N*VARDO*POINT*VENUS*EVENTS*GEOMETRY*.csv",
                )
            ),
            key=os.path.getmtime,
            reverse=True,
        )
    )
    candidates.extend(
        sorted(
            glob.glob(os.path.join(OUTPUT_DIR, "IERS-0012N*.csv")),
            key=os.path.getmtime,
            reverse=True,
        )
    )

    seen = set()
    for candidate in candidates:
        absolute = os.path.abspath(os.path.expanduser(candidate))
        if absolute in seen:
            continue
        seen.add(absolute)
        if os.path.isfile(absolute):
            return absolute

    searched = "\n".join(f"  - {path}" for path in candidates[:12])
    raise FileNotFoundError(
        "The IERS-0012N geometry CSV was not found. Run IERS-0012N first, "
        "or set the IERS_0012N_CSV environment variable to its exact path.\n"
        f"Searched:\n{searched}"
    )


def read_table_sections(csv_path):
    sections = {
        SECTION_TRIGONOMETRY: [],
        SECTION_PI_SUN: [],
    }
    table_header_found = False

    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            normalized = [str(value).strip() for value in row]
            if normalized[:4] == ["section", "quantity", "value", "unit"]:
                table_header_found = True
                continue
            if not table_header_found or len(normalized) < 4:
                continue

            section, quantity, value, unit = normalized[:4]
            if section in sections:
                sections[section].append((quantity, value, unit))

    if not table_header_found:
        raise RuntimeError(
            "The source CSV does not contain the expected "
            "section,quantity,value,unit table header."
        )

    missing = [name for name, rows in sections.items() if not rows]
    if missing:
        raise RuntimeError(
            "The source CSV is missing required table sections: "
            + ", ".join(missing)
        )
    return sections


def validate_sections(sections):
    trig_rows = sections[SECTION_TRIGONOMETRY]
    geometry_rows = sections[SECTION_PI_SUN]

    expected_trig = [
        "β Vardo",
        "β Point Venus",
        "Δβ",
        "β Average",
    ]
    trig_quantities = [row[0] for row in trig_rows]
    missing_trig = [name for name in expected_trig if name not in trig_quantities]
    if missing_trig:
        raise RuntimeError(
            "The trigonometry section is incomplete; missing: "
            + ", ".join(missing_trig)
        )

    expected_geometry = [
        "Closest Vardo UTC",
        "Closest Point Venus UTC",
        "A′B′ Angular Chord",
        "A′B′ Solar-Screen Chord",
        "AB Angular Projection",
        "AB Projected Baseline",
        "A′B′ / AB",
        "Normal Separation ρ",
        "ρ Scaled To R⊕",
        "D ES",
        "D ES Source",
        "D EV / D VS",
        "D VS / D EV",
        "Raw φ",
        "Computed π⊙",
        "Reference π⊙",
        "Residual π⊙",
    ]
    geometry_quantities = [row[0] for row in geometry_rows]
    missing_geometry = [
        name for name in expected_geometry if name not in geometry_quantities
    ]
    if missing_geometry:
        raise RuntimeError(
            "The geometric-solution section is incomplete; missing: "
            + ", ".join(missing_geometry)
        )


def save_table_png(title, rows, output_path, footer, figure_width, row_height):
    row_count = len(rows)
    figure_height = max(3.5, 1.45 + row_height * (row_count + 1))

    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height),
        dpi=220,
    )
    figure.patch.set_facecolor(BACKGROUND)
    axis.set_facecolor(BACKGROUND)
    axis.axis("off")

    axis.text(
        0.5,
        0.965,
        title,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=14.0,
        fontweight="bold",
        color=TITLE,
    )
    axis.plot(
        [0.02, 0.98],
        [0.915, 0.915],
        transform=axis.transAxes,
        linewidth=0.7,
        color="#25708b",
        clip_on=False,
    )

    cell_text = [[quantity, value, unit] for quantity, value, unit in rows]
    table = axis.table(
        cellText=cell_text,
        colLabels=["Quantity", "Value", "Units"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.50, 0.34, 0.16],
        bbox=[0.02, 0.085, 0.96, 0.79],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.4)

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor(BORDER)
        cell.PAD = 0.055
        if row_index == 0:
            cell.set_facecolor(HEADER_BACKGROUND)
            cell.get_text().set_color(TITLE)
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_ha("left")
        else:
            cell.set_facecolor(TABLE_BACKGROUND)
            if column_index == 0:
                cell.get_text().set_color(TEXT)
                cell.get_text().set_ha("left")
            elif column_index == 1:
                cell.get_text().set_color(VALUE)
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_ha("right")
            else:
                cell.get_text().set_color(UNIT)
                cell.get_text().set_ha("left")

    axis.text(
        0.02,
        0.028,
        footer,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        color=NOTE,
    )

    figure.savefig(
        output_path,
        dpi=360,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
        edgecolor="none",
        pad_inches=0.08,
    )
    plt.close(figure)


def display_pngs(paths):
    try:
        from IPython.display import Image, display

        for path in paths:
            display(Image(filename=path))
        return "DISPLAYED IN COLAB"
    except Exception as exc:
        return f"NOT USED / INLINE DISPLAY UNAVAILABLE ({type(exc).__name__})"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"CODE OUTPUT: {VERSION}")
    print()
    print("CODE INPUTS")
    print(f"Program                : {PROGRAM_NAME}")
    source_csv = locate_source_csv()
    print(f"Source IERS-0012N CSV  : {source_csv}")
    print()
    print("COMMENTS")
    print("Reads the exact IERS-0012N trigonometry and geometric-solution rows from the project CSV.")
    print("Exports each table as a separate high-resolution Matplotlib PNG for presentation use.")
    print("No JPL values are recalculated, rounded differently, or manually re-entered.")
    print("No AI image generation is used.")
    print()

    sections = read_table_sections(source_csv)
    validate_sections(sections)

    trig_rows = sections[SECTION_TRIGONOMETRY]
    geometry_rows = sections[SECTION_PI_SUN]

    save_table_png(
        "TRIGONOMETRY — VARDO NORWAY → POINT VENUS TAHITI",
        trig_rows,
        TRIGONOMETRY_PNG,
        "Source: IERS-0012N JPL Horizons SITE_COORD geometry CSV.",
        figure_width=10.8,
        row_height=0.54,
    )
    save_table_png(
        "π⊙ GEOMETRIC SOLUTION — VARDO NORWAY → POINT VENUS TAHITI",
        geometry_rows,
        PI_SUN_PNG,
        "Source: IERS-0012N JPL Horizons SITE_COORD geometry CSV.",
        figure_width=12.8,
        row_height=0.43,
    )

    display_status = display_pngs([TRIGONOMETRY_PNG, PI_SUN_PNG])

    print("RESULTS")
    print(f"Trigonometry rows      : {len(trig_rows)}")
    print(f"Geometric rows         : {len(geometry_rows)}")
    print(f"Inline display         : {display_status}")
    print()
    print("OUTPUT SUMMARY")
    print(f"Trigonometry PNG       : {TRIGONOMETRY_PNG}")
    print(f"Pi sun geometry PNG    : {PI_SUN_PNG}")
    print()
    print("PAPER COMPARISON")
    print("Published/manual values: NOT USED")
    print("Source values           : IERS-0012N project CSV only")
    print()
    print("EQUATION STATUS")
    print("Scientific recalculation              : NOT USED")
    print("CSV table extraction                  : VERIFIED")
    print("Separate Matplotlib PNG table exports : VERIFIED")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# IERS-0012X
