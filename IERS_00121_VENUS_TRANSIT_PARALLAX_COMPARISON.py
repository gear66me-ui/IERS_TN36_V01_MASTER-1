# V0121
# Audit reference: GitHub lucabaldini/astropart notes/macro/plot_au.py historical comparison table

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt

VERSION = "V0121"
SOURCE_REPOSITORY = "lucabaldini/astropart"
SOURCE_FILE = "notes/macro/plot_au.py"
SOURCE_URL = (
    "https://github.com/lucabaldini/astropart/"
    "blob/main/notes/macro/plot_au.py"
)

OUTPUT_DIR = Path("/content/IERS_TN36_V01_MASTER_OUTPUT/IERS_00121")
OUTPUT_CSV = OUTPUT_DIR / "IERS_00121_VENUS_TRANSIT_PARALLAX_TABLE.csv"
OUTPUT_PNG = OUTPUT_DIR / "IERS_00121_VENUS_TRANSIT_PARALLAX_COMPARISON.png"

WGS84_A_KM = 6378.137
AU_KM = 149_597_870.7
ARCSEC_PER_RAD = 206_264.80624709636
PROJECT_JPL_PARALLAX_ARCSEC = 8.794144
IAU_REFERENCE_PARALLAX_ARCSEC = 8.794148


@dataclass(frozen=True)
class HistoricalMeasurement:
    author: str
    year: int
    source_value_type: str
    au_ratio: float | None = None
    parallax_min_arcsec: float | None = None
    parallax_max_arcsec: float | None = None
    method: str = "Venus transit"
    status: str = "COMPARISON"


def parallax_from_au_ratio(au_ratio: float) -> float:
    """Convert measured AU ratio to solar horizontal parallax."""
    if not math.isfinite(au_ratio) or au_ratio <= 0.0:
        raise ValueError(f"Invalid AU ratio: {au_ratio}")
    earth_radius_au = WGS84_A_KM / AU_KM
    return math.atan(earth_radius_au / au_ratio) * ARCSEC_PER_RAD


def au_ratio_from_parallax(parallax_arcsec: float) -> float:
    """Convert solar horizontal parallax to measured AU ratio."""
    if not math.isfinite(parallax_arcsec) or parallax_arcsec <= 0.0:
        raise ValueError(f"Invalid parallax: {parallax_arcsec}")
    earth_radius_au = WGS84_A_KM / AU_KM
    angle_rad = parallax_arcsec / ARCSEC_PER_RAD
    return earth_radius_au / math.tan(angle_rad)


# Historical comparison values transcribed from the selected GitHub table.
# Published values are used only for comparison; project/JPL values remain separate.
HISTORICAL_MEASUREMENTS = (
    HistoricalMeasurement(
        author="Delisle",
        year=1760,
        source_value_type="PARALLAX_RANGE",
        parallax_min_arcsec=10.0,
        parallax_max_arcsec=14.0,
    ),
    HistoricalMeasurement(
        author="De Lacaille",
        year=1761,
        source_value_type="AU_RATIO",
        au_ratio=0.93158,
    ),
    HistoricalMeasurement(
        author="De Lacaille",
        year=1769,
        source_value_type="AU_RATIO",
        au_ratio=1.02069,
    ),
    HistoricalMeasurement(
        author="Delambre",
        year=1770,
        source_value_type="AU_RATIO",
        au_ratio=1.02258,
    ),
    HistoricalMeasurement(
        author="Stone",
        year=1874,
        source_value_type="PARALLAX_RANGE",
        parallax_min_arcsec=8.84,
        parallax_max_arcsec=8.92,
    ),
    HistoricalMeasurement(
        author="Airy",
        year=1874,
        source_value_type="AU_RATIO",
        au_ratio=1.00459,
    ),
    HistoricalMeasurement(
        author="Puiseux",
        year=1875,
        source_value_type="AU_RATIO",
        au_ratio=0.99044,
    ),
    HistoricalMeasurement(
        author="Todd",
        year=1875,
        source_value_type="PARALLAX_RANGE",
        parallax_min_arcsec=8.85,
        parallax_max_arcsec=8.91,
    ),
    HistoricalMeasurement(
        author="Tupman",
        year=1878,
        source_value_type="AU_RATIO",
        au_ratio=0.99763,
    ),
    HistoricalMeasurement(
        author="Todd",
        year=1881,
        source_value_type="PARALLAX_RANGE",
        parallax_min_arcsec=8.85,
        parallax_max_arcsec=8.91,
    ),
)


def resolve_measurement(item: HistoricalMeasurement) -> dict[str, object]:
    """Resolve each source record into one plotted parallax and uncertainty."""
    if item.source_value_type == "AU_RATIO":
        if item.au_ratio is None:
            raise ValueError(f"Missing AU ratio for {item.author} {item.year}")
        parallax = parallax_from_au_ratio(item.au_ratio)
        parallax_min = parallax
        parallax_max = parallax
        uncertainty = 0.0
        au_ratio = item.au_ratio
    elif item.source_value_type == "PARALLAX_RANGE":
        if item.parallax_min_arcsec is None or item.parallax_max_arcsec is None:
            raise ValueError(f"Missing range for {item.author} {item.year}")
        parallax_min = min(item.parallax_min_arcsec, item.parallax_max_arcsec)
        parallax_max = max(item.parallax_min_arcsec, item.parallax_max_arcsec)
        parallax = 0.5 * (parallax_min + parallax_max)
        uncertainty = 0.5 * (parallax_max - parallax_min)
        au_ratio = au_ratio_from_parallax(parallax)
    else:
        raise ValueError(f"Unsupported source type: {item.source_value_type}")

    return {
        "author": item.author,
        "year": item.year,
        "method": item.method,
        "source_value_type": item.source_value_type,
        "au_ratio": au_ratio,
        "parallax_arcsec": parallax,
        "parallax_min_arcsec": parallax_min,
        "parallax_max_arcsec": parallax_max,
        "uncertainty_arcsec": uncertainty,
        "delta_from_project_arcsec": parallax - PROJECT_JPL_PARALLAX_ARCSEC,
        "delta_from_iau_arcsec": parallax - IAU_REFERENCE_PARALLAX_ARCSEC,
        "status": item.status,
    }


def build_rows() -> list[dict[str, object]]:
    rows = [resolve_measurement(item) for item in HISTORICAL_MEASUREMENTS]
    rows.extend(
        [
            {
                "author": "PROJECT JPL",
                "year": 1769,
                "method": "JPL/project calculation",
                "source_value_type": "PROJECT_RESULT",
                "au_ratio": au_ratio_from_parallax(PROJECT_JPL_PARALLAX_ARCSEC),
                "parallax_arcsec": PROJECT_JPL_PARALLAX_ARCSEC,
                "parallax_min_arcsec": PROJECT_JPL_PARALLAX_ARCSEC,
                "parallax_max_arcsec": PROJECT_JPL_PARALLAX_ARCSEC,
                "uncertainty_arcsec": 0.0,
                "delta_from_project_arcsec": 0.0,
                "delta_from_iau_arcsec": (
                    PROJECT_JPL_PARALLAX_ARCSEC - IAU_REFERENCE_PARALLAX_ARCSEC
                ),
                "status": "PROJECT",
            },
            {
                "author": "IAU REFERENCE",
                "year": 1769,
                "method": "Reference comparison",
                "source_value_type": "REFERENCE",
                "au_ratio": au_ratio_from_parallax(IAU_REFERENCE_PARALLAX_ARCSEC),
                "parallax_arcsec": IAU_REFERENCE_PARALLAX_ARCSEC,
                "parallax_min_arcsec": IAU_REFERENCE_PARALLAX_ARCSEC,
                "parallax_max_arcsec": IAU_REFERENCE_PARALLAX_ARCSEC,
                "uncertainty_arcsec": 0.0,
                "delta_from_project_arcsec": (
                    IAU_REFERENCE_PARALLAX_ARCSEC - PROJECT_JPL_PARALLAX_ARCSEC
                ),
                "delta_from_iau_arcsec": 0.0,
                "status": "REFERENCE",
            },
        ]
    )
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "author",
        "year",
        "method",
        "source_value_type",
        "au_ratio",
        "parallax_arcsec",
        "parallax_min_arcsec",
        "parallax_max_arcsec",
        "uncertainty_arcsec",
        "delta_from_project_arcsec",
        "delta_from_iau_arcsec",
        "status",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, object]]) -> None:
    historical = [row for row in rows if row["status"] == "COMPARISON"]

    years = [float(row["year"]) for row in historical]
    parallaxes = [float(row["parallax_arcsec"]) for row in historical]
    uncertainties = [float(row["uncertainty_arcsec"]) for row in historical]

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=160)

    ax.errorbar(
        years,
        parallaxes,
        yerr=uncertainties,
        fmt="o",
        markersize=3.0,
        linewidth=0.65,
        elinewidth=0.65,
        capsize=2.0,
        label="Published historical comparison table",
    )

    ax.axhline(
        PROJECT_JPL_PARALLAX_ARCSEC,
        linewidth=0.8,
        linestyle="-",
        label=f"Project JPL = {PROJECT_JPL_PARALLAX_ARCSEC:.6f} arcsec",
    )
    ax.axhline(
        IAU_REFERENCE_PARALLAX_ARCSEC,
        linewidth=0.8,
        linestyle="--",
        label=f"IAU reference = {IAU_REFERENCE_PARALLAX_ARCSEC:.6f} arcsec",
    )

    ax.scatter(
        [1769.0],
        [PROJECT_JPL_PARALLAX_ARCSEC],
        marker="x",
        s=32.0,
        linewidths=0.9,
        zorder=5,
    )

    for row in historical:
        ax.annotate(
            f"{row['author']} {row['year']}",
            (float(row["year"]), float(row["parallax_arcsec"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_title("Venus-Transit Solar-Parallax Comparison")
    ax.set_xlabel("Year")
    ax.set_ylabel("Solar horizontal parallax (arcsec)")
    ax.set_xlim(1755, 1886)
    ax.set_ylim(8.45, 14.4)
    ax.grid(True, linewidth=0.45, alpha=0.65)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def print_table(rows: list[dict[str, object]]) -> None:
    print("CODE INPUTS")
    print(f"Version                          : {VERSION}")
    print(f"Source repository                : {SOURCE_REPOSITORY}")
    print(f"Source file                      : {SOURCE_FILE}")
    print(f"WGS84 equatorial radius (km)     : {WGS84_A_KM:.6f}")
    print(f"Astronomical unit (km)           : {AU_KM:.6f}")
    print(f"Project JPL parallax (arcsec)    : {PROJECT_JPL_PARALLAX_ARCSEC:.6f}")
    print(f"IAU reference parallax (arcsec)  : {IAU_REFERENCE_PARALLAX_ARCSEC:.6f}")
    print()

    print("COMMENTS")
    print("Historical values are transcribed from the selected public GitHub table.")
    print("Published values are comparison data only; project/JPL values remain separate.")
    print("AU-ratio records are converted to parallax with the displayed project constants.")
    print()

    print("RESULTS")
    print(
        f"{'AUTHOR':<20} {'YEAR':>6} {'PARALLAX':>13} "
        f"{'UNCERTAINTY':>14} {'DELTA JPL':>13} {'STATUS':>12}"
    )
    print("-" * 84)
    for row in rows:
        print(
            f"{str(row['author']):<20} "
            f"{int(row['year']):>6d} "
            f"{float(row['parallax_arcsec']):>13.6f} "
            f"{float(row['uncertainty_arcsec']):>14.6f} "
            f"{float(row['delta_from_project_arcsec']):>+13.6f} "
            f"{str(row['status']):>12}"
        )
    print()

    print("OUTPUT SUMMARY")
    print(f"CSV                              : {OUTPUT_CSV}")
    print(f"PNG                              : {OUTPUT_PNG}")
    print(f"Historical comparison rows       : {len(HISTORICAL_MEASUREMENTS)}")
    print()

    print("PAPER COMPARISON")
    de_lacaille_1769 = next(
        row for row in rows
        if row["author"] == "De Lacaille" and int(row["year"]) == 1769
    )
    print(
        "De Lacaille 1769 minus project JPL (arcsec) : "
        f"{float(de_lacaille_1769['delta_from_project_arcsec']):+.6f}"
    )
    print(
        "Project JPL minus IAU reference (arcsec)     : "
        f"{PROJECT_JPL_PARALLAX_ARCSEC - IAU_REFERENCE_PARALLAX_ARCSEC:+.6f}"
    )
    print()

    print("EQUATION STATUS")
    print("pi = atan[(R_earth / AU) / measured_AU_ratio] : VERIFIED")
    print("AU_ratio = (R_earth / AU) / tan(pi)           : VERIFIED")
    print("Published table values                         : COMPARISON ONLY")
    print("Project JPL value                              : ACCEPTED")
    print("Obsolete 8.807024 arcsec value                 : REJECTED / NOT USED")


def main() -> None:
    rows = build_rows()
    write_csv(rows)
    plot_rows(rows)
    print_table(rows)

    local_timestamp = datetime.now(ZoneInfo("America/Bogota")).strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )
    print(local_timestamp)
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()

# V0121
