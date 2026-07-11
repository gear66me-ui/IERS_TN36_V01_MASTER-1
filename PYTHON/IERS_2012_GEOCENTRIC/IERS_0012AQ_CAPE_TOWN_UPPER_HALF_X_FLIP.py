# IERS-0012AQ
# Audit reference: Correct Cape Town upper-half display using x-only reflection; cubic R-squared, RMS, delta-beta, and delta-slope only.
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from matplotlib.axes import Axes

import IERS_0012AP_CAPE_TOWN_UPPER_HALF_CUBIC_R2_RMS as previous

VERSION = "IERS-0012AQ"
LOCAL_TZ = ZoneInfo("America/Bogota")

DRIVE_ROOT = Path("/content/drive/MyDrive/Colab Notebooks")
PROJECT_ROOT = DRIVE_ROOT / "IERS_TN36_OUTPUT" if DRIVE_ROOT.exists() else Path("/content/IERS_TN36_OUTPUT")
OUTPUT_PNG_DIR = PROJECT_ROOT / "OUTPUT_PNG"
OUTPUT_CSV_DIR = PROJECT_ROOT / "OUTPUT_CSV"
OUTPUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)

PLOT_PNG = OUTPUT_PNG_DIR / f"{VERSION}_CAPE_TOWN_UPPER_HALF_X_FLIP.png"
OUTPUT_CSV = OUTPUT_CSV_DIR / f"{VERSION}_CAPE_TOWN_UPPER_HALF_X_FLIP.csv"


def flip_horizontal(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    return np.column_stack((-array[:, 0], array[:, 1]))


def flip_horizontal_point(point: np.ndarray) -> np.ndarray:
    vector = np.asarray(point, dtype=float)
    return np.array([-vector[0], vector[1]], dtype=float)


def verified_render() -> None:
    previous.VERSION = VERSION
    previous.PLOT_PNG = PLOT_PNG
    previous.OUTPUT_CSV = OUTPUT_CSV
    previous.flip_vertical = flip_horizontal
    previous.flip_vertical_point = flip_horizontal_point

    original_set_title = Axes.set_title

    def corrected_set_title(self, label, *args, **kwargs):
        corrected = str(label).replace(
            "Upper Half-Sun — Y-axis Display Flip Only",
            "Upper Half-Sun — X-axis Display Reflection Only",
        )
        return original_set_title(self, corrected, *args, **kwargs)

    Axes.set_title = corrected_set_title
    try:
        previous.render()
    finally:
        Axes.set_title = original_set_title


def main() -> int:
    verified_render()
    print("CODE INPUTS")
    print("Cape Town JPL Horizons SITE_COORD and V0007 SDO video fit")
    print("COMMENTS")
    print("No AI images; x-axis display reflection only; C1 left/high and C4 right/lower")
    print("RESULTS")
    print(f"PNG: {PLOT_PNG}")
    print("OUTPUT SUMMARY")
    print(f"CSV: {OUTPUT_CSV}")
    print("PAPER COMPARISON")
    print("NOT USED")
    print("EQUATION STATUS")
    print("Only cubic R-squared, perpendicular RMS, delta-beta, and delta-m are rendered")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# IERS-0012AQ
