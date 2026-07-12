# V0031
# Audit reference: Compact runner for the corrected North Pole/South Pole JPL half-Sun plot with dynamic label placement.
from __future__ import annotations

import py_compile
import runpy
from pathlib import Path
from urllib.request import Request, urlopen

VERSION = "V0031"
ROOT = Path("/content")
ENGINE = ROOT / "IERS_0012N_NORTH_SOUTH_POLE_HALF_SUN_V0031.py"
BASE = "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main"


def fetch(name: str) -> str:
    request = Request(f"{BASE}/{name}?v=31", headers={"User-Agent": "IERS-V0031"})
    with urlopen(request, timeout=180) as response:
        data = response.read()
    if not data:
        raise RuntimeError(f"Empty source part: {name}")
    return data.decode("utf-8")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"V0031 patch failed for {label}: expected 1 match, found {count}.")
    return text.replace(old, new, 1)


def build_engine() -> str:
    source = "".join(fetch(f"V0030_PART_{index}.txt") for index in range(1, 5))
    source = replace_once(source, 'VERSION = "V0030"', 'VERSION = "V0031"', "version")
    source = replace_once(
        source,
        'OUTPUT_DIR = Path("/content/IERS_TN36_V01_MASTER_OUTPUT/V0030_NORTH_SOUTH_POLE")',
        'OUTPUT_DIR = Path("/content/IERS_TN36_V01_MASTER_OUTPUT/V0031_NORTH_SOUTH_POLE")',
        "output directory",
    )
    source = replace_once(
        source,
        'PNG = OUTPUT_DIR / "V0030_NORTH_SOUTH_POLE_IAU1976_HALF_SUN.png"',
        'PNG = OUTPUT_DIR / "V0031_NORTH_SOUTH_POLE_IAU1976_HALF_SUN.png"',
        "PNG path",
    )
    source = replace_once(
        source,
        'CSV = OUTPUT_DIR / "V0030_NORTH_SOUTH_POLE_EVENTS.csv"',
        'CSV = OUTPUT_DIR / "V0031_NORTH_SOUTH_POLE_EVENTS.csv"',
        "CSV path",
    )

    old_labels = '''        if site["key"] == "NORTH_POLE":
            label_event(axes, track["event_points"]["C1"], "N C1", color, 18, 46)
            label_event(axes, track["event_points"]["C2"], "N C2", color, -72, 36)
            label_event(axes, track["event_points"]["CA"], "North Pole", color, 18, 54)
            label_event(axes, track["event_points"]["C3"], "N C3", color, 24, 40)
            label_event(axes, track["event_points"]["C4"], "N C4", color, -64, 48)
        else:
            label_event(axes, track["event_points"]["C1"], "S C1", color, 18, -46)
            label_event(axes, track["event_points"]["C2"], "S C2", color, -72, -36)
            label_event(axes, track["event_points"]["CA"], "South Pole", color, 18, -54)
            label_event(axes, track["event_points"]["C3"], "S C3", color, 24, -38)
            label_event(axes, track["event_points"]["C4"], "S C4", color, -64, -48)
'''
    new_labels = '''        other_track = south if site["key"] == "NORTH_POLE" else north
        other_points = np.asarray(other_track["points"], dtype=float)
        above = float(np.median(points[:, 1])) >= float(np.median(other_points[:, 1]))
        sign = 1.0 if above else -1.0
        prefix = "N" if site["key"] == "NORTH_POLE" else "S"
        site_name = "North Pole" if site["key"] == "NORTH_POLE" else "South Pole"
        label_event(axes, track["event_points"]["C1"], f"{prefix} C1", color, 18, 46 * sign)
        label_event(axes, track["event_points"]["C2"], f"{prefix} C2", color, -72, 36 * sign)
        label_event(axes, track["event_points"]["CA"], site_name, color, 18, 54 * sign)
        label_event(axes, track["event_points"]["C3"], f"{prefix} C3", color, 24, 40 * sign)
        label_event(axes, track["event_points"]["C4"], f"{prefix} C4", color, -64, 48 * sign)
'''
    source = replace_once(source, old_labels, new_labels, "dynamic label placement")
    source = source.rstrip()
    if source.endswith("# V0030"):
        source = source[:-len("# V0030")] + "# V0031\n"
    if "North Pole" not in source or "South Pole" not in source:
        raise RuntimeError("Pole-label audit failed.")
    return source


def main() -> None:
    engine = build_engine()
    ENGINE.write_text(engine, encoding="utf-8")
    py_compile.compile(str(ENGINE), doraise=True)
    runpy.run_path(str(ENGINE), run_name="__main__")


if __name__ == "__main__":
    main()
# V0031
