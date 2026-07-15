# V0144
# Audit reference: reproduce the approved V0134 plot format for all six Venus transits without changing geometry or styling.

from __future__ import annotations

import urllib.request

VERSION = "V0144"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "VENUS_TRANSIT_1761_EARTH_VENUS_PROJECTED_TRACKS_V0131.py"
)

TRANSITS = {
    1761: "1761-06-06 06:00",
    1769: "1769-06-03 22:00",
    1874: "1874-12-09 04:00",
    1882: "1882-12-06 17:00",
    2004: "2004-06-08 08:00",
    2012: "2012-06-06 01:00",
}

with urllib.request.urlopen(SOURCE_URL, timeout=90) as response:
    template = response.read().decode("utf-8")

required = [
    "# V0131",
    'VERSION = "V0131"',
    "YEAR = 1761",
    'CENTER_UTC = "1761-06-06 06:00"',
    "half_length = 0.88 * solar_radius_arcsec",
    'if label in {"Earth orbit", "Venus orbit"}:',
    '"1761 Venus Transit — Earth Orbit, Venus Orbit, and Projected Relative Track"',
]
for marker in required:
    if marker not in template:
        raise RuntimeError(f"REJECTED missing V0131 marker: {marker}")

old_arrow = '''        if label in {"Earth orbit", "Venus orbit"}:
            right_index = int(np.argmax(x_line))
            left_index = int(np.argmin(x_line))
            start_x = 0.25 * x_line[left_index] + 0.75 * x_line[right_index]
            start_y = 0.25 * y_line[left_index] + 0.75 * y_line[right_index]
            end_x = 0.75 * x_line[left_index] + 0.25 * x_line[right_index]
            end_y = 0.75 * y_line[left_index] + 0.25 * y_line[right_index]
            ax.annotate(
                "",
                xy=(end_x, end_y),
                xytext=(start_x, start_y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "linewidth": 1.15,
                    "mutation_scale": 13.0,
                    "shrinkA": 0.0,
                    "shrinkB": 0.0,
                },
                zorder=6,
            )
'''

new_arrow = '''        if label in {"Earth orbit", "Venus orbit"}:
            right_index = int(np.argmax(x_line))
            left_index = int(np.argmin(x_line))

            right_x = float(x_line[right_index])
            right_y = float(y_line[right_index])
            left_x = float(x_line[left_index])
            left_y = float(y_line[left_index])

            start_x = 0.62 * right_x + 0.38 * left_x
            start_y = 0.62 * right_y + 0.38 * left_y
            end_x = 0.38 * right_x + 0.62 * left_x
            end_y = 0.38 * right_y + 0.62 * left_y

            if not end_x < start_x:
                raise RuntimeError(f"REJECTED {label} arrow is not right-to-left")

            ax.annotate(
                "",
                xy=(end_x, end_y),
                xytext=(start_x, start_y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "linewidth": 1.35,
                    "mutation_scale": 16.0,
                    "shrinkA": 0.0,
                    "shrinkB": 0.0,
                },
                zorder=9,
                clip_on=True,
            )
'''

for year, center_utc in TRANSITS.items():
    source = template
    source = source.replace("# V0131", "# V0144")
    source = source.replace('VERSION = "V0131"', 'VERSION = "V0144"')
    source = source.replace("YEAR = 1761", f"YEAR = {year}")
    source = source.replace(
        'CENTER_UTC = "1761-06-06 06:00"',
        f'CENTER_UTC = "{center_utc}"',
    )
    source = source.replace(
        'OUTPUT_DIR = Path("/content/VENUS_TRANSIT_1761_EARTH_VENUS_PROJECTED_TRACKS_V0131_OUTPUT")',
        'OUTPUT_DIR = Path("/content/VENUS_TRANSITS_SIX_PROJECTED_TRACKS_V0144_OUTPUT")',
    )
    source = source.replace(
        'PNG_NAME = "VENUS_TRANSIT_1761_EARTH_VENUS_PROJECTED_TRACKS_V0131.png"',
        f'PNG_NAME = "VENUS_TRANSIT_{year}_EARTH_VENUS_PROJECTED_TRACKS_V0144.png"',
    )
    source = source.replace(
        'CSV_NAME = "VENUS_TRANSIT_1761_EARTH_VENUS_PROJECTED_TRACKS_V0131.csv"',
        f'CSV_NAME = "VENUS_TRANSIT_{year}_EARTH_VENUS_PROJECTED_TRACKS_V0144.csv"',
    )
    source = source.replace(
        'print("Transit                              1761")',
        f'print("Transit                              {year}")',
    )
    source = source.replace(
        '"1761 Venus Transit — Earth Orbit, Venus Orbit, and Projected Relative Track"',
        f'"{year} Venus Transit — Earth Orbit, Venus Orbit, and Projected Relative Track"',
    )
    source = source.replace(
        'print("Only the requested 1761 test figure is generated; no AI images are used.")',
        f'print("The requested {year} figure is generated; no AI images are used.")',
    )
    source = source.replace(
        "half_length = 0.88 * solar_radius_arcsec",
        "half_length = 3.00 * solar_radius_arcsec",
    )
    source = source.replace(old_arrow, new_arrow)
    source = source.replace(
        'print("VERIFIED Earth and Venus arrows point from right to left")',
        'print("VERIFIED Earth and Venus arrows point from larger X to smaller X: right to left")',
    )
    source = source.replace("dpi=600", "dpi=300")

    if source.splitlines()[0] != "# V0144":
        raise RuntimeError(f"REJECTED {year} first line")
    if source.splitlines()[-1] != "# V0144":
        raise RuntimeError(f"REJECTED {year} last line")
    if "half_length = 3.00 * solar_radius_arcsec" not in source:
        raise RuntimeError(f"REJECTED {year} full-frame lines")
    if "if not end_x < start_x" not in source:
        raise RuntimeError(f"REJECTED {year} arrow direction assertion")

    namespace = {"__name__": "__main__"}
    exec(
        compile(
            source,
            f"VENUS_TRANSIT_{year}_EARTH_VENUS_PROJECTED_TRACKS_V0144.py",
            "exec",
        ),
        namespace,
        namespace,
    )
# V0144