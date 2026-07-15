# V0134
# Audit reference: full-frame 1761 Earth/Venus/projected lines with explicit right-to-left arrows on Earth and Venus orbits.

from __future__ import annotations

import urllib.request

VERSION = "V0134"
BASE_URL = (
    "https://raw.githubusercontent.com/"
    "gear66me-ui/IERS_TN36_V01_MASTER-1/main/"
    "VENUS_TRANSIT_1761_EARTH_VENUS_PROJECTED_TRACKS_V0131.py"
)

with urllib.request.urlopen(BASE_URL, timeout=60) as response:
    source = response.read().decode("utf-8")

required = [
    "# V0131",
    'VERSION = "V0131"',
    "V0131_OUTPUT",
    "V0131.png",
    "V0131.csv",
    "half_length = 0.88 * solar_radius_arcsec",
    'if label in {"Earth orbit", "Venus orbit"}:',
]

for marker in required:
    if marker not in source:
        raise RuntimeError(f"REJECTED missing source marker: {marker}")

source = source.replace("# V0131", "# V0134")
source = source.replace('VERSION = "V0131"', 'VERSION = "V0134"')
source = source.replace("V0131_OUTPUT", "V0134_OUTPUT")
source = source.replace("V0131.png", "V0134.png")
source = source.replace("V0131.csv", "V0134.csv")
source = source.replace(
    "half_length = 0.88 * solar_radius_arcsec",
    "half_length = 3.00 * solar_radius_arcsec",
)

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

            # Explicit orbital direction: start at larger X (right), end at smaller X (left).
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

if old_arrow not in source:
    raise RuntimeError("REJECTED arrow block not found")
source = source.replace(old_arrow, new_arrow)

source = source.replace(
    'print("VERIFIED Earth and Venus arrows point from right to left")',
    'print("VERIFIED Earth and Venus arrows point from larger X to smaller X: right to left")'
)

if source.splitlines()[0] != "# V0134":
    raise RuntimeError("REJECTED incorrect first line")
if source.splitlines()[-1] != "# V0134":
    raise RuntimeError("REJECTED incorrect last line")
if "if not end_x < start_x" not in source:
    raise RuntimeError("REJECTED direction assertion missing")
if "half_length = 3.00 * solar_radius_arcsec" not in source:
    raise RuntimeError("REJECTED full-frame extension missing")

exec(
    compile(
        source,
        "VENUS_TRANSIT_1761_EARTH_VENUS_PROJECTED_TRACKS_V0134.py",
        "exec",
    )
)
# V0134
