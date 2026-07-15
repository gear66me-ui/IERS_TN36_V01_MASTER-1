# V0133
# Audit reference: extend all three V0131 direction lines across the full plot frame without changing geometry, colors, or arrows.

from __future__ import annotations

import urllib.request

VERSION = "V0133"
BASE_URL = (
    "https://raw.githubusercontent.com/"
    "gear66me-ui/IERS_TN36_V01_MASTER-1/main/"
    "VENUS_TRANSIT_1761_EARTH_VENUS_PROJECTED_TRACKS_V0131.py"
)

with urllib.request.urlopen(BASE_URL, timeout=60) as response:
    source = response.read().decode("utf-8")

required_markers = [
    "# V0131",
    'VERSION = "V0131"',
    "V0131_OUTPUT",
    "V0131.png",
    "V0131.csv",
    "half_length = 0.88 * solar_radius_arcsec",
]

for marker in required_markers:
    if marker not in source:
        raise RuntimeError(f"REJECTED missing V0131 source marker: {marker}")

source = source.replace("# V0131", "# V0133")
source = source.replace('VERSION = "V0131"', 'VERSION = "V0133"')
source = source.replace("V0131_OUTPUT", "V0133_OUTPUT")
source = source.replace("V0131.png", "V0133.png")
source = source.replace("V0131.csv", "V0133.csv")
source = source.replace(
    "half_length = 0.88 * solar_radius_arcsec",
    "half_length = 3.00 * solar_radius_arcsec",
)

if source.splitlines()[0] != "# V0133":
    raise RuntimeError("REJECTED incorrect first line")
if source.splitlines()[-1] != "# V0133":
    raise RuntimeError("REJECTED incorrect last line")
if "half_length = 3.00 * solar_radius_arcsec" not in source:
    raise RuntimeError("REJECTED full-frame line extension was not applied")

exec(
    compile(
        source,
        "VENUS_TRANSIT_1761_EARTH_VENUS_PROJECTED_TRACKS_V0133.py",
        "exec",
    )
)
# V0133
