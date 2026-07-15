# V0132
# Audit reference: extend all three 1761 direction lines beyond the axes so the plot frame clips complete orbital lines.

from __future__ import annotations

import urllib.request

VERSION = "V0132"
BASE_URL = (
    "https://raw.githubusercontent.com/"
    "gear66me-ui/IERS_TN36_V01_MASTER-1/main/"
    "VENUS_TRANSIT_1761_EARTH_VENUS_PROJECTED_TRACKS_V0131.py"
)

with urllib.request.urlopen(BASE_URL, timeout=60) as response:
    source = response.read().decode("utf-8")

replacements = {
    '# V0131': '# V0132',
    'VERSION = "V0131"': 'VERSION = "V0132"',
    'V0131_OUTPUT': 'V0132_OUTPUT',
    'V0131.png': 'V0132.png',
    'V0131.csv': 'V0132.csv',
    'half_length = 0.88 * solar_radius_arcsec':
        'half_length = 2.50 * solar_radius_arcsec',
    'Only the requested 1761 test figure is generated; no AI images are used.':
        'Only the requested 1761 figure is generated; no AI images are used.',
    'No manual angles are inserted.':
        'No manual angles are inserted. All three lines extend beyond the axes and are clipped by the plot frame.',
    'VERIFIED Earth and Venus arrows point from right to left':
        'VERIFIED Earth and Venus arrows point from right to left\n'
        '    print("VERIFIED all three direction lines are clipped by the plot frame")',
}

for old, new in replacements.items():
    if old not in source:
        raise RuntimeError(f"REJECTED missing V0131 source marker: {old}")
    source = source.replace(old, new)

if source.splitlines()[0] != "# V0132":
    raise RuntimeError("REJECTED incorrect first line after V0132 patch")
if source.splitlines()[-1] != "# V0132":
    raise RuntimeError("REJECTED incorrect last line after V0132 patch")
if "half_length = 2.50 * solar_radius_arcsec" not in source:
    raise RuntimeError("REJECTED full-frame line extension was not applied")

exec(compile(source, "VENUS_TRANSIT_1761_EARTH_VENUS_PROJECTED_TRACKS_V0132.py", "exec"))
# V0132
