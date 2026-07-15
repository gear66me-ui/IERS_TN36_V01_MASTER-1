# V0149
# Audit reference: corrected 2012-only ecliptic crosshair plot with black crosshair and title-case labels; no AI images.
from __future__ import annotations
import urllib.request

VERSION = "V0149"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "VENUS_TRANSITS_1761_1769_1874_1882_2004_2012_ECLIPTIC_CROSSHAIR_V0146.py"
)

with urllib.request.urlopen(SOURCE_URL, timeout=90) as response:
    source = response.read().decode("utf-8")

start = source.index("TRANSITS = {")
stop = source.index("}\n\nwith urllib.request.urlopen", start) + 1
source = source[:start] + 'TRANSITS = {\n    2012: "2012-06-06 01:00",\n}' + source[stop:]

source = source.replace("# V0146", "# V0149")
source = source.replace('VERSION = "V0146"', 'VERSION = "V0149"')
source = source.replace("V0146_OUTPUT", "V0149_OUTPUT")
source = source.replace("V0146.png", "V0149.png")
source = source.replace("V0146.csv", "V0149.csv")
source = source.replace("ECLIPTIC_CROSSHAIR_V0146.py", "ECLIPTIC_CROSSHAIR_V0149.py")

source = source.replace(
    'color=\\"#D8D8D8\\", linewidth=0.42, alpha=0.42',
    'color=\\"#000000\\", linewidth=0.72, alpha=0.92',
)
source = source.replace(
    'color=\\"#D8D8D8\\", linewidth=0.42, alpha=0.30',
    'color=\\"#000000\\", linewidth=0.72, alpha=0.92',
)
source = source.replace(
    '\\"Ecliptic reference  0.000°\\", color=\\"#D8D8D8\\"',
    '\\"Ecliptic Reference  0.000°\\", color=\\"#000000\\"',
)

for old, new in {
    'TextArea("Ecliptic reference: 0.000°"': 'TextArea("Ecliptic Reference: 0.000°"',
    'TextArea(f"Earth track from ecliptic:': 'TextArea(f"Earth Track From Ecliptic:',
    'TextArea(f"Projected Venus transit track:': 'TextArea(f"Projected Venus Transit Track:',
    'TextArea(f"Venus transit track from ecliptic:': 'TextArea(f"Venus Transit Track From Ecliptic:',
    '("Earth track from ecliptic",': '("Earth Track From Ecliptic",',
    '("Projected Venus transit track",': '("Projected Venus Transit Track",',
    '("Venus transit track from ecliptic",': '("Venus Transit Track From Ecliptic",',
    '{"Earth track from ecliptic", "Venus transit track from ecliptic"}': '{"Earth Track From Ecliptic", "Venus Transit Track From Ecliptic"}',
}.items():
    source = source.replace(old, new)

if source.splitlines()[0] != "# V0149" or source.splitlines()[-1] != "# V0149":
    raise RuntimeError("REJECTED version boundary")
if '1761: "1761-06-06 06:00"' in source:
    raise RuntimeError("REJECTED non-2012 transit survived")
if 'color=\\"#000000\\", linewidth=0.72, alpha=0.92' not in source:
    raise RuntimeError("REJECTED black crosshair missing")
for label in (
    "Ecliptic Reference",
    "Earth Track From Ecliptic",
    "Projected Venus Transit Track",
    "Venus Transit Track From Ecliptic",
):
    if label not in source:
        raise RuntimeError(f"REJECTED missing title-case label: {label}")

exec(compile(source, "VENUS_TRANSIT_2012_ECLIPTIC_CROSSHAIR_V0149.py", "exec"))
# V0149