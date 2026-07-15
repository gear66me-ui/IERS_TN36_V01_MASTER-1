# V0147
# Audit reference: 2012-only approved transit plot with black solar crosshair and title-case description labels; no AI images.

from __future__ import annotations

import urllib.request

VERSION = "V0147"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "VENUS_TRANSITS_1761_1769_1874_1882_2004_2012_ECLIPTIC_CROSSHAIR_V0146.py"
)

with urllib.request.urlopen(SOURCE_URL, timeout=90) as response:
    source = response.read().decode("utf-8")

required = [
    "# V0146",
    'VERSION = "V0146"',
    'TRANSITS = {',
    '2012: "2012-06-06 01:00"',
    'color="#D8D8D8", linewidth=0.42, alpha=0.42',
    'color="#D8D8D8", linewidth=0.42, alpha=0.30',
    '"Ecliptic reference  0.000°"',
    'TextArea("Ecliptic reference: 0.000°"',
    'TextArea(f"Earth track from ecliptic:',
    'TextArea(f"Projected Venus transit track:',
    'TextArea(f"Venus transit track from ecliptic:',
]
for marker in required:
    if marker not in source:
        raise RuntimeError(f"REJECTED missing V0146 marker: {marker}")

start = source.index("TRANSITS = {")
stop = source.index("}\n\nwith urllib.request.urlopen", start) + 1
source = source[:start] + 'TRANSITS = {\n    2012: "2012-06-06 01:00",\n}' + source[stop:]

source = source.replace("# V0146", "# V0147")
source = source.replace('VERSION = "V0146"', 'VERSION = "V0147"')
source = source.replace("V0146_OUTPUT", "V0147_OUTPUT")
source = source.replace("V0146.png", "V0147.png")
source = source.replace("V0146.csv", "V0147.csv")
source = source.replace("ECLIPTIC_CROSSHAIR_V0146.py", "ECLIPTIC_CROSSHAIR_V0147.py")

source = source.replace(
    'color="#D8D8D8", linewidth=0.42, alpha=0.42',
    'color="#000000", linewidth=0.72, alpha=0.92',
)
source = source.replace(
    'color="#D8D8D8", linewidth=0.42, alpha=0.30',
    'color="#000000", linewidth=0.72, alpha=0.92',
)
source = source.replace(
    '"Ecliptic reference  0.000°", color="#D8D8D8"',
    '"Ecliptic Reference  0.000°", color="#000000"',
)

capitalization = {
    '"Ecliptic reference: 0.000°"': '"Ecliptic Reference: 0.000°"',
    'f"Earth track from ecliptic:': 'f"Earth Track From Ecliptic:',
    'f"Projected Venus transit track:': 'f"Projected Venus Transit Track:',
    'f"Venus transit track from ecliptic:': 'f"Venus Transit Track From Ecliptic:',
    '("Earth track from ecliptic",': '("Earth Track From Ecliptic",',
    '("Projected Venus transit track",': '("Projected Venus Transit Track",',
    '("Venus transit track from ecliptic",': '("Venus Transit Track From Ecliptic",',
    '{"Earth track from ecliptic", "Venus transit track from ecliptic"}': '{"Earth Track From Ecliptic", "Venus Transit Track From Ecliptic"}',
}
for old, new in capitalization.items():
    source = source.replace(old, new)

if source.splitlines()[0] != "# V0147":
    raise RuntimeError("REJECTED incorrect first line")
if source.splitlines()[-1] != "# V0147":
    raise RuntimeError("REJECTED incorrect last line")
if '1761: "1761-06-06 06:00"' in source:
    raise RuntimeError("REJECTED non-2012 transit survived")
if 'color="#000000", linewidth=0.72, alpha=0.92' not in source:
    raise RuntimeError("REJECTED black crosshair missing")
for label in [
    "Ecliptic Reference",
    "Earth Track From Ecliptic",
    "Projected Venus Transit Track",
    "Venus Transit Track From Ecliptic",
]:
    if label not in source:
        raise RuntimeError(f"REJECTED title-case label missing: {label}")

exec(
    compile(
        source,
        "VENUS_TRANSIT_2012_ECLIPTIC_CROSSHAIR_V0147.py",
        "exec",
    )
)
# V0147