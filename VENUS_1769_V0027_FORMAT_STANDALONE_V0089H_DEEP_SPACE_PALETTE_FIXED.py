# V0089H
# Audit reference: palette-only wrapper for verified V0089E Matplotlib/JPL widget; no AI images.
from __future__ import annotations

import urllib.request
from pathlib import Path

VERSION = "V0089H"
BASE_URL = "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/VENUS_1769_V0027_FORMAT_STANDALONE_V0089E_LIGHT_BLUE_CONTACTS.py"
SOURCE = Path("/content/VENUS_1769_V0089H_BASE_V0089E.py")

print("CODE INPUTS")
print(f"Version: {VERSION}")
print("Base verified widget: V0089E light-blue contact version")
print("Palette variant: complete deep-space violet / coral / mint / amber replacement")
print("Plotting: Python/Matplotlib only; no AI images")
print("COMMENTS")
print("Downloading verified V0089E source and applying palette replacement only.")
print("Geometry, equations, JPL Horizons vector queries, contact solver, layout, and 900 DPI output remain unchanged.")

code = urllib.request.urlopen(BASE_URL, timeout=60).read().decode("utf-8")

# Safety checks use broad semantic tokens so formatting changes in the source do not break the widget.
required = [
    "astroquery.jplhorizons",
    "Horizons(",
    ".vectors(",
    "geocentric_ca_v0067",
    "DPI = 900",
    "matplotlib.use",
    "NO AI IMAGES",
]
missing = [item for item in required if item not in code]
if missing:
    raise RuntimeError(f"Palette widget safety audit failed; missing required source tokens: {missing}")

replacements = {
    'VERSION = "V0089E"': 'VERSION = "V0089H"',
    'V0089E_LIGHT_BLUE_CONTACTS_OUTPUT': 'V0089H_DEEP_SPACE_PALETTE_FIXED_OUTPUT',
    'V0089E_LIGHT_BLUE_CONTACTS_900DPI.png': 'V0089H_DEEP_SPACE_PALETTE_FIXED_900DPI.png',
    'V0089E_LIGHT_BLUE_CONTACTS_VECTOR.pdf': 'V0089H_DEEP_SPACE_PALETTE_FIXED_VECTOR.pdf',
    'V0089E_LIGHT_BLUE_CONTACTS_VECTOR.svg': 'V0089H_DEEP_SPACE_PALETTE_FIXED_VECTOR.svg',
    'V0089E_LIGHT_BLUE_CONTACTS.csv': 'V0089H_DEEP_SPACE_PALETTE_FIXED.csv',
    '#42D7C3': '#7CF7D4',      # Point Venus: bright mint
    '#D89B18': '#FF6F91',      # Vardø: coral rose
    '#FFD34A': '#C7B8FF',      # Solar limb: lunar violet
    '#D95A1B': '#5B2A86',      # Solar fill: deep violet
    '#263A4B': '#4B5563',      # Guides: graphite slate
    '#F8FAFC': '#F7F3FF',      # Foreground: warm violet white
    '#B8CBD6': '#D7C7FF',      # Muted text: lavender
    '#000000': '#050314',      # Background: near-black indigo
    '#23466F': '#3B1E5C',      # Table header: royal plum
    '#164B55': '#145A6A',      # Teal rows: deep cyan
    '#563B0B': '#6A2C3B',      # Gold rows: wine-copper
    '#101A2E': '#120B2A',      # Body rows: indigo black
    '#4B82C3': '#70A6FF',      # Contact C1/C2: light electric blue
    '#70879A': '#8E7CC3',      # Table grid: lavender gray
}
for old, new in replacements.items():
    code = code.replace(old, new)

code = code.replace(
    "contact-row light-blue repair",
    "complete deep-space color-palette variant"
)
code = code.replace(
    "light-blue contact-row update",
    "deep-space full-palette update"
)
code = code.replace(
    "contact row coloring updated.",
    "complete deep-space palette applied; geometry unchanged."
)
code = code.replace(
    "# V0089E",
    "# V0089H"
)

exec(compile(code, str(SOURCE), "exec"), {"__name__": "__main__"})
# V0089H
