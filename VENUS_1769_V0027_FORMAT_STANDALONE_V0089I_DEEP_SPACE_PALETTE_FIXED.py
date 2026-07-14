# V0089I
# Audit reference: palette-only wrapper over verified V0089D source; no AI images; Python/Matplotlib/JPL only.
from __future__ import annotations

import urllib.request
from pathlib import Path

VERSION = "V0089I"
SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/VENUS_1769_V0027_FORMAT_STANDALONE_V0089D_CONTACT_COLORS.py"
SOURCE = Path("/content/VENUS_1769_V0089D_SOURCE_FOR_V0089I.py")

print("CODE INPUTS")
print(f"Version: {VERSION}")
print("Base verified widget: V0089D contact-colors version")
print("Palette variant: complete deep-space violet / coral / mint / amber replacement")
print("Plotting: Python/Matplotlib only; no AI images")
print("COMMENTS")
print("Downloading verified V0089D source and applying palette replacement only.")
print("Geometry, equations, JPL Horizons vector queries, contact solver, layout, and 900 DPI output remain unchanged.")

with urllib.request.urlopen(SOURCE_URL, timeout=45) as response:
    code = response.read().decode("utf-8")
SOURCE.write_text(code, encoding="utf-8")

if "# V0089D" not in code or "def geocentric_ca_v0067" not in code or "DPI = 900" not in code:
    raise RuntimeError("Palette wrapper could not verify the V0089D source file.")

replacements = {
    'VERSION = "V0089D"': 'VERSION = "V0089I"',
    'OUT = ROOT / "VENUS_1769_V0089D_CONTACT_COLORS_OUTPUT"': 'OUT = ROOT / "VENUS_1769_V0089I_DEEP_SPACE_PALETTE_OUTPUT"',
    'PNG = OUT / "VENUS_1769_V0089D_CONTACT_COLORS_900DPI.png"': 'PNG = OUT / "VENUS_1769_V0089I_DEEP_SPACE_PALETTE_900DPI.png"',
    'PDF = OUT / "VENUS_1769_V0089D_CONTACT_COLORS_VECTOR.pdf"': 'PDF = OUT / "VENUS_1769_V0089I_DEEP_SPACE_PALETTE_VECTOR.pdf"',
    'SVG = OUT / "VENUS_1769_V0089D_CONTACT_COLORS_VECTOR.svg"': 'SVG = OUT / "VENUS_1769_V0089I_DEEP_SPACE_PALETTE_VECTOR.svg"',
    'CSV = OUT / "VENUS_1769_V0089D_CONTACT_COLORS.csv"': 'CSV = OUT / "VENUS_1769_V0089I_DEEP_SPACE_PALETTE.csv"',
    'color="#42D7C3"': 'color="#FF6B7A"',
    'color="#D89B18"': 'color="#A78BFA"',
    'SUN_COLOR = "#FFD34A"': 'SUN_COLOR = "#F8C471"',
    'SUN_FILL_COLOR = "#D95A1B"': 'SUN_FILL_COLOR = "#5B2A86"',
    'SUN_FILL_ALPHA = 0.260': 'SUN_FILL_ALPHA = 0.315',
    'GUIDE_COLOR = "#263A4B"': 'GUIDE_COLOR = "#2A2045"',
    'FG = "#F8FAFC"': 'FG = "#FFF7ED"',
    'MUTED = "#B8CBD6"': 'MUTED = "#D8B4FE"',
    'BG = "#000000"': 'BG = "#06030F"',
    'TABLE_HEADER = "#23466F"': 'TABLE_HEADER = "#3B0764"',
    'TABLE_TEAL = "#164B55"': 'TABLE_TEAL = "#065F46"',
    'TABLE_GOLD = "#563B0B"': 'TABLE_GOLD = "#7C2D12"',
    'TABLE_BODY = "#101A2E"': 'TABLE_BODY = "#160B2E"',
    'TABLE_BLUE2 = "#5DADEC"': 'TABLE_BLUE2 = "#0F766E"',
    'TABLE_BLUE2 = "#173A63"': 'TABLE_BLUE2 = "#0F766E"',
    'TABLE_GRID = "#70879A"': 'TABLE_GRID = "#C084FC"',
    'Publication plot retained; Halley reduction rows removed from the plot; contact rows recolored by event class. No AI images.': 'Publication plot retained; complete alternate palette applied only. Halley reduction rows remain removed from the plot. No AI images.',
    'PASS: V0089D uses corrected V0067 equations; Halley audit rows removed from plot; contact row coloring updated.': 'PASS: V0089I uses corrected V0067 equations; complete palette replacement only; geometry unchanged.'
}

for old, new in replacements.items():
    code = code.replace(old, new)

code = code.replace('# V0089D', '# V0089I')
code = code.replace('NO AI IMAGES — Python/Matplotlib only. Standalone JPL Horizons geometric ecliptic vector reconstruction; V0067 seconds-space geocentric CA restored.', 'NO AI IMAGES — Python/Matplotlib only. Deep-space alternate palette; V0067 seconds-space geocentric CA restored.')

exec(compile(code, str(SOURCE), "exec"), {"__name__": "__main__"})
# V0089I
