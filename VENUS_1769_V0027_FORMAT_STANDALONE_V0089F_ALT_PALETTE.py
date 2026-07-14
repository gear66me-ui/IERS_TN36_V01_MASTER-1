# V0089F
# Audit reference: alternate color palette wrapper for V0089E; Python/Matplotlib/JPL widget; no AI images.
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

VERSION = "V0089F"
SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/VENUS_1769_V0027_FORMAT_STANDALONE_V0089E_LIGHT_BLUE_CONTACTS.py"
SOURCE = Path("/content/VENUS_1769_V0027_FORMAT_STANDALONE_V0089E_LIGHT_BLUE_CONTACTS.py")

print("CODE INPUTS")
print(f"Version: {VERSION}")
print("Task: alternate publication color palette only; V0089E equations and geometry retained.")
print("COMMENTS")
print("Downloading the prior standalone widget and applying a palette-only source transformation before execution.")
print("No AI images. Python/Matplotlib only. JPL Horizons vectors are still fetched by the executed widget.")

subprocess.check_call(["curl", "-L", "-o", str(SOURCE), SOURCE_URL])
text = SOURCE.read_text(encoding="utf-8")

replacements = {
    'VERSION = "V0089E"': 'VERSION = "V0089F"',
    'OUT = ROOT / "VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_OUTPUT"': 'OUT = ROOT / "VENUS_1769_V0089F_ALT_PALETTE_OUTPUT"',
    'PNG = OUT / "VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_900DPI.png"': 'PNG = OUT / "VENUS_1769_V0089F_ALT_PALETTE_900DPI.png"',
    'PDF = OUT / "VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_VECTOR.pdf"': 'PDF = OUT / "VENUS_1769_V0089F_ALT_PALETTE_VECTOR.pdf"',
    'SVG = OUT / "VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_VECTOR.svg"': 'SVG = OUT / "VENUS_1769_V0089F_ALT_PALETTE_VECTOR.svg"',
    'CSV = OUT / "VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS.csv"': 'CSV = OUT / "VENUS_1769_V0089F_ALT_PALETTE.csv"',
    'color="#42D7C3"': 'color="#7DD3FC"',
    'color="#D89B18"': 'color="#F472B6"',
    'SUN_COLOR = "#FFD34A"': 'SUN_COLOR = "#FDE68A"',
    'SUN_FILL_COLOR = "#D95A1B"': 'SUN_FILL_COLOR = "#7C2D12"',
    'SUN_FILL_ALPHA = 0.260': 'SUN_FILL_ALPHA = 0.300',
    'GUIDE_COLOR = "#263A4B"': 'GUIDE_COLOR = "#334155"',
    'FG = "#F8FAFC"': 'FG = "#F8FAFC"',
    'MUTED = "#B8CBD6"': 'MUTED = "#CBD5E1"',
    'TABLE_HEADER = "#23466F"': 'TABLE_HEADER = "#312E81"',
    'TABLE_TEAL = "#164B55"': 'TABLE_TEAL = "#0F766E"',
    'TABLE_GOLD = "#563B0B"': 'TABLE_GOLD = "#7C2D12"',
    'TABLE_BODY = "#101A2E"': 'TABLE_BODY = "#111827"',
    'TABLE_BLUE2 = "#2F6FA8"': 'TABLE_BLUE2 = "#2563EB"',
    'TABLE_GRID = "#70879A"': 'TABLE_GRID = "#94A3B8"',
    'Publication plot retained; C1/C2 contact rows changed to lighter blue for readability. No AI images.': 'Publication plot retained; alternate palette applied for visual comparison only. No AI images.',
    'PASS: V0089E uses corrected V0067 equations; C1/C2 rows use a lighter blue contact color.': 'PASS: V0089F uses corrected V0067 equations; alternate palette only; geometry unchanged.',
    '# V0089E': '# V0089F',
}
for old, new in replacements.items():
    text = text.replace(old, new)

# Make title/subtitle reflect palette experiment without changing geometry.
text = text.replace(
    'fig.suptitle("1769 Venus Transit Between Vardø, Norway, and Point Venus, Tahiti", fontsize=15, fontweight="bold", y=0.970)',
    'fig.suptitle("1769 Venus Transit Between Vardø, Norway, and Point Venus, Tahiti", fontsize=15, fontweight="bold", y=0.970)'
)

code_obj = compile(text, "VENUS_1769_V0027_FORMAT_STANDALONE_V0089F_ALT_PALETTE_GENERATED", "exec")
exec_globals = {"__name__": "__main__", "__file__": str(SOURCE)}
exec(code_obj, exec_globals)

# V0089F
