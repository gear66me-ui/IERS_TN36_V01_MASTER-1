# V0089G
# Audit reference: complete palette-variant launcher for the verified V0089E JPL/Matplotlib widget; no AI images.
from __future__ import annotations

import os
import re
import urllib.request
from pathlib import Path

VERSION = "V0089G"
REPO_RAW = "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/VENUS_1769_V0027_FORMAT_STANDALONE_V0089E_LIGHT_BLUE_CONTACTS.py"
BASE = Path("/content")
SOURCE = BASE / "VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_BASE_FOR_V0089G.py"
print("CODE INPUTS")
print("Version: V0089G")
print("Base verified widget: V0089E light-blue contact version")
print("Palette variant: deep-space violet / coral / mint / amber")
print("Plotting: Python/Matplotlib only; no AI images")
print("COMMENTS")
print("Downloading verified V0089E source and applying a complete color-palette replacement only.")
print("Geometry, equations, JPL Horizons vector queries, contact solver, and 900 DPI output remain unchanged.")
urllib.request.urlretrieve(REPO_RAW, SOURCE)
code = SOURCE.read_text(encoding="utf-8")
replacements = {
    'VERSION = "V0089E"': 'VERSION = "V0089G"',
    'OUT = ROOT / "VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_OUTPUT"': 'OUT = ROOT / "VENUS_1769_V0089G_DEEP_SPACE_PALETTE_OUTPUT"',
    'PNG = OUT / "VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_900DPI.png"': 'PNG = OUT / "VENUS_1769_V0089G_DEEP_SPACE_PALETTE_900DPI.png"',
    'PDF = OUT / "VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_VECTOR.pdf"': 'PDF = OUT / "VENUS_1769_V0089G_DEEP_SPACE_PALETTE_VECTOR.pdf"',
    'SVG = OUT / "VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_VECTOR.svg"': 'SVG = OUT / "VENUS_1769_V0089G_DEEP_SPACE_PALETTE_VECTOR.svg"',
    'CSV = OUT / "VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS.csv"': 'CSV = OUT / "VENUS_1769_V0089G_DEEP_SPACE_PALETTE.csv"',
    'color="#42D7C3"': 'color="#FF6B6B"',
    'color="#D89B18"': 'color="#7CFFCB"',
    'SUN_COLOR = "#FFD34A"': 'SUN_COLOR = "#E9D5FF"',
    'SUN_FILL_COLOR = "#D95A1B"': 'SUN_FILL_COLOR = "#4C1D95"',
    'SUN_FILL_ALPHA = 0.260': 'SUN_FILL_ALPHA = 0.330',
    'GUIDE_COLOR = "#263A4B"': 'GUIDE_COLOR = "#3B2F63"',
    'FG = "#F8FAFC"': 'FG = "#FFF7ED"',
    'MUTED = "#B8CBD6"': 'MUTED = "#D8B4FE"',
    'BG = "#000000"': 'BG = "#050014"',
    'TABLE_HEADER = "#23466F"': 'TABLE_HEADER = "#5B21B6"',
    'TABLE_TEAL = "#164B55"': 'TABLE_TEAL = "#065F46"',
    'TABLE_GOLD = "#563B0B"': 'TABLE_GOLD = "#7C2D12"',
    'TABLE_BODY = "#101A2E"': 'TABLE_BODY = "#180B2F"',
    'TABLE_BLUE2 = "#3E7CBF"': 'TABLE_BLUE2 = "#BE185D"',
    'TABLE_GRID = "#70879A"': 'TABLE_GRID = "#C4B5FD"',
    'contact rows recolored with lighter C1/C2 blue.': 'full alternate deep-space palette applied to Sun, tracks, disks, guides, labels, and all table row families.',
    'PASS: V0089E uses corrected V0067 equations; contact C1/C2 rows use lighter blue for contrast.': 'PASS: V0089G uses corrected V0067 equations; full alternate deep-space palette applied only to colors.'
}
for old, new in replacements.items():
    code = code.replace(old, new)
code = code.replace('# V0089E', '# V0089G')
# Safety guard: preserve the verified equation path and JPL runtime vector fetch.
required = [
    'Horizons(id=target, location=location',
    'vectors(refplane="ecliptic", aberrations="geometric", cache=False)',
    'def geocentric_ca_v0067',
    'DPI = 900',
    'matplotlib.use("Agg", force=True)',
]
missing = [item for item in required if item not in code]
if missing:
    raise RuntimeError(f"Palette widget safety audit failed; missing required source tokens: {missing}")
exec(compile(code, str(SOURCE), "exec"), {"__name__": "__main__"})
# V0089G
