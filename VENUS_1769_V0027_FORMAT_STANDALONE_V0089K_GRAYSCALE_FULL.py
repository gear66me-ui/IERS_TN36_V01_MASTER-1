# V0089K
# Audit reference: full V0089 publication geometry with grayscale palette only; Python/Matplotlib; no AI images.
from __future__ import annotations

import urllib.request
from pathlib import Path

VERSION = "V0089K"
BASE_URLS = [
    "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/VENUS_1769_V0027_FORMAT_STANDALONE_V0089E_LIGHT_BLUE_CONTACTS.py",
    "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/VENUS_1769_V0027_FORMAT_STANDALONE_V0089D_CONTACT_COLORS.py",
]
SOURCE = Path("/content/VENUS_1769_V0089K_GRAYSCALE_SOURCE.py")


def download_source() -> str:
    last_error = None
    for url in BASE_URLS:
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                text = response.read().decode("utf-8")
            if "Horizons" in text and "matplotlib" in text and "def plot" in text:
                return text
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not download verified V0089 source: {last_error}")


code = download_source()

# Identity / output replacement.
identity_replacements = {
    'VERSION = "V0089E"': 'VERSION = "V0089K"',
    'VERSION = "V0089D"': 'VERSION = "V0089K"',
    'VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_OUTPUT': 'VENUS_1769_V0089K_GRAYSCALE_FULL_OUTPUT',
    'VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_900DPI.png': 'VENUS_1769_V0089K_GRAYSCALE_FULL_900DPI.png',
    'VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_VECTOR.pdf': 'VENUS_1769_V0089K_GRAYSCALE_FULL_VECTOR.pdf',
    'VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_VECTOR.svg': 'VENUS_1769_V0089K_GRAYSCALE_FULL_VECTOR.svg',
    'VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS.csv': 'VENUS_1769_V0089K_GRAYSCALE_FULL.csv',
    'VENUS_1769_V0089D_CONTACT_COLORS_OUTPUT': 'VENUS_1769_V0089K_GRAYSCALE_FULL_OUTPUT',
    'VENUS_1769_V0089D_CONTACT_COLORS_900DPI.png': 'VENUS_1769_V0089K_GRAYSCALE_FULL_900DPI.png',
    'VENUS_1769_V0089D_CONTACT_COLORS_VECTOR.pdf': 'VENUS_1769_V0089K_GRAYSCALE_FULL_VECTOR.pdf',
    'VENUS_1769_V0089D_CONTACT_COLORS_VECTOR.svg': 'VENUS_1769_V0089K_GRAYSCALE_FULL_VECTOR.svg',
    'VENUS_1769_V0089D_CONTACT_COLORS.csv': 'VENUS_1769_V0089K_GRAYSCALE_FULL.csv',
    'contact-row color repair': 'full grayscale publication palette',
    'contact row coloring updated': 'full grayscale publication palette applied',
    'contact rows recolored by event class': 'full grayscale publication palette applied by event class',
    'light-blue contact version': 'grayscale full publication version',
}
for old, new in identity_replacements.items():
    code = code.replace(old, new)

# Full grayscale palette replacement. Geometry and equations are untouched.
palette_replacements = {
    'color="#42D7C3"': 'color="#FFFFFF"',      # Point Venus: white
    'color="#D89B18"': 'color="#B8B8B8"',      # Vardø: silver gray
    'SUN_COLOR = "#FFD34A"': 'SUN_COLOR = "#F5F5F5"',
    'SUN_FILL_COLOR = "#D95A1B"': 'SUN_FILL_COLOR = "#7A7A7A"',
    'SUN_FILL_ALPHA = 0.260': 'SUN_FILL_ALPHA = 0.240',
    'GUIDE_COLOR = "#263A4B"': 'GUIDE_COLOR = "#444444"',
    'FG = "#F8FAFC"': 'FG = "#FFFFFF"',
    'MUTED = "#B8CBD6"': 'MUTED = "#D5D5D5"',
    'BG = "#000000"': 'BG = "#000000"',
    'TABLE_HEADER = "#23466F"': 'TABLE_HEADER = "#181818"',
    'TABLE_TEAL = "#164B55"': 'TABLE_TEAL = "#303030"',
    'TABLE_GOLD = "#563B0B"': 'TABLE_GOLD = "#585858"',
    'TABLE_BODY = "#101A2E"': 'TABLE_BODY = "#070707"',
    'TABLE_BLUE2 = "#173A63"': 'TABLE_BLUE2 = "#454545"',
    'TABLE_BLUE2 = "#6EA8D9"': 'TABLE_BLUE2 = "#454545"',
    'TABLE_GRID = "#70879A"': 'TABLE_GRID = "#BFBFBF"',
    'NO AI IMAGES — Python/Matplotlib only. Standalone JPL Horizons geometric ecliptic vector reconstruction; V0067 seconds-space geocentric CA restored.': 'NO AI IMAGES — Python/Matplotlib only. Full grayscale publication palette; V0067 seconds-space geocentric CA restored.',
}
for old, new in palette_replacements.items():
    code = code.replace(old, new)

# Preserve contract: the executed source is still the full V0089 code path; only constants/colors/output names change.
SOURCE.write_text(code, encoding="utf-8")
exec(compile(code, str(SOURCE), "exec"), {"__name__": "__main__"})
# V0089K
