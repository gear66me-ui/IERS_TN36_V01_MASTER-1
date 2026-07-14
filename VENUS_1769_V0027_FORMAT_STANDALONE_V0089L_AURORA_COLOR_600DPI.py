# V0089L
# Audit reference: full V0089 publication geometry with aurora color palette only; Python/Matplotlib; no AI images.
from __future__ import annotations

import urllib.request
from pathlib import Path

VERSION = "V0089L"
BASE_URLS = [
    "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/VENUS_1769_V0027_FORMAT_STANDALONE_V0089E_LIGHT_BLUE_CONTACTS.py",
    "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/VENUS_1769_V0027_FORMAT_STANDALONE_V0089D_CONTACT_COLORS.py",
]
SOURCE = Path("/content/VENUS_1769_V0089L_AURORA_COLOR_SOURCE.py")


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

identity_replacements = {
    'VERSION = "V0089E"': 'VERSION = "V0089L"',
    'VERSION = "V0089D"': 'VERSION = "V0089L"',
    'VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_OUTPUT': 'VENUS_1769_V0089L_AURORA_COLOR_600DPI_OUTPUT',
    'VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_900DPI.png': 'VENUS_1769_V0089L_AURORA_COLOR_600DPI.png',
    'VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_VECTOR.pdf': 'VENUS_1769_V0089L_AURORA_COLOR_VECTOR.pdf',
    'VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_VECTOR.svg': 'VENUS_1769_V0089L_AURORA_COLOR_VECTOR.svg',
    'VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS.csv': 'VENUS_1769_V0089L_AURORA_COLOR.csv',
    'VENUS_1769_V0089D_CONTACT_COLORS_OUTPUT': 'VENUS_1769_V0089L_AURORA_COLOR_600DPI_OUTPUT',
    'VENUS_1769_V0089D_CONTACT_COLORS_900DPI.png': 'VENUS_1769_V0089L_AURORA_COLOR_600DPI.png',
    'VENUS_1769_V0089D_CONTACT_COLORS_VECTOR.pdf': 'VENUS_1769_V0089L_AURORA_COLOR_VECTOR.pdf',
    'VENUS_1769_V0089D_CONTACT_COLORS_VECTOR.svg': 'VENUS_1769_V0089L_AURORA_COLOR_VECTOR.svg',
    'VENUS_1769_V0089D_CONTACT_COLORS.csv': 'VENUS_1769_V0089L_AURORA_COLOR.csv',
    'contact-row color repair': 'aurora color publication palette, 600 DPI raster',
    'contact row coloring updated': 'aurora color publication palette applied, 600 DPI raster',
    'contact rows recolored by event class': 'aurora color publication palette applied by event class, 600 DPI raster',
    'light-blue contact version': 'aurora color publication version, 600 DPI raster',
    'Raster DPI: 900; vector outputs: PDF and SVG': 'Raster DPI: 600; vector outputs: PDF and SVG',
}
for old, new in identity_replacements.items():
    code = code.replace(old, new)

palette_replacements = {
    'DPI = 900': 'DPI = 600',
    'DPI=900': 'DPI=600',
    'color="#42D7C3"': 'color="#00E5FF"',      # Point Venus: electric cyan
    'color="#D89B18"': 'color="#FF4D8D"',      # Vardø: aurora rose
    'SUN_COLOR = "#FFD34A"': 'SUN_COLOR = "#FFD166"',
    'SUN_FILL_COLOR = "#D95A1B"': 'SUN_FILL_COLOR = "#4C1D95"',
    'SUN_FILL_ALPHA = 0.260': 'SUN_FILL_ALPHA = 0.300',
    'GUIDE_COLOR = "#263A4B"': 'GUIDE_COLOR = "#385A66"',
    'FG = "#F8FAFC"': 'FG = "#F7F7FF"',
    'MUTED = "#B8CBD6"': 'MUTED = "#B9F2FF"',
    'BG = "#000000"': 'BG = "#020617"',
    'TABLE_HEADER = "#23466F"': 'TABLE_HEADER = "#5B21B6"',
    'TABLE_TEAL = "#164B55"': 'TABLE_TEAL = "#0F766E"',
    'TABLE_GOLD = "#563B0B"': 'TABLE_GOLD = "#B45309"',
    'TABLE_BODY = "#101A2E"': 'TABLE_BODY = "#111827"',
    'TABLE_BLUE2 = "#173A63"': 'TABLE_BLUE2 = "#1D4ED8"',
    'TABLE_BLUE2 = "#6EA8D9"': 'TABLE_BLUE2 = "#2563EB"',
    'TABLE_GRID = "#70879A"': 'TABLE_GRID = "#67E8F9"',
    'NO AI IMAGES — Python/Matplotlib only. Standalone JPL Horizons geometric ecliptic vector reconstruction; V0067 seconds-space geocentric CA restored.': 'NO AI IMAGES — Python/Matplotlib only. Aurora color publication palette; 600 DPI raster; V0067 seconds-space geocentric CA restored.',
}
for old, new in palette_replacements.items():
    code = code.replace(old, new)

# Preserve contract: full V0089 code path executes; only colors, output names, and PNG DPI change.
SOURCE.write_text(code, encoding="utf-8")
exec(compile(code, str(SOURCE), "exec"), {"__name__": "__main__"})
# V0089L
