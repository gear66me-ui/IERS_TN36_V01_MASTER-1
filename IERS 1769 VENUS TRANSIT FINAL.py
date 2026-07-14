# V0089K
# Audit reference: exact notebook-name publication widget; robust footer patch; V0089G clipped layout and geometry preserved.
from __future__ import annotations

import re
import urllib.request

VERSION = "V0089K"
NOTEBOOK_TITLE = "IERS 1769 VENUS TRANSIT FINAL.ipynb"
NOTEBOOK_ID = "1w10eJgjYJDnOPGoVHSFZXzq9izVuCgN_"
NOTEBOOK_URL = f"https://colab.research.google.com/drive/{NOTEBOOK_ID}"
REPO = "gear66me-ui/IERS_TN36_V01_MASTER-1"
FILE_NAME = "IERS 1769 VENUS TRANSIT FINAL.py"
RAW_FILE_NAME = FILE_NAME.replace(" ", "%20")
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/main/{RAW_FILE_NAME}"
GITHUB_URL = f"https://github.com/{REPO}/blob/main/{RAW_FILE_NAME}"
SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/VENUS_1769_V0027_FORMAT_STANDALONE_V0089G_FINAL_CLIPPED.py"

print("CODE INPUTS")
print(f"Version: {VERSION}")
print(f"Notebook: {NOTEBOOK_TITLE}")
print(f"Notebook ID: {NOTEBOOK_ID}")
print(f"Widget file: {FILE_NAME}")
print(f"Source: {SOURCE_URL}")
print("Patch: preserve V0089G clipped plot/layout; exact notebook-name widget file; robust footer rewrite")
print("COMMENTS")
print("This widget performs no geometry edits. It patches only filenames, version strings, output names, and visible footer metadata.")

source = urllib.request.urlopen(SOURCE_URL, timeout=60).read().decode("utf-8")

replacements = {
    "# V0089G": "# V0089K",
    'VERSION = "V0089G"': 'VERSION = "V0089K"',
    'FILE_NAME = "VENUS_1769_V0027_FORMAT_STANDALONE_V0089G_FINAL_CLIPPED.py"': f'FILE_NAME = "{FILE_NAME}"',
    'OUT = ROOT / "VENUS_1769_V0089G_FINAL_CLIPPED_OUTPUT"': 'OUT = ROOT / "IERS_1769_VENUS_TRANSIT_FINAL_OUTPUT"',
    'PNG = OUT / "VENUS_1769_V0089G_FINAL_CLIPPED_900DPI.png"': 'PNG = OUT / "IERS_1769_VENUS_TRANSIT_FINAL_900DPI.png"',
    'PDF = OUT / "VENUS_1769_V0089G_FINAL_CLIPPED_VECTOR.pdf"': 'PDF = OUT / "IERS_1769_VENUS_TRANSIT_FINAL_VECTOR.pdf"',
    'SVG = OUT / "VENUS_1769_V0089G_FINAL_CLIPPED_VECTOR.svg"': 'SVG = OUT / "IERS_1769_VENUS_TRANSIT_FINAL_VECTOR.svg"',
    'CSV = OUT / "VENUS_1769_V0089G_FINAL_CLIPPED.csv"': 'CSV = OUT / "IERS_1769_VENUS_TRANSIT_FINAL.csv"',
}
for old, new in replacements.items():
    source = source.replace(old, new)

source = re.sub(
    r'^\s*footer_1\s*=\s*.*$',
    '    footer_1 = "Python/Matplotlib/JPL Horizons geometric vector reconstruction. Final publication widget."',
    source,
    count=1,
    flags=re.MULTILINE,
)
source = re.sub(
    r'^\s*footer_2\s*=\s*.*$',
    f'    footer_2 = f"Notebook: {NOTEBOOK_TITLE} | ID: {NOTEBOOK_ID} | Colab: {NOTEBOOK_URL} | GitHub: {{GITHUB_URL}} | Raw: {{RAW_URL}} | File: {{FILE_NAME}} | Run: {{CURL_COMMAND}} ; {{RUN_COMMAND}}"',
    source,
    count=1,
    flags=re.MULTILINE,
)

required = [
    'VERSION = "V0089K"',
    f'FILE_NAME = "{FILE_NAME}"',
    NOTEBOOK_TITLE,
    NOTEBOOK_ID,
    'Python/Matplotlib/JPL Horizons geometric vector reconstruction. Final publication widget.',
]
missing = [x for x in required if x not in source]
if missing:
    raise RuntimeError(f"V0089K audit failed; missing: {missing}")
if "NO AI IMAGES" in source:
    raise RuntimeError("V0089K audit failed: removed phrase still present in executable source.")
if 'bbox_inches="tight"' in source:
    raise RuntimeError("V0089K audit failed: clipped save boundary from V0089G was not preserved.")

print("RESULTS")
print("Footer phrase removed: PASS")
print("Exact widget filename embedded: PASS")
print("Notebook title and ID embedded: PASS")
print("Clipped save boundary preserved: PASS")
print("OUTPUT SUMMARY")
print(f"GitHub file: {GITHUB_URL}")
print(f"Raw file: {RAW_URL}")
print(f"Notebook: {NOTEBOOK_URL}")
print("PAPER COMPARISON")
print("NOT USED: plotting metadata patch only; geometry source remains V0089G/V0067-equation path.")
print("EQUATION STATUS")
print("PASS: no geometry equations changed by this wrapper patch.")

exec(compile(source, FILE_NAME.replace('.py', '_EXPANDED.py'), globals()), globals())
# V0089K