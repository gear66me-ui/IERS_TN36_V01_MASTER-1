# V0089I
# Audit reference: Notebook-footered publication widget; V0089G clipped layout preserved; publication footer wording repaired; notebook title and ID added.
from __future__ import annotations

import urllib.request

VERSION = "V0089I"
NOTEBOOK_TITLE = "IERS 1769 VENUS TRANSIT FINAL.ipynb"
NOTEBOOK_ID = "1w10eJgjYJDnOPGoVHSFZXzq9izVuCgN_"
NOTEBOOK_URL = f"https://colab.research.google.com/drive/{NOTEBOOK_ID}"
REPO = "gear66me-ui/IERS_TN36_V01_MASTER-1"
FILE_NAME = "IERS_1769_VENUS_TRANSIT_FINAL_1w10eJgjYJDnOPGoVHSFZXzq9izVuCgN__V0089I_NOTEBOOK_FOOTER.py"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/main/{FILE_NAME}"
GITHUB_URL = f"https://github.com/{REPO}/blob/main/{FILE_NAME}"
SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/VENUS_1769_V0027_FORMAT_STANDALONE_V0089G_FINAL_CLIPPED.py"

print("CODE INPUTS")
print(f"Version: {VERSION}")
print(f"Notebook: {NOTEBOOK_TITLE}")
print(f"Notebook ID: {NOTEBOOK_ID}")
print(f"Source: {SOURCE_URL}")
print("Patch: preserve V0089G clipped plot/layout; remove visible publication footer phrase requested by user; add notebook title and ID")
print("COMMENTS")
print("This widget performs no geometry edits. It patches only filenames, version strings, and visible footer metadata.")

source = urllib.request.urlopen(SOURCE_URL, timeout=60).read().decode("utf-8")

replacements = {
    "# V0089G": "# V0089I",
    "VERSION = \"V0089G\"": "VERSION = \"V0089I\"",
    'FILE_NAME = "VENUS_1769_V0027_FORMAT_STANDALONE_V0089G_FINAL_CLIPPED.py"': f'FILE_NAME = "{FILE_NAME}"',
    'OUT = ROOT / "VENUS_1769_V0089G_FINAL_CLIPPED_OUTPUT"': 'OUT = ROOT / "IERS_1769_VENUS_TRANSIT_FINAL_V0089I_OUTPUT"',
    'PNG = OUT / "VENUS_1769_V0089G_FINAL_CLIPPED_900DPI.png"': 'PNG = OUT / "IERS_1769_VENUS_TRANSIT_FINAL_V0089I_900DPI.png"',
    'PDF = OUT / "VENUS_1769_V0089G_FINAL_CLIPPED_VECTOR.pdf"': 'PDF = OUT / "IERS_1769_VENUS_TRANSIT_FINAL_V0089I_VECTOR.pdf"',
    'SVG = OUT / "VENUS_1769_V0089G_FINAL_CLIPPED_VECTOR.svg"': 'SVG = OUT / "IERS_1769_VENUS_TRANSIT_FINAL_V0089I_VECTOR.svg"',
    'CSV = OUT / "VENUS_1769_V0089G_FINAL_CLIPPED.csv"': 'CSV = OUT / "IERS_1769_VENUS_TRANSIT_FINAL_V0089I.csv"',
    'Final V0089E-derived publication widget.': 'Final publication widget.',
}
for old, new in replacements.items():
    source = source.replace(old, new)

source = source.replace('REPO_PATH = FILE_NAME', 'REPO_PATH = FILE_NAME')
source = source.replace('CURL_COMMAND = f"!curl -L -o {FILE_NAME} {RAW_URL}"', 'CURL_COMMAND = f"!curl -L -o {FILE_NAME} {RAW_URL}"')

footer_old_1 = '    footer_1 = "NO AI IMAGES — Python/Matplotlib/JPL Horizons only. Final V0089E-derived publication widget."'
footer_old_1b = '    footer_1 = "Python/Matplotlib/JPL Horizons geometric vector reconstruction. Final V0089E-derived publication widget."'
footer_new_1 = '    footer_1 = "Python/Matplotlib/JPL Horizons geometric vector reconstruction. Final publication widget."'
source = source.replace(footer_old_1, footer_new_1)
source = source.replace(footer_old_1b, footer_new_1)

footer_old_2 = '    footer_2 = f"GitHub: {GITHUB_URL} | Raw: {RAW_URL} | File: {FILE_NAME} | Run: {CURL_COMMAND} ; {RUN_COMMAND}"'
footer_new_2 = f'    footer_2 = f"Notebook: {NOTEBOOK_TITLE} | ID: {NOTEBOOK_ID} | Colab: {NOTEBOOK_URL} | GitHub: {{GITHUB_URL}} | Raw: {{RAW_URL}} | File: {{FILE_NAME}} | Run: {{CURL_COMMAND}} ; {{RUN_COMMAND}}"'
if footer_old_2 not in source:
    raise RuntimeError("V0089I footer anchor not found; source layout changed.")
source = source.replace(footer_old_2, footer_new_2)

if 'footer_1 = "NO AI IMAGES' in source or 'footer_2 = "NO AI IMAGES' in source:
    raise RuntimeError("V0089I audit failed: visible footer still contains the removed phrase.")
if NOTEBOOK_TITLE not in source or NOTEBOOK_ID not in source:
    raise RuntimeError("V0089I audit failed: notebook title or ID not embedded.")
if 'bbox_inches="tight"' in source:
    raise RuntimeError("V0089I audit failed: clipped save boundary from V0089G was not preserved.")

print("RESULTS")
print("Visible footer phrase removed: PASS")
print("Notebook title embedded: PASS")
print("Notebook ID embedded: PASS")
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
# V0089I
