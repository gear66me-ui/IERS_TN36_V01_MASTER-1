# V0089H
# Audit reference: clean-footer Colab Python/Matplotlib/JPL widget; V0089G format preserved; publication footer wording removes AI-image comment.
from __future__ import annotations

import urllib.request
from pathlib import Path

VERSION = "V0089H"
REPO = "gear66me-ui/IERS_TN36_V01_MASTER-1"
SOURCE_NAME = "VENUS_1769_V0027_FORMAT_STANDALONE_V0089G_FINAL_CLIPPED.py"
FILE_NAME = "VENUS_1769_V0027_FORMAT_STANDALONE_V0089H_FINAL_CLEAN_FOOTER.py"
RAW_SOURCE_URL = f"https://raw.githubusercontent.com/{REPO}/main/{SOURCE_NAME}"
RAW_THIS_URL = f"https://raw.githubusercontent.com/{REPO}/main/{FILE_NAME}"
GITHUB_THIS_URL = f"https://github.com/{REPO}/blob/main/{FILE_NAME}"

print("CODE INPUTS")
print(f"Version: {VERSION}")
print(f"Source: {RAW_SOURCE_URL}")
print("Patch: preserve V0089G format and clipped save boundary; remove the publication footer phrase requested by user")
print("COMMENTS")
print("This widget performs no geometry edits and no plot-format changes except the footer wording update.")

source = urllib.request.urlopen(RAW_SOURCE_URL, timeout=60).read().decode("utf-8")

replacements = {
    "# V0089G": "# V0089H",
    "VERSION = \"V0089G\"": "VERSION = \"V0089H\"",
    "FILE_NAME = \"VENUS_1769_V0027_FORMAT_STANDALONE_V0089G_FINAL_CLIPPED.py\"": "FILE_NAME = \"VENUS_1769_V0027_FORMAT_STANDALONE_V0089H_FINAL_CLEAN_FOOTER.py\"",
    "OUT = ROOT / \"VENUS_1769_V0089G_FINAL_CLIPPED_OUTPUT\"": "OUT = ROOT / \"VENUS_1769_V0089H_FINAL_CLEAN_FOOTER_OUTPUT\"",
    "PNG = OUT / \"VENUS_1769_V0089G_FINAL_CLIPPED_900DPI.png\"": "PNG = OUT / \"VENUS_1769_V0089H_FINAL_CLEAN_FOOTER_900DPI.png\"",
    "PDF = OUT / \"VENUS_1769_V0089G_FINAL_CLIPPED_VECTOR.pdf\"": "PDF = OUT / \"VENUS_1769_V0089H_FINAL_CLEAN_FOOTER_VECTOR.pdf\"",
    "SVG = OUT / \"VENUS_1769_V0089G_FINAL_CLIPPED_VECTOR.svg\"": "SVG = OUT / \"VENUS_1769_V0089H_FINAL_CLEAN_FOOTER_VECTOR.svg\"",
    "CSV = OUT / \"VENUS_1769_V0089G_FINAL_CLIPPED.csv\"": "CSV = OUT / \"VENUS_1769_V0089H_FINAL_CLEAN_FOOTER.csv\"",
    "Final V0089E-derived publication widget.": "Final V0089E-derived publication widget.",
    "NO AI IMAGES — Python/Matplotlib/JPL Horizons only. Final V0089E-derived publication widget.": "Python/Matplotlib/JPL Horizons geometric vector reconstruction. Final V0089E-derived publication widget.",
    "V0089G format preserved; clipped save boundary retained; footer includes public retrieval address.": "V0089G format preserved; clipped save boundary retained; publication footer wording cleaned.",
    "print(\"PASS: V0089G format preserved; clipped save boundary retained; footer includes public retrieval address.\")": "print(\"PASS: V0089G format preserved; clipped save boundary retained; publication footer wording cleaned.\")",
    "# V0089G": "# V0089H",
}

for old, new in replacements.items():
    source = source.replace(old, new)

source = source.replace(RAW_SOURCE_URL, RAW_THIS_URL)
source = source.replace(f"https://github.com/{REPO}/blob/main/{SOURCE_NAME}", GITHUB_THIS_URL)
source = source.replace("NO AI IMAGES", "Python/Matplotlib/JPL Horizons")

required_absent = ["NO AI IMAGES"]
failed_absent = [item for item in required_absent if item in source]
if failed_absent:
    raise RuntimeError(f"Footer cleanup failed; remaining text: {failed_absent}")

required_present = [
    "VERSION = \"V0089H\"",
    "VENUS_1769_V0027_FORMAT_STANDALONE_V0089H_FINAL_CLEAN_FOOTER.py",
    "bbox_inches=None",
    "Python/Matplotlib/JPL Horizons geometric vector reconstruction. Final V0089E-derived publication widget.",
]
failed_present = [item for item in required_present if item not in source]
if failed_present:
    raise RuntimeError(f"V0089H audit failed; missing: {failed_present}")

print("RESULTS")
print("V0089H source patch audit: PASS")
print("OUTPUT SUMMARY")
print(f"This file: {GITHUB_THIS_URL}")
print(f"Raw URL: {RAW_THIS_URL}")
print("PAPER COMPARISON")
print("No scientific values changed from V0089G.")
print("EQUATION STATUS")
print("PASS: V0089G equations and clipped plot boundary preserved; only footer wording changed.")

exec(compile(source, FILE_NAME.replace(".py", "_EXPANDED.py"), "exec"), globals())
# V0089H