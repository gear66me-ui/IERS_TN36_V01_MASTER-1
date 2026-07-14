# V0089I
# Audit reference: clean clipped publication widget; V0089F geometry preserved; fixed canvas clipping and public retrieval footer retained.
from __future__ import annotations

import urllib.request

VERSION = "V0089I"
REPO = "gear66me-ui/IERS_TN36_V01_MASTER-1"
SOURCE_NAME = "VENUS_1769_V0027_FORMAT_STANDALONE_V0089F_FINAL.py"
FILE_NAME = "VENUS_1769_V0027_FORMAT_STANDALONE_V0089I_FINAL_CLEAN_CLIPPED.py"
RAW_SOURCE_URL = f"https://raw.githubusercontent.com/{REPO}/main/{SOURCE_NAME}"
RAW_THIS_URL = f"https://raw.githubusercontent.com/{REPO}/main/{FILE_NAME}"
GITHUB_THIS_URL = f"https://github.com/{REPO}/blob/main/{FILE_NAME}"

print("CODE INPUTS")
print(f"Version: {VERSION}")
print(f"Source: {RAW_SOURCE_URL}")
print("Patch: preserve V0089F geometry and format; remove publication AI-image phrase; clip saved canvas at fixed figure/table boundary.")
print("COMMENTS")
print("This widget changes only file/version names, footer wording, and savefig clipping. No geometry or numerical equations are edited.")

source = urllib.request.urlopen(RAW_SOURCE_URL, timeout=60).read().decode("utf-8")
replacements = {
    "# V0089F\n": "# V0089I\n",
    "# V0089F": "# V0089I",
    "VERSION = \"V0089F\"": "VERSION = \"V0089I\"",
    "VENUS_1769_V0027_FORMAT_STANDALONE_V0089F_FINAL.py": FILE_NAME,
    "VENUS_1769_V0089F_FINAL_OUTPUT": "VENUS_1769_V0089I_FINAL_CLEAN_CLIPPED_OUTPUT",
    "VENUS_1769_V0089F_FINAL_900DPI.png": "VENUS_1769_V0089I_FINAL_CLEAN_CLIPPED_900DPI.png",
    "VENUS_1769_V0089F_FINAL_VECTOR.pdf": "VENUS_1769_V0089I_FINAL_CLEAN_CLIPPED_VECTOR.pdf",
    "VENUS_1769_V0089F_FINAL_VECTOR.svg": "VENUS_1769_V0089I_FINAL_CLEAN_CLIPPED_VECTOR.svg",
    "VENUS_1769_V0089F_FINAL.csv": "VENUS_1769_V0089I_FINAL_CLEAN_CLIPPED.csv",
    "NO AI IMAGES — Python/Matplotlib/JPL Horizons only. Final V0089E-derived publication widget.": "Python/Matplotlib/JPL Horizons geometric vector reconstruction. Final V0089E-derived publication widget.",
    "fig.savefig(PNG, dpi=DPI, bbox_inches=\"tight\", pad_inches=0.02, facecolor=BG)": "fig.savefig(PNG, dpi=DPI, facecolor=BG)",
    "fig.savefig(PDF, bbox_inches=\"tight\", pad_inches=0.02, facecolor=BG)": "fig.savefig(PDF, facecolor=BG)",
    "fig.savefig(SVG, bbox_inches=\"tight\", pad_inches=0.02, facecolor=BG)": "fig.savefig(SVG, facecolor=BG)",
    "Final V0089E-derived publication widget.": "Final V0089E-derived publication widget.",
}
for old, new in replacements.items():
    if old not in source:
        raise RuntimeError(f"Patch anchor missing: {old}")
    source = source.replace(old, new)

source = source.replace("no AI images.", "publication footer cleaned.")
source = source.replace("NO AI IMAGES", "Python/Matplotlib/JPL Horizons")
source = source.replace(f"https://raw.githubusercontent.com/{REPO}/main/{SOURCE_NAME}", RAW_THIS_URL)
source = source.replace(f"https://github.com/{REPO}/blob/main/{SOURCE_NAME}", GITHUB_THIS_URL)

required = {
    "version": 'VERSION = "V0089I"' in source,
    "file_name": FILE_NAME in source,
    "footer_clean": "Python/Matplotlib/JPL Horizons geometric vector reconstruction. Final V0089E-derived publication widget." in source,
    "png_fixed_boundary": 'fig.savefig(PNG, dpi=DPI, facecolor=BG)' in source,
    "pdf_fixed_boundary": 'fig.savefig(PDF, facecolor=BG)' in source,
    "svg_fixed_boundary": 'fig.savefig(SVG, facecolor=BG)' in source,
    "removed_requested_phrase": "NO AI IMAGES" not in source,
    "removed_tight_bbox": 'bbox_inches="tight"' not in source,
}
failed = [k for k, ok in required.items() if not ok]
if failed:
    raise RuntimeError(f"V0089I audit failed: {failed}")

print("RESULTS")
for key, ok in required.items():
    print(f"{key}: {'PASS' if ok else 'FAIL'}")
print("OUTPUT SUMMARY")
print(f"This file: {GITHUB_THIS_URL}")
print(f"Raw URL: {RAW_THIS_URL}")
print("PAPER COMPARISON")
print("No scientific values changed from V0089F.")
print("EQUATION STATUS")
print("PASS: V0089F equations preserved; footer phrase removed; fixed clipped save boundary retained.")

exec(compile(source, FILE_NAME.replace(".py", "_EXPANDED.py"), "exec"), globals())
# V0089I