# V0089G
# Audit reference: final clipped-boundary widget; V0089F final source preserved; savefig bounding box clipped to figure/table line; no AI images.
from __future__ import annotations

import urllib.request

VERSION = "V0089G"
SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/VENUS_1769_V0027_FORMAT_STANDALONE_V0089F_FINAL.py"
SOURCE_NAME = "VENUS_1769_V0027_FORMAT_STANDALONE_V0089F_FINAL.py"
FILE_NAME = "VENUS_1769_V0027_FORMAT_STANDALONE_V0089G_FINAL_CLIPPED.py"

print("CODE INPUTS")
print(f"Version: {VERSION}")
print(f"Source: {SOURCE_NAME}")
print("Patch: preserve plot format; clip saved PNG/PDF/SVG at the fixed figure boundary/table line by removing bbox_inches='tight'.")

source = urllib.request.urlopen(SOURCE_URL, timeout=60).read().decode("utf-8")
replacements = {
    "# V0089F\n": "# V0089G\n",
    "# V0089F": "# V0089G",
    "VERSION = \"V0089F\"": "VERSION = \"V0089G\"",
    "VENUS_1769_V0027_FORMAT_STANDALONE_V0089F_FINAL.py": FILE_NAME,
    "VENUS_1769_V0089F_FINAL_OUTPUT": "VENUS_1769_V0089G_FINAL_CLIPPED_OUTPUT",
    "VENUS_1769_V0089F_FINAL_900DPI.png": "VENUS_1769_V0089G_FINAL_CLIPPED_900DPI.png",
    "VENUS_1769_V0089F_FINAL_VECTOR.pdf": "VENUS_1769_V0089G_FINAL_CLIPPED_VECTOR.pdf",
    "VENUS_1769_V0089F_FINAL_VECTOR.svg": "VENUS_1769_V0089G_FINAL_CLIPPED_VECTOR.svg",
    "VENUS_1769_V0089F_FINAL.csv": "VENUS_1769_V0089G_FINAL_CLIPPED.csv",
    "Final V0089E-derived publication widget.": "Final V0089E-derived publication widget; clipped at fixed figure/table boundary.",
    "fig.savefig(PNG, dpi=DPI, bbox_inches=\"tight\", pad_inches=0.02, facecolor=BG)": "fig.savefig(PNG, dpi=DPI, facecolor=BG)",
    "fig.savefig(PDF, bbox_inches=\"tight\", pad_inches=0.02, facecolor=BG)": "fig.savefig(PDF, facecolor=BG)",
    "fig.savefig(SVG, bbox_inches=\"tight\", pad_inches=0.02, facecolor=BG)": "fig.savefig(SVG, facecolor=BG)",
}
for old, new in replacements.items():
    if old not in source:
        raise RuntimeError(f"Patch anchor missing: {old}")
    source = source.replace(old, new)

required = {
    "version": 'VERSION = "V0089G"' in source,
    "filename": FILE_NAME in source,
    "png_fixed_boundary": 'fig.savefig(PNG, dpi=DPI, facecolor=BG)' in source,
    "pdf_fixed_boundary": 'fig.savefig(PDF, facecolor=BG)' in source,
    "svg_fixed_boundary": 'fig.savefig(SVG, facecolor=BG)' in source,
    "no_tight_bbox": 'bbox_inches="tight"' not in source,
    "no_ai_images": "NO AI IMAGES" in source,
}
failed = [k for k, ok in required.items() if not ok]
if failed:
    raise RuntimeError(f"V0089G patch audit failed: {failed}")

print("COMMENTS")
print("The plot layout is unchanged; only the saved canvas clipping behavior is changed to avoid the right-side black extension from long footer text.")
print("RESULTS")
for k, ok in required.items():
    print(f"{k}: {'PASS' if ok else 'FAIL'}")

exec(compile(source, FILE_NAME.replace('.py', '_EXPANDED.py'), "exec"), globals())
# V0089G