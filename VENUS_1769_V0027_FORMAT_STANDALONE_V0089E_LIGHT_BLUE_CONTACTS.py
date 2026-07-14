# V0089E
# Audit reference: Colab Python/Matplotlib/JPL widget; V0089D geometry preserved; C1/C2 contact-row blue changed lighter; no AI images.
from __future__ import annotations

import urllib.request
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/VENUS_1769_V0027_FORMAT_STANDALONE_V0089D_CONTACT_COLORS.py"
SOURCE_NAME = "VENUS_1769_V0027_FORMAT_STANDALONE_V0089D_CONTACT_COLORS.py"
PATCHED_NAME = "VENUS_1769_V0027_FORMAT_STANDALONE_V0089E_LIGHT_BLUE_CONTACTS_EXPANDED.py"

print("CODE INPUTS")
print("Version: V0089E")
print("Source: V0089D corrected V0067-equation standalone widget")
print("Patch: contact-time table C1/C2 rows changed to lighter blue; geometry and all numerical equations preserved")

text = urllib.request.urlopen(SOURCE_URL, timeout=60).read().decode("utf-8")
replacements = {
    "# V0089D\n": "# V0089E\n",
    "# V0089D": "# V0089E",
    "VERSION = \"V0089D\"": "VERSION = \"V0089E\"",
    "VENUS_1769_V0089D_CONTACT_COLORS_OUTPUT": "VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_OUTPUT",
    "VENUS_1769_V0089D_CONTACT_COLORS_900DPI.png": "VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_900DPI.png",
    "VENUS_1769_V0089D_CONTACT_COLORS_VECTOR.pdf": "VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_VECTOR.pdf",
    "VENUS_1769_V0089D_CONTACT_COLORS_VECTOR.svg": "VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_VECTOR.svg",
    "VENUS_1769_V0089D_CONTACT_COLORS.csv": "VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS.csv",
    "TABLE_BLUE2 = \"#173A63\"": "TABLE_BLUE2 = \"#2F6FA3\"",
    "Publication plot retained; Halley reduction rows removed from the plot; contact rows recolored by event class. No AI images.": "Publication plot retained; Halley reduction rows removed from the plot; C1/C2 contact rows changed to lighter blue for stronger separation. No AI images.",
    "PASS: V0089D uses corrected V0067 equations; Halley audit rows removed from plot; contact row coloring updated.": "PASS: V0089E uses corrected V0067 equations; Halley audit rows removed from plot; C1/C2 contact-row blue lightened."
}
for old, new in replacements.items():
    text = text.replace(old, new)

Path(PATCHED_NAME).write_text(text, encoding="utf-8")
print("COMMENTS")
print("Expanded patched source written locally before execution; no AI images; Matplotlib only.")
print("RESULTS")
print(f"Expanded source: {PATCHED_NAME}")
print("Running expanded widget now...")

compiled = compile(text, PATCHED_NAME, "exec")
exec_globals = {"__name__": "__main__", "__file__": PATCHED_NAME}
exec(compiled, exec_globals)
# V0089E
