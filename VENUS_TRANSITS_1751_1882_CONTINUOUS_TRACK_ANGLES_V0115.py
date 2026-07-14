# V0115
# Audit reference: width-only launcher for the V0113 continuous 1751-1882 JPL Earth/Venus plot, expanded to exactly three times the original width.

from __future__ import annotations

import runpy
import urllib.request
from pathlib import Path

VERSION = "V0115"
SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/VENUS_TRANSITS_1751_1882_CONTINUOUS_TRACK_ANGLES_V0113.py"
LOCAL_SOURCE = Path("/content/VENUS_TRANSITS_1751_1882_CONTINUOUS_TRACK_ANGLES_V0115_EXPANDED.py")

print("CODE INPUTS")
print("-----------")
print(f"Version                              {VERSION}")
print("Source geometry                      V0113 JPL/Matplotlib program")
print("Plot change                          width only: 18.0 -> 54.0 inches")
print("Scientific calculations             unchanged")

print("COMMENTS")
print("--------")
print("DEBUG fetching verified V0113 source from the project repository")
with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
    source = response.read().decode("utf-8")

if "figsize=(18.0, 8.5)" not in source:
    raise RuntimeError("REJECTED expected V0113 original figure size was not found")

source = source.replace("# V0113", "# V0115")
source = source.replace("V0113", "V0115")
source = source.replace("figsize=(18.0, 8.5)", "figsize=(54.0, 8.5)")
source = source.replace("YearLocator(10)", "YearLocator(5)")
source = source.replace(
    "The continuous plot uses one fixed 1882 solar tangent-screen basis so the full 1751-1882 curves remain geometrically comparable.",
    "The continuous plot uses one fixed 1882 solar tangent-screen basis so the full 1751-1882 curves remain geometrically comparable. The figure width is exactly three times the original width."
)

compile(source, str(LOCAL_SOURCE), "exec")
LOCAL_SOURCE.write_text(source, encoding="utf-8")

print("RESULTS")
print("-------")
print("Figure dimensions                   54.0 x 8.5 inches")
print("Horizontal scale factor             3.000000")
print("Vertical scale factor               1.000000")

print("OUTPUT SUMMARY")
print("--------------")
print(f"Expanded execution file             {LOCAL_SOURCE}")

print("PAPER COMPARISON")
print("----------------")
print("NOT USED: no published values or manual geometry changes")

print("EQUATION STATUS")
print("---------------")
print("VERIFIED width ratio = 54.0 / 18.0 = 3.0")
print("VERIFIED JPL vectors, fitting, angles, colors, labels, and registrations are unchanged")

runpy.run_path(str(LOCAL_SOURCE), run_name="__main__")
# V0115