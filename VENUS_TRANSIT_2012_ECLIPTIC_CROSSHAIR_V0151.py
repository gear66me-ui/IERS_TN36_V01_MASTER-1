# V0151
# Audit reference: audited 2012 ecliptic-crosshair plot with closest-approach UTC added to the color-coded information box.
from __future__ import annotations
import urllib.request

SOURCE_URL=("https://raw.githubusercontent.com/gear66me-ui/"
"IERS_TN36_V01_MASTER-1/main/"
"VENUS_TRANSIT_2012_ECLIPTIC_CROSSHAIR_V0150.py")
with urllib.request.urlopen(SOURCE_URL,timeout=90) as response:
    wrapper=response.read().decode("utf-8")

required=[
    "# V0150",
    'TextArea("Ecliptic Reference: 0.000°"',
    'TextArea(f"Earth Track From Ecliptic:',
    'TextArea(f"Projected Venus Transit Track:',
    'TextArea(f"Venus Transit Track From Ecliptic:',
    'exec(compile(source,"VENUS_TRANSIT_2012_ECLIPTIC_CROSSHAIR_V0150.py","exec"))',
]
for marker in required:
    if marker not in wrapper:
        raise RuntimeError(f"REJECTED missing V0150 marker: {marker}")

wrapper=wrapper.replace("# V0150","# V0151")
wrapper=wrapper.replace("V0150","V0151")
old_line='        TextArea("Ecliptic Reference: 0.000°", textprops={"color":"#000000","fontsize":9.6}),' 
new_lines=(
    '        TextArea(f"Closest Approach (UTC): {Time(jd_ca, format=\'jd\', scale=\'tdb\').utc.strftime(\'%Y-%m-%d %H:%M:%S\')}", '
    'textprops={"color":"#F5F5F5","fontsize":9.6}),\n'
    + old_line
)
if old_line not in wrapper:
    raise RuntimeError("REJECTED V0150 angle-box insertion marker missing")
wrapper=wrapper.replace(old_line,new_lines,1)

if 'Closest Approach (UTC)' not in wrapper:
    raise RuntimeError("REJECTED closest-approach UTC line missing")
if wrapper.splitlines()[0]!="# V0151" or wrapper.splitlines()[-1]!="# V0151":
    raise RuntimeError("REJECTED version boundary")
if 'color="#000000"' not in wrapper:
    raise RuntimeError("REJECTED black crosshair missing")
for label in [
    "Ecliptic Reference",
    "Earth Track From Ecliptic",
    "Projected Venus Transit Track",
    "Venus Transit Track From Ecliptic",
]:
    if label not in wrapper:
        raise RuntimeError(f"REJECTED missing title-case label: {label}")

compile(wrapper,"VENUS_TRANSIT_2012_ECLIPTIC_CROSSHAIR_V0151_WRAPPER.py","exec")
exec(compile(wrapper,"VENUS_TRANSIT_2012_ECLIPTIC_CROSSHAIR_V0151.py","exec"))
# V0151