# V0152
# Audit reference: audited 1769-only ecliptic-crosshair plot with closest-approach UTC in the color-coded information box.
from __future__ import annotations
import urllib.request

SOURCE_URL=("https://raw.githubusercontent.com/gear66me-ui/"
"IERS_TN36_V01_MASTER-1/main/"
"VENUS_TRANSIT_2012_ECLIPTIC_CROSSHAIR_V0150.py")
with urllib.request.urlopen(SOURCE_URL,timeout=90) as response:
    wrapper=response.read().decode("utf-8")

required=[
    "# V0150",
    'source=source.replace("YEAR = 1761","YEAR = 2012")',
    'source=source.replace(\'CENTER_UTC = "1761-06-06 06:00"\',\'CENTER_UTC = "2012-06-06 01:00"\')',
    'TextArea("Ecliptic Reference: 0.000°"',
    'color="#000000"',
    'exec(compile(source,"VENUS_TRANSIT_2012_ECLIPTIC_CROSSHAIR_V0150.py","exec"))',
]
for marker in required:
    if marker not in wrapper:
        raise RuntimeError(f"REJECTED missing V0150 marker: {marker}")

wrapper=wrapper.replace("# V0150","# V0152")
wrapper=wrapper.replace('VERSION = "V0150"','VERSION = "V0152"')
wrapper=wrapper.replace(
    'source=source.replace("YEAR = 1761","YEAR = 2012")',
    'source=source.replace("YEAR = 1761","YEAR = 1769")',
)
wrapper=wrapper.replace(
    'source=source.replace(\'CENTER_UTC = "1761-06-06 06:00"\',\'CENTER_UTC = "2012-06-06 01:00"\')',
    'source=source.replace(\'CENTER_UTC = "1761-06-06 06:00"\',\'CENTER_UTC = "1769-06-03 22:00"\')',
)
wrapper=wrapper.replace(
    'VENUS_TRANSIT_2012_ECLIPTIC_CROSSHAIR_V0150',
    'VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152',
)
wrapper=wrapper.replace(
    'print("Transit                              2012")',
    'print("Transit                              1769")',
)
wrapper=wrapper.replace(
    '"2012 Venus Transit — Ecliptic Reference And Transit Tracks"',
    '"1769 Venus Transit — Ecliptic Reference And Transit Tracks"',
)

old_line='        TextArea("Ecliptic Reference: 0.000°", textprops={"color":"#000000","fontsize":9.6}),' 
new_lines=(
    '        TextArea(f"Closest Approach (UTC): {Time(jd_ca, format=\'jd\', scale=\'tdb\').utc.strftime(\'%Y-%m-%d %H:%M:%S\')}", '
    'textprops={"color":"#F5F5F5","fontsize":9.6}),\n'
    + old_line
)
if old_line not in wrapper:
    raise RuntimeError("REJECTED angle-box insertion marker missing")
wrapper=wrapper.replace(old_line,new_lines,1)

if wrapper.splitlines()[0]!="# V0152" or wrapper.splitlines()[-1]!="# V0152":
    raise RuntimeError("REJECTED version boundary")
for marker in [
    'YEAR = 1769',
    'CENTER_UTC = "1769-06-03 22:00"',
    'Closest Approach (UTC)',
    'Ecliptic Reference',
    'Earth Track From Ecliptic',
    'Projected Venus Transit Track',
    'Venus Transit Track From Ecliptic',
    'color="#000000"',
]:
    if marker not in wrapper:
        raise RuntimeError(f"REJECTED final marker missing: {marker}")
if 'YEAR = 2012' in wrapper or 'CENTER_UTC = "2012-06-06 01:00"' in wrapper:
    raise RuntimeError("REJECTED 2012 configuration survived")

compile(wrapper,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152_WRAPPER.py","exec")
exec(compile(wrapper,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152.py","exec"))
# V0152