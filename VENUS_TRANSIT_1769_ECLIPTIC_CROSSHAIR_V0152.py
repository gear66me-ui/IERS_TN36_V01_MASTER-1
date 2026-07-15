# V0152
# Audit reference: audited 1769-only ecliptic-crosshair plot with closest-approach UTC in the color-coded information box.
from __future__ import annotations
import urllib.request

SOURCE_URL=("https://raw.githubusercontent.com/gear66me-ui/"
"IERS_TN36_V01_MASTER-1/main/"
"VENUS_TRANSIT_2012_ECLIPTIC_CROSSHAIR_V0151.py")
with urllib.request.urlopen(SOURCE_URL,timeout=90) as response:
    wrapper=response.read().decode("utf-8")

required=[
    "# V0151",
    'YEAR = 2012',
    'CENTER_UTC = "2012-06-06 01:00"',
    'Closest Approach (UTC)',
    'Ecliptic Reference',
    'Earth Track From Ecliptic',
    'Projected Venus Transit Track',
    'Venus Transit Track From Ecliptic',
    'color="#000000"',
]
for marker in required:
    if marker not in wrapper:
        raise RuntimeError(f"REJECTED missing V0151 marker: {marker}")

wrapper=wrapper.replace("# V0151","# V0152")
wrapper=wrapper.replace("V0151","V0152")
wrapper=wrapper.replace('YEAR = 2012','YEAR = 1769')
wrapper=wrapper.replace('CENTER_UTC = "2012-06-06 01:00"','CENTER_UTC = "1769-06-03 22:00"')
wrapper=wrapper.replace('print("Transit                              2012")','print("Transit                              1769")')
wrapper=wrapper.replace('"2012 Venus Transit — Ecliptic Reference And Transit Tracks"','"1769 Venus Transit — Ecliptic Reference And Transit Tracks"')
wrapper=wrapper.replace("VENUS_TRANSIT_2012_ECLIPTIC_CROSSHAIR_V0152","VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152")
wrapper=wrapper.replace(
    "VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0150.py",
    "VENUS_TRANSIT_2012_ECLIPTIC_CROSSHAIR_V0150.py",
)

if wrapper.splitlines()[0]!="# V0152" or wrapper.splitlines()[-1]!="# V0152":
    raise RuntimeError("REJECTED version boundary")
if 'YEAR = 2012' in wrapper or 'CENTER_UTC = "2012-06-06 01:00"' in wrapper:
    raise RuntimeError("REJECTED 2012 configuration survived")
for marker in [
    'YEAR = 1769',
    'CENTER_UTC = "1769-06-03 22:00"',
    'Closest Approach (UTC)',
    'color="#000000"',
    'Ecliptic Reference',
    'Earth Track From Ecliptic',
    'Projected Venus Transit Track',
    'Venus Transit Track From Ecliptic',
]:
    if marker not in wrapper:
        raise RuntimeError(f"REJECTED final marker missing: {marker}")

compile(wrapper,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152_WRAPPER.py","exec")
exec(compile(wrapper,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152.py","exec"))
# V0152