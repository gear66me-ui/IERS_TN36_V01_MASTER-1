# V0152Q
# Audit reference: preserve V0152P plot and arrows; explicitly retain the prior reference-angle labels and the same locked closest-approach formulation and UTC reporting.
from __future__ import annotations
import urllib.request

SOURCE_URL=(
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152P.py"
)
with urllib.request.urlopen(SOURCE_URL,timeout=90) as response:
    source=response.read().decode("utf-8")

required=[
    "# V0152P",
    'VERSION="V0152P"',
    'CA_UTC="1769-06-03 22:19:04.388"',
    'Earth Track From Ecliptic',
    'Projected Venus Transit Track',
    'Venus Transit Track From Ecliptic',
    'Closest Approach (UTC): {CA_UTC}',
    'ca=Time(CA_UTC,scale="utc").tdb.jd',
    'rho0=math.hypot(ca_x,ca_y)',
]
for marker in required:
    if marker not in source:
        raise RuntimeError(f"REJECTED missing V0152P marker: {marker}")

source=source.replace("# V0152P","# V0152Q")
source=source.replace('VERSION="V0152P"','VERSION="V0152Q"')
source=source.replace('VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152P_OUTPUT','VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152Q_OUTPUT')
source=source.replace('VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152P.png','VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152Q.png')
source=source.replace(
    'Approved solar plot, three reference lines, callouts, legend, and angle box are preserved; six small right-to-left arrows remain centered directly on each blue and green line.',
    'Approved solar plot and six centered right-to-left arrows are preserved. The prior references are retained explicitly: Earth Track From Ecliptic, Projected Venus Transit Track, Venus Transit Track From Ecliptic, plus the same locked closest-approach UTC formulation.'
)
source=source.replace(
    'PASS: the approved plot and normal-size black box are preserved; six small arrows lie on the central portion of each blue and green line, with no arrows outside the axes.',
    'PASS: the plot preserves the prior three reference-angle labels and the identical closest-approach definition ca=Time(CA_UTC,scale="utc").tdb.jd with rho from the Sun-Venus tangent-plane separation at 1769-06-03 22:19:04.388 UTC.'
)

if source.splitlines()[0]!="# V0152Q" or source.splitlines()[-1]!="# V0152Q":
    raise RuntimeError("REJECTED version boundary")
for marker in [
    'VERSION="V0152Q"',
    'CA_UTC="1769-06-03 22:19:04.388"',
    'Earth Track From Ecliptic',
    'Projected Venus Transit Track',
    'Venus Transit Track From Ecliptic',
    'Closest Approach (UTC): {CA_UTC}',
    'ca=Time(CA_UTC,scale="utc").tdb.jd',
]:
    if marker not in source:
        raise RuntimeError(f"REJECTED final marker missing: {marker}")

compile(source,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152Q_COMPILED.py","exec")
exec(compile(source,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152Q.py","exec"))
# V0152Q