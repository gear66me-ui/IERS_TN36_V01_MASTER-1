# V0152J
# Audit reference: preserve V0152I exactly and move only the black information box upward, clear of the red rho(t) parabola.
from __future__ import annotations
import urllib.request

SOURCE_URL=(
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152I.py"
)
with urllib.request.urlopen(SOURCE_URL,timeout=90) as response:
    source=response.read().decode("utf-8")

required=[
    "# V0152I",
    'VERSION="V0152I"',
    'offset_x=.18*rs if ca_x<=0 else -.18*rs',
    'offset_y=.16*rs if ca_y<=0 else -.16*rs',
    'bbox_to_anchor=(ca_x+offset_x,ca_y+offset_y)',
    'Angular Venus–Sun separation ρ(t), ±2 h',
    'Projected Venus Transit Track',
]
for marker in required:
    if marker not in source:
        raise RuntimeError(f"REJECTED missing V0152I marker: {marker}")

source=source.replace("# V0152I","# V0152J")
source=source.replace('VERSION="V0152I"','VERSION="V0152J"')
source=source.replace(
    'VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152I_OUTPUT',
    'VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152J_OUTPUT'
)
source=source.replace(
    'VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152I.png',
    'VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152J.png'
)
source=source.replace(
    'offset_y=.16*rs if ca_y<=0 else -.16*rs',
    'offset_y=.50*rs'
)
source=source.replace(
    'Approved solar plot, three reference lines, callouts, legend, and angle box are preserved.',
    'Approved solar plot, three reference lines, callouts, legend, and angle box are preserved; only the black information box is moved upward.'
)
source=source.replace(
    'PASS: all three approved reference lines and callouts are retained; the red rho(t) parabola has its vertex at the locked closest approach and is tangent to the white projected track there.',
    'PASS: all three approved reference lines, callouts, and the red rho(t) parabola are unchanged; only the black information box is moved upward clear of the parabola.'
)

if source.splitlines()[0]!="# V0152J" or source.splitlines()[-1]!="# V0152J":
    raise RuntimeError("REJECTED version boundary")
for marker in [
    'VERSION="V0152J"',
    'offset_y=.50*rs',
    'Angular Venus–Sun separation ρ(t), ±2 h',
    'Projected Venus Transit Track',
    'Earth Track From Ecliptic',
    'Venus Transit Track From Ecliptic',
]:
    if marker not in source:
        raise RuntimeError(f"REJECTED final marker missing: {marker}")

compile(source,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152J_COMPILED.py","exec")
exec(compile(source,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152J.py","exec"))
# V0152J