# V0152K
# Audit reference: preserve V0152J plot exactly; keep the raised black box and add clear direction arrows only to the blue and green reference lines.
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
    'offset_y=.16*rs if ca_y<=0 else -.16*rs',
    'if label in {"Earth Track From Ecliptic","Venus Transit Track From Ecliptic"}:',
    'arrowprops={"arrowstyle":"-|>","color":color,"linewidth":1.15,"mutation_scale":13.0}',
    'Angular Venus–Sun separation ρ(t), ±2 h',
]
for marker in required:
    if marker not in source:
        raise RuntimeError(f"REJECTED missing V0152I marker: {marker}")

source=source.replace("# V0152I","# V0152K")
source=source.replace('VERSION="V0152I"','VERSION="V0152K"')
source=source.replace(
    'VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152I_OUTPUT',
    'VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152K_OUTPUT'
)
source=source.replace(
    'VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152I.png',
    'VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152K.png'
)
source=source.replace(
    'offset_y=.16*rs if ca_y<=0 else -.16*rs',
    'offset_y=.50*rs'
)
source=source.replace(
    'arrowprops={"arrowstyle":"-|>","color":color,"linewidth":1.15,"mutation_scale":13.0}',
    'arrowprops={"arrowstyle":"-|>","color":color,"linewidth":1.45,"mutation_scale":20.0}'
)
source=source.replace(
    'Approved solar plot, three reference lines, callouts, legend, and angle box are preserved.',
    'Approved solar plot, three reference lines, callouts, legend, and angle box are preserved; the black box remains raised and clear direction arrows are shown on the blue and green lines.'
)
source=source.replace(
    'PASS: all three approved reference lines and callouts are retained; the red rho(t) parabola has its vertex at the locked closest approach and is tangent to the white projected track there.',
    'PASS: the approved plot is unchanged except for the raised black box and enlarged direction arrows on the blue Earth line and green Venus line.'
)

if source.splitlines()[0]!="# V0152K" or source.splitlines()[-1]!="# V0152K":
    raise RuntimeError("REJECTED version boundary")
for marker in [
    'VERSION="V0152K"',
    'offset_y=.50*rs',
    'mutation_scale":20.0',
    'Earth Track From Ecliptic',
    'Projected Venus Transit Track',
    'Venus Transit Track From Ecliptic',
    'Angular Venus–Sun separation ρ(t), ±2 h',
]:
    if marker not in source:
        raise RuntimeError(f"REJECTED final marker missing: {marker}")

compile(source,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152K_COMPILED.py","exec")
exec(compile(source,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152K.py","exec"))
# V0152K