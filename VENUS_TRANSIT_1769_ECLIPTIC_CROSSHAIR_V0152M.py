# V0152M
# Audit reference: preserve V0152L exactly and reverse only the blue and green temporal-direction arrows to the verified right-to-left transit direction.
from __future__ import annotations
import urllib.request

SOURCE_URL=(
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152L.py"
)
with urllib.request.urlopen(SOURCE_URL,timeout=90) as response:
    source=response.read().decode("utf-8")

required=[
    "# V0152L",
    'VERSION="V0152L"',
    'direction /= float(np.linalg.norm(direction))',
    'arrow_center = np.array([ca_x, ca_y], dtype=float) - 430.0 * direction',
    'arrow_tail = arrow_center - 95.0 * direction',
    'arrow_head = arrow_center + 95.0 * direction',
    'Earth Track From Ecliptic',
    'Venus Transit Track From Ecliptic',
]
for marker in required:
    if marker not in source:
        raise RuntimeError(f"REJECTED missing V0152L marker: {marker}")

source=source.replace("# V0152L","# V0152M")
source=source.replace('VERSION="V0152L"','VERSION="V0152M"')
source=source.replace('VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152L_OUTPUT','VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152M_OUTPUT')
source=source.replace('VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152L.png','VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152M.png')
source=source.replace(
    'direction /= float(np.linalg.norm(direction))',
    'direction /= float(np.linalg.norm(direction))\n            direction *= -1.0',
    1,
)
source=source.replace(
    'Approved solar plot, three reference lines, callouts, legend, and angle box are preserved; only explicit temporal-direction arrows are added to the blue Earth and green Venus lines.',
    'Approved solar plot, three reference lines, callouts, legend, and angle box are preserved; only the blue Earth and green Venus arrow directions are reversed to the verified right-to-left transit direction.'
)
source=source.replace(
    'PASS: the approved plot is unchanged; explicit arrows on the blue Earth and green Venus lines point in the increasing-JPL-time orbit direction.',
    'PASS: the approved plot is unchanged; the blue Earth and green Venus arrows are reversed and now point right-to-left across the solar disk.'
)

if source.splitlines()[0]!="# V0152M" or source.splitlines()[-1]!="# V0152M":
    raise RuntimeError("REJECTED version boundary")
for marker in [
    'VERSION="V0152M"',
    'direction *= -1.0',
    'Earth Track From Ecliptic',
    'Venus Transit Track From Ecliptic',
    'Angular Venus–Sun separation ρ(t), ±2 h',
]:
    if marker not in source:
        raise RuntimeError(f"REJECTED final marker missing: {marker}")

compile(source,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152M_COMPILED.py","exec")
exec(compile(source,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152M.py","exec"))
# V0152M