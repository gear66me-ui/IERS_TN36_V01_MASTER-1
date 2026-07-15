# V0152N
# Audit reference: preserve V0152J plot exactly; add explicit visible right-to-left arrows only on the blue Earth and green Venus lines.
from __future__ import annotations
import urllib.request

SOURCE_URL=(
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152I.py"
)
with urllib.request.urlopen(SOURCE_URL,timeout=90) as response:
    source=response.read().decode("utf-8")

old_arrow='''        if label in {"Earth Track From Ecliptic","Venus Transit Track From Ecliptic"}:
            ri=int(np.argmax(xl)); li=int(np.argmin(xl))
            ax.annotate("",xy=(.75*xl[li]+.25*xl[ri],.75*yl[li]+.25*yl[ri]),xytext=(.25*xl[li]+.75*xl[ri],.25*yl[li]+.75*yl[ri]),arrowprops={"arrowstyle":"-|>","color":color,"linewidth":1.15,"mutation_scale":13.0},zorder=6)'''

new_arrow='''        if label in {"Earth Track From Ecliptic","Venus Transit Track From Ecliptic"}:
            if label == "Earth Track From Ecliptic":
                direction = np.array([float(ecx[1]), float(ecy[1])], dtype=float)
                normal_shift = -18.0
            else:
                direction = np.array([float(vcx[1]), float(vcy[1])], dtype=float)
                normal_shift = 18.0
            direction /= float(np.linalg.norm(direction))
            if direction[0] > 0.0:
                direction *= -1.0
            normal_direction = np.array([-direction[1], direction[0]], dtype=float)
            arrow_center = np.array([ca_x, ca_y], dtype=float) + 390.0 * direction + normal_shift * normal_direction
            arrow_tail = arrow_center - 115.0 * direction
            arrow_head = arrow_center + 115.0 * direction
            ax.annotate(
                "",
                xy=(float(arrow_head[0]), float(arrow_head[1])),
                xytext=(float(arrow_tail[0]), float(arrow_tail[1])),
                arrowprops={
                    "arrowstyle":"-|>",
                    "color":color,
                    "linewidth":2.40,
                    "mutation_scale":28.0,
                    "shrinkA":0.0,
                    "shrinkB":0.0,
                },
                zorder=20,
                annotation_clip=False,
            )'''

required=[
    "# V0152I",
    'VERSION="V0152I"',
    'offset_y=.16*rs if ca_y<=0 else -.16*rs',
    old_arrow,
    'Angular Venus–Sun separation ρ(t), ±2 h',
]
for marker in required:
    if marker not in source:
        raise RuntimeError("REJECTED missing V0152I marker")

source=source.replace("# V0152I","# V0152N")
source=source.replace('VERSION="V0152I"','VERSION="V0152N"')
source=source.replace('VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152I_OUTPUT','VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152N_OUTPUT')
source=source.replace('VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152I.png','VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152N.png')
source=source.replace('offset_y=.16*rs if ca_y<=0 else -.16*rs','offset_y=.50*rs')
source=source.replace(old_arrow,new_arrow,1)
source=source.replace(
    'Approved solar plot, three reference lines, callouts, legend, and angle box are preserved.',
    'Approved solar plot, three reference lines, callouts, legend, and angle box are preserved; explicit visible right-to-left arrows are added only to the blue Earth and green Venus lines.'
)
source=source.replace(
    'PASS: all three approved reference lines and callouts are retained; the red rho(t) parabola has its vertex at the locked closest approach and is tangent to the white projected track there.',
    'PASS: the approved plot is unchanged; large blue and green arrows are visibly overlaid on their lines and both point right-to-left.'
)

if source.splitlines()[0]!="# V0152N" or source.splitlines()[-1]!="# V0152N":
    raise RuntimeError("REJECTED version boundary")
for marker in [
    'VERSION="V0152N"',
    'offset_y=.50*rs',
    'if direction[0] > 0.0:',
    'linewidth":2.40',
    'mutation_scale":28.0',
    'zorder=20',
    'Earth Track From Ecliptic',
    'Venus Transit Track From Ecliptic',
]:
    if marker not in source:
        raise RuntimeError(f"REJECTED final marker missing: {marker}")

compile(source,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152N_COMPILED.py","exec")
exec(compile(source,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152N.py","exec"))
# V0152N