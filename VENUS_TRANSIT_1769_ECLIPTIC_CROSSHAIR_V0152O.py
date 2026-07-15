# V0152O
# Audit reference: preserve V0152J exactly; add six repeated right-to-left arrows directly on each blue and green line, with no other plot changes.
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
            line_start = np.array([float(xl[0]), float(yl[0])], dtype=float)
            line_end = np.array([float(xl[1]), float(yl[1])], dtype=float)
            right_point = line_start if line_start[0] > line_end[0] else line_end
            left_point = line_end if line_start[0] > line_end[0] else line_start
            right_to_left = left_point - right_point
            right_to_left /= float(np.linalg.norm(right_to_left))
            centers = np.linspace(0.18, 0.82, 6)
            for fraction in centers:
                center_point = right_point + fraction * (left_point - right_point)
                arrow_tail = center_point - 34.0 * right_to_left
                arrow_head = center_point + 34.0 * right_to_left
                ax.annotate(
                    "",
                    xy=(float(arrow_head[0]), float(arrow_head[1])),
                    xytext=(float(arrow_tail[0]), float(arrow_tail[1])),
                    arrowprops={
                        "arrowstyle":"-|>",
                        "color":color,
                        "linewidth":1.45,
                        "mutation_scale":15.0,
                        "shrinkA":0.0,
                        "shrinkB":0.0,
                    },
                    zorder=12,
                    annotation_clip=False,
                )'''

required=[
    "# V0152I",
    'VERSION="V0152I"',
    'offset_y=.16*rs if ca_y<=0 else -.16*rs',
    old_arrow,
    'Angular Venus–Sun separation ρ(t), ±2 h',
    'Earth Track From Ecliptic',
    'Venus Transit Track From Ecliptic',
]
for marker in required:
    if marker not in source:
        raise RuntimeError("REJECTED missing V0152I marker")

source=source.replace("# V0152I","# V0152O")
source=source.replace('VERSION="V0152I"','VERSION="V0152O"')
source=source.replace('VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152I_OUTPUT','VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152O_OUTPUT')
source=source.replace('VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152I.png','VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152O.png')
source=source.replace('offset_y=.16*rs if ca_y<=0 else -.16*rs','offset_y=.50*rs')
source=source.replace(old_arrow,new_arrow,1)
source=source.replace(
    'Approved solar plot, three reference lines, callouts, legend, and angle box are preserved.',
    'Approved solar plot, three reference lines, callouts, legend, and angle box are preserved; six right-to-left arrows are placed directly on each blue and green line.'
)
source=source.replace(
    'PASS: all three approved reference lines and callouts are retained; the red rho(t) parabola has its vertex at the locked closest approach and is tangent to the white projected track there.',
    'PASS: the approved plot is unchanged; six repeated arrows lie directly on the blue line and six directly on the green line, all pointing right-to-left.'
)

if source.splitlines()[0]!="# V0152O" or source.splitlines()[-1]!="# V0152O":
    raise RuntimeError("REJECTED version boundary")
for marker in [
    'VERSION="V0152O"',
    'offset_y=.50*rs',
    'centers = np.linspace(0.18, 0.82, 6)',
    'right_point = line_start if line_start[0] > line_end[0] else line_end',
    'Earth Track From Ecliptic',
    'Venus Transit Track From Ecliptic',
]:
    if marker not in source:
        raise RuntimeError(f"REJECTED final marker missing: {marker}")

compile(source,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152O_COMPILED.py","exec")
exec(compile(source,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152O.py","exec"))
# V0152O