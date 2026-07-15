# V0152G
# Audit reference: preserve V0152F solar plot and all three lines; place rho(t) parabola on the main tangent plane with vertex at CA and tangent aligned to the white projected Venus track.
from __future__ import annotations
import urllib.request

SOURCE_URL=("https://raw.githubusercontent.com/gear66me-ui/"
"IERS_TN36_V01_MASTER-1/main/"
"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152F.py")
with urllib.request.urlopen(SOURCE_URL,timeout=90) as response:
    source=response.read().decode("utf-8")

for marker in ["# V0152F",'VERSION="V0152F"','CA_UTC="1769-06-03 22:19:04.388"',
               'Projected Venus Transit Track','rho_minutes=np.linspace(-RHO_HALF_MINUTES,RHO_HALF_MINUTES,241)',
               'inset=ax.inset_axes([0.075,0.635,0.39,0.255])']:
    if marker not in source:
        raise RuntimeError(f"REJECTED missing V0152F marker: {marker}")

source=source.replace("# V0152F","# V0152G")
source=source.replace('VERSION="V0152F"','VERSION="V0152G"')
source=source.replace("VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152F_OUTPUT","VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152G_OUTPUT")
source=source.replace("VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152F.png","VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152G.png")
source=source.replace(
    "Original solar plot and all three reference lines preserved. The rho circle alone is replaced by a red rho(t) parabola inset over +/-2 hours.",
    "Original solar plot and all three reference lines preserved. The rho circle alone is replaced by a red rho(t) parabola on the main tangent plane, with its vertex at closest approach and tangent aligned to the white projected Venus track."
)

start=source.index('    rho_minutes=np.linspace(-RHO_HALF_MINUTES,RHO_HALF_MINUTES,241)')
end=source.index('    box=VPacker(children=[',start)
new_block='''    rho_minutes=np.linspace(-RHO_HALF_MINUTES,RHO_HALF_MINUTES,241)
    rho_values=np.array([rho(sc,vc,ca+float(m)/1440.0)*AS_PER_RAD for m in rho_minutes],dtype=float)
    rho_excess=rho_values-rho0
    white_angle=((math.degrees(math.atan2(float(pcy[1]),float(pcx[1])))+90)%180)-90
    white_rad=math.radians(white_angle)
    tangent_u=np.array([math.cos(white_rad),math.sin(white_rad)],dtype=float)
    normal_u=np.array([-tangent_u[1],tangent_u[0]],dtype=float)
    speed_arcsec_per_min=float(math.hypot(float(pcx[1]),float(pcy[1])))
    along=speed_arcsec_per_min*rho_minutes
    parabola_xy=np.array([cax,cay],dtype=float)+along[:,None]*tangent_u[None,:]+rho_excess[:,None]*normal_u[None,:]
    parabola_x=parabola_xy[:,0]
    parabola_y=parabola_xy[:,1]
    ax.plot(parabola_x,parabola_y,color="#FF3B30",linewidth=1.55,zorder=8,label="rho(t) tangent parabola +/-2 h")
    ax.scatter(parabola_x[::12],parabola_y[::12],s=7,color="#FF3B30",zorder=9)
    ax.scatter([cax],[cay],s=44,facecolor="#FF3B30",edgecolor="white",linewidth=.70,zorder=11)
    ax.annotate("rho(t) minimum\\n22:19:04.388",xy=(cax,cay),xytext=(cax-210,cay+115),color="#FF3B30",fontsize=8.0,
        arrowprops={"arrowstyle":"-","color":"#FF3B30","linewidth":.65},zorder=12)
'''
source=source[:start]+new_block+source[end:]

source=source.replace('TextArea("rho(t) parabola inset: +/-120.000 min",textprops={"color":"#FF3B30","fontsize":9.6})',
                      'TextArea("rho(t) tangent parabola: +/-120.000 min",textprops={"color":"#FF3B30","fontsize":9.6})')
source=source.replace(
    "PASS: the requested 22:19:04.388 UTC reference is used; blue, white, and green lines are retained; the rho circle alone is replaced by the red rho",
    "PASS: the requested 22:19:04.388 UTC reference is used; blue, white, and green lines are retained; the red rho(t) parabola has its vertex at CA and zero normal derivative there, so its tangent equals the white projected Venus track"
)

if source.splitlines()[0]!="# V0152G" or source.splitlines()[-1]!="# V0152G":
    raise RuntimeError("REJECTED version boundary")
for marker in ["rho(t) tangent parabola +/-2 h","tangent_u","normal_u","Projected Venus Transit Track","Earth Track From Ecliptic","Venus Transit Track From Ecliptic"]:
    if marker not in source:
        raise RuntimeError(f"REJECTED final marker missing: {marker}")

compile(source,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152G.py","exec")
exec(compile(source,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152G.py","exec"))
# V0152G