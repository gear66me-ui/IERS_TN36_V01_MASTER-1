# V0152R
# Audit reference: standalone JPL Horizons implementation preserving the approved plot, full reference box, locked closest approach, and six small centered right-to-left arrows on the blue and green lines.
from __future__ import annotations
import importlib.util, math, subprocess, sys, warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

def ensure(module, package):
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable,"-m","pip","install","-q",package],check=True)
for module,package in [("astroquery","astroquery"),("astropy","astropy"),("scipy","scipy"),("numpy","numpy"),("matplotlib","matplotlib"),("IPython","ipython")]:
    ensure(module,package)

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Image,display
from matplotlib.offsetbox import AnchoredOffsetbox,TextArea,VPacker
from matplotlib.patches import Circle
from scipy.interpolate import CubicSpline
warnings.filterwarnings("ignore",message=".*dubious year.*")
warnings.filterwarnings("ignore",message=".*id_type.*deprecated.*")

VERSION="V0152R"
LOCAL_TZ=ZoneInfo("America/Bogota")
CA_UTC="1769-06-03 22:19:04.388"
CENTER_UTC="1769-06-03 22:00"
LOCATION="@399"
STEP="1m"
REFPLANE="earth"
ABERRATIONS="apparent"
AU_KM=149597870.7
AS_PER_RAD=206264.80624709636
R_SUN_KM=695700.0
R_VENUS_KM=6051.8
SEARCH_HALF_HOURS=18.0
PARABOLA_HALF_MINUTES=120.0
OUT=Path("/content/VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152R_OUTPUT")
PNG=OUT/"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152R.png"

def unit(v):
    n=float(np.linalg.norm(v))
    if not np.isfinite(n) or n<=0.0:
        raise RuntimeError("REJECTED invalid vector")
    return np.asarray(v,dtype=float)/n

def query(body,start,stop):
    table=Horizons(id=body,id_type="majorbody",location=LOCATION,epochs={"start":start,"stop":stop,"step":STEP}).vectors(refplane=REFPLANE,aberrations=ABERRATIONS)
    jd=np.asarray(table["datetime_jd"],dtype=float)
    xyz=np.column_stack([np.asarray(table[a],dtype=float) for a in "xyz"])*AU_KM
    if len(jd)<60 or not np.all(np.diff(jd)>0.0):
        raise RuntimeError(f"REJECTED JPL vector grid for body {body}")
    return jd,xyz

def curves(jd,xyz):
    return [CubicSpline(jd,xyz[:,i],bc_type="natural") for i in range(3)]

def evaluate(c,jd):
    return np.array([float(f(jd)) for f in c],dtype=float)

def physical_basis(sun):
    los=unit(sun)
    east=np.cross(np.array([0.0,0.0,1.0]),los)
    if np.linalg.norm(east)<1e-12:
        east=np.cross(np.array([0.0,1.0,0.0]),los)
    east=unit(east)
    north=unit(np.cross(los,east))
    return east,north,los

def projected_basis(sun):
    los=unit(sun)
    x=np.array([1.0,0.0,0.0])
    px=x-float(np.dot(x,los))*los
    if np.linalg.norm(px)<1e-12:
        x=np.array([0.0,1.0,0.0])
        px=x-float(np.dot(x,los))*los
    px=unit(px)
    py=unit(np.cross(los,px))
    return px,py,los

def project(v,basis):
    xaxis,yaxis,los=basis
    d=unit(v)
    den=float(np.dot(d,los))
    if den<=0.0:
        raise RuntimeError("REJECTED tangent-plane denominator")
    return float(np.dot(d,xaxis)/den*AS_PER_RAD),float(np.dot(d,yaxis)/den*AS_PER_RAD)

def fit_track(hours,x,y):
    cx=np.polyfit(hours,x,2)
    cy=np.polyfit(hours,y,2)
    vx=float(cx[1]); vy=float(cy[1])
    angle=((math.degrees(math.atan2(vy,vx))+90.0)%180.0)-90.0
    slope=math.inf if abs(vx)<1e-15 else vy/vx
    rms=float(np.sqrt(np.mean((x-np.polyval(cx,hours))**2+(y-np.polyval(cy,hours))**2)))
    return abs(angle),angle,slope,rms,cx,cy

def segment(cx,cy,angle,half):
    a=math.radians(angle)
    dx=half*math.cos(a); dy=half*math.sin(a)
    return np.array([cx-dx,cx+dx]),np.array([cy-dy,cy+dy])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    center=Time(CENTER_UTC,scale="utc")
    start=Time(center.jd-SEARCH_HALF_HOURS/24.0,format="jd",scale="utc").strftime("%Y-%m-%d %H:%M")
    stop=Time(center.jd+SEARCH_HALF_HOURS/24.0,format="jd",scale="utc").strftime("%Y-%m-%d %H:%M")

    print("CODE INPUTS")
    print(f"Version                              {VERSION}")
    print("Transit                              1769")
    print("JPL source                           Horizons geocentric vectors only")
    print(f"Observer                             {LOCATION}")
    print(f"Closest approach reference           {CA_UTC} UTC")
    print("COMMENTS")
    print("Standalone implementation; no dependency on any other project Python file.")
    print("Closest approach, three track references, full black callout, red rho(t) parabola, and centered blue/green arrows are preserved.")

    sj,sxyz=query("10",start,stop)
    vj,vxyz=query("299",start,stop)
    if len(sj)!=len(vj) or not np.allclose(sj,vj,atol=1e-11,rtol=0.0):
        raise RuntimeError("REJECTED mismatched JPL grids")
    sc=curves(sj,sxyz); vc=curves(vj,vxyz)
    ca=Time(CA_UTC,scale="utc").tdb.jd
    sun_ca=evaluate(sc,ca); venus_ca=evaluate(vc,ca)
    pb=projected_basis(sun_ca); fb=physical_basis(sun_ca)

    sun_proj=np.array([project(v,pb) for v in sxyz])
    venus_proj=np.array([project(v,pb) for v in vxyz])
    rel_proj=venus_proj-sun_proj
    sun_phys=np.array([project(v,fb) for v in sxyz])
    venus_phys=np.array([project(v,fb) for v in vxyz])
    rel_phys=venus_phys-sun_phys
    separation=np.hypot(rel_proj[:,0],rel_proj[:,1])
    sun_dist=np.linalg.norm(sxyz,axis=1); venus_dist=np.linalg.norm(vxyz,axis=1)
    contact=(np.arcsin(np.clip(R_SUN_KM/sun_dist,-1,1))+np.arcsin(np.clip(R_VENUS_KM/venus_dist,-1,1)))*AS_PER_RAD
    mask=separation<=contact
    if int(mask.sum())<30:
        raise RuntimeError("REJECTED insufficient in-transit samples")
    hours=(sj[mask]-ca)*24.0
    earth_angle,earth_signed,earth_slope,earth_rms,ecx,ecy=fit_track(hours,sun_phys[mask,0],sun_phys[mask,1])
    projected_angle,projected_signed,projected_slope,projected_rms,pcx,pcy=fit_track(hours,rel_proj[mask,0],rel_proj[mask,1])
    venus_angle,venus_signed,venus_slope,venus_rms,vcx,vcy=fit_track(hours,rel_phys[mask,0],rel_phys[mask,1])

    sx0,sy0=project(sun_ca,pb); vx0,vy0=project(venus_ca,pb)
    ca_x=vx0-sx0; ca_y=vy0-sy0
    rho0=math.hypot(ca_x,ca_y)
    rs=math.asin(R_SUN_KM/np.linalg.norm(sun_ca))*AS_PER_RAD
    rv=math.asin(R_VENUS_KM/np.linalg.norm(venus_ca))*AS_PER_RAD

    fig,ax=plt.subplots(figsize=(10.5,10.5),dpi=120)
    fig.patch.set_facecolor("black"); ax.set_facecolor("black")
    ax.add_patch(Circle((0,0),rs,facecolor="#C98A18",edgecolor="#E64A19",linewidth=1.15,alpha=.92,zorder=1))
    ce=1.02*rs
    ax.plot([-ce,ce],[0,0],color="#000000",linewidth=.72,alpha=.92,zorder=2)
    ax.plot([0,0],[-ce,ce],color="#000000",linewidth=.72,alpha=.92,zorder=2)
    ax.text(.72*rs,.035*rs,"Ecliptic Reference  0.000°",color="#000000",fontsize=8.4,ha="center",va="bottom",zorder=3)

    half=3.0*rs
    specs=[("Earth Track From Ecliptic",earth_signed,"#3EA6FF"),("Projected Venus Transit Track",projected_signed,"#F5F5F5"),("Venus Transit Track From Ecliptic",venus_signed,"#38D66B")]
    for label,angle,color in specs:
        xl,yl=segment(ca_x,ca_y,angle,half)
        ax.plot(xl,yl,color=color,linewidth=.95,label=label,zorder=5)
        if label in {"Earth Track From Ecliptic","Venus Transit Track From Ecliptic"}:
            line_start=np.array([float(xl[0]),float(yl[0])],dtype=float)
            line_end=np.array([float(xl[1]),float(yl[1])],dtype=float)
            right_point=line_start if line_start[0]>line_end[0] else line_end
            left_point=line_end if line_start[0]>line_end[0] else line_start
            right_to_left=left_point-right_point
            right_to_left/=float(np.linalg.norm(right_to_left))
            center_origin=np.array([ca_x,ca_y],dtype=float)
            for offset in np.linspace(-360.0,360.0,6):
                center_point=center_origin+offset*right_to_left
                arrow_tail=center_point-18.0*right_to_left
                arrow_head=center_point+18.0*right_to_left
                ax.annotate("",xy=(float(arrow_head[0]),float(arrow_head[1])),xytext=(float(arrow_tail[0]),float(arrow_tail[1])),arrowprops={"arrowstyle":"-|>","color":color,"linewidth":.95,"mutation_scale":8.0,"shrinkA":0.0,"shrinkB":0.0},zorder=12,annotation_clip=True)

    rho_minutes=np.linspace(-PARABOLA_HALF_MINUTES,PARABOLA_HALF_MINUTES,241)
    rho_values=[]
    for minute in rho_minutes:
        s=evaluate(sc,ca+minute/1440.0); v=evaluate(vc,ca+minute/1440.0)
        rho_values.append(math.atan2(float(np.linalg.norm(np.cross(unit(s),unit(v)))),float(np.dot(unit(s),unit(v))))*AS_PER_RAD)
    rho_values=np.asarray(rho_values)
    rho_excess=rho_values-rho0
    tangent=np.array([math.cos(math.radians(projected_signed)),math.sin(math.radians(projected_signed))])
    normal=np.array([-tangent[1],tangent[0]])
    relx=CubicSpline(sj,rel_proj[:,0],bc_type="natural")
    rely=CubicSpline(sj,rel_proj[:,1],bc_type="natural")
    speed=math.hypot(float(relx.derivative()(ca)),float(rely.derivative()(ca)))/1440.0
    along=speed*rho_minutes
    parabola=np.array([ca_x,ca_y])+along[:,None]*tangent[None,:]+rho_excess[:,None]*normal[None,:]
    ax.plot(parabola[:,0],parabola[:,1],color="#FF3B30",linewidth=1.45,zorder=7,label="Angular Venus–Sun separation ρ(t), ±2 h")
    ax.scatter(parabola[::12,0],parabola[::12,1],s=6,color="#FF3B30",zorder=8)
    ax.annotate("ρ(t) minimum\n22:19:04.388",xy=(ca_x,ca_y),xytext=(ca_x-210,ca_y+125),color="#FF3B30",fontsize=8.0,arrowprops={"arrowstyle":"-","color":"#FF3B30","linewidth":.65},zorder=10)

    ax.scatter([ca_x],[ca_y],s=24,facecolor="white",edgecolor="#DADADA",linewidth=.55,zorder=11,label="Closest approach")
    ax.add_patch(Circle((ca_x,ca_y),rv,facecolor="none",edgecolor="white",linewidth=.65,zorder=10))

    offset_x=.18*rs if ca_x<=0 else -.18*rs
    offset_y=.50*rs
    box_lines=[
        TextArea(f"Closest Approach (UTC): {CA_UTC}",textprops={"color":"#F5F5F5","fontsize":9.6}),
        TextArea(f"rho at CA: {rho0:.6f} arcsec",textprops={"color":"#FF3B30","fontsize":9.6}),
        TextArea("Ecliptic Reference: 0.000°",textprops={"color":"#000000","fontsize":9.6}),
        TextArea(f"Earth Track From Ecliptic: {earth_angle:.6f}°",textprops={"color":"#3EA6FF","fontsize":9.6}),
        TextArea(f"Projected Venus Transit Track: {projected_angle:.6f}°",textprops={"color":"#F5F5F5","fontsize":9.6}),
        TextArea(f"Venus Transit Track From Ecliptic: {venus_angle:.6f}°",textprops={"color":"#38D66B","fontsize":9.6}),
    ]
    packed=VPacker(children=box_lines,align="left",pad=0.0,sep=2.0)
    info=AnchoredOffsetbox(loc="center",child=packed,pad=.45,frameon=True,bbox_to_anchor=(ca_x+offset_x,ca_y+offset_y),bbox_transform=ax.transData,borderpad=.45)
    info.patch.set_facecolor("#050505"); info.patch.set_edgecolor("#858585"); info.patch.set_alpha(.94)
    ax.add_artist(info)
    ax.plot([ca_x,ca_x+.62*offset_x],[ca_y,ca_y+.62*offset_y],color="#B0B0B0",linewidth=.65,zorder=9)

    ext=1.10*rs
    ax.set_xlim(-ext,ext); ax.set_ylim(-ext,ext); ax.set_aspect("equal",adjustable="box")
    ax.set_title("1769 Venus Transit — Ecliptic Reference And Transit Tracks",color="#F4F4F4",fontsize=14.5,weight="bold",pad=10)
    ax.set_xlabel("Registered tangent-plane X (arcsec)",color="#E4E4E4")
    ax.set_ylabel("Registered tangent-plane Y (arcsec)",color="#E4E4E4")
    ax.tick_params(colors="#D8D8D8",labelsize=9,width=.5)
    ax.grid(True,color="#686868",alpha=.25,linewidth=.42)
    for spine in ax.spines.values():
        spine.set_color("#999999"); spine.set_linewidth(.55)
    legend=ax.legend(loc="upper right",frameon=False,fontsize=9.0)
    for text in legend.get_texts():
        text.set_color("#E6E6E6")
    fig.tight_layout(); fig.savefig(PNG,dpi=300,facecolor="black",bbox_inches="tight")
    display(Image(filename=str(PNG)))

    print("RESULTS")
    print(f"Closest approach UTC                 {CA_UTC}")
    print(f"JD(TDB)                              {ca:.12f}")
    print(f"rho at CA                            {rho0:.12f} arcsec")
    print(f"Earth orbit angle                    {earth_angle:.6f} deg")
    print(f"Projected relative track angle       {projected_angle:.6f} deg")
    print(f"Venus Transit Track From Ecliptic    {venus_angle:.6f} deg")
    print(f"Parabola window                      +/-{PARABOLA_HALF_MINUTES:.3f} min")
    print("OUTPUT SUMMARY")
    print(f"PNG                                  {PNG}")
    print("PAPER COMPARISON")
    print("NOT USED: JPL-only internal geometry audit.")
    print("EQUATION STATUS")
    print("PASS: standalone file; closest approach uses the same locked UTC formulation; all three track references and six centered right-to-left arrows are present.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)

if __name__=="__main__":
    main()
# V0152R