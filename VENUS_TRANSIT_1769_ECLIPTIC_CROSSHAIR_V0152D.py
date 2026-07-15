# V0152D
# Audit reference: restore all V0152 reference lines; use V0102C d-rho/dt-zero CA; replace rho circle only with red +/-30-minute curved segment.
from __future__ import annotations
import importlib.util, math, subprocess, sys, warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

def ensure(m,p):
    if importlib.util.find_spec(m) is None:
        subprocess.run([sys.executable,"-m","pip","install","-q",p],check=True)
for m,p in [("astroquery","astroquery"),("astropy","astropy"),("scipy","scipy"),("numpy","numpy"),("matplotlib","matplotlib"),("IPython","ipython")]: ensure(m,p)

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Image,display
from matplotlib.offsetbox import AnchoredOffsetbox,TextArea,VPacker
from matplotlib.patches import Circle
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq,minimize_scalar
warnings.filterwarnings("ignore",message=".*dubious year.*")
warnings.filterwarnings("ignore",message=".*id_type.*deprecated.*")

VERSION="V0152D"; LOCAL_TZ=ZoneInfo("America/Bogota")
YEAR=1769; CENTER_UTC="1769-06-03 22:00"; LOCATION="500@399"; STEP="1m"
REFPLANE="ecliptic"; ABERRATIONS="geometric"; AU_KM=149597870.7
AS_PER_RAD=206264.80624709636; R_SUN_KM=695700.0; R_VENUS_KM=6051.8
SEARCH_HALF_HOURS=3.0; LOCAL_HALF_MINUTES=30.0
OUT=Path("/content/VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152D_OUTPUT")
PNG=OUT/"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152D.png"

def unit(v):
    n=float(np.linalg.norm(v))
    if not np.isfinite(n) or n<=0: raise RuntimeError("REJECTED invalid vector")
    return np.asarray(v,float)/n

def query(body,start,stop):
    t=Horizons(id=body,id_type="majorbody",location=LOCATION,epochs={"start":start,"stop":stop,"step":STEP}).vectors(refplane=REFPLANE,aberrations=ABERRATIONS,cache=False)
    jd=np.asarray(t["datetime_jd"],float)
    xyz=np.column_stack([np.asarray(t[a],float) for a in "xyz"])*AU_KM
    if len(jd)<60 or not np.all(np.diff(jd)>0): raise RuntimeError(f"REJECTED JPL grid {body}")
    return jd,xyz

def spl(jd,xyz): return [CubicSpline(jd,xyz[:,i],bc_type="natural") for i in range(3)]
def ev(c,j): return np.array([float(f(j)) for f in c])
def rho(sc,vc,j):
    su,vu=unit(ev(sc,j)),unit(ev(vc,j))
    return math.atan2(float(np.linalg.norm(np.cross(su,vu))),float(np.dot(su,vu)))
def rhodot(sc,vc,j):
    h=.5/1440.0
    return rho(sc,vc,j+h)-rho(sc,vc,j-h)
def solve_ca(jd,sc,vc):
    ys=np.array([rho(sc,vc,float(t)) for t in jd]); i=int(np.argmin(ys))
    r=minimize_scalar(lambda t:rho(sc,vc,float(t)),bounds=(float(jd[max(0,i-3)]),float(jd[min(len(jd)-1,i+3)])),method="bounded",options={"xatol":1e-13,"maxiter":600})
    if not r.success: raise RuntimeError("REJECTED rho seed")
    lo=float(r.x)-5/1440.0; hi=float(r.x)+5/1440.0
    if rhodot(sc,vc,lo)*rhodot(sc,vc,hi)>0: raise RuntimeError("REJECTED d rho/dt root")
    return float(brentq(lambda t:rhodot(sc,vc,float(t)),lo,hi,xtol=1e-14,rtol=1e-14,maxiter=200))

def projected_basis(sun):
    los=unit(sun); x=np.array([1.,0.,0.]); px=x-np.dot(x,los)*los
    if np.linalg.norm(px)<1e-12:
        x=np.array([0.,1.,0.]); px=x-np.dot(x,los)*los
    px=unit(px); py=unit(np.cross(los,px)); return px,py,los

def physical_basis(sun):
    los=unit(sun); east=np.cross(np.array([0.,0.,1.]),los)
    if np.linalg.norm(east)<1e-12: east=np.cross(np.array([0.,1.,0.]),los)
    east=unit(east); north=unit(np.cross(los,east)); return east,north,los

def project(v,b):
    x,y,los=b; d=unit(v); den=float(np.dot(d,los))
    if den<=0: raise RuntimeError("REJECTED tangent denominator")
    return float(np.dot(d,x)/den*AS_PER_RAD),float(np.dot(d,y)/den*AS_PER_RAD)

def fit_track(minutes,x,y):
    cx=np.polyfit(minutes,x,2); cy=np.polyfit(minutes,y,2)
    vx,vy=float(cx[1]),float(cy[1]); ax,ay=2*float(cx[0]),2*float(cy[0])
    angle=((math.degrees(math.atan2(vy,vx))+90)%180)-90
    slope=math.inf if abs(vx)<1e-15 else vy/vx
    rms=float(np.sqrt(np.mean((x-np.polyval(cx,minutes))**2+(y-np.polyval(cy,minutes))**2)))
    curv=abs(vx*ay-vy*ax)/(vx*vx+vy*vy)**1.5
    return abs(angle),slope,rms,curv,cx,cy

def segment(cx,cy,angle,half):
    a=math.radians(angle); dx=half*math.cos(a); dy=half*math.sin(a)
    return [cx-dx,cx+dx],[cy-dy,cy+dy]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    center=Time(CENTER_UTC,scale="utc")
    start=Time(center.jd-SEARCH_HALF_HOURS/24,format="jd",scale="utc").strftime("%Y-%m-%d %H:%M")
    stop=Time(center.jd+SEARCH_HALF_HOURS/24,format="jd",scale="utc").strftime("%Y-%m-%d %H:%M")
    print("CODE INPUTS"); print(f"Version                              {VERSION}"); print(f"Transit                              {YEAR}"); print("JPL source                           Horizons geocentric geometric vectors"); print(f"Observer                             {LOCATION}")
    print("COMMENTS"); print("All original reference lines are restored. Only the rho circle is removed and replaced by the red local +/-30-minute curved projected track.")
    sj,sx=query("10",start,stop); vj,vx=query("299",start,stop)
    if len(sj)!=len(vj) or not np.allclose(sj,vj,atol=1e-11,rtol=0): raise RuntimeError("REJECTED mismatched grids")
    sc,vc=spl(sj,sx),spl(vj,vx); ca=solve_ca(sj,sc,vc); sun_ca,ven_ca=ev(sc,ca),ev(vc,ca)
    pb,fb=projected_basis(sun_ca),physical_basis(sun_ca)
    mins=np.linspace(-30,30,121); jds=ca+mins/1440.0
    rx=[]; ry=[]; ex=[]; ey=[]; prx=[]; pry=[]
    for j in jds:
        s,v=ev(sc,float(j)),ev(vc,float(j)); sxp,syp=project(s,pb); vxp,vyp=project(v,pb); rx.append(vxp-sxp); ry.append(vyp-syp)
        sxf,syf=project(s,fb); vxf,vyf=project(v,fb); ex.append(sxf); ey.append(syf); prx.append(vxf-sxf); pry.append(vyf-syf)
    rx,ry,ex,ey,prx,pry=map(np.asarray,(rx,ry,ex,ey,prx,pry))
    sx0,sy0=project(sun_ca,pb); vx0,vy0=project(ven_ca,pb); cax,cay=vx0-sx0,vy0-sy0
    rho0=float(math.hypot(cax,cay)); rs=float(math.asin(R_SUN_KM/np.linalg.norm(sun_ca))*AS_PER_RAD); rv=float(math.asin(R_VENUS_KM/np.linalg.norm(ven_ca))*AS_PER_RAD)
    ea,es,er,ec,ecx,ecy=fit_track(mins,ex,ey); pa,ps,pr,pc,pcx,pcy=fit_track(mins,rx,ry); va,vs,vr,vcv,vcx,vcy=fit_track(mins,prx,pry)
    fig,ax=plt.subplots(figsize=(10.5,10.5),dpi=120); fig.patch.set_facecolor("black"); ax.set_facecolor("black")
    ax.add_patch(Circle((0,0),rs,facecolor="#C98A18",edgecolor="#E64A19",linewidth=1.15,alpha=.92,zorder=1))
    ce=1.02*rs; ax.plot([-ce,ce],[0,0],color="black",linewidth=.72,zorder=2); ax.plot([0,0],[-ce,ce],color="black",linewidth=.72,zorder=2)
    half=3.0*rs
    for label,angle,color in [("Earth Track From Ecliptic",((math.degrees(math.atan2(float(ecy[1]),float(ecx[1])))+90)%180)-90,"#3EA6FF"),("Venus Transit Track From Ecliptic",((math.degrees(math.atan2(float(vcy[1]),float(vcx[1])))+90)%180)-90,"#38D66B")]:
        xl,yl=segment(cax,cay,angle,half); ax.plot(xl,yl,color=color,linewidth=.95,label=label,zorder=5)
        ri=int(np.argmax(xl)); li=int(np.argmin(xl)); ax.annotate("",xy=(.75*xl[li]+.25*xl[ri],.75*yl[li]+.25*yl[ri]),xytext=(.25*xl[li]+.75*xl[ri],.25*yl[li]+.75*yl[ri]),arrowprops={"arrowstyle":"-|>","color":color,"linewidth":1.15,"mutation_scale":13},zorder=6)
    qx=np.polyval(pcx,mins); qy=np.polyval(pcy,mins)
    ax.plot(qx,qy,color="#FF3B30",linewidth=1.55,zorder=7,label="Projected Venus Transit Track +/-30 min")
    ax.scatter(qx[::8],qy[::8],s=7,color="#FF3B30",zorder=8)
    ax.scatter([cax],[cay],s=36,facecolor="#FF3B30",edgecolor="white",linewidth=.65,zorder=10,label="d rho/dt = 0 closest approach")
    ax.add_patch(Circle((cax,cay),rv,facecolor="none",edgecolor="white",linewidth=.65,zorder=9))
    box=VPacker(children=[TextArea(f"Closest Approach (UTC): {Time(ca,format='jd',scale='tdb').utc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}",textprops={"color":"#F5F5F5","fontsize":9.6}),TextArea(f"rho at CA: {rho0:.9f} arcsec",textprops={"color":"#FF3B30","fontsize":9.6}),TextArea("Local projected curve: +/-30.000 min",textprops={"color":"#FF3B30","fontsize":9.6}),TextArea(f"Earth Track From Ecliptic: {ea:.6f} deg",textprops={"color":"#3EA6FF","fontsize":9.6}),TextArea(f"Projected Venus Transit Track: {pa:.6f} deg",textprops={"color":"#FF3B30","fontsize":9.6}),TextArea(f"Venus Transit Track From Ecliptic: {va:.6f} deg",textprops={"color":"#38D66B","fontsize":9.6}),TextArea("Ecliptic Reference: 0.000 deg",textprops={"color":"#C0C0C0","fontsize":9.6})],align="left",pad=0,sep=2)
    ab=AnchoredOffsetbox(loc="upper right",child=box,pad=.45,frameon=True,borderpad=.45); ab.patch.set_facecolor("#050505"); ab.patch.set_edgecolor("#858585"); ab.patch.set_alpha(.94); ax.add_artist(ab)
    ext=1.10*rs; ax.set_xlim(-ext,ext); ax.set_ylim(-ext,ext); ax.set_aspect("equal",adjustable="box")
    ax.set_title("1769 Venus Transit — Ecliptic Reference And Transit Tracks",color="#F4F4F4",fontsize=14.5,weight="bold",pad=10); ax.set_xlabel("Registered tangent-plane X (arcsec)",color="#E4E4E4"); ax.set_ylabel("Registered tangent-plane Y (arcsec)",color="#E4E4E4")
    ax.tick_params(colors="#D8D8D8",labelsize=9,width=.5); ax.grid(True,color="#686868",alpha=.25,linewidth=.42)
    for s in ax.spines.values(): s.set_color("#999999"); s.set_linewidth(.55)
    leg=ax.legend(loc="lower left",frameon=False,fontsize=8.5)
    for t in leg.get_texts(): t.set_color("#E6E6E6")
    fig.tight_layout(); fig.savefig(PNG,dpi=300,facecolor="black",bbox_inches="tight"); display(Image(filename=str(PNG)))
    print("RESULTS"); print(f"Closest approach UTC                 {Time(ca,format='jd',scale='tdb').utc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}"); print(f"JD(TDB)                              {ca:.12f}"); print(f"rho at CA                            {rho0:.12f} arcsec"); print(f"Raw d rho/dt at CA                   {rhodot(sc,vc,ca)*AS_PER_RAD:.15e} arcsec/min"); print(f"Earth orbit angle                    {ea:.6f} deg"); print(f"Projected relative track angle       {pa:.6f} deg"); print(f"Venus Transit Track From Ecliptic    {va:.6f} deg"); print(f"Projected RMS                        {pr:.12f} arcsec"); print(f"Projected curvature                  {pc:.12e} arcsec^-1"); print(f"Local plotted samples                {len(mins)}")
    print("OUTPUT SUMMARY"); print(f"PNG                                  {PNG}"); print("PAPER COMPARISON"); print("NOT USED: JPL-only internal geometry audit."); print("EQUATION STATUS"); print("PASS: V0102C d rho/dt zero defines CA; all original reference lines are retained; only the local red +/-30-minute curved projected track replaces the rho circle."); print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z")); print(VERSION)
if __name__=="__main__": main()
# V0152D