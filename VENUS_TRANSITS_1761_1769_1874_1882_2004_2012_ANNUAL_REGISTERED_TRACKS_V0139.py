# V0139
# Audit reference: standalone V0139 preserving the approved annual plots while updating only closest-approach calculation/time and the three V0152P track-angle references.
from __future__ import annotations
import importlib.util, math, subprocess, sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

def need(module: str, package: str) -> None:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)
for module, package in [("astroquery","astroquery"),("astropy","astropy"),("scipy","scipy"),("pandas","pandas"),("matplotlib","matplotlib"),("IPython","ipython")]:
    need(module, package)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

VERSION="V0139"
FILENAME="VENUS_TRANSITS_1761_1769_1874_1882_2004_2012_ANNUAL_REGISTERED_TRACKS_V0139.py"
OUT=Path("/content/VENUS_TRANSITS_1761_1769_1874_1882_2004_2012_ANNUAL_REGISTERED_TRACKS_V0139_OUTPUT")
CSV_NAME="VENUS_TRANSITS_1761_1769_1874_1882_2004_2012_ANNUAL_REGISTERED_TRACKS_V0139.csv"
AU_KM=149597870.700; R_SUN_KM=695700.000; AS_PER_RAD=206264.80624709636
LOCATION="@0"; REFPLANE="earth"; ABERRATIONS="geometric"
FINE_STEP="1m"; YEAR_STEP="6h"; SEARCH_HALF_H=18.0; FIT_HALF_H=10.0; VISUAL_SCALE=2.0
TRANSITS: Dict[int,str]={1761:"1761-06-06 06:00",1769:"1769-06-03 22:00",1874:"1874-12-09 04:00",1882:"1882-12-06 17:00",2004:"2004-06-08 08:00",2012:"2012-06-06 01:00"}
PNG_NAMES={year:f"VENUS_TRANSIT_{year}_ANNUAL_REGISTERED_TRACKS_V0139.png" for year in TRANSITS}

@dataclass(frozen=True)
class Series:
    jd: np.ndarray
    xyz: np.ndarray
@dataclass(frozen=True)
class Fit:
    raw_direction_deg: float
    signed_line_deg: float
    positive_horizontal_deg: float
    slope: float
    rms_km: float
    curvature_per_km: float

def section(name:str)->None:
    print(name); print("-"*len(name))
def unit(v:np.ndarray)->np.ndarray:
    n=float(np.linalg.norm(v))
    if not np.isfinite(n) or n<=0: raise RuntimeError("REJECTED invalid vector")
    return np.asarray(v,float)/n
def query(body:str,start:str,stop:str,step:str)->Series:
    t=Horizons(id=body,id_type="majorbody",location=LOCATION,epochs={"start":start,"stop":stop,"step":step}).vectors(refplane=REFPLANE,aberrations=ABERRATIONS)
    jd=np.asarray(t["datetime_jd"],float)
    xyz=np.column_stack([np.asarray(t[a],float) for a in "xyz"])*AU_KM
    if len(jd)<3 or not np.all(np.diff(jd)>0): raise RuntimeError(f"REJECTED JPL grid {body}")
    return Series(jd,xyz)
def splines(jd:np.ndarray,xyz:np.ndarray)->List[CubicSpline]:
    return [CubicSpline(jd,xyz[:,i],bc_type="natural") for i in range(3)]
def evaluate(c:List[CubicSpline],jd:float)->np.ndarray:
    return np.array([float(f(jd)) for f in c])
def angular_sep_single(e:np.ndarray,s:np.ndarray,v:np.ndarray)->float:
    su=unit(s-e); vu=unit(v-e)
    return math.atan2(float(np.linalg.norm(np.cross(su,vu))),float(np.dot(su,vu)))
def closest_approach(jd:np.ndarray,earth:np.ndarray,sun:np.ndarray,venus:np.ndarray)->tuple[float,float]:
    es=np.linalg.norm(sun-earth,axis=1); ev=np.linalg.norm(venus-earth,axis=1)
    su=(sun-earth)/es[:,None]; vu=(venus-earth)/ev[:,None]
    sep=np.arctan2(np.linalg.norm(np.cross(su,vu),axis=1),np.einsum("ij,ij->i",su,vu))
    i=int(np.argmin(sep)); lo=max(0,i-3); hi=min(len(jd)-1,i+3)
    ec,sc,vc=splines(jd,earth),splines(jd,sun),splines(jd,venus)
    def objective(x:float)->float:
        return angular_sep_single(evaluate(ec,x),evaluate(sc,x),evaluate(vc,x))
    r=minimize_scalar(objective,bounds=(float(jd[lo]),float(jd[hi])),method="bounded",options={"xatol":1e-12,"maxiter":500})
    if not r.success: raise RuntimeError("REJECTED closest approach")
    return float(r.x),float(r.fun)
def tangent_basis(e:np.ndarray,s:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    n=unit(s-e); x=np.cross(np.array([0.,0.,1.]),n)
    if np.linalg.norm(x)<1e-10: x=np.cross(np.array([0.,1.,0.]),n)
    x=unit(x); y=unit(np.cross(n,x)); return x,y
def signed_line_angle(raw:float)->float:
    return ((raw+90)%180)-90
def fit_track(hours:np.ndarray,xy:np.ndarray)->Fit:
    cx=np.polyfit(hours,xy[:,0],2); cy=np.polyfit(hours,xy[:,1],2)
    model=np.column_stack((np.polyval(cx,hours),np.polyval(cy,hours)))
    rms=float(np.sqrt(np.mean(np.sum((xy-model)**2,axis=1))))
    vx,vy=float(cx[1]),float(cy[1]); ax,ay=2*float(cx[0]),2*float(cy[0]); speed2=vx*vx+vy*vy
    if speed2<=0: raise RuntimeError("REJECTED degenerate fit")
    raw=math.degrees(math.atan2(vy,vx))%360; signed=signed_line_angle(raw)
    slope=math.inf if abs(vx)<1e-15 else vy/vx
    curvature=abs(vx*ay-vy*ax)/(speed2**1.5)
    return Fit(raw,signed,abs(signed),slope,rms,curvature)
def add_solar_limb(ax,center_date,y_center,radius_arcsec):
    left,right=ax.get_xlim(); bottom,top=ax.get_ylim(); width_days=right-left; height_arcsec=top-bottom
    bbox=ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted()); pixel_ratio=bbox.width/bbox.height
    radius_days=radius_arcsec*width_days/height_arcsec/pixel_ratio
    ax.add_patch(Ellipse((mdates.date2num(center_date),y_center),2*radius_days,2*radius_arcsec,facecolor="#C98A18",edgecolor="#E64A19",linewidth=1.05,alpha=.90,zorder=4,label="Solar limb"))
def make_plot(year,dates,earth_y,venus_y,ca_date,ca_y,solar_radius,earth_fit,venus_fit,projected_angle,output_path):
    fig,ax=plt.subplots(figsize=(18.0,8.5),dpi=100); fig.patch.set_facecolor("black"); ax.set_facecolor("black")
    ax.plot(dates,venus_y,color="#1E78B4",linewidth=.72,label="Venus trajectory",zorder=3)
    ax.plot(dates,earth_y,color="#2FAA45",linewidth=.72,label="Earth trajectory",zorder=2)
    ax.set_xlim(ca_date-pd.Timedelta(days=183),ca_date+pd.Timedelta(days=183))
    y_extent=max(1200.,float(np.max(np.abs(earth_y)))*1.08,float(np.max(np.abs(venus_y)))*1.08,abs(ca_y)+solar_radius*1.35)
    ax.set_ylim(-y_extent,y_extent); add_solar_limb(ax,ca_date,ca_y,solar_radius)
    ax.axvline(ca_date,color="#B0B0B0",linewidth=.52,linestyle="--",alpha=.72,zorder=1)
    ax.scatter([ca_date],[ca_y],s=22,facecolor="white",edgecolor="#DADADA",linewidth=.55,zorder=7,label="Closest approach")
    ca_text=Time(ca_date,scale="utc").strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    annotation="\n".join([f"Closest Approach (UTC): {ca_text}",f"Earth Track From Ecliptic: {earth_fit.positive_horizontal_deg:.6f}°",f"Projected Venus Transit Track: {projected_angle:.6f}°",f"Venus Transit Track From Ecliptic: {venus_fit.positive_horizontal_deg:.6f}°"])
    xoff=30. if ca_date.month<=8 else -30.; ha="left" if xoff>0 else "right"; yoff=-.18*y_extent if ca_y>=0 else .18*y_extent; va="top" if yoff<0 else "bottom"
    ax.annotate(annotation,xy=(ca_date,ca_y),xytext=(ca_date+pd.Timedelta(days=xoff),ca_y+yoff),color="#ECECEC",fontsize=10.2,ha=ha,va=va,arrowprops={"arrowstyle":"-","color":"#AFAFAF","linewidth":.65},bbox={"boxstyle":"round,pad=0.32","facecolor":"#050505","edgecolor":"#858585","alpha":.94},zorder=8)
    ax.set_title(f"{year} Venus Transit — Registered Earth–Venus Crossing and Track Angles",color="#F0F0F0",fontsize=15,weight="bold",pad=8)
    ax.set_xlabel(f"Calendar month — {year}",color="#E0E0E0",fontsize=10.5)
    ax.set_ylabel(f"Registered tangent-plane displacement (arcsec, {VISUAL_SCALE:.0f}× visual scale)",color="#E0E0E0",fontsize=10.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(colors="#D7D7D7",labelsize=9,width=.48); ax.grid(True,color="#646464",alpha=.28,linewidth=.42)
    for s in ax.spines.values(): s.set_color("#8E8E8E"); s.set_linewidth(.55)
    leg=ax.legend(loc="upper right",frameon=False,fontsize=9.4)
    for t in leg.get_texts(): t.set_color("#DFDFDF")
    fig.tight_layout(); fig.savefig(output_path,dpi=300,facecolor="black",bbox_inches="tight"); plt.close(fig); display(Image(filename=str(output_path)))
def process(year:int,center_text:str)->dict:
    center=Time(center_text,scale="utc"); delta=SEARCH_HALF_H/24
    start=Time(center.jd-delta,format="jd",scale="utc").strftime("%Y-%m-%d %H:%M"); stop=Time(center.jd+delta,format="jd",scale="utc").strftime("%Y-%m-%d %H:%M")
    earth=query("399",start,stop,FINE_STEP); venus=query("299",start,stop,FINE_STEP); sun=query("10",start,stop,FINE_STEP)
    if not(len(earth.jd)==len(venus.jd)==len(sun.jd) and np.allclose(earth.jd,venus.jd,atol=1e-11,rtol=0) and np.allclose(earth.jd,sun.jd,atol=1e-11,rtol=0)): raise RuntimeError("REJECTED mismatched minute grids")
    ca_jd,minsep=closest_approach(earth.jd,earth.xyz,sun.xyz,venus.xyz)
    ec,vc,sc=splines(earth.jd,earth.xyz),splines(venus.jd,venus.xyz),splines(sun.jd,sun.xyz)
    e0,v0,s0=evaluate(ec,ca_jd),evaluate(vc,ca_jd),evaluate(sc,ca_jd); xaxis,yaxis=tangent_basis(e0,s0)
    mask=np.abs((earth.jd-ca_jd)*24)<=FIT_HALF_H; hours=(earth.jd[mask]-ca_jd)*24
    earth_xy=np.column_stack(((earth.xyz[mask]-e0)@xaxis,(earth.xyz[mask]-e0)@yaxis)); venus_xy=np.column_stack(((venus.xyz[mask]-v0)@xaxis,(venus.xyz[mask]-v0)@yaxis))
    earth_fit,venus_fit=fit_track(hours,earth_xy),fit_track(hours,venus_xy)
    projected=abs(earth_fit.signed_line_deg-venus_fit.signed_line_deg); projected=180-projected if projected>90 else projected; projected=abs(projected)
    annual_start=Time(ca_jd-183,format="jd",scale="tdb").utc.strftime("%Y-%m-%d %H:%M"); annual_stop=Time(ca_jd+183,format="jd",scale="tdb").utc.strftime("%Y-%m-%d %H:%M")
    ey=query("399",annual_start,annual_stop,YEAR_STEP); vy=query("299",annual_start,annual_stop,YEAR_STEP)
    if len(ey.jd)!=len(vy.jd) or not np.allclose(ey.jd,vy.jd,atol=1e-11,rtol=0): raise RuntimeError("REJECTED mismatched annual grids")
    es=float(np.linalg.norm(s0-e0)); scale=AS_PER_RAD/es; reg=float(np.dot(v0-s0,yaxis))*scale
    earth_y=VISUAL_SCALE*(((ey.xyz-e0)@yaxis)*scale+reg); venus_y=VISUAL_SCALE*(((vy.xyz-v0)@yaxis)*scale+reg); ca_y=VISUAL_SCALE*reg
    ca_time=Time(ca_jd,format="jd",scale="tdb"); ca_date=ca_time.utc.to_datetime(); dates=Time(ey.jd,format="jd",scale="tdb").utc.to_datetime(); solar_radius=math.asin(R_SUN_KM/es)*AS_PER_RAD
    output=OUT/PNG_NAMES[year]; make_plot(year,dates,earth_y,venus_y,ca_date,ca_y,solar_radius,earth_fit,venus_fit,projected,output)
    return {"transit_year":year,"closest_approach_utc":ca_time.utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],"jd_tdb":ca_jd,"earth_track_from_ecliptic_deg":earth_fit.positive_horizontal_deg,"projected_venus_transit_track_deg":projected,"venus_transit_track_from_ecliptic_deg":venus_fit.positive_horizontal_deg,"minimum_separation_arcsec":minsep*AS_PER_RAD,"png_file":str(output)}
def main()->None:
    OUT.mkdir(parents=True,exist_ok=True)
    section("CODE INPUTS"); print(f"Version                              {VERSION}"); print(f"Program                              {FILENAME}"); print("JPL source                           Horizons vectors API"); print(f"Observer/reference                   {LOCATION}; Sun=10 Venus=299 Earth=399"); print(f"Reference plane/aberrations          {REFPLANE}/{ABERRATIONS}")
    section("COMMENTS"); print("Standalone V0139; no dependency on another project Python file."); print("Original V0139 plot design, geometry, colors, axes, and annual registration are preserved."); print("Only closest approach uses the verified atan2 cross/dot angular-separation minimization and annotations now list the three V0152P track references.")
    rows=[]
    for year,center in TRANSITS.items(): print(f"DEBUG processing {year}",flush=True); rows.append(process(year,center))
    csv_path=OUT/CSV_NAME; pd.DataFrame(rows).to_csv(csv_path,index=False,float_format="%.12g")
    section("RESULTS")
    for r in rows:
        print(f"{r['transit_year']}  CA {r['closest_approach_utc']}  JD(TDB) {r['jd_tdb']:.9f}")
        print(f"Earth / Projected Venus / Venus      {r['earth_track_from_ecliptic_deg']:.6f}  {r['projected_venus_transit_track_deg']:.6f}  {r['venus_transit_track_from_ecliptic_deg']:.6f} deg")
    section("OUTPUT SUMMARY"); print(f"CSV                                  {csv_path}")
    for r in rows: print(f"PNG {r['transit_year']}                            {r['png_file']}")
    section("PAPER COMPARISON"); print("NOT USED: published or manual closest-approach times.")
    section("EQUATION STATUS"); print("VERIFIED closest approach = minimum atan2(||Sun_hat × Venus_hat||, Sun_hat · Venus_hat) from JPL vectors."); print("VERIFIED plot design unchanged except requested CA times and three angle-reference labels.")
    print(datetime.now().astimezone().isoformat(timespec="seconds")); print(VERSION)
if __name__=="__main__": main()
# V0139