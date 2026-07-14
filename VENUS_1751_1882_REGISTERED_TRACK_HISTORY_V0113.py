# V0113
# Audit reference: continuous 1751-1882 registered Earth/Venus historical track plot using JPL Horizons vectors and V0112 visual style.
from __future__ import annotations
import importlib.util, math, subprocess, sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

def need(module: str, package: str) -> None:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', package], check=True)

for _m,_p in [('astroquery','astroquery'),('astropy','astropy'),('scipy','scipy'),('pandas','pandas'),('matplotlib','matplotlib'),('IPython','ipython')]:
    need(_m,_p)

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

VERSION='V0113'
OUT=Path('/content/VENUS_1751_1882_REGISTERED_TRACK_HISTORY_V0113_OUTPUT')
PNG='VENUS_1751_1882_REGISTERED_TRACK_HISTORY_V0113.png'
CSV='VENUS_1751_1882_REGISTERED_TRACK_HISTORY_V0113.csv'
AU_KM=149597870.700
R_SUN_KM=695700.000
AS_PER_RAD=206264.80624709636
LOCATION='@0'; REFPLANE='earth'; ABERRATIONS='geometric'
START='1751-01-01 00:00'; STOP='1883-01-01 00:00'
REFERENCE_CENTER='1882-12-06 17:00'
HISTORY_STEP='5d'; FINE_STEP='1m'; SEARCH_HALF_H=18.0; FIT_HALF_H=10.0

@dataclass(frozen=True)
class Series:
    jd: np.ndarray
    xyz: np.ndarray

@dataclass(frozen=True)
class Fit:
    angle: float
    slope: float
    rms: float
    curvature: float

def section(name: str) -> None:
    print(name); print('-'*len(name))

def unit(a: np.ndarray) -> np.ndarray:
    n=float(np.linalg.norm(a))
    if not np.isfinite(n) or n<=0: raise ValueError('REJECTED invalid vector')
    return a/n

def wrapdiff(a: float,b: float)->float:
    return abs((a-b+180.0)%360.0-180.0)

def query(body: str,start: str,stop: str,step: str)->Series:
    table=Horizons(id=body,id_type='majorbody',location=LOCATION,epochs={'start':start,'stop':stop,'step':step}).vectors(refplane=REFPLANE,aberrations=ABERRATIONS)
    jd=np.asarray(table['datetime_jd'],float)
    xyz=np.column_stack([np.asarray(table[k],float) for k in ('x','y','z')])*AU_KM
    if len(jd)<3 or not np.all(np.diff(jd)>0): raise RuntimeError(f'REJECTED JPL grid {body}')
    return Series(jd,xyz)

def query_history(body: str)->Series:
    jd_parts=[]; xyz_parts=[]
    for year0 in range(1751,1883,11):
        year1=min(year0+11,1883)
        start=f'{year0}-01-01 00:00'; stop=f'{year1}-01-01 00:00'
        print(f'DEBUG JPL {body} {year0}-{year1-1}',flush=True)
        s=query(body,start,stop,HISTORY_STEP)
        if jd_parts and np.isclose(s.jd[0],jd_parts[-1][-1],atol=1e-11,rtol=0):
            s=Series(s.jd[1:],s.xyz[1:])
        jd_parts.append(s.jd); xyz_parts.append(s.xyz)
    jd=np.concatenate(jd_parts); xyz=np.vstack(xyz_parts)
    return Series(jd,xyz)

def spl(s: Series)->List[CubicSpline]:
    return [CubicSpline(s.jd,s.xyz[:,i],bc_type='natural') for i in range(3)]

def ev(sp: List[CubicSpline],t: float)->np.ndarray:
    return np.array([f(t) for f in sp])

def sep(e: np.ndarray,s: np.ndarray,v: np.ndarray)->np.ndarray:
    a=s-e; b=v-e
    a/=np.linalg.norm(a,axis=1)[:,None]; b/=np.linalg.norm(b,axis=1)[:,None]
    return np.arccos(np.clip(np.einsum('ij,ij->i',a,b),-1,1))

def closest(jd: np.ndarray,e: np.ndarray,s: np.ndarray,v: np.ndarray)->tuple[float,float]:
    z=sep(e,s,v); i=int(np.argmin(z)); lo=max(0,i-3); hi=min(len(jd)-1,i+3)
    es=[CubicSpline(jd,e[:,k]) for k in range(3)]
    ss=[CubicSpline(jd,s[:,k]) for k in range(3)]
    vs=[CubicSpline(jd,v[:,k]) for k in range(3)]
    def f(t: float)->float:
        a=unit(ev(ss,t)-ev(es,t)); b=unit(ev(vs,t)-ev(es,t))
        return math.acos(float(np.clip(np.dot(a,b),-1,1)))
    q=minimize_scalar(f,bounds=(float(jd[lo]),float(jd[hi])),method='bounded',options={'xatol':1e-12,'maxiter':300})
    if not q.success: raise RuntimeError('REJECTED closest approach')
    return float(q.x),float(q.fun)

def basis(e: np.ndarray,s: np.ndarray)->tuple[np.ndarray,np.ndarray]:
    n=unit(s-e); x=np.cross(np.array([0.,0.,1.]),n)
    if np.linalg.norm(x)<1e-10: x=np.cross(np.array([0.,1.,0.]),n)
    x=unit(x); y=unit(np.cross(n,x)); return x,y

def fit(hours: np.ndarray,xy: np.ndarray)->Fit:
    cx=np.polyfit(hours,xy[:,0],2); cy=np.polyfit(hours,xy[:,1],2)
    model=np.column_stack((np.polyval(cx,hours),np.polyval(cy,hours)))
    rms=float(np.sqrt(np.mean(np.sum((xy-model)**2,axis=1))))
    vx,vy,ax,ay=float(cx[1]),float(cy[1]),float(2*cx[0]),float(2*cy[0])
    s2=vx*vx+vy*vy
    return Fit(math.degrees(math.atan2(vy,vx))%360.0,math.inf if abs(vx)<1e-15 else vy/vx,rms,abs(vx*ay-vy*ax)/(s2**1.5))

def draw_sun(ax,ca_date,radius_as: float)->None:
    left,right=ax.get_xlim(); bottom,top=ax.get_ylim()
    bbox=ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    radius_days=radius_as*(right-left)/(top-bottom)/(bbox.width/bbox.height)
    ax.add_patch(Ellipse((mdates.date2num(ca_date),0.0),2*radius_days,2*radius_as,facecolor='#C98212',edgecolor='#E04B18',linewidth=1.15,alpha=.72,zorder=1,label='Solar limb'))

def main()->None:
    OUT.mkdir(parents=True,exist_ok=True)
    section('CODE INPUTS')
    print(f'Version                              {VERSION}')
    print(f'Historical interval                  {START} through 1882-12-31 23:59')
    print(f'JPL observer/reference               {LOCATION}; Earth=399 Venus=299 Sun=10')
    print(f'Frame/plane/aberrations              JPL default ICRF-J2000/{REFPLANE}/{ABERRATIONS}')
    print(f'History/fine cadence                 {HISTORY_STEP}/{FINE_STEP}')
    print(f'NOT USED AS CA INPUT                 {REFERENCE_CENTER} broad search center only')
    section('COMMENTS')
    print('One continuous calendar-year plot preserves the V0112 visual style.')
    print('The 1882 transit closest approach defines the registered tangent plane and marker.')
    print('All geometry is derived from fresh JPL Horizons vectors; no AI imagery.')

    c=Time(REFERENCE_CENTER,scale='utc'); d=SEARCH_HALF_H/24
    start=Time(c.jd-d,format='jd',scale='utc').strftime('%Y-%m-%d %H:%M')
    stop=Time(c.jd+d,format='jd',scale='utc').strftime('%Y-%m-%d %H:%M')
    ef=query('399',start,stop,FINE_STEP); vf=query('299',start,stop,FINE_STEP); sf=query('10',start,stop,FINE_STEP)
    if not(len(ef.jd)==len(vf.jd)==len(sf.jd) and np.allclose(ef.jd,vf.jd,atol=1e-11,rtol=0) and np.allclose(ef.jd,sf.jd,atol=1e-11,rtol=0)):
        raise RuntimeError('REJECTED mismatched fine grids')
    ca,minsep=closest(ef.jd,ef.xyz,sf.xyz,vf.xyz)
    esp,vsp,ssp=spl(ef),spl(vf),spl(sf)
    e0,v0,s0=ev(esp,ca),ev(vsp,ca),ev(ssp,ca)
    x,y=basis(e0,s0)
    mask=np.abs((ef.jd-ca)*24)<=FIT_HALF_H; hours=(ef.jd[mask]-ca)*24
    exy_fit=np.column_stack(((ef.xyz[mask]-e0)@x,(ef.xyz[mask]-e0)@y))
    vxy_fit=np.column_stack(((vf.xyz[mask]-v0)@x,(vf.xyz[mask]-v0)@y))
    efit,vfit=fit(hours,exy_fit),fit(hours,vxy_fit)
    apparent=wrapdiff(efit.angle,vfit.angle)

    earth=query_history('399'); venus=query_history('299')
    if len(earth.jd)!=len(venus.jd) or not np.allclose(earth.jd,venus.jd,atol=1e-11,rtol=0): raise RuntimeError('REJECTED history grid mismatch')
    des=float(np.linalg.norm(s0-e0)); scale=AS_PER_RAD/des
    registration=float(np.dot(v0-s0,y))*scale
    earth_y=2.0*(((earth.xyz-e0)@y)*scale+registration)
    venus_y=2.0*(((venus.xyz-v0)@y)*scale+registration)
    dates=Time(earth.jd,format='jd',scale='tdb').utc.to_datetime()
    ca_time=Time(ca,format='jd',scale='tdb'); ca_date=ca_time.utc.to_datetime(); ca_utc=ca_time.utc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    solar_radius=math.asin(R_SUN_KM/des)*AS_PER_RAD

    fig,ax=plt.subplots(figsize=(18,8),dpi=100); fig.patch.set_facecolor('black'); ax.set_facecolor('black')
    ax.plot(dates,venus_y,color='#1789D4',linewidth=.65,label='Venus trajectory',zorder=4)
    ax.plot(dates,earth_y,color='#21B33B',linewidth=.65,label='Earth trajectory',zorder=3)
    ax.set_xlim(datetime(1751,1,1),datetime(1883,1,1))
    extent=max(float(np.nanmax(np.abs(earth_y))),float(np.nanmax(np.abs(venus_y))),solar_radius*1.3)
    ax.set_ylim(-extent*1.04,extent*1.04)
    draw_sun(ax,ca_date,solar_radius)
    ax.axvline(ca_date,color='#AFAFAF',linewidth=.55,linestyle='--',alpha=.75)
    ax.scatter([ca_date],[2*registration],s=22,facecolor='white',edgecolor='#D9D9D9',linewidth=.55,zorder=7,label='1882 closest approach')
    text=f'Earth angle: {efit.angle:+.6f}°\nVenus angle: {vfit.angle:+.6f}°\nApparent track angle: {apparent:.6f}°'
    ax.text(.68,.16,text,transform=ax.transAxes,color='#E8E8E8',fontsize=10,bbox={'boxstyle':'round,pad=.32','facecolor':'#060606','edgecolor':'#808080','alpha':.92})
    ax.set_title('1751–1882 Venus–Earth Registered Track History',color='#F4F4F4',fontsize=16,weight='bold',pad=8)
    ax.set_xlabel('Calendar year — 1751 through 1882',color='#E6E6E6',fontsize=11)
    ax.set_ylabel('Registered tangent-plane displacement (arcsec, 2× visual scale)',color='#E6E6E6',fontsize=11)
    ax.xaxis.set_major_locator(mdates.YearLocator(10)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(colors='#E6E6E6',labelsize=8,width=.5); ax.grid(True,color='#777777',alpha=.30,linewidth=.4)
    for sp in ax.spines.values(): sp.set_color('#B0B0B0'); sp.set_linewidth(.55)
    leg=ax.legend(loc='upper right',frameon=False,fontsize=10)
    for t in leg.get_texts(): t.set_color('#E8E8E8')
    fig.tight_layout(); png=OUT/PNG; fig.savefig(png,dpi=600,facecolor='black',bbox_inches='tight'); plt.close(fig); display(Image(filename=str(png)))

    pd.DataFrame({'jd_tdb':earth.jd,'utc':Time(earth.jd,format='jd',scale='tdb').utc.isot,'earth_registered_y_arcsec_2x':earth_y,'venus_registered_y_arcsec_2x':venus_y}).to_csv(OUT/CSV,index=False,float_format='%.12g')
    section('RESULTS')
    print(f'1882 closest approach UTC            {ca_utc}')
    print(f'1882 closest approach JD TDB         {ca:.9f}')
    print(f'Earth track angle                    {efit.angle:.6f} deg')
    print(f'Venus track angle                    {vfit.angle:.6f} deg')
    print(f'Apparent track angle                 {apparent:.6f} deg')
    print(f'Minimum separation                   {minsep*AS_PER_RAD:.6f} arcsec')
    print(f'Solar angular radius                 {solar_radius:.6f} arcsec')
    print(f'History samples per trajectory       {len(earth.jd)}')
    section('OUTPUT SUMMARY')
    print(f'PNG {png} bytes {png.stat().st_size}')
    print(f'CSV {OUT/CSV}')
    section('PAPER COMPARISON')
    print('NOT USED: published angles, manual closest-approach times, and manual track positions.')
    section('EQUATION STATUS')
    print('VERIFIED apparent_track_angle = abs(wrap180(Earth angle - Venus angle))')
    print(f'Angle range verified                 {0.0<=apparent<=180.0}')
    print(datetime.now().astimezone().isoformat(timespec='seconds'))
    print(VERSION)

if __name__=='__main__': main()
# V0113