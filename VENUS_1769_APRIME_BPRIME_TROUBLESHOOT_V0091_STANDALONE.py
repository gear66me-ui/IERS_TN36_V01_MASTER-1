# V0091
# Audit reference: standalone A prime B prime troubleshooting; compares V0067 seconds-space CA and V0089 JD-space CA; Python/Matplotlib only, no AI images.
from __future__ import annotations

import math, subprocess, sys, time, warnings
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

VERSION = 'V0091'
OUT = Path('/content/VENUS_1769_V0091_APBP_TROUBLESHOOT')
PNG = OUT / 'VENUS_1769_APBP_TROUBLESHOOT_V0091.png'
CSV = OUT / 'VENUS_1769_APBP_TROUBLESHOOT_V0091.csv'
ARC = 206_264.80624709636
JPL_AU_KM = 149_597_870.700000
SUN_RADIUS_KM = 695_700.000000
VENUS_RADIUS_KM = 6_051.800000
START, STOP, STEP = '1769-06-03 18:00', '1769-06-04 06:00', '1m'
GEOCENTER = '@399'
PV = dict(key='POINT_VENUS', label='Point Venus, Tahiti', lat=-17.4956, lon=-149.4939, elevation=0.0, body=399, color='#42D7C3')
VA = dict(key='VARDO', label='Vardø, Norway', lat=70.3724, lon=31.1103, elevation=0.0, body=399, color='#D89B18')
SITES = (PV, VA)
TARGETS = (('SUN','10'), ('VENUS','299'))
PREFIXES = ('GEOCENTER_SUN','GEOCENTER_VENUS','POINT_VENUS_SUN','POINT_VENUS_VENUS','VARDO_SUN','VARDO_VENUS')
BG, FG, MUTED = '#000000', '#F8FAFC', '#B8CBD6'
SUN_FILL, SUN_LIMB, TEAL, GOLD, HEADER, BODY = '#D95A1B', '#FFD34A', '#164B55', '#563B0B', '#23466F', '#101A2E'
LOCAL_TZ = ZoneInfo('America/Bogota')

def require(import_name, package_name):
    try: __import__(import_name)
    except ImportError: subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', package_name])
for a,b in [('numpy','numpy'),('pandas','pandas'),('scipy','scipy'),('astropy','astropy'),('astroquery','astroquery'),('matplotlib','matplotlib'),('IPython','ipython')]: require(a,b)
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
warnings.filterwarnings('ignore', message='.*id_type.*deprecated.*'); warnings.filterwarnings('ignore', message='.*dubious year.*')

def norm(v): return float(np.linalg.norm(np.asarray(v, dtype=float)))
def unit(v):
    a=np.asarray(v,dtype=float); n=norm(a)
    if n<=0: raise RuntimeError('zero vector')
    return a/n
def utc(jd): return Time(float(jd), format='jd', scale='tdb').utc.datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
def loc(site): return dict(lon=float(site['lon']), lat=float(site['lat']), elevation=float(site['elevation']), body=int(site['body']))

def query(prefix, target, location):
    last=None
    for k in range(4):
        try:
            tab=Horizons(id=target, location=location, epochs={'start':START,'stop':STOP,'step':STEP}, id_type=None).vectors(refplane='ecliptic', aberrations='geometric', cache=False)
            fr=tab.to_pandas(); out=pd.DataFrame({'JD_TDB':pd.to_numeric(fr['datetime_jd'], errors='coerce')})
            for ax in 'xyz': out[f'{prefix}_{ax.upper()}_KM']=pd.to_numeric(fr[ax], errors='coerce')*JPL_AU_KM
            out=out.dropna().drop_duplicates('JD_TDB').sort_values('JD_TDB').reset_index(drop=True)
            if len(out)<600: raise RuntimeError(f'incomplete {prefix}: {len(out)} rows')
            return out
        except Exception as e:
            last=e; time.sleep(1.5*(k+1))
    raise RuntimeError(f'JPL query failed {prefix}: {last}')

def master():
    frames=[]
    for name,target in TARGETS: frames.append(query(f'GEOCENTER_{name}', target, GEOCENTER))
    for site in SITES:
        for name,target in TARGETS: frames.append(query(f"{site['key']}_{name}", target, loc(site)))
    m=frames[0]
    for f in frames[1:]: m=m.merge(f,on='JD_TDB',how='inner',validate='one_to_one')
    if len(m)<600: raise RuntimeError('master too short')
    return m.sort_values('JD_TDB').reset_index(drop=True)

def cache(m):
    c={'JD_TDB':m['JD_TDB'].to_numpy(float)}
    for p in PREFIXES:
        for ax in 'XYZ': c[f'{p}_{ax}_KM']=CubicSpline(c['JD_TDB'], m[f'{p}_{ax}_KM'].to_numpy(float), bc_type='natural')
    return c

def vec(c,p,jd): return np.array([float(c[f'{p}_{ax}_KM'](float(jd))) for ax in 'XYZ'], dtype=float)
def sep(c, a, b, jd): return math.atan2(norm(np.cross(unit(vec(c,a,jd)), unit(vec(c,b,jd)))), float(np.dot(unit(vec(c,a,jd)), unit(vec(c,b,jd)))))

def basis(sun):
    cen=unit(sun); pole=np.array([0.,0.,1.]); east=np.cross(pole,cen)
    if norm(east)<1e-14: east=np.cross(np.array([0.,1.,0.]), cen)
    east=unit(east); north=unit(np.cross(cen,east))
    if float(np.dot(north,pole))<0: east,north=-east,-north
    return cen,east,north

def gnom(ray, bas):
    cen,east,north=bas; h=unit(ray); den=float(np.dot(h,cen))
    if den<=0: raise RuntimeError('ray outside tangent hemisphere')
    return np.array([float(np.dot(h,east)), float(np.dot(h,north))])/den

def rel(c, site, jd, bas): return ARC*(gnom(vec(c,f'{site}_VENUS',jd), bas)-gnom(vec(c,f'{site}_SUN',jd), bas))

def ca_v0067_seconds(c):
    j=np.asarray(c['JD_TDB'],float); vals=np.array([sep(c,'GEOCENTER_SUN','GEOCENTER_VENUS',x) for x in j]); i=int(np.argmin(vals))
    lo=float(j[max(0,i-3)]); hi=float(j[min(len(j)-1,i+3)]); ref=0.5*(lo+hi)
    slo=(lo-ref)*86400.; shi=(hi-ref)*86400.
    r=minimize_scalar(lambda s: sep(c,'GEOCENTER_SUN','GEOCENTER_VENUS',ref+float(s)/86400.), bounds=(slo,shi), method='bounded', options={'xatol':1e-4,'maxiter':500})
    if not r.success: raise RuntimeError('V0067 seconds CA failed')
    return ref+float(r.x)/86400.

def ca_v0089_jd(c):
    j=np.asarray(c['JD_TDB'],float); vals=np.array([sep(c,'GEOCENTER_SUN','GEOCENTER_VENUS',x) for x in j]); i=int(np.argmin(vals))
    r=minimize_scalar(lambda jd: sep(c,'GEOCENTER_SUN','GEOCENTER_VENUS',jd), bounds=(float(j[max(0,i-2)]), float(j[min(len(j)-1,i+2)])), method='bounded')
    if not r.success: raise RuntimeError('V0089 JD CA failed')
    return float(r.x)

def apbp(c, jd, label):
    bas=basis(vec(c,'GEOCENTER_SUN',jd)); h=0.5/86400.
    qpv=rel(c,'POINT_VENUS',jd,bas); qv=rel(c,'VARDO',jd,bas)
    vp=rel(c,'POINT_VENUS',jd+h,bas)-rel(c,'POINT_VENUS',jd-h,bas)
    vv=rel(c,'VARDO',jd+h,bas)-rel(c,'VARDO',jd-h,bas)
    direction=unit(unit(vp)+unit(vv)); normal2=np.array([-direction[1], direction[0]])
    if float(np.dot(qv-qpv,normal2))<0: normal2=-normal2
    cen,east,north=bas; normal3=unit(normal2[0]*east+normal2[1]*north)
    geo_sun=vec(c,'GEOCENTER_SUN',jd)
    station_pv=geo_sun-vec(c,'POINT_VENUS_SUN',jd); station_v=geo_sun-vec(c,'VARDO_SUN',jd); baseline=station_v-station_pv
    if float(np.dot(baseline,normal3))<0: normal2=-normal2; normal3=-normal3
    apbp_as=float(np.dot(qv-qpv,normal2)); km_per_as=norm(geo_sun)/ARC; ab_km=float(np.dot(baseline,normal3))
    return dict(method=label, jd=jd, utc=utc(jd), apbp_as=apbp_as, apbp_km=apbp_as*km_per_as, ab_as=ab_km/km_per_as, ab_km=ab_km, km_per_as=km_per_as)

def make_plot(rows):
    plt.close('all'); plt.rcParams.update({'figure.facecolor':BG,'savefig.facecolor':BG,'axes.facecolor':BG,'text.color':FG,'axes.labelcolor':FG,'xtick.color':MUTED,'ytick.color':MUTED,'axes.edgecolor':MUTED,'font.family':'DejaVu Serif'})
    fig=plt.figure(figsize=(13.5,7.5)); gs=fig.add_gridspec(2,2,width_ratios=(1.24,1.0),height_ratios=(1,0.78),left=.055,right=.975,top=.90,bottom=.09,wspace=.20,hspace=.34)
    ax=fig.add_subplot(gs[:,0]); ax2=fig.add_subplot(gs[0,1]); ax3=fig.add_subplot(gs[1,1]); fig.suptitle("1769 VENUS TRANSIT — A′B′ TROUBLESHOOT",fontsize=16,fontweight='bold')
    methods=[r['method'] for r in rows]; x=np.arange(len(rows)); ap=[r['apbp_as'] for r in rows]; ab=[r['ab_as'] for r in rows]
    ax.axhline(ap[0],color=TEAL,lw=.9,alpha=.6,label='V0067 reference A′B′')
    ax.plot(x,ap,marker='o',ms=8,lw=1.2,color=SUN_LIMB,label='A′B′ arcsec')
    for i,r in enumerate(rows): ax.annotate(f"{r['apbp_as']:.9f}\n{r['apbp_km']:,.6f} km",(i,ap[i]),xytext=(0,20 if i==0 else -38),textcoords='offset points',ha='center',fontsize=8,arrowprops=dict(arrowstyle='-',color=SUN_LIMB,lw=.45))
    ax.set_xticks(x,methods); ax.set_ylabel("A′B′ common-normal separation (arcsec)"); ax.grid(True,alpha=.18,lw=.35); ax.legend(frameon=False,fontsize=8)
    ax2.plot(x,ab,marker='o',ms=7,lw=1.1,color='#D89B18'); ax2.set_xticks(x,methods); ax2.set_ylabel('AB projected baseline (arcsec)'); ax2.grid(True,alpha=.18,lw=.35)
    for i,r in enumerate(rows): ax2.annotate(f"{r['ab_as']:.9f}\n{r['ab_km']:,.6f} km",(i,ab[i]),xytext=(0,18 if i==0 else -34),textcoords='offset points',ha='center',fontsize=7.5,arrowprops=dict(arrowstyle='-',color='#D89B18',lw=.45))
    ax3.axis('off')
    d=rows[1]
    table=[['Metric','V0067 seconds-space','V0089 JD-space','Delta'],['CA UTC',rows[0]['utc'],rows[1]['utc'],''],['A′B′ arcsec',f"{rows[0]['apbp_as']:.12f}",f"{rows[1]['apbp_as']:.12f}",f"{rows[1]['apbp_as']-rows[0]['apbp_as']:+.12f}"],['A′B′ km',f"{rows[0]['apbp_km']:,.9f}",f"{rows[1]['apbp_km']:,.9f}",f"{rows[1]['apbp_km']-rows[0]['apbp_km']:+,.9f}"],['AB arcsec',f"{rows[0]['ab_as']:.12f}",f"{rows[1]['ab_as']:.12f}",f"{rows[1]['ab_as']-rows[0]['ab_as']:+.12f}"],['AB km',f"{rows[0]['ab_km']:,.9f}",f"{rows[1]['ab_km']:,.9f}",f"{rows[1]['ab_km']-rows[0]['ab_km']:+,.9f}"],['Conclusion','KEEP','REJECT','CA solver changed']]
    t=ax3.table(cellText=table,cellLoc='left',bbox=[0,0,1,1],colWidths=[.20,.28,.28,.24]); t.auto_set_font_size(False)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor('#70879A'); cell.set_linewidth(.35); cell.get_text().set_color(FG); cell.get_text().set_fontsize(7.3 if r else 7.8)
        cell.set_facecolor(HEADER if r==0 else (GOLD if r in (2,3,4,5) else (TEAL if r==6 else BODY)))
        if r in (0,6): cell.get_text().set_fontweight('bold')
    fig.text(.5,.025,'NO AI IMAGES — Python/Matplotlib only. Independent JPL Horizons download; A′B′ troubleshooting only.',ha='center',fontsize=7,color=MUTED)
    fig.savefig(PNG,dpi=170,bbox_inches='tight',pad_inches=.02); plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True); print('V0091 A′B′ TROUBLESHOOT — downloading fresh JPL vectors')
    m=master(); c=cache(m)
    rows=[apbp(c, ca_v0067_seconds(c), 'V0067 seconds-space CA'), apbp(c, ca_v0089_jd(c), 'V0089 JD-space CA')]
    pd.DataFrame(rows).to_csv(CSV,index=False,float_format='%.15f')
    make_plot(rows); display(Image(filename=str(PNG)))
    print('RESULTS')
    for r in rows: print(f"{r['method']} | CA {r['utc']} | A′B′ {r['apbp_as']:.12f} arcsec | {r['apbp_km']:,.9f} km | AB {r['ab_as']:.12f} arcsec | {r['ab_km']:,.9f} km")
    print('DELTA V0089 - V0067')
    print(f"A′B′: {rows[1]['apbp_as']-rows[0]['apbp_as']:+.12f} arcsec | {rows[1]['apbp_km']-rows[0]['apbp_km']:+,.9f} km")
    print(f"AB:   {rows[1]['ab_as']-rows[0]['ab_as']:+.12f} arcsec | {rows[1]['ab_km']-rows[0]['ab_km']:+,.9f} km")
    print('CAUSE: V0089 used direct JD-space geocentric closest-approach minimization with a different bracket/tolerance. V0067 seconds-space CA is the reference path.')
    print(f'CSV: {CSV}'); print(datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S %z')); print(VERSION)

if __name__ == '__main__': main()
# V0091
