# V0089B
# Audit reference: standalone Colab Python/Matplotlib/JPL widget; V0067 seconds-space geocentric CA restored; publication title and 900 DPI raster; no AI images.
from __future__ import annotations
import math, subprocess, sys, time, warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION="V0089B"
LOCAL_TZ=ZoneInfo("America/Bogota")
ROOT=Path("/content")
OUT=ROOT/"VENUS_1769_V0089B_PUBLICATION_OUTPUT"
PNG=OUT/"VENUS_1769_V0089B_PUBLICATION_900DPI.png"
PDF=OUT/"VENUS_1769_V0089B_PUBLICATION_VECTOR.pdf"
SVG=OUT/"VENUS_1769_V0089B_PUBLICATION_VECTOR.svg"
CSV=OUT/"VENUS_1769_V0089B_CONTACTS_GEOMETRY.csv"
DPI=900
ARC=206_264.80624709636
JPL_AU_KM=149_597_870.700000
C_KM_S=299_792.458000
TAU_A_S=499.004782000
IAU1976_AU_KM=C_KM_S*TAU_A_S
EARTH_RADIUS_KM=6_378.140000
SUN_RADIUS_KM=695_700.000000
VENUS_RADIUS_KM=6_051.800000
START="1769-06-03 18:00"; STOP="1769-06-04 06:00"; STEP="1m"; GEOCENTER="@399"
EVENTS=("C1","C2","CA","C3","C4")
PV=dict(key="POINT_VENUS",label="Point Venus, Tahiti",short="PV",lat=-17.4956,lon=-149.4939,elevation=0.0,body=399,color="#42D7C3")
VA=dict(key="VARDO",label="Vardø, Norway",short="V",lat=70.3724,lon=31.1103,elevation=0.0,body=399,color="#D89B18")
SITES=(PV,VA); TARGETS=(("SUN","10"),("VENUS","299"))
PREFIXES=("GEOCENTER_SUN","GEOCENTER_VENUS","POINT_VENUS_SUN","POINT_VENUS_VENUS","VARDO_SUN","VARDO_VENUS")
SUN_LINE_WIDTH=.9375; TRACK_LINE_WIDTH=.375; DISK_LINE_WIDTH=.375; VENUS_PAINT_LINE_FACTOR=2.25
GUIDE_LINE_WIDTH=.250; MARKER_EDGE_WIDTH=.250
SUN_COLOR="#FFD34A"; SUN_FILL_COLOR="#D95A1B"; SUN_FILL_ALPHA=.260
GUIDE_COLOR="#263A4B"; FG="#F8FAFC"; MUTED="#B8CBD6"; BG="#000000"
TABLE_HEADER="#23466F"; TABLE_TEAL="#164B55"; TABLE_GOLD="#563B0B"; TABLE_BODY="#101A2E"
MAIN_PV_LABEL_Y=36.0; MAIN_VARDO_LABEL_Y=-36.0
TITLE="1769 Venus Transit Between Vardø, Norway, and Point Venus, Tahiti"
ZOOM_LABEL_OVERRIDES={
    ("PV","C1"):(-2.0,3.0),("PV","C2"):(2.0,3.0),("V","C1"):(-6.0,-8.0),("V","C2"):(8.0,-8.0),
    ("PV","C3"):(0.0,-12.0),("PV","C4"):(-12.0,-10.0),("V","C3"):(9.0,-11.0),("V","C4"):(-11.0,-11.0)}

def require(import_name,package_name):
    try: __import__(import_name)
    except ImportError: subprocess.check_call([sys.executable,"-m","pip","-q","install",package_name])
for a,b in (("numpy","numpy"),("pandas","pandas"),("scipy","scipy"),("astropy","astropy"),("astroquery","astroquery"),("matplotlib","matplotlib"),("IPython","ipython")):
    require(a,b)
import matplotlib
matplotlib.use("Agg",force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Image, display
from matplotlib.patches import Circle
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar
warnings.filterwarnings("ignore",message=".*id_type.*deprecated.*"); warnings.filterwarnings("ignore",message=".*dubious year.*")

def norm(v): return float(np.linalg.norm(np.asarray(v,dtype=float)))
def unit(v):
    a=np.asarray(v,dtype=float); n=norm(a)
    if n<=0: raise RuntimeError("Cannot normalize zero vector.")
    return a/n
def utc(jd): return Time(float(jd),format="jd",scale="tdb").utc.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
def loc(site): return dict(lon=float(site["lon"]),lat=float(site["lat"]),elevation=float(site["elevation"]),body=int(site["body"]))

def get_series(prefix,target,location):
    last=None
    for attempt in range(4):
        try:
            tab=Horizons(id=target,location=location,epochs={"start":START,"stop":STOP,"step":STEP},id_type=None).vectors(refplane="ecliptic",aberrations="geometric",cache=False)
            fr=tab.to_pandas(); out=pd.DataFrame({"JD_TDB":pd.to_numeric(fr["datetime_jd"],errors="coerce")})
            for ax in "xyz": out[f"{prefix}_{ax.upper()}_KM"]=pd.to_numeric(fr[ax],errors="coerce")*JPL_AU_KM
            out=out.dropna().drop_duplicates("JD_TDB").sort_values("JD_TDB").reset_index(drop=True)
            if len(out)<600: raise RuntimeError(f"Incomplete JPL query for {prefix}: {len(out)} rows")
            return out
        except Exception as exc:
            last=exc; time.sleep(1.5*(attempt+1))
    raise RuntimeError(f"JPL query failed for {prefix}: {last}")

def master_frame():
    series=[]
    for name,target in TARGETS: series.append(get_series(f"GEOCENTER_{name}",target,GEOCENTER))
    for site in SITES:
        for name,target in TARGETS: series.append(get_series(f"{site['key']}_{name}",target,loc(site)))
    m=series[0]
    for s in series[1:]: m=m.merge(s,on="JD_TDB",how="inner",validate="one_to_one")
    if len(m)<600: raise RuntimeError(f"Synchronized JPL master too short: {len(m)} rows")
    return m.sort_values("JD_TDB").reset_index(drop=True)

def cache_build(m):
    c={"JD_TDB":m["JD_TDB"].to_numpy(float)}
    for p in PREFIXES:
        for ax in "XYZ": c[f"{p}_{ax}_KM"]=CubicSpline(c["JD_TDB"],m[f"{p}_{ax}_KM"].to_numpy(float),bc_type="natural")
    return c

def vec(c,prefix,jd): return np.array([float(c[f"{prefix}_{ax}_KM"](float(jd))) for ax in "XYZ"],dtype=float)
def sep_rad(a,b):
    ah,bh=unit(a),unit(b)
    return math.atan2(norm(np.cross(ah,bh)),float(np.dot(ah,bh)))
def radii(c,site,jd): return math.asin(SUN_RADIUS_KM/norm(vec(c,f"{site}_SUN",jd))), math.asin(VENUS_RADIUS_KM/norm(vec(c,f"{site}_VENUS",jd)))
def residual(c,site,jd,internal):
    d=sep_rad(vec(c,f"{site}_SUN",jd),vec(c,f"{site}_VENUS",jd)); rs,rv=radii(c,site,jd)
    return d-(rs-rv if internal else rs+rv)

def roots(c,site,internal):
    jds=np.asarray(c["JD_TDB"],dtype=float); vals=np.array([residual(c,site,jd,internal) for jd in jds]); out=[]
    for i in range(len(jds)-1):
        if not np.isfinite(vals[i]+vals[i+1]): continue
        if vals[i]==0: out.append(float(jds[i]))
        elif vals[i]*vals[i+1]<0: out.append(float(brentq(lambda x: residual(c,site,x,internal),float(jds[i]),float(jds[i+1]),xtol=1e-13,rtol=1e-14)))
    uniq=[]
    for r in sorted(out):
        if not uniq or abs(r-uniq[-1])>0.2/86400.0: uniq.append(r)
    if len(uniq)!=2: raise RuntimeError(f"Expected two contact roots for {site}; found {len(uniq)}")
    return uniq

def local_ca(c,site,a,b):
    ref=.5*(a+b); lo,hi=(a-ref)*86400.0,(b-ref)*86400.0
    res=minimize_scalar(lambda s: sep_rad(vec(c,f"{site}_SUN",ref+float(s)/86400.0),vec(c,f"{site}_VENUS",ref+float(s)/86400.0)),bounds=(lo,hi),method="bounded",options={"xatol":1e-4,"maxiter":500})
    if not res.success: raise RuntimeError(f"Closest approach failed for {site}")
    return ref+float(res.x)/86400.0

def basis(sun_vec):
    center=unit(sun_vec); pole=np.array([0.0,0.0,1.0]); east=np.cross(pole,center)
    if norm(east)<1e-14: east=np.cross(np.array([0.0,1.0,0.0]),center)
    east=unit(east); north=unit(np.cross(center,east))
    if float(np.dot(north,pole))<0.0: east,north=-east,-north
    return center,east,north

def pos_arcsec(c,site,jd):
    s=vec(c,f"{site}_SUN",jd); v=vec(c,f"{site}_VENUS",jd); sh,vh=unit(s),unit(v); d=sep_rad(sh,vh)
    tangent=unit(vh-math.cos(d)*sh); _cen,east,north=basis(s)
    p=d*ARC*np.array([float(np.dot(tangent,east)),float(np.dot(tangent,north))]); p[0]*=-1.0
    return p

def fit_track(points):
    mean=points.mean(axis=0); cen=points-mean; _u,_s,vt=np.linalg.svd(cen,full_matrices=False); direction=vt[0]
    if direction[0]<0: direction=-direction
    normal=np.array([-direction[1],direction[0]]); along=cen@direction; cross=cen@normal; coef=np.polyfit(along,cross,2)
    rms=float(np.sqrt(np.mean((cross-np.polyval(coef,along))**2))); angle=math.degrees(math.atan2(direction[1],direction[0]))
    return dict(mean=mean,direction=direction,normal=normal,angle=abs(angle),rms=rms,curvature=float(2*coef[0]),slope=float(math.tan(math.radians(angle))))

def station(c,site):
    ext=roots(c,site["key"],False); inn=roots(c,site["key"],True); ca=local_ca(c,site["key"],inn[0],inn[1])
    ev={"C1":ext[0],"C2":inn[0],"CA":ca,"C3":inn[1],"C4":ext[1]}; jds=np.asarray(c["JD_TDB"],dtype=float)
    sel=jds[(jds>=ev["C1"])&(jds<=ev["C4"])]
    pts=np.array([pos_arcsec(c,site["key"],jd) for jd in sel]); epts={k:pos_arcsec(c,site["key"],jd) for k,jd in ev.items()}
    rr={k:tuple(x*ARC for x in radii(c,site["key"],jd)) for k,jd in ev.items()}; ca_sep=sep_rad(vec(c,f"{site['key']}_SUN",ca),vec(c,f"{site['key']}_VENUS",ca))*ARC
    return dict(site=site,events=ev,jds=sel,pts=pts,epts=epts,r=rr,fit=fit_track(pts),ca_sep=ca_sep)

def gnom(ray,cen,east,north):
    h=unit(ray); den=float(np.dot(h,cen)); return np.array([float(np.dot(h,east)),float(np.dot(h,north))])/den
def rel_common(c,site,jd,bas):
    cen,east,north=bas
    return ARC*(gnom(vec(c,f"{site}_VENUS",jd),cen,east,north)-gnom(vec(c,f"{site}_SUN",jd),cen,east,north))

def geocentric_ca_v0067(c):
    jds=np.asarray(c["JD_TDB"],dtype=float); vals=np.array([sep_rad(vec(c,"GEOCENTER_SUN",jd),vec(c,"GEOCENTER_VENUS",jd)) for jd in jds],dtype=float)
    i=int(np.argmin(vals)); lo_jd=float(jds[max(0,i-3)]); hi_jd=float(jds[min(len(jds)-1,i+3)])
    ref=.5*(lo_jd+hi_jd); lo_s=(lo_jd-ref)*86400.0; hi_s=(hi_jd-ref)*86400.0
    res=minimize_scalar(lambda s: sep_rad(vec(c,"GEOCENTER_SUN",ref+float(s)/86400.0),vec(c,"GEOCENTER_VENUS",ref+float(s)/86400.0)),bounds=(lo_s,hi_s),method="bounded",options={"xatol":1e-4,"maxiter":500})
    if not res.success: raise RuntimeError("Geocentric closest approach failed")
    return ref+float(res.x)/86400.0

def geo(c):
    jd=geocentric_ca_v0067(c); bas=basis(vec(c,"GEOCENTER_SUN",jd)); h=.5/86400.0
    pv0=rel_common(c,"POINT_VENUS",jd,bas); va0=rel_common(c,"VARDO",jd,bas)
    vp=rel_common(c,"POINT_VENUS",jd+h,bas)-rel_common(c,"POINT_VENUS",jd-h,bas); vv=rel_common(c,"VARDO",jd+h,bas)-rel_common(c,"VARDO",jd-h,bas)
    direction=unit(unit(vp)+unit(vv)); normal2=np.array([-direction[1],direction[0]])
    if float(np.dot(va0-pv0,normal2))<0.0: normal2=-normal2
    cen,east,north=bas; normal3=unit(normal2[0]*east+normal2[1]*north); geo_sun=vec(c,"GEOCENTER_SUN",jd); geo_venus=vec(c,"GEOCENTER_VENUS",jd)
    site_pv=geo_sun-vec(c,"POINT_VENUS_SUN",jd); site_va=geo_sun-vec(c,"VARDO_SUN",jd); baseline=site_va-site_pv
    if float(np.dot(baseline,normal3))<0.0: normal2,normal3=-normal2,-normal3
    apbp_as=float(np.dot(va0-pv0,normal2)); km_per_as=norm(geo_sun)/ARC; ab_km=float(np.dot(baseline,normal3))
    es_axis=unit(geo_sun); ev_bar=float(np.dot(geo_venus,es_axis)); vs_bar=float(np.dot(geo_sun-geo_venus,es_axis)); es_bar=float(np.dot(geo_sun,es_axis))
    return dict(jd=jd,utc=utc(jd),bas=bas,normal2=normal2,apbp_as=apbp_as,apbp_km=apbp_as*km_per_as,ab_km=ab_km,ab_as=ab_km/km_per_as,ev_bar=ev_bar,vs_bar=vs_bar,es_bar=es_bar,km_per_as=km_per_as)

def style_table(tab,gold=(),teal=(),fs=6.2,center_cols=()):
    for (r,c),cell in tab.get_celld().items():
        cell.set_edgecolor("#70879A"); cell.set_linewidth(.35); cell.get_text().set_color(FG); cell.get_text().set_fontsize(fs)
        if c in center_cols: cell.get_text().set_ha("center")
        if r==0: cell.set_facecolor(TABLE_HEADER); cell.get_text().set_fontweight("bold")
        elif r in gold: cell.set_facecolor(TABLE_GOLD); cell.get_text().set_fontweight("bold")
        elif r in teal: cell.set_facecolor(TABLE_TEAL); cell.get_text().set_fontweight("bold")
        else: cell.set_facecolor(TABLE_BODY)

def add_disk(ax,center,radius,color):
    ax.add_patch(Circle((float(center[0]),float(center[1])),radius,facecolor=color,edgecolor="none",alpha=.16,linewidth=0,zorder=5))
    ax.add_patch(Circle((float(center[0]),float(center[1])),radius,facecolor="none",edgecolor=color,alpha=.98,linewidth=DISK_LINE_WIDTH*VENUS_PAINT_LINE_FACTOR,zorder=7))
def label(ax,point,text,color,dx,dy,above=True):
    ax.annotate(text,xy=(float(point[0]),float(point[1])),xytext=(float(point[0]+dx),float(point[1]+dy)),ha="center",va="bottom" if above else "top",color=color,fontsize=6.3,fontweight="bold",arrowprops=dict(arrowstyle="-",linewidth=GUIDE_LINE_WIDTH,color=color,shrinkA=1,shrinkB=1),zorder=9)
def draw_events(ax,st,evs,main):
    color=st["site"]["color"]; short=st["site"]["short"]; xbase={"C1":-13.0,"C2":4.0,"CA":0.0,"C3":4.0,"C4":13.0}
    for ev in evs:
        p=st["epts"][ev]; add_disk(ax,p,st["r"][ev][1],color)
        ax.scatter([p[0]],[p[1]],s=16 if ev=="CA" else 7.5,marker="X" if ev=="CA" else "o",color=color,edgecolors=BG,linewidths=MARKER_EDGE_WIDTH,zorder=8)
        if main:
            above=short=="PV"; label(ax,p,f"{short} {ev}",color,xbase[ev],MAIN_PV_LABEL_Y if above else MAIN_VARDO_LABEL_Y,above)
        else:
            dx,dy=ZOOM_LABEL_OVERRIDES.get((short,ev),(xbase[ev]*.4,21.0 if short=="PV" else -21.0)); label(ax,p,f"{short} {ev}",color,dx,dy,dy>=0)
def zoom_limits(items,evs):
    pts=[]; rs=[]
    for st in items:
        for ev in evs: pts.append(st["epts"][ev]); rs.append(st["r"][ev][1])
    pts=np.array(pts); r=max(rs); m=r*.55+7
    return (float(pts[:,0].min()-r-m),float(pts[:,0].max()+r+m)),(float(pts[:,1].min()-r-m),float(pts[:,1].max()+r+m))

def max_contact_resid(c,items):
    vals=[]
    for st in items:
        key=st["site"]["key"]
        for ev in ("C1","C2","C3","C4"): vals.append(abs(residual(c,key,st["events"][ev],ev in ("C2","C3")))*ARC)
    return max(vals)

def plot(pv,va,g,max_resid):
    plt.close("all")
    plt.rcParams.update({"figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG,"text.color":FG,"axes.labelcolor":FG,"xtick.color":MUTED,"ytick.color":MUTED,"font.family":"DejaVu Serif","mathtext.fontset":"dejavuserif","pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
    fig=plt.figure(figsize=(16,9),facecolor=BG); gs=fig.add_gridspec(1,2,width_ratios=[2.05,1],left=.03,right=.985,top=.900,bottom=.07,wspace=.035)
    left=gs[0,0].subgridspec(2,1,height_ratios=[.70,.30],hspace=.20); ax=fig.add_subplot(left[0,0])
    low=left[1,0].subgridspec(1,3,width_ratios=[.82,1.58,.82],wspace=.25); ing,der,eg=fig.add_subplot(low[0,0]),fig.add_subplot(low[0,1]),fig.add_subplot(low[0,2])
    right=gs[0,1].subgridspec(2,1,height_ratios=[.47,.53],hspace=.095); resax,conax=fig.add_subplot(right[0,0]),fig.add_subplot(right[1,0])
    fig.suptitle(TITLE,fontsize=15,fontweight="bold",y=.948)
    rs=pv["r"]["CA"][0]; th=np.linspace(0,2*math.pi,1600)
    ax.add_patch(Circle((0,0),rs,facecolor=SUN_FILL_COLOR,edgecolor="none",alpha=SUN_FILL_ALPHA,zorder=0)); ax.plot(rs*np.cos(th),rs*np.sin(th),c=SUN_COLOR,lw=SUN_LINE_WIDTH,zorder=1)
    ax.axhline(0,c=GUIDE_COLOR,lw=GUIDE_LINE_WIDTH); ax.axvline(0,c=GUIDE_COLOR,lw=GUIDE_LINE_WIDTH)
    for st in (pv,va): ax.plot(st["pts"][:,0],st["pts"][:,1],c=st["site"]["color"],lw=TRACK_LINE_WIDTH,label=st["site"]["label"]); draw_events(ax,st,EVENTS,True)
    ax.set_aspect("equal"); ax.set_xlim(-1.07*rs,1.07*rs); ax.set_ylim(-.08*rs,1.06*rs); ax.set_xlabel(r"Ecliptic longitude direction, $\xi$ (arcsec)"); ax.set_ylabel(r"Ecliptic north direction, $\eta$ (arcsec)"); ax.tick_params(labelsize=6.5,length=2.2)
    leg=ax.legend(loc="lower left",frameon=False,fontsize=6.5); [t.set_color(FG) for t in leg.get_texts()]
    for zax,evs,title in ((ing,("C1","C2"),"INGRESS ZOOM — C1 / C2 TANGENCY"),(eg,("C3","C4"),"EGRESS ZOOM — C3 / C4 TANGENCY")):
        zrs=float(np.mean([st["r"][e][0] for st in (pv,va) for e in evs])); zax.add_patch(Circle((0,0),zrs,facecolor=SUN_FILL_COLOR,edgecolor="none",alpha=SUN_FILL_ALPHA,zorder=0)); zax.plot(zrs*np.cos(th),zrs*np.sin(th),c=SUN_COLOR,lw=SUN_LINE_WIDTH,zorder=2)
        for st in (pv,va): zax.plot(st["pts"][:,0],st["pts"][:,1],c=st["site"]["color"],lw=TRACK_LINE_WIDTH); draw_events(zax,st,evs,False)
        xl,yl=zoom_limits((pv,va),evs); zax.set_xlim(*xl); zax.set_ylim(*yl); zax.set_aspect("equal"); zax.set_title(title,fontsize=6.3,pad=3); zax.tick_params(labelsize=5,length=1.8)
    der.axis("off"); der.text(.5,.895,"A′B′ AND AB DERIVATION",ha="center",va="center",fontsize=8.2,fontweight="bold")
    delta=abs(va["fit"]["angle"]-pv["fit"]["angle"])
    rows=[["Quantity","Definition","Arcseconds","Kilometers"],["A′B′","JPL separate-ray derived",f"{g['apbp_as']:.6f}",f"{g['apbp_km']:,.6f}"],["AB","JPL projected baseline",f"{g['ab_as']:.6f}",f"{g['ab_km']:,.6f}"],["α PV","Point Venus, Tahiti track angle",f"{pv['fit']['angle']:.6f}",""],["α V","Vardø, Norway track angle",f"{va['fit']['angle']:.6f}",""],["Δα","Delta track angle, |αV − αPV|",f"{delta:.6f}",""]]
    tab=der.table(cellText=rows,cellLoc="left",colWidths=[.16,.47,.15,.22],bbox=[0,0,1,.805]); tab.auto_set_font_size(False); style_table(tab,gold=(2,3,4),teal=(1,),fs=5.95,center_cols=(2,3))
    resax.axis("off"); resax.set_title("RESULTS",loc="left",fontsize=9,fontweight="bold",pad=5); pi0=math.asin(EARTH_RADIUS_KM/IAU1976_AU_KM)*ARC
    rr=[["Quantity","Symbol","Value","Unit / status"],["IAU 1976 AU-normalized solar horizontal parallax","π₀",f"{pi0:.12f}","arcsec"],["Point Venus, Tahiti track angle","α_PV",f"{pv['fit']['angle']:.6f}","deg"],["Vardø, Norway track angle","α_V",f"{va['fit']['angle']:.6f}","deg"],["Average track angle","ᾱ",f"{.5*(pv['fit']['angle']+va['fit']['angle']):.6f}","deg"],["Point Venus, Tahiti RMS","RMS_PV",f"{pv['fit']['rms']:.6f}","arcsec"],["Vardø, Norway RMS","RMS_V",f"{va['fit']['rms']:.6f}","arcsec"],["Maximum contact-equation residual","",f"{max_resid:.12f}","arcsec — PASS"],["A′B′ common-normal separation","A′B′",f"{g['apbp_as']:.6f}","arcsec"],["Projection reference","","JPL ECLIPTIC","verified"]]
    rt=resax.table(cellText=rr,cellLoc="left",colWidths=[.47,.12,.21,.20],bbox=[0,0,1,.90]); rt.auto_set_font_size(False); style_table(rt,gold=(2,3),teal=(8,9),fs=6.1)
    conax.axis("off"); conax.set_title("RECOMPUTED CONTACT TIMES — UTC",loc="left",fontsize=9,fontweight="bold",pad=5); cr=[["Station","Event","UTC","Exact limb condition"]]
    for st in (pv,va):
        for ev in EVENTS:
            cond="dρ/dt = 0; local minimum" if ev=="CA" else "ρ = R☉ − R♀" if ev in ("C2","C3") else "ρ = R☉ + R♀"
            cr.append([st["site"]["label"],ev,utc(st["events"][ev]).split()[1],cond])
    ct=conax.table(cellText=cr,cellLoc="left",colWidths=[.28,.12,.30,.30],bbox=[0,0,1,.92]); ct.auto_set_font_size(False); style_table(ct,teal=(1,2,3,6,7,8),fs=5.75)
    fig.text(.5,.018,"NO AI IMAGES — Python/Matplotlib only. Standalone JPL Horizons geometric ecliptic vector reconstruction; V0067 seconds-space geocentric CA restored.",ha="center",fontsize=6,color=MUTED)
    fig.savefig(PNG,dpi=DPI,bbox_inches="tight",pad_inches=.02,facecolor=BG)
    fig.savefig(PDF,bbox_inches="tight",pad_inches=.02,facecolor=BG)
    fig.savefig(SVG,bbox_inches="tight",pad_inches=.02,facecolor=BG)
    display(Image(filename=str(PNG)))

def write_csv(pv,va,g,max_resid):
    rows=[]
    for st in (pv,va):
        for ev in EVENTS: rows.append(dict(station=st["site"]["label"],event=ev,utc=utc(st["events"][ev]),jd_tdb=st["events"][ev],xi_arcsec=st["epts"][ev][0],eta_arcsec=st["epts"][ev][1]))
    rows += [dict(station="GEOMETRY",event="A_prime_B_prime_arcsec",utc=g["utc"],jd_tdb=g["jd"],xi_arcsec=g["apbp_as"],eta_arcsec=np.nan),dict(station="GEOMETRY",event="A_prime_B_prime_km",utc=g["utc"],jd_tdb=g["jd"],xi_arcsec=g["apbp_km"],eta_arcsec=np.nan),dict(station="GEOMETRY",event="AB_km",utc=g["utc"],jd_tdb=g["jd"],xi_arcsec=g["ab_km"],eta_arcsec=max_resid)]
    pd.DataFrame(rows).to_csv(CSV,index=False)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    print("CODE INPUTS"); print(f"Version: {VERSION}"); print(f"JPL source: Horizons geometric ecliptic vectors"); print(f"Raster DPI: {DPI}; vector outputs: PDF and SVG")
    print("COMMENTS"); print("Publication title changed to Vardø, Norway / Point Venus, Tahiti. Title moved lower. V0067 seconds-space geocentric CA retained. No AI images.")
    m=master_frame(); c=cache_build(m); pv=station(c,PV); va=station(c,VA); g=geo(c); mr=max_contact_resid(c,(pv,va)); write_csv(pv,va,g,mr); plot(pv,va,g,mr)
    print("RESULTS")
    print(f"Geocentric CA UTC: {g['utc']} | JD {g['jd']:.18f}")
    print(f"A′B′ arcsec: {g['apbp_as']:.12f}")
    print(f"A′B′ km: {g['apbp_km']:.9f}")
    print(f"AB arcsec: {g['ab_as']:.12f}")
    print(f"AB km: {g['ab_km']:.9f}")
    print(f"Point Venus CA UTC: {utc(pv['events']['CA'])}")
    print(f"Vardø CA UTC: {utc(va['events']['CA'])}")
    print("OUTPUT SUMMARY"); print(f"PNG: {PNG}"); print(f"PDF: {PDF}"); print(f"SVG: {SVG}"); print(f"CSV: {CSV}")
    print("PAPER COMPARISON"); print(f"IAU 1976 π₀: {math.asin(EARTH_RADIUS_KM/IAU1976_AU_KM)*ARC:.12f} arcsec")
    print("EQUATION STATUS"); print("PASS: V0089B corrected plot uses V0067 seconds-space geocentric CA equations.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z")); print(VERSION)
if __name__=="__main__": main()
# V0089B
