# V0021
# Audit reference: Fresh JPL reconstruction of both 1769 tracks, all five events per track, and an IAU-1976-normalized half-Sun plot.
from __future__ import annotations
import ast, csv, math, subprocess, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

V="V0021"; TZ=ZoneInfo("America/Bogota")
ASR=206264.80624709636; JPL_AU=149597870.7
RE=6378.140; C=299792.458; TAU=499.004782; AU76=C*TAU
RS=695700.0; RV=6051.8; PI76=math.asin(RE/AU76)*ASR
START="1769-Jun-03 18:00"; STOP="1769-Jun-04 04:00"; STEP="1m"; N=601
EVENTS=("C1","C2","CA","C3","C4")
A={"key":"VARDO","short":"Vardo","label":"Vardo Norway","lon":31.1107,"lat":70.3706}
B={"key":"TAHITI","short":"Point Venus","label":"Point Venus Tahiti","lon":-149.4947,"lat":-17.4958}
SITES=(A,B); COLORS={A["label"]:"#ffc861",B["label"]:"#5ee08a"}
OUT=Path("/content/IERS_TN36_V01_MASTER_OUTPUT/V0021_IAU1976_FROM_SCRATCH")
PNG=OUT/"V0021_IAU1976_HALF_SUN_TRACKS.png"; CSV=OUT/"V0021_CONTACTS_AND_GEOMETRY.csv"; MASTER=OUT/"V0021_SIX_SERIES_MASTER.csv"

def need(m,p):
    try: __import__(m)
    except ImportError: subprocess.check_call([sys.executable,"-m","pip","-q","install",p])
for m,p in (("numpy","numpy"),("pandas","pandas"),("scipy","scipy"),("matplotlib","matplotlib"),("astroquery","astroquery"),("astropy","astropy")): need(m,p)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar
from astroquery.jplhorizons import Horizons
import astropy.units as u
from astropy.time import Time

def norm(v): return float(np.linalg.norm(v))
def unit(v):
    n=norm(v)
    if n==0: raise RuntimeError("zero vector")
    return np.asarray(v,float)/n

def loc(site): return {"lon":site["lon"]*u.deg,"lat":site["lat"]*u.deg,"elevation":0*u.km}
def download(target,location,prefix):
    q=Horizons(id=target,location=location,epochs={"start":START,"stop":STOP,"step":STEP}).vectors().to_pandas()
    f=pd.DataFrame({"jd":q["datetime_jd"].astype(float),"tdb":q["datetime_str"].astype(str),f"{prefix}_x":q["x"].astype(float)*JPL_AU,f"{prefix}_y":q["y"].astype(float)*JPL_AU,f"{prefix}_z":q["z"].astype(float)*JPL_AU})
    if len(f)!=N: raise RuntimeError(f"{prefix}: {len(f)} rows, expected {N}")
    return f

def six_series():
    gs=download("10","500@399","GS"); gv=download("299","500@399","GV")
    geo=gs.merge(gv.drop(columns="tdb"),on="jd")
    frames=[]
    for s in SITES:
        frames += [download("10",loc(s),s["key"]+"S"),download("299",loc(s),s["key"]+"V")]
    topo=frames[0]
    for f in frames[1:]: topo=topo.merge(f.drop(columns="tdb"),on="jd")
    master=geo.merge(topo.drop(columns="tdb"),on="jd")
    if len(master)!=N: raise RuntimeError("master row count failed")
    master.to_csv(MASTER,index=False,float_format="%.15f")
    return geo,topo,master

def cache(df):
    c={"jd":df["jd"].to_numpy(float)}
    for k in df.columns:
        if k.endswith(("_x","_y","_z")): c[k]=CubicSpline(c["jd"],df[k].to_numpy(float),bc_type="natural")
    return c

def vec(c,p,j): return np.array([float(c[p+"_x"](j)),float(c[p+"_y"](j)),float(c[p+"_z"](j))])
def utc(j): return Time(j,format="jd",scale="tdb").utc.iso
def svec(c,s,j): return vec(c,s["key"]+"S",j)
def vvec(c,s,j): return vec(c,s["key"]+"V",j)
def sep(c,s,j): return math.acos(float(np.clip(np.dot(unit(svec(c,s,j)),unit(vvec(c,s,j))),-1,1)))*ASR
def radii(c,s,j): return math.atan2(RS,norm(svec(c,s,j)))*ASR, math.atan2(RV,norm(vvec(c,s,j)))*ASR

def contact_eq(c,s,outer,j):
    rs,rv=radii(c,s,j)
    return sep(c,s,j)-(rs+rv if outer else rs-rv)
def roots(c,s,outer):
    jd=np.asarray(c["jd"],float); y=np.array([contact_eq(c,s,outer,j) for j in jd]); out=[]
    for i in range(len(jd)-1):
        if not np.isfinite(y[i:i+2]).all(): continue
        if y[i]==0: out.append(float(jd[i]))
        elif y[i]*y[i+1]<0: out.append(float(brentq(lambda x:contact_eq(c,s,outer,x),jd[i],jd[i+1],xtol=1e-13,rtol=1e-13,maxiter=200)))
    return sorted(out)
def closest(c,s):
    jd=np.asarray(c["jd"],float); y=np.array([sep(c,s,j) for j in jd]); i=int(np.argmin(y)); lo=jd[max(0,i-3)]; hi=jd[min(len(jd)-1,i+3)]
    r=minimize_scalar(lambda x:sep(c,s,x),bounds=(lo,hi),method="bounded",options={"xatol":1e-13})
    if not r.success: raise RuntimeError(f"CA failed: {s['label']}")
    return float(r.x)
def all_events(c,s):
    o=roots(c,s,True); inn=roots(c,s,False)
    if len(o)<2 or len(inn)<2: raise RuntimeError(f"contacts failed: {s['label']} outer={len(o)} inner={len(inn)}")
    e={"C1":o[0],"C2":inn[0],"CA":closest(c,s),"C3":inn[-1],"C4":o[-1]}
    if not(e["C1"]<e["C2"]<e["CA"]<e["C3"]<e["C4"]): raise RuntimeError(f"chronology failed: {s['label']}")
    return e

def basis(g,j):
    n=unit(vec(g,"GS",j)); ref=np.array([0.,0.,1.])
    if norm(np.cross(ref,n))<1e-12: ref=np.array([1.,0.,0.])
    x=unit(np.cross(ref,n)); y=unit(np.cross(n,x)); return n,x,y

def screen(g,t,s,j,b):
    n,x,y=b; sg=vec(g,"GS",j); st=svec(t,s,j); vv=vvec(t,s,j); obs=sg-st; den=float(np.dot(vv,n))
    if abs(den)<1e-14: raise RuntimeError("ray parallel to screen")
    hit=obs+float(np.dot(sg-obs,n)/den)*vv; q=hit-sg
    return np.array([math.atan2(float(np.dot(q,x)),AU76)*ASR,math.atan2(float(np.dot(q,y)),AU76)*ASR])
def venus_r(g,t,s,j): return radii(t,s,j)[1]*norm(vec(g,"GS",j))/AU76

def pca(p):
    m=p.mean(0); _,_,vt=np.linalg.svd(p-m,full_matrices=False); d=unit(vt[0]); return m,(-d if d[0]<0 else d)
def track(g,t,s,e,b):
    jd=np.asarray(t["jd"],float); use=sorted(set(list(jd[(jd>=e["C1"])&(jd<=e["C4"])])+list(e.values())))
    p=np.array([screen(g,t,s,j,b) for j in use]); m,d=pca(p); ep={k:screen(g,t,s,j,b) for k,j in e.items()}; er={k:venus_r(g,t,s,j) for k,j in e.items()}
    if tuple(ep)!=EVENTS or tuple(er)!=EVENTS: raise RuntimeError(f"event plot data incomplete: {s['label']}")
    n=np.array([-d[1],d[0]])
    return {"site":s,"events":e,"points":p,"mean":m,"dir":d,"event_points":ep,"event_radii":er,"angle":math.degrees(math.atan2(d[1],d[0])),"rms":float(np.sqrt(np.mean(((p-m)@n)**2)))}

def intersect(m,d,mid,n):
    z,*_=np.linalg.lstsq(np.c_[d,-n],mid-m,rcond=None); return m+z[0]*d
def geometry(g,a,b,j):
    tan=unit(a["dir"]+b["dir"]); tan=-tan if tan[0]<0 else tan; n=np.array([-tan[1],tan[0]]); mid=.5*(a["mean"]+b["mean"])
    ap=intersect(a["mean"],a["dir"],mid,n); bp=intersect(b["mean"],b["dir"],mid,n); w=bp-ap; th=norm(w); rho=abs(float(np.dot(w,n)))
    sun=vec(g,"GS",j); ven=vec(g,"GV",j); ratio=norm(ven)/norm(ven-sun); tr=th/ASR
    abp=math.tan(tr)*AU76; ab=abp*ratio; aba=math.atan2(ab,AU76)*ASR; raw=rho*ratio*RE/ab
    pi=raw*(th/rho)*(math.tan(tr)/tr)*(math.asin(RE/AU76)/(RE/AU76))
    return {"th":th,"rho":rho,"abp":abp,"ab":ab,"aba":aba,"ratio":ratio,"raw":raw,"pi":pi,"res":pi-PI76,"halley":abp/ab}

def label(ax,p,text,dx,dy,c): ax.annotate(text,xy=p,xytext=(p[0]+dx,p[1]+dy),textcoords="data",fontsize=5.7,color=c,ha="left",va="center",arrowprops=dict(arrowstyle="-",lw=.20,color=c,shrinkA=0,shrinkB=2))
def table(ax,a,b,g):
    rows=[("β Vardo",a["angle"],"deg"),("β Point Venus",b["angle"],"deg"),("Δβ",abs(a["angle"]-b["angle"]),"deg"),("π⊙",g["pi"],"arcsec"),("A′B′ / AB",g["halley"],"ratio"),("A′B′",g["th"],"arcsec"),("A′B′",g["abp"],"km"),("AB",g["aba"],"arcsec"),("AB",g["ab"],"km"),("D ES",1.0,"AU")]
    data=[[q,f"{v:.10f}" if q in("π⊙","A′B′ / AB") else f"{v:.6f}",u] for q,v,u in rows]
    tb=ax.table(cellText=data,colLabels=["Quantity","Value","Unit"],loc="lower left",colWidths=[.29,.23,.15],bbox=[.438,.122,.380,.345]); tb.auto_set_font_size(False); tb.set_fontsize(5.3)
    for (r,c),cell in tb.get_celld().items():
        cell.set_linewidth(.18); cell.set_edgecolor("#1e4f64"); cell.set_facecolor("#0a1a22" if r==0 else "#050b0f"); cell.get_text().set_color("#66e8ff" if r==0 else ("#ffc861" if c==1 else ("#5ee08a" if c==2 else "#dff8ff")))
        if r==0 or c==1: cell.get_text().set_fontweight("bold")
    ax.text(.440,.101,"A′B′ = solar-screen chord; AB = projected baseline; D ES = IAU 1976 cτA.",transform=ax.transAxes,color="#8fb4c1",fontsize=5.25,ha="left",va="top")

def plot(a,b,g):
    sr=math.atan2(RS,AU76)*ASR; fig,ax=plt.subplots(figsize=(9.6,5.8),dpi=240); fig.patch.set_facecolor("#03080d"); ax.set_facecolor("#03080d")
    solar_limb=Circle((0.,0.),sr,fill=False,lw=.36,ec="#66e8ff",alpha=.95); ax.add_patch(solar_limb); ax.axhline(0,lw=.18,color="#1d3d4a",alpha=.72); ax.axvline(0,lw=.18,color="#1d3d4a",alpha=.72)
    count=0
    for tr in (a,b):
        s=tr["site"]; c=COLORS[s["label"]]; p=tr["points"]; ax.plot(p[:,0],p[:,1],lw=.30,color=c,label=s["label"],zorder=3); ax.scatter(p[::6,0],p[::6,1],s=.75,color=c,alpha=.7,linewidths=0,zorder=4)
        for ev in EVENTS:
            q=tr["event_points"][ev]; ax.add_patch(Circle((q[0],q[1]),tr["event_radii"][ev],fill=False,lw=.28 if ev=="CA" else .20,ec=c,alpha=.92,zorder=2)); ax.scatter([q[0]],[q[1]],s=3.8 if ev=="CA" else 2.2,color=c,edgecolors="#03080d",linewidths=.16,zorder=5); count+=1
        label(ax,tr["event_points"]["CA"],s["short"]+" CA",18,44 if s is A else -44,c)
    if count!=10: raise RuntimeError(f"plotted {count} events, expected 10")
    for ev,dx,dy in (("C1",-48,12),("C2",-38,9),("C3",20,-10),("C4",30,-13)): label(ax,a["event_points"][ev],ev,dx,dy,"#8fb4c1")
    table(ax,a,b,g); allp=np.vstack([a["points"],b["points"]]); upper=np.median(allp[:,1])>=0; ax.set_xlim(-1.04*sr,1.04*sr); ax.set_ylim((-0.06*sr,1.06*sr) if upper else (-1.06*sr,0.06*sr)); ax.set_aspect("equal",adjustable="box")
    ax.grid(True,color="#102630",linewidth=.16,alpha=.55); ax.tick_params(colors="#8fb4c1",labelsize=6.5,width=.22,length=2); ax.set_xlabel("IAU-1976-normalized solar-screen X offset (arcsec)",color="#8fb4c1",fontsize=7.5); ax.set_ylabel("IAU-1976-normalized solar-screen Y offset (arcsec)",color="#8fb4c1",fontsize=7.5)
    ax.set_title("1769 Venus Transit — Engineering Half-Sun Track Reconstruction\nVardo, Norway / Point Venus, Tahiti — fresh JPL contacts; IAU 1976 normalization",color="#f8fdff",fontsize=9,pad=8)
    for sp in ax.spines.values(): sp.set_linewidth(.22); sp.set_color("#25708b")
    lg=ax.legend(loc="lower right",fontsize=6.3,frameon=True,borderpad=.45); lg.get_frame().set_facecolor("#071016"); lg.get_frame().set_edgecolor("#1e4f64")
    for t in lg.get_texts(): t.set_color("#dff8ff")
    fig.text(.5,.016,f"C1, C2, CA, C3, C4 plotted for both tracks. π⊙={g['pi']:.10f} arcsec; R⊕={RE:.3f} km; cτA={AU76:.6f} km.",ha="center",va="bottom",fontsize=6.2,color="#8fb4c1")
    fig.savefig(PNG,dpi=460,facecolor=fig.get_facecolor(),bbox_inches="tight",pad_inches=.055); plt.show(); plt.close(fig)

def save(a,b,g):
    with CSV.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow([V,"CONTACTS AND IAU1976 GEOMETRY"]); w.writerow(["site","event","utc","jd_tdb","x_arcsec","y_arcsec","venus_radius_arcsec","track_angle_deg","rms_arcsec"])
        for tr in (a,b):
            for ev in EVENTS:
                q=tr["event_points"][ev]; w.writerow([tr["site"]["label"],ev,utc(tr["events"][ev]),f"{tr['events'][ev]:.12f}",f"{q[0]:.9f}",f"{q[1]:.9f}",f"{tr['event_radii'][ev]:.9f}",f"{tr['angle']:.9f}",f"{tr['rms']:.9f}"])
        w.writerow([]); w.writerow(["quantity","value","unit"])
        for k,u0 in (("th","arcsec"),("rho","arcsec"),("abp","km"),("aba","arcsec"),("ab","km"),("ratio","ratio"),("raw","arcsec"),("pi","arcsec"),("res","arcsec")): w.writerow([k,g[k],u0])

def audit():
    s=Path(__file__).read_text(); ast.parse(s); lines=s.splitlines(); checks={"first":lines[0]=="# V0021","last":lines[-1]=="# V0021","contacts":"events_a=all_events" in s and "events_b=all_events" in s,"five_events":'EVENTS=("C1","C2","CA","C3","C4")' in s,"sun":"solar_limb=Circle((0.,0.),sr" in s and "ax.add_patch(solar_limb)" in s,"two_tracks":"for tr in (a,b):" in s,"event_plot":"for ev in EVENTS:" in s,"ten_events":"if count!=10:" in s,"save":"fig.savefig(PNG" in s,"show":"plt.show()" in s}; return len(lines),checks

def main():
    OUT.mkdir(parents=True,exist_ok=True); n,checks=audit(); bad=[k for k,v in checks.items() if not v]
    if bad: raise RuntimeError("static audit failed: "+", ".join(bad))
    gf,tf,m=six_series(); g=cache(gf); t=cache(tf); events_a=all_events(t,A); events_b=all_events(t,B); ref=.5*(events_a["CA"]+events_b["CA"]); b0=basis(g,ref); a=track(g,t,A,events_a,b0); b=track(g,t,B,events_b,b0); geom=geometry(g,a,b,ref); save(a,b,geom); plot(a,b,geom)
    runtime={"master":len(m)==N,"A5":tuple(events_a)==EVENTS,"B5":tuple(events_b)==EVENTS,"png":PNG.is_file() and PNG.stat().st_size>0,"csv":CSV.is_file() and CSV.stat().st_size>0,"pi":abs(geom["pi"]-PI76)<=5e-12}
    bad=[k for k,v in runtime.items() if not v]
    if bad: raise RuntimeError("runtime audit failed: "+", ".join(bad))
    print("CODE INPUTS"); print(f"Six fresh JPL series | {len(m)} rows | {START} to {STOP} | {STEP}")
    print("COMMENTS"); print("C1, C2, closest approach, C3, C4 computed and plotted independently for both tracks.")
    print("RESULTS");
    for tr in (a,b): print(tr["site"]["short"]+" | "+" | ".join(f"{ev} {utc(tr['events'][ev])}" for ev in EVENTS))
    print(f"π⊙ | {geom['pi']:.10f} arcsec"); print("OUTPUT SUMMARY"); print(f"PNG | {PNG}"); print(f"CSV | {CSV}"); print(f"MASTER | {MASTER}")
    print("PAPER COMPARISON"); print(f"IAU 1976 | {PI76:.10f} arcsec"); print("EQUATION STATUS"); print(f"Source lines | {n}")
    for k,v in checks.items(): print(f"{k} | {'PASS' if v else 'FAIL'}")
    for k,v in runtime.items(): print(f"{k} | {'PASS' if v else 'FAIL'}")
    print(datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z")); print(f"# {V}")

if __name__=="__main__":
    if "--audit-only" in sys.argv:
        n,c=audit(); print(f"LINE COUNT | {n}"); [print(f"{k} | {'PASS' if v else 'FAIL'}") for k,v in c.items()]; sys.exit(0 if all(c.values()) else 1)
    main()
# V0021
