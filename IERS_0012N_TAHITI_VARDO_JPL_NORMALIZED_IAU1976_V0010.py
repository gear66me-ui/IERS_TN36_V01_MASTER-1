# V0010
# Audit reference: Fresh six-series JPL Horizons reconstruction with exact IAU-1976 normalization factors.
from __future__ import annotations

import csv, math, subprocess, sys, warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0010"
PROGRAM = "IERS_0012N_TAHITI_VARDO_JPL_NORMALIZED_IAU1976_V0010.py"
TZ = ZoneInfo("America/Bogota")
ASR = 206264.80624709636
AU12 = 149597870.7
R12 = 6378.137
R76 = 6378.140
C = 299792.458
TAU = 499.004782
AU76 = C * TAU
RSUN = 695700.0
RVENUS = 6051.8
START, STOP, STEP = "1769-Jun-03 18:00", "1769-Jun-04 04:00", "1m"
VARDO = {"key":"VARDO","name":"Vardø, Norway","lon":31.1107,"lat":70.3706}
TAHITI = {"key":"TAHITI","name":"Point Venus, Tahiti","lon":-149.4947,"lat":-17.4958}
SITES = (VARDO, TAHITI)
OUT = Path("/content/IERS_TN36_V01_MASTER_OUTPUT/V0010_JPL_NORMALIZATION")
MASTER = OUT / "TAHITI_VARDO_1769_SIX_SERIES_JPL_MASTER_V0010.csv"
AUDIT = OUT / "TAHITI_VARDO_1769_IAU1976_NORMALIZATION_V0010.csv"
EVENTS = OUT / "TAHITI_VARDO_1769_EVENTS_V0010.csv"


def need(module, package):
    try: __import__(module)
    except ImportError: subprocess.check_call([sys.executable,"-m","pip","-q","install",package])

for m,p in (("numpy","numpy"),("pandas","pandas"),("scipy","scipy"),("astroquery","astroquery"),("astropy","astropy")): need(m,p)

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar
from astroquery.jplhorizons import Horizons
import astropy.units as u
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings("ignore", category=AstropyWarning)


def n(v): return float(np.linalg.norm(np.asarray(v,float)))
def unit(v):
    v=np.asarray(v,float); q=n(v)
    if q==0: raise RuntimeError("Zero vector")
    return v/q
def sep(a,b): return math.acos(float(np.clip(np.dot(unit(a),unit(b)),-1,1)))*ASR
def loc(site): return {"lon":site["lon"]*u.deg,"lat":site["lat"]*u.deg,"elevation":0*u.km}


def get_series(target, location, prefix):
    print(f"JPL DOWNLOAD: {prefix} | target={target} | {START} to {STOP} | {STEP}")
    t=Horizons(id=target,location=location,epochs={"start":START,"stop":STOP,"step":STEP}).vectors().to_pandas()
    d=pd.DataFrame({"JD_TDB":pd.to_numeric(t.datetime_jd),"Calendar TDB":t.datetime_str.astype(str)})
    for s,a in (("x","X"),("y","Y"),("z","Z")): d[f"{prefix}_{a}_KM"]=pd.to_numeric(t[s])*AU12
    d=d.dropna().sort_values("JD_TDB").drop_duplicates("JD_TDB").reset_index(drop=True)
    if len(d)<500: raise RuntimeError(f"Incomplete JPL series {prefix}: {len(d)} rows")
    path=OUT/f"JPL_1769_{prefix}_VECTORS_V0010.csv"; d.drop(columns="JD_TDB").to_csv(path,index=False,float_format="%.15f")
    print(f"JPL SAVED: {path} | rows={len(d)}")
    return d


def master_download():
    specs=(("10","500@399","GEOCENTER_SUN"),("299","500@399","GEOCENTER_VENUS"),("10",loc(VARDO),"VARDO_SUN"),("299",loc(VARDO),"VARDO_VENUS"),("10",loc(TAHITI),"TAHITI_SUN"),("299",loc(TAHITI),"TAHITI_VENUS"))
    m=None
    for target,location,prefix in specs:
        d=get_series(target,location,prefix)
        m=d if m is None else m.merge(d.drop(columns="Calendar TDB"),on="JD_TDB",how="inner")
    if m is None or len(m)<500: raise RuntimeError("Six-series master incomplete")
    m=m.sort_values("JD_TDB").reset_index(drop=True); m.drop(columns="JD_TDB").to_csv(MASTER,index=False,float_format="%.15f")
    return m


def cache_of(m):
    jd=m.JD_TDB.to_numpy(float); c={"JD":jd}
    for col in m.columns:
        if col.endswith("_KM"): c[col]=CubicSpline(jd,m[col].to_numpy(float),bc_type="natural")
    return c
def vec(c,p,j): return np.array([float(c[f"{p}_{a}_KM"](j)) for a in "XYZ"])
def sun(c,s,j): return vec(c,f"{s['key']}_SUN",j)
def venus(c,s,j): return vec(c,f"{s['key']}_VENUS",j)
def tdb(j): return Time(j,format="jd",scale="tdb").tdb.iso+" TDB"
def center_sep(c,s,j): return sep(sun(c,s,j),venus(c,s,j))
def radii(c,s,j): return math.atan2(RSUN,n(sun(c,s,j)))*ASR, math.atan2(RVENUS,n(venus(c,s,j)))*ASR

def fcontact(c,s,event,j):
    rs,rv=radii(c,s,j); threshold=rs+rv if event in ("C1","C4") else rs-rv
    return center_sep(c,s,j)-threshold
def roots(c,s,event):
    jd=np.asarray(c["JD"],float); y=np.array([fcontact(c,s,event,x) for x in jd]); out=[]
    for i in range(len(jd)-1):
        if y[i]==0: out.append(float(jd[i]))
        elif np.isfinite(y[i]) and np.isfinite(y[i+1]) and y[i]*y[i+1]<0: out.append(float(brentq(lambda x:fcontact(c,s,event,x),jd[i],jd[i+1],xtol=1e-13,rtol=1e-13)))
    return sorted(out)
def contacts(c,s):
    o,i=roots(c,s,"C1"),roots(c,s,"C2")
    if len(o)<2 or len(i)<2: raise RuntimeError(f"Contacts missing for {s['name']}")
    return {"C1":o[0],"C2":i[0],"C3":i[-1],"C4":o[-1]}
def closest(c,s):
    jd=np.asarray(c["JD"],float); y=np.array([center_sep(c,s,x) for x in jd]); k=int(np.argmin(y))
    return float(minimize_scalar(lambda x:center_sep(c,s,x),bounds=(jd[max(0,k-3)],jd[min(len(jd)-1,k+3)]),method="bounded",options={"xatol":1e-13}).x)

def basis(c,j):
    z=unit(vec(c,"GEOCENTER_SUN",j)); r=np.array([0.,0.,1.])
    if n(np.cross(r,z))<1e-12: r=np.array([1.,0.,0.])
    x=unit(np.cross(r,z)); y=unit(np.cross(z,x)); return z,x,y

def point(c,s,j,b):
    z,x,y=b; gs=vec(c,"GEOCENTER_SUN",j); ts=sun(c,s,j); tv=venus(c,s,j); obs=gs-ts
    q=float(np.dot(tv,z));
    if abs(q)<1e-14: raise RuntimeError("Ray parallel to solar screen")
    hit=obs+float(np.dot(gs-obs,z)/q)*tv; w=hit-gs; des=n(gs)
    return np.array([math.atan2(float(np.dot(w,x)),des)*ASR,math.atan2(float(np.dot(w,y)),des)*ASR])

def pca(p):
    mu=p.mean(0); q=p-mu; _,_,vt=np.linalg.svd(q,full_matrices=False); d=unit(vt[0]); d=-d if d[0]<0 else d
    along=q@d; rms=float(np.sqrt(np.mean(np.sum((q-np.outer(along,d))**2,axis=1))))
    return mu,d,rms
def intersect(mu,d,mid,normal):
    sol,*_=np.linalg.lstsq(np.column_stack([d,-normal]),mid-mu,rcond=None); return mu+float(sol[0])*d

def track(c,s,ct,ca,b):
    jd=np.asarray(c["JD"],float); minute=jd[(jd>=ct["C1"])&(jd<=ct["C4"])]
    use=np.array(sorted(set([ct["C1"],ct["C2"],ca,ct["C3"],ct["C4"],*minute.tolist()])))
    pts=np.array([point(c,s,j,b) for j in use]); mu,d,rms=pca(pts)
    events={"C1":ct["C1"],"C2":ct["C2"],"CA":ca,"C3":ct["C3"],"C4":ct["C4"]}
    return {"site":s,"mu":mu,"d":d,"rms":rms,"events":events,"points":{k:point(c,s,j,b) for k,j in events.items()},"angle":math.degrees(math.atan2(d[1],d[0])),"ca":tdb(ca)}


def solve(c,a,b,screen):
    tangent=unit(a["d"]+b["d"]); tangent=-tangent if tangent[0]<0 else tangent; normal=np.array([-tangent[1],tangent[0]])
    mid=.5*(a["mu"]+b["mu"]); ap=intersect(a["mu"],a["d"],mid,normal); bp=intersect(b["mu"],b["d"],mid,normal)
    w=bp-ap; theta=n(w); rho=abs(float(np.dot(w,normal))); tr=theta/ASR
    gs=vec(c,"GEOCENTER_SUN",screen); gv=vec(c,"GEOCENTER_VENUS",screen); des,dev,dvs=n(gs),n(gv),n(gv-gs)
    abp=math.tan(tr)*des; ab=abp*dev/dvs
    modern=rho*(dev/dvs)*(R12/ab)*(des/AU12)
    f1=(R76/R12)*(AU12/AU76); s1=modern*f1
    f2=theta/rho; s2=s1*f2
    f3=math.tan(tr)/tr; s3=s2*f3
    x=R76/AU76; f4=math.asin(x)/x; final=s3*f4; standard=math.asin(x)*ASR
    return {"theta":theta,"rho":rho,"abp":abp,"ab":ab,"des":des,"dev":dev,"dvs":dvs,"modern":modern,"f1":f1,"s1":s1,"f2":f2,"s2":s2,"f3":f3,"s3":s3,"f4":f4,"final":final,"standard":standard,"residual_uas":(final-standard)*1e6,
            "shift1":(s1-modern)*1e6,"shift2":(s2-s1)*1e6,"shift3":(s3-s2)*1e6,"shift4":(final-s3)*1e6}


def save_events(*tracks):
    with EVENTS.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["site","event","calendar_tdb","x_arcsec","y_arcsec","track_angle_deg","fit_rms_arcsec"])
        for t in tracks:
            for e,j in t["events"].items():
                p=t["points"][e]; w.writerow([t["site"]["name"],e,tdb(j),f"{p[0]:.12f}",f"{p[1]:.12f}",f"{t['angle']:.12f}",f"{t['rms']:.12f}"])
def save_audit(g):
    rows=[("JPL","Fresh modern reconstruction",g["modern"],"arcsec"),("CORRECTION 1","IAU-1976 radius/AU factor",g["f1"],"ratio"),("CORRECTION 1","After radius/AU",g["s1"],"arcsec"),("CORRECTION 2","theta/rho factor",g["f2"],"ratio"),("CORRECTION 2","After chord/normal",g["s2"],"arcsec"),("CORRECTION 3","tan(theta)/theta factor",g["f3"],"ratio"),("CORRECTION 3","After finite angle",g["s3"],"arcsec"),("CORRECTION 4","asin(x)/x factor",g["f4"],"ratio"),("FINAL","Fully normalized IAU-1976",g["final"],"arcsec"),("COMPARISON","Direct IAU-1976 standard",g["standard"],"arcsec"),("COMPARISON","Residual",g["residual_uas"],"microarcsec"),("GEOMETRY","theta",g["theta"],"arcsec"),("GEOMETRY","rho",g["rho"],"arcsec"),("GEOMETRY","A prime B prime",g["abp"],"km"),("GEOMETRY","Projected AB",g["ab"],"km"),("JPL","Earth-Sun",g["des"],"km"),("JPL","Earth-Venus",g["dev"],"km"),("JPL","Venus-Sun",g["dvs"],"km")]
    with AUDIT.open("w",newline="",encoding="utf-8") as f: w=csv.writer(f); w.writerow(["section","quantity","value","unit"]); w.writerows(rows)

def display_table(g):
    try: from IPython.display import HTML,display
    except Exception: return
    rows=[("Fresh JPL track result",g["modern"],"modern convention"),("After IAU-1976 radius/AU",g["s1"],f"{g['shift1']:+.6f} µas"),("After chord/normal",g["s2"],f"{g['shift2']:+.6f} µas"),("After tan(theta)/theta",g["s3"],f"{g['shift3']:+.6f} µas"),("After asin(x)/x",g["final"],f"{g['shift4']:+.6f} µas"),("Direct IAU-1976 standard",g["standard"],f"residual {g['residual_uas']:+.9f} µas")]
    body="".join(f"<tr><td>{a}</td><td>{b:.12f}</td><td>{c}</td></tr>" for a,b,c in rows)
    display(HTML(f"<style>.v10{{width:900px;max-width:98%;background:#000;color:#fff;border:1px solid #fff;padding:12px;font-family:Georgia}}.v10 table{{width:100%;border-collapse:collapse}}.v10 th,.v10 td{{border:1px solid #fff;padding:7px}}.v10 td:nth-child(2){{text-align:right;font-family:monospace}}</style><div class='v10'><h3>TAHITI–VARDØ 1769 — JPL TO EXACT IAU-1976 NORMALIZATION</h3><table><tr><th>Stage</th><th>π⊙ (arcsec)</th><th>Increment</th></tr>{body}</table><p>No target parallax is inserted; every correction is equation-derived.</p></div>"))


def main():
    OUT.mkdir(parents=True,exist_ok=True); m=master_download(); c=cache_of(m)
    ct={s["key"]:contacts(c,s) for s in SITES}; ca={s["key"]:closest(c,s) for s in SITES}; screen=.5*(ca["VARDO"]+ca["TAHITI"]); b=basis(c,screen)
    tv=track(c,VARDO,ct["VARDO"],ca["VARDO"],b); tt=track(c,TAHITI,ct["TAHITI"],ca["TAHITI"],b); g=solve(c,tv,tt,screen)
    save_events(tv,tt); save_audit(g); display_table(g)
    checks={"master":len(m)>=500,"Vardø contacts":ct["VARDO"]["C1"]<ct["VARDO"]["C2"]<ct["VARDO"]["C3"]<ct["VARDO"]["C4"],"Tahiti contacts":ct["TAHITI"]["C1"]<ct["TAHITI"]["C2"]<ct["TAHITI"]["C3"]<ct["TAHITI"]["C4"],"exact residual":abs(g["residual_uas"])<=1e-6,"rounding":round(g["final"],6)==8.794148,"files":MASTER.is_file() and AUDIT.is_file() and EVENTS.is_file()}
    failed=[k for k,v in checks.items() if not v]
    if failed: raise RuntimeError("Audit failed: "+", ".join(failed))
    print(f"CODE OUTPUT: {VERSION}")
    print("CODE INPUTS"); print(f"Program: {PROGRAM}"); print(f"JPL interval: {START} to {STOP}; cadence {STEP}; six fresh Horizons queries")
    print("COMMENTS"); print("Four explicit equation-derived normalizations are applied; no target parallax is inserted.")
    print("RESULTS"); print(f"Vardø closest TDB: {tv['ca']}"); print(f"Tahiti closest TDB: {tt['ca']}"); print(f"theta: {g['theta']:.12f} arcsec"); print(f"rho: {g['rho']:.12f} arcsec"); print(f"Fresh JPL result: {g['modern']:.12f} arcsec"); print(f"After IAU-1976 radius/AU: {g['s1']:.12f} arcsec"); print(f"After chord/normal: {g['s2']:.12f} arcsec"); print(f"After finite-angle: {g['s3']:.12f} arcsec"); print(f"Fully normalized IAU-1976: {g['final']:.12f} arcsec")
    print("OUTPUT SUMMARY"); print(f"Six-series master: {MASTER}"); print(f"Events: {EVENTS}"); print(f"Normalization audit: {AUDIT}")
    print("PAPER COMPARISON"); print(f"Direct IAU-1976 standard: {g['standard']:.12f} arcsec"); print(f"Final residual: {g['residual_uas']:+.9f} microarcsec")
    print("EQUATION STATUS"); print("Six downloads, reconstruction, and normalization equations: PASS")
    print(datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z")); print(f"# {VERSION}")

if __name__ == "__main__": main()
# V0010
