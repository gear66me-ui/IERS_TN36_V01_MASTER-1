# V0040
# Audit reference: IERS-scope and JPL instantaneous distance/velocity sanity check before the Halley P reduction.
from __future__ import annotations
import math, subprocess, sys, time, urllib.request, warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

def need(name: str, pip: str) -> None:
    try: __import__(name)
    except ImportError: subprocess.check_call([sys.executable,"-m","pip","-q","install",pip])
for a,b in (("numpy","numpy"),("pandas","pandas"),("scipy","scipy"),("astropy","astropy"),("astroquery","astroquery"),("IPython","ipython")): need(a,b)
try:
    from erfa import ErfaWarning
    warnings.filterwarnings("ignore", category=ErfaWarning)
except Exception: warnings.filterwarnings("ignore", message=".*dubious year.*")
import numpy as np, pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import HTML, display
from scipy.optimize import minimize_scalar

VERSION="V0040"; TZ=ZoneInfo("America/Bogota"); ROOT=Path("/content")
OUT=ROOT/"VENUS_1769_IERS_TN36_DISTANCE_VELOCITY_SANITY_V0040_OUTPUT"; OUT.mkdir(parents=True,exist_ok=True)
CSV=OUT/"VENUS_1769_IERS_TN36_DISTANCE_VELOCITY_SANITY_V0040.csv"; HTML_PATH=OUT/"VENUS_1769_IERS_TN36_DISTANCE_VELOCITY_SANITY_V0040.html"
BASE_COMMIT="d55bc2274359a3014c19f257e42cc149bc458d57"
BASE_URL=f"https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/{BASE_COMMIT}/VENUS_1769_TAHITI_VARDO_HALLEY_THUMB_CHECK_V0031.py"
BASE_PATH=ROOT/"VENUS_1769_TAHITI_VARDO_HALLEY_THUMB_CHECK_V0031_BASE.py"
MASTER_CANDIDATES=(ROOT/"VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0034.csv",ROOT/"VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0033.csv",ROOT/"VENUS_1769_TAHITI_VARDO_JPL_ECLIPTIC_V0040.csv")
AU_KM=149_597_870.0; DAY=86400.0; ASR=206_264.80624709636

def norm(v): return float(np.linalg.norm(np.asarray(v,dtype=float)))
def unit(v):
    n=norm(v)
    if n==0: raise RuntimeError("Zero vector.")
    return np.asarray(v,dtype=float)/n

def load_base():
    req=urllib.request.Request(f"{BASE_URL}?cache={time.time_ns()}",headers={"User-Agent":"Mozilla/5.0","Cache-Control":"no-cache"})
    with urllib.request.urlopen(req,timeout=90) as r: src=r.read().decode("utf-8")
    if not src.startswith("# V0031\n") or not src.rstrip().endswith("# V0031"): raise RuntimeError("Pinned V0031 audit failed.")
    BASE_PATH.write_text(src,encoding="utf-8"); ns={"__name__":"v0031_base","__file__":str(BASE_PATH)}; exec(compile(src,str(BASE_PATH),"exec"),ns); return ns

def compatible(path, cols):
    if not path.is_file(): return False
    try: got=pd.read_csv(path,nrows=0).columns
    except Exception: return False
    return all(c in got for c in cols)

def master(base):
    cols=["JD_TDB"]+[f"{p}_{a}_KM" for p in base["PREFIXES"] for a in "XYZ"]
    for p in MASTER_CANDIDATES:
        if compatible(p,cols): return pd.read_csv(p),str(p)
    df=base["build_master"](); df.to_csv(MASTER_CANDIDATES[-1],index=False,float_format="%.15f"); return df,"NEW JPL POSITION DOWNLOAD"

def sep_arcsec(base,cache,jd):
    s=base["vector_at"](cache,"GEOCENTER_SUN",jd); v=base["vector_at"](cache,"GEOCENTER_VENUS",jd)
    return math.acos(float(np.clip(np.dot(unit(s),unit(v)),-1,1)))*ASR

def closest(base,cache):
    j=np.asarray(cache["JD_TDB"],float); vals=np.array([sep_arcsec(base,cache,float(x)) for x in j]); i=int(np.argmin(vals)); lo=max(0,i-5); hi=min(len(j)-1,i+5); j0=float(j[lo]); span=(float(j[hi])-j0)*DAY
    f=lambda sec: sep_arcsec(base,cache,j0+float(sec)/DAY)**2
    r=minimize_scalar(f,bounds=(0.0,span),method="bounded",options={"xatol":1e-7,"maxiter":1000})
    if not r.success: raise RuntimeError("Closest approach failed.")
    jd=j0+float(r.x)/DAY; return jd,{"center":f(r.x),"minus":f(r.x-1),"plus":f(r.x+1),"span":span}

def jpl_states(target_id,epochs):
    q=Horizons(id=target_id,location="500@399",epochs=epochs); t=q.vectors(refplane="ecliptic",aberrations="geometric").to_pandas()
    rows=[]
    for _,r in t.iterrows():
        pos=np.array([float(r[a])*AU_KM for a in "xyz"]); vel=np.array([float(r[f"v{a}"])*AU_KM/DAY for a in "xyz"]); rows.append((float(r["datetime_jd"]),pos,vel))
    return rows

def state(sun,venus):
    rs,vs=sun; rv,vv=venus; rvs=rs-rv; vvs=vs-vv
    def dr(r,v): d=norm(r); return d,float(np.dot(r,v)/d)
    ES,ESd=dr(rs,vs); EV,EVd=dr(rv,vv); VS,VSd=dr(rvs,vvs)
    R={"EV/VS":EV/VS,"VS/EV":VS/EV,"ES/VS":ES/VS,"VS/ES":VS/ES,"EV/ES":EV/ES,"ES/EV":ES/EV}
    D={"EV/VS":(EVd*VS-EV*VSd)/VS**2,"VS/EV":(VSd*EV-VS*EVd)/EV**2,"ES/VS":(ESd*VS-ES*VSd)/VS**2,"VS/ES":(VSd*ES-VS*ESd)/ES**2,"EV/ES":(EVd*ES-EV*ESd)/ES**2,"ES/EV":(ESd*EV-ES*EVd)/EV**2}
    return {"EV":EV,"VS":VS,"ES":ES,"EVd":EVd,"VSd":VSd,"ESd":ESd,"R":R,"D":D}

def table(df,formats=None):
    x=df.copy(); formats=formats or {}
    for c,f in formats.items():
        if c in x: x[c]=x[c].map(lambda v:f.format(v) if pd.notna(v) else "")
    return '<div class="wrap">'+x.to_html(index=False,border=0,classes="audit",escape=False)+'</div>'

def main():
    base=load_base(); df,source=master(base); cache=base["build_cache"](df); jd,diag=closest(base,cache); utc=Time(jd,format="jd",scale="tdb").utc.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    epochs=[jd-60/DAY,jd-1/DAY,jd,jd+1/DAY,jd+60/DAY]; sun=jpl_states("10",epochs); ven=jpl_states("299",epochs)
    states={off:state((sun[k][1],sun[k][2]),(ven[k][1],ven[k][2])) for k,off in enumerate((-60,-1,0,1,60))}; s=states[0]
    scope=pd.DataFrame([
        ["ITRS station → GCRS observer","IERS TN36 Chapter 5 / ERFA-Astropy","NOT USED in geocentric distance audit"],
        ["Sun/Venus position and velocity vectors","JPL Horizons geometric ecliptic","USED"],
        ["EV, VS, ES and all ratios","Direct vector norms and derivatives","USED"],
        ["JPL A′, JPL B′, common screen, Halley P","Project reduction, not specified by TN36","NOT USED in V0040"],
    ],columns=["Component","Authority / definition","Status"])
    epoch=pd.DataFrame([["Geocentric instantaneous closest approach",utc+" UTC",jd],["Local −1 s objective", "arcsec²",diag["minus"]],["Center objective","arcsec²",diag["center"]],["Local +1 s objective","arcsec²",diag["plus"]]],columns=["Quantity","Description","Value"])
    distances=pd.DataFrame([["Earth → Venus","EV",s["EV"],s["EVd"]],["Venus → Sun","VS",s["VS"],s["VSd"]],["Earth → Sun","ES",s["ES"],s["ESd"]]],columns=["JPL distance","Symbol","Distance km","Range rate km/s"])
    labels=(("Earth–Venus / Venus–Sun","EV/VS"),("Venus–Sun / Earth–Venus","VS/EV"),("Earth–Sun / Venus–Sun","ES/VS"),("Venus–Sun / Earth–Sun","VS/ES"),("Earth–Venus / Earth–Sun","EV/ES"),("Earth–Sun / Earth–Venus","ES/EV"))
    ratios=pd.DataFrame([[a,b,s["R"][b],s["D"][b],s["D"][b]*60,1e6*s["D"][b]*3600/s["R"][b]] for a,b in labels],columns=["Ratio","Definition","Instantaneous value","Rate per second","Change per 60 s","Fractional ppm/hour"])
    around=pd.DataFrame([[off,st["EV"],st["VS"],st["ES"],st["R"]["EV/VS"],st["R"]["EV/ES"],st["R"]["ES/VS"]] for off,st in states.items()],columns=["Offset s","EV km","VS km","ES km","EV/VS","EV/ES","ES/VS"])
    identities=pd.DataFrame([["(EV/VS)(VS/EV)",s["R"]["EV/VS"]*s["R"]["VS/EV"],1.0],["(EV/ES)(ES/EV)",s["R"]["EV/ES"]*s["R"]["ES/EV"],1.0],["(ES/VS)(VS/ES)",s["R"]["ES/VS"]*s["R"]["VS/ES"],1.0],["(EV/ES)(ES/VS)",s["R"]["EV/ES"]*s["R"]["ES/VS"],s["R"]["EV/VS"]]],columns=["Identity","Calculated","Expected"]); identities["Residual"]=identities["Calculated"]-identities["Expected"]
    rows=[]
    for sec,frame in (("SCOPE",scope),("EPOCH",epoch),("DISTANCES",distances),("RATIOS",ratios),("AROUND_CA",around),("IDENTITIES",identities)):
        for i,r in frame.iterrows(): d={"section":sec,"row":int(i)}; d.update(r.to_dict()); rows.append(d)
    pd.DataFrame(rows).to_csv(CSV,index=False,float_format="%.15f")
    css='''<style>.r{background:#fff;color:#000;font-family:Arial;padding:16px;border:1px solid #000}.r *{background:#fff;color:#000;box-sizing:border-box}.r h1{font-size:22px;border-bottom:2px solid #000;padding-bottom:8px}.r h2{font-size:16px;border-top:1px solid #000;border-bottom:1px solid #000;padding:5px 0;margin-top:22px}.r h3{font-size:14px}.wrap{overflow-x:auto}.audit{border-collapse:collapse;min-width:100%;width:max-content;font-size:12px}.audit th,.audit td{border:1px solid #000;padding:7px 9px;white-space:nowrap}.audit th{font-weight:700;text-align:center}.audit td{text-align:right}.audit td:first-child,.audit td:nth-child(2){text-align:left}.note{border:1px solid #000;padding:9px;font-weight:600}.path{font-family:monospace;font-size:11px;overflow-wrap:anywhere;white-space:normal}</style>'''
    h=[css,'<div class="r"><h1>1769 Venus Transit — IERS Scope and JPL Instantaneous Distance/Velocity Sanity Check</h1>',f'<h2>CODE INPUTS</h2><p><b>JPL source:</b> {source}</p><p><b>Instantaneous epoch:</b> {utc} UTC</p><p><b>Frame:</b> JPL geometric ecliptic, geocenter 500@399</p>','<h2>COMMENTS</h2><p class="note">IERS TN36 governs the terrestrial-to-celestial observer transformation. It does not define the Venus-transit A′B′, Halley triangle, or P reduction. This audit stops before JPL A′ and JPL B′.</p><p class="note">Direct JPL velocity vectors, range rates, ratio rates, and ±60-second values test whether motion omitted by an instantaneous ratio can explain a later residual.</p>','<h2>RESULTS</h2><h3>IERS and project scope</h3>',table(scope),'<h3>Corrected instantaneous closest approach</h3>',table(epoch,{"Value":"{:.15f}"}),'<h3>JPL instantaneous distances and range rates</h3>',table(distances,{"Distance km":"{:,.12f}","Range rate km/s":"{:+.12f}"}),'<h3>All ratio permutations and derivatives</h3>',table(ratios,{"Instantaneous value":"{:.15f}","Rate per second":"{:+.15e}","Change per 60 s":"{:+.15e}","Fractional ppm/hour":"{:+.9f}"}),'<h3>Direct JPL values around closest approach</h3>',table(around,{"Offset s":"{:+.0f}","EV km":"{:,.12f}","VS km":"{:,.12f}","ES km":"{:,.12f}","EV/VS":"{:.15f}","EV/ES":"{:.15f}","ES/VS":"{:.15f}"}),'<h3>Ratio identities</h3>',table(identities,{"Calculated":"{:.15f}","Expected":"{:.15f}","Residual":"{:+.15e}"}),f'<h2>OUTPUT SUMMARY</h2><p class="path">{CSV}</p><p class="path">{HTML_PATH}</p>','<h2>PAPER COMPARISON</h2><p class="note">This is a pre-P distance-only audit. The next stage will explicitly label JPL A′ and JPL B′ and will not reuse AB/A′B′ as a distance factor.</p>']
    local=diag["center"]<=diag["minus"] and diag["center"]<=diag["plus"]; mx=float(np.max(np.abs(identities["Residual"])))
    status=pd.DataFrame([["Seconds-offset local ±1 s minimum","PASS" if local else "FAIL",diag["center"]],["Ratio reciprocal/chain identities","PASS" if mx<1e-12 else "FAIL",mx],["IERS observer transformation","NOT USED",0.0],["JPL A′, JPL B′, P reduction","NOT USED",0.0]],columns=["Equation / test","Status","Residual / diagnostic"])
    h+=['<h2>EQUATION STATUS</h2>',table(status,{"Residual / diagnostic":"{:.15e}"}),'</div>']; report=''.join(h); HTML_PATH.write_text("<html><body style='margin:0;background:#fff;color:#000'>"+report+"</body></html>",encoding="utf-8"); display(HTML(report))
    if not local or mx>=1e-12: raise RuntimeError("Audit check failed.")
    print(datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z")); print(VERSION)
if __name__=="__main__": main()
# V0040