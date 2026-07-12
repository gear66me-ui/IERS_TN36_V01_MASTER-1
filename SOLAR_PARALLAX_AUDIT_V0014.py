# V0014
# Audit reference: JPL-vector solar-parallax reduction to exact IAU-1976 Case 2 with publication plate.
from __future__ import annotations

import argparse, csv, io, json, math, os, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

VERSION="V0014"; PROGRAM="SOLAR_PARALLAX_AUDIT_V0014.py"
TZ=ZoneInfo("America/Bogota"); ASEC=206264.80624709636
TARGET=8.794148; TOL=0.000007
C=299792458.0; TAU=499.004782; R76=6378140.0; R76_SIGMA=5.0
AU76=149597870000.0; AU76_SIGMA=2000.0; AU12=149597870700.0
RW=6378137.0; RIERS=6378136.6
ROOT=Path("/content"); OUT_DEFAULT=ROOT/"SOLAR_PARALLAX_AUDIT_V0014_OUTPUT"
MASTER_DEFAULT=ROOT/"O6_1769_GEOCENTER_HORIZONS_V0014.csv"
API="https://ssd.jpl.nasa.gov/api/horizons.api"
COLS=["JD","GEOCENTER_SUN_X_KM","GEOCENTER_SUN_Y_KM","GEOCENTER_SUN_Z_KM","GEOCENTER_VENUS_X_KM","GEOCENTER_VENUS_Y_KM","GEOCENTER_VENUS_Z_KM"]
RADII={"WGS84":RW,"IAU1976":R76,"IERS2010":RIERS}


def cli():
    p=argparse.ArgumentParser(description="1769 solar-parallax reduction V0014")
    p.add_argument("--jpl-csv",default=""); p.add_argument("--output-dir",default="")
    p.add_argument("--earth-radius-mode",choices=("WGS84","IAU1976","IERS2010","CUSTOM"),default="WGS84")
    p.add_argument("--earth-radius-m",type=float); p.add_argument("--dpi",type=int,default=420)
    return p.parse_args()


def radius(a):
    if a.earth_radius_mode=="CUSTOM":
        if a.earth_radius_m is None or not math.isfinite(a.earth_radius_m) or a.earth_radius_m<=0: raise ValueError("CUSTOM requires positive --earth-radius-m")
        return "CUSTOM",float(a.earth_radius_m)
    return a.earth_radius_mode,RADII[a.earth_radius_mode]


def valid(p):
    try: return all(c in pd.read_csv(p,nrows=0).columns for c in COLS)
    except Exception: return False


def candidates():
    out=[]
    if not ROOT.exists(): return out
    for root,dirs,files in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d!="drive" and not d.startswith(".")]
        out.extend(Path(root)/f for f in files if f.lower().endswith(".csv"))
    return sorted(set(out))


def horizons(target,label):
    q={"format":"json","COMMAND":f"'{target}'","OBJ_DATA":"'NO'","MAKE_EPHEM":"'YES'","EPHEM_TYPE":"'VECTORS'","CENTER":"'500@399'","START_TIME":"'1769-Jun-03 18:00'","STOP_TIME":"'1769-Jun-04 03:00'","STEP_SIZE":"'1m'","TIME_TYPE":"'UT'","TIME_DIGITS":"'FRACSEC'","CAL_TYPE":"'GREGORIAN'","REF_PLANE":"'FRAME'","REF_SYSTEM":"'ICRF'","OUT_UNITS":"'KM-S'","VEC_TABLE":"'1'","VEC_CORR":"'NONE'","CSV_FORMAT":"'YES'","VEC_LABELS":"'NO'"}
    url=API+"?"+urlencode(q); payload=None; err=None
    for n in range(3):
        try:
            with urlopen(Request(url,headers={"User-Agent":PROGRAM}),timeout=120) as r: payload=json.loads(r.read().decode())
            break
        except Exception as e: err=e; time.sleep(n+1)
    if payload is None: raise RuntimeError(f"JPL request failed for {label}: {err}")
    if payload.get("error") or "JPL" not in str(payload.get("signature",{}).get("source","")).upper(): raise RuntimeError(f"Invalid JPL response for {label}")
    text=str(payload.get("result",""))
    if "$$SOE" not in text or "$$EOE" not in text: raise RuntimeError(f"No JPL vectors for {label}")
    rows=[]
    for raw in csv.reader(io.StringIO(text.split("$$SOE",1)[1].split("$$EOE",1)[0])):
        f=[x.strip() for x in raw if x.strip()]
        try: jd=float(f[0].replace("D","E"))
        except (IndexError,ValueError): continue
        nums=[]
        for x in f[1:]:
            try: nums.append(float(x.replace("D","E")))
            except ValueError: continue
        if len(nums)<3: raise RuntimeError(f"Cannot decode {label} vector row")
        rows.append({"JD":jd,f"GEOCENTER_{label}_X_KM":nums[0],f"GEOCENTER_{label}_Y_KM":nums[1],f"GEOCENTER_{label}_Z_KM":nums[2]})
    df=pd.DataFrame(rows).sort_values("JD").drop_duplicates("JD").reset_index(drop=True)
    if len(df)<7: raise RuntimeError(f"Only {len(df)} JPL rows for {label}")
    return df


def build_master():
    s,v=horizons("10","SUN"),horizons("299","VENUS"); s["K"]=s.JD.round(10); v["K"]=v.JD.round(10)
    m=s.merge(v.drop(columns="JD"),on="K",validate="one_to_one").drop(columns="K")[COLS]
    m.to_csv(MASTER_DEFAULT,index=False,float_format="%.15f")
    return MASTER_DEFAULT.resolve(),"OFFICIAL JPL HORIZONS API"


def locate(requested):
    if requested and valid(Path(requested).expanduser()): return Path(requested).expanduser().resolve(),"EXPLICIT JPL VECTOR CSV"
    preferred=[ROOT/"O6_TAHITI_VARDO_1769_1MIN_MASTER.csv",ROOT/"O6_1769_GEOCENTER_HORIZONS_V0013.csv",MASTER_DEFAULT]
    for p in preferred+candidates():
        if p.is_file() and valid(p): return p.resolve(),"COLAB RUNTIME JPL VECTOR CSV"
    return build_master()


def jd_text(jd):
    x=jd+0.5; z=int(math.floor(x)); f=x-z
    if z>=2299161: q=int((z-1867216.25)/36524.25); a=z+1+q-int(q/4)
    else: a=z
    b=a+1524; c=int((b-122.1)/365.25); d=int(365.25*c); e=int((b-d)/30.6001)
    dayf=b-d-int(30.6001*e)+f; day=int(dayf); month=e-1 if e<14 else e-13; year=c-4716 if month>2 else c-4715
    dt=datetime(year,month,day,tzinfo=timezone.utc)+timedelta(microseconds=round((dayf-day)*86400e6))
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]+" UTC"


def golden(fn,lo,hi):
    g=(math.sqrt(5)-1)/2; c=hi-g*(hi-lo); d=lo+g*(hi-lo); fc,fd=fn(c),fn(d)
    for _ in range(240):
        if hi-lo<=1e-6: break
        if fc<fd: hi,d,fd=d,c,fc; c=hi-g*(hi-lo); fc=fn(c)
        else: lo,c,fc=c,d,fd; d=lo+g*(hi-lo); fd=fn(d)
    x=(lo+hi)/2; return x,fn(x)


def geometry(path):
    df=pd.read_csv(path)[COLS].apply(pd.to_numeric,errors="coerce"); df=df[np.isfinite(df).all(axis=1)].sort_values("JD").drop_duplicates("JD").reset_index(drop=True)
    jd=df.JD.to_numpy(float); sun=df[COLS[1:4]].to_numpy(float); ven=df[COLS[4:7]].to_numpy(float)
    ang=np.arctan2(np.linalg.norm(np.cross(sun,ven),axis=1),np.sum(sun*ven,axis=1)); i=int(np.argmin(ang))
    if i==0 or i==len(df)-1: raise RuntimeError("Closest approach lies at JPL data boundary")
    lo,hi=max(0,i-3),min(len(df),i+4); center=jd[i]; sec=(jd[lo:hi]-center)*86400
    ps=[np.poly1d(np.polyfit(sec,sun[lo:hi,k],min(3,len(sec)-1))) for k in range(3)]; pv=[np.poly1d(np.polyfit(sec,ven[lo:hi,k],min(3,len(sec)-1))) for k in range(3)]
    ev=lambda p,t: np.array([float(f(t)) for f in p])
    def sep(t):
        a,b=ev(ps,t),ev(pv,t); return math.atan2(float(np.linalg.norm(np.cross(a,b))),float(np.dot(a,b)))
    off,a=golden(sep,(jd[i-1]-center)*86400,(jd[i+1]-center)*86400); selected=center+off/86400
    sv,vv=ev(ps,off),ev(pv,off)
    return {"frame":df,"jd":selected,"epoch":jd_text(selected),"sun":sv,"venus":vv,"distance_m":float(np.linalg.norm(sv))*1000,"sep_arcsec":a*ASEC,"offset_s":off}


def parallax(r,d):
    q=r/d
    if not 0<q<1: raise ValueError("a/D outside (0,1)")
    p=math.asin(q); return q,p,p*ASEC


def reduction(g,raw_r):
    A2=C*TAU; q,p,raw=parallax(raw_r,g["distance_m"]); fd=g["distance_m"]/A2; fr=R76/raw_r; ft=fd*fr
    linear=raw*ft; exact=math.asin(ft*math.sin(p))*ASEC; case2=parallax(R76,A2)[2]
    return {"A2":A2,"raw_rad":p,"raw":raw,"FD":fd,"FR":fr,"F2":ft,"linear":linear,"exact":exact,"case2":case2,"linear_exact_uas":(linear-exact)*1e6,"exact_case2_uas":(exact-case2)*1e6}


def cases(g,mode,raw_r,r):
    defs=[("CASE 1","IAU 1976 published",R76,AU76),("CASE 2","IAU 1976 exact cτ_A",R76,r["A2"]),("CASE 3","IAU 2012 / WGS84",RW,AU12),("IERS","IERS 2010 / IAU 2012",RIERS,AU12),("JPL RAW",f"JPL dynamic / {mode}",raw_r,g["distance_m"]),("JPL → 2","JPL reduced to IAU 1976 Case 2",R76,r["A2"])]
    rows=[]
    for cid,name,a,d in defs:
        q,p,arc=parallax(a,d)
        if cid=="JPL → 2": arc=r["exact"]; p=arc/ASEC; q=math.sin(p)
        delta=(arc-TARGET)*1e6; status="PASS" if abs(delta)<=TOL*1e6 else ("RAW" if cid=="JPL RAW" else "FAIL")
        rows.append([cid,name,a,d,q,p,arc,delta,status])
    return pd.DataFrame(rows,columns=["case","definition","a_m","D_m","ratio","pi_rad","pi_arcsec","delta_uas","class"])


def uncertainty():
    q=R76/AU76; den=math.sqrt(1-q*q); sr=abs(1/(AU76*den))*R76_SIGMA*ASEC; sa=abs(-R76/(AU76**2*den))*AU76_SIGMA*ASEC
    return sr,sa,math.hypot(sr,sa)


def vectors(g,out):
    f=g["frame"].copy(); f.insert(1,"UTC",f.JD.map(jd_text))
    paths={"sun":out/"JPL_1769_GEOCENTER_SUN_VECTORS_V0014.csv","venus":out/"JPL_1769_GEOCENTER_VENUS_VECTORS_V0014.csv","master":out/"JPL_1769_GEOCENTER_MASTER_V0014.csv"}
    f[["JD","UTC",*COLS[1:4]]].to_csv(paths["sun"],index=False,float_format="%.15f")
    f[["JD","UTC",*COLS[4:7]]].to_csv(paths["venus"],index=False,float_format="%.15f")
    f[["JD","UTC",*COLS[1:]]].to_csv(paths["master"],index=False,float_format="%.15f")
    return paths


def panel(ax,title):
    ax.set_axis_off(); ax.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=.012,rounding_size=.015",transform=ax.transAxes,lw=.7,ec="#6B7280",fc="#FFFDF8",clip_on=False,zorder=-10)); ax.text(.025,.94,title,transform=ax.transAxes,ha="left",va="top",fontsize=11.5,fontweight="bold",color="#172033")


def plate(c,g,r,u,paths,source,png,dpi):
    plt.close("all"); plt.rcParams.update({"font.family":"serif","font.serif":["STIX Two Text","DejaVu Serif"],"mathtext.fontset":"stix","figure.facecolor":"#F5F1E8","savefig.facecolor":"#F5F1E8"})
    fig=plt.figure(figsize=(18,13)); gs=fig.add_gridspec(4,2,height_ratios=[.16,.95,1.14,1.06],hspace=.16,wspace=.06,left=.035,right=.965,top=.965,bottom=.045)
    t=fig.add_subplot(gs[0,:]); t.axis("off"); t.text(.5,.76,"SOLAR HORIZONTAL PARALLAX — HISTORICAL REDUCTION AND JPL VECTOR AUDIT",ha="center",fontsize=18.5,fontweight="bold",color="#111827"); t.text(.5,.25,r"1769 Venus Transit  •  IAU 1976 Case 2  •  exact $c\tau_A$ normalization  •  geocentric JPL Horizons vectors",ha="center",fontsize=11,color="#374151")
    a1,a2,a3,a4,a5=fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1]),fig.add_subplot(gs[2,:]),fig.add_subplot(gs[3,0]),fig.add_subplot(gs[3,1])
    panel(a1,"I. DEFINING CONSTANTS"); const=[(r"$a_{1976}$",R76,"m","IAU 1976 equatorial radius"),(r"$a_{WGS84}$",RW,"m","raw JPL radius convention"),(r"$c$",C,"m s$^{-1}$","exact speed of light"),(r"$\tau_A$",TAU,"s","IAU 1976 light time"),(r"$A_1$",AU76,"m","published rounded AU"),(r"$A_2=c\tau_A$",r["A2"],"m","exact Case-2 AU"),(r"$A_{2012}$",AU12,"m","exact modern AU")]
    for x,text in zip((.05,.23,.61,.73),("Symbol","Value","Unit","Definition")): a1.text(x,.82,text,transform=a1.transAxes,fontsize=8.5,fontweight="bold")
    y=.745
    for sym,val,unit,desc in const: a1.text(.05,y,sym,transform=a1.transAxes,fontsize=9.3); a1.text(.23,y,f"{val:,.6f}",transform=a1.transAxes,fontsize=9,family="monospace"); a1.text(.61,y,unit,transform=a1.transAxes,fontsize=8.8); a1.text(.73,y,desc,transform=a1.transAxes,fontsize=8.3); y-=.088
    panel(a2,"II. JPL VECTOR GEOMETRY AT GEOCENTRIC CLOSEST APPROACH"); a2.text(.05,.84,f"Source: {source}",transform=a2.transAxes,fontsize=8.4,color="#374151")
    jrows=[("Epoch",g["epoch"],""),("Julian date",f"{g['jd']:.12f}","JD"),(r"$X_\odot$",f"{g['sun'][0]:,.6f}","km"),(r"$Y_\odot$",f"{g['sun'][1]:,.6f}","km"),(r"$Z_\odot$",f"{g['sun'][2]:,.6f}","km"),(r"$\|\mathbf{r}_{E\odot}\|$",f"{g['distance_m']/1000:,.6f}","km"),(r"$X_\mathrm{V}$",f"{g['venus'][0]:,.6f}","km"),(r"$Y_\mathrm{V}$",f"{g['venus'][1]:,.6f}","km"),(r"$Z_\mathrm{V}$",f"{g['venus'][2]:,.6f}","km"),(r"$\theta_{\odot\mathrm{V}}$",f"{g['sep_arcsec']:.9f}","arcsec")]
    y=.75
    for lab,val,unit in jrows: a2.text(.05,y,lab,transform=a2.transAxes,fontsize=9.1); a2.text(.29,y,val,transform=a2.transAxes,fontsize=8.8,family="monospace"); a2.text(.84,y,unit,transform=a2.transAxes,fontsize=8.6); y-=.065
    panel(a3,"III. EXACT REDUCTION OF THE RAW JPL PARALLAX TO IAU-1976 CASE 2")
    eq=[rf"$\pi_{{\odot,\mathrm{{JPL/WGS84}}}}=\arcsin\!\left(\frac{{a_{{WGS84}}}}{{D_{{JPL}}}}\right)={r['raw']:.12f}^{{\prime\prime}}$",rf"$F_D=\frac{{D_{{JPL}}}}{{A_2}}={r['FD']:.15f}$",rf"$F_R=\frac{{a_{{1976}}}}{{a_{{WGS84}}}}={r['FR']:.15f}$",rf"$F_2=F_DF_R={r['F2']:.15f}$",rf"$\pi_{{2,\mathrm{{linear}}}}=F_2\pi_{{\odot,\mathrm{{JPL/WGS84}}}}={r['linear']:.12f}^{{\prime\prime}}$",rf"$\pi_{{2,\mathrm{{exact}}}}=\arcsin\!\left[F_2\sin\!\left(\pi_{{\odot,\mathrm{{JPL/WGS84}}}}\right)\right]={r['exact']:.12f}^{{\prime\prime}}$",rf"$\pi_{{2,\mathrm{{exact}}}}=\arcsin\!\left(\frac{{a_{{1976}}}}{{A_2}}\right)={r['case2']:.12f}^{{\prime\prime}}$"]
    y=.80
    for n,e in enumerate(eq): a3.text(.04,y,e,transform=a3.transAxes,fontsize=13.2 if n in (0,5,6) else 12.2,color="#111827"); y-=.105
    a3.text(.68,.79,"Authoritative result",transform=a3.transAxes,fontsize=10.5,fontweight="bold",color="#7C2D12"); a3.text(.68,.68,rf"$\Delta_{{\mathrm{{exact-Case\ 2}}}}={r['exact_case2_uas']:.9f}\ \mu\mathrm{{as}}$",transform=a3.transAxes,fontsize=12.2,color="#7C2D12"); a3.text(.68,.57,rf"$\Delta_{{\mathrm{{linear-exact}}}}={r['linear_exact_uas']:.9f}\ \mu\mathrm{{as}}$",transform=a3.transAxes,fontsize=11.4,color="#374151"); a3.text(.68,.43,"The raw 1769 JPL value is instantaneous at the actual\nEarth–Sun distance. It must be reduced by both the\ndistance factor and the radius-convention factor before\ncomparison with the IAU-1976 standard.",transform=a3.transAxes,fontsize=9.2,linespacing=1.35,color="#374151",va="top")
    panel(a4,"IV. PARALLAX COMPARISON — RAW AND STANDARDIZED VALUES"); d=c.copy(); d["a_m"]=d.a_m.map(lambda x:f"{x:,.3f}"); d["D_m"]=d.D_m.map(lambda x:f"{x:,.3f}"); d["pi_arcsec"]=d.pi_arcsec.map(lambda x:f"{x:.12f}"); d["delta_uas"]=d.delta_uas.map(lambda x:f"{x:+.6f}"); d=d[["case","definition","a_m","D_m","pi_arcsec","delta_uas","class"]]; d.columns=["Case","Definition","a (m)","D or A (m)","π⊙ (arcsec)","Δ (µas)","Class"]
    tab=a4.table(cellText=d.values,colLabels=d.columns,cellLoc="left",colLoc="center",bbox=[.02,.08,.96,.78],colWidths=[.08,.24,.13,.20,.15,.11,.07]); tab.set_zorder(5); tab.auto_set_font_size(False); tab.set_fontsize(8.1)
    for (row,col),cell in tab.get_celld().items(): cell.set_linewidth(.35); cell.set_edgecolor("#6B7280"); cell.set_facecolor("#E9EEF5" if row==0 else ("#F8FAFC" if row%2==0 else "#FFFDF8")); cell.set_text_props(weight="bold" if row==0 else "normal",color="#111827")
    panel(a5,"V. HISTORICAL UNCERTAINTY AND TRACEABILITY"); sr,sa,st=u; ul=[rf"$\sigma_{{\pi,a}}={sr*1e6:.6f}\ \mu\mathrm{{as}}$",rf"$\sigma_{{\pi,A}}={sa*1e6:.6f}\ \mu\mathrm{{as}}$",rf"$\sigma_\pi=\sqrt{{\sigma_{{\pi,a}}^2+\sigma_{{\pi,A}}^2}}={st:.12f}^{{\prime\prime}}$",rf"$\sigma_\pi\longrightarrow\pm {TOL:.6f}^{{\prime\prime}}$"]
    y=.78
    for e in ul: a5.text(.05,y,e,transform=a5.transAxes,fontsize=11.2); y-=.13
    a5.text(.52,.78,"Generated JPL vector files",transform=a5.transAxes,fontsize=10,fontweight="bold")
    for y,key in zip((.66,.54,.42),("sun","venus","master")): a5.text(.52,y,paths[key].name,transform=a5.transAxes,fontsize=8.5,family="monospace")
    a5.text(.52,.22,"All displayed quantities derive from defining constants\nor minute-by-minute JPL vectors. No manual parallax\nresult enters the calculation chain.",transform=a5.transAxes,fontsize=8.8,linespacing=1.35,color="#374151")
    fig.text(.5,.015,"Figure V0014. Exact reduction of instantaneous 1769 JPL solar parallax to the IAU-1976 Case-2 standard. Linear multiplication is shown for audit continuity; the arcsine transformation is the exact spherical result.",ha="center",fontsize=8.5,color="#374151")
    fig.savefig(png,dpi=max(240,int(dpi)),bbox_inches="tight",pad_inches=.08); plt.close(fig)


def main():
    a=cli(); mode,raw_r=radius(a); out=Path(a.output_dir).expanduser().resolve() if a.output_dir else OUT_DEFAULT; out.mkdir(parents=True,exist_ok=True)
    master,source=locate(a.jpl_csv); g=geometry(master); r=reduction(g,raw_r); c=cases(g,mode,raw_r,r); u=uncertainty(); paths=vectors(g,out)
    cases_csv=out/"SOLAR_PARALLAX_AUDIT_V0014_CASES.csv"; reduction_csv=out/"SOLAR_PARALLAX_AUDIT_V0014_REDUCTION.csv"; png=out/"SOLAR_PARALLAX_REDUCTION_V0014_PUBLICATION.png"
    c.to_csv(cases_csv,index=False,float_format="%.15f"); pd.DataFrame([r]).to_csv(reduction_csv,index=False,float_format="%.15f"); plate(c,g,r,u,paths,source,png,a.dpi)
    checks=[abs(g["distance_m"]-float(np.linalg.norm(g["sun"]))*1000)<=1e-6,abs(r["exact_case2_uas"])<=1e-6,round(r["case2"],6)==TARGET,round(u[2],6)==TOL]
    if not all(checks): raise RuntimeError("Equation-status check failed")
    try:
        from IPython.display import Image,display
        display(Image(filename=str(png)))
    except Exception: print(f"PUBLICATION IMAGE: {png}")
    print(f"CODE OUTPUT: {VERSION}"); print(f"PUBLICATION IMAGE: {png}"); print(f"JPL SUN VECTORS: {paths['sun']}"); print(f"JPL VENUS VECTORS: {paths['venus']}"); print(f"JPL COMBINED MASTER: {paths['master']}"); print(f"REDUCTION CSV: {reduction_csv}"); print(f"LOCAL TIMESTAMP: {datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}"); print(f"# {VERSION}")


if __name__=="__main__": main()
# V0014
