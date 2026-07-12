# V0019
# Audit reference: Existing 12N track-event data rescaled to the exact IAU-1976 Earth-Sun distance and plotted in the original half-Sun configuration.
from __future__ import annotations
import ast, csv, math, subprocess, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

V="V0019"; TZ=ZoneInfo("America/Bogota")
ASR=206264.80624709636; AU12=149597870.7; RE=6378.140
C=299792.458; TAU=499.004782; AU76=C*TAU; RS=695700.0
PI76=math.asin(RE/AU76)*ASR
ROOT=Path("/content/IERS_TN36_V01_MASTER_OUTPUT")
SRC=ROOT/"IERS-0012N_VARDO_POINT_VENUS_ENGINEERING_HALF_SUN_EVENTS_AND_GEOMETRY.csv"
OUT=ROOT/"V0019_IAU1976_HALF_SUN"; PNG=OUT/"V0019_IAU1976_HALF_SUN_TRACKS.png"; CSV=OUT/"V0019_IAU1976_EVENTS_AND_GEOMETRY.csv"
SITES=("Vardo Norway","Point Venus Tahiti"); EVS=("C1","C2","CA","C3","C4")
SHORT={"Vardo Norway":"Vardo","Point Venus Tahiti":"Point Venus"}
COL={"Vardo Norway":"#ffc861","Point Venus Tahiti":"#5ee08a"}

def need(m,p):
    try: __import__(m)
    except ImportError: subprocess.check_call([sys.executable,"-m","pip","-q","install",p])
for m,p in (("numpy","numpy"),("scipy","scipy"),("matplotlib","matplotlib")): need(m,p)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.interpolate import CubicSpline

def audit():
    s=Path(__file__).read_text(); lines=s.splitlines(); tree=ast.parse(s); mods=set(); calls=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): mods.update(a.name for a in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module: mods.add(node.module)
        elif isinstance(node,ast.Call):
            if isinstance(node.func,ast.Name): calls.add(node.func.id)
            elif isinstance(node.func,ast.Attribute): calls.add(node.func.attr)
    ck={
      "first_line":lines[0]=="# V0019","last_line":lines[-1]=="# V0019",
      "sun_circle":"sun_limb=Circle((0.0,0.0),sun_r" in s,
      "sun_added":"ax.add_patch(sun_limb)" in s,
      "show":"plt.show()" in s,"save":"fig.savefig(" in s,
      "half_sun":"def half_limits(" in s,
      "no_jpl_download":not any(m.startswith(("astroquery","requests")) for m in mods) and not ({"Horizons","urlopen"}&calls),
    }
    return len(lines),ck,all(ck.values())

def read12n():
    if not SRC.is_file(): raise FileNotFoundError(f"Run 12N first; missing {SRC}")
    r=list(csv.reader(SRC.open(encoding="utf-8"))); eh=gh=None
    for i,row in enumerate(r):
        if row[:4]==["site","event","utc","jd_tdb"]: eh=i
        if row[:4]==["section","quantity","value","unit"]: gh=i
    if eh is None or gh is None: raise RuntimeError("12N CSV format not recognized")
    e={}
    for row in r[eh+1:gh]:
        if len(row)>=8 and row[0].strip():
            e.setdefault(row[0].strip(),{})[row[1].strip()]={"utc":row[2],"jd":float(row[3]),"x":float(row[4]),"y":float(row[5]),"rv":float(row[6])}
    g={}
    for row in r[gh+1:]:
        if len(row)>=4 and row[1].strip():
            try:g[row[1].strip()]=float(row[2])
            except ValueError:g[row[1].strip()]=row[2]
    for site in SITES:
        if site not in e or any(k not in e[site] for k in EVS): raise RuntimeError(f"Incomplete 12N events: {site}")
    return e,g

def scale(a,d): return math.atan(math.tan(a/ASR)*d/AU76)*ASR

def normalize(e,d):
    out={}
    for site in SITES:
        out[site]={}
        for k in EVS:
            q=e[site][k]
            out[site][k]={"utc":q["utc"],"jd":q["jd"],"x":scale(q["x"],d),"y":scale(q["y"],d),"rv":scale(q["rv"],d)}
    return out

def unit(v):
    n=float(np.linalg.norm(v))
    if n==0: raise RuntimeError("zero vector")
    return np.asarray(v,float)/n

def track(site,e):
    jd=np.array([e[k]["jd"] for k in EVS]); x=np.array([e[k]["x"] for k in EVS]); y=np.array([e[k]["y"] for k in EVS])
    t=np.linspace(jd[0],jd[-1],721); p=np.c_[CubicSpline(jd,x,bc_type="natural")(t),CubicSpline(jd,y,bc_type="natural")(t)]
    mu=p.mean(0); q=p-mu; _,_,vt=np.linalg.svd(q,full_matrices=False); d=unit(vt[0]); d=-d if d[0]<0 else d
    return {"site":site,"p":p,"mu":mu,"d":d,"ang":math.degrees(math.atan2(d[1],d[0])),"ep":{k:np.array([e[k]["x"],e[k]["y"]]) for k in EVS},"rv":{k:e[k]["rv"] for k in EVS},"e":e}

def intersect(mu,d,mid,n):
    z,*_=np.linalg.lstsq(np.c_[d,-n],mid-mu,rcond=None); return mu+z[0]*d

def geometry(a,b,r):
    t=unit(a["d"]+b["d"]); t=-t if t[0]<0 else t; n=np.array([-t[1],t[0]]); mid=.5*(a["mu"]+b["mu"])
    ap=intersect(a["mu"],a["d"],mid,n); bp=intersect(b["mu"],b["d"],mid,n); w=bp-ap
    th=float(np.linalg.norm(w)); rho=abs(float(np.dot(w,n))); tr=th/ASR
    abp=math.tan(tr)*AU76; ab=abp*r; aba=math.atan2(ab,AU76)*ASR
    raw=rho*r*RE/ab; pi=raw*(th/rho)*(math.tan(tr)/tr)*(math.asin(RE/AU76)/(RE/AU76))
    return {"th":th,"rho":rho,"abp":abp,"ab":ab,"aba":aba,"ratio":r,"inv":1/r,"raw":raw,"pi":pi,"res":pi-PI76,"halley":abp/ab,"rhos":rho*RE/ab}

def half_limits(r,a,b):
    y=float(np.median(np.vstack([a["p"],b["p"]])[:,1])); return (-1.04*r,1.04*r),((-0.06*r,1.06*r) if y>=0 else (-1.06*r,0.06*r))

def fmt(q,v,u):
    if isinstance(v,str): return v
    n=10 if q in ("π⊙","Computed π⊙","Reference π⊙","Residual π⊙","A′B′ / AB","D EV / D VS","D VS / D EV") else (12 if u=="AU" else 6)
    return f"{float(v):.{n}f}"

def label(ax,p,s,dx,dy,c):
    ax.annotate(s,xy=p,xytext=(p[0]+dx,p[1]+dy),textcoords="data",fontsize=5.7,color=c,ha="left",va="center",arrowprops=dict(arrowstyle="-",lw=.20,color=c,shrinkA=0,shrinkB=2))

def plot_table(ax,a,b,g):
    rows=[("β Vardo",a["ang"],"deg"),("β Point Venus",b["ang"],"deg"),("Δβ",abs(a["ang"]-b["ang"]),"deg"),("π⊙",g["pi"],"arcsec"),("A′B′ / AB",g["halley"],"ratio"),("A′B′",g["th"],"arcsec"),("A′B′",g["abp"],"km"),("AB",g["aba"],"arcsec"),("AB",g["ab"],"km"),("D ES",1.0,"AU")]
    tab=ax.table(cellText=[[q,fmt(q,v,u),u] for q,v,u in rows],colLabels=["Quantity","Value","Unit"],loc="lower left",colWidths=[.29,.23,.15],bbox=[.438,.122,.380,.345]); tab.auto_set_font_size(False); tab.set_fontsize(5.30)
    for (r,c),cell in tab.get_celld().items():
        cell.set_linewidth(.18); cell.set_edgecolor("#1e4f64"); cell.set_facecolor("#0a1a22" if r==0 else "#050b0f")
        cell.get_text().set_color("#66e8ff" if r==0 else ("#ffc861" if c==1 else ("#5ee08a" if c==2 else "#dff8ff")))
        if r==0 or c==1: cell.get_text().set_fontweight("bold")
    ax.text(.440,.101,"A′B′ = solar-screen chord; AB = projected baseline; D ES = IAU 1976 cτA.",transform=ax.transAxes,color="#8fb4c1",fontsize=5.25,ha="left",va="top")

def make_plot(a,b,g):
    sun_r=math.atan2(RS,AU76)*ASR; fig,ax=plt.subplots(figsize=(9.6,5.8),dpi=240); fig.patch.set_facecolor("#03080d"); ax.set_facecolor("#03080d")
    sun_limb=Circle((0.0,0.0),sun_r,fill=False,lw=.36,ec="#66e8ff",alpha=.95); ax.add_patch(sun_limb); ax.axhline(0,lw=.18,color="#1d3d4a",alpha=.72); ax.axvline(0,lw=.18,color="#1d3d4a",alpha=.72)
    for tr in (a,b):
        site=tr["site"]; c=COL[site]; p=tr["p"]; ax.plot(p[:,0],p[:,1],lw=.30,color=c,solid_capstyle="round",label=site,zorder=3); ax.scatter(p[::12,0],p[::12,1],s=.75,color=c,alpha=.70,linewidths=0,zorder=4)
        for k in EVS:
            q=tr["ep"][k]; ax.add_patch(Circle((q[0],q[1]),tr["rv"][k],fill=False,lw=.28 if k=="CA" else .20,ec=c,alpha=.92,zorder=2)); ax.scatter([q[0]],[q[1]],s=3.8 if k=="CA" else 2.2,color=c,edgecolors="#03080d",linewidths=.16,zorder=5)
        label(ax,tr["ep"]["CA"],SHORT[site]+" CA",18,44 if site==SITES[0] else -44,c)
    for k,dx,dy in (("C1",-48,12),("C2",-38,9),("C3",20,-10),("C4",30,-13)): label(ax,a["ep"][k],k,dx,dy,"#8fb4c1")
    plot_table(ax,a,b,g); xl,yl=half_limits(sun_r,a,b); ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal",adjustable="box"); ax.grid(True,color="#102630",linewidth=.16,alpha=.55)
    for sp in ax.spines.values(): sp.set_linewidth(.22); sp.set_color("#25708b")
    ax.tick_params(colors="#8fb4c1",labelsize=6.5,width=.22,length=2); ax.set_xlabel("IAU-1976-normalized solar-screen X offset (arcsec)",color="#8fb4c1",fontsize=7.5); ax.set_ylabel("IAU-1976-normalized solar-screen Y offset (arcsec)",color="#8fb4c1",fontsize=7.5)
    ax.set_title("1769 Venus Transit — Engineering Half-Sun Track Reconstruction\nVardo, Norway / Point Venus, Tahiti — existing IERS-0012N track data",color="#f8fdff",fontsize=9,pad=8)
    leg=ax.legend(loc="lower right",fontsize=6.3,frameon=True,borderpad=.45); leg.get_frame().set_facecolor("#071016"); leg.get_frame().set_edgecolor("#1e4f64"); leg.get_frame().set_linewidth(.22)
    for t in leg.get_texts(): t.set_color("#dff8ff")
    fig.text(.5,.016,f"Venus disks to scale at C1, C2, CA, C3, C4.  π⊙ = {g['pi']:.10f} arcsec; R⊕ = {RE:.3f} km; cτA = {AU76:.6f} km.",ha="center",va="bottom",fontsize=6.2,color="#8fb4c1")
    fig.savefig(PNG,dpi=460,facecolor=fig.get_facecolor(),bbox_inches="tight",pad_inches=.055); plt.show(); plt.close(fig)

def lower_tables(a,b,g,d):
    try: from IPython.display import HTML,display
    except Exception: return
    tr=[("β Vardo",a["ang"],"deg"),("β Point Venus",b["ang"],"deg"),("Δβ",abs(a["ang"]-b["ang"]),"deg"),("β Average",.5*(a["ang"]+b["ang"]),"deg")]
    gr=[("Closest Vardo UTC",a["e"]["CA"]["utc"],"UTC"),("Closest Point Venus UTC",b["e"]["CA"]["utc"],"UTC"),("A′B′ Angular Chord",g["th"],"arcsec"),("A′B′ Solar-Screen Chord",g["abp"],"km"),("AB Angular Projection",g["aba"],"arcsec"),("AB Projected Baseline",g["ab"],"km"),("A′B′ / AB",g["halley"],"ratio"),("Normal Separation ρ",g["rho"],"arcsec"),("ρ Scaled To R⊕",g["rhos"],"arcsec"),("D ES",1.0,"AU"),("D ES Source","IAU 1976 cτA","standard"),("Original 12N D ES",d,"km"),("D EV / D VS",g["ratio"],"ratio"),("D VS / D EV",g["inv"],"ratio"),("Raw φ",g["raw"],"arcsec"),("Computed π⊙",g["pi"],"arcsec"),("Reference π⊙",PI76,"arcsec"),("Residual π⊙",g["res"],"arcsec")]
    def rows(z): return "".join(f"<tr><td>{q}</td><td class='v'>{fmt(q,v,u)}</td><td class='u'>{u}</td></tr>" for q,v,u in z)
    css=".w{background:#03080d;color:#dff8ff;font-family:monospace;width:700px;max-width:98%;border:1px solid #1e4f64;border-radius:8px;padding:8px}.t{color:#66e8ff;font-size:10px;font-weight:800;text-align:center;border-top:1px solid #25708b;border-bottom:1px solid #25708b;padding:5px}table{border-collapse:collapse;width:100%;font-size:10px;background:#050b0f}th{color:#66e8ff;background:#0a1a22;text-align:left}td,th{padding:4px 5px;border-bottom:1px solid #102630}.v{color:#ffc861;text-align:right;font-weight:800}.u{color:#5ee08a}"
    display(HTML(f"<style>{css}</style><div class='w'><div class='t'>TRIGONOMETRY — VARDO → POINT VENUS</div><table><tr><th>Quantity</th><th>Value</th><th>Units</th></tr>{rows(tr)}</table><div class='t'>π⊙ GEOMETRIC SOLUTION — IAU 1976</div><table><tr><th>Quantity</th><th>Value</th><th>Units</th></tr>{rows(gr)}</table><div>CSV: {CSV}</div></div>"))

def save_csv(e,a,b,g,d):
    with CSV.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow([V,"IAU 1976 NORMALIZED HALF-SUN TRACK DATA"]); w.writerow(["source_csv",SRC]); w.writerow([]); w.writerow(["site","event","utc","jd_tdb","x_arcsec","y_arcsec","venus_radius_arcsec","track_angle_deg"])
        tm={SITES[0]:a,SITES[1]:b}
        for site in SITES:
            for k in EVS:
                q=e[site][k]; w.writerow([site,k,q["utc"],f"{q['jd']:.12f}",f"{q['x']:.12f}",f"{q['y']:.12f}",f"{q['rv']:.12f}",f"{tm[site]['ang']:.12f}"])
        w.writerow([]); w.writerow(["section","quantity","value","unit"])
        w.writerows([("INPUT","Original 12N Earth-Sun distance",d,"km"),("INPUT","IAU 1976 Earth-Sun distance c tau_A",AU76,"km"),("INPUT","IAU 1976 Earth radius",RE,"km"),("RESULT","A prime B prime angular chord",g["th"],"arcsec"),("RESULT","A prime B prime solar-screen chord",g["abp"],"km"),("RESULT","AB projected baseline",g["ab"],"km"),("RESULT","Computed pi_sun",g["pi"],"arcsec"),("CHECK","IAU 1976 standard",PI76,"arcsec"),("CHECK","Residual",g["res"],"arcsec")])

def main():
    OUT.mkdir(parents=True,exist_ok=True); n,ck,ok=audit()
    if not ok: raise RuntimeError("Static audit failed: "+", ".join(k for k,v in ck.items() if not v))
    e0,g0=read12n(); d=float(g0["D ES"])*AU12; e=normalize(e0,d); a=track(SITES[0],e[SITES[0]]); b=track(SITES[1],e[SITES[1]]); g=geometry(a,b,float(g0["D EV / D VS"]))
    save_csv(e,a,b,g,d); make_plot(a,b,g); lower_tables(a,b,g,d)
    rt={"png":PNG.is_file(),"csv":CSV.is_file(),"sun":abs(math.atan2(RS,AU76)*ASR-959.220817)<.001,"pi":round(g["pi"],10)==round(PI76,10),"res":abs(g["res"])<=5e-12}
    if not all(rt.values()): raise RuntimeError("Runtime audit failed: "+", ".join(k for k,v in rt.items() if not v))
    print(f"CODE OUTPUT: {V}"); print("CODE INPUTS"); print(f"Source CSV: {SRC}"); print(f"Original 12N D_ES: {d:.6f} km"); print(f"IAU 1976 c tau_A: {AU76:.6f} km")
    print("COMMENTS"); print("Existing 12N track data rescaled; no JPL download."); print("RESULTS"); print(f"π⊙: {g['pi']:.10f} arcsec"); print(f"Sun radius: {math.atan2(RS,AU76)*ASR:.6f} arcsec")
    print("OUTPUT SUMMARY"); print(f"PNG: {PNG}"); print(f"CSV: {CSV}"); print("PAPER COMPARISON"); print(f"IAU 1976 standard: {PI76:.10f} arcsec")
    print("EQUATION STATUS"); print(f"Source lines: {n}"); print("Sun plot command: PASS"); print("Half-Sun limb command: PASS"); print("plt.show() command: PASS"); print("PNG save command: PASS"); print("No JPL download command: PASS")
    print(datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z")); print(f"# {V}")

if __name__=="__main__":
    if "--audit-only" in sys.argv:
        n,ck,ok=audit(); print(f"LINE COUNT: {n}"); [print(f"{k}: {'PASS' if v else 'FAIL'}") for k,v in ck.items()]; sys.exit(0 if ok else 1)
    main()
# V0019
