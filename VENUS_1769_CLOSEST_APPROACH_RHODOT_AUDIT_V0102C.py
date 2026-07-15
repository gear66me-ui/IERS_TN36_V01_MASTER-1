# V0102C
# Audit reference: V0102 plot preserved; closest approach is the raw d rho/dt zero.
from __future__ import annotations
import math, subprocess, sys, time, warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
VERSION="V0102C"
LOCAL_TZ=ZoneInfo("America/Bogota")
OUT=Path("/content/VENUS_1769_CLOSEST_APPROACH_RHODOT_AUDIT_V0102C_OUTPUT")
PNG=OUT/"VENUS_1769_CLOSEST_APPROACH_RHODOT_AUDIT_V0102C.png"
CSV=OUT/"VENUS_1769_CLOSEST_APPROACH_RHODOT_AUDIT_V0102C.csv"
ARCSEC_PER_RAD=206264.80624709636
AU_KM=149597870.7
START="1769-06-03 21:30"; STOP="1769-06-03 23:15"; STEP="1m"; WINDOW_MIN=30.0
SUN_TARGET="10"; VENUS_TARGET="299"; GEOCENTER_LOCATION="500@399"
BG="#000000"; FG="#F8FAFC"; MUTED="#B8CBD6"; GRID="#263A4B"
BLUE="#42D7C3"; GOLD="#D89B18"; GREEN="#74D680"; RED="#FF6B6B"
TABLE_HEADER="#23466F"; TABLE_BODY="#101A2E"; TABLE_TEAL="#164B55"; TABLE_GOLD="#563B0B"
def require(a,b):
    try: __import__(a)
    except ImportError: subprocess.check_call([sys.executable,"-m","pip","-q","install",b])
for a,b in (("numpy","numpy"),("pandas","pandas"),("scipy","scipy"),("astropy","astropy"),("astroquery","astroquery"),("matplotlib","matplotlib"),("IPython","ipython")): require(a,b)
import matplotlib
matplotlib.use("Agg",force=True)
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from IPython.display import Image,display
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq,minimize_scalar
warnings.filterwarnings("ignore",message=".*id_type.*deprecated.*")
warnings.filterwarnings("ignore",message=".*dubious year.*")
def norm(v): return float(np.linalg.norm(np.asarray(v,dtype=float)))
def unit(v):
    a=np.asarray(v,dtype=float); n=norm(a)
    if n<=0: raise RuntimeError("Zero vector cannot be normalized.")
    return a/n
def utc_from_jd(jd): return Time(float(jd),format="jd",scale="tdb").utc.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
def download(prefix,target_id):
    last=None
    for attempt in range(4):
        try:
            raw=Horizons(id=target_id,location=GEOCENTER_LOCATION,epochs={"start":START,"stop":STOP,"step":STEP},id_type=None).vectors(refplane="ecliptic",aberrations="geometric",cache=False).to_pandas()
            df=pd.DataFrame({"JD_TDB":pd.to_numeric(raw["datetime_jd"],errors="coerce")})
            for ax in "xyz": df[f"{prefix}_{ax.upper()}_KM"]=pd.to_numeric(raw[ax],errors="coerce")*AU_KM
            return df.dropna().drop_duplicates("JD_TDB").sort_values("JD_TDB").reset_index(drop=True)
        except Exception as exc:
            last=exc; time.sleep(1.5*(attempt+1))
    raise RuntimeError(f"JPL Horizons download failed for {prefix}: {last}")
def build_master():
    m=download("SUN",SUN_TARGET).merge(download("VENUS",VENUS_TARGET),on="JD_TDB",how="inner")
    if len(m)<80: raise RuntimeError(f"Insufficient JPL samples: {len(m)}")
    return m
def splines(df):
    jd=df["JD_TDB"].to_numpy(float); c={"JD_TDB":jd}
    for col in df.columns:
        if col!="JD_TDB": c[col]=CubicSpline(jd,df[col].to_numpy(float),bc_type="natural")
    return c
def vec(c,p,jd): return np.array([float(c[f"{p}_X_KM"](jd)),float(c[f"{p}_Y_KM"](jd)),float(c[f"{p}_Z_KM"](jd))])
def angle_arcsec_between(a,b):
    ua,ub=unit(a),unit(b)
    return ARCSEC_PER_RAD*math.atan2(norm(np.cross(ua,ub)),float(np.dot(ua,ub)))
def rho_arcsec(c,jd): return angle_arcsec_between(vec(c,"SUN",jd),vec(c,"VENUS",jd))
def rho_dot_arcsec_per_min(c,jd):
    h=0.5/1440.0
    return rho_arcsec(c,jd+h)-rho_arcsec(c,jd-h)
def ratio_ev_es(c,jd): return norm(vec(c,"VENUS",jd))/norm(vec(c,"SUN",jd))
def solve_seed(c):
    jds=np.asarray(c["JD_TDB"],float); ys=np.array([rho_arcsec(c,float(j)) for j in jds])
    i=int(np.argmin(ys)); lo=float(jds[max(0,i-2)]); hi=float(jds[min(len(jds)-1,i+2)])
    r=minimize_scalar(lambda x:rho_arcsec(c,float(x)),bounds=(lo,hi),method="bounded",options={"xatol":1e-13,"maxiter":600})
    if not r.success: raise RuntimeError("rho minimum seed failed")
    return float(r.x)
def solve_ca(c):
    seed=solve_seed(c); lo=seed-5/1440.0; hi=seed+5/1440.0
    if rho_dot_arcsec_per_min(c,lo)*rho_dot_arcsec_per_min(c,hi)>0: raise RuntimeError("d rho/dt zero not bracketed")
    return float(brentq(lambda x:rho_dot_arcsec_per_min(c,float(x)),lo,hi,xtol=1e-14,rtol=1e-14,maxiter=200))
def analyze():
    c=splines(build_master()); ca=solve_ca(c); grid=np.linspace(-WINDOW_MIN,WINDOW_MIN,61)
    rho0=rho_arcsec(c,ca); ratio0=ratio_ev_es(c,ca); rows=[]
    for minute in grid:
        jd=ca+float(minute)/1440.0; rho=rho_arcsec(c,jd); ratio=ratio_ev_es(c,jd)
        rows.append({"minute_from_true_rho_min":float(minute),"jd_tdb":jd,"utc":utc_from_jd(jd),"rho_arcsec":rho,"rho_excess_arcsec":rho-rho0,"rho_dot_arcsec_per_min":rho_dot_arcsec_per_min(c,jd),"ev_es_ratio":ratio,"ev_es_ratio_delta_ppm":(ratio/ratio0-1)*1e6})
    df=pd.DataFrame(rows)
    stats={"ca_jd_tdb":ca,"ca_utc":utc_from_jd(ca),"rho_min_arcsec":rho0,"rhodot_at_ca":rho_dot_arcsec_per_min(c,ca),"rhodot_zero_offset_sec":0.0,"rho_minus30_excess":float(df.loc[df.minute_from_true_rho_min==-30,"rho_excess_arcsec"].iloc[0]),"rho_plus30_excess":float(df.loc[df.minute_from_true_rho_min==30,"rho_excess_arcsec"].iloc[0]),"ev_es_ppm_range":float(df.ev_es_ratio_delta_ppm.max()-df.ev_es_ratio_delta_ppm.min()),"samples":len(df)}
    OUT.mkdir(parents=True,exist_ok=True); df.to_csv(CSV,index=False,float_format="%.15f")
    return df,stats
def style_axis(ax):
    ax.grid(True,color=GRID,linewidth=.30,alpha=.58); ax.tick_params(labelsize=7.2,width=.35,length=2.4)
    for s in ax.spines.values(): s.set_color(MUTED); s.set_linewidth(.35)
def style_table(table,teal_rows=(),gold_rows=(),fontsize=6.0):
    table.auto_set_font_size(False)
    for (row,_),cell in table.get_celld().items():
        cell.set_edgecolor("#70879A"); cell.set_linewidth(.30); cell.get_text().set_color(FG); cell.get_text().set_fontsize(fontsize)
        if row==0: cell.set_facecolor(TABLE_HEADER); cell.get_text().set_fontweight("bold")
        elif row in teal_rows: cell.set_facecolor(TABLE_TEAL); cell.get_text().set_fontweight("bold")
        elif row in gold_rows: cell.set_facecolor(TABLE_GOLD); cell.get_text().set_fontweight("bold")
        else: cell.set_facecolor(TABLE_BODY)
def plot(df,stats):
    plt.close("all"); plt.rcParams.update({"font.family":"DejaVu Serif","figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG,"text.color":FG,"axes.labelcolor":FG,"xtick.color":MUTED,"ytick.color":MUTED,"axes.edgecolor":MUTED})
    x=df.minute_from_true_rho_min.to_numpy(float); rho=df.rho_arcsec.to_numpy(float); excess=df.rho_excess_arcsec.to_numpy(float); rd=df.rho_dot_arcsec_per_min.to_numpy(float); ppm=df.ev_es_ratio_delta_ppm.to_numpy(float)
    fig=plt.figure(figsize=(16,9),facecolor=BG); gs=fig.add_gridspec(4,1,height_ratios=[.32,.20,.20,.28],left=.060,right=.985,top=.895,bottom=.095,hspace=.280)
    ax1=fig.add_subplot(gs[0]); ax2=fig.add_subplot(gs[1],sharex=ax1); ax3=fig.add_subplot(gs[2],sharex=ax1); tax=fig.add_subplot(gs[3])
    fig.suptitle("1769 Venus Transit — Clean Geocentric Closest-Approach Audit",fontsize=15.2,fontweight="bold",y=.955)
    fig.text(.5,.925,"Fresh JPL Horizons geometric geocentric vectors. The raw d rho/dt zero defines t = 0 and closest approach.",ha="center",fontsize=7.5,color=MUTED)
    ax1.plot(x,rho,color=GOLD,linewidth=.62,label="rho(t): Venus–Sun center distance"); ax1.scatter(x,rho,s=5,color=GOLD,edgecolors="none",alpha=.72); ax1.axvline(0,color=BLUE,linewidth=.58,alpha=.85,label="true d rho/dt-zero CA"); ax1.scatter([0],[stats["rho_min_arcsec"]],marker="D",s=64,color=BLUE,edgecolors=FG,linewidths=.35,zorder=8)
    ax1.annotate(f"true geocentric CA\n{stats['ca_utc']}\nρ = {stats['rho_min_arcsec']:.12f}″",xy=(0,stats["rho_min_arcsec"]),xytext=(3.5,stats["rho_min_arcsec"]+.75),fontsize=7,color=FG,arrowprops={"arrowstyle":"-","linewidth":.30,"color":FG}); ax1.set_title("PHYSICAL DISTANCE CURVE: d rho/dt ZERO DEFINES CLOSEST APPROACH",fontsize=10,fontweight="bold"); ax1.set_ylabel("rho arcsec",fontsize=8.5); ax1.legend(loc="upper right",fontsize=6.8,frameon=False); style_axis(ax1)
    ax2.plot(x,rd,color=RED,linewidth=.62,label="raw d rho/dt"); ax2.axhline(0,color=MUTED,linewidth=.42,alpha=.75); ax2.axvline(0,color=BLUE,linewidth=.58,alpha=.85); ax2.scatter([0],[stats["rhodot_at_ca"]],marker="X",s=52,color=BLUE,edgecolors=FG,linewidths=.30,zorder=8); ax2.annotate(f"dρ/dt at CA = {stats['rhodot_at_ca']:+.3e} ″/min",xy=(0,stats["rhodot_at_ca"]),xytext=(-28,max(rd)*.64),fontsize=7,color=FG,arrowprops={"arrowstyle":"-","linewidth":.28,"color":FG}); ax2.set_title("RAW DERIVATIVE CHECK: d rho/dt CROSSES ZERO AT TRUE CA",fontsize=10,fontweight="bold"); ax2.set_ylabel("arcsec/min",fontsize=8.5); ax2.legend(loc="upper right",fontsize=6.8,frameon=False); style_axis(ax2)
    ax3.plot(x,excess,color=GREEN,linewidth=.62,label="rho(t) − rho at CA"); ax3.axvline(0,color=BLUE,linewidth=.58,alpha=.85); ax3.set_ylabel("rho excess arcsec",fontsize=8.5); ax3.set_xlabel("Minutes from true geocentric d rho/dt-zero closest approach",fontsize=8.8); ax3b=ax3.twinx(); ax3b.plot(x,ppm,color=GOLD,linewidth=.48,linestyle=":",label="EV/ES ratio change"); ax3b.tick_params(labelsize=7.2,colors=MUTED); ax3b.set_ylabel("EV/ES ppm",fontsize=8.2,color=MUTED); ax3.set_title("SCALE RATIO CHANGES SLOWLY; CA IS SELECTED BY d rho/dt = 0",fontsize=10,fontweight="bold"); l1,a1=ax3.get_legend_handles_labels(); l2,a2=ax3b.get_legend_handles_labels(); ax3.legend(l1+l2,a1+a2,loc="upper right",fontsize=6.8,frameon=False); style_axis(ax3)
    tax.axis("off"); rows=[["Quantity","Value","Unit / status"],["True geocentric d rho/dt-zero CA UTC",stats["ca_utc"],"JPL solve: raw derivative root"],["rho at CA",f"{stats['rho_min_arcsec']:.12f}","arcsec"],["Raw d rho/dt at CA",f"{stats['rhodot_at_ca']:+.15e}","arcsec/min; zero by construction"],["Raw d rho/dt zero offset",f"{stats['rhodot_zero_offset_sec']:+.9f}","seconds from CA"],["rho excess at −30 min",f"{stats['rho_minus30_excess']:.12f}","arcsec above CA"],["rho excess at +30 min",f"{stats['rho_plus30_excess']:.12f}","arcsec above CA"],["EV/ES ratio range over ±30 min",f"{stats['ev_es_ppm_range']:.12f}","ppm"],["Samples",str(stats["samples"]),"one-minute plotted window"]]
    table=tax.table(cellText=rows,cellLoc="left",colWidths=[.30,.40,.30],bbox=[0,.05,1,.88]); style_table(table,teal_rows=(1,2,3,4),gold_rows=(5,6,7,8),fontsize=6.1); fig.text(.5,.043,f"File: {Path(__file__).name} | Output: {PNG.name} | CSV: {CSV.name}",ha="center",fontsize=5.9,color=MUTED); fig.savefig(PNG,dpi=220,facecolor=BG); display(Image(filename=str(PNG)))
def main():
    print("CODE INPUTS"); print(f"Version: {VERSION}"); print(f"UTC window: {START} to {STOP}; step: {STEP}"); print(f"Observer: Earth geocenter {GEOCENTER_LOCATION}"); print("Data source: fresh JPL Horizons geometric ecliptic vectors"); print("COMMENTS"); print("V0102 format preserved. Closest approach is redefined as the raw d rho/dt zero.")
    df,stats=analyze(); plot(df,stats)
    print("RESULTS"); print(f"True geocentric d rho/dt-zero CA UTC: {stats['ca_utc']}"); print(f"rho at CA: {stats['rho_min_arcsec']:.12f} arcsec"); print(f"Raw d rho/dt at CA: {stats['rhodot_at_ca']:+.15e} arcsec/min"); print(f"Raw d rho/dt zero offset: {stats['rhodot_zero_offset_sec']:+.9f} sec"); print(f"rho excess at -30 min: {stats['rho_minus30_excess']:.12f} arcsec"); print(f"rho excess at +30 min: {stats['rho_plus30_excess']:.12f} arcsec"); print(f"EV/ES ratio range over ±30 min: {stats['ev_es_ppm_range']:.12f} ppm"); print("OUTPUT SUMMARY"); print(f"PNG: {PNG}"); print(f"CSV: {CSV}"); print("PAPER COMPARISON"); print("NOT USED: JPL-only internal geometry audit."); print("EQUATION STATUS"); print("PASS: CA is the bracketing root of raw central-difference d rho/dt; zero offset is exactly zero by definition."); print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z")); print(VERSION)
if __name__=="__main__": main()
# V0102C