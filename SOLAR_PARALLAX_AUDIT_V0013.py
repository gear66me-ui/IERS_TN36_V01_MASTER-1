# V0013
# Audit reference: Lucky-13 runtime-first IAU/JPL solar-parallax audit.
from __future__ import annotations

import argparse, csv, io, json, math, os, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VERSION = "V0013"
PROGRAM = "SOLAR_PARALLAX_AUDIT_V0013.py"
LOCAL_TZ = ZoneInfo("America/Bogota")
ASEC = 206264.80624709636
TARGET = 8.794148
TOL = 0.000007
C = 299792458.0
TAU = 499.004782
R76 = 6378140.0
R76_SIGMA = 5.0
AU76 = 149597870000.0
AU76_SIGMA = 2000.0
R_WGS84 = 6378137.0
R_IERS = 6378136.6
AU12 = 149597870700.0
ROOT = Path("/content")
OUT_NAME = "SOLAR_PARALLAX_AUDIT_V0013_OUTPUT"
API = "https://ssd.jpl.nasa.gov/api/horizons.api"
RUNTIME_MASTER = ROOT / "O6_1769_GEOCENTER_HORIZONS_V0013.csv"
COLS = ["JD", "GEOCENTER_SUN_X_KM", "GEOCENTER_SUN_Y_KM", "GEOCENTER_SUN_Z_KM", "GEOCENTER_VENUS_X_KM", "GEOCENTER_VENUS_Y_KM", "GEOCENTER_VENUS_Z_KM"]
RADII = {"IAU1976": R76, "WGS84": R_WGS84, "IERS2010": R_IERS}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="1769 solar parallax audit V0013")
    p.add_argument("--jpl-csv", default="")
    p.add_argument("--earth-radius-mode", choices=("IAU1976", "WGS84", "IERS2010", "CUSTOM"), default="WGS84")
    p.add_argument("--earth-radius-m", type=float)
    p.add_argument("--output-dir", default="")
    p.add_argument("--dpi", type=int, default=420)
    return p.parse_args()


def radius(a: argparse.Namespace) -> tuple[str, float]:
    if a.earth_radius_mode == "CUSTOM":
        if a.earth_radius_m is None or not math.isfinite(a.earth_radius_m) or a.earth_radius_m <= 0:
            raise ValueError("CUSTOM mode requires positive --earth-radius-m.")
        return "CUSTOM", float(a.earth_radius_m)
    return a.earth_radius_mode, RADII[a.earth_radius_mode]


def valid(path: Path) -> bool:
    try:
        return all(c in pd.read_csv(path, nrows=0).columns for c in COLS)
    except Exception:
        return False


def runtime_csvs() -> list[Path]:
    found = []
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d != "drive" and not d.startswith(".")]
        found += [Path(root) / f for f in files if f.lower().endswith(".csv")]
    return sorted(set(found))


def horizons(target: str, label: str) -> pd.DataFrame:
    q = {
        "format": "json", "COMMAND": f"'{target}'", "OBJ_DATA": "'NO'", "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'", "CENTER": "'500@399'", "START_TIME": "'1769-Jun-03 18:00'",
        "STOP_TIME": "'1769-Jun-04 03:00'", "STEP_SIZE": "'1m'", "TIME_TYPE": "'UT'",
        "TIME_DIGITS": "'FRACSEC'", "CAL_TYPE": "'GREGORIAN'", "REF_PLANE": "'FRAME'",
        "REF_SYSTEM": "'ICRF'", "OUT_UNITS": "'KM-S'", "VEC_TABLE": "'1'", "VEC_CORR": "'NONE'",
        "CSV_FORMAT": "'YES'", "VEC_LABELS": "'NO'",
    }
    url = API + "?" + urlencode(q)
    payload = None
    last = None
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "SOLAR_PARALLAX_AUDIT_V0013"}), timeout=120) as r:
                payload = json.loads(r.read().decode("utf-8"))
            break
        except Exception as exc:
            last = exc
            time.sleep(attempt + 1)
    if payload is None:
        raise RuntimeError(f"JPL Horizons request failed for {label}: {last}") from last
    if "JPL" not in str(payload.get("signature", {}).get("source", "")) or payload.get("error"):
        raise RuntimeError(f"Invalid JPL Horizons response for {label}: {payload.get('error', payload.get('signature'))}")
    text = str(payload.get("result", ""))
    if "$$SOE" not in text or "$$EOE" not in text:
        raise RuntimeError(f"No JPL vector table returned for {label}.")
    rows = []
    for raw in csv.reader(io.StringIO(text.split("$$SOE", 1)[1].split("$$EOE", 1)[0])):
        fields = [x.strip() for x in raw if x.strip()]
        try:
            jd = float(fields[0].replace("D", "E"))
        except (ValueError, IndexError):
            continue
        nums = []
        for x in fields[1:]:
            try: nums.append(float(x.replace("D", "E")))
            except ValueError: continue
        if len(nums) < 3:
            raise RuntimeError(f"Cannot decode JPL {label} row: {raw}")
        rows.append({"JD": jd, f"GEOCENTER_{label}_X_KM": nums[0], f"GEOCENTER_{label}_Y_KM": nums[1], f"GEOCENTER_{label}_Z_KM": nums[2]})
    frame = pd.DataFrame(rows).sort_values("JD").drop_duplicates("JD").reset_index(drop=True)
    if len(frame) < 7: raise RuntimeError(f"Only {len(frame)} usable JPL {label} rows.")
    return frame


def build_master() -> Path:
    sun, venus = horizons("10", "SUN"), horizons("299", "VENUS")
    sun["K"], venus["K"] = sun.JD.round(10), venus.JD.round(10)
    merged = sun.merge(venus.drop(columns="JD"), on="K", validate="one_to_one").drop(columns="K")[COLS]
    ROOT.mkdir(parents=True, exist_ok=True)
    merged.to_csv(RUNTIME_MASTER, index=False, float_format="%.15f")
    return RUNTIME_MASTER


def locate(requested: str) -> tuple[Path, str]:
    if requested and valid(Path(requested).expanduser()): return Path(requested).expanduser().resolve(), "EXPLICIT CSV"
    for path in [ROOT / "O6_TAHITI_VARDO_1769_1MIN_MASTER.csv", RUNTIME_MASTER] + runtime_csvs():
        if path.is_file() and valid(path): return path.resolve(), "COLAB RUNTIME CSV"
    return build_master().resolve(), "OFFICIAL JPL HORIZONS FALLBACK"


def norms(v: np.ndarray) -> np.ndarray: return np.sqrt(np.sum(v * v, axis=1))
def angles(a: np.ndarray, b: np.ndarray) -> np.ndarray: return np.arctan2(norms(np.cross(a, b)), np.sum(a * b, axis=1))
def eval_poly(p: list[np.poly1d], x: float) -> np.ndarray: return np.array([float(f(x)) for f in p])


def golden(f, lo: float, hi: float) -> tuple[float, float]:
    g = (math.sqrt(5) - 1) / 2
    c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(240):
        if hi - lo <= 1e-6: break
        if fc < fd: hi, d, fd, c = d, c, fc, hi - g * (hi - lo); fc = f(c)
        else: lo, c, fc, d = c, d, fd, lo + g * (hi - lo); fd = f(d)
    x = (lo + hi) / 2
    return x, f(x)


def jd_utc(jd: float) -> str:
    x, z = jd + 0.5, int(math.floor(jd + 0.5)); f = x - z
    if z >= 2299161: alpha = int((z - 1867216.25) / 36524.25); a = z + 1 + alpha - int(alpha / 4)
    else: a = z
    b = a + 1524; c = int((b - 122.1) / 365.25); d = int(365.25 * c); e = int((b - d) / 30.6001)
    dayf = b - d - int(30.6001 * e) + f; day = int(dayf); month = e - 1 if e < 14 else e - 13; year = c - 4716 if month > 2 else c - 4715
    dt = datetime(year, month, day, tzinfo=timezone.utc) + timedelta(microseconds=round((dayf - day) * 86400e6))
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"


def dynamic(path: Path) -> dict:
    df = pd.read_csv(path)[COLS].apply(pd.to_numeric, errors="coerce"); df = df[np.isfinite(df).all(axis=1)].sort_values("JD").drop_duplicates("JD").reset_index(drop=True)
    jd = df.JD.to_numpy(float); sun = df[COLS[1:4]].to_numpy(float); ven = df[COLS[4:7]].to_numpy(float); i = int(np.argmin(angles(sun, ven)))
    if i == 0 or i == len(df) - 1: raise RuntimeError("JPL closest approach is at a data boundary.")
    lo, hi, center = max(0, i - 3), min(len(df), i + 4), jd[i]; sec = (jd[lo:hi] - center) * 86400
    ps = [np.poly1d(np.polyfit(sec, sun[lo:hi, k], min(3, len(sec)-1))) for k in range(3)]; pv = [np.poly1d(np.polyfit(sec, ven[lo:hi, k], min(3, len(sec)-1))) for k in range(3)]
    def sep(t):
        a, b = eval_poly(ps, t), eval_poly(pv, t); return math.atan2(float(np.linalg.norm(np.cross(a,b))), float(np.dot(a,b)))
    off, ang = golden(sep, (jd[i-1]-center)*86400, (jd[i+1]-center)*86400); vec = eval_poly(ps, off); selected = center + off/86400
    return {"jd": selected, "epoch": jd_utc(selected), "vec": vec, "distance_m": float(np.linalg.norm(vec))*1000, "sep_arcsec": ang*ASEC, "offset_s": off}


def p_arcsec(r: float, d: float) -> tuple[float, float]:
    ratio = r/d
    if not 0 < ratio < 1: raise ValueError("a/D outside (0,1).")
    return ratio, math.asin(ratio)*ASEC


def case(cid, label, r, d, source, epoch="") -> dict:
    ratio, p = p_arcsec(r,d); delta = p-TARGET
    return {"case_id":cid,"case":label,"source":source,"earth_radius_m":r,"au_or_distance_m":d,"ratio_a_over_distance":ratio,"solar_parallax_rad":p/ASEC,"solar_parallax_deg":p/3600,"solar_parallax_arcmin":p/60,"solar_parallax_arcsec":p,"difference_arcsec":delta,"difference_microarcsec":delta*1e6,"percent_difference":delta/TARGET*100,"tolerance_arcsec":TOL,"pass_fail":"PASS" if abs(delta)<=TOL else "FAIL","epoch_utc":epoch}


def main() -> None:
    a = args(); mode, rj = radius(a); master, source = locate(a.jpl_csv); j = dynamic(master); exact_au = C*TAU
    cases = pd.DataFrame([
        case("CASE_1","IAU 1976 Published",R76,AU76,"IAU-1976 constants"), case("CASE_2","IAU 1976 Exact c×τA",R76,exact_au,"c×τA"),
        case("CASE_3","IAU 2012 / WGS84",R_WGS84,AU12,"IAU-2012/WGS84"), case("CASE_IERS","IERS 2010 / IAU 2012",R_IERS,AU12,"IERS-2010/IAU-2012"),
        case("CASE_4",f"JPL Dynamic / {mode}",rj,j["distance_m"],source,j["epoch"]),
    ])
    p1, p2, p3, piers = [float(cases.set_index("case_id").loc[x,"solar_parallax_arcsec"]) for x in ("CASE_1","CASE_2","CASE_3","CASE_IERS")]
    old_wgs = p_arcsec(R_WGS84,AU76)[1]; oldr_newau = p_arcsec(R76,AU12)[1]; q=R76/AU76; den=math.sqrt(1-q*q)
    sr=abs(1/(AU76*den))*R76_SIGMA*ASEC; sau=abs(-R76/(AU76**2*den))*AU76_SIGMA*ASEC; sigma=math.hypot(sr,sau)
    inv = pd.DataFrame([
        ("Exact c × τA product",exact_au,"m"),("Published AU minus exact c × τA",AU76-exact_au,"m"),("AU rounding contribution",(p1-p2)*1e6,"microarcsec"),
        ("IAU-1976 to WGS84 radius contribution",(old_wgs-p1)*1e6,"microarcsec"),("IAU-1976 to IAU-2012 AU contribution",(oldr_newau-p1)*1e6,"microarcsec"),
        ("Combined modern shift",(p3-p1)*1e6,"microarcsec"),("WGS84 to IERS radius shift",(piers-p3)*1e6,"microarcsec"),("Radius uncertainty contribution",sr*1e6,"microarcsec"),
        ("AU uncertainty contribution",sau*1e6,"microarcsec"),("Combined historical uncertainty",sigma,"arcsec"),
    ],columns=["investigation","value","unit"])
    out=Path(a.output_dir).expanduser().resolve() if a.output_dir else ROOT/OUT_NAME; out.mkdir(parents=True,exist_ok=True)
    cases.to_csv(out/"SOLAR_PARALLAX_AUDIT_V0013_CASES.csv",index=False,float_format="%.15f"); inv.to_csv(out/"SOLAR_PARALLAX_AUDIT_V0013_INVESTIGATION.csv",index=False,float_format="%.15f")
    pd.DataFrame([("source",source,""),("master",str(master),""),("epoch",j["epoch"],"UTC"),("JD",j["jd"],"JD"),("X",j["vec"][0],"km"),("Y",j["vec"][1],"km"),("Z",j["vec"][2],"km"),("distance",j["distance_m"],"m"),("separation",j["sep_arcsec"],"arcsec"),("offset",j["offset_s"],"s")],columns=["quantity","value","unit"]).to_csv(out/"SOLAR_PARALLAX_AUDIT_V0013_JPL_VECTOR.csv",index=False)
    fig,ax=plt.subplots(figsize=(16,6)); ax.axis("off"); show=cases[["case","earth_radius_m","au_or_distance_m","solar_parallax_arcsec","difference_microarcsec","pass_fail"]].copy()
    for c,d in (("earth_radius_m",3),("au_or_distance_m",3),("solar_parallax_arcsec",12),("difference_microarcsec",6)): show[c]=show[c].map(lambda x,n=d:f"{x:.{n}f}")
    table=ax.table(cellText=show.values,colLabels=["Case","a (m)","AU / D (m)","π⊙ (arcsec)","Δ (µas)","Status"],loc="center",cellLoc="left",colLoc="center"); table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1,1.6)
    for (row,_),cell in table.get_celld().items(): cell.set_linewidth(.35); cell.set_text_props(weight="bold" if row==0 else "normal")
    ax.set_title("1769 VENUS TRANSIT — SOLAR PARALLAX AUDIT V0013",fontweight="bold"); fig.savefig(out/"SOLAR_PARALLAX_AUDIT_V0013_ENGINEERING_TABLE.png",dpi=max(180,a.dpi),bbox_inches="tight"); plt.close(fig)
    print(f"CODE OUTPUT: {VERSION}\n\nCODE INPUTS\nProgram : {PROGRAM}\nJPL source : {source}\nJPL master : {master}\nEarth radius mode : {mode}\nEarth radius : {rj:.6f} m\nOutput : {out}")
    print("\nCOMMENTS\nRuntime CSVs are searched first without traversing mounted Drive.\nIf absent, official JPL Horizons vectors are downloaded to /content.\nπ⊙ = asin(a/D); 8.794148 arcsec is comparison-only.")
    print("\nRESULTS"); print(cases[["case_id","case","earth_radius_m","au_or_distance_m","solar_parallax_arcsec","difference_microarcsec","pass_fail"]].to_string(index=False,float_format=lambda x:f"{x:.12f}"))
    print(f"\nOUTPUT SUMMARY\n{out}\n\nPAPER COMPARISON\nIAU-1976 published : {p1:.12f} arcsec\nIAU-1976 exact c×τA : {p2:.12f} arcsec\nPublished 6 dp : {p1:.6f}\nExact c×τA 6 dp : {p2:.6f}\nHistorical σπ : {sigma:.12f} arcsec")
    checks=[("AU = c×τA",abs(exact_au-C*TAU)<=1e-6),("Published rounds to 8.794148",round(p1,6)==TARGET),("Exact c×τA rounds to 8.794148",round(p2,6)==TARGET),("σπ rounds to ±0.000007",round(sigma,6)==TOL),("JPL vector magnitude",abs(j["distance_m"]-float(np.linalg.norm(j["vec"]))*1000)<=1e-6)]
    print("\nEQUATION STATUS"); [print(f"{name:<48} : {'PASS' if ok else 'FAIL'}") for name,ok in checks]
    failed=[name for name,ok in checks if not ok]
    if failed: raise RuntimeError("Equation checks failed: "+", ".join(failed))
    print(f"LOCAL TIMESTAMP: {datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"# {VERSION}")


if __name__ == "__main__": main()
# V0013
