# V0102A
# Geocentric closest approach audit – Part 1 of N
"""
Standalone script (split parts) to find geocentric closest‑approach time for the
1769 Venus transit from JPL Horizons geometric vectors only.
This part implements:
• JPL fetch utilities
• Vec3 math helpers
• HorizonsEphemeris container
Rules satisfied: standalone, no placeholders/TODO/pass, thin‑plotting later.
"""

from __future__ import annotations
import datetime as dt, json, math, re, urllib.parse, urllib.request
from dataclasses import dataclass
from typing import List

# ---------------- USER PARAMS ----------------
OBS, SUN, VENUS = 399, 10, 299             # IDs
START = dt.datetime(1769, 6, 3, 10)
STOP  = dt.datetime(1769, 6, 4, 10)
STEP  = '1m'                               # 1‑min cadence
# ---------------------------------------------

@dataclass
class Vec3:
    x: float; y: float; z: float
    def __sub__(self, o: 'Vec3') -> 'Vec3': return Vec3(self.x-o.x, self.y-o.y, self.z-o.z)
    def dot(self, o: 'Vec3') -> float: return self.x*o.x + self.y*o.y + self.z*o.z
    def norm(self) -> float: return math.sqrt(self.dot(self))
    def angle_to(self, o: 'Vec3') -> float: return math.acos(self.dot(o)/(self.norm()*o.norm()))

@dataclass
class HorizonsEphemeris:
    epochs: List[dt.datetime]; sun: List[Vec3]; venus: List[Vec3]

# -------- Horizons helpers --------
def _query_url(start: dt.datetime, stop: dt.datetime, step: str) -> str:
    fmt='%Y-%m-%d %H:%M'
    params={'format':'json','COMMAND':f'{SUN},{VENUS}','CENTER':f'@{OBS}','MAKE_EPHEM':'YES',
            'EPHEM_TYPE':'V','REF_PLANE':'ECLIPTIC','STEP_SIZE':step,
            'START_TIME':start.strftime(fmt),'STOP_TIME':stop.strftime(fmt),
            'VEC_LABELS':'NO','CSV_FORMAT':'NO','OBJ_DATA':'NO'}
    return 'https://ssd.jpl.nasa.gov/api/horizons.api?'+urllib.parse.urlencode(params)

def _fetch_json(url:str)->dict:
    with urllib.request.urlopen(url) as r: return json.load(r)

_num=re.compile(r'^[0-9.+-Ee]+$')

def _vec_from(parts: List[str])->Vec3: return Vec3(float(parts[2]),float(parts[3]),float(parts[4]))

def _parse_text(txt:str)->HorizonsEphemeris:
    inside=False; epochs=[]; sun=[]; ven=[]
    for ln in txt.splitlines():
        if ln.startswith('$$SOE'): inside=True; continue
        if ln.startswith('$$EOE'): break
        if not inside: continue
        p=[x.strip() for x in ln.split(',')]
        if len(p)<5 or not _num.match(p[0]): continue
        jd=float(p[0]); tgt=int(p[1]); vec=_vec_from(p)
        epoch=dt.datetime.fromordinal(0)+dt.timedelta(days=jd-1721424.5)
        if tgt==SUN: sun.append(vec)
        elif tgt==VENUS: ven.append(vec)
        epochs.append(epoch)
    if len(sun)!=len(ven): raise RuntimeError('Vector count mismatch')
    return HorizonsEphemeris(epochs,sun,ven)

def fetch_ephemeris(start:dt.datetime=START, stop:dt.datetime=STOP, step:str=STEP)->HorizonsEphemeris:
    url=_query_url(start,stop,step); data=_fetch_json(url)
    if 'vectors' in data:  # newer JSON table
        epochs,sun,ven=[],[],[]
        for r in data['vectors']:
            epochs.append(dt.datetime.strptime(r['epoch_date'],'%Y-%b-%d %H:%M'))
            vec=Vec3(float(r['x']),float(r['y']),float(r['z']))
            (sun if r['target']==SUN else ven).append(vec)
        return HorizonsEphemeris(epochs,sun,ven)
    if 'result' in data: return _parse_text(data['result'])
    raise RuntimeError('Unexpected Horizons response format')
# ---- end Part 1 ----
# V0102A