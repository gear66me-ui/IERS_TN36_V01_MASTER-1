# V0139
# Audit reference: exact V0120 plot and geometry, changing only the annual window to closest approach ±183 days and PNG output to 300 DPI.
from __future__ import annotations
import urllib.request

VERSION = "V0139"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/"
    "b6e9e1176ab91846c3a1312abac3008caa40e5eb/"
    "VENUS_TRANSITS_1761_1769_1874_1882_2004_2012_ANNUAL_REGISTERED_TRACKS_V0120.py"
)

with urllib.request.urlopen(SOURCE_URL, timeout=90) as response:
    source = response.read().decode("utf-8")

source = source.replace("# V0120", "# V0139")
source = source.replace("V0120", "V0139")
source = source.replace("dpi=600", "dpi=300")

old_xlim = "    ax.set_xlim(datetime(year, 1, 1), datetime(year + 1, 1, 1))"
new_xlim = (
    "    ax.set_xlim("
    "ca_date - pd.Timedelta(days=183), "
    "ca_date + pd.Timedelta(days=183)"
    ")"
)
if old_xlim not in source:
    raise RuntimeError("REJECTED V0120 x-limit statement not found")
source = source.replace(old_xlim, new_xlim)

old_query = '''    year_start = f"{year}-01-01 00:00"
    year_stop = f"{year + 1}-01-01 00:00"
    earth_year = query("399", year_start, year_stop, YEAR_STEP)
    venus_year = query("299", year_start, year_stop, YEAR_STEP)
'''
new_query = '''    annual_start = Time(ca_jd - 183.0, format="jd", scale="tdb").utc.strftime("%Y-%m-%d %H:%M")
    annual_stop = Time(ca_jd + 183.0, format="jd", scale="tdb").utc.strftime("%Y-%m-%d %H:%M")
    earth_year = query("399", annual_start, annual_stop, YEAR_STEP)
    venus_year = query("299", annual_start, annual_stop, YEAR_STEP)
'''
if old_query not in source:
    raise RuntimeError("REJECTED V0120 annual-query block not found")
source = source.replace(old_query, new_query)

if source.splitlines()[0] != "# V0139":
    raise RuntimeError("REJECTED first line")
if source.splitlines()[-1] != "# V0139":
    raise RuntimeError("REJECTED last line")
if "dpi=300" not in source or "dpi=600" in source:
    raise RuntimeError("REJECTED DPI")
if "ca_date - pd.Timedelta(days=183)" not in source:
    raise RuntimeError("REJECTED centered x-limits")
if "ca_jd - 183.0" not in source or "ca_jd + 183.0" not in source:
    raise RuntimeError("REJECTED centered annual query")

exec(
    compile(
        source,
        "VENUS_TRANSITS_1761_1769_1874_1882_2004_2012_ANNUAL_REGISTERED_TRACKS_V0139.py",
        "exec",
    )
)
# V0139
