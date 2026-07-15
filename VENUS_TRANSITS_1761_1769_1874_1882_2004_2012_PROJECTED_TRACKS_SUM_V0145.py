# V0145
# Audit reference: preserve V0144 six-plot geometry and styling while replacing the incorrect Venus angle report with Earth plus projected-angle sum.

from __future__ import annotations

import urllib.request

VERSION = "V0145"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "VENUS_TRANSITS_1761_1769_1874_1882_2004_2012_PROJECTED_TRACKS_V0144.py"
)

with urllib.request.urlopen(SOURCE_URL, timeout=90) as response:
    source = response.read().decode("utf-8")

if source.splitlines()[0] != "# V0144":
    raise RuntimeError("REJECTED V0144 first line")
if source.splitlines()[-1] != "# V0144":
    raise RuntimeError("REJECTED V0144 last line")

source = source.replace("# V0144", "# V0145")
source = source.replace('VERSION = "V0144"', 'VERSION = "V0145"')
source = source.replace("V0144_OUTPUT", "V0145_OUTPUT")
source = source.replace("V0144.png", "V0145.png")
source = source.replace("V0144.csv", "V0145.csv")
source = source.replace("PROJECTED_TRACKS_V0144.py", "PROJECTED_TRACKS_SUM_V0145.py")

anchor = '    namespace = {"__name__": "__main__"}\n'
if anchor not in source:
    raise RuntimeError("REJECTED V0144 execution anchor missing")

patch = '''    source = source.replace(
        'f"Venus orbit: {venus_track.positive_angle_deg:.6f}°",',
        'f"Earth + projected sum: {earth_track.positive_angle_deg + projected_relative_track.positive_angle_deg:.6f}°",',
    )
    source = source.replace(
        '"venus_apparent_track_angle_deg": venus_track.positive_angle_deg,',
        '"earth_plus_projected_angle_sum_deg": earth_track.positive_angle_deg + projected_relative_track.positive_angle_deg,',
    )
    source = source.replace(
        'print(f"Venus orbit angle                    {result[\'venus_apparent_track_angle_deg\']:.6f} deg")',
        'print(f"Earth + projected angle sum          {result[\'earth_plus_projected_angle_sum_deg\']:.6f} deg")',
    )
    if 'Venus orbit: {venus_track.positive_angle_deg:.6f}°' in source:
        raise RuntimeError(f"REJECTED {year} incorrect 25-degree annotation remains")
    if 'venus_apparent_track_angle_deg' in source:
        raise RuntimeError(f"REJECTED {year} incorrect Venus angle field remains")
    if 'earth_plus_projected_angle_sum_deg' not in source:
        raise RuntimeError(f"REJECTED {year} angle-sum field missing")

'''
source = source.replace(anchor, patch + anchor)

if source.splitlines()[0] != "# V0145":
    raise RuntimeError("REJECTED V0145 first line")
if source.splitlines()[-1] != "# V0145":
    raise RuntimeError("REJECTED V0145 last line")
if "Earth + projected sum" not in source:
    raise RuntimeError("REJECTED sum annotation patch missing")

namespace = {"__name__": "__main__"}
exec(
    compile(
        source,
        "VENUS_TRANSITS_1761_1769_1874_1882_2004_2012_PROJECTED_TRACKS_SUM_V0145.py",
        "exec",
    ),
    namespace,
    namespace,
)
# V0145