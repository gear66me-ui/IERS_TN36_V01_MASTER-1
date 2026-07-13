# V0037
# Audit reference: Re-render V0036 with pure black-and-white, non-overlapping, horizontally scrollable tables.
from __future__ import annotations

import re
import time
import urllib.request
from pathlib import Path

VERSION = "V0037"
ROOT = Path("/content")
FULL_PATH = ROOT / "VENUS_1769_TAHITI_VARDO_DISTANCE_RATIO_AUDIT_V0037_FULL.py"
SOURCE_COMMIT = "fddae67a6761ef68b049ca43b768f4742c71b4eb"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/"
    f"{SOURCE_COMMIT}/VENUS_1769_TAHITI_VARDO_DISTANCE_RATIO_AUDIT_V0036.py"
)


def fetch_source() -> str:
    request = urllib.request.Request(
        f"{SOURCE_URL}?cache={time.time_ns()}",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        source = response.read().decode("utf-8")
    if not source.startswith("# V0036\n"):
        raise RuntimeError("Pinned V0036 source header audit failed.")
    if not source.rstrip().endswith("# V0036"):
        raise RuntimeError("Pinned V0036 source footer audit failed.")
    return source


def build_v0037(source: str) -> str:
    source = source.replace("# V0036", "# V0037")
    source = source.replace(
        "# Audit reference: Compare JPL vector distances and all distance ratios at the project φ0 epoch and the instantaneous geocentric closest-approach epoch.",
        "# Audit reference: Pure black-and-white JPL distance-ratio audit with non-overlapping responsive tables.",
    )
    source = source.replace('VERSION = "V0036"', 'VERSION = "V0037"')
    source = source.replace("V0036_OUTPUT", "V0037_OUTPUT")
    source = source.replace("V0036.csv", "V0037.csv")
    source = source.replace("V0036.html", "V0037.html")
    source = source.replace("v0036-report", "v0037-report")

    old_return = (
        '    return display_frame.to_html(index=False, border=0, classes="audit-table", escape=False)\n'
    )
    new_return = (
        '    table = display_frame.to_html(index=False, border=0, classes="audit-table", escape=False)\n'
        '    return f\'<div class="table-wrap">{table}</div>\'\n'
    )
    if old_return not in source:
        raise RuntimeError("V0036 html_table return statement was not found.")
    source = source.replace(old_return, new_return, 1)

    css_pattern = re.compile(r'    css = """\n<style>.*?</style>\n"""', re.DOTALL)
    new_css = '''    css = """
<style>
.v0037-report{
  background:#ffffff;
  color:#000000;
  font-family:Arial,Helvetica,sans-serif;
  padding:16px;
  border:2px solid #000000;
  width:100%;
  max-width:none;
  box-sizing:border-box;
}
.v0037-report h1{
  font-size:24px;
  margin:0 0 20px 0;
  padding-bottom:10px;
  border-bottom:3px solid #000000;
  color:#000000;
  background:#ffffff;
}
.v0037-report h2{
  font-size:18px;
  margin:26px 0 12px 0;
  padding:8px 0;
  border-top:2px solid #000000;
  border-bottom:2px solid #000000;
  color:#000000;
  background:#ffffff;
}
.v0037-report h3{
  font-size:16px;
  margin:20px 0 8px 0;
  padding-bottom:5px;
  border-bottom:1px solid #000000;
  color:#000000;
  background:#ffffff;
}
.v0037-report p{
  margin:7px 0 12px 0;
  line-height:1.45;
  color:#000000;
  background:#ffffff;
}
.table-wrap{
  display:block;
  width:100%;
  overflow-x:auto;
  overflow-y:hidden;
  margin:8px 0 18px 0;
  padding-bottom:5px;
  -webkit-overflow-scrolling:touch;
}
.audit-table{
  border-collapse:collapse;
  width:max-content;
  min-width:100%;
  table-layout:auto;
  font-size:14px;
  line-height:1.35;
  color:#000000;
  background:#ffffff;
}
.audit-table th{
  border:1.5px solid #000000;
  padding:10px 14px;
  text-align:center;
  font-weight:700;
  white-space:nowrap;
  color:#000000;
  background:#ffffff;
}
.audit-table td{
  border:1.5px solid #000000;
  padding:10px 14px;
  text-align:right;
  vertical-align:middle;
  white-space:nowrap;
  color:#000000;
  background:#ffffff;
}
.audit-table td:first-child{
  text-align:left;
  white-space:normal;
  min-width:230px;
  max-width:330px;
}
.audit-table td:nth-child(2){
  text-align:left;
  white-space:nowrap;
}
.audit-table tbody tr,
.audit-table tbody tr:nth-child(even),
.audit-table tbody tr:nth-child(odd),
.audit-table tbody tr td{
  color:#000000;
  background:#ffffff;
}
.note{
  border:1.5px solid #000000;
  padding:12px;
  color:#000000;
  background:#ffffff;
  font-weight:600;
}
.path{
  font-family:monospace;
  font-size:12px;
  overflow-wrap:anywhere;
  color:#000000;
  background:#ffffff;
}
@media (max-width:900px){
  .v0037-report{padding:9px;}
  .v0037-report h1{font-size:20px;}
  .v0037-report h2{font-size:16px;}
  .v0037-report h3{font-size:15px;}
  .audit-table{font-size:12px;}
  .audit-table th,.audit-table td{padding:8px 10px;}
  .audit-table td:first-child{min-width:190px;max-width:250px;}
}
</style>
"""'''
    source, replacements = css_pattern.subn(new_css, source, count=1)
    if replacements != 1:
        raise RuntimeError("V0036 CSS block was not replaced exactly once.")

    compile(source, str(FULL_PATH), "exec")
    return source


def main() -> None:
    source = build_v0037(fetch_source())
    FULL_PATH.write_text(source, encoding="utf-8")
    namespace = {
        "__name__": "__main__",
        "__file__": str(FULL_PATH),
    }
    exec(compile(source, str(FULL_PATH), "exec"), namespace)


if __name__ == "__main__":
    main()
# V0037
