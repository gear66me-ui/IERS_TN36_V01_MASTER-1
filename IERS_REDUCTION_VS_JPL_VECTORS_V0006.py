# V0006
# Audit reference: Self-healing bootstrap for IERS-0012N geometry generation followed by the exact IAU-1976 V0005 post-audit.
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

VERSION = "V0006"
PROGRAM = "IERS_REDUCTION_VS_JPL_VECTORS_V0006.py"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUTPUT_DIR = ROOT / "IERS_TN36_V01_MASTER_OUTPUT"
SOURCE_CSV = OUTPUT_DIR / "IERS-0012N_VARDO_POINT_VENUS_ENGINEERING_HALF_SUN_EVENTS_AND_GEOMETRY.csv"
SOURCE_SCRIPT = ROOT / "IERS_0012N_VARDO_POINT_VENUS_ENGINEERING_TRACK_PLOT_PI_SUN.py"
AUDIT_SCRIPT = ROOT / "IERS_REDUCTION_VS_JPL_VECTORS_V0005.py"

REPOSITORY_RAW = "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main"
SOURCE_URL = f"{REPOSITORY_RAW}/{SOURCE_SCRIPT.name}?v=6"
AUDIT_URL = f"{REPOSITORY_RAW}/{AUDIT_SCRIPT.name}?v=6"


def download_text(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": PROGRAM})
    with urlopen(request, timeout=180) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError(f"Downloaded empty file from {url}")
    destination.write_bytes(payload)


def run_python(script: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{script.name} failed with return code {completed.returncode}."
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated_source = False
    if not SOURCE_CSV.is_file():
        download_text(SOURCE_URL, SOURCE_SCRIPT)
        run_python(SOURCE_SCRIPT)
        generated_source = True

    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(
            "IERS-0012N completed without generating its expected geometry CSV: "
            f"{SOURCE_CSV}"
        )

    download_text(AUDIT_URL, AUDIT_SCRIPT)
    run_python(AUDIT_SCRIPT)

    print(f"CODE OUTPUT: {VERSION}")
    print("CODE INPUTS")
    print(f"Program: {PROGRAM}")
    print(f"IERS-0012N geometry CSV: {SOURCE_CSV}")
    print("COMMENTS")
    print(
        "IERS-0012N was generated automatically in this runtime."
        if generated_source
        else "Existing IERS-0012N geometry CSV was reused."
    )
    print("RESULTS")
    print("The IAU-1976 post-audit completed through V0005.")
    print("OUTPUT SUMMARY")
    print(f"Geometry CSV: {SOURCE_CSV}")
    print(f"Audit script: {AUDIT_SCRIPT}")
    print("PAPER COMPARISON")
    print("Modern WGS84/IAU-2012 and exact IAU-1976 Case-2 reductions were compared.")
    print("EQUATION STATUS")
    print("Bootstrap and audit: PASS")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# V0006
