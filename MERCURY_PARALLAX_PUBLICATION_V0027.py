# V0027
# Audit reference: Preserve approved V0026 Mercury publication and move only the V C3 egress annotation farther left outside the Mercury limb.
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

VERSION = "V0027"
SOURCE_COMMIT = "72ca54c4e8f05d5b28061fd208df30f6c9044706"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/"
    f"{SOURCE_COMMIT}/MERCURY_PARALLAX_PUBLICATION_V0026.py"
)
ISOLATED_PATH = Path("/content/MERCURY_PARALLAX_PUBLICATION_V0027_ISOLATED.py")

OLD_LABEL_SHIFT = '_v0023_shift_label(egress_axis, "V C3", -0.18)'
NEW_LABEL_SHIFT = '_v0023_shift_label(egress_axis, "V C3", -0.34)'


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
        return response.read().decode("utf-8")


def build_isolated_source() -> str:
    source = fetch_source()
    if OLD_LABEL_SHIFT not in source:
        raise RuntimeError("The approved V0026 V C3 label shift was not found.")
    source = source.replace(OLD_LABEL_SHIFT, NEW_LABEL_SHIFT, 1)
    source = source.replace("V0026", VERSION)
    if not source.startswith(f"# {VERSION}\n"):
        raise RuntimeError("V0027 source boundary check failed.")
    compile(source, str(ISOLATED_PATH), "exec")
    return source


def main() -> None:
    source = build_isolated_source()
    ISOLATED_PATH.write_text(source, encoding="utf-8")
    subprocess.run(
        [sys.executable, str(ISOLATED_PATH)],
        check=True,
    )

    candidates = sorted(
        Path("/content").rglob("*V0027*.png"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("V0027 completed but no PNG output was found.")

    from IPython.display import Image, display
    display(Image(filename=str(candidates[0])))


if __name__ == "__main__":
    main()
# V0027
