# V0026
# Audit reference: Preserve approved V0025 Mercury publication and adjust only the four Vardø zoom annotations.
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

VERSION = "V0026"
SOURCE_COMMIT = "9ce85d3ca66abaf58cee5a6693bd90b1c7cf371f"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/"
    f"{SOURCE_COMMIT}/MERCURY_PARALLAX_PUBLICATION_V0025.py"
)
ISOLATED_PATH = Path("/content/MERCURY_PARALLAX_PUBLICATION_V0026_ISOLATED.py")

OLD_PATCH = r'''    _v0023_shift_label(ingress_axis, "MB C2", -0.12)
    _v0023_shift_label(egress_axis, "V C4", +0.10)
    _v0023_shift_label(egress_axis, "MB C3", +0.10)
    _v0023_shift_label(egress_axis, "V C3", -0.10)
'''

NEW_PATCH = r'''    _v0023_shift_label(ingress_axis, "MB C2", -0.12)
    _v0023_shift_label(ingress_axis, "V C2", -0.12)
    _v0023_shift_label(ingress_axis, "V C1", +0.14)
    _v0023_shift_label(egress_axis, "V C4", +0.14)
    _v0023_shift_label(egress_axis, "MB C3", +0.10)
    _v0023_shift_label(egress_axis, "V C3", -0.18)
'''


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
    if OLD_PATCH not in source:
        raise RuntimeError("The approved V0025 zoom-label block was not found.")
    source = source.replace(OLD_PATCH, NEW_PATCH, 1)
    source = source.replace("V0025", VERSION)
    if not source.startswith(f"# {VERSION}\n"):
        raise RuntimeError("V0026 source boundary check failed.")
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
        Path("/content").rglob("*V0026*.png"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("V0026 completed but no PNG output was found.")

    from IPython.display import Image, display
    display(Image(filename=str(candidates[0])))


if __name__ == "__main__":
    main()
# V0026
