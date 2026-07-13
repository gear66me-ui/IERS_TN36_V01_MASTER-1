# V0024
# Audit reference: Run the corrected V0023 Mercury publication in a clean subprocess to prevent stale Matplotlib monkeypatches from prior notebook runs.
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

VERSION = "V0024"
SOURCE_COMMIT = "388aae990157cd88db8f1314d174d48f9dd2f4c2"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/"
    f"{SOURCE_COMMIT}/MERCURY_PARALLAX_PUBLICATION_V0023.py"
)
ISOLATED_WRAPPER = Path("/content/MERCURY_PARALLAX_PUBLICATION_V0024_ISOLATED.py")


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
        text = response.read().decode("utf-8")
    text = text.replace("V0023", VERSION)
    if not text.startswith(f"# {VERSION}\n"):
        raise RuntimeError("V0024 source boundary check failed.")
    compile(text, str(ISOLATED_WRAPPER), "exec")
    return text


def main() -> None:
    source = fetch_source()
    ISOLATED_WRAPPER.write_text(source, encoding="utf-8")
    subprocess.run(
        [sys.executable, str(ISOLATED_WRAPPER)],
        check=True,
    )

    candidates = sorted(
        Path("/content").rglob("*V0024*.png"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("V0024 completed but no PNG output was found.")

    from IPython.display import Image, display
    display(Image(filename=str(candidates[0])))


if __name__ == "__main__":
    main()
# V0024
