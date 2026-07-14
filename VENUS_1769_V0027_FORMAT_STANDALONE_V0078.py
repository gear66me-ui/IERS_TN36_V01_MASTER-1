# V0078
# Audit reference: Plot-only correction from verified V0077; redder and less translucent solar fill, geometry unchanged.
from __future__ import annotations

import re
import urllib.request

VERSION = "V0078"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "VENUS_1769_V0027_FORMAT_STANDALONE_V0077.py"
)
SOLAR_FILL_COLOR = "#FF5A1F"
SOLAR_FILL_ALPHA = 0.240


def fetch_source() -> str:
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": VERSION})
    with urllib.request.urlopen(request, timeout=60) as response:
        source = response.read().decode("utf-8")
    if "# V0077" not in source or "def promote_to_v0077(" not in source:
        raise RuntimeError("Verified V0077 source was not loaded correctly.")
    return source


def patch_redder_less_translucent_sun(source: str) -> str:
    source = re.sub(
        r'SOLAR_FILL_COLOR\s*=\s*"#[0-9A-Fa-f]{6}"',
        f'SOLAR_FILL_COLOR = "{SOLAR_FILL_COLOR}"',
        source,
        count=1,
    )
    source = re.sub(
        r'SOLAR_FILL_ALPHA\s*=\s*[0-9]+(?:\.[0-9]+)?',
        f'SOLAR_FILL_ALPHA = {SOLAR_FILL_ALPHA:.3f}',
        source,
        count=1,
    )
    return source


def promote_to_v0078(source: str) -> str:
    source = patch_redder_less_translucent_sun(source)
    source = source.replace("V0077", VERSION)
    source = source.replace(
        "# Audit reference: Plot-only correction from verified V0076; Venus disk paint outlines retained, zoom disks outlined, derivation-table last row changed from average track angle to delta track angle, geometry unchanged.",
        "# Audit reference: Plot-only correction from verified V0077; redder and less translucent solar fill, geometry unchanged.",
    )
    source = source.replace(
        "VENUS_1769_V0027_FORMAT_STANDALONE_V0077.py",
        "VENUS_1769_V0027_FORMAT_STANDALONE_V0078.py",
    )
    return source


def main() -> None:
    source = promote_to_v0078(fetch_source())
    namespace = {"__name__": "__main__", "__file__": "VENUS_1769_V0027_FORMAT_STANDALONE_V0078.py"}
    exec(compile(source, "VENUS_1769_V0027_FORMAT_STANDALONE_V0078.py", "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
# V0078
