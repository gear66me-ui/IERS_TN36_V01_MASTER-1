# V0084
# Audit reference: WIDGET ONLY; no AI images; uses verified IERS V0080 lineage and writes standalone expanded source locally before execution.
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

VERSION = "V0084"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "VENUS_1769_V0027_FORMAT_STANDALONE_V0080.py"
)
EXPANDED_OUT = Path("/content/VENUS_1769_V0027_FORMAT_STANDALONE_V0084_EXPANDED.py")


def fetch_text(url: str, version: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": version})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def extract_source_url(source: str) -> str:
    match = re.search(
        r'SOURCE_URL\s*=\s*\(\s*"([^"]+)"\s*"([^"]+)"\s*"([^"]+)"\s*\)',
        source,
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError("SOURCE_URL chain could not be traced.")
    return "".join(match.groups())


def build_expanded_v0080_source() -> str:
    v0080 = fetch_text(SOURCE_URL, VERSION)
    if "# V0080" not in v0080 or "VENUS_1769_V0027_FORMAT_STANDALONE_V0079.py" not in v0080:
        raise RuntimeError("Verified IERS V0080 source was not loaded.")
    v0079_url = extract_source_url(v0080)
    v0079 = fetch_text(v0079_url, VERSION)
    if "# V0079" not in v0079 or "VENUS_1769_V0027_FORMAT_STANDALONE_V0067.py" not in v0079:
        raise RuntimeError("Verified IERS V0079 source was not loaded.")
    namespace = {"__name__": "v0079_builder"}
    exec(compile(v0079, "V0079_builder.py", "exec"), namespace, namespace)
    expanded = namespace["build_v0079_source"]()
    promote = {"__name__": "v0080_promoter"}
    exec(compile(v0080, "V0080_promoter.py", "exec"), promote, promote)
    expanded = promote["promote_to_v0080"](expanded)
    expanded = expanded.replace("V0080", VERSION)
    expanded = expanded.replace(
        "VENUS_1769_V0027_FORMAT_STANDALONE_V0080.py",
        "VENUS_1769_V0027_FORMAT_STANDALONE_V0084_WIDGET.py",
    )
    return expanded


def audit_source(source: str) -> None:
    checks = {
        "version": "# V0084" in source and 'VERSION = "V0084"' in source,
        "plot_function": "def plot_publication(" in source,
        "main_solar_fill": "SUN_FILL_COLOR" in source and "SUN_FILL_ALPHA" in source,
        "zoom_solar_limb": "zoom_solar_radius" in source and "zoom_axis.plot(" in source,
        "venus_paint_outline": "VENUS_PAINT_LINE_FACTOR" in source,
        "label_offsets": "MAIN_PV_LABEL_Y = 36.0" in source and "MAIN_VARDO_LABEL_Y = -36.0" in source,
        "local_ca_seconds": "reference_jd" in source and "closest_jd" in source,
        "no_ai_images": "image_gen" not in source.lower(),
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise RuntimeError(f"V0084 audit failed: {failed}")


def main() -> None:
    source = build_expanded_v0080_source()
    audit_source(source)
    EXPANDED_OUT.write_text(source, encoding="utf-8")
    namespace = {"__name__": "__main__", "__file__": str(EXPANDED_OUT)}
    exec(compile(source, str(EXPANDED_OUT), "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
# V0084
