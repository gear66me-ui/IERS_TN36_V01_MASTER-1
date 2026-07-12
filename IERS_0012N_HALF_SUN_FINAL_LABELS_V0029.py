# V0029
# Audit reference: Leaves V0025 unchanged except for the two closest-approach display labels; site keys and all calculations remain untouched.
from __future__ import annotations

import ast
import base64
import gzip
import py_compile
import runpy
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path("/content")
SOURCE = ROOT / "IERS_0012N_HALF_SUN_TABLE_LABELS_SOURCE_V0029.py"
ENGINE = ROOT / "IERS_0012N_HALF_SUN_FINAL_LABELS_ENGINE_V0029.py"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "IERS_0012N_HALF_SUN_TABLE_LABELS_V0025.py?v=29"
)


def download_text() -> str:
    request = Request(SOURCE_URL, headers={"User-Agent": "IERS-V0029"})
    with urlopen(request, timeout=180) as response:
        data = response.read()
    if not data:
        raise RuntimeError("V0025 source download was empty.")
    SOURCE.write_bytes(data)
    return data.decode("utf-8")


def extract_payload(source: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "PAYLOAD":
                    value = ast.literal_eval(node.value)
                    if isinstance(value, str) and value:
                        return value
    raise RuntimeError("PAYLOAD was not found in V0025.")


def patch_ca_label_line(engine: str) -> str:
    lines = engine.splitlines(keepends=True)
    matches = [
        index
        for index, line in enumerate(lines)
        if "label =" in line
        and "site['short']" in line
        and "event ==" in line
        and "CA" in line
        and "prefix" in line
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one closest-approach label assignment; found {len(matches)}.")

    index = matches[0]
    indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
    lines[index] = (
        indent
        + 'label = (("Vardo, Norway" if site["key"] == "VARDO" else "Point Venus, Tahiti") '
        + 'if event == "CA" else f"{prefix} {event}")\n'
    )
    patched = "".join(lines)

    if 'layouts[str(site["key"])][event]' not in patched:
        raise RuntimeError("Site-key layout lookup was altered unexpectedly.")
    if 'site["key"] == "VARDO"' not in patched:
        raise RuntimeError("VARDO site-key test is missing.")
    if "Point Venus, Tahiti" not in patched or "Vardo, Norway" not in patched:
        raise RuntimeError("Final display labels were not inserted.")
    return patched


def main() -> None:
    source = download_text()
    payload = extract_payload(source)
    engine = gzip.decompress(base64.b64decode(payload)).decode("utf-8")
    engine = patch_ca_label_line(engine)
    ENGINE.write_text(engine, encoding="utf-8")
    py_compile.compile(str(ENGINE), doraise=True)
    runpy.run_path(str(ENGINE), run_name="__main__")


if __name__ == "__main__":
    main()
# V0029
