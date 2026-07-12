# V0026
# Audit reference: Leaves V0025 unchanged except for the two closest-approach labels: Point Venus, Tahiti and Vardo, Norway.
from __future__ import annotations

import ast
import base64
import gzip
import py_compile
import runpy
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path("/content")
SOURCE = ROOT / "IERS_0012N_HALF_SUN_TABLE_LABELS_SOURCE_V0026.py"
ENGINE = ROOT / "IERS_0012N_HALF_SUN_FINAL_LABELS_ENGINE_V0026.py"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "IERS_0012N_HALF_SUN_TABLE_LABELS_V0025.py?v=26"
)


def download_text() -> str:
    request = Request(SOURCE_URL, headers={"User-Agent": "IERS-V0026"})
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


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"V0026 label patch failed for {label}: expected 1 match, found {count}.")
    return text.replace(old, new, 1)


def main() -> None:
    source = download_text()
    payload = extract_payload(source)
    engine = gzip.decompress(base64.b64decode(payload)).decode("utf-8")
    engine = replace_once(engine, '"Vardo CA"', '"Vardo, Norway"', "Vardo label")
    engine = replace_once(engine, '"Tahiti CA"', '"Point Venus, Tahiti"', "Tahiti label")
    ENGINE.write_text(engine, encoding="utf-8")
    py_compile.compile(str(ENGINE), doraise=True)
    runpy.run_path(str(ENGINE), run_name="__main__")


if __name__ == "__main__":
    main()
# V0026
