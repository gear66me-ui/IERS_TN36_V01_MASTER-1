# V0027
# Audit reference: Leaves V0025 unchanged except for robustly replacing the two closest-approach labels with Vardo, Norway and Point Venus, Tahiti.
from __future__ import annotations

import ast
import base64
import gzip
import py_compile
import runpy
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path("/content")
SOURCE = ROOT / "IERS_0012N_HALF_SUN_TABLE_LABELS_SOURCE_V0027.py"
ENGINE = ROOT / "IERS_0012N_HALF_SUN_FINAL_LABELS_ENGINE_V0027.py"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "IERS_0012N_HALF_SUN_TABLE_LABELS_V0025.py?v=27"
)


def download_text() -> str:
    request = Request(SOURCE_URL, headers={"User-Agent": "IERS-V0027"})
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


def is_ca_point(node: ast.AST) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    if not isinstance(node.slice, ast.Constant) or node.slice.value != "CA":
        return False
    inner = node.value
    if not isinstance(inner, ast.Subscript):
        return False
    return isinstance(inner.slice, ast.Constant) and inner.slice.value == "event_points"


def replace_string_at_node(source: str, node: ast.Constant, replacement: str) -> str:
    lines = source.splitlines(keepends=True)
    if node.lineno != node.end_lineno:
        raise RuntimeError("Unexpected multiline label literal.")
    index = node.lineno - 1
    line = lines[index]
    lines[index] = line[: node.col_offset] + repr(replacement) + line[node.end_col_offset :]
    return "".join(lines)


def patch_labels(engine: str) -> str:
    tree = ast.parse(engine)
    label_nodes: list[ast.Constant] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "label_event":
            continue
        if len(node.args) < 3 or not is_ca_point(node.args[1]):
            continue
        if isinstance(node.args[2], ast.Constant) and isinstance(node.args[2].value, str):
            label_nodes.append(node.args[2])

    label_nodes.sort(key=lambda item: (item.lineno, item.col_offset))
    if len(label_nodes) != 2:
        raise RuntimeError(f"Expected two closest-approach label calls; found {len(label_nodes)}.")

    for node, replacement in reversed(
        list(zip(label_nodes, ("Vardo, Norway", "Point Venus, Tahiti"), strict=True))
    ):
        engine = replace_string_at_node(engine, node, replacement)

    if "Vardo, Norway" not in engine or "Point Venus, Tahiti" not in engine:
        raise RuntimeError("Final label audit failed.")
    return engine


def main() -> None:
    source = download_text()
    payload = extract_payload(source)
    engine = gzip.decompress(base64.b64decode(payload)).decode("utf-8")
    engine = patch_labels(engine)
    ENGINE.write_text(engine, encoding="utf-8")
    py_compile.compile(str(ENGINE), doraise=True)
    runpy.run_path(str(ENGINE), run_name="__main__")


if __name__ == "__main__":
    main()
# V0027
