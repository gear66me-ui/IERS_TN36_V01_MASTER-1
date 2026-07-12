# V0028
# Audit reference: Leaves V0025 unchanged except for robust token-level replacement of the two site labels with Vardo, Norway and Point Venus, Tahiti.
from __future__ import annotations

import ast
import base64
import gzip
import io
import py_compile
import runpy
import token
import tokenize
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path("/content")
SOURCE = ROOT / "IERS_0012N_HALF_SUN_TABLE_LABELS_SOURCE_V0028.py"
ENGINE = ROOT / "IERS_0012N_HALF_SUN_FINAL_LABELS_ENGINE_V0028.py"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "IERS_0012N_HALF_SUN_TABLE_LABELS_V0025.py?v=28"
)


def download_text() -> str:
    request = Request(SOURCE_URL, headers={"User-Agent": "IERS-V0028"})
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


def map_label(value: str) -> str | None:
    normalized = value.strip().lower().replace("ø", "o")
    tahiti_variants = {
        "tahiti",
        "tahiti ca",
        "tahiti, california",
        "tahiti california",
        "point venus tahiti",
        "point venus, tahiti",
        "t ca",
    }
    vardo_variants = {
        "vardo",
        "vardo ca",
        "vardo, norway",
        "vardo norway",
        "v ca",
    }
    if normalized in tahiti_variants:
        return "Point Venus, Tahiti"
    if normalized in vardo_variants:
        return "Vardo, Norway"
    if "tahiti" in normalized and "california" in normalized:
        return "Point Venus, Tahiti"
    return None


def patch_string_tokens(source: str) -> tuple[str, int]:
    input_stream = io.StringIO(source)
    output_tokens: list[tokenize.TokenInfo] = []
    replacements = 0

    for item in tokenize.generate_tokens(input_stream.readline):
        if item.type == token.STRING:
            try:
                value = ast.literal_eval(item.string)
            except Exception:
                value = None
            if isinstance(value, str):
                replacement = map_label(value)
                if replacement is not None and replacement != value:
                    item = tokenize.TokenInfo(
                        item.type,
                        repr(replacement),
                        item.start,
                        item.end,
                        item.line,
                    )
                    replacements += 1
        output_tokens.append(item)

    return tokenize.untokenize(output_tokens), replacements


def audit_labels(source: str) -> None:
    if "Point Venus, Tahiti" not in source:
        raise RuntimeError("Point Venus, Tahiti label was not inserted.")
    if "Vardo, Norway" not in source:
        raise RuntimeError("Vardo, Norway label was not inserted.")
    forbidden = ("Tahiti, California", "Tahiti California", "Tahiti CA", "Vardo CA")
    remaining = [item for item in forbidden if item in source]
    if remaining:
        raise RuntimeError("Old labels remain: " + ", ".join(remaining))


def main() -> None:
    source = download_text()
    payload = extract_payload(source)
    engine = gzip.decompress(base64.b64decode(payload)).decode("utf-8")
    engine, replacements = patch_string_tokens(engine)
    if replacements < 2:
        raise RuntimeError(f"Expected at least two label replacements; found {replacements}.")
    audit_labels(engine)
    ENGINE.write_text(engine, encoding="utf-8")
    py_compile.compile(str(ENGINE), doraise=True)
    runpy.run_path(str(ENGINE), run_name="__main__")


if __name__ == "__main__":
    main()
# V0028
