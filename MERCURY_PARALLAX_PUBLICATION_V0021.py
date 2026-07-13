# V0021
# Audit reference: Execute the exact V0013 publication layout with corrected line weights while preserving both side zooms and the center geometry table.
from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import time
import urllib.request
from pathlib import Path

VERSION = "V0021"
SOURCE_COMMIT = "83d03041feeb6ebd5b95e9f0edbf68c4b8b99f44"
SOURCE_NAME = "MERCURY_PARALLAX_PUBLICATION_V0013.py"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/"
    f"{SOURCE_COMMIT}/{SOURCE_NAME}"
)
EXPANDED_PATH = Path("/content/MERCURY_PARALLAX_PUBLICATION_V0021_FULL.py")


def fetch_exact_v0013_wrapper() -> str:
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


def literal_assignment(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise RuntimeError(f"Required V0013 wrapper assignment not found: {name}")


def contains_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(item, ast.Name) and item.id == name
        for item in ast.walk(node)
    )


def set_keyword(call: ast.Call, keyword_name: str, value: float) -> None:
    for keyword in call.keywords:
        if keyword.arg == keyword_name:
            keyword.value = ast.Constant(value=value)
            return
    call.keywords.append(
        ast.keyword(arg=keyword_name, value=ast.Constant(value=value))
    )


class V0013PublicationPatch(ast.NodeTransformer):
    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id.upper()
            if target.id == "VERSION":
                node.value = ast.Constant(value=VERSION)
            elif "LINE_WIDTH" in name or name.endswith("_LW"):
                if "SUN" in name or "SOLAR" in name:
                    node.value = ast.Constant(value=0.500)
                else:
                    node.value = ast.Constant(value=0.375)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign):
        self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            name = node.target.id.upper()
            if node.target.id == "VERSION":
                node.value = ast.Constant(value=VERSION)
            elif "LINE_WIDTH" in name or name.endswith("_LW"):
                if "SUN" in name or "SOLAR" in name:
                    node.value = ast.Constant(value=0.500)
                else:
                    node.value = ast.Constant(value=0.375)
        return node

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        function_name = ""
        if isinstance(node.func, ast.Attribute):
            function_name = node.func.attr

        if function_name in {"plot", "axhline", "axvline"}:
            solar_limb = (
                function_name == "plot"
                and any(
                    contains_name(argument, "reference_solar_radius")
                    for argument in node.args
                )
            )
            set_keyword(
                node,
                "linewidth",
                0.500 if solar_limb else 0.375,
            )

        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "Circle"
        ):
            set_keyword(node, "linewidth", 0.375)

        return node


def build_corrected_source() -> str:
    wrapper_text = fetch_exact_v0013_wrapper()
    wrapper_tree = ast.parse(wrapper_text, filename=SOURCE_NAME)
    payload = literal_assignment(wrapper_tree, "PAYLOAD")
    expected_sha256 = literal_assignment(wrapper_tree, "EXPECTED_SHA256")
    original_bytes = gzip.decompress(base64.b64decode(payload))
    actual_sha256 = hashlib.sha256(original_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Exact V0013 source verification failed: "
            f"expected {expected_sha256}, received {actual_sha256}"
        )

    original_text = original_bytes.decode("utf-8")
    original_text = original_text.replace("V0013", VERSION)
    original_tree = ast.parse(
        original_text,
        filename=str(EXPANDED_PATH),
    )
    corrected_tree = V0013PublicationPatch().visit(original_tree)
    ast.fix_missing_locations(corrected_tree)
    corrected_body = ast.unparse(corrected_tree)
    corrected_text = (
        f"# {VERSION}\n"
        "# Audit reference: Exact V0013 figure geometry retained; "
        "solar limb 0.500 pt and plotted geometry 0.375 pt.\n"
        f"{corrected_body}\n"
        f"# {VERSION}\n"
    )
    compile(corrected_text, str(EXPANDED_PATH), "exec")
    return corrected_text


def main() -> None:
    corrected_text = build_corrected_source()
    EXPANDED_PATH.write_text(corrected_text, encoding="utf-8")
    namespace = {
        "__name__": "__main__",
        "__file__": str(EXPANDED_PATH),
    }
    exec(
        compile(corrected_text, str(EXPANDED_PATH), "exec"),
        namespace,
    )


if __name__ == "__main__":
    main()
# V0021
