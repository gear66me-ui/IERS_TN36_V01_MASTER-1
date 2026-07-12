# V0024
# Audit reference: Repairs V0023 source-patch delivery and runs the corrected half-Sun plot with summary table and decluttered labels.
from __future__ import annotations

import py_compile
import runpy
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path("/content")
SOURCE = ROOT / "IERS_0012N_HALF_SUN_TABLE_LABELS_SOURCE_V0024.py"
PATCHED = ROOT / "IERS_0012N_HALF_SUN_TABLE_LABELS_PATCHED_V0024.py"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "IERS_0012N_HALF_SUN_TABLE_LABELS_V0023.py?v=24"
)


def download_source() -> str:
    request = Request(SOURCE_URL, headers={"User-Agent": "IERS-V0024"})
    with urlopen(request, timeout=180) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError("V0024 source download was empty.")
    SOURCE.write_bytes(payload)
    return payload.decode("utf-8")


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"V0024 repair failed for {label}: expected 1 match, found {count}.")
    return text.replace(old, new, 1)


def repair(source: str) -> str:
    text = source
    text = replace_exact(
        text,
        '    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL | re.MULTILINE)',
        '    updated, count = re.subn(pattern, lambda _match: replacement, text, count=1, flags=re.DOTALL | re.MULTILINE)',
        "regex replacement escaping",
    )
    text = replace_exact(
        text,
        '    text = replace_once(text, "# V0021", "# V0023", "opening version marker")',
        '''    text = replace_once(
        text,
        "# V0021\\n# Audit reference: Fresh JPL Horizons derivation of Vardo and Point Venus tracks with C1, C2, CA, C3, C4 and an IAU-1976-normalized half-Sun plot only.",
        "# V0023\\n# Audit reference: Fresh JPL Horizons half-Sun plot with embedded summary table and decluttered Tahiti-above/Vardo-below labels.",
        "opening version marker",
    )''',
        "opening marker uniqueness",
    )
    text = replace_exact(
        text,
        '    text = replace_once(text, "# V0021", "# V0023", "closing version marker")',
        '''    if not text.rstrip().endswith("# V0021"):
        raise RuntimeError("V0023 closing marker was not found.")
    text = text.rstrip()[:-len("# V0021")] + "# V0023\\n"''',
        "closing marker uniqueness",
    )
    return text


def main() -> None:
    source = download_source()
    patched = repair(source)
    PATCHED.write_text(patched, encoding="utf-8")
    py_compile.compile(str(PATCHED), doraise=True)
    runpy.run_path(str(PATCHED), run_name="__main__")


if __name__ == "__main__":
    main()
# V0024
