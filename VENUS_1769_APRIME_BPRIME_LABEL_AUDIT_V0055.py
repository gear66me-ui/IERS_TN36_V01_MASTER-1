# V0055
# Audit reference: Remove all A′D′/AD/ED/DS notation and retain only A′B′/AB with EV/VS/ES.
from __future__ import annotations

import re
import time
import urllib.request
from pathlib import Path

VERSION = "V0055"
ROOT = Path("/content")
SOURCE_SHA = "64e36b59052c514d36ef5cb30a684bc87beb5fef"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/"
    f"{SOURCE_SHA}/VENUS_1769_SINGLE_APRIME_BPRIME_HALLEY_AUDIT_V0054.py"
)
FULL_PATH = ROOT / "VENUS_1769_APRIME_BPRIME_LABEL_AUDIT_V0055_FULL.py"


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
        source = response.read().decode("utf-8")
    if not source.startswith("# V0054\n") or not source.rstrip().endswith("# V0054"):
        raise RuntimeError("Pinned V0054 source audit failed.")
    return source


def corrected_source(source: str) -> str:
    source = source.replace("V0054", VERSION)
    source = source.replace(
        "# Audit reference: Enforce one exact A′B′ definition, correct EV/VS/ES labels, and verify AB and π₀ reductions.",
        "# Audit reference: Use only A′B′/AB and EV/VS/ES notation throughout the report.",
    )

    forbidden_notation_row = '''            [
                "V0053 label correction",
                "A′D′/AD and ED/DS removed; use A′B′/AB and EV/VS",
            ],
'''
    if forbidden_notation_row not in source:
        raise RuntimeError("Expected obsolete notation row was not found.")
    source = source.replace(forbidden_notation_row, "")

    old_comment = (
        "'<p class=\"note\">V0053 incorrectly presented the collinear Halley model "
        "prediction as a second “TN36/Halley A′B′.” It was not a second TN36 point. "
        "V0055 removes that label and uses one exact A′B′ everywhere.</p>',"
    )
    new_comment = (
        "'<p class=\"note\">A′B′ and AB are the only point-pair symbols used in this "
        "audit. EV, VS, and ES are the only distance symbols.</p>',"
    )
    if old_comment not in source:
        raise RuntimeError("Expected legacy comment was not found after version update.")
    source = source.replace(old_comment, new_comment)

    source = source.replace(
        '"<h3>Epoch and corrected notation</h3>"',
        '"<h3>Epoch and notation</h3>"',
    )

    forbidden_patterns = {
        "A-prime D-prime": r"A′D′",
        "standalone AD": r"(?<![A-Z])AD(?![A-Z])",
        "standalone ED": r"(?<![A-Z])ED(?![A-Z])",
        "standalone DS": r"(?<![A-Z])DS(?![A-Z])",
    }
    for label, pattern in forbidden_patterns.items():
        if re.search(pattern, source):
            raise RuntimeError(f"Forbidden notation remains in V0055 source: {label}.")

    if "A′B′" not in source or "AB" not in source:
        raise RuntimeError("Required A′B′/AB notation is missing.")
    if not all(symbol in source for symbol in ("EV", "VS", "ES")):
        raise RuntimeError("Required EV/VS/ES notation is missing.")

    compile(source, str(FULL_PATH), "exec")
    return source


def main() -> None:
    source = corrected_source(fetch_source())
    FULL_PATH.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {
        "__name__": "__main__",
        "__file__": str(FULL_PATH),
    }
    exec(compile(source, str(FULL_PATH), "exec"), namespace)


if __name__ == "__main__":
    main()
# V0055