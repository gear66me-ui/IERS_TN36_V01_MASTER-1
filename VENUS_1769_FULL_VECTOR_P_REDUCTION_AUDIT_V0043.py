# V0043
# Audit reference: Correct mixed-type table formatting in the full-vector differential-parallax and P audit.
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

VERSION = "V0043"
ROOT = Path("/content")
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/"
    "19582a94aa34dbfcbed36551518de44c50ea6154/"
    "VENUS_1769_FULL_VECTOR_P_REDUCTION_AUDIT_V0042.py"
)
PATCHED_PATH = ROOT / "VENUS_1769_FULL_VECTOR_P_REDUCTION_AUDIT_V0043_FULL.py"


def download_source() -> str:
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
    if not source.startswith("# V0042\n") or not source.rstrip().endswith("# V0042"):
        raise RuntimeError("Pinned V0042 source-boundary audit failed.")
    return source


def corrected_table_function() -> str:
    return '''def table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    shown = frame.copy()

    def format_value(value, pattern: str) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
            return pattern.format(float(value))
        return str(value)

    for column, pattern in (formats or {}).items():
        if column in shown.columns:
            shown[column] = shown[column].map(
                lambda value, p=pattern: format_value(value, p)
            )
    return (
        '<div class="wrap">'
        + shown.to_html(index=False, border=0, classes="audit", escape=False)
        + "</div>"
    )
'''


def patch_source(source: str) -> str:
    source = source.replace("V0042", "V0043")
    table_start = source.index("def table(")
    main_start = source.index("\ndef main()", table_start)
    source = source[:table_start] + corrected_table_function() + source[main_start:]
    if not source.startswith("# V0043\n") or not source.rstrip().endswith("# V0043"):
        raise RuntimeError("Patched V0043 source-boundary audit failed.")
    compile(source, str(PATCHED_PATH), "exec")
    return source


def main() -> None:
    source = patch_source(download_source())
    PATCHED_PATH.write_text(source, encoding="utf-8")
    namespace = {
        "__name__": "__main__",
        "__file__": str(PATCHED_PATH),
    }
    exec(compile(source, str(PATCHED_PATH), "exec"), namespace)


if __name__ == "__main__":
    main()
# V0043
