# V0049
# Audit reference: Correct the final certificate label to classical Halley AB minus corrected exact AB.
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

VERSION = "V0049"
ROOT = Path("/content")
SOURCE_COMMIT = "963b0faf6760b51f0e2e902b3553dab93d1619e4"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/"
    f"{SOURCE_COMMIT}/VENUS_1769_FINAL_HALLEY_CLOSURE_CERTIFICATE_V0048.py"
)
GENERATED = ROOT / "VENUS_1769_FINAL_HALLEY_CLOSURE_CERTIFICATE_V0049_FULL.py"


def main() -> None:
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

    if not source.startswith("# V0048\n") or not source.rstrip().endswith("# V0048"):
        raise RuntimeError("Pinned V0048 source audit failed.")

    source = source.replace("V0048", "V0049")
    source = source.replace(
        "# Audit reference: Consolidate physical distances, geometric corrections, AB closure, and normalized solar parallax π.",
        "# Audit reference: Final closure certificate with corrected classical-minus-exact residual labeling.",
    )
    source = source.replace(
        '["Total exact minus classical", total_residual_km, 100.0]',
        '["Total classical minus exact", total_residual_km, 100.0]',
    )
    source = source.replace(
        "The 19.568706 km discrepancy is not a distance or epoch error.",
        "The positive 19.568706 km value is Classical Halley AB minus Corrected exact AB; it is not a distance or epoch error.",
    )
    source = source.replace(
        '<p class="note">The classical factor EV/VS becomes exact only after applying the four full-screen geometric corrections.</p>',
        '<p class="note">The classical factor EV/VS becomes exact only after applying the four full-screen geometric corrections.</p>'
        '<p class="note">Sign convention: +19.568706 km means Classical Halley AB − Corrected exact AB.</p>',
    )

    if "Total exact minus classical" in source:
        raise RuntimeError("Rejected legacy sign label remains in generated V0049 source.")
    if "Total classical minus exact" not in source:
        raise RuntimeError("Corrected sign label was not inserted.")
    if not source.startswith("# V0049\n") or not source.rstrip().endswith("# V0049"):
        raise RuntimeError("Generated V0049 boundary audit failed.")

    compile(source, str(GENERATED), "exec")
    GENERATED.write_text(source, encoding="utf-8")

    namespace: dict[str, object] = {
        "__name__": "__main__",
        "__file__": str(GENERATED),
    }
    exec(compile(source, str(GENERATED), "exec"), namespace)


if __name__ == "__main__":
    main()
# V0049
