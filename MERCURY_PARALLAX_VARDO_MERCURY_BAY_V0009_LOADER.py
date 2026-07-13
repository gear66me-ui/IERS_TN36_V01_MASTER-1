# V0009
# Audit reference: Verify immutable V0007 source, patch Matplotlib MathText syntax, audit the equation, and execute as V0009.
from __future__ import annotations

import base64
import gzip
import hashlib
import re
import urllib.request
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/4f02ae33ad70862621edfbeca77e67df8ddf082c/MERCURY_PARALLAX_VARDO_MERCURY_BAY_V0007.py.gz.b64"
ORIGINAL_SHA256 = "ef4fb4000f70c31bfd790bf146cbf5aed9a4d04536b6554efa1c8dbf09f48e68"
OUTPUT_PATH = Path("/content/MERCURY_PARALLAX_VARDO_MERCURY_BAY_V0009.py")

request = urllib.request.Request(
    SOURCE_URL,
    headers={"Cache-Control": "no-cache", "Pragma": "no-cache", "User-Agent": "Colab-V0009"},
)
encoded = urllib.request.urlopen(request, timeout=60).read()
original_source = gzip.decompress(base64.b64decode(encoded))
original_digest = hashlib.sha256(original_source).hexdigest()
if original_digest != ORIGINAL_SHA256:
    raise RuntimeError(
        "Immutable V0007 source verification failed: "
        f"expected {ORIGINAL_SHA256}, received {original_digest}"
    )

text = original_source.decode("utf-8")
if not text.startswith("# V0007\n") or not text.rstrip().endswith("# V0007"):
    raise RuntimeError("V0007 source boundary audit failed.")

# Correct every invalid Matplotlib MathText construction such as \mathbf q and \mathbf P.
text, bold_vector_repairs = re.subn(
    r"\\mathbf\s+([A-Za-z])",
    lambda match: rf"\mathbf{{{match.group(1)}}}",
    text,
)
if bold_vector_repairs < 1:
    raise RuntimeError("Expected at least one invalid MathText bold-vector expression to repair.")

# Ensure a failed save cannot leave an open figure that IPython tries to draw a second time.
old_save_block = (
    '    figure.savefig(output_png, dpi=300, bbox_inches="tight", '
    'pad_inches=0.08, facecolor="black")\n'
    '    plt.close(figure)'
)
new_save_block = (
    '    try:\n'
    '        figure.savefig(output_png, dpi=300, bbox_inches="tight", '
    'pad_inches=0.08, facecolor="black")\n'
    '    finally:\n'
    '        plt.close(figure)'
)
if old_save_block not in text:
    raise RuntimeError("V0007 publication save block was not found for cleanup hardening.")
text = text.replace(old_save_block, new_save_block, 1)

# Promote the patched, complete implementation to V0009 and keep output names traceable.
text = text.replace("V0007", "V0009")
text = text.replace(
    "MERCURY_PARALLAX_VARDO_MERCURY_BAY_V0007.py",
    "MERCURY_PARALLAX_VARDO_MERCURY_BAY_V0009.py",
)
if not text.startswith("# V0009\n") or not text.rstrip().endswith("# V0009"):
    raise RuntimeError("V0009 source boundary audit failed after promotion.")

# Compile the complete patched source.
compile(text, str(OUTPUT_PATH), "exec")
OUTPUT_PATH.write_text(text, encoding="utf-8")
patched_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

# Render the exact formerly failing equation before running the scientific program.
import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib.mathtext import MathTextParser
import matplotlib.pyplot as plt

mathtext_equation = (
    r"$\Delta\mathbf{q}=\left(\mathbf{P}_S/D_{ES}"
    r"-\mathbf{P}_M/D_{EM}\right)\mathbf{B}$"
)
MathTextParser("agg").parse(mathtext_equation, dpi=100)
plt.close("all")

print("Loader version: V0009")
print(f"Verified immutable V0007 SHA-256: {original_digest}")
print(f"MathText bold-vector repairs: {bold_vector_repairs}")
print("MathText equation smoke test: PASS")
print(f"Patched complete V0009 SHA-256: {patched_digest}")

exec(
    compile(text, str(OUTPUT_PATH), "exec"),
    {"__name__": "__main__", "__file__": str(OUTPUT_PATH)},
)
# V0009
