# V0008
# Audit reference: Cache-proof loader pinned to immutable GitHub commits for the V0007 Mercury parallax source.
from __future__ import annotations

import base64
import gzip
import hashlib
import urllib.request
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/4f02ae33ad70862621edfbeca77e67df8ddf082c/MERCURY_PARALLAX_VARDO_MERCURY_BAY_V0007.py.gz.b64"
OUTPUT_PATH = Path("/content/MERCURY_PARALLAX_VARDO_MERCURY_BAY_V0007.py")
EXPECTED_SHA256 = "ef4fb4000f70c31bfd790bf146cbf5aed9a4d04536b6554efa1c8dbf09f48e68"

request = urllib.request.Request(
    SOURCE_URL,
    headers={"Cache-Control": "no-cache", "Pragma": "no-cache", "User-Agent": "Colab-V0008"},
)
encoded = urllib.request.urlopen(request, timeout=60).read()
source = gzip.decompress(base64.b64decode(encoded))
digest = hashlib.sha256(source).hexdigest()
if digest != EXPECTED_SHA256:
    raise RuntimeError(
        "Immutable V0007 source verification failed: "
        f"expected {EXPECTED_SHA256}, received {digest}"
    )
text = source.decode("utf-8")
if not text.startswith("# V0007\n") or not text.rstrip().endswith("# V0007"):
    raise RuntimeError("V0007 source boundary audit failed.")
compile(text, str(OUTPUT_PATH), "exec")
OUTPUT_PATH.write_text(text, encoding="utf-8")
print("Loader version: V0008")
print(f"Verified immutable V0007 SHA-256: {digest}")
exec(compile(text, str(OUTPUT_PATH), "exec"), {"__name__": "__main__", "__file__": str(OUTPUT_PATH)})
# V0008
