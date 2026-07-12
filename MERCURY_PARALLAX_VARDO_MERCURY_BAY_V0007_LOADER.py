# V0007
# Audit reference: Download, verify, expand, and execute the complete V0007 Mercury parallax source from this repository.
from __future__ import annotations

import base64
import gzip
import hashlib
import urllib.request
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main/MERCURY_PARALLAX_VARDO_MERCURY_BAY_V0007.py.gz.b64"
OUTPUT_PATH = Path("/content/MERCURY_PARALLAX_VARDO_MERCURY_BAY_V0007.py")
EXPECTED_SHA256 = "c0d50b2eec97979b3b22c60f3aae887f7178259fac1d4246c649296420291054"

encoded = urllib.request.urlopen(SOURCE_URL, timeout=60).read()
source = gzip.decompress(base64.b64decode(encoded))
digest = hashlib.sha256(source).hexdigest()
if digest != EXPECTED_SHA256:
    raise RuntimeError(f"V0007 SHA-256 mismatch: {digest}")
text = source.decode("utf-8")
if not text.startswith("# V0007\n") or not text.rstrip().endswith("# V0007"):
    raise RuntimeError("V0007 boundary audit failed.")
compile(text, str(OUTPUT_PATH), "exec")
OUTPUT_PATH.write_text(text, encoding="utf-8")
exec(compile(text, str(OUTPUT_PATH), "exec"), {"__name__": "__main__", "__file__": str(OUTPUT_PATH)})
# V0007
