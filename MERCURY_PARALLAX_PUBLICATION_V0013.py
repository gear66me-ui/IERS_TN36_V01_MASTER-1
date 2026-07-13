# V0013
# Audit reference: Reconstruct and run the complete visible-disk JPL ecliptic V0013 program from pinned GitHub payload parts.
from __future__ import annotations

import base64
import gzip
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

REPOSITORY = "gear66me-ui/IERS_TN36_V01_MASTER-1"
PAYLOAD_COMMIT = "be22bf536af485319c96789fed1f26d9d15b55bf"
PART_NAMES = (
    "MERCURY_PARALLAX_PUBLICATION_V0013_PART1.txt",
    "MERCURY_PARALLAX_PUBLICATION_V0013_PART2.txt",
    "MERCURY_PARALLAX_PUBLICATION_V0013_PART3.txt",
)
EXPECTED_SHA256 = "d42222c68657d764d2d41e6075a67318110231733c76a2968eddad2b0d75f6ce"
EXPANDED_PATH = Path("/content/MERCURY_PARALLAX_PUBLICATION_V0013_FULL.py")


def fetch_text(url: str) -> str:
    request = Request(url, headers={"Cache-Control": "no-cache", "User-Agent": "Colab-V0013"})
    with urlopen(request, timeout=60) as response:
        return response.read().decode("ascii").strip()


base_url = f"https://raw.githubusercontent.com/{REPOSITORY}/{PAYLOAD_COMMIT}"
payload = "".join(fetch_text(f"{base_url}/{name}") for name in PART_NAMES)
source = gzip.decompress(base64.b64decode(payload))
digest = hashlib.sha256(source).hexdigest()
if digest != EXPECTED_SHA256:
    raise RuntimeError(
        f"V0013 source verification failed: expected {EXPECTED_SHA256}, received {digest}"
    )
text = source.decode("utf-8")
if not text.startswith("# V0013\n") or not text.rstrip().endswith("# V0013"):
    raise RuntimeError("V0013 source boundary audit failed.")
compile(text, str(EXPANDED_PATH), "exec")
EXPANDED_PATH.write_text(text, encoding="utf-8")
print(f"Expanded complete V0013 source: {EXPANDED_PATH}")
print(f"Verified V0013 SHA-256: {digest}")
exec(
    compile(text, str(EXPANDED_PATH), "exec"),
    {"__name__": "__main__", "__file__": str(EXPANDED_PATH)},
)
# V0013
