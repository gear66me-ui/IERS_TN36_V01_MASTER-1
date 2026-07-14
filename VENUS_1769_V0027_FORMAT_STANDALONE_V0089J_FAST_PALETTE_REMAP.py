# V0089J
# Audit reference: fast local Matplotlib/Pillow palette remap of verified V0089D/E render; no JPL refetch; no AI images.
from __future__ import annotations

import sys
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0089J"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUT = ROOT / "VENUS_1769_V0089J_FAST_ALT_PALETTE_OUTPUT"
PNG = OUT / "VENUS_1769_V0089J_FAST_ALT_PALETTE_900DPI.png"
PDF = OUT / "VENUS_1769_V0089J_FAST_ALT_PALETTE_VECTOR_WRAPPER.pdf"

CANDIDATES = [
    ROOT / "VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_OUTPUT" / "VENUS_1769_V0089E_LIGHT_BLUE_CONTACTS_900DPI.png",
    ROOT / "VENUS_1769_V0089D_CONTACT_COLORS_OUTPUT" / "VENUS_1769_V0089D_CONTACT_COLORS_900DPI.png",
    ROOT / "VENUS_1769_V0089C_PUBLICATION_AUDIT_OUTPUT" / "VENUS_1769_V0089C_PUBLICATION_AUDIT_900DPI.png",
    ROOT / "VENUS_1769_V0089B_PUBLICATION_OUTPUT" / "VENUS_1769_V0089B_PUBLICATION_900DPI.png",
]

def require(import_name: str, package_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", package_name])

for import_name, package_name in (("numpy", "numpy"), ("PIL", "pillow"), ("matplotlib", "matplotlib"), ("IPython", "ipython")):
    require(import_name, package_name)

import numpy as np
from PIL import Image as PILImage
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from IPython.display import Image, display

# Source palette sampled from the verified V0089D/E plot family.
SOURCE_TO_TARGET = {
    "#000000": "#070815",  # background: black -> midnight violet
    "#F8FAFC": "#F4EFFF",  # foreground text: white -> pearl lavender
    "#B8CBD6": "#B8B1D9",  # muted text: blue gray -> muted lavender
    "#70879A": "#6E5E91",  # table grid: steel -> violet slate
    "#263A4B": "#33244F",  # guide lines: blue slate -> deep plum
    "#FFD34A": "#FF6F91",  # solar limb: yellow -> coral rose
    "#D95A1B": "#7A2E8E",  # solar fill: orange -> royal violet
    "#42D7C3": "#9DFFCB",  # Point Venus: teal -> mint
    "#D89B18": "#FFB86B",  # Vardo: gold -> amber peach
    "#23466F": "#3B1E66",  # table header: blue -> aubergine
    "#164B55": "#116466",  # teal rows -> dark peacock
    "#563B0B": "#7F3D3D",  # brown rows -> wine clay
    "#101A2E": "#111328",  # body rows -> deep indigo
    "#173A63": "#5B4BA8",  # C1/C2 blue rows -> violet blue
    "#4DA3FF": "#C084FC",  # light contact blue -> orchid
}

TOLERANCE = 54.0

def hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32)

def find_source() -> Path:
    for p in CANDIDATES:
        if p.exists() and p.stat().st_size > 1000:
            return p
    found = sorted(ROOT.glob("**/VENUS_1769*900DPI*.png"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    for p in found:
        if "V0089J" not in p.name and p.stat().st_size > 1000:
            return p
    raise RuntimeError("No previous V0089 rendered PNG was found in /content. Run the verified V0089D or V0089E widget once, then run this fast palette widget again.")

def recolor_image(src: Path) -> np.ndarray:
    img = PILImage.open(src).convert("RGBA")
    arr = np.asarray(img).astype(np.float32)
    rgb = arr[..., :3]
    alpha = arr[..., 3:4]
    out = rgb.copy()
    for old_hex, new_hex in SOURCE_TO_TARGET.items():
        old = hex_to_rgb(old_hex)
        new = hex_to_rgb(new_hex)
        dist = np.sqrt(np.sum((rgb - old) ** 2, axis=2))
        mask = (dist <= TOLERANCE) & (alpha[..., 0] > 0)
        strength = np.clip(1.0 - dist / TOLERANCE, 0.0, 1.0)[..., None]
        out[mask] = (rgb * (1.0 - strength) + new * strength)[mask]
    final = np.concatenate([np.clip(out, 0, 255), alpha], axis=2).astype(np.uint8)
    return final

def save_and_display(arr: np.ndarray) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(arr, mode="RGBA").save(PNG)
    plt.close("all")
    fig = plt.figure(figsize=(16, 9), facecolor="#070815")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(arr)
    ax.axis("off")
    fig.savefig(PDF, bbox_inches="tight", pad_inches=0.0, facecolor="#070815")
    plt.close(fig)
    display(Image(filename=str(PNG)))

def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print("Task: fast alternate palette preview from existing verified V0089 PNG")
    print("Plotting: Python/Matplotlib/Pillow only; no AI images; no JPL refetch")
    print("COMMENTS")
    print("Complete palette remap: background, Sun, station tracks/disks, table headers, contact rows, body rows, grid lines, and text accents.")
    src = find_source()
    print(f"Source PNG: {src}")
    arr = recolor_image(src)
    save_and_display(arr)
    print("RESULTS")
    print("Palette: midnight violet / coral rose Sun / mint Point Venus / amber Vardø / aubergine tables")
    print("Geometry: unchanged from already-rendered verified widget")
    print("JPL Horizons calls: 0")
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"PDF wrapper: {PDF}")
    print("PAPER COMPARISON")
    print("NOT USED: color-only visual comparison; scientific numbers unchanged from source render")
    print("EQUATION STATUS")
    print("PASS: no equations changed; no AI images; no JPL network calls; palette-only remap.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)

if __name__ == "__main__":
    main()
# V0089J
