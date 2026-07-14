# V0098
# Audit reference: palette-option preview widget only; Python/Matplotlib; no AI images; no JPL calls.
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0098"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUT = ROOT / "VENUS_1769_PALETTE_OPTIONS_V0098_OUTPUT"
PNG = OUT / "VENUS_1769_PALETTE_OPTIONS_V0098_900DPI.png"
PDF = OUT / "VENUS_1769_PALETTE_OPTIONS_V0098_VECTOR.pdf"
SVG = OUT / "VENUS_1769_PALETTE_OPTIONS_V0098_VECTOR.svg"
DPI = 900

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from IPython.display import Image, display

PALETTES = [
    {
        "name": "01 — Deep Space Violet / Coral / Mint / Amber",
        "bg": "#090713", "fg": "#F5F1FF", "muted": "#B7A9D6", "grid": "#2D2446",
        "sun_fill": "#7B2CBF", "sun_limb": "#FF9F1C", "pv": "#2EC4B6", "vardo": "#FF5D73",
        "header": "#3B1D5E", "row1": "#073B4C", "row2": "#5A189A", "row3": "#6A3D0A", "body": "#14101F",
    },
    {
        "name": "02 — Arctic Night / Ice / Magenta / Electric Cyan",
        "bg": "#020617", "fg": "#E0F2FE", "muted": "#93A4B8", "grid": "#1E3A5F",
        "sun_fill": "#1D4ED8", "sun_limb": "#BAE6FD", "pv": "#22D3EE", "vardo": "#F472B6",
        "header": "#0F172A", "row1": "#164E63", "row2": "#701A75", "row3": "#334155", "body": "#07111F",
    },
    {
        "name": "03 — Antique Observatory / Parchment / Rust / Verdigris",
        "bg": "#120F0A", "fg": "#FFF4D6", "muted": "#D6C7A1", "grid": "#4A3B2A",
        "sun_fill": "#6B2E1F", "sun_limb": "#D9A441", "pv": "#76B7A0", "vardo": "#D46A3A",
        "header": "#3E2C1C", "row1": "#23423A", "row2": "#5B2F1B", "row3": "#6A5420", "body": "#1C1710",
    },
    {
        "name": "04 — Black Background / White Ink Monochrome",
        "bg": "#000000", "fg": "#FFFFFF", "muted": "#C8C8C8", "grid": "#4A4A4A",
        "sun_fill": "#202020", "sun_limb": "#FFFFFF", "pv": "#FFFFFF", "vardo": "#BDBDBD",
        "header": "#111111", "row1": "#2A2A2A", "row2": "#3A3A3A", "row3": "#4A4A4A", "body": "#080808",
    },
    {
        "name": "05 — White Background / Black Ink Reverse Monochrome",
        "bg": "#FFFFFF", "fg": "#000000", "muted": "#4A4A4A", "grid": "#B8B8B8",
        "sun_fill": "#E8E8E8", "sun_limb": "#000000", "pv": "#000000", "vardo": "#595959",
        "header": "#E2E2E2", "row1": "#D0D0D0", "row2": "#C0C0C0", "row3": "#B0B0B0", "body": "#F5F5F5",
    },
]


def draw_sample(ax, p: dict) -> None:
    ax.set_facecolor(p["bg"])
    ax.set_aspect("equal")
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-0.78, 0.88)
    ax.axis("off")

    ax.add_patch(Circle((0, 0), 0.72, facecolor=p["sun_fill"], edgecolor="none", alpha=0.34, zorder=1))
    ax.add_patch(Circle((0, 0), 0.72, facecolor="none", edgecolor=p["sun_limb"], linewidth=1.15, zorder=2))
    ax.plot([-0.82, 0.82], [0.0, 0.0], color=p["grid"], linewidth=0.45, zorder=3)
    ax.plot([0.0, 0.0], [-0.25, 0.82], color=p["grid"], linewidth=0.45, zorder=3)

    xs = [-0.66, -0.34, 0.00, 0.34, 0.66]
    pv_y = [-0.02, 0.09, 0.20, 0.31, 0.42]
    va_y = [-0.18, -0.07, 0.04, 0.15, 0.26]
    ax.plot(xs, pv_y, color=p["pv"], linewidth=1.15, zorder=4)
    ax.plot(xs, va_y, color=p["vardo"], linewidth=1.15, zorder=4)
    for x, y in zip(xs, pv_y):
        ax.add_patch(Circle((x, y), 0.042, facecolor=p["pv"], edgecolor=p["bg"], linewidth=0.45, alpha=0.95, zorder=5))
    for x, y in zip(xs, va_y):
        ax.add_patch(Circle((x, y), 0.042, facecolor=p["vardo"], edgecolor=p["bg"], linewidth=0.45, alpha=0.95, zorder=5))

    x0, y0, w, h = -1.06, -0.70, 2.12, 0.28
    ax.add_patch(Rectangle((x0, y0 + 0.21), w, 0.07, facecolor=p["header"], edgecolor=p["grid"], linewidth=0.35))
    for i, col in enumerate([p["row1"], p["row2"], p["row3"], p["body"]]):
        ax.add_patch(Rectangle((x0, y0 + 0.21 - 0.052 * (i + 1)), w, 0.052, facecolor=col, edgecolor=p["grid"], linewidth=0.25))
    ax.text(-1.00, -0.465, "table header / C1 C2 / C3 C4 / body", color=p["fg"], fontsize=4.5, ha="left", va="center")

    ax.text(-1.06, 0.82, p["name"], color=p["fg"], fontsize=7.0, fontweight="bold", ha="left", va="top")
    ax.text(-1.06, 0.70, f"Sun {p['sun_limb']}  PV {p['pv']}  Vardø {p['vardo']}", color=p["muted"], fontsize=4.7, ha="left", va="top")


def plot() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.close("all")
    fig = plt.figure(figsize=(15, 8.6), facecolor="#050505")
    fig.suptitle("1769 Venus Transit — Five Complete Color Palette Options", color="#FFFFFF", fontsize=16, fontweight="bold", y=0.975)
    fig.text(0.5, 0.944, "Palette preview only: no JPL calls, no geometry changes, no AI images. Use one palette for the final V0089 plot rebuild.", color="#D6D6D6", fontsize=8.5, ha="center")

    gs = fig.add_gridspec(2, 3, left=0.025, right=0.985, top=0.905, bottom=0.07, wspace=0.05, hspace=0.10)
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    for pal, pos in zip(PALETTES, positions):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        draw_sample(ax, pal)
    ax_last = fig.add_subplot(gs[1, 2])
    ax_last.set_facecolor("#050505")
    ax_last.axis("off")
    ax_last.text(0.02, 0.92, "Selection note", color="#FFFFFF", fontsize=12, fontweight="bold", ha="left")
    ax_last.text(0.02, 0.78, "Pick one palette number.\nI will then rebuild the actual\n900-DPI JPL plot with only\nthat color scheme changed.", color="#D6D6D6", fontsize=10, ha="left", va="top", linespacing=1.45)
    ax_last.text(0.02, 0.30, "Recommended starting point:\n01 for dramatic publication color,\n03 for historical paper tone,\n04/05 for monochrome proofs.", color="#AEB9C7", fontsize=8.6, ha="left", va="top", linespacing=1.45)

    fig.savefig(PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.03, facecolor=fig.get_facecolor())
    fig.savefig(PDF, bbox_inches="tight", pad_inches=0.03, facecolor=fig.get_facecolor())
    fig.savefig(SVG, bbox_inches="tight", pad_inches=0.03, facecolor=fig.get_facecolor())
    display(Image(filename=str(PNG)))


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print("Task: five complete color palette options for the 1769 Venus Transit Matplotlib plot")
    print("COMMENTS")
    print("This widget is a palette preview only. It does not call JPL Horizons and does not alter geometry.")
    print("RESULTS")
    for i, p in enumerate(PALETTES, 1):
        print(f"{i:02d}: {p['name']}")
        print(f"    BG {p['bg']} | FG {p['fg']} | Sun limb {p['sun_limb']} | Sun fill {p['sun_fill']} | PV {p['pv']} | Vardø {p['vardo']}")
    plot()
    print("OUTPUT SUMMARY")
    print(f"PNG: {PNG}")
    print(f"PDF: {PDF}")
    print(f"SVG: {SVG}")
    print("PAPER COMPARISON")
    print("NOT USED: palette selection preview only; no scientific quantities calculated.")
    print("EQUATION STATUS")
    print("PASS: no equations changed; no AI images; Python/Matplotlib palette widget only.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0098
