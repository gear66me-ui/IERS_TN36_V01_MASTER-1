# V0114
# Audit reference: centered Sun, solar crosshairs, and JPL Venus trajectory direction arrows.

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

import VENUS_TRANSIT_MONTHLY_SOLAR_PLANE_1761_2012_V0112 as base

VERSION = "V0114"
OUTPUT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PNG_NAMES = (
    "VENUS_TRANSIT_MONTHLY_1761_1769_V0114.png",
    "VENUS_TRANSIT_MONTHLY_1874_1882_V0114.png",
    "VENUS_TRANSIT_MONTHLY_2004_2012_V0114.png",
)
CSV_NAME = "VENUS_TRANSIT_MONTHLY_SOLAR_PLANE_1761_2012_V0114.csv"


def add_direction_arrows(ax: plt.Axes, track: base.TransitTrack) -> None:
    visible = [
        i for i, (x, y) in enumerate(zip(track.x_arcsec, track.y_arcsec))
        if abs(float(x)) <= base.PLOT_LIMIT_ARCSEC and abs(float(y)) <= base.PLOT_LIMIT_ARCSEC
    ]
    if len(visible) < 12:
        return

    fractions = (0.18, 0.38, 0.58, 0.78)
    for fraction in fractions:
        k = int(round(fraction * (len(visible) - 2)))
        i0 = visible[max(0, k)]
        i1 = visible[min(len(visible) - 1, k + 1)]
        x0 = float(track.x_arcsec[i0])
        y0 = float(track.y_arcsec[i0])
        x1 = float(track.x_arcsec[i1])
        y1 = float(track.y_arcsec[i1])
        arrow = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=11.0,
            linewidth=0.9,
            zorder=8,
        )
        ax.add_patch(arrow)


def make_pair_figure(tracks: tuple[base.TransitTrack, base.TransitTrack], path: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "grid.linewidth": 0.4,
    })

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 12.8))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.06, hspace=0.24)

    for ax, track in zip(axes, tracks):
        sun_radius = float(track.sun_radius_arcsec[track.ca_index])
        ax.add_patch(Circle(
            (0.0, 0.0),
            sun_radius,
            facecolor="orangered",
            edgecolor="darkred",
            linewidth=0.9,
            alpha=0.78,
            zorder=0,
            label="Sun limb",
        ))

        ax.axhline(0.0, linewidth=0.8, zorder=1)
        ax.axvline(0.0, linewidth=0.8, zorder=1, label="Solar-center crosshairs")

        ax.plot(
            track.x_arcsec,
            track.y_arcsec,
            linewidth=1.45,
            zorder=5,
            label="Venus geocentric trajectory",
        )
        add_direction_arrows(ax, track)

        for idx in base.monthly_indices(track.dates):
            x = float(track.x_arcsec[idx])
            y = float(track.y_arcsec[idx])
            if abs(x) > base.PLOT_LIMIT_ARCSEC or abs(y) > base.PLOT_LIMIT_ARCSEC:
                continue
            radius = float(track.venus_radius_arcsec[idx])
            ax.add_patch(Circle((x, y), radius, fill=False, linewidth=0.8, zorder=7))
            ax.annotate(
                track.dates[idx].strftime("%b"),
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7.5,
                zorder=9,
            )

        ca_x = float(track.x_arcsec[track.ca_index])
        ca_y = float(track.y_arcsec[track.ca_index])
        ca_radius = float(track.venus_radius_arcsec[track.ca_index])
        ax.plot([0.0, ca_x], [0.0, ca_y], linestyle="--", linewidth=1.0, zorder=4, label="Sun center to closest approach")
        ax.add_patch(Circle((ca_x, ca_y), ca_radius, fill=False, linewidth=1.35, zorder=10))
        ax.plot(ca_x, ca_y, marker="o", markersize=3.0, zorder=11)

        ax.set_xlim(-base.PLOT_LIMIT_ARCSEC, base.PLOT_LIMIT_ARCSEC)
        ax.set_ylim(-base.PLOT_LIMIT_ARCSEC, base.PLOT_LIMIT_ARCSEC)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Solar east–west X (arcsec)")
        ax.set_ylabel("Projected solar north Y (arcsec)")
        ax.set_title(
            f"{track.label}: three months before to three months after\n"
            f"JPL closest approach: {track.ca_utc}; ρ = {track.rho_arcsec[track.ca_index]:.6f} arcsec",
            fontweight="bold",
        )
        ax.grid(True, which="major", alpha=0.48)
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.16)
        ax.legend(frameon=False, loc="lower left")
        base.add_earth_observer(ax)

    fig.suptitle(
        "Venus Transit Trajectories — Sun Centered, Solar North Up",
        fontsize=13.0,
        fontweight="bold",
    )
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.show()


def main() -> None:
    base.section("CODE INPUTS", [
        f"Version: {VERSION}",
        "Transits: 1761, 1769, 1874, 1882, 2004, 2012",
        f"Window: ±{base.WINDOW_DAYS} days",
        f"Fixed plot range: ±{base.PLOT_LIMIT_ARCSEC:.0f} arcsec",
        f"JPL cadence: {base.STEP_SIZE}",
    ])
    base.section("COMMENTS", [
        "Fresh geometric vectors are requested from NASA/JPL Horizons.",
        "The Sun center is fixed at the crosshair origin in every panel.",
        "Direction arrows follow increasing JPL epoch along the Venus trajectory.",
        "The closest-approach radius is drawn from the Sun center to the Venus center.",
    ])

    tracks = tuple(base.project_track(label, date) for label, date in base.TRANSITS)
    for (a, b), png_name in zip(base.PAIR_GROUPS, PNG_NAMES):
        make_pair_figure((tracks[a], tracks[b]), OUTPUT_DIR / png_name)
    base.write_csv(OUTPUT_DIR / CSV_NAME, tracks)

    base.section("RESULTS", [
        f"{track.label}: closest approach {track.ca_utc}; ρ={track.rho_arcsec[track.ca_index]:.6f} arcsec; "
        f"Sun radius={track.sun_radius_arcsec[track.ca_index]:.6f} arcsec; "
        f"Venus diameter={2.0 * track.venus_radius_arcsec[track.ca_index]:.6f} arcsec"
        for track in tracks
    ])
    base.section("OUTPUT SUMMARY", [
        *(f"PNG: {OUTPUT_DIR / name}" for name in PNG_NAMES),
        f"CSV: {OUTPUT_DIR / CSV_NAME}",
    ])
    base.section("PAPER COMPARISON", ["Not used; plotted geometry is derived directly from JPL vectors."])
    base.section("EQUATION STATUS", [
        "PASS: Sun center is the fixed image-plane origin for every panel.",
        "PASS: closest-approach radius begins at the solar-center crosshairs.",
        "PASS: trajectory arrows point toward increasing JPL epoch.",
    ])
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"{VERSION} COMPLETE")


if __name__ == "__main__":
    main()

# V0114
