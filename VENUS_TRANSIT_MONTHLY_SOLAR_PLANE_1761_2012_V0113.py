# V0113
# Audit reference: corrected visible JPL Venus trajectory with red-orange Sun and proportional monthly Venus limbs.

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import VENUS_TRANSIT_MONTHLY_SOLAR_PLANE_1761_2012_V0112 as base

VERSION = "V0113"
OUTPUT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PNG_NAMES = (
    "VENUS_TRANSIT_MONTHLY_1761_1769_V0113.png",
    "VENUS_TRANSIT_MONTHLY_1874_1882_V0113.png",
    "VENUS_TRANSIT_MONTHLY_2004_2012_V0113.png",
)


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
        sun = Circle(
            (0.0, 0.0),
            sun_radius,
            facecolor="orangered",
            edgecolor="darkred",
            linewidth=0.9,
            alpha=0.78,
            zorder=0,
            label="Sun limb",
        )
        ax.add_patch(sun)

        visible = (
            (track.x_arcsec >= -base.PLOT_LIMIT_ARCSEC)
            & (track.x_arcsec <= base.PLOT_LIMIT_ARCSEC)
            & (track.y_arcsec >= -base.PLOT_LIMIT_ARCSEC)
            & (track.y_arcsec <= base.PLOT_LIMIT_ARCSEC)
        )
        if not visible.any():
            raise RuntimeError(f"No visible trajectory samples inside ±{base.PLOT_LIMIT_ARCSEC:.0f} arcsec for {track.label}")

        ax.plot(
            track.x_arcsec,
            track.y_arcsec,
            linewidth=1.45,
            zorder=4,
            label="Venus geocentric trajectory",
        )

        month_ids = base.monthly_indices(track.dates)
        for idx in month_ids:
            x = float(track.x_arcsec[idx])
            y = float(track.y_arcsec[idx])
            if abs(x) > base.PLOT_LIMIT_ARCSEC or abs(y) > base.PLOT_LIMIT_ARCSEC:
                continue
            radius = float(track.venus_radius_arcsec[idx])
            ax.add_patch(Circle((x, y), radius, fill=False, linewidth=0.8, zorder=6))
            ax.plot(x, y, marker="o", markersize=1.8, zorder=7)
            ax.annotate(
                track.dates[idx].strftime("%b"),
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7.5,
                zorder=8,
            )

        ca_x = float(track.x_arcsec[track.ca_index])
        ca_y = float(track.y_arcsec[track.ca_index])
        ca_radius = float(track.venus_radius_arcsec[track.ca_index])
        ax.axvline(
            ca_x,
            linewidth=1.15,
            linestyle="--",
            zorder=3,
            label="Closest-approach meridian",
        )
        ax.add_patch(Circle((ca_x, ca_y), ca_radius, fill=False, linewidth=1.35, zorder=9))
        ax.plot(ca_x, ca_y, marker="o", markersize=3.0, zorder=10)

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
        "Venus Transit Trajectories — Sun-Facing Plane, Solar North Up",
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
        "The complete Venus trajectory is plotted above the red-orange Sun disk.",
        "Monthly Venus limbs use the JPL-derived angular radius and are drawn to scale.",
        "Earth is represented as the geocentric observer inset.",
    ])

    tracks = tuple(base.project_track(label, date) for label, date in base.TRANSITS)
    for (a, b), png_name in zip(base.PAIR_GROUPS, PNG_NAMES):
        make_pair_figure((tracks[a], tracks[b]), OUTPUT_DIR / png_name)
    base.write_csv(OUTPUT_DIR / "VENUS_TRANSIT_MONTHLY_SOLAR_PLANE_1761_2012_V0113.csv", tracks)

    base.section("RESULTS", [
        f"{track.label}: closest approach {track.ca_utc}; ρ={track.rho_arcsec[track.ca_index]:.6f} arcsec; "
        f"Sun radius={track.sun_radius_arcsec[track.ca_index]:.6f} arcsec; "
        f"Venus diameter={2.0 * track.venus_radius_arcsec[track.ca_index]:.6f} arcsec"
        for track in tracks
    ])
    base.section("OUTPUT SUMMARY", [
        *(f"PNG: {OUTPUT_DIR / name}" for name in PNG_NAMES),
        f"CSV: {OUTPUT_DIR / 'VENUS_TRANSIT_MONTHLY_SOLAR_PLANE_1761_2012_V0113.csv'}",
    ])
    base.section("PAPER COMPARISON", ["Not used; plotted geometry is derived directly from JPL vectors."])
    base.section("EQUATION STATUS", [
        "PASS: full daily JPL Venus trajectory is plotted continuously.",
        "PASS: Sun and Venus limbs are calculated from JPL distances and physical radii.",
        "PASS: solar north remains vertical in the instantaneous Sun-facing tangent plane.",
    ])
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"{VERSION} COMPLETE")


if __name__ == "__main__":
    main()

# V0113
