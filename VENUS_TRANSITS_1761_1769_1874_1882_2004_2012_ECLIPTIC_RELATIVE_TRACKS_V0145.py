# V0145
# Audit reference: six approved-format transit plots with Earth orbit, projected relative track, and physical/ecliptic relative track; raw Venus orbit removed.

from __future__ import annotations

import urllib.request

VERSION = "V0145"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "VENUS_TRANSIT_1761_EARTH_VENUS_PROJECTED_TRACKS_V0131.py"
)

TRANSITS = {
    1761: "1761-06-06 06:00",
    1769: "1769-06-03 22:00",
    1874: "1874-12-09 04:00",
    1882: "1882-12-06 17:00",
    2004: "2004-06-08 08:00",
    2012: "2012-06-06 01:00",
}

with urllib.request.urlopen(SOURCE_URL, timeout=90) as response:
    template = response.read().decode("utf-8")

for year, center_utc in TRANSITS.items():
    source = template
    source = source.replace("# V0131", "# V0145")
    source = source.replace('VERSION = "V0131"', 'VERSION = "V0145"')
    source = source.replace("YEAR = 1761", f"YEAR = {year}")
    source = source.replace('CENTER_UTC = "1761-06-06 06:00"', f'CENTER_UTC = "{center_utc}"')
    source = source.replace(
        'OUTPUT_DIR = Path("/content/VENUS_TRANSIT_1761_EARTH_VENUS_PROJECTED_TRACKS_V0131_OUTPUT")',
        'OUTPUT_DIR = Path("/content/VENUS_TRANSITS_SIX_ECLIPTIC_RELATIVE_TRACKS_V0145_OUTPUT")',
    )
    source = source.replace(
        'PNG_NAME = "VENUS_TRANSIT_1761_EARTH_VENUS_PROJECTED_TRACKS_V0131.png"',
        f'PNG_NAME = "VENUS_TRANSIT_{year}_ECLIPTIC_RELATIVE_TRACKS_V0145.png"',
    )
    source = source.replace(
        'CSV_NAME = "VENUS_TRANSIT_1761_EARTH_VENUS_PROJECTED_TRACKS_V0131.csv"',
        f'CSV_NAME = "VENUS_TRANSIT_{year}_ECLIPTIC_RELATIVE_TRACKS_V0145.csv"',
    )
    source = source.replace('print("Transit                              1761")', f'print("Transit                              {year}")')
    source = source.replace(
        '"1761 Venus Transit — Earth Orbit, Venus Orbit, and Projected Relative Track"',
        f'"{year} Venus Transit — Earth Orbit and Venus−Sun Relative Tracks"',
    )

    source = source.replace(
        "relative_x_projected = venus_x_projected - sun_x_projected\n"
        "    relative_y_projected = venus_y_projected - sun_y_projected",
        "relative_x_projected = venus_x_projected - sun_x_projected\n"
        "    relative_y_projected = venus_y_projected - sun_y_projected\n"
        "    relative_x_physical = venus_x_physical - sun_x_physical\n"
        "    relative_y_physical = venus_y_physical - sun_y_physical",
    )

    source = source.replace(
        "venus_track = fit_track(hours, venus_x_physical[mask], venus_y_physical[mask])\n"
        "    projected_relative_track = fit_track(",
        "physical_relative_track = fit_track(\n"
        "        hours,\n"
        "        relative_x_physical[mask],\n"
        "        relative_y_physical[mask],\n"
        "    )\n"
        "    projected_relative_track = fit_track(",
    )

    source = source.replace(
        '("Venus orbit", venus_track.signed_angle_deg, "#38D66B"),',
        '("Physical Venus−Sun relative track", physical_relative_track.signed_angle_deg, "#38D66B"),',
    )
    source = source.replace(
        'if label in {"Earth orbit", "Venus orbit"}:',
        'if label in {"Earth orbit", "Physical Venus−Sun relative track"}:',
    )
    source = source.replace(
        'f"Venus orbit: {venus_track.positive_angle_deg:.6f}°",',
        'f"Physical relative track: {physical_relative_track.positive_angle_deg:.6f}°",',
    )

    replacements = {
        '"venus_apparent_track_angle_deg": venus_track.positive_angle_deg,':
            '"physical_relative_track_angle_deg": physical_relative_track.positive_angle_deg,',
        '"venus_apparent_slope": venus_track.slope,':
            '"physical_relative_slope": physical_relative_track.slope,',
        '"venus_apparent_rms_arcsec": venus_track.rms_arcsec,':
            '"physical_relative_rms_arcsec": physical_relative_track.rms_arcsec,',
        '"venus_apparent_curvature_per_arcsec": venus_track.curvature_per_arcsec,':
            '"physical_relative_curvature_per_arcsec": physical_relative_track.curvature_per_arcsec,',
        'print(f"Venus orbit angle                    {result[\'venus_apparent_track_angle_deg\']:.6f} deg")':
            'print(f"Physical relative track angle        {result[\'physical_relative_track_angle_deg\']:.6f} deg")',
        'print(f"Venus orbit slope                    {result[\'venus_apparent_slope\']:.9f}")':
            'print(f"Physical relative slope              {result[\'physical_relative_slope\']:.9f}")',
        'print("Earth and Venus orbit-direction lines use the physical east-north tangent plane.")':
            'print("Earth orbit and physical Venus−Sun relative track use the physical east-north tangent plane.")',
        'print("VERIFIED Earth and Venus arrows point from right to left")':
            'print("VERIFIED Earth and physical-relative arrows point from right to left")',
        "half_length = 0.88 * solar_radius_arcsec":
            "half_length = 3.00 * solar_radius_arcsec",
        "dpi=600":
            "dpi=300",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)

    if "venus_track" in source:
        raise RuntimeError(f"REJECTED {year}: raw Venus orbit survived")
    if "physical_relative_track" not in source:
        raise RuntimeError(f"REJECTED {year}: physical relative track missing")
    if source.splitlines()[0] != "# V0145" or source.splitlines()[-1] != "# V0145":
        raise RuntimeError(f"REJECTED {year}: version boundary")

    namespace = {"__name__": "__main__"}
    exec(
        compile(source, f"VENUS_TRANSIT_{year}_ECLIPTIC_RELATIVE_TRACKS_V0145.py", "exec"),
        namespace,
        namespace,
    )
# V0145