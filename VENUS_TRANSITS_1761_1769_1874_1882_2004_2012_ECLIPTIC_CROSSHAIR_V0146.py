# V0146
# Audit reference: six approved-format transit plots with ecliptic crosshair and color-coded angle box; no AI images.

from __future__ import annotations

import urllib.request

VERSION = "V0146"
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

required = [
    "# V0131",
    'VERSION = "V0131"',
    "YEAR = 1761",
    'CENTER_UTC = "1761-06-06 06:00"',
    "relative_x_projected = venus_x_projected - sun_x_projected",
    "venus_track = fit_track(hours, venus_x_physical[mask], venus_y_physical[mask])",
    'from matplotlib.patches import Circle',
    'annotation = "\\n".join([',
]
for marker in required:
    if marker not in template:
        raise RuntimeError(f"REJECTED missing V0131 marker: {marker}")

for year, center_utc in TRANSITS.items():
    source = template
    source = source.replace("# V0131", "# V0146")
    source = source.replace('VERSION = "V0131"', 'VERSION = "V0146"')
    source = source.replace("YEAR = 1761", f"YEAR = {year}")
    source = source.replace('CENTER_UTC = "1761-06-06 06:00"', f'CENTER_UTC = "{center_utc}"')
    source = source.replace(
        'OUTPUT_DIR = Path("/content/VENUS_TRANSIT_1761_EARTH_VENUS_PROJECTED_TRACKS_V0131_OUTPUT")',
        'OUTPUT_DIR = Path("/content/VENUS_TRANSITS_SIX_ECLIPTIC_CROSSHAIR_V0146_OUTPUT")',
    )
    source = source.replace(
        'PNG_NAME = "VENUS_TRANSIT_1761_EARTH_VENUS_PROJECTED_TRACKS_V0131.png"',
        f'PNG_NAME = "VENUS_TRANSIT_{year}_ECLIPTIC_CROSSHAIR_V0146.png"',
    )
    source = source.replace(
        'CSV_NAME = "VENUS_TRANSIT_1761_EARTH_VENUS_PROJECTED_TRACKS_V0131.csv"',
        f'CSV_NAME = "VENUS_TRANSIT_{year}_ECLIPTIC_CROSSHAIR_V0146.csv"',
    )
    source = source.replace('print("Transit                              1761")', f'print("Transit                              {year}")')
    source = source.replace(
        '"1761 Venus Transit — Earth Orbit, Venus Orbit, and Projected Relative Track"',
        f'"{year} Venus Transit — Ecliptic, Earth Track, and Venus Transit Tracks"',
    )

    source = source.replace(
        "from matplotlib.patches import Circle",
        "from matplotlib.patches import Circle\nfrom matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker",
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
        '        ("Earth orbit", earth_track.signed_angle_deg, "#3EA6FF"),\n'
        '        ("Projected Venus−Sun relative track", projected_relative_track.signed_angle_deg, "#F5F5F5"),\n'
        '        ("Venus orbit", venus_track.signed_angle_deg, "#38D66B"),',
        '        ("Earth track from ecliptic", earth_track.signed_angle_deg, "#3EA6FF"),\n'
        '        ("Projected Venus transit track", projected_relative_track.signed_angle_deg, "#F5F5F5"),\n'
        '        ("Venus transit track from ecliptic", physical_relative_track.signed_angle_deg, "#38D66B"),',
    )
    source = source.replace(
        'if label in {"Earth orbit", "Venus orbit"}:',
        'if label in {"Earth track from ecliptic", "Venus transit track from ecliptic"}:',
    )

    source = source.replace(
        "    ax.add_patch(Circle(\n"
        "        (0.0, 0.0),\n"
        "        solar_radius_arcsec,\n"
        "        facecolor=\"#C98A18\",\n"
        "        edgecolor=\"#E64A19\",\n"
        "        linewidth=1.15,\n"
        "        alpha=0.92,\n"
        "        zorder=1,\n"
        "    ))",
        "    ax.add_patch(Circle(\n"
        "        (0.0, 0.0),\n"
        "        solar_radius_arcsec,\n"
        "        facecolor=\"#C98A18\",\n"
        "        edgecolor=\"#E64A19\",\n"
        "        linewidth=1.15,\n"
        "        alpha=0.92,\n"
        "        zorder=1,\n"
        "    ))\n"
        "    crosshair_extent = 1.02 * solar_radius_arcsec\n"
        "    ax.plot([-crosshair_extent, crosshair_extent], [0.0, 0.0], color=\"#D8D8D8\", linewidth=0.42, alpha=0.42, zorder=2)\n"
        "    ax.plot([0.0, 0.0], [-crosshair_extent, crosshair_extent], color=\"#D8D8D8\", linewidth=0.42, alpha=0.30, zorder=2)\n"
        "    ax.text(0.72 * solar_radius_arcsec, 0.035 * solar_radius_arcsec, \"Ecliptic reference  0.000°\", color=\"#D8D8D8\", fontsize=8.4, ha=\"center\", va=\"bottom\", zorder=3)",
    )

    old_annotation_start = source.index('    annotation = "\\n".join([')
    old_annotation_end = source.index('    extent = 1.10 * solar_radius_arcsec', old_annotation_start)
    new_annotation = '''    offset_x = 0.18 * solar_radius_arcsec if ca_x <= 0.0 else -0.18 * solar_radius_arcsec
    offset_y = 0.16 * solar_radius_arcsec if ca_y <= 0.0 else -0.16 * solar_radius_arcsec

    box_lines = [
        TextArea("Ecliptic reference: 0.000°", textprops={"color": "#D8D8D8", "fontsize": 9.6}),
        TextArea(f"Earth track from ecliptic: {earth_track.positive_angle_deg:.6f}°", textprops={"color": "#3EA6FF", "fontsize": 9.6}),
        TextArea(f"Projected Venus transit track: {projected_relative_track.positive_angle_deg:.6f}°", textprops={"color": "#F5F5F5", "fontsize": 9.6}),
        TextArea(f"Venus transit track from ecliptic: {physical_relative_track.positive_angle_deg:.6f}°", textprops={"color": "#38D66B", "fontsize": 9.6}),
    ]
    packed_box = VPacker(children=box_lines, align="left", pad=0.0, sep=2.0)
    angle_box = AnchoredOffsetbox(
        loc="center",
        child=packed_box,
        pad=0.45,
        frameon=True,
        bbox_to_anchor=(ca_x + offset_x, ca_y + offset_y),
        bbox_transform=ax.transData,
        borderpad=0.45,
    )
    angle_box.patch.set_facecolor("#050505")
    angle_box.patch.set_edgecolor("#858585")
    angle_box.patch.set_alpha(0.94)
    ax.add_artist(angle_box)
    ax.plot(
        [ca_x, ca_x + 0.62 * offset_x],
        [ca_y, ca_y + 0.62 * offset_y],
        color="#B0B0B0",
        linewidth=0.65,
        zorder=7,
    )

'''
    source = source[:old_annotation_start] + new_annotation + source[old_annotation_end:]

    replacements = {
        '"venus_apparent_track_angle_deg": venus_track.positive_angle_deg,':
            '"physical_relative_track_angle_deg": physical_relative_track.positive_angle_deg,',
        '"venus_apparent_slope": venus_track.slope,':
            '"physical_relative_slope": physical_relative_track.slope,',
        '"venus_apparent_rms_arcsec": venus_track.rms_arcsec,':
            '"physical_relative_rms_arcsec": physical_relative_track.rms_arcsec,',
        '"venus_apparent_curvature_per_arcsec": venus_track.curvature_per_arcsec,':
            '"physical_relative_curvature_per_arcsec": physical_relative_track.curvature_per_arcsec,',
        'print(f"Earth orbit angle                    {result[\'earth_apparent_track_angle_deg\']:.6f} deg")':
            'print(f"Earth track from ecliptic             {result[\'earth_apparent_track_angle_deg\']:.6f} deg")',
        'print(f"Projected relative track angle       {result[\'projected_relative_track_angle_deg\']:.6f} deg")':
            'print(f"Projected Venus transit track         {result[\'projected_relative_track_angle_deg\']:.6f} deg")',
        'print(f"Venus orbit angle                    {result[\'venus_apparent_track_angle_deg\']:.6f} deg")':
            'print(f"Venus transit track from ecliptic     {result[\'physical_relative_track_angle_deg\']:.6f} deg")',
        'print(f"Venus orbit slope                    {result[\'venus_apparent_slope\']:.9f}")':
            'print(f"Venus transit ecliptic slope          {result[\'physical_relative_slope\']:.9f}")',
        'print("Earth and Venus orbit-direction lines use the physical east-north tangent plane.")':
            'print("Earth and Venus transit ecliptic angles use the physical east-north tangent plane.")',
        'print("VERIFIED Earth and Venus arrows point from right to left")':
            'print("VERIFIED Earth and Venus transit arrows point from right to left")',
        "half_length = 0.88 * solar_radius_arcsec":
            "half_length = 3.00 * solar_radius_arcsec",
        "dpi=600":
            "dpi=300",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)

    if "venus_track" in source:
        raise RuntimeError(f"REJECTED {year}: raw Venus orbit survived")
    if "Ecliptic reference: 0.000°" not in source:
        raise RuntimeError(f"REJECTED {year}: ecliptic label missing")
    if source.splitlines()[0] != "# V0146" or source.splitlines()[-1] != "# V0146":
        raise RuntimeError(f"REJECTED {year}: version boundary")

    namespace = {"__name__": "__main__"}
    exec(
        compile(source, f"VENUS_TRANSIT_{year}_ECLIPTIC_CROSSHAIR_V0146.py", "exec"),
        namespace,
        namespace,
    )
# V0146