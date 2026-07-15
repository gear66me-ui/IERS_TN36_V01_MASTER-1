# V0143
# Audit reference: V0141 true Sun-centered ecliptic-Z geometry with explicit physical y-limits equal to the largest JPL |Z| plus 20 percent.
from __future__ import annotations
import urllib.request

VERSION = "V0143"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "VENUS_TRANSIT_1761_TRUE_HELIOCENTRIC_ECLIPTIC_Z_V0141.py"
)

with urllib.request.urlopen(SOURCE_URL, timeout=90) as response:
    source = response.read().decode("utf-8")

source = source.replace("# V0141", "# V0143")
source = source.replace("V0141", "V0143")

old_extrema = '''    earth_peak_to_peak = earth_max - earth_min
    venus_peak_to_peak = venus_max - venus_min
'''
new_extrema = '''    earth_peak_to_peak = earth_max - earth_min
    venus_peak_to_peak = venus_max - venus_min

    maximum_absolute_z_km = max(
        abs(earth_min),
        abs(earth_max),
        abs(venus_min),
        abs(venus_max),
    )
    y_limit_km = 1.20 * maximum_absolute_z_km
    if not np.isfinite(y_limit_km) or y_limit_km <= 0.0:
        raise RuntimeError("REJECTED invalid physical Z-axis limit")
'''
if old_extrema not in source:
    raise RuntimeError("REJECTED V0141 extrema block not found")
source = source.replace(old_extrema, new_extrema)

source = source.replace(
'''        linewidth=1.0,
        label="Venus heliocentric ecliptic Z",
''',
'''        linewidth=1.05,
        color="#3EA6FF",
        label="Venus heliocentric ecliptic Z",
''',
1,
)
source = source.replace(
'''        linewidth=1.0,
        label="Earth heliocentric ecliptic Z",
''',
'''        linewidth=1.05,
        color="#38D66B",
        label="Earth heliocentric ecliptic Z",
''',
1,
)

old_axes = '''    ax.axhline(0.0, linewidth=0.65, linestyle="--", alpha=0.75)
    ax.axvline(center.utc.to_datetime(), linewidth=0.65, linestyle="--", alpha=0.75)
'''
new_axes = '''    ax.axhline(
        0.0,
        color="#B8B8B8",
        linewidth=0.65,
        linestyle="--",
        alpha=0.75,
    )
    ax.axvline(
        center.utc.to_datetime(),
        color="#B8B8B8",
        linewidth=0.65,
        linestyle="--",
        alpha=0.75,
    )
    ax.set_ylim(-y_limit_km, y_limit_km)
'''
if old_axes not in source:
    raise RuntimeError("REJECTED V0141 axes block not found")
source = source.replace(old_axes, new_axes)

source = source.replace(
'''        s=24,
        zorder=5,
        label="1761 transit epoch",
''',
'''        s=28,
        facecolor="#FFE082",
        edgecolor="#FFFFFF",
        linewidth=0.65,
        zorder=5,
        label="1761 transit epoch",
''',
)

source = source.replace(
'''        fontsize=9.6,
        ha="left",
''',
'''        fontsize=9.6,
        color="#F2F2F2",
        ha="left",
''',
)
source = source.replace(
'''        arrowprops={
            "arrowstyle": "-",
            "linewidth": 0.65,
        },
''',
'''        arrowprops={
            "arrowstyle": "-",
            "color": "#B8B8B8",
            "linewidth": 0.65,
        },
''',
)

old_labels = '''    ax.set_title(
        "1761 Venus Transit — True Sun-Centered Ecliptic-Z Amplitudes",
        fontsize=14.5,
        weight="bold",
        pad=10,
    )
    ax.set_xlabel("Date (closest approach centered; ±183 days)")
    ax.set_ylabel("Heliocentric ecliptic Z (km)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, alpha=0.24, linewidth=0.42)
    ax.tick_params(labelsize=9)

    ax.legend(loc="upper right", frameon=False, fontsize=9.2)
'''
new_labels = '''    ax.set_title(
        "1761 Venus Transit — True Sun-Centered Ecliptic-Z Amplitudes",
        color="#F2F2F2",
        fontsize=14.5,
        weight="bold",
        pad=10,
    )
    ax.set_xlabel(
        "Date (closest approach centered; ±183 days)",
        color="#E5E5E5",
    )
    ax.set_ylabel("Heliocentric ecliptic Z (km)", color="#E5E5E5")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, color="#686868", alpha=0.24, linewidth=0.42)
    ax.tick_params(colors="#D8D8D8", labelsize=9)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(6, 6))
    ax.yaxis.get_offset_text().set_color("#D8D8D8")
    for spine in ax.spines.values():
        spine.set_color("#909090")
        spine.set_linewidth(0.55)

    legend = ax.legend(loc="upper right", frameon=False, fontsize=9.2)
    for label in legend.get_texts():
        label.set_color("#E6E6E6")
'''
if old_labels not in source:
    raise RuntimeError("REJECTED V0141 label block not found")
source = source.replace(old_labels, new_labels)

source = source.replace(
'''    print(f"Venus peak-to-peak Z                {venus_peak_to_peak:,.6f} km")
    print(f"Sample count                        {len(result)}")
''',
'''    print(f"Venus peak-to-peak Z                {venus_peak_to_peak:,.6f} km")
    print(f"Maximum absolute plotted Z          {maximum_absolute_z_km:,.6f} km")
    print(f"Y-axis limit (maximum + 20%)        ±{y_limit_km:,.6f} km")
    print(f"Sample count                        {len(result)}")
''',
)
source = source.replace(
'''    print("VERIFIED amplitude multiplier = 1.0")
''',
'''    print("VERIFIED amplitude multiplier = 1.0")
    print("VERIFIED y-axis = ±1.20 × maximum absolute JPL Z")
''',
)

if source.splitlines()[0] != "# V0143":
    raise RuntimeError("REJECTED incorrect first line")
if source.splitlines()[-1] != "# V0143":
    raise RuntimeError("REJECTED incorrect last line")
if "y_limit_km = 1.20 * maximum_absolute_z_km" not in source:
    raise RuntimeError("REJECTED 20-percent y-axis margin missing")
if "Amplitude multiplier                 NONE" not in source:
    raise RuntimeError("REJECTED physical amplitude audit missing")

exec(
    compile(
        source,
        "VENUS_TRANSIT_1761_TRUE_HELIOCENTRIC_ECLIPTIC_Z_V0143.py",
        "exec",
    )
)
# V0143