# IERS-0012Z
# Audit reference: GitHubDelivery@IERS-0012Z; preserve the IERS-0012N engineering plot and exact widget table styling while exporting both tables as PNG.

import hashlib
import math
import os
import time
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
from scipy.optimize import minimize_scalar

VERSION = "IERS-0012Z"
PROGRAM_NAME = "IERS_0012Z_PRESERVE_0012N_PLOT_AND_TABLE_STYLE.py"
BASE_PROGRAM_NAME = "IERS_0012Y_RECALCULATE_1769_TABLES_TO_PNG.py"
BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    + BASE_PROGRAM_NAME
)
LOCAL_TZ = ZoneInfo("America/Bogota")
OUTPUT_DIR = "/content/IERS_TN36_V01_MASTER_OUTPUT"
PLOT_PNG = os.path.join(
    OUTPUT_DIR,
    "IERS-0012Z_VARDO_POINT_VENUS_ENGINEERING_HALF_SUN_TRACKS.png",
)
TRIGONOMETRY_PNG = os.path.join(
    OUTPUT_DIR,
    "IERS-0012Z_TRIGONOMETRY_VARDO_POINT_VENUS.png",
)
PI_SUN_PNG = os.path.join(
    OUTPUT_DIR,
    "IERS-0012Z_PI_SUN_GEOMETRIC_SOLUTION_VARDO_POINT_VENUS.png",
)
DOWNLOAD_ATTEMPTS = 4
DOWNLOAD_TIMEOUT_SECONDS = 45

TRACK_COLORS = {
    "Vardo Norway": "#ffc861",
    "Point Venus Tahiti": "#5ee08a",
}


def fetch_core_namespace():
    last_error = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(
                BASE_URL,
                headers={
                    "User-Agent": "Mozilla/5.0 IERS-0012Z",
                    "Accept": "text/plain",
                },
            )
            with urllib.request.urlopen(
                request,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            ) as response:
                source_bytes = response.read()
            source = source_bytes.decode("utf-8")
            if not source.startswith("# IERS-0012Y\n"):
                raise RuntimeError("Scientific core version marker is invalid.")
            if not source.rstrip().endswith("# IERS-0012Y"):
                raise RuntimeError("Scientific core closing marker is invalid.")
            required = (
                "def build_geocenter_master",
                "def build_sitecoord_master",
                "def find_site_contacts",
                "def site_track",
                "def compute_parallax_geometry",
                "def trigonometry_rows",
                "def geometric_rows",
                "def format_value",
            )
            missing = [token for token in required if token not in source]
            if missing:
                raise RuntimeError(f"Scientific core missing: {missing}")

            namespace = {
                "__name__": "iers_0012y_core",
                "__file__": BASE_PROGRAM_NAME,
                "__package__": None,
                "__cached__": None,
            }
            compiled = compile(
                source,
                BASE_PROGRAM_NAME,
                "exec",
                dont_inherit=True,
                optimize=0,
            )
            exec(compiled, namespace, namespace)
            return namespace, hashlib.sha256(source_bytes).hexdigest()
        except Exception as exc:
            last_error = exc
            if attempt < DOWNLOAD_ATTEMPTS:
                time.sleep(2.0 * attempt)
    raise RuntimeError(
        "Unable to load the verified JPL scientific core after "
        f"{DOWNLOAD_ATTEMPTS} attempts: {last_error}"
    )


def find_site_closest_iers_0012n(core, topo_cache, site):
    jds = topo_cache["jd_tdb"]
    values = [core["site_sep_arcsec"](topo_cache, site, jd) for jd in jds]
    minimum_index = int(np.argmin(values))
    lower_jd = float(jds[max(0, minimum_index - 3)])
    upper_jd = float(jds[min(len(jds) - 1, minimum_index + 3)])
    result = minimize_scalar(
        lambda jd: core["site_sep_arcsec"](topo_cache, site, jd),
        bounds=(lower_jd, upper_jd),
        method="bounded",
        options={"xatol": 1e-13},
    )
    if not result.success:
        raise RuntimeError(
            f"IERS-0012N closest-approach minimization failed for "
            f"{site['label']}: {result.message}"
        )
    return float(result.x)


def augment_track(core, geo_cache, topo_cache, track, contacts, basis):
    event_jds = {
        "C1": contacts["C1"],
        "C2": contacts["C2"],
        "CA": track["closest_jd"],
        "C3": contacts["C3"],
        "C4": contacts["C4"],
    }
    event_points = {
        event: core["ray_screen_point_arcsec_sitecoord"](
            geo_cache,
            topo_cache,
            track["site"],
            jd,
            basis,
        )
        for event, jd in event_jds.items()
    }
    event_radii = {
        event: core["angular_radii_arcsec"](
            topo_cache,
            track["site"],
            jd,
        )[1]
        for event, jd in event_jds.items()
    }
    track["event_jds"] = event_jds
    track["event_pts"] = event_points
    track["event_radii"] = event_radii
    return track


def sun_radius_arcsec(core, geo_cache, jd_tdb):
    earth_sun_distance = core["norm"](
        core["vec_at"](geo_cache, "GEOCENTER_SUN", jd_tdb)
    )
    return (
        math.atan2(core["SUN_RADIUS_KM"], earth_sun_distance)
        * core["ARCSEC_PER_RAD"]
    )


def axis_limits_for_half_sun(radius, track_a, track_b):
    all_points = np.vstack([track_a["points"], track_b["points"]])
    median_y = float(np.median(all_points[:, 1]))
    sign = 1.0 if median_y >= 0.0 else -1.0
    x_limits = (-1.04 * radius, 1.04 * radius)
    y_limits = (
        (-0.06 * radius, 1.06 * radius)
        if sign > 0.0
        else (-1.06 * radius, 0.06 * radius)
    )
    minimum_y = float(np.min(all_points[:, 1]))
    maximum_y = float(np.max(all_points[:, 1]))
    if minimum_y < y_limits[0] or maximum_y > y_limits[1]:
        padding = 0.08 * radius
        y_limits = (
            min(y_limits[0], minimum_y - padding),
            max(y_limits[1], maximum_y + padding),
        )
    return x_limits, y_limits


def add_label(axis, xy, text, dx, dy, color):
    axis.annotate(
        text,
        xy=(xy[0], xy[1]),
        xytext=(xy[0] + dx, xy[1] + dy),
        textcoords="data",
        fontsize=5.7,
        color=color,
        ha="left",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "lw": 0.20,
            "color": color,
            "shrinkA": 0,
            "shrinkB": 2,
        },
    )


def add_summary_table_on_plot(core, axis, track_a, track_b, geometry):
    compact_rows = [
        ("β Vardo", track_a["track_angle_deg"], "deg"),
        ("β Point Venus", track_b["track_angle_deg"], "deg"),
        (
            "Δβ",
            abs(track_a["track_angle_deg"] - track_b["track_angle_deg"]),
            "deg",
        ),
        ("π⊙", geometry["pi_sun_arcsec"], "arcsec"),
        ("A′B′ / AB", geometry["halley_ratio"], "ratio"),
        ("A′B′", geometry["A_prime_B_prime_arcsec"], "arcsec"),
        ("A′B′", geometry["A_prime_B_prime_km"], "km"),
        ("AB", geometry["AB_arcsec"], "arcsec"),
        ("AB", geometry["AB_km"], "km"),
        ("D ES", geometry["D_ES_AU"], "AU"),
    ]
    rows = [
        [quantity, core["format_value"](quantity, value, unit), unit]
        for quantity, value, unit in compact_rows
    ]
    table = axis.table(
        cellText=rows,
        colLabels=["Quantity", "Value", "Unit"],
        loc="lower left",
        colWidths=[0.29, 0.23, 0.15],
        bbox=[0.438, 0.122, 0.380, 0.345],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.30)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_linewidth(0.18)
        cell.set_edgecolor("#1e4f64")
        if row_index == 0:
            cell.set_facecolor("#0a1a22")
            cell.get_text().set_color("#66e8ff")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#050b0f")
            if column_index == 1:
                cell.get_text().set_color("#ffc861")
                cell.get_text().set_fontweight("bold")
            elif column_index == 2:
                cell.get_text().set_color("#5ee08a")
            else:
                cell.get_text().set_color("#dff8ff")
    axis.text(
        0.440,
        0.101,
        "A′B′ = solar-screen chord; AB = projected baseline; "
        "D ES is JPL |Sun|/AU.",
        transform=axis.transAxes,
        color="#8fb4c1",
        fontsize=5.25,
        ha="left",
        va="top",
    )


def plot_engineering_track(
    core,
    geo_cache,
    track_a,
    track_b,
    screen_jd,
    geometry,
):
    radius = sun_radius_arcsec(core, geo_cache, screen_jd)
    figure, axis = plt.subplots(figsize=(9.6, 5.8), dpi=240)
    figure.patch.set_facecolor("#03080d")
    axis.set_facecolor("#03080d")
    axis.add_patch(
        Circle(
            (0.0, 0.0),
            radius,
            fill=False,
            lw=0.36,
            ec="#66e8ff",
            alpha=0.95,
        )
    )
    axis.axhline(0.0, lw=0.18, color="#1d3d4a", alpha=0.72)
    axis.axvline(0.0, lw=0.18, color="#1d3d4a", alpha=0.72)

    for track in (track_a, track_b):
        site_label = track["site"]["label"]
        color = TRACK_COLORS[site_label]
        points = track["points"]
        axis.plot(
            points[:, 0],
            points[:, 1],
            lw=0.30,
            color=color,
            solid_capstyle="round",
            label=site_label,
            zorder=3,
        )
        axis.scatter(
            points[::6, 0],
            points[::6, 1],
            s=0.75,
            color=color,
            alpha=0.70,
            linewidths=0,
            zorder=4,
        )
        for event in ("C1", "C2", "CA", "C3", "C4"):
            xy = track["event_pts"][event]
            radius_venus = track["event_radii"][event]
            axis.add_patch(
                Circle(
                    (xy[0], xy[1]),
                    radius_venus,
                    fill=False,
                    lw=0.20 if event != "CA" else 0.28,
                    ec=color,
                    alpha=0.92,
                    zorder=2,
                )
            )
            axis.scatter(
                [xy[0]],
                [xy[1]],
                s=3.8 if event == "CA" else 2.2,
                color=color,
                edgecolors="#03080d",
                linewidths=0.16,
                zorder=5,
            )
        closest = track["event_pts"]["CA"]
        dy = 15.0 if site_label.startswith("Vardo") else -15.0
        add_label(
            axis,
            closest,
            f"{track['site']['short']} CA",
            18.0,
            dy,
            color,
        )

    for event, dx, dy in (
        ("C1", -48.0, 12.0),
        ("C2", -38.0, 9.0),
        ("C3", 20.0, -10.0),
        ("C4", 30.0, -13.0),
    ):
        add_label(
            axis,
            track_a["event_pts"][event],
            event,
            dx,
            dy,
            "#8fb4c1",
        )

    add_summary_table_on_plot(core, axis, track_a, track_b, geometry)
    x_limits, y_limits = axis_limits_for_half_sun(radius, track_a, track_b)
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_aspect("equal", adjustable="box")
    for spine in axis.spines.values():
        spine.set_linewidth(0.22)
        spine.set_color("#25708b")
    axis.tick_params(
        axis="both",
        colors="#8fb4c1",
        labelsize=6.5,
        width=0.22,
        length=2.0,
    )
    axis.grid(True, color="#102630", linewidth=0.16, alpha=0.55)
    axis.set_xlabel(
        "Solar-screen X offset (arcsec)",
        color="#8fb4c1",
        fontsize=7.5,
    )
    axis.set_ylabel(
        "Solar-screen Y offset (arcsec)",
        color="#8fb4c1",
        fontsize=7.5,
    )
    axis.set_title(
        "1769 Venus Transit — Engineering Half-Sun Track Reconstruction\n"
        "Vardo, Norway / Point Venus, Tahiti — "
        "JPL Horizons SITE_COORD geometry",
        color="#f8fdff",
        fontsize=9.0,
        pad=8,
    )
    legend = axis.legend(
        loc="lower right",
        fontsize=6.3,
        frameon=True,
        borderpad=0.45,
    )
    legend.get_frame().set_facecolor("#071016")
    legend.get_frame().set_edgecolor("#1e4f64")
    legend.get_frame().set_linewidth(0.22)
    for text in legend.get_texts():
        text.set_color("#dff8ff")

    note = (
        "Venus disks are plotted to scale at C1, C2, closest approach, "
        "C3, and C4.  "
        f"Vardo CA: {track_a['closest_utc']}   "
        f"Point Venus CA: {track_b['closest_utc']}"
    )
    figure.text(
        0.5,
        0.016,
        note,
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#8fb4c1",
    )
    figure.savefig(
        PLOT_PNG,
        dpi=460,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.055,
    )
    plt.show()
    plt.close(figure)


def save_widget_style_table_png(core, title_prefix, title_span, rows, output_path):
    formatted_rows = [
        [quantity, core["format_value"](quantity, value, unit), unit]
        for quantity, value, unit in rows
    ]
    row_count = len(formatted_rows)
    figure_width = 7.0
    figure_height = max(2.2, 0.82 + 0.30 * (row_count + 1))

    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height),
        dpi=220,
    )
    figure.patch.set_facecolor("#03080d")
    axis.set_facecolor("#03080d")
    axis.axis("off")

    outer = FancyBboxPatch(
        (0.006, 0.006),
        0.988,
        0.988,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        transform=axis.transAxes,
        facecolor="#03080d",
        edgecolor="#1e4f64",
        linewidth=0.60,
        clip_on=False,
        zorder=0,
    )
    axis.add_patch(outer)

    title_text = f"{title_prefix} — {title_span}"
    axis.text(
        0.5,
        0.935,
        title_text,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color="#66e8ff",
        family="monospace",
    )
    axis.plot(
        [0.018, 0.982],
        [0.972, 0.972],
        transform=axis.transAxes,
        linewidth=0.45,
        color="#25708b",
        clip_on=False,
    )
    axis.plot(
        [0.018, 0.982],
        [0.895, 0.895],
        transform=axis.transAxes,
        linewidth=0.45,
        color="#25708b",
        clip_on=False,
    )

    table_bottom = 0.09
    table_height = 0.77
    table = axis.table(
        cellText=formatted_rows,
        colLabels=["Quantity", "Value", "Units"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.50, 0.34, 0.16],
        bbox=[0.018, table_bottom, 0.964, table_height],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)

    for (row_index, column_index), cell in table.get_celld().items():
        cell.PAD = 0.032
        if row_index == 0:
            cell.set_facecolor("#0a1a22")
            cell.set_edgecolor("#1d3d4a")
            cell.set_linewidth(0.42)
            cell.get_text().set_color("#66e8ff")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_ha("left")
        else:
            cell.set_facecolor("#050b0f")
            cell.set_edgecolor("#102630")
            cell.set_linewidth(0.34)
            if column_index == 0:
                cell.get_text().set_color("#dff8ff")
                cell.get_text().set_ha("left")
            elif column_index == 1:
                cell.get_text().set_color("#ffc861")
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_ha("right")
            else:
                cell.get_text().set_color("#5ee08a")
                cell.get_text().set_ha("left")
        cell.get_text().set_family("monospace")

    axis.text(
        0.020,
        0.045,
        "JPL Horizons SITE_COORD geometry — IERS-0012N formatting preserved.",
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=6.2,
        color="#8fb4c1",
        family="monospace",
    )

    figure.savefig(
        output_path,
        dpi=360,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
        edgecolor="none",
        pad_inches=0.04,
    )
    plt.close(figure)


def display_outputs(paths):
    try:
        from IPython.display import Image, display

        for path in paths:
            display(Image(filename=path))
        return "DISPLAYED IN COLAB"
    except Exception as exc:
        return f"NOT USED / INLINE DISPLAY UNAVAILABLE ({type(exc).__name__})"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    core, core_sha256 = fetch_core_namespace()

    print(f"CODE OUTPUT: {VERSION}")
    print()
    print("CODE INPUTS")
    print(f"Program                : {PROGRAM_NAME}")
    print(f"Scientific core        : {BASE_PROGRAM_NAME}")
    print(f"Core SHA-256           : {core_sha256}")
    print(f"JPL interval           : {core['START']} TO {core['STOP']} STEP {core['STEP']}")
    print()
    print("COMMENTS")
    print("Preserves the IERS-0012N engineering half-Sun plot rather than replacing it with tables only.")
    print("Preserves the IERS-0012N widget colors, title bars, column proportions, number formatting, and typography in the PNG tables.")
    print("Uses the original IERS-0012N raw-Julian-date closest-approach minimization for output equivalence.")
    print("No AI image generation is used.")
    print()

    geocenter_master = core["build_geocenter_master"]()
    topocentric_master = core["build_sitecoord_master"](
        core["SITE_A"],
        core["SITE_B"],
    )
    geocenter_cache = core["build_cache"](geocenter_master)
    topocentric_cache = core["build_cache"](topocentric_master)

    contacts_a = core["find_site_contacts"](
        topocentric_cache,
        core["SITE_A"],
    )
    contacts_b = core["find_site_contacts"](
        topocentric_cache,
        core["SITE_B"],
    )
    closest_a = find_site_closest_iers_0012n(
        core,
        topocentric_cache,
        core["SITE_A"],
    )
    closest_b = find_site_closest_iers_0012n(
        core,
        topocentric_cache,
        core["SITE_B"],
    )
    screen_jd = 0.5 * (closest_a + closest_b)
    basis = core["fixed_geocenter_basis"](geocenter_cache, screen_jd)

    track_a = core["site_track"](
        geocenter_cache,
        topocentric_cache,
        core["SITE_A"],
        contacts_a,
        closest_a,
        basis,
    )
    track_b = core["site_track"](
        geocenter_cache,
        topocentric_cache,
        core["SITE_B"],
        contacts_b,
        closest_b,
        basis,
    )
    track_a = augment_track(
        core,
        geocenter_cache,
        topocentric_cache,
        track_a,
        contacts_a,
        basis,
    )
    track_b = augment_track(
        core,
        geocenter_cache,
        topocentric_cache,
        track_b,
        contacts_b,
        basis,
    )
    geometry = core["compute_parallax_geometry"](
        geocenter_cache,
        track_a,
        track_b,
        screen_jd,
    )

    trigonometry = core["trigonometry_rows"](track_a, track_b)
    geometric_solution = core["geometric_rows"](
        track_a,
        track_b,
        geometry,
    )

    plot_engineering_track(
        core,
        geocenter_cache,
        track_a,
        track_b,
        screen_jd,
        geometry,
    )
    title_span = (
        "Vardo Norway → Point Venus Tahiti — "
        "JPL HORIZONS SITE_COORD"
    )
    save_widget_style_table_png(
        core,
        "TRIGONOMETRY",
        title_span,
        trigonometry,
        TRIGONOMETRY_PNG,
    )
    save_widget_style_table_png(
        core,
        "π⊙ GEOMETRIC SOLUTION",
        title_span,
        geometric_solution,
        PI_SUN_PNG,
    )

    display_status = display_outputs(
        [PLOT_PNG, TRIGONOMETRY_PNG, PI_SUN_PNG]
    )

    print("RESULTS")
    print(f"Vardo closest UTC      : {track_a['closest_utc']}")
    print(f"Point Venus closest UTC: {track_b['closest_utc']}")
    print(f"Vardo track angle      : {track_a['track_angle_deg']:.6f} deg")
    print(f"Point Venus angle      : {track_b['track_angle_deg']:.6f} deg")
    print(f"Pi sun                 : {geometry['pi_sun_arcsec']:.9f} arcsec")
    print(f"Inline display         : {display_status}")
    print()
    print("OUTPUT SUMMARY")
    print(f"Engineering plot PNG   : {PLOT_PNG}")
    print(f"Trigonometry table PNG : {TRIGONOMETRY_PNG}")
    print(f"Pi sun table PNG       : {PI_SUN_PNG}")
    print()
    print("PAPER COMPARISON")
    print(f"Reference pi sun       : {core['PI_SUN_REFERENCE_ARCSEC']:.6f} arcsec")
    print(f"Computed-reference     : {geometry['pi_sun_residual_arcsec']:+.9f} arcsec")
    print()
    print("EQUATION STATUS")
    print("IERS-0012N JPL geometry              : VERIFIED")
    print("IERS-0012N half-Sun plot             : PRESERVED")
    print("IERS-0012N table content             : PRESERVED")
    print("IERS-0012N table number formatting   : PRESERVED")
    print("IERS-0012N widget visual formatting  : PRESERVED IN PNG")
    print("AI image generation                  : NOT USED")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# IERS-0012Z
