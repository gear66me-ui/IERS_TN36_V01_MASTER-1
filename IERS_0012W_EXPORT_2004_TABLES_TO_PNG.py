# IERS-0012W
# Audit reference: GitHubDelivery@IERS-0012W; verified IERS-0012U JPL core with separate presentation-ready PNG exports for both scientific tables.

import hashlib
import time
import urllib.request

VERSION = "IERS-0012W"
PROGRAM_NAME = "IERS_0012W_EXPORT_2004_TABLES_TO_PNG.py"
BASE_PROGRAM_NAME = "IERS_0012U_JPL_2004_NASA_FIGURE2_RECONSTRUCTION_LABEL_LAYOUT.py"
BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    + BASE_PROGRAM_NAME
)
DOWNLOAD_ATTEMPTS = 4
DOWNLOAD_TIMEOUT_SECONDS = 45


def fetch_verified_base_source():
    last_error = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(
                BASE_URL,
                headers={
                    "User-Agent": "Mozilla/5.0 IERS-0012W",
                    "Accept": "text/plain",
                },
            )
            with urllib.request.urlopen(
                request,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            ) as response:
                source_bytes = response.read()
            source = source_bytes.decode("utf-8")
            if not source.startswith("# IERS-0012U\n"):
                raise RuntimeError(
                    "Downloaded scientific core does not begin with the "
                    "required IERS-0012U version marker."
                )
            if not source.rstrip().endswith("# IERS-0012U"):
                raise RuntimeError(
                    "Downloaded scientific core does not end with the "
                    "required IERS-0012U version marker."
                )
            required_tokens = (
                "def query_jpl_ephemerides",
                "def derive_events",
                "def build_track",
                "def build_ecliptic_fit",
                "def display_widgets",
                "def write_csv",
                "def plot_reconstruction",
                "Greatest-transit centered-seconds minimum",
                "No AI image generation is used.",
            )
            missing = [token for token in required_tokens if token not in source]
            if missing:
                raise RuntimeError(
                    f"Downloaded scientific core is incomplete; missing {missing}."
                )
            return source, hashlib.sha256(source_bytes).hexdigest()
        except Exception as exc:
            last_error = exc
            if attempt < DOWNLOAD_ATTEMPTS:
                time.sleep(2.0 * attempt)
    raise RuntimeError(
        "Unable to download the verified IERS-0012U scientific core "
        f"after {DOWNLOAD_ATTEMPTS} attempts: {last_error}"
    )


def replace_exactly_once(source, old, new, description):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Patch audit failed for {description}: expected 1 occurrence, "
            f"found {count}."
        )
    return source.replace(old, new, 1)


def build_iers_0012w_source(base_source):
    source = base_source

    source = replace_exactly_once(
        source,
        "# Audit reference: GitHubDelivery@IERS-0012U; preserve IERS-0012U "
        "JPL geometry and separate hourly/contact labels from the transit track.",
        "# Audit reference: GitHubDelivery@IERS-0012W; preserve verified JPL "
        "geometry and export both scientific tables as presentation-ready PNG files.",
        "audit reference",
    )
    source = replace_exactly_once(
        source,
        BASE_PROGRAM_NAME,
        PROGRAM_NAME,
        "program name",
    )
    source = source.replace("IERS-0012U", VERSION)

    constants_anchor = "NASA_FALLBACK = {"
    constants_block = '''GEOMETRY_TABLE_PNG_PATH = os.path.join(
    OUT_DIR,
    "IERS-0012W_JPL_GEOCENTRIC_TRACK_GEOMETRY_TABLE.png",
)
CONTACT_TABLE_PNG_PATH = os.path.join(
    OUT_DIR,
    "IERS-0012W_NASA_GSFC_VS_JPL_HORIZONS_CONTACT_TABLE.png",
)


'''
    source = replace_exactly_once(
        source,
        constants_anchor,
        constants_block + constants_anchor,
        "table PNG output paths",
    )

    function_anchor = "def write_csv(\n"
    table_functions = '''def save_styled_table_png(
    title,
    columns,
    rows,
    output_path,
    column_widths,
    footer=None,
):
    row_count = len(rows)
    figure_height = max(4.2, 1.55 + 0.46 * (row_count + 1))
    figure_width = 13.6

    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height),
        dpi=220,
    )
    figure.patch.set_facecolor("#03080d")
    axis.set_facecolor("#03080d")
    axis.axis("off")

    axis.text(
        0.5,
        0.965,
        title,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=15.0,
        fontweight="bold",
        color="#66e8ff",
    )
    axis.plot(
        [0.02, 0.98],
        [0.918, 0.918],
        transform=axis.transAxes,
        linewidth=0.7,
        color="#25708b",
        clip_on=False,
    )

    bottom_margin = 0.085 if footer else 0.045
    table = axis.table(
        cellText=[[str(value) for value in row] for row in rows],
        colLabels=[str(value) for value in columns],
        cellLoc="left",
        colLoc="left",
        colWidths=column_widths,
        bbox=[0.02, bottom_margin, 0.96, 0.80],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.3)

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor("#16333f")
        cell.PAD = 0.055
        if row_index == 0:
            cell.set_facecolor("#0a1a22")
            cell.get_text().set_color("#66e8ff")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_ha("left")
        else:
            cell.set_facecolor("#050b0f")
            cell.get_text().set_color("#e8f7ff")
            if column_index == 1:
                cell.get_text().set_color("#ffc861")
                cell.get_text().set_fontweight("bold")
            elif column_index >= 2:
                cell.get_text().set_color("#b6f5ca")

    if footer:
        axis.text(
            0.02,
            0.028,
            footer,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.0,
            color="#8fb4c1",
        )

    figure.savefig(
        output_path,
        dpi=360,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
        edgecolor="none",
        pad_inches=0.08,
    )
    plt.close(figure)


def export_table_pngs(
    greatest_state,
    track,
    ecliptic,
    comparison,
    nasa_minimum_separation,
    nasa_status,
):
    track_angle = track["fit"]["angle_deg"]
    ecliptic_angle = ecliptic["fit"]["angle_deg"]
    relative_angle = acute_line_difference_deg(
        track_angle,
        ecliptic_angle,
    )

    geometry_rows = [
        ["JPL greatest UTC", utc_hms(greatest_state["jd_utc"], 3), "UT"],
        ["JPL minimum separation", f"{greatest_state['separation_arcsec']:.6f}", "arcsec"],
        ["NASA minimum separation", f"{nasa_minimum_separation:.3f}", "arcsec"],
        ["JPL greatest position angle", f"{greatest_state['position_angle_deg']:.6f}", "deg"],
        ["Track angle from horizontal", f"{track_angle:.6f}", "deg"],
        ["Average local track angle", f"{track['fit']['average_local_angle_deg']:.6f}", "deg"],
        ["Ecliptic angle from horizontal", f"{ecliptic_angle:.6f}", "deg"],
        ["Track angle from ecliptic", f"{relative_angle:.6f}", "deg"],
        ["Track slope", f"{track['fit']['slope']:.10f}", "dy/dx"],
        ["Linear-fit RMS", f"{track['fit']['rms_arcsec']:.9f}", "arcsec"],
        ["Quadratic-fit RMS", f"{track['fit']['quadratic_rms_arcsec']:.9f}", "arcsec"],
        ["Curvature", f"{track['fit']['curvature_per_arcsec']:.12e}", "1/arcsec"],
        ["JPL Sun semidiameter", f"{greatest_state['sun_radius_arcsec']:.6f}", "arcsec"],
        ["JPL Venus semidiameter", f"{greatest_state['venus_radius_arcsec']:.6f}", "arcsec"],
        ["Sun apparent RA", format_ra(greatest_state["sun_ra_rad"]), "h:m:s"],
        ["Sun apparent Dec", format_dec(greatest_state["sun_dec_rad"]), "d:m:s"],
        ["Venus apparent RA", format_ra(greatest_state["venus_ra_rad"]), "h:m:s"],
        ["Venus apparent Dec", format_dec(greatest_state["venus_dec_rad"]), "d:m:s"],
    ]

    contact_rows = []
    for row in comparison:
        contact_rows.append(
            [
                row["event"],
                row["nasa_utc"],
                row["jpl_utc"],
                f"{row['delta_seconds']:+.3f}",
                f"{row['nasa_pa_deg']:.3f}",
                f"{row['jpl_pa_deg']:.6f}",
                f"{row['delta_pa_deg']:+.6f}",
            ]
        )

    save_styled_table_png(
        "JPL GEOCENTRIC TRACK GEOMETRY — 2004 VENUS TRANSIT",
        ["Quantity", "Value", "Unit"],
        geometry_rows,
        GEOMETRY_TABLE_PNG_PATH,
        [0.55, 0.27, 0.18],
        footer=(
            "JPL Horizons geocentric apparent ephemerides; NASA values are "
            "published comparisons only."
        ),
    )
    save_styled_table_png(
        "NASA GSFC CONTACT TABLE vs JPL HORIZONS",
        [
            "Event",
            "NASA UT",
            "JPL UT",
            "Δt s",
            "NASA PA°",
            "JPL PA°",
            "ΔPA°",
        ],
        contact_rows,
        CONTACT_TABLE_PNG_PATH,
        [0.10, 0.15, 0.18, 0.12, 0.14, 0.16, 0.15],
        footer=f"NASA table source: {NASA_URL} | {nasa_status}",
    )

    try:
        from IPython.display import Image, display

        display(Image(filename=GEOMETRY_TABLE_PNG_PATH))
        display(Image(filename=CONTACT_TABLE_PNG_PATH))
    except Exception:
        pass


'''
    source = replace_exactly_once(
        source,
        function_anchor,
        table_functions + function_anchor,
        "PNG table export functions",
    )

    call_anchor = '''    plot_reconstruction(
        splines,
        events,
        track,
        ecliptic,
        hourly,
        greatest_state,
    )
'''
    call_replacement = '''    export_table_pngs(
        greatest_state,
        track,
        ecliptic,
        comparison,
        nasa_minimum_separation,
        nasa_status,
    )
    plot_reconstruction(
        splines,
        events,
        track,
        ecliptic,
        hourly,
        greatest_state,
    )
'''
    source = replace_exactly_once(
        source,
        call_anchor,
        call_replacement,
        "PNG table export call",
    )

    output_anchor = '    print(f"CSV output             : {CSV_PATH}")'
    output_replacement = '''    print(f"CSV output             : {CSV_PATH}")
    print(f"Geometry table PNG     : {GEOMETRY_TABLE_PNG_PATH}")
    print(f"Contact table PNG      : {CONTACT_TABLE_PNG_PATH}")'''
    source = replace_exactly_once(
        source,
        output_anchor,
        output_replacement,
        "PNG output summary",
    )

    no_ai_line = '    print("No AI image generation is used.")'
    expanded_comments = '''    print("Both scientific tables are exported as separate high-resolution Matplotlib PNG files.")
    print("The PNG tables preserve the dark engineering-widget presentation style.")
    print("No AI image generation is used.")'''
    source = replace_exactly_once(
        source,
        no_ai_line,
        expanded_comments,
        "table export comments",
    )

    required_revisions = (
        "GEOMETRY_TABLE_PNG_PATH",
        "CONTACT_TABLE_PNG_PATH",
        "def save_styled_table_png",
        "def export_table_pngs",
        "JPL GEOCENTRIC TRACK GEOMETRY — 2004 VENUS TRANSIT",
        "NASA GSFC CONTACT TABLE vs JPL HORIZONS",
        "Geometry table PNG",
        "Contact table PNG",
    )
    missing = [token for token in required_revisions if token not in source]
    if missing:
        raise RuntimeError(f"IERS-0012W revision audit failed; missing {missing}.")
    if "TODO" in source:
        raise RuntimeError("IERS-0012W audit rejected a TODO statement.")
    if "placeholder" in source.lower():
        raise RuntimeError("IERS-0012W audit rejected placeholder content.")
    if not source.startswith("# IERS-0012W\n"):
        raise RuntimeError("IERS-0012W first-line version audit failed.")
    if not source.rstrip().endswith("# IERS-0012W"):
        raise RuntimeError("IERS-0012W last-line version audit failed.")
    return source


def main():
    base_source, base_sha256 = fetch_verified_base_source()
    expanded_source = build_iers_0012w_source(base_source)
    compiled = compile(
        expanded_source,
        PROGRAM_NAME,
        "exec",
        dont_inherit=True,
        optimize=0,
    )

    print(f"[SUCCESS] Verified scientific core fetched from GitHub: {BASE_PROGRAM_NAME}")
    print(f"[AUDIT] Base SHA-256: {base_sha256}")
    print("[AUDIT] IERS-0012W expanded source compile check: PASS")
    print("[AUDIT] JPL calculations unchanged; two PNG table exports added")
    print("-" * 74)

    namespace = {
        "__name__": "__main__",
        "__file__": PROGRAM_NAME,
        "__package__": None,
        "__cached__": None,
    }
    exec(compiled, namespace, namespace)


if __name__ == "__main__":
    main()
# IERS-0012W
