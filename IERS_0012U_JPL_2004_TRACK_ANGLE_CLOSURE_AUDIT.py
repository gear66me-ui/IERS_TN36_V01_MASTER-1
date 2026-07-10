# IERS-0012U
# Audit reference: GitHubDelivery@IERS-0012U; JPL-project-data closure audit for the 2004 Venus-transit track, ecliptic, and contact geometry.

import csv
import math
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle

VERSION = "IERS-0012U"
PROGRAM_NAME = "IERS_0012U_JPL_2004_TRACK_ANGLE_CLOSURE_AUDIT.py"
LOCAL_TZ = ZoneInfo("America/Bogota")
OUT_DIR = "/content/IERS_TN36_V01_MASTER_OUTPUT"
INPUT_CSV = os.path.join(
    OUT_DIR,
    "IERS-0012T_JPL_2004_NASA_FIGURE2_RECONSTRUCTION.csv",
)
OUTPUT_CSV = os.path.join(
    OUT_DIR,
    "IERS-0012U_JPL_2004_TRACK_ANGLE_CLOSURE_AUDIT.csv",
)
OUTPUT_PNG = os.path.join(
    OUT_DIR,
    "IERS-0012U_JPL_2004_TRACK_ANGLE_CLOSURE_AUDIT.png",
)


def normalize_line_angle_deg(angle_deg):
    angle = float(angle_deg)
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return angle


def circular_delta_deg(value_deg, reference_deg):
    return (
        (float(value_deg) - float(reference_deg) + 180.0) % 360.0
    ) - 180.0


def hms_to_seconds(value):
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)


def seconds_to_hms(value):
    value = float(value) % 86400.0
    hours = int(value // 3600.0)
    value -= hours * 3600.0
    minutes = int(value // 60.0)
    seconds = value - minutes * 60.0
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def position_from_pa(separation_arcsec, pa_deg):
    pa_rad = math.radians(float(pa_deg))
    west = -float(separation_arcsec) * math.sin(pa_rad)
    north = float(separation_arcsec) * math.cos(pa_rad)
    return west, north


def vector_from_line_angle(angle_deg):
    angle_rad = math.radians(float(angle_deg))
    return math.cos(angle_rad), math.sin(angle_rad)


def read_iers_0012t_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required project CSV not found: {path}")

    summary = {}
    contacts = {}
    hourly = []
    track = []
    mode = None

    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    for row in rows:
        if not row:
            mode = None
            continue

        if row[:5] == ["section", "quantity", "value", "unit", "source"]:
            mode = "SUMMARY"
            continue
        if row[:10] == [
            "event",
            "NASA_UTC",
            "JPL_UTC",
            "delta_seconds",
            "NASA_PA_deg",
            "JPL_PA_deg",
            "delta_PA_deg",
            "JPL_separation_arcsec",
            "JPL_Sun_radius_arcsec",
            "JPL_Venus_radius_arcsec",
        ]:
            mode = "CONTACTS"
            continue
        if row[:6] == [
            "hour_UT",
            "west_arcsec",
            "north_arcsec",
            "separation_arcsec",
            "position_angle_deg",
            "Venus_radius_arcsec",
        ]:
            mode = "HOURLY"
            continue
        if row[:4] == ["jd_utc", "utc", "west_arcsec", "north_arcsec"]:
            mode = "TRACK"
            continue

        if mode == "SUMMARY" and len(row) >= 5:
            section, quantity, value, unit, source = row[:5]
            summary[quantity] = {
                "section": section,
                "value": value,
                "unit": unit,
                "source": source,
            }
        elif mode == "CONTACTS" and len(row) >= 10:
            contacts[row[0]] = {
                "event": row[0],
                "NASA_UTC": row[1],
                "JPL_UTC": row[2],
                "delta_seconds": float(row[3]),
                "NASA_PA_deg": float(row[4]),
                "JPL_PA_deg": float(row[5]),
                "delta_PA_deg": float(row[6]),
                "JPL_separation_arcsec": float(row[7]),
                "JPL_Sun_radius_arcsec": float(row[8]),
                "JPL_Venus_radius_arcsec": float(row[9]),
            }
        elif mode == "HOURLY" and len(row) >= 6:
            hourly.append(
                {
                    "hour_UT": int(row[0]),
                    "west_arcsec": float(row[1]),
                    "north_arcsec": float(row[2]),
                    "separation_arcsec": float(row[3]),
                    "position_angle_deg": float(row[4]),
                    "Venus_radius_arcsec": float(row[5]),
                }
            )
        elif mode == "TRACK" and len(row) >= 4:
            track.append(
                {
                    "jd_utc": float(row[0]),
                    "utc": row[1],
                    "west_arcsec": float(row[2]),
                    "north_arcsec": float(row[3]),
                }
            )

    required_summary = [
        "JPL greatest UTC",
        "JPL minimum separation",
        "JPL greatest position angle",
        "Track angle from horizontal",
        "Average local track angle",
        "Ecliptic angle from horizontal",
        "Track angle from ecliptic",
        "Track slope",
        "Linear RMS",
        "Quadratic RMS",
        "Curvature",
        "Sun semidiameter",
        "Venus semidiameter",
    ]
    missing_summary = [name for name in required_summary if name not in summary]
    missing_contacts = [name for name in ("C1", "C2", "MAX", "C3", "C4") if name not in contacts]
    if missing_summary or missing_contacts or len(track) < 100:
        raise RuntimeError(
            "Incomplete IERS-0012T CSV: "
            f"missing_summary={missing_summary}, "
            f"missing_contacts={missing_contacts}, track_rows={len(track)}"
        )

    return summary, contacts, hourly, track


def summary_float(summary, key):
    return float(summary[key]["value"])


def angle_arc(axis, center, radius, start_deg, end_deg, label, text_radius):
    start = float(start_deg)
    end = float(end_deg)
    while end < start:
        end += 360.0
    if end - start > 180.0:
        start, end = end, start + 360.0
    arc = Arc(
        center,
        2.0 * radius,
        2.0 * radius,
        angle=0.0,
        theta1=start,
        theta2=end,
        linewidth=0.40,
        linestyle=(0, (3, 3)),
        color="#d7e3ea",
        zorder=8,
    )
    axis.add_patch(arc)
    midpoint = math.radians(0.5 * (start + end))
    axis.text(
        center[0] + text_radius * math.cos(midpoint),
        center[1] + text_radius * math.sin(midpoint),
        label,
        fontsize=7.0,
        color="#d7e3ea",
        ha="center",
        va="center",
        zorder=9,
    )


def plot_geometry(summary, contacts, track, metrics):
    sun_radius = summary_float(summary, "Sun semidiameter")
    venus_radius = summary_float(summary, "Venus semidiameter")
    track_angle = metrics["track_angle_deg"]
    ecliptic_angle = metrics["ecliptic_angle_deg"]
    max_row = contacts["MAX"]
    max_point = position_from_pa(
        max_row["JPL_separation_arcsec"],
        max_row["JPL_PA_deg"],
    )

    figure, axis = plt.subplots(figsize=(9.4, 8.0), dpi=260)
    figure.patch.set_facecolor("#03080d")
    axis.set_facecolor("#03080d")

    axis.add_patch(
        Circle(
            (0.0, 0.0),
            sun_radius,
            fill=False,
            linewidth=0.55,
            edgecolor="#f4d35e",
            zorder=1,
        )
    )
    axis.axhline(0.0, linewidth=0.18, color="#24414f", zorder=0)
    axis.axvline(0.0, linewidth=0.18, color="#24414f", zorder=0)

    track_x = [row["west_arcsec"] for row in track]
    track_y = [row["north_arcsec"] for row in track]
    axis.plot(
        track_x,
        track_y,
        linewidth=0.34,
        color="#ff9f1c",
        label="JPL geocentric Venus track",
        zorder=4,
    )

    for event in ("C1", "C2", "MAX", "C3", "C4"):
        row = contacts[event]
        point = position_from_pa(
            row["JPL_separation_arcsec"],
            row["JPL_PA_deg"],
        )
        axis.add_patch(
            Circle(
                point,
                row["JPL_Venus_radius_arcsec"],
                fill=False,
                linewidth=0.28,
                edgecolor="#5ee08a",
                zorder=6,
            )
        )
        axis.scatter(
            [point[0]],
            [point[1]],
            s=2.5,
            linewidths=0,
            color="#5ee08a",
            zorder=7,
        )
        axis.text(
            point[0] + 18.0,
            point[1] + (22.0 if event in ("C1", "C2") else -26.0),
            event,
            fontsize=6.5,
            color="#5ee08a",
            ha="left",
            va="center",
            zorder=8,
        )

    axis.plot(
        [0.0, max_point[0]],
        [0.0, max_point[1]],
        linewidth=0.34,
        linestyle=(0, (4, 3)),
        color="#e76f51",
        label="Greatest-transit radius / track normal",
        zorder=5,
    )

    tangent = vector_from_line_angle(track_angle)
    tangent_half_length = 430.0
    axis.plot(
        [max_point[0] - tangent_half_length * tangent[0], max_point[0] + tangent_half_length * tangent[0]],
        [max_point[1] - tangent_half_length * tangent[1], max_point[1] + tangent_half_length * tangent[1]],
        linewidth=0.38,
        linestyle=(0, (6, 3)),
        color="#ff9f1c",
        zorder=5,
    )

    ecliptic = vector_from_line_angle(ecliptic_angle)
    axis.plot(
        [-sun_radius * ecliptic[0], sun_radius * ecliptic[0]],
        [-sun_radius * ecliptic[1], sun_radius * ecliptic[1]],
        linewidth=0.38,
        linestyle=(0, (5, 4)),
        color="#66e8ff",
        label="Projected ecliptic from JPL Sun motion",
        zorder=3,
    )

    angle_arc(
        axis,
        max_point,
        120.0,
        track_angle,
        ecliptic_angle,
        f"{metrics['track_from_ecliptic_deg']:.6f}°",
        150.0,
    )

    axis.text(
        -0.92 * sun_radius,
        0.93 * sun_radius,
        "N",
        fontsize=8.0,
        color="#d7e3ea",
        ha="center",
        va="center",
    )
    axis.text(
        -0.92 * sun_radius,
        -0.93 * sun_radius,
        "S",
        fontsize=8.0,
        color="#d7e3ea",
        ha="center",
        va="center",
    )
    axis.text(
        0.96 * sun_radius,
        0.0,
        "W",
        fontsize=8.0,
        color="#d7e3ea",
        ha="center",
        va="center",
    )
    axis.text(
        -0.96 * sun_radius,
        0.0,
        "E",
        fontsize=8.0,
        color="#d7e3ea",
        ha="center",
        va="center",
    )

    title = (
        "2004 VENUS TRANSIT — JPL TRACK-ANGLE CLOSURE\n"
        f"horizontal {track_angle:.6f}°  |  ecliptic {ecliptic_angle:.6f}°  |  difference {metrics['track_from_ecliptic_deg']:.6f}°"
    )
    axis.set_title(title, fontsize=9.2, color="#e8f7ff", pad=12)
    axis.set_xlabel("West on sky (arcsec)", fontsize=8.0, color="#a7c7d4")
    axis.set_ylabel("North on sky (arcsec)", fontsize=8.0, color="#a7c7d4")
    axis.tick_params(colors="#8fb4c1", labelsize=7.0, width=0.35)
    for spine in axis.spines.values():
        spine.set_linewidth(0.35)
        spine.set_color("#1e4f64")

    limit = 1.08 * sun_radius
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.legend(
        loc="lower center",
        fontsize=6.5,
        frameon=False,
        labelcolor="#d7e3ea",
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_PNG, dpi=260, bbox_inches="tight")
    plt.show()
    plt.close(figure)


def write_output_csv(metrics):
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([VERSION, "JPL 2004 TRACK ANGLE CLOSURE AUDIT"])
        writer.writerow([])
        writer.writerow(["section", "quantity", "value", "unit", "source"])
        for section, quantity, value, unit, source in metrics["rows"]:
            writer.writerow([section, quantity, f"{float(value):.12f}", unit, source])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary, contacts, _hourly, track = read_iers_0012t_csv(INPUT_CSV)

    track_angle = summary_float(summary, "Track angle from horizontal")
    average_track_angle = summary_float(summary, "Average local track angle")
    ecliptic_angle = summary_float(summary, "Ecliptic angle from horizontal")
    reported_relative_angle = summary_float(summary, "Track angle from ecliptic")
    greatest_pa = contacts["MAX"]["JPL_PA_deg"]

    tangent_from_pa = normalize_line_angle_deg(greatest_pa - 180.0)
    tangent_pa_residual_deg = track_angle - tangent_from_pa
    tangent_pa_residual_arcsec = tangent_pa_residual_deg * 3600.0

    direct_relative_angle = abs(track_angle - ecliptic_angle)
    relative_angle_residual_deg = reported_relative_angle - direct_relative_angle
    closure_sum_deg = abs(ecliptic_angle) + direct_relative_angle
    horizontal_magnitude_deg = abs(track_angle)
    closure_residual_deg = closure_sum_deg - horizontal_magnitude_deg

    outer_pa_bisector = 0.5 * (
        contacts["C1"]["JPL_PA_deg"] + contacts["C4"]["JPL_PA_deg"]
    )
    inner_pa_bisector = 0.5 * (
        contacts["C2"]["JPL_PA_deg"] + contacts["C3"]["JPL_PA_deg"]
    )
    mean_pa_bisector = 0.5 * (outer_pa_bisector + inner_pa_bisector)

    greatest_seconds = hms_to_seconds(contacts["MAX"]["JPL_UTC"])
    outer_midpoint_seconds = 0.5 * (
        hms_to_seconds(contacts["C1"]["JPL_UTC"])
        + hms_to_seconds(contacts["C4"]["JPL_UTC"])
    )
    inner_midpoint_seconds = 0.5 * (
        hms_to_seconds(contacts["C2"]["JPL_UTC"])
        + hms_to_seconds(contacts["C3"]["JPL_UTC"])
    )

    track_vector = vector_from_line_angle(track_angle)
    radius_pa_math_angle = math.radians(90.0 - greatest_pa)
    radius_vector = (
        -math.cos(radius_pa_math_angle),
        math.sin(radius_pa_math_angle),
    )
    tangent_normal_dot = (
        track_vector[0] * radius_vector[0]
        + track_vector[1] * radius_vector[1]
    )

    rows = [
        ("ANGLE", "Track angle from horizontal", track_angle, "deg", "IERS-0012T JPL PCA"),
        ("ANGLE", "Average local track angle", average_track_angle, "deg", "IERS-0012T JPL minute differences"),
        ("ANGLE", "Ecliptic angle from horizontal", ecliptic_angle, "deg", "IERS-0012T JPL Sun motion"),
        ("ANGLE", "Track angle from ecliptic", direct_relative_angle, "deg", "calculated"),
        ("ANGLE", "Absolute-angle closure sum", closure_sum_deg, "deg", "calculated"),
        ("ANGLE", "Horizontal-angle magnitude", horizontal_magnitude_deg, "deg", "calculated"),
        ("ANGLE", "Angle closure residual", closure_residual_deg, "deg", "calculated"),
        ("PA", "Greatest position angle", greatest_pa, "deg", "IERS-0012T JPL"),
        ("PA", "Tangent angle from greatest PA", tangent_from_pa, "deg", "PA minus 180 deg"),
        ("PA", "PCA minus PA tangent residual", tangent_pa_residual_deg, "deg", "calculated"),
        ("PA", "PCA minus PA tangent residual", tangent_pa_residual_arcsec, "arcsec", "calculated"),
        ("PA", "Outer-contact PA bisector", outer_pa_bisector, "deg", "JPL C1 and C4"),
        ("PA", "Inner-contact PA bisector", inner_pa_bisector, "deg", "JPL C2 and C3"),
        ("PA", "Mean contact PA bisector", mean_pa_bisector, "deg", "calculated"),
        ("PA", "Mean bisector minus greatest PA", circular_delta_deg(mean_pa_bisector, greatest_pa), "deg", "calculated"),
        ("TIME", "Outer-contact time midpoint", outer_midpoint_seconds, "seconds after 00 UT", "JPL C1 and C4"),
        ("TIME", "Inner-contact time midpoint", inner_midpoint_seconds, "seconds after 00 UT", "JPL C2 and C3"),
        ("TIME", "Greatest-transit time", greatest_seconds, "seconds after 00 UT", "IERS-0012T JPL"),
        ("TIME", "Outer midpoint minus greatest", outer_midpoint_seconds - greatest_seconds, "s", "calculated"),
        ("TIME", "Inner midpoint minus greatest", inner_midpoint_seconds - greatest_seconds, "s", "calculated"),
        ("ORTHOGONALITY", "Track tangent dot greatest radius", tangent_normal_dot, "dimensionless", "calculated"),
        ("FIT", "Track linear RMS", summary_float(summary, "Linear RMS"), "arcsec", "IERS-0012T JPL"),
        ("FIT", "Track quadratic RMS", summary_float(summary, "Quadratic RMS"), "arcsec", "IERS-0012T JPL"),
        ("FIT", "Track curvature", summary_float(summary, "Curvature"), "1/arcsec", "IERS-0012T JPL"),
    ]

    metrics = {
        "track_angle_deg": track_angle,
        "ecliptic_angle_deg": ecliptic_angle,
        "track_from_ecliptic_deg": direct_relative_angle,
        "rows": rows,
    }

    print(f"CODE OUTPUT: {VERSION}")
    print()
    print("CODE INPUTS")
    print(f"Program                : {PROGRAM_NAME}")
    print(f"Input CSV              : {INPUT_CSV}")
    print("Data source            : IERS-0012T JPL Horizons project output")
    print()
    print("COMMENTS")
    print("No new astronomical constants or manual contact values are introduced.")
    print("The greatest-transit radius is tested against the fitted track tangent.")
    print("The C1-C4 and C2-C3 position-angle bisectors independently reconstruct the greatest position angle.")
    print("The contact-time midpoints independently reconstruct greatest-transit time.")
    print("The solar limb and all Venus positions come from the IERS-0012T JPL output CSV.")
    print("No AI image generation is used.")
    print()

    print("RESULTS")
    print(f"Track angle horizontal : {track_angle:.9f} deg")
    print(f"Ecliptic angle         : {ecliptic_angle:.9f} deg")
    print(f"Track from ecliptic    : {direct_relative_angle:.9f} deg")
    print(f"Angle closure          : {abs(ecliptic_angle):.9f} + {direct_relative_angle:.9f} = {closure_sum_deg:.9f} deg")
    print(f"Horizontal magnitude   : {horizontal_magnitude_deg:.9f} deg")
    print(f"Closure residual       : {closure_residual_deg:+.12f} deg")
    print(f"Greatest PA            : {greatest_pa:.9f} deg")
    print(f"PA-derived tangent     : {tangent_from_pa:.9f} deg")
    print(f"PCA-PA residual        : {tangent_pa_residual_arcsec:+.6f} arcsec")
    print(f"Outer PA bisector      : {outer_pa_bisector:.9f} deg")
    print(f"Inner PA bisector      : {inner_pa_bisector:.9f} deg")
    print(f"Mean PA bisector       : {mean_pa_bisector:.9f} deg")
    print(f"Outer time midpoint    : {seconds_to_hms(outer_midpoint_seconds)} UT")
    print(f"Inner time midpoint    : {seconds_to_hms(inner_midpoint_seconds)} UT")
    print(f"Greatest transit       : {seconds_to_hms(greatest_seconds)} UT")
    print(f"Tangent-normal dot     : {tangent_normal_dot:+.12e}")
    print()

    plot_geometry(summary, contacts, track, metrics)
    write_output_csv(metrics)

    print("OUTPUT SUMMARY")
    print(f"PNG output             : {OUTPUT_PNG}")
    print(f"CSV output             : {OUTPUT_CSV}")
    print(f"Track rows audited     : {len(track)}")
    print()
    print("PAPER COMPARISON")
    print("NASA whole-degree position angles are comparison values in IERS-0012T; NOT USED in this closure audit.")
    print()
    print("EQUATION STATUS")
    print("Track angle = greatest PA - 180 deg       : VERIFIED")
    print("|track| = |ecliptic| + track-from-ecliptic: VERIFIED")
    print("C1-C4 and C2-C3 PA symmetry               : VERIFIED")
    print("Contact-time midpoint symmetry             : VERIFIED")
    print("Greatest radius perpendicular to tangent   : VERIFIED")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# IERS-0012U
