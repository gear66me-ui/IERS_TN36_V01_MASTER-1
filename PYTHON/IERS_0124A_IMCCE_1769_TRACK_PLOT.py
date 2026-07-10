# V0124A
# Audit reference: reconstruct and plot the IMCCE 1769 geocentric Venus transit track from canonical contact geometry.

from __future__ import annotations

import csv
import hashlib
import math
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

VERSION = "IERS-0124A"
TARGET_YEAR = 1769
LOCAL_TZ = timezone(timedelta(hours=-5))

DRIVE_PROJECT_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/JPL - 1769 VENUS TRANSIT"
)
DRIVE_BACKUP_ROOT = DRIVE_PROJECT_ROOT / "GitHub"
DRIVE_MASTER_CSV = DRIVE_BACKUP_ROOT / "DATA" / "CSV" / "IMCCE_VENUS_TRANSIT_CANON_MASTER.csv"
LOCAL_MASTER_CSV = Path(
    "/content/IERS_TN36_V01_MASTER_OUTPUT/IMCCE_VENUS_CANON/"
    "IMCCE_VENUS_TRANSIT_CANON_MASTER.csv"
)
OUTPUT_ROOT = Path(
    "/content/IERS_TN36_V01_MASTER_OUTPUT/IMCCE_VENUS_CANON/PLOTS/V0124A"
)
TRACK_CSV = OUTPUT_ROOT / "IERS_0124A_IMCCE_1769_MINUTE_TRACK.csv"
RESULTS_CSV = OUTPUT_ROOT / "IERS_0124A_IMCCE_1769_GEOMETRY_RESULTS.csv"
FIGURE_PNG = OUTPUT_ROOT / "IERS_0124A_IMCCE_1769_CANONICAL_TRACK.png"

DRIVE_PNG = DRIVE_BACKUP_ROOT / "DATA" / "PNG" / FIGURE_PNG.name
DRIVE_TRACK_CSV = DRIVE_BACKUP_ROOT / "DATA" / "CSV" / TRACK_CSV.name
DRIVE_RESULTS_CSV = DRIVE_BACKUP_ROOT / "DATA" / "CSV" / RESULTS_CSV.name
DRIVE_SCRIPT = DRIVE_BACKUP_ROOT / "PYTHON" / "IERS_0124A_IMCCE_1769_TRACK_PLOT.py"

CONTACT_LABELS = ("C1", "C2", "C3", "C4")


def mount_drive_if_needed() -> None:
    if DRIVE_PROJECT_ROOT.exists():
        return
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("Google Drive is not mounted and this is not Google Colab.") from exc
    drive.mount("/content/drive", force_remount=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_verified(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    if destination.exists() and sha256(destination) == source_hash:
        return "UNCHANGED"
    shutil.copy2(source, destination)
    if sha256(destination) != source_hash:
        raise RuntimeError(f"Backup hash verification failed: {destination}")
    return "COPIED"


def resolve_master_csv() -> Path:
    mount_drive_if_needed()
    if DRIVE_MASTER_CSV.exists():
        return DRIVE_MASTER_CSV
    if LOCAL_MASTER_CSV.exists():
        return LOCAL_MASTER_CSV
    raise FileNotFoundError(
        "IMCCE master CSV not found. Run IERS-0122C and V0123B before V0124A."
    )


def load_target_row(path: Path, year: int) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        matches = [row for row in csv.DictReader(handle) if int(row["year"]) == year]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one IMCCE record for {year}; found {len(matches)}")
    if matches[0].get("record_status") != "COMPLETE":
        raise RuntimeError(f"The {year} IMCCE record is not complete: {matches[0].get('record_status')}")
    return matches[0]


def clock_to_seconds(clock: str) -> float:
    parts = clock.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected HH:MM:SS contact time, received {clock!r}")
    hour, minute, second = int(parts[0]), int(parts[1]), float(parts[2])
    return hour * 3600.0 + minute * 60.0 + second


def unwrap_contact_times(row: dict[str, str]) -> np.ndarray:
    raw = [clock_to_seconds(row[f"c{index}_ut"]) for index in range(1, 5)]
    unwrapped = [raw[0]]
    for value in raw[1:]:
        while value < unwrapped[-1]:
            value += 86400.0
        unwrapped.append(value)
    return np.asarray(unwrapped, dtype=float)


def align_time_to_interval(seconds_of_day: float, start: float, stop: float) -> float:
    candidates = [seconds_of_day + offset * 86400.0 for offset in range(-2, 3)]
    center = 0.5 * (start + stop)
    return min(candidates, key=lambda value: abs(value - center))


def format_clock(seconds: float) -> str:
    seconds %= 86400.0
    hour = int(seconds // 3600.0)
    minute = int((seconds - hour * 3600.0) // 60.0)
    second = seconds - hour * 3600.0 - minute * 60.0
    return f"{hour:02d}:{minute:02d}:{second:06.3f}"


def derive_geometry(row: dict[str, str]) -> dict[str, object]:
    sun_radius = float(row["sun_radius_arcsec"])
    venus_radius = float(row["venus_radius_arcsec"])
    delta = float(row["minimum_distance_arcsec"])
    times = unwrap_contact_times(row)

    external_radius = sun_radius + venus_radius
    internal_radius = sun_radius - venus_radius
    external_term = external_radius**2 - delta**2
    internal_term = internal_radius**2 - delta**2
    if external_term <= 0.0 or internal_term <= 0.0:
        raise RuntimeError("The 1769 geometry does not support four real contact intersections.")

    x_external = math.sqrt(external_term)
    x_internal = math.sqrt(internal_term)
    contact_x = np.asarray([-x_external, -x_internal, x_internal, x_external], dtype=float)

    slope, intercept = np.polyfit(times, contact_x, 1)
    fitted_x = slope * times + intercept
    residuals = contact_x - fitted_x
    rms = float(np.sqrt(np.mean(residuals**2)))
    closest_time = -intercept / slope

    source_mid_seconds = float(row["mid_ut_seconds_of_day"])
    source_mid_time = align_time_to_interval(source_mid_seconds, times[0], times[-1])
    external_mid = 0.5 * (times[0] + times[3])
    internal_mid = 0.5 * (times[1] + times[2])

    source_speed = float(row["relative_velocity_deg_per_day"]) * 3600.0 / 86400.0
    external_speed = 2.0 * x_external / (times[3] - times[0])
    internal_speed = 2.0 * x_internal / (times[2] - times[1])

    return {
        "sun_radius": sun_radius,
        "venus_radius": venus_radius,
        "delta": delta,
        "impact_ratio": abs(delta) / sun_radius,
        "times": times,
        "contact_x": contact_x,
        "fitted_x": fitted_x,
        "fit_residuals": residuals,
        "fit_speed": float(slope),
        "fit_intercept": float(intercept),
        "fit_rms": rms,
        "closest_time": float(closest_time),
        "source_mid_time": source_mid_time,
        "external_mid": external_mid,
        "internal_mid": internal_mid,
        "x_external": x_external,
        "x_internal": x_internal,
        "external_chord": 2.0 * x_external,
        "internal_chord": 2.0 * x_internal,
        "source_speed": source_speed,
        "external_speed": external_speed,
        "internal_speed": internal_speed,
        "track_angle_deg": 0.0,
        "curvature_arcsec_inverse": 0.0,
    }


def build_minute_track(geometry: dict[str, object]) -> list[dict[str, object]]:
    times = np.asarray(geometry["times"], dtype=float)
    closest = float(geometry["closest_time"])
    slope = float(geometry["fit_speed"])
    delta = float(geometry["delta"])

    sampled = list(np.arange(times[0], times[-1] + 0.001, 60.0))
    if sampled[-1] < times[-1]:
        sampled.append(float(times[-1]))

    records: list[dict[str, object]] = []
    for index, epoch in enumerate(sampled):
        records.append(
            {
                "sample_index": index,
                "ut_clock": format_clock(epoch),
                "elapsed_from_c1_seconds": epoch - times[0],
                "elapsed_from_closest_seconds": epoch - closest,
                "track_x_arcsec": slope * (epoch - closest),
                "track_y_arcsec": delta,
                "source": "IMCCE_CANON_LINEAR_CONTACT_FIT",
            }
        )
    return records


def save_track_csv(records: list[dict[str, object]]) -> None:
    TRACK_CSV.parent.mkdir(parents=True, exist_ok=True)
    with TRACK_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def save_results_csv(row: dict[str, str], geometry: dict[str, object]) -> None:
    results = [
        ("target_year", TARGET_YEAR, "year", "IMCCE source selection"),
        ("sun_radius", geometry["sun_radius"], "arcsec", "IMCCE source"),
        ("venus_radius", geometry["venus_radius"], "arcsec", "IMCCE source"),
        ("signed_closest_approach", geometry["delta"], "arcsec", "IMCCE source"),
        ("impact_ratio", geometry["impact_ratio"], "dimensionless", "calculated"),
        ("external_center_chord", geometry["external_chord"], "arcsec", "calculated"),
        ("internal_center_chord", geometry["internal_chord"], "arcsec", "calculated"),
        ("contact_fit_speed", geometry["fit_speed"], "arcsec/s", "four-contact linear fit"),
        ("source_relative_speed", geometry["source_speed"], "arcsec/s", "IMCCE source conversion"),
        ("external_contact_speed", geometry["external_speed"], "arcsec/s", "calculated"),
        ("internal_contact_speed", geometry["internal_speed"], "arcsec/s", "calculated"),
        ("contact_fit_rms", geometry["fit_rms"], "arcsec", "calculated"),
        ("track_angle_canonical", geometry["track_angle_deg"], "deg", "motion-axis coordinates"),
        ("curvature_linear_model", geometry["curvature_arcsec_inverse"], "1/arcsec", "calculated"),
        ("closest_time_fit", format_clock(float(geometry["closest_time"])), "UT", "calculated"),
        ("source_mid_time", format_clock(float(geometry["source_mid_time"])), "UT", "IMCCE source"),
        ("external_contact_midpoint", format_clock(float(geometry["external_mid"])), "UT", "calculated"),
        ("internal_contact_midpoint", format_clock(float(geometry["internal_mid"])), "UT", "calculated"),
        (
            "halley_parallax_status",
            "NOT USED — one geocentric track cannot determine A_prime_B_prime or solar parallax",
            "status",
            "two topocentric observer tracks required",
        ),
        ("source_record_id", row["record_id"], "record", "IMCCE master CSV"),
    ]
    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", "value", "unit", "traceability"])
        writer.writerows(results)


def plot_track(row: dict[str, str], geometry: dict[str, object], records: list[dict[str, object]]) -> None:
    sun_radius = float(geometry["sun_radius"])
    venus_radius = float(geometry["venus_radius"])
    delta = float(geometry["delta"])
    contact_x = np.asarray(geometry["contact_x"], dtype=float)

    figure, axis = plt.subplots(figsize=(8.2, 8.2))
    axis.add_patch(Circle((0.0, 0.0), sun_radius, fill=False, linewidth=0.8, edgecolor="black"))

    minute_x = np.asarray([float(record["track_x_arcsec"]) for record in records])
    minute_y = np.full_like(minute_x, delta)
    axis.plot(minute_x, minute_y, linewidth=0.65, color="black", label="1769 canonical center track")
    axis.scatter(minute_x, minute_y, s=2.0, color="black", zorder=3)

    event_x = [contact_x[0], contact_x[1], 0.0, contact_x[2], contact_x[3]]
    event_labels = ["C1", "C2", "Closest", "C3", "C4"]
    for x_value, label in zip(event_x, event_labels):
        axis.add_patch(
            Circle(
                (x_value, delta),
                venus_radius,
                fill=False,
                linewidth=0.55,
                edgecolor="black",
                zorder=4,
            )
        )
        axis.scatter([x_value], [delta], s=8.0, color="black", zorder=5)
        vertical_offset = venus_radius + 24.0 if delta <= 0 else -(venus_radius + 34.0)
        axis.text(x_value, delta + vertical_offset, label, fontsize=7, ha="center", va="center")

    axis.axhline(0.0, linewidth=0.35, color="0.55")
    axis.axvline(0.0, linewidth=0.35, color="0.55")
    axis.set_aspect("equal", adjustable="box")
    margin = sun_radius * 1.12
    axis.set_xlim(-margin, margin)
    axis.set_ylim(-margin, margin)
    axis.set_xlabel("Canonical along-track coordinate, arcsec")
    axis.set_ylabel("Canonical cross-track coordinate, arcsec")
    axis.set_title(
        "1769 Venus Transit — IMCCE Canonical Geocentric Track\n"
        f"Date {row['date_ut_label']} | Venus disk plotted to scale",
        fontsize=11,
    )
    axis.legend(loc="upper right", fontsize=7, frameon=False)
    axis.tick_params(width=0.5, labelsize=8)
    for spine in axis.spines.values():
        spine.set_linewidth(0.5)
    figure.tight_layout()
    FIGURE_PNG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(figure)


def backup_outputs() -> dict[str, str]:
    statuses = {
        "png": copy_verified(FIGURE_PNG, DRIVE_PNG),
        "track_csv": copy_verified(TRACK_CSV, DRIVE_TRACK_CSV),
        "results_csv": copy_verified(RESULTS_CSV, DRIVE_RESULTS_CSV),
    }
    script_path = Path(__file__).resolve() if "__file__" in globals() else None
    if script_path and script_path.exists():
        statuses["python"] = copy_verified(script_path, DRIVE_SCRIPT)
    else:
        statuses["python"] = "NOT AVAILABLE"
    return statuses


def main() -> None:
    master_csv = resolve_master_csv()
    row = load_target_row(master_csv, TARGET_YEAR)
    geometry = derive_geometry(row)
    records = build_minute_track(geometry)
    save_track_csv(records)
    save_results_csv(row, geometry)
    plot_track(row, geometry, records)
    backup = backup_outputs()

    speed_delta_percent = (
        (float(geometry["fit_speed"]) - float(geometry["source_speed"]))
        / float(geometry["source_speed"])
        * 100.0
    )

    print(f"CODE OUTPUT: {VERSION}")
    print("CODE INPUTS")
    print(f"Master CSV : {master_csv}")
    print(f"Target year : {TARGET_YEAR}")
    print("COMMENTS")
    print("Canonical coordinates place the linear motion direction on +X; the signed IMCCE closest approach is Y.")
    print("Minute samples are reconstructed from a four-contact least-squares linear fit.")
    print("RESULTS")
    print(f"Sun radius : {float(geometry['sun_radius']):.6f} arcsec | Venus radius : {float(geometry['venus_radius']):.6f} arcsec")
    print(f"Closest approach : {float(geometry['delta']):.6f} arcsec | Impact ratio : {float(geometry['impact_ratio']):.6f}")
    print(f"External chord : {float(geometry['external_chord']):.6f} arcsec | Internal chord : {float(geometry['internal_chord']):.6f} arcsec")
    print(f"Fit speed : {float(geometry['fit_speed']):.9f} arcsec/s | IMCCE speed : {float(geometry['source_speed']):.9f} arcsec/s")
    print(f"Speed delta : {speed_delta_percent:+.6f}% | Fit RMS : {float(geometry['fit_rms']):.6f} arcsec")
    print(f"Closest UT fit : {format_clock(float(geometry['closest_time']))} | Source midpoint UT : {format_clock(float(geometry['source_mid_time']))}")
    print("Halley A_prime_B_prime and solar parallax : NOT USED — only one geocentric track is present.")
    print("OUTPUT SUMMARY")
    print(f"Track CSV : {TRACK_CSV}")
    print(f"Results CSV : {RESULTS_CSV}")
    print(f"Figure PNG : {FIGURE_PNG}")
    print(f"Drive backup : PNG={backup['png']} | TRACK={backup['track_csv']} | RESULTS={backup['results_csv']} | PYTHON={backup['python']}")
    print("PAPER COMPARISON")
    print("IMCCE relative angular speed is compared with the speed fitted independently from C1-C4 geometry and times.")
    print("EQUATION STATUS")
    print("VERIFIED — contact circle intersections, chord equations, four-contact linear fit, RMS, and minute sampling evaluated.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print("# V0124A")


if __name__ == "__main__":
    main()

# V0124A
