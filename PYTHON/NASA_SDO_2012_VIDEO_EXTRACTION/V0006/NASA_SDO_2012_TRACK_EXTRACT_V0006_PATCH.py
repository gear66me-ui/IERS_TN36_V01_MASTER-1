# V0006
# Audit reference: Relax sparse-track acceptance while preserving span, speed, RMS, and angle safeguards.
from __future__ import annotations

import argparse
import runpy
from pathlib import Path

import numpy as np
import pandas as pd


def patch_source(source_path: Path) -> None:
    text = source_path.read_text(encoding="utf-8")

    if 'VERSION = "V0005"' not in text:
        raise RuntimeError("Expected V0005 source was not found.")
    text = text.replace('VERSION = "V0005"', 'VERSION = "V0006"', 1)

    csv_marker = 'CSV_NAME = "NASA_SDO_2012_TRACK.csv"\n'
    if csv_marker not in text:
        raise RuntimeError("CSV constant marker was not found.")
    text = text.replace(
        csv_marker,
        csv_marker + 'RAW_CSV_NAME = "NASA_SDO_2012_RAW_DETECTIONS.csv"\n',
        1,
    )

    old_acceptance = '''    minimum_count = max(180, int(round(0.045 * len(table))))
    if selected_count < minimum_count:
        raise RuntimeError(
            f"Continuous Venus track has only {selected_count} inliers; "
            f"minimum required is {minimum_count}."
        )
    if selected_span < 0.55:
        raise RuntimeError(
            f"Continuous Venus track spans only {selected_span:.3f} of the video."
        )
    if not (0.45 <= speed <= 2.80):
'''

    new_acceptance = '''    dense_minimum_count = max(180, int(round(0.045 * len(table))))
    sparse_minimum_count = max(100, int(round(0.018 * len(table))))
    print(
        "DEBUG | Venus track acceptance candidate | "
        f"selected={selected_count} | sparse_min={sparse_minimum_count} | "
        f"dense_min={dense_minimum_count} | span={selected_span:.6f} | "
        f"speed={speed:.6f} R_sun/video"
    )
    if selected_span < 0.55:
        raise RuntimeError(
            f"Continuous Venus track spans only {selected_span:.3f} of the video."
        )
    if selected_count < sparse_minimum_count:
        raise RuntimeError(
            f"Continuous Venus track has only {selected_count} inliers; "
            f"sparse minimum required is {sparse_minimum_count}."
        )
    if selected_count < dense_minimum_count and selected_span < 0.80:
        raise RuntimeError(
            f"Sparse Venus track has {selected_count} inliers but spans only "
            f"{selected_span:.3f}; sparse acceptance requires span >= 0.800."
        )
    if not (0.45 <= speed <= 2.80):
'''

    if old_acceptance not in text:
        raise RuntimeError("Track-acceptance patch location was not found.")
    text = text.replace(old_acceptance, new_acceptance, 1)

    function_marker = '\ndef read_frame(video_path: Path, frame_index: int) -> np.ndarray:\n'
    if function_marker not in text:
        raise RuntimeError("Metric-validation insertion marker was not found.")

    validation_function = '''

def validate_track_metrics(metrics: TrackMetrics) -> None:
    angle_delta = abs(
        (metrics.track_angle_deg - metrics.ols_angle_deg + 90.0) % 180.0 - 90.0
    )
    print(
        "DEBUG | final geometry validation | "
        f"tls_angle={metrics.track_angle_deg:.6f} deg | "
        f"ols_angle={metrics.ols_angle_deg:.6f} deg | "
        f"angle_delta={angle_delta:.6f} deg | "
        f"rms_perpendicular={metrics.rms_perpendicular_norm:.8f} R_sun"
    )
    if angle_delta > 5.0:
        raise RuntimeError(
            f"TLS and OLS track angles disagree by {angle_delta:.6f} deg."
        )
    if metrics.rms_perpendicular_norm > 0.080:
        raise RuntimeError(
            "Final track RMS is too large for a credible Venus trajectory: "
            f"{metrics.rms_perpendicular_norm:.8f} R_sun."
        )
'''
    text = text.replace(function_marker, validation_function + function_marker, 1)

    old_main = '''    raw_table, metadata = process_video(config)
    filtered_table = select_continuous_venus_track(raw_table)
    normalized_table = normalize_and_fill(filtered_table)
    final_table, metrics = fit_track(normalized_table)
'''

    new_main = '''    raw_table, metadata = process_video(config)
    raw_csv_path = config.output_dir / RAW_CSV_NAME
    raw_table.to_csv(raw_csv_path, index=False, float_format="%.10f")
    print(f"DEBUG | raw detections saved | {raw_csv_path}")
    filtered_table = select_continuous_venus_track(raw_table)
    normalized_table = normalize_and_fill(filtered_table)
    final_table, metrics = fit_track(normalized_table)
    validate_track_metrics(metrics)
'''

    if old_main not in text:
        raise RuntimeError("Main-workflow patch location was not found.")
    text = text.replace(old_main, new_main, 1)

    source_path.write_text(text, encoding="utf-8")


def run_regression_test(source_path: Path) -> None:
    namespace = runpy.run_path(str(source_path))
    select_track = namespace["select_continuous_venus_track"]
    normalize_and_fill = namespace["normalize_and_fill"]
    fit_track = namespace["fit_track"]
    validate_track_metrics = namespace["validate_track_metrics"]

    rng = np.random.default_rng(20120606)
    count = 5570
    frame = np.arange(count, dtype=float)
    solar_radius = np.full(count, 200.0)
    solar_cx = np.full(count, 256.0)
    solar_cy = np.full(count, 256.0)

    venus_cx = np.full(count, np.nan)
    venus_cy = np.full(count, np.nan)
    venus_radius = np.full(count, np.nan)
    venus_quality = np.full(count, np.nan)
    detected = np.zeros(count, dtype=bool)

    true_indices = np.unique(np.linspace(0, count - 1, 141).astype(int))
    available = np.setdiff1d(np.arange(count), true_indices)
    artifact_indices = rng.choice(available, size=520, replace=False)

    u_true = true_indices / (count - 1)
    true_x = -0.84 + 1.68 * u_true + rng.normal(0.0, 0.005, true_indices.size)
    true_y = 0.24 - 0.31 * u_true + rng.normal(0.0, 0.005, true_indices.size)

    venus_cx[true_indices] = solar_cx[true_indices] + true_x * solar_radius[true_indices]
    venus_cy[true_indices] = solar_cy[true_indices] - true_y * solar_radius[true_indices]
    venus_radius[true_indices] = 6.2 + rng.normal(0.0, 0.12, true_indices.size)
    venus_quality[true_indices] = 0.88
    detected[true_indices] = True

    venus_cx[artifact_indices] = solar_cx[artifact_indices] + (
        0.19 + rng.normal(0.0, 0.008, artifact_indices.size)
    ) * solar_radius[artifact_indices]
    venus_cy[artifact_indices] = solar_cy[artifact_indices] - (
        0.30 + rng.normal(0.0, 0.008, artifact_indices.size)
    ) * solar_radius[artifact_indices]
    venus_radius[artifact_indices] = 5.8 + rng.normal(0.0, 0.18, artifact_indices.size)
    venus_quality[artifact_indices] = 0.74
    detected[artifact_indices] = True

    table = pd.DataFrame(
        {
            "frame_index": frame,
            "time_s": frame / 29.97,
            "solar_cx_px": solar_cx,
            "solar_cy_px": solar_cy,
            "solar_radius_px": solar_radius,
            "solar_quality": np.ones(count),
            "solar_method": ["TEST"] * count,
            "venus_cx_px": venus_cx,
            "venus_cy_px": venus_cy,
            "venus_radius_px": venus_radius,
            "venus_quality": venus_quality,
            "venus_detected": detected,
            "venus_source": ["TEST"] * count,
        }
    )

    filtered = select_track(table)
    selected = filtered["venus_track_inlier"].to_numpy(dtype=bool)
    true_selected = int(np.count_nonzero(selected[true_indices]))
    artifact_selected = int(np.count_nonzero(selected[artifact_indices]))
    if true_selected < 110 or artifact_selected > 20:
        raise RuntimeError(
            "Sparse-track regression failed: "
            f"true_selected={true_selected}, artifact_selected={artifact_selected}."
        )

    normalized = normalize_and_fill(filtered)
    _, metrics = fit_track(normalized)
    validate_track_metrics(metrics)
    print(
        "Regression test: sparse full-span Venus track accepted | "
        f"true_selected={true_selected} | artifact_selected={artifact_selected}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    patch_source(args.target)
    run_regression_test(args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# V0006
