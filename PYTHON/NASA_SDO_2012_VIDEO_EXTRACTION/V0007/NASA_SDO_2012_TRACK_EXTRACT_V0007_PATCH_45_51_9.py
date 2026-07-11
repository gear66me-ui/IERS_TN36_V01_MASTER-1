# V0007
# Audit reference: Restrict NASA SDO extraction to the user-verified full-Sun interval 45.000-51.900 seconds.
from __future__ import annotations

import argparse
import math
import runpy
from pathlib import Path


DEFAULT_ANALYSIS_START_S = 45.000
DEFAULT_ANALYSIS_END_S = 51.900


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one patch location, found {count}.")
    return text.replace(old, new, 1)


def patch_source(source_path: Path) -> None:
    text = source_path.read_text(encoding="utf-8")

    text = replace_once(text, 'VERSION = "V0006"', 'VERSION = "V0007"', "version")

    text = replace_once(
        text,
        'OVERLAY_NAME = "NASA_SDO_2012_TRACK_OVERLAY.png"\n',
        'OVERLAY_NAME = "NASA_SDO_2012_TRACK_OVERLAY.png"\n'
        'DEFAULT_ANALYSIS_START_S = 45.000\n'
        'DEFAULT_ANALYSIS_END_S = 51.900\n',
        "analysis constants",
    )

    text = replace_once(
        text,
        '    min_venus_quality: float = 0.20\n',
        '    min_venus_quality: float = 0.20\n'
        '    analysis_start_s: float = DEFAULT_ANALYSIS_START_S\n'
        '    analysis_end_s: float = DEFAULT_ANALYSIS_END_S\n',
        "Config time fields",
    )

    text = replace_once(
        text,
        '    parser.add_argument("--global-limb-retry", type=int, default=90)\n'
        '    args = parser.parse_args(argv)\n',
        '    parser.add_argument("--global-limb-retry", type=int, default=90)\n'
        '    parser.add_argument(\n'
        '        "--analysis-start-s", type=float, default=DEFAULT_ANALYSIS_START_S\n'
        '    )\n'
        '    parser.add_argument(\n'
        '        "--analysis-end-s", type=float, default=DEFAULT_ANALYSIS_END_S\n'
        '    )\n'
        '    args = parser.parse_args(argv)\n'
        '    if not math.isfinite(args.analysis_start_s):\n'
        '        parser.error("--analysis-start-s must be finite.")\n'
        '    if not math.isfinite(args.analysis_end_s):\n'
        '        parser.error("--analysis-end-s must be finite.")\n'
        '    if args.analysis_start_s < 0.0:\n'
        '        parser.error("--analysis-start-s must be non-negative.")\n'
        '    if args.analysis_end_s <= args.analysis_start_s:\n'
        '        parser.error("--analysis-end-s must exceed --analysis-start-s.")\n',
        "parser time arguments",
    )

    text = replace_once(
        text,
        '        global_limb_retry=max(1, args.global_limb_retry),\n'
        '    )\n',
        '        global_limb_retry=max(1, args.global_limb_retry),\n'
        '        analysis_start_s=float(args.analysis_start_s),\n'
        '        analysis_end_s=float(args.analysis_end_s),\n'
        '    )\n',
        "Config constructor time fields",
    )

    text = replace_once(
        text,
        '''    metadata = video_metadata(cap)
    cap.set(cv2.CAP_PROP_POS_FRAMES, config.start_frame)

    records: list[FrameRecord] = []
''',
        '''    metadata = video_metadata(cap)
    fps = float(metadata["fps"])
    time_start_frame = int(math.ceil(config.analysis_start_s * fps))
    time_end_frame = int(math.ceil(config.analysis_end_s * fps)) - 1
    effective_start_frame = max(config.start_frame, time_start_frame)
    effective_end_frame = time_end_frame
    if config.end_frame is not None:
        effective_end_frame = min(effective_end_frame, config.end_frame)
    if effective_end_frame < effective_start_frame:
        raise RuntimeError(
            "The requested frame limits do not overlap the analysis time window."
        )
    cap.set(cv2.CAP_PROP_POS_FRAMES, effective_start_frame)
    metadata["analysis_start_s"] = float(config.analysis_start_s)
    metadata["analysis_end_s_exclusive"] = float(config.analysis_end_s)
    metadata["analysis_start_frame"] = int(effective_start_frame)
    metadata["analysis_end_frame"] = int(effective_end_frame)
    metadata["analysis_frame_count_expected"] = int(
        effective_end_frame - effective_start_frame + 1
    )
    print(
        "DEBUG | verified full-Sun analysis window | "
        f"start_s={config.analysis_start_s:.3f} | "
        f"end_s_exclusive={config.analysis_end_s:.3f} | "
        f"start_frame={effective_start_frame} | "
        f"end_frame={effective_end_frame}"
    )

    records: list[FrameRecord] = []
''',
        "process-video time window",
    )

    text = replace_once(
        text,
        '    frame_index = config.start_frame\n',
        '    frame_index = effective_start_frame\n',
        "effective start frame",
    )

    text = replace_once(
        text,
        '            if config.end_frame is not None and frame_index > config.end_frame:\n'
        '                break\n',
        '            if frame_index > effective_end_frame:\n'
        '                break\n',
        "effective end frame",
    )

    text = text.replace('if np.count_nonzero(finite) < 80:', 'if np.count_nonzero(finite) < 30:')
    text = text.replace('if candidate_indices.size < 80:', 'if candidate_indices.size < 30:')
    text = text.replace('if count < 40:', 'if count < 24:')

    text = replace_once(
        text,
        '    dense_minimum_count = max(180, int(round(0.045 * len(table))))\n'
        '    sparse_minimum_count = max(100, int(round(0.018 * len(table))))\n',
        '    dense_minimum_count = max(90, int(round(0.45 * len(table))))\n'
        '    sparse_minimum_count = max(30, int(round(0.15 * len(table))))\n',
        "short-window acceptance counts",
    )

    text = text.replace('0.45 <= speed <= 2.80', '0.08 <= speed <= 3.50')
    text = text.replace('R_sun/video', 'R_sun/window')

    text = replace_once(
        text,
        '            ("last_processed_frame", metadata.get("last_processed_frame", 0), "frame"),\n',
        '            ("last_processed_frame", metadata.get("last_processed_frame", 0), "frame"),\n'
        '            ("analysis_start_time", metadata.get("analysis_start_s", 0.0), "s"),\n'
        '            ("analysis_end_time_exclusive", metadata.get("analysis_end_s_exclusive", 0.0), "s"),\n'
        '            ("analysis_start_frame", metadata.get("analysis_start_frame", 0), "frame"),\n'
        '            ("analysis_end_frame", metadata.get("analysis_end_frame", 0), "frame"),\n'
        '            ("analysis_frame_count_expected", metadata.get("analysis_frame_count_expected", 0), "frame"),\n',
        "report time-window values",
    )

    text = replace_once(
        text,
        '            ("manual_scientific_values", "NOT USED", ""),\n',
        '            ("manual_scientific_values", "NOT USED", ""),\n'
        '            ("manual_video_window", "USED: 45.000 TO 51.900 S", ""),\n'
        '            ("alternate_view_at_52_s", "REJECTED / NOT USED", ""),\n',
        "manual video-window audit rows",
    )

    source_path.write_text(text, encoding="utf-8")


def run_regression_test(source_path: Path) -> None:
    namespace = runpy.run_path(str(source_path))
    Config = namespace["Config"]
    if namespace["VERSION"] != "V0007":
        raise RuntimeError("Version regression failed.")
    config = Config(Path("video.mp4"), Path("output"))
    if config.analysis_start_s != DEFAULT_ANALYSIS_START_S:
        raise RuntimeError("Default analysis start regression failed.")
    if config.analysis_end_s != DEFAULT_ANALYSIS_END_S:
        raise RuntimeError("Default analysis end regression failed.")
    fps = 29.97002997
    start_frame = int(math.ceil(DEFAULT_ANALYSIS_START_S * fps))
    end_frame = int(math.ceil(DEFAULT_ANALYSIS_END_S * fps)) - 1
    if start_frame != 1349 or end_frame != 1555:
        raise RuntimeError(
            f"Time-to-frame regression failed: start={start_frame}, end={end_frame}."
        )
    print(
        "Regression test: verified full-Sun window configured | "
        f"start={DEFAULT_ANALYSIS_START_S:.3f}s | "
        f"end_exclusive={DEFAULT_ANALYSIS_END_S:.3f}s | "
        f"frames={start_frame}-{end_frame} at {fps:.8f} fps"
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
# V0007
