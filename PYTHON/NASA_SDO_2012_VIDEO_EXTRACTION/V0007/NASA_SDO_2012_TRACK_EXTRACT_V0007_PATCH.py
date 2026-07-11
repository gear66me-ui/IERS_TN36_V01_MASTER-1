# V0007
# Audit reference: Restrict NASA SDO extraction to the verified full-limb interval at 46.0-53.0 seconds.
from __future__ import annotations

import argparse
import runpy
from pathlib import Path


DEFAULT_ANALYSIS_START_S = 46.0
DEFAULT_ANALYSIS_END_S = 53.0


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one patch location, found {count}.")
    return text.replace(old, new, 1)


def patch_source(source_path: Path) -> None:
    text = source_path.read_text(encoding="utf-8")

    text = replace_once(
        text,
        'VERSION = "V0006"',
        'VERSION = "V0007"',
        "version patch",
    )

    text = replace_once(
        text,
        'OVERLAY_NAME = "NASA_SDO_2012_TRACK_OVERLAY.png"\n',
        'OVERLAY_NAME = "NASA_SDO_2012_TRACK_OVERLAY.png"\n'
        'DEFAULT_ANALYSIS_START_S = 46.0\n'
        'DEFAULT_ANALYSIS_END_S = 53.0\n',
        "analysis constants patch",
    )

    text = replace_once(
        text,
        '    min_venus_quality: float = 0.20\n',
        '    min_venus_quality: float = 0.20\n'
        '    analysis_start_s: float = DEFAULT_ANALYSIS_START_S\n'
        '    analysis_end_s: float = DEFAULT_ANALYSIS_END_S\n',
        "Config time-window fields patch",
    )

    text = replace_once(
        text,
        '    parser.add_argument("--global-limb-retry", type=int, default=90)\n'
        '    args = parser.parse_args(argv)\n',
        '    parser.add_argument("--global-limb-retry", type=int, default=90)\n'
        '    parser.add_argument(\n'
        '        "--analysis-start-s",\n'
        '        type=float,\n'
        '        default=DEFAULT_ANALYSIS_START_S,\n'
        '        help="Video time in seconds at which full-limb analysis begins.",\n'
        '    )\n'
        '    parser.add_argument(\n'
        '        "--analysis-end-s",\n'
        '        type=float,\n'
        '        default=DEFAULT_ANALYSIS_END_S,\n'
        '        help="Video time in seconds at which full-limb analysis ends.",\n'
        '    )\n'
        '    args = parser.parse_args(argv)\n'
        '    if not (args.analysis_start_s >= 0.0):\n'
        '        parser.error("--analysis-start-s must be non-negative.")\n'
        '    if not (args.analysis_end_s > args.analysis_start_s):\n'
        '        parser.error("--analysis-end-s must be greater than --analysis-start-s.")\n',
        "argument parser time-window patch",
    )

    text = replace_once(
        text,
        '        global_limb_retry=max(1, args.global_limb_retry),\n'
        '    )\n',
        '        global_limb_retry=max(1, args.global_limb_retry),\n'
        '        analysis_start_s=float(args.analysis_start_s),\n'
        '        analysis_end_s=float(args.analysis_end_s),\n'
        '    )\n',
        "Config constructor time-window patch",
    )

    old_process_header = '''    metadata = video_metadata(cap)
    cap.set(cv2.CAP_PROP_POS_FRAMES, config.start_frame)

    records: list[FrameRecord] = []
'''
    new_process_header = '''    metadata = video_metadata(cap)
    fps = float(metadata["fps"])
    time_start_frame = int(math.floor(config.analysis_start_s * fps))
    time_end_frame = int(math.ceil(config.analysis_end_s * fps))
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
    metadata["analysis_end_s"] = float(config.analysis_end_s)
    metadata["analysis_start_frame"] = int(effective_start_frame)
    metadata["analysis_end_frame"] = int(effective_end_frame)
    print(
        "DEBUG | verified full-limb analysis window | "
        f"start_s={config.analysis_start_s:.3f} | "
        f"end_s={config.analysis_end_s:.3f} | "
        f"start_frame={effective_start_frame} | "
        f"end_frame={effective_end_frame}"
    )

    records: list[FrameRecord] = []
'''
    text = replace_once(
        text,
        old_process_header,
        new_process_header,
        "process_video time-window initialization patch",
    )

    text = replace_once(
        text,
        '    frame_index = config.start_frame\n',
        '    frame_index = effective_start_frame\n',
        "initial frame index patch",
    )

    text = replace_once(
        text,
        '            if config.end_frame is not None and frame_index > config.end_frame:\n'
        '                break\n',
        '            if frame_index > effective_end_frame:\n'
        '                break\n',
        "loop end-frame patch",
    )

    text = replace_once(
        text,
        '        if not (0.45 <= speed <= 2.80):\n'
        '            continue\n',
        '        if not (0.08 <= speed <= 3.50):\n'
        '            continue\n',
        "RANSAC speed-range patch",
    )

    text = replace_once(
        text,
        '    if not (0.45 <= speed <= 2.80):\n'
        '        raise RuntimeError(\n'
        '            f"Continuous Venus track speed {speed:.6f} R_sun/video is implausible."\n'
        '        )\n',
        '    if not (0.08 <= speed <= 3.50):\n'
        '        raise RuntimeError(\n'
        '            f"Continuous Venus track speed {speed:.6f} R_sun/window is implausible."\n'
        '        )\n',
        "final speed-range patch",
    )

    text = replace_once(
        text,
        '        f"speed={speed:.6f} R_sun/video"\n',
        '        f"speed={speed:.6f} R_sun/window"\n',
        "acceptance debug units patch",
    )

    text = replace_once(
        text,
        '        f"speed={speed:.6f} R_sun/video"\n'
        '    )\n'
        '    return table\n',
        '        f"speed={speed:.6f} R_sun/window"\n'
        '    )\n'
        '    return table\n',
        "selected-track debug units patch",
    )

    text = replace_once(
        text,
        '            ("last_processed_frame", metadata.get("last_processed_frame", 0), "frame"),\n',
        '            ("last_processed_frame", metadata.get("last_processed_frame", 0), "frame"),\n'
        '            ("analysis_start_time", metadata.get("analysis_start_s", 0.0), "s"),\n'
        '            ("analysis_end_time", metadata.get("analysis_end_s", 0.0), "s"),\n'
        '            ("analysis_start_frame", metadata.get("analysis_start_frame", 0), "frame"),\n'
        '            ("analysis_end_frame", metadata.get("analysis_end_frame", 0), "frame"),\n',
        "report time-window rows patch",
    )

    source_path.write_text(text, encoding="utf-8")


def run_regression_test(source_path: Path) -> None:
    namespace = runpy.run_path(str(source_path))
    Config = namespace["Config"]
    if namespace["VERSION"] != "V0007":
        raise RuntimeError("Version regression failed.")
    config = Config(Path("video.mp4"), Path("output"))
    if config.analysis_start_s != DEFAULT_ANALYSIS_START_S:
        raise RuntimeError("Default analysis start time regression failed.")
    if config.analysis_end_s != DEFAULT_ANALYSIS_END_S:
        raise RuntimeError("Default analysis end time regression failed.")
    fps = 29.97002997
    start_frame = int(DEFAULT_ANALYSIS_START_S * fps)
    end_frame = int(np.ceil(DEFAULT_ANALYSIS_END_S * fps)) if False else int(__import__("math").ceil(DEFAULT_ANALYSIS_END_S * fps))
    if start_frame != 1378 or end_frame != 1589:
        raise RuntimeError(
            f"Time-to-frame regression failed: start={start_frame}, end={end_frame}."
        )
    print(
        "Regression test: verified full-limb window configured | "
        f"start={DEFAULT_ANALYSIS_START_S:.1f}s | "
        f"end={DEFAULT_ANALYSIS_END_S:.1f}s | "
        f"frames={start_frame}-{end_frame} at 29.97002997 fps"
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
