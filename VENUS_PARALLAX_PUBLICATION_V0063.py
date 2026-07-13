# V0063
# Audit reference: Generate V0059 then V0061 before executing the unchanged-format V0062 Venus publication figure.
from __future__ import annotations

import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0063"
TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
REPOSITORY_RAW = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1"
)

STAGES = (
    {
        "version": "V0059",
        "commit": "c20ca305b9e25e3ad46c7bc7556316c8ae3083e3",
        "filename": "VENUS_1769_VECTOR_TO_PROJECTED_GOLD_STANDARD_V0059.py",
        "output_dir": "VENUS_1769_VECTOR_TO_PROJECTED_GOLD_STANDARD_V0059_OUTPUT",
    },
    {
        "version": "V0061",
        "commit": "b0a2df3cb9d5ea2f7fa36af79c39bd60e4986f60",
        "filename": "VENUS_1769_SEPARATE_RAY_VECTOR_TRANSFER_GOLD_STANDARD_V0061.py",
        "output_dir": "VENUS_1769_SEPARATE_RAY_VECTOR_TRANSFER_GOLD_STANDARD_V0061_OUTPUT",
    },
    {
        "version": "V0062",
        "commit": "010dafcef86f7eb47d040a1d9f72788e626843aa",
        "filename": "VENUS_PARALLAX_PUBLICATION_V0062.py",
        "output_dir": "VENUS_PARALLAX_PUBLICATION_V0062_OUTPUT",
    },
)


def fetch_text(url: str) -> str:
    request = urllib.request.Request(
        f"{url}?cache={time.time_ns()}",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read().decode("utf-8")


def stage_url(stage: dict[str, str]) -> str:
    return (
        f"{REPOSITORY_RAW}/{stage['commit']}/{stage['filename']}"
    )


def materialize_stage(stage: dict[str, str]) -> Path:
    version = stage["version"]
    path = ROOT / stage["filename"]
    source = fetch_text(stage_url(stage))

    if not source.startswith(f"# {version}\n"):
        raise RuntimeError(
            f"{version} opening-version audit failed for {path}."
        )
    if not source.rstrip().endswith(f"# {version}"):
        raise RuntimeError(
            f"{version} closing-version audit failed for {path}."
        )

    compile(source, str(path), "exec")
    path.write_text(source, encoding="utf-8")
    return path


def run_stage(stage: dict[str, str], path: Path) -> str:
    completed = subprocess.run(
        [sys.executable, str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{stage['version']} execution failed.\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
    return completed.stdout


def clean_target_outputs() -> None:
    for stage in STAGES:
        output_dir = ROOT / stage["output_dir"]
        if output_dir.exists():
            shutil.rmtree(output_dir)

    for pattern in (
        "VENUS_PARALLAX_PUBLICATION_V0062_FULL.py",
        "VENUS_PARALLAX_PUBLICATION_V0062_ISOLATED.py",
    ):
        path = ROOT / pattern
        path.unlink(missing_ok=True)


def newest_publication_png() -> Path:
    candidates = sorted(
        [
            path
            for path in ROOT.rglob("*.png")
            if "V0062" in path.name
            and "VENUS" in path.name.upper()
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(
            "V0062 completed but no Venus publication PNG was found."
        )
    return candidates[0]


def main() -> None:
    clean_target_outputs()

    stage_paths: list[Path] = []
    stage_logs: list[str] = []

    for stage in STAGES:
        path = materialize_stage(stage)
        stage_paths.append(path)
        stage_logs.append(run_stage(stage, path))

    v0059_csv = (
        ROOT
        / "VENUS_1769_VECTOR_TO_PROJECTED_GOLD_STANDARD_V0059_OUTPUT"
        / "VENUS_1769_VECTOR_TO_PROJECTED_GOLD_STANDARD_V0059.csv"
    )
    v0061_csv = (
        ROOT
        / "VENUS_1769_SEPARATE_RAY_VECTOR_TRANSFER_GOLD_STANDARD_V0061_OUTPUT"
        / "VENUS_1769_SEPARATE_RAY_VECTOR_TRANSFER_GOLD_STANDARD_V0061.csv"
    )

    if not v0059_csv.is_file() or v0059_csv.stat().st_size == 0:
        raise RuntimeError("V0059 dependency CSV was not generated.")
    if not v0061_csv.is_file() or v0061_csv.stat().st_size == 0:
        raise RuntimeError("V0061 gold-standard CSV was not generated.")

    output_png = newest_publication_png()

    from IPython.display import Image, display

    display(Image(filename=str(output_png)))

    print("CODE INPUTS")
    print("Pinned V0059 vector projection, pinned V0061 separate-ray transfer, and exact-format V0062/V0027 publication source.")
    print("COMMENTS")
    print("The missing V0059 prerequisite is now generated automatically before V0061. No plot formatting was changed.")
    print("RESULTS")
    print("Transit pair: Point Venus, Tahiti and Vardo, Norway")
    print("OUTPUT SUMMARY")
    print(f"PNG: {output_png}")
    print(f"V0059 CSV: {v0059_csv}")
    print(f"V0061 CSV: {v0061_csv}")
    print(f"PY: {stage_paths[-1]}")
    print("PAPER COMPARISON")
    print("NOT USED. This is the updated JPL Venus publication figure.")
    print("EQUATION STATUS")
    print("Dependency chain V0059 → V0061 → V0062 and unchanged V0027 figure format: PASS")
    print(datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0063