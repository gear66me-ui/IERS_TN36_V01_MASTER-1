# V0123B
# Audit reference: create the Google Drive GitHub backup tree and synchronize current IMCCE project files.

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

VERSION = "V0123B"
LOCAL_TZ = timezone(timedelta(hours=-5))

DRIVE_PROJECT_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/JPL - 1769 VENUS TRANSIT"
)
BACKUP_ROOT = DRIVE_PROJECT_ROOT / "GitHub"
PROJECT_OUTPUT_ROOT = Path(
    "/content/IERS_TN36_V01_MASTER_OUTPUT/IMCCE_VENUS_CANON"
)

SUBFOLDERS = {
    "python": BACKUP_ROOT / "PYTHON",
    "csv": BACKUP_ROOT / "DATA" / "CSV",
    "xlsx": BACKUP_ROOT / "DATA" / "XLSX",
    "json": BACKUP_ROOT / "DATA" / "JSON",
    "source": BACKUP_ROOT / "DATA" / "SOURCE",
    "stage": BACKUP_ROOT / "DATA" / "STAGE",
    "audit": BACKUP_ROOT / "AUDIT",
}

PYTHON_FILES = [
    Path(f"/content/IERS_0122{letter}_IMCCE_TO_CSV.py")
    for letter in "ABCDE"
] + [Path("/content/IERS_0123B_GITHUB_DRIVE_BACKUP.py")]

CONFIG_JSON = BACKUP_ROOT / "GITHUB_BACKUP_CONFIG.json"
README_TXT = BACKUP_ROOT / "README_BACKUP.txt"
AUDIT_CSV = SUBFOLDERS["audit"] / "V0123B_BACKUP_AUDIT.csv"
AUDIT_JSON = SUBFOLDERS["audit"] / "V0123B_BACKUP_AUDIT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_backup_tree() -> list[Path]:
    if not DRIVE_PROJECT_ROOT.exists():
        raise FileNotFoundError(f"Drive project folder not found: {DRIVE_PROJECT_ROOT}")
    created: list[Path] = []
    for folder in [BACKUP_ROOT, *SUBFOLDERS.values()]:
        existed = folder.exists()
        folder.mkdir(parents=True, exist_ok=True)
        if not existed:
            created.append(folder)
    return created


def build_copy_plan() -> list[tuple[Path, Path, str]]:
    plan: list[tuple[Path, Path, str]] = []

    for source in PYTHON_FILES:
        plan.append((source, SUBFOLDERS["python"] / source.name, "PYTHON"))

    fixed_outputs = [
        (
            PROJECT_OUTPUT_ROOT / "IMCCE_VENUS_TRANSIT_CANON_MASTER.csv",
            SUBFOLDERS["csv"] / "IMCCE_VENUS_TRANSIT_CANON_MASTER.csv",
            "CSV",
        ),
        (
            PROJECT_OUTPUT_ROOT / "IMCCE_VENUS_TRANSIT_CANON_MASTER.xlsx",
            SUBFOLDERS["xlsx"] / "IMCCE_VENUS_TRANSIT_CANON_MASTER.xlsx",
            "XLSX",
        ),
        (
            PROJECT_OUTPUT_ROOT / "IERS_0122E_FINAL_DELIVERY_AUDIT.json",
            SUBFOLDERS["audit"] / "IERS_0122E_FINAL_DELIVERY_AUDIT.json",
            "AUDIT",
        ),
    ]
    plan.extend(fixed_outputs)

    recursive_sources = [
        (PROJECT_OUTPUT_ROOT / "SOURCE", SUBFOLDERS["source"], "SOURCE"),
        (PROJECT_OUTPUT_ROOT / "STAGE", SUBFOLDERS["stage"], "STAGE"),
    ]
    for source_root, destination_root, category in recursive_sources:
        if source_root.exists():
            for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
                relative = source.relative_to(source_root)
                plan.append((source, destination_root / relative, category))

    for source in sorted(PROJECT_OUTPUT_ROOT.glob("*.json")):
        if source.name != "IERS_0122E_FINAL_DELIVERY_AUDIT.json":
            plan.append((source, SUBFOLDERS["json"] / source.name, "JSON"))

    unique: dict[str, tuple[Path, Path, str]] = {}
    for source, destination, category in plan:
        unique[str(destination)] = (source, destination, category)
    return list(unique.values())


def synchronize_file(source: Path, destination: Path, category: str) -> dict[str, object]:
    record: dict[str, object] = {
        "category": category,
        "source": str(source),
        "destination": str(destination),
        "status": "NOT FOUND",
        "bytes": 0,
        "sha256": "",
    }
    if not source.exists():
        return record

    source_hash = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and sha256(destination) == source_hash:
        status = "UNCHANGED"
    else:
        shutil.copy2(source, destination)
        status = "COPIED"

    destination_hash = sha256(destination)
    if destination_hash != source_hash:
        raise RuntimeError(f"Hash verification failed: {source} -> {destination}")

    record.update(
        status=status,
        bytes=destination.stat().st_size,
        sha256=destination_hash,
    )
    return record


def write_configuration() -> None:
    config = {
        "version": VERSION,
        "drive_project_root": str(DRIVE_PROJECT_ROOT),
        "backup_root": str(BACKUP_ROOT),
        "project_output_root": str(PROJECT_OUTPUT_ROOT),
        "subfolders": {name: str(path) for name, path in SUBFOLDERS.items()},
        "policy": (
            "Future JPL 1769 project scripts should save or synchronize Python, CSV, "
            "XLSX, JSON, source, stage, and audit files into this backup tree."
        ),
    }
    CONFIG_JSON.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    readme = (
        "JPL - 1769 VENUS TRANSIT / GitHub BACKUP\n\n"
        "PYTHON contains project scripts downloaded from GitHub.\n"
        "DATA/CSV contains final CSV deliverables.\n"
        "DATA/XLSX contains final Excel workbooks.\n"
        "DATA/JSON contains project JSON metadata.\n"
        "DATA/SOURCE contains preserved source downloads and manifests.\n"
        "DATA/STAGE contains intermediate parsed data.\n"
        "AUDIT contains backup and delivery audit records.\n\n"
        "Files are copied only when new or changed and are verified by SHA-256.\n"
    )
    README_TXT.write_text(readme, encoding="utf-8")


def write_audits(records: list[dict[str, object]], created: list[Path]) -> None:
    fieldnames = ["category", "source", "destination", "status", "bytes", "sha256"]
    with AUDIT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    payload = {
        "version": VERSION,
        "timestamp_local": datetime.now(LOCAL_TZ).isoformat(timespec="seconds"),
        "drive_project_root": str(DRIVE_PROJECT_ROOT),
        "backup_root": str(BACKUP_ROOT),
        "created_folders": [str(path) for path in created],
        "records": records,
        "summary": {
            "copied": sum(record["status"] == "COPIED" for record in records),
            "unchanged": sum(record["status"] == "UNCHANGED" for record in records),
            "not_found": sum(record["status"] == "NOT FOUND" for record in records),
        },
    }
    AUDIT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    created = create_backup_tree()
    write_configuration()
    records = [
        synchronize_file(source, destination, category)
        for source, destination, category in build_copy_plan()
    ]
    write_audits(records, created)

    copied = sum(record["status"] == "COPIED" for record in records)
    unchanged = sum(record["status"] == "UNCHANGED" for record in records)
    missing = sum(record["status"] == "NOT FOUND" for record in records)

    print(f"CODE OUTPUT: {VERSION}")
    print("CODE INPUTS")
    print(f"Drive project root : {DRIVE_PROJECT_ROOT}")
    print(f"Backup root : {BACKUP_ROOT}")
    print("COMMENTS")
    print("Creates the GitHub backup tree, copies new or changed project files, and verifies SHA-256 hashes.")
    print("RESULTS")
    print(f"Folders created : {len(created)} | Files copied : {copied} | Unchanged : {unchanged} | NOT FOUND : {missing}")
    print("OUTPUT SUMMARY")
    print(f"Configuration : {CONFIG_JSON}")
    print(f"Backup audit CSV : {AUDIT_CSV}")
    print(f"Backup audit JSON : {AUDIT_JSON}")
    print("PAPER COMPARISON")
    print("NOT USED — Google Drive backup stage.")
    print("EQUATION STATUS")
    print("VERIFIED — every copied file passed SHA-256 source/destination comparison.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print("# V0123B")


if __name__ == "__main__":
    main()

# V0123B
