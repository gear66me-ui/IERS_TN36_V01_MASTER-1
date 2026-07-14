# V0093
# Audit reference: line-by-line derivation-method audit of V0067 versus V0089; no AI images; Python text/CSV report only.
from __future__ import annotations

import csv
import difflib
import hashlib
import re
import textwrap
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "V0093"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUT = ROOT / "VENUS_1769_V0067_V0089_LINE_AUDIT_V0093_OUTPUT"
OUT.mkdir(parents=True, exist_ok=True)

REPO_RAW = "https://raw.githubusercontent.com/gear66me-ui/IERS_TN36_V01_MASTER-1/main"
FILES = {
    "V0067": f"{REPO_RAW}/VENUS_1769_V0027_FORMAT_STANDALONE_V0067.py",
    "V0089": f"{REPO_RAW}/VENUS_1769_V0027_FORMAT_STANDALONE_V0089_STANDALONE.py",
}

CSV_LINE_AUDIT = OUT / "V0093_LINE_BY_LINE_V0067_V0089_AUDIT.csv"
TXT_LINE_AUDIT = OUT / "V0093_LINE_BY_LINE_V0067_V0089_AUDIT.txt"
MD_REPORT = OUT / "V0093_EXECUTIVE_DERIVATION_AUDIT.md"

CRITICAL_PATTERNS = {
    "constants": ["ARC", "ARCSEC_PER_RAD", "JPL_AU", "IAU1976", "EARTH_RADIUS", "SUN_RADIUS", "VENUS_RADIUS"],
    "jpl_download": ["Horizons", "vectors", "refplane", "aberrations", "cache", "JPL"],
    "contact_roots": ["brentq", "contact", "residual", "C1", "C2", "C3", "C4", "internal", "external"],
    "closest_approach": ["closest", "minimize_scalar", "xatol", "reference_jd", "lower_seconds", "upper_seconds", "jd"],
    "basis_projection": ["basis", "ecliptic", "gnom", "gnomonic", "east", "north", "center"],
    "apbp_geometry": ["A_prime", "B_prime", "apbp", "aprime", "A′B′", "normal", "velocity", "track_direction"],
    "ab_geometry": ["baseline", "AB", "ab_km", "ab_as", "km_per", "ES", "EV", "VS"],
    "plot_only": ["plot", "Circle", "label", "annotate", "table", "color", "figure", "axis", "mirror", "p[0]"],
}

EQUATION_LINES = {
    "V0067": {
        "local_ca": "seconds-space minimizer around midpoint of bracketing contact/minimum interval; reference path",
        "geocentric_ca": "seconds-space minimizer with reference_jd, lower_seconds, upper_seconds, xatol=1e-4 sec",
        "common_relative_position": "gnomonic Venus-Sun tangent-plane separation in arcsec",
        "separate_ray_geometry": "uses q_pv/q_v and velocity-derived common normal at geocentric closest approach",
    },
    "V0089": {
        "local_ca": "seconds-space local CA for station contacts retained",
        "geo": "direct JD-space geocentric minimizer with different bracket and default tolerance; regression source",
        "rel_common": "gnomonic Venus-Sun tangent-plane separation in arcsec",
        "geo_apbp": "compact reimplementation of separate-ray geometry; not line-identical to V0067",
    },
}


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": VERSION})
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.read().decode("utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def line_records(a_lines: list[str], b_lines: list[str]) -> list[dict[str, object]]:
    matcher = difflib.SequenceMatcher(a=a_lines, b=b_lines, autojunk=False)
    records: list[dict[str, object]] = []
    serial = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for off in range(i2 - i1):
                serial += 1
                records.append({
                    "serial": serial,
                    "status": "UNCHANGED",
                    "v0067_line": i1 + off + 1,
                    "v0089_line": j1 + off + 1,
                    "v0067_text": a_lines[i1 + off],
                    "v0089_text": b_lines[j1 + off],
                    "audit_category": classify_line(a_lines[i1 + off] + " " + b_lines[j1 + off]),
                })
        elif tag == "replace":
            span = max(i2 - i1, j2 - j1)
            for off in range(span):
                serial += 1
                left_exists = i1 + off < i2
                right_exists = j1 + off < j2
                left = a_lines[i1 + off] if left_exists else ""
                right = b_lines[j1 + off] if right_exists else ""
                records.append({
                    "serial": serial,
                    "status": "REPLACED",
                    "v0067_line": i1 + off + 1 if left_exists else "",
                    "v0089_line": j1 + off + 1 if right_exists else "",
                    "v0067_text": left,
                    "v0089_text": right,
                    "audit_category": classify_line(left + " " + right),
                })
        elif tag == "delete":
            for idx in range(i1, i2):
                serial += 1
                records.append({
                    "serial": serial,
                    "status": "DELETED_FROM_V0089",
                    "v0067_line": idx + 1,
                    "v0089_line": "",
                    "v0067_text": a_lines[idx],
                    "v0089_text": "",
                    "audit_category": classify_line(a_lines[idx]),
                })
        elif tag == "insert":
            for idx in range(j1, j2):
                serial += 1
                records.append({
                    "serial": serial,
                    "status": "ADDED_IN_V0089",
                    "v0067_line": "",
                    "v0089_line": idx + 1,
                    "v0067_text": "",
                    "v0089_text": b_lines[idx],
                    "audit_category": classify_line(b_lines[idx]),
                })
    return records


def classify_line(text: str) -> str:
    low = text.lower()
    hits = []
    for category, patterns in CRITICAL_PATTERNS.items():
        if any(p.lower() in low for p in patterns):
            hits.append(category)
    return ";".join(hits) if hits else "general"


def write_csv(records: list[dict[str, object]]) -> None:
    fields = ["serial", "status", "v0067_line", "v0089_line", "audit_category", "v0067_text", "v0089_text"]
    with CSV_LINE_AUDIT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row[key] for key in fields})


def write_text(records: list[dict[str, object]]) -> None:
    with TXT_LINE_AUDIT.open("w", encoding="utf-8") as handle:
        handle.write("V0093 LINE-BY-LINE AUDIT: V0067 vs V0089\n")
        handle.write("Each record is one aligned diff unit. UNCHANGED means exact line match after SequenceMatcher alignment.\n\n")
        for row in records:
            handle.write(
                f"#{row['serial']:05d} | {row['status']} | category={row['audit_category']} | "
                f"V0067:{row['v0067_line']} | V0089:{row['v0089_line']}\n"
            )
            handle.write(f"  V0067: {row['v0067_text']}\n")
            handle.write(f"  V0089: {row['v0089_text']}\n\n")


def function_spans(lines: list[str]) -> list[tuple[str, int, int]]:
    starts: list[tuple[str, int]] = []
    for idx, line in enumerate(lines, start=1):
        m = re.match(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
        if m:
            starts.append((m.group(1), idx))
    spans: list[tuple[str, int, int]] = []
    for n, (name, start) in enumerate(starts):
        end = starts[n + 1][1] - 1 if n + 1 < len(starts) else len(lines)
        spans.append((name, start, end))
    return spans


def find_lines(lines: list[str], patterns: list[str]) -> list[int]:
    out = []
    low_patterns = [p.lower() for p in patterns]
    for idx, line in enumerate(lines, start=1):
        low = line.lower()
        if any(p in low for p in low_patterns):
            out.append(idx)
    return out


def write_report(v67: str, v89: str, records: list[dict[str, object]]) -> None:
    a_lines = v67.splitlines()
    b_lines = v89.splitlines()
    status_counts = {}
    category_counts = {}
    for row in records:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        for cat in str(row["audit_category"]).split(";"):
            category_counts[cat] = category_counts.get(cat, 0) + 1

    critical_lines = {
        name: {
            "V0067": find_lines(a_lines, pats),
            "V0089": find_lines(b_lines, pats),
        }
        for name, pats in CRITICAL_PATTERNS.items()
    }

    with MD_REPORT.open("w", encoding="utf-8") as handle:
        handle.write("# V0093 Executive Derivation Audit — V0067 vs V0089\n\n")
        handle.write("## Scope\n\n")
        handle.write("This report compares the actual GitHub source files line by line. It does not extract numbers from prior notebook output.\n\n")
        handle.write(f"- V0067 URL: `{FILES['V0067']}`\n")
        handle.write(f"- V0089 URL: `{FILES['V0089']}`\n")
        handle.write(f"- V0067 lines: {len(a_lines)}\n")
        handle.write(f"- V0089 lines: {len(b_lines)}\n")
        handle.write(f"- V0067 SHA256: `{sha256_text(v67)}`\n")
        handle.write(f"- V0089 SHA256: `{sha256_text(v89)}`\n\n")
        handle.write("## Diff counts\n\n")
        for key in sorted(status_counts):
            handle.write(f"- {key}: {status_counts[key]}\n")
        handle.write("\n## Category counts across aligned diff records\n\n")
        for key in sorted(category_counts):
            handle.write(f"- {key}: {category_counts[key]}\n")
        handle.write("\n## Critical conclusion\n\n")
        handle.write(textwrap.dedent("""
        V0089 is not a literal standalone expansion of V0067/V0080. It is a compact rewrite.
        The important numerical regression is in the separate-ray geometry path, specifically the geocentric closest-approach epoch used to form the ecliptic basis, tangent velocity direction, common normal, A′B′, km-per-arcsec, and AB projection.

        V0067 reference method:
        - seconds-space geocentric minimization
        - explicit reference_jd, lower_seconds, upper_seconds
        - xatol = 1.0e-4 seconds
        - common_relative_position uses gnomonic Venus-Sun tangent-plane position at the solved geocentric CA
        - separate_ray_geometry computes velocities at jd ± 0.5 seconds and derives the common normal

        V0089 regression method:
        - direct JD-space geocentric minimize_scalar
        - different ±2 sample bracket
        - default bounded-method tolerance
        - compact geo() rewrite combines the same conceptual quantities but is not line-identical
        - this shifts geocentric CA by about -4.900 seconds in the V0091 numeric audit and changes A′B′ by +0.002258107279 arcsec
        """).strip() + "\n\n")
        handle.write("## Function spans\n\n")
        handle.write("### V0067 functions\n\n")
        for name, start, end in function_spans(a_lines):
            handle.write(f"- `{name}`: lines {start}-{end}\n")
        handle.write("\n### V0089 functions\n\n")
        for name, start, end in function_spans(b_lines):
            handle.write(f"- `{name}`: lines {start}-{end}\n")
        handle.write("\n## Critical line index by category\n\n")
        for category, pair in critical_lines.items():
            handle.write(f"### {category}\n\n")
            for version, lines in pair.items():
                preview = ", ".join(str(x) for x in lines[:80])
                if len(lines) > 80:
                    preview += f", ... ({len(lines)} total)"
                handle.write(f"- {version}: {preview if preview else 'none'}\n")
            handle.write("\n")
        handle.write("## Equation-method audit summary\n\n")
        for version, equations in EQUATION_LINES.items():
            handle.write(f"### {version}\n\n")
            for key, desc in equations.items():
                handle.write(f"- **{key}**: {desc}\n")
            handle.write("\n")
        handle.write("## Full line-by-line audit files\n\n")
        handle.write(f"- CSV: `{CSV_LINE_AUDIT}`\n")
        handle.write(f"- TXT: `{TXT_LINE_AUDIT}`\n")


def print_console_summary(v67: str, v89: str, records: list[dict[str, object]]) -> None:
    a_lines = v67.splitlines()
    b_lines = v89.splitlines()
    status_counts = {}
    for row in records:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    print("CODE INPUTS")
    print(f"V0067: {FILES['V0067']}")
    print(f"V0089: {FILES['V0089']}")
    print("COMMENTS")
    print("Full line-by-line audit written to CSV and TXT. Console prints only the executive derivation audit to keep Colab readable.")
    print("RESULTS")
    print(f"V0067 line count: {len(a_lines)}")
    print(f"V0089 line count: {len(b_lines)}")
    print(f"V0067 SHA256: {sha256_text(v67)}")
    print(f"V0089 SHA256: {sha256_text(v89)}")
    for key in sorted(status_counts):
        print(f"{key}: {status_counts[key]}")
    print("OUTPUT SUMMARY")
    print(f"Line-by-line CSV: {CSV_LINE_AUDIT}")
    print(f"Line-by-line TXT: {TXT_LINE_AUDIT}")
    print(f"Executive report: {MD_REPORT}")
    print("PAPER COMPARISON")
    print("Not applicable. This is source-method audit only.")
    print("EQUATION STATUS")
    print("FAIL: V0089 is not equation-path identical to V0067 for geocentric CA / A′B′ reduction.")
    print("Reference equation path is V0067 seconds-space separate-ray geometry.")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(VERSION)


def main() -> None:
    v67 = fetch(FILES["V0067"])
    v89 = fetch(FILES["V0089"])
    if "# V0067" not in v67:
        raise RuntimeError("V0067 source did not validate.")
    if "# V0089" not in v89:
        raise RuntimeError("V0089 source did not validate.")
    records = line_records(v67.splitlines(), v89.splitlines())
    write_csv(records)
    write_text(records)
    write_report(v67, v89, records)
    print_console_summary(v67, v89, records)


if __name__ == "__main__":
    main()
# V0093
