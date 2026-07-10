# IERS-0012AA
# Audit reference: GitHubDelivery@IERS-0012AA; preserve IERS-0012Z outputs and move only the Vardo and Point Venus plot labels.

import hashlib
import time
import urllib.request

VERSION = "IERS-0012AA"
PROGRAM_NAME = "IERS_0012AA_MOVE_VARDO_POINT_VENUS_LABELS.py"
BASE_PROGRAM_NAME = "IERS_0012Z_PRESERVE_0012N_PLOT_AND_TABLE_STYLE.py"
BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    + BASE_PROGRAM_NAME
)
DOWNLOAD_ATTEMPTS = 4
DOWNLOAD_TIMEOUT_SECONDS = 45


def fetch_verified_source():
    last_error = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(
                BASE_URL,
                headers={
                    "User-Agent": "Mozilla/5.0 IERS-0012AA",
                    "Accept": "text/plain",
                },
            )
            with urllib.request.urlopen(
                request,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            ) as response:
                source_bytes = response.read()
            source = source_bytes.decode("utf-8")
            if not source.startswith("# IERS-0012Z\n"):
                raise RuntimeError("Base source opening version marker is invalid.")
            if not source.rstrip().endswith("# IERS-0012Z"):
                raise RuntimeError("Base source closing version marker is invalid.")
            required_tokens = (
                "def plot_engineering_track",
                "def save_widget_style_table_png",
                "Vardo CA:",
                "Point Venus CA:",
                "f\"{track['site']['short']} CA\"",
            )
            missing = [token for token in required_tokens if token not in source]
            if missing:
                raise RuntimeError(f"Base source audit failed; missing {missing}.")
            return source, hashlib.sha256(source_bytes).hexdigest()
        except Exception as exc:
            last_error = exc
            if attempt < DOWNLOAD_ATTEMPTS:
                time.sleep(2.0 * attempt)
    raise RuntimeError(
        "Unable to download the verified IERS-0012Z base source after "
        f"{DOWNLOAD_ATTEMPTS} attempts: {last_error}"
    )


def replace_exactly_once(source, old, new, description):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Patch audit failed for {description}: expected 1 occurrence, "
            f"found {count}."
        )
    return source.replace(old, new, 1)


def build_iers_0012aa_source(base_source):
    source = base_source

    source = replace_exactly_once(
        source,
        "# Audit reference: GitHubDelivery@IERS-0012Z; preserve the "
        "IERS-0012N engineering plot and exact widget table styling while "
        "exporting both tables as PNG.",
        "# Audit reference: GitHubDelivery@IERS-0012AA; preserve IERS-0012Z "
        "geometry, plot, and table styling while moving only the Vardo and "
        "Point Venus labels.",
        "audit reference",
    )
    source = replace_exactly_once(
        source,
        BASE_PROGRAM_NAME,
        PROGRAM_NAME,
        "program name",
    )
    source = source.replace("IERS-0012Z", VERSION)

    old_label_block = '''        closest = track["event_pts"]["CA"]
        dy = 15.0 if site_label.startswith("Vardo") else -15.0
        add_label(
            axis,
            closest,
            f"{track['site']['short']} CA",
            18.0,
            dy,
            color,
        )
'''
    new_label_block = '''        closest = track["event_pts"]["CA"]
        if site_label.startswith("Vardo"):
            label_text = "Vardo, Norway"
            label_dx = 18.0
            label_dy = -24.0
        else:
            label_text = "Point Venus, Tahiti"
            label_dx = 18.0
            label_dy = 24.0
        add_label(
            axis,
            closest,
            label_text,
            label_dx,
            label_dy,
            color,
        )
'''
    source = replace_exactly_once(
        source,
        old_label_block,
        new_label_block,
        "observer closest-approach labels",
    )

    old_note = '''        f"Vardo CA: {track_a['closest_utc']}   "
        f"Point Venus CA: {track_b['closest_utc']}"
'''
    new_note = '''        f"Vardo, Norway closest: {track_a['closest_utc']}   "
        f"Point Venus, Tahiti closest: {track_b['closest_utc']}"
'''
    source = replace_exactly_once(
        source,
        old_note,
        new_note,
        "bottom closest-approach note",
    )

    old_comment = (
        '    print("No AI image generation is used.")'
    )
    new_comment = '''    print("Point Venus, Tahiti is labeled above the green track.")
    print("Vardo, Norway is labeled below the yellow track.")
    print("No AI image generation is used.")'''
    source = replace_exactly_once(
        source,
        old_comment,
        new_comment,
        "label-layout comments",
    )

    required_revisions = (
        'label_text = "Vardo, Norway"',
        'label_dy = -24.0',
        'label_text = "Point Venus, Tahiti"',
        'label_dy = 24.0',
        "Vardo, Norway closest:",
        "Point Venus, Tahiti closest:",
    )
    missing = [token for token in required_revisions if token not in source]
    if missing:
        raise RuntimeError(f"IERS-0012AA revision audit failed; missing {missing}.")
    if "f\"{track['site']['short']} CA\"" in source:
        raise RuntimeError("Rejected old abbreviated CA plot label remains.")
    if "TODO" in source:
        raise RuntimeError("Rejected TODO statement found.")
    if "placeholder" in source.lower():
        raise RuntimeError("Rejected placeholder content found.")
    if "\n    pass\n" in source:
        raise RuntimeError("Rejected incomplete pass statement found.")
    if not source.startswith("# IERS-0012AA\n"):
        raise RuntimeError("Opening version marker audit failed.")
    if not source.rstrip().endswith("# IERS-0012AA"):
        raise RuntimeError("Closing version marker audit failed.")
    return source


def main():
    base_source, base_sha256 = fetch_verified_source()
    expanded_source = build_iers_0012aa_source(base_source)
    compiled = compile(
        expanded_source,
        PROGRAM_NAME,
        "exec",
        dont_inherit=True,
        optimize=0,
    )

    print(f"[SUCCESS] Verified base fetched from GitHub: {BASE_PROGRAM_NAME}")
    print(f"[AUDIT] Base SHA-256: {base_sha256}")
    print("[AUDIT] IERS-0012AA expanded source compile check: PASS")
    print("[AUDIT] Geometry, plot, and table styling unchanged; labels only")
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
# IERS-0012AA
