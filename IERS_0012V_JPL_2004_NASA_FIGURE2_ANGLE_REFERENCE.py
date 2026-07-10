# IERS-0012V
# Audit reference: GitHubDelivery@IERS-0012V; verified IERS-0012U JPL core with unsigned plot angles and a central two-reference angle panel.

import hashlib
import time
import urllib.request

VERSION = "IERS-0012V"
PROGRAM_NAME = "IERS_0012V_JPL_2004_NASA_FIGURE2_ANGLE_REFERENCE.py"
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
                    "User-Agent": "Mozilla/5.0 IERS-0012V",
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
                    "Downloaded scientific core does not begin with "
                    "the required IERS-0012U version marker."
                )
            if not source.rstrip().endswith("# IERS-0012U"):
                raise RuntimeError(
                    "Downloaded scientific core does not end with "
                    "the required IERS-0012U version marker."
                )
            required_tokens = (
                "def query_jpl_ephemerides",
                "def derive_events",
                "def build_track",
                "def build_ecliptic_fit",
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


def build_iers_0012v_source(base_source):
    source = base_source

    source = replace_exactly_once(
        source,
        "# Audit reference: GitHubDelivery@IERS-0012U; preserve IERS-0012U "
        "JPL geometry and separate hourly/contact labels from the transit track.",
        "# Audit reference: GitHubDelivery@IERS-0012V; preserve verified JPL "
        "geometry, show unsigned plot angles, and add a central two-reference "
        "angle panel.",
        "audit reference",
    )
    source = replace_exactly_once(
        source,
        BASE_PROGRAM_NAME,
        PROGRAM_NAME,
        "program name",
    )
    source = source.replace("IERS-0012U", VERSION)

    source = replace_exactly_once(
        source,
        'f"TRACK/HORIZONTAL = {track_angle:+.4f}°\\n"',
        'f"TRACK/HORIZONTAL = {abs(track_angle):.4f}°\\n"',
        "unsigned track-to-horizontal plot value",
    )
    source = replace_exactly_once(
        source,
        'f"ECLIPTIC/HORIZONTAL = {ecliptic_angle:+.4f}°\\n"',
        'f"ECLIPTIC/HORIZONTAL = {abs(ecliptic_angle):.4f}°\\n"',
        "unsigned ecliptic-to-horizontal plot value",
    )

    plot_anchor = "    axis.set_xlim(-limit, limit)"
    angle_panel = '''    angle_reference_text = (
        "ANGLE REFERENCES\\n"
        f"TRACK TO HORIZONTAL  {abs(track_angle):.4f}°\\n"
        f"VENUS TRACK FROM ECLIPTIC  {relative_angle:.4f}°"
    )
    axis.text(
        0.0,
        315.0,
        angle_reference_text,
        fontsize=8.0,
        color="#e8f7ff",
        ha="center",
        va="center",
        linespacing=1.50,
        fontweight="bold",
        zorder=15,
        bbox={
            "boxstyle": "round,pad=0.40",
            "facecolor": "#03080d",
            "edgecolor": "#24414f",
            "linewidth": 0.40,
            "alpha": 0.96,
        },
    )

'''
    source = replace_exactly_once(
        source,
        plot_anchor,
        angle_panel + plot_anchor,
        "central angle-reference panel",
    )

    no_ai_line = '    print("No AI image generation is used.")'
    expanded_comments = '''    print("Plot angle labels use unsigned magnitudes for immediate visual interpretation.")
    print("A central panel reports track-to-horizontal and Venus-track-to-ecliptic angles.")
    print("No AI image generation is used.")'''
    source = replace_exactly_once(
        source,
        no_ai_line,
        expanded_comments,
        "plot revision comments",
    )

    required_revisions = (
        "TRACK TO HORIZONTAL",
        "VENUS TRACK FROM ECLIPTIC",
        "abs(track_angle):.4f",
        "abs(ecliptic_angle):.4f",
        "angle_reference_text",
    )
    missing = [token for token in required_revisions if token not in source]
    if missing:
        raise RuntimeError(f"IERS-0012V revision audit failed; missing {missing}.")
    if source.count("ANGLE REFERENCES") != 1:
        raise RuntimeError("IERS-0012V must contain exactly one central angle panel.")
    if "TODO" in source:
        raise RuntimeError("IERS-0012V audit rejected a TODO statement.")
    if "placeholder" in source.lower():
        raise RuntimeError("IERS-0012V audit rejected placeholder content.")
    if "\n    pass\n" in source:
        raise RuntimeError("IERS-0012V audit rejected an incomplete pass statement.")
    if not source.startswith("# IERS-0012V\n"):
        raise RuntimeError("IERS-0012V first-line version audit failed.")
    if not source.rstrip().endswith("# IERS-0012V"):
        raise RuntimeError("IERS-0012V last-line version audit failed.")
    return source


def main():
    base_source, base_sha256 = fetch_verified_base_source()
    expanded_source = build_iers_0012v_source(base_source)
    compiled = compile(
        expanded_source,
        PROGRAM_NAME,
        "exec",
        dont_inherit=True,
        optimize=0,
    )

    print(f"[SUCCESS] Verified scientific core fetched from GitHub: {BASE_PROGRAM_NAME}")
    print(f"[AUDIT] Base SHA-256: {base_sha256}")
    print("[AUDIT] IERS-0012V expanded source compile check: PASS")
    print("[AUDIT] JPL calculations unchanged; plot presentation patch only")
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
# IERS-0012V
