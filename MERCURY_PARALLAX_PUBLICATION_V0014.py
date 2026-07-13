# V0014
# Audit reference: Patch verified V0013 JPL source for stronger lines, yellow solar limb, and the A/B/A-prime/B-prime track table.
from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import re
import urllib.request
from pathlib import Path

SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/"
    "27c4cdf9f216478698f1fa90c8a5ce22aeebef71/"
    "MERCURY_PARALLAX_PUBLICATION_V0013.py"
)
OUTPUT_PATH = Path("/content/MERCURY_PARALLAX_PUBLICATION_V0014_FULL.py")


def extract_literal(module_text: str, variable_name: str):
    tree = ast.parse(module_text)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"{variable_name} was not found in the immutable V0013 wrapper.")


request = urllib.request.Request(
    SOURCE_URL,
    headers={
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "User-Agent": "Colab-V0014",
    },
)
wrapper_bytes = urllib.request.urlopen(request, timeout=90).read()
wrapper_text = wrapper_bytes.decode("utf-8")
payload = extract_literal(wrapper_text, "PAYLOAD")
expected_sha256 = extract_literal(wrapper_text, "EXPECTED_SHA256")

source_bytes = gzip.decompress(base64.b64decode(payload))
source_digest = hashlib.sha256(source_bytes).hexdigest()
if source_digest != expected_sha256:
    raise RuntimeError(
        "Immutable V0013 source verification failed: "
        f"expected {expected_sha256}, received {source_digest}"
    )

text = source_bytes.decode("utf-8")
if not text.startswith("# V0013\n") or not text.rstrip().endswith("# V0013"):
    raise RuntimeError("V0013 source boundary audit failed.")

text = text.replace("V0013", "V0014")

text, sun_color_count = re.subn(
    r'(?m)^(\s*SUN_COLOR\s*=\s*)["\'][^"\']+["\']\s*$',
    r'\1"#FACC15"',
    text,
)
text, sun_width_count = re.subn(
    r'(?m)^(\s*(?:SUN|SOLAR|LIMB)[A-Z_]*LINE_WIDTH[A-Z_]*\s*=\s*)[0-9.]+\s*$',
    r'\g<1>1.000',
    text,
)

width_patterns = (
    r'(?m)^(\s*TRACK[A-Z_]*LINE_WIDTH[A-Z_]*\s*=\s*)[0-9.]+\s*$',
    r'(?m)^(\s*DISK[A-Z_]*LINE_WIDTH[A-Z_]*\s*=\s*)[0-9.]+\s*$',
    r'(?m)^(\s*MERCURY[A-Z_]*LINE_WIDTH[A-Z_]*\s*=\s*)[0-9.]+\s*$',
    r'(?m)^(\s*MARKER[A-Z_]*LINE_WIDTH[A-Z_]*\s*=\s*)[0-9.]+\s*$',
)
width_assignment_count = 0
for pattern in width_patterns:
    text, count = re.subn(pattern, r'\g<1>0.500', text)
    width_assignment_count += count

text = text.replace("linewidth=0.375", "linewidth=0.500")
text = text.replace("linewidth = 0.375", "linewidth = 0.500")
text = text.replace("linewidth=0.3750", "linewidth=0.500")
text = text.replace("linewidth = 0.3750", "linewidth = 0.500")

solar_plot_pattern = re.compile(
    r'(ax_plot\.plot\(\s*'
    r'reference_solar_radius\s*\*\s*np\.cos\(theta\)\s*,\s*'
    r'reference_solar_radius\s*\*\s*np\.sin\(theta\)\s*\,)'
    r'(.*?)'
    r'(\n\s*\))',
    re.DOTALL,
)


def patch_solar_plot(match: re.Match[str]) -> str:
    body = match.group(2)
    body = re.sub(r'color\s*=\s*[^,\n]+', 'color="#FACC15"', body, count=1)
    body = re.sub(r'linewidth\s*=\s*[^,\n]+', 'linewidth=1.000', body, count=1)
    return match.group(1) + body + match.group(3)


text, solar_plot_count = solar_plot_pattern.subn(patch_solar_plot, text, count=1)

table_block = r'''
    # V0014 inset table: definitions and track angles, shifted upward by about two rows.
    for _existing_plot_table in list(ax_plot.tables):
        _existing_plot_table.set_visible(False)

    _a_prime = np.asarray(
        site_results["MERCURY_BAY"]["event_points"]["MAX"], dtype=float
    )
    _b_prime = np.asarray(
        site_results["VARDO"]["event_points"]["MAX"], dtype=float
    )
    _angle_a = float(fit_mb["angle_deg"])
    _angle_b = float(fit_v["angle_deg"])

    _ab_rows = [
        ["Point", "Definition", r"$\xi$ (arcsec)", r"$\eta$ (arcsec)", "Track angle"],
        ["A", "Mercury Bay observer / cyan JPL track", "—", "—", f"{_angle_a:.6f}°"],
        ["A′", "Mercury Bay closest-approach center", f"{_a_prime[0]:.6f}", f"{_a_prime[1]:.6f}", "—"],
        ["B", "Vardø observer / amber virtual JPL track", "—", "—", f"{_angle_b:.6f}°"],
        ["B′", "Vardø closest-approach center", f"{_b_prime[0]:.6f}", f"{_b_prime[1]:.6f}", "—"],
        ["A/B mean", "Average fitted track angle", "—", "—", f"{average_angle:.6f}°"],
    ]
    _ab_table = ax_plot.table(
        cellText=_ab_rows,
        cellLoc="left",
        colWidths=[0.10, 0.43, 0.17, 0.17, 0.18],
        bbox=[0.075, 0.145, 0.850, 0.205],
        zorder=20,
    )
    _ab_table.auto_set_font_size(False)
    _ab_table.set_fontsize(6.8)
    for (_row, _column), _cell in _ab_table.get_celld().items():
        _cell.set_edgecolor("#CBD5E1")
        _cell.set_linewidth(0.500)
        _cell.get_text().set_color("white")
        if _row == 0:
            _cell.set_facecolor("#1E3A5F")
            _cell.get_text().set_fontweight("bold")
        elif _row in (1, 2):
            _cell.set_facecolor("#123B48")
        elif _row in (3, 4):
            _cell.set_facecolor("#4A3510")
        else:
            _cell.set_facecolor("#0F172A")
'''

save_marker = "    figure.savefig("
if save_marker not in text:
    raise RuntimeError("V0013 figure-save marker was not found.")
text = text.replace(save_marker, table_block + "\n" + save_marker, 1)

comment_marker = '    print("COMMENTS")'
comment_insert = (
    '    print("COMMENTS")\n'
    '    print("V0014 solar limb: yellow, 1.000-point line.")\n'
    '    print("V0014 tracks, Mercury disks, centers, and inset-table borders: 0.500-point lines.")\n'
    '    print("A and B identify the Mercury Bay and Vardø observer tracks; A′ and B′ are their closest-approach Mercury centers.")'
)
if comment_marker not in text:
    raise RuntimeError("V0013 COMMENTS marker was not found.")
text = text.replace(comment_marker, comment_insert, 1)

if not text.startswith("# V0014\n") or not text.rstrip().endswith("# V0014"):
    raise RuntimeError("V0014 source boundary audit failed after promotion.")

compile(text, str(OUTPUT_PATH), "exec")
OUTPUT_PATH.write_text(text, encoding="utf-8")
patched_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

print("Loader version: V0014")
print(f"Verified immutable V0013 SHA-256: {source_digest}")
print(f"Solar color assignment repairs: {sun_color_count}")
print(f"Solar width assignment repairs: {sun_width_count}")
print(f"Track/disk width assignment repairs: {width_assignment_count}")
print(f"Solar plotting-call repairs: {solar_plot_count}")
print(f"Patched complete V0014 SHA-256: {patched_sha256}")

exec(
    compile(text, str(OUTPUT_PATH), "exec"),
    {"__name__": "__main__", "__file__": str(OUTPUT_PATH)},
)
# V0014
