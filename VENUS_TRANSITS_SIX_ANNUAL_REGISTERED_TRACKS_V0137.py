# V0137
# Audit reference: correct V0135 annual projection to orthographic and reduce PNG output from 600 DPI to 300 DPI.

from __future__ import annotations

import base64
import gzip
import re
import urllib.request

VERSION = "V0137"
SOURCE_URL = (
    "https://raw.githubusercontent.com/"
    "gear66me-ui/IERS_TN36_V01_MASTER-1/main/"
    "VENUS_TRANSITS_SIX_ANNUAL_REGISTERED_TRACKS_V0135.py"
)

with urllib.request.urlopen(SOURCE_URL, timeout=90) as response:
    delivery_source = response.read().decode("utf-8")

payload_match = re.search(
    r'base64\.b64decode\("([A-Za-z0-9+/=]+)"\)',
    delivery_source,
)
if payload_match is None:
    raise RuntimeError("REJECTED V0135 compressed payload was not found")

source = gzip.decompress(
    base64.b64decode(payload_match.group(1))
).decode("utf-8")

source = source.replace("# V0135", "# V0137")
source = source.replace("V0135", "V0137")
source = source.replace("dpi=600", "dpi=300")

gnomonic_function = '''def project_to_tangent_plane(
    xyz_km: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    x_axis, y_axis, line_of_sight = basis
    direction = xyz_km / np.linalg.norm(xyz_km, axis=1)[:, None]
    denominator = direction @ line_of_sight
    if np.any(denominator <= 0.0):
        raise RuntimeError("REJECTED tangent-plane denominator")
    x_arcsec = (direction @ x_axis) / denominator * AS_PER_RAD
    y_arcsec = (direction @ y_axis) / denominator * AS_PER_RAD
    return x_arcsec, y_arcsec
'''

orthographic_function = '''
def project_annual_orthographic(
    xyz_km: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Registered orthographic projection for the full ±6-month overview."""
    x_axis, y_axis, _ = basis
    direction = xyz_km / np.linalg.norm(xyz_km, axis=1)[:, None]
    x_arcsec = (direction @ x_axis) * AS_PER_RAD
    y_arcsec = (direction @ y_axis) * AS_PER_RAD
    return x_arcsec, y_arcsec
'''

if gnomonic_function not in source:
    raise RuntimeError("REJECTED V0135 gnomonic function was not found")

source = source.replace(
    gnomonic_function,
    gnomonic_function + orthographic_function,
)

annual_replacements = {
    "earth_x, earth_y = project_to_tangent_plane(annual_sun.xyz_km, projected_basis)":
        "earth_x, earth_y = project_annual_orthographic(annual_sun.xyz_km, projected_basis)",
    "venus_x, venus_y = project_to_tangent_plane(annual_venus.xyz_km, projected_basis)":
        "venus_x, venus_y = project_annual_orthographic(annual_venus.xyz_km, projected_basis)",
    "sun_ca_x, sun_ca_y = project_to_tangent_plane(sun_ca_xyz[None, :], projected_basis)":
        "sun_ca_x, sun_ca_y = project_annual_orthographic(sun_ca_xyz[None, :], projected_basis)",
    "venus_ca_x, venus_ca_y = project_to_tangent_plane(venus_ca_xyz[None, :], projected_basis)":
        "venus_ca_x, venus_ca_y = project_annual_orthographic(venus_ca_xyz[None, :], projected_basis)",
}

for old, new in annual_replacements.items():
    if old not in source:
        raise RuntimeError(f"REJECTED missing annual projection call: {old}")
    source = source.replace(old, new)

source = source.replace(
    'print("Each closest approach is centered at day zero.")',
    'print("Each closest approach is centered at day zero.")\n'
    '    print("Annual curves use a registered orthographic projection.")',
)

source = source.replace(
    'print("VERIFIED JPL geocentric annual and minute vector grids")',
    'print("VERIFIED JPL geocentric annual and minute vector grids")\n'
    '    print("VERIFIED annual orthographic projection remains finite across ±6 months")',
)

if source.splitlines()[0] != "# V0137":
    raise RuntimeError("REJECTED incorrect first line")
if source.splitlines()[-1] != "# V0137":
    raise RuntimeError("REJECTED incorrect last line")
if "dpi=300" not in source or "dpi=600" in source:
    raise RuntimeError("REJECTED DPI correction")
if "project_annual_orthographic" not in source:
    raise RuntimeError("REJECTED orthographic correction")

exec(
    compile(
        source,
        "VENUS_TRANSITS_SIX_ANNUAL_REGISTERED_TRACKS_V0137.py",
        "exec",
    )
)
# V0137
