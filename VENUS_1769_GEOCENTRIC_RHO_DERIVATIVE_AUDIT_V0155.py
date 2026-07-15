# V0155
# Audit reference: corrected numerical-time conditioning for the 1769 geocentric rho minimum, derivative root, rolling means, and CA y-axis crossing.
from __future__ import annotations

import urllib.request

SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "VENUS_1769_GEOCENTRIC_RHO_DERIVATIVE_AUDIT_V0154.py"
)

with urllib.request.urlopen(SOURCE_URL, timeout=90) as response:
    source = response.read().decode("utf-8")

required = [
    "# V0154",
    'VERSION = "V0154"',
    "jd = jd_sun",
    "sun_splines = make_splines(jd, sun_xyz)",
    "jd_ca = 0.5 * (jd_min + jd_root)",
    "analytic_at_ca = rho_dot_per_day(jd_ca) / 1440.0",
    "dense_jd = jd_ca + dense_minutes / 1440.0",
]
for marker in required:
    if marker not in source:
        raise RuntimeError(f"REJECTED missing V0154 marker: {marker}")

source = source.replace("# V0154", "# V0155")
source = source.replace('VERSION = "V0154"', 'VERSION = "V0155"')
source = source.replace("V0154_OUTPUT", "V0155_OUTPUT")
source = source.replace("V0154.png", "V0155.png")
source = source.replace("V0154.csv", "V0155.csv")
source = source.replace(
    "    jd = jd_sun\n\n    sun_splines = make_splines(jd, sun_xyz)\n    venus_splines = make_splines(jd, venus_xyz)",
    "    jd = jd_sun\n"
    "    jd_origin = float(jd[len(jd) // 2])\n"
    "    t_seconds = (jd - jd_origin) * 86400.0\n\n"
    "    sun_splines = make_splines(t_seconds, sun_xyz)\n"
    "    venus_splines = make_splines(t_seconds, venus_xyz)",
)
source = source.replace(
    "    x_spline = CubicSpline(jd, relative_x, bc_type=\"natural\")\n"
    "    y_spline = CubicSpline(jd, relative_y, bc_type=\"natural\")",
    "    x_spline = CubicSpline(t_seconds, relative_x, bc_type=\"natural\")\n"
    "    y_spline = CubicSpline(t_seconds, relative_y, bc_type=\"natural\")",
)
source = source.replace("def rho2(jd_value: float)", "def rho2(t_value: float)")
source = source.replace("x_spline(jd_value)", "x_spline(t_value)")
source = source.replace("y_spline(jd_value)", "y_spline(t_value)")
source = source.replace("def rho_dot_per_day(jd_value: float)", "def rho_dot_per_second(t_value: float)")
source = source.replace("dx_spline(jd_value)", "dx_spline(t_value)")
source = source.replace("dy_spline(jd_value)", "dy_spline(t_value)")
source = source.replace(
    "        bounds=(float(jd[lo]), float(jd[hi])),\n"
    "        method=\"bounded\",\n"
    "        options={\"xatol\": 1.0e-13, \"maxiter\": 500},",
    "        bounds=(float(t_seconds[lo]), float(t_seconds[hi])),\n"
    "        method=\"bounded\",\n"
    "        options={\"xatol\": 1.0e-7, \"maxiter\": 500},",
)
source = source.replace("    jd_min = float(minimum.x)", "    t_min = float(minimum.x)")
source = source.replace(
    "    root_lo = float(jd[lo])\n"
    "    root_hi = float(jd[hi])\n"
    "    if rho_dot_per_day(root_lo) * rho_dot_per_day(root_hi) >= 0.0:\n"
    "        raise RuntimeError(\"REJECTED drho/dt root not bracketed\")\n"
    "    jd_root = float(brentq(rho_dot_per_day, root_lo, root_hi, xtol=1.0e-13, rtol=1.0e-14, maxiter=500))\n\n"
    "    jd_ca = 0.5 * (jd_min + jd_root)",
    "    root_lo = float(t_seconds[lo])\n"
    "    root_hi = float(t_seconds[hi])\n"
    "    if rho_dot_per_second(root_lo) * rho_dot_per_second(root_hi) >= 0.0:\n"
    "        raise RuntimeError(\"REJECTED drho/dt root not bracketed\")\n"
    "    t_root = float(brentq(rho_dot_per_second, root_lo, root_hi, xtol=1.0e-9, rtol=1.0e-14, maxiter=500))\n\n"
    "    t_ca = t_root\n"
    "    jd_ca = jd_origin + t_ca / 86400.0",
)
source = source.replace("method_agreement_seconds = abs(jd_min - jd_root) * 86400.0", "method_agreement_seconds = abs(t_min - t_root)")
source = source.replace("x_spline(jd_ca)", "x_spline(t_ca)")
source = source.replace("y_spline(jd_ca)", "y_spline(t_ca)")
source = source.replace("dx_spline(jd_ca)", "dx_spline(t_ca)")
source = source.replace("dy_spline(jd_ca)", "dy_spline(t_ca)")
source = source.replace("minutes = (jd - jd_ca) * 1440.0", "minutes = (t_seconds - t_ca) / 60.0")
source = source.replace("dx_spline(jd)", "dx_spline(t_seconds)")
source = source.replace("dy_spline(jd)", "dy_spline(t_seconds)")
source = source.replace("/ sample_rho / 1440.0", "/ sample_rho * 60.0")
source = source.replace("CubicSpline(jd[valid], rolling[valid]", "CubicSpline(t_seconds[valid], rolling[valid]")
source = source.replace("rolling_spline(jd_ca)", "rolling_spline(t_ca)")
source = source.replace("analytic_at_ca = rho_dot_per_day(jd_ca) / 1440.0", "analytic_at_ca = rho_dot_per_second(t_ca) * 60.0")
source = source.replace(
    "    dense_jd = jd_ca + dense_minutes / 1440.0\n"
    "    dense_x = x_spline(dense_jd)\n"
    "    dense_y = y_spline(dense_jd)",
    "    dense_t = t_ca + dense_minutes * 60.0\n"
    "    dense_x = x_spline(dense_t)\n"
    "    dense_y = y_spline(dense_t)",
)
source = source.replace("dx_spline(dense_jd)", "dx_spline(dense_t)")
source = source.replace("dy_spline(dense_jd)", "dy_spline(dense_t)")
source = source.replace("/ dense_rho / 1440.0", "/ dense_rho * 60.0")
source = source.replace("rho2(jd_ca)", "rho2(t_ca)")
source = source.replace(
    'print("The closest-approach epoch is calculated from geometric geocentric vectors, not apparent RA/DEC ephemerides.")',
    'print("The closest-approach epoch is solved in seconds relative to the JPL grid midpoint to avoid Julian-date precision loss.")',
)
source = source.replace(
    'print("The transit track is rotated only after the minimum is solved, so closest approach lies on the y-axis by construction.")',
    'print("Closest approach is the analytic drho/dt root; rho minimization is an independent agreement check.")\n'
    '    print("The transit track is rotated only after the minimum is solved, so closest approach lies on the y-axis by construction.")',
)

for forbidden in [
    "jd_ca = 0.5 * (jd_min + jd_root)",
    "rho_dot_per_day",
    "dense_jd",
    "x_spline(jd_ca)",
    "y_spline(jd_ca)",
]:
    if forbidden in source:
        raise RuntimeError(f"REJECTED stale V0154 expression survived: {forbidden}")

for marker in [
    "t_seconds = (jd - jd_origin) * 86400.0",
    "t_ca = t_root",
    "analytic_at_ca = rho_dot_per_second(t_ca) * 60.0",
    "dense_t = t_ca + dense_minutes * 60.0",
]:
    if marker not in source:
        raise RuntimeError(f"REJECTED corrected marker missing: {marker}")

if source.splitlines()[0] != "# V0155" or source.splitlines()[-1] != "# V0155":
    raise RuntimeError("REJECTED version boundary")

compile(source, "VENUS_1769_GEOCENTRIC_RHO_DERIVATIVE_AUDIT_V0155_GENERATED.py", "exec")
exec(compile(source, "VENUS_1769_GEOCENTRIC_RHO_DERIVATIVE_AUDIT_V0155.py", "exec"))
# V0155