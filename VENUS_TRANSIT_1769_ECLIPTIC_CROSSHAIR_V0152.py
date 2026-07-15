# V0152
# Audit reference: 1769 ecliptic crosshair using exact d-rho/dt-zero closest approach with rho overlay at the vertex.
from __future__ import annotations
import urllib.request

SOURCE_URL=("https://raw.githubusercontent.com/gear66me-ui/"
"IERS_TN36_V01_MASTER-1/main/"
"VENUS_TRANSIT_2012_ECLIPTIC_CROSSHAIR_V0150.py")
with urllib.request.urlopen(SOURCE_URL,timeout=90) as response:
    wrapper=response.read().decode("utf-8")

required=[
    "# V0150",
    'source=source.replace("YEAR = 1761","YEAR = 2012")',
    'source=source.replace(\'CENTER_UTC = "1761-06-06 06:00"\',\'CENTER_UTC = "2012-06-06 01:00"\')',
    'exec(compile(source,"VENUS_TRANSIT_2012_ECLIPTIC_CROSSHAIR_V0150.py","exec"))',
]
for marker in required:
    if marker not in wrapper:
        raise RuntimeError(f"REJECTED missing V0150 marker: {marker}")

wrapper=wrapper.replace("# V0150","# V0152")
wrapper=wrapper.replace('VERSION = "V0150"','VERSION = "V0152"')
wrapper=wrapper.replace(
    'source=source.replace("YEAR = 1761","YEAR = 2012")',
    'source=source.replace("YEAR = 1761","YEAR = 1769")')
wrapper=wrapper.replace(
    'source=source.replace(\'CENTER_UTC = "1761-06-06 06:00"\',\'CENTER_UTC = "2012-06-06 01:00"\')',
    'source=source.replace(\'CENTER_UTC = "1761-06-06 06:00"\',\'CENTER_UTC = "1769-06-03 22:00"\')')
wrapper=wrapper.replace(
    'VENUS_TRANSIT_2012_ECLIPTIC_CROSSHAIR_V0150',
    'VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152')
wrapper=wrapper.replace('print("Transit                              2012")','print("Transit                              1769")')
wrapper=wrapper.replace(
    '"2012 Venus Transit — Ecliptic Reference And Transit Tracks"',
    '"1769 Venus Transit — Ecliptic Reference And Transit Tracks"')

exec_marker='exec(compile(source,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152.py","exec"))'
if exec_marker not in wrapper:
    raise RuntimeError("REJECTED nested execution marker missing")

patch=r'''
if "from scipy.optimize import minimize_scalar" not in source:
    raise RuntimeError("REJECTED scipy optimizer import missing in generated source")
source=source.replace("from scipy.optimize import minimize_scalar","from scipy.optimize import minimize_scalar, brentq",1)

start=source.index("def closest_approach(")
end=source.index("\ndef physical_east_north_basis",start)
new_solver='''def closest_approach(
    sun: VectorSeries,
    venus: VectorSeries,
) -> tuple[float, float, np.ndarray]:
    if len(sun.jd) != len(venus.jd) or not np.allclose(sun.jd, venus.jd, atol=1.0e-11, rtol=0.0):
        raise RuntimeError("REJECTED mismatched JPL grids")
    sun_curves = cubic_splines(sun)
    venus_curves = cubic_splines(venus)
    def rho(jd_value: float) -> float:
        su = unit(evaluate(sun_curves, jd_value))
        vu = unit(evaluate(venus_curves, jd_value))
        return math.atan2(float(np.linalg.norm(np.cross(su, vu))), float(np.dot(su, vu)))
    sampled = np.array([rho(float(jd)) for jd in sun.jd], dtype=float)
    index = int(np.argmin(sampled))
    lower = float(sun.jd[max(0,index-3)])
    upper = float(sun.jd[min(len(sun.jd)-1,index+3)])
    seed = minimize_scalar(rho,bounds=(lower,upper),method="bounded",options={"xatol":1.0e-13,"maxiter":600})
    if not seed.success:
        raise RuntimeError("REJECTED rho-minimum seed")
    h = 0.5 / 1440.0
    def rho_dot(jd_value: float) -> float:
        return rho(jd_value+h)-rho(jd_value-h)
    lower_root = float(seed.x)-5.0/1440.0
    upper_root = float(seed.x)+5.0/1440.0
    if rho_dot(lower_root)*rho_dot(upper_root)>0.0:
        raise RuntimeError("REJECTED d rho/dt zero not bracketed")
    jd_ca = float(brentq(rho_dot,lower_root,upper_root,xtol=1.0e-14,rtol=1.0e-14,maxiter=200))
    return jd_ca, rho(jd_ca), evaluate(sun_curves,jd_ca)

'''
source=source[:start]+new_solver+source[end+1:]

old='''    index_ca = int(np.argmin(np.abs(sun.jd - jd_ca)))
    ca_x = float(relative_x_projected[index_ca])
    ca_y = float(relative_y_projected[index_ca])
    solar_radius_arcsec = float(math.asin(R_SUN_KM / sun_distance_km[index_ca]) * AS_PER_RAD)
    venus_radius_arcsec = float(math.asin(R_VENUS_KM / venus_distance_km[index_ca]) * AS_PER_RAD)'''
new='''    index_ca = int(np.argmin(np.abs(sun.jd - jd_ca)))
    sun_curves_exact = cubic_splines(sun)
    venus_curves_exact = cubic_splines(venus)
    sun_exact = evaluate(sun_curves_exact, jd_ca)
    venus_exact = evaluate(venus_curves_exact, jd_ca)
    px, py, los = projected_basis
    sun_dir_exact = unit(sun_exact)
    venus_dir_exact = unit(venus_exact)
    sun_x_exact = float(np.dot(sun_dir_exact, px) / np.dot(sun_dir_exact, los) * AS_PER_RAD)
    sun_y_exact = float(np.dot(sun_dir_exact, py) / np.dot(sun_dir_exact, los) * AS_PER_RAD)
    venus_x_exact = float(np.dot(venus_dir_exact, px) / np.dot(venus_dir_exact, los) * AS_PER_RAD)
    venus_y_exact = float(np.dot(venus_dir_exact, py) / np.dot(venus_dir_exact, los) * AS_PER_RAD)
    ca_x = venus_x_exact - sun_x_exact
    ca_y = venus_y_exact - sun_y_exact
    rho_ca_arcsec = float(math.hypot(ca_x, ca_y))
    solar_radius_arcsec = float(math.asin(R_SUN_KM / np.linalg.norm(sun_exact)) * AS_PER_RAD)
    venus_radius_arcsec = float(math.asin(R_VENUS_KM / np.linalg.norm(venus_exact)) * AS_PER_RAD)'''
if old not in source:
    raise RuntimeError("REJECTED exact-CA coordinate marker missing")
source=source.replace(old,new,1)

anchor='''    ax.scatter(
        [ca_x],
        [ca_y],'''
overlay='''    ax.add_patch(Circle(
        (0.0, 0.0),
        rho_ca_arcsec,
        facecolor="none",
        edgecolor="#42D7C3",
        linewidth=0.72,
        linestyle="--",
        alpha=0.92,
        zorder=4,
        label=f"rho at CA = {rho_ca_arcsec:.6f} arcsec",
    ))
    ax.plot([0.0, ca_x], [0.0, ca_y], color="#42D7C3", linewidth=0.58, alpha=0.90, zorder=6)
    ax.text(0.52*ca_x, 0.52*ca_y, f"rho = {rho_ca_arcsec:.6f} arcsec", color="#42D7C3", fontsize=8.0, ha="center", va="bottom", zorder=8)

'''+anchor
if anchor not in source:
    raise RuntimeError("REJECTED closest-approach scatter marker missing")
source=source.replace(anchor,overlay,1)

old_line='        TextArea("Ecliptic Reference: 0.000°", textprops={"color":"#000000","fontsize":9.6}),' 
new_lines=(
    '        TextArea(f"Closest Approach (UTC): {Time(jd_ca, format=\'jd\', scale=\'tdb\').utc.strftime(\'%Y-%m-%d %H:%M:%S.%f\')[:-3]}", textprops={"color":"#F5F5F5","fontsize":9.6}),\n'
    '        TextArea(f"rho at CA: {rho_ca_arcsec:.6f} arcsec", textprops={"color":"#42D7C3","fontsize":9.6}),\n'
    + old_line)
if old_line not in source:
    raise RuntimeError("REJECTED angle-box insertion marker missing")
source=source.replace(old_line,new_lines,1)

for marker in ["brentq","rho_ca_arcsec","rho at CA","Closest Approach (UTC)"]:
    if marker not in source:
        raise RuntimeError(f"REJECTED generated-source marker missing: {marker}")
'''

wrapper=wrapper.replace(exec_marker,patch+"\n"+exec_marker,1)
if wrapper.splitlines()[0]!="# V0152" or wrapper.splitlines()[-1]!="# V0152":
    raise RuntimeError("REJECTED version boundary")
compile(wrapper,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152_WRAPPER.py","exec")
exec(compile(wrapper,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152.py","exec"))
# V0152