# V0152A
# Audit reference: cache-busted 1769 ecliptic crosshair using the V0102C d-rho/dt-zero closest approach, with rho circle through the exact CA vertex.
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
    'TextArea("Ecliptic Reference: 0.000°"',
    'exec(compile(source,"VENUS_TRANSIT_2012_ECLIPTIC_CROSSHAIR_V0150.py","exec"))',
]
for marker in required:
    if marker not in wrapper:
        raise RuntimeError(f"REJECTED missing V0150 marker: {marker}")

wrapper=wrapper.replace("# V0150","# V0152A")
wrapper=wrapper.replace('VERSION = "V0150"','VERSION = "V0152A"')
wrapper=wrapper.replace(
    'source=source.replace("YEAR = 1761","YEAR = 2012")',
    'source=source.replace("YEAR = 1761","YEAR = 1769")')
wrapper=wrapper.replace(
    'source=source.replace(\'CENTER_UTC = "1761-06-06 06:00"\',\'CENTER_UTC = "2012-06-06 01:00"\')',
    'source=source.replace(\'CENTER_UTC = "1761-06-06 06:00"\',\'CENTER_UTC = "1769-06-03 22:00"\')')
wrapper=wrapper.replace(
    'VENUS_TRANSIT_2012_ECLIPTIC_CROSSHAIR_V0150',
    'VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152A')
wrapper=wrapper.replace('print("Transit                              2012")','print("Transit                              1769")')
wrapper=wrapper.replace(
    '"2012 Venus Transit — Ecliptic Reference And Transit Tracks"',
    '"1769 Venus Transit — Ecliptic Reference And Transit Tracks"')

start=wrapper.index('def closest_approach(')
end=wrapper.index('\ndef physical_east_north_basis',start)
new_solver='''def closest_approach(\n    sun: VectorSeries,\n    venus: VectorSeries,\n) -> tuple[float, float, np.ndarray]:\n    if len(sun.jd) != len(venus.jd) or not np.allclose(sun.jd, venus.jd, atol=1.0e-11, rtol=0.0):\n        raise RuntimeError("REJECTED mismatched JPL grids")\n    sun_curves = cubic_splines(sun)\n    venus_curves = cubic_splines(venus)\n    def rho(jd_value: float) -> float:\n        su = unit(evaluate(sun_curves, jd_value))\n        vu = unit(evaluate(venus_curves, jd_value))\n        return math.atan2(float(np.linalg.norm(np.cross(su, vu))), float(np.dot(su, vu)))\n    sampled = np.array([rho(float(jd)) for jd in sun.jd], dtype=float)\n    index = int(np.argmin(sampled))\n    lo_seed = float(sun.jd[max(0,index-3)])\n    hi_seed = float(sun.jd[min(len(sun.jd)-1,index+3)])\n    seed = minimize_scalar(rho,bounds=(lo_seed,hi_seed),method="bounded",options={"xatol":1.0e-13,"maxiter":600})\n    if not seed.success:\n        raise RuntimeError("REJECTED rho-minimum seed")\n    h = 0.5 / 1440.0\n    def rhodot(jd_value: float) -> float:\n        return rho(jd_value+h)-rho(jd_value-h)\n    lo = float(seed.x)-5.0/1440.0\n    hi = float(seed.x)+5.0/1440.0\n    flo = rhodot(lo)\n    fhi = rhodot(hi)\n    if flo*fhi > 0.0:\n        raise RuntimeError("REJECTED d rho/dt zero not bracketed")\n    for _ in range(120):\n        mid = 0.5*(lo+hi)\n        fmid = rhodot(mid)\n        if flo*fmid <= 0.0:\n            hi = mid\n            fhi = fmid\n        else:\n            lo = mid\n            flo = fmid\n    jd_ca = 0.5*(lo+hi)\n    return jd_ca, rho(jd_ca), evaluate(sun_curves,jd_ca)\n\n'''
wrapper=wrapper[:start]+new_solver+wrapper[end+1:]

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
if old not in wrapper:
    raise RuntimeError("REJECTED exact-CA coordinate marker missing")
wrapper=wrapper.replace(old,new,1)

anchor='''    ax.scatter(
        [ca_x],
        [ca_y],'''
rho_overlay='''    ax.add_patch(Circle(
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
    ax.text(0.52*ca_x, 0.52*ca_y, f"rho = {rho_ca_arcsec:.6f} arcsec",
        color="#42D7C3", fontsize=8.0, ha="center", va="bottom", zorder=8)

'''+anchor
if anchor not in wrapper:
    raise RuntimeError("REJECTED CA scatter marker missing")
wrapper=wrapper.replace(anchor,rho_overlay,1)

old_line='        TextArea("Ecliptic Reference: 0.000°", textprops={"color":"#000000","fontsize":9.6}),' 
new_lines=(
    '        TextArea(f"Closest Approach (UTC): {Time(jd_ca, format=\'jd\', scale=\'tdb\').utc.strftime(\'%Y-%m-%d %H:%M:%S.%f\')[:-3]}", '
    'textprops={"color":"#F5F5F5","fontsize":9.6}),\n'
    '        TextArea(f"rho at CA: {rho_ca_arcsec:.6f} arcsec", textprops={"color":"#42D7C3","fontsize":9.6}),\n'
    + old_line)
if old_line not in wrapper:
    raise RuntimeError("REJECTED angle-box insertion marker missing")
wrapper=wrapper.replace(old_line,new_lines,1)

if wrapper.splitlines()[0]!="# V0152A" or wrapper.splitlines()[-1]!="# V0152A":
    raise RuntimeError("REJECTED version boundary")
for marker in ['YEAR = 1769','CENTER_UTC = "1769-06-03 22:00"','rho at CA','rho_ca_arcsec','Closest Approach (UTC)','Ecliptic Reference']:
    if marker not in wrapper:
        raise RuntimeError(f"REJECTED final marker missing: {marker}")

compile(wrapper,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152A_WRAPPER.py","exec")
exec(compile(wrapper,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152A.py","exec"))
# V0152A