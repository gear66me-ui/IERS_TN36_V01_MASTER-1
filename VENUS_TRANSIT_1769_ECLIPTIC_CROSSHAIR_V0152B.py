# V0152B
# Audit reference: direct V0131-based 1769 crosshair plot; d-rho/dt zero defines CA; rho circle overlays exact vertex.
from __future__ import annotations
import urllib.request

SOURCE_URL=("https://raw.githubusercontent.com/gear66me-ui/"
"IERS_TN36_V01_MASTER-1/main/"
"VENUS_TRANSIT_1761_EARTH_VENUS_PROJECTED_TRACKS_V0131.py")
with urllib.request.urlopen(SOURCE_URL,timeout=90) as response:
    source=response.read().decode("utf-8")

for marker in ["# V0131",'VERSION = "V0131"',"YEAR = 1761",'CENTER_UTC = "1761-06-06 06:00"',"def closest_approach(","from scipy.optimize import minimize_scalar"]:
    if marker not in source:
        raise RuntimeError(f"REJECTED missing V0131 marker: {marker}")

source=source.replace("# V0131","# V0152B")
source=source.replace('VERSION = "V0131"','VERSION = "V0152B"')
source=source.replace("YEAR = 1761","YEAR = 1769")
source=source.replace('CENTER_UTC = "1761-06-06 06:00"','CENTER_UTC = "1769-06-03 22:00"')
source=source.replace("VENUS_TRANSIT_1761_EARTH_VENUS_PROJECTED_TRACKS_V0131","VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152B")
source=source.replace('print("Transit                              1761")','print("Transit                              1769")')
source=source.replace('"1761 Venus Transit — Earth Orbit, Venus Orbit, and Projected Relative Track"','"1769 Venus Transit — Ecliptic Reference And Transit Tracks"')
source=source.replace("from scipy.optimize import minimize_scalar","from scipy.optimize import minimize_scalar, brentq")
source=source.replace("from matplotlib.patches import Circle","from matplotlib.patches import Circle\nfrom matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker")

start=source.index('def closest_approach(')
end=source.index('\ndef physical_east_north_basis',start)
new_solver='''def closest_approach(\n    sun: VectorSeries,\n    venus: VectorSeries,\n) -> tuple[float, float, np.ndarray]:\n    if len(sun.jd) != len(venus.jd) or not np.allclose(sun.jd, venus.jd, atol=1.0e-11, rtol=0.0):\n        raise RuntimeError("REJECTED mismatched JPL grids")\n    sun_curves = cubic_splines(sun)\n    venus_curves = cubic_splines(venus)\n    def rho(jd_value: float) -> float:\n        su = unit(evaluate(sun_curves, jd_value))\n        vu = unit(evaluate(venus_curves, jd_value))\n        return math.atan2(float(np.linalg.norm(np.cross(su, vu))), float(np.dot(su, vu)))\n    sampled = np.array([rho(float(jd)) for jd in sun.jd], dtype=float)\n    index = int(np.argmin(sampled))\n    lo_seed = float(sun.jd[max(0,index-3)])\n    hi_seed = float(sun.jd[min(len(sun.jd)-1,index+3)])\n    seed = minimize_scalar(rho,bounds=(lo_seed,hi_seed),method="bounded",options={"xatol":1.0e-13,"maxiter":600})\n    if not seed.success:\n        raise RuntimeError("REJECTED rho-minimum seed")\n    h = 0.5 / 1440.0\n    def rhodot(jd_value: float) -> float:\n        return rho(jd_value+h)-rho(jd_value-h)\n    lo = float(seed.x)-5.0/1440.0\n    hi = float(seed.x)+5.0/1440.0\n    if rhodot(lo)*rhodot(hi) > 0.0:\n        raise RuntimeError("REJECTED d rho/dt zero not bracketed")\n    jd_ca = float(brentq(rhodot,lo,hi,xtol=1.0e-14,rtol=1.0e-14,maxiter=200))\n    return jd_ca, rho(jd_ca), evaluate(sun_curves,jd_ca)\n\n'''
source=source[:start]+new_solver+source[end+1:]

source=source.replace(
"relative_x_projected = venus_x_projected - sun_x_projected\n    relative_y_projected = venus_y_projected - sun_y_projected",
"relative_x_projected = venus_x_projected - sun_x_projected\n    relative_y_projected = venus_y_projected - sun_y_projected\n    relative_x_physical = venus_x_physical - sun_x_physical\n    relative_y_physical = venus_y_physical - sun_y_physical")
source=source.replace(
"venus_track = fit_track(hours, venus_x_physical[mask], venus_y_physical[mask])\n    projected_relative_track = fit_track(",
"physical_relative_track = fit_track(hours, relative_x_physical[mask], relative_y_physical[mask])\n    projected_relative_track = fit_track(")
source=source.replace(
'        ("Earth orbit", earth_track.signed_angle_deg, "#3EA6FF"),\n        ("Projected Venus−Sun relative track", projected_relative_track.signed_angle_deg, "#F5F5F5"),\n        ("Venus orbit", venus_track.signed_angle_deg, "#38D66B"),',
'        ("Earth Track From Ecliptic", earth_track.signed_angle_deg, "#3EA6FF"),\n        ("Projected Venus Transit Track", projected_relative_track.signed_angle_deg, "#F5F5F5"),\n        ("Venus Transit Track From Ecliptic", physical_relative_track.signed_angle_deg, "#38D66B"),')
source=source.replace('if label in {"Earth orbit", "Venus orbit"}:','if label in {"Earth Track From Ecliptic", "Venus Transit Track From Ecliptic"}:')

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

solar='''    ax.add_patch(Circle(
        (0.0, 0.0),
        solar_radius_arcsec,
        facecolor="#C98A18",
        edgecolor="#E64A19",
        linewidth=1.15,
        alpha=0.92,
        zorder=1,
    ))'''
cross=solar+'''\n    crosshair_extent = 1.02 * solar_radius_arcsec\n    ax.plot([-crosshair_extent, crosshair_extent], [0.0, 0.0], color="#000000", linewidth=0.72, alpha=0.92, zorder=2)\n    ax.plot([0.0, 0.0], [-crosshair_extent, crosshair_extent], color="#000000", linewidth=0.72, alpha=0.92, zorder=2)\n    ax.text(0.72 * solar_radius_arcsec, 0.035 * solar_radius_arcsec, "Ecliptic Reference  0.000°", color="#000000", fontsize=8.4, ha="center", va="bottom", zorder=3)'''
source=source.replace(solar,cross,1)

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
source=source.replace(anchor,rho_overlay,1)

ann_start=source.index('    annotation = "\\n".join([')
ann_end=source.index('    extent = 1.10 * solar_radius_arcsec',ann_start)
new_ann='''    offset_x = 0.18 * solar_radius_arcsec if ca_x <= 0.0 else -0.18 * solar_radius_arcsec
    offset_y = 0.16 * solar_radius_arcsec if ca_y <= 0.0 else -0.16 * solar_radius_arcsec
    box_lines = [
        TextArea(f"Closest Approach (UTC): {Time(jd_ca, format='jd', scale='tdb').utc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}", textprops={"color":"#F5F5F5","fontsize":9.6}),
        TextArea(f"rho at CA: {rho_ca_arcsec:.6f} arcsec", textprops={"color":"#42D7C3","fontsize":9.6}),
        TextArea("Ecliptic Reference: 0.000°", textprops={"color":"#000000","fontsize":9.6}),
        TextArea(f"Earth Track From Ecliptic: {earth_track.positive_angle_deg:.6f}°", textprops={"color":"#3EA6FF","fontsize":9.6}),
        TextArea(f"Projected Venus Transit Track: {projected_relative_track.positive_angle_deg:.6f}°", textprops={"color":"#F5F5F5","fontsize":9.6}),
        TextArea(f"Venus Transit Track From Ecliptic: {physical_relative_track.positive_angle_deg:.6f}°", textprops={"color":"#38D66B","fontsize":9.6}),
    ]
    packed_box = VPacker(children=box_lines, align="left", pad=0.0, sep=2.0)
    angle_box = AnchoredOffsetbox(loc="center", child=packed_box, pad=0.45, frameon=True,
        bbox_to_anchor=(ca_x + offset_x, ca_y + offset_y), bbox_transform=ax.transData, borderpad=0.45)
    angle_box.patch.set_facecolor("#050505")
    angle_box.patch.set_edgecolor("#858585")
    angle_box.patch.set_alpha(0.94)
    ax.add_artist(angle_box)
    ax.plot([ca_x, ca_x + 0.62 * offset_x], [ca_y, ca_y + 0.62 * offset_y],
        color="#B0B0B0", linewidth=0.65, zorder=7)

'''
source=source[:ann_start]+new_ann+source[ann_end:]

source=source.replace("half_length = 0.88 * solar_radius_arcsec","half_length = 3.00 * solar_radius_arcsec")
source=source.replace("dpi=600","dpi=300")
source=source.replace('"venus_apparent_track_angle_deg": venus_track.positive_angle_deg,','"physical_relative_track_angle_deg": physical_relative_track.positive_angle_deg,')
source=source.replace('"venus_apparent_slope": venus_track.slope,','"physical_relative_slope": physical_relative_track.slope,')
source=source.replace('"venus_apparent_rms_arcsec": venus_track.rms_arcsec,','"physical_relative_rms_arcsec": physical_relative_track.rms_arcsec,')
source=source.replace('"venus_apparent_curvature_per_arcsec": venus_track.curvature_per_arcsec,','"physical_relative_curvature_per_arcsec": physical_relative_track.curvature_per_arcsec,')
source=source.replace('print(f"Venus orbit angle                    {result[\'venus_apparent_track_angle_deg\']:.6f} deg")','print(f"Venus Transit Track From Ecliptic     {result[\'physical_relative_track_angle_deg\']:.6f} deg")')
source=source.replace('print(f"Venus orbit slope                    {result[\'venus_apparent_slope\']:.9f}")','print(f"Venus Transit Ecliptic Slope          {result[\'physical_relative_slope\']:.9f}")')

if source.splitlines()[0]!="# V0152B" or source.splitlines()[-1]!="# V0152B":
    raise RuntimeError("REJECTED version boundary")
for marker in ["YEAR = 1769",'CENTER_UTC = "1769-06-03 22:00"',"rho_ca_arcsec","Closest Approach (UTC)","Ecliptic Reference","physical_relative_track"]:
    if marker not in source:
        raise RuntimeError(f"REJECTED final marker missing: {marker}")

compile(source,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152B.py","exec")
exec(compile(source,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152B.py","exec"))
# V0152B