# V0152H
# Audit reference: preserve the V0152B reference plot exactly; replace only the rho circle with a red +/-2-hour rho(t) parabola tangent to the white projected Venus track at the locked closest-approach vertex.
from __future__ import annotations
import urllib.request

SOURCE_URL=("https://raw.githubusercontent.com/gear66me-ui/"
"IERS_TN36_V01_MASTER-1/main/"
"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152B.py")
with urllib.request.urlopen(SOURCE_URL,timeout=90) as response:
    source=response.read().decode("utf-8")

required=[
    "# V0152B",
    'VERSION = "V0152B"',
    'jd_ca, minimum_separation_rad, sun_at_ca = closest_approach(sun, venus)',
    'Projected Venus Transit Track',
    'Earth Track From Ecliptic',
    'Venus Transit Track From Ecliptic',
    'rho_ca_arcsec',
    'ax.add_patch(Circle(',
]
for marker in required:
    if marker not in source:
        raise RuntimeError(f"REJECTED missing V0152B marker: {marker}")

source=source.replace("# V0152B","# V0152H")
source=source.replace('VERSION = "V0152B"','VERSION = "V0152H"')
source=source.replace("VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152B_OUTPUT","VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152H_OUTPUT")
source=source.replace("VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152B.png","VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152H.png")
source=source.replace("VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152B.csv","VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152H.csv")

old_ca='''    jd_ca, minimum_separation_rad, sun_at_ca = closest_approach(sun, venus)'''
new_ca='''    jd_ca = Time("1769-06-03 22:19:04.388", scale="utc").tdb.jd
    sun_at_ca = evaluate(cubic_splines(sun), jd_ca)
    venus_at_ca_locked = evaluate(cubic_splines(venus), jd_ca)
    minimum_separation_rad = math.atan2(
        float(np.linalg.norm(np.cross(unit(sun_at_ca), unit(venus_at_ca_locked)))),
        float(np.dot(unit(sun_at_ca), unit(venus_at_ca_locked))),
    )'''
if old_ca not in source:
    raise RuntimeError("REJECTED closest-approach replacement marker missing")
source=source.replace(old_ca,new_ca,1)

old_circle='''    ax.add_patch(Circle(
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

'''
new_parabola='''    rho_minutes = np.linspace(-120.0, 120.0, 241)
    sun_exact_curves = cubic_splines(sun)
    venus_exact_curves = cubic_splines(venus)
    rho_values = []
    for minute_value in rho_minutes:
        jd_value = jd_ca + float(minute_value) / 1440.0
        sun_value = evaluate(sun_exact_curves, jd_value)
        venus_value = evaluate(venus_exact_curves, jd_value)
        rho_values.append(math.atan2(
            float(np.linalg.norm(np.cross(unit(sun_value), unit(venus_value)))),
            float(np.dot(unit(sun_value), unit(venus_value))),
        ) * AS_PER_RAD)
    rho_values = np.asarray(rho_values, dtype=float)
    rho_excess = rho_values - rho_ca_arcsec
    white_angle_rad = math.radians(projected_relative_track.signed_angle_deg)
    tangent_unit = np.array([math.cos(white_angle_rad), math.sin(white_angle_rad)], dtype=float)
    normal_unit = np.array([-tangent_unit[1], tangent_unit[0]], dtype=float)
    relative_x_curve = CubicSpline(sun.jd, relative_x_projected, bc_type="natural")
    relative_y_curve = CubicSpline(sun.jd, relative_y_projected, bc_type="natural")
    velocity_x = float(relative_x_curve.derivative()(jd_ca)) / 1440.0
    velocity_y = float(relative_y_curve.derivative()(jd_ca)) / 1440.0
    speed_arcsec_per_min = float(math.hypot(velocity_x, velocity_y))
    along_track = speed_arcsec_per_min * rho_minutes
    parabola_xy = (
        np.array([ca_x, ca_y], dtype=float)
        + along_track[:, None] * tangent_unit[None, :]
        + rho_excess[:, None] * normal_unit[None, :]
    )
    ax.plot(
        parabola_xy[:, 0], parabola_xy[:, 1],
        color="#FF3B30", linewidth=1.35, zorder=7,
        label="rho(t) parabola +/-2 h",
    )
    ax.scatter(parabola_xy[::12, 0], parabola_xy[::12, 1], s=6,
        color="#FF3B30", edgecolors="none", zorder=8)
    ax.annotate(
        "rho(t) minimum\\n22:19:04.388",
        xy=(ca_x, ca_y), xytext=(ca_x-210.0, ca_y+120.0),
        color="#FF3B30", fontsize=8.0,
        arrowprops={"arrowstyle":"-","color":"#FF3B30","linewidth":0.65},
        zorder=10,
    )

'''
if old_circle not in source:
    raise RuntimeError("REJECTED rho-circle block missing")
source=source.replace(old_circle,new_parabola,1)

source=source.replace(
    'TextArea(f"rho at CA: {rho_ca_arcsec:.6f} arcsec", textprops={"color":"#42D7C3","fontsize":9.6}),',
    'TextArea(f"rho at CA: {rho_ca_arcsec:.6f} arcsec", textprops={"color":"#FF3B30","fontsize":9.6}),\n        TextArea("rho(t) parabola: +/-120.000 min", textprops={"color":"#FF3B30","fontsize":9.6}),',
    1,
)

if source.splitlines()[0]!="# V0152H" or source.splitlines()[-1]!="# V0152H":
    raise RuntimeError("REJECTED version boundary")
for marker in [
    '1769-06-03 22:19:04.388',
    'rho(t) parabola +/-2 h',
    'Projected Venus Transit Track',
    'Earth Track From Ecliptic',
    'Venus Transit Track From Ecliptic',
    'projected_relative_track.signed_angle_deg',
]:
    if marker not in source:
        raise RuntimeError(f"REJECTED final marker missing: {marker}")

compile(source,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152H_COMPILED.py","exec")
exec(compile(source,"VENUS_TRANSIT_1769_ECLIPTIC_CROSSHAIR_V0152H.py","exec"))
# V0152H