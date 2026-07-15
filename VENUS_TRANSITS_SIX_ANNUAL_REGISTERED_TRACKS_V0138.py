# V0138
# Audit reference: restore original annual plot style, center CA, retain ±6 months, 300 DPI.
from __future__ import annotations
import base64,gzip,re,urllib.request

URL=("https://raw.githubusercontent.com/gear66me-ui/"
"IERS_TN36_V01_MASTER-1/main/"
"VENUS_TRANSITS_SIX_ANNUAL_REGISTERED_TRACKS_V0135.py")
with urllib.request.urlopen(URL,timeout=90) as r:
    delivery=r.read().decode("utf-8")
m=re.search(r'base64\.b64decode\("([A-Za-z0-9+/=]+)"\)',delivery)
if m is None:
    raise RuntimeError("REJECTED V0135 payload not found")
s=gzip.decompress(base64.b64decode(m.group(1))).decode("utf-8")
s=s.replace("# V0135","# V0138").replace("V0135","V0138").replace("dpi=600","dpi=300")
s=s.replace("import matplotlib.pyplot as plt\n","import matplotlib.pyplot as plt\nimport matplotlib.dates as mdates\n")

fn='''def project_to_tangent_plane(
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
ortho='''

def project_annual_registered(xyz_km, basis):
    x_axis, y_axis, _ = basis
    direction = xyz_km / np.linalg.norm(xyz_km, axis=1)[:, None]
    return (direction @ x_axis) * AS_PER_RAD, (direction @ y_axis) * AS_PER_RAD
'''
if fn not in s: raise RuntimeError("REJECTED projection function missing")
s=s.replace(fn,fn+ortho)

for a,b in {
"earth_x, earth_y = project_to_tangent_plane(annual_sun.xyz_km, projected_basis)":"earth_x, earth_y = project_annual_registered(annual_sun.xyz_km, projected_basis)",
"venus_x, venus_y = project_to_tangent_plane(annual_venus.xyz_km, projected_basis)":"venus_x, venus_y = project_annual_registered(annual_venus.xyz_km, projected_basis)",
"sun_ca_x, sun_ca_y = project_to_tangent_plane(sun_ca_xyz[None, :], projected_basis)":"sun_ca_x, sun_ca_y = project_annual_registered(sun_ca_xyz[None, :], projected_basis)",
"venus_ca_x, venus_ca_y = project_to_tangent_plane(venus_ca_xyz[None, :], projected_basis)":"venus_ca_x, venus_ca_y = project_annual_registered(venus_ca_xyz[None, :], projected_basis)",
}.items():
    if a not in s: raise RuntimeError("REJECTED annual call missing")
    s=s.replace(a,b)

old='''    origin_x = float(venus_ca_x[0])
    origin_y = float(venus_ca_y[0])

    earth_plot_x = (earth_x - origin_x) * VISUAL_SCALE
    earth_plot_y = (earth_y - origin_y) * VISUAL_SCALE
    venus_plot_x = (venus_x - origin_x) * VISUAL_SCALE
    venus_plot_y = (venus_y - origin_y) * VISUAL_SCALE

    days_from_ca = annual_sun.jd - jd_ca
'''
new='''    earth_plot_y = (earth_y - float(sun_ca_y[0])) * VISUAL_SCALE
    venus_plot_y = (venus_y - float(venus_ca_y[0])) * VISUAL_SCALE
    plot_dates = Time(annual_sun.jd, format="jd", scale="tdb").utc.to_datetime()
    ca_date = Time(jd_ca, format="jd", scale="tdb").utc.to_datetime()
'''
if old not in s: raise RuntimeError("REJECTED annual centering block missing")
s=s.replace(old,new)

s=s.replace('ax.plot(days_from_ca, venus_plot_y','ax.plot(plot_dates, venus_plot_y')
s=s.replace('ax.plot(days_from_ca, earth_plot_y','ax.plot(plot_dates, earth_plot_y')
s=s.replace('add_right_to_left_arrow(ax, days_from_ca, venus_plot_y','add_right_to_left_arrow(ax, mdates.date2num(plot_dates), venus_plot_y')
s=s.replace('add_right_to_left_arrow(ax, days_from_ca, earth_plot_y','add_right_to_left_arrow(ax, mdates.date2num(plot_dates), earth_plot_y')

a=s.index("    solar_disk = Circle(")
b=s.index("    ax.axvline",a)
center='''    ax.scatter([ca_date],[0.0],s=92,facecolor="#C98A18",edgecolor="#E64A19",
               linewidth=1.15,label="Solar limb",zorder=6)
    ax.scatter([ca_date],[0.0],s=18,facecolor="white",edgecolor="white",
               linewidth=0.5,label="Closest approach",zorder=7)
    annotation = "\\n".join([
        f"Earth orbit angle: {earth_track.positive_angle_deg:.6f}°",
        f"Venus orbit angle: {venus_track.positive_angle_deg:.6f}°",
        f"Projected relative angle: {projected_relative_track.positive_angle_deg:.6f}°",
    ])
    ax.annotate(annotation,xy=(ca_date,0.0),
        xytext=(ca_date + pd.Timedelta(days=28),0.18*y_span),
        color="#F0F0F0",fontsize=9.5,ha="left",va="bottom",
        arrowprops={"arrowstyle":"-","color":"#B0B0B0","linewidth":0.65},
        bbox={"boxstyle":"round,pad=0.32","facecolor":"#050505",
              "edgecolor":"#858585","alpha":0.94},zorder=9)

'''
s=s[:a]+center+s[b:]
s=s.replace('ax.axvline(0.0,','ax.axvline(ca_date,')
s=s.replace('ax.set_xlim(-ANNUAL_HALF_WINDOW_DAYS, ANNUAL_HALF_WINDOW_DAYS)',
'''ax.set_xlim(plot_dates[0], plot_dates[-1])
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))''')
s=s.replace('ax.set_xlabel("Days from closest approach (±6 months)", color="#E4E4E4")',
            'ax.set_xlabel(f"Calendar month — {year}", color="#E4E4E4")')
if s.splitlines()[0]!="# V0138" or s.splitlines()[-1]!="# V0138":
    raise RuntimeError("REJECTED version boundary")
if "dpi=600" in s or "dpi=300" not in s:
    raise RuntimeError("REJECTED DPI")
exec(compile(s,"VENUS_TRANSITS_SIX_ANNUAL_REGISTERED_TRACKS_V0138.py","exec"))
# V0138
