# V0023
# Audit reference: Reliable source-patch delivery for the fresh JPL half-Sun plot with embedded summary table and decluttered Vardo/Tahiti labels.
from __future__ import annotations

import py_compile
import re
import runpy
from pathlib import Path
from urllib.request import Request, urlopen

VERSION = "V0023"
ROOT = Path("/content")
SOURCE = ROOT / "IERS_0012N_HALF_SUN_JPL_PLOT_ONLY_SOURCE_V0023.py"
ENGINE = ROOT / "IERS_0012N_HALF_SUN_TABLE_LABELS_ENGINE_V0023.py"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "IERS_0012N_HALF_SUN_JPL_PLOT_ONLY_V0021.py?v=23"
)


def download_source() -> str:
    request = Request(SOURCE_URL, headers={"User-Agent": "IERS-V0023"})
    with urlopen(request, timeout=180) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError("V0023 source download was empty.")
    SOURCE.write_bytes(payload)
    return payload.decode("utf-8")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"V0023 patch failed for {label}: expected 1 match, found {count}.")
    return text.replace(old, new, 1)


def regex_once(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL | re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"V0023 regex patch failed for {label}: expected 1 match, found {count}.")
    return updated


def build_engine(source: str) -> str:
    text = source
    text = replace_once(text, "# V0021", "# V0023", "opening version marker")
    text = replace_once(text, 'VERSION = "V0021"', 'VERSION = "V0023"', "version constant")
    text = replace_once(
        text,
        'VENUS_RADIUS_KM = 6_051.8',
        'VENUS_RADIUS_KM = 6_051.8\nEARTH_RADIUS_KM = 6_378.140',
        "Earth radius constant",
    )
    text = replace_once(
        text,
        'OUTPUT_DIR = Path("/content/IERS_TN36_V01_MASTER_OUTPUT/V0021_HALF_SUN_PLOT_ONLY")\nPNG = OUTPUT_DIR / "V0021_VARDO_TAHITI_IAU1976_HALF_SUN.png"',
        'OUTPUT_DIR = Path("/content/IERS_TN36_V01_MASTER_OUTPUT/V0023_HALF_SUN_TABLE_LABELS")\nPNG = OUTPUT_DIR / "V0023_VARDO_TAHITI_IAU1976_HALF_SUN_TABLE.png"',
        "output paths",
    )

    text = regex_once(
        text,
        r"def merge_vectors\(\) -> tuple\[pd\.DataFrame, pd\.DataFrame\]:\n.*?\n    return geocenter, topocentric\n",
        '''def merge_vectors() -> tuple[pd.DataFrame, pd.DataFrame]:
    geocenter_sun = get_vectors("10", "500@399", "GEO_SUN")
    geocenter_venus = get_vectors("299", "500@399", "GEO_VENUS")
    geocenter = geocenter_sun.merge(geocenter_venus, on="jd", how="inner")
    topocentric: pd.DataFrame | None = None
    for site in SITES:
        location = site_location(site)
        for target, suffix in (("10", "SUN"), ("299", "VENUS")):
            frame = get_vectors(target, location, f"{site['key']}_{suffix}")
            topocentric = frame if topocentric is None else topocentric.merge(frame, on="jd", how="inner")
    if topocentric is None or len(geocenter) < 500 or len(topocentric) < 500:
        raise RuntimeError("Incomplete JPL vector retrieval.")
    return geocenter, topocentric
''',
        "geocenter Sun and Venus retrieval",
    )

    helper_block = '''def track_fit(track: dict[str, object]) -> tuple[np.ndarray, np.ndarray, float]:
    points = np.asarray(track["points"], dtype=float)
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    direction = unit(vt[0])
    if direction[0] < 0.0:
        direction = -direction
    angle = math.degrees(math.atan2(direction[1], direction[0]))
    return center, direction, angle


def line_intersection(mu: np.ndarray, direction: np.ndarray, midpoint: np.ndarray, normal: np.ndarray) -> np.ndarray:
    solution, *_ = np.linalg.lstsq(np.column_stack((direction, -normal)), midpoint - mu, rcond=None)
    return mu + float(solution[0]) * direction


def compute_summary(
    geo: dict[str, object],
    track_vardo: dict[str, object],
    track_tahiti: dict[str, object],
    reference_jd: float,
) -> dict[str, float]:
    mu_v, direction_v, beta_v = track_fit(track_vardo)
    mu_t, direction_t, beta_t = track_fit(track_tahiti)
    tangent = unit(direction_v + direction_t)
    if tangent[0] < 0.0:
        tangent = -tangent
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    midpoint = 0.5 * (mu_v + mu_t)
    aprime = line_intersection(mu_v, direction_v, midpoint, normal)
    bprime = line_intersection(mu_t, direction_t, midpoint, normal)
    chord = bprime - aprime
    abp_arcsec = float(np.linalg.norm(chord))
    rho_arcsec = abs(float(np.dot(chord, normal)))

    earth_sun = vector_at(geo, "GEO_SUN", reference_jd)
    earth_venus = vector_at(geo, "GEO_VENUS", reference_jd)
    venus_sun = earth_venus - earth_sun
    ev_over_vs = norm(earth_venus) / norm(venus_sun)

    theta = abp_arcsec / ARCSEC_PER_RAD
    abp_km = math.tan(theta) * IAU1976_AU_KM
    ab_km = abp_km * ev_over_vs
    ab_arcsec = math.atan2(ab_km, IAU1976_AU_KM) * ARCSEC_PER_RAD
    raw_phi = rho_arcsec * ev_over_vs * EARTH_RADIUS_KM / ab_km
    pi_sun = (
        raw_phi
        * (abp_arcsec / rho_arcsec)
        * (math.tan(theta) / theta)
        * (math.asin(EARTH_RADIUS_KM / IAU1976_AU_KM) / (EARTH_RADIUS_KM / IAU1976_AU_KM))
    )
    return {
        "beta_vardo": beta_v,
        "beta_tahiti": beta_t,
        "delta_beta": abs(beta_v - beta_t),
        "pi_sun": pi_sun,
        "halley_ratio": abp_km / ab_km,
        "abp_arcsec": abp_arcsec,
        "abp_km": abp_km,
        "ab_arcsec": ab_arcsec,
        "ab_km": ab_km,
        "des_au": 1.0,
    }


def summary_table(axes, summary: dict[str, float]) -> None:
    rows = [
        ("β Vardo", f"{summary['beta_vardo']:.6f}", "deg"),
        ("β Point Venus", f"{summary['beta_tahiti']:.6f}", "deg"),
        ("Δβ", f"{summary['delta_beta']:.6f}", "deg"),
        ("π⊙", f"{summary['pi_sun']:.10f}", "arcsec"),
        ("A′B′ / AB", f"{summary['halley_ratio']:.10f}", "ratio"),
        ("A′B′", f"{summary['abp_arcsec']:.6f}", "arcsec"),
        ("A′B′", f"{summary['abp_km']:.6f}", "km"),
        ("AB", f"{summary['ab_arcsec']:.6f}", "arcsec"),
        ("AB", f"{summary['ab_km']:.6f}", "km"),
        ("D ES", f"{summary['des_au']:.12f}", "AU"),
    ]
    table = axes.table(
        cellText=rows,
        colLabels=["Quantity", "Value", "Unit"],
        loc="lower left",
        colWidths=[0.29, 0.23, 0.15],
        bbox=[0.438, 0.122, 0.380, 0.345],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.30)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0.18)
        cell.set_edgecolor("#1e4f64")
        cell.set_facecolor("#0a1a22" if row == 0 else "#050b0f")
        cell.get_text().set_color(
            "#66e8ff" if row == 0 else ("#ffc861" if column == 1 else ("#5ee08a" if column == 2 else "#dff8ff"))
        )
        if row == 0 or column == 1:
            cell.get_text().set_fontweight("bold")
    axes.text(
        0.440,
        0.101,
        "A′B′ = solar-screen chord; AB = projected baseline; D ES = IAU 1976 cτA.",
        transform=axes.transAxes,
        color="#8fb4c1",
        fontsize=5.25,
        ha="left",
        va="top",
    )


'''
    text = replace_once(text, "def label_event(", helper_block + "def label_event(", "summary helper functions")

    text = regex_once(
        text,
        r"def make_plot\(track_a: dict\[str, object\], track_b: dict\[str, object\]\) -> None:\n.*?\n\ndef utc\(jd: float\) -> str:",
        '''def make_plot(
    track_a: dict[str, object],
    track_b: dict[str, object],
    summary: dict[str, float],
) -> None:
    solar_radius_arcsec = math.atan2(SUN_RADIUS_KM, IAU1976_AU_KM) * ARCSEC_PER_RAD
    figure, axes = plt.subplots(figsize=(9.6, 5.8), dpi=240)
    figure.patch.set_facecolor("#03080d")
    axes.set_facecolor("#03080d")

    solar_limb = Circle((0.0, 0.0), solar_radius_arcsec, fill=False, lw=0.36, ec="#66e8ff", alpha=0.95)
    axes.add_patch(solar_limb)
    axes.axhline(0.0, lw=0.18, color="#1d3d4a")
    axes.axvline(0.0, lw=0.18, color="#1d3d4a")

    plotted_events = 0
    for track in (track_a, track_b):
        site = track["site"]
        color = str(site["color"])
        points = np.asarray(track["points"], dtype=float)
        axes.plot(points[:, 0], points[:, 1], lw=0.30, color=color, label=str(site["label"]), zorder=3)
        axes.scatter(points[::6, 0], points[::6, 1], s=0.8, color=color, linewidths=0, zorder=4)
        for event in EVENTS:
            point = np.asarray(track["event_points"][event], dtype=float)
            radius = float(track["event_radii"][event])
            axes.add_patch(Circle((point[0], point[1]), radius, fill=False, lw=0.28 if event == "CA" else 0.20, ec=color, alpha=0.95, zorder=5))
            axes.scatter([point[0]], [point[1]], s=3.5, color=color, linewidths=0, zorder=6)
            plotted_events += 1

        if site["key"] == "VARDO":
            label_event(axes, track["event_points"]["C1"], "V C1", color, 18, -46)
            label_event(axes, track["event_points"]["C2"], "V C2", color, -72, -36)
            label_event(axes, track["event_points"]["CA"], "Vardo CA", color, 18, -54)
            label_event(axes, track["event_points"]["C3"], "V C3", color, 24, -38)
            label_event(axes, track["event_points"]["C4"], "V C4", color, -64, -48)
        else:
            label_event(axes, track["event_points"]["C1"], "T C1", color, 18, 46)
            label_event(axes, track["event_points"]["C2"], "T C2", color, -72, 36)
            label_event(axes, track["event_points"]["CA"], "Tahiti CA", color, 18, 54)
            label_event(axes, track["event_points"]["C3"], "T C3", color, 24, 40)
            label_event(axes, track["event_points"]["C4"], "T C4", color, -64, 48)

    if plotted_events != 10:
        raise RuntimeError(f"Expected 10 plotted event disks; found {plotted_events}.")

    summary_table(axes, summary)
    combined = np.vstack((track_a["points"], track_b["points"]))
    median_y = float(np.median(combined[:, 1]))
    axes.set_xlim(-1.04 * solar_radius_arcsec, 1.04 * solar_radius_arcsec)
    axes.set_ylim((-0.06 * solar_radius_arcsec, 1.06 * solar_radius_arcsec) if median_y >= 0.0 else (-1.06 * solar_radius_arcsec, 0.06 * solar_radius_arcsec))
    axes.set_aspect("equal", adjustable="box")
    axes.grid(True, color="#102630", linewidth=0.16, alpha=0.55)
    axes.tick_params(colors="#8fb4c1", labelsize=6.5, width=0.22, length=2)
    axes.set_xlabel("IAU-1976-normalized solar-screen X offset (arcsec)", color="#8fb4c1", fontsize=7.5)
    axes.set_ylabel("IAU-1976-normalized solar-screen Y offset (arcsec)", color="#8fb4c1", fontsize=7.5)
    axes.set_title("1769 Venus Transit — Engineering Half-Sun Track Reconstruction\nVardo, Norway / Point Venus, Tahiti — JPL Horizons SITE_COORD geometry", color="#f8fdff", fontsize=9, pad=8)
    legend = axes.legend(loc="lower right", fontsize=6.3, frameon=True)
    legend.get_frame().set_facecolor("#071016")
    legend.get_frame().set_edgecolor("#1e4f64")
    for item in legend.get_texts():
        item.set_color("#dff8ff")
    figure.text(0.5, 0.016, "Venus disks are plotted to scale at C1, C2, closest approach, C3, and C4. Tahiti labels above; Vardo labels below.", ha="center", fontsize=6.2, color="#8fb4c1")
    figure.savefig(PNG, dpi=460, facecolor=figure.get_facecolor(), bbox_inches="tight", pad_inches=0.055)
    plt.show()
    plt.close(figure)
    if not PNG.is_file() or PNG.stat().st_size == 0:
        raise RuntimeError("PNG plot was not generated.")
    display(Image(filename=str(PNG)))


def utc(jd: float) -> str:''',
        "plot, table, and label layout",
    )

    text = replace_once(
        text,
        '    track_vardo = build_track(geo, topo, VARDO, events_vardo, frame)\n    track_tahiti = build_track(geo, topo, TAHITI, events_tahiti, frame)\n    make_plot(track_vardo, track_tahiti)',
        '    track_vardo = build_track(geo, topo, VARDO, events_vardo, frame)\n    track_tahiti = build_track(geo, topo, TAHITI, events_tahiti, frame)\n    summary = compute_summary(geo, track_vardo, track_tahiti, reference_jd)\n    make_plot(track_vardo, track_tahiti, summary)',
        "main summary computation",
    )
    text = replace_once(text, "# V0021", "# V0023", "closing version marker")
    return text


def main() -> None:
    source = download_source()
    engine = build_engine(source)
    ENGINE.write_text(engine, encoding="utf-8")
    py_compile.compile(str(ENGINE), doraise=True)
    runpy.run_path(str(ENGINE), run_name="__main__")


if __name__ == "__main__":
    main()
# V0023
