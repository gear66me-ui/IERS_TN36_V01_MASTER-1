# V0003
# Audit reference: Calendar-date black-and-white summary plate derived from the verified V0002 audit engine.
from __future__ import annotations

import importlib.util
import math
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


VERSION = "V0003"
PROGRAM = "IERS_REDUCTION_VS_JPL_VECTORS_V0003.py"
TITLE = "SOLAR HORIZONTAL PARALLAX — HISTORICAL REDUCTION AND JPL VECTOR AUDIT"
LOCAL_TZ = ZoneInfo("America/Bogota")
ROOT = Path("/content")
OUTPUT_DIR_DEFAULT = ROOT / "IERS_REDUCTION_VS_JPL_VECTORS_V0003_OUTPUT"
BASE_PATH = ROOT / "IERS_REDUCTION_VS_JPL_VECTORS_V0002_BASE.py"
BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "IERS_TN36_V01_MASTER-1/main/"
    "IERS_REDUCTION_VS_JPL_VECTORS_V0002.py"
)


def load_verified_engine():
    if not BASE_PATH.is_file():
        request = Request(BASE_URL, headers={"User-Agent": PROGRAM})
        with urlopen(request, timeout=120) as response:
            source = response.read()
        BASE_PATH.write_bytes(source)

    specification = importlib.util.spec_from_file_location(
        "iers_reduction_v0002_engine",
        BASE_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Unable to initialize the verified V0002 audit engine.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def build_summary_table(engine, reduction: dict[str, float]) -> pd.DataFrame:
    definitions = [
        (
            "IAU 1976",
            "Published astronomical unit",
            engine.IAU1976_RADIUS_M,
            engine.IAU1976_PUBLISHED_AU_M,
        ),
        (
            "IAU 1976",
            "Exact astronomical unit, A₂ = cτ_A",
            engine.IAU1976_RADIUS_M,
            reduction["exact_iau1976_au_m"],
        ),
        (
            "IAU 2012",
            "WGS84 equatorial radius",
            engine.WGS84_RADIUS_M,
            engine.IAU2012_AU_M,
        ),
        (
            "IERS 2010",
            "IAU 2012 exact astronomical unit",
            engine.IERS2010_RADIUS_M,
            engine.IAU2012_AU_M,
        ),
        (
            "JPL reduced",
            "Exact reduction to IAU 1976 Case 2",
            engine.IAU1976_RADIUS_M,
            reduction["exact_iau1976_au_m"],
        ),
    ]

    rows: list[dict[str, object]] = []
    for case_name, convention, radius_m, distance_m in definitions:
        _, _, parallax_arcsec = engine.solar_parallax(radius_m, distance_m)
        if case_name == "JPL reduced":
            parallax_arcsec = reduction["exact_arcsec"]

        difference_microarcsec = (
            parallax_arcsec - engine.TARGET_ARCSEC
        ) * 1_000_000.0
        classification = (
            "PASS"
            if abs(difference_microarcsec)
            <= engine.PASS_TOLERANCE_ARCSEC * 1_000_000.0
            else "FAIL"
        )
        rows.append(
            {
                "Case": case_name,
                "Convention": convention,
                "a (m)": radius_m,
                "A (m)": distance_m,
                "π⊙ (arcsec)": parallax_arcsec,
                "Δ (µas)": difference_microarcsec,
                "Class": classification,
            }
        )
    return pd.DataFrame(rows)


def save_calendar_vector_files(
    engine,
    geometry: dict[str, object],
    output_dir: Path,
) -> dict[str, Path]:
    frame = geometry["frame"].copy()
    frame.insert(
        0,
        "Calendar UTC",
        frame["JD"].map(engine.julian_date_to_utc_text),
    )

    paths = {
        "sun": output_dir / "JPL_1769_GEOCENTER_SUN_VECTORS_V0003.csv",
        "venus": output_dir / "JPL_1769_GEOCENTER_VENUS_VECTORS_V0003.csv",
        "master": output_dir / "JPL_1769_GEOCENTER_MASTER_V0003.csv",
    }
    frame[
        ["Calendar UTC", *engine.VECTOR_COLUMNS[1:4]]
    ].to_csv(
        paths["sun"],
        index=False,
        float_format="%.15f",
    )
    frame[
        ["Calendar UTC", *engine.VECTOR_COLUMNS[4:7]]
    ].to_csv(
        paths["venus"],
        index=False,
        float_format="%.15f",
    )
    frame[
        ["Calendar UTC", *engine.VECTOR_COLUMNS[1:]]
    ].to_csv(
        paths["master"],
        index=False,
        float_format="%.15f",
    )
    return paths


def add_black_panel(axis, title: str) -> None:
    axis.set_axis_off()
    axis.add_patch(
        FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            transform=axis.transAxes,
            linewidth=0.8,
            edgecolor="#FFFFFF",
            facecolor="#000000",
            clip_on=False,
            zorder=-10,
        )
    )
    axis.text(
        0.025,
        0.94,
        title,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color="#FFFFFF",
    )


def render_publication_plate(
    engine,
    summary: pd.DataFrame,
    geometry: dict[str, object],
    reduction: dict[str, float],
    uncertainty: tuple[float, float, float],
    jpl_source: str,
    vector_paths: dict[str, Path],
    output_png: Path,
    dpi: int,
) -> None:
    plt.close("all")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "figure.facecolor": "#000000",
            "savefig.facecolor": "#000000",
            "text.color": "#FFFFFF",
            "axes.edgecolor": "#FFFFFF",
            "axes.labelcolor": "#FFFFFF",
            "xtick.color": "#FFFFFF",
            "ytick.color": "#FFFFFF",
        }
    )

    figure = plt.figure(figsize=(18.0, 11.5), facecolor="#000000")
    grid = figure.add_gridspec(
        4,
        2,
        height_ratios=(0.18, 0.82, 1.02, 0.88),
        hspace=0.17,
        wspace=0.06,
        left=0.035,
        right=0.965,
        top=0.965,
        bottom=0.05,
    )

    title_axis = figure.add_subplot(grid[0, :])
    title_axis.set_axis_off()
    title_axis.text(
        0.5,
        0.75,
        TITLE,
        ha="center",
        va="center",
        fontsize=19.0,
        fontweight="bold",
        color="#FFFFFF",
    )
    title_axis.text(
        0.5,
        0.24,
        (
            r"1769 Venus Transit  •  exact $c\tau_A$ normalization  •  "
            r"geocentric JPL Horizons vectors"
        ),
        ha="center",
        va="center",
        fontsize=11.0,
        color="#D1D5DB",
    )

    reduction_axis = figure.add_subplot(grid[1, 0])
    vector_axis = figure.add_subplot(grid[1, 1])
    table_axis = figure.add_subplot(grid[2, :])
    audit_axis = figure.add_subplot(grid[3, :])

    add_black_panel(reduction_axis, "I. HISTORICAL REDUCTION — SUMMARY")
    equations = [
        rf"$F_D=D_{{\mathrm{{JPL}}}}/A_2={reduction['distance_factor']:.15f}$",
        rf"$F_R=a_{{1976}}/a_{{WGS84}}={reduction['radius_factor']:.15f}$",
        rf"$F_2=F_DF_R={reduction['total_factor']:.15f}$",
        (
            rf"$\pi_{{2,\mathrm{{exact}}}}="
            rf"\arcsin\!\left[F_2\sin\!\left("
            rf"\pi_{{\odot,\mathrm{{JPL}}}}\right)\right]$"
        ),
        rf"$\pi_{{2,\mathrm{{exact}}}}={reduction['exact_arcsec']:.12f}^{{\prime\prime}}$",
        rf"$\Delta_{{\mathrm{{exact-Case\ 2}}}}={reduction['exact_minus_case_2_microarcsec']:.9f}\ \mu\mathrm{{as}}$",
    ]
    y_position = 0.78
    for index, equation in enumerate(equations):
        reduction_axis.text(
            0.05,
            y_position,
            equation,
            transform=reduction_axis.transAxes,
            fontsize=13.0 if index in (3, 4) else 11.8,
            color="#FFFFFF",
        )
        y_position -= 0.125

    reduction_axis.text(
        0.05,
        0.06,
        (
            "The instantaneous JPL geometry is standardized by both the\n"
            "Earth–Sun distance factor and the radius-convention factor."
        ),
        transform=reduction_axis.transAxes,
        fontsize=8.8,
        color="#D1D5DB",
        linespacing=1.35,
    )

    add_black_panel(vector_axis, "II. JPL VECTOR AUDIT — CALENDAR UTC")
    sun_vector = geometry["sun_vector_km"]
    venus_vector = geometry["venus_vector_km"]
    vector_rows = [
        ("Source", jpl_source, ""),
        ("Calendar date", str(geometry["epoch_utc"]), ""),
        (r"$X_\odot$", f"{sun_vector[0]:,.6f}", "km"),
        (r"$Y_\odot$", f"{sun_vector[1]:,.6f}", "km"),
        (r"$Z_\odot$", f"{sun_vector[2]:,.6f}", "km"),
        (
            r"$\|\mathbf{r}_{E\odot}\|$",
            f"{float(geometry['distance_m']) / 1000.0:,.6f}",
            "km",
        ),
        (r"$X_\mathrm{V}$", f"{venus_vector[0]:,.6f}", "km"),
        (r"$Y_\mathrm{V}$", f"{venus_vector[1]:,.6f}", "km"),
        (r"$Z_\mathrm{V}$", f"{venus_vector[2]:,.6f}", "km"),
        (
            r"$\theta_{\odot\mathrm{V}}$",
            f"{float(geometry['separation_arcsec']):.9f}",
            "arcsec",
        ),
    ]
    y_position = 0.80
    for label, value, unit in vector_rows:
        vector_axis.text(
            0.05,
            y_position,
            label,
            transform=vector_axis.transAxes,
            fontsize=9.0,
            color="#FFFFFF",
        )
        vector_axis.text(
            0.29,
            y_position,
            value,
            transform=vector_axis.transAxes,
            fontsize=8.6,
            family="monospace",
            color="#FFFFFF",
        )
        vector_axis.text(
            0.87,
            y_position,
            unit,
            transform=vector_axis.transAxes,
            fontsize=8.4,
            color="#D1D5DB",
        )
        y_position -= 0.071

    add_black_panel(table_axis, "III. PARALLAX SUMMARY")
    display_table = summary.copy()
    display_table["a (m)"] = display_table["a (m)"].map(
        lambda value: f"{float(value):,.3f}"
    )
    display_table["A (m)"] = display_table["A (m)"].map(
        lambda value: f"{float(value):,.3f}"
    )
    display_table["π⊙ (arcsec)"] = display_table["π⊙ (arcsec)"].map(
        lambda value: f"{float(value):.12f}"
    )
    display_table["Δ (µas)"] = display_table["Δ (µas)"].map(
        lambda value: f"{float(value):+.6f}"
    )

    publication_table = table_axis.table(
        cellText=display_table.values,
        colLabels=display_table.columns,
        cellLoc="left",
        colLoc="center",
        bbox=(0.02, 0.10, 0.96, 0.75),
        colWidths=(0.09, 0.27, 0.12, 0.19, 0.14, 0.10, 0.07),
    )
    publication_table.set_zorder(5)
    publication_table.auto_set_font_size(False)
    publication_table.set_fontsize(8.3)

    for (row, _column), cell in publication_table.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor("#FFFFFF")
        cell.set_facecolor(
            "#000000"
            if row == 0
            else ("#111111" if row % 2 else "#202020")
        )
        cell.set_text_props(
            color="#FFFFFF",
            weight="bold" if row == 0 else "normal",
        )

    add_black_panel(
        audit_axis,
        "IV. UNCERTAINTY, TRACEABILITY, AND GENERATED FILES",
    )
    radius_sigma, au_sigma, combined_sigma = uncertainty
    audit_axis.text(
        0.04,
        0.76,
        rf"$\sigma_{{\pi,a}}={radius_sigma * 1_000_000.0:.6f}\ \mu\mathrm{{as}}$",
        transform=audit_axis.transAxes,
        fontsize=11.5,
        color="#FFFFFF",
    )
    audit_axis.text(
        0.04,
        0.58,
        rf"$\sigma_{{\pi,A}}={au_sigma * 1_000_000.0:.6f}\ \mu\mathrm{{as}}$",
        transform=audit_axis.transAxes,
        fontsize=11.5,
        color="#FFFFFF",
    )
    audit_axis.text(
        0.04,
        0.40,
        (
            rf"$\sigma_\pi={combined_sigma:.12f}^{{\prime\prime}}"
            rf"\rightarrow\pm {engine.PASS_TOLERANCE_ARCSEC:.6f}^{{\prime\prime}}$"
        ),
        transform=audit_axis.transAxes,
        fontsize=11.5,
        color="#FFFFFF",
    )
    audit_axis.text(
        0.04,
        0.15,
        (
            "Displayed dates use the Gregorian calendar in UTC. Every numerical\n"
            "result derives from defining constants or minute-by-minute JPL vectors."
        ),
        transform=audit_axis.transAxes,
        fontsize=9.0,
        color="#D1D5DB",
        linespacing=1.35,
    )

    audit_axis.text(
        0.53,
        0.76,
        "Generated JPL vector files — calendar UTC only",
        transform=audit_axis.transAxes,
        fontsize=10.0,
        fontweight="bold",
        color="#FFFFFF",
    )
    for y_position, key in zip(
        (0.58, 0.43, 0.28),
        ("sun", "venus", "master"),
    ):
        audit_axis.text(
            0.53,
            y_position,
            vector_paths[key].name,
            transform=audit_axis.transAxes,
            fontsize=8.7,
            family="monospace",
            color="#FFFFFF",
        )

    figure.text(
        0.5,
        0.017,
        (
            "Figure V0003. Calendar-date black-and-white summary of the "
            "historical solar-parallax reduction and geocentric JPL vector audit."
        ),
        ha="center",
        fontsize=8.5,
        color="#D1D5DB",
    )
    figure.savefig(
        output_png,
        dpi=max(240, int(dpi)),
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="#000000",
    )
    plt.close(figure)


def main() -> None:
    engine = load_verified_engine()
    arguments = engine.parse_arguments()
    radius_mode, raw_radius_m = engine.selected_radius(arguments)

    output_dir = (
        Path(arguments.output_dir).expanduser().resolve()
        if arguments.output_dir
        else OUTPUT_DIR_DEFAULT
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    master_path, jpl_source = engine.locate_jpl_master(arguments.jpl_csv)
    geometry = engine.derive_jpl_geometry(master_path)
    reduction = engine.derive_reduction(geometry, raw_radius_m)
    uncertainty = engine.historical_uncertainty()
    summary = build_summary_table(engine, reduction)
    vector_paths = save_calendar_vector_files(
        engine,
        geometry,
        output_dir,
    )

    summary_csv = (
        output_dir
        / "IERS_REDUCTION_VS_JPL_VECTORS_V0003_SUMMARY.csv"
    )
    reduction_csv = (
        output_dir
        / "IERS_REDUCTION_VS_JPL_VECTORS_V0003_REDUCTION.csv"
    )
    publication_png = (
        output_dir
        / "SOLAR_PARALLAX_HISTORICAL_REDUCTION_JPL_AUDIT_V0003.png"
    )

    summary.to_csv(
        summary_csv,
        index=False,
        float_format="%.15f",
    )
    pd.DataFrame([reduction]).to_csv(
        reduction_csv,
        index=False,
        float_format="%.15f",
    )
    render_publication_plate(
        engine,
        summary,
        geometry,
        reduction,
        uncertainty,
        jpl_source,
        vector_paths,
        publication_png,
        arguments.dpi,
    )

    generated_headers = {
        key: list(pd.read_csv(path, nrows=0).columns)
        for key, path in vector_paths.items()
    }
    checks = {
        "JPL vector magnitude": abs(
            float(geometry["distance_m"])
            - float(np.linalg.norm(geometry["sun_vector_km"])) * 1000.0
        ) <= 0.000001,
        "Exact JPL reduction equals Case 2": abs(
            reduction["exact_minus_case_2_microarcsec"]
        ) <= 0.000001,
        "Case 2 rounds to 8.794148": (
            round(reduction["case_2_arcsec"], 6)
            == engine.TARGET_ARCSEC
        ),
        "Historical uncertainty rounds to ±0.000007": (
            round(uncertainty[2], 6)
            == engine.PASS_TOLERANCE_ARCSEC
        ),
        "Raw JPL row omitted": (
            "JPL raw" not in summary["Case"].tolist()
        ),
        "Generated vectors omit Julian date": all(
            "JD" not in columns
            for columns in generated_headers.values()
        ),
        "Publication image generated": publication_png.is_file(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Equation checks failed: " + ", ".join(failed))

    try:
        from IPython.display import Image, display
        display(Image(filename=str(publication_png)))
    except Exception:
        print(f"PUBLICATION IMAGE: {publication_png}")

    print(f"CODE OUTPUT: {VERSION}")
    print("CODE INPUTS")
    print(f"Program: {PROGRAM}")
    print(f"JPL source: {jpl_source}")
    print(f"JPL master: {master_path}")
    print("COMMENTS")
    print(
        "Black-and-white Matplotlib publication plate; Gregorian calendar UTC; "
        "no raw-parallax table row; no AI imagery."
    )
    print("RESULTS")
    print(f"Calendar closest approach: {geometry['epoch_utc']}")
    print(
        "Exact reduced parallax: "
        f"{reduction['exact_arcsec']:.12f} arcsec"
    )
    print("OUTPUT SUMMARY")
    print(f"Publication image: {publication_png}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Reduction CSV: {reduction_csv}")
    print(f"JPL Sun vectors: {vector_paths['sun']}")
    print(f"JPL Venus vectors: {vector_paths['venus']}")
    print(f"JPL combined master: {vector_paths['master']}")
    print("PAPER COMPARISON")
    print(
        "IAU-1976 Case 2: "
        f"{reduction['case_2_arcsec']:.12f} arcsec"
    )
    print("EQUATION STATUS")
    print("All checks: PASS")
    print(
        "LOCAL TIMESTAMP: "
        f"{datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}"
    )
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# V0003
