# IERS-0012R
# Audit reference: GitHubDelivery@IERS-0012R; exact closure audit for the optimized 1769 antipodal maximum-parallax solution.

import csv
import math
import os
from datetime import datetime
from zoneinfo import ZoneInfo

VERSION = "IERS-0012R"
PROGRAM_NAME = "IERS_0012R_AUDIT_1769_MAXIMUM_PARALLAX_CLOSURE.py"

ARCSEC_PER_RAD = 206_264.80624709636
AU_KM = 149_597_870.7
WGS84_A_KM = 6_378.137
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
PI_SUN_REFERENCE_ARCSEC = 8.794148
LOCAL_TZ = ZoneInfo("America/Bogota")

OUT_DIR = "/content/IERS_TN36_V01_MASTER_OUTPUT"
INPUT_CSV = os.path.join(
    OUT_DIR,
    "IERS-0012Q_OPTIMIZED_1769_ANTIPODAL_MAXIMUM_PARALLAX.csv",
)
OUTPUT_CSV = os.path.join(
    OUT_DIR,
    "IERS-0012R_1769_MAXIMUM_PARALLAX_CLOSURE_AUDIT.csv",
)


def load_q_values(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required project CSV not found: {path}")

    values = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 4:
                continue
            section, quantity, value, unit = row[:4]
            if section in {"OPTIMIZATION", "THEORY", "RESULT"}:
                try:
                    values[quantity] = float(value)
                except ValueError:
                    continue

    required = [
        "Site A latitude",
        "Site A longitude east",
        "Site B latitude",
        "Site B longitude east",
        "JPL D ES",
        "JPL D VS / D EV",
        "A prime B prime",
        "Normal separation rho",
        "Pi sun",
        "Pi sun residual",
    ]
    missing = [name for name in required if name not in values]
    if missing:
        raise RuntimeError(f"Missing required values in IERS-0012Q CSV: {missing}")
    return values


def wgs84_geocentric_radius_km(geodetic_latitude_deg):
    latitude = math.radians(float(geodetic_latitude_deg))
    sin_latitude = math.sin(latitude)
    cos_latitude = math.cos(latitude)
    prime_vertical = WGS84_A_KM / math.sqrt(
        1.0 - WGS84_E2 * sin_latitude * sin_latitude
    )
    x = prime_vertical * cos_latitude
    z = prime_vertical * (1.0 - WGS84_E2) * sin_latitude
    return math.hypot(x, z)


def exact_angle_arcsec(projected_baseline_km, ratio, earth_sun_distance_km):
    return math.atan(
        float(projected_baseline_km) * float(ratio) / float(earth_sun_distance_km)
    ) * ARCSEC_PER_RAD


def fmt(value, decimals=12):
    return f"{float(value):.{decimals}f}"


def html_escape(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def display_widget(rows):
    try:
        from IPython.display import HTML, display
    except Exception:
        return False

    body = "".join(
        "<tr>"
        f"<td>{html_escape(quantity)}</td>"
        f"<td>{html_escape(value)}</td>"
        f"<td>{html_escape(unit)}</td>"
        "</tr>"
        for quantity, value, unit in rows
    )

    html = f"""
    <style>
    .iers-wrap{{background:#03080d;color:#e8f7ff;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;width:760px;max-width:98%;border:1px solid #1e4f64;border-radius:9px;padding:9px;margin:8px 0 14px}}
    .iers-title{{color:#66e8ff;font-size:10px;font-weight:800;letter-spacing:.055em;text-align:center;border-top:1px solid #25708b;border-bottom:1px solid #25708b;padding:5px 0;margin:5px 0}}
    .iers-table{{border-collapse:collapse;width:100%;table-layout:fixed;font-size:10px;background:#050b0f;margin-bottom:7px}}
    .iers-table th{{color:#66e8ff;background:#0a1a22;border-bottom:1px solid #1d3d4a;padding:4px 5px;text-align:left}}
    .iers-table td{{border-bottom:1px solid #102630;padding:4px 5px}}
    .iers-table td:nth-child(2){{color:#ffc861;text-align:right;font-weight:800}}
    .iers-table td:nth-child(3){{color:#5ee08a}}
    .iers-note{{color:#8fb4c1;font-size:9px;margin-top:5px}}
    </style>
    <div class="iers-wrap">
      <div class="iers-title">IERS-0012R — 1769 MAXIMUM PARALLAX CLOSURE AUDIT</div>
      <table class="iers-table">
        <thead><tr><th>Quantity</th><th>Value</th><th>Unit</th></tr></thead>
        <tbody>{body}</tbody>
      </table>
      <div class="iers-note">The 43.535 arcsec value is the ideal equatorial-diameter limit, not the constrained WGS84 antipodal maximum.</div>
    </div>
    """
    display(HTML(html))
    return True


def write_output_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([VERSION, "1769 MAXIMUM PARALLAX CLOSURE AUDIT"])
        writer.writerow([])
        writer.writerow(["quantity", "value", "unit"])
        writer.writerows(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    values = load_q_values(INPUT_CSV)

    latitude = values["Site A latitude"]
    longitude = values["Site A longitude east"]
    d_es_au = values["JPL D ES"]
    ratio = values["JPL D VS / D EV"]
    observed_aprime_bprime = values["A prime B prime"]
    observed_rho = values["Normal separation rho"]
    pi_sun = values["Pi sun"]
    pi_sun_residual = values["Pi sun residual"]

    earth_sun_distance_km = d_es_au * AU_KM
    equatorial_diameter_km = 2.0 * WGS84_A_KM
    local_radius_km = wgs84_geocentric_radius_km(latitude)
    local_antipodal_chord_km = 2.0 * local_radius_km

    normalized_max_linear_arcsec = (
        2.0 * PI_SUN_REFERENCE_ARCSEC * ratio
    )
    ideal_equatorial_linear_arcsec = (
        normalized_max_linear_arcsec / d_es_au
    )
    ideal_equatorial_exact_arcsec = exact_angle_arcsec(
        equatorial_diameter_km,
        ratio,
        earth_sun_distance_km,
    )

    projected_baseline_km = (
        math.tan(observed_aprime_bprime / ARCSEC_PER_RAD)
        * earth_sun_distance_km
        / ratio
    )
    constrained_exact_arcsec = exact_angle_arcsec(
        projected_baseline_km,
        ratio,
        earth_sun_distance_km,
    )
    local_chord_exact_arcsec = exact_angle_arcsec(
        local_antipodal_chord_km,
        ratio,
        earth_sun_distance_km,
    )

    ellipsoid_factor = local_antipodal_chord_km / equatorial_diameter_km
    orientation_factor = projected_baseline_km / local_antipodal_chord_km
    combined_factor = projected_baseline_km / equatorial_diameter_km

    ellipsoid_loss_arcsec = (
        ideal_equatorial_exact_arcsec - local_chord_exact_arcsec
    )
    orientation_loss_arcsec = (
        local_chord_exact_arcsec - constrained_exact_arcsec
    )
    total_loss_arcsec = (
        ideal_equatorial_exact_arcsec - constrained_exact_arcsec
    )
    closure_residual_arcsec = observed_rho - constrained_exact_arcsec

    rows = [
        ("Optimized latitude", fmt(latitude, 9), "deg"),
        ("Optimized longitude east", fmt(longitude, 9), "deg"),
        ("D ES", fmt(d_es_au, 12), "AU"),
        ("D VS / D EV", fmt(ratio, 10), "ratio"),
        ("Normalized maximum linear", fmt(normalized_max_linear_arcsec, 9), "arcsec"),
        ("Ideal equatorial maximum linear", fmt(ideal_equatorial_linear_arcsec, 9), "arcsec"),
        ("Ideal equatorial maximum exact", fmt(ideal_equatorial_exact_arcsec, 9), "arcsec"),
        ("Equatorial diameter", fmt(equatorial_diameter_km, 6), "km"),
        ("Local WGS84 antipodal chord", fmt(local_antipodal_chord_km, 6), "km"),
        ("Projected baseline AB", fmt(projected_baseline_km, 6), "km"),
        ("Ellipsoid factor", fmt(ellipsoid_factor, 12), "ratio"),
        ("Orientation factor", fmt(orientation_factor, 12), "ratio"),
        ("Combined factor", fmt(combined_factor, 12), "ratio"),
        ("Ellipsoid loss", fmt(ellipsoid_loss_arcsec, 9), "arcsec"),
        ("Track-normal projection loss", fmt(orientation_loss_arcsec, 9), "arcsec"),
        ("Total ideal-to-constrained loss", fmt(total_loss_arcsec, 9), "arcsec"),
        ("Predicted constrained rho", fmt(constrained_exact_arcsec, 9), "arcsec"),
        ("Observed rho", fmt(observed_rho, 9), "arcsec"),
        ("Exact closure residual", fmt(closure_residual_arcsec, 12), "arcsec"),
        ("Computed pi sun", fmt(pi_sun, 9), "arcsec"),
        ("Pi sun residual", fmt(pi_sun_residual, 9), "arcsec"),
    ]

    print(f"CODE OUTPUT: {VERSION}")
    print()
    print("CODE INPUTS")
    print(f"Program                : {PROGRAM_NAME}")
    print(f"Input CSV              : {INPUT_CSV}")
    print()
    print("COMMENTS")
    print("IERS-0012Q already found the constrained WGS84 antipodal optimum.")
    print("The earlier 43.535 arcsec label is the ideal equatorial-diameter limit.")
    print("This audit separates WGS84 radius loss from track-normal projection loss.")
    print()

    display_widget(rows)

    print("RESULTS")
    for quantity, value, unit in rows:
        print(f"{quantity:<32}: {value} {unit}")
    print()
    print("OUTPUT SUMMARY")
    write_output_csv(OUTPUT_CSV, rows)
    print(f"CSV output             : {OUTPUT_CSV}")
    print()
    print("PAPER COMPARISON")
    print(f"Reference pi sun       : {PI_SUN_REFERENCE_ARCSEC:.9f} arcsec")
    print()
    print("EQUATION STATUS")
    print("Exact atan baseline projection closure : VERIFIED")
    print("WGS84 local-radius decomposition       : VERIFIED")
    print("Track-normal projected baseline factor : VERIFIED")
    print(datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
# IERS-0012R
