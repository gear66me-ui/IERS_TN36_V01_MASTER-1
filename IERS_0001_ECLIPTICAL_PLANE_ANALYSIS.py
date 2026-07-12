"""
IERS TN36 — Ecliptical Plane Analysis
Parts A+B+C+D
Bucaramanga observer, 2012 Venus transit
"""

from pathlib import Path
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

VERSION = "IERS-0001-D"
AU_KM = 149_597_870.7
ARCSEC_PER_RAD = 206_264.80624709636

OBSERVER = {
    "name": "Bucaramanga, Colombia",
    "latitude_deg": 7.125390,
    "longitude_deg": -73.119800,
    "elevation_m": 959.0,
}

TRANSIT = {
    "target": "Venus",
    "center": "Sun",
    "date_utc": "2012-06-06",
}

SUN_VECTOR_FILE = "JPL_2012_BUCARAMANGA_SUN_VECTORS.csv"
VENUS_VECTOR_FILE = "JPL_2012_BUCARAMANGA_VENUS_VECTORS.csv"


def normalize(vector):
    vector = np.asarray(vector, dtype=float)
    magnitude = np.linalg.norm(vector)
    if not np.isfinite(magnitude) or magnitude == 0.0:
        raise ValueError("Cannot normalize a zero or non-finite vector.")
    return vector / magnitude


def normalize_rows(vectors):
    vectors = np.asarray(vectors, dtype=float)
    magnitudes = np.linalg.norm(vectors, axis=1)
    if np.any(~np.isfinite(magnitudes)) or np.any(magnitudes == 0.0):
        raise ValueError("Cannot normalize zero or non-finite vector rows.")
    return vectors / magnitudes[:, None]


def project_to_plane(vector, line_of_sight):
    line_of_sight = normalize(line_of_sight)
    vector = np.asarray(vector, dtype=float)
    return vector - np.dot(vector, line_of_sight) * line_of_sight


def angular_separation(vector_a, vector_b):
    cosine = np.clip(np.dot(normalize(vector_a), normalize(vector_b)), -1.0, 1.0)
    return np.arccos(cosine) * ARCSEC_PER_RAD


def linear_track_fit(x_values, y_values):
    return np.polyfit(np.asarray(x_values, float), np.asarray(y_values, float), 1)


def evaluate_track(coefficients, x_values):
    return np.polyval(coefficients, np.asarray(x_values, float))


def build_screen_basis(line_of_sight):
    w_axis = normalize(line_of_sight)
    trial_axis = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(w_axis, trial_axis)) > 0.95:
        trial_axis = np.array([0.0, 1.0, 0.0])
    u_axis = normalize(np.cross(trial_axis, w_axis))
    v_axis = normalize(np.cross(w_axis, u_axis))
    return u_axis, v_axis, w_axis


def screen_coordinates(vector, line_of_sight):
    u_axis, v_axis, w_axis = build_screen_basis(line_of_sight)
    projected = project_to_plane(vector, w_axis)
    return float(np.dot(projected, u_axis)), float(np.dot(projected, v_axis))


def relative_sun_venus(sun_vector, venus_vector):
    return np.asarray(venus_vector, float) - np.asarray(sun_vector, float)


def interpolate_vectors(times, vectors, kind="cubic"):
    times = np.asarray(times, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    if times.ndim != 1 or vectors.shape != (times.size, 3):
        raise ValueError("Expected times shape (N,) and vectors shape (N,3).")
    interpolation_kind = kind if times.size >= 4 else "linear"
    components = [
        interp1d(times, vectors[:, axis], kind=interpolation_kind, bounds_error=True)
        for axis in range(3)
    ]

    def sample(epoch):
        epoch = np.asarray(epoch, dtype=float)
        sampled = np.stack([component(epoch) for component in components], axis=-1)
        return sampled.astype(float)

    return sample


def _canonical_header(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _find_header(lines, soe_index):
    for index in range(soe_index - 1, max(-1, soe_index - 12), -1):
        candidate = next(csv.reader([lines[index]], skipinitialspace=True))
        canonical = [_canonical_header(value) for value in candidate]
        if all(axis in canonical for axis in ("X", "Y", "Z")):
            return candidate
    return None


def read_horizons_vector_file(file_path):
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing JPL Horizons vector file: {path.name}")

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        soe_index = next(i for i, line in enumerate(lines) if "$$SOE" in line)
        eoe_index = next(i for i, line in enumerate(lines) if "$$EOE" in line and i > soe_index)
    except StopIteration as exc:
        raise ValueError(f"{path.name}: missing $$SOE/$$EOE vector-data markers.") from exc

    header = _find_header(lines, soe_index)
    if header is None:
        raise ValueError(f"{path.name}: X, Y, Z header columns were not found.")

    canonical = [_canonical_header(value) for value in header]
    x_index, y_index, z_index = (canonical.index(axis) for axis in ("X", "Y", "Z"))
    jd_candidates = ["JDTDB", "JDUT", "JD"]
    jd_index = next((canonical.index(name) for name in jd_candidates if name in canonical), 0)

    epochs = []
    vectors = []
    for row in csv.reader(lines[soe_index + 1:eoe_index], skipinitialspace=True):
        if not row or max(jd_index, x_index, y_index, z_index) >= len(row):
            continue
        try:
            epochs.append(float(row[jd_index]))
            vectors.append([float(row[x_index]), float(row[y_index]), float(row[z_index])])
        except ValueError:
            continue

    result = {
        "path": path,
        "jd": np.asarray(epochs, dtype=float),
        "position_km": np.asarray(vectors, dtype=float),
    }
    validate_vector_table(result, path.name)
    return result


def validate_vector_table(table, label):
    epochs = np.asarray(table["jd"], dtype=float)
    vectors = np.asarray(table["position_km"], dtype=float)
    if epochs.size < 2:
        raise ValueError(f"{label}: fewer than two valid vector records.")
    if vectors.shape != (epochs.size, 3):
        raise ValueError(f"{label}: vector array must have shape (N,3).")
    if not np.all(np.isfinite(epochs)) or not np.all(np.isfinite(vectors)):
        raise ValueError(f"{label}: non-finite values detected.")
    if np.any(np.diff(epochs) <= 0.0):
        raise ValueError(f"{label}: Julian dates must increase strictly.")
    if np.any(np.linalg.norm(vectors, axis=1) == 0.0):
        raise ValueError(f"{label}: zero-length Cartesian vector detected.")
    return True


def display_vector_summary(label, table):
    epochs = table["jd"]
    vectors = table["position_km"]
    radii = np.linalg.norm(vectors, axis=1)
    print(f"{label:<8} {epochs.size:6d}  {epochs[0]:16.8f}  {epochs[-1]:16.8f}  "
          f"{radii.min():15.3f}  {radii.max():15.3f}")


def load_available_horizons_vectors():
    required = [("Sun", SUN_VECTOR_FILE), ("Venus", VENUS_VECTOR_FILE)]
    loaded = {}
    for label, filename in required:
        path = Path(filename)
        if path.is_file():
            loaded[label.lower()] = read_horizons_vector_file(path)

    if loaded:
        print("\nBODY      ROWS          JD START            JD END        R MIN km        R MAX km")
        for label in ("Sun", "Venus"):
            key = label.lower()
            if key in loaded:
                display_vector_summary(label, loaded[key])
    else:
        print("\nJPL vector files not yet present in the Colab working directory.")
        print(f"Required: {SUN_VECTOR_FILE}")
        print(f"Required: {VENUS_VECTOR_FILE}")
    return loaded


def common_epoch_grid(sun_table, venus_table):
    sun_jd = np.asarray(sun_table["jd"], dtype=float)
    venus_jd = np.asarray(venus_table["jd"], dtype=float)
    start = max(sun_jd[0], venus_jd[0])
    stop = min(sun_jd[-1], venus_jd[-1])
    if stop <= start:
        raise ValueError("Sun and Venus vector tables have no common epoch interval.")

    sun_step = np.median(np.diff(sun_jd))
    venus_step = np.median(np.diff(venus_jd))
    step = max(sun_step, venus_step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("Unable to derive a valid common epoch cadence.")

    count = int(np.floor((stop - start) / step)) + 1
    epochs = start + np.arange(count, dtype=float) * step
    tolerance = 64.0 * np.finfo(float).eps * max(abs(start), abs(stop))
    if stop - epochs[-1] > tolerance:
        epochs = np.append(epochs, stop)
    if epochs.size < 3:
        raise ValueError("At least three synchronized epochs are required.")
    return epochs


def gnomonic_coordinates(unit_vectors, basis):
    u_axis, v_axis, w_axis = basis
    unit_vectors = np.asarray(unit_vectors, dtype=float)
    denominator = unit_vectors @ w_axis
    if np.any(denominator <= 0.0):
        raise ValueError("A direction lies outside the forward tangent hemisphere.")
    x = (unit_vectors @ u_axis) / denominator
    y = (unit_vectors @ v_axis) / denominator
    return np.column_stack((x, y))


def empirical_solar_motion_basis(sun_unit_vectors):
    sun_unit_vectors = normalize_rows(sun_unit_vectors)
    reference_axis = normalize(np.mean(sun_unit_vectors, axis=0))
    u0, v0, _ = build_screen_basis(reference_axis)
    preliminary = gnomonic_coordinates(sun_unit_vectors, (u0, v0, reference_axis))
    centered = preliminary - np.mean(preliminary, axis=0)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if singular_values[0] == 0.0:
        raise ValueError("Solar directions do not define a measurable motion axis.")

    direction_2d = vh[0]
    chronological_displacement = preliminary[-1] - preliminary[0]
    if np.dot(direction_2d, chronological_displacement) < 0.0:
        direction_2d = -direction_2d

    u_axis = normalize(direction_2d[0] * u0 + direction_2d[1] * v0)
    v_axis = normalize(np.cross(reference_axis, u_axis))
    return u_axis, v_axis, reference_axis


def orthogonal_track_fit(x_values, y_values):
    points = np.column_stack((x_values, y_values)).astype(float)
    centroid = np.mean(points, axis=0)
    _, singular_values, vh = np.linalg.svd(points - centroid, full_matrices=False)
    direction = vh[0]
    if np.dot(direction, points[-1] - points[0]) < 0.0:
        direction = -direction
    normal = np.array([-direction[1], direction[0]])
    along = (points - centroid) @ direction
    cross = (points - centroid) @ normal
    angle_deg = float(np.degrees(np.arctan2(direction[1], direction[0])) % 360.0)
    if np.isclose(angle_deg, 360.0, atol=1.0e-10):
        angle_deg = 0.0
    return {
        "centroid": centroid,
        "direction": direction,
        "normal": normal,
        "along": along,
        "cross": cross,
        "angle_deg": angle_deg,
        "rms_cross_arcsec": float(np.sqrt(np.mean(cross ** 2))),
        "singular_values": singular_values,
    }


def derive_apparent_track(sun_table, venus_table):
    epochs = common_epoch_grid(sun_table, venus_table)
    sun_sample = interpolate_vectors(sun_table["jd"], sun_table["position_km"])
    venus_sample = interpolate_vectors(venus_table["jd"], venus_table["position_km"])
    sun_vectors = sun_sample(epochs)
    venus_vectors = venus_sample(epochs)

    sun_directions = normalize_rows(sun_vectors)
    venus_directions = normalize_rows(venus_vectors)
    basis = empirical_solar_motion_basis(sun_directions)

    sun_xy = gnomonic_coordinates(sun_directions, basis)
    venus_xy = gnomonic_coordinates(venus_directions, basis)
    relative_xy_arcsec = (venus_xy - sun_xy) * ARCSEC_PER_RAD

    x_arcsec = relative_xy_arcsec[:, 0]
    y_arcsec = relative_xy_arcsec[:, 1]
    separation_arcsec = np.hypot(x_arcsec, y_arcsec)
    closest_index = int(np.argmin(separation_arcsec))
    fit = orthogonal_track_fit(x_arcsec, y_arcsec)

    return {
        "jd": epochs,
        "sun_vectors_km": sun_vectors,
        "venus_vectors_km": venus_vectors,
        "x_arcsec": x_arcsec,
        "y_arcsec": y_arcsec,
        "separation_arcsec": separation_arcsec,
        "closest_index": closest_index,
        "fit": fit,
        "basis": basis,
    }


def display_track_summary(track):
    index = track["closest_index"]
    fit = track["fit"]
    print("\nAPPARENT VENUS TRACK — EMPIRICAL SOLAR-MOTION BASIS")
    print("EPOCHS       TRACK ANGLE deg       RMS NORMAL arcsec       CA JD TDB       CA SEP arcsec")
    print(f"{track['jd'].size:6d}       {fit['angle_deg']:15.6f}       "
          f"{fit['rms_cross_arcsec']:17.6f}       {track['jd'][index]:13.8f}       "
          f"{track['separation_arcsec'][index]:13.6f}")


def plot_apparent_track(track):
    x_values = track["x_arcsec"]
    y_values = track["y_arcsec"]
    index = track["closest_index"]
    fit = track["fit"]

    span = np.array([fit["along"].min(), fit["along"].max()])
    fitted = fit["centroid"] + span[:, None] * fit["direction"]

    fig, axis = plt.subplots(figsize=(8.0, 6.0))
    axis.plot(x_values, y_values, linewidth=1.2, label="JPL apparent Venus track")
    axis.plot(fitted[:, 0], fitted[:, 1], "--", linewidth=1.0, label="Orthogonal linear fit")
    axis.scatter([x_values[index]], [y_values[index]], s=34, label="Sampled closest approach")
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("Along empirical solar motion (arcsec)")
    axis.set_ylabel("Normal to empirical solar motion (arcsec)")
    axis.set_title(f"Bucaramanga 2012 Venus Transit — Track angle {fit['angle_deg']:.6f}°")
    axis.grid(True, linewidth=0.5, alpha=0.35)
    axis.legend()
    plt.show()


def geometry_self_test():
    line_of_sight = normalize([1.0, 0.2, 0.1])
    relative_vector = relative_sun_venus([0.0, 0.0, 0.0], [1200.0, -350.0, 800.0])
    screen_x, screen_y = screen_coordinates(relative_vector, line_of_sight)
    return screen_x, screen_y


def main():
    screen_x, screen_y = geometry_self_test()
    print("IERS TN36 - Ecliptical Plane Analysis")
    print(f"Version : {VERSION}")
    print(f"Observer: {OBSERVER['name']}")
    print(f"Transit : {TRANSIT['date_utc']}")
    print(f"Geometry self-test: X={screen_x:.6f} km  Y={screen_y:.6f} km")

    loaded = load_available_horizons_vectors()
    if "sun" in loaded and "venus" in loaded:
        track = derive_apparent_track(loaded["sun"], loaded["venus"])
        display_track_summary(track)
        plot_apparent_track(track)


if __name__ == "__main__":
    main()
