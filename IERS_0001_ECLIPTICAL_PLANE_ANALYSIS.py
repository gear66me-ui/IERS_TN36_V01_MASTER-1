"""
IERS TN36 — Ecliptical Plane Analysis
Parts A+B+C+D+E
Bucaramanga observer, 2012 Venus transit
"""

from pathlib import Path
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

VERSION = "IERS-0001-E"
ARCSEC_PER_RAD = 206_264.80624709636
SECONDS_PER_DAY = 86_400.0

OBSERVER = {
    "name": "Bucaramanga, Colombia",
    "latitude_deg": 7.125390,
    "longitude_deg": -73.119800,
    "elevation_m": 959.0,
}
TRANSIT = {"target": "Venus", "center": "Sun", "date_utc": "2012-06-06"}

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


def build_screen_basis(line_of_sight):
    w_axis = normalize(line_of_sight)
    trial = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(w_axis, trial)) > 0.95:
        trial = np.array([0.0, 1.0, 0.0])
    u_axis = normalize(np.cross(trial, w_axis))
    v_axis = normalize(np.cross(w_axis, u_axis))
    return u_axis, v_axis, w_axis


def interpolate_vectors(times, vectors):
    times = np.asarray(times, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    if times.ndim != 1 or vectors.shape != (times.size, 3):
        raise ValueError("Expected times shape (N,) and vectors shape (N,3).")
    kind = "cubic" if times.size >= 4 else "linear"
    components = [
        interp1d(times, vectors[:, axis], kind=kind, bounds_error=True)
        for axis in range(3)
    ]

    def sample(epoch):
        epoch = np.asarray(epoch, dtype=float)
        return np.stack([component(epoch) for component in components], axis=-1)

    return sample


def _canonical_header(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _find_header(lines, soe_index):
    for index in range(soe_index - 1, max(-1, soe_index - 16), -1):
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
        soe = next(i for i, line in enumerate(lines) if "$$SOE" in line)
        eoe = next(i for i, line in enumerate(lines) if "$$EOE" in line and i > soe)
    except StopIteration as exc:
        raise ValueError(f"{path.name}: missing $$SOE/$$EOE markers.") from exc

    header = _find_header(lines, soe)
    if header is None:
        raise ValueError(f"{path.name}: X, Y, Z header columns were not found.")

    canonical = [_canonical_header(value) for value in header]
    x_index, y_index, z_index = (canonical.index(axis) for axis in ("X", "Y", "Z"))
    jd_index = next(
        (canonical.index(name) for name in ("JDTDB", "JDUT", "JD") if name in canonical),
        0,
    )

    epochs, vectors = [], []
    for row in csv.reader(lines[soe + 1:eoe], skipinitialspace=True):
        if not row or max(jd_index, x_index, y_index, z_index) >= len(row):
            continue
        try:
            epochs.append(float(row[jd_index]))
            vectors.append([float(row[x_index]), float(row[y_index]), float(row[z_index])])
        except ValueError:
            continue

    table = {
        "path": path,
        "jd": np.asarray(epochs, dtype=float),
        "position_km": np.asarray(vectors, dtype=float),
    }
    validate_vector_table(table, path.name)
    return table


def validate_vector_table(table, label):
    epochs = np.asarray(table["jd"], dtype=float)
    vectors = np.asarray(table["position_km"], dtype=float)
    if epochs.size < 3:
        raise ValueError(f"{label}: at least three valid vector records are required.")
    if vectors.shape != (epochs.size, 3):
        raise ValueError(f"{label}: vector array must have shape (N,3).")
    if not np.all(np.isfinite(epochs)) or not np.all(np.isfinite(vectors)):
        raise ValueError(f"{label}: non-finite values detected.")
    if np.any(np.diff(epochs) <= 0.0):
        raise ValueError(f"{label}: Julian dates must increase strictly.")
    if np.any(np.linalg.norm(vectors, axis=1) == 0.0):
        raise ValueError(f"{label}: zero-length Cartesian vector detected.")


def load_available_horizons_vectors():
    loaded = {}
    for label, filename in (("sun", SUN_VECTOR_FILE), ("venus", VENUS_VECTOR_FILE)):
        if Path(filename).is_file():
            loaded[label] = read_horizons_vector_file(filename)

    if len(loaded) == 2:
        print("\nBODY      ROWS          JD START            JD END        R MIN km        R MAX km")
        for label in ("sun", "venus"):
            table = loaded[label]
            radii = np.linalg.norm(table["position_km"], axis=1)
            print(
                f"{label.title():<8} {table['jd'].size:6d}  "
                f"{table['jd'][0]:16.8f}  {table['jd'][-1]:16.8f}  "
                f"{radii.min():15.3f}  {radii.max():15.3f}"
            )
    else:
        print("\nJPL vector files not yet present in the Colab working directory.")
        print(f"Required: {SUN_VECTOR_FILE}")
        print(f"Required: {VENUS_VECTOR_FILE}")
    return loaded


def common_epoch_grid(sun_table, venus_table):
    start = max(sun_table["jd"][0], venus_table["jd"][0])
    stop = min(sun_table["jd"][-1], venus_table["jd"][-1])
    if stop <= start:
        raise ValueError("Sun and Venus vector tables have no common interval.")
    step = max(
        np.median(np.diff(sun_table["jd"])),
        np.median(np.diff(venus_table["jd"])),
    )
    count = int(np.floor((stop - start) / step)) + 1
    epochs = start + np.arange(count, dtype=float) * step
    if stop - epochs[-1] > 1.0e-10:
        epochs = np.append(epochs, stop)
    if epochs.size < 3:
        raise ValueError("At least three synchronized epochs are required.")
    return epochs


def gnomonic_coordinates(unit_vectors, basis):
    u_axis, v_axis, w_axis = basis
    unit_vectors = np.asarray(unit_vectors, dtype=float)
    denominator = unit_vectors @ w_axis
    if np.any(denominator <= 0.0):
        raise ValueError("Direction outside the forward tangent hemisphere.")
    return np.column_stack(
        ((unit_vectors @ u_axis) / denominator, (unit_vectors @ v_axis) / denominator)
    )


def empirical_solar_motion_basis(sun_directions):
    sun_directions = normalize_rows(sun_directions)
    w_axis = normalize(np.mean(sun_directions, axis=0))
    u0, v0, _ = build_screen_basis(w_axis)
    xy = gnomonic_coordinates(sun_directions, (u0, v0, w_axis))
    centered = xy - np.mean(xy, axis=0)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if singular_values[0] == 0.0:
        raise ValueError("Solar directions do not define a motion axis.")
    direction = vh[0]
    if np.dot(direction, xy[-1] - xy[0]) < 0.0:
        direction = -direction
    u_axis = normalize(direction[0] * u0 + direction[1] * v0)
    v_axis = normalize(np.cross(w_axis, u_axis))
    return u_axis, v_axis, w_axis


def signed_line_angle_deg(direction):
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    while angle > 90.0:
        angle -= 180.0
    while angle <= -90.0:
        angle += 180.0
    return angle


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
    return {
        "centroid": centroid,
        "direction": direction,
        "normal": normal,
        "along": along,
        "cross": cross,
        "angle_deg": signed_line_angle_deg(direction),
        "rms_cross_arcsec": float(np.sqrt(np.mean(cross ** 2))),
        "singular_values": singular_values,
    }


def derive_track_in_basis(epochs, sun_vectors, venus_vectors, basis):
    sun_directions = normalize_rows(sun_vectors)
    venus_directions = normalize_rows(venus_vectors)
    sun_xy = gnomonic_coordinates(sun_directions, basis)
    venus_xy = gnomonic_coordinates(venus_directions, basis)
    relative = (venus_xy - sun_xy) * ARCSEC_PER_RAD
    separation = np.linalg.norm(relative, axis=1)
    closest_index = int(np.argmin(separation))
    fit = orthogonal_track_fit(relative[:, 0], relative[:, 1])
    return {
        "jd": epochs,
        "sun_vectors_km": sun_vectors,
        "venus_vectors_km": venus_vectors,
        "x_arcsec": relative[:, 0],
        "y_arcsec": relative[:, 1],
        "separation_arcsec": separation,
        "closest_index": closest_index,
        "fit": fit,
        "basis": basis,
    }


def derive_apparent_track(sun_table, venus_table):
    epochs = common_epoch_grid(sun_table, venus_table)
    sun_vectors = interpolate_vectors(
        sun_table["jd"], sun_table["position_km"]
    )(epochs)
    venus_vectors = interpolate_vectors(
        venus_table["jd"], venus_table["position_km"]
    )(epochs)
    basis = empirical_solar_motion_basis(normalize_rows(sun_vectors))
    return derive_track_in_basis(epochs, sun_vectors, venus_vectors, basis)


def vector_derivative(epochs_jd, vectors):
    seconds = (np.asarray(epochs_jd, dtype=float) - epochs_jd[0]) * SECONDS_PER_DAY
    edge_order = 2 if len(seconds) >= 3 else 1
    return np.gradient(np.asarray(vectors, dtype=float), seconds, axis=0, edge_order=edge_order)


def derive_ecliptic_plane_normal(epochs_jd, sun_vectors):
    velocities = vector_derivative(epochs_jd, sun_vectors)
    angular_momentum = np.cross(sun_vectors, velocities)
    norms = np.linalg.norm(angular_momentum, axis=1)
    valid = np.isfinite(norms) & (norms > 0.0)
    if np.count_nonzero(valid) < 3:
        raise ValueError("Insufficient vectors to derive the local ecliptic plane.")

    normals = angular_momentum[valid] / norms[valid, None]
    reference = normals[len(normals) // 2].copy()
    normals[np.einsum("ij,j->i", normals, reference) < 0.0] *= -1.0
    normal = normalize(np.average(normals, axis=0, weights=norms[valid]))

    sun_directions = normalize_rows(sun_vectors)
    departures = np.arcsin(np.clip(sun_directions @ normal, -1.0, 1.0))
    rms_plane_arcsec = float(np.sqrt(np.mean(departures ** 2)) * ARCSEC_PER_RAD)
    scatter = np.arccos(np.clip(normals @ normal, -1.0, 1.0))
    normal_scatter_arcsec = float(np.sqrt(np.mean(scatter ** 2)) * ARCSEC_PER_RAD)
    return normal, rms_plane_arcsec, normal_scatter_arcsec


def ecliptic_screen_basis(ecliptic_normal, reference_axis, sun_directions):
    w_axis = normalize(reference_axis)
    u_axis = normalize(np.cross(ecliptic_normal, w_axis))
    displacement = sun_directions[-1] - sun_directions[0]
    if np.dot(u_axis, displacement) < 0.0:
        u_axis = -u_axis
    v_axis = normalize(np.cross(w_axis, u_axis))
    return u_axis, v_axis, w_axis


def derive_ecliptic_track(apparent_track):
    sun_vectors = apparent_track["sun_vectors_km"]
    venus_vectors = apparent_track["venus_vectors_km"]
    epochs = apparent_track["jd"]
    normal, plane_rms, normal_scatter = derive_ecliptic_plane_normal(epochs, sun_vectors)
    sun_directions = normalize_rows(sun_vectors)
    reference_axis = apparent_track["basis"][2]
    basis = ecliptic_screen_basis(normal, reference_axis, sun_directions)
    track = derive_track_in_basis(epochs, sun_vectors, venus_vectors, basis)
    track["ecliptic_normal"] = normal
    track["plane_rms_arcsec"] = plane_rms
    track["normal_scatter_arcsec"] = normal_scatter

    empirical_u = apparent_track["basis"][0]
    rotation = np.degrees(
        np.arctan2(np.dot(empirical_u, basis[1]), np.dot(empirical_u, basis[0]))
    )
    while rotation > 90.0:
        rotation -= 180.0
    while rotation <= -90.0:
        rotation += 180.0
    track["basis_rotation_deg"] = float(rotation)
    return track


def display_track_comparison(apparent_track, ecliptic_track):
    a_index = apparent_track["closest_index"]
    e_index = ecliptic_track["closest_index"]
    print("\nTRACK ANGLE COMPARISON — JPL VECTORS ONLY")
    print("FRAME                     ANGLE deg     RMS NORMAL arcsec       CA JD TDB       CA SEP arcsec")
    print(
        f"Empirical solar motion   {apparent_track['fit']['angle_deg']:10.6f}"
        f"     {apparent_track['fit']['rms_cross_arcsec']:17.6f}"
        f"     {apparent_track['jd'][a_index]:13.8f}"
        f"     {apparent_track['separation_arcsec'][a_index]:13.6f}"
    )
    print(
        f"Derived ecliptic plane   {ecliptic_track['fit']['angle_deg']:10.6f}"
        f"     {ecliptic_track['fit']['rms_cross_arcsec']:17.6f}"
        f"     {ecliptic_track['jd'][e_index]:13.8f}"
        f"     {ecliptic_track['separation_arcsec'][e_index]:13.6f}"
    )
    print("\nECLIPTIC PLANE DIAGNOSTICS")
    print("BASIS ROTATION deg     SUN-PLANE RMS arcsec     NORMAL SCATTER arcsec")
    print(
        f"{ecliptic_track['basis_rotation_deg']:18.6f}"
        f"     {ecliptic_track['plane_rms_arcsec']:20.6f}"
        f"     {ecliptic_track['normal_scatter_arcsec']:21.6f}"
    )


def plot_track_comparison(apparent_track, ecliptic_track):
    fig, axis = plt.subplots(figsize=(8.0, 6.0))
    axis.plot(
        apparent_track["x_arcsec"],
        apparent_track["y_arcsec"],
        linewidth=1.1,
        label=f"Empirical basis {apparent_track['fit']['angle_deg']:.6f}°",
    )
    axis.plot(
        ecliptic_track["x_arcsec"],
        ecliptic_track["y_arcsec"],
        linewidth=1.1,
        label=f"Ecliptic basis {ecliptic_track['fit']['angle_deg']:.6f}°",
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("Tangent-plane X (arcsec)")
    axis.set_ylabel("Tangent-plane Y (arcsec)")
    axis.set_title("Bucaramanga 2012 Venus Transit — Derived Coordinate Frames")
    axis.grid(True, linewidth=0.5, alpha=0.35)
    axis.legend()
    plt.show()


def geometry_self_test():
    line_of_sight = normalize([1.0, 0.2, 0.1])
    relative = np.array([1200.0, -350.0, 800.0])
    u_axis, v_axis, w_axis = build_screen_basis(line_of_sight)
    projected = relative - np.dot(relative, w_axis) * w_axis
    return float(np.dot(projected, u_axis)), float(np.dot(projected, v_axis))


def main():
    screen_x, screen_y = geometry_self_test()
    print("IERS TN36 - Ecliptical Plane Analysis")
    print(f"Version : {VERSION}")
    print(f"Observer: {OBSERVER['name']}")
    print(f"Transit : {TRANSIT['date_utc']}")
    print(f"Geometry self-test: X={screen_x:.6f} km  Y={screen_y:.6f} km")

    loaded = load_available_horizons_vectors()
    if len(loaded) == 2:
        apparent_track = derive_apparent_track(loaded["sun"], loaded["venus"])
        ecliptic_track = derive_ecliptic_track(apparent_track)
        display_track_comparison(apparent_track, ecliptic_track)
        plot_track_comparison(apparent_track, ecliptic_track)


if __name__ == "__main__":
    main()
