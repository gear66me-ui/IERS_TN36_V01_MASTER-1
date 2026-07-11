    median_radius = float(np.median(radius_norm[finite]))
    radius_tolerance = max(0.008, 0.35 * median_radius)
    candidate = finite & (quality >= 0.12) & (
        np.abs(radius_norm - median_radius) <= radius_tolerance
    )
    candidate_indices = np.flatnonzero(candidate)
    if candidate_indices.size < 80:
        candidate_indices = np.flatnonzero(finite)
    if candidate_indices.size < 80:
        raise RuntimeError("Too few radius-consistent Venus candidates.")

    frame_min = float(np.min(frame[candidate_indices]))
    frame_span = float(np.max(frame[candidate_indices]) - frame_min)
    if frame_span <= 0.0:
        raise RuntimeError("Trajectory frame span is zero.")
    u_all = (frame - frame_min) / frame_span
    u = u_all[candidate_indices]
    x = x_norm[candidate_indices]
    y = y_norm[candidate_indices]
    rn = radius_norm[candidate_indices]
    q = np.clip(quality[candidate_indices], 0.0, 1.0)

    residual_limit = max(0.030, 1.80 * median_radius)
    radius_limit = max(0.008, 0.30 * median_radius)
    rng = np.random.default_rng(20120605)
    best_mask: Optional[np.ndarray] = None
    best_score = -np.inf
    trial_count = min(9000, max(2500, 3 * candidate_indices.size))

    for _ in range(trial_count):
        i, j = map(int, rng.integers(0, candidate_indices.size, size=2))
        if i == j:
            continue
        du = float(u[j] - u[i])
        if abs(du) < 0.18:
            continue
        vx = float((x[j] - x[i]) / du)
        vy = float((y[j] - y[i]) / du)
        speed = math.hypot(vx, vy)
        if not (0.45 <= speed <= 2.80):
            continue
        x0 = float(x[i] - vx * u[i])
        y0 = float(y[i] - vy * u[i])
        residual = np.hypot(x - (x0 + vx * u), y - (y0 + vy * u))
        inlier = (residual <= residual_limit) & (
            np.abs(rn - median_radius) <= radius_limit
        )
        count = int(np.count_nonzero(inlier))
        if count < 40:
            continue
        span = float(np.ptp(u[inlier]))
        if span < 0.45:
            continue
        score = (
            count + 350.0 * span + 80.0 * float(np.mean(q[inlier]))
            - 900.0 * float(np.median(residual[inlier]))
        )
        if score > best_score:
            best_score = score
            best_mask = inlier

    if best_mask is None:
        raise RuntimeError(
            "No continuous moving Venus trajectory passed the RANSAC constraints."
        )

    inlier = best_mask.copy()
    coeff_x = np.zeros(2, dtype=float)
    coeff_y = np.zeros(2, dtype=float)
    for _ in range(8):
        if np.count_nonzero(inlier) < 20:
            break
        design = np.column_stack((np.ones(np.count_nonzero(inlier)), u[inlier]))
        coeff_x, *_ = np.linalg.lstsq(design, x[inlier], rcond=None)
        coeff_y, *_ = np.linalg.lstsq(design, y[inlier], rcond=None)
        residual = np.hypot(
            x - (coeff_x[0] + coeff_x[1] * u),
            y - (coeff_y[0] + coeff_y[1] * u),
        )
        median_residual, sigma_residual = robust_scale(residual[inlier])
        adaptive_limit = min(
            0.085,
            max(0.018, 1.20 * median_radius, median_residual + 3.5 * sigma_residual),
        )
        updated = (residual <= adaptive_limit) & (
            np.abs(rn - median_radius) <= radius_limit
        )
        if np.array_equal(updated, inlier):
            break
        inlier = updated

    selected_count = int(np.count_nonzero(inlier))
    selected_span = float(np.ptp(u[inlier])) if selected_count else 0.0
    speed = math.hypot(float(coeff_x[1]), float(coeff_y[1]))
    minimum_count = max(180, int(round(0.045 * len(table))))
    if selected_count < minimum_count:
        raise RuntimeError(
            f"Continuous Venus track has only {selected_count} inliers; "
            f"minimum required is {minimum_count}."
        )
    if selected_span < 0.55:
        raise RuntimeError(
            f"Continuous Venus track spans only {selected_span:.3f} of the video."
        )
    if not (0.45 <= speed <= 2.80):
        raise RuntimeError(
            f"Continuous Venus track speed {speed:.6f} R_sun/video is implausible."
        )

    selected_global = np.zeros(len(table), dtype=bool)
    selected_global[candidate_indices[inlier]] = True
    predicted_x_all = coeff_x[0] + coeff_x[1] * u_all
    predicted_y_all = coeff_y[0] + coeff_y[1] * u_all
    residual_all = np.hypot(x_norm - predicted_x_all, y_norm - predicted_y_all)

    table["venus_track_inlier"] = selected_global
    table["trajectory_model_x_norm"] = predicted_x_all
    table["trajectory_model_y_norm"] = predicted_y_all
    table["trajectory_residual_norm"] = residual_all
    table["trajectory_speed_norm_per_video"] = speed
    table["venus_detected"] = selected_global
    table["venus_source"] = np.where(
        selected_global, "DETECTED_CONTINUOUS_TRACK", "REJECTED_SUNSPOT_OR_ARTIFACT"
    )
    table.loc[
        ~selected_global,
        ["venus_cx_px", "venus_cy_px", "venus_radius_px", "venus_quality"],
    ] = np.nan

    print(
        "DEBUG | continuous Venus track selected | "
        f"raw_detections={int(np.count_nonzero(detected))} | "
        f"selected={selected_count} | span={selected_span:.6f} | "
        f"speed={speed:.6f} R_sun/video"
    )
    return table

def normalize_and_fill(frame_table: pd.DataFrame) -> pd.DataFrame:
    table = frame_table.copy()
    table["x_norm_raw"] = (
        table["venus_cx_px"] - table["solar_cx_px"]
    ) / table["solar_radius_px"]
    table["y_norm_raw"] = (
        table["solar_cy_px"] - table["venus_cy_px"]
    ) / table["solar_radius_px"]
    table["venus_radius_norm_raw"] = (
        table["venus_radius_px"] / table["solar_radius_px"]
    )

    columns = [
        "venus_cx_px",
        "venus_cy_px",
        "venus_radius_px",
        "x_norm_raw",
        "y_norm_raw",
        "venus_radius_norm_raw",
    ]
    invalid = ~table["venus_detected"].astype(bool)
    table.loc[invalid, columns] = np.nan
    for column in columns:
        table[column] = (
            table[column]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )
    if table[columns].isna().any().any():
        raise RuntimeError("Venus track could not be filled across every frame.")

    table["x_norm"] = table["x_norm_raw"]
    table["y_norm"] = table["y_norm_raw"]
    table["venus_radius_norm"] = table["venus_radius_norm_raw"]
    table["venus_source"] = np.where(
        table["venus_detected"], "DETECTED", "INTERPOLATED"
    )
    return table


def fit_track(table: pd.DataFrame) -> tuple[pd.DataFrame, TrackMetrics]:
    detected = table["venus_detected"].to_numpy(dtype=bool)
    quality = table["venus_quality"].to_numpy(dtype=float)
    fit_mask = detected & np.isfinite(quality) & (quality >= 0.15)
    if np.count_nonzero(fit_mask) < 5:
        fit_mask = detected
    if np.count_nonzero(fit_mask) < 5:
        fit_mask = np.ones(len(table), dtype=bool)

    x = table.loc[fit_mask, "x_norm"].to_numpy(dtype=float)
    y = table.loc[fit_mask, "y_norm"].to_numpy(dtype=float)
    points = np.column_stack((x, y))
    center = np.mean(points, axis=0)
    centered = points - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0].astype(float)

    first_point = table[["x_norm", "y_norm"]].iloc[0].to_numpy(dtype=float)
    last_point = table[["x_norm", "y_norm"]].iloc[-1].to_numpy(dtype=float)
    if np.dot(last_point - first_point, direction) < 0.0:
        direction *= -1.0
    direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]], dtype=float)

    if np.ptp(x) <= 1.0e-12:
        raise RuntimeError("Normalized x range is too small for y(x) regression.")
    ols_slope, ols_intercept = np.polyfit(x, y, 1)
    y_ols = ols_slope * x + ols_intercept
    rms_vertical = float(np.sqrt(np.mean((y - y_ols) ** 2)))
    perpendicular = centered @ normal
    rms_perpendicular = float(np.sqrt(np.mean(perpendicular ** 2)))

    along = centered @ direction
    quadratic_a, quadratic_b, quadratic_c = np.polyfit(
        along, perpendicular, 2
    )
    quadratic_fit = (
        quadratic_a * along * along
        + quadratic_b * along
        + quadratic_c
    )
    curvature_rms = float(
        np.sqrt(np.mean((perpendicular - quadratic_fit) ** 2))
    )
    curvature = float(
        2.0 * quadratic_a
        / (1.0 + quadratic_b * quadratic_b) ** 1.5
    )

    all_points = table[["x_norm", "y_norm"]].to_numpy(dtype=float)
    all_centered = all_points - center
    all_along = all_centered @ direction
    all_perpendicular = all_centered @ normal
    table = table.copy()
    table["along_track_norm"] = all_along
    table["cross_track_norm"] = all_perpendicular
    table["x_fit_norm"] = table["x_norm"]
    table["y_fit_norm"] = (
        ols_slope * table["x_norm"] + ols_intercept
    )
    table["vertical_residual_norm"] = (
        table["y_norm"] - table["y_fit_norm"]
    )
    table["quadratic_cross_track_fit_norm"] = (
        quadratic_a * all_along * all_along
        + quadratic_b * all_along
        + quadratic_c
    )
    table["curvature_residual_norm"] = (
        all_perpendicular - table["quadratic_cross_track_fit_norm"]
    )

    median_solar_radius = float(np.median(table["solar_radius_px"]))
    median_venus_radius = float(np.median(table["venus_radius_px"]))
    metrics = TrackMetrics(
        fit_count=int(np.count_nonzero(fit_mask)),
        detected_count=int(np.count_nonzero(detected)),
        total_count=int(len(table)),
        tls_center_x_norm=float(center[0]),
        tls_center_y_norm=float(center[1]),
        tls_direction_x=float(direction[0]),
        tls_direction_y=float(direction[1]),
        track_angle_deg=float(
            (np.degrees(np.arctan2(direction[1], direction[0])) + 90.0)
            % 180.0
            - 90.0
        ),
        ols_slope=float(ols_slope),
        ols_intercept=float(ols_intercept),
        ols_angle_deg=float(np.degrees(np.arctan(ols_slope))),
        rms_vertical_norm=rms_vertical,
        rms_perpendicular_norm=rms_perpendicular,
        quadratic_a=float(quadratic_a),
        quadratic_b=float(quadratic_b),
        quadratic_c=float(quadratic_c),
        curvature_norm_inverse=curvature,
        curvature_px_inverse=float(curvature / median_solar_radius),
        curvature_rms_norm=curvature_rms,
        median_solar_radius_px=median_solar_radius,
        median_venus_radius_px=median_venus_radius,
        median_venus_radius_norm=float(
            np.median(table["venus_radius_norm"])
        ),
    )
    return table, metrics


def read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(
                f"Could not read overlay frame {frame_index}."
            )
        return frame
    finally:
        cap.release()


def save_overlay(
    config: Config,
    table: pd.DataFrame,
    metrics: TrackMetrics,
    output_path: Path,
) -> int:
    detected_indices = table.index[table["venus_detected"]].to_numpy()
    representative_row = (
        int(detected_indices[len(detected_indices) // 2])
        if detected_indices.size
        else len(table) // 2
    )
    row = table.iloc[representative_row]
    frame_index = int(row["frame_index"])
    frame = read_frame(config.input_video, frame_index)
    gray = to_gray(frame)

    solar_cx = float(row["solar_cx_px"])
    solar_cy = float(row["solar_cy_px"])
    solar_radius = float(row["solar_radius_px"])
    x_overlay = solar_cx + table["x_norm"].to_numpy(dtype=float) * solar_radius
    y_overlay = solar_cy - table["y_norm"].to_numpy(dtype=float) * solar_radius

    figure, axis = plt.subplots(figsize=(8.0, 8.0), constrained_layout=True)
    axis.imshow(gray, cmap="gray", origin="upper")
    theta = np.linspace(0.0, 2.0 * np.pi, 1200)
    axis.plot(
        solar_cx + solar_radius * np.cos(theta),
        solar_cy + solar_radius * np.sin(theta),
        linewidth=0.65,
        label="Recovered solar limb",
    )
    stride = max(1, len(table) // 900)
    axis.plot(
        x_overlay[::stride],
        y_overlay[::stride],
        linestyle="None",
        marker=".",
        markersize=1.5,
        label="Frame-by-frame Venus centroid",
    )

    x_line_norm = np.linspace(
        float(table["x_norm"].min()),
        float(table["x_norm"].max()),
        500,
    )
    y_line_norm = metrics.ols_slope * x_line_norm + metrics.ols_intercept
    axis.plot(
