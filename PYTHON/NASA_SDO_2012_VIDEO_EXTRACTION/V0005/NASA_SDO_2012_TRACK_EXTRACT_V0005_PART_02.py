        else:
            size_penalty = abs(
                math.log(max(equivalent_radius, 1.0) / max(predicted.radius, 1.0))
            )
            distance_penalty = (
                math.hypot(cx - predicted.cx, cy - predicted.cy)
                / max(config.venus_search_ratio * solar.radius, 1.0)
            )
        score = (
            0.40 * math.tanh(max(0.0, darkness_z) / 4.0)
            + 0.35 * circularity
            + 0.25 * math.exp(-size_penalty)
            - 0.35 * distance_penalty
        )
        results.append(
            CircleResult(cx, cy, equivalent_radius, float(score), "COMPONENT")
        )
    return results


def refine_venus_subpixel(
    gray: np.ndarray,
    darkness: np.ndarray,
    initial: CircleResult,
    solar: CircleResult,
    config: Config,
) -> CircleResult:
    radius0 = float(
        np.clip(
            initial.radius,
            config.venus_radius_min_ratio * solar.radius,
            config.venus_radius_max_ratio * solar.radius,
        )
    )
    half = int(math.ceil(max(6.0, 2.4 * radius0)))
    x0 = max(0, int(math.floor(initial.cx)) - half)
    x1 = min(gray.shape[1], int(math.floor(initial.cx)) + half + 1)
    y0 = max(0, int(math.floor(initial.cy)) - half)
    y1 = min(gray.shape[0], int(math.floor(initial.cy)) + half + 1)
    if x1 - x0 < 7 or y1 - y0 < 7:
        raise RuntimeError("Venus refinement patch is too small.")

    patch_dark = darkness[y0:y1, x0:x1].astype(float)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    local_median, local_sigma = robust_scale(patch_dark)
    weights = np.clip(patch_dark - local_median, 0.0, None)
    cutoff = np.percentile(weights, 55.0)
    weights[weights < cutoff] = 0.0
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise RuntimeError("Venus subpixel weights are empty.")
    cx_weighted = float(np.sum(weights * xx) / weight_sum)
    cy_weighted = float(np.sum(weights * yy) / weight_sum)

    patch = gray[y0:y1, x0:x1].astype(np.float32)
    grad_x = cv2.Scharr(patch, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(patch, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(grad_x, grad_y).astype(float)
    distance = np.hypot(xx - cx_weighted, yy - cy_weighted)
    annulus = (distance >= 0.45 * radius0) & (distance <= 1.65 * radius0)
    if np.count_nonzero(annulus) < 12:
        raise RuntimeError("Venus annulus has too few pixels.")
    gradient_cutoff = float(np.percentile(gradient[annulus], 70.0))
    edge_mask = annulus & (gradient >= gradient_cutoff)
    edge_points = np.column_stack((xx[edge_mask], yy[edge_mask]))
    if edge_points.shape[0] < 12:
        raise RuntimeError("Venus edge fit has too few points.")

    circle = robust_circle(edge_points)
    min_radius = config.venus_radius_min_ratio * solar.radius
    max_radius = config.venus_radius_max_ratio * solar.radius
    if not (min_radius <= circle.radius <= max_radius):
        circle.radius = radius0
    center_shift = math.hypot(circle.cx - cx_weighted, circle.cy - cy_weighted)
    if center_shift > 0.75 * radius0:
        circle.cx = cx_weighted
        circle.cy = cy_weighted

    disk = distance <= max(circle.radius, radius0)
    ring = (distance >= 1.25 * max(circle.radius, radius0)) & (
        distance <= 1.85 * max(circle.radius, radius0)
    )
    if np.any(disk) and np.any(ring):
        contrast = float(np.median(patch[ring]) - np.median(patch[disk]))
        contrast_quality = math.tanh(max(0.0, contrast) / max(local_sigma, 1.0))
    else:
        contrast_quality = 0.0
    circle.quality = float(
        np.clip(
            0.35 * initial.quality
            + 0.35 * circle.quality
            + 0.30 * contrast_quality,
            0.0,
            1.0,
        )
    )
    circle.method = "SUBPIXEL_WEIGHTED_CENTROID+EDGE_CIRCLE"
    return circle


def detect_venus(
    gray: np.ndarray,
    solar: CircleResult,
    predicted: Optional[CircleResult],
    config: Config,
    force_global: bool,
) -> CircleResult:
    darkness = estimate_darkness(gray, solar.radius)
    allowed = solar_mask(gray.shape, solar, config.venus_inner_limit)
    if predicted is not None and not force_global:
        yy, xx = np.ogrid[:gray.shape[0], :gray.shape[1]]
        search_radius = max(
            config.venus_search_ratio * solar.radius,
            5.0 * predicted.radius,
        )
        local = (
            (xx - predicted.cx) ** 2 + (yy - predicted.cy) ** 2
            <= search_radius ** 2
        )
        allowed &= local

    candidates = candidate_components(
        darkness, allowed, solar, None if force_global else predicted, config
    )
    if not candidates and predicted is not None and not force_global:
        candidates = candidate_components(
            darkness,
            solar_mask(gray.shape, solar, config.venus_inner_limit),
            solar,
            None,
            config,
        )
    if not candidates:
        raise RuntimeError("No Venus candidate was found.")

    candidates.sort(key=lambda item: item.quality, reverse=True)
    errors: list[str] = []
    for candidate in candidates[:12]:
        try:
            refined = refine_venus_subpixel(
                gray, darkness, candidate, solar, config
            )
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        solar_distance = math.hypot(
            refined.cx - solar.cx, refined.cy - solar.cy
        )
        if solar_distance + refined.radius > config.venus_inner_limit * solar.radius:
            continue
        if refined.quality >= config.min_venus_quality:
            return refined
    raise RuntimeError(
        "Venus candidates failed subpixel refinement: " + "; ".join(errors[:3])
    )


def video_metadata(cap: cv2.VideoCapture) -> dict[str, float]:
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 30.0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_s": frame_count / fps if frame_count > 0 else float("nan"),
    }


def process_video(config: Config) -> tuple[pd.DataFrame, dict[str, float]]:
    if not config.input_video.exists():
        raise FileNotFoundError(f"Input video not found: {config.input_video}")
    cap = cv2.VideoCapture(str(config.input_video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open: {config.input_video}")
    metadata = video_metadata(cap)
    cap.set(cv2.CAP_PROP_POS_FRAMES, config.start_frame)

    records: list[FrameRecord] = []
    solar_previous: Optional[CircleResult] = None
    venus_history: list[CircleResult] = []
    failures = 0
    frame_index = config.start_frame
    processed = 0
    bootstrap_skipped = 0
    first_processed_frame: Optional[int] = None
    last_processed_frame: Optional[int] = None
    maximum_bootstrap_skip = min(900, max(120, int(metadata["frame_count"] * 0.20)))

    try:
        while True:
            if config.end_frame is not None and frame_index > config.end_frame:
                break
            if config.max_frames is not None and processed >= config.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            gray = to_gray(frame)

            need_global = (
                solar_previous is None
                or processed % config.global_limb_retry == 0
            )
            try:
                if need_global:
                    solar_raw = detect_solar_limb_global(gray)
                else:
                    solar_raw = refine_solar_limb(
                        gray, solar_previous, config.radial_samples
                    )
                solar = (
                    solar_raw
                    if solar_previous is None
                    else smooth_circle(
                        solar_previous, solar_raw, config.solar_smoothing
                    )
                )
            except RuntimeError:
                if solar_previous is None:
                    bootstrap_skipped += 1
                    frame_index += 1
                    if bootstrap_skipped > maximum_bootstrap_skip:
                        raise RuntimeError(
                            "Solar limb was not found during the allowed bootstrap window."
                        )
                    continue
                solar = CircleResult(
                    solar_previous.cx,
                    solar_previous.cy,
                    solar_previous.radius,
                    0.0,
                    "SOLAR_FALLBACK_PREVIOUS",
                )

            if solar_previous is None:
                first_processed_frame = frame_index
                print(
                    "DEBUG | first solar frame acquired | "
                    f"frame={frame_index} | "
                    f"title_or_blank_frames_skipped={bootstrap_skipped}"
                )
            solar_previous = solar

            predicted = predict_venus(venus_history)
            force_global = predicted is None or failures >= config.venus_reacquire_after
            detected = False
            source = "DETECTED"
            try:
                venus = detect_venus(
                    gray, solar, predicted, config, force_global=force_global
                )
                detected = True
                failures = 0
                venus_history.append(venus)
                venus_history = venus_history[-3:]
            except RuntimeError:
                failures += 1
                if predicted is not None:
                    venus = predicted
                    venus.quality = 0.0
                    venus.method = "PREDICTED_AFTER_FAILURE"
                    source = "PREDICTED"
                else:
                    venus = CircleResult(np.nan, np.nan, np.nan, 0.0, "MISSING")
                    source = "MISSING"

            pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            time_s = (
                pos_msec / 1000.0
                if np.isfinite(pos_msec) and pos_msec >= 0.0
                else frame_index / metadata["fps"]
            )
            records.append(
                FrameRecord(
                    frame_index=frame_index,
                    time_s=time_s,
                    solar_cx_px=solar.cx,
                    solar_cy_px=solar.cy,
                    solar_radius_px=solar.radius,
                    solar_quality=solar.quality,
                    solar_method=solar.method,
                    venus_cx_px=venus.cx,
                    venus_cy_px=venus.cy,
                    venus_radius_px=venus.radius,
                    venus_quality=venus.quality,
                    venus_detected=detected,
                    venus_source=source,
                )
            )
            last_processed_frame = frame_index
            frame_index += 1
            processed += 1
            if processed % 500 == 0:
                print(
                    "DEBUG | video progress | "
                    f"processed={processed} | frame={last_processed_frame}"
                )
    finally:
        cap.release()

    if not records:
        raise RuntimeError("No video frames were processed.")
    metadata["bootstrap_frames_skipped"] = int(bootstrap_skipped)
    metadata["first_processed_frame"] = int(first_processed_frame or records[0].frame_index)
    metadata["last_processed_frame"] = int(last_processed_frame or records[-1].frame_index)
    frame_table = pd.DataFrame([asdict(record) for record in records])
    return frame_table, metadata

def select_continuous_venus_track(frame_table: pd.DataFrame) -> pd.DataFrame:
    table = frame_table.copy()
    required = [
        "frame_index", "solar_cx_px", "solar_cy_px", "solar_radius_px",
        "venus_cx_px", "venus_cy_px", "venus_radius_px",
        "venus_quality", "venus_detected",
    ]
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise RuntimeError(f"Trajectory filter missing columns: {missing}")

    table["venus_detected_raw"] = table["venus_detected"].astype(bool)
    for name in ("venus_cx_px", "venus_cy_px", "venus_radius_px", "venus_quality"):
        table[f"{name}_raw_detector"] = table[name]

    solar_radius = table["solar_radius_px"].to_numpy(dtype=float)
    x_norm = (
        table["venus_cx_px"].to_numpy(dtype=float)
        - table["solar_cx_px"].to_numpy(dtype=float)
    ) / solar_radius
    y_norm = (
        table["solar_cy_px"].to_numpy(dtype=float)
        - table["venus_cy_px"].to_numpy(dtype=float)
    ) / solar_radius
    radius_norm = table["venus_radius_px"].to_numpy(dtype=float) / solar_radius
    quality = table["venus_quality"].to_numpy(dtype=float)
    detected = table["venus_detected"].to_numpy(dtype=bool)
    frame = table["frame_index"].to_numpy(dtype=float)

    finite = (
        detected & np.isfinite(x_norm) & np.isfinite(y_norm)
        & np.isfinite(radius_norm) & np.isfinite(quality)
        & np.isfinite(frame) & (solar_radius > 0.0)
    )
    if np.count_nonzero(finite) < 80:
        raise RuntimeError("Too few finite detections for trajectory filtering.")

