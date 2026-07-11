    x = points[:, 0]
    y = points[:, 1]
    design = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    rhs = x * x + y * y
    solution, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    cx, cy, c = solution
    radius_sq = c + cx * cx + cy * cy
    if radius_sq <= 0.0:
        raise ValueError("Circle fit produced a non-positive radius.")
    return float(cx), float(cy), float(math.sqrt(radius_sq))


def robust_circle(points_xy: np.ndarray) -> CircleResult:
    points = np.asarray(points_xy, dtype=float)
    cx0, cy0, r0 = algebraic_circle(points)

    def residuals(params: np.ndarray) -> np.ndarray:
        cx, cy, radius = params
        distances = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
        return distances - radius

    result = least_squares(
        residuals,
        x0=np.array([cx0, cy0, r0], dtype=float),
        loss="soft_l1",
        f_scale=max(0.5, 0.002 * r0),
        bounds=([-np.inf, -np.inf, 1.0], [np.inf, np.inf, np.inf]),
        max_nfev=120,
    )
    cx, cy, radius = result.x
    rms = float(np.sqrt(np.mean(residuals(result.x) ** 2)))
    quality = float(1.0 / (1.0 + rms / max(radius, 1.0)))
    return CircleResult(float(cx), float(cy), float(radius), quality, "ROBUST_CIRCLE")


def sample_ring_mean(gray: np.ndarray, cx: float, cy: float, radius: float) -> float:
    angles = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    map_x = (cx + radius * np.cos(angles)).astype(np.float32).reshape(1, -1)
    map_y = (cy + radius * np.sin(angles)).astype(np.float32).reshape(1, -1)
    values = cv2.remap(
        gray, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )
    return float(np.mean(values))


def solar_candidate_score(
    gray: np.ndarray, cx: float, cy: float, radius: float
) -> float:
    height, width = gray.shape
    min_dim = float(min(height, width))
    if not (0.20 * min_dim <= radius <= 0.58 * min_dim):
        return -np.inf
    center_distance = math.hypot(cx - 0.5 * width, cy - 0.5 * height)
    if center_distance > 0.34 * min_dim:
        return -np.inf
    inside = sample_ring_mean(gray, cx, cy, 0.88 * radius)
    limb_in = sample_ring_mean(gray, cx, cy, 0.98 * radius)
    outside = sample_ring_mean(gray, cx, cy, 1.04 * radius)
    _, sigma = robust_scale(gray)
    brightness_term = (inside - outside) / sigma
    edge_term = abs(limb_in - outside) / sigma
    center_term = 1.0 - center_distance / (0.34 * min_dim)
    return float(brightness_term + edge_term + 2.0 * center_term)


def detect_solar_limb_global(gray: np.ndarray) -> CircleResult:
    height, width = gray.shape
    min_dim = float(min(height, width))
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blurred = cv2.GaussianBlur(
        normalized, (0, 0), sigmaX=max(1.0, min_dim / 500.0)
    )
    candidates: list[tuple[float, float, float, str]] = []

    for percentile in (45.0, 55.0, 65.0, 75.0):
        threshold = float(np.percentile(blurred, percentile))
        binary = np.where(blurred >= threshold, 255, 0).astype(np.uint8)
        kernel_size = max(3, int(round(min_dim * 0.012)) | 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
            area = float(cv2.contourArea(contour))
            if not (0.05 * height * width <= area <= 0.92 * height * width):
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0.0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.35:
                continue
            points = contour[:, 0, :].astype(float)
            try:
                circle = robust_circle(points[:: max(1, len(points) // 1800)])
            except (ValueError, np.linalg.LinAlgError):
                continue
            candidates.append(
                (circle.cx, circle.cy, circle.radius, f"PERCENTILE_{percentile:.0f}")
            )

    edges = cv2.Canny(blurred, 35, 110)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        points = contour[:, 0, :].astype(float)
        if len(points) < 80:
            continue
        try:
            circle = robust_circle(points[:: max(1, len(points) // 1800)])
        except (ValueError, np.linalg.LinAlgError):
            continue
        candidates.append((circle.cx, circle.cy, circle.radius, "CANNY"))

    scale = min(1.0, 900.0 / max(height, width))
    small = cv2.resize(blurred, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for param2 in (45, 34, 26):
        hough = cv2.HoughCircles(
            small,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(20.0, 0.18 * min(small.shape)),
            param1=90,
            param2=param2,
            minRadius=max(5, int(0.16 * min(small.shape))),
            maxRadius=int(0.64 * min(small.shape)),
        )
        if hough is not None:
            for x_small, y_small, r_small in hough[0, :12]:
                candidates.append(
                    (
                        float(x_small / scale),
                        float(y_small / scale),
                        float(r_small / scale),
                        f"HOUGH_{param2}",
                    )
                )

    if not candidates:
        raise RuntimeError("Solar limb could not be initialized.")

    validated: list[tuple[float, CircleResult, str]] = []
    for cx, cy, radius, method in candidates:
        if not (0.16 * min_dim <= radius <= 0.64 * min_dim):
            continue
        if not (-0.10 * width <= cx <= 1.10 * width):
            continue
        if not (-0.10 * height <= cy <= 1.10 * height):
            continue
        try:
            score = solar_candidate_score(gray, cx, cy, radius)
        except Exception:
            continue
        if not np.isfinite(score):
            continue
        try:
            refined = refine_solar_limb(
                gray,
                CircleResult(cx, cy, radius, 0.0, method),
                radial_samples=720,
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
        inside = sample_ring_mean(gray, refined.cx, refined.cy, 0.72 * refined.radius)
        outside = sample_ring_mean(gray, refined.cx, refined.cy, 1.12 * refined.radius)
        _, sigma_gray = robust_scale(gray)
        contrast = (inside - outside) / max(sigma_gray, 1.0e-9)
        center_distance = math.hypot(
            refined.cx - 0.5 * width, refined.cy - 0.5 * height
        ) / min_dim
        validation = float(score + 1.5 * contrast - 2.0 * center_distance)
        if contrast <= 0.15:
            continue
        refined.method = f"GLOBAL_{method}+{refined.method}"
        refined.quality = float(np.clip(0.5 + 0.08 * validation, 0.0, 1.0))
        validated.append((validation, refined, method))

    if not validated:
        raise RuntimeError("Solar limb candidates failed filled-disk validation.")
    validated.sort(key=lambda item: item[0], reverse=True)
    return validated[0][1]

def refine_solar_limb(
    gray: np.ndarray, prior: CircleResult, radial_samples: int
) -> CircleResult:
    angles = np.linspace(0.0, 2.0 * np.pi, radial_samples, endpoint=False)
    radial_offsets = np.linspace(-0.09, 0.09, 91)
    radii = prior.radius * (1.0 + radial_offsets)
    map_x = (
        prior.cx
        + np.cos(angles)[:, None] * radii[None, :]
    ).astype(np.float32)
    map_y = (
        prior.cy
        + np.sin(angles)[:, None] * radii[None, :]
    ).astype(np.float32)
    profiles = cv2.remap(
        gray,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    profiles = cv2.GaussianBlur(profiles, (0, 0), sigmaX=1.0, sigmaY=0.0)
    gradients = np.gradient(profiles.astype(float), axis=1)
    indices = np.argmin(gradients, axis=1)
    edge_strength = -gradients[np.arange(radial_samples), indices]
    median_strength, sigma_strength = robust_scale(edge_strength)
    keep = edge_strength > median_strength - 0.5 * sigma_strength
    chosen_radii = radii[indices]
    points = np.column_stack(
        (
            prior.cx + chosen_radii * np.cos(angles),
            prior.cy + chosen_radii * np.sin(angles),
        )
    )
    points = points[keep]
    if points.shape[0] < max(40, radial_samples // 5):
        raise RuntimeError("Solar limb refinement had too few edge points.")
    circle = robust_circle(points)
    edge_snr = float(
        max(0.0, (np.median(edge_strength[keep]) - median_strength) / sigma_strength)
    )
    circle.quality = float(
        np.clip(0.55 * circle.quality + 0.45 * math.tanh(edge_snr + 0.5), 0.0, 1.0)
    )
    circle.method = "RADIAL_GRADIENT"
    return circle


def smooth_circle(
    previous: CircleResult, current: CircleResult, alpha: float
) -> CircleResult:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return CircleResult(
        cx=(1.0 - alpha) * previous.cx + alpha * current.cx,
        cy=(1.0 - alpha) * previous.cy + alpha * current.cy,
        radius=(1.0 - alpha) * previous.radius + alpha * current.radius,
        quality=current.quality,
        method=current.method,
    )


def solar_mask(
    shape: tuple[int, int], circle: CircleResult, radius_ratio: float
) -> np.ndarray:
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    return (
        (xx - circle.cx) ** 2 + (yy - circle.cy) ** 2
        <= (radius_ratio * circle.radius) ** 2
    )


def estimate_darkness(gray: np.ndarray, solar_radius: float) -> np.ndarray:
    sigma = max(2.0, 0.030 * solar_radius)
    background = cv2.GaussianBlur(
        gray, (0, 0), sigmaX=sigma, sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101
    )
    darkness = background.astype(np.float32) - gray.astype(np.float32)
    darkness[darkness < 0.0] = 0.0
    return darkness


def component_circularity(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0.0:
        return 0.0
    return float(np.clip(4.0 * np.pi * area / (perimeter * perimeter), 0.0, 1.0))


def predict_venus(
    history: list[CircleResult], frame_step: int = 1
) -> Optional[CircleResult]:
    if not history:
        return None
    if len(history) == 1:
        last = history[-1]
        return CircleResult(last.cx, last.cy, last.radius, last.quality, "PREDICT_HOLD")
    first = history[-2]
    last = history[-1]
    return CircleResult(
        cx=last.cx + frame_step * (last.cx - first.cx),
        cy=last.cy + frame_step * (last.cy - first.cy),
        radius=last.radius,
        quality=last.quality,
        method="PREDICT_VELOCITY",
    )


def candidate_components(
    darkness: np.ndarray,
    allowed_mask: np.ndarray,
    solar: CircleResult,
    predicted: Optional[CircleResult],
    config: Config,
) -> list[CircleResult]:
    mask = np.asarray(allowed_mask, dtype=bool)
    if mask.shape != darkness.shape:
        raise ValueError(
            "Venus candidate mask shape does not match the darkness image."
        )
    values = np.asarray(darkness[mask], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return []
    median, sigma = robust_scale(values)
    percentile = 97.0 if predicted is None else 92.0
    threshold = max(
        median + (2.5 if predicted is None else 1.5) * sigma,
        float(np.percentile(values, percentile)),
    )
    binary = np.zeros_like(darkness, dtype=np.uint8)
    binary[(darkness >= threshold) & allowed_mask] = 255
    kernel_radius = max(1, int(round(0.004 * solar.radius)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * kernel_radius + 1, 2 * kernel_radius + 1)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    min_radius = config.venus_radius_min_ratio * solar.radius
    max_radius = config.venus_radius_max_ratio * solar.radius
    results: list[CircleResult] = []

    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        equivalent_radius = math.sqrt(area / np.pi)
        if not (0.45 * min_radius <= equivalent_radius <= 1.8 * max_radius):
            continue
        cx, cy = map(float, centroids[label])
        local = labels == label
        circularity = component_circularity(local)
        mean_darkness = float(np.mean(darkness[local]))
        darkness_z = (mean_darkness - median) / sigma
        if predicted is None:
            expected_radius = 0.035 * solar.radius
            size_penalty = abs(math.log(max(equivalent_radius, 1.0) / expected_radius))
            distance_penalty = 0.0
