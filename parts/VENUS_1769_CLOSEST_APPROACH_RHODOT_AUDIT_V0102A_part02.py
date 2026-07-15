# V0102A
# Audit reference: angular separation, interpolation, derivatives, and roots

ARCSEC_PER_RAD = 206264.80624709636
LEGACY_CA = dt.datetime(1769, 6, 3, 22, 18, 59, 487000)

@dataclass
class SampleSeries:
    epochs: List[dt.datetime]
    seconds: List[float]
    rho_rad: List[float]
    rho_arcsec: List[float]
    rho2: List[float]


def clamp_unit(value: float) -> float:
    return max(-1.0, min(1.0, value))


def angular_separation(sun_vec: Vec3, venus_vec: Vec3) -> float:
    denominator = sun_vec.norm() * venus_vec.norm()
    if denominator == 0.0:
        raise ZeroDivisionError('Zero-length JPL vector encountered')
    cosine = clamp_unit(sun_vec.dot(venus_vec) / denominator)
    return math.acos(cosine)


def build_series(eph: HorizonsEphemeris) -> SampleSeries:
    if not eph.epochs or len(eph.sun) != len(eph.venus):
        raise RuntimeError('Incomplete paired Sun/Venus vector series')
    count = min(len(eph.epochs), len(eph.sun), len(eph.venus))
    epochs = eph.epochs[:count]
    origin = epochs[0]
    seconds = [(epoch-origin).total_seconds() for epoch in epochs]
    rho_rad = [angular_separation(eph.sun[i], eph.venus[i]) for i in range(count)]
    rho_arcsec = [value * ARCSEC_PER_RAD for value in rho_rad]
    rho2 = [value * value for value in rho_rad]
    return SampleSeries(epochs, seconds, rho_rad, rho_arcsec, rho2)


def derivative(values: List[float], seconds: List[float]) -> List[float]:
    n = len(values)
    if n < 3 or n != len(seconds):
        raise ValueError('Derivative requires at least three aligned samples')
    out = [0.0] * n
    out[0] = (values[1]-values[0])/(seconds[1]-seconds[0])
    out[-1] = (values[-1]-values[-2])/(seconds[-1]-seconds[-2])
    for i in range(1, n-1):
        out[i] = (values[i+1]-values[i-1])/(seconds[i+1]-seconds[i-1])
    return out


def rolling_mean(values: List[float], width: int) -> List[float]:
    if width < 1 or width % 2 == 0:
        raise ValueError('Rolling width must be a positive odd integer')
    half = width // 2
    result = []
    for i in range(len(values)):
        lo = max(0, i-half)
        hi = min(len(values), i+half+1)
        result.append(sum(values[lo:hi])/(hi-lo))
    return result


def bracket_zero(seconds: List[float], values: List[float], near_index: int) -> tuple:
    candidates = []
    for i in range(len(values)-1):
        if values[i] == 0.0:
            return seconds[i], seconds[i]
        if values[i] * values[i+1] < 0.0:
            distance = abs(i-near_index)
            candidates.append((distance, seconds[i], seconds[i+1]))
    if not candidates:
        raise RuntimeError('No derivative zero crossing found')
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], candidates[0][2]


def linear_zero_time(t0: float, t1: float, y0: float, y1: float) -> float:
    if y1 == y0:
        return 0.5*(t0+t1)
    return t0-y0*(t1-t0)/(y1-y0)


def zero_crossing(seconds: List[float], values: List[float], near_index: int) -> float:
    t0, t1 = bracket_zero(seconds, values, near_index)
    if t0 == t1:
        return t0
    i = seconds.index(t0)
    return linear_zero_time(t0, t1, values[i], values[i+1])


def quadratic_vertex(x0: float, y0: float, x1: float, y1: float,
                     x2: float, y2: float) -> tuple:
    d01 = (x0-x1); d02 = (x0-x2); d12 = (x1-x2)
    a = y0/(d01*d02) + y1/((-d01)*d12) + y2/((-d02)*(-d12))
    b = -y0*(x1+x2)/(d01*d02) - y1*(x0+x2)/((-d01)*d12) \
        - y2*(x0+x1)/((-d02)*(-d12))
    c = y0-a*x0*x0-b*x0
    xv = -b/(2.0*a)
    yv = a*xv*xv+b*xv+c
    return xv, yv, a, b, c


def minimum_from_three(seconds: List[float], values: List[float]) -> tuple:
    i = min(range(len(values)), key=values.__getitem__)
    if i == 0 or i == len(values)-1:
        raise RuntimeError('Minimum lies at series boundary')
    return quadratic_vertex(seconds[i-1], values[i-1],
                            seconds[i], values[i],
                            seconds[i+1], values[i+1])


def epoch_from_seconds(origin: dt.datetime, seconds: float) -> dt.datetime:
    return origin + dt.timedelta(seconds=float(seconds))

# V0102A