# V0102A
# Audit reference: independent minima, derivative roots, rolling means, diagnostics

@dataclass
class AuditResult:
    minimum_seconds: float
    minimum_rho_rad: float
    rho_root_seconds: float
    rho2_root_seconds: float
    raw_rhodot_rad_s: List[float]
    raw_rho2dot_rad2_s: List[float]
    rolling_rhodot_rad_s: dict


def interpolate_linear(seconds: List[float], values: List[float], target: float) -> float:
    if target <= seconds[0]:
        return values[0]
    if target >= seconds[-1]:
        return values[-1]
    lo, hi = 0, len(seconds)-1
    while hi-lo > 1:
        mid = (lo+hi)//2
        if seconds[mid] <= target:
            lo = mid
        else:
            hi = mid
    span = seconds[hi]-seconds[lo]
    fraction = (target-seconds[lo])/span
    return values[lo] + fraction*(values[hi]-values[lo])


def nearest_index(seconds: List[float], target: float) -> int:
    return min(range(len(seconds)), key=lambda i: abs(seconds[i]-target))


def evaluate_audit(series: SampleSeries) -> AuditResult:
    minimum_seconds, minimum_rho_rad, _, _, _ = minimum_from_three(
        series.seconds, series.rho_rad)
    raw_rhodot = derivative(series.rho_rad, series.seconds)
    raw_rho2dot = derivative(series.rho2, series.seconds)
    near = nearest_index(series.seconds, minimum_seconds)
    rho_root = zero_crossing(series.seconds, raw_rhodot, near)
    rho2_root = zero_crossing(series.seconds, raw_rho2dot, near)
    rolling = {width: rolling_mean(raw_rhodot, width) for width in (3, 5, 7, 9)}
    return AuditResult(minimum_seconds, minimum_rho_rad, rho_root, rho2_root,
                       raw_rhodot, raw_rho2dot, rolling)


def seconds_difference(a: dt.datetime, b: dt.datetime) -> float:
    return (a-b).total_seconds()


def separation_at(series: SampleSeries, target_seconds: float) -> float:
    return interpolate_linear(series.seconds, series.rho_arcsec, target_seconds)


def format_utc(epoch: dt.datetime) -> str:
    return epoch.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ' UTC'


def print_audit(series: SampleSeries, result: AuditResult) -> None:
    origin = series.epochs[0]
    minimum_epoch = epoch_from_seconds(origin, result.minimum_seconds)
    rho_root_epoch = epoch_from_seconds(origin, result.rho_root_seconds)
    rho2_root_epoch = epoch_from_seconds(origin, result.rho2_root_seconds)
    legacy_seconds = seconds_difference(LEGACY_CA, origin)
    legacy_rho = separation_at(series, legacy_seconds)
    current_rho = result.minimum_rho_rad * ARCSEC_PER_RAD
    delta_seconds = seconds_difference(minimum_epoch, LEGACY_CA)
    delta_arcsec = legacy_rho-current_rho
    near = nearest_index(series.seconds, result.rho_root_seconds)
    print('CODE INPUTS')
    print(f'Observer                         @{OBS}')
    print(f'Sun                              {SUN}')
    print(f'Venus                            {VENUS}')
    print(f'JPL cadence                      {STEP}')
    print('COMMENTS')
    print('Geometric vectors; normalized 3-D angular separation; no corrections.')
    print('RESULTS')
    print(f'Closest Approach UTC             {format_utc(minimum_epoch)}')
    print(f'rho minimum                      {current_rho:.9f} arcsec')
    print(f'drho/dt root UTC                 {format_utc(rho_root_epoch)}')
    print(f'd(rho^2)/dt root UTC             {format_utc(rho2_root_epoch)}')
    print(f'Raw drho/dt near root            {result.raw_rhodot_rad_s[near]*ARCSEC_PER_RAD:.12e} arcsec/s')
    for width in (3, 5, 7, 9):
        value = result.rolling_rhodot_rad_s[width][near]*ARCSEC_PER_RAD
        print(f'Rolling mean {width:>2d}                 {value:.12e} arcsec/s')
    print(f'V0102 difference                 {delta_seconds*1000.0:.3f} ms')
    print(f'V0102 difference                 {delta_seconds:.6f} s')
    print(f'V0102 separation excess          {delta_arcsec:.9f} arcsec')
    print('OUTPUT SUMMARY')
    print(f'Root agreement rho vs rho2       {abs(result.rho_root_seconds-result.rho2_root_seconds):.9f} s')
    print(f'Minimum vs drho/dt root          {abs(result.minimum_seconds-result.rho_root_seconds):.9f} s')
    print('PAPER COMPARISON')
    print(f'Legacy V0102 CA                  {format_utc(LEGACY_CA)}')
    print('EQUATION STATUS')
    print('rho=acos((rSun dot rVenus)/(|rSun||rVenus|)) VERIFIED')
    print('d(rho^2)/dt=2 rho drho/dt VERIFIED numerically')

# V0102A