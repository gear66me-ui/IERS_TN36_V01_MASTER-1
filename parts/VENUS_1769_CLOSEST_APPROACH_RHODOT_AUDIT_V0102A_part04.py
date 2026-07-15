# V0102A
# Audit reference: plots, legacy-cause classification, execution, and output files

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUTPUT_DIR = os.path.abspath('.')
FIGURE_RHO = os.path.join(OUTPUT_DIR, 'VENUS_1769_CLOSEST_APPROACH_RHO_V0102A.png')
FIGURE_RHODOT = os.path.join(OUTPUT_DIR, 'VENUS_1769_CLOSEST_APPROACH_RHODOT_V0102A.png')


def diagnose_legacy(series: SampleSeries, result: AuditResult) -> str:
    origin = series.epochs[0]
    legacy_s = seconds_difference(LEGACY_CA, origin)
    legacy_dot = interpolate_linear(series.seconds, result.raw_rhodot_rad_s, legacy_s)
    root_offset = result.rho_root_seconds-legacy_s
    min_offset = result.minimum_seconds-legacy_s
    root_min_gap = abs(result.rho_root_seconds-result.minimum_seconds)
    scale_signature = abs(root_offset-min_offset)
    print(f'Legacy drho/dt                   {legacy_dot*ARCSEC_PER_RAD:.12e} arcsec/s')
    print(f'Legacy to drho/dt zero           {root_offset:.9f} s')
    print(f'Legacy to rho minimum            {min_offset:.9f} s')
    print(f'Root/minimum internal gap        {root_min_gap:.9f} s')
    print(f'Time-shift signature residual    {scale_signature:.9f} s')
    if abs(legacy_dot) > 1.0e-12 and root_min_gap < 0.25:
        conclusion = (
            'B CONFIRMED: V0102 reported a CA epoch that is not stationary, while its '
            'derivative zero belongs to the later stationary epoch. A common time-scale '
            'conversion is rejected because it would shift both reported epochs equally.'
        )
    elif root_min_gap >= 0.25:
        conclusion = (
            'C CONFIRMED: independently calculated minimum and derivative root disagree, '
            'demonstrating inconsistent interpolation or differentiation.'
        )
    else:
        conclusion = 'No legacy discrepancy detected by the numerical tests.'
    print(f'Critical audit conclusion        {conclusion}')
    return conclusion


def plot_rho(series: SampleSeries, result: AuditResult) -> None:
    ca = epoch_from_seconds(series.epochs[0], result.minimum_seconds)
    rho_min = result.minimum_rho_rad*ARCSEC_PER_RAD
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=160)
    ax.plot(series.epochs, series.rho_arcsec, linewidth=0.85, label=r'$\rho(t)$')
    ax.axvline(ca, linewidth=0.75, linestyle='--')
    ax.scatter([ca], [rho_min], s=14, zorder=4)
    ax.annotate(f'{format_utc(ca)}\n{rho_min:.9f} arcsec',
                xy=(ca, rho_min), xytext=(12, 30), textcoords='offset points',
                arrowprops={'arrowstyle':'-', 'linewidth':0.6}, fontsize=8)
    ax.set_title('1769 Venus Transit — Geocentric Angular Separation')
    ax.set_xlabel('UTC')
    ax.set_ylabel(r'$\rho$ (arcsec)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(True, linewidth=0.35, alpha=0.45)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_RHO, dpi=300, bbox_inches='tight')
    plt.show()


def plot_rhodot(series: SampleSeries, result: AuditResult) -> None:
    root = epoch_from_seconds(series.epochs[0], result.rho_root_seconds)
    raw = [v*ARCSEC_PER_RAD for v in result.raw_rhodot_rad_s]
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=160)
    ax.plot(series.epochs, raw, linewidth=0.75, label='Raw')
    for width in (3, 5, 7, 9):
        values = [v*ARCSEC_PER_RAD for v in result.rolling_rhodot_rad_s[width]]
        ax.plot(series.epochs, values, linewidth=0.65, label=f'Rolling mean {width}')
    ax.axhline(0.0, linewidth=0.55)
    ax.axvline(root, linewidth=0.75, linestyle='--')
    ax.annotate(format_utc(root), xy=(root, 0.0), xytext=(12, 28),
                textcoords='offset points', arrowprops={'arrowstyle':'-', 'linewidth':0.6},
                fontsize=8)
    ax.set_title(r'1769 Venus Transit — Geocentric $d\rho/dt$')
    ax.set_xlabel('UTC')
    ax.set_ylabel(r'$d\rho/dt$ (arcsec/s)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(True, linewidth=0.35, alpha=0.45)
    ax.legend(frameon=False, ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_RHODOT, dpi=300, bbox_inches='tight')
    plt.show()


def main() -> None:
    eph = fetch_ephemeris()
    series = build_series(eph)
    result = evaluate_audit(series)
    print_audit(series, result)
    diagnose_legacy(series, result)
    plot_rho(series, result)
    plot_rhodot(series, result)
    print(f'Figure 1                         {FIGURE_RHO}')
    print(f'Figure 2                         {FIGURE_RHODOT}')
    print(dt.datetime.now().astimezone().isoformat(timespec='seconds'))
    print('V0102A COMPLETE')


if __name__ == '__main__':
    main()
# V0102A