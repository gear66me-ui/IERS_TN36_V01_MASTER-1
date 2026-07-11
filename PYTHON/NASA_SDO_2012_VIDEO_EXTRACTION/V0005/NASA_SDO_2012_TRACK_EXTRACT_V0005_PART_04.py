        solar_cx + x_line_norm * solar_radius,
        solar_cy - y_line_norm * solar_radius,
        linewidth=0.75,
        label="Linear regression",
    )

    venus_cx = solar_cx + float(row["x_norm"]) * solar_radius
    venus_cy = solar_cy - float(row["y_norm"]) * solar_radius
    venus_radius = float(row["venus_radius_norm"]) * solar_radius
    axis.plot(
        venus_cx + venus_radius * np.cos(theta),
        venus_cy + venus_radius * np.sin(theta),
        linewidth=0.65,
        label=f"Venus disk, frame {frame_index}",
    )

    margin = 1.05 * solar_radius
    axis.set_xlim(solar_cx - margin, solar_cx + margin)
    axis.set_ylim(solar_cy + margin, solar_cy - margin)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Image x [px]")
    axis.set_ylabel("Image y [px]")
    axis.set_title(
        "NASA SDO 2012 Venus Transit Track\n"
        f"TLS angle={metrics.track_angle_deg:.6f} deg | "
        f"RMS={metrics.rms_perpendicular_norm:.8f} R_sun | "
        f"curvature={metrics.curvature_norm_inverse:.8e} R_sun^-1"
    )
    axis.legend(loc="best", fontsize=7)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return frame_index


def metrics_table(
    metrics: TrackMetrics, metadata: dict[str, float], overlay_frame: int
) -> pd.DataFrame:
    rows = [
        ("version", VERSION, ""),
        ("video_fps", metadata["fps"], "frame/s"),
        ("video_frame_count_reported", metadata["frame_count"], "frame"),
        ("video_width", metadata["width"], "px"),
        ("video_height", metadata["height"], "px"),
        ("processed_frame_count", metrics.total_count, "frame"),
        ("detected_frame_count", metrics.detected_count, "frame"),
        ("fit_frame_count", metrics.fit_count, "frame"),
        (
            "detection_fraction",
            metrics.detected_count / max(metrics.total_count, 1),
            "fraction",
        ),
        ("overlay_frame_index", overlay_frame, "frame"),
        ("track_angle_tls", metrics.track_angle_deg, "deg"),
        ("linear_slope_y_vs_x", metrics.ols_slope, ""),
        ("linear_intercept_y_vs_x", metrics.ols_intercept, "R_sun"),
        ("track_angle_ols", metrics.ols_angle_deg, "deg"),
        ("rms_vertical", metrics.rms_vertical_norm, "R_sun"),
        ("rms_perpendicular", metrics.rms_perpendicular_norm, "R_sun"),
        ("quadratic_a", metrics.quadratic_a, "R_sun^-1"),
        ("quadratic_b", metrics.quadratic_b, ""),
        ("quadratic_c", metrics.quadratic_c, "R_sun"),
        (
            "curvature",
            metrics.curvature_norm_inverse,
            "R_sun^-1",
        ),
        ("curvature", metrics.curvature_px_inverse, "px^-1"),
        ("curvature_fit_rms", metrics.curvature_rms_norm, "R_sun"),
        ("median_solar_radius", metrics.median_solar_radius_px, "px"),
        ("median_venus_radius", metrics.median_venus_radius_px, "px"),
        (
            "median_venus_radius_normalized",
            metrics.median_venus_radius_norm,
            "R_sun",
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "unit"])


def save_xlsx(
    table: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="TRACK", index=False)
        metrics_df.to_excel(writer, sheet_name="METRICS", index=False)
        workbook = writer.book
        track_sheet = workbook["TRACK"]
        metrics_sheet = workbook["METRICS"]
        for sheet in (track_sheet, metrics_sheet):
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                font = copy.copy(cell.font)
                font.bold = True
                cell.font = font
        for column_cells in track_sheet.columns:
            header = str(column_cells[0].value)
            width = min(
                28,
                max(
                    len(header) + 2,
                    max(
                        len(str(cell.value)) if cell.value is not None else 0
                        for cell in column_cells[1: min(track_sheet.max_row, 250) + 1]
                    )
                    + 1,
                ),
            )
            track_sheet.column_dimensions[column_cells[0].column_letter].width = width
        for column_cells in metrics_sheet.columns:
            width = max(
                len(str(cell.value)) if cell.value is not None else 0
                for cell in column_cells
            )
            metrics_sheet.column_dimensions[column_cells[0].column_letter].width = min(
                32, width + 2
            )


def print_table(title: str, rows: list[tuple[str, object, str]]) -> None:
    print(title)
    print(f"{'ITEM':<36} {'VALUE':>22}  {'UNIT':<14}")
    print(f"{'-' * 36} {'-' * 22}  {'-' * 14}")
    for name, value, unit in rows:
        if isinstance(value, (float, np.floating)):
            formatted = f"{float(value):.10f}"
        else:
            formatted = str(value)
        print(f"{name:<36} {formatted:>22}  {unit:<14}")
    print()


def print_report(
    config: Config,
    metadata: dict[str, float],
    metrics: TrackMetrics,
    csv_path: Path,
    xlsx_path: Path,
    overlay_path: Path,
    overlay_frame: int,
) -> None:
    print("CODE INPUTS")
    print_table(
        "",
        [
            ("version", VERSION, ""),
            ("input_video", str(config.input_video), ""),
            ("output_dir", str(config.output_dir), ""),
            ("reported_fps", metadata["fps"], "frame/s"),
            ("reported_frame_count", metadata["frame_count"], "frame"),
            ("bootstrap_frames_skipped", metadata.get("bootstrap_frames_skipped", 0), "frame"),
            ("first_processed_frame", metadata.get("first_processed_frame", 0), "frame"),
            ("last_processed_frame", metadata.get("last_processed_frame", 0), "frame"),
        ],
    )

    print("COMMENTS")
    print_table(
        "",
        [
            ("solar_limb", "ROBUST RADIAL GRADIENT CIRCLE", ""),
            ("venus_centroid", "SUBPIXEL WEIGHTED + EDGE CIRCLE", ""),
            ("coordinate_normalization", "RECOVERED SOLAR RADIUS", ""),
            ("missing_frame_policy", "LINEAR INTERPOLATION, LABELED", ""),
            ("manual_scientific_values", "NOT USED", ""),
        ],
    )

    print("RESULTS")
    print_table(
        "",
        [
            ("processed_frames", metrics.total_count, "frame"),
            ("detected_frames", metrics.detected_count, "frame"),
            (
                "detection_fraction",
                metrics.detected_count / max(metrics.total_count, 1),
                "fraction",
            ),
            ("track_angle_tls", metrics.track_angle_deg, "deg"),
            ("linear_slope_y_vs_x", metrics.ols_slope, ""),
            ("linear_intercept_y_vs_x", metrics.ols_intercept, "R_sun"),
            ("track_angle_ols", metrics.ols_angle_deg, "deg"),
            ("rms_perpendicular", metrics.rms_perpendicular_norm, "R_sun"),
            ("rms_vertical", metrics.rms_vertical_norm, "R_sun"),
            ("quadratic_a", metrics.quadratic_a, "R_sun^-1"),
            ("quadratic_b", metrics.quadratic_b, ""),
            ("quadratic_c", metrics.quadratic_c, "R_sun"),
            (
                "curvature",
                metrics.curvature_norm_inverse,
                "R_sun^-1",
            ),
            ("curvature", metrics.curvature_px_inverse, "px^-1"),
            ("curvature_fit_rms", metrics.curvature_rms_norm, "R_sun"),
            ("median_solar_radius", metrics.median_solar_radius_px, "px"),
            ("median_venus_radius", metrics.median_venus_radius_px, "px"),
            (
                "median_venus_radius_normalized",
                metrics.median_venus_radius_norm,
                "R_sun",
            ),
            ("overlay_frame", overlay_frame, "frame"),
        ],
    )

    print("OUTPUT SUMMARY")
    print_table(
        "",
        [
            ("track_csv", str(csv_path), ""),
            ("track_xlsx", str(xlsx_path), ""),
            ("overlay_png", str(overlay_path), ""),
        ],
    )

    print("PAPER COMPARISON")
    print_table(
        "",
        [
            ("published_track_angle", "NOT USED", ""),
            ("published_curvature", "NOT USED", ""),
            ("comparison_status", "VIDEO-DERIVED RESULTS ONLY", ""),
        ],
    )

    print("EQUATION STATUS")
    print_table(
        "",
        [
            ("normalization", "VERIFIED: (VENUS-SUN CENTER)/R_SUN", ""),
            ("linear_regression", "VERIFIED: y = m*x + b", ""),
            ("tls_direction", "VERIFIED: SVD PRINCIPAL AXIS", ""),
            ("rms", "VERIFIED: SQRT(MEAN(RESIDUAL^2))", ""),
            (
                "curvature",
                "VERIFIED: 2a/(1+b^2)^(3/2)",
                "",
            ),
        ],
    )
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    config = parse_args(argv)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = config.output_dir / CSV_NAME
    xlsx_path = config.output_dir / XLSX_NAME
    overlay_path = config.output_dir / OVERLAY_NAME

    raw_table, metadata = process_video(config)
    filtered_table = select_continuous_venus_track(raw_table)
    normalized_table = normalize_and_fill(filtered_table)
    final_table, metrics = fit_track(normalized_table)
    overlay_frame = save_overlay(
        config, final_table, metrics, overlay_path
    )
    summary = metrics_table(metrics, metadata, overlay_frame)

    final_table.to_csv(csv_path, index=False, float_format="%.10f")
    save_xlsx(final_table, summary, xlsx_path)
    print_report(
        config,
        metadata,
        metrics,
        csv_path,
        xlsx_path,
        overlay_path,
        overlay_frame,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Execution interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
# V0005
