#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "results" / "research" / "g1_grass" / "aggregated_metrics.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "research" / "g1_grass" / "paper_tables"

B3_METHOD = "B3_success_gate"
OURS_METHOD = "Ours_risk_gate"

STAT_SUFFIXES = ("mean", "std", "median", "ci95_low", "ci95_high")
METRIC_ORDER = (
    "success",
    "distance_m",
    "mean_tracking_error",
    "touchdown_timing_error_mean",
    "foot_slip_ratio",
    "stance_duration_deviation_mean",
    "unexpected_contact_count",
    "roll_rms",
    "pitch_rms",
    "base_ang_vel_rms",
    "com_height_fluctuation",
    "recovery_time_s",
    "ankle_action_mean",
    "ankle_action_max",
    "torque_peak",
    "torque_rms",
    "torque_saturation_ratio",
    "joint_limit_margin_min",
    "action_jerk",
    "compensation_phase_alignment",
    "compensation_efficiency",
)
HIGHER_IS_BETTER = {
    "success",
    "distance_m",
    "joint_limit_margin_min",
    "compensation_phase_alignment",
    "compensation_efficiency",
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def require_columns(frame: pd.DataFrame, columns: list[str], table_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot export {table_name}; missing column(s): {', '.join(missing)}")


def metric_stat_columns(frame: pd.DataFrame, metrics: list[str]) -> list[str]:
    columns = []
    for metric in metrics:
        columns.extend([f"{metric}_{suffix}" for suffix in STAT_SUFFIXES if f"{metric}_{suffix}" in frame.columns])
    return columns


def export_metric_table(
    frame: pd.DataFrame,
    output_dir: Path,
    filename: str,
    metrics: list[str],
    *,
    include_success_counts: bool = False,
) -> None:
    base_columns = ["method", "scenario", "n_episodes"]
    if include_success_counts:
        base_columns.extend(["n_success", "success_rate"])
    columns = base_columns + metric_stat_columns(frame, metrics)
    require_columns(frame, base_columns, filename)
    output = frame.loc[:, columns].copy()
    output.to_csv(output_dir / filename, index=False)


def available_metrics(frame: pd.DataFrame) -> list[str]:
    metrics = []
    for column in frame.columns:
        if column.endswith("_mean"):
            metrics.append(column.removesuffix("_mean"))
    ordered = [metric for metric in METRIC_ORDER if metric in metrics]
    ordered.extend(sorted(metric for metric in metrics if metric not in ordered))
    return ordered


def compare_value(row: pd.Series | None, metric: str, suffix: str) -> float:
    if row is None:
        return np.nan
    return row.get(f"{metric}_{suffix}", np.nan)


def export_b3_vs_ours(frame: pd.DataFrame, output_dir: Path) -> None:
    require_columns(frame, ["method", "scenario"], "table_b3_vs_ours.csv")
    metrics = available_metrics(frame)
    scenarios = sorted(frame["scenario"].dropna().unique().tolist())
    rows = []

    for scenario in scenarios:
        b3_rows = frame[(frame["scenario"] == scenario) & (frame["method"] == B3_METHOD)]
        ours_rows = frame[(frame["scenario"] == scenario) & (frame["method"] == OURS_METHOD)]
        b3 = b3_rows.iloc[0] if not b3_rows.empty else None
        ours = ours_rows.iloc[0] if not ours_rows.empty else None

        for metric in metrics:
            b3_mean = compare_value(b3, metric, "mean")
            ours_mean = compare_value(ours, metric, "mean")
            delta = ours_mean - b3_mean if pd.notna(ours_mean) and pd.notna(b3_mean) else np.nan
            relative_delta_pct = (
                100.0 * delta / abs(b3_mean)
                if pd.notna(delta) and pd.notna(b3_mean) and abs(b3_mean) > 0.0
                else np.nan
            )
            better_direction = "higher" if metric in HIGHER_IS_BETTER else "lower"
            if pd.isna(delta):
                ours_better = np.nan
            elif better_direction == "higher":
                ours_better = bool(delta > 0.0)
            else:
                ours_better = bool(delta < 0.0)

            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "better_direction": better_direction,
                    "B3_success_gate_mean": b3_mean,
                    "B3_success_gate_ci95_low": compare_value(b3, metric, "ci95_low"),
                    "B3_success_gate_ci95_high": compare_value(b3, metric, "ci95_high"),
                    "Ours_risk_gate_mean": ours_mean,
                    "Ours_risk_gate_ci95_low": compare_value(ours, metric, "ci95_low"),
                    "Ours_risk_gate_ci95_high": compare_value(ours, metric, "ci95_high"),
                    "delta_ours_minus_b3": delta,
                    "relative_delta_pct": relative_delta_pct,
                    "ours_better": ours_better,
                }
            )

    pd.DataFrame(rows).to_csv(output_dir / "table_b3_vs_ours.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper-ready tables from G1 grass aggregated metrics.")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to aggregated_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where paper table CSV files will be written.",
    )
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    frame = pd.read_csv(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_metric_table(
        frame,
        output_dir,
        "table_success_rate.csv",
        ["success"],
        include_success_counts=True,
    )
    export_metric_table(
        frame,
        output_dir,
        "table_contact_risk.csv",
        [
            "touchdown_timing_error_mean",
            "foot_slip_ratio",
            "stance_duration_deviation_mean",
            "unexpected_contact_count",
        ],
    )
    export_metric_table(
        frame,
        output_dir,
        "table_posture_risk.csv",
        [
            "roll_rms",
            "pitch_rms",
            "base_ang_vel_rms",
            "com_height_fluctuation",
            "recovery_time_s",
        ],
    )
    export_metric_table(
        frame,
        output_dir,
        "table_compensation_quality.csv",
        [
            "ankle_action_mean",
            "ankle_action_max",
            "torque_peak",
            "torque_rms",
            "torque_saturation_ratio",
            "joint_limit_margin_min",
            "action_jerk",
            "compensation_phase_alignment",
            "compensation_efficiency",
        ],
    )
    export_b3_vs_ours(frame, output_dir)
    print(f"[INFO] Wrote paper tables to {output_dir}")


if __name__ == "__main__":
    main()
