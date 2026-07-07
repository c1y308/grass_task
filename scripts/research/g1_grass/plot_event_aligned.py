#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "research" / "g1_grass" / "figures"
EVENT_TYPES = ("touchdown_error", "foot_slip", "terrain_transition")
DEFAULT_WINDOW = (-0.5, 1.0)
DEFAULT_BIN_SIZE = 0.02
DEFAULT_MAX_EVENTS = 500

TIME_COLUMN_CANDIDATES = (
    "time_s",
    "time",
    "timestamp_s",
    "sim_time_s",
    "episode_time_s",
    "elapsed_s",
)
TRAJECTORY_COLUMN_CANDIDATES = (
    "method",
    "scenario",
    "train_seed",
    "eval_seed",
    "seed",
    "run_id",
    "episode",
    "env_id",
    "env",
)
EVENT_COLUMN_CANDIDATES = ("event_type", "event", "event_name")

METRIC_CANDIDATES = {
    "slip_velocity": (
        "slip_velocity",
        "slip_velocity_mps",
        "foot_slip_velocity",
        "foot_slip_velocity_mps",
        "foot_xy_speed",
        "foot_speed",
        "foot_velocity_xy_norm",
    ),
    "roll": ("roll_error", "roll_abs", "roll", "base_roll", "root_roll"),
    "pitch": ("pitch_error", "pitch_abs", "pitch", "base_pitch", "root_pitch"),
    "ankle_action": (
        "ankle_action_amplitude",
        "ankle_action_abs",
        "ankle_action_mean",
        "ankle_action",
        "ankle_action_norm",
    ),
    "torque_saturation": (
        "torque_saturation_indicator",
        "torque_saturated",
        "torque_saturation",
        "torque_saturation_ratio",
    ),
    "action_jerk": ("action_jerk", "action_jerk_norm", "policy_action_jerk"),
}

PANEL_SPECS = (
    ("slip_velocity", "Slip velocity", "m/s"),
    ("roll_pitch", "Roll / pitch error", "rad"),
    ("ankle_action", "Ankle action amplitude", "action unit"),
    ("torque_saturation", "Torque saturation indicator", "ratio"),
    ("action_jerk", "Action jerk", "action unit/step^2"),
)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def parse_event_types(value: str) -> list[str]:
    event_types = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [event_type for event_type in event_types if event_type not in EVENT_TYPES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown event type(s): {', '.join(unknown)}. Valid types: {', '.join(EVENT_TYPES)}"
        )
    if not event_types:
        raise argparse.ArgumentTypeError("At least one event type is required.")
    return event_types


def find_column(frame: pd.DataFrame, candidates: Iterable[str], explicit: str | None = None) -> str:
    if explicit:
        if explicit not in frame.columns:
            raise ValueError(f"Column not found: {explicit}")
        return explicit
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"None of these columns were found: {', '.join(candidates)}")


def optional_column(frame: pd.DataFrame, candidates: Iterable[str], explicit: str | None = None) -> str | None:
    if explicit:
        return find_column(frame, candidates, explicit)
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def read_diagnostics(input_path: Path) -> pd.DataFrame:
    input_path = resolve_path(input_path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(input_path)
    raise ValueError(f"Unsupported input format: {input_path}. Use CSV or parquet.")


def coerce_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def detect_event_mask(frame: pd.DataFrame, event_type: str, event_column: str | None) -> pd.Series:
    if event_column is not None:
        events = frame[event_column]
        if events.dtype == object:
            event_tokens = events.fillna("").astype(str).str.split(r"[|,; ]+", regex=True)
            return event_tokens.apply(lambda tokens: event_type in tokens)
        return events.fillna("").astype(str).eq(event_type)

    indicator_candidates = (
        f"event_{event_type}",
        f"{event_type}_event",
        f"is_{event_type}",
        f"{event_type}_indicator",
        event_type,
    )
    for column in indicator_candidates:
        if column in frame.columns:
            values = frame[column]
            if values.dtype == object:
                normalized = values.fillna("").astype(str).str.lower()
                return normalized.isin({"1", "true", "yes", event_type})
            return pd.to_numeric(values, errors="coerce").fillna(0.0) > 0.0
    raise ValueError(
        f"No event column found for {event_type}. Provide event_type/event_name column or an indicator such as "
        f"event_{event_type}."
    )


def metric_column_map(frame: pd.DataFrame, args: argparse.Namespace) -> dict[str, str | None]:
    return {
        "slip_velocity": optional_column(frame, METRIC_CANDIDATES["slip_velocity"], args.slip_column),
        "roll": optional_column(frame, METRIC_CANDIDATES["roll"], args.roll_column),
        "pitch": optional_column(frame, METRIC_CANDIDATES["pitch"], args.pitch_column),
        "ankle_action": optional_column(frame, METRIC_CANDIDATES["ankle_action"], args.ankle_action_column),
        "torque_saturation": optional_column(
            frame, METRIC_CANDIDATES["torque_saturation"], args.torque_saturation_column
        ),
        "action_jerk": optional_column(frame, METRIC_CANDIDATES["action_jerk"], args.action_jerk_column),
    }


def add_plot_metrics(frame: pd.DataFrame, columns: dict[str, str | None]) -> dict[str, list[tuple[str, str]]]:
    metrics: dict[str, list[tuple[str, str]]] = {}

    if columns["slip_velocity"] is not None:
        metrics["slip_velocity"] = [(columns["slip_velocity"], "slip velocity")]

    roll_pitch = []
    if columns["roll"] is not None:
        if columns["roll"] in {"roll", "base_roll", "root_roll"}:
            frame["_roll_error_abs"] = frame[columns["roll"]].abs()
            roll_pitch.append(("_roll_error_abs", "roll"))
        else:
            roll_pitch.append((columns["roll"], "roll"))
    if columns["pitch"] is not None:
        if columns["pitch"] in {"pitch", "base_pitch", "root_pitch"}:
            frame["_pitch_error_abs"] = frame[columns["pitch"]].abs()
            roll_pitch.append(("_pitch_error_abs", "pitch"))
        else:
            roll_pitch.append((columns["pitch"], "pitch"))
    if roll_pitch:
        metrics["roll_pitch"] = roll_pitch

    if columns["ankle_action"] is not None:
        frame["_ankle_action_abs"] = frame[columns["ankle_action"]].abs()
        metrics["ankle_action"] = [("_ankle_action_abs", "ankle action")]

    if columns["torque_saturation"] is not None:
        metrics["torque_saturation"] = [(columns["torque_saturation"], "torque saturation")]

    if columns["action_jerk"] is not None:
        metrics["action_jerk"] = [(columns["action_jerk"], "action jerk")]

    missing_panels = [panel for panel, _, _ in PANEL_SPECS if panel not in metrics]
    if missing_panels:
        print(f"[WARN] Missing metric columns for panel(s): {', '.join(missing_panels)}")
    return metrics


def trajectory_columns(frame: pd.DataFrame, time_column: str) -> list[str]:
    return [column for column in TRAJECTORY_COLUMN_CANDIDATES if column in frame.columns and column != time_column]


def event_records(
    frame: pd.DataFrame,
    event_type: str,
    time_column: str,
    event_column: str | None,
    trajectory_cols: list[str],
    max_events: int,
) -> pd.DataFrame:
    mask = detect_event_mask(frame, event_type, event_column)
    events = frame.loc[mask, [time_column] + trajectory_cols].copy()
    events = events.dropna(subset=[time_column]).sort_values(time_column)
    if max_events > 0 and len(events) > max_events:
        events = events.iloc[:max_events].copy()
    events["_event_id"] = np.arange(len(events), dtype=int)
    events = events.rename(columns={time_column: "_event_time"})
    return events


def align_to_events(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    time_column: str,
    trajectory_cols: list[str],
    window: tuple[float, float],
    bin_size: float,
) -> pd.DataFrame:
    aligned_parts = []
    for _, event in events.iterrows():
        same_trajectory = pd.Series(True, index=frame.index)
        for column in trajectory_cols:
            event_value = event[column]
            if pd.isna(event_value):
                same_trajectory &= frame[column].isna()
            else:
                same_trajectory &= frame[column].eq(event_value)

        relative_time = frame[time_column] - event["_event_time"]
        window_mask = same_trajectory & relative_time.between(window[0], window[1], inclusive="both")
        aligned = frame.loc[window_mask].copy()
        if aligned.empty:
            continue
        aligned["event_id"] = int(event["_event_id"])
        aligned["event_time_s"] = float(event["_event_time"])
        aligned["relative_time_s"] = relative_time.loc[window_mask].to_numpy(dtype=float)
        aligned["relative_time_bin_s"] = np.round(aligned["relative_time_s"] / bin_size) * bin_size
        aligned_parts.append(aligned)

    if not aligned_parts:
        return pd.DataFrame()
    return pd.concat(aligned_parts, ignore_index=True, sort=False)


def grouped_metric_summary(
    aligned: pd.DataFrame,
    value_column: str,
    group_column: str | None,
) -> pd.DataFrame:
    group_cols = ["relative_time_bin_s"]
    if group_column is not None and group_column in aligned.columns:
        group_cols.insert(0, group_column)

    summary = (
        aligned.groupby(group_cols, dropna=False)[value_column]
        .agg(
            mean="mean",
            median="median",
            ci95_low=lambda values: values.quantile(0.025),
            ci95_high=lambda values: values.quantile(0.975),
            n="count",
        )
        .reset_index()
    )
    return summary


def plot_panel(
    ax: plt.Axes,
    aligned: pd.DataFrame,
    panel_key: str,
    panel_title: str,
    y_label: str,
    metric_specs: list[tuple[str, str]],
    group_column: str | None,
) -> None:
    if panel_key not in {key for key, _, _ in PANEL_SPECS} or not metric_specs:
        ax.set_axis_off()
        return

    for value_column, metric_label in metric_specs:
        summary = grouped_metric_summary(aligned, value_column, group_column)
        if summary.empty:
            continue
        if group_column is not None and group_column in summary.columns:
            groups = summary[group_column].fillna("unknown").unique().tolist()
        else:
            groups = ["all"]
            summary[group_column or "_group"] = "all"
            group_column = group_column or "_group"

        for group in groups:
            group_summary = summary[summary[group_column].fillna("unknown").eq(group)].sort_values(
                "relative_time_bin_s"
            )
            if group_summary.empty:
                continue
            label = str(group) if len(metric_specs) == 1 else f"{group} {metric_label}"
            x = group_summary["relative_time_bin_s"].to_numpy(dtype=float)
            mean = group_summary["mean"].to_numpy(dtype=float)
            low = group_summary["ci95_low"].to_numpy(dtype=float)
            high = group_summary["ci95_high"].to_numpy(dtype=float)
            line = ax.plot(x, mean, linewidth=1.8, label=label)[0]
            ax.fill_between(x, low, high, color=line.get_color(), alpha=0.18, linewidth=0.0)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title(panel_title)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)


def plot_event_figure(
    aligned: pd.DataFrame,
    event_type: str,
    metrics: dict[str, list[tuple[str, str]]],
    group_column: str | None,
    output_dir: Path,
    prefix: str,
) -> None:
    fig, axes = plt.subplots(len(PANEL_SPECS), 1, figsize=(9.0, 12.0), sharex=True)
    for ax, (panel_key, panel_title, y_label) in zip(axes, PANEL_SPECS):
        plot_panel(ax, aligned, panel_key, panel_title, y_label, metrics.get(panel_key, []), group_column)

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        for ax in axes[1:]:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)), frameon=False)

    axes[-1].set_xlabel("Time from event (s)")
    fig.suptitle(f"Event-aligned compensation diagnostics: {event_type}", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{prefix}_{event_type}.png"
    pdf_path = output_dir / f"{prefix}_{event_type}.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[INFO] Wrote {png_path}")
    print(f"[INFO] Wrote {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot event-aligned G1 grass compensation diagnostics.")
    parser.add_argument("input", type=Path, help="Per-step diagnostic CSV or parquet file.")
    parser.add_argument(
        "--event-types",
        type=parse_event_types,
        default=list(EVENT_TYPES),
        help=f"Comma-separated event types. Valid: {', '.join(EVENT_TYPES)}.",
    )
    parser.add_argument(
        "--window",
        nargs=2,
        type=float,
        default=DEFAULT_WINDOW,
        metavar=("START", "END"),
        help="Time window around each event in seconds. Default: -0.5 1.0.",
    )
    parser.add_argument("--bin-size", type=float, default=DEFAULT_BIN_SIZE, help="Aligned time bin size in seconds.")
    parser.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS, help="Maximum events per type; <=0 uses all.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Figure output directory.")
    parser.add_argument("--prefix", type=str, default="event_aligned", help="Output file prefix.")
    parser.add_argument("--time-column", type=str, default=None, help="Override time column.")
    parser.add_argument("--event-column", type=str, default=None, help="Override event type/name column.")
    parser.add_argument("--group-column", type=str, default="method", help="Column used for plot grouping; use none to disable.")
    parser.add_argument("--slip-column", type=str, default=None, help="Override slip velocity column.")
    parser.add_argument("--roll-column", type=str, default=None, help="Override roll error column.")
    parser.add_argument("--pitch-column", type=str, default=None, help="Override pitch error column.")
    parser.add_argument("--ankle-action-column", type=str, default=None, help="Override ankle action amplitude column.")
    parser.add_argument(
        "--torque-saturation-column", type=str, default=None, help="Override torque saturation indicator column."
    )
    parser.add_argument("--action-jerk-column", type=str, default=None, help="Override action jerk column.")
    args = parser.parse_args()

    if args.window[0] >= args.window[1]:
        parser.error("--window START must be smaller than END.")
    if args.bin_size <= 0.0:
        parser.error("--bin-size must be positive.")

    frame = read_diagnostics(args.input)
    time_column = find_column(frame, TIME_COLUMN_CANDIDATES, args.time_column)
    event_column = optional_column(frame, EVENT_COLUMN_CANDIDATES, args.event_column)
    group_column = None if args.group_column.lower() == "none" else args.group_column
    if group_column is not None and group_column not in frame.columns:
        print(f"[WARN] Group column not found: {group_column}. Plotting all rows as one group.")
        group_column = None

    columns = metric_column_map(frame, args)
    numeric_columns = [time_column] + [column for column in columns.values() if column is not None]
    coerce_numeric(frame, numeric_columns)
    frame = frame.dropna(subset=[time_column]).sort_values(time_column).reset_index(drop=True)
    metrics = add_plot_metrics(frame, columns)
    trajectory_cols = trajectory_columns(frame, time_column)

    for event_type in args.event_types:
        events = event_records(frame, event_type, time_column, event_column, trajectory_cols, args.max_events)
        if events.empty:
            print(f"[WARN] No events found for {event_type}; skipping.")
            continue
        aligned = align_to_events(frame, events, time_column, trajectory_cols, tuple(args.window), args.bin_size)
        if aligned.empty:
            print(f"[WARN] No aligned samples found for {event_type}; skipping.")
            continue
        plot_event_figure(aligned, event_type, metrics, group_column, resolve_path(args.output_dir), args.prefix)


if __name__ == "__main__":
    main()
