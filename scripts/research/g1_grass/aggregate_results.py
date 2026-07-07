#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "research" / "g1_grass" / "aggregated_metrics.csv"
GROUP_COLUMNS = ("method", "scenario")
REQUIRED_COLUMNS = ("method", "scenario", "success")
ID_NUMERIC_COLUMNS = {"train_seed", "eval_seed", "episode"}
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 12345


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def collect_csv_files(inputs: Iterable[Path], output_path: Path) -> list[tuple[Path, bool]]:
    output_path = output_path.resolve()
    files: list[tuple[Path, bool]] = []
    for input_path in inputs:
        resolved = resolve_path(input_path)
        if resolved.is_dir():
            files.extend((path, True) for path in sorted(resolved.rglob("*.csv")) if path.is_file())
        elif resolved.is_file():
            files.append((resolved, False))
        else:
            raise FileNotFoundError(f"Input does not exist: {input_path}")

    unique_files = []
    seen: dict[Path, int] = {}
    for file_path, allow_skip in files:
        resolved = file_path.resolve()
        if resolved == output_path:
            continue
        if resolved in seen:
            existing_index = seen[resolved]
            existing_path, existing_allow_skip = unique_files[existing_index]
            unique_files[existing_index] = (existing_path, existing_allow_skip and allow_skip)
            continue
        seen[resolved] = len(unique_files)
        unique_files.append((file_path, allow_skip))
    if not unique_files:
        raise ValueError("No input CSV files found.")
    return unique_files


def read_evaluation_csv(file_path: Path, *, allow_skip: bool) -> pd.DataFrame | None:
    frame = pd.read_csv(file_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        if allow_skip:
            print(f"[INFO] Skipping non-evaluation CSV {file_path}: missing {', '.join(missing)}")
            return None
        raise ValueError(f"{file_path} is missing required column(s): {', '.join(missing)}")

    frame = frame.copy()
    frame["source_file"] = str(file_path)
    return frame


def load_evaluation_data(inputs: list[tuple[Path, bool]]) -> pd.DataFrame:
    frames = []
    for file_path, allow_skip in inputs:
        frame = read_evaluation_csv(file_path, allow_skip=allow_skip)
        if frame is not None:
            frames.append(frame)

    if not frames:
        raise ValueError("No evaluation CSV files with the required columns were found.")

    data = pd.concat(frames, ignore_index=True, sort=False)
    for column in REQUIRED_COLUMNS:
        if column not in data.columns:
            raise ValueError(f"Combined data is missing required column: {column}")
    data["success"] = pd.to_numeric(data["success"], errors="coerce").fillna(0).astype(float)
    return data


def numeric_metric_columns(data: pd.DataFrame) -> list[str]:
    for column in data.columns:
        if column not in GROUP_COLUMNS and column != "source_file":
            converted = pd.to_numeric(data[column], errors="coerce")
            if converted.notna().any():
                data[column] = converted

    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    return [column for column in numeric_columns if column not in ID_NUMERIC_COLUMNS]


def bootstrap_mean_ci(values: pd.Series, rng: np.random.Generator, n_samples: int) -> tuple[float, float]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if array.size == 0:
        return np.nan, np.nan
    if array.size == 1 or n_samples <= 0:
        mean_value = float(array.mean())
        return mean_value, mean_value

    sample_indices = rng.integers(0, array.size, size=(n_samples, array.size))
    sample_means = array[sample_indices].mean(axis=1)
    lower, upper = np.quantile(sample_means, [0.025, 0.975])
    return float(lower), float(upper)


def summarize_group(
    method: str,
    scenario: str,
    group: pd.DataFrame,
    metrics: list[str],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "method": method,
        "scenario": scenario,
        "n_episodes": int(len(group)),
        "n_success": int(pd.to_numeric(group["success"], errors="coerce").fillna(0).sum()),
    }
    row["success_rate"] = row["n_success"] / row["n_episodes"] if row["n_episodes"] else np.nan

    for metric in metrics:
        values = pd.to_numeric(group[metric], errors="coerce")
        row[f"{metric}_mean"] = float(values.mean()) if values.notna().any() else np.nan
        row[f"{metric}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan
        row[f"{metric}_median"] = float(values.median()) if values.notna().any() else np.nan
        ci_low, ci_high = bootstrap_mean_ci(values, rng, bootstrap_samples)
        row[f"{metric}_ci95_low"] = ci_low
        row[f"{metric}_ci95_high"] = ci_high
    return row


def aggregate(data: pd.DataFrame, bootstrap_samples: int, bootstrap_seed: int) -> pd.DataFrame:
    metrics = numeric_metric_columns(data)
    rng = np.random.default_rng(bootstrap_seed)
    rows = []
    grouped = data.groupby(list(GROUP_COLUMNS), sort=True, dropna=False)
    for (method, scenario), group in grouped:
        rows.append(summarize_group(method, scenario, group, metrics, rng, bootstrap_samples))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate G1 grass evaluation CSV files.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Evaluation CSV file(s) or directory/directories.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output aggregated metrics CSV path.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Number of bootstrap resamples for 95%% mean confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Random seed for bootstrap resampling.",
    )
    args = parser.parse_args()

    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must be non-negative.")

    output_path = resolve_path(args.output)
    input_files = collect_csv_files(args.inputs, output_path)
    data = load_evaluation_data(input_files)
    aggregated = aggregate(data, args.bootstrap_samples, args.bootstrap_seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(output_path, index=False)
    print(f"[INFO] Wrote {len(aggregated)} groups to {output_path}")


if __name__ == "__main__":
    main()
