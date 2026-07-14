#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


METRIC_VERSION = "g1_grass_risk_v2"
RISK_COLUMNS = ("contact_risk", "posture_risk", "compensation_risk")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_window_q95(values: np.ndarray, window: int, samples: int, seed: int) -> float:
    if values.size < window:
        raise ValueError(f"Need at least {window} episodes, found {values.size}.")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        stop = min(start + 256, samples)
        indices = rng.integers(0, values.size, size=(stop - start, window))
        means[start:stop] = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.95))


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze preregistered G1 grass risk thresholds.")
    parser.add_argument("input", type=Path, help="Flat-reference per-episode CSV.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--safety-cap", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    missing = [column for column in RISK_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Missing risk columns: {', '.join(missing)}")
    thresholds = {}
    reference = {}
    for offset, column in enumerate(RISK_COLUMNS):
        values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=np.float64)
        q95 = bootstrap_window_q95(values, args.window, args.bootstrap_samples, args.seed + offset)
        thresholds[f"{column.replace('_risk', '')}_threshold"] = min(q95 + args.margin, args.safety_cap)
        reference[column] = {"episodes": int(values.size), "bootstrap_window_q95": q95}

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"
    output = {
        "metric_version": METRIC_VERSION,
        "frozen": True,
        "input_csv": str(args.input.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "git_commit": git_commit,
        "window_episodes": args.window,
        "bootstrap_samples": args.bootstrap_samples,
        "margin": args.margin,
        "safety_cap": args.safety_cap,
        "thresholds": thresholds,
        "reference": reference,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(output, stream, sort_keys=False)
    print(f"[INFO] Wrote frozen thresholds to {args.output}")


if __name__ == "__main__":
    main()
