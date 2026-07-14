#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FLAGS = ("success_pass", "contact_pass", "posture_pass", "compensation_pass", "gate_pass")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot B3/Ours curriculum mechanism diagnostics.")
    parser.add_argument("--b3", type=Path, required=True)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = {"B3 success gate": pd.read_csv(args.b3), "Ours risk gate": pd.read_csv(args.ours)}

    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    colors = {"B3 success gate": "#0072B2", "Ours risk gate": "#D55E00"}
    for label, frame in frames.items():
        x = frame["local_ppo_iteration"]
        axes[0].step(x, frame["stage_after"], where="post", label=label, color=colors[label], linewidth=2)
        for metric, linestyle in (("success_rate", "-"), ("contact_risk", "--"), ("posture_risk", ":")):
            axes[1].plot(x, frame[metric], label=f"{label}: {metric}", color=colors[label], linestyle=linestyle)
    ours = frames["Ours risk gate"]
    for index, flag in enumerate(FLAGS):
        if flag in ours:
            axes[2].step(
                ours["local_ppo_iteration"],
                ours[flag].astype(float) + index * 1.2,
                where="post",
                label=flag,
            )
    axes[0].set_ylabel("Stage id")
    axes[0].set_yticks(range(5))
    axes[1].set_ylabel("Window metric")
    axes[2].set_ylabel("Gate flags (offset)")
    axes[2].set_xlabel("Local PPO iteration")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=8, ncol=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    figure.savefig(args.output.with_suffix(".pdf"))
    print(f"[INFO] Wrote curriculum diagnostic plot to {args.output}")


if __name__ == "__main__":
    main()
