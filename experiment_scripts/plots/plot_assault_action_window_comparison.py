#!/usr/bin/env python3
"""
Plot peak performance bar chart for the Assault action window comparison experiment.

For each action window value the peak mean score across all eval checkpoints
is selected, with 95% CI.

Usage:
    python experiment_scripts/plots/plot_assault_action_window_comparison.py
    python experiment_scripts/plots/plot_assault_action_window_comparison.py --log-dir logs/assault_action_window_comparison
"""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


LOG_DIR = "logs/assault_action_window_comparison"
CONFIDENCE = 0.95

EXPERIMENTS = [
    ("assault_aw_cmp_w5",  "5"),
    ("assault_aw_cmp_w15", "15"),
    ("assault_aw_cmp_w30", "30"),
    ("assault_aw_cmp_w60", "60"),
]


def find_latest_run(log_dir: str, exp_prefix: str) -> str | None:
    timestamp_re = re.compile(rf"^{re.escape(exp_prefix)}_\d{{8}}_\d{{6}}$")
    matches = [
        d for d in os.listdir(log_dir)
        if timestamp_re.match(d) and os.path.isdir(os.path.join(log_dir, d))
    ]
    if not matches:
        return None
    matches.sort()
    return os.path.join(log_dir, matches[-1])


def load_eval_series(
    run_dir: str,
) -> tuple[list[int], list[float], list[float], list[float]]:
    eval_dir = os.path.join(run_dir, "evaluation")
    if not os.path.isdir(eval_dir):
        return [], [], [], []

    pattern = re.compile(r".*_eval_stats\.json$")
    records = []
    for fname in os.listdir(eval_dir):
        if not pattern.match(fname):
            continue
        with open(os.path.join(eval_dir, fname)) as f:
            data = json.load(f)
        step = data.get("global_step")
        episodes = data.get("per_episode", {}).get("scores")
        if step is None or not episodes:
            continue
        episodes = np.array(episodes, dtype=float)
        n = len(episodes)
        mean = episodes.mean()
        se = episodes.std(ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf((1 + CONFIDENCE) / 2, df=n - 1)
        records.append((step, mean, mean - t_crit * se, mean + t_crit * se))

    records.sort(key=lambda x: x[0])
    if not records:
        return [], [], [], []
    steps, means, lows, highs = zip(*records)
    return list(steps), list(means), list(lows), list(highs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default=LOG_DIR)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        raise SystemExit(f"Log directory not found: {log_dir}")

    output_path = args.output or os.path.join(log_dir, "assault_action_window_bar_chart.png")

    labels = []
    bar_heights = []
    err_lo = []
    err_hi = []

    for exp_prefix, label in EXPERIMENTS:
        run_dir = find_latest_run(log_dir, exp_prefix)
        if run_dir is None:
            print(f"  [skip] no run found for prefix '{exp_prefix}'")
            continue

        steps, means, ci_lo, ci_hi = load_eval_series(run_dir)
        if not steps:
            print(f"  [skip] no eval stats found in {run_dir}")
            continue

        peak_idx = int(np.argmax(means))
        peak_mean = means[peak_idx]
        peak_lo   = ci_lo[peak_idx]
        peak_hi   = ci_hi[peak_idx]

        labels.append(label)
        bar_heights.append(peak_mean)
        err_lo.append(peak_mean - peak_lo)
        err_hi.append(peak_hi - peak_mean)

        print(f"  action_window={label}: peak at step {steps[peak_idx]:,}, "
              f"score = {peak_mean:.2f}")

    if not labels:
        raise SystemExit("No data found — check --log-dir.")

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x, bar_heights, yerr=[err_lo, err_hi],
           capsize=5, width=0.6, color="darkorange", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_xlabel("Action Window", fontsize=18)
    ax.set_ylabel("Score", fontsize=18)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
