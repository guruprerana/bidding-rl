#!/usr/bin/env python3
"""
Plot peak performance bar chart for the bid upper bound comparison experiment.

For each condition the peak mean avg_min_targets_reached across all eval
checkpoints is selected. Bar height = 8 * peak_mean.

Usage:
    python experiment_scripts/plots/plot_bid_upper_bound_comparison.py
    python experiment_scripts/plots/plot_bid_upper_bound_comparison.py --log-dir logs/gridworld_bid_upper_bound_comparison
"""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


EXPERIMENTS = [
    ("bid_ub_cmp_bub0", "0"),
    ("bid_ub_cmp_bub1", "1"),
    ("bid_ub_cmp_bub2", "2"),
    ("bid_ub_cmp_bub4", "4"),
    ("bid_ub_cmp_bub6", "6"),
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
    rollouts_dir = os.path.join(run_dir, "rollouts")
    if not os.path.isdir(rollouts_dir):
        return [], [], [], []

    CONFIDENCE = 0.95
    pattern = re.compile(r".*_eval_stats\.json$")
    records = []
    for fname in os.listdir(rollouts_dir):
        if not pattern.match(fname):
            continue
        path = os.path.join(rollouts_dir, fname)
        with open(path) as f:
            data = json.load(f)
        step = data.get("global_step")
        episodes = data.get("per_episode_data", {}).get("min_targets_reached")
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
    parser.add_argument("--log-dir", default="logs/gridworld_bid_upper_bound_comparison")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        raise SystemExit(f"Log directory not found: {log_dir}")

    output_path = args.output or os.path.join(log_dir, "gridworld_bid_upper_bound_bar_chart.png")

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

        height = 8 * peak_mean
        labels.append(label)
        bar_heights.append(height)
        err_lo.append(8 * (peak_mean - peak_lo))
        err_hi.append(8 * (peak_hi - peak_mean))

        print(f"  {label}: peak at step {steps[peak_idx]:,}, "
              f"peak avg_min_targets_reached = {peak_mean:.3f}, "
              f"bar height (8x) = {height:.3f}")

    if not labels:
        raise SystemExit("No data found — check --log-dir.")

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x, bar_heights, yerr=[err_lo, err_hi],
           capsize=5, width=0.6, color="mediumpurple", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18)
    ax.set_xlabel("Bid Upper Bound", fontsize=20)
    ax.set_ylabel("Avg. Performance", fontsize=20)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
