#!/usr/bin/env python3
"""
Plot average env reward per agent for the bidding mechanism comparison experiment.

Reads eval stats from each experiment's rollouts/ directory and produces a
figure with avg episode return / num_agents vs environment steps for all
algorithms (excluding DWN).

The total episode return is divided by num_agents to give a per-agent figure
comparable across mechanisms, letting us check whether the performance drop
visible in avg_min_targets_reached is also present in rewards.

Usage:
    python experiment_scripts/plots/plot_bidding_mechanism_reward_per_agent.py
    python experiment_scripts/plots/plot_bidding_mechanism_reward_per_agent.py --log-dir logs/gridworld_bidding_mechanism_comparison
    python experiment_scripts/plots/plot_bidding_mechanism_reward_per_agent.py --log-dir logs/gridworld_bidding_mechanism_comparison --smooth 5
"""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# Experiment name prefixes to include (DWN is excluded)
EXPERIMENTS = [
    ("bidding_cmp_all_pay",                   "All-Pay"),
    ("bidding_cmp_winner_pays",               "Winner-Pays"),
    ("bidding_cmp_winner_pays_others_reward", "Winner-Pays (Others Rewarded)"),
    ("bidding_cmp_single_agent",              "Single-Agent"),
]


def find_latest_run(log_dir: str, exp_prefix: str) -> str | None:
    """Return the path to the most recent subdirectory matching exp_prefix."""
    timestamp_re = re.compile(rf"^{re.escape(exp_prefix)}_\d{{8}}_\d{{6}}$")
    matches = [
        d for d in os.listdir(log_dir)
        if timestamp_re.match(d) and os.path.isdir(os.path.join(log_dir, d))
    ]
    if not matches:
        return None
    matches.sort()
    return os.path.join(log_dir, matches[-1])


def load_return_per_agent_series(
    run_dir: str,
) -> tuple[list[int], list[float], list[float], list[float]]:
    """
    Load all eval stats from rollouts/ and return
    (steps, means, ci_lower, ci_upper) of avg episode return / num_agents,
    sorted by step.

    The total episode return is divided by num_agents so the scale is
    comparable across mechanisms and matches the single-agent baseline.
    Confidence intervals use a 95% t-distribution over per-episode values.
    """
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
        episode_returns = data.get("per_episode_data", {}).get("returns")
        if step is None or not episode_returns:
            continue

        # num_agents for multi-agent runs; single-agent JSONs use num_targets
        # and have only one effective agent, so default to 1.
        num_agents = data.get("num_agents") or data.get("num_targets") or 1

        episodes = np.array(episode_returns, dtype=float) / num_agents
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


def smooth(values: list[float], window: int) -> list[float]:
    """Simple uniform moving average."""
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        default="logs/gridworld_bidding_mechanism_comparison",
        help="Base log directory produced by bidding_mechanism_comparison.py",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        metavar="W",
        help="Moving-average window size for smoothing (default: 1 = no smoothing)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output filename (default: reward_per_agent.png inside --log-dir)",
    )
    args = parser.parse_args()

    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        raise SystemExit(f"Log directory not found: {log_dir}")

    output_path = args.output or os.path.join(log_dir, "gridworld_bidding_mechanisms_reward_per_agent.png")

    fig, ax = plt.subplots(figsize=(8, 6))

    any_data = False
    for exp_prefix, label in EXPERIMENTS:
        run_dir = find_latest_run(log_dir, exp_prefix)
        if run_dir is None:
            print(f"  [skip] no run found for prefix '{exp_prefix}'")
            continue

        steps, means, ci_lo, ci_hi = load_return_per_agent_series(run_dir)
        if not steps:
            print(f"  [skip] no eval stats found in {run_dir}")
            continue

        smoothed_mean = smooth(means, args.smooth)
        smoothed_lo   = smooth(ci_lo, args.smooth)
        smoothed_hi   = smooth(ci_hi, args.smooth)

        (line,) = ax.plot(steps, smoothed_mean, label=label, linewidth=2)
        ax.fill_between(steps, smoothed_lo, smoothed_hi,
                        color=line.get_color(), alpha=0.15)

        print(f"  {label}: {len(steps)} eval points, "
              f"steps {steps[0]:,} – {steps[-1]:,}, "
              f"final avg_return_per_agent = {means[-1]:.3f}")
        any_data = True

    if not any_data:
        raise SystemExit("No data found — check --log-dir.")

    ax.set_xlabel("Environment Steps", fontsize=16)
    ax.set_ylabel("Avg Episode Return per Agent", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
