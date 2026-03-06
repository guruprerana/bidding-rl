#!/usr/bin/env python3
"""
Plot learning curves for the assault bidding mechanism comparison experiment.

X axis: env steps (per agent; multi-agent global_step divided by 3).
Y axis: average score with ±1 std shaded region across seeds.
All runs are capped at the per-agent step count corresponding to multi-agent
iter 150 (150 × 128 × 512 = 9,830,400 steps per agent).

Usage:
    python experiment_scripts/plots/plot_assault_bidding_comparison.py
    python experiment_scripts/plots/plot_assault_bidding_comparison.py --log-dir logs/assault_bidding_mechanism_comparison
    python experiment_scripts/plots/plot_assault_bidding_comparison.py --smooth 3
"""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np


LOG_DIR = "logs/assault_bidding_mechanism_comparison"
NUM_AGENTS = 3
# Per-agent step cap: matches multi-agent at iter 150 (150 × 128 × 512 = 9,830,400)
MAX_STEPS_PER_AGENT = 150 * 128 * 512
SEEDS = [1825, 410, 4507, 4013, 3658]

# (prefix, label, multi_agent)
# multi_agent=True → x axis divided by NUM_AGENTS
EXPERIMENTS = [
    ("assault_cmp_winner_pays",          "Winner-Pays",            True),
    ("assault_cmp_winner_pays_localobs", "Winner-Pays (Local Obs)", True),
    ("assault_cmp_all_pay",              "All-Pay",                True),
    ("assault_cmp_all_pay_localobs",     "All-Pay (Local Obs)",    True),
    ("assault_cmp_single_agent",         "Single-Agent PPO",       False),
]


def find_all_seed_runs(log_dir: str, exp_prefix: str, seeds: list[int]) -> dict[int, str]:
    """Find run directories for all seeds of an experiment.

    Returns dict mapping seed → run_dir path.
    """
    seed_runs = {}
    for seed in seeds:
        pattern_re = re.compile(rf"^{re.escape(exp_prefix)}_s{seed}_\d{{8}}_\d{{6}}$")
        matches = [
            d for d in os.listdir(log_dir)
            if pattern_re.match(d) and os.path.isdir(os.path.join(log_dir, d))
        ]
        if matches:
            matches.sort()
            seed_runs[seed] = os.path.join(log_dir, matches[-1])
    return seed_runs


def load_eval_series_single_seed(
    run_dir: str,
    multi_agent: bool,
) -> tuple[list[float], list[float]]:
    """Return (steps, means) sorted by step for a single seed.

    steps are per-agent steps (global_step / NUM_AGENTS) for multi-agent runs.
    y values are mean score across eval episodes.
    Only includes points within MAX_STEPS_PER_AGENT.
    """
    eval_dir = os.path.join(run_dir, "evaluation")
    if not os.path.isdir(eval_dir):
        return [], []

    records = []
    for fname in os.listdir(eval_dir):
        if not re.match(r"iter_\d+_eval_stats\.json$", fname):
            continue
        with open(os.path.join(eval_dir, fname)) as f:
            data = json.load(f)

        global_step = data.get("global_step")
        episodes = data.get("per_episode", {}).get("scores")
        if global_step is None or not episodes:
            continue

        x = global_step / NUM_AGENTS if multi_agent else global_step
        if x > MAX_STEPS_PER_AGENT:
            continue

        mean = np.mean(episodes)
        records.append((x, mean))

    records.sort(key=lambda r: r[0])
    if not records:
        return [], []
    steps, means = zip(*records)
    return list(steps), list(means)


def aggregate_across_seeds(
    seed_runs: dict[int, str],
    multi_agent: bool,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Aggregate data across multiple seeds.

    Returns (steps, mean_across_seeds, mean - std, mean + std) at common steps.
    """
    all_seed_data = {}

    for seed, run_dir in seed_runs.items():
        steps, means = load_eval_series_single_seed(run_dir, multi_agent)
        if steps:
            all_seed_data[seed] = dict(zip(steps, means))

    if not all_seed_data:
        return [], [], [], []

    common_steps = set.intersection(*[set(data.keys()) for data in all_seed_data.values()])
    common_steps = sorted(common_steps)

    if not common_steps:
        return [], [], [], []

    steps_out, means_out, std_lower, std_upper = [], [], [], []
    for step in common_steps:
        values = [all_seed_data[seed][step] for seed in all_seed_data]
        mean = np.mean(values)
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0
        steps_out.append(step)
        means_out.append(mean)
        std_lower.append(mean - std)
        std_upper.append(mean + std)

    return steps_out, means_out, std_lower, std_upper


def smooth(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default=LOG_DIR)
    parser.add_argument("--smooth", type=int, default=1, metavar="W")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        raise SystemExit(f"Log directory not found: {log_dir}")

    output_path = args.output or os.path.join(log_dir, "assault_bidding_mechanisms_learning_curves.png")

    fig, ax = plt.subplots(figsize=(8, 6))

    any_data = False
    for exp_prefix, label, multi_agent in EXPERIMENTS:
        seed_runs = find_all_seed_runs(log_dir, exp_prefix, SEEDS)
        if not seed_runs:
            print(f"  [skip] no runs found for '{exp_prefix}'")
            continue

        print(f"  {label}: found {len(seed_runs)}/{len(SEEDS)} seeds")

        steps, means, std_lo, std_hi = aggregate_across_seeds(seed_runs, multi_agent)
        if not steps:
            print(f"  [skip] no common eval data across seeds for '{exp_prefix}'")
            continue

        s_steps  = smooth(steps,   args.smooth)
        s_means  = smooth(means,   args.smooth)
        s_lo     = smooth(std_lo,  args.smooth)
        s_hi     = smooth(std_hi,  args.smooth)

        (line,) = ax.plot(s_steps, s_means, label=label, linewidth=2)
        ax.fill_between(s_steps, s_lo, s_hi, color=line.get_color(), alpha=0.15)

        print(f"    {len(steps)} points, "
              f"steps {steps[0]:,.0f}–{steps[-1]:,.0f}, "
              f"final score = {means[-1]:.1f} ± {means[-1] - std_lo[-1]:.1f}")
        any_data = True

    if not any_data:
        raise SystemExit("No data found — check --log-dir.")

    ax.set_xlabel("Env. Steps", fontsize=18)
    ax.set_ylabel("Score", fontsize=18)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(loc="upper left", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
