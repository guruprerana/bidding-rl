#!/usr/bin/env python3
"""
Learning curves for gridworld_all_methods_comparison.

X axis: env steps (divided by 8 for multi-agent methods to give per-agent steps).
Y axis: 8 × avg_avg_performance (with 95% CI across 20 eval episodes).
Only iterations up to 400 are included.

Usage:
    python experiment_scripts/plots/plot_bidding_mechanism_comparison.py
    python experiment_scripts/plots/plot_bidding_mechanism_comparison.py --log-dir logs/gridworld_all_methods_comparison
    python experiment_scripts/plots/plot_bidding_mechanism_comparison.py --smooth 5
"""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


LOG_DIR = "logs/gridworld_all_methods_comparison"
NUM_AGENTS = 8
MAX_ITERATION = 400
CONFIDENCE = 0.95

# (prefix, label, multi_agent)
# multi_agent=True → x axis divided by NUM_AGENTS
EXPERIMENTS = [
    ("bidding_cmp_all_pay",                   "All-Pay",                      True),
    ("bidding_cmp_winner_pays",               "Winner-Pays",                  True),
    ("bidding_cmp_winner_pays_others_reward", "Winner-Pays (Others Rewarded)", True),
    ("multiagentppo_localobs",                "All-Pay (Local Obs)",   True),
    ("ppo_singleagent_optunait21",            "Single-Policy PPO",              False),
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
    multi_agent: bool,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return (steps, means, ci_lower, ci_upper) sorted by step.

    steps are per-agent steps (global_step / NUM_AGENTS) for multi-agent runs.
    y values are NUM_AGENTS × avg_avg_performance with 95% t-CI over 20 episodes.
    Only includes iterations ≤ MAX_ITERATION.
    """
    rollouts_dir = os.path.join(run_dir, "rollouts")
    if not os.path.isdir(rollouts_dir):
        return [], [], [], []

    records = []
    for fname in os.listdir(rollouts_dir):
        m = re.match(r"iter_(\d+)_eval_stats\.json$", fname)
        if not m:
            continue
        if int(m.group(1)) > MAX_ITERATION:
            continue
        with open(os.path.join(rollouts_dir, fname)) as f:
            data = json.load(f)

        global_step = data.get("global_step")
        episodes = data.get("per_episode_data", {}).get("avg_performance")
        if global_step is None or not episodes:
            continue

        episodes = np.array(episodes, dtype=float) * NUM_AGENTS
        n = len(episodes)
        mean = episodes.mean()
        se = episodes.std(ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf((1 + CONFIDENCE) / 2, df=n - 1)
        half = t_crit * se

        x = global_step / NUM_AGENTS if multi_agent else global_step
        records.append((x, mean, mean - half, mean + half))

    records.sort(key=lambda r: r[0])
    if not records:
        return [], [], [], []
    steps, means, lows, highs = zip(*records)
    return list(steps), list(means), list(lows), list(highs)


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

    output_path = args.output or os.path.join(log_dir, "gridworld_bidding_mechanisms_learning_curves.png")

    fig, ax = plt.subplots(figsize=(8, 6))

    any_data = False
    for exp_prefix, label, multi_agent in EXPERIMENTS:
        run_dir = find_latest_run(log_dir, exp_prefix)
        if run_dir is None:
            print(f"  [skip] no run found for '{exp_prefix}'")
            continue

        steps, means, ci_lo, ci_hi = load_eval_series(run_dir, multi_agent)
        if not steps:
            print(f"  [skip] no eval data in {run_dir}")
            continue

        steps     = smooth(steps,  args.smooth)
        s_means   = smooth(means,  args.smooth)
        s_lo      = smooth(ci_lo,  args.smooth)
        s_hi      = smooth(ci_hi,  args.smooth)

        (line,) = ax.plot(steps, s_means, label=label, linewidth=2)
        ax.fill_between(steps, s_lo, s_hi, color=line.get_color(), alpha=0.15)

        print(f"  {label}: {len(steps)} points, "
              f"steps {steps[0]:,.0f}–{steps[-1]:,.0f}, "
              f"final {NUM_AGENTS}×avg_perf = {means[-1]:.2f}")
        any_data = True

    if not any_data:
        raise SystemExit("No data found — check --log-dir.")

    ax.set_xlabel("Env. Steps", fontsize=18)
    ax.set_ylabel("Avg. Performance", fontsize=18)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(loc="lower right", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
