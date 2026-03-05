#!/usr/bin/env python3
"""
Plot learning curves for the assault all-methods comparison experiment.

X axis: env steps (per agent; multi-agent global_step divided by 3).
Y axis: average score with 95% CI across eval episodes.
All runs are capped at the per-agent step count corresponding to multi-agent
iter 150 (150 × 128 × 512 = 9,830,400 steps per agent).

Usage:
    python experiment_scripts/plots/plot_assault_bidding_comparison.py
    python experiment_scripts/plots/plot_assault_bidding_comparison.py --log-dir logs/assault_all_methods_comparison
    python experiment_scripts/plots/plot_assault_bidding_comparison.py --smooth 3
"""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


LOG_DIR = "logs/assault_all_methods_comparison"
NUM_AGENTS = 3
# Per-agent step cap: matches multi-agent at iter 150 (150 × 128 × 512 = 9,830,400)
MAX_STEPS_PER_AGENT = 150 * 128 * 512
CONFIDENCE = 0.95

# (prefix, label, multi_agent)
# multi_agent=True → x axis divided by NUM_AGENTS
EXPERIMENTS = [
    ("assault_cmp_winner_pays",               "Winner-Pays",                  True),
    ("assault_cmp_all_pay",                   "All-Pay",                      True),
    # ("assault_cmp_winner_pays_others_reward", "Winner-Pays (Others Rewarded)", True),
    ("assault_ppo_multiagent_localobs",       "All-Pay (Local Obs)",   True),
    ("assault_ppo_single_agent_ppo_default_params", "Single-Policy PPO",       False),
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
    Only includes iterations <= MAX_ITERATION.
    """
    eval_dir = os.path.join(run_dir, "evaluation")
    if not os.path.isdir(eval_dir):
        return [], [], [], []

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

        episodes = np.array(episodes, dtype=float)
        n = len(episodes)
        mean = episodes.mean()
        se = episodes.std(ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf((1 + CONFIDENCE) / 2, df=n - 1)
        half = t_crit * se

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

    output_path = args.output or os.path.join(log_dir, "assault_bidding_mechanisms_learning_curves.png")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Collect all series first so we can match single-agent freq to multi-agent
    all_series = []
    for exp_prefix, label, multi_agent in EXPERIMENTS:
        run_dir = find_latest_run(log_dir, exp_prefix)
        if run_dir is None:
            print(f"  [skip] no run found for '{exp_prefix}'")
            continue
        steps, means, ci_lo, ci_hi = load_eval_series(run_dir, multi_agent)
        if not steps:
            print(f"  [skip] no eval data in {run_dir}")
            continue
        all_series.append((label, multi_agent, steps, means, ci_lo, ci_hi))

    # Target step gap = median step gap of multi-agent series
    def step_gap(steps):
        if len(steps) < 2:
            return None
        return np.median(np.diff(steps))

    ma_gaps = [step_gap(s[2]) for s in all_series if s[1] and step_gap(s[2]) is not None]
    target_gap = np.median(ma_gaps) if ma_gaps else None

    # Align all series to start at the same step (max of all first steps)
    start_step = max(s[2][0] for s in all_series if s[2])

    any_data = False
    for label, multi_agent, steps, means, ci_lo, ci_hi in all_series:
        # Trim to common start
        start_idx = next((i for i, s in enumerate(steps) if s >= start_step), 0)
        steps = steps[start_idx:]
        means = means[start_idx:]
        ci_lo = ci_lo[start_idx:]
        ci_hi = ci_hi[start_idx:]

        if not multi_agent and target_gap:
            # Keep only points at least target_gap apart
            keep = [0]
            for i in range(1, len(steps)):
                if steps[i] - steps[keep[-1]] >= target_gap:
                    keep.append(i)
            steps = [steps[i] for i in keep]
            means = [means[i] for i in keep]
            ci_lo = [ci_lo[i] for i in keep]
            ci_hi = [ci_hi[i] for i in keep]

        s_steps = smooth(steps,  args.smooth)
        s_means = smooth(means,  args.smooth)
        s_lo    = smooth(ci_lo,  args.smooth)
        s_hi    = smooth(ci_hi,  args.smooth)

        (line,) = ax.plot(s_steps, s_means, label=label, linewidth=2)
        ax.fill_between(s_steps, s_lo, s_hi, color=line.get_color(), alpha=0.15)

        print(f"  {label}: {len(steps)} points, "
              f"steps {steps[0]:,.0f}–{steps[-1]:,.0f}, "
              f"final score = {means[-1]:.1f}")
        any_data = True

    if not any_data:
        raise SystemExit("No data found — check --log-dir.")

    ax.set_xlabel("Env. Steps", fontsize=18)
    ax.set_ylabel("Score", fontsize=18)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(loc="lower right", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
