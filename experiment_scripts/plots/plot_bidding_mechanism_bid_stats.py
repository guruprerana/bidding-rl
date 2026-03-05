#!/usr/bin/env python3
"""
Plot bid-distribution and per-agent control-timesteps for each multi-agent bidding mechanism.

Reads the final eval stats from each experiment's rollouts/ directory and produces two
figures saved into the base log directory:

  1. bid_distribution.png  — grouped bar chart: bid value vs. count, one group per mechanism.
  2. control_timesteps.png — grouped bar chart: agent index vs. avg timesteps controlled,
                             one group per mechanism.

Only multi-agent PPO experiments are included (DWN and single-agent baseline are excluded
because they don't record bid/control-timestep data in the same format).

Usage:
    python experiment_scripts/plots/plot_bidding_mechanism_bid_stats.py
    python experiment_scripts/plots/plot_bidding_mechanism_bid_stats.py \\
        --log-dir logs/gridworld_bidding_mechanism_comparison
"""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# Multi-agent PPO experiments to include
EXPERIMENTS = [
     ("bidding_cmp_winner_pays",               "Winner-Pays"),
    ("bidding_cmp_all_pay",                   "All-Pay"),
    # ("bidding_cmp_winner_pays_others_reward", "Winner-Pays (Others Rewarded)"),
    ("multiagentppo_localobs",                "All-Pay (Local Obs)"),
]

CONFIDENCE = 0.95


def find_latest_run(log_dir: str, exp_prefix: str) -> str | None:
    """Return path to the most recent subdirectory matching exp_prefix_YYYYMMDD_HHMMSS."""
    timestamp_re = re.compile(rf"^{re.escape(exp_prefix)}_\d{{8}}_\d{{6}}$")
    matches = [
        d for d in os.listdir(log_dir)
        if timestamp_re.match(d) and os.path.isdir(os.path.join(log_dir, d))
    ]
    if not matches:
        return None
    matches.sort()
    return os.path.join(log_dir, matches[-1])


def find_eval_stats(run_dir: str, iteration: int | None = None) -> dict | None:
    """
    Return the eval-stats dict from rollouts/.

    If iteration is given, load iter_<iteration>_eval_stats.json exactly.
    Otherwise load the file with the highest avg_avg_performance.
    """
    rollouts_dir = os.path.join(run_dir, "rollouts")
    if not os.path.isdir(rollouts_dir):
        return None

    if iteration is not None:
        path = os.path.join(rollouts_dir, f"iter_{iteration}_eval_stats.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    pattern = re.compile(r"^iter_(\d+)_eval_stats\.json$")
    best_score = float("-inf")
    best_data = None
    for fname in os.listdir(rollouts_dir):
        if not pattern.match(fname):
            continue
        with open(os.path.join(rollouts_dir, fname)) as f:
            data = json.load(f)
        score = data.get("statistics", {}).get("avg_avg_performance")
        if score is not None and score > best_score:
            best_score = score
            best_data = data

    return best_data


def ci_half_width(values: list[float]) -> float:
    """Return half-width of 95% t-confidence interval."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n < 2:
        return 0.0
    se = arr.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf((1 + CONFIDENCE) / 2, df=n - 1)
    return t_crit * se


# ---------------------------------------------------------------------------
# Plot 1: bid distribution
# ---------------------------------------------------------------------------

def load_bid_distribution(data: dict) -> tuple[list[int], list[float], list[float]]:
    """
    Return (bid_values, means, errors) from an eval-stats dict.

    Uses per-episode bid_counts for confidence intervals when available,
    falling back to the pre-averaged statistics dict.
    """
    per_ep = data.get("per_episode_data", {}).get("bid_counts", [])
    if per_ep:
        # Collect all bid values present across episodes
        all_bids: set[int] = set()
        for ep_counts in per_ep:
            all_bids.update(int(k) for k in ep_counts.keys())
        bid_values = sorted(all_bids)

        means, errors = [], []
        for bv in bid_values:
            vals = [ep.get(str(bv), ep.get(bv, 0)) for ep in per_ep]
            means.append(float(np.mean(vals)))
            errors.append(ci_half_width(vals))
        return bid_values, means, errors

    # Fallback: pre-averaged statistics (no CI available)
    avg_counts = data.get("statistics", {}).get("avg_bid_counts", {})
    if not avg_counts:
        return [], [], []
    bid_values = sorted(int(k) for k in avg_counts.keys())
    means = [float(avg_counts[str(bv)] if str(bv) in avg_counts else avg_counts.get(bv, 0))
             for bv in bid_values]
    errors = [0.0] * len(bid_values)
    return bid_values, means, errors


def plot_bid_distribution(experiments_data: list[tuple[str, dict]], output_path: str):
    """Save a grouped bar chart of bid distributions."""
    # Determine the union of all bid values across all experiments
    all_bid_values: set[int] = set()
    parsed = []
    for label, data in experiments_data:
        bvs, means, errs = load_bid_distribution(data)
        parsed.append((label, bvs, means, errs))
        all_bid_values.update(bvs)

    if not all_bid_values:
        print("  [skip] no bid distribution data found")
        return

    bid_values = sorted(all_bid_values)
    n_groups = len(bid_values)
    n_bars = len(parsed)
    bar_width = 0.7 / max(n_bars, 1)
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bar_width
    group_positions = np.arange(n_groups, dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, n_groups * 0.6 + 2), 6))

    for bar_idx, (label, bvs, means, errs) in enumerate(parsed):
        # Map bid values to the common bid_values axis
        bv_to_mean = dict(zip(bvs, means))
        bv_to_err = dict(zip(bvs, errs))
        heights = [bv_to_mean.get(bv, 0.0) for bv in bid_values]
        errors  = [bv_to_err.get(bv, 0.0)  for bv in bid_values]

        x = group_positions + offsets[bar_idx]
        ax.bar(x, heights, width=bar_width, label=label,
               yerr=errors, capsize=3, error_kw={"linewidth": 1.2})

        print(f"  {label} bid dist: " + ", ".join(
            f"{bv}→{h:.1f}" for bv, h in zip(bid_values, heights)
        ))

    ax.set_xticks(group_positions)
    ax.set_xticklabels([str(bv) for bv in bid_values], fontsize=13)
    ax.set_xlabel("Bid Value", fontsize=18)
    ax.set_ylabel("Avg Count per Episode", fontsize=18)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(loc="center right", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: per-agent control timesteps
# ---------------------------------------------------------------------------

def load_control_timesteps(data: dict) -> tuple[list[int], list[float], list[float]]:
    """
    Return (agent_indices, means, errors) from an eval-stats dict.

    Uses per-episode control_steps_per_agent for confidence intervals when available,
    falling back to the pre-averaged statistics dict.
    """
    per_ep = data.get("per_episode_data", {}).get("control_steps_per_agent", [])
    if per_ep:
        arr = np.array(per_ep, dtype=float)  # shape: (num_episodes, num_agents)
        if arr.ndim != 2 or arr.shape[1] == 0:
            return [], [], []
        num_agents = arr.shape[1]
        means = arr.mean(axis=0).tolist()
        errors = [ci_half_width(arr[:, a].tolist()) for a in range(num_agents)]
        return list(range(num_agents)), means, errors

    # Fallback: pre-averaged statistics
    avg_ctrl = data.get("statistics", {}).get("avg_control_timesteps_per_agent", [])
    if not avg_ctrl:
        return [], [], []
    return list(range(len(avg_ctrl))), list(avg_ctrl), [0.0] * len(avg_ctrl)


def plot_control_timesteps(experiments_data: list[tuple[str, dict]], output_path: str):
    """Save a grouped bar chart of avg control timesteps per agent."""
    parsed = []
    max_agents = 0
    for label, data in experiments_data:
        idxs, means, errs = load_control_timesteps(data)
        parsed.append((label, idxs, means, errs))
        max_agents = max(max_agents, len(idxs))

    if max_agents == 0:
        print("  [skip] no control-timestep data found")
        return

    n_groups = max_agents
    n_bars = len(parsed)
    bar_width = 0.7 / max(n_bars, 1)
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bar_width
    group_positions = np.arange(n_groups, dtype=float)

    fig, ax = plt.subplots(figsize=(max(6, n_groups * 0.9 + 2), 6))

    for bar_idx, (label, idxs, means, errs) in enumerate(parsed):
        # Pad to max_agents if necessary
        heights = means + [0.0] * (n_groups - len(means))
        errors  = errs  + [0.0] * (n_groups - len(errs))

        x = group_positions + offsets[bar_idx]
        ax.bar(x, heights, width=bar_width, label=label,
               yerr=errors, capsize=3, error_kw={"linewidth": 1.2})

        print(f"  {label} control steps: " + ", ".join(
            f"agent{i}→{h:.1f}" for i, h in enumerate(heights)
        ))

    ax.set_xticks(group_positions)
    ax.set_xticklabels([f"Policy {i}" for i in range(n_groups)], fontsize=13)
    ax.set_xlabel("Policy", fontsize=18)
    ax.set_ylabel("Avg Timesteps Controlled per Episode", fontsize=18)
    ax.set_ylim(bottom=150)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(loc="lower right", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        default="logs/gridworld_all_methods_comparison",
        help="Base log directory produced by bidding_mechanism_comparison.py",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        metavar="N",
        help="Which eval iteration to load (default: 200). Pass -1 to use the final iteration.",
    )
    parser.add_argument(
        "--output-bid",
        default=None,
        help="Override output path for the bid-distribution plot",
    )
    parser.add_argument(
        "--output-control",
        default=None,
        help="Override output path for the control-timesteps plot",
    )
    args = parser.parse_args()

    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        raise SystemExit(f"Log directory not found: {log_dir}")

    bid_output     = args.output_bid     or os.path.join(log_dir, "bidding_mechanisms_bid_distribution.png")
    control_output = args.output_control or os.path.join(log_dir, "bidding_mechanisms_control_timesteps.png")

    target_iter = None if args.iteration == -1 else args.iteration

    experiments_data: list[tuple[str, dict]] = []
    for exp_prefix, label in EXPERIMENTS:
        run_dir = find_latest_run(log_dir, exp_prefix)
        if run_dir is None:
            print(f"  [skip] no run found for prefix '{exp_prefix}'")
            continue
        data = find_eval_stats(run_dir, target_iter)
        if data is None:
            iter_desc = "final" if target_iter is None else str(target_iter)
            print(f"  [skip] no eval stats for iter={iter_desc} in {run_dir}")
            continue
        it = data.get("iteration", "?")
        print(f"  {label}: loaded eval stats (iter {it}) from {run_dir}")
        experiments_data.append((label, data))

    if not experiments_data:
        raise SystemExit("No data found — check --log-dir.")

    print("\n--- Bid Distribution ---")
    plot_bid_distribution(experiments_data, bid_output)

    print("\n--- Control Timesteps per Agent ---")
    plot_control_timesteps(experiments_data, control_output)


if __name__ == "__main__":
    main()
