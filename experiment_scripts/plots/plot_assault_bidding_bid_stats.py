#!/usr/bin/env python3
"""
Plot bid-distribution and per-agent control-timesteps for the all_pay Assault experiment.

Reads eval stats from the experiment's evaluation/ directory and produces two
figures saved into the base log directory:

  1. bid_distribution.png  — grouped bar chart: agent on x-axis, one bar per bid value.
  2. control_timesteps.png — bar chart: avg timesteps controlled per agent.

Usage:
    python experiment_scripts/plots/plot_assault_bidding_bid_stats.py
    python experiment_scripts/plots/plot_assault_bidding_bid_stats.py \\
        --log-dir logs/assault_bidding_mechanism_comparison
"""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


BID_DIST_EXPERIMENTS = [
    ("assault_cmp_winner_pays",                      "Winner-Pays"),
    ("assault_cmp_all_pay",                          "All-Pay"),
    # ("assault_cmp_winner_pays_others_reward",         "Winner-Pays (Others Rewarded)"),
    # ("assault_ppo_multiagent_localobs",              "All-Pay (Local Obs)"),
]

CONTROL_EXPERIMENTS = [
    ("assault_cmp_winner_pays",                      "Winner-Pays"),
    ("assault_cmp_all_pay",                          "All-Pay"),
    # ("assault_cmp_winner_pays_others_reward",         "Winner-Pays (Others Rewarded)"),
    # ("assault_ppo_multiagent_localobs",              "All-Pay (Local Obs)"),
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
    Return the eval-stats dict from evaluation/.

    If iteration is given, load iter_<iteration>_eval_stats.json exactly.
    Otherwise load the file with the highest iteration number.
    """
    eval_dir = os.path.join(run_dir, "evaluation")
    if not os.path.isdir(eval_dir):
        return None

    if iteration is not None:
        path = os.path.join(eval_dir, f"iter_{iteration}_eval_stats.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    pattern = re.compile(r"^iter_(\d+)_eval_stats\.json$")
    best_score = float("-inf")
    best_data = None
    for fname in os.listdir(eval_dir):
        if not pattern.match(fname):
            continue
        with open(os.path.join(eval_dir, fname)) as f:
            data = json.load(f)
        score = data.get("statistics", {}).get("avg_score")
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
# Plot 1: bid distribution per agent (all_pay only)
# ---------------------------------------------------------------------------

def load_bid_distribution_per_agent(
    data: dict,
) -> tuple[list[int], list[int], list[list[float]], list[list[float]]]:
    """
    Return (agent_indices, bid_values, means_per_agent, errors_per_agent).

    means_per_agent[agent_i][bid_idx]  = avg count of that bid value for that agent.
    errors_per_agent[agent_i][bid_idx] = 95% CI half-width.

    Uses per-episode bid_counts_per_agent for CIs when available, falling back to
    the pre-averaged statistics dict.
    """
    per_ep = data.get("per_episode", {}).get("bid_counts_per_agent", [])
    if per_ep:
        # per_ep: list[episodes] of list[agents] of dict{bid_val: count}
        num_agents = max(len(ep) for ep in per_ep)
        all_bids: set[int] = set()
        for ep_agents in per_ep:
            for agent_bc in ep_agents:
                all_bids.update(int(k) for k in agent_bc.keys())
        bid_values = sorted(all_bids)

        means_pa, errors_pa = [], []
        for agent_i in range(num_agents):
            means, errors = [], []
            for bv in bid_values:
                vals = [
                    ep[agent_i].get(str(bv), ep[agent_i].get(bv, 0))
                    for ep in per_ep if agent_i < len(ep)
                ]
                means.append(float(np.mean(vals)) if vals else 0.0)
                errors.append(ci_half_width(vals) if vals else 0.0)
            means_pa.append(means)
            errors_pa.append(errors)
        return list(range(num_agents)), bid_values, means_pa, errors_pa

    # Fallback: pre-averaged statistics (no CI)
    avg_per_agent = data.get("statistics", {}).get("avg_bid_counts_per_agent", [])
    if not avg_per_agent:
        return [], [], [], []
    all_bids: set[int] = set()
    for agent_bc in avg_per_agent:
        all_bids.update(int(k) for k in agent_bc.keys())
    bid_values = sorted(all_bids)
    means_pa = [
        [float(agent_bc.get(str(bv), agent_bc.get(bv, 0))) for bv in bid_values]
        for agent_bc in avg_per_agent
    ]
    errors_pa = [[0.0] * len(bid_values)] * len(avg_per_agent)
    return list(range(len(avg_per_agent))), bid_values, means_pa, errors_pa


def plot_bid_distribution(data: dict, output_path: str):
    """
    Save a grouped bar chart: bid values on the x-axis, one bar per agent.
    """
    agent_indices, bid_values, means_pa, errors_pa = load_bid_distribution_per_agent(data)

    if not agent_indices:
        print("  [skip] no bid distribution data found")
        return

    num_agents = len(agent_indices)
    n_bars = num_agents
    bar_width = 0.7 / max(n_bars, 1)
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bar_width
    group_positions = np.arange(len(bid_values), dtype=float)

    fig, ax = plt.subplots(figsize=(max(6, len(bid_values) * 1.2 + 2), 6))

    for a in range(num_agents):
        heights = [means_pa[a][bid_idx] for bid_idx in range(len(bid_values))]
        x = group_positions + offsets[a]
        ax.bar(x, heights, width=bar_width, label=f"Policy {a}")

    print("  Bid distribution per agent:")
    for a in range(num_agents):
        print(f"    Policy {a}: " + ", ".join(
            f"bid{bv}→{means_pa[a][i]:.1f}" for i, bv in enumerate(bid_values)
        ))

    ax.set_xticks(group_positions)
    ax.set_xticklabels([str(bv) for bv in bid_values], fontsize=13)
    ax.set_xlabel("Bid Value", fontsize=18)
    ax.set_ylabel("Avg Count per Episode", fontsize=18)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(loc="upper right", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: per-agent control timesteps
# ---------------------------------------------------------------------------

def load_control_timesteps(data: dict) -> tuple[list[int], list[float]]:
    """Return (agent_indices, means) from an eval-stats dict."""
    per_ep = data.get("per_episode", {}).get("control_steps_per_agent", [])
    if per_ep:
        arr = np.array(per_ep, dtype=float)  # (num_episodes, num_agents)
        if arr.ndim != 2 or arr.shape[1] == 0:
            return [], []
        return list(range(arr.shape[1])), arr.mean(axis=0).tolist()

    avg_ctrl = data.get("statistics", {}).get("avg_control_timesteps_per_agent", [])
    if not avg_ctrl:
        return [], []
    return list(range(len(avg_ctrl))), list(avg_ctrl)


def plot_control_timesteps(experiments_data: list[tuple[str, dict]], output_path: str):
    """Save a grouped bar chart of avg control timesteps per policy."""
    parsed = []
    max_agents = 0
    for label, data in experiments_data:
        idxs, means = load_control_timesteps(data)
        if idxs:
            parsed.append((label, idxs, means))
            max_agents = max(max_agents, len(idxs))

    if not parsed:
        print("  [skip] no control-timestep data found")
        return

    n_groups = max_agents
    n_bars = len(parsed)
    bar_width = 0.7 / max(n_bars, 1)
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bar_width
    group_positions = np.arange(n_groups, dtype=float)

    fig, ax = plt.subplots(figsize=(max(6, n_groups * 0.9 + 2), 6))

    for bar_idx, (label, idxs, means) in enumerate(parsed):
        heights = means + [0.0] * (n_groups - len(means))
        x = group_positions + offsets[bar_idx]
        ax.bar(x, heights, width=bar_width, label=label)
        print(f"  {label} control steps: " + ", ".join(
            f"policy{i}→{h:.1f}" for i, h in enumerate(means)
        ))

    ax.set_xticks(group_positions)
    ax.set_xticklabels([f"Policy {i}" for i in range(n_groups)], fontsize=13)
    ax.set_xlabel("Policy", fontsize=18)
    ax.set_ylabel("Avg Timesteps Controlled per Episode", fontsize=18)
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
        default="logs/assault_all_methods_comparison",
        help="Base log directory produced by assault_bidding_comparison.py",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=150,
        metavar="N",
        help="Which eval iteration to load (default: 150). Pass -1 to use the final iteration.",
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

    control_output = args.output_control or os.path.join(log_dir, "assault_bidding_mechanisms_control_timesteps.png")

    target_iter = None if args.iteration == -1 else args.iteration

    # Load data for all experiments once
    all_data: dict[str, tuple[str, dict]] = {}
    all_prefixes = {prefix for prefix, _ in BID_DIST_EXPERIMENTS + CONTROL_EXPERIMENTS}
    for exp_prefix in all_prefixes:
        run_dir = find_latest_run(log_dir, exp_prefix)
        if run_dir is None:
            print(f"  [skip] no run found for prefix '{exp_prefix}'")
            continue
        data = find_eval_stats(run_dir, target_iter)
        if data is None:
            iter_desc = "final" if target_iter is None else str(target_iter)
            print(f"  [skip] no eval stats for iter={iter_desc} in {run_dir}")
            continue
        all_data[exp_prefix] = data

    # Bid distribution — one plot per experiment
    print("\n--- Bid Distribution per Agent ---")
    for exp_prefix, label in BID_DIST_EXPERIMENTS:
        if exp_prefix not in all_data:
            continue
        safe_label = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        bid_output = args.output_bid or os.path.join(
            log_dir, f"assault_bid_distribution_{safe_label}.png"
        )
        print(f"  {label} (iter {all_data[exp_prefix].get('iteration', '?')})")
        plot_bid_distribution(all_data[exp_prefix], bid_output)

    # Control timesteps — grouped bars across all experiments
    print("\n--- Control Timesteps per Agent ---")
    control_experiments_data = [
        (label, all_data[exp_prefix])
        for exp_prefix, label in CONTROL_EXPERIMENTS
        if exp_prefix in all_data
    ]
    plot_control_timesteps(control_experiments_data, control_output)


if __name__ == "__main__":
    main()
