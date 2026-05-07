#!/usr/bin/env python3
"""
Plot bid-distribution and per-agent control timesteps for one AirRaid PPO seed.

This mirrors plot_assault_bidding_bid_stats.py: it loads one run per method
instead of aggregating across seeds.

Usage:
    python experiment_scripts/plots/plot_airraid_bidding_bid_stats.py
    python experiment_scripts/plots/plot_airraid_bidding_bid_stats.py --seed 1825
    python experiment_scripts/plots/plot_airraid_bidding_bid_stats.py \\
        --log-dir logs/airraid_bidding_mechanism_comparison --iteration 400
"""

from __future__ import annotations

import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BID_DIST_EXPERIMENTS = [
    ("airraid_cmp_winner_pays", "Winner-Pays (Local Obs)"),
    ("airraid_cmp_all_pay", "All-Pay (Local Obs)"),
]

CONTROL_EXPERIMENTS = [
    ("airraid_cmp_winner_pays", "Winner-Pays (Local Obs)"),
    ("airraid_cmp_all_pay", "All-Pay (Local Obs)"),
]


def find_seed_run(log_dir: str, exp_prefix: str, seed: int) -> str | None:
    """Return the newest run directory matching exp_prefix_s<seed>_timestamp."""
    pattern = re.compile(rf"^{re.escape(exp_prefix)}_s{seed}_\d{{8}}_\d{{6}}$")
    matches = [
        d
        for d in os.listdir(log_dir)
        if pattern.match(d) and os.path.isdir(os.path.join(log_dir, d))
    ]
    if not matches:
        return None
    matches.sort()
    return os.path.join(log_dir, matches[-1])


def find_eval_stats(run_dir: str, iteration: int | None) -> dict | None:
    """Load eval stats for one run.

    If iteration is None, use the eval checkpoint with the highest avg_score.
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


def load_bid_distribution_per_agent(data: dict) -> tuple[list[int], list[int], list[list[float]]]:
    """Return agent indices, bid values, and mean bid counts per agent."""
    avg_per_agent = data.get("statistics", {}).get("avg_bid_counts_per_agent", [])
    if not avg_per_agent:
        return [], [], []

    all_bids: set[int] = set()
    for agent_counts in avg_per_agent:
        all_bids.update(int(k) for k in agent_counts.keys())
    bid_values = sorted(all_bids)
    means_per_agent = [
        [float(agent_counts.get(str(bid), agent_counts.get(bid, 0.0))) for bid in bid_values]
        for agent_counts in avg_per_agent
    ]
    return list(range(len(avg_per_agent))), bid_values, means_per_agent


def plot_bid_distribution(data: dict, output_path: str) -> None:
    """Save grouped bars: bid value on x-axis, one bar per policy."""
    agent_indices, bid_values, means_per_agent = load_bid_distribution_per_agent(data)
    if not agent_indices:
        print("  [skip] no bid distribution data found")
        return

    num_agents = len(agent_indices)
    bar_width = 0.7 / max(num_agents, 1)
    offsets = np.linspace(-(num_agents - 1) / 2, (num_agents - 1) / 2, num_agents) * bar_width
    positions = np.arange(len(bid_values), dtype=float)

    fig, ax = plt.subplots(figsize=(max(6, len(bid_values) * 1.2 + 2), 6))

    for agent_idx in range(num_agents):
        heights = means_per_agent[agent_idx]
        ax.bar(positions + offsets[agent_idx], heights, width=bar_width, label=f"Policy {agent_idx}")

    print("  Bid distribution per agent:")
    for agent_idx in range(num_agents):
        print(
            f"    Policy {agent_idx}: "
            + ", ".join(
                f"bid{bid}->{means_per_agent[agent_idx][i]:.1f}"
                for i, bid in enumerate(bid_values)
            )
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(bid) for bid in bid_values], fontsize=13)
    ax.set_xlabel("Bid Value", fontsize=18)
    ax.set_ylabel("Avg Count per Episode", fontsize=18)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(loc="upper left", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved: {output_path}")
    plt.close(fig)


def load_control_timesteps(data: dict) -> tuple[list[int], list[float], list[float]]:
    """Return policy indices, average timesteps, and std per episode."""
    avg_ctrl = data.get("statistics", {}).get("avg_control_timesteps_per_agent", [])
    if not avg_ctrl:
        return [], [], []

    std_ctrl = data.get("statistics", {}).get("std_control_timesteps_per_agent", [])
    if not std_ctrl:
        std_ctrl = [
            data.get("statistics", {}).get(f"std_agent_{i}_control_steps", 0.0)
            for i in range(len(avg_ctrl))
        ]
    return (
        list(range(len(avg_ctrl))),
        [float(v) for v in avg_ctrl],
        [float(v) for v in std_ctrl],
    )


def plot_control_timesteps(experiments_data: list[tuple[str, dict]], output_path: str) -> None:
    """Save grouped bars of average control timesteps per policy."""
    parsed = []
    max_agents = 0
    for label, data in experiments_data:
        indices, means, stds = load_control_timesteps(data)
        if indices:
            parsed.append((label, means, stds))
            max_agents = max(max_agents, len(indices))

    if not parsed:
        print("  [skip] no control-timestep data found")
        return

    bar_width = 0.7 / max(len(parsed), 1)
    offsets = np.linspace(-(len(parsed) - 1) / 2, (len(parsed) - 1) / 2, len(parsed)) * bar_width
    positions = np.arange(max_agents, dtype=float)

    fig, ax = plt.subplots(figsize=(max(6, max_agents * 0.9 + 2), 6))

    for bar_idx, (label, means, stds) in enumerate(parsed):
        heights = means + [0.0] * (max_agents - len(means))
        errors = stds + [0.0] * (max_agents - len(stds))
        ax.bar(positions + offsets[bar_idx], heights, width=bar_width, label=label)
        ax.errorbar(
            positions + offsets[bar_idx],
            heights,
            yerr=errors,
            fmt="none",
            ecolor="black",
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
        )
        print(
            f"  {label} control steps: "
            + ", ".join(
                f"policy{i}->{h:.1f} +/- {stds[i]:.1f}"
                for i, h in enumerate(means)
            )
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([f"Policy {i}" for i in range(max_agents)], fontsize=13)
    ax.set_xlabel("Policy", fontsize=18)
    ax.set_ylabel("Avg Timesteps Controlled per Episode", fontsize=18)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(loc="lower right", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved: {output_path}")
    plt.close(fig)


def safe_label(label: str) -> str:
    return (
        label.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        default="logs/airraid_bidding_mechanism_comparison",
        help="Base log directory produced by airraid_bidding_comparison.py",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1825,
        help="Seed to plot. Default 1825 has completed all-pay and winner-pays PPO runs.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=400,
        metavar="N",
        help="Which eval iteration to load. Pass -1 to use the best-scoring eval checkpoint.",
    )
    parser.add_argument("--output-bid", default=None)
    parser.add_argument("--output-control", default=None)
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        raise SystemExit(f"Log directory not found: {args.log_dir}")

    target_iter = None if args.iteration == -1 else args.iteration
    all_data: dict[str, dict] = {}

    print(f"Using seed {args.seed}")
    all_prefixes = {prefix for prefix, _ in BID_DIST_EXPERIMENTS + CONTROL_EXPERIMENTS}
    for exp_prefix in all_prefixes:
        run_dir = find_seed_run(args.log_dir, exp_prefix, args.seed)
        if run_dir is None:
            print(f"  [skip] no run found for prefix '{exp_prefix}' seed {args.seed}")
            continue
        data = find_eval_stats(run_dir, target_iter)
        if data is None:
            iter_desc = "best" if target_iter is None else str(target_iter)
            print(f"  [skip] no eval stats for iter={iter_desc} in {run_dir}")
            continue
        all_data[exp_prefix] = data

    print("\n--- Bid Distribution per Agent ---")
    for exp_prefix, label in BID_DIST_EXPERIMENTS:
        if exp_prefix not in all_data:
            continue
        bid_output = args.output_bid or os.path.join(
            args.log_dir,
            f"airraid_bid_distribution_{exp_prefix.removeprefix('airraid_cmp_')}.png",
        )
        print(f"  {label} (iter {all_data[exp_prefix].get('iteration', '?')})")
        plot_bid_distribution(all_data[exp_prefix], bid_output)

    print("\n--- Control Timesteps per Agent ---")
    control_data = [
        (label, all_data[exp_prefix])
        for exp_prefix, label in CONTROL_EXPERIMENTS
        if exp_prefix in all_data
    ]
    control_output = args.output_control or os.path.join(
        args.log_dir,
        "airraid_bidding_mechanisms_control_timesteps.png",
    )
    plot_control_timesteps(control_data, control_output)


if __name__ == "__main__":
    main()
