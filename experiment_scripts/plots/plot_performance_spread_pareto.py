#!/usr/bin/env python3
"""
Scatter plot of performance vs spread across evaluation iterations.

For each method and each eval checkpoint, plots:
  x = 8 × avg_avg_performance
  y = max(avg_performance_per_target) - min(avg_performance_per_target)

Each point is one eval iteration; each method gets its own colour/marker.
Baselines (single scalar) are shown as horizontal/vertical reference lines.

Usage:
    python experiment_scripts/plots/plot_performance_spread_pareto.py
"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np


# ── configuration ──────────────────────────────────────────────────────────

LOG_DIR = "logs/gridworld_all_methods_comparison"
NUM_AGENTS = 8

TRAINED_METHODS = [
    ("bidding_cmp_all_pay",                   "All-Pay"),
    ("bidding_cmp_winner_pays",               "Winner-Pays"),
    ("bidding_cmp_winner_pays_others_reward", "Winner-Pays (Others Rewarded)"),
    ("bidding_cmp_all_pay_norewshaping",      "All-Pay (No Reward Shaping)"),
    ("multiagentppo_localobs",                "Multi-Agent PPO (Local Obs)"),
    ("bidding_cmp_dwn",                       "DWN"),
    ("ppo_singleagent_optunait21",            "Single-Agent PPO"),
]

BASELINES = [
    ("nearest_target",  "Nearest Target"),
    ("least_time_left", "Least Time Left"),
]


# ── helpers ─────────────────────────────────────────────────────────────────

def find_latest_run(log_dir: str, prefix: str) -> str | None:
    timestamp_re = re.compile(rf"^{re.escape(prefix)}_\d{{8}}_\d{{6}}$")
    try:
        entries = os.listdir(log_dir)
    except FileNotFoundError:
        return None
    matches = [
        d for d in entries
        if timestamp_re.match(d) and os.path.isdir(os.path.join(log_dir, d))
    ]
    if not matches:
        return None
    matches.sort()
    return os.path.join(log_dir, matches[-1])


def load_trained_points(run_dir: str) -> tuple[list[float], list[float]]:
    """Return (perf_list, spread_list) — one value per eval checkpoint."""
    rollouts_dir = os.path.join(run_dir, "rollouts")
    if not os.path.isdir(rollouts_dir):
        return [], []

    xs, ys = [], []
    for fname in os.listdir(rollouts_dir):
        if not fname.endswith("_eval_stats.json"):
            continue
        with open(os.path.join(rollouts_dir, fname)) as fh:
            data = json.load(fh)
        stat = data.get("statistics", {})
        avg_avg = stat.get("avg_avg_performance")
        per_target = stat.get("avg_performance_per_target")
        if avg_avg is None or not per_target:
            continue
        xs.append(NUM_AGENTS * avg_avg)
        ys.append(float(max(per_target) - min(per_target)))

    return xs, ys


def load_baseline_point(log_dir: str, name: str) -> tuple[float, float] | None:
    """Return (perf, spread) for a named baseline."""
    pattern = os.path.join(log_dir, "algorithmic_baselines_eval_*.json")
    found = sorted(glob.glob(pattern))
    if not found:
        return None
    with open(found[-1]) as fh:
        data = json.load(fh)
    entry = data.get("results", {}).get(name, {})
    avg_avg    = entry.get("summary", {}).get("avg_avg_performance")
    per_target = entry.get("summary", {}).get("avg_performance_per_target")
    if avg_avg is None or not per_target:
        return None
    return (
        NUM_AGENTS * avg_avg,
        float(max(per_target) - min(per_target)),
    )


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    markers = ["o", "s", "^", "D", "v", "P", "X"]
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    any_data = False
    for i, (prefix, label) in enumerate(TRAINED_METHODS):
        run_dir = find_latest_run(LOG_DIR, prefix)
        if run_dir is None:
            print(f"  [skip] no run found for '{prefix}'")
            continue
        xs, ys = load_trained_points(run_dir)
        if not xs:
            print(f"  [skip] no eval data in {run_dir}")
            continue

        color  = prop_cycle[i % len(prop_cycle)]
        marker = markers[i % len(markers)]
        ax.scatter(xs, ys, label=label, color=color, marker=marker,
                   s=50, alpha=0.8, zorder=3)
        print(f"  {label}: {len(xs)} points, "
              f"perf=[{min(xs):.1f}, {max(xs):.1f}], "
              f"spread=[{min(ys):.2f}, {max(ys):.2f}]")
        any_data = True

    # Baselines as star markers with dashed outlines
    baseline_colors = ["black", "dimgray"]
    for j, (name, label) in enumerate(BASELINES):
        pt = load_baseline_point(LOG_DIR, name)
        if pt is None:
            print(f"  [skip] baseline '{name}' not found")
            continue
        x, y = pt
        ax.scatter([x], [y], label=label, color=baseline_colors[j],
                   marker="*", s=200, zorder=4)
        print(f"  {label}: perf={x:.2f}, spread={y:.2f}")
        any_data = True

    if not any_data:
        raise SystemExit("No data found — check LOG_DIR.")

    ax.set_xlabel(f"{NUM_AGENTS} × Avg Performance", fontsize=14)
    ax.set_ylabel("Spread (max − min avg target perf)", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(LOG_DIR, "performance_spread_pareto.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
