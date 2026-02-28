#!/usr/bin/env python3
"""
Plot avg min targets reached vs number of eval targets for each bidding mechanism.

Reads the JSON results produced by eval_bidding_mechanism_scaling.py and saves
a grouped bar chart to the same log directory.

Usage:
    python experiment_scripts/plots/plot_bidding_mechanism_scaling.py
    python experiment_scripts/plots/plot_bidding_mechanism_scaling.py --log-dir logs/eval_bidding_mechanism_scaling
    python experiment_scripts/plots/plot_bidding_mechanism_scaling.py --log-dir logs/eval_bidding_mechanism_scaling/20260227_123456
"""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


MECHANISMS = [
    ("all_pay",                   "All-Pay"),
    ("winner_pays",               "Winner-Pays"),
    ("winner_pays_others_reward", "Winner-Pays + Others Reward"),
]

EVAL_NUM_AGENTS_LIST = [8, 10, 12, 14]

CONFIDENCE = 0.95


def find_latest_run(log_dir: str) -> str:
    """Find the most recent timestamped subdirectory under log_dir."""
    timestamp_re = re.compile(r"^\d{8}_\d{6}$")
    matches = [
        d for d in os.listdir(log_dir)
        if timestamp_re.match(d) and os.path.isdir(os.path.join(log_dir, d))
    ]
    if not matches:
        return None
    matches.sort()
    return os.path.join(log_dir, matches[-1])


def load_summary(run_dir: str) -> dict | None:
    """Load all_results_summary.json from run_dir if present."""
    path = os.path.join(run_dir, "all_results_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def ci(values: list[float]) -> float:
    """Return half-width of 95% t-confidence interval."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n < 2:
        return 0.0
    se = arr.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf((1 + CONFIDENCE) / 2, df=n - 1)
    return t_crit * se


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        default="logs/eval_bidding_mechanism_scaling",
        help="Path to the eval log directory (parent or specific timestamped run)",
    )
    parser.add_argument("--output", default=None, help="Override output path for the plot")
    args = parser.parse_args()

    log_dir = args.log_dir

    # Accept either a timestamped run dir directly or the parent (find latest)
    timestamp_re = re.compile(r"^\d{8}_\d{6}$")
    if timestamp_re.match(os.path.basename(log_dir)):
        run_dir = log_dir
    else:
        run_dir = find_latest_run(log_dir)
        if run_dir is None:
            raise SystemExit(f"No timestamped run directories found in: {log_dir}")

    print(f"Loading results from: {run_dir}")
    summary = load_summary(run_dir)
    if summary is None:
        raise SystemExit(f"all_results_summary.json not found in {run_dir}")

    output_path = args.output or os.path.join(run_dir, "bidding_mechanism_scaling.png")

    n_groups = len(EVAL_NUM_AGENTS_LIST)
    n_bars = len(MECHANISMS)
    bar_width = 0.22
    group_positions = np.arange(n_groups)
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bar_width

    fig, ax = plt.subplots(figsize=(9, 6))

    any_data = False
    for bar_idx, (mech_key, mech_label) in enumerate(MECHANISMS):
        mech_data = summary.get(mech_key)
        if mech_data is None:
            print(f"  [skip] mechanism '{mech_key}' not found in summary")
            continue

        means, errors = [], []
        for n_agents in EVAL_NUM_AGENTS_LIST:
            entry = mech_data.get(str(n_agents))
            if entry is None:
                means.append(0.0)
                errors.append(0.0)
                continue
            per_ep = entry.get("per_episode_data", {}).get("min_targets_reached", [])
            if per_ep:
                means.append(float(np.mean(per_ep)))
                errors.append(ci(per_ep))
            else:
                means.append(entry["statistics"]["avg_min_targets_reached"])
                errors.append(0.0)

        x = group_positions + offsets[bar_idx]
        bars = ax.bar(x, means, width=bar_width, label=mech_label, yerr=errors,
                      capsize=4, error_kw={"linewidth": 1.2})

        print(f"  {mech_label}: " + ", ".join(
            f"{n}→{m:.2f}" for n, m in zip(EVAL_NUM_AGENTS_LIST, means)
        ))
        any_data = True

    if not any_data:
        raise SystemExit("No data to plot — check --log-dir.")

    ax.set_xticks(group_positions)
    ax.set_xticklabels([str(n) for n in EVAL_NUM_AGENTS_LIST], fontsize=12)
    ax.set_xlabel("Number of Eval Targets", fontsize=16)
    ax.set_ylabel("Min Targets Reached", fontsize=16)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(loc="upper right", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
