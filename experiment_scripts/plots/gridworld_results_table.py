#!/usr/bin/env python3
"""
Print a LaTeX table comparing gridworld algorithm performance.

Reads trained runs from logs/gridworld_bidding_mechanism_comparison/ across
5 seeds. Picks the best evaluation iteration for each seed (highest
avg_avg_performance), computes 8 × avg_performance mean across seeds with
standard deviation, and renders a LaTeX table.

Usage:
    python experiment_scripts/plots/gridworld_results_table.py
"""

import json
import os
import re

import numpy as np


# ── configuration ──────────────────────────────────────────────────────────

LOG_DIR = "logs/gridworld_bidding_mechanism_comparison"
NUM_AGENTS = 8  # multiplier applied to avg_performance values
SEEDS = [410, 1825, 3658, 4013, 4507]

# Ordered table rows: (prefix, display_label)
METHODS = [
    ("bidding_cmp_winner_pays",                           "Winner-Pays"),
    ("bidding_cmp_winner_pays_localobs",                  "Winner-Pays (Local Obs)"),
    ("bidding_cmp_winner_pays_no_attn",                   "Winner-Pays (No Attn. Pool.)"),
    ("bidding_cmp_all_pay",                               "All-Pay"),
    ("bidding_cmp_all_pay_localobs",                      "All-Pay (Local Obs)"),
    ("bidding_cmp_all_pay_no_attn",                       "All-Pay (No Attn. Pool.)"),
    ("bidding_cmp_dwn",                                   "DWN"),
    ("bidding_cmp_single_agent",                          "Single-Agent PPO"),
    ("bidding_cmp_single_agent_nearest_shaping",          "Single-Agent (NS)"),
    ("bidding_cmp_single_agent_nearest_expiry_shaping",   "Single-Agent (ES)"),
]


# ── helpers ─────────────────────────────────────────────────────────────────

def find_all_seed_runs(log_dir: str, exp_prefix: str, seeds: list[int]) -> dict[int, str]:
    """Find run directories for all seeds of an experiment.

    Returns dict mapping seed → run_dir path.
    """
    seed_runs = {}
    try:
        entries = os.listdir(log_dir)
    except FileNotFoundError:
        return seed_runs

    for seed in seeds:
        # Pattern: {exp_prefix}_s{seed}_{timestamp}
        pattern_re = re.compile(rf"^{re.escape(exp_prefix)}_s{seed}_\d{{8}}_\d{{6}}$")
        matches = [
            d for d in entries
            if pattern_re.match(d) and os.path.isdir(os.path.join(log_dir, d))
        ]
        if matches:
            matches.sort()
            seed_runs[seed] = os.path.join(log_dir, matches[-1])  # Use latest if multiple
    return seed_runs


def best_iter_mean_performance(run_dir: str, is_dwn: bool = False) -> float | None:
    """Return mean performance from the best iteration for a single seed.

    'Best' = highest statistics.avg_avg_performance (or avg_return if not available).
    For methods with avg_performance (per-agent), scales by NUM_AGENTS.
    For DWN: uses 8*avg_targets_reached - avg_expired_targets.
    """
    rollouts_dir = os.path.join(run_dir, "rollouts")
    if not os.path.isdir(rollouts_dir):
        return None

    files = [f for f in os.listdir(rollouts_dir) if f.endswith("_eval_stats.json")]
    if not files:
        return None

    records = []
    for fname in files:
        path = os.path.join(rollouts_dir, fname)
        with open(path) as fh:
            data = json.load(fh)
        stat = data.get("statistics", {})

        # For DWN, compute custom performance metric for sorting
        if is_dwn:
            targets = stat.get("avg_targets_reached")
            expired = stat.get("avg_expired_targets")
            if targets is not None and expired is not None:
                sort_key = 8 * targets - expired
            else:
                continue
        else:
            sort_key = stat.get("avg_avg_performance", stat.get("avg_return"))
            if sort_key is None:
                continue

        records.append((sort_key, data))

    if not records:
        return None

    records.sort(key=lambda x: x[0])
    best = records[-1][1]
    stat = best.get("statistics", {})

    # For DWN, return the custom metric
    if is_dwn:
        targets = stat.get("avg_targets_reached")
        expired = stat.get("avg_expired_targets")
        if targets is not None and expired is not None:
            return 8 * targets - expired
        return None

    # For other methods, use per_episode_data
    ped = best.get("per_episode_data", {})

    # Check for avg_performance first (per-agent metric, needs scaling)
    avg_perf = ped.get("avg_performance")
    if avg_perf:
        return float(np.mean(avg_perf)) * NUM_AGENTS

    return None


def aggregate_best_performance_across_seeds(seed_runs: dict[int, str], is_dwn: bool = False) -> tuple[float, float] | None:
    """Compute mean and std of best iteration performance across seeds.

    Returns (mean, std) or None if insufficient data.
    """
    values = []
    for seed, run_dir in seed_runs.items():
        perf = best_iter_mean_performance(run_dir, is_dwn=is_dwn)
        if perf is not None:
            values.append(perf)

    if not values:
        return None

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std




# ── main ────────────────────────────────────────────────────────────────────

def _fmt(mean_std: tuple[float, float] | None) -> str:
    """Format mean with std in grey brackets."""
    if mean_std is None:
        return "---"
    mean, std = mean_std
    std_str = rf"{{\small\textcolor{{gray}}{{$(\pm {std:.2f})$}}}}"
    return rf"${mean:.2f}$ {std_str}"


def main() -> None:
    perf_results: dict[str, tuple[float, float] | None] = {}

    for key, label in METHODS:
        seed_runs = find_all_seed_runs(LOG_DIR, key, SEEDS)
        if not seed_runs:
            print(f"  [skip] no runs found for '{key}'")
            perf_results[key] = None
            continue

        # Special handling for DWN
        is_dwn = (key == "bidding_cmp_dwn")
        mean_std = aggregate_best_performance_across_seeds(seed_runs, is_dwn=is_dwn)
        perf_results[key] = mean_std

        if mean_std:
            mean, std = mean_std
            print(f"  {label}: {len(seed_runs)}/{len(SEEDS)} seeds, "
                  f"performance = {mean:.2f} ± {std:.2f}")

    # ── render LaTeX table ──────────────────────────────────────────────────
    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Gridworld performance (mean $\pm$ std across 5 seeds) at the best"
        r" evaluation iteration. Values are $8 \times \text{avg performance}$ for"
        r" bidding/single-agent methods, and $8 \times \text{targets reached} - \text{expired}$ for DWN.}"
    )
    lines.append(r"  \begin{tabular}{ll}")
    lines.append(r"    \hline")
    lines.append(r"    {\bf Algorithm} & {\bf Performance} \\")
    lines.append(r"    \hline")

    for key, label in METHODS:
        cell_ap = _fmt(perf_results[key])
        lines.append(f"    {label} & {cell_ap} \\\\")

    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \label{tab:gridworld_methods}")
    lines.append(r"\end{table}")

    table = "\n".join(lines)
    out_path = os.path.join(LOG_DIR, "gridworld_results_table.tex")
    with open(out_path, "w") as fh:
        fh.write(table + "\n")
    print(f"\nTable written to {out_path}")
    print(table)


if __name__ == "__main__":
    main()
