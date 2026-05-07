#!/usr/bin/env python3
"""
Print a LaTeX table comparing gridworld algorithm performance.

Reads trained runs from logs/gridworld_bidding_mechanism_comparison/ across
available seeds. Averages performance over the last 5 available evaluation
iterations for each seed, then computes the mean and standard deviation across
seed averages, and renders a LaTeX table.

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
SEEDS = [410, 1825, 3658, 4013, 4507, 5215, 6803, 6861, 7819, 8057]

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


def _eval_index_from_filename(fname: str) -> int | None:
    match = re.match(r"(?:iter|step)_(\d+)_eval_stats\.json$", fname)
    if not match:
        return None
    return int(match.group(1))


def _performance_from_eval(data: dict, is_dwn: bool = False) -> float | None:
    """Return table performance from one eval stats payload."""
    stat = data.get("statistics", {})

    if is_dwn:
        targets = stat.get("avg_targets_reached")
        expired = stat.get("avg_expired_targets")
        if targets is None or expired is None:
            return None
        return float(NUM_AGENTS * targets - expired)

    ped = data.get("per_episode_data", {})
    avg_perf = ped.get("avg_performance")
    if avg_perf:
        return float(np.mean(avg_perf)) * NUM_AGENTS

    return None


def recent_eval_mean_performance(run_dir: str, is_dwn: bool = False, num_evals: int = 5) -> float | None:
    """Return mean performance over the last available eval iterations for one seed."""
    rollouts_dir = os.path.join(run_dir, "rollouts")
    if not os.path.isdir(rollouts_dir):
        return None

    files = []
    for fname in os.listdir(rollouts_dir):
        eval_index = _eval_index_from_filename(fname)
        if eval_index is not None:
            files.append((eval_index, fname))

    if not files:
        return None

    values = []
    for _, fname in sorted(files)[-num_evals:]:
        path = os.path.join(rollouts_dir, fname)
        with open(path) as fh:
            data = json.load(fh)
        perf = _performance_from_eval(data, is_dwn=is_dwn)
        if perf is not None:
            values.append(perf)

    if not values:
        return None

    return float(np.mean(values))


def aggregate_recent_performance_across_seeds(seed_runs: dict[int, str], is_dwn: bool = False) -> tuple[float, float] | None:
    """Compute mean and std of recent-eval seed averages across seeds.

    Returns (mean, std) or None if insufficient data.
    """
    values = []
    for seed, run_dir in seed_runs.items():
        perf = recent_eval_mean_performance(run_dir, is_dwn=is_dwn)
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
        mean_std = aggregate_recent_performance_across_seeds(seed_runs, is_dwn=is_dwn)
        perf_results[key] = mean_std

        if mean_std:
            mean, std = mean_std
            print(f"  {label}: {len(seed_runs)}/{len(SEEDS)} candidate seeds, "
                  f"performance = {mean:.2f} ± {std:.2f}")

    # ── render LaTeX table ──────────────────────────────────────────────────
    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Gridworld performance (mean $\pm$ std across available seeds) averaged over"
        r" each seed's last 5 available evaluation iterations. Values are $8 \times \text{avg performance}$ for"
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
