#!/usr/bin/env python3
"""
Print a LaTeX table comparing AirRaid bidding mechanism performance.

Reads trained runs from logs/airraid_bidding_mechanism_comparison/ across
available completed seeds. Computes each seed's mean score over its last N
evaluation iterations, then computes mean score across seeds with standard
deviation, and renders a LaTeX table.

Usage:
    python experiment_scripts/plots/airraid_results_table.py
    python experiment_scripts/plots/airraid_results_table.py --last 5
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np


LOG_DIR = "logs/airraid_bidding_mechanism_comparison"
NUM_ITERATIONS = 400
MAX_DWN_STEPS = 50_000_000

METHODS = [
    ("airraid_cmp_winner_pays_global_obs", "Winner-Pays"),
    ("airraid_cmp_all_pay_global_obs", "All-Pay"),
    ("airraid_cmp_single_agent", "Single-Agent PPO"),
]


def discover_seeds(log_dir: str, exp_prefix: str) -> list[int]:
    """Discover seeds with run directories for one experiment prefix."""
    if not os.path.isdir(log_dir):
        return []

    pattern_re = re.compile(rf"^{re.escape(exp_prefix)}_s(\d+)_\d{{8}}_\d{{6}}$")
    seeds = []
    for name in os.listdir(log_dir):
        match = pattern_re.match(name)
        if match and os.path.isdir(os.path.join(log_dir, name)):
            seeds.append(int(match.group(1)))
    return sorted(set(seeds))


def is_completed_run(run_dir: str) -> bool:
    final_ppo_eval = os.path.join(run_dir, "evaluation", f"iter_{NUM_ITERATIONS}_eval_stats.json")
    final_dwn_eval = os.path.join(run_dir, "rollouts", f"step_{MAX_DWN_STEPS}_eval_stats.json")
    return os.path.isfile(final_ppo_eval) or os.path.isfile(final_dwn_eval)


def find_completed_seed_runs(log_dir: str, exp_prefix: str) -> dict[int, str]:
    """Find the latest completed run directory for each available seed."""
    seed_runs = {}
    try:
        names = os.listdir(log_dir)
    except FileNotFoundError:
        return seed_runs

    for seed in discover_seeds(log_dir, exp_prefix):
        pattern_re = re.compile(rf"^{re.escape(exp_prefix)}_s{seed}_\d{{8}}_\d{{6}}$")
        matches = [
            name
            for name in names
            if pattern_re.match(name) and os.path.isdir(os.path.join(log_dir, name))
        ]
        matches.sort()
        for name in reversed(matches):
            run_dir = os.path.join(log_dir, name)
            if is_completed_run(run_dir):
                seed_runs[seed] = run_dir
                break
    return seed_runs


def last_n_eval_mean_score(run_dir: str, n: int) -> float | None:
    """Return mean avg_score over the last n evaluation iterations for one seed."""
    records = []
    for subdir, pattern in (
        ("evaluation", re.compile(r"^iter_(\d+)_eval_stats\.json$")),
        ("rollouts", re.compile(r"^step_(\d+)_eval_stats\.json$")),
    ):
        eval_dir = os.path.join(run_dir, subdir)
        if not os.path.isdir(eval_dir):
            continue

        for fname in os.listdir(eval_dir):
            match = pattern.match(fname)
            if not match:
                continue

            path = os.path.join(eval_dir, fname)
            with open(path) as f:
                data = json.load(f)

            avg_score = data.get("statistics", {}).get("avg_score")
            if avg_score is None:
                continue
            records.append((int(match.group(1)), float(avg_score)))

    if not records:
        return None

    records.sort(key=lambda record: record[0])
    return float(np.mean([score for _, score in records[-n:]]))


def aggregate_last_n_score_across_seeds(
    seed_runs: dict[int, str],
    n: int,
) -> tuple[float, float, list[int]] | None:
    """Compute mean and std of last-n-eval mean score across seeds."""
    values = []
    used_seeds = []
    for seed, run_dir in seed_runs.items():
        score = last_n_eval_mean_score(run_dir, n=n)
        if score is not None:
            values.append(score)
            used_seeds.append(seed)

    if not values:
        return None

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std, used_seeds


def fmt(mean_std_seeds: tuple[float, float, list[int]] | None) -> str:
    if mean_std_seeds is None:
        return "---"
    mean, std, _ = mean_std_seeds
    std_str = rf"{{\small\textcolor{{gray}}{{$(\pm {std:.2f})$}}}}"
    return rf"${mean:.2f}$ {std_str}"


def render_table(
    results: dict[str, tuple[float, float, list[int]] | None],
    last: int,
) -> str:
    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"  \centering")
    lines.append(
        rf"  \caption{{AirRaid score (mean $\pm$ std across completed seeds) averaged "
        rf"over each seed's last {last} evaluation iterations.}}"
    )
    lines.append(r"  \begin{tabular}{ll}")
    lines.append(r"    \hline")
    lines.append(r"    {\bf Algorithm} & {\bf Score} \\")
    lines.append(r"    \hline")

    for prefix, label in METHODS:
        lines.append(f"    {label} & {fmt(results[prefix])} \\\\")

    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \label{tab:airraid_methods}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default=LOG_DIR)
    parser.add_argument("--last", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results: dict[str, tuple[float, float, list[int]] | None] = {}
    for prefix, label in METHODS:
        seed_runs = find_completed_seed_runs(args.log_dir, prefix)
        result = aggregate_last_n_score_across_seeds(seed_runs, n=args.last)
        results[prefix] = result

        if result is None:
            print(f"  [skip] no completed runs found for '{prefix}'")
            continue

        mean, std, used_seeds = result
        print(
            f"  {label}: {len(used_seeds)} completed seeds "
            f"({', '.join(str(seed) for seed in used_seeds)}), "
            f"score = {mean:.2f} +/- {std:.2f}"
        )

    table = render_table(results, last=args.last)
    out_path = args.output or os.path.join(args.log_dir, "airraid_results_table.tex")
    with open(out_path, "w") as f:
        f.write(table + "\n")

    print(f"\nTable written to {out_path}")
    print(table)


if __name__ == "__main__":
    main()
