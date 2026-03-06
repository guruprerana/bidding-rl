#!/usr/bin/env python3
"""
Print a LaTeX table comparing Assault algorithm performance.

Reads trained runs from logs/assault_bidding_mechanism_comparison/ across
5 seeds. Picks the best evaluation iteration for each seed (highest
statistics.avg_score), computes mean score across seeds with standard
deviation, and renders a LaTeX table.

Usage:
    python experiment_scripts/plots/assault_results_table.py
"""

import json
import os
import re

import numpy as np


# ── configuration ──────────────────────────────────────────────────────────

LOG_DIR = "logs/assault_bidding_mechanism_comparison"
SEEDS = [1825, 410, 4507, 4013, 3658]

# Ordered table rows: (prefix, display_label)
METHODS = [
    ("assault_cmp_winner_pays",          "Winner-Pays"),
    ("assault_cmp_winner_pays_localobs", "Winner-Pays (Local Obs)"),
    ("assault_cmp_all_pay",              "All-Pay"),
    ("assault_cmp_all_pay_localobs",     "All-Pay (Local Obs)"),
    ("assault_cmp_dwn",                  "DWN"),
    ("assault_cmp_single_agent",         "Single-Agent PPO"),
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
        pattern_re = re.compile(rf"^{re.escape(exp_prefix)}_s{seed}_\d{{8}}_\d{{6}}$")
        matches = [
            d for d in entries
            if pattern_re.match(d) and os.path.isdir(os.path.join(log_dir, d))
        ]
        if matches:
            matches.sort()
            seed_runs[seed] = os.path.join(log_dir, matches[-1])
    return seed_runs


def best_iter_mean_score(run_dir: str) -> float | None:
    """Return mean score from the best iteration for a single seed.

    'Best' = highest statistics.avg_score.
    Checks evaluation/ then rollouts/ to support different run formats.
    """
    eval_dir = None
    for subdir in ("evaluation", "rollouts"):
        candidate = os.path.join(run_dir, subdir)
        if os.path.isdir(candidate):
            files = [f for f in os.listdir(candidate) if f.endswith("_eval_stats.json")]
            if files:
                eval_dir = candidate
                break

    if eval_dir is None:
        return None

    records = []
    for fname in os.listdir(eval_dir):
        if not fname.endswith("_eval_stats.json"):
            continue
        with open(os.path.join(eval_dir, fname)) as fh:
            data = json.load(fh)
        avg_score = data.get("statistics", {}).get("avg_score")
        scores = data.get("per_episode", {}).get("scores")
        if avg_score is None or not scores:
            continue
        records.append((avg_score, scores))

    if not records:
        return None

    records.sort(key=lambda x: x[0])
    best_scores = records[-1][1]
    return float(np.mean(best_scores))


def aggregate_best_score_across_seeds(seed_runs: dict[int, str]) -> tuple[float, float] | None:
    """Compute mean and std of best iteration score across seeds.

    Returns (mean, std) or None if no data.
    """
    values = []
    for seed, run_dir in seed_runs.items():
        score = best_iter_mean_score(run_dir)
        if score is not None:
            values.append(score)

    if not values:
        return None

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


# ── main ────────────────────────────────────────────────────────────────────

def _fmt(mean_std: tuple[float, float] | None) -> str:
    if mean_std is None:
        return "---"
    mean, std = mean_std
    std_str = rf"{{\small\textcolor{{gray}}{{$(\pm {std:.2f})$}}}}"
    return rf"${mean:.2f}$ {std_str}"


def main() -> None:
    results: dict[str, tuple[float, float] | None] = {}

    for prefix, label in METHODS:
        seed_runs = find_all_seed_runs(LOG_DIR, prefix, SEEDS)
        if not seed_runs:
            print(f"  [skip] no runs found for '{prefix}'")
            results[prefix] = None
            continue

        mean_std = aggregate_best_score_across_seeds(seed_runs)
        results[prefix] = mean_std

        if mean_std:
            mean, std = mean_std
            print(f"  {label}: {len(seed_runs)}/{len(SEEDS)} seeds, "
                  f"score = {mean:.2f} ± {std:.2f}")

    # ── render LaTeX table ──────────────────────────────────────────────────
    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Assault score (mean $\pm$ std across 5 seeds) at the best"
        r" evaluation iteration.}"
    )
    lines.append(r"  \begin{tabular}{ll}")
    lines.append(r"    \hline")
    lines.append(r"    {\bf Algorithm} & {\bf Score} \\")
    lines.append(r"    \hline")

    for prefix, label in METHODS:
        cell = _fmt(results[prefix])
        lines.append(f"    {label} & {cell} \\\\")

    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \label{tab:assault_methods}")
    lines.append(r"\end{table}")

    table = "\n".join(lines)
    out_path = os.path.join(LOG_DIR, "assault_results_table.tex")
    with open(out_path, "w") as fh:
        fh.write(table + "\n")
    print(f"\nTable written to {out_path}")
    print(table)


if __name__ == "__main__":
    main()
