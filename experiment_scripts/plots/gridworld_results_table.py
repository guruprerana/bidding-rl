#!/usr/bin/env python3
"""
Print a LaTeX table comparing gridworld algorithm performance.

Reads trained runs from logs/gridworld_all_methods_comparison/ and
algorithmic baseline data from algorithmic_baselines_eval_*.json in the same
directory. Picks the best evaluation iteration for each trained method
(highest avg_avg_performance), computes 8 × avg_performance with 95% CI
from the 20 per-episode values at that iteration, and renders a LaTeX table.

Usage:
    python experiment_scripts/plots/gridworld_results_table.py
"""

import glob
import json
import os
import re

import numpy as np
from scipy import stats


# ── configuration ──────────────────────────────────────────────────────────

LOG_DIR = "logs/gridworld_all_methods_comparison"
NUM_AGENTS = 8  # multiplier applied to avg_performance values

# Ordered table rows: (type, key, display_label)
#   type "run"      → find_latest_run(LOG_DIR, key) → best_iter_episodes()
#   type "baseline" → load_baseline_episodes(LOG_DIR, key)
METHODS = [
    ("run",      "bidding_cmp_all_pay",                    "All-Pay"),
    ("run",      "bidding_cmp_winner_pays",                "Winner-Pays"),
    ("run",      "bidding_cmp_winner_pays_others_reward",  "Winner-Pays (Others Rewarded)"),
    ("run",      "bidding_cmp_all_pay_norewshaping",       "All-Pay (No Reward Shaping)"),
    ("run",      "multiagentppo_localobs",                 "Multi-Agent PPO (Local Obs)"),
    ("run",      "bidding_cmp_dwn",                        "DWN"),
    ("run",      "ppo_singleagent_optunait21",             "Single-Agent PPO"),
    ("baseline", "nearest_target",                         "Nearest Target"),
    ("baseline", "least_time_left",                        "Least Time Left"),
]


# ── helpers ─────────────────────────────────────────────────────────────────

def find_latest_run(log_dir: str, prefix: str) -> str | None:
    """Return the path of the latest timestamped run directory for prefix."""
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


def _iter_number(fname: str) -> int | None:
    """Extract iteration number from iter_N_eval_stats.json, or None."""
    m = re.match(r"iter_(\d+)_eval_stats\.json$", fname)
    return int(m.group(1)) if m else None


def best_iter_data(run_dir: str) -> dict | None:
    """Return avg_performance and spread lists from the best iteration.

    'Best' = highest statistics.avg_avg_performance.  Falls back to
    statistics.avg_return if avg_avg_performance is absent (step-based files).
    Returns None when no usable data is found.
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
        sort_key = stat.get("avg_avg_performance", stat.get("avg_return"))
        if sort_key is None:
            continue
        records.append((sort_key, data))

    if not records:
        return None

    records.sort(key=lambda x: x[0])
    best = records[-1][1]
    ped  = best.get("per_episode_data", {})

    avg_perf = ped.get("avg_performance") or ped.get("returns")
    return {"avg_performance": avg_perf}


def load_baseline_data(log_dir: str, baseline_name: str) -> dict | None:
    """Return avg_performance and spread lists for a named baseline."""
    pattern = os.path.join(log_dir, "algorithmic_baselines_eval_*.json")
    found = sorted(glob.glob(pattern))
    if not found:
        return None
    with open(found[-1]) as fh:
        data = json.load(fh)
    per_ep = (
        data.get("results", {})
            .get(baseline_name, {})
            .get("per_episode", {})
    )
    avg_perf = per_ep.get("avg_performance_per_episode")
    return {"avg_performance": avg_perf}


def ci95(values: list[float], scale: float = 1.0) -> tuple[float, float]:
    """Return (mean, half_width) of a 95% t-interval after scaling."""
    arr = np.array(values, dtype=float) * scale
    n = len(arr)
    mean = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(n)
    half_width = stats.t.ppf(0.975, df=n - 1) * se
    return mean, half_width


# ── main ────────────────────────────────────────────────────────────────────

def _fmt(ci_val: tuple[float, float] | None) -> str:
    if ci_val is None:
        return "---"
    mean, hw = ci_val
    lo, hi = mean - hw, mean + hw
    ci = rf"{{\small\textcolor{{gray}}{{$[{lo:.2f},\ {hi:.2f}]$}}}}"
    return rf"${mean:.2f}$ {ci}"


def main() -> None:
    avg_perf_ci: dict[str, tuple[float, float] | None] = {}

    for mtype, key, label in METHODS:
        if mtype == "run":
            run_dir = find_latest_run(LOG_DIR, key)
            data = best_iter_data(run_dir) if run_dir else None
        else:
            data = load_baseline_data(LOG_DIR, key)

        if data is None:
            avg_perf_ci[key] = None
            continue

        ap = data.get("avg_performance")
        avg_perf_ci[key] = ci95(ap, scale=NUM_AGENTS) if ap else None

    # ── render LaTeX table ──────────────────────────────────────────────────
    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Gridworld performance (mean with 95\% CI) at the best"
        r" evaluation iteration. Values are $8 \times \text{avg performance}$.}"
    )
    lines.append(r"  \begin{tabular}{ll}")
    lines.append(r"    \hline")
    lines.append(r"    {\bf Algorithm} & {\bf Avg. Performance} \\")
    lines.append(r"    \hline")

    for mtype, key, label in METHODS:
        cell_ap = _fmt(avg_perf_ci[key])
        lines.append(f"    {label} & {cell_ap} \\\\")

    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \label{tab:gridworld_all_methods}")
    lines.append(r"\end{table}")

    table = "\n".join(lines)
    out_path = os.path.join(LOG_DIR, "gridworld_results_table.tex")
    with open(out_path, "w") as fh:
        fh.write(table + "\n")
    print(f"Table written to {out_path}")
    print(table)


if __name__ == "__main__":
    main()
