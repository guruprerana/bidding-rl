#!/usr/bin/env python3
"""
Print a LaTeX table comparing Assault algorithm performance.

Reads trained runs from logs/assault_all_methods_comparison/ and picks the
best evaluation iteration for each method (highest statistics.avg_score),
then computes mean score with 95% CI from per_episode.scores.

Usage:
    python experiment_scripts/plots/assault_results_table.py
"""

import json
import os
import re

import numpy as np
from scipy import stats


# ── configuration ──────────────────────────────────────────────────────────

LOG_DIR = "logs/assault_all_methods_comparison"

# Ordered table rows: (prefix, display_label)
METHODS = [
    ("assault_cmp_all_pay",                          "All-Pay"),
    ("assault_cmp_winner_pays",                      "Winner-Pays"),
    ("assault_cmp_winner_pays_others_reward",         "Winner-Pays (Others Rewarded)"),
    ("assault_ppo_multiagent_localobs",              "All-Pay (Local Obs)"),
    ("assault_cmp_dwn",                              "DWN"),
    ("assault_ppo_single_agent_ppo_default_params",  "Single-Policy PPO"),
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


def best_iter_scores(run_dir: str) -> list[float] | None:
    """Return per-episode scores from the best iteration (highest avg_score).

    Checks evaluation/ (iter_N_eval_stats.json) and rollouts/
    (step_N_eval_stats.json) to support different run formats.
    """
    for subdir in ("evaluation", "rollouts"):
        candidate = os.path.join(run_dir, subdir)
        if os.path.isdir(candidate):
            files = [f for f in os.listdir(candidate) if f.endswith("_eval_stats.json")]
            if files:
                eval_dir = candidate
                break
    else:
        return None

    if not files:
        return None

    records = []
    for fname in files:
        path = os.path.join(eval_dir, fname)  # type: ignore[possibly-undefined]
        with open(path) as fh:
            data = json.load(fh)
        avg_score = data.get("statistics", {}).get("avg_score")
        if avg_score is None:
            continue
        scores = data.get("per_episode", {}).get("scores")
        if not scores:
            continue
        records.append((avg_score, scores))

    if not records:
        return None

    records.sort(key=lambda x: x[0])
    return records[-1][1]


def ci95(values: list[float]) -> tuple[float, float]:
    """Return (mean, half_width) of a 95% t-interval."""
    arr = np.array(values, dtype=float)
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
    results: dict[str, tuple[float, float] | None] = {}

    for prefix, label in METHODS:
        run_dir = find_latest_run(LOG_DIR, prefix)
        if run_dir is None:
            print(f"  [skip] no run found for '{prefix}'")
            results[prefix] = None
            continue

        scores = best_iter_scores(run_dir)
        if scores is None:
            print(f"  [skip] no eval data in {run_dir}")
            results[prefix] = None
            continue

        results[prefix] = ci95(scores)
        mean, hw = results[prefix]
        print(f"  {label}: {mean:.2f} ± {hw:.2f}  (n={len(scores)})")

    # ── render LaTeX table ──────────────────────────────────────────────────
    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Assault score (mean with 95\% CI) at the best"
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
    lines.append(r"  \label{tab:assault_all_methods}")
    lines.append(r"\end{table}")

    table = "\n".join(lines)
    out_path = os.path.join(LOG_DIR, "assault_results_table.tex")
    with open(out_path, "w") as fh:
        fh.write(table + "\n")
    print(f"\nTable written to {out_path}")
    print(table)


if __name__ == "__main__":
    main()
