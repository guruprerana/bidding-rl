#!/usr/bin/env python3
"""
Print a LaTeX table comparing algorithm performance across environments.

Pools per-episode data from the last N eval checkpoints for each algorithm,
then reports mean ± std.

Usage:
    python experiment_scripts/plots/latex_results_table.py
    python experiment_scripts/plots/latex_results_table.py --last 5
"""

import argparse
import json
import os
import re

import numpy as np


# ── experiment registry ────────────────────────────────────────────────────

ALGORITHMS = [
    ("all_pay",                   "All-Pay"),
    ("winner_pays",               "Winner-Pays"),
    ("winner_pays_others_reward", "Winner-Pays (Others Rewarded)"),
    ("single_agent",              "Single-Agent"),
    ("dwn",                       "DWN"),
]

ENVIRONMENTS = [
    {
        "label":       "Gridworld",
        "log_dir":     "logs/gridworld_bidding_mechanism_comparison",
        "prefix_tmpl": "bidding_cmp_{key}",
        "eval_subdirs": ["rollouts"],
        "episode_key": ("per_episode_data", "min_targets_reached"),
        "col_header":  "Min Targets Reached",
    },
    {
        "label":       "Assault",
        "log_dir":     "logs/assault_bidding_mechanism_comparison",
        "prefix_tmpl": "assault_cmp_{key}",
        "eval_subdirs": ["evaluation", "rollouts"],  # PPO uses evaluation/, DWN uses rollouts/
        "episode_key": ("per_episode", "scores"),
        "col_header":  "Score",
    },
]


# ── helpers ────────────────────────────────────────────────────────────────

def find_latest_run(log_dir: str, exp_prefix: str) -> str | None:
    timestamp_re = re.compile(rf"^{re.escape(exp_prefix)}_\d{{8}}_\d{{6}}$")
    matches = [
        d for d in os.listdir(log_dir)
        if timestamp_re.match(d) and os.path.isdir(os.path.join(log_dir, d))
    ]
    if not matches:
        return None
    matches.sort()
    return os.path.join(log_dir, matches[-1])


def load_last_n_episodes(
    run_dir: str,
    eval_subdirs: list[str],
    episode_key: tuple[str, str],
    n: int,
) -> np.ndarray | None:
    """Return a flat array of per-episode values from the last n eval files.
    Tries each subdir in eval_subdirs in order, using the first that has files.
    """
    pattern = re.compile(r".*_eval_stats\.json$")

    subdir = None
    files = []
    for candidate in eval_subdirs:
        path = os.path.join(run_dir, candidate)
        if os.path.isdir(path):
            found = [f for f in os.listdir(path) if pattern.match(f)]
            if found:
                subdir, files = path, found
                break

    if subdir is None:
        return None

    # Parse global_step from each file and sort
    records = []
    for fname in files:
        path = os.path.join(subdir, fname)
        with open(path) as f:
            data = json.load(f)
        step = data.get("global_step")
        if step is not None:
            records.append((step, data))
    records.sort(key=lambda x: x[0])

    last_n = records[-n:]
    top_key, sub_key = episode_key
    episodes = []
    for _, data in last_n:
        values = data.get(top_key, {}).get(sub_key, [])
        episodes.extend(values)

    return np.array(episodes, dtype=float) if episodes else None


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--last", type=int, default=5, metavar="N",
        help="Number of final eval checkpoints to average over (default: 5)",
    )
    args = parser.parse_args()

    # Collect results[env_label][algo_key] = (mean, std) or None
    results = {env["label"]: {} for env in ENVIRONMENTS}

    for env in ENVIRONMENTS:
        for algo_key, _ in ALGORITHMS:
            prefix = env["prefix_tmpl"].format(key=algo_key)
            run_dir = find_latest_run(env["log_dir"], prefix)
            if run_dir is None:
                results[env["label"]][algo_key] = None
                continue
            episodes = load_last_n_episodes(
                run_dir, env["eval_subdirs"], env["episode_key"], args.last
            )
            if episodes is None or len(episodes) == 0:
                results[env["label"]][algo_key] = None
            else:
                from scipy import stats
                n = len(episodes)
                mean = episodes.mean()
                se = episodes.std(ddof=1) / np.sqrt(n)
                half_width = stats.t.ppf(0.975, df=n - 1) * se
                results[env["label"]][algo_key] = (mean, half_width)

    # ── render LaTeX table ─────────────────────────────────────────────────
    col_headers = [env["col_header"] for env in ENVIRONMENTS]
    env_labels  = [env["label"] for env in ENVIRONMENTS]

    col_spec = "l" * (1 + len(ENVIRONMENTS))

    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"  \centering")
    lines.append(
        rf"  \caption{{Performance (mean with 95\% CI) averaged over the last {args.last} "
        r"evaluation checkpoints.}"
    )
    lines.append(rf"  \begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \hline")

    # Bold header row
    header_cells = [r"{\bf Algorithm}"] + [
        r"{{\bf {0} ({1})}}".format(env_labels[i], col_headers[i])
        for i in range(len(ENVIRONMENTS))
    ]
    lines.append("    " + " & ".join(header_cells))
    lines.append(r"    \\ \hline")

    for algo_key, algo_label in ALGORITHMS:
        cells = [algo_label]
        for env in ENVIRONMENTS:
            val = results[env["label"]][algo_key]
            if val is None:
                cells.append("—")
            else:
                mean, half_width = val
                lo, hi = mean - half_width, mean + half_width
                ci = rf"{{\small\textcolor{{gray}}{{$[{lo:.2f},\ {hi:.2f}]$}}}}"
                cells.append(rf"${mean:.2f}$ {ci}")
        lines.append("    " + " & ".join(cells) + r" \\")

    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \label{tab:bidding_comparison}")
    lines.append(r"\end{table}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
