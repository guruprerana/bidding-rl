#!/usr/bin/env python3
"""
Plot PPO learning curves for the AirRaid local-observation bidding comparison.

X axis: env steps per agent. Multi-agent global_step is divided by 3 so it is
comparable to single-agent PPO.
Y axis: average eval score with +/- 1 std shaded across seed means.

Usage:
    python experiment_scripts/plots/plot_airraid_bidding_comparison.py
    python experiment_scripts/plots/plot_airraid_bidding_comparison.py --log-dir logs/airraid_bidding_mechanism_comparison
    python experiment_scripts/plots/plot_airraid_bidding_comparison.py --smooth 3 --min-seeds 2
    python experiment_scripts/plots/plot_airraid_bidding_comparison.py --seeds 5215,6803,6861
    python experiment_scripts/plots/plot_airraid_bidding_comparison.py --include-partial
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOG_DIR = "logs/airraid_bidding_mechanism_comparison"
NUM_AGENTS = 3
NUM_ITERATIONS = 400
NUM_ENVS = 128
NUM_STEPS = 512
MAX_STEPS_PER_AGENT = NUM_ITERATIONS * NUM_ENVS * NUM_STEPS
DEFAULT_SEEDS = None

# (prefix, label, multi_agent)
# multi_agent=True means x axis is global_step / NUM_AGENTS.
EXPERIMENTS = [
    ("airraid_cmp_winner_pays_global_obs", "Winner-Pays", True),
    ("airraid_cmp_winner_pays", "Winner-Pays (Local Obs)", True),
    ("airraid_cmp_all_pay_global_obs", "All-Pay", True),
    ("airraid_cmp_all_pay", "All-Pay (Local Obs)", True),
    ("airraid_cmp_single_agent", "Single-Agent PPO", False),
]


def is_completed_run(log_dir: str, exp_prefix: str, seed: int, run_dir: str) -> bool:
    """Return whether this seed finished.

    Prefer the final iteration eval file because worker logs can be truncated
    while the copied result folder is complete. If the final eval is missing,
    fall back to the worker log completion marker.
    """
    final_eval = os.path.join(run_dir, "evaluation", f"iter_{NUM_ITERATIONS}_eval_stats.json")
    if os.path.isfile(final_eval):
        return True

    log_path = os.path.join(log_dir, f"seed_{seed}", f"{exp_prefix}_s{seed}.log")
    if os.path.isfile(log_path):
        with open(log_path, errors="ignore") as f:
            return "EXPERIMENT COMPLETED" in f.read()

    return False


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


def parse_seeds(value: str | None) -> list[int] | None:
    if not value:
        return DEFAULT_SEEDS
    return [int(seed.strip()) for seed in value.split(",") if seed.strip()]


def find_seed_runs(
    log_dir: str,
    exp_prefix: str,
    seeds: list[int],
    include_partial: bool,
) -> dict[int, str]:
    """Find the latest run directory for each seed of an experiment."""
    seed_runs = {}
    if not os.path.isdir(log_dir):
        return seed_runs

    names = os.listdir(log_dir)
    for seed in seeds:
        pattern_re = re.compile(rf"^{re.escape(exp_prefix)}_s{seed}_\d{{8}}_\d{{6}}$")
        matches = [
            name
            for name in names
            if pattern_re.match(name) and os.path.isdir(os.path.join(log_dir, name))
        ]
        if matches:
            matches.sort()
            run_dir = os.path.join(log_dir, matches[-1])
            if include_partial or is_completed_run(log_dir, exp_prefix, seed, run_dir):
                seed_runs[seed] = run_dir
    return seed_runs


def load_eval_series(run_dir: str, multi_agent: bool, metric: str) -> dict[float, float]:
    """Return {per-agent step: metric mean} for one seed."""
    eval_dir = os.path.join(run_dir, "evaluation")
    if not os.path.isdir(eval_dir):
        return {}

    records = {}
    for fname in os.listdir(eval_dir):
        if not re.match(r"iter_\d+_eval_stats\.json$", fname):
            continue
        path = os.path.join(eval_dir, fname)
        with open(path) as f:
            data = json.load(f)

        global_step = data.get("global_step")
        stats = data.get("statistics", {})
        value = stats.get(metric)
        if global_step is None or value is None:
            continue

        step = global_step / NUM_AGENTS if multi_agent else global_step
        if step <= MAX_STEPS_PER_AGENT:
            records[step] = float(value)

    return dict(sorted(records.items()))


def aggregate_across_seeds(
    seed_runs: dict[int, str],
    multi_agent: bool,
    metric: str,
    min_seeds: int,
) -> tuple[list[float], list[float], list[float], list[float], list[int]]:
    """Aggregate seed curves at each available step.

    Returns steps, mean, mean - std, mean + std, and seed counts. Points with
    fewer than min_seeds available are omitted.
    """
    by_step = defaultdict(list)
    for run_dir in seed_runs.values():
        for step, value in load_eval_series(run_dir, multi_agent, metric).items():
            by_step[step].append(value)

    steps_out, means_out, std_lower, std_upper, counts_out = [], [], [], [], []
    for step in sorted(by_step):
        values = by_step[step]
        if len(values) < min_seeds:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        steps_out.append(step)
        means_out.append(mean)
        std_lower.append(mean - std)
        std_upper.append(mean + std)
        counts_out.append(len(values))

    return steps_out, means_out, std_lower, std_upper, counts_out


def smooth(values: list[float], window: int) -> list[float]:
    if window <= 1 or not values:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default=LOG_DIR)
    parser.add_argument("--smooth", type=int, default=1, metavar="W")
    parser.add_argument("--min-seeds", type=int, default=1)
    parser.add_argument("--metric", default="avg_score")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed allowlist. By default, discover all seeds with matching run directories.",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Include active/incomplete seeds. By default only completed seeds are plotted.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        raise SystemExit(f"Log directory not found: {args.log_dir}")

    output_path = args.output or os.path.join(
        args.log_dir,
        f"airraid_ppo_bidding_learning_curves_{args.metric}.png",
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    any_data = False
    requested_seeds = parse_seeds(args.seeds)
    for exp_prefix, label, multi_agent in EXPERIMENTS:
        seeds = requested_seeds if requested_seeds is not None else discover_seeds(args.log_dir, exp_prefix)
        seed_runs = find_seed_runs(args.log_dir, exp_prefix, seeds, args.include_partial)
        if not seed_runs:
            print(f"  [skip] no runs found for '{exp_prefix}'")
            continue

        seed_note = "completed" if not args.include_partial else "available"
        print(
            f"  {label}: found {len(seed_runs)}/{len(seeds)} {seed_note} seeds "
            f"({', '.join(str(seed) for seed in sorted(seed_runs))})"
        )
        steps, means, std_lo, std_hi, counts = aggregate_across_seeds(
            seed_runs,
            multi_agent,
            args.metric,
            args.min_seeds,
        )
        if not steps:
            print(f"  [skip] no eval points with min_seeds={args.min_seeds}")
            continue

        s_steps = smooth(steps, args.smooth)
        s_means = smooth(means, args.smooth)
        s_lo = smooth(std_lo, args.smooth)
        s_hi = smooth(std_hi, args.smooth)

        (line,) = ax.plot(s_steps, s_means, label=label, linewidth=2)
        ax.fill_between(s_steps, s_lo, s_hi, color=line.get_color(), alpha=0.15)

        print(
            f"    {len(steps)} points, steps {steps[0]:,.0f}-{steps[-1]:,.0f}, "
            f"seeds/point {min(counts)}-{max(counts)}, "
            f"final {args.metric} = {means[-1]:.1f} +/- {means[-1] - std_lo[-1]:.1f}"
        )
        any_data = True

    if not any_data:
        raise SystemExit("No data found. Check --log-dir, --metric, or --min-seeds.")

    ax.set_xlabel("Env. Steps", fontsize=18)
    ax.set_ylabel(args.metric.replace("_", " ").title(), fontsize=18)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(loc="upper left", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
