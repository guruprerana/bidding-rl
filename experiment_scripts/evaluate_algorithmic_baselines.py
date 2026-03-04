#!/usr/bin/env python3
"""
Evaluate algorithmic baseline policies on the BiddingGridworld.

Environment config matches bidding_mechanism_comparison.py exactly, using
single_agent_mode=True so the baselines act as centralized coordinators
without bidding overhead.

Policies evaluated:
  1. RoundRobinPolicy    — cycles through targets in order
  2. NearestTargetPolicy — always pursues the closest target
  3. LeastTimeLeftPolicy — prioritises the target closest to expiry
"""

import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch

from bidding_gridworld.bidding_gridworld_torch import BiddingGridworld, BiddingGridworldConfig
from bidding_gridworld.algorithmic_baselines import (
    RoundRobinPolicy,
    NearestTargetPolicy,
    LeastTimeLeftPolicy,
    evaluate_algorithmic_policy,
)

# ============================================================================
# Environment config — mirrors bidding_mechanism_comparison.py exactly
# ============================================================================

GRID_SIZE = 30
NUM_AGENTS = 8
TARGET_REWARD = 50.0
MAX_STEPS = 2000
DISTANCE_REWARD_SCALE = 0.6
TARGET_EXPIRY_STEPS = 200
TARGET_EXPIRY_PENALTY = 50.0
MOVING_TARGETS = True
DIRECTION_CHANGE_PROB = 0.1
TARGET_MOVE_INTERVAL = 5
ACTION_WINDOW = 5

# Eval settings
NUM_EVAL_EPISODES = 20
SEED = 1

OUTPUT_DIR = "logs/algorithmic_baselines"


# ============================================================================
# Main
# ============================================================================

def make_env(seed: int) -> BiddingGridworld:
    cfg = BiddingGridworldConfig(
        grid_size=GRID_SIZE,
        num_agents=NUM_AGENTS,
        bid_upper_bound=6,
        bid_penalty=0.1,
        target_reward=TARGET_REWARD,
        max_steps=MAX_STEPS,
        action_window=ACTION_WINDOW,
        distance_reward_scale=DISTANCE_REWARD_SCALE,
        target_expiry_steps=TARGET_EXPIRY_STEPS,
        target_expiry_penalty=TARGET_EXPIRY_PENALTY,
        moving_targets=MOVING_TARGETS,
        direction_change_prob=DIRECTION_CHANGE_PROB,
        target_move_interval=TARGET_MOVE_INTERVAL,
        window_bidding=False,
        window_penalty=0.05,
        visible_targets=None,
        single_agent_mode=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return BiddingGridworld(cfg, num_envs=1, device=device, seed=seed)


def summarise(stats: dict, num_agents: int) -> dict:
    returns = stats["episode_returns"]
    lengths = stats["episode_lengths"]
    targets = stats["targets_reached_per_episode"]
    expired = stats["expired_targets_per_episode"]
    min_reached = stats["min_targets_reached_per_episode"]
    success_rate = sum(1 for t in targets if t == num_agents) / len(targets)

    # Per-target arrays: shape (num_episodes, num_agents)
    reached_arr = np.array(stats["targets_reached_count_per_episode"], dtype=float)   # (E, A)
    expired_arr = np.array(stats["expired_count_per_target_per_episode"], dtype=float) # (E, A)
    perf_arr    = reached_arr - expired_arr                                             # (E, A)

    return {
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "avg_length": float(np.mean(lengths)),
        "avg_targets_reached": float(np.mean(targets)),
        "avg_expired": float(np.mean(expired)),
        "std_expired": float(np.std(expired)),
        "avg_min_reached": float(np.mean(min_reached)),
        "success_rate": float(success_rate),
        # Per-target means across episodes
        "avg_reached_per_target":     reached_arr.mean(axis=0).tolist(),
        "avg_expired_per_target":     expired_arr.mean(axis=0).tolist(),
        "max_expired_per_target":     expired_arr.max(axis=0).tolist(),
        "avg_performance_per_target": perf_arr.mean(axis=0).tolist(),
        # Scalar summaries
        "avg_avg_performance": float(np.mean(stats["avg_performance_per_episode"])),
        "avg_min_performance": float(np.mean(stats["min_performance_per_episode"])),
    }


def print_table(results: dict, num_agents: int) -> None:
    header = (
        f"{'Policy':<25}  {'AvgReturn':>10}  {'AvgTargets':>10}  "
        f"{'AvgExpired':>10}  {'AvgMinPerf':>10}  {'SuccessRate':>11}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, s in results.items():
        print(
            f"{name:<25}  {s['avg_return']:>10.2f}  "
            f"{s['avg_targets_reached']:>10.2f}/{num_agents}  "
            f"{s['avg_expired']:>10.2f}  "
            f"{s['avg_min_performance']:>10.2f}  "
            f"{s['success_rate']*100:>10.1f}%"
        )
    print(sep)
    print()


def main() -> None:
    env = make_env(SEED)

    policies = [
        # ("round_robin",     RoundRobinPolicy(NUM_AGENTS, GRID_SIZE, TARGET_EXPIRY_STEPS)),
        ("nearest_target",  NearestTargetPolicy(NUM_AGENTS, GRID_SIZE, TARGET_EXPIRY_STEPS)),
        ("least_time_left", LeastTimeLeftPolicy(NUM_AGENTS, GRID_SIZE, TARGET_EXPIRY_STEPS)),
    ]

    all_results = {}
    for name, policy in policies:
        stats = evaluate_algorithmic_policy(
            env,
            policy,
            num_episodes=NUM_EVAL_EPISODES,
            target_expiry_penalty=TARGET_EXPIRY_PENALTY,
            verbose=True,
        )
        all_results[name] = {"summary": summarise(stats, NUM_AGENTS), "raw": stats}

    summaries = {name: v["summary"] for name, v in all_results.items()}
    print_table(summaries, NUM_AGENTS)

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"eval_{timestamp}.json")

    output = {
        "config": {
            "grid_size": GRID_SIZE,
            "num_agents": NUM_AGENTS,
            "target_reward": TARGET_REWARD,
            "max_steps": MAX_STEPS,
            "distance_reward_scale": DISTANCE_REWARD_SCALE,
            "target_expiry_steps": TARGET_EXPIRY_STEPS,
            "target_expiry_penalty": TARGET_EXPIRY_PENALTY,
            "moving_targets": MOVING_TARGETS,
            "direction_change_prob": DIRECTION_CHANGE_PROB,
            "target_move_interval": TARGET_MOVE_INTERVAL,
            "action_window": ACTION_WINDOW,
            "num_eval_episodes": NUM_EVAL_EPISODES,
            "seed": SEED,
        },
        "results": {
            name: {"summary": v["summary"], "per_episode": v["raw"]}
            for name, v in all_results.items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
