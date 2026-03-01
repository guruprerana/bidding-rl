#!/usr/bin/env python3
"""
Evaluate iter-200 checkpoints produced by bidding_mechanism_comparison.py.

For each experiment sub-directory found under BASE_LOG_DIR that contains
checkpoints/iter_200/agent.pt, the script:
  1. Reads training_config.json to reconstruct the environment and agent.
  2. Loads the iter_200 checkpoint.
  3. Runs evaluate_multi_agent_policy (or evaluate_single_agent_policy for
     single-agent experiments) for NUM_EVAL_EPISODES episodes.
  4. Writes eval_iter200_<timestamp>.json into the experiment directory.

Usage:
  python experiment_scripts/evaluate_bidding_mechanism_comparison.py
  python experiment_scripts/evaluate_bidding_mechanism_comparison.py --log-dir logs/my_run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch

from bidding_gridworld.bidding_gridworld_torch import (
    BiddingGridworld,
    BiddingGridworldConfig,
    evaluate_multi_agent_policy,
    evaluate_single_agent_policy,
)
from bidding_gridworld.bidding_ppo import SharedAgent
from bidding_gridworld.single_agent_ppo import SingleAgent

# ============================================================================
# Settings
# ============================================================================

BASE_LOG_DIR = "logs/gridworld_bidding_mechanism_comparison"
CHECKPOINT_ITER = 200
NUM_EVAL_EPISODES = 20
EVAL_MAX_STEPS = 2000
SEED = 1


# ============================================================================
# Helpers: detect mode, build env + agent
# ============================================================================

def _is_single_agent(cfg: dict) -> bool:
    return "num_targets" in cfg and "num_agents" not in cfg


def _build_multi_agent_env_config(cfg: dict) -> BiddingGridworldConfig:
    return BiddingGridworldConfig(
        grid_size=cfg["grid_size"],
        num_agents=cfg["num_agents"],
        bid_upper_bound=cfg["bid_upper_bound"],
        bid_penalty=cfg["bid_penalty"],
        target_reward=cfg["target_reward"],
        max_steps=EVAL_MAX_STEPS,
        action_window=cfg.get("action_window", 1),
        distance_reward_scale=cfg.get("distance_reward_scale", 0.0),
        target_expiry_steps=cfg.get("target_expiry_steps", None),
        target_expiry_penalty=cfg.get("target_expiry_penalty", 0.0),
        moving_targets=cfg.get("moving_targets", False),
        direction_change_prob=cfg.get("direction_change_prob", 0.0),
        target_move_interval=cfg.get("target_move_interval", 1),
        window_bidding=cfg.get("window_bidding", False),
        window_penalty=cfg.get("window_penalty", 0.0),
        visible_targets=cfg.get("visible_targets", None),
        single_agent_mode=False,
        bidding_mechanism=cfg.get("bidding_mechanism", "all_pay"),
    )


def _build_single_agent_env_config(cfg: dict) -> BiddingGridworldConfig:
    return BiddingGridworldConfig(
        grid_size=cfg["grid_size"],
        num_agents=cfg["num_targets"],
        bid_upper_bound=0,
        bid_penalty=0.0,
        target_reward=cfg["target_reward"],
        max_steps=EVAL_MAX_STEPS,
        action_window=1,
        distance_reward_scale=cfg.get("distance_reward_scale", 0.0),
        target_expiry_steps=cfg.get("target_expiry_steps", None),
        target_expiry_penalty=cfg.get("target_expiry_penalty", 0.0),
        moving_targets=cfg.get("moving_targets", False),
        direction_change_prob=cfg.get("direction_change_prob", 0.0),
        target_move_interval=cfg.get("target_move_interval", 1),
        window_bidding=False,
        window_penalty=0.0,
        visible_targets=None,
        single_agent_mode=True,
        reward_decay_factor=cfg.get("reward_decay_factor", 0.0),
    )


def _build_multi_agent_agent(cfg: dict, obs_dim: int, device: torch.device) -> SharedAgent:
    window_bidding  = cfg.get("window_bidding", False)
    bid_upper_bound = cfg["bid_upper_bound"]
    action_window   = cfg.get("action_window", 1)
    visible_targets = cfg.get("visible_targets", None)
    moving_targets  = cfg.get("moving_targets", True)

    agent = SharedAgent(
        obs_dim=obs_dim,
        num_actions_per_agent=3 if window_bidding else 2,
        window_bidding=window_bidding,
        actor_hidden_sizes=cfg.get("actor_hidden_sizes", [128, 128, 128]),
        critic_hidden_sizes=cfg.get("critic_hidden_sizes", [256, 256, 256]),
        use_target_attention_pooling=cfg.get("use_target_attention_pooling", False),
        target_embed_dim=cfg.get("target_embed_dim", 64),
        target_encoder_hidden_sizes=cfg.get("target_encoder_hidden_sizes", [64, 64]),
        attention_pooling_layout="centralized" if visible_targets is None else "visible",
        include_target_reached=not moving_targets,
    ).to(device)
    agent.set_bid_head(bid_upper_bound)
    if window_bidding:
        agent.set_window_head(action_window)
    return agent


def _build_single_agent_agent(cfg: dict, obs_dim: int, device: torch.device) -> SingleAgent:
    moving_targets = cfg.get("moving_targets", True)
    return SingleAgent(
        obs_dim=obs_dim,
        num_targets=cfg["num_targets"],
        actor_hidden_sizes=cfg.get("actor_hidden_sizes", [128, 128, 128]),
        critic_hidden_sizes=cfg.get("critic_hidden_sizes", [256, 256, 256]),
        use_target_attention_pooling=cfg.get("use_target_attention_pooling", False),
        target_embed_dim=cfg.get("target_embed_dim", 64),
        target_encoder_hidden_sizes=cfg.get("target_encoder_hidden_sizes", [64, 64]),
        include_target_reached=not moving_targets,
    ).to(device)


# ============================================================================
# Per-experiment evaluation
# ============================================================================

def evaluate_experiment(exp_dir: Path, device: torch.device) -> None:
    """Load iter-200 checkpoint and evaluate; write JSON into exp_dir."""
    checkpoint_path = exp_dir / "checkpoints" / f"iter_{CHECKPOINT_ITER}" / "agent.pt"
    config_path     = exp_dir / "config" / "training_config.json"

    if not checkpoint_path.exists():
        print(f"  [SKIP] No iter_{CHECKPOINT_ITER} checkpoint: {exp_dir.name}")
        return
    if not config_path.exists():
        print(f"  [SKIP] No training_config.json: {exp_dir.name}")
        return

    print(f"\n{'='*70}")
    print(f"  {exp_dir.name}")
    print(f"{'='*70}")

    with open(config_path) as f:
        cfg = json.load(f)

    single_agent = _is_single_agent(cfg)
    num_targets  = cfg["num_targets"] if single_agent else cfg["num_agents"]

    if single_agent:
        env_config = _build_single_agent_env_config(cfg)
        env = BiddingGridworld(env_config, num_envs=1, device=device, seed=SEED)
        agent = _build_single_agent_agent(cfg, env.obs_dim, device)
    else:
        env_config = _build_multi_agent_env_config(cfg)
        env = BiddingGridworld(env_config, num_envs=1, device=device, seed=SEED)
        agent = _build_multi_agent_agent(cfg, env.per_agent_obs_dim, device)

    state_dict = torch.load(str(checkpoint_path), map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()

    target_expiry_penalty = cfg.get("target_expiry_penalty", 0.0)

    if single_agent:
        def policy_fn(obs):
            obs_t = obs if torch.is_tensor(obs) else torch.tensor(obs, dtype=torch.float32)
            obs_t = obs_t.to(device)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_t.unsqueeze(0))
            return action.squeeze(0)

        eval_stats = evaluate_single_agent_policy(
            env=env,
            policy_fn=policy_fn,
            num_episodes=NUM_EVAL_EPISODES,
            target_expiry_penalty=target_expiry_penalty,
            verbose=True,
        )
    else:
        def policy_fn(obs):
            obs_t = obs if torch.is_tensor(obs) else torch.tensor(obs, dtype=torch.float32)
            obs_t = obs_t.to(device)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_t)
            return action

        eval_stats = evaluate_multi_agent_policy(
            env=env,
            policy_fn=policy_fn,
            num_episodes=NUM_EVAL_EPISODES,
            target_expiry_penalty=target_expiry_penalty,
            verbose=True,
        )

    env.close()

    # Aggregate stats common to both modes
    avg_return      = float(np.mean(eval_stats["episode_returns"]))
    avg_length      = float(np.mean(eval_stats["episode_lengths"]))
    avg_targets     = float(np.mean(eval_stats["targets_reached_per_episode"]))
    avg_expired     = float(np.mean(eval_stats["expired_targets_per_episode"]))
    avg_min_reached = float(np.mean(eval_stats["min_targets_reached_per_episode"]))
    avg_avg_perf    = float(np.mean(eval_stats["avg_performance_per_episode"]))
    avg_min_perf    = float(np.mean(eval_stats["min_performance_per_episode"]))
    success_rate    = sum(1 for t in eval_stats["targets_reached_per_episode"]
                          if t == num_targets) / NUM_EVAL_EPISODES

    reached_arr = np.array(eval_stats["targets_reached_count_per_episode"], dtype=float)
    expired_arr = np.array(eval_stats["expired_count_per_target_per_episode"], dtype=float)
    perf_arr    = reached_arr - expired_arr

    statistics: dict = {
        "avg_return":              avg_return,
        "std_return":              float(np.std(eval_stats["episode_returns"])),
        "avg_length":              avg_length,
        "avg_targets_reached":     avg_targets,
        "avg_expired":             avg_expired,
        "avg_min_targets_reached": avg_min_reached,
        "success_rate":            success_rate,
        "avg_avg_performance":     avg_avg_perf,
        "avg_min_performance":     avg_min_perf,
        "avg_reached_per_target":  reached_arr.mean(axis=0).tolist(),
        "avg_expired_per_target":  expired_arr.mean(axis=0).tolist(),
        "avg_performance_per_target": perf_arr.mean(axis=0).tolist(),
    }

    per_episode: dict = {
        "returns":               [float(r) for r in eval_stats["episode_returns"]],
        "lengths":               [int(l)   for l in eval_stats["episode_lengths"]],
        "targets_reached":       [int(t)   for t in eval_stats["targets_reached_per_episode"]],
        "expired_targets":       [int(e)   for e in eval_stats["expired_targets_per_episode"]],
        "min_targets_reached":   [int(m)   for m in eval_stats["min_targets_reached_per_episode"]],
        "avg_performance":       [float(p) for p in eval_stats["avg_performance_per_episode"]],
        "min_performance":       [float(p) for p in eval_stats["min_performance_per_episode"]],
        "expired_count_per_target": eval_stats["expired_count_per_target_per_episode"],
        "targets_reached_count": eval_stats["targets_reached_count_per_episode"],
    }

    # Multi-agent-only extras
    if not single_agent:
        avg_return_no_bid = float(np.mean(eval_stats["episode_returns_no_bid"])) if eval_stats.get("episode_returns_no_bid") else float("nan")
        statistics["avg_return_no_bid"] = avg_return_no_bid

        all_bid_counts = eval_stats.get("bid_counts_per_episode", [])
        bid_upper_bound = cfg["bid_upper_bound"]
        statistics["avg_bid_counts"] = {
            bid_val: float(np.mean([bc.get(bid_val, 0) for bc in all_bid_counts]))
            for bid_val in range(bid_upper_bound + 1)
        }
        all_control_steps = eval_stats.get("control_steps_per_agent_per_episode", [])
        statistics["avg_control_steps_per_agent"] = (
            np.array(all_control_steps).mean(axis=0).tolist() if all_control_steps else []
        )
        per_episode["returns_no_bid"]        = [float(r) for r in eval_stats.get("episode_returns_no_bid", [])]
        per_episode["bid_counts"]            = [dict(sorted(bc.items())) for bc in all_bid_counts]
        per_episode["control_steps_per_agent"] = all_control_steps

    output = {
        "exp_name":          cfg.get("exp_name", exp_dir.name),
        "exp_dir":           str(exp_dir),
        "mode":              "single_agent" if single_agent else "multi_agent",
        "bidding_mechanism": cfg.get("bidding_mechanism", "single_agent" if single_agent else "unknown"),
        "checkpoint_iter":   CHECKPOINT_ITER,
        "num_eval_episodes": NUM_EVAL_EPISODES,
        "eval_max_steps":    EVAL_MAX_STEPS,
        "timestamp":         datetime.now().isoformat(),
        "statistics":        statistics,
        "per_episode":       per_episode,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = exp_dir / f"eval_iter{CHECKPOINT_ITER}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Mode:            {'single_agent' if single_agent else 'multi_agent'}")
    print(f"  Avg return:      {avg_return:.2f}")
    print(f"  Avg targets:     {avg_targets:.2f}/{num_targets}")
    print(f"  Avg expired:     {avg_expired:.2f}")
    print(f"  Avg min perf:    {avg_min_perf:.2f}")
    print(f"  Success rate:    {success_rate*100:.1f}%")
    print(f"  Saved → {out_path}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir", default=BASE_LOG_DIR,
        help=f"Base log directory to search (default: {BASE_LOG_DIR})"
    )
    args = parser.parse_args()

    base = Path(args.log_dir)
    if not base.exists():
        print(f"Log directory not found: {base}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Searching for iter_{CHECKPOINT_ITER} checkpoints in: {base}\n")

    exp_dirs = sorted(
        d for d in base.iterdir()
        if d.is_dir() and (d / "checkpoints" / f"iter_{CHECKPOINT_ITER}" / "agent.pt").exists()
    )

    if not exp_dirs:
        print(f"No experiments with iter_{CHECKPOINT_ITER} checkpoint found under {base}")
        sys.exit(1)

    print(f"Found {len(exp_dirs)} experiment(s):")
    for d in exp_dirs:
        print(f"  {d.name}")

    for exp_dir in exp_dirs:
        evaluate_experiment(exp_dir, device)

    print(f"\nDone. Evaluated {len(exp_dirs)} experiment(s).")


if __name__ == "__main__":
    main()
