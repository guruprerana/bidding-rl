#!/usr/bin/env python3
"""
Evaluate multi-agent PPO policies from bidding_mechanism_comparison.py
on increasing numbers of eval targets (8, 10, 12, 14).

Finds the latest run directories for each mechanism under TRAINING_LOG_DIR,
loads the final checkpoint, and evaluates with each value in EVAL_NUM_AGENTS_LIST.
Results are saved as JSON files under BASE_LOG_DIR/<timestamp>/.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path (one level up from experiment_scripts/)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch

from bidding_gridworld.bidding_gridworld_torch import (
    BiddingGridworld,
    BiddingGridworldConfig,
    evaluate_multi_agent_policy,
)
from bidding_gridworld.bidding_ppo import SharedAgent


# ============================================================================
# CONFIG
# ============================================================================

TRAINING_LOG_DIR = "logs/gridworld_bidding_mechanism_comparison"
BASE_LOG_DIR = "logs/eval_bidding_mechanism_scaling"

# Multi-agent PPO mechanisms to evaluate (mechanism name → experiment dir prefix)
MECHANISMS = [
    ("all_pay",                   "bidding_cmp_all_pay"),
    ("winner_pays",               "bidding_cmp_winner_pays"),
    ("winner_pays_others_reward", "bidding_cmp_winner_pays_others_reward"),
]

# Numbers of eval agents/targets to sweep over
EVAL_NUM_AGENTS_LIST = [8, 10, 12, 14]
CHECKPOINT_ITERATION = 200  # Which training iteration checkpoint to load

NUM_EVAL_EPISODES = 20
EVAL_MAX_STEPS = 2000
SEED = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# HELPERS
# ============================================================================

def find_latest_run_dir(base_log_dir: str, exp_name_prefix: str) -> Path:
    """Find the most recent run directory matching `exp_name_prefix_*`."""
    base = Path(base_log_dir)
    candidates = sorted(base.glob(f"{exp_name_prefix}_*"))
    if not candidates:
        raise FileNotFoundError(
            f"No run directories found matching '{exp_name_prefix}_*' in {base_log_dir}"
        )
    return candidates[-1]


def find_checkpoint_path(run_dir: Path, iteration: int) -> Path:
    """Return agent.pt path for a specific iter checkpoint in run_dir/checkpoints/."""
    ckpt_path = run_dir / "checkpoints" / f"iter_{iteration}" / "agent.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def load_config(run_dir: Path) -> dict:
    config_path = run_dir / "config" / "training_config.json"
    with open(config_path) as f:
        return json.load(f)


def build_agent(config: dict, device: str) -> SharedAgent:
    """Reconstruct a SharedAgent from a training config dict."""
    num_agents = config["num_agents"]
    moving_targets = config.get("moving_targets", False)
    visible_targets = config.get("visible_targets")
    include_reached = not moving_targets
    window_bidding = config.get("window_bidding", False)

    # Compute per-agent obs dim matching BiddingGridworld's formula
    if visible_targets is None:
        per_agent_obs_dim = 3 + (4 if include_reached else 3) * num_agents
    else:
        per_agent_obs_dim = (
            7 + 3 * visible_targets if include_reached else 6 + 2 * visible_targets
        )

    agent = SharedAgent(
        obs_dim=per_agent_obs_dim,
        num_actions_per_agent=3 if window_bidding else 2,
        window_bidding=window_bidding,
        actor_hidden_sizes=config.get("actor_hidden_sizes"),
        critic_hidden_sizes=config.get("critic_hidden_sizes"),
        use_target_attention_pooling=config.get("use_target_attention_pooling", False),
        target_embed_dim=config.get("target_embed_dim", 64),
        target_encoder_hidden_sizes=config.get("target_encoder_hidden_sizes"),
        attention_pooling_layout="centralized" if visible_targets is None else "visible",
        include_target_reached=include_reached,
    )
    agent.set_bid_head(config["bid_upper_bound"])
    if window_bidding:
        agent.set_window_head(config["action_window"])
    return agent.to(device)


def load_agent(run_dir: Path, device: str):
    """Load the iter_CHECKPOINT_ITERATION checkpoint into a reconstructed SharedAgent."""
    config = load_config(run_dir)
    agent = build_agent(config, device)
    ckpt_path = find_checkpoint_path(run_dir, CHECKPOINT_ITERATION)
    state_dict = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    print(f"  Loaded checkpoint: {ckpt_path}")
    return agent, config


def evaluate_for_num_agents(
    agent: SharedAgent,
    config: dict,
    eval_num_agents: int,
    num_eval_episodes: int,
    eval_max_steps: int,
    device: str,
    seed: int,
) -> dict:
    """Run evaluation with eval_num_agents agents/targets and return a stats dict."""
    env_config = BiddingGridworldConfig(
        grid_size=config["grid_size"],
        num_agents=eval_num_agents,
        bid_upper_bound=config["bid_upper_bound"],
        bid_penalty=config["bid_penalty"],
        target_reward=config["target_reward"],
        max_steps=eval_max_steps,
        action_window=config["action_window"],
        distance_reward_scale=config["distance_reward_scale"],
        target_expiry_steps=config.get("target_expiry_steps"),
        target_expiry_penalty=config.get("target_expiry_penalty", 0.0),
        moving_targets=config.get("moving_targets", False),
        direction_change_prob=config.get("direction_change_prob", 0.1),
        target_move_interval=config.get("target_move_interval", 5),
        window_bidding=config.get("window_bidding", False),
        window_penalty=config.get("window_penalty", 0.0),
        visible_targets=config.get("visible_targets"),
        single_agent_mode=False,
        bidding_mechanism=config["bidding_mechanism"],
    )
    eval_env = BiddingGridworld(env_config, num_envs=1, device=device, seed=seed)

    def policy_fn(obs):
        obs_tensor = obs if torch.is_tensor(obs) else torch.tensor(obs, dtype=torch.float32)
        obs_tensor = obs_tensor.to(device)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
        return action

    eval_stats = evaluate_multi_agent_policy(
        env=eval_env,
        policy_fn=policy_fn,
        num_episodes=num_eval_episodes,
        target_expiry_penalty=config.get("target_expiry_penalty", 0.0),
        verbose=True,
    )
    eval_env.close()

    # Aggregate statistics
    all_bid_counts = eval_stats.get("bid_counts_per_episode", [])
    bid_upper_bound = config["bid_upper_bound"]
    avg_bid_counts = {
        bid_val: float(np.mean([bc.get(bid_val, 0) for bc in all_bid_counts]))
        for bid_val in range(bid_upper_bound + 1)
    }

    all_control_steps = eval_stats.get("control_steps_per_agent_per_episode", [])
    avg_control_steps = (
        np.array(all_control_steps).mean(axis=0).tolist() if all_control_steps else []
    )

    success_rate = sum(
        1 for t in eval_stats["targets_reached_per_episode"] if t == eval_num_agents
    ) / num_eval_episodes

    return {
        "eval_num_agents": eval_num_agents,
        "train_num_agents": config["num_agents"],
        "bidding_mechanism": config["bidding_mechanism"],
        "num_episodes": num_eval_episodes,
        "timestamp": datetime.now().isoformat(),
        "statistics": {
            "avg_return": float(np.mean(eval_stats["episode_returns"])),
            "std_return": float(np.std(eval_stats["episode_returns"])),
            "avg_return_no_bid": (
                float(np.mean(eval_stats["episode_returns_no_bid"]))
                if eval_stats.get("episode_returns_no_bid")
                else float("nan")
            ),
            "avg_length": float(np.mean(eval_stats["episode_lengths"])),
            "std_length": float(np.std(eval_stats["episode_lengths"])),
            "avg_targets_reached": float(np.mean(eval_stats["targets_reached_per_episode"])),
            "std_targets_reached": float(np.std(eval_stats["targets_reached_per_episode"])),
            "avg_expired_targets": float(np.mean(eval_stats["expired_targets_per_episode"])),
            "avg_min_targets_reached": float(
                np.mean(eval_stats["min_targets_reached_per_episode"])
            ),
            "success_rate": float(success_rate),
            "avg_bid_counts": avg_bid_counts,
            "avg_control_timesteps_per_agent": avg_control_steps,
        },
        "per_episode_data": {
            "returns": [float(r) for r in eval_stats["episode_returns"]],
            "returns_no_bid": [
                float(r) for r in eval_stats.get("episode_returns_no_bid", [])
            ],
            "lengths": [int(l) for l in eval_stats["episode_lengths"]],
            "targets_reached": [int(t) for t in eval_stats["targets_reached_per_episode"]],
            "expired_targets": [int(e) for e in eval_stats["expired_targets_per_episode"]],
            "min_targets_reached": [
                int(m) for m in eval_stats["min_targets_reached_per_episode"]
            ],
            "bid_counts": [dict(sorted(bc.items())) for bc in all_bid_counts],
            "control_steps_per_agent": all_control_steps,
        },
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(BASE_LOG_DIR) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDevice: {DEVICE}")
    print(f"Results will be saved to: {out_dir}\n")

    all_results = {}

    for mechanism, exp_name_prefix in MECHANISMS:
        print(f"\n{'='*72}")
        print(f"  Mechanism: {mechanism}")
        print(f"{'='*72}")

        run_dir = find_latest_run_dir(TRAINING_LOG_DIR, exp_name_prefix)
        print(f"  Run dir: {run_dir}")

        agent, config = load_agent(run_dir, DEVICE)
        mech_results = {}

        for eval_num_agents in EVAL_NUM_AGENTS_LIST:
            print(f"\n  -- eval_num_agents = {eval_num_agents} --")
            stats = evaluate_for_num_agents(
                agent=agent,
                config=config,
                eval_num_agents=eval_num_agents,
                num_eval_episodes=NUM_EVAL_EPISODES,
                eval_max_steps=EVAL_MAX_STEPS,
                device=DEVICE,
                seed=SEED,
            )
            mech_results[str(eval_num_agents)] = stats

            # Save per-mechanism per-num-agents JSON
            out_file = out_dir / f"{exp_name_prefix}_eval_{eval_num_agents}agents.json"
            with open(out_file, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"  Saved: {out_file.name}")

        all_results[mechanism] = mech_results

    # Save combined summary
    summary_path = out_dir / "all_results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
