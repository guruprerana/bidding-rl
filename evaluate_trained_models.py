#!/usr/bin/env python3
"""
Evaluate trained PPO models using refactored evaluation functions.

This script loads trained models from experiment directories and evaluates them,
creating GIFs and comprehensive statistics using the new evaluation utilities.

USAGE:
    python evaluate_trained_models.py

CONFIGURATION:
    Edit the main() function to specify:
    - SINGLE_AGENT_EXP: Path to single-agent experiment directory
    - MULTI_AGENT_EXP: Path to multi-agent experiment directory
    - NUM_EVAL_EPISODES: Number of episodes to evaluate (default: 20)
    - NUM_GIF_EPISODES: Number of episodes to save as GIFs (default: 5)
    - DEVICE: "cuda" or "cpu" (auto-detected by default)

OUTPUT:
    For each evaluated model, creates:
    logs/[experiment_name]/evaluation/eval_[timestamp]/
    ├── episode_0.gif ... episode_N.gif  # Visual rollouts
    └── eval_stats.json                   # Comprehensive statistics

FEATURES:
    - Automatically loads latest checkpoint from experiment
    - Uses deterministic (greedy) policy for evaluation
    - Correctly tracks targets with moving targets that respawn
    - Generates per-episode and aggregate statistics
    - Supports both single-agent and multi-agent models
    - Handles window bidding and target expiry mechanisms
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bidding_gridworld import (
    MovingTargetBiddingGridworld,
    evaluate_multi_agent_policy,
    evaluate_single_agent_policy
)
from bidding_ppo import SharedAgent as MultiAgentNetwork, reorder_observation_for_agent
from single_agent_ppo import SingleAgent as SingleAgentNetwork


def load_config(experiment_dir: Path) -> dict:
    """Load training configuration from experiment directory."""
    config_path = experiment_dir / "config" / "training_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def find_latest_checkpoint(experiment_dir: Path) -> Path:
    """Find the latest checkpoint in the experiment directory."""
    checkpoints_dir = experiment_dir / "checkpoints"

    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    # Find all checkpoint directories
    checkpoint_dirs = sorted([d for d in checkpoints_dir.iterdir() if d.is_dir()])

    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")

    # Return the latest checkpoint
    latest_checkpoint = checkpoint_dirs[-1]
    model_path = latest_checkpoint / "agent.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return model_path


def evaluate_single_agent_model(
    experiment_dir: Path,
    num_eval_episodes: int = 10,
    num_gif_episodes: int = 3,
    device: str = "cpu"
):
    """Evaluate a trained single-agent model."""
    print(f"\n{'='*80}")
    print(f"EVALUATING SINGLE-AGENT MODEL: {experiment_dir.name}")
    print(f"{'='*80}\n")

    # Load config
    config = load_config(experiment_dir)
    print(f"Loaded config from {experiment_dir / 'config' / 'training_config.json'}")

    # Find latest checkpoint
    model_path = find_latest_checkpoint(experiment_dir)
    print(f"Using checkpoint: {model_path}")

    # Create environment
    print(f"\nCreating environment...")
    if config.get('moving_targets', True):
        env = MovingTargetBiddingGridworld(
            grid_size=config['grid_size'],
            num_agents=config['num_targets'],
            target_reward=config['target_reward'],
            max_steps=600,  # Longer for evaluation
            distance_reward_scale=config.get('distance_reward_scale', 0.0),
            target_expiry_steps=config.get('target_expiry_steps'),
            target_expiry_penalty=config.get('target_expiry_penalty', 5.0),
            direction_change_prob=config.get('direction_change_prob', 0.1),
            target_move_interval=config.get('target_move_interval', 1),
            single_agent_mode=True
        )
    else:
        from bidding_gridworld import BiddingGridworld
        env = BiddingGridworld(
            grid_size=config['grid_size'],
            num_agents=config['num_targets'],
            target_reward=config['target_reward'],
            max_steps=600,
            distance_reward_scale=config.get('distance_reward_scale', 0.0),
            target_expiry_steps=config.get('target_expiry_steps'),
            target_expiry_penalty=config.get('target_expiry_penalty', 5.0),
            single_agent_mode=True
        )

    print(f"Environment: {env.__class__.__name__}")
    print(f"  Grid size: {config['grid_size']}x{config['grid_size']}")
    print(f"  Num targets: {config['num_targets']}")
    print(f"  Max steps: 600")

    # Load model
    print(f"\nLoading model...")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    network = SingleAgentNetwork(obs_dim).to(device)
    network.load_state_dict(torch.load(model_path, map_location=device))
    network.eval()

    print(f"Model loaded: obs_dim={obs_dim}, action_dim={action_dim}")

    # Create policy function
    def policy_fn(obs):
        """Policy function for evaluation - deterministic (greedy)."""
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            action_logits = network.actor(obs_tensor.unsqueeze(0))
            action = torch.argmax(action_logits, dim=-1)
            return action.squeeze(0).cpu().numpy()

    # Run evaluation
    print(f"\nRunning evaluation ({num_eval_episodes} episodes)...")
    eval_stats = evaluate_single_agent_policy(
        env=env,
        policy_fn=policy_fn,
        num_episodes=num_eval_episodes,
        target_expiry_penalty=config.get('target_expiry_penalty', 0.0),
        verbose=True
    )

    # Create output directory
    output_dir = experiment_dir / "evaluation" / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create GIFs
    print(f"\nCreating GIFs...")
    for episode_idx in range(min(num_gif_episodes, num_eval_episodes)):
        episode_data = eval_stats["episode_data_list"][episode_idx]
        gif_path = output_dir / f"episode_{episode_idx}.gif"
        env.create_single_agent_gif(episode_data, gif_path, fps=2)

    # Save statistics
    stats_summary = {
        "experiment": experiment_dir.name,
        "model_path": str(model_path),
        "timestamp": datetime.now().isoformat(),
        "num_episodes": num_eval_episodes,
        "config": config,
        "statistics": {
            "avg_return": float(np.mean(eval_stats["episode_returns"])),
            "std_return": float(np.std(eval_stats["episode_returns"])),
            "avg_length": float(np.mean(eval_stats["episode_lengths"])),
            "avg_targets_reached": float(np.mean(eval_stats["targets_reached_per_episode"])),
            "avg_expired_targets": float(np.mean(eval_stats["expired_targets_per_episode"])),
            "avg_min_targets_reached": float(np.mean(eval_stats["min_targets_reached_per_episode"])),
            "success_rate": float(sum(1 for t in eval_stats["targets_reached_per_episode"]
                                     if t == config['num_targets']) / num_eval_episodes),
        },
        "per_episode": {
            "returns": [float(r) for r in eval_stats["episode_returns"]],
            "lengths": [int(l) for l in eval_stats["episode_lengths"]],
            "targets_reached": [int(t) for t in eval_stats["targets_reached_per_episode"]],
            "targets_reached_counts": eval_stats["targets_reached_count_per_episode"],
        }
    }

    stats_file = output_dir / "eval_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats_summary, f, indent=2)

    print(f"\n✅ Evaluation complete!")
    print(f"   Results saved to: {output_dir}")
    print(f"   GIFs: {num_gif_episodes} episodes")
    print(f"   Stats: {stats_file}")

    env.close()
    return eval_stats


def evaluate_multi_agent_model(
    experiment_dir: Path,
    num_eval_episodes: int = 10,
    num_gif_episodes: int = 3,
    device: str = "cpu"
):
    """Evaluate a trained multi-agent model."""
    print(f"\n{'='*80}")
    print(f"EVALUATING MULTI-AGENT MODEL: {experiment_dir.name}")
    print(f"{'='*80}\n")

    # Load config
    config = load_config(experiment_dir)
    print(f"Loaded config from {experiment_dir / 'config' / 'training_config.json'}")

    # Find latest checkpoint
    model_path = find_latest_checkpoint(experiment_dir)
    print(f"Using checkpoint: {model_path}")

    # Create environment
    print(f"\nCreating environment...")
    env = MovingTargetBiddingGridworld(
        grid_size=config['grid_size'],
        num_agents=config['num_agents'],
        bid_upper_bound=config['bid_upper_bound'],
        bid_penalty=config['bid_penalty'],
        target_reward=config['target_reward'],
        max_steps=600,  # Longer for evaluation
        action_window=config.get('action_window', 1),
        distance_reward_scale=config.get('distance_reward_scale', 0.0),
        target_expiry_steps=config.get('target_expiry_steps'),
        target_expiry_penalty=config.get('target_expiry_penalty', 5.0),
        direction_change_prob=config.get('direction_change_prob', 0.1),
        target_move_interval=config.get('target_move_interval', 1),
        window_bidding=config.get('window_bidding', False),
        window_penalty=config.get('window_penalty', 0.0),
    )

    print(f"Environment: {env.__class__.__name__}")
    print(f"  Grid size: {config['grid_size']}x{config['grid_size']}")
    print(f"  Num agents: {config['num_agents']}")
    print(f"  Max steps: 600")

    # Load model
    print(f"\nLoading model...")
    obs_dim = env.observation_space.shape[0]

    # Determine number of action components per agent
    num_actions_per_agent = 3 if config.get('window_bidding', False) else 2
    window_bidding = config.get('window_bidding', False)

    network = MultiAgentNetwork(obs_dim, num_actions_per_agent, window_bidding).to(device)
    network.set_bid_head(config['bid_upper_bound'])
    if window_bidding:
        network.set_window_head(config.get('action_window', 1))

    network.load_state_dict(torch.load(model_path, map_location=device))
    network.eval()

    print(f"Model loaded: obs_dim={obs_dim}, num_actions={num_actions_per_agent}, window_bidding={window_bidding}")

    # Create policy function
    def policy_fn(base_obs):
        """Policy function for evaluation with observation reordering."""
        # Prepare observations for all agents (reorder targets)
        obs_list = []
        for agent_idx in range(config['num_agents']):
            reordered_obs = reorder_observation_for_agent(
                base_obs, agent_idx, config['num_agents']
            )
            obs_list.append(reordered_obs)

        obs = torch.tensor(np.stack(obs_list), dtype=torch.float32).to(device)

        # Get actions (deterministic for evaluation - use argmax)
        with torch.no_grad():
            shared_features = network.actor_shared(obs)

            # Get direction actions
            direction_logits = network.direction_head(shared_features)
            directions = torch.argmax(direction_logits, dim=-1)

            # Get bid actions
            bid_logits = network.bid_head(shared_features)
            bids = torch.argmax(bid_logits, dim=-1)

            # Stack actions
            if window_bidding:
                window_logits = network.window_head(shared_features)
                windows = torch.argmax(window_logits, dim=-1)
                actions = torch.stack([directions, bids, windows], dim=-1)
            else:
                actions = torch.stack([directions, bids], dim=-1)

            return actions.cpu().numpy()

    # Run evaluation
    print(f"\nRunning evaluation ({num_eval_episodes} episodes)...")
    eval_stats = evaluate_multi_agent_policy(
        env=env,
        policy_fn=policy_fn,
        num_episodes=num_eval_episodes,
        target_expiry_penalty=config.get('target_expiry_penalty', 0.0),
        verbose=True
    )

    # Create output directory
    output_dir = experiment_dir / "evaluation" / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create GIFs
    print(f"\nCreating GIFs...")
    for episode_idx in range(min(num_gif_episodes, num_eval_episodes)):
        episode_data = eval_stats["episode_data_list"][episode_idx]
        gif_path = output_dir / f"episode_{episode_idx}.gif"
        env.create_competition_gif(episode_data, gif_path, fps=2)

    # Save statistics
    stats_summary = {
        "experiment": experiment_dir.name,
        "model_path": str(model_path),
        "timestamp": datetime.now().isoformat(),
        "num_episodes": num_eval_episodes,
        "config": config,
        "statistics": {
            "avg_return": float(np.mean(eval_stats["episode_returns"])),
            "std_return": float(np.std(eval_stats["episode_returns"])),
            "avg_length": float(np.mean(eval_stats["episode_lengths"])),
            "avg_targets_reached": float(np.mean(eval_stats["targets_reached_per_episode"])),
            "avg_expired_targets": float(np.mean(eval_stats["expired_targets_per_episode"])),
            "avg_min_targets_reached": float(np.mean(eval_stats["min_targets_reached_per_episode"])),
            "success_rate": float(sum(1 for t in eval_stats["targets_reached_per_episode"]
                                     if t == config['num_agents']) / num_eval_episodes),
        },
        "per_episode": {
            "returns": [float(r) for r in eval_stats["episode_returns"]],
            "lengths": [int(l) for l in eval_stats["episode_lengths"]],
            "targets_reached": [int(t) for t in eval_stats["targets_reached_per_episode"]],
            "targets_reached_counts": eval_stats["targets_reached_count_per_episode"],
        }
    }

    stats_file = output_dir / "eval_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats_summary, f, indent=2)

    print(f"\n✅ Evaluation complete!")
    print(f"   Results saved to: {output_dir}")
    print(f"   GIFs: {num_gif_episodes} episodes")
    print(f"   Stats: {stats_file}")

    env.close()
    return eval_stats


def main():
    """
    Main evaluation function.

    CONFIGURATION - Edit these values to customize evaluation:
    """

    # ========================================================================
    # CONFIGURATION - Modify parameters here
    # ========================================================================

    # Experiment directories (change these to evaluate different models)
    SINGLE_AGENT_EXP = "logs/ppo_moving_targets_single_agent_exp5_20251216_214906"
    MULTI_AGENT_EXP = "logs/ppo_moving_targets_exp8_20251216_214612"

    # Evaluation settings
    NUM_EVAL_EPISODES = 20  # Number of episodes to evaluate per model
    NUM_GIF_EPISODES = 5    # Number of episodes to save as GIFs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Force "cpu" if needed

    # ========================================================================
    # End of configuration
    # ========================================================================

    print(f"Device: {DEVICE}")
    print(f"Evaluation episodes: {NUM_EVAL_EPISODES}")
    print(f"GIF episodes: {NUM_GIF_EPISODES}")

    # Evaluate single-agent model
    single_agent_dir = Path(SINGLE_AGENT_EXP)
    if single_agent_dir.exists():
        try:
            evaluate_single_agent_model(
                single_agent_dir,
                num_eval_episodes=NUM_EVAL_EPISODES,
                num_gif_episodes=NUM_GIF_EPISODES,
                device=DEVICE
            )
        except Exception as e:
            print(f"\n❌ Error evaluating single-agent model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠️  Single-agent experiment directory not found: {single_agent_dir}")

    # Evaluate multi-agent model
    multi_agent_dir = Path(MULTI_AGENT_EXP)
    if multi_agent_dir.exists():
        try:
            evaluate_multi_agent_model(
                multi_agent_dir,
                num_eval_episodes=NUM_EVAL_EPISODES,
                num_gif_episodes=NUM_GIF_EPISODES,
                device=DEVICE
            )
        except Exception as e:
            print(f"\n❌ Error evaluating multi-agent model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠️  Multi-agent experiment directory not found: {multi_agent_dir}")

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
