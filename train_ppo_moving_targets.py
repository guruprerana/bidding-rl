#!/usr/bin/env python3
"""
PPO Training Script for Bidding Gridworld

This script trains PPO policies for both single-agent and multi-agent bidding.

Features:
- Single-agent mode: One agent navigates to collect all targets
- Multi-agent mode: Multiple agents bid for control to reach their targets
- Periodic checkpointing
- Regular rollout evaluations with GIF generation
- Comprehensive wandb logging
- Moving target environment support
- All configuration in one place (no CLI arguments needed)

Usage:
    python train_ppo_moving_targets.py

Configure all parameters in the CONFIGURATION section of the main() function.
Set SINGLE_AGENT_MODE = True for single-agent navigation, False for multi-agent bidding.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import wandb
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bidding_ppo import PPOTrainer, Args, reorder_observation_for_agent
from single_agent_ppo import SingleAgentPPOTrainer, SingleAgentArgs
from bidding_gridworld import (
    BiddingGridworld,
    MovingTargetBiddingGridworld,
    evaluate_multi_agent_policy,
    evaluate_single_agent_policy
)


class PPOMovingTargetsExperiment:
    """Experiment wrapper for PPO training with periodic evaluation and checkpointing."""

    def __init__(
        self,
        base_log_dir: str = "logs",
        experiment_name: str = "",
        checkpoint_freq: int = 50,
        eval_freq: int = 25,
        num_eval_episodes: int = 3,
        num_gif_episodes: int = 3,
        single_agent_mode: bool = False,
        eval_max_steps: int = 600,
    ):
        """
        Initialize the experiment.

        Args:
            base_log_dir: Base directory for logs
            experiment_name: Name for this experiment
            checkpoint_freq: Save checkpoint every N iterations
            eval_freq: Evaluate every N iterations
            num_eval_episodes: Number of episodes per evaluation
            num_gif_episodes: Number of episodes to save as GIFs
            single_agent_mode: If True, use single-agent PPO; if False, use multi-agent PPO
            eval_max_steps: Maximum steps per episode during evaluation
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not experiment_name:
            mode_prefix = "single_agent" if single_agent_mode else "multi_agent"
            experiment_name = f"ppo_{mode_prefix}"

        self.log_dir = Path(base_log_dir) / f"{experiment_name}_{timestamp}"
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.num_eval_episodes = num_eval_episodes
        self.num_gif_episodes = num_gif_episodes
        self.single_agent_mode = single_agent_mode
        self.eval_max_steps = eval_max_steps

        # Create directory structure
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.log_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.rollouts_dir = self.log_dir / "rollouts"
        self.rollouts_dir.mkdir(exist_ok=True)
        self.config_dir = self.log_dir / "config"
        self.config_dir.mkdir(exist_ok=True)

        print(f"📁 Experiment directory: {self.log_dir}")

    def log_codebase_to_wandb(self, run):
        """Log src folder and training script to wandb as an artifact."""
        if not run:
            return

        print("📦 Logging codebase to wandb...")

        # Get project root (parent of logs directory)
        project_root = self.log_dir.parent.parent

        # Create wandb artifact
        artifact = wandb.Artifact(
            name=f"codebase-{run.id}",
            type="code",
            description="Codebase snapshot (src folder + training script)"
        )

        # Add src directory
        src_dir = project_root / "src"
        if src_dir.exists():
            artifact.add_dir(str(src_dir), name="src")
            num_files = len(list(src_dir.rglob("*.py")))
            print(f"  Added src/ ({num_files} Python files)")

        # Add training script
        train_script = project_root / "train_ppo_moving_targets.py"
        if train_script.exists():
            artifact.add_file(str(train_script), name="train_ppo_moving_targets.py")
            print(f"  Added train_ppo_moving_targets.py")

        # Log artifact
        run.log_artifact(artifact)
        print(f"✅ Codebase logged to wandb artifact")

    def save_config(self, args):
        """Save training configuration."""
        config = vars(args)
        config_file = self.config_dir / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"💾 Config saved to {config_file}")

    def save_checkpoint(self, trainer, iteration: int, global_step: int):
        """Save model checkpoint."""
        checkpoint_dir = self.checkpoints_dir / f"iter_{iteration}"
        checkpoint_dir.mkdir(exist_ok=True)

        # Save model
        model_path = checkpoint_dir / "agent.pt"
        torch.save(trainer.agent.state_dict(), str(model_path))

        # Save checkpoint info
        checkpoint_info = {
            "iteration": iteration,
            "global_step": global_step,
            "timestamp": datetime.now().isoformat(),
        }
        info_path = checkpoint_dir / "checkpoint_info.json"
        with open(info_path, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)

        print(f"💾 Checkpoint saved: {checkpoint_dir}")

        # Save to wandb
        if trainer.args.track:
            wandb.save(str(model_path))

    def evaluate_policy(self, trainer: PPOTrainer, iteration: int, global_step: int):
        """Evaluate the current policy with rollouts and create visualizations."""
        print(f"\n{'='*60}")
        print(f"EVALUATION - Iteration {iteration}")
        print(f"Running {self.num_eval_episodes} episodes (saving GIFs for first {self.num_gif_episodes})")
        print(f"{'='*60}\n")

        # Create evaluation environment with longer max_steps
        eval_env = MovingTargetBiddingGridworld(
            grid_size=trainer.args.grid_size,
            num_agents=trainer.args.num_agents,
            bid_upper_bound=trainer.args.bid_upper_bound,
            bid_penalty=trainer.args.bid_penalty,
            target_reward=trainer.args.target_reward,
            max_steps=self.eval_max_steps,
            action_window=trainer.args.action_window,
            distance_reward_scale=trainer.args.distance_reward_scale,
            target_expiry_steps=trainer.args.target_expiry_steps,
            target_expiry_penalty=trainer.args.target_expiry_penalty,
            direction_change_prob=trainer.args.direction_change_prob,
            target_move_interval=trainer.args.target_move_interval,
            window_bidding=trainer.args.window_bidding,
            window_penalty=trainer.args.window_penalty,
            visible_targets=trainer.args.visible_targets,
        )

        # Check if we're using per-agent observations (decentralized mode)
        use_per_agent_obs = trainer.args.visible_targets is not None and trainer.args.visible_targets < trainer.args.num_agents

        # Create policy wrapper function
        def policy_fn(base_obs):
            """Convert base observation to agent-specific observations and get actions."""
            if use_per_agent_obs:
                # Decentralized mode: base_obs is already a Dict with per-agent observations
                # Stack them in order: agent_0, agent_1, ..., agent_n
                obs_list = [base_obs[f"agent_{i}"] for i in range(trainer.args.num_agents)]
                obs = torch.tensor(np.stack(obs_list), dtype=torch.float32).to(trainer.device)
            else:
                # Centralized mode: reorder targets for each agent
                obs_list = []
                for agent_idx in range(trainer.args.num_agents):
                    reordered_obs = reorder_observation_for_agent(
                        base_obs, agent_idx, trainer.args.num_agents
                    )
                    obs_list.append(reordered_obs)
                obs = torch.tensor(np.stack(obs_list), dtype=torch.float32).to(trainer.device)

            # Get actions (deterministic for evaluation)
            with torch.no_grad():
                action, _, _, _ = trainer.agent.get_action_and_value(obs)
                return action.cpu().numpy()

        # Run evaluation using refactored function
        eval_stats = evaluate_multi_agent_policy(
            env=eval_env,
            policy_fn=policy_fn,
            num_episodes=self.num_eval_episodes,
            target_expiry_penalty=trainer.args.target_expiry_penalty,
            verbose=True
        )

        # Create GIFs for first num_gif_episodes
        episode_data_list = eval_stats.get("episode_data_list", [])
        max_gif_episodes = min(self.num_gif_episodes, self.num_eval_episodes, len(episode_data_list))
        for episode_idx in range(max_gif_episodes):
            episode_data = episode_data_list[episode_idx]
            gif_path = self.rollouts_dir / f"iter_{iteration}_ep_{episode_idx}.gif"
            eval_env.create_competition_gif(episode_data, gif_path, fps=2)

            # Log to wandb if the GIF exists
            if trainer.args.track and gif_path.exists() and gif_path.stat().st_size > 0:
                wandb.log({
                    f"eval/rollout_ep_{episode_idx}": wandb.Video(str(gif_path), fps=2, format="gif"),
                }, step=global_step)
            elif trainer.args.track:
                print(f"⚠️  Skipping wandb.Video for missing GIF: {gif_path}")

        eval_env.close()

        # Compute aggregate statistics
        avg_return = np.mean(eval_stats["episode_returns"])
        avg_length = np.mean(eval_stats["episode_lengths"])
        avg_targets = np.mean(eval_stats["targets_reached_per_episode"])
        avg_expired = np.mean(eval_stats["expired_targets_per_episode"])
        avg_min_reached = np.mean(eval_stats["min_targets_reached_per_episode"])
        success_rate = sum(1 for t in eval_stats["targets_reached_per_episode"]
                          if t == trainer.args.num_agents) / self.num_eval_episodes

        # Log to wandb
        if trainer.args.track:
            wandb.log({
                "eval/avg_return": avg_return,
                "eval/avg_length": avg_length,
                "eval/avg_targets_reached": avg_targets,
                "eval/avg_expired_targets": avg_expired,
                "eval/avg_min_targets_reached": avg_min_reached,
                "eval/success_rate": success_rate,
            }, step=global_step)

        # Save eval stats to local JSON file
        eval_summary = {
            "iteration": iteration,
            "global_step": global_step,
            "num_episodes": self.num_eval_episodes,
            "num_agents": trainer.args.num_agents,
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "avg_return": float(avg_return),
                "avg_length": float(avg_length),
                "avg_targets_reached": float(avg_targets),
                "avg_expired_targets": float(avg_expired),
                "avg_min_targets_reached": float(avg_min_reached),
                "success_rate": float(success_rate),
                "std_return": float(np.std(eval_stats["episode_returns"])),
                "std_length": float(np.std(eval_stats["episode_lengths"])),
                "std_targets_reached": float(np.std(eval_stats["targets_reached_per_episode"])),
            },
            "per_episode_data": {
                "returns": [float(r) for r in eval_stats["episode_returns"]],
                "lengths": [int(l) for l in eval_stats["episode_lengths"]],
                "targets_reached": [int(t) for t in eval_stats["targets_reached_per_episode"]],
                "expired_targets": [int(e) for e in eval_stats["expired_targets_per_episode"]],
                "min_targets_reached": [int(m) for m in eval_stats["min_targets_reached_per_episode"]],
            }
        }

        stats_file = self.rollouts_dir / f"iter_{iteration}_eval_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(eval_summary, f, indent=2)
        print(f"📊 Eval stats saved to {stats_file}")

        return eval_stats

    def evaluate_single_agent_policy(self, trainer: SingleAgentPPOTrainer, iteration: int, global_step: int):
        """Evaluate the single-agent policy with rollouts and create visualizations."""
        print(f"\n{'='*60}")
        print(f"EVALUATION - Iteration {iteration}")
        print(f"Running {self.num_eval_episodes} episodes (saving GIFs for first {self.num_gif_episodes})")
        print(f"{'='*60}\n")

        # Create evaluation environment with longer max_steps
        if trainer.args.moving_targets:
            eval_env = MovingTargetBiddingGridworld(
                grid_size=trainer.args.grid_size,
                num_agents=trainer.args.num_targets,
                target_reward=trainer.args.target_reward,
                max_steps=self.eval_max_steps,
                distance_reward_scale=trainer.args.distance_reward_scale,
                target_expiry_steps=trainer.args.target_expiry_steps,
                target_expiry_penalty=trainer.args.target_expiry_penalty,
                reward_decay_factor=trainer.args.reward_decay_factor,
                direction_change_prob=trainer.args.direction_change_prob,
                target_move_interval=trainer.args.target_move_interval,
                single_agent_mode=True
            )
        else:
            eval_env = BiddingGridworld(
                grid_size=trainer.args.grid_size,
                num_agents=trainer.args.num_targets,
                target_reward=trainer.args.target_reward,
                max_steps=self.eval_max_steps,
                distance_reward_scale=trainer.args.distance_reward_scale,
                target_expiry_steps=trainer.args.target_expiry_steps,
                target_expiry_penalty=trainer.args.target_expiry_penalty,
                reward_decay_factor=trainer.args.reward_decay_factor,
                single_agent_mode=True
            )

        # Create policy wrapper function
        def policy_fn(obs):
            """Convert observation to tensor and get action."""
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(trainer.device)
            # Get action (deterministic for evaluation)
            with torch.no_grad():
                action, _, _, _ = trainer.agent.get_action_and_value(obs_tensor.unsqueeze(0))
                return action.squeeze(0).cpu().numpy()

        # Run evaluation using refactored function
        eval_stats = evaluate_single_agent_policy(
            env=eval_env,
            policy_fn=policy_fn,
            num_episodes=self.num_eval_episodes,
            target_expiry_penalty=trainer.args.target_expiry_penalty,
            verbose=True
        )

        # Create GIFs for first num_gif_episodes
        episode_data_list = eval_stats.get("episode_data_list", [])
        max_gif_episodes = min(self.num_gif_episodes, self.num_eval_episodes, len(episode_data_list))
        for episode_idx in range(max_gif_episodes):
            episode_data = episode_data_list[episode_idx]
            gif_path = self.rollouts_dir / f"iter_{iteration}_ep_{episode_idx}.gif"
            eval_env.create_single_agent_gif(episode_data, gif_path, fps=2)

            # Log to wandb if the GIF exists
            if trainer.args.track and gif_path.exists() and gif_path.stat().st_size > 0:
                wandb.log({
                    f"eval/rollout_ep_{episode_idx}": wandb.Video(str(gif_path), fps=2, format="gif"),
                }, step=global_step)
            elif trainer.args.track:
                print(f"⚠️  Skipping wandb.Video for missing GIF: {gif_path}")

        eval_env.close()

        # Compute aggregate statistics
        avg_return = np.mean(eval_stats["episode_returns"])
        avg_length = np.mean(eval_stats["episode_lengths"])
        avg_targets = np.mean(eval_stats["targets_reached_per_episode"])
        avg_expired = np.mean(eval_stats["expired_targets_per_episode"])
        avg_min_reached = np.mean(eval_stats["min_targets_reached_per_episode"])
        success_rate = sum(1 for t in eval_stats["targets_reached_per_episode"]
                          if t == trainer.args.num_targets) / self.num_eval_episodes

        # Log to wandb
        if trainer.args.track:
            wandb.log({
                "eval/avg_return": avg_return,
                "eval/avg_length": avg_length,
                "eval/avg_targets_reached": avg_targets,
                "eval/avg_expired_targets": avg_expired,
                "eval/avg_min_targets_reached": avg_min_reached,
                "eval/success_rate": success_rate,
            }, step=global_step)

        # Save eval stats to local JSON file
        eval_summary = {
            "iteration": iteration,
            "global_step": global_step,
            "num_episodes": self.num_eval_episodes,
            "num_targets": trainer.args.num_targets,
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "avg_return": float(avg_return),
                "avg_length": float(avg_length),
                "avg_targets_reached": float(avg_targets),
                "avg_expired_targets": float(avg_expired),
                "avg_min_targets_reached": float(avg_min_reached),
                "success_rate": float(success_rate),
                "std_return": float(np.std(eval_stats["episode_returns"])),
                "std_length": float(np.std(eval_stats["episode_lengths"])),
                "std_targets_reached": float(np.std(eval_stats["targets_reached_per_episode"])),
            },
            "per_episode_data": {
                "returns": [float(r) for r in eval_stats["episode_returns"]],
                "lengths": [int(l) for l in eval_stats["episode_lengths"]],
                "targets_reached": [int(t) for t in eval_stats["targets_reached_per_episode"]],
                "expired_targets": [int(e) for e in eval_stats["expired_targets_per_episode"]],
                "min_targets_reached": [int(m) for m in eval_stats["min_targets_reached_per_episode"]],
            }
        }

        stats_file = self.rollouts_dir / f"iter_{iteration}_eval_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(eval_summary, f, indent=2)
        print(f"📊 Eval stats saved to {stats_file}")

        return eval_stats

    def run(self, args):
        """Run the training experiment."""
        mode_str = "SINGLE-AGENT" if self.single_agent_mode else "MULTI-AGENT"
        print(f"\n{'='*80}")
        print(f"PPO TRAINING - {mode_str} MODE")
        print(f"{'='*80}\n")

        # Save config
        self.save_config(args)

        # Define callbacks
        if self.single_agent_mode:
            # Single-agent callbacks
            def on_iteration_end(trainer, iteration, global_step):
                # Checkpoint saving
                if iteration % self.checkpoint_freq == 0:
                    self.save_checkpoint(trainer, iteration, global_step)

                # Evaluation
                if iteration % self.eval_freq == 0:
                    self.evaluate_single_agent_policy(trainer, iteration, global_step)

            def on_training_end(trainer, global_step):
                # Final evaluation
                print("\n" + "="*80)
                print("FINAL EVALUATION")
                print("="*80)
                self.evaluate_single_agent_policy(trainer, trainer.args.num_iterations, global_step)
        else:
            # Multi-agent callbacks
            def on_iteration_end(trainer, iteration, global_step):
                # Checkpoint saving
                if iteration % self.checkpoint_freq == 0:
                    self.save_checkpoint(trainer, iteration, global_step)

                # Evaluation
                if iteration % self.eval_freq == 0:
                    self.evaluate_policy(trainer, iteration, global_step)

            def on_training_end(trainer, global_step):
                # Final evaluation
                print("\n" + "="*80)
                print("FINAL EVALUATION")
                print("="*80)
                self.evaluate_policy(trainer, trainer.args.num_iterations, global_step)

        callbacks = {
            "on_iteration_end": on_iteration_end,
            "on_training_end": on_training_end,
        }

        # Create and run trainer
        if self.single_agent_mode:
            trainer = SingleAgentPPOTrainer(args, callbacks=callbacks)
        else:
            trainer = PPOTrainer(args, callbacks=callbacks)

        trainer.setup()

        # Log codebase to wandb
        self.log_codebase_to_wandb(wandb.run)

        print(f"Checkpoint frequency: every {self.checkpoint_freq} iterations")
        print(f"Evaluation frequency: every {self.eval_freq} iterations")
        print(f"Evaluation episodes: {self.num_eval_episodes} (saving GIFs for first {self.num_gif_episodes})\n")

        trainer.train()
        trainer.save_model()
        trainer.cleanup()

        print(f"\n✅ Training complete! Results saved to {self.log_dir}")


def main():
    """Main training function."""

    # ========================================================================
    # CONFIGURATION - Modify parameters here
    # ========================================================================

    # Mode selection
    SINGLE_AGENT_MODE = False  # Set to True for single-agent navigation, False for multi-agent bidding
    MOVING_TARGETS = True  # Set to True for moving targets

    # Experiment settings
    EXPERIMENT_NAME = "ppo_moving_targets_exp9"  # Leave empty for default name with timestamp
    CHECKPOINT_FREQ = 5000  # Save checkpoint every N iterations
    EVAL_FREQ = 5000  # Evaluate every N iterations
    NUM_EVAL_EPISODES = 20  # Number of episodes per evaluation
    NUM_GIF_EPISODES = 0  # Number of episodes to save as GIFs

    # Environment parameters
    GRID_SIZE = 50
    NUM_AGENTS = 10  # For multi-agent: number of bidding agents; For single-agent: number of targets
    TARGET_REWARD = 20.0
    MAX_STEPS = 1000  # Maximum steps per episode during training
    EVAL_MAX_STEPS = 2000  # Maximum steps per episode during evaluation (typically longer than training)
    DISTANCE_REWARD_SCALE = 0.01
    TARGET_EXPIRY_STEPS = 40
    TARGET_EXPIRY_PENALTY = 100.0
    REWARD_DECAY_FACTOR = 0.0  # Single-agent only: decay rewards for over-visited targets (0.0 = no decay, 0.5 = moderate)

    # Multi-agent specific parameters (ignored in single-agent mode)
    BID_UPPER_BOUND = 10
    BID_PENALTY = 0.05
    ACTION_WINDOW = 10
    WINDOW_BIDDING = True  # Set to True to let agents choose their window length
    WINDOW_PENALTY = 0.05  # Penalty per window step (only applies when WINDOW_BIDDING = True)
    VISIBLE_TARGETS = 3  # Set to None for centralized (all targets visible), or N for decentralized (each agent sees own target + N nearest others)

    # Moving targets parameters (only used if MOVING_TARGETS = True)
    DIRECTION_CHANGE_PROB = 0.1
    TARGET_MOVE_INTERVAL = 2

    # Training parameters
    TOTAL_TIMESTEPS = int(1e8)
    LEARNING_RATE = 2.5e-4
    NUM_ENVS = 4
    NUM_STEPS = 128
    SEED = 1

    # Wandb tracking
    WANDB_PROJECT = "bidding-rl"
    WANDB_ENTITY = None
    TRACK = True  # Set to False to disable wandb tracking

    # ========================================================================
    # End of configuration
    # ========================================================================

    # Create appropriate Args based on mode
    if SINGLE_AGENT_MODE:
        args = SingleAgentArgs(
            exp_name=EXPERIMENT_NAME or "single_agent_ppo",
            seed=SEED,
            track=TRACK,
            wandb_project_name=WANDB_PROJECT,
            wandb_entity=WANDB_ENTITY,

            # Environment config
            grid_size=GRID_SIZE,
            num_targets=NUM_AGENTS,  # In single-agent mode, this is number of targets
            target_reward=TARGET_REWARD,
            max_steps=MAX_STEPS,
            distance_reward_scale=DISTANCE_REWARD_SCALE,
            target_expiry_steps=TARGET_EXPIRY_STEPS,
            target_expiry_penalty=TARGET_EXPIRY_PENALTY,
            reward_decay_factor=REWARD_DECAY_FACTOR,
            moving_targets=MOVING_TARGETS,
            direction_change_prob=DIRECTION_CHANGE_PROB,
            target_move_interval=TARGET_MOVE_INTERVAL,

            # Training config
            total_timesteps=TOTAL_TIMESTEPS,
            learning_rate=LEARNING_RATE,
            num_envs=NUM_ENVS,
            num_steps=NUM_STEPS,
        )
    else:
        args = Args(
            exp_name=EXPERIMENT_NAME or "multi_agent_ppo",
            seed=SEED,
            track=TRACK,
            wandb_project_name=WANDB_PROJECT,
            wandb_entity=WANDB_ENTITY,

            # Environment config
            grid_size=GRID_SIZE,
            num_agents=NUM_AGENTS,
            bid_upper_bound=BID_UPPER_BOUND,
            bid_penalty=BID_PENALTY,
            target_reward=TARGET_REWARD,
            max_steps=MAX_STEPS,
            action_window=ACTION_WINDOW,
            distance_reward_scale=DISTANCE_REWARD_SCALE,
            target_expiry_steps=TARGET_EXPIRY_STEPS,
            target_expiry_penalty=TARGET_EXPIRY_PENALTY,
            moving_targets=MOVING_TARGETS,
            direction_change_prob=DIRECTION_CHANGE_PROB,
            target_move_interval=TARGET_MOVE_INTERVAL,
            window_bidding=WINDOW_BIDDING,
            window_penalty=WINDOW_PENALTY,
            visible_targets=VISIBLE_TARGETS,

            # Training config
            total_timesteps=TOTAL_TIMESTEPS,
            learning_rate=LEARNING_RATE,
            num_envs=NUM_ENVS,
            num_steps=NUM_STEPS,
        )

    # Create and run experiment
    experiment = PPOMovingTargetsExperiment(
        experiment_name=EXPERIMENT_NAME,
        checkpoint_freq=CHECKPOINT_FREQ,
        eval_freq=EVAL_FREQ,
        num_eval_episodes=NUM_EVAL_EPISODES,
        num_gif_episodes=NUM_GIF_EPISODES,
        single_agent_mode=SINGLE_AGENT_MODE,
        eval_max_steps=EVAL_MAX_STEPS,
    )

    experiment.run(args)


if __name__ == "__main__":
    main()
