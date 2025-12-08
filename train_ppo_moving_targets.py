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

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bidding_ppo import PPOTrainer, Args, reorder_observation_for_agent
from single_agent_ppo import SingleAgentPPOTrainer, SingleAgentArgs
from bidding_gridworld import BiddingGridworld, MovingTargetBiddingGridworld


class PPOMovingTargetsExperiment:
    """Experiment wrapper for PPO training with periodic evaluation and checkpointing."""

    def __init__(
        self,
        base_log_dir: str = "logs",
        experiment_name: str = "",
        checkpoint_freq: int = 50,
        eval_freq: int = 25,
        num_eval_episodes: int = 3,
        single_agent_mode: bool = False,
    ):
        """
        Initialize the experiment.

        Args:
            base_log_dir: Base directory for logs
            experiment_name: Name for this experiment
            checkpoint_freq: Save checkpoint every N iterations
            eval_freq: Evaluate every N iterations
            num_eval_episodes: Number of episodes per evaluation
            single_agent_mode: If True, use single-agent PPO; if False, use multi-agent PPO
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not experiment_name:
            mode_prefix = "single_agent" if single_agent_mode else "multi_agent"
            experiment_name = f"ppo_{mode_prefix}"

        self.log_dir = Path(base_log_dir) / f"{experiment_name}_{timestamp}"
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.num_eval_episodes = num_eval_episodes
        self.single_agent_mode = single_agent_mode

        # Create directory structure
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.log_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.rollouts_dir = self.log_dir / "rollouts"
        self.rollouts_dir.mkdir(exist_ok=True)
        self.config_dir = self.log_dir / "config"
        self.config_dir.mkdir(exist_ok=True)

        print(f"📁 Experiment directory: {self.log_dir}")

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
        print(f"{'='*60}")

        # Create evaluation environment
        eval_env = MovingTargetBiddingGridworld(
            grid_size=trainer.args.grid_size,
            num_agents=trainer.args.num_agents,
            bid_upper_bound=trainer.args.bid_upper_bound,
            bid_penalty=trainer.args.bid_penalty,
            target_reward=trainer.args.target_reward,
            max_steps=trainer.args.max_steps,
            action_window=trainer.args.action_window,
            distance_reward_scale=trainer.args.distance_reward_scale,
            target_expiry_steps=trainer.args.target_expiry_steps,
            target_expiry_penalty=trainer.args.target_expiry_penalty,
            direction_change_prob=trainer.args.direction_change_prob,
            target_move_interval=trainer.args.target_move_interval,
        )

        eval_stats = {
            "episode_returns": [],
            "episode_lengths": [],
            "targets_reached_per_episode": [],
        }

        for episode_idx in range(self.num_eval_episodes):
            # Reset environment
            base_obs, _ = eval_env.reset()

            # Prepare observations for all agents (reorder targets)
            obs_list = []
            for agent_idx in range(trainer.args.num_agents):
                reordered_obs = reorder_observation_for_agent(
                    base_obs, agent_idx, trainer.args.num_agents
                )
                obs_list.append(reordered_obs)

            obs = torch.tensor(np.stack(obs_list), dtype=torch.float32).to(trainer.device)

            # Episode tracking
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_step_details = []
            episode_return = 0
            step_count = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                # Store state
                episode_states.append(base_obs.copy())

                # Get actions (deterministic for evaluation)
                with torch.no_grad():
                    action, _, _, _ = trainer.agent.get_action_and_value(obs)
                    action = action.cpu().numpy()

                # Convert to environment action format
                env_action = {}
                for agent_idx in range(trainer.args.num_agents):
                    env_action[f"agent_{agent_idx}"] = {
                        "direction": int(action[agent_idx, 0]),
                        "bid": int(action[agent_idx, 1]),
                    }

                episode_actions.append(env_action)

                # Step environment
                base_obs, rewards_dict, terminated, truncated, info = eval_env.step(env_action)

                episode_return += sum(rewards_dict.values())
                episode_rewards.append(rewards_dict)

                # Store step details
                episode_step_details.append({
                    "winning_agent": info.get("winning_agent", -1),
                    "bids": info.get("bids", {}),
                    "window_agent": info.get("window_agent", None),
                    "window_steps_remaining": info.get("window_steps_remaining", 0),
                    "bid_penalty_applied": info.get("bid_penalty_applied", False),
                })

                # Prepare next observation
                obs_list = []
                for agent_idx in range(trainer.args.num_agents):
                    reordered_obs = reorder_observation_for_agent(
                        base_obs, agent_idx, trainer.args.num_agents
                    )
                    obs_list.append(reordered_obs)

                obs = torch.tensor(np.stack(obs_list), dtype=torch.float32).to(trainer.device)
                step_count += 1

            # Count targets reached (from final observation)
            targets_reached = sum(1 for i in range(trainer.args.num_agents)
                                 if base_obs[2 + 2 * trainer.args.num_agents + i] == 1)

            eval_stats["episode_returns"].append(episode_return)
            eval_stats["episode_lengths"].append(step_count)
            eval_stats["targets_reached_per_episode"].append(targets_reached)

            print(f"  Episode {episode_idx + 1}: Return={episode_return:.2f}, "
                  f"Length={step_count}, Targets={targets_reached}/{trainer.args.num_agents}")

            # Create GIF
            episode_data = {
                "states": episode_states,
                "actions": episode_actions,
                "rewards": episode_rewards,
                "step_details": episode_step_details,
            }

            gif_path = self.rollouts_dir / f"iter_{iteration}_ep_{episode_idx}.gif"
            eval_env.create_competition_gif(episode_data, gif_path, fps=2)

            # Log to wandb
            if trainer.args.track:
                wandb.log({
                    f"eval/rollout_ep_{episode_idx}": wandb.Video(str(gif_path), fps=2, format="gif"),
                }, step=global_step)

        eval_env.close()

        # Compute statistics
        avg_return = np.mean(eval_stats["episode_returns"])
        avg_length = np.mean(eval_stats["episode_lengths"])
        avg_targets = np.mean(eval_stats["targets_reached_per_episode"])
        success_rate = sum(1 for t in eval_stats["targets_reached_per_episode"]
                          if t == trainer.args.num_agents) / self.num_eval_episodes

        print(f"\nEvaluation Summary:")
        print(f"  Average Return: {avg_return:.2f}")
        print(f"  Average Length: {avg_length:.1f}")
        print(f"  Average Targets: {avg_targets:.2f}/{trainer.args.num_agents}")
        print(f"  Success Rate: {success_rate*100:.1f}%\n")

        # Log to wandb
        if trainer.args.track:
            wandb.log({
                "eval/avg_return": avg_return,
                "eval/avg_length": avg_length,
                "eval/avg_targets_reached": avg_targets,
                "eval/success_rate": success_rate,
            }, step=global_step)

        return eval_stats

    def evaluate_single_agent_policy(self, trainer: SingleAgentPPOTrainer, iteration: int, global_step: int):
        """Evaluate the single-agent policy with rollouts and create visualizations."""
        print(f"\n{'='*60}")
        print(f"EVALUATION - Iteration {iteration}")
        print(f"{'='*60}")

        # Create evaluation environment
        if trainer.args.moving_targets:
            eval_env = MovingTargetBiddingGridworld(
                grid_size=trainer.args.grid_size,
                num_agents=trainer.args.num_targets,
                target_reward=trainer.args.target_reward,
                max_steps=trainer.args.max_steps,
                distance_reward_scale=trainer.args.distance_reward_scale,
                target_expiry_steps=trainer.args.target_expiry_steps,
                target_expiry_penalty=trainer.args.target_expiry_penalty,
                direction_change_prob=trainer.args.direction_change_prob,
                target_move_interval=trainer.args.target_move_interval,
                single_agent_mode=True
            )
        else:
            eval_env = BiddingGridworld(
                grid_size=trainer.args.grid_size,
                num_agents=trainer.args.num_targets,
                target_reward=trainer.args.target_reward,
                max_steps=trainer.args.max_steps,
                distance_reward_scale=trainer.args.distance_reward_scale,
                target_expiry_steps=trainer.args.target_expiry_steps,
                target_expiry_penalty=trainer.args.target_expiry_penalty,
                single_agent_mode=True
            )

        eval_stats = {
            "episode_returns": [],
            "episode_lengths": [],
            "targets_reached_per_episode": [],
        }

        for episode_idx in range(self.num_eval_episodes):
            # Reset environment
            obs, _ = eval_env.reset()
            obs = torch.tensor(obs, dtype=torch.float32).to(trainer.device)

            # Episode tracking
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_return = 0
            step_count = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                # Store state
                episode_states.append(obs.cpu().numpy().copy())

                # Get action (deterministic for evaluation)
                with torch.no_grad():
                    action, _, _, _ = trainer.agent.get_action_and_value(obs.unsqueeze(0))
                    action = action.squeeze(0).cpu().numpy()

                episode_actions.append(int(action))

                # Step environment
                obs, reward, terminated, truncated, info = eval_env.step(int(action))

                episode_return += reward
                episode_rewards.append(reward)

                obs = torch.tensor(obs, dtype=torch.float32).to(trainer.device)
                step_count += 1

            # Count targets reached (from final observation)
            targets_reached = sum(1 for i in range(trainer.args.num_targets)
                                 if obs[2 + 2 * trainer.args.num_targets + i].item() == 1)

            eval_stats["episode_returns"].append(episode_return)
            eval_stats["episode_lengths"].append(step_count)
            eval_stats["targets_reached_per_episode"].append(targets_reached)

            print(f"  Episode {episode_idx + 1}: Return={episode_return:.2f}, "
                  f"Length={step_count}, Targets={targets_reached}/{trainer.args.num_targets}")

            # Create GIF
            episode_data = {
                "states": episode_states,
                "actions": episode_actions,
                "rewards": episode_rewards,
            }

            gif_path = self.rollouts_dir / f"iter_{iteration}_ep_{episode_idx}.gif"
            eval_env.create_single_agent_gif(episode_data, gif_path, fps=2)

            # Log to wandb
            if trainer.args.track:
                wandb.log({
                    f"eval/rollout_ep_{episode_idx}": wandb.Video(str(gif_path), fps=2, format="gif"),
                }, step=global_step)

        eval_env.close()

        # Compute statistics
        avg_return = np.mean(eval_stats["episode_returns"])
        avg_length = np.mean(eval_stats["episode_lengths"])
        avg_targets = np.mean(eval_stats["targets_reached_per_episode"])
        success_rate = sum(1 for t in eval_stats["targets_reached_per_episode"]
                          if t == trainer.args.num_targets) / self.num_eval_episodes

        print(f"\nEvaluation Summary:")
        print(f"  Average Return: {avg_return:.2f}")
        print(f"  Average Length: {avg_length:.1f}")
        print(f"  Average Targets: {avg_targets:.2f}/{trainer.args.num_targets}")
        print(f"  Success Rate: {success_rate*100:.1f}%\n")

        # Log to wandb
        if trainer.args.track:
            wandb.log({
                "eval/avg_return": avg_return,
                "eval/avg_length": avg_length,
                "eval/avg_targets_reached": avg_targets,
                "eval/success_rate": success_rate,
            }, step=global_step)

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

        print(f"Checkpoint frequency: every {self.checkpoint_freq} iterations")
        print(f"Evaluation frequency: every {self.eval_freq} iterations\n")

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
    SINGLE_AGENT_MODE = True  # Set to True for single-agent navigation, False for multi-agent bidding
    MOVING_TARGETS = True  # Set to True for moving targets

    # Experiment settings
    EXPERIMENT_NAME = "ppo_moving_targets_single_agent"  # Leave empty for default name with timestamp
    CHECKPOINT_FREQ = 1000  # Save checkpoint every N iterations
    EVAL_FREQ = 1000  # Evaluate every N iterations
    NUM_EVAL_EPISODES = 3  # Number of episodes per evaluation

    # Environment parameters
    GRID_SIZE = 15
    NUM_AGENTS = 3  # For multi-agent: number of bidding agents; For single-agent: number of targets
    TARGET_REWARD = 10.0
    MAX_STEPS = 300
    DISTANCE_REWARD_SCALE = 0.1
    TARGET_EXPIRY_STEPS = 40
    TARGET_EXPIRY_PENALTY = 50.0

    # Multi-agent specific parameters (ignored in single-agent mode)
    BID_UPPER_BOUND = 3
    BID_PENALTY = 0.01
    ACTION_WINDOW = 6

    # Moving targets parameters (only used if MOVING_TARGETS = True)
    DIRECTION_CHANGE_PROB = 0.1
    TARGET_MOVE_INTERVAL = 2

    # Training parameters
    TOTAL_TIMESTEPS = int(5e7)
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
        single_agent_mode=SINGLE_AGENT_MODE,
    )

    experiment.run(args)


if __name__ == "__main__":
    main()
