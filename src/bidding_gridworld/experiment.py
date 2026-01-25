"""Experiment orchestration for PPO training with evaluations and checkpoints."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb

from bidding_gridworld.bidding_gridworld_torch import (
    BiddingGridworld,
    BiddingGridworldConfig,
    evaluate_multi_agent_policy,
    evaluate_single_agent_policy,
)
from bidding_gridworld.bidding_ppo import PPOTrainer
from bidding_gridworld.single_agent_ppo import SingleAgentPPOTrainer


class PPOMovingTargetsExperiment:
    """Experiment wrapper for PPO training with periodic evaluation and checkpointing."""

    def __init__(
        self,
        base_log_dir: str = "logs",
        experiment_name: str = "",
        checkpoint_freq: int = 50,
        eval_freq: int = 25,
        video_freq: int = 0,
        num_eval_episodes: int = 3,
        num_video_episodes: int = 3,
        log_videos_to_wandb: bool = False,
        single_agent_mode: bool = False,
        eval_max_steps: int = 600,
        eval_num_agents: int | None = None,
        eval_num_targets: int | None = None,
    ):
        """
        Initialize the experiment.

        Args:
            base_log_dir: Base directory for logs
            experiment_name: Name for this experiment
            checkpoint_freq: Save checkpoint every N iterations
            eval_freq: Evaluate every N iterations
            video_freq: Save video rollouts every N iterations (0 = use eval_freq)
            num_eval_episodes: Number of episodes per evaluation
            num_video_episodes: Number of episodes to save as MP4s
            log_videos_to_wandb: If True, upload MP4s to wandb
            single_agent_mode: If True, use single-agent PPO; if False, use multi-agent PPO
            eval_max_steps: Maximum steps per episode during evaluation
            eval_num_agents: Optional override for number of agents/targets during eval (multi-agent only)
            eval_num_targets: Optional override for number of targets during eval (single-agent only)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not experiment_name:
            mode_prefix = "single_agent" if single_agent_mode else "multi_agent"
            experiment_name = f"ppo_{mode_prefix}"

        self.log_dir = Path(base_log_dir) / f"{experiment_name}_{timestamp}"
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.video_freq = eval_freq if video_freq in {0, None} else video_freq
        self.num_eval_episodes = num_eval_episodes
        self.num_video_episodes = num_video_episodes
        self.log_videos_to_wandb = log_videos_to_wandb
        self.single_agent_mode = single_agent_mode
        self.eval_max_steps = eval_max_steps
        self.eval_num_agents = eval_num_agents
        self.eval_num_targets = eval_num_targets

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
        disabled = os.environ.get("WANDB_MODE", "").lower() == "disabled" or \
            os.environ.get("WANDB_DISABLED", "").lower() in {"true", "1"}
        if not run or disabled:
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
        print("✅ Codebase logged to wandb artifact")

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

    def evaluate_policy(
        self,
        trainer: PPOTrainer,
        iteration: int,
        global_step: int,
        create_videos: bool = True,
    ):
        """Evaluate the current policy with rollouts and create visualizations."""
        eval_num_agents = self.eval_num_agents or trainer.args.num_agents
        if eval_num_agents != trainer.args.num_agents and not trainer.args.use_target_attention_pooling:
            raise ValueError(
                "Eval num_agents differs from training, but use_target_attention_pooling=False. "
                "Enable USE_TARGET_ATTENTION_POOLING to support variable target counts during eval."
            )
        print(f"\n{'='*60}")
        print(f"EVALUATION - Iteration {iteration}")
        if create_videos:
            print(f"Running {self.num_eval_episodes} episodes (saving videos for first {self.num_video_episodes})")
        else:
            print(f"Running {self.num_eval_episodes} episodes (no videos this iteration)")
        print(f"{'='*60}\n")

        # Create evaluation environment with longer max_steps
        env_config = BiddingGridworldConfig(
            grid_size=trainer.args.grid_size,
            num_agents=eval_num_agents,
            bid_upper_bound=trainer.args.bid_upper_bound,
            bid_penalty=trainer.args.bid_penalty,
            target_reward=trainer.args.target_reward,
            max_steps=self.eval_max_steps,
            action_window=trainer.args.action_window,
            distance_reward_scale=trainer.args.distance_reward_scale,
            target_expiry_steps=trainer.args.target_expiry_steps,
            target_expiry_penalty=trainer.args.target_expiry_penalty,
            moving_targets=True,
            direction_change_prob=trainer.args.direction_change_prob,
            target_move_interval=trainer.args.target_move_interval,
            window_bidding=trainer.args.window_bidding,
            window_penalty=trainer.args.window_penalty,
            visible_targets=trainer.args.visible_targets,
            single_agent_mode=False,
        )
        eval_env = BiddingGridworld(
            env_config,
            num_envs=1,
            device=trainer.device,
            seed=trainer.args.seed,
        )

        # Create policy wrapper function
        def policy_fn(obs):
            """Get actions for a single env using per-agent observations."""
            obs_tensor = obs if torch.is_tensor(obs) else torch.tensor(obs, dtype=torch.float32)
            obs_tensor = obs_tensor.to(trainer.device)

            with torch.no_grad():
                action, _, _, _ = trainer.agent.get_action_and_value(obs_tensor)
                return action

        # Run evaluation using refactored function
        eval_stats = evaluate_multi_agent_policy(
            env=eval_env,
            policy_fn=policy_fn,
            num_episodes=self.num_eval_episodes,
            target_expiry_penalty=trainer.args.target_expiry_penalty,
            verbose=True
        )

        episode_data_list = eval_stats.get("episode_data_list", [])
        if create_videos:
            # Create videos for first num_video_episodes
            max_video_episodes = min(self.num_video_episodes, self.num_eval_episodes, len(episode_data_list))
            for episode_idx in range(max_video_episodes):
                episode_data = episode_data_list[episode_idx]
                video_path = self.rollouts_dir / f"iter_{iteration}_ep_{episode_idx}.mp4"
                eval_env.create_competition_gif(episode_data, video_path, fps=2)

                # Log to wandb if enabled and the video exists
                if trainer.args.track and self.log_videos_to_wandb and video_path.exists() and video_path.stat().st_size > 0:
                    wandb.log({
                        f"eval/rollout_ep_{episode_idx}": wandb.Video(str(video_path), fps=2, format="mp4"),
                    }, step=global_step)
                elif trainer.args.track and self.log_videos_to_wandb:
                    print(f"⚠️  Skipping wandb.Video for missing video: {video_path}")

        eval_env.close()

        # Compute aggregate statistics
        avg_return = np.mean(eval_stats["episode_returns"])
        avg_length = np.mean(eval_stats["episode_lengths"])
        avg_targets = np.mean(eval_stats["targets_reached_per_episode"])
        avg_expired = np.mean(eval_stats["expired_targets_per_episode"])
        avg_min_reached = np.mean(eval_stats["min_targets_reached_per_episode"])
        success_rate = sum(1 for t in eval_stats["targets_reached_per_episode"]
                           if t == eval_num_agents) / self.num_eval_episodes

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
            "num_agents": eval_num_agents,
            "train_num_agents": trainer.args.num_agents,
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

    def evaluate_single_agent_policy(
        self,
        trainer: SingleAgentPPOTrainer,
        iteration: int,
        global_step: int,
        create_videos: bool = True,
    ):
        """Evaluate the single-agent policy with rollouts and create visualizations."""
        eval_num_targets = self.eval_num_targets or trainer.args.num_targets
        if eval_num_targets != trainer.args.num_targets:
            raise ValueError(
                "Eval num_targets differs from training in single-agent mode, which uses a fixed "
                "observation size. Train with the desired target count or add a variable-target encoder."
            )
        print(f"\n{'='*60}")
        print(f"EVALUATION - Iteration {iteration}")
        if create_videos:
            print(f"Running {self.num_eval_episodes} episodes (saving videos for first {self.num_video_episodes})")
        else:
            print(f"Running {self.num_eval_episodes} episodes (no videos this iteration)")
        print(f"{'='*60}\n")

        # Create evaluation environment with longer max_steps
        env_config = BiddingGridworldConfig(
            grid_size=trainer.args.grid_size,
            num_agents=eval_num_targets,
            bid_upper_bound=0,
            bid_penalty=0.0,
            target_reward=trainer.args.target_reward,
            max_steps=self.eval_max_steps,
            action_window=1,
            distance_reward_scale=trainer.args.distance_reward_scale,
            target_expiry_steps=trainer.args.target_expiry_steps,
            target_expiry_penalty=trainer.args.target_expiry_penalty,
            moving_targets=trainer.args.moving_targets,
            direction_change_prob=trainer.args.direction_change_prob,
            target_move_interval=trainer.args.target_move_interval,
            window_bidding=False,
            window_penalty=0.0,
            visible_targets=None,
            single_agent_mode=True,
            reward_decay_factor=trainer.args.reward_decay_factor,
        )
        eval_env = BiddingGridworld(
            env_config,
            num_envs=1,
            device=trainer.device,
            seed=trainer.args.seed,
        )

        # Create policy wrapper function
        def policy_fn(obs):
            """Get action for a single env."""
            obs_tensor = obs if torch.is_tensor(obs) else torch.tensor(obs, dtype=torch.float32)
            obs_tensor = obs_tensor.to(trainer.device)
            with torch.no_grad():
                action, _, _, _ = trainer.agent.get_action_and_value(obs_tensor.unsqueeze(0))
                return action.squeeze(0)

        # Run evaluation using refactored function
        eval_stats = evaluate_single_agent_policy(
            env=eval_env,
            policy_fn=policy_fn,
            num_episodes=self.num_eval_episodes,
            target_expiry_penalty=trainer.args.target_expiry_penalty,
            verbose=True
        )

        episode_data_list = eval_stats.get("episode_data_list", [])
        if create_videos:
            # Create videos for first num_video_episodes
            max_video_episodes = min(self.num_video_episodes, self.num_eval_episodes, len(episode_data_list))
            for episode_idx in range(max_video_episodes):
                episode_data = episode_data_list[episode_idx]
                video_path = self.rollouts_dir / f"iter_{iteration}_ep_{episode_idx}.mp4"
                eval_env.create_single_agent_gif(episode_data, video_path, fps=2)

                # Log to wandb if enabled and the video exists
                if trainer.args.track and self.log_videos_to_wandb and video_path.exists() and video_path.stat().st_size > 0:
                    wandb.log({
                        f"eval/rollout_ep_{episode_idx}": wandb.Video(str(video_path), fps=2, format="mp4"),
                    }, step=global_step)
                elif trainer.args.track and self.log_videos_to_wandb:
                    print(f"⚠️  Skipping wandb.Video for missing video: {video_path}")

        eval_env.close()

        # Compute aggregate statistics
        avg_return = np.mean(eval_stats["episode_returns"])
        avg_length = np.mean(eval_stats["episode_lengths"])
        avg_targets = np.mean(eval_stats["targets_reached_per_episode"])
        avg_expired = np.mean(eval_stats["expired_targets_per_episode"])
        avg_min_reached = np.mean(eval_stats["min_targets_reached_per_episode"])
        success_rate = sum(1 for t in eval_stats["targets_reached_per_episode"]
                           if t == eval_num_targets) / self.num_eval_episodes

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
            "num_targets": eval_num_targets,
            "train_num_targets": trainer.args.num_targets,
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

        if hasattr(args, "num_iterations"):
            if self.single_agent_mode:
                args.total_timesteps = args.num_iterations * args.num_envs * args.num_steps
            else:
                args.total_timesteps = args.num_iterations * args.num_envs * args.num_steps * args.num_agents

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
                    create_videos = iteration % self.video_freq == 0
                    self.evaluate_single_agent_policy(trainer, iteration, global_step, create_videos=create_videos)

            def on_training_end(trainer, global_step):
                # Final evaluation
                print("\n" + "="*80)
                print("FINAL EVALUATION")
                print("="*80)
                self.evaluate_single_agent_policy(trainer, trainer.args.num_iterations, global_step, create_videos=True)
        else:
            # Multi-agent callbacks
            def on_iteration_end(trainer, iteration, global_step):
                # Checkpoint saving
                if iteration % self.checkpoint_freq == 0:
                    self.save_checkpoint(trainer, iteration, global_step)

                # Evaluation
                if iteration % self.eval_freq == 0:
                    create_videos = iteration % self.video_freq == 0
                    self.evaluate_policy(trainer, iteration, global_step, create_videos=create_videos)

            def on_training_end(trainer, global_step):
                # Final evaluation
                print("\n" + "="*80)
                print("FINAL EVALUATION")
                print("="*80)
                self.evaluate_policy(trainer, trainer.args.num_iterations, global_step, create_videos=True)

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
        print(f"Video frequency: every {self.video_freq} iterations")
        print(f"Evaluation episodes: {self.num_eval_episodes} (saving videos for first {self.num_video_episodes})\n")

        trainer.train()
        trainer.save_model()
        trainer.cleanup()

        print(f"\n✅ Training complete! Results saved to {self.log_dir}")
