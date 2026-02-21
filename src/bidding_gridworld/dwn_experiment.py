"""Experiment orchestration for DWN training with evaluations and checkpoints."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import wandb

from bidding_gridworld.bidding_gridworld_torch import (
    BiddingGridworld,
    BiddingGridworldConfig,
    evaluate_multi_agent_policy,
)
from bidding_gridworld.dwn import GridworldDWNArgs, GridworldDWNTrainer


class DWNExperiment:
    """Experiment wrapper for DWN training with periodic evaluation and checkpointing.

    Mirrors PPOMovingTargetsExperiment but adapted for the DWN trainer:
    - Step-based frequencies (not iteration-based) via the on_step callback
    - DWN greedy policy for evaluation (argmax Q for directions, argmax W for winner)
    """

    def __init__(
        self,
        base_log_dir: str = "logs",
        experiment_name: str = "",
        checkpoint_freq: int = 2_500_000,
        eval_freq: int = 2_500_000,
        video_freq: int = 0,
        num_eval_episodes: int = 5,
        num_video_episodes: int = 2,
        log_videos_to_wandb: bool = False,
        eval_max_steps: Optional[int] = None,
    ):
        """
        Args:
            base_log_dir: Base directory for logs.
            experiment_name: Name for this experiment.
            checkpoint_freq: Save checkpoint every N global steps (0 = disabled).
            eval_freq: Evaluate every N global steps (0 = disabled).
            video_freq: Save videos every N global steps (0 = same as eval_freq).
            num_eval_episodes: Number of episodes per evaluation.
            num_video_episodes: Number of episodes to save as MP4s.
            log_videos_to_wandb: Upload MP4s to wandb.
            eval_max_steps: Max steps per eval episode (None = use training max_steps).
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not experiment_name:
            experiment_name = "dwn"

        self.log_dir = Path(base_log_dir) / f"{experiment_name}_{timestamp}"
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.video_freq = eval_freq if video_freq in {0, None} else video_freq
        self.num_eval_episodes = num_eval_episodes
        self.num_video_episodes = num_video_episodes
        self.log_videos_to_wandb = log_videos_to_wandb
        self.eval_max_steps = eval_max_steps

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.log_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.rollouts_dir = self.log_dir / "rollouts"
        self.rollouts_dir.mkdir(exist_ok=True)
        self.config_dir = self.log_dir / "config"
        self.config_dir.mkdir(exist_ok=True)

        print(f"Experiment directory: {self.log_dir}")

    def save_config(self, args: GridworldDWNArgs):
        config_file = self.config_dir / "training_config.json"
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=2, default=str)
        print(f"Config saved to {config_file}")

    def save_checkpoint(self, trainer: GridworldDWNTrainer, global_step: int):
        ckpt_dir = self.checkpoints_dir / f"step_{global_step}"
        trainer.save_model(str(ckpt_dir))

        info = {
            "global_step": global_step,
            "timestamp": datetime.now().isoformat(),
        }
        with open(ckpt_dir / "checkpoint_info.json", "w") as f:
            json.dump(info, f, indent=2)

        if trainer.args.track:
            for pt_file in ckpt_dir.glob("*.pt"):
                wandb.save(str(pt_file))

    def evaluate_policy(
        self,
        trainer: GridworldDWNTrainer,
        global_step: int,
        create_videos: bool = True,
    ):
        """Run greedy evaluation episodes and optionally save videos."""
        args = trainer.args
        max_steps = self.eval_max_steps or args.max_steps

        print(f"\n{'='*60}")
        print(f"EVALUATION — step {global_step}")
        if create_videos:
            print(f"Running {self.num_eval_episodes} episodes "
                  f"(saving videos for first {self.num_video_episodes})")
        else:
            print(f"Running {self.num_eval_episodes} episodes (no videos this eval)")
        print(f"{'='*60}\n")

        env_config = BiddingGridworldConfig(
            grid_size=args.grid_size,
            num_agents=args.num_targets,
            bid_upper_bound=1,
            bid_penalty=0.0,
            target_reward=args.target_reward,
            max_steps=max_steps,
            action_window=1,
            distance_reward_scale=args.distance_reward_scale,
            target_expiry_steps=args.target_expiry_steps,
            target_expiry_penalty=args.target_expiry_penalty,
            moving_targets=args.moving_targets,
            direction_change_prob=args.direction_change_prob,
            target_move_interval=args.target_move_interval,
            window_bidding=False,
            window_penalty=0.0,
            visible_targets=args.visible_targets,
            single_agent_mode=False,
        )
        eval_env = BiddingGridworld(
            env_config,
            num_envs=1,
            device=trainer.device,
            seed=args.seed,
        )

        num_agents = args.num_targets

        def policy_fn(obs: torch.Tensor) -> torch.Tensor:
            """Greedy DWN policy: argmax Q for directions, argmax W for winner.

            Args:
                obs: (num_agents, per_agent_obs_dim)
            Returns:
                actions: (num_agents, 2) — [direction, bid]
            """
            obs_t = obs.to(trainer.device)
            with torch.no_grad():
                q_values = trainer.q_network(obs_t)          # (A, 4)
                directions = q_values.argmax(dim=1)           # (A,)
                w_values = trainer.w_network(obs_t)           # (A,)
                winner = w_values.argmax(dim=0)               # scalar

                bids = torch.zeros(num_agents, dtype=torch.int64, device=trainer.device)
                bids[winner] = 1
                return torch.stack([directions, bids], dim=-1)  # (A, 2)

        eval_stats = evaluate_multi_agent_policy(
            env=eval_env,
            policy_fn=policy_fn,
            num_episodes=self.num_eval_episodes,
            target_expiry_penalty=args.target_expiry_penalty,
            verbose=True,
        )

        if create_videos:
            episode_data_list = eval_stats.get("episode_data_list", [])
            max_vids = min(self.num_video_episodes, len(episode_data_list))
            for ep_idx in range(max_vids):
                video_path = self.rollouts_dir / f"step_{global_step}_ep_{ep_idx}.mp4"
                eval_env.create_competition_gif(episode_data_list[ep_idx], video_path, fps=2)

                if args.track and self.log_videos_to_wandb and video_path.exists() and video_path.stat().st_size > 0:
                    wandb.log(
                        {f"eval/rollout_ep_{ep_idx}": wandb.Video(str(video_path), fps=2, format="mp4")},
                        step=global_step,
                    )

        eval_env.close()

        avg_return = np.mean(eval_stats["episode_returns"])
        avg_length = np.mean(eval_stats["episode_lengths"])
        avg_targets = np.mean(eval_stats["targets_reached_per_episode"])
        avg_expired = np.mean(eval_stats["expired_targets_per_episode"])
        avg_min_reached = np.mean(eval_stats["min_targets_reached_per_episode"])
        success_rate = sum(
            1 for t in eval_stats["targets_reached_per_episode"] if t == num_agents
        ) / self.num_eval_episodes

        if args.track:
            wandb.log({
                "eval/avg_return": avg_return,
                "eval/avg_length": avg_length,
                "eval/avg_targets_reached": avg_targets,
                "eval/avg_expired_targets": avg_expired,
                "eval/avg_min_targets_reached": avg_min_reached,
                "eval/success_rate": success_rate,
            }, step=global_step)

        eval_summary = {
            "global_step": global_step,
            "num_episodes": self.num_eval_episodes,
            "num_agents": num_agents,
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
            },
        }
        stats_file = self.rollouts_dir / f"step_{global_step}_eval_stats.json"
        with open(stats_file, "w") as f:
            json.dump(eval_summary, f, indent=2)
        print(f"Eval stats saved to {stats_file}")

        return eval_stats

    def run(self, args: GridworldDWNArgs):
        """Run the full training experiment."""
        print(f"\n{'='*80}")
        print(f"DWN TRAINING")
        print(f"{'='*80}\n")

        self.save_config(args)

        last_checkpoint_step = [0]
        last_eval_step = [0]
        last_video_step = [0]

        def on_step(trainer: GridworldDWNTrainer, global_step: int):
            if self.checkpoint_freq > 0 and global_step - last_checkpoint_step[0] >= self.checkpoint_freq:
                self.save_checkpoint(trainer, global_step)
                last_checkpoint_step[0] = global_step

            if self.eval_freq > 0 and global_step - last_eval_step[0] >= self.eval_freq:
                create_videos = (
                    self.video_freq <= 0
                    or global_step - last_video_step[0] >= self.video_freq
                )
                self.evaluate_policy(trainer, global_step, create_videos=create_videos)
                last_eval_step[0] = global_step
                if create_videos:
                    last_video_step[0] = global_step

        def on_training_end(trainer: GridworldDWNTrainer, global_step: int):
            self.save_checkpoint(trainer, global_step)
            print("\n" + "=" * 80)
            print("FINAL EVALUATION")
            print("=" * 80)
            self.evaluate_policy(trainer, global_step, create_videos=True)

        trainer = GridworldDWNTrainer(
            args,
            callbacks={"on_step": on_step, "on_training_end": on_training_end},
        )

        try:
            trainer.setup()

            print(f"Checkpoint frequency: every {self.checkpoint_freq:,} steps")
            print(f"Evaluation frequency: every {self.eval_freq:,} steps")
            print(f"Video frequency:      every {self.video_freq:,} steps")
            print(f"Eval episodes: {self.num_eval_episodes} "
                  f"(saving videos for first {self.num_video_episodes})\n")

            trainer.train()
        finally:
            trainer.cleanup()

        print(f"\nTraining complete! Results saved to {self.log_dir}")
