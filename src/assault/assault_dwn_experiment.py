"""Experiment orchestration for Assault DWN training with evaluations and checkpoints."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import wandb

from assault.assault_torch import AssaultConfig, AssaultEnv
from assault.assault_dwn import AssaultDWNArgs, AssaultDWNTrainer


class AssaultDWNExperiment:
    """Experiment wrapper for Assault DWN training with periodic evaluation and checkpointing.

    Mirrors DWNExperiment (bidding_gridworld/dwn_experiment.py) but uses
    step-based callbacks with the AssaultDWNTrainer.
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
    ):
        """
        Args:
            base_log_dir: Base directory for logs.
            experiment_name: Name for this experiment.
            checkpoint_freq: Save checkpoint every N global steps (0 = disabled).
            eval_freq: Evaluate every N global steps (0 = disabled).
            video_freq: Save videos every N global steps (0 = same as eval_freq).
            num_eval_episodes: Number of greedy episodes per evaluation.
            num_video_episodes: Number of episodes to save as MP4s.
            log_videos_to_wandb: Upload MP4s to wandb.
        """
        if not experiment_name:
            experiment_name = "assault_dwn"

        # Reuse an existing directory for this experiment if one exists
        base = Path(base_log_dir)
        existing_dirs = sorted(base.glob(f"{experiment_name}_*")) if base.exists() else []
        if existing_dirs:
            self.log_dir = existing_dirs[-1]
            print(f"Resuming experiment: {self.log_dir}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = base / f"{experiment_name}_{timestamp}"
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.video_freq = eval_freq if video_freq in {0, None} else video_freq
        self.num_eval_episodes = num_eval_episodes
        self.num_video_episodes = num_video_episodes
        self.log_videos_to_wandb = log_videos_to_wandb

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.log_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.rollouts_dir = self.log_dir / "rollouts"
        self.rollouts_dir.mkdir(exist_ok=True)
        self.config_dir = self.log_dir / "config"
        self.config_dir.mkdir(exist_ok=True)

        print(f"Experiment directory: {self.log_dir}")

    def save_config(self, args: AssaultDWNArgs):
        config_file = self.config_dir / "training_config.json"
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=2, default=str)
        print(f"Config saved to {config_file}")

    def save_checkpoint(self, trainer: AssaultDWNTrainer, global_step: int):
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
        trainer: AssaultDWNTrainer,
        global_step: int,
        create_videos: bool = True,
    ):
        """Run greedy evaluation episodes and optionally save MP4 videos."""
        args = trainer.args
        num_agents = args.num_agents

        need_render = create_videos and self.num_video_episodes > 0
        render_mode = "rgb_array" if need_render else None

        print(f"\n{'='*60}")
        print(f"EVALUATION — step {global_step}")
        print(
            f"Running {self.num_eval_episodes} episodes"
            + (f" (videos for first {self.num_video_episodes})" if need_render else "")
        )
        print(f"{'='*60}\n")

        env_config = AssaultConfig(
            num_agents=args.num_agents,
            max_enemies=args.max_enemies,
            bid_upper_bound=1,
            bid_penalty=0.0,
            action_window=1,
            window_bidding=False,
            window_penalty=0.0,
            enemy_destroy_reward=args.enemy_destroy_reward,
            hit_penalty=args.hit_penalty,
            life_loss_penalty=args.life_loss_penalty,
            raw_score_scale=args.raw_score_scale,
            fire_while_hot_penalty=args.fire_while_hot_penalty,
            max_steps=args.max_steps,
            hud=args.hud,
            single_agent_mode=False,
            allow_variable_enemies=args.allow_variable_enemies,
            allow_sideward_fire=args.allow_sideward_fire,
            bidding_mechanism="all_pay",
        )
        eval_env = AssaultEnv(
            env_config,
            num_envs=1,
            device=trainer.device,
            seed=args.seed,
            render_mode=render_mode,
        )

        returns: List[float] = []
        scores: List[float] = []
        lengths: List[int] = []
        all_agent_control_counts: List[List[int]] = []
        episode_frames: List[List[np.ndarray]] = []

        for ep_idx in range(self.num_eval_episodes):
            obs, _ = eval_env.reset()
            # obs: (1, num_agents, per_agent_obs_dim)

            done = False
            ep_return = 0.0
            ep_length = 0
            last_score = 0.0
            agent_counts = [0] * num_agents
            frames: List[np.ndarray] = []

            should_record = need_render and ep_idx < self.num_video_episodes
            if should_record:
                frame = eval_env.render(env_idx=0, show_agent_overlay=True)
                if frame is not None:
                    frames.append(frame)

            while not done:
                obs_t = obs.to(trainer.device)  # (1, A, obs_dim)
                with torch.no_grad():
                    flat_obs = obs_t.reshape(num_agents, -1)           # (A, obs_dim)
                    q_values = trainer.q_network(flat_obs)             # (A, num_actions)
                    directions = q_values.argmax(dim=1)                # (A,)
                    w_values = trainer.w_network(flat_obs)             # (A,)
                    winner = w_values.argmax(dim=0)                    # scalar

                    bids = torch.zeros(
                        num_agents, dtype=torch.int64, device=trainer.device
                    )
                    bids[winner] = 1
                    actions = torch.stack([directions, bids], dim=-1)  # (A, 2)
                    actions = actions.unsqueeze(0)                      # (1, A, 2)

                obs, reward, terminated, truncated, info = eval_env.step(actions)

                if should_record:
                    frame = eval_env.render(env_idx=0, show_agent_overlay=True)
                    if frame is not None:
                        frames.append(frame)

                ep_return += reward[0].sum().item()
                ep_length += 1

                if "score" in info and info["score"] is not None:
                    last_score = info["score"][0].item()

                winning_agent = info.get("winning_agent")
                if winning_agent is not None:
                    winner_idx = int(winning_agent[0].item())
                    if 0 <= winner_idx < num_agents:
                        agent_counts[winner_idx] += 1

                done = bool(terminated[0].item() or truncated[0].item())

            returns.append(ep_return)
            scores.append(last_score)
            lengths.append(ep_length)
            all_agent_control_counts.append(agent_counts)
            if should_record and frames:
                episode_frames.append(frames)

            print(
                f"  ep {ep_idx}: return={ep_return:.2f}, score={last_score:.0f}, "
                f"length={ep_length}, control={agent_counts}"
            )

        eval_env.close()

        # Write MP4 videos
        if need_render and episode_frames:
            for ep_idx, frames in enumerate(episode_frames):
                if not frames:
                    continue
                video_path = self.rollouts_dir / f"step_{global_step}_ep_{ep_idx}.mp4"
                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(str(video_path), fourcc, 30, (w, h))
                for frame in frames:
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                out.release()
                print(f"Video saved: {video_path} ({len(frames)} frames)")

                if args.track and self.log_videos_to_wandb and video_path.exists():
                    wandb.log(
                        {
                            f"eval/rollout_ep_{ep_idx}": wandb.Video(
                                str(video_path), fps=30, format="mp4"
                            )
                        },
                        step=global_step,
                    )

        # Aggregate stats
        avg_return = float(np.mean(returns))
        avg_score = float(np.mean(scores))
        avg_length = float(np.mean(lengths))

        avg_agent_control: Dict[str, float] = {}
        for i in range(num_agents):
            avg_agent_control[f"avg_agent_{i}_control_steps"] = float(
                np.mean([ep[i] for ep in all_agent_control_counts])
            )

        if args.track:
            log_dict = {
                "eval/avg_return": avg_return,
                "eval/avg_score": avg_score,
                "eval/avg_length": avg_length,
                **{f"eval/{k}": v for k, v in avg_agent_control.items()},
            }
            wandb.log(log_dict, step=global_step)

        eval_summary = {
            "global_step": global_step,
            "num_episodes": self.num_eval_episodes,
            "num_agents": num_agents,
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "avg_return": avg_return,
                "avg_score": avg_score,
                "avg_length": avg_length,
                "std_return": float(np.std(returns)),
                "std_score": float(np.std(scores)),
                "std_length": float(np.std(lengths)),
                **avg_agent_control,
            },
            "per_episode": {
                "returns": [float(r) for r in returns],
                "scores": [float(s) for s in scores],
                "lengths": [int(l) for l in lengths],
                "agent_control_counts": all_agent_control_counts,
            },
        }
        stats_file = self.rollouts_dir / f"step_{global_step}_eval_stats.json"
        with open(stats_file, "w") as f:
            json.dump(eval_summary, f, indent=2)
        print(f"Eval stats saved to {stats_file}")

        return eval_summary

    def _find_latest_checkpoint_step(self) -> Optional[int]:
        """Return the global_step of the latest checkpoint, or None."""
        if not self.checkpoints_dir.exists():
            return None
        step_dirs = []
        for p in self.checkpoints_dir.iterdir():
            if p.is_dir() and p.name.startswith("step_"):
                try:
                    step_dirs.append(int(p.name.split("_")[1]))
                except (IndexError, ValueError):
                    pass
        return max(step_dirs) if step_dirs else None

    def run(self, args: AssaultDWNArgs):
        """Run the full training experiment."""
        print(f"\n{'='*80}")
        print("ASSAULT DWN TRAINING")
        print(f"{'='*80}\n")

        latest_step = self._find_latest_checkpoint_step()
        if latest_step is not None and latest_step >= args.total_timesteps:
            print(f"Already complete (step {latest_step}/{args.total_timesteps}), skipping.")
            return

        self.save_config(args)

        last_checkpoint_step = [0]
        last_eval_step = [0]
        last_video_step = [0]

        def on_step(trainer: AssaultDWNTrainer, global_step: int):
            if (
                self.checkpoint_freq > 0
                and global_step - last_checkpoint_step[0] >= self.checkpoint_freq
            ):
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

        def on_training_end(trainer: AssaultDWNTrainer, global_step: int):
            self.save_checkpoint(trainer, global_step)
            print("\n" + "=" * 80)
            print("FINAL EVALUATION")
            print("=" * 80)
            self.evaluate_policy(trainer, global_step, create_videos=True)

        trainer = AssaultDWNTrainer(
            args,
            callbacks={"on_step": on_step, "on_training_end": on_training_end},
        )

        try:
            trainer.setup()

            print(f"Checkpoint frequency: every {self.checkpoint_freq:,} steps")
            print(f"Evaluation frequency: every {self.eval_freq:,} steps")
            print(f"Video frequency:      every {self.video_freq:,} steps")
            print(
                f"Eval episodes: {self.num_eval_episodes} "
                f"(saving videos for first {self.num_video_episodes})\n"
            )

            trainer.train()
        finally:
            trainer.cleanup()

        print(f"\nTraining complete! Results saved to {self.log_dir}")
