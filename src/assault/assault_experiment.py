"""Experiment wrapper for Assault PPO training."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import wandb

from assault.assault_torch import AssaultConfig, AssaultEnv
from assault.assault_bidding_ppo import AssaultPPOTrainer, AssaultArgs
from assault.assault_single_agent_ppo import AssaultSingleAgentPPOTrainer, AssaultSingleAgentArgs


class AssaultExperiment:
    """Experiment wrapper for PPO training with periodic evaluation and checkpointing."""

    def __init__(
        self,
        base_log_dir: str = "logs",
        experiment_name: str = "",
        checkpoint_freq: int = 50,
        eval_freq: int = 25,
        video_freq: int = 0,
        num_eval_episodes: int = 5,
        num_video_episodes: int = 3,
        log_videos_to_wandb: bool = False,
        single_agent_mode: bool = False,
        render_oc_overlay: bool = False,
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not experiment_name:
            mode_prefix = "single_agent" if single_agent_mode else "multi_agent"
            experiment_name = f"assault_ppo_{mode_prefix}"

        self.log_dir = Path(base_log_dir) / f"{experiment_name}_{timestamp}"
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.video_freq = eval_freq if video_freq in {0, None} else video_freq
        self.num_eval_episodes = num_eval_episodes
        self.num_video_episodes = num_video_episodes
        self.log_videos_to_wandb = log_videos_to_wandb
        self.single_agent_mode = single_agent_mode
        self.render_oc_overlay = render_oc_overlay

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.log_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.config_dir = self.log_dir / "config"
        self.config_dir.mkdir(exist_ok=True)
        self.eval_dir = self.log_dir / "evaluation"
        self.eval_dir.mkdir(exist_ok=True)
        self.rollouts_dir = self.log_dir / "rollouts"
        self.rollouts_dir.mkdir(exist_ok=True)

        print(f"📁 Experiment directory: {self.log_dir}")

    def log_codebase_to_wandb(self, run):
        disabled = os.environ.get("WANDB_MODE", "").lower() == "disabled" or \
            os.environ.get("WANDB_DISABLED", "").lower() in {"true", "1"}
        if not run or disabled:
            return

        artifact = wandb.Artifact(
            name=f"codebase-{run.id}",
            type="code",
            description="Codebase snapshot (src folder + training script)"
        )
        project_root = self.log_dir.parent.parent
        src_dir = project_root / "src"
        if src_dir.exists():
            artifact.add_dir(str(src_dir), name="src")
        train_script = project_root / "train_ppo_moving_targets.py"
        if train_script.exists():
            artifact.add_file(str(train_script), name="train_ppo_moving_targets.py")
        run.log_artifact(artifact)

    def save_config(self, args):
        config = vars(args)
        config_file = self.config_dir / "training_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2, default=str)
        print(f"💾 Config saved to {config_file}")

    def save_checkpoint(self, trainer, iteration: int, global_step: int):
        checkpoint_dir = self.checkpoints_dir / f"iter_{iteration}"
        checkpoint_dir.mkdir(exist_ok=True)
        model_path = checkpoint_dir / "agent.pt"
        torch.save(trainer.agent.state_dict(), str(model_path))
        info_path = checkpoint_dir / "checkpoint_info.json"
        with open(info_path, "w") as f:
            json.dump(
                {
                    "iteration": iteration,
                    "global_step": global_step,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        if trainer.args.track:
            wandb.save(str(model_path))

    def _evaluate(
        self,
        trainer,
        args,
        single_agent_mode: bool,
        iteration: int | None = None,
        global_step: int | None = None,
        create_videos: bool = True,
    ) -> Dict[str, float]:
        # Determine if we need rendering
        need_render = create_videos and self.num_video_episodes > 0
        render_mode = "rgb_array" if need_render else None

        env_config = AssaultConfig(
            num_agents=args.num_agents,
            max_enemies=args.max_enemies,
            bid_upper_bound=getattr(args, "bid_upper_bound", 0),
            bid_penalty=getattr(args, "bid_penalty", 0.0),
            action_window=getattr(args, "action_window", 1),
            window_bidding=getattr(args, "window_bidding", False),
            window_penalty=getattr(args, "window_penalty", 0.0),
            enemy_destroy_reward=args.enemy_destroy_reward,
            hit_penalty=args.hit_penalty,
            life_loss_penalty=args.life_loss_penalty,
            raw_score_scale=getattr(args, "raw_score_scale", 0.0),
            max_steps=args.max_steps,
            hud=args.hud,
            single_agent_mode=single_agent_mode,
            allow_variable_enemies=args.allow_variable_enemies,
            allow_sideward_fire=getattr(args, "allow_sideward_fire", True),
        )
        env = AssaultEnv(
            env_config,
            num_envs=1,
            device=trainer.device,
            seed=args.seed,
            render_mode=render_mode,
            render_oc_overlay=self.render_oc_overlay,
        )

        returns = []
        scores = []
        lengths = []
        episode_frames: List[List[np.ndarray]] = []

        for ep_idx in range(self.num_eval_episodes):
            obs, _ = env.reset()
            done = False
            ep_return = 0.0
            ep_len = 0
            last_score = 0.0
            frames: List[np.ndarray] = []

            # Capture initial frame if recording this episode
            should_record = need_render and ep_idx < self.num_video_episodes
            if should_record:
                frame = env.render(env_idx=0)
                if frame is not None:
                    frames.append(frame)

            while not done:
                with torch.no_grad():
                    if single_agent_mode:
                        action, _, _, _ = trainer.agent.get_action_and_value(obs)
                    else:
                        flat_obs = obs.reshape(-1, trainer.obs_dim)
                        action, _, _, _ = trainer.agent.get_action_and_value(flat_obs)
                        action = action.reshape(1, args.num_agents, -1)

                obs, reward, terminated, truncated, info = env.step(action)

                # Capture frame after step
                if should_record:
                    frame = env.render(env_idx=0)
                    if frame is not None:
                        frames.append(frame)

                if isinstance(info, dict):
                    score_tensor = info.get("score", None)
                    if torch.is_tensor(score_tensor):
                        last_score = float(score_tensor.item())
                ep_return += reward.sum().item() if not single_agent_mode else reward.item()
                done = bool(terminated.item() or truncated.item())
                ep_len += 1

            returns.append(ep_return)
            scores.append(last_score)
            lengths.append(ep_len)

            if should_record and frames:
                episode_frames.append(frames)

        env.close()

        # Write videos
        if create_videos and episode_frames and iteration is not None:
            self._write_videos(episode_frames, iteration, global_step, trainer.args.track)

        stats = {
            "avg_score": float(np.mean(scores)),
            "avg_length": float(np.mean(lengths)),
            "std_score": float(np.std(scores)),
            "std_length": float(np.std(lengths)),
            "avg_return": float(np.mean(returns)),
        }
        if iteration is not None:
            eval_summary = {
                "iteration": iteration,
                "global_step": global_step,
                "num_episodes": self.num_eval_episodes,
                "timestamp": datetime.now().isoformat(),
                "statistics": stats,
                "per_episode": {
                    "scores": [float(s) for s in scores],
                    "returns": [float(r) for r in returns],
                    "lengths": [int(l) for l in lengths],
                },
            }
            eval_file = self.eval_dir / f"iter_{iteration}_eval_stats.json"
            with open(eval_file, "w") as f:
                json.dump(eval_summary, f, indent=2)
        return stats

    def _write_videos(
        self,
        episode_frames: List[List[np.ndarray]],
        iteration: int,
        global_step: int | None,
        track: bool,
    ) -> None:
        """Write episode frames to MP4 videos."""
        for ep_idx, frames in enumerate(episode_frames):
            if not frames:
                continue

            video_path = self.rollouts_dir / f"iter_{iteration}_ep_{ep_idx}.mp4"

            # Get frame dimensions - OCAtari returns (H, W, C) format
            first_frame = frames[0]
            h, w = first_frame.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, 30, (w, h))

            for frame in frames:
                # Ensure frame is in correct format
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            out.release()
            print(f"📹 Video saved: {video_path} ({len(frames)} frames)")

            # Log to wandb if enabled
            if track and self.log_videos_to_wandb and video_path.exists():
                wandb.log(
                    {f"eval/rollout_ep_{ep_idx}": wandb.Video(str(video_path), fps=30, format="mp4")},
                    step=global_step,
                )

    def run(self, args):
        mode_str = "SINGLE-AGENT" if self.single_agent_mode else "MULTI-AGENT"
        print(f"\n{'='*80}")
        print(f"ASSAULT PPO TRAINING - {mode_str} MODE")
        print(f"{'='*80}\n")

        if hasattr(args, "num_iterations"):
            if self.single_agent_mode:
                args.total_timesteps = args.num_iterations * args.num_envs * args.num_steps
            else:
                args.total_timesteps = args.num_iterations * args.num_envs * args.num_steps * args.num_agents

        self.save_config(args)

        print(f"Checkpoint frequency: every {self.checkpoint_freq} iterations")
        print(f"Evaluation frequency: every {self.eval_freq} iterations")
        print(f"Video frequency: every {self.video_freq} iterations")
        print(f"Evaluation episodes: {self.num_eval_episodes} (saving videos for first {self.num_video_episodes})\n")

        def on_iteration_end(trainer, iteration, global_step):
            if iteration % self.checkpoint_freq == 0:
                self.save_checkpoint(trainer, iteration, global_step)
            if iteration % self.eval_freq == 0:
                create_videos = iteration % self.video_freq == 0
                stats = self._evaluate(
                    trainer, args, self.single_agent_mode, iteration, global_step, create_videos=create_videos
                )
                if trainer.args.track:
                    wandb.log({f"eval/{k}": v for k, v in stats.items()}, step=global_step)

        def on_training_end(trainer, global_step):
            # Always create videos for final evaluation
            stats = self._evaluate(
                trainer, args, self.single_agent_mode, args.num_iterations, global_step, create_videos=True
            )
            if trainer.args.track:
                wandb.log({f"eval/{k}": v for k, v in stats.items()}, step=global_step)

        callbacks = {
            "on_iteration_end": on_iteration_end,
            "on_training_end": on_training_end,
        }

        if self.single_agent_mode:
            trainer = AssaultSingleAgentPPOTrainer(args, callbacks=callbacks)
        else:
            trainer = AssaultPPOTrainer(args, callbacks=callbacks)

        trainer.setup()
        self.log_codebase_to_wandb(wandb.run)
        trainer.train()

        # Save final model to logs directory
        model_filename = "assault_single_agent.pt" if self.single_agent_mode else "assault_agent.pt"
        model_path = self.log_dir / model_filename
        trainer.save_model(str(model_path))
        print(f"💾 Final model saved to {model_path}")

        trainer.cleanup()

        print(f"\n✅ Training complete! Results saved to {self.log_dir}")
