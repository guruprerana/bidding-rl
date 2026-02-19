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
            fire_while_hot_penalty=getattr(args, "fire_while_hot_penalty", 0.0),
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
        # Track reward components across episodes
        all_episode_components: List[Dict[str, float]] = []
        # Track agent control counts (multi-agent only)
        all_agent_control_counts: List[List[int]] = []
        # Track per-agent returns (multi-agent only)
        all_agent_returns: List[List[float]] = []

        for ep_idx in range(self.num_eval_episodes):
            obs, _ = env.reset()
            done = False
            ep_return = 0.0
            ep_len = 0
            last_score = 0.0
            frames: List[np.ndarray] = []
            ep_components: Dict[str, float] = {}
            agent_control_counts = [0] * args.num_agents
            agent_returns = [0.0] * args.num_agents

            # Capture initial frame if recording this episode
            should_record = need_render and ep_idx < self.num_video_episodes
            if should_record:
                frame = env.render(env_idx=0, show_agent_overlay=not single_agent_mode)
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
                    frame = env.render(env_idx=0, show_agent_overlay=not single_agent_mode)
                    if frame is not None:
                        frames.append(frame)

                if isinstance(info, dict):
                    score_tensor = info.get("score", None)
                    if torch.is_tensor(score_tensor):
                        last_score = float(score_tensor.item())
                    # Track which agent has control (multi-agent only)
                    if not single_agent_mode:
                        winning_agent = info.get("winning_agent")
                        if winning_agent is not None:
                            agent_idx = int(winning_agent.item())
                            if 0 <= agent_idx < len(agent_control_counts):
                                agent_control_counts[agent_idx] += 1
                    # Accumulate reward components
                    rc = info.get("reward_components", {})
                    for key, val in rc.items():
                        v = float(val.item()) if torch.is_tensor(val) else float(val)
                        ep_components[key] = ep_components.get(key, 0.0) + v

                if not single_agent_mode:
                    for i in range(args.num_agents):
                        agent_returns[i] += reward[0, i].item()
                ep_return += reward.sum().item() if not single_agent_mode else reward.item()
                done = bool(terminated.item() or truncated.item())
                ep_len += 1

            returns.append(ep_return)
            scores.append(last_score)
            lengths.append(ep_len)
            all_episode_components.append(ep_components)
            if not single_agent_mode:
                all_agent_control_counts.append(agent_control_counts)
                all_agent_returns.append(agent_returns)

            if should_record and frames:
                episode_frames.append(frames)

        env.close()

        # Write videos
        if create_videos and episode_frames and iteration is not None:
            self._write_videos(episode_frames, iteration, global_step, trainer.args.track)

        # Compute average reward components across episodes
        component_keys = all_episode_components[0].keys() if all_episode_components else []
        avg_components = {
            f"avg_{key}": float(np.mean([ep.get(key, 0.0) for ep in all_episode_components]))
            for key in component_keys
        }

        # Compute average agent control counts (multi-agent only)
        avg_agent_control = {}
        if not single_agent_mode and all_agent_control_counts:
            for i in range(args.num_agents):
                avg_agent_control[f"avg_agent_{i}_control_steps"] = float(
                    np.mean([ep[i] for ep in all_agent_control_counts])
                )

        # Compute per-agent return stats (multi-agent only)
        per_agent_return_stats = {}
        if not single_agent_mode and all_agent_returns:
            for i in range(args.num_agents):
                agent_rets = [ep[i] for ep in all_agent_returns]
                per_agent_return_stats[f"avg_agent_{i}_return"] = float(np.mean(agent_rets))
                per_agent_return_stats[f"std_agent_{i}_return"] = float(np.std(agent_rets))

        stats = {
            "avg_score": float(np.mean(scores)),
            "avg_length": float(np.mean(lengths)),
            "std_score": float(np.std(scores)),
            "std_length": float(np.std(lengths)),
            "avg_return": float(np.mean(returns)),
            **avg_components,
            **avg_agent_control,
            **per_agent_return_stats,
        }
        if iteration is not None:
            per_episode = {
                "scores": [float(s) for s in scores],
                "returns": [float(r) for r in returns],
                "lengths": [int(l) for l in lengths],
            }
            if not single_agent_mode and all_agent_control_counts:
                per_episode["agent_control_counts"] = all_agent_control_counts
            if not single_agent_mode and all_agent_returns:
                per_episode["agent_returns"] = all_agent_returns

            eval_summary = {
                "iteration": iteration,
                "global_step": global_step,
                "num_episodes": self.num_eval_episodes,
                "timestamp": datetime.now().isoformat(),
                "statistics": stats,
                "per_episode": per_episode,
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

    @classmethod
    def evaluate_checkpoint(
        cls,
        exp_dir: str | Path,
        model_filename: str = "assault_agent.pt",
        num_eval_episodes: int = 5,
        num_video_episodes: int = 3,
        output_subdir: str = "eval_videos",
        render_oc_overlay: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate a trained model from an experiment directory.

        Args:
            exp_dir: Path to experiment directory containing config and model
            model_filename: Name of the model file (default: assault_agent.pt)
            num_eval_episodes: Number of episodes to evaluate
            num_video_episodes: Number of episodes to save as videos
            output_subdir: Subdirectory for output videos and stats
            render_oc_overlay: Whether to render OCAtari object detection overlay

        Returns:
            Dictionary of evaluation statistics
        """
        from assault.assault_bidding_ppo import AssaultSharedAgent
        from assault.assault_single_agent_ppo import AssaultSingleAgent

        exp_dir = Path(exp_dir)
        model_path = exp_dir / model_filename
        config_path = exp_dir / "config" / "training_config.json"
        output_dir = exp_dir / output_subdir
        output_dir.mkdir(exist_ok=True)

        with open(config_path) as f:
            config = json.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        # Determine if single-agent based on model filename or config
        single_agent_mode = "single_agent" in model_filename

        env_config = AssaultConfig(
            num_agents=config["num_agents"],
            max_enemies=config["max_enemies"],
            bid_upper_bound=config.get("bid_upper_bound", 0),
            bid_penalty=config.get("bid_penalty", 0.0),
            action_window=config.get("action_window", 1),
            window_bidding=config.get("window_bidding", False),
            window_penalty=config.get("window_penalty", 0.0),
            enemy_destroy_reward=config["enemy_destroy_reward"],
            hit_penalty=config["hit_penalty"],
            life_loss_penalty=config["life_loss_penalty"],
            raw_score_scale=config.get("raw_score_scale", 0.0),
            fire_while_hot_penalty=config.get("fire_while_hot_penalty", 0.0),
            max_steps=config["max_steps"],
            hud=config["hud"],
            single_agent_mode=single_agent_mode,
            allow_variable_enemies=config["allow_variable_enemies"],
            allow_sideward_fire=config.get("allow_sideward_fire", True),
        )

        env = AssaultEnv(
            env_config,
            num_envs=1,
            device=device,
            seed=config["seed"],
            render_mode="rgb_array",
            render_oc_overlay=render_oc_overlay,
        )

        # Create and load agent
        if single_agent_mode:
            agent = AssaultSingleAgent(
                obs_dim=env.obs_dim,
                action_space_n=env.action_space_n,
                actor_hidden_sizes=tuple(config["actor_hidden_sizes"]),
                critic_hidden_sizes=tuple(config["critic_hidden_sizes"]),
            ).to(device)
            obs_dim = env.obs_dim
        else:
            agent = AssaultSharedAgent(
                obs_dim=env.per_agent_obs_dim,
                action_space_n=env.action_space_n,
                bid_upper_bound=config["bid_upper_bound"],
                window_bidding=config["window_bidding"],
                action_window=config["action_window"],
                actor_hidden_sizes=tuple(config["actor_hidden_sizes"]),
                critic_hidden_sizes=tuple(config["critic_hidden_sizes"]),
            ).to(device)
            obs_dim = env.per_agent_obs_dim

        agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        agent.eval()
        print(f"Loaded model from {model_path}")

        # Run evaluation
        returns = []
        scores = []
        lengths = []
        episode_frames: List[List[np.ndarray]] = []
        all_episode_components: List[Dict[str, float]] = []
        all_agent_control_counts: List[List[int]] = []

        for ep_idx in range(num_eval_episodes):
            obs, _ = env.reset()
            done = False
            ep_return = 0.0
            ep_len = 0
            last_score = 0.0
            frames: List[np.ndarray] = []
            ep_components: Dict[str, float] = {}
            agent_control_counts = [0] * config["num_agents"]

            should_record = ep_idx < num_video_episodes
            if should_record:
                frame = env.render(env_idx=0, show_agent_overlay=not single_agent_mode)
                if frame is not None:
                    frames.append(frame)

            while not done:
                with torch.no_grad():
                    if single_agent_mode:
                        action, _, _, _ = agent.get_action_and_value(obs)
                    else:
                        flat_obs = obs.reshape(-1, obs_dim)
                        action, _, _, _ = agent.get_action_and_value(flat_obs)
                        action = action.reshape(1, config["num_agents"], -1)

                obs, reward, terminated, truncated, info = env.step(action)

                if should_record:
                    frame = env.render(env_idx=0, show_agent_overlay=not single_agent_mode)
                    if frame is not None:
                        frames.append(frame)

                if isinstance(info, dict):
                    score_tensor = info.get("score", None)
                    if torch.is_tensor(score_tensor):
                        last_score = float(score_tensor.item())
                    # Track which agent has control (multi-agent only)
                    if not single_agent_mode:
                        winning_agent = info.get("winning_agent")
                        if winning_agent is not None:
                            agent_idx = int(winning_agent.item())
                            if 0 <= agent_idx < len(agent_control_counts):
                                agent_control_counts[agent_idx] += 1
                    # Accumulate reward components
                    rc = info.get("reward_components", {})
                    for key, val in rc.items():
                        v = float(val.item()) if torch.is_tensor(val) else float(val)
                        ep_components[key] = ep_components.get(key, 0.0) + v

                ep_return += reward.sum().item() if not single_agent_mode else reward.item()
                done = bool(terminated.item() or truncated.item())
                ep_len += 1

            returns.append(ep_return)
            scores.append(last_score)
            lengths.append(ep_len)
            all_episode_components.append(ep_components)
            if not single_agent_mode:
                all_agent_control_counts.append(agent_control_counts)

            print(f"Episode {ep_idx}: score={last_score:.0f}, return={ep_return:.2f}, length={ep_len}")

            # Write video
            if should_record and frames:
                video_path = output_dir / f"episode_{ep_idx}.mp4"
                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(str(video_path), fourcc, 30, (w, h))
                for frame in frames:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                out.release()
                print(f"  Saved: {video_path} ({len(frames)} frames)")

        env.close()

        # Compute statistics
        component_keys = all_episode_components[0].keys() if all_episode_components else []
        avg_components = {
            f"avg_{key}": float(np.mean([ep.get(key, 0.0) for ep in all_episode_components]))
            for key in component_keys
        }

        avg_agent_control = {}
        if not single_agent_mode and all_agent_control_counts:
            for i in range(config["num_agents"]):
                avg_agent_control[f"avg_agent_{i}_control_steps"] = float(
                    np.mean([ep[i] for ep in all_agent_control_counts])
                )

        stats = {
            "avg_score": float(np.mean(scores)),
            "avg_length": float(np.mean(lengths)),
            "std_score": float(np.std(scores)),
            "std_length": float(np.std(lengths)),
            "avg_return": float(np.mean(returns)),
            **avg_components,
            **avg_agent_control,
        }

        per_episode = {
            "scores": [float(s) for s in scores],
            "returns": [float(r) for r in returns],
            "lengths": [int(l) for l in lengths],
        }
        if not single_agent_mode and all_agent_control_counts:
            per_episode["agent_control_counts"] = all_agent_control_counts

        eval_summary = {
            "model_path": str(model_path),
            "num_episodes": num_eval_episodes,
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "per_episode": per_episode,
        }

        stats_path = output_dir / "eval_stats.json"
        with open(stats_path, "w") as f:
            json.dump(eval_summary, f, indent=2)
        print(f"\nStats saved to {stats_path}")
        print(f"Videos saved to {output_dir}")

        return stats
