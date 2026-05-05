"""Experiment wrapper for Air Raid PPO training."""

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

from airraid.airraid_bidding_ppo import AirRaidArgs, AirRaidPPOTrainer
from airraid.airraid_single_agent_ppo import AirRaidSingleAgentArgs, AirRaidSingleAgentPPOTrainer
from airraid.airraid_torch import AirRaidConfig, AirRaidEnv


class AirRaidExperiment:
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
        if not experiment_name:
            mode_prefix = "single_agent" if single_agent_mode else "multi_agent"
            experiment_name = f"airraid_ppo_{mode_prefix}"
        base = Path(base_log_dir)
        existing_dirs = sorted(base.glob(f"{experiment_name}_*")) if base.exists() else []
        if existing_dirs:
            self.log_dir = existing_dirs[-1]
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = base / f"{experiment_name}_{timestamp}"
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

    def log_codebase_to_wandb(self, run):
        disabled = os.environ.get("WANDB_MODE", "").lower() == "disabled" or os.environ.get("WANDB_DISABLED", "").lower() in {"true", "1"}
        if not run or disabled:
            return
        artifact = wandb.Artifact(name=f"codebase-{run.id}", type="code", description="Codebase snapshot")
        project_root = self.log_dir.parent.parent
        src_dir = project_root / "src"
        if src_dir.exists():
            artifact.add_dir(str(src_dir), name="src")
        train_script = project_root / "train_airraid_ppo.py"
        if train_script.exists():
            artifact.add_file(str(train_script), name="train_airraid_ppo.py")
        run.log_artifact(artifact)

    def save_config(self, args):
        with open(self.config_dir / "training_config.json", "w") as f:
            json.dump(vars(args), f, indent=2, default=str)

    def save_checkpoint(self, trainer, iteration: int, global_step: int):
        checkpoint_dir = self.checkpoints_dir / f"iter_{iteration}"
        checkpoint_dir.mkdir(exist_ok=True)
        model_path = checkpoint_dir / "agent.pt"
        torch.save(trainer.agent.state_dict(), str(model_path))
        with open(checkpoint_dir / "checkpoint_info.json", "w") as f:
            json.dump({"iteration": iteration, "global_step": global_step, "timestamp": datetime.now().isoformat()}, f, indent=2)
        if trainer.args.track:
            wandb.save(str(model_path))

    def _evaluate(self, trainer, args, single_agent_mode: bool, iteration: int | None = None, global_step: int | None = None, create_videos: bool = True) -> Dict[str, float]:
        need_render = create_videos and self.num_video_episodes > 0
        render_mode = "rgb_array" if need_render else None
        env_config = AirRaidConfig(
            num_agents=args.num_agents,
            max_enemies=args.max_enemies,
            bid_upper_bound=getattr(args, "bid_upper_bound", 0),
            bid_penalty=getattr(args, "bid_penalty", 0.0),
            action_window=getattr(args, "action_window", 1),
            window_bidding=getattr(args, "window_bidding", False),
            window_penalty=getattr(args, "window_penalty", 0.0),
            enemy_destroy_reward=args.enemy_destroy_reward,
            building_hit_penalty=args.building_hit_penalty,
            life_loss_penalty=args.life_loss_penalty,
            raw_score_scale=getattr(args, "raw_score_scale", 0.0),
            enemy_missile_danger_penalty=getattr(args, "enemy_missile_danger_penalty", 0.0),
            enemy_missile_danger_y_threshold=getattr(args, "enemy_missile_danger_y_threshold", 110.0),
            enemy_missile_danger_x_radius=getattr(args, "enemy_missile_danger_x_radius", 18.0),
            enemy_missile_danger_y_radius=getattr(args, "enemy_missile_danger_y_radius", 80.0),
            enemy_missile_near_hit_penalty=getattr(args, "enemy_missile_near_hit_penalty", 0.0),
            enemy_missile_near_hit_y_margin=getattr(args, "enemy_missile_near_hit_y_margin", 35.0),
            enemy_missile_near_hit_x_radius=getattr(args, "enemy_missile_near_hit_x_radius", 25.0),
            enemy_disappear_penalty=getattr(args, "enemy_disappear_penalty", 0.0),
            max_steps=args.max_steps,
            hud=args.hud,
            single_agent_mode=single_agent_mode,
            allow_sideward_fire=getattr(args, "allow_sideward_fire", True),
            bidding_mechanism=getattr(args, "bidding_mechanism", "all_pay"),
            only_own_enemy=getattr(args, "only_own_enemy", False),
            obs_stack=getattr(args, "obs_stack", 1),
            building_penalty_mode=getattr(args, "building_penalty_mode", "lane"),
            include_agent_id=getattr(args, "separate_agent_networks", False),
        )
        env = AirRaidEnv(env_config, num_envs=1, device=trainer.device, seed=args.seed, render_mode=render_mode, render_oc_overlay=self.render_oc_overlay)

        returns = []
        scores = []
        lengths = []
        episode_frames: List[List[np.ndarray]] = []
        all_episode_components: List[Dict[str, float]] = []
        all_agent_control_counts: List[List[int]] = []
        all_agent_bid_counts: List[List[Dict[int, int]]] = []
        all_agent_returns: List[List[float]] = []
        all_hit_agent_counts: List[List[int]] = []
        all_missile_owner_counts: List[List[int]] = []
        all_owner_hit_counts: List[List[List[int]]] = []

        for ep_idx in range(self.num_eval_episodes):
            obs, _ = env.reset()
            done = False
            ep_return = 0.0
            ep_len = 0
            last_score = 0.0
            frames: List[np.ndarray] = []
            ep_components: Dict[str, float] = {}
            agent_control_counts = [0] * args.num_agents
            agent_bid_counts: List[Dict[int, int]] = [{} for _ in range(args.num_agents)]
            agent_returns = [0.0] * args.num_agents
            hit_agent_counts = [0] * args.num_agents
            missile_owner_counts = [0] * args.num_agents
            owner_hit_counts = [[0] * args.num_agents for _ in range(args.num_agents)]
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
                if should_record:
                    frame = env.render(env_idx=0, show_agent_overlay=not single_agent_mode)
                    if frame is not None:
                        frames.append(frame)
                if isinstance(info, dict):
                    score_tensor = info.get("score", None)
                    if torch.is_tensor(score_tensor):
                        last_score = float(score_tensor.item())
                    if not single_agent_mode:
                        winning_agent = info.get("winning_agent")
                        if winning_agent is not None:
                            agent_idx = int(winning_agent.item())
                            if 0 <= agent_idx < len(agent_control_counts):
                                agent_control_counts[agent_idx] += 1
                        bids_tensor = info.get("bids")
                        is_bidding_round_info = info.get("is_bidding_round")
                        if bids_tensor is not None and is_bidding_round_info is not None and bool(is_bidding_round_info[0].item()):
                            bids_list = bids_tensor[0].cpu().tolist()
                            for agent_i, bid_val in enumerate(bids_list):
                                bc = agent_bid_counts[agent_i]
                                bc[int(bid_val)] = bc.get(int(bid_val), 0) + 1
                    rc = info.get("reward_components", {})
                    for key, val in rc.items():
                        v = float(val.item()) if torch.is_tensor(val) else float(val)
                        if key.endswith("_current"):
                            ep_components[key] = v
                        else:
                            ep_components[key] = ep_components.get(key, 0.0) + v
                    if rc:
                        score_delta_val = rc.get("score_delta", None)
                        hit_agent_val = rc.get("hit_agent", None)
                        missile_owner_val = rc.get("missile_owner", None)
                        score_delta = float(score_delta_val.item()) if torch.is_tensor(score_delta_val) else float(score_delta_val or 0.0)
                        if score_delta > 0.0:
                            hit_agent = int(hit_agent_val.item()) if torch.is_tensor(hit_agent_val) else int(hit_agent_val)
                            if 0 <= hit_agent < args.num_agents:
                                hit_agent_counts[hit_agent] += 1
                            if not single_agent_mode:
                                missile_owner = int(missile_owner_val.item()) if torch.is_tensor(missile_owner_val) else int(missile_owner_val)
                                if 0 <= missile_owner < args.num_agents:
                                    missile_owner_counts[missile_owner] += 1
                                if 0 <= missile_owner < args.num_agents and 0 <= hit_agent < args.num_agents:
                                    owner_hit_counts[missile_owner][hit_agent] += 1
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
            all_hit_agent_counts.append(hit_agent_counts)
            if not single_agent_mode:
                all_agent_control_counts.append(agent_control_counts)
                all_agent_bid_counts.append([dict(bc) for bc in agent_bid_counts])
                all_agent_returns.append(agent_returns)
                all_missile_owner_counts.append(missile_owner_counts)
                all_owner_hit_counts.append(owner_hit_counts)
            if should_record and frames:
                episode_frames.append(frames)

        env.close()
        if create_videos and episode_frames and iteration is not None:
            self._write_videos(episode_frames, iteration, global_step, trainer.args.track)

        component_keys = all_episode_components[0].keys() if all_episode_components else []
        avg_components = {f"avg_{key}": float(np.mean([ep.get(key, 0.0) for ep in all_episode_components])) for key in component_keys}
        avg_agent_control = {}
        if not single_agent_mode and all_agent_control_counts:
            for i in range(args.num_agents):
                avg_agent_control[f"avg_agent_{i}_control_steps"] = float(np.mean([ep[i] for ep in all_agent_control_counts]))
        avg_bid_counts_per_agent: List[Dict[int, float]] = []
        if not single_agent_mode and all_agent_bid_counts:
            bid_upper_bound = getattr(args, "bid_upper_bound", 0)
            for agent_i in range(args.num_agents):
                avg_bc = {}
                for bid_val in range(bid_upper_bound + 1):
                    avg_bc[bid_val] = float(np.mean([ep[agent_i].get(bid_val, 0) for ep in all_agent_bid_counts]))
                avg_bid_counts_per_agent.append(avg_bc)
        avg_control_timesteps_per_agent: List[float] = []
        if not single_agent_mode and all_agent_control_counts:
            avg_control_timesteps_per_agent = np.array(all_agent_control_counts).mean(axis=0).tolist()
        per_agent_return_stats = {}
        if not single_agent_mode and all_agent_returns:
            for i in range(args.num_agents):
                agent_rets = [ep[i] for ep in all_agent_returns]
                per_agent_return_stats[f"avg_agent_{i}_return"] = float(np.mean(agent_rets))
                per_agent_return_stats[f"std_agent_{i}_return"] = float(np.std(agent_rets))
        hit_credit_stats = {}
        if all_hit_agent_counts:
            hit_credit_stats["avg_hit_agent_counts"] = np.array(all_hit_agent_counts, dtype=float).mean(axis=0).tolist()
            hit_credit_stats["avg_hit_lane_counts"] = hit_credit_stats["avg_hit_agent_counts"]
        if not single_agent_mode and all_missile_owner_counts:
            hit_credit_stats["avg_missile_owner_counts"] = np.array(all_missile_owner_counts, dtype=float).mean(axis=0).tolist()
            hit_credit_stats["avg_owner_hit_counts"] = np.array(all_owner_hit_counts, dtype=float).mean(axis=0).tolist()

        stats = {
            "avg_score": float(np.mean(scores)),
            "avg_length": float(np.mean(lengths)),
            "std_score": float(np.std(scores)),
            "std_length": float(np.std(lengths)),
            "avg_return": float(np.mean(returns)),
            **avg_components,
            **avg_agent_control,
            **per_agent_return_stats,
            **hit_credit_stats,
            "avg_bid_counts_per_agent": avg_bid_counts_per_agent,
            "avg_control_timesteps_per_agent": avg_control_timesteps_per_agent,
        }
        if iteration is not None:
            eval_summary = {
                "iteration": iteration,
                "global_step": global_step,
                "num_episodes": self.num_eval_episodes,
                "timestamp": datetime.now().isoformat(),
                "statistics": stats,
            }
            with open(self.eval_dir / f"iter_{iteration}_eval_stats.json", "w") as f:
                json.dump(eval_summary, f, indent=2)
        return stats

    def _write_videos(self, episode_frames: List[List[np.ndarray]], iteration: int, global_step: int | None, track: bool) -> None:
        for ep_idx, frames in enumerate(episode_frames):
            if not frames:
                continue
            video_path = self.rollouts_dir / f"iter_{iteration}_ep_{ep_idx}.mp4"
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, 30, (w, h))
            for frame in frames:
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            if track and self.log_videos_to_wandb and video_path.exists():
                wandb.log({f"eval/rollout_ep_{ep_idx}": wandb.Video(str(video_path), fps=30, format="mp4")}, step=global_step)

    @staticmethod
    def _find_latest_checkpoint(checkpoints_dir: Path):
        if not checkpoints_dir.exists():
            return None
        iter_dirs = []
        for p in checkpoints_dir.iterdir():
            if p.is_dir() and p.name.startswith("iter_"):
                try:
                    iter_dirs.append((int(p.name.split("_")[1]), p))
                except (IndexError, ValueError):
                    pass
        if not iter_dirs:
            return None
        iter_dirs.sort(key=lambda x: x[0])
        _, latest = iter_dirs[-1]
        info_path = latest / "checkpoint_info.json"
        model_path = latest / "agent.pt"
        if not info_path.exists() or not model_path.exists():
            return None
        with open(info_path) as f:
            info = json.load(f)
        return {"iteration": info["iteration"], "global_step": info["global_step"], "path": model_path}

    def run(self, args):
        if hasattr(args, "num_iterations"):
            if self.single_agent_mode:
                args.total_timesteps = args.num_iterations * args.num_envs * args.num_steps
            else:
                args.total_timesteps = args.num_iterations * args.num_envs * args.num_steps * args.num_agents
        ckpt = self._find_latest_checkpoint(self.checkpoints_dir)
        if ckpt and ckpt["iteration"] >= args.num_iterations:
            return
        start_iteration = 1
        initial_global_step = 0
        resume_model_path = None
        if ckpt:
            start_iteration = ckpt["iteration"] + 1
            initial_global_step = ckpt["global_step"]
            resume_model_path = ckpt["path"]

        self.save_config(args)

        def on_iteration_end(trainer, iteration, global_step):
            if iteration % self.checkpoint_freq == 0:
                self.save_checkpoint(trainer, iteration, global_step)
            if iteration % self.eval_freq == 0:
                create_videos = iteration % self.video_freq == 0
                stats = self._evaluate(trainer, args, self.single_agent_mode, iteration, global_step, create_videos=create_videos)
                if trainer.args.track:
                    wandb.log({f"eval/{k}": v for k, v in stats.items()}, step=global_step)

        def on_training_end(trainer, global_step):
            stats = self._evaluate(trainer, args, self.single_agent_mode, args.num_iterations, global_step, create_videos=True)
            if trainer.args.track:
                wandb.log({f"eval/{k}": v for k, v in stats.items()}, step=global_step)

        callbacks = {"on_iteration_end": on_iteration_end, "on_training_end": on_training_end}
        trainer = AirRaidSingleAgentPPOTrainer(args, callbacks=callbacks) if self.single_agent_mode else AirRaidPPOTrainer(args, callbacks=callbacks)
        trainer.setup()
        if resume_model_path is not None:
            trainer.agent.load_state_dict(torch.load(resume_model_path, map_location=trainer.device, weights_only=True))
        self.log_codebase_to_wandb(wandb.run)
        trainer.train(start_iteration=start_iteration, initial_global_step=initial_global_step)
        model_filename = "airraid_single_agent.pt" if self.single_agent_mode else "airraid_agent.pt"
        trainer.save_model(str(self.log_dir / model_filename))
        trainer.cleanup()
