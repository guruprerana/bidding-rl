"""Experiment orchestration for PPO training with evaluations and checkpoints."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb

from bidding_gridworld.bidding_gridworld_torch import (
    BiddingGridworld,
    BiddingGridworldConfig,
    evaluate_multi_agent_policy,
    evaluate_multi_agent_policy_batched,
    evaluate_single_agent_policy,
    evaluate_single_agent_policy_batched,
)
from bidding_gridworld.bidding_ppo import PPOTrainer
from bidding_gridworld.single_agent_ppo import SingleAgentPPOTrainer


@contextmanager
def isolated_torch_rng(seed: int):
    """Use a reproducible policy-sampling stream without perturbing training."""
    cpu_state = torch.get_rng_state()
    cuda_initialized = torch.cuda.is_available() and torch.cuda.is_initialized()
    cuda_states = torch.cuda.get_rng_state_all() if cuda_initialized else None
    torch.manual_seed(seed)
    if cuda_initialized:
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


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
        deterministic_eval: bool = False,
        eval_seed: int | None = None,
        policy_sample_seed: int | None = None,
        final_model_in_log_dir: bool = False,
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
            deterministic_eval: Use highest-logit actions instead of sampling
                the trained policy during evaluation.
            eval_seed: Optional held-out environment seed. Defaults to the
                training seed for backward compatibility.
            policy_sample_seed: Optional isolated action-sampling seed.
                Defaults to the evaluation seed.
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
        self.deterministic_eval = deterministic_eval
        self.eval_seed = eval_seed
        self.policy_sample_seed = policy_sample_seed
        self.final_model_in_log_dir = final_model_in_log_dir

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
        project_root = Path(__file__).resolve().parents[2]

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
        if getattr(trainer, "charging_agent", None) is not None:
            charging_model_path = checkpoint_dir / "charging_agent.pt"
            torch.save(
                trainer.charging_agent.state_dict(),
                str(charging_model_path),
            )

        # Save checkpoint info
        checkpoint_info = {
            "iteration": iteration,
            "global_step": global_step,
            "timestamp": datetime.now().isoformat(),
            "feeder_warm_start": getattr(
                trainer, "feeder_warm_start_report", None
            ),
            "feeder_freeze_iterations": getattr(
                trainer.args, "feeder_freeze_iterations", 0
            ),
            "feeder_finetune_learning_rate": getattr(
                trainer.args, "feeder_finetune_learning_rate", None
            ),
            "charging_behavior_cloning": getattr(
                trainer, "charging_bc_report", None
            ),
            "charging_bc_refresh": getattr(
                trainer, "charging_bc_refresh_report", None
            ),
        }
        info_path = checkpoint_dir / "checkpoint_info.json"
        with open(info_path, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)

        print(f"💾 Checkpoint saved: {checkpoint_dir}")

        # Save to wandb
        if trainer.args.track:
            wandb.save(str(model_path))
            if getattr(trainer, "charging_agent", None) is not None:
                wandb.save(str(charging_model_path))

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
            bidding_mechanism=trainer.args.bidding_mechanism,
            use_target_priorities=trainer.args.use_target_priorities,
            programmatic_bidding=trainer.args.programmatic_bidding,
            battery_capacity=trainer.args.battery_capacity,
            recharge_station_positions=trainer.args.recharge_station_positions,
            moving_recharge_stations=getattr(
                trainer.args, "moving_recharge_stations", False
            ),
            recharge_station_direction_change_prob=getattr(
                trainer.args,
                "recharge_station_direction_change_prob",
                0.1,
            ),
            recharge_station_move_interval=getattr(
                trainer.args, "recharge_station_move_interval", 5
            ),
            movement_energy_cost=trainer.args.movement_energy_cost,
            battery_depletion_penalty=trainer.args.battery_depletion_penalty,
            charging_agent_enabled=trainer.args.charging_agent_enabled,
            charging_low_battery_threshold=trainer.args.charging_low_battery_threshold,
            charging_distance_reward_scale=trainer.args.charging_distance_reward_scale,
            charging_recharge_bonus=trainer.args.charging_recharge_bonus,
            charging_depletion_penalty=trainer.args.charging_depletion_penalty,
            charging_high_battery_control_penalty=(
                trainer.args.charging_high_battery_control_penalty
            ),
            feeder_low_battery_control_penalty=(
                trainer.args.feeder_low_battery_control_penalty
            ),
            charging_low_battery_bid_boost=(
                trainer.args.charging_low_battery_bid_boost
            ),
            charging_bid_boost_threshold=getattr(
                trainer.args, "charging_bid_boost_threshold", None
            ),
            charging_activation_margin=trainer.args.charging_activation_margin,
            charging_release_window_on_recharge=(
                trainer.args.charging_release_window_on_recharge
            ),
            charging_programmatic_navigation=(
                trainer.args.charging_programmatic_navigation
            ),
            charging_reserve_features_enabled=getattr(
                trainer.args, "charging_reserve_features_enabled", False
            ),
            charging_nearest_station_features_enabled=getattr(
                trainer.args,
                "charging_nearest_station_features_enabled",
                False,
            ),
        )
        evaluation_seed = (
            trainer.args.seed if self.eval_seed is None else self.eval_seed
        )
        policy_sample_seed = (
            evaluation_seed
            if self.policy_sample_seed is None
            else self.policy_sample_seed
        )
        eval_env = BiddingGridworld(
            env_config,
            num_envs=self.num_eval_episodes,
            device=trainer.device,
            seed=evaluation_seed,
        )

        # Create batched policy wrapper function
        # obs shape: (N, num_agents, obs_dim) — reshape for shared network, then reshape back
        def policy_fn(obs):
            """Get actions for all envs in a batched call."""
            obs_tensor = obs if torch.is_tensor(obs) else torch.tensor(obs, dtype=torch.float32)
            obs_tensor = obs_tensor.to(trainer.device)
            N_envs, n_agents, obs_dim = obs_tensor.shape
            obs_flat = obs_tensor.reshape(N_envs * n_agents, obs_dim)
            with torch.no_grad():
                if trainer.args.programmatic_bidding != "none":
                    action_fn = trainer.agent.get_direction_action_and_value
                elif trainer.args.bid_only_ppo:
                    action_fn = (
                        trainer.agent.get_bid_action_and_value_with_direction
                    )
                else:
                    action_fn = trainer.agent.get_action_and_value
                action, _, _, _ = action_fn(
                    obs_flat, deterministic=self.deterministic_eval
                )
                feeder_action = action.reshape(N_envs, n_agents, -1)
                if trainer.charging_agent is None:
                    return feeder_action
                charging_obs = eval_env.get_charging_observation()
                charging_action_fn = (
                    trainer.charging_agent.get_bid_action_and_value
                    if trainer.args.charging_programmatic_navigation
                    else trainer.charging_agent.get_action_and_value
                )
                charging_action, _, _, _ = charging_action_fn(
                    charging_obs,
                    deterministic=self.deterministic_eval,
                    **(
                        {
                            "deterministic_direction": True,
                        }
                        if (
                            not trainer.args.charging_programmatic_navigation
                            and getattr(
                                trainer.args,
                                "charging_greedy_navigation_eval",
                                False,
                            )
                        )
                        else {}
                    ),
                )
                return torch.cat(
                    [feeder_action, charging_action.unsqueeze(1)], dim=1
                )

        # Run batched evaluation
        with isolated_torch_rng(policy_sample_seed):
            eval_stats = evaluate_multi_agent_policy_batched(
                env=eval_env,
                policy_fn=policy_fn,
                num_episodes=self.num_eval_episodes,
                target_expiry_penalty=trainer.args.target_expiry_penalty,
                verbose=True,
                capture_episode_count=(
                    min(self.num_video_episodes, self.num_eval_episodes)
                    if create_videos
                    else 0
                ),
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
        avg_return_no_bid = np.mean(eval_stats["episode_returns_no_bid"]) if eval_stats.get("episode_returns_no_bid") else float("nan")
        avg_length = np.mean(eval_stats["episode_lengths"])
        avg_targets = np.mean(eval_stats["targets_reached_per_episode"])
        avg_priority_sum = np.mean(eval_stats["reached_priority_sum_per_episode"])
        avg_reached_count_by_priority = np.mean(
            eval_stats["reached_count_by_priority_per_episode"], axis=0
        ).tolist()
        avg_expired = np.mean(eval_stats["expired_targets_per_episode"])
        avg_min_reached = np.mean(eval_stats["min_targets_reached_per_episode"])
        success_rate = sum(1 for t in eval_stats["targets_reached_per_episode"]
                           if t == eval_num_agents) / self.num_eval_episodes

        # Average bid counts across episodes
        all_bid_counts = eval_stats.get("bid_counts_per_episode", [])
        bid_upper_bound = trainer.args.bid_upper_bound
        avg_bid_counts = {}
        for bid_val in range(bid_upper_bound + 1):
            avg_bid_counts[bid_val] = float(np.mean([bc.get(bid_val, 0) for bc in all_bid_counts]))
        all_charging_bid_counts = eval_stats.get(
            "charging_bid_counts_per_episode", []
        )
        avg_charging_bid_counts = {}
        for bid_val in range(bid_upper_bound + 1):
            avg_charging_bid_counts[bid_val] = float(
                np.mean(
                    [
                        bc.get(bid_val, 0)
                        for bc in all_charging_bid_counts
                    ]
                )
            )

        # Average control timesteps per agent across episodes
        all_control_steps = eval_stats.get("control_steps_per_agent_per_episode", [])
        avg_control_steps_per_agent = (
            np.array(all_control_steps).mean(axis=0).tolist()
            if all_control_steps else []
        )
        all_depletions_per_agent = eval_stats.get(
            "battery_depletions_per_agent_per_episode", []
        )
        avg_depletions_per_agent = (
            np.array(all_depletions_per_agent).mean(axis=0).tolist()
            if all_depletions_per_agent
            else []
        )
        avg_battery_depletions = float(
            np.mean(eval_stats["battery_depletions_per_episode"])
        )
        avg_battery_recharges = float(
            np.mean(eval_stats["battery_recharges_per_episode"])
        )
        charging_navigation_steps = np.asarray(
            eval_stats["charging_navigation_steps_per_episode"],
            dtype=float,
        )
        charging_optimal_direction_steps = np.asarray(
            eval_stats["charging_optimal_direction_steps_per_episode"],
            dtype=float,
        )
        charging_optimal_direction_rate = float(
            np.mean(
                np.divide(
                    charging_optimal_direction_steps,
                    charging_navigation_steps,
                    out=np.zeros_like(charging_navigation_steps),
                    where=charging_navigation_steps > 0,
                )
            )
        )
        charging_activation_steps = float(
            np.mean(eval_stats["charging_activation_steps_per_episode"])
        )
        charging_activation_fraction = float(
            np.mean(
                np.divide(
                    np.asarray(
                        eval_stats["charging_activation_steps_per_episode"],
                        dtype=float,
                    ),
                    np.asarray(eval_stats["episode_lengths"], dtype=float),
                    out=np.zeros(self.num_eval_episodes, dtype=float),
                    where=np.asarray(eval_stats["episode_lengths"]) > 0,
                )
            )
        )
        active_auctions = np.asarray(
            eval_stats["charging_active_auction_steps_per_episode"],
            dtype=float,
        )
        active_auction_wins = np.asarray(
            eval_stats["charging_active_auction_wins_per_episode"],
            dtype=float,
        )
        active_feeder_max_bid_sum = np.asarray(
            eval_stats[
                "charging_active_feeder_max_bid_sum_per_episode"
            ],
            dtype=float,
        )
        active_feeder_tie_or_outbid = np.asarray(
            eval_stats[
                "charging_active_feeder_tie_or_outbid_steps_per_episode"
            ],
            dtype=float,
        )
        charging_active_auction_win_rate = float(
            np.mean(
                np.divide(
                    active_auction_wins,
                    active_auctions,
                    out=np.zeros_like(active_auctions),
                    where=active_auctions > 0,
                )
            )
        )
        charging_active_avg_feeder_max_bid = float(
            np.mean(
                np.divide(
                    active_feeder_max_bid_sum,
                    active_auctions,
                    out=np.zeros_like(active_auctions),
                    where=active_auctions > 0,
                )
            )
        )
        charging_active_feeder_tie_or_outbid_rate = float(
            np.mean(
                np.divide(
                    active_feeder_tie_or_outbid,
                    active_auctions,
                    out=np.zeros_like(active_auctions),
                    where=active_auctions > 0,
                )
            )
        )
        charging_control_steps = 0.0
        charging_control_fraction = 0.0
        if trainer.args.charging_agent_enabled and all_control_steps:
            control_arr = np.asarray(all_control_steps, dtype=float)
            charging_control_steps = float(control_arr[:, -1].mean())
            total_control = control_arr.sum(axis=1)
            charging_control_fraction = float(
                np.mean(
                    np.divide(
                        control_arr[:, -1],
                        total_control,
                        out=np.zeros_like(total_control),
                        where=total_control > 0,
                    )
                )
            )

        # Log to wandb
        if trainer.args.track:
            wandb.log({
                "eval/avg_return": avg_return,
                "eval/avg_return_no_bid": avg_return_no_bid,
                "eval/avg_length": avg_length,
                "eval/avg_targets_reached": avg_targets,
                "eval/avg_reached_priority_sum": avg_priority_sum,
                "eval/avg_expired_targets": avg_expired,
                "eval/avg_min_targets_reached": avg_min_reached,
                "eval/success_rate": success_rate,
                "eval/avg_battery_depletions": avg_battery_depletions,
                "eval/avg_battery_recharges": avg_battery_recharges,
                "eval/charging_optimal_direction_rate": (
                    charging_optimal_direction_rate
                ),
                "eval/charging_activation_steps": charging_activation_steps,
                "eval/charging_activation_fraction": (
                    charging_activation_fraction
                ),
                "eval/charging_active_auction_win_rate": (
                    charging_active_auction_win_rate
                ),
                "eval/charging_active_avg_feeder_max_bid": (
                    charging_active_avg_feeder_max_bid
                ),
                "eval/charging_active_feeder_tie_or_outbid_rate": (
                    charging_active_feeder_tie_or_outbid_rate
                ),
                "eval/charging_control_steps": charging_control_steps,
                "eval/charging_control_fraction": charging_control_fraction,
            }, step=global_step)

        # Save eval stats to local JSON file
        reached_arr = np.array(eval_stats["targets_reached_count_per_episode"], dtype=float)
        expired_arr = np.array(eval_stats["expired_count_per_target_per_episode"], dtype=float)
        perf_arr    = reached_arr - expired_arr
        eval_summary = {
            "iteration": iteration,
            "global_step": global_step,
            "policy_action_selection": (
                "logit_argmax" if self.deterministic_eval else "sampled"
            ),
            "charging_navigation_action_selection": (
                "programmatic"
                if trainer.args.charging_programmatic_navigation
                else (
                    "logit_argmax"
                    if getattr(
                        trainer.args,
                        "charging_greedy_navigation_eval",
                        False,
                    )
                    else (
                        "logit_argmax"
                        if self.deterministic_eval
                        else "sampled"
                    )
                )
            ),
            "environment_seed": evaluation_seed,
            "policy_sample_seed": policy_sample_seed,
            "num_episodes": self.num_eval_episodes,
            "num_agents": eval_num_agents,
            "train_num_agents": trainer.args.num_agents,
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "avg_return": float(avg_return),
                "avg_return_no_bid": float(avg_return_no_bid),
                "avg_length": float(avg_length),
                "avg_targets_reached": float(avg_targets),
                "avg_reached_priority_sum": float(avg_priority_sum),
                "avg_reached_count_by_priority": avg_reached_count_by_priority,
                "avg_battery_depletions": avg_battery_depletions,
                "avg_battery_recharges": avg_battery_recharges,
                "avg_charging_optimal_direction_rate": (
                    charging_optimal_direction_rate
                ),
                "avg_charging_activation_steps": charging_activation_steps,
                "avg_charging_activation_fraction": (
                    charging_activation_fraction
                ),
                "avg_charging_active_auction_win_rate": (
                    charging_active_auction_win_rate
                ),
                "avg_charging_active_feeder_max_bid": (
                    charging_active_avg_feeder_max_bid
                ),
                "avg_charging_active_feeder_tie_or_outbid_rate": (
                    charging_active_feeder_tie_or_outbid_rate
                ),
                "avg_charging_control_steps": charging_control_steps,
                "avg_charging_control_fraction": charging_control_fraction,
                "avg_expired_targets": float(avg_expired),
                "avg_min_targets_reached": float(avg_min_reached),
                "success_rate": float(success_rate),
                "std_return": float(np.std(eval_stats["episode_returns"])),
                "std_length": float(np.std(eval_stats["episode_lengths"])),
                "std_targets_reached": float(np.std(eval_stats["targets_reached_per_episode"])),
                "avg_avg_performance": float(np.mean(eval_stats["avg_performance_per_episode"])),
                "avg_min_performance": float(np.mean(eval_stats["min_performance_per_episode"])),
                "avg_reached_per_target": reached_arr.mean(axis=0).tolist(),
                "avg_expired_per_target": expired_arr.mean(axis=0).tolist(),
                "avg_performance_per_target": perf_arr.mean(axis=0).tolist(),
                "avg_bid_counts": avg_bid_counts,
                "avg_charging_bid_counts": avg_charging_bid_counts,
                "avg_control_timesteps_per_agent": avg_control_steps_per_agent,
                "avg_battery_depletions_per_agent": (
                    avg_depletions_per_agent
                ),
            },
            "per_episode_data": {
                "returns": [float(r) for r in eval_stats["episode_returns"]],
                "returns_no_bid": [float(r) for r in eval_stats.get("episode_returns_no_bid", [])],
                "lengths": [int(l) for l in eval_stats["episode_lengths"]],
                "targets_reached": [int(t) for t in eval_stats["targets_reached_per_episode"]],
                "reached_priority_sum": [
                    int(p) for p in eval_stats["reached_priority_sum_per_episode"]
                ],
                "reached_priority_sum_per_target": eval_stats[
                    "reached_priority_sum_per_target_per_episode"
                ],
                "reached_count_by_priority": eval_stats[
                    "reached_count_by_priority_per_episode"
                ],
                "battery_depletions": eval_stats["battery_depletions_per_episode"],
                "battery_depletions_per_agent": eval_stats.get(
                    "battery_depletions_per_agent_per_episode", []
                ),
                "battery_recharges": eval_stats["battery_recharges_per_episode"],
                "charging_navigation_steps": eval_stats[
                    "charging_navigation_steps_per_episode"
                ],
                "charging_optimal_direction_steps": eval_stats[
                    "charging_optimal_direction_steps_per_episode"
                ],
                "charging_activation_steps": (
                    eval_stats["charging_activation_steps_per_episode"]
                ),
                "charging_active_auction_steps": eval_stats[
                    "charging_active_auction_steps_per_episode"
                ],
                "charging_active_auction_wins": eval_stats[
                    "charging_active_auction_wins_per_episode"
                ],
                "charging_active_feeder_max_bid_sum": eval_stats[
                    "charging_active_feeder_max_bid_sum_per_episode"
                ],
                "charging_active_feeder_tie_or_outbid_steps": eval_stats[
                    "charging_active_feeder_tie_or_outbid_steps_per_episode"
                ],
                "expired_targets": [int(e) for e in eval_stats["expired_targets_per_episode"]],
                "min_targets_reached": [int(m) for m in eval_stats["min_targets_reached_per_episode"]],
                "avg_performance": [float(p) for p in eval_stats["avg_performance_per_episode"]],
                "min_performance": [float(p) for p in eval_stats["min_performance_per_episode"]],
                "expired_count_per_target": eval_stats["expired_count_per_target_per_episode"],
                "targets_reached_count": eval_stats["targets_reached_count_per_episode"],
                "bid_counts": [dict(sorted(bc.items())) for bc in all_bid_counts],
                "charging_bid_counts": [
                    dict(sorted(bc.items()))
                    for bc in all_charging_bid_counts
                ],
                "control_steps_per_agent": all_control_steps,
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
            urgency_weighted_scalarization=(
                trainer.args.urgency_weighted_scalarization
            ),
            use_target_priorities=trainer.args.use_target_priorities,
            battery_capacity=trainer.args.battery_capacity,
            recharge_station_positions=trainer.args.recharge_station_positions,
            moving_recharge_stations=getattr(
                trainer.args, "moving_recharge_stations", False
            ),
            recharge_station_direction_change_prob=getattr(
                trainer.args,
                "recharge_station_direction_change_prob",
                0.1,
            ),
            recharge_station_move_interval=getattr(
                trainer.args, "recharge_station_move_interval", 5
            ),
            movement_energy_cost=trainer.args.movement_energy_cost,
            battery_depletion_penalty=trainer.args.battery_depletion_penalty,
        )
        evaluation_seed = (
            trainer.args.seed if self.eval_seed is None else self.eval_seed
        )
        policy_sample_seed = (
            evaluation_seed
            if self.policy_sample_seed is None
            else self.policy_sample_seed
        )
        eval_env = BiddingGridworld(
            env_config,
            num_envs=self.num_eval_episodes,
            device=trainer.device,
            seed=evaluation_seed,
        )

        # Create batched policy wrapper function
        # obs shape: (N, obs_dim) — network already handles batch dimension
        def policy_fn(obs):
            """Get actions for all envs in a batched call."""
            obs_tensor = obs if torch.is_tensor(obs) else torch.tensor(obs, dtype=torch.float32)
            obs_tensor = obs_tensor.to(trainer.device)
            with torch.no_grad():
                action, _, _, _ = trainer.agent.get_action_and_value(
                    obs_tensor, deterministic=self.deterministic_eval
                )
            return action

        # Run batched evaluation
        with isolated_torch_rng(policy_sample_seed):
            eval_stats = evaluate_single_agent_policy_batched(
                env=eval_env,
                policy_fn=policy_fn,
                num_episodes=self.num_eval_episodes,
                target_expiry_penalty=trainer.args.target_expiry_penalty,
                verbose=True,
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
        avg_priority_sum = np.mean(eval_stats["reached_priority_sum_per_episode"])
        avg_reached_count_by_priority = np.mean(
            eval_stats["reached_count_by_priority_per_episode"], axis=0
        ).tolist()
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
                "eval/avg_reached_priority_sum": avg_priority_sum,
                "eval/avg_expired_targets": avg_expired,
                "eval/avg_min_targets_reached": avg_min_reached,
                "eval/success_rate": success_rate,
            }, step=global_step)

        # Save eval stats to local JSON file
        reached_arr = np.array(eval_stats["targets_reached_count_per_episode"], dtype=float)
        expired_arr = np.array(eval_stats["expired_count_per_target_per_episode"], dtype=float)
        perf_arr    = reached_arr - expired_arr
        total_reaches = reached_arr.sum(axis=1)
        total_expiries = expired_arr.sum(axis=1)
        total_performance = total_reaches - total_expiries
        eval_summary = {
            "iteration": iteration,
            "global_step": global_step,
            "policy_action_selection": (
                "logit_argmax" if self.deterministic_eval else "sampled"
            ),
            "environment_seed": evaluation_seed,
            "policy_sample_seed": policy_sample_seed,
            "num_episodes": self.num_eval_episodes,
            "num_targets": eval_num_targets,
            "train_num_targets": trainer.args.num_targets,
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "avg_return": float(avg_return),
                "avg_length": float(avg_length),
                "avg_targets_reached": float(avg_targets),
                "avg_reached_priority_sum": float(avg_priority_sum),
                "avg_reached_count_by_priority": avg_reached_count_by_priority,
                "avg_battery_depletions": float(
                    np.mean(eval_stats["battery_depletions_per_episode"])
                ),
                "avg_battery_recharges": float(
                    np.mean(eval_stats["battery_recharges_per_episode"])
                ),
                "avg_expired_targets": float(avg_expired),
                "avg_min_targets_reached": float(avg_min_reached),
                "success_rate": float(success_rate),
                "std_return": float(np.std(eval_stats["episode_returns"])),
                "std_length": float(np.std(eval_stats["episode_lengths"])),
                "std_targets_reached": float(np.std(eval_stats["targets_reached_per_episode"])),
                "avg_total_reaches": float(np.mean(total_reaches)),
                "std_total_reaches": float(np.std(total_reaches)),
                "avg_total_expiries": float(np.mean(total_expiries)),
                "std_total_expiries": float(np.std(total_expiries)),
                "avg_total_performance": float(np.mean(total_performance)),
                "std_total_performance": float(np.std(total_performance)),
                "avg_avg_performance": float(np.mean(eval_stats["avg_performance_per_episode"])),
                "avg_min_performance": float(np.mean(eval_stats["min_performance_per_episode"])),
                "avg_reached_per_target": reached_arr.mean(axis=0).tolist(),
                "avg_expired_per_target": expired_arr.mean(axis=0).tolist(),
                "avg_performance_per_target": perf_arr.mean(axis=0).tolist(),
            },
            "per_episode_data": {
                "returns": [float(r) for r in eval_stats["episode_returns"]],
                "lengths": [int(l) for l in eval_stats["episode_lengths"]],
                "targets_reached": [int(t) for t in eval_stats["targets_reached_per_episode"]],
                "reached_priority_sum": [
                    int(p) for p in eval_stats["reached_priority_sum_per_episode"]
                ],
                "reached_priority_sum_per_target": eval_stats[
                    "reached_priority_sum_per_target_per_episode"
                ],
                "reached_count_by_priority": eval_stats[
                    "reached_count_by_priority_per_episode"
                ],
                "battery_depletions": eval_stats["battery_depletions_per_episode"],
                "battery_recharges": eval_stats["battery_recharges_per_episode"],
                "expired_targets": [int(e) for e in eval_stats["expired_targets_per_episode"]],
                "min_targets_reached": [int(m) for m in eval_stats["min_targets_reached_per_episode"]],
                "avg_performance": [float(p) for p in eval_stats["avg_performance_per_episode"]],
                "min_performance": [float(p) for p in eval_stats["min_performance_per_episode"]],
                "expired_count_per_target": eval_stats["expired_count_per_target_per_episode"],
                "targets_reached_count": eval_stats["targets_reached_count_per_episode"],
                "total_reaches": total_reaches.astype(int).tolist(),
                "total_expiries": total_expiries.astype(int).tolist(),
                "total_performance": total_performance.astype(int).tolist(),
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
        trainer.save_model(
            str(self.log_dir / "models")
            if self.final_model_in_log_dir
            else None
        )
        trainer.cleanup()

        print(f"\n✅ Training complete! Results saved to {self.log_dir}")
