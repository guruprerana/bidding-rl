#!/usr/bin/env python3
"""Evaluate a saved Air Raid PPO checkpoint for a fixed number of episodes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from airraid.airraid_bidding_ppo import AirRaidArgs, AirRaidPPOTrainer
from airraid.airraid_experiment import AirRaidExperiment
from airraid.airraid_torch import AirRaidConfig, AirRaidEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint-iter", type=int, required=True)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-num-envs", type=int, default=32)
    return parser.parse_args()


def evaluate_batched(trainer: AirRaidPPOTrainer, args: AirRaidArgs, num_episodes: int, num_envs: int) -> dict:
    env_config = AirRaidConfig(
        num_agents=args.num_agents,
        max_enemies=args.max_enemies,
        bid_upper_bound=args.bid_upper_bound,
        bid_penalty=args.bid_penalty,
        action_window=args.action_window,
        window_bidding=args.window_bidding,
        window_penalty=args.window_penalty,
        enemy_destroy_reward=args.enemy_destroy_reward,
        building_hit_penalty=args.building_hit_penalty,
        life_loss_penalty=args.life_loss_penalty,
        raw_score_scale=args.raw_score_scale,
        enemy_missile_danger_penalty=getattr(args, "enemy_missile_danger_penalty", 0.0),
        enemy_missile_danger_y_threshold=getattr(args, "enemy_missile_danger_y_threshold", 110.0),
        enemy_missile_danger_x_radius=getattr(args, "enemy_missile_danger_x_radius", 18.0),
        enemy_missile_danger_y_radius=getattr(args, "enemy_missile_danger_y_radius", 80.0),
        enemy_missile_near_hit_penalty=getattr(args, "enemy_missile_near_hit_penalty", 0.0),
        enemy_missile_near_hit_y_margin=getattr(args, "enemy_missile_near_hit_y_margin", 35.0),
        enemy_missile_near_hit_x_radius=getattr(args, "enemy_missile_near_hit_x_radius", 25.0),
        max_steps=args.max_steps,
        hud=args.hud,
        single_agent_mode=False,
        allow_sideward_fire=args.allow_sideward_fire,
        bidding_mechanism=args.bidding_mechanism,
        only_own_enemy=args.only_own_enemy,
        obs_stack=args.obs_stack,
        building_penalty_mode=args.building_penalty_mode,
        include_agent_id=getattr(args, "separate_agent_networks", False),
    )
    env = AirRaidEnv(env_config, num_envs=num_envs, device=trainer.device, seed=args.seed)
    obs, _ = env.reset()

    ep_returns = torch.zeros(num_envs, device=trainer.device)
    ep_lengths = torch.zeros(num_envs, dtype=torch.int32, device=trainer.device)
    ep_scores = torch.zeros(num_envs, device=trainer.device)
    ep_components = [dict() for _ in range(num_envs)]

    returns = []
    lengths = []
    scores = []
    components = []

    while len(returns) < num_episodes:
        with torch.no_grad():
            flat_obs = obs.reshape(-1, trainer.obs_dim)
            action, _, _, _ = trainer.agent.get_action_and_value(flat_obs)
            action = action.reshape(num_envs, args.num_agents, -1)
        obs, reward, terminated, truncated, info = env.step(action)

        ep_returns += reward.sum(dim=1)
        ep_lengths += 1
        if torch.is_tensor(info.get("score")):
            ep_scores = info["score"].to(trainer.device)

        rc = info.get("reward_components", {})
        for key, values in rc.items():
            values = values.detach().cpu()
            for env_idx in range(num_envs):
                v = float(values[env_idx].item())
                if key.endswith("_current"):
                    ep_components[env_idx][key] = v
                else:
                    ep_components[env_idx][key] = ep_components[env_idx].get(key, 0.0) + v

        done = terminated | truncated
        done_indices = torch.where(done)[0].detach().cpu().tolist()
        for env_idx in done_indices:
            returns.append(float(ep_returns[env_idx].item()))
            lengths.append(int(ep_lengths[env_idx].item()))
            scores.append(float(ep_scores[env_idx].item()))
            components.append(dict(ep_components[env_idx]))
            ep_returns[env_idx] = 0.0
            ep_lengths[env_idx] = 0
            ep_scores[env_idx] = 0.0
            ep_components[env_idx] = {}
            if len(returns) >= num_episodes:
                break

    env.close()

    returns = returns[:num_episodes]
    lengths = lengths[:num_episodes]
    scores = scores[:num_episodes]
    components = components[:num_episodes]
    component_keys = sorted({key for ep in components for key in ep.keys()})
    avg_components = {
        f"avg_{key}": sum(ep.get(key, 0.0) for ep in components) / len(components)
        for key in component_keys
    }
    return {
        "avg_score": sum(scores) / len(scores),
        "avg_length": sum(lengths) / len(lengths),
        "std_score": float(torch.tensor(scores, dtype=torch.float32).std(unbiased=False).item()),
        "std_length": float(torch.tensor(lengths, dtype=torch.float32).std(unbiased=False).item()),
        "avg_return": sum(returns) / len(returns),
        **avg_components,
    }


def main() -> None:
    cli = parse_args()
    run_dir = Path(cli.run_dir)
    config_path = run_dir / "config" / "training_config.json"
    checkpoint_path = run_dir / "checkpoints" / f"iter_{cli.checkpoint_iter}" / "agent.pt"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    with open(config_path) as f:
        config = json.load(f)
    config["track"] = False
    if cli.seed is not None:
        config["seed"] = cli.seed
    args = AirRaidArgs(**{k: v for k, v in config.items() if k in AirRaidArgs.__dataclass_fields__})

    trainer = AirRaidPPOTrainer(args)
    trainer.setup()
    trainer.agent.load_state_dict(torch.load(checkpoint_path, map_location=trainer.device, weights_only=True))
    trainer.agent.eval()

    if cli.eval_num_envs > 1:
        stats = evaluate_batched(trainer, args, cli.num_episodes, cli.eval_num_envs)
    else:
        experiment = AirRaidExperiment(
            base_log_dir=str(run_dir.parent),
            experiment_name=args.exp_name,
            num_eval_episodes=cli.num_episodes,
            num_video_episodes=0,
            log_videos_to_wandb=False,
            single_agent_mode=False,
        )
        stats = experiment._evaluate(
            trainer,
            args,
            single_agent_mode=False,
            iteration=None,
            global_step=None,
            create_videos=False,
        )
    trainer.cleanup()

    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    output_name = cli.output_name or f"checkpoint_iter_{cli.checkpoint_iter}_eval_{cli.num_episodes}eps.json"
    output_path = eval_dir / output_name
    with open(output_path, "w") as f:
        json.dump(
            {
                "checkpoint_iter": cli.checkpoint_iter,
                "num_episodes": cli.num_episodes,
                "timestamp": datetime.now().isoformat(),
                "checkpoint_path": str(checkpoint_path),
                "statistics": stats,
            },
            f,
            indent=2,
        )
    print(output_path)


if __name__ == "__main__":
    main()
