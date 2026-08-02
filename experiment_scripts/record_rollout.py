#!/usr/bin/env python3
"""Search for and record a priority- and charging-aware rollout video."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from bidding_gridworld.bidding_gridworld_torch import (
    BiddingGridworld,
    evaluate_multi_agent_policy_batched,
)
from bidding_gridworld.experiment import isolated_torch_rng
from experiment_scripts.evaluate_gridworld_charging_checkpoint import (
    build_trainer,
    latest_iteration,
)


DEFAULT_EXPERIMENT = REPO_ROOT / (
    "logs/cat_feeder_complexity_experiments/"
    "moving_station_learned_nav_separate_bcap50_interval15_"
    "s7777_20260726_041414"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_EXPERIMENT,
    )
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--environment-seed", type=int, default=12001)
    parser.add_argument("--policy-seed", type=int, default=12001)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    for name in ("candidates", "max_steps", "fps", "frame_stride"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def candidate_record(stats: dict, index: int) -> dict:
    navigation_steps = stats["charging_navigation_steps_per_episode"][index]
    optimal_steps = stats[
        "charging_optimal_direction_steps_per_episode"
    ][index]
    navigation_rate = (
        float(optimal_steps) / float(navigation_steps)
        if navigation_steps > 0
        else 0.0
    )
    control_steps = stats["control_steps_per_agent_per_episode"][index]
    episode_length = stats["episode_lengths"][index]
    return {
        "candidate_index": index,
        "priority_sum": stats["reached_priority_sum_per_episode"][index],
        "battery_depletions": stats["battery_depletions_per_episode"][index],
        "battery_recharges": stats["battery_recharges_per_episode"][index],
        "expired_targets": stats["expired_targets_per_episode"][index],
        "minimum_feeds_per_target": stats[
            "min_targets_reached_per_episode"
        ][index],
        "reached_count_by_priority": stats[
            "reached_count_by_priority_per_episode"
        ][index],
        "charging_navigation_steps": navigation_steps,
        "charging_optimal_direction_rate": navigation_rate,
        "charging_control_fraction": (
            float(control_steps[-1]) / float(episode_length)
            if control_steps and episode_length > 0
            else 0.0
        ),
        "episode_length": episode_length,
    }


def selection_key(record: dict) -> tuple:
    """Prefer safety, correct navigation, priority, and balanced feeding."""
    return (
        record["battery_depletions"] == 0,
        record["charging_optimal_direction_rate"],
        record["priority_sum"],
        record["minimum_feeds_per_target"],
        -record["expired_targets"],
        record["battery_recharges"],
    )


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    config_path = experiment_dir / "config" / "training_config.json"
    with config_path.open() as source:
        config = json.load(source)

    iteration = args.iteration or latest_iteration(experiment_dir)
    checkpoint_dir = experiment_dir / "checkpoints" / f"iter_{iteration}"
    trainer = build_trainer(config, args.device)
    trainer.agent.load_state_dict(
        torch.load(
            checkpoint_dir / "agent.pt",
            map_location=trainer.device,
            weights_only=True,
        )
    )
    trainer.agent.eval()
    if trainer.charging_agent is None:
        raise RuntimeError("the selected checkpoint has no charging policy")
    trainer.charging_agent.load_state_dict(
        torch.load(
            checkpoint_dir / "charging_agent.pt",
            map_location=trainer.device,
            weights_only=True,
        )
    )
    trainer.charging_agent.eval()

    env_config = replace(trainer.envs.config, max_steps=args.max_steps)
    trainer.envs.close()
    env = BiddingGridworld(
        env_config,
        num_envs=args.candidates,
        device=trainer.device,
        seed=args.environment_seed,
    )

    def policy_fn(obs):
        num_envs, num_agents, obs_dim = obs.shape
        with torch.no_grad():
            feeder_actions, _, _, _ = trainer.agent.get_action_and_value(
                obs.reshape(num_envs * num_agents, obs_dim),
                deterministic=False,
            )
            feeder_actions = feeder_actions.reshape(
                num_envs, num_agents, -1
            )
            charging_actions, _, _, _ = (
                trainer.charging_agent.get_action_and_value(
                    env.get_charging_observation(),
                    deterministic=False,
                    deterministic_direction=True,
                )
            )
        return torch.cat(
            [feeder_actions, charging_actions.unsqueeze(1)], dim=1
        )

    print(
        f"Searching {args.candidates} sampled-policy episodes from "
        f"iteration {iteration} on {trainer.device}"
    )
    with isolated_torch_rng(args.policy_seed):
        stats = evaluate_multi_agent_policy_batched(
            env,
            policy_fn,
            num_episodes=args.candidates,
            target_expiry_penalty=env_config.target_expiry_penalty,
            verbose=False,
            capture_episode_count=args.candidates,
        )

    candidates = [
        candidate_record(stats, index) for index in range(args.candidates)
    ]
    selected = max(candidates, key=selection_key)
    selected_index = selected["candidate_index"]
    episode_data = stats["episode_data_list"][selected_index]

    if args.output is None:
        output_path = (
            experiment_dir
            / "rollouts"
            / "showcase"
            / (
                f"priority_charging_iter_{iteration}_env_"
                f"{args.environment_seed}_candidate_{selected_index}.mp4"
            )
        )
    else:
        output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source_experiment": str(experiment_dir),
        "checkpoint_iteration": iteration,
        "environment_seed": args.environment_seed,
        "policy_seed": args.policy_seed,
        "candidate_count": args.candidates,
        "selection_order": [
            "zero battery depletions",
            "charging direction accuracy",
            "reached priority sum",
            "minimum feeds per target",
            "fewer expirations",
            "recharges",
        ],
        "selected": selected,
        "candidates": candidates,
        "video": {
            "path": str(output_path),
            "fps": args.fps,
            "frame_stride": args.frame_stride,
        },
        "runtime_programmatic_navigation": env_config.charging_programmatic_navigation,
        "moving_recharge_stations": env_config.moving_recharge_stations,
    }
    metadata_path = output_path.with_suffix(".json")
    with metadata_path.open("w") as destination:
        json.dump(metadata, destination, indent=2)

    print("Selected rollout:")
    print(json.dumps(selected, indent=2))
    env.create_competition_gif(
        episode_data,
        output_path,
        fps=args.fps,
        frame_stride=args.frame_stride,
    )
    env.close()
    print(f"Metadata saved: {metadata_path}")


if __name__ == "__main__":
    main()
