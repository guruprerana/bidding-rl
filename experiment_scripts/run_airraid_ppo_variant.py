#!/usr/bin/env python3
"""Run one Air Raid multi-agent PPO variant."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from airraid.airraid_bidding_ppo import AirRaidArgs
from airraid.airraid_experiment import AirRaidExperiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--action-window", type=int, required=True)
    parser.add_argument("--bid-penalty", type=float, default=0.01)
    parser.add_argument("--building-hit-penalty", type=float, required=True)
    parser.add_argument("--life-loss-penalty", type=float, required=True)
    parser.add_argument("--raw-score-scale", type=float, required=True)
    parser.add_argument("--enemy-missile-danger-penalty", type=float, default=0.0)
    parser.add_argument("--enemy-missile-danger-y-threshold", type=float, default=110.0)
    parser.add_argument("--enemy-missile-danger-x-radius", type=float, default=18.0)
    parser.add_argument("--enemy-missile-danger-y-radius", type=float, default=80.0)
    parser.add_argument("--enemy-missile-near-hit-penalty", type=float, default=0.0)
    parser.add_argument("--enemy-missile-near-hit-y-margin", type=float, default=35.0)
    parser.add_argument("--enemy-missile-near-hit-x-radius", type=float, default=25.0)
    parser.add_argument("--num-iterations", type=int, default=250)
    parser.add_argument("--num-eval-episodes", type=int, default=20)
    parser.add_argument("--obs-stack", type=int, default=1)
    parser.add_argument("--only-own-enemy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--building-penalty-mode", choices=("lane", "all_agents", "controller"), default="lane")
    parser.add_argument("--separate-agent-networks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--track", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    args = AirRaidArgs(
        exp_name=cli.exp_name,
        seed=cli.seed,
        track=cli.track,
        wandb_project_name="bidding-rl",
        wandb_entity=None,
        num_agents=3,
        max_enemies=3,
        bid_upper_bound=2,
        bid_penalty=cli.bid_penalty,
        action_window=cli.action_window,
        window_bidding=False,
        window_penalty=0.0,
        bidding_mechanism="all_pay",
        only_own_enemy=cli.only_own_enemy,
        obs_stack=cli.obs_stack,
        building_penalty_mode=cli.building_penalty_mode,
        separate_agent_networks=cli.separate_agent_networks,
        enemy_destroy_reward=10.0,
        building_hit_penalty=cli.building_hit_penalty,
        life_loss_penalty=cli.life_loss_penalty,
        raw_score_scale=cli.raw_score_scale,
        enemy_missile_danger_penalty=cli.enemy_missile_danger_penalty,
        enemy_missile_danger_y_threshold=cli.enemy_missile_danger_y_threshold,
        enemy_missile_danger_x_radius=cli.enemy_missile_danger_x_radius,
        enemy_missile_danger_y_radius=cli.enemy_missile_danger_y_radius,
        enemy_missile_near_hit_penalty=cli.enemy_missile_near_hit_penalty,
        enemy_missile_near_hit_y_margin=cli.enemy_missile_near_hit_y_margin,
        enemy_missile_near_hit_x_radius=cli.enemy_missile_near_hit_x_radius,
        max_steps=10000,
        hud=True,
        allow_sideward_fire=True,
        actor_hidden_sizes=(128, 128, 128, 128),
        critic_hidden_sizes=(256, 256, 256, 256),
        num_iterations=cli.num_iterations,
        learning_rate=1e-4,
        num_envs=128,
        num_steps=512,
        num_minibatches=8,
        update_epochs=8,
        anneal_lr=True,
        gamma=0.99,
        gae_lambda=0.95,
        norm_adv=True,
        clip_coef=0.05,
        clip_vloss=False,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
    )
    experiment = AirRaidExperiment(
        experiment_name=cli.exp_name,
        checkpoint_freq=10,
        eval_freq=10,
        video_freq=10,
        num_eval_episodes=cli.num_eval_episodes,
        num_video_episodes=1,
        log_videos_to_wandb=False,
        single_agent_mode=False,
        render_oc_overlay=False,
    )
    experiment.run(args)


if __name__ == "__main__":
    main()
