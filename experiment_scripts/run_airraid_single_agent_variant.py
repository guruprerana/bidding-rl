#!/usr/bin/env python3
"""Run one Air Raid single-agent PPO variant."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from airraid.airraid_experiment import AirRaidExperiment
from airraid.airraid_single_agent_ppo import AirRaidSingleAgentArgs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--building-hit-penalty", type=float, required=True)
    parser.add_argument("--life-loss-penalty", type=float, default=10.0)
    parser.add_argument("--raw-score-scale", type=float, default=0.0)
    parser.add_argument("--obs-stack", type=int, default=1)
    parser.add_argument("--num-iterations", type=int, default=750)
    parser.add_argument("--num-eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--track", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    args = AirRaidSingleAgentArgs(
        exp_name=cli.exp_name,
        seed=cli.seed,
        track=cli.track,
        wandb_project_name="bidding-rl",
        wandb_entity=None,
        num_agents=3,
        max_enemies=3,
        enemy_destroy_reward=10.0,
        building_hit_penalty=cli.building_hit_penalty,
        life_loss_penalty=cli.life_loss_penalty,
        raw_score_scale=cli.raw_score_scale,
        max_steps=10000,
        hud=True,
        allow_sideward_fire=True,
        obs_stack=cli.obs_stack,
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
        checkpoint_freq=50,
        eval_freq=50,
        video_freq=50,
        num_eval_episodes=cli.num_eval_episodes,
        num_video_episodes=1,
        log_videos_to_wandb=False,
        single_agent_mode=True,
        render_oc_overlay=False,
    )
    experiment.run(args)


if __name__ == "__main__":
    main()
