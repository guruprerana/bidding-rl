#!/usr/bin/env python3
"""
PPO Training Script for OCAtari Air Raid (single-agent or bidding).
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from airraid.airraid_bidding_ppo import AirRaidArgs
from airraid.airraid_experiment import AirRaidExperiment
from airraid.airraid_single_agent_ppo import AirRaidSingleAgentArgs


def main():
    SINGLE_AGENT_MODE = False

    EXPERIMENT_NAME = "airraid_ppo_multiagent"
    CHECKPOINT_FREQ = 10
    EVAL_FREQ = 10
    VIDEO_FREQ = 10
    NUM_EVAL_EPISODES = 5
    NUM_VIDEO_EPISODES = 1
    LOG_VIDEOS_TO_WANDB = False
    RENDER_OC_OVERLAY = False

    NUM_AGENTS = 3
    MAX_ENEMIES = 3
    MAX_STEPS = 10000
    HUD = True
    ALLOW_SIDEWARD_FIRE = True

    ENEMY_DESTROY_REWARD = 10.0
    BUILDING_HIT_PENALTY = 5.0
    LIFE_LOSS_PENALTY = 10.0
    RAW_SCORE_SCALE_SINGLE = 0.00
    RAW_SCORE_SCALE_MULTI = 0.0

    BID_UPPER_BOUND = 2
    BID_PENALTY = 0.01
    ACTION_WINDOW = 50
    WINDOW_BIDDING = False
    WINDOW_PENALTY = 0.0
    BIDDING_MECHANISM = "all_pay"
    ONLY_OWN_ENEMY = True

    NUM_ITERATIONS = 400
    LEARNING_RATE = 1e-4
    NUM_ENVS = 128
    NUM_STEPS = 512
    NUM_MINIBATCHES = 8
    UPDATE_EPOCHS = 8
    ANNEAL_LR = True
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    NORM_ADV = True
    CLIP_COEF = 0.05
    CLIP_VLOSS = False
    ENT_COEF = 0.05
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    TARGET_KL = None

    ACTOR_HIDDEN_SIZES = (128, 128, 128, 128)
    CRITIC_HIDDEN_SIZES = (256, 256, 256, 256)

    WANDB_PROJECT = "bidding-rl"
    WANDB_ENTITY = None
    TRACK = True

    if SINGLE_AGENT_MODE:
        args = AirRaidSingleAgentArgs(
            exp_name=EXPERIMENT_NAME or "airraid_single_agent_ppo",
            seed=1,
            track=TRACK,
            wandb_project_name=WANDB_PROJECT,
            wandb_entity=WANDB_ENTITY,
            num_agents=NUM_AGENTS,
            max_enemies=MAX_ENEMIES,
            enemy_destroy_reward=ENEMY_DESTROY_REWARD,
            building_hit_penalty=BUILDING_HIT_PENALTY,
            life_loss_penalty=LIFE_LOSS_PENALTY,
            raw_score_scale=RAW_SCORE_SCALE_SINGLE,
            max_steps=MAX_STEPS,
            hud=HUD,
            allow_sideward_fire=ALLOW_SIDEWARD_FIRE,
            num_iterations=NUM_ITERATIONS,
            learning_rate=LEARNING_RATE,
            num_envs=NUM_ENVS,
            num_steps=NUM_STEPS,
            num_minibatches=NUM_MINIBATCHES,
            update_epochs=UPDATE_EPOCHS,
            anneal_lr=ANNEAL_LR,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            norm_adv=NORM_ADV,
            clip_coef=CLIP_COEF,
            clip_vloss=CLIP_VLOSS,
            ent_coef=ENT_COEF,
            vf_coef=VF_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            target_kl=TARGET_KL,
            actor_hidden_sizes=ACTOR_HIDDEN_SIZES,
            critic_hidden_sizes=CRITIC_HIDDEN_SIZES,
        )
    else:
        args = AirRaidArgs(
            exp_name=EXPERIMENT_NAME or "airraid_bidding_ppo",
            seed=1,
            track=TRACK,
            wandb_project_name=WANDB_PROJECT,
            wandb_entity=WANDB_ENTITY,
            num_agents=NUM_AGENTS,
            max_enemies=MAX_ENEMIES,
            bid_upper_bound=BID_UPPER_BOUND,
            bid_penalty=BID_PENALTY,
            action_window=ACTION_WINDOW,
            window_bidding=WINDOW_BIDDING,
            window_penalty=WINDOW_PENALTY,
            bidding_mechanism=BIDDING_MECHANISM,
            only_own_enemy=ONLY_OWN_ENEMY,
            enemy_destroy_reward=ENEMY_DESTROY_REWARD,
            building_hit_penalty=BUILDING_HIT_PENALTY,
            life_loss_penalty=LIFE_LOSS_PENALTY,
            raw_score_scale=RAW_SCORE_SCALE_MULTI,
            max_steps=MAX_STEPS,
            hud=HUD,
            allow_sideward_fire=ALLOW_SIDEWARD_FIRE,
            num_iterations=NUM_ITERATIONS,
            learning_rate=LEARNING_RATE,
            num_envs=NUM_ENVS,
            num_steps=NUM_STEPS,
            num_minibatches=NUM_MINIBATCHES,
            update_epochs=UPDATE_EPOCHS,
            anneal_lr=ANNEAL_LR,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            norm_adv=NORM_ADV,
            clip_coef=CLIP_COEF,
            clip_vloss=CLIP_VLOSS,
            ent_coef=ENT_COEF,
            vf_coef=VF_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            target_kl=TARGET_KL,
            actor_hidden_sizes=ACTOR_HIDDEN_SIZES,
            critic_hidden_sizes=CRITIC_HIDDEN_SIZES,
        )

    experiment = AirRaidExperiment(
        experiment_name=EXPERIMENT_NAME,
        checkpoint_freq=CHECKPOINT_FREQ,
        eval_freq=EVAL_FREQ,
        video_freq=VIDEO_FREQ,
        num_eval_episodes=NUM_EVAL_EPISODES,
        num_video_episodes=NUM_VIDEO_EPISODES,
        log_videos_to_wandb=LOG_VIDEOS_TO_WANDB,
        single_agent_mode=SINGLE_AGENT_MODE,
        render_oc_overlay=RENDER_OC_OVERLAY,
    )
    experiment.run(args)


if __name__ == "__main__":
    main()
