#!/usr/bin/env python3
"""
PPO Training Script for OCAtari Assault (single-agent or bidding).
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from assault.assault_bidding_ppo import AssaultArgs
from assault.assault_single_agent_ppo import AssaultSingleAgentArgs
from assault.assault_experiment import AssaultExperiment


def main():
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    SINGLE_AGENT_MODE = True

    # Experiment settings
    EXPERIMENT_NAME = "assault_ppo_single_agent_exp1"
    CHECKPOINT_FREQ = 20
    EVAL_FREQ = 10
    VIDEO_FREQ = 0  # Save videos every N iterations (0 = same as EVAL_FREQ)
    NUM_EVAL_EPISODES = 5
    NUM_VIDEO_EPISODES = 3  # Number of episodes to save as videos
    LOG_VIDEOS_TO_WANDB = False  # Upload videos to wandb
    RENDER_OC_OVERLAY = False  # Draw object detection bounding boxes

    # Environment settings
    NUM_AGENTS = 3
    MAX_ENEMIES = 3
    MAX_STEPS = 10000
    HUD = True
    ALLOW_VARIABLE_ENEMIES = True
    ALLOW_SIDEWARD_FIRE = True  # Enable RIGHTFIRE and LEFTFIRE to destroy horizontal enemy missiles

    # Reward coefficients
    ENEMY_DESTROY_REWARD = 1.0  # Reward for destroying an enemy (based on visibility)
    OVERHEAT_PENALTY = 3.0      # Penalty when temperature bar turns red (moderate)
    LIFE_LOSS_PENALTY = 10.0     # Penalty for losing a life
    RAW_SCORE_SCALE = 0.01       # Scale for raw Atari score (dense reward for hits)
    FIRE_WHILE_HOT_PENALTY = 2.0  # Penalty for firing when health bar is red

    # Bidding settings (multi-agent mode only)
    BID_UPPER_BOUND = 10
    BID_PENALTY = 0.1
    ACTION_WINDOW = 3
    WINDOW_BIDDING = False
    WINDOW_PENALTY = 0.0

    # Training settings
    NUM_ITERATIONS = 300
    LEARNING_RATE = 1e-4
    NUM_ENVS = 128
    NUM_STEPS = 512
    NUM_MINIBATCHES = 8
    UPDATE_EPOCHS = 8
    ANNEAL_LR = True
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    NORM_ADV = True
    CLIP_COEF = 0.2
    CLIP_VLOSS = True
    ENT_COEF = 0.03
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    TARGET_KL = 0.015

    # Network
    ACTOR_HIDDEN_SIZES = (128, 128, 128, 128)
    CRITIC_HIDDEN_SIZES = (256, 256, 256, 256)

    # Wandb
    WANDB_PROJECT = "bidding-rl"
    WANDB_ENTITY = None
    TRACK = True

    # ========================================================================
    # End configuration
    # ========================================================================

    if SINGLE_AGENT_MODE:
        args = AssaultSingleAgentArgs(
            exp_name=EXPERIMENT_NAME or "assault_single_agent_ppo",
            seed=1,
            track=TRACK,
            wandb_project_name=WANDB_PROJECT,
            wandb_entity=WANDB_ENTITY,
            num_agents=NUM_AGENTS,
            max_enemies=MAX_ENEMIES,
            enemy_destroy_reward=ENEMY_DESTROY_REWARD,
            hit_penalty=OVERHEAT_PENALTY,
            life_loss_penalty=LIFE_LOSS_PENALTY,
            raw_score_scale=RAW_SCORE_SCALE,
            fire_while_hot_penalty=FIRE_WHILE_HOT_PENALTY,
            max_steps=MAX_STEPS,
            hud=HUD,
            allow_variable_enemies=ALLOW_VARIABLE_ENEMIES,
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
        args = AssaultArgs(
            exp_name=EXPERIMENT_NAME or "assault_bidding_ppo",
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
            enemy_destroy_reward=ENEMY_DESTROY_REWARD,
            hit_penalty=OVERHEAT_PENALTY,
            life_loss_penalty=LIFE_LOSS_PENALTY,
            raw_score_scale=RAW_SCORE_SCALE,
            fire_while_hot_penalty=FIRE_WHILE_HOT_PENALTY,
            max_steps=MAX_STEPS,
            hud=HUD,
            allow_variable_enemies=ALLOW_VARIABLE_ENEMIES,
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

    experiment = AssaultExperiment(
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
