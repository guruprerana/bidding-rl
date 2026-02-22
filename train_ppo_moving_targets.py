#!/usr/bin/env python3
"""
PPO Training Script for Bidding Gridworld

This script trains PPO policies for both single-agent and multi-agent bidding.

Features:
- Single-agent mode: One agent navigates to collect all targets
- Multi-agent mode: Multiple agents bid for control to reach their targets
- Periodic checkpointing
- Regular rollout evaluations with MP4 generation
- Comprehensive wandb logging
- Moving target environment support
- All configuration in one place (no CLI arguments needed)

Usage:
    python train_ppo_moving_targets.py

Configure all parameters in the CONFIGURATION section of the main() function.
Set SINGLE_AGENT_MODE = True for single-agent navigation, False for multi-agent bidding.
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bidding_gridworld.bidding_ppo import Args
from bidding_gridworld.single_agent_ppo import SingleAgentArgs
from bidding_gridworld.experiment import PPOMovingTargetsExperiment


def main():
    """Main training function."""

    # ========================================================================
    # CONFIGURATION - Modify parameters here
    # ========================================================================

    # Mode selection
    SINGLE_AGENT_MODE = False  # Set to True for single-agent navigation, False for multi-agent bidding
    MOVING_TARGETS = True  # Set to True for moving targets

    # Experiment settings
    EXPERIMENT_NAME = "ppo_10moving_targets_emb_exp1"  # Leave empty for default name with timestamp
    CHECKPOINT_FREQ = 100  # Save checkpoint every N iterations
    EVAL_FREQ = 20  # Evaluate every N iterations
    NUM_EVAL_EPISODES = 20  # Number of episodes per evaluation
    NUM_VIDEO_EPISODES = 1  # Number of episodes to save as MP4s
    VIDEO_FREQ = 100  # Save video rollouts every N iterations (0 = use eval freq)
    EVAL_NUM_AGENTS = 10  # Multi-agent only: override number of agents/targets during eval (requires attention pooling)
    EVAL_NUM_TARGETS = None  # Single-agent only: override number of targets during eval (fixed obs; keep None)

    # Environment parameters
    GRID_SIZE = 30
    NUM_AGENTS = 6  # For multi-agent: number of bidding agents; For single-agent: number of targets
    TARGET_REWARD = 50.0
    MAX_STEPS = 2000  # Maximum steps per episode during training
    EVAL_MAX_STEPS = 2000  # Maximum steps per episode during evaluation (typically longer than training)
    DISTANCE_REWARD_SCALE = 0.6
    TARGET_EXPIRY_STEPS = 200
    TARGET_EXPIRY_PENALTY = 50.0
    REWARD_DECAY_FACTOR = 0.0  # Single-agent only: decay rewards for over-visited targets (0.0 = no decay, 0.5 = moderate)

    # Multi-agent specific parameters (ignored in single-agent mode)
    BID_UPPER_BOUND = 6
    BID_PENALTY = 0.1
    ACTION_WINDOW = 5
    WINDOW_BIDDING = False  # Set to True to let agents choose their window length
    WINDOW_PENALTY = 0.05  # Penalty per window step (only applies when WINDOW_BIDDING = True)
    VISIBLE_TARGETS = None  # Set to None for centralized (all targets visible), or N for decentralized (each agent sees own target + N nearest others)
    BIDDING_MECHANISM = "all_pay"  # "all_pay" | "winner_pays" | "winner_pays_others_reward"

    # Moving targets parameters (only used if MOVING_TARGETS = True)
    DIRECTION_CHANGE_PROB = 0.1
    TARGET_MOVE_INTERVAL = 5

    # Training parameters
    NUM_ITERATIONS = 400
    LEARNING_RATE = 2.5e-4
    NUM_ENVS = 4096
    NUM_STEPS = 256
    NUM_MINIBATCHES = 256
    UPDATE_EPOCHS = 4
    ANNEAL_LR = True
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    NORM_ADV = True
    CLIP_COEF = 0.3
    CLIP_VLOSS = False
    ENT_COEF = 0.03
    VF_COEF = 1.0
    MAX_GRAD_NORM = 0.5
    TARGET_KL = None
    SEED = 1
    # Network architecture
    ACTOR_HIDDEN_SIZES = [128, 128, 128, 128]
    CRITIC_HIDDEN_SIZES = [256, 256, 256, 256]
    USE_TARGET_ATTENTION_POOLING = True
    TARGET_EMBED_DIM = 64
    TARGET_ENCODER_HIDDEN_SIZES = [64, 64]

    # Wandb tracking
    WANDB_PROJECT = "bidding-rl"
    WANDB_ENTITY = None
    TRACK = True  # Set to False to disable wandb tracking
    LOG_VIDEOS_TO_WANDB = False  # Set to True to upload MP4s to wandb

    # ========================================================================
    # End of configuration
    # ========================================================================

    # Create appropriate Args based on mode
    if SINGLE_AGENT_MODE:
        args = SingleAgentArgs(
            exp_name=EXPERIMENT_NAME or "single_agent_ppo",
            seed=SEED,
            track=TRACK,
            wandb_project_name=WANDB_PROJECT,
            wandb_entity=WANDB_ENTITY,

            # Environment config
            grid_size=GRID_SIZE,
            num_targets=NUM_AGENTS,  # In single-agent mode, this is number of targets
            target_reward=TARGET_REWARD,
            max_steps=MAX_STEPS,
            distance_reward_scale=DISTANCE_REWARD_SCALE,
            target_expiry_steps=TARGET_EXPIRY_STEPS,
            target_expiry_penalty=TARGET_EXPIRY_PENALTY,
            reward_decay_factor=REWARD_DECAY_FACTOR,
            moving_targets=MOVING_TARGETS,
            direction_change_prob=DIRECTION_CHANGE_PROB,
            target_move_interval=TARGET_MOVE_INTERVAL,

            # Training config
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
            use_target_attention_pooling=USE_TARGET_ATTENTION_POOLING,
            target_embed_dim=TARGET_EMBED_DIM,
            target_encoder_hidden_sizes=TARGET_ENCODER_HIDDEN_SIZES,
        )
    else:
        args = Args(
            exp_name=EXPERIMENT_NAME or "multi_agent_ppo",
            seed=SEED,
            track=TRACK,
            wandb_project_name=WANDB_PROJECT,
            wandb_entity=WANDB_ENTITY,

            # Environment config
            grid_size=GRID_SIZE,
            num_agents=NUM_AGENTS,
            bid_upper_bound=BID_UPPER_BOUND,
            bid_penalty=BID_PENALTY,
            target_reward=TARGET_REWARD,
            max_steps=MAX_STEPS,
            action_window=ACTION_WINDOW,
            distance_reward_scale=DISTANCE_REWARD_SCALE,
            target_expiry_steps=TARGET_EXPIRY_STEPS,
            target_expiry_penalty=TARGET_EXPIRY_PENALTY,
            moving_targets=MOVING_TARGETS,
            direction_change_prob=DIRECTION_CHANGE_PROB,
            target_move_interval=TARGET_MOVE_INTERVAL,
            window_bidding=WINDOW_BIDDING,
            window_penalty=WINDOW_PENALTY,
            visible_targets=VISIBLE_TARGETS,
            bidding_mechanism=BIDDING_MECHANISM,

            # Training config
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
            use_target_attention_pooling=USE_TARGET_ATTENTION_POOLING,
            target_embed_dim=TARGET_EMBED_DIM,
            target_encoder_hidden_sizes=TARGET_ENCODER_HIDDEN_SIZES,
        )

    # Create and run experiment
    experiment = PPOMovingTargetsExperiment(
        experiment_name=EXPERIMENT_NAME,
        checkpoint_freq=CHECKPOINT_FREQ,
        eval_freq=EVAL_FREQ,
        video_freq=VIDEO_FREQ,
        num_eval_episodes=NUM_EVAL_EPISODES,
        num_video_episodes=NUM_VIDEO_EPISODES,
        log_videos_to_wandb=LOG_VIDEOS_TO_WANDB,
        single_agent_mode=SINGLE_AGENT_MODE,
        eval_max_steps=EVAL_MAX_STEPS,
        eval_num_agents=EVAL_NUM_AGENTS,
        eval_num_targets=EVAL_NUM_TARGETS,
    )

    experiment.run(args)


if __name__ == "__main__":
    main()
