#!/usr/bin/env python3
"""
DWN Training Script for OCAtari Assault (Multi-Agent Bidding)

Trains a Deep W-Learning policy for multi-agent bidding on the Assault Atari
game.  One shared Q-network proposes Atari actions per agent; one shared
W-network selects the winner.  The winning agent bids 1, all others bid 0.

Usage:
    python train_assault_dwn.py

Configure all parameters in the CONFIGURATION section below.
No CLI arguments needed.
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from assault.assault_dwn import AssaultDWNArgs
from assault.assault_dwn_experiment import AssaultDWNExperiment


def main():
    """Main training function."""

    # ========================================================================
    # CONFIGURATION — modify parameters here
    # ========================================================================

    # Experiment
    EXPERIMENT_NAME = "assault_dwn_exp1"
    SEED = 1

    # Checkpointing and evaluation
    CHECKPOINT_FREQ    = 2_500_000   # Save checkpoint every N global steps (0 = disabled)
    EVAL_FREQ          = 2_500_000   # Evaluate every N global steps (0 = disabled)
    VIDEO_FREQ         = 0           # Save videos every N steps (0 = same as EVAL_FREQ)
    NUM_EVAL_EPISODES  = 5           # Episodes per evaluation
    NUM_VIDEO_EPISODES = 1           # Episodes to save as MP4s
    LOG_VIDEOS_TO_WANDB = False      # Upload MP4s to wandb

    # ---- Environment --------------------------------------------------------

    NUM_AGENTS  = 3
    MAX_ENEMIES = 3
    MAX_STEPS   = 10000              # Max env steps per episode

    ENEMY_DESTROY_REWARD = 1.0
    HIT_PENALTY          = 1.0      # Penalty when temperature bar turns red
    LIFE_LOSS_PENALTY    = 10.0
    RAW_SCORE_SCALE      = 0.5      # Scale for raw Atari score signal
    FIRE_WHILE_HOT_PENALTY = 0.0    # Extra penalty for firing while overheated

    ALLOW_SIDEWARD_FIRE    = True   # Use 7 actions (including RIGHTFIRE/LEFTFIRE)
    ALLOW_VARIABLE_ENEMIES = True
    HUD                    = True

    # ---- DWN core -----------------------------------------------------------

    TOTAL_TIMESTEPS  = 10_000_000
    NUM_ENVS         = 8            # Parallel envs (CPU-bound; keep modest)
    GAMMA            = 0.99

    BUFFER_SIZE      = 500_000
    BATCH_SIZE       = 256
    LEARNING_STARTS  = 10_000       # Steps before first Q update
    TRAIN_FREQUENCY  = 80           # Update Q every N global steps (≈10× NUM_ENVS)
    W_TRAIN_DELAY    = 50_000       # Extra steps after LEARNING_STARTS before W updates
    TARGET_NETWORK_FREQ = 1_000     # Q-target sync period (in update counts)
    TAU              = 1.0          # 1.0 = hard copy, <1.0 = Polyak update

    # ---- Networks -----------------------------------------------------------

    Q_HIDDEN_SIZES = (256, 256)
    W_HIDDEN_SIZES = (128, 128)

    Q_LEARNING_RATE = 1e-4
    W_LEARNING_RATE = 1e-4

    # ---- Epsilon schedules (per-episode exponential decay) ------------------

    Q_EPSILON_START = 0.99
    Q_EPSILON_MIN   = 0.01
    Q_EPSILON_DECAY = 0.995

    W_EPSILON_START = 0.99
    W_EPSILON_MIN   = 0.01
    W_EPSILON_DECAY = 0.995

    # ---- Logging ------------------------------------------------------------

    LOG_FREQUENCY = 1_000           # Print + wandb log every N steps

    # ---- Wandb --------------------------------------------------------------

    TRACK          = True
    WANDB_PROJECT  = "bidding-rl"
    WANDB_ENTITY   = None           # Set to your wandb username/team or leave None

    # ========================================================================
    # End of configuration
    # ========================================================================

    args = AssaultDWNArgs(
        exp_name=EXPERIMENT_NAME,
        seed=SEED,
        track=TRACK,
        wandb_project_name=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,

        # Environment
        num_agents=NUM_AGENTS,
        max_enemies=MAX_ENEMIES,
        max_steps=MAX_STEPS,
        enemy_destroy_reward=ENEMY_DESTROY_REWARD,
        hit_penalty=HIT_PENALTY,
        life_loss_penalty=LIFE_LOSS_PENALTY,
        raw_score_scale=RAW_SCORE_SCALE,
        fire_while_hot_penalty=FIRE_WHILE_HOT_PENALTY,
        allow_sideward_fire=ALLOW_SIDEWARD_FIRE,
        allow_variable_enemies=ALLOW_VARIABLE_ENEMIES,
        hud=HUD,

        # DWN core
        total_timesteps=TOTAL_TIMESTEPS,
        num_envs=NUM_ENVS,
        gamma=GAMMA,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        learning_starts=LEARNING_STARTS,
        train_frequency=TRAIN_FREQUENCY,
        w_train_delay=W_TRAIN_DELAY,
        target_network_freq=TARGET_NETWORK_FREQ,
        tau=TAU,

        # Networks
        q_hidden_sizes=Q_HIDDEN_SIZES,
        w_hidden_sizes=W_HIDDEN_SIZES,
        q_learning_rate=Q_LEARNING_RATE,
        w_learning_rate=W_LEARNING_RATE,

        # Epsilon
        q_epsilon_start=Q_EPSILON_START,
        q_epsilon_min=Q_EPSILON_MIN,
        q_epsilon_decay=Q_EPSILON_DECAY,
        w_epsilon_start=W_EPSILON_START,
        w_epsilon_min=W_EPSILON_MIN,
        w_epsilon_decay=W_EPSILON_DECAY,

        # Logging
        log_frequency=LOG_FREQUENCY,
    )

    experiment = AssaultDWNExperiment(
        experiment_name=EXPERIMENT_NAME,
        checkpoint_freq=CHECKPOINT_FREQ,
        eval_freq=EVAL_FREQ,
        video_freq=VIDEO_FREQ,
        num_eval_episodes=NUM_EVAL_EPISODES,
        num_video_episodes=NUM_VIDEO_EPISODES,
        log_videos_to_wandb=LOG_VIDEOS_TO_WANDB,
    )

    experiment.run(args)


if __name__ == "__main__":
    main()
