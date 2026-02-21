#!/usr/bin/env python3
"""
DWN Training Script for Bidding Gridworld — small environment (10x10, 3 targets)

Same algorithm as train_dwn.py but with a smaller, faster environment.
Parameters are scaled proportionally: the 10x10 grid has ~3× shorter max
manhattan distance and ~10× shorter episodes than the 30x30 version.

Usage:
    python train_dwn_small.py

Configure all parameters in the CONFIGURATION section below.
No CLI arguments needed.
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bidding_gridworld.dwn import GridworldDWNArgs
from bidding_gridworld.dwn_experiment import DWNExperiment


def main():
    """Main training function."""

    # ========================================================================
    # CONFIGURATION — modify parameters here
    # ========================================================================

    # Experiment
    EXPERIMENT_NAME = "dwn_small_moving_targets_exp1"
    SEED = 1

    # Checkpointing and evaluation
    CHECKPOINT_FREQ  = 500_000   # Save checkpoint every N global steps (0 = disabled)
    EVAL_FREQ        = 500_000   # Evaluate every N global steps (0 = disabled)
    VIDEO_FREQ       = 0         # Save videos every N steps (0 = same as EVAL_FREQ)
    NUM_EVAL_EPISODES  = 10      # Episodes per evaluation
    NUM_VIDEO_EPISODES = 3       # Episodes to save as MP4s
    LOG_VIDEOS_TO_WANDB = False  # Upload MP4s to wandb
    EVAL_MAX_STEPS   = 200       # Max steps per eval episode

    # ---- Environment --------------------------------------------------------

    GRID_SIZE = 10
    NUM_TARGETS = 3            # Number of bidding agents / objectives
    TARGET_REWARD = 50.0
    MAX_STEPS = 200            # Max steps per episode (~10x less than 30x30)

    DISTANCE_REWARD_SCALE = 0.6  # Reward for closing distance to own target

    TARGET_EXPIRY_STEPS = 50     # Steps before an unreached target expires (~10x less than 30x30)
    TARGET_EXPIRY_PENALTY = 50.0  # Penalty applied on expiry

    MOVING_TARGETS = True
    DIRECTION_CHANGE_PROB = 0.1  # Probability a moving target changes direction
    TARGET_MOVE_INTERVAL = 2    # Steps between target moves (~2x less than 30x30)

    VISIBLE_TARGETS = None      # None = centralized (all targets visible to each agent)
                                 # int N = each agent sees own target + N nearest others

    # ---- DWN core -----------------------------------------------------------

    TOTAL_TIMESTEPS = 5_000_000   # ~10x less than 30x30
    NUM_ENVS = 256
    GAMMA = 0.99

    BUFFER_SIZE = 1_000_000
    BATCH_SIZE = 256
    LEARNING_STARTS = 10_000    # Steps before first Q update (~5x less than 30x30)
    TRAIN_FREQUENCY = 512       # Update Q every N global steps; must be >= NUM_ENVS (global_step increments by NUM_ENVS each iteration, so any value < NUM_ENVS fires every step)
    W_TRAIN_DELAY = 1_000_000      # Additional steps after LEARNING_STARTS before W updates begin (~5x less than 30x30)
    TARGET_NETWORK_FREQ = 1_000 # Q-target sync period (in update counts)
    TAU = 1.0                   # 1.0 = hard copy, <1.0 = soft (Polyak) update

    # ---- Networks -----------------------------------------------------------

    Q_HIDDEN_SIZES = (256, 256, 256, 256)
    W_HIDDEN_SIZES = (128, 128)

    Q_LEARNING_RATE = 1e-3
    W_LEARNING_RATE = 1e-3

    # ---- Epsilon schedules (per-episode exponential decay) ------------------

    Q_EPSILON_START = 0.99
    Q_EPSILON_MIN   = 0.01
    Q_EPSILON_DECAY = 0.99

    W_EPSILON_START = 0.99
    W_EPSILON_MIN   = 0.01
    W_EPSILON_DECAY = 0.99

    # ---- Logging ------------------------------------------------------------

    LOG_FREQUENCY = 5_120       # Print + wandb log every N steps (~20 global steps at NUM_ENVS=256)

    # ---- Wandb --------------------------------------------------------------

    TRACK = True
    WANDB_PROJECT = "bidding-rl"
    WANDB_ENTITY = None         # Set to your wandb username/team or leave None

    # ========================================================================
    # End of configuration
    # ========================================================================

    args = GridworldDWNArgs(
        exp_name=EXPERIMENT_NAME,
        seed=SEED,
        track=TRACK,
        wandb_project_name=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,

        # Environment
        grid_size=GRID_SIZE,
        num_targets=NUM_TARGETS,
        target_reward=TARGET_REWARD,
        max_steps=MAX_STEPS,
        distance_reward_scale=DISTANCE_REWARD_SCALE,
        target_expiry_steps=TARGET_EXPIRY_STEPS,
        target_expiry_penalty=TARGET_EXPIRY_PENALTY,
        moving_targets=MOVING_TARGETS,
        direction_change_prob=DIRECTION_CHANGE_PROB,
        target_move_interval=TARGET_MOVE_INTERVAL,
        visible_targets=VISIBLE_TARGETS,

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

    experiment = DWNExperiment(
        experiment_name=EXPERIMENT_NAME,
        checkpoint_freq=CHECKPOINT_FREQ,
        eval_freq=EVAL_FREQ,
        video_freq=VIDEO_FREQ,
        num_eval_episodes=NUM_EVAL_EPISODES,
        num_video_episodes=NUM_VIDEO_EPISODES,
        log_videos_to_wandb=LOG_VIDEOS_TO_WANDB,
        eval_max_steps=EVAL_MAX_STEPS,
    )

    experiment.run(args)


if __name__ == "__main__":
    main()
