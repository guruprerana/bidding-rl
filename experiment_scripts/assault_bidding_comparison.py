#!/usr/bin/env python3
"""
Assault Bidding Mechanism Comparison Experiment

Runs 5 sequential experiments to compare bidding mechanisms and baselines
in the OCAtari Assault environment:
  1. Multi-agent PPO — all_pay
  2. Multi-agent PPO — winner_pays
  3. Multi-agent PPO — winner_pays_others_reward
  4. Single-agent PPO baseline
  5. DWN baseline

All configuration is at the top of this file.
"""

import os
import sys

# Add src to path (one level up from experiment_scripts/)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from assault.assault_bidding_ppo import AssaultArgs
from assault.assault_single_agent_ppo import AssaultSingleAgentArgs
from assault.assault_dwn import AssaultDWNArgs
from assault.assault_experiment import AssaultExperiment
from assault.assault_dwn_experiment import AssaultDWNExperiment


# ============================================================================
# OUTPUT
# ============================================================================

BASE_LOG_DIR = "logs/assault_bidding_mechanism_comparison"

# ============================================================================
# CONFIGURATION
# ============================================================================

# Wandb
WANDB_PROJECT = "bidding-rl"
WANDB_ENTITY = None
TRACK = True

# ============================================================================
# ENVIRONMENT CONFIG
# ============================================================================

NUM_AGENTS = 3
MAX_ENEMIES = 3
MAX_STEPS = 10000

BID_UPPER_BOUND = 2
BID_PENALTY = 0.3
ACTION_WINDOW = 15
WINDOW_BIDDING = False
WINDOW_PENALTY = 0.0

ENEMY_DESTROY_REWARD = 10.0
HIT_PENALTY = 0.0
LIFE_LOSS_PENALTY = 10.0
RAW_SCORE_SCALE = 0.0
FIRE_WHILE_HOT_PENALTY = 8.0

HUD = True
ALLOW_VARIABLE_ENEMIES = True
ALLOW_SIDEWARD_FIRE = True

# ============================================================================
# MULTI-AGENT PPO TRAINING CONFIG
# ============================================================================

NUM_ITERATIONS = 150
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
ENT_COEF = 0.05
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.02
SEED = 1

# Network
ACTOR_HIDDEN_SIZES = (128, 128, 128, 128)
CRITIC_HIDDEN_SIZES = (256, 256, 256, 256)

# Eval / logging
EVAL_FREQ = 10         # eval every 10 iterations (~6.7% of 150)
CHECKPOINT_FREQ = 50   # save every 50 iterations
NUM_EVAL_EPISODES = 5
NUM_VIDEO_EPISODES = 0
VIDEO_FREQ = 0
LOG_VIDEOS_TO_WANDB = False

# ============================================================================
# SINGLE-AGENT BASELINE CONFIG
# ============================================================================

# 150 × 128 × 512 × 3 = 450 × 128 × 512 → same total env steps
SINGLE_AGENT_ITERATIONS = 450
SINGLE_AGENT_EVAL_FREQ = 30  # 450 * ~6.7% ≈ 30 iterations

# ============================================================================
# DWN BASELINE CONFIG  (matches train_assault_dwn.py exactly)
# ============================================================================

DWN_TOTAL_TIMESTEPS = 10_000_000
DWN_NUM_ENVS = 8
DWN_GAMMA = 0.99

DWN_BUFFER_SIZE = 500_000
DWN_BATCH_SIZE = 256
DWN_LEARNING_STARTS = 10_000
DWN_TRAIN_FREQUENCY = 80
DWN_W_TRAIN_DELAY = 100_000
DWN_TARGET_NETWORK_FREQ = 1_000
DWN_TAU = 1.0

DWN_Q_HIDDEN_SIZES = (256, 256, 256)
DWN_W_HIDDEN_SIZES = (128, 128, 128)
DWN_Q_LEARNING_RATE = 1e-4
DWN_W_LEARNING_RATE = 1e-4

DWN_Q_EPSILON_START = 0.99
DWN_Q_EPSILON_MIN = 0.01
DWN_Q_EPSILON_DECAY = 0.995
DWN_W_EPSILON_START = 0.99
DWN_W_EPSILON_MIN = 0.01
DWN_W_EPSILON_DECAY = 0.995

DWN_LOG_FREQUENCY = 1_000
DWN_EVAL_FREQ = 2_500_000
DWN_CHECKPOINT_FREQ = 2_500_000
DWN_NUM_EVAL_EPISODES = 5
DWN_NUM_VIDEO_EPISODES = 0

# ============================================================================
# End of configuration
# ============================================================================


def run_ppo_experiment(bidding_mechanism: str, exp_name: str) -> None:
    """Create and run a multi-agent Assault PPO experiment."""
    args = AssaultArgs(
        exp_name=exp_name,
        seed=SEED,
        track=TRACK,
        wandb_project_name=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,

        # Environment
        num_agents=NUM_AGENTS,
        max_enemies=MAX_ENEMIES,
        bid_upper_bound=BID_UPPER_BOUND,
        bid_penalty=BID_PENALTY,
        action_window=ACTION_WINDOW,
        window_bidding=WINDOW_BIDDING,
        window_penalty=WINDOW_PENALTY,
        enemy_destroy_reward=ENEMY_DESTROY_REWARD,
        hit_penalty=HIT_PENALTY,
        life_loss_penalty=LIFE_LOSS_PENALTY,
        raw_score_scale=RAW_SCORE_SCALE,
        fire_while_hot_penalty=FIRE_WHILE_HOT_PENALTY,
        max_steps=MAX_STEPS,
        hud=HUD,
        allow_variable_enemies=ALLOW_VARIABLE_ENEMIES,
        allow_sideward_fire=ALLOW_SIDEWARD_FIRE,
        bidding_mechanism=bidding_mechanism,

        # Network
        actor_hidden_sizes=ACTOR_HIDDEN_SIZES,
        critic_hidden_sizes=CRITIC_HIDDEN_SIZES,

        # Training
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
    )
    experiment = AssaultExperiment(
        base_log_dir=BASE_LOG_DIR,
        experiment_name=exp_name,
        checkpoint_freq=CHECKPOINT_FREQ,
        eval_freq=EVAL_FREQ,
        video_freq=VIDEO_FREQ,
        num_eval_episodes=NUM_EVAL_EPISODES,
        num_video_episodes=NUM_VIDEO_EPISODES,
        log_videos_to_wandb=LOG_VIDEOS_TO_WANDB,
        single_agent_mode=False,
    )
    experiment.run(args)


def run_single_agent_baseline() -> None:
    """Create and run the single-agent Assault PPO baseline."""
    exp_name = "assault_cmp_single_agent"
    args = AssaultSingleAgentArgs(
        exp_name=exp_name,
        seed=SEED,
        track=TRACK,
        wandb_project_name=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,

        # Environment
        num_agents=NUM_AGENTS,
        max_enemies=MAX_ENEMIES,
        enemy_destroy_reward=ENEMY_DESTROY_REWARD,
        hit_penalty=HIT_PENALTY,
        life_loss_penalty=LIFE_LOSS_PENALTY,
        raw_score_scale=RAW_SCORE_SCALE,
        fire_while_hot_penalty=FIRE_WHILE_HOT_PENALTY,
        max_steps=MAX_STEPS,
        hud=HUD,
        allow_variable_enemies=ALLOW_VARIABLE_ENEMIES,
        allow_sideward_fire=ALLOW_SIDEWARD_FIRE,

        # Network
        actor_hidden_sizes=ACTOR_HIDDEN_SIZES,
        critic_hidden_sizes=CRITIC_HIDDEN_SIZES,

        # Training
        num_iterations=SINGLE_AGENT_ITERATIONS,
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
    )
    experiment = AssaultExperiment(
        base_log_dir=BASE_LOG_DIR,
        experiment_name=exp_name,
        checkpoint_freq=CHECKPOINT_FREQ,
        eval_freq=SINGLE_AGENT_EVAL_FREQ,
        video_freq=VIDEO_FREQ,
        num_eval_episodes=NUM_EVAL_EPISODES,
        num_video_episodes=NUM_VIDEO_EPISODES,
        log_videos_to_wandb=LOG_VIDEOS_TO_WANDB,
        single_agent_mode=True,
    )
    experiment.run(args)


def run_dwn_baseline() -> None:
    """Create and run the Assault DWN baseline."""
    exp_name = "assault_cmp_dwn"
    args = AssaultDWNArgs(
        exp_name=exp_name,
        seed=SEED,
        track=TRACK,
        wandb_project_name=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,

        # Environment
        num_agents=NUM_AGENTS,
        max_enemies=MAX_ENEMIES,
        enemy_destroy_reward=ENEMY_DESTROY_REWARD,
        hit_penalty=HIT_PENALTY,
        life_loss_penalty=LIFE_LOSS_PENALTY,
        raw_score_scale=RAW_SCORE_SCALE,
        fire_while_hot_penalty=FIRE_WHILE_HOT_PENALTY,
        allow_sideward_fire=ALLOW_SIDEWARD_FIRE,
        allow_variable_enemies=ALLOW_VARIABLE_ENEMIES,
        hud=HUD,
        max_steps=MAX_STEPS,

        # DWN core
        total_timesteps=DWN_TOTAL_TIMESTEPS,
        num_envs=DWN_NUM_ENVS,
        gamma=DWN_GAMMA,
        buffer_size=DWN_BUFFER_SIZE,
        batch_size=DWN_BATCH_SIZE,
        learning_starts=DWN_LEARNING_STARTS,
        train_frequency=DWN_TRAIN_FREQUENCY,
        w_train_delay=DWN_W_TRAIN_DELAY,
        target_network_freq=DWN_TARGET_NETWORK_FREQ,
        tau=DWN_TAU,

        # Networks
        q_hidden_sizes=DWN_Q_HIDDEN_SIZES,
        w_hidden_sizes=DWN_W_HIDDEN_SIZES,
        q_learning_rate=DWN_Q_LEARNING_RATE,
        w_learning_rate=DWN_W_LEARNING_RATE,

        # Epsilon schedules
        q_epsilon_start=DWN_Q_EPSILON_START,
        q_epsilon_min=DWN_Q_EPSILON_MIN,
        q_epsilon_decay=DWN_Q_EPSILON_DECAY,
        w_epsilon_start=DWN_W_EPSILON_START,
        w_epsilon_min=DWN_W_EPSILON_MIN,
        w_epsilon_decay=DWN_W_EPSILON_DECAY,

        # Logging
        log_frequency=DWN_LOG_FREQUENCY,
    )
    experiment = AssaultDWNExperiment(
        base_log_dir=BASE_LOG_DIR,
        experiment_name=exp_name,
        checkpoint_freq=DWN_CHECKPOINT_FREQ,
        eval_freq=DWN_EVAL_FREQ,
        video_freq=0,
        num_eval_episodes=DWN_NUM_EVAL_EPISODES,
        num_video_episodes=DWN_NUM_VIDEO_EPISODES,
        log_videos_to_wandb=False,
    )
    experiment.run(args)


# ============================================================================
# MAIN
# ============================================================================

def experiment_exists(exp_name: str) -> bool:
    """Return True if a folder starting with exp_name already exists in BASE_LOG_DIR."""
    base = os.path.join(os.path.dirname(__file__), '..', BASE_LOG_DIR)
    if not os.path.isdir(base):
        return False
    return any(entry.startswith(exp_name) for entry in os.listdir(base))


def main():
    experiments = [
        ("Multi-agent PPO — all_pay",                  "assault_cmp_all_pay",                   lambda: run_ppo_experiment("all_pay",                   "assault_cmp_all_pay")),
        ("Multi-agent PPO — winner_pays",               "assault_cmp_winner_pays",               lambda: run_ppo_experiment("winner_pays",               "assault_cmp_winner_pays")),
        ("Multi-agent PPO — winner_pays_others_reward", "assault_cmp_winner_pays_others_reward", lambda: run_ppo_experiment("winner_pays_others_reward", "assault_cmp_winner_pays_others_reward")),
        ("Single-agent PPO baseline",                   "assault_cmp_single_agent",              run_single_agent_baseline),
        ("DWN baseline",                                "assault_cmp_dwn",                       run_dwn_baseline),
    ]

    total = len(experiments)
    for idx, (label, exp_name, run_fn) in enumerate(experiments, start=1):
        print()
        print("=" * 72)
        print(f"  EXPERIMENT {idx}/{total}: {label}")
        print("=" * 72)
        if experiment_exists(exp_name):
            print(f"  SKIPPING — folder for '{exp_name}' already exists in {BASE_LOG_DIR}")
            continue
        run_fn()

    print()
    print("=" * 72)
    print("  All experiments complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
