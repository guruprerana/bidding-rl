#!/usr/bin/env python3
"""
Bidding Mechanism Comparison Experiment

Runs 5 experiments in parallel to compare bidding mechanisms and baselines:
  1. Multi-agent PPO — all_pay
  2. Multi-agent PPO — winner_pays
  3. Multi-agent PPO — winner_pays_others_reward
  4. Single-agent PPO baseline
  5. DWN baseline

All config is defined at the top of this file.
Each experiment runs in its own subprocess (required for separate CUDA contexts).
Per-experiment output is written to BASE_LOG_DIR/<exp_name>.log.
"""

import functools
import multiprocessing
import os
import sys

# Add src to path (one level up from experiment_scripts/)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from bidding_gridworld.bidding_ppo import Args
from bidding_gridworld.single_agent_ppo import SingleAgentArgs
from bidding_gridworld.dwn import GridworldDWNArgs
from bidding_gridworld.experiment import PPOMovingTargetsExperiment
from bidding_gridworld.dwn_experiment import DWNExperiment


# ============================================================================
# OUTPUT
# ============================================================================

BASE_LOG_DIR = "logs/gridworld_bidding_mechanism_comparison"

# ============================================================================
# SHARED ENVIRONMENT CONFIG
# ============================================================================

GRID_SIZE = 30
NUM_AGENTS = 8
TARGET_REWARD = 50.0
MAX_STEPS = 2000
EVAL_MAX_STEPS = 2000
DISTANCE_REWARD_SCALE = 0.6
TARGET_EXPIRY_STEPS = 200
TARGET_EXPIRY_PENALTY = 50.0
MOVING_TARGETS = True
DIRECTION_CHANGE_PROB = 0.1
TARGET_MOVE_INTERVAL = 5

# ============================================================================
# MULTI-AGENT PPO CONFIG
# ============================================================================

BID_UPPER_BOUND = 6
BID_PENALTY = 0.1
ACTION_WINDOW = 5
WINDOW_BIDDING = False
WINDOW_PENALTY = 0.05
VISIBLE_TARGETS = None

# Training
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
CLIP_COEF = 0.05
CLIP_VLOSS = False
ENT_COEF = 0.03
VF_COEF = 1.0
MAX_GRAD_NORM = 0.5
TARGET_KL = None
SEED = 1

# Network
ACTOR_HIDDEN_SIZES = [128, 128, 128, 128]
CRITIC_HIDDEN_SIZES = [256, 256, 256, 256]
USE_TARGET_ATTENTION_POOLING = True
TARGET_EMBED_DIM = 64
TARGET_ENCODER_HIDDEN_SIZES = [64, 64]

# Eval / logging
EVAL_NUM_AGENTS = None  # None = use NUM_AGENTS (keeps PPO eval comparable to DWN, which has no variable-agent support)
CHECKPOINT_FREQ = 10  # Save every 10 iterations
EVAL_FREQ = 10         # Eval every 10 iterations (2.5% of 400)
NUM_EVAL_EPISODES = 20
NUM_VIDEO_EPISODES = 0
VIDEO_FREQ = 0
LOG_VIDEOS_TO_WANDB = False

WANDB_PROJECT = "bidding-rl"
WANDB_ENTITY = None
TRACK = True

# Set to True to skip single-agent PPO and DWN baselines
MULTI_AGENT_ONLY = True

# ============================================================================
# SINGLE-AGENT BASELINE CONFIG
# ============================================================================

# 400 iters × 8 agents = 3200 single-agent iterations to match env steps
NUM_SINGLE_AGENT_ITERATIONS = 3200
REWARD_DECAY_FACTOR = 0.0
# Eval every 160 iters → same 5% cadence as multi-agent (160/3200 = 10/400)
SINGLE_AGENT_EVAL_FREQ = 160

# These differ from the multi-agent PPO config above
SINGLE_AGENT_CLIP_COEF = 0.1
SINGLE_AGENT_CLIP_VLOSS = True
SINGLE_AGENT_VF_COEF = 0.5

# ============================================================================
# DWN BASELINE CONFIG
# ============================================================================

DWN_TOTAL_TIMESTEPS = 50_000_000
DWN_NUM_ENVS = 256
DWN_GAMMA = 0.99

DWN_BUFFER_SIZE = 1_000_000
DWN_BATCH_SIZE = 256
DWN_LEARNING_STARTS = 100_000
DWN_TRAIN_FREQUENCY = 2_560
DWN_W_TRAIN_DELAY = 1_000_000
DWN_TARGET_NETWORK_FREQ = 1_000
DWN_TAU = 1.0

DWN_Q_HIDDEN_SIZES = (256, 256, 256, 256)
DWN_W_HIDDEN_SIZES = (128, 128, 128)
DWN_Q_LEARNING_RATE = 1e-4
DWN_W_LEARNING_RATE = 1e-4

DWN_Q_EPSILON_START = 0.99
DWN_Q_EPSILON_MIN = 0.01
DWN_Q_EPSILON_DECAY = 0.99
DWN_W_EPSILON_START = 0.99
DWN_W_EPSILON_MIN = 0.01
DWN_W_EPSILON_DECAY = 0.99

DWN_LOG_FREQUENCY = 25_600
DWN_CHECKPOINT_FREQ = 2_500_000
DWN_EVAL_FREQ = 2_500_000
DWN_VIDEO_FREQ = 0
DWN_NUM_EVAL_EPISODES = 10
DWN_NUM_VIDEO_EPISODES = 0
DWN_LOG_VIDEOS_TO_WANDB = False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def make_ppo_args(bidding_mechanism: str, exp_name: str) -> Args:
    """Build an Args instance for a multi-agent PPO run."""
    return Args(
        exp_name=exp_name,
        seed=SEED,
        track=TRACK,
        wandb_project_name=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,

        # Environment
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
        bidding_mechanism=bidding_mechanism,

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

        # Network
        actor_hidden_sizes=ACTOR_HIDDEN_SIZES,
        critic_hidden_sizes=CRITIC_HIDDEN_SIZES,
        use_target_attention_pooling=USE_TARGET_ATTENTION_POOLING,
        target_embed_dim=TARGET_EMBED_DIM,
        target_encoder_hidden_sizes=TARGET_ENCODER_HIDDEN_SIZES,
    )


def run_ppo_experiment(bidding_mechanism: str, exp_name: str) -> None:
    """Create and run a multi-agent PPO experiment."""
    args = make_ppo_args(bidding_mechanism, exp_name)
    experiment = PPOMovingTargetsExperiment(
        base_log_dir=BASE_LOG_DIR,
        experiment_name=exp_name,
        checkpoint_freq=CHECKPOINT_FREQ,
        eval_freq=EVAL_FREQ,
        video_freq=VIDEO_FREQ,
        num_eval_episodes=NUM_EVAL_EPISODES,
        num_video_episodes=NUM_VIDEO_EPISODES,
        log_videos_to_wandb=LOG_VIDEOS_TO_WANDB,
        single_agent_mode=False,
        eval_max_steps=EVAL_MAX_STEPS,
        eval_num_agents=EVAL_NUM_AGENTS,
    )
    experiment.run(args)


def run_single_agent_baseline() -> None:
    """Create and run the single-agent PPO baseline."""
    exp_name = "bidding_cmp_single_agent"
    args = SingleAgentArgs(
        exp_name=exp_name,
        seed=SEED,
        track=TRACK,
        wandb_project_name=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,

        # Environment
        grid_size=GRID_SIZE,
        num_targets=NUM_AGENTS,
        target_reward=TARGET_REWARD,
        max_steps=MAX_STEPS,
        distance_reward_scale=DISTANCE_REWARD_SCALE,
        target_expiry_steps=TARGET_EXPIRY_STEPS,
        target_expiry_penalty=TARGET_EXPIRY_PENALTY,
        reward_decay_factor=REWARD_DECAY_FACTOR,
        moving_targets=MOVING_TARGETS,
        direction_change_prob=DIRECTION_CHANGE_PROB,
        target_move_interval=TARGET_MOVE_INTERVAL,

        # Training
        num_iterations=NUM_SINGLE_AGENT_ITERATIONS,
        learning_rate=LEARNING_RATE,
        num_envs=NUM_ENVS,
        num_steps=NUM_STEPS,
        num_minibatches=NUM_MINIBATCHES,
        update_epochs=UPDATE_EPOCHS,
        anneal_lr=ANNEAL_LR,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        norm_adv=NORM_ADV,
        clip_coef=SINGLE_AGENT_CLIP_COEF,
        clip_vloss=SINGLE_AGENT_CLIP_VLOSS,
        ent_coef=ENT_COEF,
        vf_coef=SINGLE_AGENT_VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        target_kl=TARGET_KL,

        # Network
        actor_hidden_sizes=ACTOR_HIDDEN_SIZES,
        critic_hidden_sizes=CRITIC_HIDDEN_SIZES,
        use_target_attention_pooling=USE_TARGET_ATTENTION_POOLING,
        target_embed_dim=TARGET_EMBED_DIM,
        target_encoder_hidden_sizes=TARGET_ENCODER_HIDDEN_SIZES,
    )
    experiment = PPOMovingTargetsExperiment(
        base_log_dir=BASE_LOG_DIR,
        experiment_name=exp_name,
        checkpoint_freq=CHECKPOINT_FREQ,
        eval_freq=SINGLE_AGENT_EVAL_FREQ,
        video_freq=VIDEO_FREQ,
        num_eval_episodes=NUM_EVAL_EPISODES,
        num_video_episodes=NUM_VIDEO_EPISODES,
        log_videos_to_wandb=LOG_VIDEOS_TO_WANDB,
        single_agent_mode=True,
        eval_max_steps=EVAL_MAX_STEPS,
    )
    experiment.run(args)


def run_dwn_baseline() -> None:
    """Create and run the DWN baseline."""
    exp_name = "bidding_cmp_dwn"
    args = GridworldDWNArgs(
        exp_name=exp_name,
        seed=SEED,
        track=TRACK,
        wandb_project_name=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,

        # Environment
        grid_size=GRID_SIZE,
        num_targets=NUM_AGENTS,
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
    experiment = DWNExperiment(
        base_log_dir=BASE_LOG_DIR,
        experiment_name=exp_name,
        checkpoint_freq=DWN_CHECKPOINT_FREQ,
        eval_freq=DWN_EVAL_FREQ,
        video_freq=DWN_VIDEO_FREQ,
        num_eval_episodes=DWN_NUM_EVAL_EPISODES,
        num_video_episodes=DWN_NUM_VIDEO_EPISODES,
        log_videos_to_wandb=DWN_LOG_VIDEOS_TO_WANDB,
        eval_max_steps=EVAL_MAX_STEPS,
    )
    experiment.run(args)


# ============================================================================
# PARALLEL EXECUTION
# ============================================================================

def _run_worker(label: str, run_fn, log_path: str, gpu_id: int) -> None:
    """Subprocess entry point: pins to a GPU, redirects stdout/stderr, and runs the experiment."""
    import traceback
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "w", buffering=1) as f:
        sys.stdout = f
        sys.stderr = f
        print(f"{'=' * 72}")
        print(f"  {label}")
        print(f"{'=' * 72}")
        try:
            run_fn()
            print(f"\nEXPERIMENT COMPLETED: {label}")
        except Exception as e:
            print(f"\nEXPERIMENT FAILED: {label}")
            print(f"Error: {e}")
            traceback.print_exc()
            raise


# ============================================================================
# MAIN
# ============================================================================

def main():
    multi_agent_experiments = [
        ("Multi-agent PPO — all_pay",                  "bidding_cmp_all_pay",                   functools.partial(run_ppo_experiment, "all_pay",                   "bidding_cmp_all_pay")),
        ("Multi-agent PPO — winner_pays",               "bidding_cmp_winner_pays",               functools.partial(run_ppo_experiment, "winner_pays",               "bidding_cmp_winner_pays")),
        ("Multi-agent PPO — winner_pays_others_reward", "bidding_cmp_winner_pays_others_reward", functools.partial(run_ppo_experiment, "winner_pays_others_reward",  "bidding_cmp_winner_pays_others_reward")),
    ]
    baseline_experiments = [
        ("Single-agent PPO baseline",                   "bidding_cmp_single_agent",              run_single_agent_baseline),
        ("DWN baseline",                                "bidding_cmp_dwn",                       run_dwn_baseline),
    ]

    experiments = multi_agent_experiments if MULTI_AGENT_ONLY else multi_agent_experiments + baseline_experiments

    os.makedirs(BASE_LOG_DIR, exist_ok=True)
    ctx = multiprocessing.get_context("spawn")

    procs = []
    for gpu_id, (label, exp_name, run_fn) in enumerate(experiments):
        log_path = os.path.join(BASE_LOG_DIR, f"{exp_name}.log")
        p = ctx.Process(target=_run_worker, args=(label, run_fn, log_path, gpu_id), name=exp_name)
        procs.append((label, log_path, p, gpu_id))

    print()
    print("=" * 72)
    print(f"  Launching {len(procs)} experiments in parallel (one per GPU)")
    if MULTI_AGENT_ONLY:
        print("  (baselines skipped — MULTI_AGENT_ONLY=True)")
    print("=" * 72)
    for label, log_path, p, gpu_id in procs:
        p.start()
        print(f"  [PID {p.pid:>6}]  GPU {gpu_id}  {label}")
        print(f"              log → {log_path}")
    print()

    for label, log_path, p, gpu_id in procs:
        p.join()

    print()
    print("=" * 72)
    print("  Results")
    print("=" * 72)
    any_failed = False
    for label, log_path, p, gpu_id in procs:
        status = "COMPLETED" if p.exitcode == 0 else f"FAILED (exit {p.exitcode})"
        if p.exitcode != 0:
            any_failed = True
        print(f"  {status:<30}  {label}")
        print(f"  {'':30}  log → {log_path}")
    print()

    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
