#!/usr/bin/env python3
"""
Reward Decay Comparison Experiment

Runs 3 single-agent PPO experiments in parallel to compare reward decay factors:
  1. Single-agent PPO — reward_decay_factor = 0.3
  2. Single-agent PPO — reward_decay_factor = 0.5
  3. Single-agent PPO — reward_decay_factor = 0.7

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

from bidding_gridworld.single_agent_ppo import SingleAgentArgs
from bidding_gridworld.experiment import PPOMovingTargetsExperiment


# ============================================================================
# OUTPUT
# ============================================================================

BASE_LOG_DIR = "logs/gridworld_reward_decay_comparison"

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
# SINGLE-AGENT PPO CONFIG
# ============================================================================

# 400 iters × 8 agents = 3200 single-agent iterations to match env steps
NUM_ITERATIONS = 3200
# Eval every 160 iters → same 5% cadence (160/3200 = 10/400)
EVAL_FREQ = 160

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

ACTOR_HIDDEN_SIZES = [128, 128, 128, 128]
CRITIC_HIDDEN_SIZES = [256, 256, 256, 256]
USE_TARGET_ATTENTION_POOLING = True
TARGET_EMBED_DIM = 64
TARGET_ENCODER_HIDDEN_SIZES = [64, 64]

CHECKPOINT_FREQ = 100
NUM_EVAL_EPISODES = 20
NUM_VIDEO_EPISODES = 0
VIDEO_FREQ = 0
LOG_VIDEOS_TO_WANDB = False

WANDB_PROJECT = "bidding-rl"
WANDB_ENTITY = None
TRACK = True

# Decay factors to compare
REWARD_DECAY_FACTORS = [0.3, 0.5, 0.7]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_single_agent_experiment(reward_decay_factor: float, exp_name: str) -> None:
    """Create and run a single-agent PPO experiment with a given decay factor."""
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
        reward_decay_factor=reward_decay_factor,
        moving_targets=MOVING_TARGETS,
        direction_change_prob=DIRECTION_CHANGE_PROB,
        target_move_interval=TARGET_MOVE_INTERVAL,

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
    experiment = PPOMovingTargetsExperiment(
        base_log_dir=BASE_LOG_DIR,
        experiment_name=exp_name,
        checkpoint_freq=CHECKPOINT_FREQ,
        eval_freq=EVAL_FREQ,
        video_freq=VIDEO_FREQ,
        num_eval_episodes=NUM_EVAL_EPISODES,
        num_video_episodes=NUM_VIDEO_EPISODES,
        log_videos_to_wandb=LOG_VIDEOS_TO_WANDB,
        single_agent_mode=True,
        eval_max_steps=EVAL_MAX_STEPS,
    )
    experiment.run(args)


# ============================================================================
# PARALLEL EXECUTION
# ============================================================================

def _run_worker(label: str, run_fn, log_path: str) -> None:
    """Subprocess entry point: redirects stdout/stderr to a log file and runs the experiment."""
    import traceback
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
    experiments = [
        (
            f"Single-agent PPO — reward_decay_factor={d}",
            f"decay_cmp_{str(d).replace('.', '_')}",
            functools.partial(
                run_single_agent_experiment,
                d,
                f"decay_cmp_{str(d).replace('.', '_')}",
            ),
        )
        for d in REWARD_DECAY_FACTORS
    ]

    os.makedirs(BASE_LOG_DIR, exist_ok=True)
    ctx = multiprocessing.get_context("spawn")

    procs = []
    for label, exp_name, run_fn in experiments:
        log_path = os.path.join(BASE_LOG_DIR, f"{exp_name}.log")
        p = ctx.Process(target=_run_worker, args=(label, run_fn, log_path), name=exp_name)
        procs.append((label, log_path, p))

    print()
    print("=" * 72)
    print(f"  Launching {len(procs)} experiments in parallel")
    print("=" * 72)
    for label, log_path, p in procs:
        p.start()
        print(f"  [PID {p.pid:>6}]  {label}")
        print(f"              log → {log_path}")
    print()

    for label, log_path, p in procs:
        p.join()

    print()
    print("=" * 72)
    print("  Results")
    print("=" * 72)
    any_failed = False
    for label, log_path, p in procs:
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
