#!/usr/bin/env python3
"""
LR-min Floor Sweep — All-Pay Mechanism

Runs 4 experiments in parallel, all using the all_pay bidding mechanism but
with different lr_min floor values to test whether preventing the LR from
decaying all the way to zero reduces the post-peak performance drop.

LR schedule: linear anneal from learning_rate down to lr_min (not to zero).

Tested values (learning_rate = 2.5e-4):
  1e-5  — tiny floor (~4% of initial LR)
  5e-5  — moderate floor (~20% of initial LR)
  1e-4  — substantial floor (~40% of initial LR)
  2e-4  — high floor (~80% of initial LR, near-constant LR)

All other config is identical to the bidding_mechanism_comparison all_pay run.

Usage:
    python experiment_scripts/lr_min_sweep_all_pay.py
"""

import functools
import multiprocessing
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from bidding_gridworld.bidding_ppo import Args
from bidding_gridworld.experiment import PPOMovingTargetsExperiment


# ============================================================================
# OUTPUT
# ============================================================================

BASE_LOG_DIR = "logs/lr_min_sweep_all_pay"

# ============================================================================
# ENVIRONMENT CONFIG  (matches bidding_mechanism_comparison)
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

BID_UPPER_BOUND = 6
BID_PENALTY = 0.1
ACTION_WINDOW = 5
WINDOW_BIDDING = False
WINDOW_PENALTY = 0.05
VISIBLE_TARGETS = None

# ============================================================================
# TRAINING CONFIG  (matches bidding_mechanism_comparison)
# ============================================================================

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

ACTOR_HIDDEN_SIZES = [128, 128, 128, 128]
CRITIC_HIDDEN_SIZES = [256, 256, 256, 256]
USE_TARGET_ATTENTION_POOLING = True
TARGET_EMBED_DIM = 64
TARGET_ENCODER_HIDDEN_SIZES = [64, 64]

CHECKPOINT_FREQ = 100
EVAL_FREQ = 10
NUM_EVAL_EPISODES = 20
NUM_VIDEO_EPISODES = 0
VIDEO_FREQ = 0
LOG_VIDEOS_TO_WANDB = False

WANDB_PROJECT = "bidding-rl"
WANDB_ENTITY = None
TRACK = True

# ============================================================================
# LR-MIN VALUES TO SWEEP
# ============================================================================

LR_MIN_VALUES = [1e-5, 5e-5, 1e-4, 2e-4]


# ============================================================================
# HELPERS
# ============================================================================

def exp_name_for(lr_min: float) -> str:
    """Stable, filesystem-safe experiment name encoding the lr_min value."""
    if lr_min == 0.0:
        return "lr_min_allpay_0"
    # Format as e.g. "lr_min_allpay_1e-05" -> sanitise for filenames
    return f"lr_min_allpay_{lr_min:.0e}".replace("-", "neg")


def make_args(lr_min: float) -> Args:
    return Args(
        exp_name=exp_name_for(lr_min),
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
        bidding_mechanism="all_pay",

        # Training
        num_iterations=NUM_ITERATIONS,
        learning_rate=LEARNING_RATE,
        lr_min=lr_min,
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


def run_experiment(lr_min: float) -> None:
    args = make_args(lr_min)
    experiment = PPOMovingTargetsExperiment(
        base_log_dir=BASE_LOG_DIR,
        experiment_name=exp_name_for(lr_min),
        checkpoint_freq=CHECKPOINT_FREQ,
        eval_freq=EVAL_FREQ,
        video_freq=VIDEO_FREQ,
        num_eval_episodes=NUM_EVAL_EPISODES,
        num_video_episodes=NUM_VIDEO_EPISODES,
        log_videos_to_wandb=LOG_VIDEOS_TO_WANDB,
        single_agent_mode=False,
        eval_max_steps=EVAL_MAX_STEPS,
    )
    experiment.run(args)


# ============================================================================
# PARALLEL EXECUTION
# ============================================================================

def _run_worker(label: str, run_fn, log_path: str, gpu_id: int) -> None:
    import traceback
    # Must be set before any CUDA initialisation happens in the subprocess.
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


def main():
    experiments = [
        (
            f"All-Pay  lr_min={lr_min:.0e}",
            exp_name_for(lr_min),
            functools.partial(run_experiment, lr_min),
        )
        for lr_min in LR_MIN_VALUES
    ]

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
    print(f"  learning_rate={LEARNING_RATE:.2e}   lr_min values: {LR_MIN_VALUES}")
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
