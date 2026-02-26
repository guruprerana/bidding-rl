#!/usr/bin/env python3
"""
Bid Penalty Comparison Experiment

Runs 5 experiments in parallel: multi-agent PPO with all_pay mechanism,
varying bid_penalty across 0, 0.1, 1, 2, 5.

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
from bidding_gridworld.experiment import PPOMovingTargetsExperiment


# ============================================================================
# OUTPUT
# ============================================================================

BASE_LOG_DIR = "logs/gridworld_bid_penalty_comparison"

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

BIDDING_MECHANISM = "all_pay"
BID_UPPER_BOUND = 6
ACTION_WINDOW = 5
WINDOW_BIDDING = False
WINDOW_PENALTY = 0.05
VISIBLE_TARGETS = None

# bid_penalty values to sweep
BID_PENALTIES = [0, 0.1, 1, 2, 5]

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
CLIP_COEF = 0.3
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
EVAL_NUM_AGENTS = None
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
# HELPER FUNCTIONS
# ============================================================================

def make_ppo_args(bid_penalty: float, exp_name: str) -> Args:
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
        bid_penalty=bid_penalty,
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


def run_ppo_experiment(bid_penalty: float, exp_name: str) -> None:
    """Create and run a multi-agent PPO experiment."""
    args = make_ppo_args(bid_penalty, exp_name)
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


# ============================================================================
# PARALLEL EXECUTION
# ============================================================================

def _run_worker(label: str, run_fn, log_path: str, gpu_id: int = 0) -> None:
    """Subprocess entry point: redirects stdout/stderr to a log file and runs the experiment."""
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

def _penalty_tag(p: float) -> str:
    """Format a penalty value into a filesystem-safe tag (e.g. 0.1 -> '0p1')."""
    return str(p).replace(".", "p")


def main():
    experiments = [
        (
            f"Multi-agent PPO — all_pay — bid_penalty={p}",
            f"bid_penalty_cmp_p{_penalty_tag(p)}",
            functools.partial(run_ppo_experiment, p, f"bid_penalty_cmp_p{_penalty_tag(p)}"),
        )
        for p in BID_PENALTIES
    ]

    num_gpus = int(os.popen("nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l").read().strip() or 1)

    os.makedirs(BASE_LOG_DIR, exist_ok=True)
    ctx = multiprocessing.get_context("spawn")

    procs = []
    for i, (label, exp_name, run_fn) in enumerate(experiments):
        log_path = os.path.join(BASE_LOG_DIR, f"{exp_name}.log")
        gpu_id = i % num_gpus
        p = ctx.Process(target=_run_worker, args=(label, run_fn, log_path, gpu_id), name=exp_name)
        procs.append((label, log_path, p, gpu_id))

    print()
    print("=" * 72)
    print(f"  Launching {len(procs)} experiments in parallel across {num_gpus} GPU(s)")
    print("=" * 72)
    for label, log_path, p, gpu_id in procs:
        p.start()
        print(f"  [PID {p.pid:>6}]  [GPU {gpu_id}]  {label}")
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
