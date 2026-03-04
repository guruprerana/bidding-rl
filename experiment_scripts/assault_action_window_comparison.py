#!/usr/bin/env python3
"""
Assault Action Window Comparison Experiment

Runs experiments in parallel: multi-agent PPO with all_pay mechanism,
varying action_window across [1, 5, 15, 30, 60] in the OCAtari Assault environment.

All configuration is at the top of this file.
Each experiment runs in its own subprocess (required for separate CUDA contexts).
Per-experiment output is written to BASE_LOG_DIR/<exp_name>.log.
"""

import functools
import multiprocessing
import os
import sys

# Add src to path (one level up from experiment_scripts/)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from assault.assault_bidding_ppo import AssaultArgs
from assault.assault_experiment import AssaultExperiment


# ============================================================================
# OUTPUT
# ============================================================================

BASE_LOG_DIR = "logs/assault_action_window_comparison"

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

BIDDING_MECHANISM = "all_pay"

# action_window values to sweep
ACTION_WINDOWS = [5, 15, 30, 60]

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
CLIP_COEF = 0.05
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
EVAL_FREQ = 10
CHECKPOINT_FREQ = 10
NUM_EVAL_EPISODES = 5
NUM_VIDEO_EPISODES = 0
VIDEO_FREQ = 0
LOG_VIDEOS_TO_WANDB = False

# ============================================================================
# End of configuration
# ============================================================================


def run_ppo_experiment(action_window: int, exp_name: str) -> None:
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
        action_window=action_window,
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
        bidding_mechanism=BIDDING_MECHANISM,

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

def experiment_exists(exp_name: str) -> bool:
    """Return True if a folder starting with exp_name already exists in BASE_LOG_DIR."""
    base = os.path.join(os.path.dirname(__file__), '..', BASE_LOG_DIR)
    if not os.path.isdir(base):
        return False
    return any(entry.startswith(exp_name) for entry in os.listdir(base))


def main():
    experiments = [
        (
            f"Multi-agent PPO — all_pay — action_window={w}",
            f"assault_aw_cmp_w{w}",
            functools.partial(run_ppo_experiment, w, f"assault_aw_cmp_w{w}"),
        )
        for w in ACTION_WINDOWS
    ]

    # Skip experiments whose output folder already exists
    experiments = [(label, exp_name, run_fn) for label, exp_name, run_fn in experiments if not experiment_exists(exp_name)]

    if not experiments:
        print("All experiments already exist — nothing to run.")
        return

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
