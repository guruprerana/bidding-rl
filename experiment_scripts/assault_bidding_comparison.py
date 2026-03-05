#!/usr/bin/env python3
"""
Assault Bidding Mechanism Comparison Experiment

Runs 4 experiments in parallel to compare bidding mechanisms and baselines
in the OCAtari Assault environment:
  1. Multi-agent PPO — all_pay
  2. Multi-agent PPO — winner_pays
  3. Single-agent PPO baseline
  4. DWN baseline

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
CLIP_COEF = 0.05
CLIP_VLOSS = True
ENT_COEF = 0.05
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.02
SEEDS = [1825, 410, 4507, 4013, 3658]

# Network
ACTOR_HIDDEN_SIZES = (128, 128, 128, 128)
CRITIC_HIDDEN_SIZES = (256, 256, 256, 256)

# Eval / logging
EVAL_FREQ = 10         # eval every 10 iterations (~6.7% of 150)
CHECKPOINT_FREQ = 10   # save every 50 iterations
NUM_EVAL_EPISODES = 5
NUM_VIDEO_EPISODES = 0
VIDEO_FREQ = 0
LOG_VIDEOS_TO_WANDB = False

# Set to True to skip single-agent PPO and DWN baselines
MULTI_AGENT_ONLY = False

# ============================================================================
# SINGLE-AGENT BASELINE CONFIG
# ============================================================================

SINGLE_AGENT_ITERATIONS = 150
SINGLE_AGENT_EVAL_FREQ = 10

# These differ from the multi-agent PPO config above
SINGLE_AGENT_LEARNING_RATE = 2.5e-4
SINGLE_AGENT_NUM_STEPS = 512
SINGLE_AGENT_NUM_MINIBATCHES = 8
SINGLE_AGENT_UPDATE_EPOCHS = 8
SINGLE_AGENT_CLIP_COEF = 0.1
SINGLE_AGENT_ENT_COEF = 0.01
SINGLE_AGENT_TARGET_KL = None

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


def run_ppo_experiment(bidding_mechanism: str, exp_name: str, seed: int) -> None:
    """Create and run a multi-agent Assault PPO experiment."""
    args = AssaultArgs(
        exp_name=exp_name,
        seed=seed,
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


def run_single_agent_baseline(exp_name: str, seed: int) -> None:
    """Create and run the single-agent Assault PPO baseline."""
    args = AssaultSingleAgentArgs(
        exp_name=exp_name,
        seed=seed,
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
        learning_rate=SINGLE_AGENT_LEARNING_RATE,
        num_envs=NUM_ENVS,
        num_steps=SINGLE_AGENT_NUM_STEPS,
        num_minibatches=SINGLE_AGENT_NUM_MINIBATCHES,
        update_epochs=SINGLE_AGENT_UPDATE_EPOCHS,
        anneal_lr=ANNEAL_LR,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        norm_adv=NORM_ADV,
        clip_coef=SINGLE_AGENT_CLIP_COEF,
        clip_vloss=CLIP_VLOSS,
        ent_coef=SINGLE_AGENT_ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        target_kl=SINGLE_AGENT_TARGET_KL,
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


def run_dwn_baseline(exp_name: str, seed: int) -> None:
    """Create and run the Assault DWN baseline."""
    args = AssaultDWNArgs(
        exp_name=exp_name,
        seed=seed,
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
# PARALLEL EXECUTION
# ============================================================================

def _run_worker(label: str, run_fn, log_path: str, gpu_id: int) -> None:
    """Subprocess entry point: pins to a GPU, redirects stdout/stderr, and runs the experiment."""
    import traceback
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "a", buffering=1) as f:
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


def _guarded_worker(sem, label: str, run_fn, log_path: str, gpu_id: int) -> None:
    """Acquire semaphore slot, run experiment, then release — limits concurrency."""
    with sem:
        _run_worker(label, run_fn, log_path, gpu_id)


# ============================================================================
# MAIN
# ============================================================================

def main():
    def make_experiments(seed: int):
        s = f"_s{seed}"
        multi_agent = [
            (f"Multi-agent PPO — all_pay (seed={seed})",     f"assault_cmp_all_pay{s}",    functools.partial(run_ppo_experiment, "all_pay",    f"assault_cmp_all_pay{s}",    seed)),
            (f"Multi-agent PPO — winner_pays (seed={seed})", f"assault_cmp_winner_pays{s}", functools.partial(run_ppo_experiment, "winner_pays", f"assault_cmp_winner_pays{s}", seed)),
        ]
        baselines = [
            (f"Single-agent PPO baseline (seed={seed})", f"assault_cmp_single_agent{s}", functools.partial(run_single_agent_baseline, f"assault_cmp_single_agent{s}", seed)),
            (f"DWN baseline (seed={seed})",              f"assault_cmp_dwn{s}",          functools.partial(run_dwn_baseline,          f"assault_cmp_dwn{s}",          seed)),
        ]
        return multi_agent if MULTI_AGENT_ONLY else multi_agent + baselines

    all_experiments = []
    for seed in SEEDS:
        all_experiments.extend((seed, label, exp_name, run_fn) for label, exp_name, run_fn in make_experiments(seed))

    os.makedirs(BASE_LOG_DIR, exist_ok=True)
    ctx = multiprocessing.get_context("spawn")

    gpu_ids = [4, 7, 8, 9]
    max_concurrent = len(gpu_ids) * 2
    sem = ctx.Semaphore(max_concurrent)

    procs = []
    for idx, (seed, label, exp_name, run_fn) in enumerate(all_experiments):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        seed_log_dir = os.path.join(BASE_LOG_DIR, f"seed_{seed}")
        os.makedirs(seed_log_dir, exist_ok=True)
        log_path = os.path.join(seed_log_dir, f"{exp_name}.log")
        p = ctx.Process(target=_guarded_worker, args=(sem, label, run_fn, log_path, gpu_id), name=exp_name)
        procs.append((label, log_path, p, gpu_id))

    print()
    print("=" * 72)
    print(f"  Launching {len(procs)} experiments (max {max_concurrent} concurrent)")
    print(f"  Seeds: {SEEDS}")
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
