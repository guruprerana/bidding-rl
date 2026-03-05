#!/usr/bin/env python3
"""
Bidding Mechanism Variants Comparison

Trains 4 variants of all_pay and winner_pays across 5 seeds:
  1. all_pay       — local observation  (visible_targets=0)
  2. all_pay       — no attention pooling (visible_targets=None, use_target_attention_pooling=False)
  3. winner_pays   — local observation  (visible_targets=0)
  4. winner_pays   — no attention pooling (visible_targets=None, use_target_attention_pooling=False)

Each experiment runs in its own subprocess (required for separate CUDA contexts).
Per-experiment output is written to BASE_LOG_DIR/seed_<seed>/<exp_name>.log.
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

BASE_LOG_DIR = "logs/gridworld_bidding_mechanism_variants"

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
SEEDS = [1825, 410, 4507, 4013, 3658]

# Network
ACTOR_HIDDEN_SIZES = [128, 128, 128, 128]
CRITIC_HIDDEN_SIZES = [256, 256, 256, 256]
TARGET_EMBED_DIM = 64
TARGET_ENCODER_HIDDEN_SIZES = [64, 64]

# Eval / logging
EVAL_NUM_AGENTS = None
CHECKPOINT_FREQ = 10
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

def make_ppo_args(
    bidding_mechanism: str,
    exp_name: str,
    seed: int,
    visible_targets,
    use_target_attention_pooling: bool,
) -> Args:
    """Build an Args instance for a multi-agent PPO run."""
    return Args(
        exp_name=exp_name,
        seed=seed,
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
        visible_targets=visible_targets,
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
        use_target_attention_pooling=use_target_attention_pooling,
        target_embed_dim=TARGET_EMBED_DIM,
        target_encoder_hidden_sizes=TARGET_ENCODER_HIDDEN_SIZES,
    )


def run_ppo_experiment(
    bidding_mechanism: str,
    exp_name: str,
    seed: int,
    visible_targets,
    use_target_attention_pooling: bool,
) -> None:
    """Create and run a multi-agent PPO experiment."""
    args = make_ppo_args(
        bidding_mechanism=bidding_mechanism,
        exp_name=exp_name,
        seed=seed,
        visible_targets=visible_targets,
        use_target_attention_pooling=use_target_attention_pooling,
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
        single_agent_mode=False,
        eval_max_steps=EVAL_MAX_STEPS,
        eval_num_agents=EVAL_NUM_AGENTS,
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
    def make_experiments(seed: int):
        s = f"_s{seed}"
        return [
            (
                f"all_pay — local obs (seed={seed})",
                f"bidding_cmp_all_pay_localobs{s}",
                functools.partial(
                    run_ppo_experiment,
                    "all_pay",
                    f"bidding_cmp_all_pay_localobs{s}",
                    seed,
                    0,     # visible_targets=0 (local observation)
                    False, # use_target_attention_pooling=False (safe with 0 visible targets)
                ),
            ),
            (
                f"all_pay — no attention pooling (seed={seed})",
                f"bidding_cmp_all_pay_no_attn{s}",
                functools.partial(
                    run_ppo_experiment,
                    "all_pay",
                    f"bidding_cmp_all_pay_no_attn{s}",
                    seed,
                    None,  # visible_targets=None (centralized/global obs)
                    False, # use_target_attention_pooling=False
                ),
            ),
            (
                f"winner_pays — local obs (seed={seed})",
                f"bidding_cmp_winner_pays_localobs{s}",
                functools.partial(
                    run_ppo_experiment,
                    "winner_pays",
                    f"bidding_cmp_winner_pays_localobs{s}",
                    seed,
                    0,     # visible_targets=0 (local observation)
                    False, # use_target_attention_pooling=False (safe with 0 visible targets)
                ),
            ),
            (
                f"winner_pays — no attention pooling (seed={seed})",
                f"bidding_cmp_winner_pays_no_attn{s}",
                functools.partial(
                    run_ppo_experiment,
                    "winner_pays",
                    f"bidding_cmp_winner_pays_no_attn{s}",
                    seed,
                    None,  # visible_targets=None (centralized/global obs)
                    False, # use_target_attention_pooling=False
                ),
            ),
        ]

    all_experiments = []
    for seed in SEEDS:
        all_experiments.extend(
            (seed, label, exp_name, run_fn)
            for label, exp_name, run_fn in make_experiments(seed)
        )

    os.makedirs(BASE_LOG_DIR, exist_ok=True)
    ctx = multiprocessing.get_context("spawn")

    procs = []
    gpu_ids = [0, 1, 2, 3]
    for idx, (seed, label, exp_name, run_fn) in enumerate(all_experiments):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        seed_log_dir = os.path.join(BASE_LOG_DIR, f"seed_{seed}")
        os.makedirs(seed_log_dir, exist_ok=True)
        log_path = os.path.join(seed_log_dir, f"{exp_name}.log")
        p = ctx.Process(target=_run_worker, args=(label, run_fn, log_path, gpu_id), name=exp_name)
        procs.append((label, log_path, p, gpu_id))

    print()
    print("=" * 72)
    print(f"  Launching {len(procs)} experiments in parallel")
    print(f"  Seeds: {SEEDS}")
    print(f"  Variants: local_obs (visible_targets=0), no_attention_pooling")
    print(f"  Mechanisms: all_pay, winner_pays")
    print("=" * 72)
    for label, log_path, p, gpu_id in procs:
        p.start()
        print(f"  [PID {p.pid:>6}]  GPU {gpu_id}  {label}")
        print(f"              log -> {log_path}")
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
        print(f"  {'':30}  log -> {log_path}")
    print()

    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
