#!/usr/bin/env python3
"""
Optuna HPO Script for Single-Agent PPO (decay=0.7)

Uses Optuna with SQLite backend for parallel hyperparameter search.
N_WORKERS subprocesses provide CUDA isolation; each runs
N_TRIALS_TOTAL // N_WORKERS Optuna trials in-process.

Optimises for the final avg_min_targets_reached from a single end-of-training
evaluation (no periodic evals during training).

Re-running the script resumes an existing study and adds more trials.
"""

import glob
import json
import multiprocessing
import os
import sys
import traceback

# ============================================================================
# OUTPUT
# ============================================================================

BASE_LOG_DIR = "logs/optuna_hparam_search"

# ============================================================================
# FIXED CONFIG (same as decay_cmp_0_7)
# ============================================================================

GRID_SIZE = 30
NUM_TARGETS = 8          # also used as num_agents for the env
TARGET_REWARD = 50.0
MAX_STEPS = 2000
EVAL_MAX_STEPS = 2000
DISTANCE_REWARD_SCALE = 0.6
TARGET_EXPIRY_STEPS = 200
TARGET_EXPIRY_PENALTY = 50.0
MOVING_TARGETS = True
DIRECTION_CHANGE_PROB = 0.1
TARGET_MOVE_INTERVAL = 5

NUM_ITERATIONS = 1600
# Disable periodic evals/checkpoints during HPO — only the final eval runs.
# Setting these above NUM_ITERATIONS ensures the iteration % freq == 0 condition
# never fires during training; PPOMovingTargetsExperiment always runs a final eval
# in on_training_end regardless.
EVAL_FREQ = NUM_ITERATIONS + 1
CHECKPOINT_FREQ = NUM_ITERATIONS + 1

NUM_ENVS = 4096
NUM_STEPS = 256          # fixed — compute budget, not algo quality

# Fixed network architecture
ACTOR_HIDDEN_SIZES = [128, 128, 128, 128]
CRITIC_HIDDEN_SIZES = [256, 256, 256, 256]
USE_TARGET_ATTENTION_POOLING = True
TARGET_EMBED_DIM = 64
TARGET_ENCODER_HIDDEN_SIZES = [64, 64]

# Fixed experiment-level params
NUM_EVAL_EPISODES = 20
NUM_VIDEO_EPISODES = 0
VIDEO_FREQ = 0
LOG_VIDEOS_TO_WANDB = False
ANNEAL_LR = True
NORM_ADV = True
CLIP_VLOSS = False
TARGET_KL = None
SEED = 1
TRACK = False            # disable W&B per trial to avoid clutter

# Optuna parallelism
N_WORKERS = 4
N_TRIALS_TOTAL = 100

STUDY_NAME = "optuna_ppo_hparam_search"


# ============================================================================
# OBJECTIVE
# ============================================================================

def objective(trial) -> float:
    """
    Optuna objective: run one PPO experiment and return the final
    avg_min_targets_reached from the end-of-training evaluation.
    """
    # ------------------------------------------------------------------
    # Sample hyperparameters
    # ------------------------------------------------------------------
    learning_rate    = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma            = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda       = trial.suggest_float("gae_lambda", 0.9, 0.99)
    clip_coef        = trial.suggest_float("clip_coef", 0.1, 0.4)
    ent_coef         = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
    vf_coef          = trial.suggest_float("vf_coef", 0.25, 2.0)
    update_epochs    = trial.suggest_categorical("update_epochs", [2, 4, 8])
    # all choices are valid divisors of batch_size = 4096 × 256 = 1_048_576
    num_minibatches  = trial.suggest_categorical("num_minibatches", [64, 128, 256, 512])
    max_grad_norm    = trial.suggest_float("max_grad_norm", 0.3, 1.0)
    reward_decay_factor = trial.suggest_float("reward_decay_factor", 0.5, 1.2)

    exp_name = f"optuna_trial_{trial.number:03d}"
    print(f"\n[Trial {trial.number}] Starting experiment '{exp_name}'")
    print(f"  lr={learning_rate:.2e}, gamma={gamma:.4f}, gae_lambda={gae_lambda:.3f}, "
          f"clip={clip_coef:.3f}, ent={ent_coef:.2e}, vf={vf_coef:.3f}, "
          f"epochs={update_epochs}, minibatches={num_minibatches}, grad_norm={max_grad_norm:.3f}, "
          f"decay={reward_decay_factor:.3f}")

    from bidding_gridworld.single_agent_ppo import SingleAgentArgs
    from bidding_gridworld.experiment import PPOMovingTargetsExperiment

    args = SingleAgentArgs(
        exp_name=exp_name,
        seed=SEED,
        track=TRACK,

        # Environment
        grid_size=GRID_SIZE,
        num_targets=NUM_TARGETS,
        target_reward=TARGET_REWARD,
        max_steps=MAX_STEPS,
        distance_reward_scale=DISTANCE_REWARD_SCALE,
        target_expiry_steps=TARGET_EXPIRY_STEPS,
        target_expiry_penalty=TARGET_EXPIRY_PENALTY,
        reward_decay_factor=reward_decay_factor,
        moving_targets=MOVING_TARGETS,
        direction_change_prob=DIRECTION_CHANGE_PROB,
        target_move_interval=TARGET_MOVE_INTERVAL,

        # Training (sampled)
        num_iterations=NUM_ITERATIONS,
        learning_rate=learning_rate,
        num_envs=NUM_ENVS,
        num_steps=NUM_STEPS,
        num_minibatches=num_minibatches,
        update_epochs=update_epochs,
        anneal_lr=ANNEAL_LR,
        gamma=gamma,
        gae_lambda=gae_lambda,
        norm_adv=NORM_ADV,
        clip_coef=clip_coef,
        clip_vloss=CLIP_VLOSS,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        target_kl=TARGET_KL,

        # Network (fixed)
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

    # ------------------------------------------------------------------
    # Read the final eval stats file written by on_training_end.
    # PPOMovingTargetsExperiment saves it as iter_{NUM_ITERATIONS}_eval_stats.json.
    # ------------------------------------------------------------------
    pattern = os.path.join(
        BASE_LOG_DIR,
        f"{exp_name}_*",
        "rollouts",
        f"iter_{NUM_ITERATIONS}_eval_stats.json",
    )
    matches = glob.glob(pattern)

    if not matches:
        print(f"[Trial {trial.number}] WARNING: final eval stats not found — pattern: {pattern}")
        return 0.0

    fpath = matches[0]
    try:
        with open(fpath) as f:
            data = json.load(f)
        score = float(data.get("statistics", {}).get("avg_min_targets_reached", 0.0))
    except Exception as e:
        print(f"[Trial {trial.number}] Error reading {fpath}: {e}")
        return 0.0

    print(f"[Trial {trial.number}] Final avg_min_targets_reached = {score:.4f}")
    return score


# ============================================================================
# WORKER SUBPROCESS ENTRY POINT
# ============================================================================

def _worker_main(
    worker_id: int,
    storage_url: str,
    study_name: str,
    n_trials: int,
    log_path: str,
    gpu_id: int = 0,
) -> None:
    """Run n_trials Optuna trials in-process within a worker subprocess."""
    # Pin this worker to a single GPU before any CUDA-touching import.
    # CUDA_VISIBLE_DEVICES must be set before torch is imported.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Add src to path (this is a spawned process, so we must re-add it)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "w", buffering=1) as f:
        sys.stdout = f
        sys.stderr = f

        print(f"{'='*72}")
        print(f"  Worker {worker_id} — {n_trials} trials")
        print(f"  Study: {study_name}")
        print(f"  Storage: {storage_url}")
        print(f"{'='*72}")

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            best_path = os.path.join(BASE_LOG_DIR, "best_so_far.json")

            def _save_best(study, trial):
                if trial.state != optuna.trial.TrialState.COMPLETE:
                    return
                try:
                    best = study.best_trial
                except ValueError:
                    return  # no completed trials yet
                n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                payload = {
                    "best_trial_number": best.number,
                    "best_value": best.value,
                    "best_params": best.params,
                    "n_completed": n_done,
                }
                tmp = best_path + ".tmp"
                with open(tmp, "w") as _f:
                    json.dump(payload, _f, indent=2)
                os.replace(tmp, best_path)  # atomic on POSIX
                print(f"[Trial {trial.number}] best so far: #{best.number} = {best.value:.4f}  ({n_done} done)")

            study = optuna.load_study(study_name=study_name, storage=storage_url)
            study.optimize(objective, n_trials=n_trials, callbacks=[_save_best])
            print(f"\nWorker {worker_id} completed {n_trials} trials.")
        except Exception as e:
            print(f"\nWorker {worker_id} FAILED: {e}")
            traceback.print_exc()
            raise


# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(BASE_LOG_DIR, exist_ok=True)

    storage_url = f"sqlite:///{os.path.abspath(BASE_LOG_DIR)}/optuna_study.db"

    # Create (or resume) the study in the main process
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
    )

    existing = len(study.trials)
    remaining = max(0, N_TRIALS_TOTAL - existing)

    print()
    print("=" * 72)
    print("  Optuna HPO — Single-Agent PPO (reward_decay_factor=0.7)")
    print(f"  Study:      {STUDY_NAME}")
    print(f"  Storage:    {storage_url}")
    print(f"  Trials:     {N_TRIALS_TOTAL} total  ({existing} done, {remaining} remaining)")
    print(f"  Workers:    {N_WORKERS}")
    print("=" * 72)

    if remaining == 0:
        print("\n  All trials already completed — printing results only.\n")
    else:
        ctx = multiprocessing.get_context("spawn")

        import torch as _torch
        n_gpus = max(1, _torch.cuda.device_count())

        trials_per_worker, remainder = divmod(remaining, N_WORKERS)

        procs = []
        for worker_id in range(N_WORKERS):
            n_trials = trials_per_worker + (1 if worker_id < remainder else 0)
            if n_trials == 0:
                continue
            gpu_id = worker_id % n_gpus
            log_path = os.path.join(BASE_LOG_DIR, f"worker_{worker_id}.log")
            p = ctx.Process(
                target=_worker_main,
                args=(worker_id, storage_url, STUDY_NAME, n_trials, log_path),
                kwargs={"gpu_id": gpu_id},
                name=f"optuna_worker_{worker_id}",
            )
            procs.append((worker_id, gpu_id, log_path, p))

        print()
        for worker_id, gpu_id, log_path, p in procs:
            p.start()
            print(f"  [PID {p.pid:>6}]  Worker {worker_id}  cuda:{gpu_id}  "
                  f"({trials_per_worker + (1 if worker_id < remainder else 0)} trials)"
                  f"  log → {log_path}")
        print()

        for worker_id, gpu_id, log_path, p in procs:
            p.join()

        print()
        print("=" * 72)
        print("  Worker results")
        print("=" * 72)
        any_failed = False
        for worker_id, gpu_id, log_path, p in procs:
            status = "COMPLETED" if p.exitcode == 0 else f"FAILED (exit {p.exitcode})"
            if p.exitcode != 0:
                any_failed = True
            print(f"  Worker {worker_id} (cuda:{gpu_id}): {status:<30}  log → {log_path}")
        print()

        if any_failed:
            print("  WARNING: some workers failed — results may be incomplete.\n")

    # ------------------------------------------------------------------
    # Reload study and print summary
    # ------------------------------------------------------------------
    study = optuna.load_study(study_name=STUDY_NAME, storage=storage_url)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print()
    print("=" * 72)
    print(f"  Study Summary — {len(completed)} completed trials")
    print("=" * 72)

    if not completed:
        print("  No completed trials found.")
        return

    sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)
    top_n = min(10, len(sorted_trials))

    print(f"\n  Top {top_n} trials (by final avg_min_targets_reached):\n")
    header = f"  {'Rank':<5} {'#Trial':>7} {'Score':>8}  Params"
    print(header)
    print(f"  {'-'*5} {'-'*7} {'-'*8}  {'-'*50}")
    for rank, t in enumerate(sorted_trials[:top_n], start=1):
        params_str = "  ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in sorted(t.params.items())
        )
        print(f"  {rank:<5} {t.number:>7} {t.value:>8.4f}  {params_str}")

    best = sorted_trials[0]
    print(f"\n  Best trial: #{best.number}  score={best.value:.4f}")
    print("  Best params:")
    for k, v in sorted(best.params.items()):
        fmt = f"{v:.6g}" if isinstance(v, float) else str(v)
        print(f"    {k}: {fmt}")

    # Save summary JSON
    summary = {
        "study_name": STUDY_NAME,
        "storage_url": storage_url,
        "n_trials_completed": len(completed),
        "best_trial": {
            "number": best.number,
            "value": best.value,
            "params": best.params,
        },
        "top_10_trials": [
            {
                "rank": rank,
                "number": t.number,
                "value": t.value,
                "params": t.params,
            }
            for rank, t in enumerate(sorted_trials[:top_n], start=1)
        ],
    }
    summary_path = os.path.join(BASE_LOG_DIR, "study_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Study summary saved to {summary_path}")
    print()


if __name__ == "__main__":
    main()
