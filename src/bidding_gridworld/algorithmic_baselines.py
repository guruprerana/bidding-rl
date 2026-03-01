"""
Algorithmic baseline policies for BiddingGridworld (single-agent mode).

These policies serve as oracle upper-bounds: they navigate optimally (one
Manhattan step at a time) but differ in how they choose which target to
pursue.

Observation layout (single_agent_mode=True, moving_targets=True):
  obs = [agent_pos(2), target_pos(2*n), target_counters(n), window_steps(1), relative_counts(n)]
  - agent_pos      = obs[0:2] * (grid_size - 1)         → int [row, col]
  - target_pos     = obs[2:2+2n].reshape(n,2) * (grid_size-1)  → int (n,2)
  - counter_norm   = obs[2+2n:2+3n]                     → float in [0,1]
                     (0=just spawned, 1=at expiry)
  - window_steps   = obs[2+3n]                           → float
  - relative_counts= obs[3+3n:3+4n]                     → float

Actions (directions):
  0 = col - 1 (left)
  1 = col + 1 (right)
  2 = row - 1 (up)
  3 = row + 1 (down)
"""

from __future__ import annotations

from typing import Optional, List, Dict

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Navigation helper
# ---------------------------------------------------------------------------

def _move_toward(agent_row: int, agent_col: int, target_row: int, target_col: int) -> int:
    """
    Return a single direction that moves (agent_row, agent_col) one step
    closer to (target_row, target_col) using Manhattan distance.

    Prefers the row axis when both row and column differ.
    If already at target, returns 2 (up, a no-op that stays in-bounds).
    """
    dr = target_row - agent_row
    dc = target_col - agent_col

    if dr == 0 and dc == 0:
        return 2  # already there; any valid direction

    # Prefer row axis on tie
    if abs(dr) >= abs(dc):
        return 3 if dr > 0 else 2
    else:
        return 1 if dc > 0 else 0


def _decode_obs(obs: np.ndarray, grid_size: int, num_targets: int):
    """
    Decode a flat single-agent observation into (agent_pos, target_pos, counter_norm).

    Returns:
        agent_pos   : np.ndarray shape (2,) int, [row, col]
        target_pos  : np.ndarray shape (n, 2) int, [row, col]
        counter_norm: np.ndarray shape (n,) float in [0, 1]
    """
    n = num_targets
    denom = float(grid_size - 1) if grid_size > 1 else 1.0

    agent_pos = np.round(obs[0:2] * denom).astype(int)
    target_pos = np.round(obs[2:2 + 2 * n].reshape(n, 2) * denom).astype(int)
    counter_norm = obs[2 + 2 * n: 2 + 3 * n]

    return agent_pos, target_pos, counter_norm


# ---------------------------------------------------------------------------
# Base policy
# ---------------------------------------------------------------------------

class _BasePolicy:
    def __init__(self, num_targets: int, grid_size: int, target_expiry_steps: Optional[int] = None):
        self.num_targets = num_targets
        self.grid_size = grid_size
        self.target_expiry_steps = target_expiry_steps

    def reset(self) -> None:
        """Called at the start of each episode."""
        pass

    def act(self, obs: np.ndarray, targets_just_reached: Optional[np.ndarray]) -> int:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Policy 1: Round-Robin
# ---------------------------------------------------------------------------

class RoundRobinPolicy(_BasePolicy):
    """
    Cycles through targets in order 0, 1, ..., n-1, 0, 1, ...
    Advances to the next target when the current one is just reached.
    """

    def reset(self) -> None:
        self._idx = 0

    def act(self, obs: np.ndarray, targets_just_reached: Optional[np.ndarray]) -> int:
        n = self.num_targets

        # Advance index if current target was just reached
        if targets_just_reached is not None and targets_just_reached[self._idx]:
            self._idx = (self._idx + 1) % n

        agent_pos, target_pos, _ = _decode_obs(obs, self.grid_size, n)
        t = target_pos[self._idx]
        return _move_toward(agent_pos[0], agent_pos[1], t[0], t[1])


# ---------------------------------------------------------------------------
# Policy 2: Nearest Target
# ---------------------------------------------------------------------------

class NearestTargetPolicy(_BasePolicy):
    """
    At each step, navigates toward the closest target by Manhattan distance.
    """

    def act(self, obs: np.ndarray, targets_just_reached: Optional[np.ndarray]) -> int:
        n = self.num_targets
        agent_pos, target_pos, _ = _decode_obs(obs, self.grid_size, n)

        distances = np.abs(target_pos[:, 0] - agent_pos[0]) + np.abs(target_pos[:, 1] - agent_pos[1])
        chosen = int(np.argmin(distances))

        t = target_pos[chosen]
        return _move_toward(agent_pos[0], agent_pos[1], t[0], t[1])


# ---------------------------------------------------------------------------
# Policy 3: Least Time Left
# ---------------------------------------------------------------------------

class LeastTimeLeftPolicy(_BasePolicy):
    """
    Prioritises the target closest to expiry (highest counter_norm).
    Falls back to NearestTargetPolicy when target_expiry_steps is None.
    """

    def act(self, obs: np.ndarray, targets_just_reached: Optional[np.ndarray]) -> int:
        n = self.num_targets
        agent_pos, target_pos, counter_norm = _decode_obs(obs, self.grid_size, n)

        if self.target_expiry_steps is None:
            # No expiry — fall back to nearest target
            distances = np.abs(target_pos[:, 0] - agent_pos[0]) + np.abs(target_pos[:, 1] - agent_pos[1])
            chosen = int(np.argmin(distances))
        else:
            chosen = int(np.argmax(counter_norm))

        t = target_pos[chosen]
        return _move_toward(agent_pos[0], agent_pos[1], t[0], t[1])


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

def evaluate_algorithmic_policy(
    env,
    policy: _BasePolicy,
    num_episodes: int,
    target_expiry_penalty: float = 0.0,
    verbose: bool = True,
) -> Dict[str, List]:
    """
    Evaluate an algorithmic policy on a BiddingGridworld (single-agent mode,
    num_envs=1).

    Mirrors evaluate_single_agent_policy() in bidding_gridworld_torch.py
    exactly, adding policy.reset() at episode start and passing
    targets_just_reached to policy.act().

    Returns the same stats dict:
        episode_returns, episode_lengths, targets_reached_per_episode,
        expired_targets_per_episode, min_targets_reached_per_episode,
        targets_reached_count_per_episode
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating {policy.__class__.__name__}")
        print(f"Running {num_episodes} episodes")
        print(f"{'='*60}\n")

    eval_stats = {
        "episode_returns": [],
        "episode_lengths": [],
        "targets_reached_per_episode": [],
        "expired_targets_per_episode": [],
        "min_targets_reached_per_episode": [],
        "targets_reached_count_per_episode": [],
        "expired_count_per_target_per_episode": [],
        "avg_expired_per_episode": [],
        "max_expired_per_episode": [],
        "avg_reached_per_episode": [],
        "performance_per_episode": [],
        "avg_performance_per_episode": [],
        "min_performance_per_episode": [],
    }

    for episode_idx in range(num_episodes):
        obs, _ = env.reset()
        policy.reset()

        episode_return = 0.0
        step_count = 0
        terminated = False
        truncated = False

        targets_reached_count = np.zeros(env.num_agents, dtype=np.int32)
        expired_targets_count = np.zeros(env.num_agents, dtype=np.int32)
        targets_just_reached = None

        while not (terminated or truncated):
            direction = policy.act(obs[0].cpu().numpy() if torch.is_tensor(obs) else obs[0],
                                   targets_just_reached)

            obs, rewards, terminations, truncations, info = env.step(
                torch.tensor([direction], device=env.device)
            )
            terminated = bool(terminations[0].item())
            truncated = bool(truncations[0].item())

            reward_val = float(rewards[0].item())
            episode_return += reward_val

            tje = info.get("targets_just_expired")
            if isinstance(tje, torch.Tensor):
                just_expired = tje[0].detach().cpu().numpy().astype(bool)
                expired_targets_count += just_expired.astype(int)

            tjr_tensor = info.get("targets_just_reached")
            if isinstance(tjr_tensor, torch.Tensor):
                tjr = tjr_tensor[0].detach().cpu().numpy().astype(bool)
                targets_reached_count += tjr.astype(int)
                targets_just_reached = tjr
            else:
                targets_just_reached = None

            step_count += 1

        targets_reached = int(np.sum(targets_reached_count > 0))
        min_targets_reached = int(np.min(targets_reached_count))
        episode_expired_count = int(expired_targets_count.sum())
        performance = targets_reached_count - expired_targets_count

        eval_stats["episode_returns"].append(episode_return)
        eval_stats["episode_lengths"].append(step_count)
        eval_stats["targets_reached_per_episode"].append(targets_reached)
        eval_stats["expired_targets_per_episode"].append(episode_expired_count)
        eval_stats["min_targets_reached_per_episode"].append(min_targets_reached)
        eval_stats["targets_reached_count_per_episode"].append(targets_reached_count.tolist())
        eval_stats["expired_count_per_target_per_episode"].append(expired_targets_count.tolist())
        eval_stats["avg_expired_per_episode"].append(float(np.mean(expired_targets_count)))
        eval_stats["max_expired_per_episode"].append(float(np.max(expired_targets_count)))
        eval_stats["avg_reached_per_episode"].append(float(np.mean(targets_reached_count)))
        eval_stats["performance_per_episode"].append(performance.tolist())
        eval_stats["avg_performance_per_episode"].append(float(np.mean(performance)))
        eval_stats["min_performance_per_episode"].append(float(np.min(performance)))

        if verbose:
            print(f"  Episode {episode_idx + 1}: Return={episode_return:.2f}, "
                  f"Length={step_count}, Targets={targets_reached}/{env.num_agents}, "
                  f"Expired={episode_expired_count}, MinReached={min_targets_reached}, "
                  f"AvgPerf={float(np.mean(performance)):.2f}")

    if verbose:
        avg_return = np.mean(eval_stats["episode_returns"])
        avg_length = np.mean(eval_stats["episode_lengths"])
        avg_targets = np.mean(eval_stats["targets_reached_per_episode"])
        avg_expired = np.mean(eval_stats["expired_targets_per_episode"])
        avg_min_reached = np.mean(eval_stats["min_targets_reached_per_episode"])
        avg_avg_perf = np.mean(eval_stats["avg_performance_per_episode"])
        avg_min_perf = np.mean(eval_stats["min_performance_per_episode"])
        success_rate = sum(1 for t in eval_stats["targets_reached_per_episode"]
                          if t == env.num_agents) / num_episodes

        print("\nEvaluation Summary:")
        print(f"  Average Return:      {avg_return:.2f}")
        print(f"  Average Length:      {avg_length:.1f}")
        print(f"  Average Targets:     {avg_targets:.2f}/{env.num_agents}")
        print(f"  Average Expired:     {avg_expired:.2f} ± {np.std(eval_stats['expired_targets_per_episode']):.2f}")
        print(f"  Average Min Reached: {avg_min_reached:.2f} ± {np.std(eval_stats['min_targets_reached_per_episode']):.2f}")
        print(f"  Avg Performance (reaches-exp): {avg_avg_perf:.2f}")
        print(f"  Avg Min Performance: {avg_min_perf:.2f}")
        print(f"  Success Rate:        {success_rate*100:.1f}%\n")

    return eval_stats
