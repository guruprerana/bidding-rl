"""
Torch-batched GPU environment skeleton for BiddingGridworld.

This is a starting point for a full CUDA-native env. It mirrors the
state/step layout of BiddingGridworld but keeps all tensors on GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2


@dataclass
class BiddingGridworldConfig:
    grid_size: int
    num_agents: int
    bid_upper_bound: int
    bid_penalty: float
    target_reward: float
    max_steps: int
    action_window: int
    distance_reward_scale: float
    target_expiry_steps: Optional[int]
    target_expiry_penalty: float
    moving_targets: bool
    direction_change_prob: float
    target_move_interval: int
    window_bidding: bool
    window_penalty: float
    visible_targets: Optional[int]
    single_agent_mode: bool = False
    reward_decay_factor: float = 0.0
    bidding_mechanism: str = "all_pay"
    nearest_target_shaping: bool = False
    nearest_expiry_shaping: bool = False
    # "all_pay" | "winner_pays" | "winner_pays_others_reward"


class BiddingGridworld:
    """
    GPU-native batched env for BiddingGridworld.

    Intended to be used directly in PPO trainers to avoid CPU env stepping.
    All state is stored in CUDA tensors and updated with torch ops.
    """

    def __init__(
        self,
        config: BiddingGridworldConfig,
        num_envs: int,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config
        self.num_envs = num_envs
        self.grid_size = config.grid_size
        self.num_agents = config.num_agents
        self.window_bidding = config.window_bidding
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(seed)

        # Core state (tensors allocated on reset)
        self.agent_pos = None
        self.target_pos = None
        self.targets_reached = None
        self.targets_reached_count = None
        self.target_counters = None
        self.window_agent = None
        self.window_steps_remaining = None
        self.step_count = None
        self.previous_distances = None

        # Moving-target state
        self.target_directions = None
        self.target_move_counters = None

        # Precompute position cache for reset sampling (exclude (0,0))
        self._reset_positions = self._build_reset_positions()

        # Precompute per-agent reorder indices for centralized observations
        if self.config.visible_targets is None and not self.config.single_agent_mode:
            base = torch.arange(self.config.num_agents, device=self.device)
            reorder = []
            for agent_id in range(self.config.num_agents):
                reorder.append(torch.cat([base[agent_id:agent_id + 1], base[:agent_id], base[agent_id + 1:]]))
            self._reorder_idx = torch.stack(reorder, dim=0)
        else:
            self._reorder_idx = None

        # Precompute diagonal mask for visible_targets
        if self.config.visible_targets is not None and not self.config.single_agent_mode:
            self._diag_mask = torch.eye(self.config.num_agents, device=self.device, dtype=torch.bool)
        else:
            self._diag_mask = None

        include_reached = not self.config.moving_targets
        if self.config.single_agent_mode:
            base_dim = 3 + (5 if include_reached else 4) * self.config.num_agents
            self.obs_dim = base_dim
            self.obs_shape = (self.num_envs, self.obs_dim)
            self.per_agent_obs_dim = None
        else:
            if self.config.visible_targets is None:
                self.per_agent_obs_dim = 3 + (4 if include_reached else 3) * self.config.num_agents
            else:
                self.per_agent_obs_dim = 7 + 3 * self.config.visible_targets if include_reached else 6 + 2 * self.config.visible_targets
            self.obs_dim = None
            self.obs_shape = (self.num_envs, self.config.num_agents, self.per_agent_obs_dim)

    def reset(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        if seed is not None:
            self.gen.manual_seed(seed)

        cfg = self.config
        device = self.device

        # Agent starts at (0, 0)
        self.agent_pos = torch.zeros((self.num_envs, 2), device=device, dtype=torch.int32)

        # Targets: sample distinct positions per env (vectorized)
        if cfg.num_agents > self._reset_positions.shape[0]:
            raise ValueError("num_agents exceeds available grid positions")
        rand = torch.rand((self.num_envs, self._reset_positions.shape[0]), generator=self.gen, device=device)
        idx = torch.topk(rand, k=cfg.num_agents, dim=1, largest=True).indices
        self.target_pos = self._reset_positions[idx].to(torch.int32)

        self.targets_reached = torch.zeros((self.num_envs, cfg.num_agents), device=device, dtype=torch.int32)
        self.targets_reached_count = torch.zeros((self.num_envs, cfg.num_agents), device=device, dtype=torch.int32)
        self.target_counters = torch.zeros((self.num_envs, cfg.num_agents), device=device, dtype=torch.int32)
        self.window_agent = torch.full((self.num_envs,), -1, device=device, dtype=torch.int32)
        self.window_steps_remaining = torch.zeros((self.num_envs,), device=device, dtype=torch.int32)
        self.step_count = torch.zeros((self.num_envs,), device=device, dtype=torch.int32)
        self.previous_distances = self._compute_distances()

        if cfg.moving_targets:
            # Directions: 0/1/2/3 for left/right/up/down
            self.target_directions = torch.randint(
                0, 4, (self.num_envs, cfg.num_agents), generator=self.gen, device=device, dtype=torch.int32
            )
            self.target_move_counters = torch.zeros((self.num_envs, cfg.num_agents), device=device, dtype=torch.int32)

        obs = self._get_observation()
        info: Dict = {}
        return obs, info

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step the environment with batched actions.

        action shape:
        - multi-agent: (num_envs, num_agents, 2) or (num_envs, num_agents, 3) if window_bidding
        - single-agent: (num_envs,) or (num_envs, 1)
        """
        cfg = self.config
        device = self.device

        self.step_count = self.step_count + 1

        if cfg.single_agent_mode:
            action_dir = action.view(self.num_envs).to(torch.int64)
            move_mask = torch.ones((self.num_envs,), device=device, dtype=torch.bool)
            winning_agent = torch.zeros((self.num_envs,), device=device, dtype=torch.int32)
            apply_bid_penalty = torch.zeros((self.num_envs,), device=device, dtype=torch.bool)
            current_window_length = torch.zeros((self.num_envs,), device=device, dtype=torch.int32)
            bids = None
        else:
            action = action.to(torch.int64)
            action_dir = action[..., 0]
            bids = action[..., 1]
            action_window = action[..., 2] if cfg.window_bidding else None

            in_window = self.window_steps_remaining > 0
            apply_bid_penalty = torch.zeros((self.num_envs,), device=device, dtype=torch.bool)
            current_window_length = torch.zeros((self.num_envs,), device=device, dtype=torch.int32)

            if torch.any(in_window):
                self.window_steps_remaining = torch.where(
                    in_window, self.window_steps_remaining - 1, self.window_steps_remaining
                )

            max_bid = bids.max(dim=1).values
            has_bid = (max_bid > 0) | (cfg.bid_upper_bound == 0)
            winners_mask = bids == max_bid.unsqueeze(1)
            rand = torch.rand(bids.shape, device=device, generator=self.gen)
            rand = torch.where(winners_mask, rand, torch.full_like(rand, -1.0))
            winner = rand.argmax(dim=1).to(torch.int32)
            winner_long = winner.to(torch.int64)

            winning_agent = torch.where(in_window, self.window_agent, winner)
            winning_agent = torch.where(has_bid | in_window, winning_agent, torch.full_like(winning_agent, -1))

            if cfg.window_bidding:
                chosen_window = torch.zeros((self.num_envs,), device=device, dtype=torch.int32)
                if torch.any(has_bid & (~in_window)):
                    chosen_window = torch.where(
                        has_bid & (~in_window),
                        action_window.gather(1, winner_long.view(-1, 1)).squeeze(1) + 1,
                        chosen_window,
                    )
                current_window_length = torch.where(
                    has_bid & (~in_window),
                    chosen_window,
                    current_window_length,
                )
                self.window_steps_remaining = torch.where(
                    has_bid & (~in_window),
                    chosen_window - 1,
                    self.window_steps_remaining,
                )
            else:
                self.window_steps_remaining = torch.where(
                    has_bid & (~in_window),
                    torch.full_like(self.window_steps_remaining, cfg.action_window - 1),
                    self.window_steps_remaining,
                )
                current_window_length = torch.where(
                    has_bid & (~in_window),
                    torch.full_like(current_window_length, cfg.action_window),
                    current_window_length,
                )

            apply_bid_penalty = has_bid & (~in_window)
            self.window_agent = torch.where(
                has_bid & (~in_window),
                winner,
                self.window_agent,
            )
            self.window_agent = torch.where(
                (~has_bid) & (~in_window),
                torch.full_like(self.window_agent, -1),
                self.window_agent,
            )

            move_mask = winning_agent >= 0

        if cfg.single_agent_mode:
            move_dir = action_dir
        else:
            move_dir = action_dir.gather(1, winning_agent.clamp(min=0).to(torch.int64).view(-1, 1)).squeeze(1)

        new_pos = self._move_position(self.agent_pos, move_dir)
        self.agent_pos = torch.where(move_mask.unsqueeze(-1), new_pos, self.agent_pos)

        # Targets reached
        targets_just_reached = self._positions_equal(self.agent_pos, self.target_pos) & (self.targets_reached == 0)
        self.targets_reached = torch.where(targets_just_reached, torch.ones_like(self.targets_reached), self.targets_reached)
        self.targets_reached_count = self.targets_reached_count + targets_just_reached.to(torch.int32)
        self.target_counters = torch.where(targets_just_reached, torch.zeros_like(self.target_counters), self.target_counters)

        # Target expiry
        if cfg.target_expiry_steps is not None:
            not_reached = self.targets_reached == 0
            self.target_counters = self.target_counters + not_reached.to(torch.int32)
            targets_expired = not_reached & (self.target_counters >= cfg.target_expiry_steps)
            self.target_counters = torch.where(targets_expired, torch.zeros_like(self.target_counters), self.target_counters)
        else:
            targets_expired = torch.zeros_like(self.targets_reached, dtype=torch.bool)

        # Rewards
        current_distances = self._compute_distances()
        if cfg.single_agent_mode:
            rewards = torch.zeros((self.num_envs,), device=device, dtype=torch.float32)
            if cfg.distance_reward_scale > 0:
                unreached = self.targets_reached == 0
                if cfg.nearest_target_shaping:
                    INF = float(self.grid_size * 2 + 1)
                    prev_masked = self.previous_distances.to(torch.float32).masked_fill(~unreached, INF)
                    curr_masked = current_distances.to(torch.float32).masked_fill(~unreached, INF)
                    has_unreached = unreached.any(dim=1)
                    nearest_improve = torch.where(
                        has_unreached,
                        prev_masked.min(dim=1).values - curr_masked.min(dim=1).values,
                        torch.zeros(self.num_envs, device=device),
                    )
                    rewards = rewards + cfg.distance_reward_scale * nearest_improve
                elif cfg.nearest_expiry_shaping and cfg.target_expiry_steps is not None:
                    # Shape toward the unreached target nearest to expiry (highest counter)
                    counters_f = self.target_counters.to(torch.float32).masked_fill(~unreached, -1.0)
                    has_unreached = unreached.any(dim=1)
                    expiry_idx = counters_f.argmax(dim=1).unsqueeze(1)  # (num_envs, 1)
                    prev_expiry = self.previous_distances.to(torch.float32).gather(1, expiry_idx).squeeze(1)
                    curr_expiry = current_distances.to(torch.float32).gather(1, expiry_idx).squeeze(1)
                    expiry_improve = torch.where(
                        has_unreached,
                        prev_expiry - curr_expiry,
                        torch.zeros(self.num_envs, device=device),
                    )
                    rewards = rewards + cfg.distance_reward_scale * expiry_improve
                else:
                    dist_improve = (self.previous_distances - current_distances).to(torch.float32)
                    rewards = rewards + cfg.distance_reward_scale * (dist_improve * unreached.to(torch.float32)).sum(dim=1)

            if cfg.reward_decay_factor > 0:
                min_count = self.targets_reached_count.min(dim=1).values
                relative_count = (self.targets_reached_count - min_count.unsqueeze(1)).to(torch.float32)
                decay = torch.exp(-cfg.reward_decay_factor * relative_count)
                rewards = rewards + (targets_just_reached.to(torch.float32) * (cfg.target_reward * decay)).sum(dim=1)
            else:
                rewards = rewards + (targets_just_reached.to(torch.float32) * cfg.target_reward).sum(dim=1)

            if cfg.target_expiry_penalty > 0:
                rewards = rewards - cfg.target_expiry_penalty * targets_expired.to(torch.float32).sum(dim=1)
        else:
            rewards = torch.zeros((self.num_envs, cfg.num_agents), device=device, dtype=torch.float32)
            bid_net_effect = torch.zeros_like(rewards)
            if torch.any(apply_bid_penalty) and bids is not None:
                bids_f = bids.to(torch.float32)
                mask = apply_bid_penalty.unsqueeze(1).to(torch.float32)

                if cfg.bidding_mechanism == "all_pay":
                    effect = mask * cfg.bid_penalty * bids_f
                    rewards = rewards - effect
                    bid_net_effect = bid_net_effect - effect
                else:
                    # Build winner mask: (num_envs, num_agents), 1.0 only at winning agent index
                    winner_mask = torch.zeros((self.num_envs, cfg.num_agents), device=device, dtype=torch.float32)
                    valid_win = winning_agent >= 0
                    if torch.any(valid_win):
                        idx = winning_agent.clamp(min=0).long().view(-1, 1)
                        winner_mask.scatter_(1, idx, 1.0)
                        winner_mask = winner_mask * valid_win.float().unsqueeze(1)

                    if cfg.bidding_mechanism == "winner_pays":
                        effect = mask * cfg.bid_penalty * winner_mask * bids_f
                        rewards = rewards - effect
                        bid_net_effect = bid_net_effect - effect
                    elif cfg.bidding_mechanism == "winner_pays_others_reward":
                        others_mask = 1.0 - winner_mask
                        win_effect = mask * cfg.bid_penalty * winner_mask * bids_f
                        other_effect = mask * cfg.bid_penalty * others_mask * bids_f
                        rewards = rewards - win_effect + other_effect
                        bid_net_effect = bid_net_effect - win_effect + other_effect

                if cfg.window_bidding and cfg.window_penalty > 0:
                    penalty = cfg.window_penalty * current_window_length.to(torch.float32)
                    valid_win = winning_agent >= 0
                    if torch.any(valid_win):
                        idx = winning_agent.clamp(min=0).to(torch.int64).view(-1, 1)
                        pen_vec = (-penalty * valid_win.to(torch.float32)).view(-1, 1)
                        rewards = rewards.scatter_add(1, idx, pen_vec)
                        bid_net_effect = bid_net_effect.scatter_add(1, idx, pen_vec)

            if cfg.distance_reward_scale > 0:
                dist_improve = (self.previous_distances - current_distances).to(torch.float32)
                rewards = rewards + cfg.distance_reward_scale * dist_improve * (self.targets_reached == 0).to(torch.float32)

            rewards = rewards + cfg.target_reward * targets_just_reached.to(torch.float32)
            if cfg.target_expiry_penalty > 0:
                rewards = rewards - cfg.target_expiry_penalty * targets_expired.to(torch.float32)

        # Per-objective (per-target) rewards for DWN and similar multi-objective methods
        if cfg.single_agent_mode:
            per_obj = torch.zeros((self.num_envs, cfg.num_agents), device=device, dtype=torch.float32)
            if cfg.distance_reward_scale > 0:
                dist_improve = (self.previous_distances - current_distances).to(torch.float32)
                per_obj = per_obj + cfg.distance_reward_scale * dist_improve * (self.targets_reached == 0).to(torch.float32)
            if cfg.reward_decay_factor > 0:
                min_count = self.targets_reached_count.min(dim=1).values
                relative_count = (self.targets_reached_count - min_count.unsqueeze(1)).to(torch.float32)
                decay = torch.exp(-cfg.reward_decay_factor * relative_count)
                per_obj = per_obj + targets_just_reached.to(torch.float32) * (cfg.target_reward * decay)
            else:
                per_obj = per_obj + targets_just_reached.to(torch.float32) * cfg.target_reward
            if cfg.target_expiry_penalty > 0:
                per_obj = per_obj - cfg.target_expiry_penalty * targets_expired.to(torch.float32)

        self.previous_distances = current_distances

        if cfg.moving_targets:
            self._move_targets(targets_just_reached, targets_expired)
            terminated = torch.zeros((self.num_envs,), device=device, dtype=torch.bool)
            truncated = self.step_count >= cfg.max_steps
        else:
            all_targets_reached = self.targets_reached.sum(dim=1) == cfg.num_agents
            terminated = all_targets_reached.to(torch.bool)
            truncated = (self.step_count >= cfg.max_steps) & (~terminated)

        obs = self._get_observation()
        info = {
            "winning_agent": winning_agent,
            "bids": bids,
            "window_agent": self.window_agent,
            "window_steps_remaining": self.window_steps_remaining,
            "bid_penalty_applied": apply_bid_penalty,
            "targets_just_reached": targets_just_reached,
            "targets_just_expired": targets_expired,
        }
        if cfg.single_agent_mode:
            info["per_objective_rewards"] = per_obj
        else:
            info["is_bidding_round"] = ~in_window
            info["reward_no_bid_sum"] = (rewards - bid_net_effect).sum(dim=1)
        return obs, rewards, terminated, truncated, info

    def _get_centralized_observation_tensor(self) -> torch.Tensor:
        """Build centralized observation tensor for all envs."""
        cfg = self.config
        denom = float(cfg.grid_size - 1) if cfg.grid_size > 1 else 1.0

        agent_pos = self.agent_pos.to(torch.float32) / denom
        target_pos = self.target_pos.to(torch.float32) / denom
        include_reached = not cfg.moving_targets
        targets_reached = self.targets_reached.to(torch.float32)

        if cfg.target_expiry_steps is not None:
            counter_denom = float(cfg.target_expiry_steps)
        else:
            counter_denom = float(cfg.max_steps)
        counter_denom = max(counter_denom, 1.0)
        target_counters = self.target_counters.to(torch.float32) / counter_denom

        window_denom = float(max(cfg.action_window, 1))
        window_steps = (self.window_steps_remaining.to(torch.float32) / window_denom).unsqueeze(-1)

        if include_reached:
            base_obs = torch.cat(
                [
                    agent_pos,
                    target_pos.reshape(self.num_envs, -1),
                    targets_reached,
                    target_counters,
                    window_steps,
                ],
                dim=-1,
            )
        else:
            base_obs = torch.cat(
                [
                    agent_pos,
                    target_pos.reshape(self.num_envs, -1),
                    target_counters,
                    window_steps,
                ],
                dim=-1,
            )

        if cfg.single_agent_mode:
            min_count = self.targets_reached_count.min(dim=1).values
            count_denom = float(max(cfg.num_agents, 1))
            relative = (self.targets_reached_count - min_count.unsqueeze(1)).to(torch.float32)
            relative = torch.clamp(relative / count_denom, 0.0, 1.0)
            base_obs = torch.cat([base_obs, relative], dim=-1)

        return base_obs

    def _get_centralized_observation(self) -> np.ndarray:
        """Return centralized observation on CPU (for evaluation/visualization)."""
        return self._get_centralized_observation_tensor().detach().cpu().numpy()

    def _get_observation(self) -> torch.Tensor:
        """
        Build observation tensor matching BiddingGridworld layout.

        For centralized mode: shape (num_envs, obs_dim)
        For decentralized mode: shape (num_envs, num_agents, obs_dim_per_agent)
        """
        cfg = self.config
        denom = float(cfg.grid_size - 1) if cfg.grid_size > 1 else 1.0

        agent_pos = self.agent_pos.to(torch.float32) / denom
        target_pos = self.target_pos.to(torch.float32) / denom
        include_reached = not cfg.moving_targets
        targets_reached = self.targets_reached.to(torch.float32)

        if cfg.target_expiry_steps is not None:
            counter_denom = float(cfg.target_expiry_steps)
        else:
            counter_denom = float(cfg.max_steps)
        counter_denom = max(counter_denom, 1.0)
        target_counters = self.target_counters.to(torch.float32) / counter_denom

        window_denom = float(max(cfg.action_window, 1))
        window_steps = self.window_steps_remaining.to(torch.float32) / window_denom
        window_steps = window_steps.unsqueeze(-1)

        if cfg.single_agent_mode:
            if include_reached:
                base_obs = torch.cat(
                    [
                        agent_pos,
                        target_pos.reshape(self.num_envs, -1),
                        targets_reached,
                        target_counters,
                        window_steps,
                    ],
                    dim=-1,
                )
            else:
                base_obs = torch.cat(
                    [
                        agent_pos,
                        target_pos.reshape(self.num_envs, -1),
                        target_counters,
                        window_steps,
                    ],
                    dim=-1,
                )
            min_count = self.targets_reached_count.min(dim=1).values
            count_denom = float(max(cfg.num_agents, 1))
            relative = (self.targets_reached_count - min_count.unsqueeze(1)).to(torch.float32)
            relative = torch.clamp(relative / count_denom, 0.0, 1.0)
            return torch.cat([base_obs, relative], dim=-1)

        if cfg.visible_targets is None:
            reordered_pos = target_pos[:, self._reorder_idx, :].reshape(self.num_envs, cfg.num_agents, -1)
            reordered_counters = target_counters[:, self._reorder_idx]
            if include_reached:
                reordered_reached = targets_reached[:, self._reorder_idx]
                return torch.cat(
                    [agent_pos.unsqueeze(1).expand(-1, cfg.num_agents, -1),
                     reordered_pos,
                     reordered_reached,
                     reordered_counters,
                     window_steps.unsqueeze(1).expand(-1, cfg.num_agents, -1)],
                    dim=-1,
                )
            return torch.cat(
                [agent_pos.unsqueeze(1).expand(-1, cfg.num_agents, -1),
                 reordered_pos,
                 reordered_counters,
                 window_steps.unsqueeze(1).expand(-1, cfg.num_agents, -1)],
                dim=-1,
            )

        if cfg.visible_targets == 0:
            own_pos = target_pos  # (num_envs, num_agents, 2) — each agent reads its own slice
            own_counter = target_counters.unsqueeze(-1)  # (num_envs, num_agents, 1)
            if include_reached:
                own_reached = targets_reached.unsqueeze(-1)  # (num_envs, num_agents, 1)
                return torch.cat(
                    [
                        agent_pos.unsqueeze(1).expand(-1, cfg.num_agents, -1),
                        own_pos,
                        own_reached,
                        own_counter,
                        window_steps.unsqueeze(1).expand(-1, cfg.num_agents, -1),
                    ],
                    dim=-1,
                )
            return torch.cat(
                [
                    agent_pos.unsqueeze(1).expand(-1, cfg.num_agents, -1),
                    own_pos,
                    own_counter,
                    window_steps.unsqueeze(1).expand(-1, cfg.num_agents, -1),
                ],
                dim=-1,
            )

        # Per-agent observations with visible nearest targets
        distances = self._compute_distances().to(torch.float32)
        dist_all = distances.unsqueeze(1).expand(-1, cfg.num_agents, -1)
        dist_all = dist_all.masked_fill(self._diag_mask.unsqueeze(0), float(cfg.grid_size * 2 + 1))
        idx = torch.topk(dist_all, k=cfg.visible_targets, dim=2, largest=False).indices

        target_pos_exp = target_pos.unsqueeze(1).expand(-1, cfg.num_agents, -1, 2)
        vis_pos = target_pos_exp.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, 2)).reshape(
            self.num_envs, cfg.num_agents, -1
        )
        own_pos = target_pos
        own_counter = target_counters.unsqueeze(-1)

        if include_reached:
            targets_reached_exp = targets_reached.unsqueeze(1).expand(-1, cfg.num_agents, -1)
            vis_reached = targets_reached_exp.gather(2, idx)
            own_reached = targets_reached.unsqueeze(-1)
            return torch.cat(
                [
                    agent_pos.unsqueeze(1).expand(-1, cfg.num_agents, -1),
                    own_pos,
                    vis_pos,
                    own_reached,
                    vis_reached,
                    own_counter,
                    window_steps.unsqueeze(1).expand(-1, cfg.num_agents, -1),
                ],
                dim=-1,
            )
        return torch.cat(
            [
                agent_pos.unsqueeze(1).expand(-1, cfg.num_agents, -1),
                own_pos,
                vis_pos,
                own_counter,
                window_steps.unsqueeze(1).expand(-1, cfg.num_agents, -1),
            ],
            dim=-1,
        )

    def _compute_distances(self) -> torch.Tensor:
        """Compute manhattan distances from agent to each target."""
        diff = (self.agent_pos.unsqueeze(1) - self.target_pos).abs()
        return diff.sum(dim=-1)

    def _positions_equal(self, pos_a: torch.Tensor, pos_b: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask of equality between (num_envs, 2) and (num_envs, num_agents, 2)."""
        return (pos_b[..., 0] == pos_a[:, 0].unsqueeze(1)) & (pos_b[..., 1] == pos_a[:, 1].unsqueeze(1))

    def _move_position(self, position: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """Move positions by one step based on direction tensor."""
        row = position[..., 0]
        col = position[..., 1]

        delta_row = torch.zeros_like(row)
        delta_col = torch.zeros_like(col)

        delta_col = torch.where(direction == 0, delta_col - 1, delta_col)
        delta_col = torch.where(direction == 1, delta_col + 1, delta_col)
        delta_row = torch.where(direction == 2, delta_row - 1, delta_row)
        delta_row = torch.where(direction == 3, delta_row + 1, delta_row)

        new_row = (row + delta_row).clamp(0, self.config.grid_size - 1)
        new_col = (col + delta_col).clamp(0, self.config.grid_size - 1)

        return torch.stack([new_row, new_col], dim=-1)

    def _move_targets(self, targets_just_reached: torch.Tensor, targets_expired: torch.Tensor) -> None:
        """Move or respawn targets for moving-target variant."""
        cfg = self.config
        device = self.device

        respawn_mask = targets_just_reached | targets_expired
        if torch.any(respawn_mask):
            new_pos = self._sample_positions_excluding_agent(respawn_mask)
            self.target_pos = torch.where(respawn_mask.unsqueeze(-1), new_pos, self.target_pos)
            self.targets_reached = torch.where(respawn_mask, torch.zeros_like(self.targets_reached), self.targets_reached)
            self.target_counters = torch.where(respawn_mask, torch.zeros_like(self.target_counters), self.target_counters)
            self.target_move_counters = torch.where(respawn_mask, torch.zeros_like(self.target_move_counters), self.target_move_counters)
            new_dirs = torch.randint(
                0, 4, self.target_directions.shape, generator=self.gen, device=device, dtype=torch.int32
            )
            self.target_directions = torch.where(respawn_mask, new_dirs, self.target_directions)

        # Increment move counters for non-respawned targets
        not_respawned = ~respawn_mask
        self.target_move_counters = torch.where(
            not_respawned,
            self.target_move_counters + 1,
            self.target_move_counters,
        )

        should_move = not_respawned & (self.target_move_counters >= cfg.target_move_interval)
        self.target_move_counters = torch.where(should_move, torch.zeros_like(self.target_move_counters), self.target_move_counters)

        if torch.any(should_move):
            if cfg.direction_change_prob > 0:
                change = torch.rand(self.target_directions.shape, device=device, generator=self.gen) < cfg.direction_change_prob
                change = change & should_move
                new_dirs = torch.randint(
                    0, 4, self.target_directions.shape, generator=self.gen, device=device, dtype=torch.int32
                )
                self.target_directions = torch.where(change, new_dirs, self.target_directions)

            current_pos = self.target_pos
            new_pos = self._move_position(current_pos, self.target_directions)

            hit_wall = (new_pos == current_pos).all(dim=-1)
            if torch.any(hit_wall):
                valid_dirs = self._valid_directions_mask(current_pos)
                rand = torch.rand(valid_dirs.shape, device=device, generator=self.gen)
                rand = torch.where(valid_dirs, rand, torch.full_like(rand, -1.0))
                new_dirs = rand.argmax(dim=-1).to(torch.int32)
                self.target_directions = torch.where(hit_wall, new_dirs, self.target_directions)
                new_pos = self._move_position(current_pos, self.target_directions)

            self.target_pos = torch.where(should_move.unsqueeze(-1), new_pos, self.target_pos)

    def _valid_directions_mask(self, position: torch.Tensor) -> torch.Tensor:
        """Return valid direction mask for each position."""
        row = position[..., 0]
        col = position[..., 1]
        can_left = col > 0
        can_right = col < (self.config.grid_size - 1)
        can_up = row > 0
        can_down = row < (self.config.grid_size - 1)
        return torch.stack([can_left, can_right, can_up, can_down], dim=-1)

    def _sample_positions_excluding_agent(self, respawn_mask: torch.Tensor) -> torch.Tensor:
        """Sample new positions for targets, avoiding current agent position."""
        cfg = self.config
        device = self.device
        shape = respawn_mask.shape
        rows = torch.randint(0, cfg.grid_size, shape, generator=self.gen, device=device)
        cols = torch.randint(0, cfg.grid_size, shape, generator=self.gen, device=device)
        new_pos = torch.stack([rows, cols], dim=-1)

        if cfg.grid_size <= 1:
            return new_pos.to(torch.int32)

        agent_row = self.agent_pos[:, 0].unsqueeze(1)
        agent_col = self.agent_pos[:, 1].unsqueeze(1)
        match = (new_pos[..., 0] == agent_row) & (new_pos[..., 1] == agent_col)

        for _ in range(5):
            if not torch.any(match & respawn_mask):
                break
            rows = torch.randint(0, cfg.grid_size, shape, generator=self.gen, device=device)
            cols = torch.randint(0, cfg.grid_size, shape, generator=self.gen, device=device)
            candidate = torch.stack([rows, cols], dim=-1)
            new_pos = torch.where((match & respawn_mask).unsqueeze(-1), candidate, new_pos)
            match = (new_pos[..., 0] == agent_row) & (new_pos[..., 1] == agent_col)

        # Final guarantee: shift column for any remaining matches (avoids agent position).
        if torch.any(match & respawn_mask) and cfg.grid_size > 1:
            col_fix = (new_pos[..., 1] + 1) % cfg.grid_size
            new_pos = torch.where((match & respawn_mask).unsqueeze(-1), torch.stack([new_pos[..., 0], col_fix], dim=-1), new_pos)

        return new_pos.to(torch.int32)

    def _build_reset_positions(self) -> torch.Tensor:
        """Build a cached grid of positions excluding (0,0) for reset sampling."""
        grid = torch.arange(self.config.grid_size, device=self.device, dtype=torch.int32)
        rows, cols = torch.meshgrid(grid, grid, indexing="ij")
        positions = torch.stack([rows, cols], dim=-1).reshape(-1, 2)
        mask = ~((positions[:, 0] == 0) & (positions[:, 1] == 0))
        return positions[mask].to(torch.int32)

    def create_single_agent_gif(
        self,
        episode_data: Dict[str, Any],
        output_path: Path,
        fps: int = 2
    ) -> None:
        """Create a video for single-agent mode showing all targets."""
        grid_size_inches = min(10, max(6, self.grid_size * 0.15))
        info_width = 4
        fig = plt.figure(figsize=(grid_size_inches + info_width, grid_size_inches))

        gs = fig.add_gridspec(1, 2, width_ratios=[grid_size_inches, info_width], wspace=0.15)
        grid_ax = fig.add_subplot(gs[0])
        info_ax = fig.add_subplot(gs[1])

        def animate(frame):
            grid_ax.clear()
            info_ax.clear()

            if frame >= len(episode_data["states"]):
                return

            state = episode_data["states"][frame]
            denom = float(self.grid_size - 1) if self.grid_size > 1 else 1.0
            agent_row = int(state[0] * denom)
            agent_col = int(state[1] * denom)

            target_positions = []
            targets_reached = []
            include_reached = not self.config.moving_targets
            for i in range(self.num_agents):
                target_idx = 2 + i * 2
                target_positions.append((int(state[target_idx] * denom),
                                       int(state[target_idx + 1] * denom)))
                if include_reached:
                    target_reached_idx = 2 + 2 * self.num_agents + i
                    targets_reached.append(int(state[target_reached_idx]))
                else:
                    targets_reached.append(0)

            grid_ax.set_xlim(-0.5, self.grid_size - 0.5)
            grid_ax.set_ylim(-0.5, self.grid_size - 0.5)
            grid_ax.set_aspect('equal')

            for i in range(self.grid_size + 1):
                grid_ax.axhline(i - 0.5, color='lightgray', linewidth=0.5)
                grid_ax.axvline(i - 0.5, color='lightgray', linewidth=0.5)

            stick_colors = ['royalblue', 'crimson', 'darkorange', 'forestgreen', 'purple', 'deeppink', 'teal', 'saddlebrown', 'mediumvioletred', 'steelblue', 'olivedrab', 'coral']

            cat_expressions = ['😸', '😺', '😼', '😽', '🙀', '😹', '😻', '😾', '😿', '🐱', '😺', '😸']

            _scale = 15 / self.grid_size
            _cat_fontsize = max(6, int(18 * _scale))
            _robot_s = 0.32 * _scale

            def draw_cat(ax, cx, cy, color, idx=0):
                expr = cat_expressions[idx % len(cat_expressions)]
                ax.text(cx, cy, expr, ha='center', va='center', fontsize=_cat_fontsize, color=color, zorder=5)

            def draw_robot(ax, cx, cy, s=_robot_s):
                import matplotlib.patches as mpatches
                ax.plot([cx, cx], [cy - s*1.05, cy - s*0.72], color='#444', linewidth=1.5, zorder=4)
                ax.add_patch(plt.Circle((cx, cy - s*1.12), s*0.08, facecolor='#FF4444', edgecolor='#444', linewidth=1, zorder=5))
                ax.add_patch(mpatches.FancyBboxPatch((cx - s*0.42, cy - s*0.70), s*0.84, s*0.60,
                             boxstyle='round,pad=0.02', facecolor='#A8C8E8', edgecolor='#444', linewidth=1.5, zorder=4))
                for ex in [cx - s*0.15, cx + s*0.15]:
                    ax.add_patch(plt.Circle((ex, cy - s*0.42), s*0.10, facecolor='#1144AA', edgecolor='#444', linewidth=1, zorder=5))
                    ax.add_patch(plt.Circle((ex + s*0.03, cy - s*0.44), s*0.04, facecolor='white', linewidth=0, zorder=6))
                ax.add_patch(mpatches.FancyBboxPatch((cx - s*0.18, cy - s*0.22), s*0.36, s*0.10,
                             boxstyle='round,pad=0.01', facecolor='#1144AA', edgecolor='#444', linewidth=1, zorder=5))
                ax.add_patch(mpatches.FancyBboxPatch((cx - s*0.46, cy + s*0.02), s*0.92, s*0.68,
                             boxstyle='round,pad=0.02', facecolor='#88AACC', edgecolor='#444', linewidth=1.5, zorder=4))
                ax.add_patch(mpatches.FancyBboxPatch((cx - s*0.26, cy + s*0.12), s*0.52, s*0.36,
                             boxstyle='round,pad=0.01', facecolor='#CCDDE8', edgecolor='#666', linewidth=1, zorder=5))
                for bx, bc in [(cx - s*0.10, '#FF4444'), (cx + s*0.10, '#44CC44')]:
                    ax.add_patch(plt.Circle((bx, cy + s*0.30), s*0.07, facecolor=bc, edgecolor='#444', linewidth=1, zorder=6))
                for side in [-1, 1]:
                    ax.plot([cx + side*s*0.46, cx + side*s*0.72], [cy + s*0.18, cy + s*0.38],
                            color='#88AACC', linewidth=4, solid_capstyle='round', zorder=3)
                    ax.plot([cx + side*s*0.46, cx + side*s*0.72], [cy + s*0.18, cy + s*0.38],
                            color='#444', linewidth=1.5, solid_capstyle='round', zorder=3)

            for i in range(self.num_agents):
                target_row, target_col = target_positions[i]
                if include_reached and targets_reached[i] != 0:
                    draw_cat(grid_ax, target_col, target_row, 'darkgreen', idx=i)
                    grid_ax.text(target_col, target_row - 0.5, '✓',
                           ha='center', va='center', fontsize=8, fontweight='bold', color='darkgreen')
                else:
                    draw_cat(grid_ax, target_col, target_row, stick_colors[i % len(stick_colors)], idx=i)

            draw_robot(grid_ax, agent_col, agent_row)

            grid_ax.set_title(f'Step {frame}', fontsize=11, fontweight='bold')

            if self.grid_size <= 15:
                tick_step = 1
            elif self.grid_size <= 30:
                tick_step = 2
            else:
                tick_step = 5

            tick_positions = list(range(0, self.grid_size, tick_step))
            grid_ax.set_xticks(tick_positions)
            grid_ax.set_yticks(tick_positions)
            grid_ax.invert_yaxis()

            info_ax.axis('off')

            reward = 0.0
            if frame < len(episode_data["rewards"]):
                reward = episode_data["rewards"][frame]

            total_reward = sum(episode_data["rewards"][:frame + 1])

            info_lines = []
            info_lines.append('SINGLE-AGENT ROLLOUT')
            info_lines.append('')
            info_lines.append(f'Grid: {self.grid_size}x{self.grid_size}')
            info_lines.append(f'Targets: {self.num_agents}')
            info_lines.append('')
            info_lines.append('REWARDS:')
            info_lines.append(f'  Step:  {reward:7.2f}')
            info_lines.append(f'  Total: {total_reward:7.2f}')
            info_lines.append('')
            if include_reached:
                info_lines.append('TARGET STATUS:')
                for target_id in range(self.num_agents):
                    target_reached_idx = 2 + 2 * self.num_agents + target_id
                    target_reached = int(state[target_reached_idx])
                    status = 'OK' if target_reached else 'NO'
                    info_lines.append(f'  {target_id}: {status}')

            info_text = '\n'.join(info_lines)
            info_ax.text(0.05, 0.95, info_text,
                        transform=info_ax.transAxes,
                        fontfamily='monospace',
                        fontsize=10,
                        verticalalignment='top',
                        horizontalalignment='left')

        frames = []
        num_frames = len(episode_data["states"]) + 5
        for frame_idx in range(num_frames):
            animate(frame_idx)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            frames.append(frame)

        plt.close(fig)

        if len(frames) > 0:
            h, w = frames[0].shape[:2]
            output_path_mp4 = str(output_path).replace('.gif', '.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path_mp4, fourcc, fps, (w, h))

            try:
                for frame in frames:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                out.release()
                print(f"OK Single-agent video saved: {output_path_mp4}")
            except Exception as e:
                print(f"Warning: Could not save video {output_path_mp4}: {e}")
                out.release()

    def create_competition_gif(
        self,
        episode_data: Dict[str, Any],
        output_path: Path,
        fps: int = 1
    ) -> None:
        """Create a video of a multi-agent competition episode."""
        grid_size_inches = min(10, max(6, self.grid_size * 0.15))
        info_width = 5
        fig = plt.figure(figsize=(grid_size_inches + info_width, grid_size_inches))

        gs = fig.add_gridspec(1, 2, width_ratios=[grid_size_inches, info_width], wspace=0.15)
        grid_ax = fig.add_subplot(gs[0])
        info_ax = fig.add_subplot(gs[1])

        def animate(frame):
            grid_ax.clear()
            info_ax.clear()

            if frame >= len(episode_data["states"]):
                return

            state = episode_data["states"][frame]
            denom = float(self.grid_size - 1) if self.grid_size > 1 else 1.0
            agent_row = int(state[0] * denom)
            agent_col = int(state[1] * denom)

            target_positions = []
            targets_reached = []
            include_reached = not self.config.moving_targets
            for i in range(self.num_agents):
                target_idx = 2 + i * 2
                target_positions.append((int(state[target_idx] * denom),
                                       int(state[target_idx + 1] * denom)))
                if include_reached:
                    target_reached_idx = 2 + 2 * self.num_agents + i
                    targets_reached.append(int(state[target_reached_idx]))
                else:
                    targets_reached.append(0)

            step_detail = episode_data["step_details"][frame] if frame < len(episode_data["step_details"]) else None
            actions = episode_data["actions"][frame] if frame < len(episode_data["actions"]) else None
            rewards = episode_data["rewards"][frame] if frame < len(episode_data["rewards"]) else None

            grid_ax.set_xlim(-0.5, self.grid_size - 0.5)
            grid_ax.set_ylim(-0.5, self.grid_size - 0.5)
            grid_ax.set_aspect('equal')

            for i in range(self.grid_size + 1):
                grid_ax.axhline(i - 0.5, color='lightgray', linewidth=0.5)
                grid_ax.axvline(i - 0.5, color='lightgray', linewidth=0.5)

            stick_colors = ['royalblue', 'crimson', 'darkorange', 'forestgreen', 'purple', 'deeppink', 'teal', 'saddlebrown', 'mediumvioletred', 'steelblue', 'olivedrab', 'coral']
            edge_colors = ['blue', 'red', 'orange', 'green', 'purple']
            winning_agent = step_detail.get("winning_agent", -1) if step_detail else None

            cat_expressions = ['😸', '😺', '😼', '😽', '🙀', '😹', '😻', '😾', '😿', '🐱', '😺', '😸']

            _scale = 15 / self.grid_size
            _cat_fontsize = max(6, int(18 * _scale))
            _robot_s = 0.32 * _scale

            def draw_cat(ax, cx, cy, color, idx=0):
                expr = cat_expressions[idx % len(cat_expressions)]
                ax.text(cx, cy, expr, ha='center', va='center', fontsize=_cat_fontsize, color=color, zorder=5)

            def draw_robot(ax, cx, cy, s=_robot_s):
                import matplotlib.patches as mpatches
                ax.plot([cx, cx], [cy - s*1.05, cy - s*0.72], color='#444', linewidth=1.5, zorder=4)
                ax.add_patch(plt.Circle((cx, cy - s*1.12), s*0.08, facecolor='#FF4444', edgecolor='#444', linewidth=1, zorder=5))
                ax.add_patch(mpatches.FancyBboxPatch((cx - s*0.42, cy - s*0.70), s*0.84, s*0.60,
                             boxstyle='round,pad=0.02', facecolor='#A8C8E8', edgecolor='#444', linewidth=1.5, zorder=4))
                for ex in [cx - s*0.15, cx + s*0.15]:
                    ax.add_patch(plt.Circle((ex, cy - s*0.42), s*0.10, facecolor='#1144AA', edgecolor='#444', linewidth=1, zorder=5))
                    ax.add_patch(plt.Circle((ex + s*0.03, cy - s*0.44), s*0.04, facecolor='white', linewidth=0, zorder=6))
                ax.add_patch(mpatches.FancyBboxPatch((cx - s*0.18, cy - s*0.22), s*0.36, s*0.10,
                             boxstyle='round,pad=0.01', facecolor='#1144AA', edgecolor='#444', linewidth=1, zorder=5))
                ax.add_patch(mpatches.FancyBboxPatch((cx - s*0.46, cy + s*0.02), s*0.92, s*0.68,
                             boxstyle='round,pad=0.02', facecolor='#88AACC', edgecolor='#444', linewidth=1.5, zorder=4))
                ax.add_patch(mpatches.FancyBboxPatch((cx - s*0.26, cy + s*0.12), s*0.52, s*0.36,
                             boxstyle='round,pad=0.01', facecolor='#CCDDE8', edgecolor='#666', linewidth=1, zorder=5))
                for bx, bc in [(cx - s*0.10, '#FF4444'), (cx + s*0.10, '#44CC44')]:
                    ax.add_patch(plt.Circle((bx, cy + s*0.30), s*0.07, facecolor=bc, edgecolor='#444', linewidth=1, zorder=6))
                for side in [-1, 1]:
                    ax.plot([cx + side*s*0.46, cx + side*s*0.72], [cy + s*0.18, cy + s*0.38],
                            color='#88AACC', linewidth=4, solid_capstyle='round', zorder=3)
                    ax.plot([cx + side*s*0.46, cx + side*s*0.72], [cy + s*0.18, cy + s*0.38],
                            color='#444', linewidth=1.5, solid_capstyle='round', zorder=3)

            for i in range(self.num_agents):
                target_row, target_col = target_positions[i]
                is_controlling = (winning_agent == i)

                if include_reached and targets_reached[i] != 0:
                    draw_cat(grid_ax, target_col, target_row, 'darkgreen', idx=i)
                    grid_ax.text(target_col, target_row - 0.5, '✓',
                           ha='center', va='center', fontsize=8, fontweight='bold', color='darkgreen')
                else:
                    draw_cat(grid_ax, target_col, target_row, stick_colors[i % len(stick_colors)], idx=i)
                    if is_controlling:
                        grid_ax.text(target_col, target_row - 0.6, '⚡',
                               ha='center', va='center', fontsize=8, color='gold')

            if winning_agent is not None and 0 <= winning_agent < self.num_agents:
                grid_ax.add_patch(plt.Circle((agent_col, agent_row), 0.35,
                                       facecolor='none', edgecolor=edge_colors[winning_agent % len(edge_colors)], linewidth=3))

            draw_robot(grid_ax, agent_col, agent_row)

            grid_ax.set_title(f'Step {frame}', fontsize=11, fontweight='bold')

            if self.grid_size <= 15:
                tick_step = 1
            elif self.grid_size <= 30:
                tick_step = 2
            else:
                tick_step = 5

            tick_positions = list(range(0, self.grid_size, tick_step))
            grid_ax.set_xticks(tick_positions)
            grid_ax.set_yticks(tick_positions)
            grid_ax.invert_yaxis()

            info_ax.axis('off')

            info_lines = []
            info_lines.append('MULTI-AGENT COMPETITION')
            info_lines.append('')
            info_lines.append(f'Grid: {self.grid_size}x{self.grid_size}')
            info_lines.append(f'Agents: {self.num_agents}')
            controller_label = str(winning_agent) if winning_agent is not None and winning_agent >= 0 else "None"
            info_lines.append(f'Controller: {controller_label}')
            info_lines.append('')

            if step_detail and actions and rewards:
                info_lines.append('AGENT DETAILS:')
                direction_names = {0: "<", 1: ">", 2: "^", 3: "v"}

                cumulative = {}
                for i in range(self.num_agents):
                    cumulative[f"agent_{i}"] = sum(
                        episode_data["rewards"][f].get(f"agent_{i}", 0)
                        for f in range(min(frame + 1, len(episode_data["rewards"])))
                    )

                for i in range(self.num_agents):
                    agent_key = f"agent_{i}"
                    if agent_key in actions:
                        action_data = actions[agent_key]
                        bid = action_data.get('bid', 0)
                        direction = direction_names.get(action_data.get('direction', 0), '?')
                        window_steps = None
                        if self.window_bidding and "window" in action_data:
                            window_steps = int(action_data.get("window", 0)) + 1
                        reward = cumulative.get(agent_key, 0)
                        window_str = f' W:{window_steps:2d}' if window_steps is not None else ''

                        if i == winning_agent:
                            info_lines.append(f'  [{i}] * Bid:{bid:2d}{window_str} {direction} R:{reward:6.1f}')
                        else:
                            info_lines.append(f'  [{i}]   Bid:{bid:2d}{window_str} {direction} R:{reward:6.1f}')

            info_text = '\n'.join(info_lines)
            info_ax.text(0.05, 0.95, info_text,
                        transform=info_ax.transAxes,
                        fontfamily='monospace',
                        fontsize=9,
                        verticalalignment='top',
                        horizontalalignment='left')

        frames = []
        num_frames = len(episode_data["states"]) + 3
        for frame_idx in range(num_frames):
            animate(frame_idx)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            frames.append(frame)

        plt.close(fig)

        if len(frames) > 0:
            h, w = frames[0].shape[:2]
            output_path_mp4 = str(output_path).replace('.gif', '.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path_mp4, fourcc, fps, (w, h))

            try:
                for frame in frames:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                out.release()
                print(f"OK Competition video saved: {output_path_mp4}")
            except Exception as e:
                print(f"Warning: Could not save video {output_path_mp4}: {e}")
                out.release()

    def partial_reset(self, mask: torch.Tensor) -> torch.Tensor:
        """Reset only the envs indicated by mask (bool, shape (num_envs,)).

        Resets all per-env state tensors for done envs in-place, then returns
        a fresh observation tensor (num_envs, ...) with reset obs spliced in
        for the masked envs.
        """
        cfg = self.config
        device = self.device

        self.agent_pos = torch.where(
            mask.unsqueeze(-1),
            torch.zeros_like(self.agent_pos),
            self.agent_pos,
        )

        rand = torch.rand(
            (self.num_envs, self._reset_positions.shape[0]), generator=self.gen, device=device
        )
        idx = torch.topk(rand, k=cfg.num_agents, dim=1, largest=True).indices
        new_target_pos = self._reset_positions[idx].to(torch.int32)
        self.target_pos = torch.where(mask.view(-1, 1, 1), new_target_pos, self.target_pos)

        zeros2d = torch.zeros((self.num_envs, cfg.num_agents), device=device, dtype=torch.int32)
        self.targets_reached = torch.where(mask.unsqueeze(-1), zeros2d, self.targets_reached)
        self.targets_reached_count = torch.where(mask.unsqueeze(-1), zeros2d, self.targets_reached_count)
        self.target_counters = torch.where(mask.unsqueeze(-1), zeros2d, self.target_counters)
        self.window_agent = torch.where(mask, torch.full_like(self.window_agent, -1), self.window_agent)
        self.window_steps_remaining = torch.where(
            mask, torch.zeros_like(self.window_steps_remaining), self.window_steps_remaining
        )
        self.step_count = torch.where(mask, torch.zeros_like(self.step_count), self.step_count)

        if cfg.moving_targets:
            new_dirs = torch.randint(
                0, 4, self.target_directions.shape, generator=self.gen, device=device, dtype=torch.int32
            )
            self.target_directions = torch.where(mask.unsqueeze(-1), new_dirs, self.target_directions)
            self.target_move_counters = torch.where(
                mask.unsqueeze(-1), torch.zeros_like(self.target_move_counters), self.target_move_counters
            )

        new_distances = self._compute_distances()
        self.previous_distances = torch.where(mask.unsqueeze(-1), new_distances, self.previous_distances)

        return self._get_observation()

    def close(self) -> None:
        """No-op close for API compatibility."""
        return None


def evaluate_multi_agent_policy(
    env: BiddingGridworld,
    policy_fn,
    num_episodes: int,
    target_expiry_penalty: float = 0.0,
    verbose: bool = True
) -> Dict[str, List]:
    """
    Evaluate a multi-agent policy on the torch-batched environment.

    Args:
        env: BiddingGridworld (num_envs=1, multi-agent mode)
        policy_fn: Callable taking obs (num_agents, obs_dim) and returning actions
                   (num_agents, action_dim).
        num_episodes: Number of episodes to evaluate
        target_expiry_penalty: Target expiry penalty value (for counting expired targets)
        verbose: Whether to print progress
    """
    if verbose:
        print(f"\n{'='*60}")
        print("Evaluating multi-agent policy")
        print(f"Running {num_episodes} episodes")
        print(f"{'='*60}\n")

    eval_stats = {
        "episode_returns": [],
        "episode_returns_no_bid": [],
        "episode_lengths": [],
        "targets_reached_per_episode": [],
        "expired_targets_per_episode": [],
        "min_targets_reached_per_episode": [],
        "targets_reached_count_per_episode": [],
        "episode_data_list": [],
        "bid_counts_per_episode": [],
        "control_steps_per_agent_per_episode": [],
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

        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_step_details = []
        episode_return = 0.0
        episode_return_no_bid = 0.0
        step_count = 0
        terminated = False
        truncated = False

        targets_reached_count = np.zeros(env.num_agents, dtype=np.int32)
        expired_targets_count = np.zeros(env.num_agents, dtype=np.int32)
        bid_counts: dict = {}
        control_steps = np.zeros(env.num_agents, dtype=np.int32)

        while not (terminated or truncated):
            episode_states.append(env._get_centralized_observation()[0].copy())

            actions = policy_fn(obs[0])
            if isinstance(actions, np.ndarray):
                actions = torch.tensor(actions, device=env.device)
            actions = actions.to(env.device)
            action_batch = actions.unsqueeze(0)

            env_action = {}
            for agent_idx in range(env.num_agents):
                agent_action = {
                    "direction": int(actions[agent_idx, 0].item()),
                    "bid": int(actions[agent_idx, 1].item()),
                }
                if env.window_bidding:
                    agent_action["window"] = int(actions[agent_idx, 2].item())
                env_action[f"agent_{agent_idx}"] = agent_action

            episode_actions.append(env_action)

            obs, rewards, terminations, truncations, info = env.step(action_batch)
            terminated = bool(terminations[0].item())
            truncated = bool(truncations[0].item())

            rewards_cpu = rewards[0].detach().cpu().numpy()
            rewards_dict = {f"agent_{i}": float(rewards_cpu[i]) for i in range(env.num_agents)}
            episode_return += float(rewards_cpu.sum())
            episode_rewards.append(rewards_dict)

            reward_no_bid_sum = info.get("reward_no_bid_sum")
            if isinstance(reward_no_bid_sum, torch.Tensor):
                episode_return_no_bid += float(reward_no_bid_sum[0].item())

            tje = info.get("targets_just_expired")
            if isinstance(tje, torch.Tensor):
                just_expired = tje[0].detach().cpu().numpy().astype(bool)
                expired_targets_count += just_expired.astype(int)

            targets_just_reached = info.get("targets_just_reached")
            if isinstance(targets_just_reached, torch.Tensor):
                just_reached = targets_just_reached[0].detach().cpu().numpy().astype(bool)
                for agent_idx in range(env.num_agents):
                    if just_reached[agent_idx]:
                        targets_reached_count[agent_idx] += 1

            winning_agent = info.get("winning_agent", torch.tensor([-1], device=env.device))
            if isinstance(winning_agent, torch.Tensor):
                winning_agent = int(winning_agent[0].item())

            bids = info.get("bids", None)
            if isinstance(bids, torch.Tensor):
                bids = bids[0].detach().cpu().tolist()

            is_bidding_round = info.get("is_bidding_round", None)
            if isinstance(is_bidding_round, torch.Tensor):
                is_bidding_round = bool(is_bidding_round[0].item())

            if is_bidding_round and bids is not None:
                for bid_val in bids:
                    bid_counts[int(bid_val)] = bid_counts.get(int(bid_val), 0) + 1

            if winning_agent >= 0:
                control_steps[winning_agent] += 1

            episode_step_details.append({
                "winning_agent": winning_agent,
                "bids": bids,
                "window_agent": int(info.get("window_agent", torch.tensor([-1]))[0].item()) if info.get("window_agent") is not None else None,
                "window_steps_remaining": int(info.get("window_steps_remaining", torch.tensor([0]))[0].item()) if info.get("window_steps_remaining") is not None else 0,
                "bid_penalty_applied": bool(info.get("bid_penalty_applied", torch.tensor([False]))[0].item()) if info.get("bid_penalty_applied") is not None else False,
            })

            step_count += 1

        targets_reached = sum(1 for count in targets_reached_count if count > 0)
        min_targets_reached = int(np.min(targets_reached_count))
        episode_expired_count = int(expired_targets_count.sum())
        performance = targets_reached_count - expired_targets_count

        eval_stats["episode_returns"].append(episode_return)
        eval_stats["episode_returns_no_bid"].append(episode_return_no_bid)
        eval_stats["episode_lengths"].append(step_count)
        eval_stats["targets_reached_per_episode"].append(targets_reached)
        eval_stats["expired_targets_per_episode"].append(episode_expired_count)
        eval_stats["min_targets_reached_per_episode"].append(min_targets_reached)
        eval_stats["targets_reached_count_per_episode"].append(targets_reached_count.tolist())
        eval_stats["bid_counts_per_episode"].append(bid_counts)
        eval_stats["control_steps_per_agent_per_episode"].append(control_steps.tolist())
        eval_stats["expired_count_per_target_per_episode"].append(expired_targets_count.tolist())
        eval_stats["avg_expired_per_episode"].append(float(np.mean(expired_targets_count)))
        eval_stats["max_expired_per_episode"].append(float(np.max(expired_targets_count)))
        eval_stats["avg_reached_per_episode"].append(float(np.mean(targets_reached_count)))
        eval_stats["performance_per_episode"].append(performance.tolist())
        eval_stats["avg_performance_per_episode"].append(float(np.mean(performance)))
        eval_stats["min_performance_per_episode"].append(float(np.min(performance)))

        eval_stats["episode_data_list"].append({
            "states": episode_states,
            "actions": episode_actions,
            "rewards": episode_rewards,
            "step_details": episode_step_details,
        })

        if verbose:
            print(f"  Episode {episode_idx + 1}: Return={episode_return:.2f}, "
                  f"Length={step_count}, Targets={targets_reached}/{env.num_agents}, "
                  f"Expired={episode_expired_count}, MinReached={min_targets_reached}, "
                  f"AvgPerf={float(np.mean(performance)):.2f}")

    if verbose:
        avg_return = np.mean(eval_stats["episode_returns"])
        avg_return_no_bid = np.mean(eval_stats["episode_returns_no_bid"])
        avg_length = np.mean(eval_stats["episode_lengths"])
        avg_targets = np.mean(eval_stats["targets_reached_per_episode"])
        avg_expired = np.mean(eval_stats["expired_targets_per_episode"])
        avg_min_reached = np.mean(eval_stats["min_targets_reached_per_episode"])
        avg_avg_perf = np.mean(eval_stats["avg_performance_per_episode"])
        avg_min_perf = np.mean(eval_stats["min_performance_per_episode"])
        success_rate = sum(1 for t in eval_stats["targets_reached_per_episode"]
                          if t == env.num_agents) / num_episodes

        print("\nEvaluation Summary:")
        print(f"  Average Return: {avg_return:.2f}")
        print(f"  Average Return (no bid penalty): {avg_return_no_bid:.2f}")
        print(f"  Average Length: {avg_length:.1f}")
        print(f"  Average Targets: {avg_targets:.2f}/{env.num_agents}")
        print(f"  Average Expired: {avg_expired:.2f} ± {np.std(eval_stats['expired_targets_per_episode']):.2f}")
        print(f"  Average Min Reached: {avg_min_reached:.2f} ± {np.std(eval_stats['min_targets_reached_per_episode']):.2f}")
        print(f"  Avg Performance (reaches-exp): {avg_avg_perf:.2f}")
        print(f"  Avg Min Performance: {avg_min_perf:.2f}")
        print(f"  Success Rate: {success_rate*100:.1f}%\n")

    return eval_stats


def evaluate_multi_agent_policy_batched(
    env: BiddingGridworld,
    policy_fn,
    num_episodes: int,
    target_expiry_penalty: float = 0.0,
    verbose: bool = True
) -> Dict[str, List]:
    """
    Batched evaluation of a multi-agent policy.

    Assumes env.num_envs == num_episodes. Runs all episodes in parallel in a single
    while loop, which is much faster than sequential evaluation when episodes have
    fixed length (e.g. moving-targets mode where episodes always run to max_steps).

    Args:
        env: BiddingGridworld with num_envs == num_episodes
        policy_fn: Callable taking obs (N, num_agents, obs_dim) and returning actions
                   (N, num_agents, action_dim).
        num_episodes: Number of episodes (must equal env.num_envs)
        target_expiry_penalty: Unused; kept for API parity.
        verbose: Whether to print progress
    """
    N = env.num_envs
    A = env.num_agents
    bid_upper_bound = env.config.bid_upper_bound
    device = env.device

    if N != num_episodes:
        raise ValueError(f"env.num_envs ({N}) must equal num_episodes ({num_episodes})")

    if verbose:
        print(f"\n{'='*60}")
        print("Evaluating multi-agent policy (batched)")
        print(f"Running {num_episodes} episodes in parallel")
        print(f"{'='*60}\n")

    obs, _ = env.reset()

    # GPU accumulators
    returns = torch.zeros(N, device=device)
    returns_no_bid = torch.zeros(N, device=device)
    lengths = torch.zeros(N, dtype=torch.long, device=device)
    targets_reached_count = torch.zeros(N, A, dtype=torch.long, device=device)
    expired_count = torch.zeros(N, A, dtype=torch.long, device=device)
    control_steps = torch.zeros(N, A, dtype=torch.long, device=device)
    bid_count_tensor = torch.zeros(N, bid_upper_bound + 1, dtype=torch.long, device=device)

    # active[i] is True while episode i has not terminated/truncated
    active = torch.ones(N, dtype=torch.bool, device=device)

    while active.any():
        actions = policy_fn(obs)
        if not torch.is_tensor(actions):
            actions = torch.tensor(actions, device=device)
        actions = actions.to(device)

        obs, rewards, terminations, truncations, info = env.step(actions)

        done = terminations | truncations  # (N,)
        # Only update accumulators for still-active envs
        active_f = active.float()

        returns += (rewards.sum(dim=1) * active_f)
        lengths += active.long()

        rnb = info.get("reward_no_bid_sum")
        if isinstance(rnb, torch.Tensor):
            returns_no_bid += (rnb * active_f)

        tje = info.get("targets_just_expired")
        if isinstance(tje, torch.Tensor):
            # tje shape: (N, A)
            expired_count += (tje.long() * active.unsqueeze(1).long())

        tjr = info.get("targets_just_reached")
        if isinstance(tjr, torch.Tensor):
            # tjr shape: (N, A)
            targets_reached_count += (tjr.long() * active.unsqueeze(1).long())

        winning_agent = info.get("winning_agent")
        if isinstance(winning_agent, torch.Tensor):
            # winning_agent shape: (N,), value -1 means no winner
            valid_winner = (winning_agent >= 0) & active  # (N,)
            if valid_winner.any():
                winner_idx = winning_agent.clamp(min=0).long()  # avoid negative index; must be int64
                control_steps.scatter_add_(
                    1,
                    winner_idx.unsqueeze(1),
                    valid_winner.long().unsqueeze(1)
                )

        bids = info.get("bids")
        is_bidding_round = info.get("is_bidding_round")
        if isinstance(bids, torch.Tensor) and isinstance(is_bidding_round, torch.Tensor):
            # bids shape: (N, A), is_bidding_round shape: (N,)
            bidding_active = is_bidding_round & active  # (N,)
            if bidding_active.any():
                bids_clamped = bids.clamp(0, bid_upper_bound).long()
                for a_idx in range(A):
                    bid_count_tensor.scatter_add_(
                        1,
                        bids_clamped[:, a_idx].unsqueeze(1),
                        bidding_active.long().unsqueeze(1)
                    )

        # Mark envs that are done as inactive
        active = active & ~done

    # Move results to CPU for output
    returns_cpu = returns.cpu().tolist()
    returns_no_bid_cpu = returns_no_bid.cpu().tolist()
    lengths_cpu = lengths.cpu().tolist()
    targets_reached_cpu = targets_reached_count.cpu().numpy()
    expired_cpu = expired_count.cpu().numpy()
    control_steps_cpu = control_steps.cpu().tolist()
    bid_count_np = bid_count_tensor.cpu().numpy()

    eval_stats = {
        "episode_returns": [],
        "episode_returns_no_bid": [],
        "episode_lengths": [],
        "targets_reached_per_episode": [],
        "expired_targets_per_episode": [],
        "min_targets_reached_per_episode": [],
        "targets_reached_count_per_episode": [],
        "episode_data_list": [],  # empty — video not supported in batched mode
        "bid_counts_per_episode": [],
        "control_steps_per_agent_per_episode": [],
        "expired_count_per_target_per_episode": [],
        "avg_expired_per_episode": [],
        "max_expired_per_episode": [],
        "avg_reached_per_episode": [],
        "performance_per_episode": [],
        "avg_performance_per_episode": [],
        "min_performance_per_episode": [],
    }

    for i in range(N):
        trc = targets_reached_cpu[i]  # (A,) numpy
        ec = expired_cpu[i]           # (A,) numpy
        performance = trc - ec

        targets_reached = int((trc > 0).sum())
        min_targets_reached = int(trc.min())
        episode_expired_count = int(ec.sum())

        bid_counts_dict = {b: int(bid_count_np[i, b]) for b in range(bid_upper_bound + 1)}

        eval_stats["episode_returns"].append(returns_cpu[i])
        eval_stats["episode_returns_no_bid"].append(returns_no_bid_cpu[i])
        eval_stats["episode_lengths"].append(int(lengths_cpu[i]))
        eval_stats["targets_reached_per_episode"].append(targets_reached)
        eval_stats["expired_targets_per_episode"].append(episode_expired_count)
        eval_stats["min_targets_reached_per_episode"].append(min_targets_reached)
        eval_stats["targets_reached_count_per_episode"].append(trc.tolist())
        eval_stats["bid_counts_per_episode"].append(bid_counts_dict)
        eval_stats["control_steps_per_agent_per_episode"].append(control_steps_cpu[i])
        eval_stats["expired_count_per_target_per_episode"].append(ec.tolist())
        eval_stats["avg_expired_per_episode"].append(float(np.mean(ec)))
        eval_stats["max_expired_per_episode"].append(float(np.max(ec)))
        eval_stats["avg_reached_per_episode"].append(float(np.mean(trc)))
        eval_stats["performance_per_episode"].append(performance.tolist())
        eval_stats["avg_performance_per_episode"].append(float(np.mean(performance)))
        eval_stats["min_performance_per_episode"].append(float(np.min(performance)))

    if verbose:
        for i in range(N):
            print(f"  Episode {i + 1}: Return={eval_stats['episode_returns'][i]:.2f}, "
                  f"Length={eval_stats['episode_lengths'][i]}, "
                  f"Targets={eval_stats['targets_reached_per_episode'][i]}/{A}, "
                  f"Expired={eval_stats['expired_targets_per_episode'][i]}, "
                  f"MinReached={eval_stats['min_targets_reached_per_episode'][i]}, "
                  f"AvgPerf={eval_stats['avg_performance_per_episode'][i]:.2f}")

        avg_return = np.mean(eval_stats["episode_returns"])
        avg_return_no_bid = np.mean(eval_stats["episode_returns_no_bid"])
        avg_length = np.mean(eval_stats["episode_lengths"])
        avg_targets = np.mean(eval_stats["targets_reached_per_episode"])
        avg_expired = np.mean(eval_stats["expired_targets_per_episode"])
        avg_min_reached = np.mean(eval_stats["min_targets_reached_per_episode"])
        avg_avg_perf = np.mean(eval_stats["avg_performance_per_episode"])
        avg_min_perf = np.mean(eval_stats["min_performance_per_episode"])
        success_rate = sum(1 for t in eval_stats["targets_reached_per_episode"]
                          if t == A) / num_episodes

        print("\nEvaluation Summary:")
        print(f"  Average Return: {avg_return:.2f}")
        print(f"  Average Return (no bid penalty): {avg_return_no_bid:.2f}")
        print(f"  Average Length: {avg_length:.1f}")
        print(f"  Average Targets: {avg_targets:.2f}/{A}")
        print(f"  Average Expired: {avg_expired:.2f} ± {np.std(eval_stats['expired_targets_per_episode']):.2f}")
        print(f"  Average Min Reached: {avg_min_reached:.2f} ± {np.std(eval_stats['min_targets_reached_per_episode']):.2f}")
        print(f"  Avg Performance (reaches-exp): {avg_avg_perf:.2f}")
        print(f"  Avg Min Performance: {avg_min_perf:.2f}")
        print(f"  Success Rate: {success_rate*100:.1f}%\n")

    return eval_stats


def evaluate_single_agent_policy(
    env: BiddingGridworld,
    policy_fn,
    num_episodes: int,
    target_expiry_penalty: float = 0.0,
    verbose: bool = True
) -> Dict[str, List]:
    """
    Evaluate a single-agent policy on the torch-batched environment.

    Args:
        env: BiddingGridworld (num_envs=1, single-agent mode)
        policy_fn: Callable taking obs (obs_dim,) and returning an action (scalar).
        num_episodes: Number of episodes to evaluate
        target_expiry_penalty: Target expiry penalty value (for counting expired targets)
        verbose: Whether to print progress
    """
    if verbose:
        print(f"\n{'='*60}")
        print("Evaluating single-agent policy")
        print(f"Running {num_episodes} episodes")
        print(f"{'='*60}\n")

    eval_stats = {
        "episode_returns": [],
        "episode_lengths": [],
        "targets_reached_per_episode": [],
        "expired_targets_per_episode": [],
        "min_targets_reached_per_episode": [],
        "targets_reached_count_per_episode": [],
        "episode_data_list": [],
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

        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_return = 0.0
        step_count = 0
        terminated = False
        truncated = False

        targets_reached_count = np.zeros(env.num_agents, dtype=np.int32)
        expired_targets_count = np.zeros(env.num_agents, dtype=np.int32)

        while not (terminated or truncated):
            episode_states.append(env._get_centralized_observation()[0].copy())

            action = policy_fn(obs[0])
            if isinstance(action, np.ndarray):
                action = int(action.item())
            elif torch.is_tensor(action):
                action = int(action.item())
            else:
                action = int(action)

            episode_actions.append(action)

            action_batch = torch.tensor([action], device=env.device)
            obs, rewards, terminations, truncations, info = env.step(action_batch)
            terminated = bool(terminations[0].item())
            truncated = bool(truncations[0].item())

            reward_val = float(rewards[0].item())
            episode_return += reward_val
            episode_rewards.append(reward_val)

            tje = info.get("targets_just_expired")
            if isinstance(tje, torch.Tensor):
                just_expired = tje[0].detach().cpu().numpy().astype(bool)
                expired_targets_count += just_expired.astype(int)

            targets_just_reached = info.get("targets_just_reached")
            if isinstance(targets_just_reached, torch.Tensor):
                just_reached = targets_just_reached[0].detach().cpu().numpy().astype(bool)
                for target_idx in range(env.num_agents):
                    if just_reached[target_idx]:
                        targets_reached_count[target_idx] += 1

            step_count += 1

        targets_reached = sum(1 for count in targets_reached_count if count > 0)
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

        eval_stats["episode_data_list"].append({
            "states": episode_states,
            "actions": episode_actions,
            "rewards": episode_rewards,
        })

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
        print(f"  Average Return: {avg_return:.2f}")
        print(f"  Average Length: {avg_length:.1f}")
        print(f"  Average Targets: {avg_targets:.2f}/{env.num_agents}")
        print(f"  Average Expired: {avg_expired:.2f} ± {np.std(eval_stats['expired_targets_per_episode']):.2f}")
        print(f"  Average Min Reached: {avg_min_reached:.2f} ± {np.std(eval_stats['min_targets_reached_per_episode']):.2f}")
        print(f"  Avg Performance (reaches-exp): {avg_avg_perf:.2f}")
        print(f"  Avg Min Performance: {avg_min_perf:.2f}")
        print(f"  Success Rate: {success_rate*100:.1f}%\n")

    return eval_stats


def evaluate_single_agent_policy_batched(
    env: BiddingGridworld,
    policy_fn,
    num_episodes: int,
    target_expiry_penalty: float = 0.0,
    verbose: bool = True
) -> Dict[str, List]:
    """
    Batched evaluation of a single-agent policy.

    Assumes env.num_envs == num_episodes. Runs all episodes in parallel in a single
    while loop.

    Args:
        env: BiddingGridworld with num_envs == num_episodes, single_agent_mode=True
        policy_fn: Callable taking obs (N, obs_dim) and returning actions (N,).
        num_episodes: Number of episodes (must equal env.num_envs)
        target_expiry_penalty: Unused; kept for API parity.
        verbose: Whether to print progress
    """
    N = env.num_envs
    A = env.num_agents
    device = env.device

    if N != num_episodes:
        raise ValueError(f"env.num_envs ({N}) must equal num_episodes ({num_episodes})")

    if verbose:
        print(f"\n{'='*60}")
        print("Evaluating single-agent policy (batched)")
        print(f"Running {num_episodes} episodes in parallel")
        print(f"{'='*60}\n")

    obs, _ = env.reset()

    # GPU accumulators
    returns = torch.zeros(N, device=device)
    lengths = torch.zeros(N, dtype=torch.long, device=device)
    targets_reached_count = torch.zeros(N, A, dtype=torch.long, device=device)
    expired_count = torch.zeros(N, A, dtype=torch.long, device=device)

    active = torch.ones(N, dtype=torch.bool, device=device)

    while active.any():
        actions = policy_fn(obs)
        if not torch.is_tensor(actions):
            actions = torch.tensor(actions, device=device)
        actions = actions.to(device)

        obs, rewards, terminations, truncations, info = env.step(actions)

        done = terminations | truncations  # (N,)
        active_f = active.float()

        returns += (rewards * active_f)
        lengths += active.long()

        tje = info.get("targets_just_expired")
        if isinstance(tje, torch.Tensor):
            expired_count += (tje.long() * active.unsqueeze(1).long())

        tjr = info.get("targets_just_reached")
        if isinstance(tjr, torch.Tensor):
            targets_reached_count += (tjr.long() * active.unsqueeze(1).long())

        active = active & ~done

    returns_cpu = returns.cpu().tolist()
    lengths_cpu = lengths.cpu().tolist()
    targets_reached_cpu = targets_reached_count.cpu().numpy()
    expired_cpu = expired_count.cpu().numpy()

    eval_stats = {
        "episode_returns": [],
        "episode_lengths": [],
        "targets_reached_per_episode": [],
        "expired_targets_per_episode": [],
        "min_targets_reached_per_episode": [],
        "targets_reached_count_per_episode": [],
        "episode_data_list": [],  # empty — video not supported in batched mode
        "expired_count_per_target_per_episode": [],
        "avg_expired_per_episode": [],
        "max_expired_per_episode": [],
        "avg_reached_per_episode": [],
        "performance_per_episode": [],
        "avg_performance_per_episode": [],
        "min_performance_per_episode": [],
    }

    for i in range(N):
        trc = targets_reached_cpu[i]  # (A,) numpy
        ec = expired_cpu[i]           # (A,) numpy
        performance = trc - ec

        targets_reached = int((trc > 0).sum())
        min_targets_reached = int(trc.min())
        episode_expired_count = int(ec.sum())

        eval_stats["episode_returns"].append(returns_cpu[i])
        eval_stats["episode_lengths"].append(int(lengths_cpu[i]))
        eval_stats["targets_reached_per_episode"].append(targets_reached)
        eval_stats["expired_targets_per_episode"].append(episode_expired_count)
        eval_stats["min_targets_reached_per_episode"].append(min_targets_reached)
        eval_stats["targets_reached_count_per_episode"].append(trc.tolist())
        eval_stats["expired_count_per_target_per_episode"].append(ec.tolist())
        eval_stats["avg_expired_per_episode"].append(float(np.mean(ec)))
        eval_stats["max_expired_per_episode"].append(float(np.max(ec)))
        eval_stats["avg_reached_per_episode"].append(float(np.mean(trc)))
        eval_stats["performance_per_episode"].append(performance.tolist())
        eval_stats["avg_performance_per_episode"].append(float(np.mean(performance)))
        eval_stats["min_performance_per_episode"].append(float(np.min(performance)))

    if verbose:
        for i in range(N):
            print(f"  Episode {i + 1}: Return={eval_stats['episode_returns'][i]:.2f}, "
                  f"Length={eval_stats['episode_lengths'][i]}, "
                  f"Targets={eval_stats['targets_reached_per_episode'][i]}/{A}, "
                  f"Expired={eval_stats['expired_targets_per_episode'][i]}, "
                  f"MinReached={eval_stats['min_targets_reached_per_episode'][i]}, "
                  f"AvgPerf={eval_stats['avg_performance_per_episode'][i]:.2f}")

        avg_return = np.mean(eval_stats["episode_returns"])
        avg_length = np.mean(eval_stats["episode_lengths"])
        avg_targets = np.mean(eval_stats["targets_reached_per_episode"])
        avg_expired = np.mean(eval_stats["expired_targets_per_episode"])
        avg_min_reached = np.mean(eval_stats["min_targets_reached_per_episode"])
        avg_avg_perf = np.mean(eval_stats["avg_performance_per_episode"])
        avg_min_perf = np.mean(eval_stats["min_performance_per_episode"])
        success_rate = sum(1 for t in eval_stats["targets_reached_per_episode"]
                          if t == A) / num_episodes

        print("\nEvaluation Summary:")
        print(f"  Average Return: {avg_return:.2f}")
        print(f"  Average Length: {avg_length:.1f}")
        print(f"  Average Targets: {avg_targets:.2f}/{A}")
        print(f"  Average Expired: {avg_expired:.2f} +/- {np.std(eval_stats['expired_targets_per_episode']):.2f}")
        print(f"  Average Min Reached: {avg_min_reached:.2f} +/- {np.std(eval_stats['min_targets_reached_per_episode']):.2f}")
        print(f"  Avg Performance (reaches-exp): {avg_avg_perf:.2f}")
        print(f"  Avg Min Performance: {avg_min_perf:.2f}")
        print(f"  Success Rate: {success_rate*100:.1f}%\n")

    return eval_stats
