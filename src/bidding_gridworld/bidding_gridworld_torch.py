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
    urgency_weighted_scalarization: bool = False
    use_target_priorities: bool = True
    bidding_mechanism: str = "all_pay"
    nearest_target_shaping: bool = False
    nearest_expiry_shaping: bool = False
    programmatic_bidding: str = "none"
    # "none" | "nearest_target". The latter uses a one-hot bid from the
    # bidder whose assigned target is currently closest to the shared body.
    # "all_pay" | "winner_pays" | "winner_pays_others_reward"
    battery_capacity: Optional[int] = None
    recharge_station_positions: Optional[Tuple[Tuple[int, int], ...]] = None
    moving_recharge_stations: bool = False
    recharge_station_direction_change_prob: float = 0.1
    recharge_station_move_interval: int = 5
    movement_energy_cost: int = 1
    battery_depletion_penalty: float = 0.0
    charging_agent_enabled: bool = False
    charging_low_battery_threshold: int = 20
    charging_distance_reward_scale: float = 0.0
    charging_recharge_bonus: float = 0.0
    charging_depletion_penalty: float = 0.0
    charging_high_battery_control_penalty: float = 0.0
    feeder_low_battery_control_penalty: float = 0.0
    charging_low_battery_bid_boost: int = 0
    charging_bid_boost_threshold: Optional[int] = None
    charging_activation_margin: Optional[int] = None
    charging_release_window_on_recharge: bool = False
    charging_programmatic_navigation: bool = False
    charging_reserve_features_enabled: bool = False
    charging_nearest_station_features_enabled: bool = False


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
        self.num_bidders = config.num_agents + int(config.charging_agent_enabled)
        self.charging_agent_idx = (
            config.num_agents if config.charging_agent_enabled else None
        )
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
        self.target_priorities = None
        self.window_agent = None
        self.window_steps_remaining = None
        self.step_count = None
        self.previous_distances = None
        self.battery_level = None

        # Moving-target state
        self.target_directions = None
        self.target_move_counters = None

        # Per-environment recharge-station state. The fixed template remains in
        # recharge_station_pos for checkpoint/config compatibility.
        self.current_recharge_station_pos = None
        self.recharge_station_directions = None
        self.recharge_station_move_counters = None

        self.battery_enabled = config.battery_capacity is not None
        self.single_agent_charging_support_enabled = (
            config.single_agent_mode
            and (
                config.charging_distance_reward_scale > 0
                or config.charging_recharge_bonus > 0
                or config.charging_depletion_penalty > 0
            )
        )
        self.recharge_station_pos = self._build_recharge_station_positions()
        self.num_recharge_stations = int(self.recharge_station_pos.shape[0])
        self.energy_feature_dim = 1 + 2 * self.num_recharge_stations if self.battery_enabled else 0
        self.charging_reserve_feature_dim = (
            2 if config.charging_reserve_features_enabled else 0
        )
        self.charging_nearest_station_feature_dim = (
            2 if config.charging_nearest_station_features_enabled else 0
        )
        self.charging_obs_dim = (
            5
            + 3 * self.num_recharge_stations
            + self.charging_reserve_feature_dim
            + self.charging_nearest_station_feature_dim
        )
        if config.charging_agent_enabled and not self.battery_enabled:
            raise ValueError("charging_agent_enabled requires battery_capacity")
        if self.single_agent_charging_support_enabled and not self.battery_enabled:
            raise ValueError(
                "single-agent charging support requires battery_capacity"
            )
        if config.action_window < 1:
            raise ValueError("action_window must be positive")
        if config.programmatic_bidding not in {"none", "nearest_target"}:
            raise ValueError(
                "programmatic_bidding must be 'none' or 'nearest_target'"
            )
        if (
            config.programmatic_bidding != "none"
            and config.bid_upper_bound < 1
        ):
            raise ValueError(
                "programmatic_bidding requires bid_upper_bound >= 1"
            )
        if config.urgency_weighted_scalarization:
            if not config.single_agent_mode:
                raise ValueError(
                    "urgency_weighted_scalarization requires single_agent_mode"
                )
            if config.target_expiry_steps is None:
                raise ValueError(
                    "urgency_weighted_scalarization requires target_expiry_steps"
                )
            if config.nearest_target_shaping or config.nearest_expiry_shaping:
                raise ValueError(
                    "urgency-weighted scalarization uses dense per-target shaping "
                    "and is incompatible with nearest-target shaping"
                )
        if (
            (config.charging_agent_enabled or self.single_agent_charging_support_enabled)
            and not 0 < config.charging_low_battery_threshold <= int(config.battery_capacity)
        ):
            raise ValueError(
                "charging_low_battery_threshold must be in [1, battery_capacity]"
            )
        if config.charging_low_battery_bid_boost < 0:
            raise ValueError("charging_low_battery_bid_boost must be non-negative")
        if (
            config.charging_bid_boost_threshold is not None
            and (
                config.battery_capacity is None
                or not 0
                <= config.charging_bid_boost_threshold
                <= int(config.battery_capacity)
            )
        ):
            raise ValueError(
                "charging_bid_boost_threshold must be in "
                "[0, battery_capacity]"
            )
        if config.feeder_low_battery_control_penalty < 0:
            raise ValueError(
                "feeder_low_battery_control_penalty must be non-negative"
            )
        if (
            config.charging_activation_margin is not None
            and config.charging_activation_margin < 0
        ):
            raise ValueError("charging_activation_margin must be non-negative")
        if not 0.0 <= config.recharge_station_direction_change_prob <= 1.0:
            raise ValueError(
                "recharge_station_direction_change_prob must be in [0, 1]"
            )
        if config.recharge_station_move_interval < 1:
            raise ValueError("recharge_station_move_interval must be positive")
        if config.moving_recharge_stations and not self.battery_enabled:
            raise ValueError(
                "moving_recharge_stations requires battery_capacity"
            )

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
            base_dim = (
                3
                + (6 if include_reached else 5) * self.config.num_agents
                + self.energy_feature_dim
            )
            if not self.config.use_target_priorities:
                base_dim -= self.config.num_agents
            self.obs_dim = base_dim
            self.obs_shape = (self.num_envs, self.obs_dim)
            self.per_agent_obs_dim = None
        else:
            if self.config.visible_targets is None:
                target_block_width = (5 if include_reached else 4) - int(
                    not self.config.use_target_priorities
                )
                self.per_agent_obs_dim = (
                    3
                    + target_block_width * self.config.num_agents
                    + self.energy_feature_dim
                )
            else:
                self.per_agent_obs_dim = (
                    8 + 4 * self.config.visible_targets
                    if include_reached
                    else 7 + 3 * self.config.visible_targets
                ) + self.energy_feature_dim
                if not self.config.use_target_priorities:
                    self.per_agent_obs_dim -= self.config.visible_targets + 1
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
        self.target_priorities = self._sample_target_priorities()
        if self.battery_enabled:
            self.battery_level = torch.full(
                (self.num_envs,),
                int(cfg.battery_capacity),
                device=device,
                dtype=torch.int32,
            )
            self.current_recharge_station_pos = (
                self.recharge_station_pos.unsqueeze(0)
                .expand(self.num_envs, -1, -1)
                .clone()
            )
            self.recharge_station_directions = torch.randint(
                0,
                4,
                (self.num_envs, self.num_recharge_stations),
                generator=self.gen,
                device=device,
                dtype=torch.int32,
            )
            self.recharge_station_move_counters = torch.zeros(
                (self.num_envs, self.num_recharge_stations),
                device=device,
                dtype=torch.int32,
            )
        else:
            self.battery_level = None
            self.current_recharge_station_pos = None
            self.recharge_station_directions = None
            self.recharge_station_move_counters = None
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
        - multi-agent: (num_envs, num_bidders, 2) or (num_envs, num_bidders, 3)
          if window_bidding. num_bidders includes the optional charging agent.
        - single-agent: (num_envs,) or (num_envs, 1)
        """
        cfg = self.config
        device = self.device

        urgency_weights = (
            self._urgency_weights()
            if cfg.urgency_weighted_scalarization
            else None
        )

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
            programmatic_charging_direction = None
            if (
                cfg.charging_agent_enabled
                and cfg.charging_programmatic_navigation
            ):
                action_dir = action_dir.clone()
                programmatic_charging_direction = (
                    self._direction_to_nearest_recharge_station(
                        self.agent_pos
                    )
                )
                action_dir[:, self.charging_agent_idx] = (
                    programmatic_charging_direction
                )
            bids = action[..., 1]
            if cfg.programmatic_bidding == "nearest_target":
                nearest_target = self._compute_distances().argmin(dim=1)
                bids = torch.zeros_like(bids)
                bids.scatter_(
                    1,
                    nearest_target.unsqueeze(1),
                    torch.ones(
                        (self.num_envs, 1),
                        device=device,
                        dtype=bids.dtype,
                    ),
                )
            effective_bids = bids
            charging_bid_active = (
                torch.ones(
                    self.num_envs, device=device, dtype=torch.bool
                )
                if cfg.charging_agent_enabled
                else None
            )
            if (
                cfg.charging_agent_enabled
                and cfg.charging_activation_margin is not None
            ):
                charging_bid_active = self.battery_level <= (
                    self._nearest_recharge_energy_requirement(self.agent_pos)
                    + cfg.charging_activation_margin
                )
                effective_bids = bids.clone()
                effective_bids[:, self.charging_agent_idx] = torch.where(
                    charging_bid_active,
                    effective_bids[:, self.charging_agent_idx],
                    torch.zeros_like(effective_bids[:, self.charging_agent_idx]),
                )
            if (
                cfg.charging_agent_enabled
                and cfg.charging_low_battery_bid_boost > 0
            ):
                if effective_bids is bids:
                    effective_bids = bids.clone()
                boost_threshold = (
                    cfg.charging_low_battery_threshold
                    if cfg.charging_bid_boost_threshold is None
                    else cfg.charging_bid_boost_threshold
                )
                boost_active = (
                    self.battery_level
                    <= boost_threshold
                ) & (effective_bids[:, self.charging_agent_idx] > 0)
                effective_bids[:, self.charging_agent_idx] += (
                    boost_active.to(effective_bids.dtype)
                    * cfg.charging_low_battery_bid_boost
                )
            action_window = action[..., 2] if cfg.window_bidding else None

            in_window = self.window_steps_remaining > 0
            apply_bid_penalty = torch.zeros((self.num_envs,), device=device, dtype=torch.bool)
            current_window_length = torch.zeros((self.num_envs,), device=device, dtype=torch.int32)

            if torch.any(in_window):
                self.window_steps_remaining = torch.where(
                    in_window, self.window_steps_remaining - 1, self.window_steps_remaining
                )

            max_bid = effective_bids.max(dim=1).values
            has_bid = (max_bid > 0) | (cfg.bid_upper_bound == 0)
            winners_mask = effective_bids == max_bid.unsqueeze(1)
            rand = torch.rand(
                effective_bids.shape, device=device, generator=self.gen
            )
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

        previous_agent_pos = self.agent_pos
        battery_before_step = (
            self.battery_level.clone() if self.battery_enabled else None
        )
        previous_station_distance = (
            self._nearest_recharge_distance(previous_agent_pos)
            if (
                self.config.charging_agent_enabled
                or self.single_agent_charging_support_enabled
            )
            else None
        )
        if cfg.charging_agent_enabled:
            optimal_charging_direction = (
                self._direction_to_nearest_recharge_station(
                    previous_agent_pos
                )
            )
            charging_navigation_active = (
                (winning_agent == self.charging_agent_idx)
                & (previous_station_distance > 0)
            )
            charging_direction_optimal = charging_navigation_active & (
                action_dir[:, self.charging_agent_idx]
                == optimal_charging_direction
            )
        else:
            optimal_charging_direction = None
            charging_navigation_active = None
            charging_direction_optimal = None
        new_pos = self._move_position(previous_agent_pos, move_dir)
        self.agent_pos = torch.where(move_mask.unsqueeze(-1), new_pos, self.agent_pos)

        if self.battery_enabled:
            moved = move_mask & (self.agent_pos != previous_agent_pos).any(dim=-1)
            energy_used = moved.to(torch.int32) * int(cfg.movement_energy_cost)
            self.battery_level = torch.clamp(self.battery_level - energy_used, min=0)
            battery_before_recharge = self.battery_level.clone()
            at_recharge_station = self._at_recharge_station(self.agent_pos)
            battery_recharged = at_recharge_station & (
                self.battery_level < int(cfg.battery_capacity)
            )
            self.battery_level = torch.where(
                at_recharge_station,
                torch.full_like(self.battery_level, int(cfg.battery_capacity)),
                self.battery_level,
            )
            battery_depleted = (self.battery_level <= 0) & (~at_recharge_station)
        else:
            battery_before_recharge = None
            at_recharge_station = torch.zeros(
                self.num_envs, device=device, dtype=torch.bool
            )
            battery_depleted = torch.zeros(
                self.num_envs, device=device, dtype=torch.bool
            )
            battery_recharged = torch.zeros(
                self.num_envs, device=device, dtype=torch.bool
            )

        # Targets reached
        targets_just_reached = self._positions_equal(self.agent_pos, self.target_pos) & (self.targets_reached == 0)
        effective_target_priorities = (
            self.target_priorities
            if cfg.use_target_priorities
            else torch.ones_like(self.target_priorities)
        )
        target_priorities_just_reached = (
            effective_target_priorities
            * targets_just_reached.to(effective_target_priorities.dtype)
        )
        self.targets_reached = torch.where(targets_just_reached, torch.ones_like(self.targets_reached), self.targets_reached)
        self.targets_reached_count = self.targets_reached_count + targets_just_reached.to(torch.int32)
        self.target_counters = torch.where(targets_just_reached, torch.zeros_like(self.target_counters), self.target_counters)

        if torch.any(battery_depleted):
            tow_pos = self._nearest_recharge_station(self.agent_pos)
            self.agent_pos = torch.where(
                battery_depleted.unsqueeze(-1), tow_pos, self.agent_pos
            )
            self.battery_level = torch.where(
                battery_depleted,
                torch.full_like(self.battery_level, int(cfg.battery_capacity)),
                self.battery_level,
            )

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
                rewards = rewards + (
                    targets_just_reached.to(torch.float32)
                    * cfg.target_reward
                    * effective_target_priorities.to(torch.float32)
                    * decay
                ).sum(dim=1)
            else:
                rewards = rewards + (
                    targets_just_reached.to(torch.float32)
                    * cfg.target_reward
                    * effective_target_priorities.to(torch.float32)
                ).sum(dim=1)

            if cfg.target_expiry_penalty > 0:
                rewards = rewards - cfg.target_expiry_penalty * targets_expired.to(torch.float32).sum(dim=1)
            if cfg.battery_depletion_penalty > 0:
                rewards = rewards - cfg.battery_depletion_penalty * battery_depleted.to(torch.float32)
            if self.single_agent_charging_support_enabled:
                current_station_distance = self._nearest_recharge_distance(
                    self.agent_pos
                )
                activation_margin = (
                    0
                    if cfg.charging_activation_margin is None
                    else cfg.charging_activation_margin
                )
                charging_active = battery_before_step <= (
                    previous_station_distance * int(cfg.movement_energy_cost)
                    + activation_margin
                )
                threshold = float(cfg.charging_low_battery_threshold)
                urgency = torch.clamp(
                    (threshold - battery_before_step.to(torch.float32))
                    / threshold,
                    min=0.0,
                    max=1.0,
                )
                station_progress = (
                    previous_station_distance - current_station_distance
                ).to(torch.float32)
                rewards = rewards + (
                    cfg.charging_distance_reward_scale
                    * urgency
                    * station_progress
                    * charging_active.to(torch.float32)
                )
                rewards = rewards + (
                    cfg.charging_recharge_bonus
                    * (
                        (
                            int(cfg.battery_capacity)
                            - battery_before_recharge
                        ).to(torch.float32)
                        / float(cfg.battery_capacity)
                    )
                    * battery_recharged.to(torch.float32)
                    * (
                        battery_before_recharge
                        <= cfg.charging_low_battery_threshold
                    ).to(torch.float32)
                )
                rewards = rewards - (
                    cfg.charging_depletion_penalty
                    * battery_depleted.to(torch.float32)
                )
        else:
            rewards = torch.zeros(
                (self.num_envs, self.num_bidders),
                device=device,
                dtype=torch.float32,
            )
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
                    winner_mask = torch.zeros(
                        (self.num_envs, self.num_bidders),
                        device=device,
                        dtype=torch.float32,
                    )
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
                rewards[:, :cfg.num_agents] += (
                    cfg.distance_reward_scale
                    * dist_improve
                    * (self.targets_reached == 0).to(torch.float32)
                )

            rewards[:, :cfg.num_agents] += (
                cfg.target_reward
                * effective_target_priorities.to(torch.float32)
                * targets_just_reached.to(torch.float32)
            )
            if cfg.target_expiry_penalty > 0:
                rewards[:, :cfg.num_agents] -= (
                    cfg.target_expiry_penalty * targets_expired.to(torch.float32)
                )
            if cfg.battery_depletion_penalty > 0 and torch.any(battery_depleted):
                valid_controller = battery_depleted & (winning_agent >= 0)
                if torch.any(valid_controller):
                    controller_idx = winning_agent.clamp(min=0).to(torch.int64).view(-1, 1)
                    penalty = (
                        -cfg.battery_depletion_penalty
                        * valid_controller.to(torch.float32)
                    ).view(-1, 1)
                    rewards = rewards.scatter_add(1, controller_idx, penalty)

            if cfg.charging_agent_enabled:
                charging_controls = winning_agent == self.charging_agent_idx
                feeder_controls = (
                    (winning_agent >= 0)
                    & (winning_agent < cfg.num_agents)
                )
                feeder_charging_conflict = (
                    feeder_controls & charging_bid_active
                )
                if (
                    cfg.feeder_low_battery_control_penalty > 0
                    and torch.any(feeder_charging_conflict)
                ):
                    feeder_idx = winning_agent.clamp(min=0).to(
                        torch.int64
                    ).view(-1, 1)
                    feeder_penalty = (
                        -cfg.feeder_low_battery_control_penalty
                        * feeder_charging_conflict.to(torch.float32)
                    ).view(-1, 1)
                    rewards = rewards.scatter_add(
                        1, feeder_idx, feeder_penalty
                    )
                threshold = float(cfg.charging_low_battery_threshold)
                urgency = torch.clamp(
                    (threshold - battery_before_step.to(torch.float32)) / threshold,
                    min=0.0,
                    max=1.0,
                )
                current_station_distance = self._nearest_recharge_distance(
                    self.agent_pos
                )
                station_progress = (
                    previous_station_distance - current_station_distance
                ).to(torch.float32)
                charging_reward = (
                    cfg.charging_distance_reward_scale
                    * urgency
                    * station_progress
                    * charging_controls.to(torch.float32)
                )
                charging_reward += (
                    cfg.charging_recharge_bonus
                    * (
                        (
                            int(cfg.battery_capacity)
                            - battery_before_recharge
                        ).to(torch.float32)
                        / float(cfg.battery_capacity)
                    )
                    * battery_recharged.to(torch.float32)
                    * (
                        battery_before_recharge
                        <= cfg.charging_low_battery_threshold
                    ).to(torch.float32)
                    * charging_controls.to(torch.float32)
                )
                charging_reward -= (
                    cfg.charging_depletion_penalty
                    * battery_depleted.to(torch.float32)
                )
                charging_reward -= (
                    cfg.charging_high_battery_control_penalty
                    * (
                        battery_before_step
                        > cfg.charging_low_battery_threshold
                    ).to(torch.float32)
                    * charging_controls.to(torch.float32)
                )
                rewards[:, self.charging_agent_idx] += charging_reward

            if (
                cfg.charging_agent_enabled
                and cfg.charging_release_window_on_recharge
            ):
                release_charging_window = battery_recharged & (
                    winning_agent == self.charging_agent_idx
                )
                self.window_steps_remaining = torch.where(
                    release_charging_window,
                    torch.zeros_like(self.window_steps_remaining),
                    self.window_steps_remaining,
                )
                self.window_agent = torch.where(
                    release_charging_window,
                    torch.full_like(self.window_agent, -1),
                    self.window_agent,
                )

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
                per_obj = per_obj + (
                    targets_just_reached.to(torch.float32)
                    * cfg.target_reward
                    * effective_target_priorities.to(torch.float32)
                    * decay
                )
            else:
                per_obj = per_obj + (
                    targets_just_reached.to(torch.float32)
                    * cfg.target_reward
                    * effective_target_priorities.to(torch.float32)
                )
            if cfg.target_expiry_penalty > 0:
                per_obj = per_obj - cfg.target_expiry_penalty * targets_expired.to(torch.float32)
            if urgency_weights is not None:
                rewards = (urgency_weights * per_obj).sum(dim=1)
                if cfg.battery_depletion_penalty > 0:
                    rewards = rewards - (
                        cfg.battery_depletion_penalty
                        * battery_depleted.to(torch.float32)
                    )

        self.previous_distances = current_distances

        if cfg.moving_targets:
            self._move_targets(targets_just_reached, targets_expired)
            terminated = torch.zeros((self.num_envs,), device=device, dtype=torch.bool)
            truncated = self.step_count >= cfg.max_steps
        else:
            all_targets_reached = self.targets_reached.sum(dim=1) == cfg.num_agents
            terminated = all_targets_reached.to(torch.bool)
            truncated = (self.step_count >= cfg.max_steps) & (~terminated)

        if cfg.moving_recharge_stations:
            self._move_recharge_stations()

        obs = self._get_observation()
        info = {
            "winning_agent": winning_agent,
            "bids": bids,
            "effective_bids": (
                effective_bids if not cfg.single_agent_mode else None
            ),
            "window_agent": self.window_agent,
            "window_steps_remaining": self.window_steps_remaining,
            "bid_penalty_applied": apply_bid_penalty,
            "targets_just_reached": targets_just_reached,
            "target_priorities_just_reached": target_priorities_just_reached,
            "targets_just_expired": targets_expired,
            "battery_level": self.battery_level,
            "battery_depleted": battery_depleted,
            "battery_recharged": battery_recharged,
            "at_recharge_station": at_recharge_station,
            "recharge_station_positions": (
                self.current_recharge_station_pos.clone()
                if self.battery_enabled
                else None
            ),
            "charging_agent_idx": self.charging_agent_idx,
            "charging_bid_active": (
                charging_bid_active if not cfg.single_agent_mode else None
            ),
            "programmatic_charging_direction": (
                programmatic_charging_direction
                if not cfg.single_agent_mode
                else None
            ),
            "optimal_charging_direction": optimal_charging_direction,
            "charging_navigation_active": charging_navigation_active,
            "charging_direction_optimal": charging_direction_optimal,
        }
        if cfg.single_agent_mode:
            info["per_objective_rewards"] = per_obj
            info["urgency_weights"] = urgency_weights
        else:
            info["is_bidding_round"] = ~in_window
            reward_without_bid = rewards - bid_net_effect
            info["reward_no_bid_sum"] = reward_without_bid.sum(dim=1)
            bid_policy_rewards = bid_net_effect.clone()
            valid_controller = winning_agent >= 0
            if torch.any(valid_controller):
                controller_index = winning_agent.clamp(min=0).long().view(-1, 1)
                controller_team_reward = (
                    reward_without_bid.sum(dim=1)
                    * valid_controller.to(torch.float32)
                ).view(-1, 1)
                bid_policy_rewards.scatter_add_(
                    1, controller_index, controller_team_reward
                )
            info["bid_policy_controller_team_rewards"] = bid_policy_rewards
        return obs, rewards, terminated, truncated, info

    def _urgency_weights(self) -> torch.Tensor:
        """Return normalized inverse-TTL weights for active targets.

        Weights describe the state in which the action is selected. Reached
        static targets receive zero weight; all active targets have at least
        one remaining step, avoiding a singularity at expiry.
        """
        expiry_steps = int(self.config.target_expiry_steps)
        active = self.targets_reached == 0
        time_to_live = torch.clamp(
            expiry_steps - self.target_counters,
            min=1,
        ).to(torch.float32)
        inverse_ttl = active.to(torch.float32) / time_to_live
        normalizer = inverse_ttl.sum(dim=1, keepdim=True)
        return torch.where(
            normalizer > 0,
            inverse_ttl / normalizer.clamp_min(torch.finfo(torch.float32).tiny),
            torch.zeros_like(inverse_ttl),
        )

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
        energy_features = self._get_energy_features()

        window_denom = float(max(cfg.action_window, 1))
        window_steps = (self.window_steps_remaining.to(torch.float32) / window_denom).unsqueeze(-1)

        target_parts = [agent_pos, target_pos.reshape(self.num_envs, -1)]
        if include_reached:
            target_parts.append(targets_reached)
        target_parts.append(target_counters)
        if cfg.use_target_priorities:
            target_parts.append(
                self.target_priorities.to(torch.float32) / 4.0
            )
        target_obs = torch.cat(target_parts, dim=-1)

        if cfg.single_agent_mode:
            min_count = self.targets_reached_count.min(dim=1).values
            count_denom = float(max(cfg.num_agents, 1))
            relative = (self.targets_reached_count - min_count.unsqueeze(1)).to(torch.float32)
            relative = torch.clamp(relative / count_denom, 0.0, 1.0)
            return torch.cat(
                [target_obs, window_steps, relative, energy_features], dim=-1
            )

        return torch.cat([target_obs, energy_features, window_steps], dim=-1)

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
        target_priorities = self.target_priorities.to(torch.float32) / 4.0
        energy_features = self._get_energy_features()

        window_denom = float(max(cfg.action_window, 1))
        window_steps = self.window_steps_remaining.to(torch.float32) / window_denom
        window_steps = window_steps.unsqueeze(-1)

        if cfg.single_agent_mode:
            base_parts = [agent_pos, target_pos.reshape(self.num_envs, -1)]
            if include_reached:
                base_parts.append(targets_reached)
            base_parts.append(target_counters)
            if cfg.use_target_priorities:
                base_parts.append(target_priorities)
            base_parts.append(window_steps)
            base_obs = torch.cat(base_parts, dim=-1)
            min_count = self.targets_reached_count.min(dim=1).values
            count_denom = float(max(cfg.num_agents, 1))
            relative = (self.targets_reached_count - min_count.unsqueeze(1)).to(torch.float32)
            relative = torch.clamp(relative / count_denom, 0.0, 1.0)
            return torch.cat([base_obs, relative, energy_features], dim=-1)

        if cfg.visible_targets is None:
            reordered_pos = target_pos[:, self._reorder_idx, :].reshape(self.num_envs, cfg.num_agents, -1)
            reordered_counters = target_counters[:, self._reorder_idx]
            reordered_priorities = target_priorities[:, self._reorder_idx]
            common_parts = [
                agent_pos.unsqueeze(1).expand(-1, cfg.num_agents, -1),
                reordered_pos,
            ]
            if include_reached:
                reordered_reached = targets_reached[:, self._reorder_idx]
                common_parts.append(reordered_reached)
            common_parts.append(reordered_counters)
            if cfg.use_target_priorities:
                common_parts.append(reordered_priorities)
            common_parts.extend(
                [
                    energy_features.unsqueeze(1).expand(-1, cfg.num_agents, -1),
                    window_steps.unsqueeze(1).expand(-1, cfg.num_agents, -1),
                ]
            )
            return torch.cat(common_parts, dim=-1)

        if cfg.visible_targets == 0:
            own_pos = target_pos  # (num_envs, num_agents, 2) — each agent reads its own slice
            own_counter = target_counters.unsqueeze(-1)  # (num_envs, num_agents, 1)
            own_priority = target_priorities.unsqueeze(-1)
            own_parts = [
                agent_pos.unsqueeze(1).expand(-1, cfg.num_agents, -1),
                own_pos,
            ]
            if include_reached:
                own_reached = targets_reached.unsqueeze(-1)  # (num_envs, num_agents, 1)
                own_parts.append(own_reached)
            own_parts.append(own_counter)
            if cfg.use_target_priorities:
                own_parts.append(own_priority)
            own_parts.extend(
                [
                    energy_features.unsqueeze(1).expand(-1, cfg.num_agents, -1),
                    window_steps.unsqueeze(1).expand(-1, cfg.num_agents, -1),
                ]
            )
            return torch.cat(own_parts, dim=-1)

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
        own_priority = target_priorities.unsqueeze(-1)
        priority_exp = target_priorities.unsqueeze(1).expand(-1, cfg.num_agents, -1)
        vis_priorities = priority_exp.gather(2, idx)

        visible_parts = [
            agent_pos.unsqueeze(1).expand(-1, cfg.num_agents, -1),
            own_pos,
            vis_pos,
        ]
        if include_reached:
            targets_reached_exp = targets_reached.unsqueeze(1).expand(-1, cfg.num_agents, -1)
            vis_reached = targets_reached_exp.gather(2, idx)
            own_reached = targets_reached.unsqueeze(-1)
            visible_parts.extend([own_reached, vis_reached])
        visible_parts.append(own_counter)
        if cfg.use_target_priorities:
            visible_parts.extend([own_priority, vis_priorities])
        visible_parts.extend(
            [
                energy_features.unsqueeze(1).expand(-1, cfg.num_agents, -1),
                window_steps.unsqueeze(1).expand(-1, cfg.num_agents, -1),
            ]
        )
        return torch.cat(visible_parts, dim=-1)

    def _get_energy_features(self) -> torch.Tensor:
        """Return normalized shared battery and recharge-station features."""
        if not self.battery_enabled:
            return torch.empty(
                (self.num_envs, 0), device=self.device, dtype=torch.float32
            )
        battery = (
            self.battery_level.to(torch.float32) / float(self.config.battery_capacity)
        ).unsqueeze(-1)
        denom = float(self.config.grid_size - 1) if self.config.grid_size > 1 else 1.0
        stations = (
            self._batched_recharge_station_positions().to(torch.float32)
            / denom
        ).reshape(self.num_envs, -1)
        return torch.cat([battery, stations], dim=-1)

    def get_charging_observation(self) -> torch.Tensor:
        """Return the compact observation used by the optional charging policy."""
        if not self.config.charging_agent_enabled:
            raise RuntimeError("Charging observations require charging_agent_enabled")

        denom = float(self.grid_size - 1) if self.grid_size > 1 else 1.0
        agent_pos = self.agent_pos.to(torch.float32) / denom
        battery = (
            self.battery_level.to(torch.float32)
            / float(self.config.battery_capacity)
        ).unsqueeze(-1)
        station_positions = self._batched_recharge_station_positions()
        physical_station_distances = (
            station_positions - self.agent_pos.unsqueeze(1)
        ).abs().sum(dim=-1)
        if self.config.charging_reserve_features_enabled:
            grid_energy_diameter = float(
                max(
                    2
                    * (self.grid_size - 1)
                    * int(self.config.movement_energy_cost),
                    1,
                )
            )
            physical_battery = (
                self.battery_level.to(torch.float32) / grid_energy_diameter
            ).unsqueeze(-1)
            reserve_slack = (
                self.battery_level.to(torch.float32)
                - (
                    physical_station_distances.min(dim=1).values
                    * int(self.config.movement_energy_cost)
                ).to(torch.float32)
            ).div(grid_energy_diameter).unsqueeze(-1)
            reserve_features = torch.cat(
                [physical_battery, reserve_slack], dim=-1
            )
        else:
            reserve_features = torch.empty(
                (self.num_envs, 0),
                device=self.device,
                dtype=torch.float32,
            )
        nearest_station_idx = physical_station_distances.argmin(dim=1)
        nearest_station = station_positions.gather(
            1,
            nearest_station_idx.view(-1, 1, 1).expand(-1, 1, 2),
        ).squeeze(1)
        if self.config.charging_nearest_station_features_enabled:
            nearest_station_features = (
                nearest_station.to(torch.float32)
                - self.agent_pos.to(torch.float32)
            ) / denom
        else:
            nearest_station_features = torch.empty(
                (self.num_envs, 0),
                device=self.device,
                dtype=torch.float32,
            )
        relative_stations = (
            station_positions.to(torch.float32)
            - self.agent_pos.unsqueeze(1).to(torch.float32)
        ) / denom
        station_distances = relative_stations.abs().sum(dim=-1) / 2.0
        window_denom = float(max(self.config.action_window, 1))
        window_steps = (
            self.window_steps_remaining.to(torch.float32) / window_denom
        ).unsqueeze(-1)
        charging_controls_window = (
            (self.window_agent == self.charging_agent_idx)
            & (self.window_steps_remaining > 0)
        ).to(torch.float32).unsqueeze(-1)
        return torch.cat(
            [
                agent_pos,
                battery,
                reserve_features,
                nearest_station_features,
                relative_stations.reshape(self.num_envs, -1),
                station_distances,
                window_steps,
                charging_controls_window,
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
            new_priorities = self._sample_target_priorities()
            self.target_pos = torch.where(respawn_mask.unsqueeze(-1), new_pos, self.target_pos)
            self.target_priorities = torch.where(respawn_mask, new_priorities, self.target_priorities)
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

    def _sample_target_priorities(self) -> torch.Tensor:
        """Sample integer feeding priorities uniformly from 1 through 4."""
        if not self.config.use_target_priorities:
            return torch.ones(
                (self.num_envs, self.config.num_agents),
                device=self.device,
                dtype=torch.int32,
            )
        return torch.randint(
            1,
            5,
            (self.num_envs, self.config.num_agents),
            generator=self.gen,
            device=self.device,
            dtype=torch.int32,
        )

    def _move_recharge_stations(self) -> None:
        """Advance independent bounded random walks for recharge stations."""
        cfg = self.config
        self.recharge_station_move_counters += 1
        should_move = (
            self.recharge_station_move_counters
            >= cfg.recharge_station_move_interval
        )
        self.recharge_station_move_counters = torch.where(
            should_move,
            torch.zeros_like(self.recharge_station_move_counters),
            self.recharge_station_move_counters,
        )
        if not torch.any(should_move):
            return

        if cfg.recharge_station_direction_change_prob > 0:
            change = (
                torch.rand(
                    self.recharge_station_directions.shape,
                    device=self.device,
                    generator=self.gen,
                )
                < cfg.recharge_station_direction_change_prob
            ) & should_move
            new_directions = torch.randint(
                0,
                4,
                self.recharge_station_directions.shape,
                generator=self.gen,
                device=self.device,
                dtype=torch.int32,
            )
            self.recharge_station_directions = torch.where(
                change, new_directions, self.recharge_station_directions
            )

        current = self.current_recharge_station_pos
        proposed = self._move_position(
            current, self.recharge_station_directions
        )
        hit_wall = (proposed == current).all(dim=-1) & should_move
        if torch.any(hit_wall):
            valid_directions = self._valid_directions_mask(current)
            random_scores = torch.rand(
                valid_directions.shape,
                device=self.device,
                generator=self.gen,
            ).masked_fill(~valid_directions, -1.0)
            replacement_directions = random_scores.argmax(dim=-1).to(
                torch.int32
            )
            self.recharge_station_directions = torch.where(
                hit_wall,
                replacement_directions,
                self.recharge_station_directions,
            )
            proposed = self._move_position(
                current, self.recharge_station_directions
            )

        self.current_recharge_station_pos = torch.where(
            should_move.unsqueeze(-1), proposed, current
        )

    def _build_recharge_station_positions(self) -> torch.Tensor:
        """Validate and materialize fixed recharge-station coordinates."""
        cfg = self.config
        if cfg.battery_capacity is None:
            return torch.empty((0, 2), device=self.device, dtype=torch.int32)
        if cfg.battery_capacity <= 0:
            raise ValueError("battery_capacity must be positive when battery mode is enabled")
        if cfg.movement_energy_cost <= 0:
            raise ValueError("movement_energy_cost must be positive")

        positions = cfg.recharge_station_positions
        if not positions:
            raise ValueError(
                "recharge_station_positions must contain at least one station "
                "when battery mode is enabled"
            )
        station_pos = torch.tensor(positions, device=self.device, dtype=torch.int32)
        if station_pos.ndim != 2 or station_pos.shape[1] != 2:
            raise ValueError("recharge_station_positions must be a sequence of (row, col) pairs")
        if torch.any(station_pos < 0) or torch.any(station_pos >= cfg.grid_size):
            raise ValueError("recharge station positions must lie within the grid")
        if torch.unique(station_pos, dim=0).shape[0] != station_pos.shape[0]:
            raise ValueError("recharge station positions must be unique")
        return station_pos

    def _at_recharge_station(self, positions: torch.Tensor) -> torch.Tensor:
        """Return whether each batched position occupies any recharge station."""
        if not self.battery_enabled:
            return torch.zeros(positions.shape[0], device=self.device, dtype=torch.bool)
        matches = (
            positions.unsqueeze(1)
            == self._batched_recharge_station_positions()
        )
        return matches.all(dim=-1).any(dim=1)

    def _batched_recharge_station_positions(self) -> torch.Tensor:
        """Return current station coordinates with shape (env, station, 2)."""
        if self.current_recharge_station_pos is not None:
            return self.current_recharge_station_pos
        return self.recharge_station_pos.unsqueeze(0).expand(
            self.num_envs, -1, -1
        )

    def _nearest_recharge_station(self, positions: torch.Tensor) -> torch.Tensor:
        """Return the nearest station to each position using Manhattan distance."""
        distances = self._recharge_station_distances(positions)
        nearest_idx = distances.argmin(dim=1)
        stations = self._batched_recharge_station_positions()
        return stations.gather(
            1, nearest_idx.view(-1, 1, 1).expand(-1, 1, 2)
        ).squeeze(1)

    def _recharge_station_distances(self, positions: torch.Tensor) -> torch.Tensor:
        return (
            positions.unsqueeze(1)
            - self._batched_recharge_station_positions()
        ).abs().sum(dim=-1)

    def _nearest_recharge_distance(self, positions: torch.Tensor) -> torch.Tensor:
        return self._recharge_station_distances(positions).min(dim=1).values

    def _nearest_recharge_energy_requirement(
        self, positions: torch.Tensor
    ) -> torch.Tensor:
        """Return battery units needed for a shortest path to a station."""
        return (
            self._nearest_recharge_distance(positions)
            * int(self.config.movement_energy_cost)
        )

    def _direction_to_nearest_recharge_station(
        self, positions: torch.Tensor
    ) -> torch.Tensor:
        """Return a deterministic shortest-path direction to a nearest station."""
        delta = self._nearest_recharge_station(positions) - positions
        return torch.where(
            delta[:, 1] < 0,
            torch.zeros_like(delta[:, 1], dtype=torch.int64),
            torch.where(
                delta[:, 1] > 0,
                torch.ones_like(delta[:, 1], dtype=torch.int64),
                torch.where(
                    delta[:, 0] < 0,
                    torch.full_like(delta[:, 0], 2, dtype=torch.int64),
                    torch.full_like(delta[:, 0], 3, dtype=torch.int64),
                ),
            ),
        )

    def _build_reset_positions(self) -> torch.Tensor:
        """Build a cached grid of positions excluding (0,0) for reset sampling."""
        grid = torch.arange(self.config.grid_size, device=self.device, dtype=torch.int32)
        rows, cols = torch.meshgrid(grid, grid, indexing="ij")
        positions = torch.stack([rows, cols], dim=-1).reshape(-1, 2)
        mask = ~((positions[:, 0] == 0) & (positions[:, 1] == 0))
        return positions[mask].to(torch.int32)

    def get_render_state(self) -> Dict[str, Any]:
        """Return exact batched state needed by rollout visualizations."""
        return {
            "agent_positions": self.agent_pos.detach().cpu().numpy().copy(),
            "target_positions": self.target_pos.detach().cpu().numpy().copy(),
            "target_priorities": self.target_priorities.detach()
            .cpu()
            .numpy()
            .copy(),
            "target_counters": self.target_counters.detach().cpu().numpy().copy(),
            "targets_reached_count": self.targets_reached_count.detach()
            .cpu()
            .numpy()
            .copy(),
            "battery_levels": (
                self.battery_level.detach().cpu().numpy().copy()
                if self.battery_level is not None
                else None
            ),
            "recharge_station_positions": (
                self._batched_recharge_station_positions()
                .detach()
                .cpu()
                .numpy()
                .copy()
                if self.battery_enabled
                else None
            ),
            "window_agents": self.window_agent.detach().cpu().numpy().copy(),
            "window_steps_remaining": self.window_steps_remaining.detach()
            .cpu()
            .numpy()
            .copy(),
        }

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
        fps: int = 10,
        frame_stride: int = 1,
    ) -> None:
        """Create a multi-agent MP4 rollout for the active environment mode."""
        import matplotlib.patches as mpatches

        if fps <= 0:
            raise ValueError("fps must be positive")
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive")

        render_states = episode_data.get("render_states", [])
        if not render_states:
            render_states = self._legacy_render_states(episode_data)
        if not render_states:
            raise ValueError("episode_data contains no renderable states")

        actions_list = episode_data.get("actions", [])
        rewards_list = episode_data.get("rewards", [])
        details = episode_data.get("step_details", [])
        cumulative_priority = []
        cumulative_reaches = []
        cumulative_expired = []
        cumulative_recharges = []
        cumulative_depletions = []
        priority_total = reaches_total = expired_total = 0
        recharge_total = depletion_total = 0
        for frame_idx in range(len(render_states)):
            detail = details[frame_idx] if frame_idx < len(details) else {}
            reached_priorities = detail.get(
                "target_priorities_just_reached", []
            )
            priority_total += int(sum(reached_priorities))
            reaches_total += sum(int(value > 0) for value in reached_priorities)
            expired_total += sum(
                bool(value) for value in detail.get("targets_just_expired", [])
            )
            recharge_total += int(detail.get("battery_recharged", False))
            depletion_total += int(detail.get("battery_depleted", False))
            cumulative_priority.append(priority_total)
            cumulative_reaches.append(reaches_total)
            cumulative_expired.append(expired_total)
            cumulative_recharges.append(recharge_total)
            cumulative_depletions.append(depletion_total)

        identity_colors = [
            "#3569A8",
            "#C43C39",
            "#8E5BA6",
            "#2F8A62",
            "#D17A22",
            "#B44E83",
            "#607D3B",
            "#6D5B4B",
        ]
        priority_colors = {
            1: "#9ECAE1",
            2: "#74C476",
            3: "#FDBE6F",
            4: "#E6550D",
        }
        direction_labels = {0: "LEFT", 1: "RIGHT", 2: "UP", 3: "DOWN"}
        direction_delta = {
            0: (-0.8, 0.0),
            1: (0.8, 0.0),
            2: (0.0, -0.8),
            3: (0.0, 0.8),
        }

        fig = plt.figure(figsize=(13.8, 8.2), facecolor="#F4F6F8")
        gs = fig.add_gridspec(
            1, 2, width_ratios=[8.2, 5.0], wspace=0.08
        )
        grid_ax = fig.add_subplot(gs[0])
        info_ax = fig.add_subplot(gs[1])

        def draw_frame(frame_idx: int) -> np.ndarray:
            grid_ax.clear()
            info_ax.clear()
            snapshot = render_states[frame_idx]
            detail = details[frame_idx] if frame_idx < len(details) else {}
            actions = (
                actions_list[frame_idx]
                if frame_idx < len(actions_list)
                else {}
            )
            agent_row, agent_col = snapshot["agent_position"]
            target_positions = snapshot["target_positions"]
            priorities = snapshot["target_priorities"]
            battery = snapshot.get("battery_level")
            battery_after = detail.get("battery_level_after", battery)
            stations = snapshot.get("recharge_station_positions", [])
            winning_agent = int(detail.get("winning_agent", -1))
            charging_idx = self.charging_agent_idx

            grid_ax.set_facecolor("#FFFFFF")
            grid_ax.set_xlim(-0.5, self.grid_size - 0.5)
            grid_ax.set_ylim(self.grid_size - 0.5, -0.5)
            grid_ax.set_aspect("equal")
            grid_ax.set_xticks(range(0, self.grid_size, 5))
            grid_ax.set_yticks(range(0, self.grid_size, 5))
            grid_ax.tick_params(labelsize=8, colors="#5C6670")
            grid_ax.grid(
                which="major", color="#DCE1E5", linewidth=0.7, alpha=0.8
            )
            for spine in grid_ax.spines.values():
                spine.set_color("#AAB2B9")

            for station_idx, (row, col) in enumerate(stations):
                grid_ax.scatter(
                    [col],
                    [row],
                    marker="D",
                    s=185,
                    facecolor="#38B7C4",
                    edgecolor="#075B66",
                    linewidth=1.8,
                    zorder=3,
                )
                grid_ax.text(
                    col,
                    row,
                    f"S{station_idx + 1}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    fontweight="bold",
                    zorder=4,
                )

            for target_idx, ((row, col), priority) in enumerate(
                zip(target_positions, priorities)
            ):
                priority = int(priority)
                identity_color = identity_colors[
                    target_idx % len(identity_colors)
                ]
                target_facecolor = (
                    priority_colors.get(priority, "#BDBDBD")
                    if self.config.use_target_priorities
                    else "#DCE6F1"
                )
                grid_ax.scatter(
                    [col],
                    [row],
                    marker="o",
                    s=205,
                    facecolor=target_facecolor,
                    edgecolor=identity_color,
                    linewidth=2.2,
                    zorder=5,
                )
                grid_ax.text(
                    col,
                    row,
                    f"F{target_idx + 1}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#17202A",
                    fontweight="bold",
                    zorder=6,
                )
                if self.config.use_target_priorities:
                    grid_ax.text(
                        col,
                        row - 0.85,
                        f"P{priority}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#17202A",
                        fontweight="bold",
                        bbox={
                            "boxstyle": "round,pad=0.18",
                            "facecolor": "white",
                            "edgecolor": priority_colors.get(priority, "#BDBDBD"),
                            "linewidth": 1.2,
                        },
                        zorder=7,
                    )

            battery_fraction = (
                float(battery) / float(self.config.battery_capacity)
                if battery is not None and self.config.battery_capacity
                else 1.0
            )
            battery_color = (
                "#2E9D61"
                if battery_fraction > 0.55
                else "#E6A23C"
                if battery_fraction > 0.25
                else "#D64545"
            )
            controller_color = "#38B7C4"
            if 0 <= winning_agent < self.num_agents:
                controller_color = identity_colors[
                    winning_agent % len(identity_colors)
                ]
            grid_ax.add_patch(
                mpatches.Circle(
                    (agent_col, agent_row),
                    0.62,
                    facecolor="white",
                    edgecolor=controller_color,
                    linewidth=3.2,
                    zorder=8,
                )
            )
            grid_ax.add_patch(
                mpatches.FancyBboxPatch(
                    (agent_col - 0.28, agent_row - 0.25),
                    0.56,
                    0.5,
                    boxstyle="round,pad=0.04",
                    facecolor="#D7E3EA",
                    edgecolor="#263238",
                    linewidth=1.4,
                    zorder=9,
                )
            )
            grid_ax.plot(
                [agent_col - 0.13, agent_col + 0.13],
                [agent_row - 0.07, agent_row - 0.07],
                marker="o",
                markersize=3,
                linestyle="None",
                color="#263238",
                zorder=10,
            )
            if battery is not None:
                grid_ax.add_patch(
                    mpatches.Rectangle(
                        (agent_col - 0.24, agent_row + 0.11),
                        0.48 * battery_fraction,
                        0.08,
                        facecolor=battery_color,
                        edgecolor="none",
                        zorder=10,
                    )
                )
            winner_action = actions.get(f"agent_{winning_agent}", {})
            if winning_agent >= 0 and winner_action:
                dx, dy = direction_delta.get(
                    int(winner_action.get("direction", 0)), (0.0, 0.0)
                )
                grid_ax.arrow(
                    agent_col,
                    agent_row,
                    dx,
                    dy,
                    width=0.05,
                    head_width=0.3,
                    head_length=0.25,
                    length_includes_head=True,
                    color=controller_color,
                    zorder=7,
                )

            if self.config.charging_agent_enabled:
                motion_label = (
                    "MOVING RECHARGE STATIONS"
                    if self.config.moving_recharge_stations
                    else "FIXED RECHARGE STATIONS"
                )
            else:
                motion_label = (
                    "MOVING TARGETS" if self.config.moving_targets else "STATIC TARGETS"
                )
            environment_label = (
                "Priority Cat Feeder"
                if self.config.use_target_priorities
                else f"{self.num_agents}-Target Gridworld"
            )
            grid_ax.set_title(
                f"{environment_label} | Step {frame_idx:,}\n{motion_label}",
                fontsize=12,
                fontweight="bold",
                color="#1F2D3D",
                pad=10,
            )
            grid_ax.set_xlabel("Column", fontsize=9, color="#5C6670")
            grid_ax.set_ylabel("Row", fontsize=9, color="#5C6670")

            info_ax.set_facecolor("#F4F6F8")
            info_ax.set_xlim(0, 1)
            info_ax.set_ylim(0, 1)
            info_ax.axis("off")
            if self.config.charging_agent_enabled and winning_agent == charging_idx:
                controller_label = "CHARGER"
            elif 0 <= winning_agent < self.num_agents:
                controller_label = (
                    f"FEEDER {winning_agent + 1}"
                    if self.config.use_target_priorities
                    else f"AGENT {winning_agent + 1}"
                )
            else:
                controller_label = "NONE"
            info_ax.text(
                0.02,
                0.975,
                "DECENTRALIZED BIDDING",
                fontsize=14,
                fontweight="bold",
                color="#1F2D3D",
                va="top",
            )
            info_ax.text(
                0.02,
                0.93,
                f"Controller: {controller_label}",
                fontsize=11,
                fontweight="bold",
                color=controller_color,
                va="top",
            )
            info_ax.text(
                0.98,
                0.93,
                "AUCTION" if detail.get("is_bidding_round") else "ACTION WINDOW",
                fontsize=9,
                color="#5C6670",
                va="top",
                ha="right",
            )

            if battery is not None:
                info_ax.text(
                    0.02,
                    0.875,
                    f"Battery {battery} -> {battery_after} / {self.config.battery_capacity}",
                    fontsize=9,
                    fontweight="bold",
                    color="#263238",
                    va="top",
                )
                info_ax.add_patch(
                    mpatches.Rectangle(
                        (0.02, 0.835),
                        0.94,
                        0.025,
                        facecolor="#D5D8DC",
                        edgecolor="#7B878D",
                        linewidth=0.7,
                    )
                )
                info_ax.add_patch(
                    mpatches.Rectangle(
                        (0.02, 0.835),
                        0.94 * battery_fraction,
                        0.025,
                        facecolor=battery_color,
                        edgecolor="none",
                    )
                )

            if self.config.charging_agent_enabled:
                active_label = (
                    "ACTIVE" if detail.get("charging_bid_active") else "STANDBY"
                )
                summary_text = (
                    f"Charging: {active_label}   Window remaining: "
                    f"{detail.get('window_steps_remaining', 0)}\n"
                    f"Priority collected: {cumulative_priority[frame_idx]}   "
                    f"Feeds: {cumulative_reaches[frame_idx]}   "
                    f"Expired: {cumulative_expired[frame_idx]}\n"
                    f"Recharges: {cumulative_recharges[frame_idx]}   "
                    f"Depletions: {cumulative_depletions[frame_idx]}"
                )
            elif self.config.use_target_priorities:
                summary_text = (
                    f"Window remaining: {detail.get('window_steps_remaining', 0)}\n"
                    f"Priority collected: {cumulative_priority[frame_idx]}   "
                    f"Feeds: {cumulative_reaches[frame_idx]}   "
                    f"Expired: {cumulative_expired[frame_idx]}"
                )
            else:
                net_targets = (
                    cumulative_reaches[frame_idx] - cumulative_expired[frame_idx]
                )
                summary_text = (
                    f"Window remaining: {detail.get('window_steps_remaining', 0)}\n"
                    f"Targets reached: {cumulative_reaches[frame_idx]}   "
                    f"Expired: {cumulative_expired[frame_idx]}\n"
                    f"Net targets: {net_targets}"
                )
            info_ax.text(
                0.02,
                0.79,
                summary_text,
                fontsize=9,
                fontfamily="monospace",
                color="#263238",
                va="top",
                linespacing=1.45,
            )
            info_ax.plot([0.02, 0.98], [0.68, 0.68], color="#C6CDD2", lw=1)
            info_ax.text(
                0.02,
                0.655,
                "BIDS / ACTIONS",
                fontsize=10,
                fontweight="bold",
                color="#1F2D3D",
                va="top",
            )
            bid_lines = []
            effective_bids = detail.get("effective_bids", [])
            for bidder_idx in range(self.num_bidders):
                action = actions.get(f"agent_{bidder_idx}", {})
                bid = int(action.get("bid", 0))
                effective = (
                    int(effective_bids[bidder_idx])
                    if bidder_idx < len(effective_bids)
                    else bid
                )
                direction = direction_labels.get(
                    int(action.get("direction", 0)), "?"
                )
                winner_mark = ">" if bidder_idx == winning_agent else " "
                if self.config.charging_agent_enabled and bidder_idx == charging_idx:
                    role = "CHG"
                    priority_label = "  "
                else:
                    role = (
                        f"F{bidder_idx + 1:02d}"
                        if self.config.use_target_priorities
                        else f"A{bidder_idx + 1:02d}"
                    )
                    priority_label = (
                        f"P{int(priorities[bidder_idx])}"
                        if self.config.use_target_priorities
                        else "  "
                    )
                effective_label = (
                    f"/{effective}" if effective != bid else "  "
                )
                bid_lines.append(
                    f"{winner_mark} {role} {priority_label}  "
                    f"bid {bid}{effective_label:>3}  {direction:<5}"
                )
            info_ax.text(
                0.02,
                0.62,
                "\n".join(bid_lines),
                fontsize=8.2,
                fontfamily="monospace",
                color="#263238",
                va="top",
                linespacing=1.25,
            )

            event_parts = []
            reached_priorities = detail.get(
                "target_priorities_just_reached", []
            )
            for feeder_idx, priority in enumerate(reached_priorities):
                if int(priority) > 0:
                    if self.config.use_target_priorities:
                        event_parts.append(f"FED F{feeder_idx + 1} (+{priority})")
                    else:
                        event_parts.append(f"REACHED TARGET {feeder_idx + 1}")
            expired_now = sum(
                bool(value) for value in detail.get("targets_just_expired", [])
            )
            if expired_now:
                event_parts.append(f"{expired_now} TARGET EXPIRED")
            if detail.get("battery_recharged"):
                event_parts.append("BATTERY RECHARGED")
            if detail.get("battery_depleted"):
                event_parts.append("BATTERY DEPLETED")
            if detail.get("charging_navigation_active"):
                event_parts.append(
                    "CHARGER DIRECTION: "
                    + (
                        "OPTIMAL"
                        if detail.get("charging_direction_optimal")
                        else "NON-OPTIMAL"
                    )
                )
            event_text = " | ".join(event_parts) if event_parts else "No event"
            info_ax.plot([0.02, 0.98], [0.16, 0.16], color="#C6CDD2", lw=1)
            info_ax.text(
                0.02,
                0.135,
                "STEP EVENT",
                fontsize=9,
                fontweight="bold",
                color="#1F2D3D",
                va="top",
            )
            info_ax.text(
                0.02,
                0.1,
                event_text,
                fontsize=8.3,
                color="#D64545" if detail.get("battery_depleted") else "#263238",
                va="top",
                wrap=True,
            )
            if self.config.charging_agent_enabled:
                legend_text = (
                    "Target fill = priority | target outline = feeder identity | "
                    "diamond = recharge station"
                )
            elif self.config.use_target_priorities:
                legend_text = "Target fill = priority | target outline = feeder identity"
            else:
                legend_text = "Target outline = agent identity | policy-generated bids and actions"
            info_ax.text(
                0.02,
                0.025,
                legend_text,
                fontsize=7.5,
                color="#6B747C",
                va="bottom",
            )

            fig.canvas.draw()
            return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

        frame_indices = list(range(0, len(render_states), frame_stride))
        if frame_indices[-1] != len(render_states) - 1:
            frame_indices.append(len(render_states) - 1)
        output_path = Path(output_path)
        output_path_mp4 = output_path.with_suffix(".mp4")
        output_path_mp4.parent.mkdir(parents=True, exist_ok=True)
        first_image = draw_frame(frame_indices[0])
        height, width = first_image.shape[:2]
        writer = cv2.VideoWriter(
            str(output_path_mp4),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            plt.close(fig)
            raise RuntimeError(f"could not open MP4 writer for {output_path_mp4}")
        frame_count = 0
        try:
            writer.write(cv2.cvtColor(first_image, cv2.COLOR_RGB2BGR))
            frame_count += 1
            last_image = first_image
            for frame_idx in frame_indices[1:]:
                last_image = draw_frame(frame_idx)
                writer.write(cv2.cvtColor(last_image, cv2.COLOR_RGB2BGR))
                frame_count += 1
            for _ in range(fps):
                image = last_image
                writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                frame_count += 1
        finally:
            writer.release()
            plt.close(fig)
        print(
            f"OK Competition video saved: {output_path_mp4} "
            f"({frame_count} frames)"
        )

    def _legacy_render_states(
        self, episode_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Decode older observation-only rollout data for video compatibility."""
        states = episode_data.get("states", [])
        if not states:
            return []
        cfg = self.config
        denom = float(max(self.grid_size - 1, 1))
        include_reached = not cfg.moving_targets
        target_position_end = 2 + 2 * self.num_agents
        reached_start = target_position_end
        counter_start = reached_start + (self.num_agents if include_reached else 0)
        priority_start = counter_start + self.num_agents
        energy_start = priority_start + (
            self.num_agents if cfg.use_target_priorities else 0
        )
        decoded = []
        for state in states:
            target_positions = np.rint(
                np.asarray(state[2:target_position_end]).reshape(-1, 2) * denom
            ).astype(int)
            priorities = (
                np.rint(
                    np.asarray(
                        state[priority_start : priority_start + self.num_agents]
                    )
                    * 4.0
                ).astype(int)
                if cfg.use_target_priorities
                else np.ones(self.num_agents, dtype=int)
            )
            battery = None
            stations = []
            if self.battery_enabled:
                battery = int(
                    round(float(state[energy_start]) * cfg.battery_capacity)
                )
                station_values = state[
                    energy_start
                    + 1 : energy_start
                    + 1
                    + 2 * self.num_recharge_stations
                ]
                stations = np.rint(
                    np.asarray(station_values).reshape(-1, 2) * denom
                ).astype(int).tolist()
            decoded.append(
                {
                    "agent_position": np.rint(
                        np.asarray(state[:2]) * denom
                    ).astype(int).tolist(),
                    "target_positions": target_positions.tolist(),
                    "target_priorities": priorities.tolist(),
                    "target_counters": [0] * self.num_agents,
                    "targets_reached_count": [0] * self.num_agents,
                    "battery_level": battery,
                    "recharge_station_positions": stations,
                    "window_agent": -1,
                    "window_steps_remaining": 0,
                }
            )
        return decoded

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
        new_priorities = self._sample_target_priorities()
        self.target_priorities = torch.where(mask.unsqueeze(-1), new_priorities, self.target_priorities)
        self.window_agent = torch.where(mask, torch.full_like(self.window_agent, -1), self.window_agent)
        self.window_steps_remaining = torch.where(
            mask, torch.zeros_like(self.window_steps_remaining), self.window_steps_remaining
        )
        self.step_count = torch.where(mask, torch.zeros_like(self.step_count), self.step_count)
        if self.battery_enabled:
            self.battery_level = torch.where(
                mask,
                torch.full_like(self.battery_level, int(cfg.battery_capacity)),
                self.battery_level,
            )
            initial_stations = self.recharge_station_pos.unsqueeze(0).expand(
                self.num_envs, -1, -1
            )
            self.current_recharge_station_pos = torch.where(
                mask.view(-1, 1, 1),
                initial_stations,
                self.current_recharge_station_pos,
            )
            new_station_directions = torch.randint(
                0,
                4,
                self.recharge_station_directions.shape,
                generator=self.gen,
                device=device,
                dtype=torch.int32,
            )
            self.recharge_station_directions = torch.where(
                mask.unsqueeze(-1),
                new_station_directions,
                self.recharge_station_directions,
            )
            self.recharge_station_move_counters = torch.where(
                mask.unsqueeze(-1),
                torch.zeros_like(self.recharge_station_move_counters),
                self.recharge_station_move_counters,
            )

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
        "reached_priority_sum_per_episode": [],
        "reached_priority_sum_per_target_per_episode": [],
        "reached_count_by_priority_per_episode": [],
        "battery_depletions_per_episode": [],
        "battery_recharges_per_episode": [],
        "charging_navigation_steps_per_episode": [],
        "charging_optimal_direction_steps_per_episode": [],
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
        reached_priority_sum = np.zeros(env.num_agents, dtype=np.int32)
        reached_count_by_priority = np.zeros(4, dtype=np.int32)
        expired_targets_count = np.zeros(env.num_agents, dtype=np.int32)
        bid_counts: dict = {}
        control_steps = np.zeros(env.num_bidders, dtype=np.int32)
        battery_depletions = 0
        battery_recharges = 0
        charging_navigation_steps = 0
        charging_optimal_direction_steps = 0

        while not (terminated or truncated):
            episode_states.append(env._get_centralized_observation()[0].copy())

            actions = policy_fn(obs[0])
            if isinstance(actions, np.ndarray):
                actions = torch.tensor(actions, device=env.device)
            actions = actions.to(env.device)
            action_batch = actions.unsqueeze(0)

            env_action = {}
            for agent_idx in range(env.num_bidders):
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
            rewards_dict = {
                f"agent_{i}": float(rewards_cpu[i])
                for i in range(env.num_bidders)
            }
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
            priorities_just_reached = info.get("target_priorities_just_reached")
            if isinstance(priorities_just_reached, torch.Tensor):
                reached_priorities = priorities_just_reached[0].detach().cpu().numpy()
                reached_priority_sum += reached_priorities
                reached_count_by_priority += np.bincount(reached_priorities, minlength=5)[1:5]
            battery_depletions += int(info["battery_depleted"][0].item())
            battery_recharges += int(info["battery_recharged"][0].item())
            if isinstance(info.get("charging_navigation_active"), torch.Tensor):
                charging_navigation_steps += int(
                    info["charging_navigation_active"][0].item()
                )
                charging_optimal_direction_steps += int(
                    info["charging_direction_optimal"][0].item()
                )

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
        eval_stats["reached_priority_sum_per_episode"].append(int(reached_priority_sum.sum()))
        eval_stats["reached_priority_sum_per_target_per_episode"].append(reached_priority_sum.tolist())
        eval_stats["reached_count_by_priority_per_episode"].append(reached_count_by_priority.tolist())
        eval_stats["battery_depletions_per_episode"].append(battery_depletions)
        eval_stats["battery_recharges_per_episode"].append(battery_recharges)
        eval_stats["charging_navigation_steps_per_episode"].append(
            charging_navigation_steps
        )
        eval_stats["charging_optimal_direction_steps_per_episode"].append(
            charging_optimal_direction_steps
        )
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
                  f"PrioritySum={int(reached_priority_sum.sum())}, "
                  f"Expired={episode_expired_count}, MinReached={min_targets_reached}, "
                  f"AvgPerf={float(np.mean(performance)):.2f}")

    if verbose:
        avg_return = np.mean(eval_stats["episode_returns"])
        avg_return_no_bid = np.mean(eval_stats["episode_returns_no_bid"])
        avg_length = np.mean(eval_stats["episode_lengths"])
        avg_targets = np.mean(eval_stats["targets_reached_per_episode"])
        avg_priority_sum = np.mean(eval_stats["reached_priority_sum_per_episode"])
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
        print(f"  Average Reached Priority Sum: {avg_priority_sum:.2f}")
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
    verbose: bool = True,
    capture_episode_count: int = 0,
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
        capture_episode_count: Number of leading batched episodes for which
            exact visualization state, actions, rewards, and events are saved.
    """
    N = env.num_envs
    A = env.num_agents
    B = env.num_bidders
    bid_upper_bound = env.config.bid_upper_bound
    device = env.device

    if N != num_episodes:
        raise ValueError(f"env.num_envs ({N}) must equal num_episodes ({num_episodes})")
    if capture_episode_count < 0:
        raise ValueError("capture_episode_count must be non-negative")
    capture_episode_count = min(capture_episode_count, N)

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
    reached_priority_sum = torch.zeros(N, A, dtype=torch.long, device=device)
    reached_count_by_priority = torch.zeros(N, 4, dtype=torch.long, device=device)
    battery_depletions = torch.zeros(N, dtype=torch.long, device=device)
    battery_recharges = torch.zeros(N, dtype=torch.long, device=device)
    charging_navigation_steps = torch.zeros(
        N, dtype=torch.long, device=device
    )
    charging_optimal_direction_steps = torch.zeros(
        N, dtype=torch.long, device=device
    )
    charging_activation_steps = torch.zeros(
        N, dtype=torch.long, device=device
    )
    charging_active_auction_steps = torch.zeros(
        N, dtype=torch.long, device=device
    )
    charging_active_auction_wins = torch.zeros(
        N, dtype=torch.long, device=device
    )
    charging_active_feeder_max_bid_sum = torch.zeros(
        N, dtype=torch.long, device=device
    )
    charging_active_feeder_tie_or_outbid_steps = torch.zeros(
        N, dtype=torch.long, device=device
    )
    expired_count = torch.zeros(N, A, dtype=torch.long, device=device)
    control_steps = torch.zeros(N, B, dtype=torch.long, device=device)
    depletion_control_steps = torch.zeros(
        N, B, dtype=torch.long, device=device
    )
    bid_count_tensor = torch.zeros(N, bid_upper_bound + 1, dtype=torch.long, device=device)
    charging_bid_count_tensor = torch.zeros(
        N, bid_upper_bound + 1, dtype=torch.long, device=device
    )

    # active[i] is True while episode i has not terminated/truncated
    active = torch.ones(N, dtype=torch.bool, device=device)
    captured_episodes = [
        {
            "render_states": [],
            "states": [],
            "actions": [],
            "rewards": [],
            "step_details": [],
        }
        for _ in range(capture_episode_count)
    ]

    while active.any():
        render_state = (
            env.get_render_state() if capture_episode_count > 0 else None
        )
        active_before_step = active.clone()
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
        priorities_just_reached = info.get("target_priorities_just_reached")
        if isinstance(priorities_just_reached, torch.Tensor):
            active_priorities = priorities_just_reached.long() * active.unsqueeze(1).long()
            reached_priority_sum += active_priorities
            for priority in range(1, 5):
                reached_count_by_priority[:, priority - 1] += (
                    (active_priorities == priority) & active.unsqueeze(1)
                ).sum(dim=1)
        battery_depletions += info["battery_depleted"].long() * active.long()
        battery_recharges += info["battery_recharged"].long() * active.long()
        if isinstance(info.get("charging_navigation_active"), torch.Tensor):
            charging_navigation_steps += (
                info["charging_navigation_active"].long() * active.long()
            )
            charging_optimal_direction_steps += (
                info["charging_direction_optimal"].long() * active.long()
            )
        charging_bid_active = info.get("charging_bid_active")
        if isinstance(charging_bid_active, torch.Tensor):
            charging_activation_steps += (
                charging_bid_active.long() * active.long()
            )

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
            depletion_with_controller = (
                info["battery_depleted"] & valid_winner
            )
            if depletion_with_controller.any():
                depletion_control_steps.scatter_add_(
                    1,
                    winner_idx.unsqueeze(1),
                    depletion_with_controller.long().unsqueeze(1),
                )

        bids = info.get("bids")
        is_bidding_round = info.get("is_bidding_round")
        if isinstance(bids, torch.Tensor) and isinstance(is_bidding_round, torch.Tensor):
            # bids shape: (N, A), is_bidding_round shape: (N,)
            bidding_active = is_bidding_round & active  # (N,)
            if bidding_active.any():
                bids_clamped = bids.clamp(0, bid_upper_bound).long()
                for a_idx in range(B):
                    bid_count_tensor.scatter_add_(
                        1,
                        bids_clamped[:, a_idx].unsqueeze(1),
                        bidding_active.long().unsqueeze(1)
                    )
                if env.charging_agent_idx is not None:
                    charging_bid_count_tensor.scatter_add_(
                        1,
                        bids_clamped[
                            :, env.charging_agent_idx
                        ].unsqueeze(1),
                        bidding_active.long().unsqueeze(1),
                    )
                    charging_active_auction = (
                        bidding_active & charging_bid_active
                    )
                    feeder_max_bid = bids[:, :A].max(dim=1).values
                    charger_bid = bids[:, env.charging_agent_idx]
                    charging_active_auction_steps += (
                        charging_active_auction.long()
                    )
                    charging_active_feeder_max_bid_sum += (
                        feeder_max_bid.long()
                        * charging_active_auction.long()
                    )
                    charging_active_feeder_tie_or_outbid_steps += (
                        (
                            feeder_max_bid >= charger_bid
                        )
                        & charging_active_auction
                    ).long()
                    if isinstance(winning_agent, torch.Tensor):
                        charging_active_auction_wins += (
                            (
                                winning_agent
                                == env.charging_agent_idx
                            )
                            & charging_active_auction
                        ).long()

        if capture_episode_count > 0:
            actions_cpu = actions.detach().cpu().numpy()
            rewards_cpu = rewards.detach().cpu().numpy()
            active_cpu = active_before_step.detach().cpu().numpy()
            info_cpu = {
                key: value.detach().cpu().numpy()
                for key, value in info.items()
                if isinstance(value, torch.Tensor)
            }
            for env_idx in range(capture_episode_count):
                if not active_cpu[env_idx]:
                    continue
                snapshot = {
                    "agent_position": render_state["agent_positions"][
                        env_idx
                    ].tolist(),
                    "target_positions": render_state["target_positions"][
                        env_idx
                    ].tolist(),
                    "target_priorities": render_state["target_priorities"][
                        env_idx
                    ].tolist(),
                    "target_counters": render_state["target_counters"][
                        env_idx
                    ].tolist(),
                    "targets_reached_count": render_state[
                        "targets_reached_count"
                    ][env_idx].tolist(),
                    "battery_level": (
                        int(render_state["battery_levels"][env_idx])
                        if render_state["battery_levels"] is not None
                        else None
                    ),
                    "recharge_station_positions": (
                        render_state["recharge_station_positions"][
                            env_idx
                        ].tolist()
                        if render_state["recharge_station_positions"]
                        is not None
                        else []
                    ),
                    "window_agent": int(
                        render_state["window_agents"][env_idx]
                    ),
                    "window_steps_remaining": int(
                        render_state["window_steps_remaining"][env_idx]
                    ),
                }
                action_dict = {}
                for bidder_idx in range(B):
                    bidder_action = {
                        "direction": int(actions_cpu[env_idx, bidder_idx, 0]),
                        "bid": int(actions_cpu[env_idx, bidder_idx, 1]),
                    }
                    if env.window_bidding:
                        bidder_action["window"] = int(
                            actions_cpu[env_idx, bidder_idx, 2]
                        )
                    action_dict[f"agent_{bidder_idx}"] = bidder_action
                reward_dict = {
                    f"agent_{bidder_idx}": float(
                        rewards_cpu[env_idx, bidder_idx]
                    )
                    for bidder_idx in range(B)
                }
                detail = {
                    "winning_agent": int(
                        info_cpu["winning_agent"][env_idx]
                    ),
                    "bids": info_cpu["bids"][env_idx].tolist(),
                    "effective_bids": info_cpu["effective_bids"][
                        env_idx
                    ].tolist(),
                    "is_bidding_round": bool(
                        info_cpu["is_bidding_round"][env_idx]
                    ),
                    "window_agent": int(
                        info_cpu["window_agent"][env_idx]
                    ),
                    "window_steps_remaining": int(
                        info_cpu["window_steps_remaining"][env_idx]
                    ),
                    "bid_penalty_applied": bool(
                        info_cpu["bid_penalty_applied"][env_idx]
                    ),
                    "battery_level_after": (
                        int(info_cpu["battery_level"][env_idx])
                        if "battery_level" in info_cpu
                        else None
                    ),
                    "battery_recharged": bool(
                        info_cpu["battery_recharged"][env_idx]
                    ),
                    "battery_depleted": bool(
                        info_cpu["battery_depleted"][env_idx]
                    ),
                    "charging_bid_active": bool(
                        info_cpu.get(
                            "charging_bid_active",
                            np.zeros(N, dtype=bool),
                        )[env_idx]
                    ),
                    "charging_navigation_active": bool(
                        info_cpu.get(
                            "charging_navigation_active",
                            np.zeros(N, dtype=bool),
                        )[env_idx]
                    ),
                    "charging_direction_optimal": bool(
                        info_cpu.get(
                            "charging_direction_optimal",
                            np.zeros(N, dtype=bool),
                        )[env_idx]
                    ),
                    "targets_just_reached": info_cpu[
                        "targets_just_reached"
                    ][env_idx].astype(bool).tolist(),
                    "target_priorities_just_reached": info_cpu[
                        "target_priorities_just_reached"
                    ][env_idx].tolist(),
                    "targets_just_expired": info_cpu[
                        "targets_just_expired"
                    ][env_idx].astype(bool).tolist(),
                }
                episode = captured_episodes[env_idx]
                episode["render_states"].append(snapshot)
                episode["actions"].append(action_dict)
                episode["rewards"].append(reward_dict)
                episode["step_details"].append(detail)

        # Mark envs that are done as inactive
        active = active & ~done

    # Move results to CPU for output
    returns_cpu = returns.cpu().tolist()
    returns_no_bid_cpu = returns_no_bid.cpu().tolist()
    lengths_cpu = lengths.cpu().tolist()
    targets_reached_cpu = targets_reached_count.cpu().numpy()
    reached_priority_cpu = reached_priority_sum.cpu().numpy()
    reached_count_by_priority_cpu = reached_count_by_priority.cpu().numpy()
    battery_depletions_cpu = battery_depletions.cpu().tolist()
    battery_recharges_cpu = battery_recharges.cpu().tolist()
    charging_navigation_steps_cpu = charging_navigation_steps.cpu().tolist()
    charging_optimal_direction_steps_cpu = (
        charging_optimal_direction_steps.cpu().tolist()
    )
    charging_activation_steps_cpu = charging_activation_steps.cpu().tolist()
    charging_active_auction_steps_cpu = (
        charging_active_auction_steps.cpu().tolist()
    )
    charging_active_auction_wins_cpu = (
        charging_active_auction_wins.cpu().tolist()
    )
    charging_active_feeder_max_bid_sum_cpu = (
        charging_active_feeder_max_bid_sum.cpu().tolist()
    )
    charging_active_feeder_tie_or_outbid_steps_cpu = (
        charging_active_feeder_tie_or_outbid_steps.cpu().tolist()
    )
    expired_cpu = expired_count.cpu().numpy()
    control_steps_cpu = control_steps.cpu().tolist()
    depletion_control_steps_cpu = depletion_control_steps.cpu().tolist()
    bid_count_np = bid_count_tensor.cpu().numpy()
    charging_bid_count_np = charging_bid_count_tensor.cpu().numpy()

    eval_stats = {
        "episode_returns": [],
        "episode_returns_no_bid": [],
        "episode_lengths": [],
        "targets_reached_per_episode": [],
        "expired_targets_per_episode": [],
        "min_targets_reached_per_episode": [],
        "targets_reached_count_per_episode": [],
        "reached_priority_sum_per_episode": [],
        "reached_priority_sum_per_target_per_episode": [],
        "reached_count_by_priority_per_episode": [],
        "battery_depletions_per_episode": [],
        "battery_recharges_per_episode": [],
        "charging_navigation_steps_per_episode": [],
        "charging_optimal_direction_steps_per_episode": [],
        "charging_activation_steps_per_episode": [],
        "charging_active_auction_steps_per_episode": [],
        "charging_active_auction_wins_per_episode": [],
        "charging_active_feeder_max_bid_sum_per_episode": [],
        "charging_active_feeder_tie_or_outbid_steps_per_episode": [],
        "episode_data_list": captured_episodes,
        "bid_counts_per_episode": [],
        "charging_bid_counts_per_episode": [],
        "control_steps_per_agent_per_episode": [],
        "battery_depletions_per_agent_per_episode": [],
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
        priority_sum = reached_priority_cpu[i]
        priority_counts = reached_count_by_priority_cpu[i]
        ec = expired_cpu[i]           # (A,) numpy
        performance = trc - ec

        targets_reached = int((trc > 0).sum())
        min_targets_reached = int(trc.min())
        episode_expired_count = int(ec.sum())

        bid_counts_dict = {b: int(bid_count_np[i, b]) for b in range(bid_upper_bound + 1)}
        charging_bid_counts_dict = {
            b: int(charging_bid_count_np[i, b])
            for b in range(bid_upper_bound + 1)
        }

        eval_stats["episode_returns"].append(returns_cpu[i])
        eval_stats["episode_returns_no_bid"].append(returns_no_bid_cpu[i])
        eval_stats["episode_lengths"].append(int(lengths_cpu[i]))
        eval_stats["targets_reached_per_episode"].append(targets_reached)
        eval_stats["expired_targets_per_episode"].append(episode_expired_count)
        eval_stats["min_targets_reached_per_episode"].append(min_targets_reached)
        eval_stats["targets_reached_count_per_episode"].append(trc.tolist())
        eval_stats["reached_priority_sum_per_episode"].append(int(priority_sum.sum()))
        eval_stats["reached_priority_sum_per_target_per_episode"].append(priority_sum.tolist())
        eval_stats["reached_count_by_priority_per_episode"].append(priority_counts.tolist())
        eval_stats["battery_depletions_per_episode"].append(battery_depletions_cpu[i])
        eval_stats["battery_recharges_per_episode"].append(battery_recharges_cpu[i])
        eval_stats["charging_navigation_steps_per_episode"].append(
            charging_navigation_steps_cpu[i]
        )
        eval_stats["charging_optimal_direction_steps_per_episode"].append(
            charging_optimal_direction_steps_cpu[i]
        )
        eval_stats["charging_activation_steps_per_episode"].append(
            charging_activation_steps_cpu[i]
        )
        eval_stats["charging_active_auction_steps_per_episode"].append(
            charging_active_auction_steps_cpu[i]
        )
        eval_stats["charging_active_auction_wins_per_episode"].append(
            charging_active_auction_wins_cpu[i]
        )
        eval_stats[
            "charging_active_feeder_max_bid_sum_per_episode"
        ].append(charging_active_feeder_max_bid_sum_cpu[i])
        eval_stats[
            "charging_active_feeder_tie_or_outbid_steps_per_episode"
        ].append(charging_active_feeder_tie_or_outbid_steps_cpu[i])
        eval_stats["bid_counts_per_episode"].append(bid_counts_dict)
        eval_stats["charging_bid_counts_per_episode"].append(
            charging_bid_counts_dict
        )
        eval_stats["control_steps_per_agent_per_episode"].append(control_steps_cpu[i])
        eval_stats["battery_depletions_per_agent_per_episode"].append(
            depletion_control_steps_cpu[i]
        )
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
                  f"PrioritySum={eval_stats['reached_priority_sum_per_episode'][i]}, "
                  f"Expired={eval_stats['expired_targets_per_episode'][i]}, "
                  f"MinReached={eval_stats['min_targets_reached_per_episode'][i]}, "
                  f"AvgPerf={eval_stats['avg_performance_per_episode'][i]:.2f}")

        avg_return = np.mean(eval_stats["episode_returns"])
        avg_return_no_bid = np.mean(eval_stats["episode_returns_no_bid"])
        avg_length = np.mean(eval_stats["episode_lengths"])
        avg_targets = np.mean(eval_stats["targets_reached_per_episode"])
        avg_priority_sum = np.mean(eval_stats["reached_priority_sum_per_episode"])
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
        print(f"  Average Reached Priority Sum: {avg_priority_sum:.2f}")
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
        "reached_priority_sum_per_episode": [],
        "reached_priority_sum_per_target_per_episode": [],
        "reached_count_by_priority_per_episode": [],
        "battery_depletions_per_episode": [],
        "battery_recharges_per_episode": [],
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
        reached_priority_sum = np.zeros(env.num_agents, dtype=np.int32)
        reached_count_by_priority = np.zeros(4, dtype=np.int32)
        expired_targets_count = np.zeros(env.num_agents, dtype=np.int32)
        battery_depletions = 0
        battery_recharges = 0

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
            priorities_just_reached = info.get("target_priorities_just_reached")
            if isinstance(priorities_just_reached, torch.Tensor):
                reached_priorities = priorities_just_reached[0].detach().cpu().numpy()
                reached_priority_sum += reached_priorities
                reached_count_by_priority += np.bincount(reached_priorities, minlength=5)[1:5]
            battery_depletions += int(info["battery_depleted"][0].item())
            battery_recharges += int(info["battery_recharged"][0].item())

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
        eval_stats["reached_priority_sum_per_episode"].append(int(reached_priority_sum.sum()))
        eval_stats["reached_priority_sum_per_target_per_episode"].append(reached_priority_sum.tolist())
        eval_stats["reached_count_by_priority_per_episode"].append(reached_count_by_priority.tolist())
        eval_stats["battery_depletions_per_episode"].append(battery_depletions)
        eval_stats["battery_recharges_per_episode"].append(battery_recharges)
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
                  f"PrioritySum={int(reached_priority_sum.sum())}, "
                  f"Expired={episode_expired_count}, MinReached={min_targets_reached}, "
                  f"AvgPerf={float(np.mean(performance)):.2f}")

    if verbose:
        avg_return = np.mean(eval_stats["episode_returns"])
        avg_length = np.mean(eval_stats["episode_lengths"])
        avg_targets = np.mean(eval_stats["targets_reached_per_episode"])
        avg_priority_sum = np.mean(eval_stats["reached_priority_sum_per_episode"])
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
        print(f"  Average Reached Priority Sum: {avg_priority_sum:.2f}")
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
    reached_priority_sum = torch.zeros(N, A, dtype=torch.long, device=device)
    reached_count_by_priority = torch.zeros(N, 4, dtype=torch.long, device=device)
    battery_depletions = torch.zeros(N, dtype=torch.long, device=device)
    battery_recharges = torch.zeros(N, dtype=torch.long, device=device)
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
        priorities_just_reached = info.get("target_priorities_just_reached")
        if isinstance(priorities_just_reached, torch.Tensor):
            active_priorities = priorities_just_reached.long() * active.unsqueeze(1).long()
            reached_priority_sum += active_priorities
            for priority in range(1, 5):
                reached_count_by_priority[:, priority - 1] += (
                    (active_priorities == priority) & active.unsqueeze(1)
                ).sum(dim=1)
        battery_depletions += info["battery_depleted"].long() * active.long()
        battery_recharges += info["battery_recharged"].long() * active.long()

        active = active & ~done

    returns_cpu = returns.cpu().tolist()
    lengths_cpu = lengths.cpu().tolist()
    targets_reached_cpu = targets_reached_count.cpu().numpy()
    reached_priority_cpu = reached_priority_sum.cpu().numpy()
    reached_count_by_priority_cpu = reached_count_by_priority.cpu().numpy()
    battery_depletions_cpu = battery_depletions.cpu().tolist()
    battery_recharges_cpu = battery_recharges.cpu().tolist()
    expired_cpu = expired_count.cpu().numpy()

    eval_stats = {
        "episode_returns": [],
        "episode_lengths": [],
        "targets_reached_per_episode": [],
        "expired_targets_per_episode": [],
        "min_targets_reached_per_episode": [],
        "targets_reached_count_per_episode": [],
        "reached_priority_sum_per_episode": [],
        "reached_priority_sum_per_target_per_episode": [],
        "reached_count_by_priority_per_episode": [],
        "battery_depletions_per_episode": [],
        "battery_recharges_per_episode": [],
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
        priority_sum = reached_priority_cpu[i]
        priority_counts = reached_count_by_priority_cpu[i]
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
        eval_stats["reached_priority_sum_per_episode"].append(int(priority_sum.sum()))
        eval_stats["reached_priority_sum_per_target_per_episode"].append(priority_sum.tolist())
        eval_stats["reached_count_by_priority_per_episode"].append(priority_counts.tolist())
        eval_stats["battery_depletions_per_episode"].append(battery_depletions_cpu[i])
        eval_stats["battery_recharges_per_episode"].append(battery_recharges_cpu[i])
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
                  f"PrioritySum={eval_stats['reached_priority_sum_per_episode'][i]}, "
                  f"Expired={eval_stats['expired_targets_per_episode'][i]}, "
                  f"MinReached={eval_stats['min_targets_reached_per_episode'][i]}, "
                  f"AvgPerf={eval_stats['avg_performance_per_episode'][i]:.2f}")

        avg_return = np.mean(eval_stats["episode_returns"])
        avg_length = np.mean(eval_stats["episode_lengths"])
        avg_targets = np.mean(eval_stats["targets_reached_per_episode"])
        avg_priority_sum = np.mean(eval_stats["reached_priority_sum_per_episode"])
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
        print(f"  Average Reached Priority Sum: {avg_priority_sum:.2f}")
        print(f"  Average Expired: {avg_expired:.2f} +/- {np.std(eval_stats['expired_targets_per_episode']):.2f}")
        print(f"  Average Min Reached: {avg_min_reached:.2f} +/- {np.std(eval_stats['min_targets_reached_per_episode']):.2f}")
        print(f"  Avg Performance (reaches-exp): {avg_avg_perf:.2f}")
        print(f"  Avg Min Performance: {avg_min_perf:.2f}")
        print(f"  Success Rate: {success_rate*100:.1f}%\n")

    return eval_stats
