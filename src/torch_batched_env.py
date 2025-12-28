"""
Torch-batched GPU environment skeleton for BiddingGridworld.

This is a starting point for a full CUDA-native env. It mirrors the
state/step layout of BiddingGridworld but keeps all tensors on GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch


@dataclass
class TorchBatchedConfig:
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


class TorchBatchedBiddingGridworld:
    """
    GPU-native batched env for BiddingGridworld.

    Intended to be used directly in PPO trainers to avoid CPU env stepping.
    All state is stored in CUDA tensors and updated with torch ops.
    """

    def __init__(
        self,
        config: TorchBatchedConfig,
        num_envs: int,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config
        self.num_envs = num_envs
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

        # Observation shape metadata
        if self.config.single_agent_mode:
            self.obs_dim = 3 + 5 * self.config.num_agents
            self.obs_shape = (self.num_envs, self.obs_dim)
            self.per_agent_obs_dim = None
        else:
            if self.config.visible_targets is None:
                self.per_agent_obs_dim = 3 + 4 * self.config.num_agents
            else:
                self.per_agent_obs_dim = 7 + 3 * self.config.visible_targets
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
            has_bid = max_bid > 0
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
                dist_improve = (self.previous_distances - current_distances).to(torch.float32)
                rewards = rewards + cfg.distance_reward_scale * (dist_improve * (self.targets_reached == 0).to(torch.float32)).sum(dim=1)

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
            if torch.any(apply_bid_penalty) and bids is not None:
                rewards = rewards - apply_bid_penalty.unsqueeze(1).to(torch.float32) * cfg.bid_penalty * bids.to(torch.float32)
                if cfg.window_bidding and cfg.window_penalty > 0:
                    penalty = cfg.window_penalty * current_window_length.to(torch.float32)
                    valid_win = winning_agent >= 0
                    if torch.any(valid_win):
                        idx = winning_agent.clamp(min=0).to(torch.int64).view(-1, 1)
                        rewards = rewards.scatter_add(1, idx, (-penalty * valid_win.to(torch.float32)).view(-1, 1))

            if cfg.distance_reward_scale > 0:
                dist_improve = (self.previous_distances - current_distances).to(torch.float32)
                rewards = rewards + cfg.distance_reward_scale * dist_improve * (self.targets_reached == 0).to(torch.float32)

            rewards = rewards + cfg.target_reward * targets_just_reached.to(torch.float32)
            if cfg.target_expiry_penalty > 0:
                rewards = rewards - cfg.target_expiry_penalty * targets_expired.to(torch.float32)

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
        }
        return obs, rewards, terminated, truncated, info

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
            min_count = self.targets_reached_count.min(dim=1).values
            count_denom = float(max(cfg.num_agents, 1))
            relative = (self.targets_reached_count - min_count.unsqueeze(1)).to(torch.float32)
            relative = torch.clamp(relative / count_denom, 0.0, 1.0)
            return torch.cat([base_obs, relative], dim=-1)

        if cfg.visible_targets is None:
            reordered_pos = target_pos[:, self._reorder_idx, :].reshape(self.num_envs, cfg.num_agents, -1)
            reordered_reached = targets_reached[:, self._reorder_idx]
            reordered_counters = target_counters[:, self._reorder_idx]
            return torch.cat(
                [agent_pos.unsqueeze(1).expand(-1, cfg.num_agents, -1),
                 reordered_pos,
                 reordered_reached,
                 reordered_counters,
                 window_steps.unsqueeze(1).expand(-1, cfg.num_agents, -1)],
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
        targets_reached_exp = targets_reached.unsqueeze(1).expand(-1, cfg.num_agents, -1)
        vis_reached = targets_reached_exp.gather(2, idx)

        own_pos = target_pos
        own_reached = targets_reached.unsqueeze(-1)
        own_counter = target_counters.unsqueeze(-1)

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

    def close(self) -> None:
        """No-op close for API compatibility."""
        return None
