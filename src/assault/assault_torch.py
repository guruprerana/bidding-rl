"""
OCAtari Assault environments with object-state tensor observations.

Supports single-agent and bidding multi-agent control over the player, where
each agent is associated with a specific Enemy slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ocatari.core import OCAtari


@dataclass
class AssaultConfig:
    num_agents: int
    max_enemies: int = 3
    bid_upper_bound: int = 10
    bid_penalty: float = 0.1
    action_window: int = 1
    window_bidding: bool = False
    window_penalty: float = 0.0
    enemy_destroy_reward: float = 1.0
    hit_penalty: float = 1.0
    life_loss_penalty: float = 10.0
    health_loss_penalty: float = 0.1
    max_steps: int = 10000
    hud: bool = True
    single_agent_mode: bool = False
    allow_variable_enemies: bool = True


class AssaultEnv:
    """
    Batched Assault environment with object-centric observations.

    - single_agent_mode=True: returns obs shape (num_envs, obs_dim)
    - single_agent_mode=False: returns obs shape (num_envs, num_agents, obs_dim)
    """

    def __init__(
        self,
        config: AssaultConfig,
        num_envs: int,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ) -> None:
        if config.num_agents > config.max_enemies:
            raise ValueError("num_agents cannot exceed max_enemies")

        self.config = config
        self.num_envs = num_envs
        self.device = device or torch.device("cpu")
        self.gen = torch.Generator(device="cpu")
        if seed is not None:
            self.gen.manual_seed(seed)

        self.envs = [
            OCAtari("ALE/Assault-v5", obs_mode="obj", hud=False, render_mode=None)
            for _ in range(num_envs)
        ]
        self.action_space_n = int(self.envs[0].action_space.n)
        self.screen_width, self.screen_height = 160, 210
        self.max_health_width = 64.0
        if seed is not None:
            for idx, env in enumerate(self.envs):
                env.reset(seed=seed + idx)

        self._slot_indices = self._build_slot_indices(self.envs[0])
        self.max_lives = 3

        self.step_count = torch.zeros((num_envs,), dtype=torch.int32)
        self.window_agent = torch.full((num_envs,), -1, dtype=torch.int32)
        self.window_steps_remaining = torch.zeros((num_envs,), dtype=torch.int32)
        self.prev_enemy_visible = torch.zeros((num_envs, config.max_enemies), dtype=torch.int32)
        self.prev_lives = torch.zeros((num_envs,), dtype=torch.int32)
        self.prev_health_width = torch.zeros((num_envs,), dtype=torch.float32)
        self.prev_health_red = torch.zeros((num_envs,), dtype=torch.int32)

        self._global_obs_dim = 18
        if config.single_agent_mode:
            self.obs_dim = self._global_obs_dim + config.max_enemies * 3
            self.obs_shape = (num_envs, self.obs_dim)
            self.per_agent_obs_dim = None
        else:
            self.per_agent_obs_dim = self._global_obs_dim + config.max_enemies * 3
            self.obs_dim = None
            self.obs_shape = (num_envs, config.num_agents, self.per_agent_obs_dim)

    def reset(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        if seed is not None:
            self.gen.manual_seed(seed)
        obs_list = []
        for idx, env in enumerate(self.envs):
            env.reset(seed=None if seed is None else seed + idx)
            state = self._extract_state(env)
            obs_list.append(self._build_obs(state, env_idx=idx))
            self.prev_enemy_visible[idx] = state["enemy_visible"]
            self.prev_lives[idx] = state["lives_count"]
            self.prev_health_width[idx] = state["health_width"]
            self.prev_health_red[idx] = state["health_red"]
            self.step_count[idx] = 0
            self.window_agent[idx] = -1
            self.window_steps_remaining[idx] = 0

        obs = torch.stack(obs_list, dim=0).to(self.device)
        info: Dict = {}
        return obs, info

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        cfg = self.config
        self.step_count += 1

        if cfg.single_agent_mode:
            action_dir = action.view(self.num_envs).to(torch.int64).cpu().numpy()
            bids = None
            winning_agent = torch.zeros((self.num_envs,), dtype=torch.int32)
            apply_bid_penalty = torch.zeros((self.num_envs,), dtype=torch.bool)
            current_window_length = torch.zeros((self.num_envs,), dtype=torch.int32)
        else:
            action = action.to(torch.int64)
            action_dir = action[..., 0].cpu().numpy()
            bids = action[..., 1].cpu()
            action_window = action[..., 2].cpu() if cfg.window_bidding else None

            in_window = self.window_steps_remaining > 0
            if torch.any(in_window):
                self.window_steps_remaining = torch.where(
                    in_window, self.window_steps_remaining - 1, self.window_steps_remaining
                )

            winning_agent = self._select_winners(bids, in_window)
            apply_bid_penalty = (winning_agent >= 0) & (~in_window)
            current_window_length = torch.zeros((self.num_envs,), dtype=torch.int32)
            if cfg.window_bidding:
                for env_idx in range(self.num_envs):
                    if apply_bid_penalty[env_idx]:
                        winner = int(winning_agent[env_idx])
                        chosen_window = int(action_window[env_idx, winner].item()) + 1
                        current_window_length[env_idx] = chosen_window
                        self.window_steps_remaining[env_idx] = chosen_window - 1
                        self.window_agent[env_idx] = winner
            else:
                for env_idx in range(self.num_envs):
                    if apply_bid_penalty[env_idx]:
                        current_window_length[env_idx] = cfg.action_window
                        self.window_steps_remaining[env_idx] = cfg.action_window - 1
                        self.window_agent[env_idx] = int(winning_agent[env_idx])

        obs_list = []
        rewards_list = []
        terminated_list = []
        truncated_list = []
        enemy_destroyed_list = []
        score_list = []
        state_list = []

        for env_idx, env in enumerate(self.envs):
            if cfg.single_agent_mode:
                chosen_action = int(action_dir[env_idx])
            else:
                winner = int(winning_agent[env_idx].item())
                if winner < 0:
                    chosen_action = 0
                else:
                    chosen_action = int(action_dir[env_idx, winner])

            _, _, terminated, truncated, info = env.step(chosen_action)
            if terminated or truncated or (cfg.max_steps and self.step_count[env_idx] >= cfg.max_steps):
                terminated = bool(terminated)
                truncated = bool(truncated or (cfg.max_steps and self.step_count[env_idx] >= cfg.max_steps))

            state = self._extract_state(env)
            score_list.append(float(info.get("score", 0.0)))

            enemy_visible = state["enemy_visible"]
            destroyed = (self.prev_enemy_visible[env_idx] == 1) & (enemy_visible == 0)
            enemy_destroyed_list.append(destroyed.clone())

            penalty = 0.0
            life_loss = max(int(self.prev_lives[env_idx]) - int(state["lives_count"]), 0)
            health_loss = max(float(self.prev_health_width[env_idx]) - float(state["health_width"]), 0.0)
            hit_event = int(state["health_red"]) == 1 and int(self.prev_health_red[env_idx]) == 0

            if life_loss > 0:
                penalty -= cfg.life_loss_penalty * life_loss
            if health_loss > 0:
                penalty -= cfg.health_loss_penalty * health_loss
            if hit_event:
                penalty -= cfg.hit_penalty

            if cfg.single_agent_mode:
                reward = cfg.enemy_destroy_reward * destroyed.to(torch.float32).sum().item() + penalty
                rewards = torch.tensor(reward, dtype=torch.float32)
            else:
                rewards = torch.zeros((cfg.num_agents,), dtype=torch.float32)
                for agent_id in range(cfg.num_agents):
                    if destroyed[agent_id] == 1:
                        rewards[agent_id] += cfg.enemy_destroy_reward
                rewards += penalty
                if bids is not None and apply_bid_penalty[env_idx]:
                    rewards -= cfg.bid_penalty * bids[env_idx].to(torch.float32)
                    if cfg.window_bidding and cfg.window_penalty > 0 and current_window_length[env_idx] > 0:
                        winner = int(winning_agent[env_idx].item())
                        rewards[winner] -= cfg.window_penalty * float(current_window_length[env_idx].item())

            if terminated or truncated:
                env.reset()
                state = self._extract_state(env)
                self.step_count[env_idx] = 0

            obs_list.append(self._build_obs(state, env_idx=env_idx))
            rewards_list.append(rewards)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            state_list.append(state)

            self.prev_enemy_visible[env_idx] = state["enemy_visible"]
            self.prev_lives[env_idx] = state["lives_count"]
            self.prev_health_width[env_idx] = state["health_width"]
            self.prev_health_red[env_idx] = state["health_red"]

        obs = torch.stack(obs_list, dim=0).to(self.device)
        terminated_t = torch.tensor(terminated_list, dtype=torch.bool, device=self.device)
        truncated_t = torch.tensor(truncated_list, dtype=torch.bool, device=self.device)

        if cfg.single_agent_mode:
            rewards_t = torch.stack(rewards_list, dim=0).to(self.device)
        else:
            rewards_t = torch.stack(rewards_list, dim=0).to(self.device)

        info = {
            "winning_agent": winning_agent.to(self.device) if not cfg.single_agent_mode else None,
            "enemy_destroyed": torch.stack(enemy_destroyed_list, dim=0).to(self.device),
            "window_agent": self.window_agent.to(self.device),
            "window_steps_remaining": self.window_steps_remaining.to(self.device),
            "bid_penalty_applied": apply_bid_penalty.to(self.device),
            "score": torch.tensor(score_list, dtype=torch.float32, device=self.device),
            "health_width": self.prev_health_width.to(self.device),
            "lives_count": self.prev_lives.to(self.device),
            "player_xy_raw": torch.stack([s["player_xy_raw"] for s in state_list], dim=0).to(self.device),
            "mothership_raw": torch.stack([s["mothership_raw"] for s in state_list], dim=0).to(self.device),
            "enemy_missile_raw": torch.stack([s["enemy_missile_raw"] for s in state_list], dim=0).to(self.device),
            "player_missile_v_raw": torch.stack([s["player_missile_v_raw"] for s in state_list], dim=0).to(self.device),
            "player_missile_h_raw": torch.stack([s["player_missile_h_raw"] for s in state_list], dim=0).to(self.device),
            "enemy_raw": torch.stack([s["enemy_raw"] for s in state_list], dim=0).to(self.device),
        }
        return obs, rewards_t, terminated_t, truncated_t, info

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def _select_winners(self, bids: torch.Tensor, in_window: torch.Tensor) -> torch.Tensor:
        max_bid = bids.max(dim=1).values
        winners = torch.full((self.num_envs,), -1, dtype=torch.int32)
        for env_idx in range(self.num_envs):
            if in_window[env_idx]:
                winners[env_idx] = self.window_agent[env_idx]
                continue
            if max_bid[env_idx] <= 0:
                continue
            candidates = torch.where(bids[env_idx] == max_bid[env_idx])[0]
            if candidates.numel() == 1:
                winners[env_idx] = candidates[0].to(torch.int32)
            else:
                choice = torch.randint(0, candidates.numel(), (1,), generator=self.gen).item()
                winners[env_idx] = candidates[choice].to(torch.int32)
        return winners

    def _build_slot_indices(self, env: OCAtari) -> Dict[str, List[int]]:
        slot_types = [slot.__class__.__name__ for slot in env._slots]
        indices: Dict[str, List[int]] = {}
        for idx, name in enumerate(slot_types):
            indices.setdefault(name, []).append(idx)
        return indices

    def _extract_state(self, env: OCAtari) -> Dict[str, torch.Tensor]:
        slots = env.objects
        idx = self._slot_indices

        def slot_obj(indices: List[int], default=None):
            if not indices:
                return default
            obj = slots[indices[0]]
            return obj

        def slot_list(indices: List[int]) -> List:
            return [slots[i] for i in indices]

        player = slot_obj(idx.get("Player", []))
        mothership = slot_obj(idx.get("MotherShip", []))
        enemy_objs = slot_list(idx.get("Enemy", []))
        enemy_missile = slot_obj(idx.get("EnemyMissile", []))
        missile_v = slot_obj(idx.get("PlayerMissileVertical", []))
        missile_h = slot_obj(idx.get("PlayerMissileHorizontal", []))

        ram_state = self._get_ram(env)
        lives = slot_list(idx.get("Lives", []))
        health = slot_obj(idx.get("Health", []))

        def to_xy_vis(obj) -> Tuple[float, float, int]:
            if obj is None or obj.__class__.__name__ == "NoObject":
                return 0.0, 0.0, 0
            x, y = obj.xy
            norm_x = float(x) / float(self.screen_width)
            norm_y = float(y) / float(self.screen_height)
            return norm_x, norm_y, 1

        def to_xy_raw(obj) -> Tuple[float, float]:
            if obj is None or obj.__class__.__name__ == "NoObject":
                return 0.0, 0.0
            x, y = obj.xy
            return float(x), float(y)

        player_x, player_y, _ = to_xy_vis(player)
        mothership_x, mothership_y, mothership_vis = to_xy_vis(mothership)
        enemy_missile_x, enemy_missile_y, enemy_missile_vis = to_xy_vis(enemy_missile)
        missile_v_x, missile_v_y, missile_v_vis = to_xy_vis(missile_v)
        missile_h_x, missile_h_y, missile_h_vis = to_xy_vis(missile_h)

        player_raw = to_xy_raw(player)
        mothership_raw = to_xy_raw(mothership)
        enemy_missile_raw = to_xy_raw(enemy_missile)
        missile_v_raw = to_xy_raw(missile_v)
        missile_h_raw = to_xy_raw(missile_h)

        enemy_features = []
        enemy_raw = []
        enemy_visible = []
        for enemy in enemy_objs[: self.config.max_enemies]:
            ex, ey, ev = to_xy_vis(enemy)
            enemy_features.append((ex, ey, ev))
            enemy_visible.append(ev)
            enemy_raw.append(to_xy_raw(enemy))
        while len(enemy_features) < self.config.max_enemies:
            enemy_features.append((0.0, 0.0, 0))
            enemy_visible.append(0)
            enemy_raw.append((0.0, 0.0))

        if ram_state is not None:
            lives_count = max(int(ram_state[101]) - 1, 0)
            health_width = self._health_width_from_ram(ram_state)
            health_red = 1 if int(ram_state[21]) == 70 else 0
        else:
            lives_count = sum(1 for obj in lives if obj.__class__.__name__ == "Lives")
            health_width = float(health.wh[0]) if health is not None and health.__class__.__name__ != "NoObject" else 0.0
            health_red = 1 if health is not None and tuple(getattr(health, "rgb", (0, 0, 0))) == (200, 72, 72) else 0
        lives_norm = float(lives_count) / float(self.max_lives)
        health_norm = float(health_width) / float(self.max_health_width)

        return {
            "player_xy": torch.tensor([player_x, player_y], dtype=torch.float32),
            "player_xy_raw": torch.tensor(player_raw, dtype=torch.float32),
            "mothership": torch.tensor([mothership_x, mothership_y, mothership_vis], dtype=torch.float32),
            "mothership_raw": torch.tensor(mothership_raw, dtype=torch.float32),
            "enemy_missile": torch.tensor([enemy_missile_x, enemy_missile_y, enemy_missile_vis], dtype=torch.float32),
            "enemy_missile_raw": torch.tensor(enemy_missile_raw, dtype=torch.float32),
            "player_missile_v": torch.tensor([missile_v_x, missile_v_y, missile_v_vis], dtype=torch.float32),
            "player_missile_v_raw": torch.tensor(missile_v_raw, dtype=torch.float32),
            "player_missile_h": torch.tensor([missile_h_x, missile_h_y, missile_h_vis], dtype=torch.float32),
            "player_missile_h_raw": torch.tensor(missile_h_raw, dtype=torch.float32),
            "enemy_features": torch.tensor(enemy_features, dtype=torch.float32),
            "enemy_raw": torch.tensor(enemy_raw, dtype=torch.float32),
            "enemy_visible": torch.tensor(enemy_visible, dtype=torch.int32),
            "lives_count": torch.tensor(lives_count, dtype=torch.int32),
            "lives_norm": torch.tensor(lives_norm, dtype=torch.float32),
            "health_width": torch.tensor(health_width, dtype=torch.float32),
            "health_norm": torch.tensor(health_norm, dtype=torch.float32),
            "health_red": torch.tensor(health_red, dtype=torch.int32),
        }

    def _build_obs(self, state: Dict[str, torch.Tensor], env_idx: int) -> torch.Tensor:
        cfg = self.config
        global_features = torch.cat(
            [
                state["player_xy"],
                state["mothership"],
                state["enemy_missile"],
                state["player_missile_v"],
                state["player_missile_h"],
                torch.tensor(
                    [
                        float(self.window_steps_remaining[env_idx].item()) / float(max(cfg.action_window, 1)),
                    ],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [
                        state["health_norm"].item(),
                        state["lives_norm"].item(),
                        state["health_red"].item(),
                    ],
                    dtype=torch.float32,
                ),
            ],
            dim=0,
        )

        if cfg.single_agent_mode:
            enemies = state["enemy_features"].reshape(-1)
            return torch.cat([global_features, enemies], dim=0)

        per_agent_obs = []
        for agent_id in range(cfg.num_agents):
            target_enemy = state["enemy_features"][agent_id]
            other_enemies = torch.cat(
                [
                    state["enemy_features"][:agent_id],
                    state["enemy_features"][agent_id + 1:],
                ],
                dim=0,
            ).reshape(-1)
            per_agent_obs.append(torch.cat([global_features, target_enemy, other_enemies], dim=0))
        return torch.stack(per_agent_obs, dim=0)

    def _get_ram(self, env: OCAtari) -> Optional[np.ndarray]:
        ale = getattr(env, "_ale", None)
        if ale is None:
            return None
        try:
            return ale.getRAM()
        except Exception:
            return None

    def _health_width_from_ram(self, ram_state: np.ndarray) -> float:
        if ram_state is None:
            return 0.0
        a = int(ram_state[28])
        b = int(ram_state[29])
        if a == 192 and b == 0:
            return 8.0
        if a == 224 and b == 0:
            return 12.0
        if a == 240 and b == 0:
            return 16.0
        if a == 248 and b == 0:
            return 20.0
        if a == 252 and b == 0:
            return 24.0
        if a == 254 and b == 0:
            return 28.0
        if a == 255 and b == 0:
            return 32.0
        if a == 255 and b == 1:
            return 36.0
        if a == 255 and b == 3:
            return 40.0
        if a == 255 and b == 7:
            return 44.0
        if a == 255 and b == 15:
            return 48.0
        if a == 255 and b == 31:
            return 52.0
        if a == 255 and b == 63:
            return 56.0
        if a == 255 and b == 127:
            return 60.0
        if a == 255 and b == 255:
            return 64.0
        return 0.0
