"""
OCAtari Air Raid environments with object-state tensor observations.

Supports single-agent and bidding multi-agent control over the player, where
each agent is associated with one of the three enemy lanes in Air Raid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from ocatari.core import OCAtari


@dataclass
class AirRaidConfig:
    num_agents: int
    max_enemies: int = 3
    bid_upper_bound: int = 10
    bid_penalty: float = 0.1
    action_window: int = 1
    window_bidding: bool = False
    window_penalty: float = 0.0
    enemy_destroy_reward: float = 1.0
    building_hit_penalty: float = 1.0
    life_loss_penalty: float = 10.0
    raw_score_scale: float = 0.0
    enemy_missile_danger_penalty: float = 0.0
    enemy_missile_danger_y_threshold: float = 110.0
    enemy_missile_danger_x_radius: float = 18.0
    enemy_missile_danger_y_radius: float = 80.0
    enemy_missile_near_hit_penalty: float = 0.0
    enemy_missile_near_hit_y_margin: float = 35.0
    enemy_missile_near_hit_x_radius: float = 25.0
    enemy_disappear_penalty: float = 0.0
    max_steps: int = 10000
    hud: bool = True
    single_agent_mode: bool = False
    allow_sideward_fire: bool = True
    bidding_mechanism: str = "all_pay"
    only_own_enemy: bool = False
    obs_stack: int = 1
    building_penalty_mode: str = "lane"
    include_agent_id: bool = False


class AirRaidEnv:
    """
    Batched Air Raid environment with object-centric observations.

    - single_agent_mode=True: returns obs shape (num_envs, obs_dim)
    - single_agent_mode=False: returns obs shape (num_envs, num_agents, obs_dim)
    """

    def __init__(
        self,
        config: AirRaidConfig,
        num_envs: int,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        render_oc_overlay: bool = False,
    ) -> None:
        if config.num_agents > config.max_enemies:
            raise ValueError("num_agents cannot exceed max_enemies")

        self.config = config
        self.num_envs = num_envs
        self.device = device or torch.device("cpu")
        self.render_mode = render_mode
        self.render_oc_overlay = render_oc_overlay
        self.gen = torch.Generator(device="cpu")
        if seed is not None:
            self.gen.manual_seed(seed)

        self.envs = [
            OCAtari(
                "AirRaidNoFrameskip-v4",
                mode="ram",
                obs_mode="obj",
                hud=True,
                render_mode=render_mode,
                render_oc_overlay=render_oc_overlay,
            )
            for _ in range(num_envs)
        ]
        self.action_space_n = int(self.envs[0].action_space.n)
        if not config.allow_sideward_fire:
            self.action_space_n = 4

        self.screen_width, self.screen_height = 160, 210
        self.max_lives = 2
        self._lane_centers = torch.tensor([25.0, 74.0, 124.0], dtype=torch.float32)

        if seed is not None:
            for idx, env in enumerate(self.envs):
                env.reset(seed=seed + idx)

        self._slot_indices = self._build_slot_indices(self.envs[0])
        self._missile_slot_count = len(self._slot_indices.get("Missile", []))

        self.step_count = torch.zeros((num_envs,), dtype=torch.int32)
        self.window_agent = torch.full((num_envs,), -1, dtype=torch.int32)
        self.window_steps_remaining = torch.zeros((num_envs,), dtype=torch.int32)
        self.prev_enemy_visible = torch.zeros((num_envs, config.max_enemies), dtype=torch.int32)
        self.prev_enemy_raw = torch.zeros((num_envs, config.max_enemies, 2), dtype=torch.float32)
        self.prev_enemy_type = torch.zeros((num_envs, config.max_enemies), dtype=torch.int32)
        self.prev_player_missile_x = torch.zeros((num_envs,), dtype=torch.float32)
        self.prev_player_missile_y = torch.zeros((num_envs,), dtype=torch.float32)
        self.prev_player_missile_visible = torch.zeros((num_envs,), dtype=torch.int32)
        self.player_missile_history_len = 5
        self.player_missile_pos_history = torch.zeros(
            (num_envs, self.player_missile_history_len, 2), dtype=torch.float32
        )
        self.player_missile_history_valid = torch.zeros(
            (num_envs, self.player_missile_history_len), dtype=torch.bool
        )
        self.prev_missile_slot_raw = torch.zeros((num_envs, self._missile_slot_count, 2), dtype=torch.float32)
        self.prev_missile_slot_visible = torch.zeros((num_envs, self._missile_slot_count), dtype=torch.int32)
        self.player_missile_owner = torch.full((num_envs,), -1, dtype=torch.int32)
        self.enemy_missile_near_hit_active = torch.zeros((num_envs,), dtype=torch.bool)
        self.prev_enemy_missile_y = torch.zeros((num_envs,), dtype=torch.float32)
        self.prev_lives = torch.zeros((num_envs,), dtype=torch.int32)
        self.prev_building_health = torch.zeros((num_envs, 3), dtype=torch.float32)
        self.prev_building_x = torch.zeros((num_envs, 3), dtype=torch.float32)
        self.prev_score = torch.zeros((num_envs,), dtype=torch.float32)
        self.cumulative_score = torch.zeros((num_envs,), dtype=torch.float32)

        self.obs_stack = max(int(config.obs_stack), 1)

        # Global features:
        # player(2) + player_missile(3) + enemy_missile(3) + missile_vy(2) +
        # building_x(3) + building_health(3) + lives/score/window(4) = 20
        self._global_obs_dim = 20
        # Per-enemy features: x, y, visible, type_onehot(4) = 7
        self._per_enemy_dim = 7
        self._per_agent_dim = 2 + (config.num_agents if config.include_agent_id else 0)
        if config.single_agent_mode:
            self.base_obs_dim = self._global_obs_dim + config.max_enemies * self._per_enemy_dim
            self.obs_dim = self.base_obs_dim * self.obs_stack
            self.obs_shape = (num_envs, self.obs_dim)
            self.per_agent_obs_dim = None
            self._obs_history = torch.zeros((num_envs, self.obs_stack, self.base_obs_dim), dtype=torch.float32)
        else:
            visible_enemy_count = 1 if config.only_own_enemy else config.max_enemies
            self.base_per_agent_obs_dim = (
                self._global_obs_dim
                + visible_enemy_count * self._per_enemy_dim
                + self._per_agent_dim
            )
            self.per_agent_obs_dim = self.base_per_agent_obs_dim * self.obs_stack
            self.obs_dim = None
            self.obs_shape = (num_envs, config.num_agents, self.per_agent_obs_dim)
            self._obs_history = torch.zeros(
                (num_envs, config.num_agents, self.obs_stack, self.base_per_agent_obs_dim),
                dtype=torch.float32,
            )

    def reset(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        if seed is not None:
            self.gen.manual_seed(seed)
        obs_list = []
        for idx, env in enumerate(self.envs):
            env.reset(seed=None if seed is None else seed + idx)
            state = self._extract_state(env, idx)
            obs_list.append(self._build_obs(state, env_idx=idx))
            self.prev_enemy_visible[idx] = state["enemy_visible"]
            self.prev_enemy_raw[idx] = state["enemy_raw"]
            self.prev_enemy_type[idx] = state["enemy_type_id"]
            self.prev_player_missile_x[idx] = state["player_missile_raw"][0].item()
            self.prev_player_missile_y[idx] = state["player_missile_raw"][1].item()
            self.prev_player_missile_visible[idx] = int(state["player_missile"][2].item())
            self._reset_player_missile_history(idx, state)
            self._update_missile_slot_trackers(idx, state)
            self.player_missile_owner[idx] = -1
            self.enemy_missile_near_hit_active[idx] = False
            self.prev_enemy_missile_y[idx] = state["enemy_missile_raw"][1].item()
            self.prev_lives[idx] = state["lives_count"]
            self.prev_building_health[idx] = state["building_health"]
            self.prev_building_x[idx] = state["building_x"]
            self.prev_score[idx] = state["score"]
            self.step_count[idx] = 0
            self.window_agent[idx] = -1
            self.window_steps_remaining[idx] = 0
            self.cumulative_score[idx] = state["score"]

        obs = torch.stack(obs_list, dim=0)
        return self._reset_obs_history(obs), {}

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
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
            for env_idx in range(self.num_envs):
                if in_window[env_idx]:
                    ctrl = int(self.window_agent[env_idx].item())
                    if ctrl >= 0 and self.prev_enemy_visible[env_idx, ctrl] == 0:
                        in_window[env_idx] = False
                        self.window_steps_remaining[env_idx] = 0
                        self.window_agent[env_idx] = -1
            is_bidding_round = ~in_window.clone()
            winning_agent = self._select_winners(bids, in_window)
            apply_bid_penalty = (winning_agent >= 0) & (~in_window)
            current_window_length = torch.zeros((self.num_envs,), dtype=torch.int32)
            for env_idx in range(self.num_envs):
                if not in_window[env_idx] and int(winning_agent[env_idx].item()) < 0:
                    self.window_agent[env_idx] = -1
                    self.window_steps_remaining[env_idx] = 0
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
        score_list = []
        state_list = []
        reward_components_list = []
        bid_effect_list = []

        for env_idx, env in enumerate(self.envs):
            if cfg.single_agent_mode:
                chosen_action = int(action_dir[env_idx])
                controller = -1
            else:
                winner = int(winning_agent[env_idx].item())
                controller = winner
                chosen_action = 0 if winner < 0 else int(action_dir[env_idx, winner])

            if not cfg.allow_sideward_fire and chosen_action >= 4:
                chosen_action = 1 if chosen_action == 4 else 3
            is_fire_action = chosen_action in (1, 4, 5)
            pre_missile_visible = int(self.prev_player_missile_visible[env_idx].item()) == 1
            if not cfg.single_agent_mode and is_fire_action and controller >= 0 and not pre_missile_visible:
                self.player_missile_owner[env_idx] = controller

            _, raw_reward, terminated, truncated, _ = env.step(chosen_action)
            if terminated or truncated or (cfg.max_steps and self.step_count[env_idx] >= cfg.max_steps):
                terminated = bool(terminated)
                truncated = bool(truncated or (cfg.max_steps and self.step_count[env_idx] >= cfg.max_steps))

            state = self._extract_state(env, env_idx)
            missile_owner = int(self.player_missile_owner[env_idx].item())
            missile_visible = int(state["player_missile"][2].item()) == 1
            score_delta = float(state["score"].item() - self.prev_score[env_idx].item())
            self.cumulative_score[env_idx] += score_delta
            score_list.append(float(self.cumulative_score[env_idx].item()))

            enemy_disappeared = (self.prev_enemy_visible[env_idx] == 1) & (state["enemy_visible"] == 0)
            life_loss = max(int(self.prev_lives[env_idx]) - int(state["lives_count"]), 0)
            building_health_delta = self.prev_building_health[env_idx] - state["building_health"]
            building_damage = torch.clamp(building_health_delta, min=0.0)
            building_damage_total = float(building_damage.sum().item())
            damaged_building_penalties: list[tuple[int, float]] = []
            if building_damage_total > 0:
                damaged_slots = torch.where(building_damage > 0)[0]
                for slot_idx in damaged_slots.tolist():
                    bx = float(self.prev_building_x[env_idx, slot_idx].item())
                    if bx > 0.0:
                        agent_idx = int(torch.argmin(torch.abs(self._lane_centers / self.screen_width - bx)).item())
                        damage_amount = float(building_damage[slot_idx].item())
                        damaged_building_penalties.append((agent_idx, damage_amount))

            hit_agent = -1
            if score_delta > 0:
                candidates = torch.where(enemy_disappeared)[0]
                if candidates.numel() == 1:
                    hit_agent = int(candidates[0].item())
                elif candidates.numel() > 1:
                    prev_missile_y = self.prev_player_missile_y[env_idx].item()
                    prev_enemy_y = self.prev_enemy_raw[env_idx, candidates, 1]
                    hit_agent = int(candidates[torch.argmin(torch.abs(prev_enemy_y - prev_missile_y))].item())
                else:
                    # Air Raid often updates score without making the OCAtari enemy object disappear
                    # in the same frame. Attribute those hits by the latest visible missile
                    # position before the score event.
                    missile_pos = self._latest_visible_player_missile_pos(env_idx)
                    if missile_pos is not None:
                        hit_agent = int(torch.argmin(torch.abs(self._lane_centers - missile_pos[0])).item())

            enemy_destroy_reward = cfg.enemy_destroy_reward * (score_delta / 25.0) if score_delta > 0 else 0.0
            building_hit_penalty = -cfg.building_hit_penalty if building_damage_total > 0 else 0.0
            life_loss_penalty = -cfg.life_loss_penalty * life_loss if life_loss > 0 else 0.0
            raw_score_reward = cfg.raw_score_scale * score_delta
            unshot_enemy_disappeared = enemy_disappeared if score_delta <= 0 else torch.zeros_like(enemy_disappeared)
            enemy_disappear_count = int(unshot_enemy_disappeared.sum().item())
            enemy_disappear_penalty = -cfg.enemy_disappear_penalty * enemy_disappear_count
            enemy_missile_danger_penalty = self._enemy_missile_danger_penalty(state)
            enemy_missile_near_hit_penalty, enemy_missile_near_hit_event = self._enemy_missile_near_hit_penalty(
                env_idx, state, life_loss > 0
            )

            reward_components_list.append(
                {
                    "enemy_destroy": enemy_destroy_reward,
                    "raw_score": raw_score_reward,
                    "building_hit_penalty": building_hit_penalty,
                    "building_damage": building_damage_total,
                    "life_loss_penalty": life_loss_penalty,
                    "enemy_disappear_penalty": enemy_disappear_penalty,
                    "enemy_disappear_count": float(enemy_disappear_count),
                    "enemy_missile_danger_penalty": enemy_missile_danger_penalty,
                    "enemy_missile_near_hit_penalty": enemy_missile_near_hit_penalty,
                    "enemy_missile_near_hit_event": enemy_missile_near_hit_event,
                    "life_loss_count": float(life_loss),
                    "score_delta": score_delta,
                    "lives_current": float(state["lives_count"].item()),
                    "hit_agent": float(hit_agent),
                    "missile_owner": float(missile_owner),
                }
            )

            if cfg.single_agent_mode:
                reward = (
                    enemy_destroy_reward
                    + building_hit_penalty
                    + life_loss_penalty
                    + enemy_disappear_penalty
                    + enemy_missile_danger_penalty
                    + enemy_missile_near_hit_penalty
                    + raw_score_reward
                )
                rewards = torch.tensor(reward, dtype=torch.float32)
            else:
                rewards = torch.zeros((cfg.num_agents,), dtype=torch.float32)
                bid_effect = torch.zeros((cfg.num_agents,), dtype=torch.float32)
                winner = int(winning_agent[env_idx].item())
                if hit_agent >= 0 and missile_owner == hit_agent:
                    rewards[hit_agent] += enemy_destroy_reward
                if damaged_building_penalties:
                    if cfg.building_penalty_mode == "all_agents":
                        rewards += -cfg.building_hit_penalty
                    elif cfg.building_penalty_mode == "controller":
                        if winner >= 0:
                            rewards[winner] += -cfg.building_hit_penalty
                    elif cfg.building_penalty_mode != "lane":
                        raise ValueError(f"Unknown building_penalty_mode: {cfg.building_penalty_mode}")
                    else:
                        penalized_agents = {agent_idx for agent_idx, _ in damaged_building_penalties}
                        for agent_idx in penalized_agents:
                            rewards[agent_idx] += -cfg.building_hit_penalty
                if cfg.enemy_disappear_penalty != 0.0 and enemy_disappear_count > 0:
                    for agent_idx in torch.where(unshot_enemy_disappeared)[0].tolist():
                        if agent_idx < cfg.num_agents:
                            rewards[agent_idx] += -cfg.enemy_disappear_penalty
                if winner >= 0 and life_loss_penalty != 0.0:
                    rewards[winner] += life_loss_penalty
                if winner >= 0 and enemy_missile_danger_penalty != 0.0:
                    rewards[winner] += enemy_missile_danger_penalty
                if winner >= 0 and enemy_missile_near_hit_penalty != 0.0:
                    rewards[winner] += enemy_missile_near_hit_penalty
                if bids is not None and apply_bid_penalty[env_idx]:
                    bids_f = bids[env_idx].to(torch.float32)
                    if cfg.bidding_mechanism == "all_pay":
                        eff = cfg.bid_penalty * bids_f
                        rewards -= eff
                        bid_effect -= eff
                    elif winner >= 0:
                        if cfg.bidding_mechanism == "winner_pays":
                            eff = cfg.bid_penalty * float(bids_f[winner].item())
                            rewards[winner] -= eff
                            bid_effect[winner] -= eff
                        elif cfg.bidding_mechanism == "winner_pays_others_reward":
                            win_eff = cfg.bid_penalty * float(bids_f[winner].item())
                            rewards[winner] -= win_eff
                            bid_effect[winner] -= win_eff
                            for agent_idx in range(cfg.num_agents):
                                if agent_idx != winner:
                                    other_eff = cfg.bid_penalty * float(bids_f[agent_idx].item())
                                    rewards[agent_idx] += other_eff
                                    bid_effect[agent_idx] += other_eff
                    if cfg.window_bidding and cfg.window_penalty > 0 and current_window_length[env_idx] > 0:
                        win_pen = cfg.window_penalty * float(current_window_length[env_idx].item())
                        rewards[winner] -= win_pen
                        bid_effect[winner] -= win_pen
                if cfg.raw_score_scale != 0.0 and hit_agent >= 0:
                    rewards[hit_agent] += raw_score_reward
                bid_effect_list.append(bid_effect)

            if terminated or truncated:
                env.reset()
                state = self._extract_state(env, env_idx)
                self.step_count[env_idx] = 0
                self.cumulative_score[env_idx] = state["score"]
                self.player_missile_owner[env_idx] = -1
                self.enemy_missile_near_hit_active[env_idx] = False
            elif not missile_visible:
                self.player_missile_owner[env_idx] = -1

            obs_list.append(self._build_obs(state, env_idx=env_idx))
            rewards_list.append(rewards)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            state_list.append(state)

            self.prev_enemy_visible[env_idx] = state["enemy_visible"]
            self.prev_enemy_raw[env_idx] = state["enemy_raw"]
            self.prev_enemy_type[env_idx] = state["enemy_type_id"]
            self.prev_player_missile_x[env_idx] = state["player_missile_raw"][0].item()
            self.prev_player_missile_y[env_idx] = state["player_missile_raw"][1].item()
            self.prev_player_missile_visible[env_idx] = int(state["player_missile"][2].item())
            self._record_player_missile_history(env_idx, state)
            self._update_missile_slot_trackers(env_idx, state)
            self.prev_enemy_missile_y[env_idx] = state["enemy_missile_raw"][1].item()
            self.prev_lives[env_idx] = state["lives_count"]
            self.prev_building_health[env_idx] = state["building_health"]
            self.prev_building_x[env_idx] = state["building_x"]
            self.prev_score[env_idx] = state["score"]

        raw_obs = torch.stack(obs_list, dim=0)
        terminated_t = torch.tensor(terminated_list, dtype=torch.bool, device=self.device)
        truncated_t = torch.tensor(truncated_list, dtype=torch.bool, device=self.device)
        obs = self._update_obs_history(raw_obs, terminated_t.cpu() | truncated_t.cpu())
        rewards_t = torch.stack(rewards_list, dim=0).to(self.device)
        reward_components = {
            key: torch.tensor([rc[key] for rc in reward_components_list], dtype=torch.float32, device=self.device)
            for key in reward_components_list[0].keys()
        }

        info = {
            "winning_agent": winning_agent.to(self.device) if not cfg.single_agent_mode else None,
            "window_agent": self.window_agent.to(self.device),
            "window_steps_remaining": self.window_steps_remaining.to(self.device),
            "bid_penalty_applied": apply_bid_penalty.to(self.device),
            "score": torch.tensor(score_list, dtype=torch.float32, device=self.device),
            "player_xy_raw": torch.stack([s["player_xy_raw"] for s in state_list], dim=0).to(self.device),
            "player_missile_raw": torch.stack([s["player_missile_raw"] for s in state_list], dim=0).to(self.device),
            "player_missile_owner": self.player_missile_owner.to(self.device),
            "enemy_missile_raw": torch.stack([s["enemy_missile_raw"] for s in state_list], dim=0).to(self.device),
            "enemy_raw": torch.stack([s["enemy_raw"] for s in state_list], dim=0).to(self.device),
            "enemy_visible": torch.stack([s["enemy_visible"] for s in state_list], dim=0).to(self.device),
            "building_health": torch.stack([s["building_health"] for s in state_list], dim=0).to(self.device),
            "lives_count": torch.stack([s["lives_count"] for s in state_list], dim=0).to(self.device),
            "reward_components": reward_components,
        }
        if not cfg.single_agent_mode:
            info["bids"] = bids
            info["is_bidding_round"] = is_bidding_round.to(self.device)
            bid_effects = torch.stack(bid_effect_list, dim=0).to(self.device)
            info["reward_no_bid_sum"] = (rewards_t - bid_effects).sum(dim=1)
        return obs, rewards_t, terminated_t, truncated_t, info

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def render(self, env_idx: int = 0, show_agent_overlay: bool = True) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        frame = self.envs[env_idx].render()
        if frame is None or not show_agent_overlay or self.config.single_agent_mode:
            return frame

        frame = frame.copy()
        controlling_agent = int(self.window_agent[env_idx].item())
        agent_colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255)]
        if controlling_agent >= 0:
            color = agent_colors[controlling_agent % len(agent_colors)]
            cv2.putText(frame, f"Agent {controlling_agent}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            state = self._extract_state(self.envs[env_idx], env_idx)
            if controlling_agent < len(state["enemy_raw"]) and state["enemy_visible"][controlling_agent]:
                ex, ey = state["enemy_raw"][controlling_agent]
                cv2.circle(frame, (int(ex.item()), int(ey.item())), 12, color, 2)
        else:
            cv2.putText(frame, "No control", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        return frame

    def _reset_env(self, env_idx: int) -> torch.Tensor:
        state = self._extract_state(self.envs[env_idx], env_idx)
        self.step_count[env_idx] = 0
        self.window_agent[env_idx] = -1
        self.window_steps_remaining[env_idx] = 0
        self.prev_enemy_visible[env_idx] = state["enemy_visible"]
        self.prev_enemy_raw[env_idx] = state["enemy_raw"]
        self.prev_enemy_type[env_idx] = state["enemy_type_id"]
        self.prev_player_missile_x[env_idx] = state["player_missile_raw"][0].item()
        self.prev_player_missile_y[env_idx] = state["player_missile_raw"][1].item()
        self.prev_player_missile_visible[env_idx] = int(state["player_missile"][2].item())
        self._reset_player_missile_history(env_idx, state)
        self._update_missile_slot_trackers(env_idx, state)
        self.player_missile_owner[env_idx] = -1
        self.enemy_missile_near_hit_active[env_idx] = False
        self.prev_enemy_missile_y[env_idx] = state["enemy_missile_raw"][1].item()
        self.prev_lives[env_idx] = state["lives_count"]
        self.prev_building_health[env_idx] = state["building_health"]
        self.prev_building_x[env_idx] = state["building_x"]
        self.prev_score[env_idx] = state["score"]
        self.cumulative_score[env_idx] = state["score"]
        raw_obs = self._build_obs(state, env_idx=env_idx)
        return self._reset_obs_history_env(env_idx, raw_obs)

    def _stacked_obs(self) -> torch.Tensor:
        if self.config.single_agent_mode:
            stacked = self._obs_history.reshape(self.num_envs, self.obs_dim)
        else:
            stacked = self._obs_history.reshape(self.num_envs, self.config.num_agents, self.per_agent_obs_dim)
        return stacked.to(self.device)

    def _reset_obs_history(self, raw_obs: torch.Tensor) -> torch.Tensor:
        if self.config.single_agent_mode:
            self._obs_history = raw_obs.unsqueeze(1).repeat(1, self.obs_stack, 1).detach().cpu()
        else:
            self._obs_history = raw_obs.unsqueeze(2).repeat(1, 1, self.obs_stack, 1).detach().cpu()
        return self._stacked_obs()

    def _reset_obs_history_env(self, env_idx: int, raw_obs: torch.Tensor) -> torch.Tensor:
        if self.config.single_agent_mode:
            self._obs_history[env_idx] = raw_obs.unsqueeze(0).repeat(self.obs_stack, 1).detach().cpu()
            return self._obs_history[env_idx].reshape(self.obs_dim).to(self.device)
        self._obs_history[env_idx] = raw_obs.unsqueeze(1).repeat(1, self.obs_stack, 1).detach().cpu()
        return self._obs_history[env_idx].reshape(self.config.num_agents, self.per_agent_obs_dim).to(self.device)

    def _update_obs_history(self, raw_obs: torch.Tensor, done_mask: torch.Tensor) -> torch.Tensor:
        raw_obs = raw_obs.detach().cpu()
        if self.obs_stack == 1:
            self._obs_history = raw_obs.unsqueeze(1) if self.config.single_agent_mode else raw_obs.unsqueeze(2)
            return raw_obs.to(self.device)
        if self.config.single_agent_mode:
            self._obs_history = torch.roll(self._obs_history, shifts=-1, dims=1)
            self._obs_history[:, -1] = raw_obs
            for env_idx in torch.where(done_mask)[0].tolist():
                self._obs_history[env_idx] = raw_obs[env_idx].unsqueeze(0).repeat(self.obs_stack, 1)
        else:
            self._obs_history = torch.roll(self._obs_history, shifts=-1, dims=2)
            self._obs_history[:, :, -1] = raw_obs
            for env_idx in torch.where(done_mask)[0].tolist():
                self._obs_history[env_idx] = raw_obs[env_idx].unsqueeze(1).repeat(1, self.obs_stack, 1)
        return self._stacked_obs()

    def _reset_player_missile_history(self, env_idx: int, state: Dict[str, torch.Tensor]) -> None:
        self.player_missile_pos_history[env_idx].zero_()
        self.player_missile_history_valid[env_idx].zero_()
        self._record_player_missile_history(env_idx, state)

    def _record_player_missile_history(self, env_idx: int, state: Dict[str, torch.Tensor]) -> None:
        if int(state["player_missile"][2].item()) != 1:
            return
        pos = state["player_missile_raw"].to(torch.float32).cpu()
        if pos[0].item() <= 0.0 or pos[1].item() <= 0.0:
            return
        self.player_missile_pos_history[env_idx] = torch.roll(
            self.player_missile_pos_history[env_idx], shifts=-1, dims=0
        )
        self.player_missile_history_valid[env_idx] = torch.roll(
            self.player_missile_history_valid[env_idx], shifts=-1, dims=0
        )
        self.player_missile_pos_history[env_idx, -1] = pos
        self.player_missile_history_valid[env_idx, -1] = True

    def _latest_visible_player_missile_pos(self, env_idx: int) -> Optional[torch.Tensor]:
        valid = torch.where(self.player_missile_history_valid[env_idx])[0]
        if valid.numel() == 0:
            return None
        return self.player_missile_pos_history[env_idx, int(valid[-1].item())]

    def _enemy_missile_danger_penalty(self, state: Dict[str, torch.Tensor]) -> float:
        cfg = self.config
        if cfg.enemy_missile_danger_penalty <= 0.0:
            return 0.0
        if int(state["enemy_missile"][2].item()) != 1:
            return 0.0

        enemy_x, enemy_y = state["enemy_missile_raw"].tolist()
        player_x, player_y = state["player_xy_raw"].tolist()
        if enemy_y < cfg.enemy_missile_danger_y_threshold:
            return 0.0
        if enemy_y > player_y + cfg.enemy_missile_danger_y_radius:
            return 0.0

        x_radius = max(float(cfg.enemy_missile_danger_x_radius), 1.0)
        y_radius = max(float(cfg.enemy_missile_danger_y_radius), 1.0)
        x_severity = max(0.0, 1.0 - abs(enemy_x - player_x) / x_radius)
        y_severity = max(0.0, 1.0 - abs(player_y - enemy_y) / y_radius)
        severity = x_severity * y_severity
        if severity <= 0.0:
            return 0.0
        return -float(cfg.enemy_missile_danger_penalty) * severity

    def _enemy_missile_near_hit_penalty(
        self, env_idx: int, state: Dict[str, torch.Tensor], life_lost: bool
    ) -> Tuple[float, float]:
        cfg = self.config
        if cfg.enemy_missile_near_hit_penalty <= 0.0:
            self.enemy_missile_near_hit_active[env_idx] = False
            return 0.0, 0.0
        if int(state["enemy_missile"][2].item()) != 1:
            self.enemy_missile_near_hit_active[env_idx] = False
            return 0.0, 0.0

        enemy_x, enemy_y = state["enemy_missile_raw"].tolist()
        player_x, player_y = state["player_xy_raw"].tolist()
        y_margin = max(float(cfg.enemy_missile_near_hit_y_margin), 1.0)
        x_radius = max(float(cfg.enemy_missile_near_hit_x_radius), 1.0)
        near_player = abs(enemy_x - player_x) <= x_radius and abs(enemy_y - player_y) <= y_margin

        if life_lost:
            self.enemy_missile_near_hit_active[env_idx] = False
            return 0.0, 0.0
        if not near_player:
            if enemy_y > player_y + y_margin or abs(enemy_x - player_x) > x_radius:
                self.enemy_missile_near_hit_active[env_idx] = False
            return 0.0, 0.0
        if self.enemy_missile_near_hit_active[env_idx]:
            return 0.0, 0.0

        self.enemy_missile_near_hit_active[env_idx] = True
        return -float(cfg.enemy_missile_near_hit_penalty), 1.0

    def _update_missile_slot_trackers(self, env_idx: int, state: Dict[str, torch.Tensor]) -> None:
        if self._missile_slot_count == 0:
            return
        self.prev_missile_slot_raw[env_idx] = state["missile_slot_raw"]
        self.prev_missile_slot_visible[env_idx] = state["missile_slot_visible"]

    def partial_reset(self, done_mask: torch.Tensor) -> torch.Tensor:
        if self.config.single_agent_mode:
            result = torch.zeros(self.num_envs, self.obs_dim, device=self.device)
        else:
            result = torch.zeros(
                self.num_envs,
                self.config.num_agents,
                self.per_agent_obs_dim,
                device=self.device,
            )
        for env_idx in range(self.num_envs):
            if done_mask[env_idx].item():
                result[env_idx] = self._reset_env(env_idx)
        return result

    def _select_winners(self, bids: torch.Tensor, in_window: torch.Tensor) -> torch.Tensor:
        max_bid = bids.max(dim=1).values
        winners = torch.full((self.num_envs,), -1, dtype=torch.int32)
        for env_idx in range(self.num_envs):
            if in_window[env_idx]:
                winners[env_idx] = self.window_agent[env_idx]
                continue
            if max_bid[env_idx] <= 0 and self.config.bid_upper_bound > 0:
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

    def _extract_state(self, env: OCAtari, env_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        slots = env.objects
        idx = self._slot_indices

        def slot_obj(indices: List[int], offset: int = 0):
            if len(indices) <= offset:
                return None
            return slots[indices[offset]]

        def slot_list(indices: List[int]) -> List:
            return [slots[i] for i in indices]

        def to_xy_vis(obj) -> Tuple[float, float, int]:
            if obj is None or obj.__class__.__name__ == "NoObject":
                return 0.0, 0.0, 0
            x, y = obj.xy
            return float(x) / self.screen_width, float(y) / self.screen_height, 1

        def to_xy_raw(obj) -> Tuple[float, float]:
            if obj is None or obj.__class__.__name__ == "NoObject":
                return 0.0, 0.0
            x, y = obj.xy
            return float(x), float(y)

        player = slot_obj(idx.get("Player", []))
        missile_slots = idx.get("Missile", [])
        missiles = slot_list(missile_slots)
        building_objs = slot_list(idx.get("Building", []))
        lives_obj = slot_obj(idx.get("Lives", []))
        score_obj = slot_obj(idx.get("PlayerScore", []))

        player_x, player_y, _ = to_xy_vis(player)
        player_raw = to_xy_raw(player)

        missile_slot_raw = []
        missile_slot_visible = []
        missile_candidates = []
        for offset, obj in enumerate(missiles):
            raw_x, raw_y = to_xy_raw(obj)
            visible = 0 if obj is None or obj.__class__.__name__ == "NoObject" else 1
            missile_slot_raw.append((raw_x, raw_y))
            missile_slot_visible.append(visible)
            if not visible:
                continue
            dy = 0.0
            if env_idx is not None and offset < self.prev_missile_slot_raw.shape[1]:
                prev_visible = int(self.prev_missile_slot_visible[env_idx, offset].item()) == 1
                prev_y = float(self.prev_missile_slot_raw[env_idx, offset, 1].item())
                if prev_visible and prev_y > 0.0:
                    dy = raw_y - prev_y
            missile_candidates.append((offset, obj, raw_x, raw_y, dy))
        while len(missile_slot_raw) < self._missile_slot_count:
            missile_slot_raw.append((0.0, 0.0))
            missile_slot_visible.append(0)

        player_missile_obj = None
        enemy_missile_obj = None
        upward = [item for item in missile_candidates if item[4] < 0.0]
        downward = [item for item in missile_candidates if item[4] > 0.0]
        if upward:
            player_missile_obj = min(upward, key=lambda item: item[4])[1]
        if downward:
            enemy_missile_obj = max(downward, key=lambda item: item[4])[1]

        # At missile spawn there may be no motion history yet. Air Raid exposes two
        # fixed Missile slots in RAM: slot 0 is the player's upward missile, slot 1
        # is the enemy's downward missile. Use that only as a fallback.
        if player_missile_obj is None:
            for offset, obj, *_ in missile_candidates:
                if offset == 0:
                    player_missile_obj = obj
                    break
        if enemy_missile_obj is None:
            for offset, obj, *_ in missile_candidates:
                if offset == 1:
                    enemy_missile_obj = obj
                    break

        pm_x, pm_y, pm_vis = to_xy_vis(player_missile_obj)
        em_x, em_y, em_vis = to_xy_vis(enemy_missile_obj)
        player_missile_raw = to_xy_raw(player_missile_obj)
        enemy_missile_raw = to_xy_raw(enemy_missile_obj)

        ram_state = env.ale.getRAM()
        enemy_features_by_lane = []
        enemy_raw_by_lane = []
        enemy_visible_by_lane = []
        enemy_type_ids = []
        type_to_bucket = {
            "Enemy25": 0,
            "Enemy50": 1,
            "Enemy75": 2,
            "Enemy100": 3,
        }
        visible_enemies = []
        for class_name in ("Enemy25", "Enemy50", "Enemy75", "Enemy100"):
            for obj in slot_list(idx.get(class_name, [])):
                if obj.__class__.__name__ == "NoObject":
                    continue
                raw_x, raw_y = to_xy_raw(obj)
                visible_enemies.append((raw_x, raw_y, type_to_bucket[class_name], obj))
        lane_assignments = [None] * self.config.max_enemies
        for raw_x, raw_y, bucket, obj in sorted(visible_enemies, key=lambda item: item[0]):
            lane_idx = int(torch.argmin(torch.abs(self._lane_centers - raw_x)).item())
            if lane_assignments[lane_idx] is None:
                lane_assignments[lane_idx] = (raw_x, raw_y, bucket, obj)
            else:
                for fallback in range(self.config.max_enemies):
                    if lane_assignments[fallback] is None:
                        lane_assignments[fallback] = (raw_x, raw_y, bucket, obj)
                        break

        for lane_idx in range(self.config.max_enemies):
            lane_obj = lane_assignments[lane_idx]
            if lane_obj is None:
                enemy_features_by_lane.append((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                enemy_raw_by_lane.append((0.0, 0.0))
                enemy_visible_by_lane.append(0)
                enemy_type_ids.append(-1)
                continue
            raw_x, raw_y, bucket, obj = lane_obj
            ex, ey, ev = to_xy_vis(obj)
            onehot = [0.0, 0.0, 0.0, 0.0]
            onehot[bucket] = 1.0
            enemy_features_by_lane.append((ex, ey, float(ev), *onehot))
            enemy_raw_by_lane.append((raw_x, raw_y))
            enemy_visible_by_lane.append(ev)
            enemy_type_ids.append(bucket)

        building_x = []
        building_health = []
        # Air Raid has two real buildings. OCAtari exposes a third Building slot
        # for the wraparound visual fragment when a building crosses the screen
        # edge; treating that fragment as a separate building creates fake damage
        # events whenever it appears or disappears.
        h_by_damage = [32, 29, 29, 29, 27, 23, 21, 19, 15, 19, 23, 25, 25, 8, 32]
        for building_idx in range(2):
            obj = building_objs[building_idx] if building_idx < len(building_objs) else None
            bx, _, _ = to_xy_vis(obj)
            damage_idx = int(ram_state[27 + building_idx])
            health = float(h_by_damage[damage_idx]) / 32.0 if 0 <= damage_idx < len(h_by_damage) else 0.0
            building_x.append(bx)
            building_health.append(health)
        building_x.append(0.0)
        building_health.append(0.0)

        score_value = float(getattr(score_obj, "score", 0.0)) if score_obj is not None else 0.0
        lives_count = int(getattr(lives_obj, "lives", 0)) if lives_obj is not None and lives_obj.__class__.__name__ != "NoObject" else 0
        # Two buildings alive is equivalent to full health.
        buildings_alive = sum(1 for health in building_health[:2] if health > 0)

        return {
            "player_xy": torch.tensor([player_x, player_y], dtype=torch.float32),
            "player_xy_raw": torch.tensor(player_raw, dtype=torch.float32),
            "player_missile": torch.tensor([pm_x, pm_y, pm_vis], dtype=torch.float32),
            "player_missile_raw": torch.tensor(player_missile_raw, dtype=torch.float32),
            "enemy_missile": torch.tensor([em_x, em_y, em_vis], dtype=torch.float32),
            "enemy_missile_raw": torch.tensor(enemy_missile_raw, dtype=torch.float32),
            "missile_slot_raw": torch.tensor(missile_slot_raw, dtype=torch.float32),
            "missile_slot_visible": torch.tensor(missile_slot_visible, dtype=torch.int32),
            "enemy_features": torch.tensor(enemy_features_by_lane, dtype=torch.float32),
            "enemy_raw": torch.tensor(enemy_raw_by_lane, dtype=torch.float32),
            "enemy_visible": torch.tensor(enemy_visible_by_lane, dtype=torch.int32),
            "enemy_type_id": torch.tensor(enemy_type_ids, dtype=torch.int32),
            "building_x": torch.tensor(building_x, dtype=torch.float32),
            "building_health": torch.tensor(building_health, dtype=torch.float32),
            "lives_count": torch.tensor(lives_count, dtype=torch.int32),
            "lives_norm": torch.tensor(float(lives_count) / float(max(self.max_lives, 1)), dtype=torch.float32),
            "buildings_alive": torch.tensor(buildings_alive, dtype=torch.int32),
            "score": torch.tensor(score_value, dtype=torch.float32),
            "score_visible": torch.tensor(1.0 if score_obj is not None else 0.0, dtype=torch.float32),
        }

    def _build_obs(self, state: Dict[str, torch.Tensor], env_idx: int) -> torch.Tensor:
        cfg = self.config
        player_missile_vy = (
            state["player_missile_raw"][1].item() - self.prev_player_missile_y[env_idx].item()
        ) / float(self.screen_height)
        enemy_missile_vy = (
            state["enemy_missile_raw"][1].item() - self.prev_enemy_missile_y[env_idx].item()
        ) / float(self.screen_height)
        prev_score = self.prev_score[env_idx].item()
        curr_score = state["score"].item()

        global_features = torch.cat(
            [
                state["player_xy"],
                state["player_missile"],
                state["enemy_missile"],
                torch.tensor([player_missile_vy, enemy_missile_vy], dtype=torch.float32),
                state["building_x"],
                state["building_health"],
                torch.tensor(
                    [
                        state["lives_norm"].item(),
                        1.0 if curr_score > prev_score else 0.0,
                        state["score_visible"].item(),
                        float(self.window_steps_remaining[env_idx].item()) / float(max(cfg.action_window, 1)),
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
        current_controller = int(self.window_agent[env_idx].item())
        is_bidding_step = 1.0 if self.window_steps_remaining[env_idx].item() == 0 else 0.0
        for agent_id in range(cfg.num_agents):
            target_enemy = state["enemy_features"][agent_id]
            agent_features = torch.tensor(
                [
                    1.0 if agent_id == current_controller else 0.0,
                    is_bidding_step,
                ],
                dtype=torch.float32,
            )
            if cfg.include_agent_id:
                agent_id_features = torch.zeros(cfg.num_agents, dtype=torch.float32)
                agent_id_features[agent_id] = 1.0
                agent_features = torch.cat([agent_features, agent_id_features], dim=0)
            if cfg.only_own_enemy:
                per_agent_obs.append(torch.cat([global_features, target_enemy, agent_features], dim=0))
            else:
                other_enemies = torch.cat(
                    [
                        state["enemy_features"][:agent_id],
                        state["enemy_features"][agent_id + 1:],
                    ],
                    dim=0,
                ).reshape(-1)
                per_agent_obs.append(torch.cat([global_features, target_enemy, other_enemies, agent_features], dim=0))
        return torch.stack(per_agent_obs, dim=0)
