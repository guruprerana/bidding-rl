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
    hit_penalty: float = 1.0  # Penalty when temperature bar turns red (overheat)
    life_loss_penalty: float = 10.0
    raw_score_scale: float = 0.0  # Scale for raw Atari score (dense reward signal)
    fire_while_hot_penalty: float = 0.0  # Penalty for firing when health bar is red
    max_steps: int = 10000
    hud: bool = True
    single_agent_mode: bool = False
    allow_variable_enemies: bool = True
    allow_sideward_fire: bool = True  # If False, disables RIGHTFIRE and LEFTFIRE actions


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
                "AssaultNoFrameskip-v4",
                obs_mode="obj",
                hud=True,
                render_mode=render_mode,
                render_oc_overlay=render_oc_overlay,
            )
            for _ in range(num_envs)
        ]
        # Action space: 0=NOOP, 1=FIRE, 2=UP, 3=RIGHT, 4=LEFT, [5=RIGHTFIRE, 6=LEFTFIRE]
        if config.allow_sideward_fire:
            self.action_space_n = int(self.envs[0].action_space.n)  # Full action space (7)
        else:
            self.action_space_n = 5  # Exclude RIGHTFIRE and LEFTFIRE
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
        self.cumulative_score = torch.zeros((num_envs,), dtype=torch.float32)
        # Track previous missile Y positions for row-based hit attribution
        self.prev_missile_v_y = torch.zeros((num_envs,), dtype=torch.float32)
        # Enemy row Y positions (approximate): Bottom ~103, Middle ~78, Top ~53
        self._enemy_row_y = torch.tensor([103.0, 78.0, 53.0], dtype=torch.float32)

        # Global obs: player(2) + mothership(3) + enemy_missile(3) + player_missiles(6) +
        #             window(1) + health/lives(3) + enemy_type_onehot(3) = 21
        self._global_obs_dim = 21
        # Per-enemy features: x, y, visible, is_small_or_split, is_doubled = 5
        self._per_enemy_dim = 5
        # Per-agent features (multi-agent only): is_in_control(1) + is_bidding_step(1) = 2
        self._per_agent_dim = 2
        if config.single_agent_mode:
            self.obs_dim = self._global_obs_dim + config.max_enemies * self._per_enemy_dim
            self.obs_shape = (num_envs, self.obs_dim)
            self.per_agent_obs_dim = None
        else:
            # Add per-agent control flag to observation
            self.per_agent_obs_dim = self._global_obs_dim + config.max_enemies * self._per_enemy_dim + self._per_agent_dim
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
            self.cumulative_score[idx] = 0.0
            self.prev_missile_v_y[idx] = 0.0

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
        # Reward component tracking
        reward_components_list = []

        for env_idx, env in enumerate(self.envs):
            if cfg.single_agent_mode:
                chosen_action = int(action_dir[env_idx])
            else:
                winner = int(winning_agent[env_idx].item())
                if winner < 0:
                    chosen_action = 0
                else:
                    chosen_action = int(action_dir[env_idx, winner])

            _, raw_reward, terminated, truncated, info = env.step(chosen_action)
            if terminated or truncated or (cfg.max_steps and self.step_count[env_idx] >= cfg.max_steps):
                terminated = bool(terminated)
                truncated = bool(truncated or (cfg.max_steps and self.step_count[env_idx] >= cfg.max_steps))

            # Track cumulative Atari score (raw reward = score delta)
            self.cumulative_score[env_idx] += float(raw_reward)

            state = self._extract_state(env)
            score_list.append(float(self.cumulative_score[env_idx]))

            enemy_visible = state["enemy_visible"]
            destroyed = (self.prev_enemy_visible[env_idx] == 1) & (enemy_visible == 0)
            enemy_destroyed_list.append(destroyed.clone())

            # Detect enemy destruction via raw score increase (more reliable than visibility)
            enemy_hit = raw_reward > 0

            # Row-based hit attribution using missile Y position
            # When score increases, determine which row was hit based on previous missile Y
            hit_agent = -1
            if enemy_hit and not cfg.single_agent_mode:
                missile_v_y = state["player_missile_v_raw"][1].item()
                prev_y = self.prev_missile_v_y[env_idx].item()
                # If missile disappeared (y went to 0) and we had a previous position
                if missile_v_y == 0.0 and prev_y > 0:
                    # Find closest enemy row to previous missile Y
                    distances = torch.abs(self._enemy_row_y - prev_y)
                    hit_agent = int(torch.argmin(distances).item())

            life_loss = max(int(self.prev_lives[env_idx]) - int(state["lives_count"]), 0)
            # Temperature bar turning red = overheating warning
            overheat_event = int(state["health_red"]) == 1 and int(self.prev_health_red[env_idx]) == 0

            # Track individual reward components
            # For single-agent: use score-based detection
            # For multi-agent: attribute to agent whose row was hit
            if cfg.single_agent_mode:
                enemy_destroy_reward = cfg.enemy_destroy_reward if enemy_hit else 0.0
            else:
                enemy_destroy_reward = cfg.enemy_destroy_reward if hit_agent >= 0 else 0.0
            raw_score_reward = cfg.raw_score_scale * float(raw_reward)
            life_loss_penalty = -cfg.life_loss_penalty * life_loss if life_loss > 0 else 0.0
            overheat_penalty = -cfg.hit_penalty if overheat_event else 0.0
            is_fire_action = chosen_action in (1, 5, 6)  # FIRE, RIGHTFIRE, LEFTFIRE
            # Use prev_health_red (pre-step state) because when life is lost, health resets
            # and state["health_red"] would be 0, missing the penalty for the fatal shot
            fire_while_hot_penalty = -cfg.fire_while_hot_penalty if (
                cfg.fire_while_hot_penalty > 0 and is_fire_action and int(self.prev_health_red[env_idx]) == 1
            ) else 0.0

            penalty = life_loss_penalty + overheat_penalty + fire_while_hot_penalty

            reward_components_list.append({
                "enemy_destroy": enemy_destroy_reward,
                "raw_score": raw_score_reward,
                "life_loss_penalty": life_loss_penalty,
                "overheat_penalty": overheat_penalty,
                "fire_while_hot_penalty": fire_while_hot_penalty,
                "life_loss_count": float(life_loss),
                "lives_current": float(state["lives_count"].item()),
                "death": 1.0 if terminated else 0.0,
                # Debug fields for overheat behavior
                "health_red_pre": float(self.prev_health_red[env_idx].item()),
                "health_red_post": float(state["health_red"].item()),
                "is_fire_action": 1.0 if is_fire_action else 0.0,
                "fired_while_hot": 1.0 if (is_fire_action and int(self.prev_health_red[env_idx]) == 1) else 0.0,
            })

            if cfg.single_agent_mode:
                reward = enemy_destroy_reward + penalty + raw_score_reward
                rewards = torch.tensor(reward, dtype=torch.float32)
            else:
                rewards = torch.zeros((cfg.num_agents,), dtype=torch.float32)
                # Agent whose row was hit gets the destroy reward
                if hit_agent >= 0 and hit_agent < cfg.num_agents:
                    rewards[hit_agent] += cfg.enemy_destroy_reward
                # Raw score reward goes to all agents
                rewards += raw_score_reward
                # All penalties only apply to the winning agent who took the action
                winner = int(winning_agent[env_idx].item())
                if winner >= 0:
                    rewards[winner] += penalty  # life_loss + overheat + fire_while_hot
                if bids is not None and apply_bid_penalty[env_idx]:
                    rewards -= cfg.bid_penalty * bids[env_idx].to(torch.float32)
                    if cfg.window_bidding and cfg.window_penalty > 0 and current_window_length[env_idx] > 0:
                        winner = int(winning_agent[env_idx].item())
                        rewards[winner] -= cfg.window_penalty * float(current_window_length[env_idx].item())

            if terminated or truncated:
                env.reset()
                state = self._extract_state(env)
                self.step_count[env_idx] = 0
                self.cumulative_score[env_idx] = 0.0
                self.prev_missile_v_y[env_idx] = 0.0

            obs_list.append(self._build_obs(state, env_idx=env_idx))
            rewards_list.append(rewards)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            state_list.append(state)

            self.prev_enemy_visible[env_idx] = state["enemy_visible"]
            self.prev_lives[env_idx] = state["lives_count"]
            self.prev_missile_v_y[env_idx] = state["player_missile_v_raw"][1].item()
            self.prev_health_width[env_idx] = state["health_width"]
            self.prev_health_red[env_idx] = state["health_red"]

        obs = torch.stack(obs_list, dim=0).to(self.device)
        terminated_t = torch.tensor(terminated_list, dtype=torch.bool, device=self.device)
        truncated_t = torch.tensor(truncated_list, dtype=torch.bool, device=self.device)

        if cfg.single_agent_mode:
            rewards_t = torch.stack(rewards_list, dim=0).to(self.device)
        else:
            rewards_t = torch.stack(rewards_list, dim=0).to(self.device)

        # Aggregate reward components into tensors
        reward_components = {
            key: torch.tensor([rc[key] for rc in reward_components_list], dtype=torch.float32, device=self.device)
            for key in reward_components_list[0].keys()
        }

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
            "reward_components": reward_components,
        }
        return obs, rewards_t, terminated_t, truncated_t, info

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def render(self, env_idx: int = 0) -> Optional[np.ndarray]:
        """
        Render the specified environment.

        Args:
            env_idx: Index of the environment to render (default: 0)

        Returns:
            RGB array if render_mode="rgb_array", None otherwise
        """
        if self.render_mode is None:
            return None
        return self.envs[env_idx].render()

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

        # Enemy type from RGB: green=(72,160,72), blue=(84,138,210), brown=(105,77,20)
        # Use first visible enemy to determine type (all enemies same type per wave)
        enemy_type_onehot = [0.0, 0.0, 0.0]
        for enemy in enemy_objs:
            if enemy is not None and enemy.__class__.__name__ != "NoObject":
                rgb = tuple(getattr(enemy, "rgb", (0, 0, 0)))
                if rgb == (72, 160, 72):
                    enemy_type_onehot[0] = 1.0  # green
                elif rgb == (84, 138, 210):
                    enemy_type_onehot[1] = 1.0  # blue
                elif rgb == (105, 77, 20):
                    enemy_type_onehot[2] = 1.0  # brown
                break  # All enemies same type, just need first one

        # Enemy features: (x, y, visible, is_small_or_split, is_doubled)
        # is_small_or_split inferred from wh: normal=(16,8), small/split=(8,8)
        # is_doubled from RAM: appearance 224 means two targets (OCAtari only tracks one position)
        # Appearance values: 0=not visible, 192=normal, 160=one small, 224=doubled, 96=other small
        ram_state = env.ale.getRAM()
        enemy_appearance = ram_state[54:57]  # Per-enemy appearance

        enemy_features = []
        enemy_raw = []
        enemy_visible = []
        for i, enemy in enumerate(enemy_objs[: self.config.max_enemies]):
            ex, ey, ev = to_xy_vis(enemy)
            # Check if enemy is in small/split form (width 8 instead of 16)
            if enemy is not None and enemy.__class__.__name__ != "NoObject":
                wh = getattr(enemy, "wh", (16, 8))
                is_small_or_split = 1.0 if wh[0] == 8 else 0.0
            else:
                is_small_or_split = 0.0
            # Check if enemy has split into two targets (appearance == 224)
            is_doubled = 1.0 if (i < len(enemy_appearance) and enemy_appearance[i] == 224) else 0.0
            enemy_features.append((ex, ey, ev, is_small_or_split, is_doubled))
            enemy_visible.append(ev)
            enemy_raw.append(to_xy_raw(enemy))
        while len(enemy_features) < self.config.max_enemies:
            enemy_features.append((0.0, 0.0, 0, 0.0, 0.0))
            enemy_visible.append(0)
            enemy_raw.append((0.0, 0.0))

        # Extract lives and health from detected objects (hud=True)
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
            "enemy_type_onehot": torch.tensor(enemy_type_onehot, dtype=torch.float32),
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
                state["enemy_type_onehot"],  # One-hot: green, blue, brown (3 features)
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
            other_enemies = torch.cat(
                [
                    state["enemy_features"][:agent_id],
                    state["enemy_features"][agent_id + 1:],
                ],
                dim=0,
            ).reshape(-1)
            # Per-agent features:
            # - is_in_control: is this agent currently controlling the action?
            # - is_bidding_step: are bids evaluated this step? (shared, but included per-agent for convenience)
            agent_features = torch.tensor(
                [
                    1.0 if agent_id == current_controller else 0.0,  # is_in_control
                    is_bidding_step,  # is_bidding_step
                ],
                dtype=torch.float32,
            )
            per_agent_obs.append(torch.cat([global_features, target_enemy, other_enemies, agent_features], dim=0))
        return torch.stack(per_agent_obs, dim=0)

