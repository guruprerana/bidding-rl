"""
Multi-agent PPO trainer for OCAtari Air Raid with bidding control.
"""

import os
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from airraid.airraid_torch import AirRaidConfig, AirRaidEnv
from ppo_trainer_base import MultiAgentPPOTrainerBase
from ppo_utils import layer_init


@dataclass
class AirRaidArgs:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "bidding-rl"
    wandb_entity: str = None

    num_agents: int = 3
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
    max_steps: int = 10000
    hud: bool = True
    allow_sideward_fire: bool = True
    bidding_mechanism: str = "all_pay"
    only_own_enemy: bool = False

    actor_hidden_sizes: Tuple[int, ...] = (128, 128, 128)
    critic_hidden_sizes: Tuple[int, ...] = (256, 256, 256)

    num_iterations: int = 1000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    batch_size: int = 0
    minibatch_size: int = 0
    total_timesteps: int = 0


class AirRaidSharedAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_space_n: int,
        bid_upper_bound: int,
        window_bidding: bool,
        action_window: int,
        actor_hidden_sizes: Tuple[int, ...],
        critic_hidden_sizes: Tuple[int, ...],
    ):
        super().__init__()
        self.window_bidding = window_bidding
        actor_layers = []
        actor_in = obs_dim
        for hidden in actor_hidden_sizes:
            actor_layers.append(layer_init(nn.Linear(actor_in, hidden)))
            actor_layers.append(nn.ELU())
            actor_in = hidden
        self.actor = nn.Sequential(*actor_layers) if actor_layers else nn.Identity()
        self.actor_feature_dim = actor_in
        self.direction_head = layer_init(nn.Linear(self.actor_feature_dim, action_space_n), std=0.01)
        self.bid_head = layer_init(nn.Linear(self.actor_feature_dim, bid_upper_bound + 1), std=0.01)
        self.window_head = None
        if self.window_bidding:
            self.window_head = layer_init(nn.Linear(self.actor_feature_dim, action_window), std=0.01)

        critic_layers = []
        critic_in = obs_dim
        for hidden in critic_hidden_sizes:
            critic_layers.append(layer_init(nn.Linear(critic_in, hidden)))
            critic_layers.append(nn.ELU())
            critic_in = hidden
        critic_layers.append(layer_init(nn.Linear(critic_in, 1), std=1.0))
        self.critic = nn.Sequential(*critic_layers)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        features = self.actor(x)
        direction_logits = self.direction_head(features)
        bid_logits = self.bid_head(features)
        direction_dist = Categorical(logits=direction_logits)
        bid_dist = Categorical(logits=bid_logits)
        window_dist = Categorical(logits=self.window_head(features)) if self.window_bidding else None
        if action is None:
            direction = direction_dist.sample()
            bid = bid_dist.sample()
            if self.window_bidding:
                window = window_dist.sample()
                action = torch.stack([direction, bid, window], dim=-1)
            else:
                action = torch.stack([direction, bid], dim=-1)
        else:
            direction = action[..., 0]
            bid = action[..., 1]
            if self.window_bidding:
                window = action[..., 2]
        log_prob = direction_dist.log_prob(direction) + bid_dist.log_prob(bid)
        entropy = direction_dist.entropy() + bid_dist.entropy()
        if self.window_bidding:
            log_prob = log_prob + window_dist.log_prob(window)
            entropy = entropy + window_dist.entropy()
        value = self.critic(x)
        return action, log_prob, entropy, value


class AirRaidPPOTrainer(MultiAgentPPOTrainerBase):
    def __init__(self, args: AirRaidArgs, callbacks=None):
        super().__init__(args, callbacks=callbacks)
        self.obs_dim = None
        self.num_action_components = 3 if self.args.window_bidding else 2

    def setup(self):
        env_config = AirRaidConfig(
            num_agents=self.args.num_agents,
            max_enemies=self.args.max_enemies,
            bid_upper_bound=self.args.bid_upper_bound,
            bid_penalty=self.args.bid_penalty,
            action_window=self.args.action_window,
            window_bidding=self.args.window_bidding,
            window_penalty=self.args.window_penalty,
            enemy_destroy_reward=self.args.enemy_destroy_reward,
            building_hit_penalty=self.args.building_hit_penalty,
            life_loss_penalty=self.args.life_loss_penalty,
            raw_score_scale=self.args.raw_score_scale,
            max_steps=self.args.max_steps,
            hud=self.args.hud,
            single_agent_mode=False,
            allow_sideward_fire=self.args.allow_sideward_fire,
            bidding_mechanism=self.args.bidding_mechanism,
            only_own_enemy=self.args.only_own_enemy,
        )
        self.envs = AirRaidEnv(env_config, num_envs=self.args.num_envs, device=self.device, seed=self.args.seed)
        self.obs_dim = self.envs.per_agent_obs_dim
        self.agent = AirRaidSharedAgent(
            obs_dim=self.obs_dim,
            action_space_n=self.envs.action_space_n,
            bid_upper_bound=self.args.bid_upper_bound,
            window_bidding=self.args.window_bidding,
            action_window=self.args.action_window,
            actor_hidden_sizes=self.args.actor_hidden_sizes,
            critic_hidden_sizes=self.args.critic_hidden_sizes,
        ).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)
        self.args.batch_size = int(self.args.num_envs * self.args.num_steps * self.args.num_agents)
        self.args.minibatch_size = int(self.args.batch_size // self.args.num_minibatches)

    def _on_iteration_start(self, iteration: int):
        if not self.args.track:
            return
        self._episode_reward_no_bid_sum = torch.zeros((), device=self.device, dtype=torch.float32)

    def _on_rollout_step(self, infos, global_step: int):
        if not self.args.track or not isinstance(infos, dict):
            return
        reward_no_bid_sum = infos.get("reward_no_bid_sum", None)
        if torch.is_tensor(reward_no_bid_sum):
            self._episode_reward_no_bid_sum += reward_no_bid_sum.sum()

    def _extra_log_dict(self, global_step: int) -> dict:
        if not self._last_rollout_stats:
            return {}
        rewards = self._last_rollout_stats["rewards"]
        log_dict = {"rewards/avg_step_reward": rewards.mean().item()}
        if self.args.track and hasattr(self, "_episode_reward_no_bid_sum"):
            n = self.args.num_envs * self.args.num_steps * self.args.num_agents
            log_dict["rewards/avg_step_reward_no_bid"] = (self._episode_reward_no_bid_sum / n).item()
        return log_dict

    def save_model(self, path: str | None = None):
        torch.save(self.agent.state_dict(), path or "airraid_agent.pt")

    def cleanup(self):
        if self.envs is not None:
            self.envs.close()
