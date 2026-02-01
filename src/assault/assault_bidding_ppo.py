"""
Multi-agent PPO trainer for OCAtari Assault with bidding control.
"""

import os
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from assault.assault_torch import AssaultConfig, AssaultEnv
from ppo_utils import layer_init
from ppo_trainer_base import MultiAgentPPOTrainerBase


@dataclass
class AssaultArgs:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "bidding-rl"
    wandb_entity: str = None

    # Environment
    num_agents: int = 3
    max_enemies: int = 3
    bid_upper_bound: int = 10
    bid_penalty: float = 0.1
    action_window: int = 1
    window_bidding: bool = False
    window_penalty: float = 0.0
    enemy_destroy_reward: float = 1.0
    hit_penalty: float = 1.0  # Penalty when temperature bar turns red (overheat)
    life_loss_penalty: float = 10.0
    raw_score_scale: float = 0.0  # Scale for raw Atari score
    fire_while_hot_penalty: float = 0.0  # Penalty for firing when health bar is red
    max_steps: int = 10000
    hud: bool = True
    allow_variable_enemies: bool = True
    allow_sideward_fire: bool = True  # If False, disables RIGHTFIRE and LEFTFIRE

    # Network
    actor_hidden_sizes: Tuple[int, ...] = (128, 128, 128)
    critic_hidden_sizes: Tuple[int, ...] = (256, 256, 256)

    # PPO
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

    # Runtime filled
    batch_size: int = 0
    minibatch_size: int = 0
    total_timesteps: int = 0


class AssaultSharedAgent(nn.Module):
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


class AssaultPPOTrainer(MultiAgentPPOTrainerBase):
    def __init__(self, args: AssaultArgs, callbacks=None):
        super().__init__(args, callbacks=callbacks)
        self.obs_dim = None
        self.num_action_components = 3 if self.args.window_bidding else 2

    def setup(self):
        env_config = AssaultConfig(
            num_agents=self.args.num_agents,
            max_enemies=self.args.max_enemies,
            bid_upper_bound=self.args.bid_upper_bound,
            bid_penalty=self.args.bid_penalty,
            action_window=self.args.action_window,
            window_bidding=self.args.window_bidding,
            window_penalty=self.args.window_penalty,
            enemy_destroy_reward=self.args.enemy_destroy_reward,
            hit_penalty=self.args.hit_penalty,
            life_loss_penalty=self.args.life_loss_penalty,
            raw_score_scale=self.args.raw_score_scale,
            fire_while_hot_penalty=self.args.fire_while_hot_penalty,
            max_steps=self.args.max_steps,
            hud=self.args.hud,
            single_agent_mode=False,
            allow_variable_enemies=self.args.allow_variable_enemies,
            allow_sideward_fire=self.args.allow_sideward_fire,
        )
        self.envs = AssaultEnv(env_config, num_envs=self.args.num_envs, device=self.device, seed=self.args.seed)

        self.obs_dim = self.envs.per_agent_obs_dim
        self.agent = AssaultSharedAgent(
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

        print("🚀 Assault PPO Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Observation dim: {self.obs_dim}")
        if self.args.window_bidding:
            print("   Actions per agent: 3 (direction + bid + window)")
        else:
            print("   Actions per agent: 2 (direction + bid)")

    def save_model(self, path: str | None = None):
        if path is None:
            path = "assault_agent.pt"
        torch.save(self.agent.state_dict(), path)

    def cleanup(self):
        if self.envs is not None:
            self.envs.close()
