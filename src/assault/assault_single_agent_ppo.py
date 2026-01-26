"""
Single-agent PPO trainer for OCAtari Assault.
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
from ppo_trainer_base import SingleAgentPPOTrainerBase


@dataclass
class AssaultSingleAgentArgs:
    exp_name: str = "assault_single_agent_ppo"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "bidding-rl"
    wandb_entity: str = None

    # Environment
    num_agents: int = 3
    max_enemies: int = 3
    enemy_destroy_reward: float = 1.0
    hit_penalty: float = 1.0
    life_loss_penalty: float = 10.0
    health_loss_penalty: float = 0.1
    max_steps: int = 10000
    hud: bool = True
    allow_variable_enemies: bool = True

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


class AssaultSingleAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_space_n: int,
        actor_hidden_sizes: Tuple[int, ...],
        critic_hidden_sizes: Tuple[int, ...],
    ):
        super().__init__()
        actor_layers = []
        actor_in = obs_dim
        for hidden in actor_hidden_sizes:
            actor_layers.append(layer_init(nn.Linear(actor_in, hidden)))
            actor_layers.append(nn.ELU())
            actor_in = hidden
        actor_layers.append(layer_init(nn.Linear(actor_in, action_space_n), std=0.01))
        self.actor = nn.Sequential(*actor_layers)

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
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(x)
        return action, log_prob, entropy, value


class AssaultSingleAgentPPOTrainer(SingleAgentPPOTrainerBase):
    def __init__(self, args: AssaultSingleAgentArgs, callbacks=None):
        super().__init__(args, callbacks=callbacks)
        self.obs_dim = None

    def setup(self):
        env_config = AssaultConfig(
            num_agents=self.args.num_agents,
            max_enemies=self.args.max_enemies,
            enemy_destroy_reward=self.args.enemy_destroy_reward,
            hit_penalty=self.args.hit_penalty,
            life_loss_penalty=self.args.life_loss_penalty,
            health_loss_penalty=self.args.health_loss_penalty,
            max_steps=self.args.max_steps,
            hud=self.args.hud,
            single_agent_mode=True,
            allow_variable_enemies=self.args.allow_variable_enemies,
        )
        self.envs = AssaultEnv(env_config, num_envs=self.args.num_envs, device=self.device, seed=self.args.seed)

        self.obs_dim = self.envs.obs_shape[1]
        self.agent = AssaultSingleAgent(
            obs_dim=self.obs_dim,
            action_space_n=self.envs.action_space_n,
            actor_hidden_sizes=self.args.actor_hidden_sizes,
            critic_hidden_sizes=self.args.critic_hidden_sizes,
        ).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)

        self.args.batch_size = int(self.args.num_envs * self.args.num_steps)
        self.args.minibatch_size = int(self.args.batch_size // self.args.num_minibatches)

        print("🚀 Assault Single-Agent PPO initialized")
        print(f"   Device: {self.device}")
        print(f"   Observation dim: {self.obs_dim}")
        print(f"   Actions: {self.envs.action_space_n}")

    def save_model(self, path: str | None = None):
        if path is None:
            path = "assault_single_agent.pt"
        torch.save(self.agent.state_dict(), path)

    def cleanup(self):
        if self.envs is not None:
            self.envs.close()
