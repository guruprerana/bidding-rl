"""Gridworld Deep W-Learning trainer with shared Q/W networks.

Analogous to bidding_ppo.py but using DWN (arXiv:2408.01188) instead of PPO.
Unlike the base DWNTrainer (which has N separate Q/W networks), this uses
**one shared Q-network and one shared W-network** for all objectives. Each
agent's per-agent observation already encodes which objective it represents
(its own target is reordered to first position), so the shared networks
process all objectives identically.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from deep_w_learning import QNetwork, WNetwork
from ppo_utils import format_duration
from bidding_gridworld.bidding_gridworld_torch import (
    BiddingGridworld,
    BiddingGridworldConfig,
)


class ReplayBuffer:
    """GPU-resident circular replay buffer with scalar rewards.

    Unlike the base DWN buffer (which stores per-objective reward vectors),
    this stores scalar rewards per transition since we flatten agents into
    individual transitions.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        device: torch.device,
    ):
        self.buffer_size = buffer_size
        self.device = device
        self.pos = 0
        self.full = False

        self.obs = torch.zeros((buffer_size, obs_dim), device=device)
        self.actions = torch.zeros((buffer_size,), device=device, dtype=torch.int64)
        self.rewards = torch.zeros((buffer_size,), device=device)
        self.next_obs = torch.zeros((buffer_size, obs_dim), device=device)
        self.dones = torch.zeros((buffer_size,), device=device)

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ):
        """Add a batch of transitions.

        All inputs are 1-D batches: obs (B, obs_dim), actions (B,),
        rewards (B,), next_obs (B, obs_dim), dones (B,).
        """
        batch_size = obs.shape[0]
        end = self.pos + batch_size

        if end <= self.buffer_size:
            self.obs[self.pos:end] = obs
            self.actions[self.pos:end] = actions
            self.rewards[self.pos:end] = rewards
            self.next_obs[self.pos:end] = next_obs
            self.dones[self.pos:end] = dones
        else:
            first = self.buffer_size - self.pos
            self.obs[self.pos:] = obs[:first]
            self.actions[self.pos:] = actions[:first]
            self.rewards[self.pos:] = rewards[:first]
            self.next_obs[self.pos:] = next_obs[:first]
            self.dones[self.pos:] = dones[:first]
            remainder = batch_size - first
            self.obs[:remainder] = obs[first:]
            self.actions[:remainder] = actions[first:]
            self.rewards[:remainder] = rewards[first:]
            self.next_obs[:remainder] = next_obs[first:]
            self.dones[:remainder] = dones[first:]

        self.pos = end % self.buffer_size
        if end >= self.buffer_size:
            self.full = True

    def sample(self, batch_size: int):
        max_idx = self.buffer_size if self.full else self.pos
        idx = torch.randint(0, max_idx, (batch_size,), device=self.device)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    @property
    def size(self) -> int:
        return self.buffer_size if self.full else self.pos


@dataclass
class GridworldDWNArgs:
    """Configuration for Gridworld Deep W-Learning trainer."""

    exp_name: str = "gridworld_dwn"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "bidding-rl"
    wandb_entity: Optional[str] = None

    # Environment
    grid_size: int = 10
    num_targets: int = 3
    target_reward: float = 10.0
    max_steps: int = 100
    distance_reward_scale: float = 0.0
    target_expiry_steps: Optional[int] = None
    target_expiry_penalty: float = 5.0
    moving_targets: bool = False
    direction_change_prob: float = 0.1
    target_move_interval: int = 1
    visible_targets: Optional[int] = None

    # DWN core
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 1_000_000
    total_timesteps: int = 500_000
    learning_starts: int = 10_000
    train_frequency: int = 4
    w_train_delay: int = 1000
    target_network_freq: int = 1000
    tau: float = 1.0

    # Network architecture
    q_hidden_sizes: tuple[int, ...] = (128, 128)
    w_hidden_sizes: tuple[int, ...] = (128, 128)

    # Learning rates
    q_learning_rate: float = 0.01
    w_learning_rate: float = 0.01

    # Epsilon schedules (per-episode decay)
    q_epsilon_start: float = 0.99
    q_epsilon_min: float = 0.001
    q_epsilon_decay: float = 0.99
    w_epsilon_start: float = 0.99
    w_epsilon_min: float = 0.001
    w_epsilon_decay: float = 0.99

    # Parallelism
    num_envs: int = 4

    # Logging
    log_frequency: int = 1000


class GridworldDWNTrainer:
    """Deep W-Learning trainer for bidding gridworld with shared networks.

    Instead of N separate Q/W networks (one per objective), uses one shared
    Q-network and one shared W-network. Each agent's per-agent observation
    encodes which objective it represents (own target reordered to first
    position), so the shared networks process all objectives identically.
    """

    def __init__(self, args: GridworldDWNArgs, callbacks: Optional[dict] = None):
        self.args = args
        self.callbacks = callbacks or {}
        self.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

        if args.track:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                config=vars(args),
                name=self.run_name,
                save_code=True,
            )

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        self.envs: Optional[BiddingGridworld] = None
        self.per_agent_obs_dim: Optional[int] = None
        self.num_actions = 4  # 4 directions

        # Networks (populated by setup)
        self.q_network: Optional[QNetwork] = None
        self.q_target: Optional[QNetwork] = None
        self.w_network: Optional[WNetwork] = None
        self.q_optimizer: Optional[optim.Optimizer] = None
        self.w_optimizer: Optional[optim.Optimizer] = None
        self.replay_buffer: Optional[ReplayBuffer] = None

        # Epsilon state
        self.q_epsilon = args.q_epsilon_start
        self.w_epsilon = args.w_epsilon_start

    def setup(self):
        """Create environment, shared networks, replay buffer, and optimizers."""
        args = self.args

        env_config = BiddingGridworldConfig(
            grid_size=args.grid_size,
            num_agents=args.num_targets,
            bid_upper_bound=1,
            bid_penalty=0.0,
            target_reward=args.target_reward,
            max_steps=args.max_steps,
            action_window=1,
            distance_reward_scale=args.distance_reward_scale,
            target_expiry_steps=args.target_expiry_steps,
            target_expiry_penalty=args.target_expiry_penalty,
            moving_targets=args.moving_targets,
            direction_change_prob=args.direction_change_prob,
            target_move_interval=args.target_move_interval,
            window_bidding=False,
            window_penalty=0.0,
            visible_targets=args.visible_targets,
            single_agent_mode=False,
        )
        self.envs = BiddingGridworld(
            env_config,
            num_envs=args.num_envs,
            device=self.device,
            seed=args.seed,
        )

        self.per_agent_obs_dim = self.envs.per_agent_obs_dim

        # Shared Q-network and target
        self.q_network = QNetwork(
            self.per_agent_obs_dim, self.num_actions, args.q_hidden_sizes
        ).to(self.device)
        self.q_target = QNetwork(
            self.per_agent_obs_dim, self.num_actions, args.q_hidden_sizes
        ).to(self.device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        self.q_target.requires_grad_(False)

        # Shared W-network
        self.w_network = WNetwork(
            self.per_agent_obs_dim, args.w_hidden_sizes
        ).to(self.device)

        # Optimizers (2 total, not 2*N)
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=args.q_learning_rate)
        self.w_optimizer = optim.Adam(self.w_network.parameters(), lr=args.w_learning_rate)

        # Replay buffer with scalar rewards
        self.replay_buffer = ReplayBuffer(
            args.buffer_size, self.per_agent_obs_dim, self.device
        )

        total_params = sum(
            p.numel() for p in self.q_network.parameters()
        ) + sum(
            p.numel() for p in self.w_network.parameters()
        )
        print(f"Gridworld DWN Trainer initialized")
        print(f"  Device: {self.device}")
        print(f"  Targets (objectives): {args.num_targets}, Actions: {self.num_actions}")
        print(f"  Per-agent obs dim: {self.per_agent_obs_dim}")
        print(f"  Shared networks — total parameters: {total_params:,}")
        print(f"  Run name: {self.run_name}")

    def select_action(self, obs: torch.Tensor):
        """Select actions using shared DWN mechanism.

        Args:
            obs: (num_envs, num_agents, per_agent_obs_dim)

        Returns:
            env_actions: (num_envs, num_agents, 2) — [direction, bid]
            winners: (num_envs,) — winning agent index
            executed_directions: (num_envs,) — direction chosen by winner
        """
        num_envs = obs.shape[0]
        num_agents = obs.shape[1]

        with torch.no_grad():
            # Flatten: (num_envs * num_agents, obs_dim)
            flat_obs = obs.reshape(num_envs * num_agents, -1)

            # Shared Q -> proposed directions (Q-epsilon-greedy)
            q_values = self.q_network(flat_obs)  # (E*A, 4)
            q_values = q_values.reshape(num_envs, num_agents, self.num_actions)
            greedy_actions = q_values.argmax(dim=2)  # (E, A)

            rand_mask = torch.rand(num_envs, num_agents, device=self.device) < self.q_epsilon
            random_actions = torch.randint(
                0, self.num_actions, (num_envs, num_agents), device=self.device
            )
            proposed_directions = torch.where(rand_mask, random_actions, greedy_actions)

            # Shared W -> priorities (W-epsilon-greedy)
            w_values = self.w_network(flat_obs)  # (E*A,)
            w_values = w_values.reshape(num_envs, num_agents)
            greedy_winners = w_values.argmax(dim=1)  # (E,)

            rand_mask_w = torch.rand(num_envs, device=self.device) < self.w_epsilon
            random_winners = torch.randint(0, num_agents, (num_envs,), device=self.device)
            winners = torch.where(rand_mask_w, random_winners, greedy_winners)

            # Build env actions: (num_envs, num_agents, 2) = [direction, bid]
            # Winner gets bid=1, others get bid=0
            bids = torch.zeros(num_envs, num_agents, device=self.device, dtype=torch.int64)
            bids.scatter_(1, winners.unsqueeze(1), 1)

            env_actions = torch.stack([proposed_directions, bids], dim=-1)  # (E, A, 2)

            # Executed direction is the winner's proposed direction
            executed_directions = proposed_directions.gather(
                1, winners.unsqueeze(1)
            ).squeeze(1)

        return env_actions, winners, executed_directions

    def _update_q(self):
        """Update shared Q-network with standard DQN loss."""
        args = self.args
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(args.batch_size)

        with torch.no_grad():
            next_q = self.q_target(next_obs)
            max_next_q = next_q.max(dim=1).values
            target = rewards + args.gamma * max_next_q * (1.0 - dones)

        current_q = self.q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.functional.mse_loss(current_q, target)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        return loss.item()

    def _update_w(self):
        """Update shared W-network with DWN priority targets.

        W_target = Q(s, a_executed) - [r + gamma * max_a Q_target(s', a)]
        """
        args = self.args
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(args.batch_size)

        with torch.no_grad():
            current_q = self.q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q = self.q_target(next_obs)
            max_next_q = next_q.max(dim=1).values
            td_target = rewards + args.gamma * max_next_q * (1.0 - dones)
            w_target = current_q - td_target

        w_pred = self.w_network(obs)
        loss = nn.functional.mse_loss(w_pred, w_target)

        self.w_optimizer.zero_grad()
        loss.backward()
        self.w_optimizer.step()

        return loss.item()

    def _update_target_network(self):
        """Hard or soft copy Q-network to target."""
        tau = self.args.tau
        if tau == 1.0:
            self.q_target.load_state_dict(self.q_network.state_dict())
        else:
            for param, target_param in zip(
                self.q_network.parameters(), self.q_target.parameters()
            ):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def _decay_epsilon(self):
        """Exponential epsilon decay (called per episode)."""
        args = self.args
        self.q_epsilon = max(args.q_epsilon_min, self.q_epsilon * args.q_epsilon_decay)
        self.w_epsilon = max(args.w_epsilon_min, self.w_epsilon * args.w_epsilon_decay)

    def train(self):
        if self.envs is None:
            raise RuntimeError("Must call setup() before train()")

        args = self.args
        num_agents = args.num_targets
        global_step = 0
        start_time = time.time()
        num_updates = 0

        # Per-env episode tracking (sum over all agents)
        episode_returns = torch.zeros(args.num_envs, device=self.device)
        episode_lengths = torch.zeros(args.num_envs, device=self.device, dtype=torch.int32)
        winner_counts = torch.zeros(num_agents, device=self.device)

        next_obs, _ = self.envs.reset(seed=args.seed)
        # next_obs: (num_envs, num_agents, per_agent_obs_dim)

        while global_step < args.total_timesteps:
            env_actions, winners, executed_directions = self.select_action(next_obs)

            obs_new, rewards, terminations, truncations, infos = self.envs.step(env_actions)
            # rewards: (num_envs, num_agents) — per-agent scalar rewards
            # obs_new: (num_envs, num_agents, per_agent_obs_dim)

            dones = (terminations | truncations).to(torch.float32)  # (num_envs,)

            # Flatten and store transitions: num_envs * num_agents entries per step
            flat_obs = next_obs.reshape(args.num_envs * num_agents, -1)
            flat_next_obs = obs_new.reshape(args.num_envs * num_agents, -1)

            # All agents saw the same executed direction
            flat_actions = executed_directions.unsqueeze(1).expand(
                -1, num_agents
            ).reshape(args.num_envs * num_agents)

            flat_rewards = rewards.reshape(args.num_envs * num_agents)
            flat_dones = dones.unsqueeze(1).expand(
                -1, num_agents
            ).reshape(args.num_envs * num_agents)

            self.replay_buffer.add(flat_obs, flat_actions, flat_rewards, flat_next_obs, flat_dones)

            # Track episode stats
            episode_returns += rewards.sum(dim=1)
            episode_lengths += 1
            winner_counts.scatter_add_(
                0, winners, torch.ones(args.num_envs, device=self.device)
            )

            # Handle episode endings
            done_mask = dones > 0.5
            if done_mask.any():
                for env_idx in done_mask.nonzero(as_tuple=True)[0]:
                    ep_return = episode_returns[env_idx].item()
                    ep_length = episode_lengths[env_idx].item()

                    if global_step % args.log_frequency < args.num_envs:
                        print(
                            f"  step={global_step}, ep_return={ep_return:.2f}, "
                            f"ep_length={int(ep_length)}, "
                            f"q_eps={self.q_epsilon:.4f}, w_eps={self.w_epsilon:.4f}"
                        )

                    if args.track:
                        wandb.log({
                            "charts/episodic_return": ep_return,
                            "charts/episodic_length": int(ep_length),
                            "charts/q_epsilon": self.q_epsilon,
                            "charts/w_epsilon": self.w_epsilon,
                        }, step=global_step)

                episode_returns = torch.where(
                    done_mask, torch.zeros_like(episode_returns), episode_returns
                )
                episode_lengths = torch.where(
                    done_mask, torch.zeros_like(episode_lengths), episode_lengths
                )
                self._decay_epsilon()

            next_obs = obs_new
            global_step += args.num_envs

            # Training updates
            if (
                global_step >= args.learning_starts
                and global_step % args.train_frequency < args.num_envs
            ):
                q_loss = self._update_q()
                num_updates += 1

                # W-network update with delay
                w_loss = None
                if global_step >= args.learning_starts + args.w_train_delay:
                    w_loss = self._update_w()

                # Target network update
                if num_updates % args.target_network_freq == 0:
                    self._update_target_network()

                # Logging
                if global_step % args.log_frequency < args.num_envs:
                    elapsed = time.time() - start_time
                    sps = int(global_step / elapsed)
                    remaining_steps = args.total_timesteps - global_step
                    eta = format_duration(remaining_steps / max(sps, 1))

                    w_str = f", W loss: {w_loss:.4f}" if w_loss is not None else ""
                    print(
                        f"Step {global_step}/{args.total_timesteps} - SPS: {sps} - "
                        f"Q loss: {q_loss:.4f}{w_str} - ETA: {eta}"
                    )

                    if args.track:
                        log_dict = {
                            "charts/SPS": sps,
                            "charts/global_step": global_step,
                            "losses/q_loss": q_loss,
                        }
                        if w_loss is not None:
                            log_dict["losses/w_loss"] = w_loss

                        # W-value stats
                        with torch.no_grad():
                            flat = next_obs.reshape(args.num_envs * num_agents, -1)
                            w_vals = self.w_network(flat)
                            log_dict["charts/w_value_mean"] = w_vals.mean().item()

                        # Winner distribution
                        total_wins = winner_counts.sum().item()
                        if total_wins > 0:
                            for i in range(num_agents):
                                log_dict[f"charts/winner_pct_{i}"] = (
                                    winner_counts[i].item() / total_wins
                                )

                        wandb.log(log_dict, step=global_step)

            # Callback
            if self.callbacks.get("on_step"):
                self.callbacks["on_step"](self, global_step)

        if self.callbacks.get("on_training_end"):
            self.callbacks["on_training_end"](self, global_step)

        elapsed = time.time() - start_time
        print(f"\nTraining complete: {global_step} steps in {format_duration(elapsed)}")

    def save_model(self, path: Optional[str] = None):
        if path is None:
            path = f"models/{self.run_name}"
        os.makedirs(path, exist_ok=True)

        torch.save(self.q_network.state_dict(), f"{path}/q.pt")
        torch.save(self.q_target.state_dict(), f"{path}/q_target.pt")
        torch.save(self.w_network.state_dict(), f"{path}/w.pt")
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        self.q_network.load_state_dict(
            torch.load(f"{path}/q.pt", map_location=self.device)
        )
        self.q_target.load_state_dict(
            torch.load(f"{path}/q_target.pt", map_location=self.device)
        )
        self.w_network.load_state_dict(
            torch.load(f"{path}/w.pt", map_location=self.device)
        )
        print(f"Model loaded from {path}")

    def cleanup(self):
        if self.envs is not None:
            self.envs.close()
        if self.args.track:
            wandb.finish()
        print("Cleanup completed")
