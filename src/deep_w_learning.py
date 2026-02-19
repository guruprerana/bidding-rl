"""Deep W-Learning (DWN) trainer for multi-objective RL.

Based on arXiv:2408.01188. Each objective has its own Q-network (action selection)
and W-network (priority). At each step, the objective with the highest W-value
selects the action via its Q-network.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from ppo_utils import build_mlp, format_duration


class ReplayBuffer:
    """GPU-resident circular replay buffer with pre-allocated tensors."""

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        num_objectives: int,
        num_envs: int,
        device: torch.device,
    ):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        self.pos = 0
        self.full = False

        self.obs = torch.zeros((buffer_size, obs_dim), device=device)
        self.actions = torch.zeros((buffer_size,), device=device, dtype=torch.int64)
        self.rewards = torch.zeros((buffer_size, num_objectives), device=device)
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
        """Add a batch of transitions. obs: (num_envs, obs_dim), etc."""
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


class QNetwork(nn.Module):
    """Q-network: obs -> Q-values per action for one objective."""

    def __init__(self, obs_dim: int, num_actions: int, hidden_sizes: tuple[int, ...] = (128, 128)):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes, num_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class WNetwork(nn.Module):
    """W-network: obs -> scalar priority for one objective."""

    def __init__(self, obs_dim: int, hidden_sizes: tuple[int, ...] = (128, 128)):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


@dataclass
class DWNArgs:
    """Configuration for Deep W-Learning trainer."""
    exp_name: str = "dwn"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "bidding-rl"
    wandb_entity: Optional[str] = None

    # DWN core
    num_objectives: int = 3
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

    # Epsilon schedules (per-episode decay as in paper)
    q_epsilon_start: float = 0.99
    q_epsilon_min: float = 0.001
    q_epsilon_decay: float = 0.99
    w_epsilon_start: float = 0.99
    w_epsilon_min: float = 0.001
    w_epsilon_decay: float = 0.99

    # Environment
    num_envs: int = 4
    max_steps: int = 100

    # Logging
    log_frequency: int = 1000


class DWNTrainer:
    """Deep W-Learning trainer for multi-objective RL."""

    def __init__(self, args: DWNArgs, callbacks: Optional[dict] = None):
        self.args = args
        self.callbacks = callbacks or {}
        self.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

        if getattr(self.args, "track", False):
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
        self.envs = None
        self.obs_dim = None
        self.num_actions = None

        # Networks (populated by _setup_networks)
        self.q_networks: list[QNetwork] = []
        self.q_targets: list[QNetwork] = []
        self.w_networks: list[WNetwork] = []
        self.q_optimizers: list[optim.Optimizer] = []
        self.w_optimizers: list[optim.Optimizer] = []
        self.replay_buffer: Optional[ReplayBuffer] = None

        # Epsilon state
        self.q_epsilon = args.q_epsilon_start
        self.w_epsilon = args.w_epsilon_start

    def setup(self):
        """Subclasses must create self.envs, set self.obs_dim and self.num_actions, then call _setup_networks()."""
        raise NotImplementedError

    def _setup_networks(self):
        """Create Q/W networks, optimizers, and replay buffer after obs_dim and num_actions are set."""
        args = self.args
        N = args.num_objectives

        self.q_networks = []
        self.q_targets = []
        self.w_networks = []
        self.q_optimizers = []
        self.w_optimizers = []

        for _ in range(N):
            q = QNetwork(self.obs_dim, self.num_actions, args.q_hidden_sizes).to(self.device)
            q_target = QNetwork(self.obs_dim, self.num_actions, args.q_hidden_sizes).to(self.device)
            q_target.load_state_dict(q.state_dict())
            q_target.requires_grad_(False)

            w = WNetwork(self.obs_dim, args.w_hidden_sizes).to(self.device)

            self.q_networks.append(q)
            self.q_targets.append(q_target)
            self.w_networks.append(w)
            self.q_optimizers.append(optim.Adam(q.parameters(), lr=args.q_learning_rate))
            self.w_optimizers.append(optim.Adam(w.parameters(), lr=args.w_learning_rate))

        self.replay_buffer = ReplayBuffer(
            args.buffer_size, self.obs_dim, N, args.num_envs, self.device
        )

        total_params = sum(
            sum(p.numel() for p in net.parameters())
            for net in self.q_networks + self.w_networks
        )
        print(f"DWN Trainer initialized")
        print(f"  Device: {self.device}")
        print(f"  Objectives: {N}, Actions: {self.num_actions}, Obs dim: {self.obs_dim}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Run name: {self.run_name}")

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Select actions using DWN mechanism.

        Args:
            obs: (num_envs, obs_dim)

        Returns:
            actions: (num_envs,) int64
        """
        num_envs = obs.shape[0]
        N = self.args.num_objectives

        with torch.no_grad():
            # Each Q_i proposes a_i = argmax_a Q_i(s, a) with Q-epsilon-greedy
            proposed_actions = torch.zeros((num_envs, N), device=self.device, dtype=torch.int64)
            for i, q_net in enumerate(self.q_networks):
                q_values = q_net(obs)  # (num_envs, num_actions)
                greedy_actions = q_values.argmax(dim=1)
                # Q-epsilon-greedy
                rand_mask = torch.rand(num_envs, device=self.device) < self.q_epsilon
                random_actions = torch.randint(0, self.num_actions, (num_envs,), device=self.device)
                proposed_actions[:, i] = torch.where(rand_mask, random_actions, greedy_actions)

            # Each W_i computes priority W_i(s)
            w_values = torch.zeros((num_envs, N), device=self.device)
            for i, w_net in enumerate(self.w_networks):
                w_values[:, i] = w_net(obs)

            # W-epsilon-greedy: winner j = argmax_i W_i(s) with exploration
            greedy_winners = w_values.argmax(dim=1)
            rand_mask = torch.rand(num_envs, device=self.device) < self.w_epsilon
            random_winners = torch.randint(0, N, (num_envs,), device=self.device)
            winners = torch.where(rand_mask, random_winners, greedy_winners)

            # Execute a_j (the action proposed by the winning objective)
            actions = proposed_actions.gather(1, winners.unsqueeze(1)).squeeze(1)

        return actions, winners

    def _update_q_networks(self):
        """Update all Q-networks with standard DQN loss per objective."""
        args = self.args
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(args.batch_size)

        q_losses = []
        for i in range(args.num_objectives):
            # Target: r_i + gamma * max_a Q_target_i(s', a) * (1 - done)
            with torch.no_grad():
                next_q = self.q_targets[i](next_obs)
                max_next_q = next_q.max(dim=1).values
                target = rewards[:, i] + args.gamma * max_next_q * (1.0 - dones)

            # Current Q_i(s, a_executed)
            current_q = self.q_networks[i](obs).gather(1, actions.unsqueeze(1)).squeeze(1)

            loss = nn.functional.mse_loss(current_q, target)
            self.q_optimizers[i].zero_grad()
            loss.backward()
            self.q_optimizers[i].step()
            q_losses.append(loss.item())

        return q_losses

    def _update_w_networks(self):
        """Update W-networks with DWN priority targets.

        W_target_i = Q_i(s, a_executed) - [r_i + gamma * max_a Q_target_i(s', a)]
        This is the negative TD-error: high W means the executed action was worse than
        expected for objective i, so it needs higher priority.
        """
        args = self.args
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(args.batch_size)

        w_losses = []
        for i in range(args.num_objectives):
            with torch.no_grad():
                current_q = self.q_networks[i](obs).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q = self.q_targets[i](next_obs)
                max_next_q = next_q.max(dim=1).values
                td_target = rewards[:, i] + args.gamma * max_next_q * (1.0 - dones)
                w_target = current_q - td_target

            w_pred = self.w_networks[i](obs)
            loss = nn.functional.mse_loss(w_pred, w_target)
            self.w_optimizers[i].zero_grad()
            loss.backward()
            self.w_optimizers[i].step()
            w_losses.append(loss.item())

        return w_losses

    def _update_target_networks(self):
        """Hard or soft copy Q-networks to targets."""
        tau = self.args.tau
        for i in range(self.args.num_objectives):
            if tau == 1.0:
                self.q_targets[i].load_state_dict(self.q_networks[i].state_dict())
            else:
                for param, target_param in zip(
                    self.q_networks[i].parameters(), self.q_targets[i].parameters()
                ):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def _decay_epsilon(self):
        """Exponential epsilon decay (called per episode, matching paper)."""
        args = self.args
        self.q_epsilon = max(args.q_epsilon_min, self.q_epsilon * args.q_epsilon_decay)
        self.w_epsilon = max(args.w_epsilon_min, self.w_epsilon * args.w_epsilon_decay)

    def train(self):
        if self.envs is None:
            raise RuntimeError("Must call setup() before train()")

        args = self.args
        global_step = 0
        start_time = time.time()
        num_updates = 0

        # Tracking
        episode_returns = torch.zeros(args.num_envs, device=self.device)
        episode_lengths = torch.zeros(args.num_envs, device=self.device, dtype=torch.int32)
        winner_counts = torch.zeros(args.num_objectives, device=self.device)

        next_obs, _ = self.envs.reset(seed=args.seed)

        while global_step < args.total_timesteps:
            actions, winners = self.select_action(next_obs)

            obs, rewards_scalar, terminations, truncations, infos = self.envs.step(actions)

            # Get per-objective rewards
            if "per_objective_rewards" in infos:
                per_obj_rewards = infos["per_objective_rewards"]
            else:
                # Fallback: broadcast scalar reward to all objectives
                per_obj_rewards = rewards_scalar.unsqueeze(-1).expand(-1, args.num_objectives)

            dones = (terminations | truncations).to(torch.float32)

            # Store transitions
            self.replay_buffer.add(next_obs, actions, per_obj_rewards, obs, dones)

            # Track episode stats
            episode_returns += rewards_scalar
            episode_lengths += 1
            winner_counts.scatter_add_(0, winners, torch.ones(args.num_envs, device=self.device))

            # Handle episode endings
            done_mask = dones > 0.5
            if done_mask.any():
                for env_idx in done_mask.nonzero(as_tuple=True)[0]:
                    ep_return = episode_returns[env_idx].item()
                    ep_length = episode_lengths[env_idx].item()

                    if global_step % args.log_frequency < args.num_envs:
                        print(
                            f"  step={global_step}, ep_return={ep_return:.2f}, "
                            f"ep_length={int(ep_length)}, q_eps={self.q_epsilon:.4f}, w_eps={self.w_epsilon:.4f}"
                        )

                    if getattr(args, "track", False):
                        wandb.log({
                            "charts/episodic_return": ep_return,
                            "charts/episodic_length": int(ep_length),
                            "charts/q_epsilon": self.q_epsilon,
                            "charts/w_epsilon": self.w_epsilon,
                        }, step=global_step)

                episode_returns = torch.where(done_mask, torch.zeros_like(episode_returns), episode_returns)
                episode_lengths = torch.where(done_mask, torch.zeros_like(episode_lengths), episode_lengths)
                self._decay_epsilon()

            next_obs = obs
            global_step += args.num_envs

            # Training updates
            if global_step >= args.learning_starts and global_step % args.train_frequency < args.num_envs:
                q_losses = self._update_q_networks()
                num_updates += 1

                # W-network update with delay
                w_losses = None
                if global_step >= args.learning_starts + args.w_train_delay:
                    w_losses = self._update_w_networks()

                # Target network update
                if num_updates % args.target_network_freq == 0:
                    self._update_target_networks()

                # Logging
                if global_step % args.log_frequency < args.num_envs:
                    elapsed = time.time() - start_time
                    sps = int(global_step / elapsed)
                    remaining_steps = args.total_timesteps - global_step
                    eta = format_duration(remaining_steps / max(sps, 1))

                    print(
                        f"Step {global_step}/{args.total_timesteps} - SPS: {sps} - "
                        f"Q losses: [{', '.join(f'{l:.4f}' for l in q_losses)}] - "
                        f"ETA: {eta}"
                    )

                    if getattr(args, "track", False):
                        log_dict = {
                            "charts/SPS": sps,
                            "charts/global_step": global_step,
                        }
                        for i, ql in enumerate(q_losses):
                            log_dict[f"losses/q_loss_{i}"] = ql
                        if w_losses is not None:
                            for i, wl in enumerate(w_losses):
                                log_dict[f"losses/w_loss_{i}"] = wl

                        # W-value stats and winner distribution
                        with torch.no_grad():
                            for i in range(args.num_objectives):
                                w_val = self.w_networks[i](next_obs).mean().item()
                                log_dict[f"charts/w_value_{i}"] = w_val

                        total_wins = winner_counts.sum().item()
                        if total_wins > 0:
                            for i in range(args.num_objectives):
                                log_dict[f"charts/winner_pct_{i}"] = winner_counts[i].item() / total_wins

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

        for i in range(self.args.num_objectives):
            torch.save(self.q_networks[i].state_dict(), f"{path}/q_network_{i}.pt")
            torch.save(self.w_networks[i].state_dict(), f"{path}/w_network_{i}.pt")
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        for i in range(self.args.num_objectives):
            self.q_networks[i].load_state_dict(
                torch.load(f"{path}/q_network_{i}.pt", map_location=self.device)
            )
            self.q_targets[i].load_state_dict(self.q_networks[i].state_dict())
            self.w_networks[i].load_state_dict(
                torch.load(f"{path}/w_network_{i}.pt", map_location=self.device)
            )
        print(f"Model loaded from {path}")

    def cleanup(self):
        if self.envs is not None:
            self.envs.close()
        if getattr(self.args, "track", False):
            wandb.finish()
        print("Cleanup completed")
