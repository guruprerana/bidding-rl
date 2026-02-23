"""Assault Deep W-Learning trainer with shared Q/W networks.

Analogous to bidding_gridworld/dwn.py but for the OCAtari Assault environment.
One shared Q-network proposes Atari actions per agent; one shared W-network
selects the winner agent. The winner bids 1, all others bid 0.

DWN paper: arXiv:2408.01188.
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
from bidding_gridworld.dwn import ReplayBuffer
from assault.assault_torch import AssaultConfig, AssaultEnv


@dataclass
class AssaultDWNArgs:
    """Configuration for Assault Deep W-Learning trainer."""

    exp_name: str = "assault_dwn"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "bidding-rl"
    wandb_entity: Optional[str] = None

    # Environment
    num_agents: int = 3
    max_enemies: int = 3
    enemy_destroy_reward: float = 1.0
    hit_penalty: float = 1.0
    life_loss_penalty: float = 10.0
    raw_score_scale: float = 0.5
    fire_while_hot_penalty: float = 0.0
    allow_sideward_fire: bool = True
    allow_variable_enemies: bool = True
    hud: bool = True
    max_steps: int = 10000

    # DWN core
    gamma: float = 0.99
    batch_size: int = 256
    buffer_size: int = 500_000
    total_timesteps: int = 10_000_000
    learning_starts: int = 10_000
    train_frequency: int = 8
    w_train_delay: int = 50_000
    target_network_freq: int = 1000
    tau: float = 1.0

    # Network architecture
    q_hidden_sizes: tuple = (256, 256)
    w_hidden_sizes: tuple = (128, 128)

    # Learning rates
    q_learning_rate: float = 1e-4
    w_learning_rate: float = 1e-4

    # Epsilon schedules (per-episode exponential decay)
    q_epsilon_start: float = 0.99
    q_epsilon_min: float = 0.01
    q_epsilon_decay: float = 0.995
    w_epsilon_start: float = 0.99
    w_epsilon_min: float = 0.01
    w_epsilon_decay: float = 0.995

    # Parallelism
    num_envs: int = 8

    # Logging
    log_frequency: int = 1000


class AssaultDWNTrainer:
    """Deep W-Learning trainer for OCAtari Assault with shared Q/W networks.

    Instead of N separate Q/W networks, uses one shared Q-network and one
    shared W-network.  Each agent's per-agent observation already encodes
    which enemy slot it represents (the agent's target enemy is listed first),
    so the shared networks process all agents identically.
    """

    def __init__(self, args: AssaultDWNArgs, callbacks: Optional[dict] = None):
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

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )

        self.envs: Optional[AssaultEnv] = None
        self.per_agent_obs_dim: Optional[int] = None
        self.num_actions: Optional[int] = None

        # Networks (populated by setup)
        self.q_network: Optional[QNetwork] = None
        self.q_target: Optional[QNetwork] = None
        self.w_network: Optional[WNetwork] = None
        self.q_optimizer: Optional[optim.Optimizer] = None
        self.w_optimizer: Optional[optim.Optimizer] = None
        self.replay_buffer: Optional[ReplayBuffer] = None
        # W-buffer stores only non-winning agent transitions (per DWN paper)
        self.w_replay_buffer: Optional[ReplayBuffer] = None

        # Epsilon state
        self.q_epsilon = args.q_epsilon_start
        self.w_epsilon = args.w_epsilon_start

    def setup(self):
        """Create environment, shared networks, replay buffers, and optimizers."""
        args = self.args

        env_config = AssaultConfig(
            num_agents=args.num_agents,
            max_enemies=args.max_enemies,
            bid_upper_bound=1,
            bid_penalty=0.0,
            action_window=1,
            window_bidding=False,
            window_penalty=0.0,
            enemy_destroy_reward=args.enemy_destroy_reward,
            hit_penalty=args.hit_penalty,
            life_loss_penalty=args.life_loss_penalty,
            raw_score_scale=args.raw_score_scale,
            fire_while_hot_penalty=args.fire_while_hot_penalty,
            max_steps=args.max_steps,
            hud=args.hud,
            single_agent_mode=False,
            allow_variable_enemies=args.allow_variable_enemies,
            allow_sideward_fire=args.allow_sideward_fire,
            bidding_mechanism="all_pay",
        )
        self.envs = AssaultEnv(
            env_config,
            num_envs=args.num_envs,
            device=self.device,
            seed=args.seed,
        )

        self.per_agent_obs_dim = self.envs.per_agent_obs_dim
        self.num_actions = self.envs.action_space_n

        # Shared Q-network and its target
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

        self.q_optimizer = optim.Adam(
            self.q_network.parameters(), lr=args.q_learning_rate
        )
        self.w_optimizer = optim.Adam(
            self.w_network.parameters(), lr=args.w_learning_rate
        )

        # Q-buffer: all agent transitions (used for Q-updates)
        self.replay_buffer = ReplayBuffer(
            args.buffer_size, self.per_agent_obs_dim, self.device
        )
        # W-buffer: only non-winning agent transitions (used for W-updates)
        self.w_replay_buffer = ReplayBuffer(
            args.buffer_size, self.per_agent_obs_dim, self.device
        )

        total_params = sum(
            p.numel() for p in self.q_network.parameters()
        ) + sum(
            p.numel() for p in self.w_network.parameters()
        )
        print("Assault DWN Trainer initialized")
        print(f"  Device: {self.device}")
        print(f"  Agents: {args.num_agents}, Actions: {self.num_actions}")
        print(f"  Per-agent obs dim: {self.per_agent_obs_dim}")
        print(f"  Shared networks — total parameters: {total_params:,}")
        print(f"  Run name: {self.run_name}")

    def select_action(self, obs: torch.Tensor):
        """Select actions using shared DWN mechanism.

        Args:
            obs: (num_envs, num_agents, per_agent_obs_dim)

        Returns:
            env_actions: (num_envs, num_agents, 2) — [direction, bid]
            winners: (num_envs,) — winning agent index per env
            executed_directions: (num_envs,) — direction chosen by each winner
        """
        num_envs = obs.shape[0]
        num_agents = obs.shape[1]

        with torch.no_grad():
            # Flatten: (num_envs * num_agents, obs_dim)
            flat_obs = obs.reshape(num_envs * num_agents, -1)

            # Shared Q -> proposed directions per agent (Q-epsilon-greedy)
            q_values = self.q_network(flat_obs)                          # (E*A, num_actions)
            q_values = q_values.reshape(num_envs, num_agents, self.num_actions)
            greedy_actions = q_values.argmax(dim=2)                      # (E, A)

            rand_mask = torch.rand(num_envs, num_agents, device=self.device) < self.q_epsilon
            random_actions = torch.randint(
                0, self.num_actions, (num_envs, num_agents), device=self.device
            )
            proposed_directions = torch.where(rand_mask, random_actions, greedy_actions)

            # Shared W -> winner per env (W-epsilon-greedy)
            w_values = self.w_network(flat_obs)                          # (E*A,)
            w_values = w_values.reshape(num_envs, num_agents)
            greedy_winners = w_values.argmax(dim=1)                      # (E,)

            rand_mask_w = torch.rand(num_envs, device=self.device) < self.w_epsilon
            random_winners = torch.randint(0, num_agents, (num_envs,), device=self.device)
            winners = torch.where(rand_mask_w, random_winners, greedy_winners)

            # Build env_actions: winner bid=1, losers bid=0
            bids = torch.zeros(num_envs, num_agents, device=self.device, dtype=torch.int64)
            bids.scatter_(1, winners.unsqueeze(1), 1)
            env_actions = torch.stack([proposed_directions, bids], dim=-1)  # (E, A, 2)

            # The executed direction is the winner's proposed direction
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

        Only samples from loser transitions (per DWN paper: W-networks are
        updated only for objectives that did not get to execute their action).
        Returns None if W-buffer is too small.
        """
        args = self.args
        if self.w_replay_buffer.size < args.batch_size:
            return None
        obs, actions, rewards, next_obs, dones = self.w_replay_buffer.sample(args.batch_size)

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
        """Hard or soft copy Q-network weights to target."""
        tau = self.args.tau
        if tau == 1.0:
            self.q_target.load_state_dict(self.q_network.state_dict())
        else:
            for param, target_param in zip(
                self.q_network.parameters(), self.q_target.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1.0 - tau) * target_param.data
                )

    def _decay_epsilon(self):
        """Exponential epsilon decay — called once per episode completion."""
        args = self.args
        self.q_epsilon = max(args.q_epsilon_min, self.q_epsilon * args.q_epsilon_decay)
        self.w_epsilon = max(args.w_epsilon_min, self.w_epsilon * args.w_epsilon_decay)

    def train(self):
        if self.envs is None:
            raise RuntimeError("Must call setup() before train()")

        args = self.args
        num_agents = args.num_agents
        global_step = 0
        start_time = time.time()
        num_updates = 0

        # Per-env episode tracking
        episode_returns = torch.zeros(args.num_envs, device=self.device)
        episode_lengths = torch.zeros(args.num_envs, device=self.device, dtype=torch.int32)
        winner_counts = torch.zeros(num_agents, device=self.device)

        next_obs, _ = self.envs.reset(seed=args.seed)
        # next_obs: (num_envs, num_agents, per_agent_obs_dim)

        while global_step < args.total_timesteps:
            env_actions, winners, executed_directions = self.select_action(next_obs)

            obs_new, rewards, terminations, truncations, infos = self.envs.step(env_actions)
            # rewards: (num_envs, num_agents)
            # obs_new: (num_envs, num_agents, per_agent_obs_dim)

            dones = (terminations | truncations).to(torch.float32)  # (num_envs,)

            # Flatten and store transitions: num_envs * num_agents entries per step
            flat_obs = next_obs.reshape(args.num_envs * num_agents, -1)
            flat_next_obs = obs_new.reshape(args.num_envs * num_agents, -1)

            # All agents receive the winner's executed direction in the Q-buffer
            flat_actions = executed_directions.unsqueeze(1).expand(
                -1, num_agents
            ).reshape(args.num_envs * num_agents)

            flat_rewards = rewards.reshape(args.num_envs * num_agents)
            flat_dones = dones.unsqueeze(1).expand(
                -1, num_agents
            ).reshape(args.num_envs * num_agents)

            self.replay_buffer.add(
                flat_obs, flat_actions, flat_rewards, flat_next_obs, flat_dones
            )

            # W-buffer: only non-winning (loser) agent transitions
            loser_mask = torch.ones(
                args.num_envs, num_agents, dtype=torch.bool, device=self.device
            )
            loser_mask.scatter_(1, winners.unsqueeze(1), False)
            loser_flat = loser_mask.reshape(-1)
            if loser_flat.any():
                self.w_replay_buffer.add(
                    flat_obs[loser_flat],
                    flat_actions[loser_flat],
                    flat_rewards[loser_flat],
                    flat_next_obs[loser_flat],
                    flat_dones[loser_flat],
                )

            # Track episode stats
            episode_returns += rewards.sum(dim=1)
            episode_lengths += 1
            winner_counts.scatter_add_(
                0, winners, torch.ones(args.num_envs, device=self.device)
            )

            # Handle episode endings
            done_mask = dones > 0.5
            if done_mask.any():
                # AssaultEnv.step() auto-resets done envs but does not reset
                # window_agent. Call partial_reset to get clean initial obs
                # (window_agent=-1, is_in_control=0 for all agents).
                reset_obs = self.envs.partial_reset(done_mask)
                obs_new = torch.where(done_mask.view(-1, 1, 1), reset_obs, obs_new)

                for env_idx in done_mask.nonzero(as_tuple=True)[0]:
                    ep_return = episode_returns[env_idx].item()
                    ep_length = episode_lengths[env_idx].item()
                    # infos["score"] holds the cumulative score just before
                    # the env auto-reset, i.e. the final episode score.
                    ep_score = (
                        infos["score"][env_idx].item()
                        if "score" in infos and infos["score"] is not None
                        else 0.0
                    )

                    if global_step % args.log_frequency < args.num_envs:
                        print(
                            f"  step={global_step}, ep_return={ep_return:.2f}, "
                            f"ep_length={int(ep_length)}, ep_score={ep_score:.0f}, "
                            f"q_eps={self.q_epsilon:.4f}, w_eps={self.w_epsilon:.4f}"
                        )

                    if args.track:
                        wandb.log(
                            {
                                "charts/episodic_return": ep_return,
                                "charts/episodic_length": int(ep_length),
                                "charts/episodic_score": ep_score,
                                "charts/q_epsilon": self.q_epsilon,
                                "charts/w_epsilon": self.w_epsilon,
                            },
                            step=global_step,
                        )

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

                # W-network update starts after an additional delay
                w_loss = None
                if global_step >= args.learning_starts + args.w_train_delay:
                    w_loss = self._update_w()

                # Target network sync
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

            # Step callback
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
