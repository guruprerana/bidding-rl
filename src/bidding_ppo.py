# PPO implementation for BiddingGridworld environments
# Based on https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
# Adapted for multi-agent bidding with shared actor-critic networks

import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import wandb

from bidding_gridworld import BiddingGridworld, MovingTargetBiddingGridworld
from ppo_utils import layer_init, compute_gae, ppo_update_step, compute_explained_variance


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "bidding-rl"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    # Environment specific arguments
    grid_size: int = 10
    """size of the gridworld"""
    num_agents: int = 2
    """number of agents in the environment"""
    bid_upper_bound: int = 10
    """maximum bid value"""
    bid_penalty: float = 0.1
    """penalty multiplier for bids"""
    target_reward: float = 10.0
    """reward for reaching target"""
    max_steps: int = 100
    """maximum steps per episode"""
    action_window: int = 1
    """number of steps a winning agent controls the action"""
    distance_reward_scale: float = 0.0
    """reward scaling for distance improvements"""
    target_expiry_steps: Optional[int] = None
    """maximum steps allowed before target expiry penalty"""
    target_expiry_penalty: float = 5.0
    """penalty for not reaching target within expiry_steps"""
    moving_targets: bool = False
    """whether to use moving targets variant"""
    direction_change_prob: float = 0.1
    """probability of target direction change (for moving targets)"""
    target_move_interval: int = 1
    """steps between target movements (for moving targets)"""

    # Algorithm specific arguments
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class BiddingEnvWrapper(gym.Wrapper):
    """
    Wrapper to convert the multi-agent BiddingGridworld to a format compatible with vectorized training.

    Key features:
    - Reorders target observations so each agent's target appears first
    - Flattens multi-agent actions/rewards into single vectors for parallel processing
    - Each agent gets the same base observation but with reordered target information

    Observation structure from BiddingGridworld:
    [agent_pos (2), target_positions (2*num_agents), target_reached (num_agents), target_counters (num_agents), window_steps_remaining (1)]

    For agent pursuing target_idx, we reorder to:
    [agent_pos (2), target_idx_pos (2), other_targets (2*(num_agents-1)),
     target_idx_reached (1), other_reached (num_agents-1), target_idx_counter (1), other_counters (num_agents-1), window_steps_remaining (1)]
    """

    def __init__(self, env, num_agents):
        super().__init__(env)
        self.num_agents = num_agents

        # Original observation dimension from BiddingGridworld (no change in size)
        base_obs_dim = env.observation_space.shape[0]

        # Observation space remains the same size, just reordered
        # Shape is (num_agents, base_obs_dim) since we return stacked observations
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_agents, base_obs_dim),
            dtype=np.float32
        )

        # Action space: each agent outputs (direction, bid)
        # We'll use a MultiDiscrete space: [dir_agent0, bid_agent0, dir_agent1, bid_agent1, ...]
        self.action_space = gym.spaces.MultiDiscrete(
            [4, env.bid_upper_bound + 1] * num_agents
        )

    def _reorder_observation(self, base_obs, target_index):
        """
        Reorder base observation so the agent's target appears first.

        Args:
            base_obs: Base observation from environment (numpy array)
            target_index: Which target this agent is pursuing (0 to num_agents-1)

        Returns:
            Reordered observation with this agent's target information first
        """
        return reorder_observation_for_agent(base_obs, target_index, self.num_agents)

    def reset(self, **kwargs):
        """Reset environment and return reordered observations for all agents."""
        base_obs, info = self.env.reset(**kwargs)

        # Create reordered observation for each agent (each pursuing different target)
        obs_list = []
        for agent_idx in range(self.num_agents):
            reordered_obs = self._reorder_observation(base_obs, agent_idx)
            obs_list.append(reordered_obs)

        # Stack all agent observations into single array for vectorized processing
        # Shape: (num_agents, obs_dim)
        stacked_obs = np.stack(obs_list, axis=0)

        return stacked_obs, info

    def step(self, action):
        """
        Execute step with actions from all agents.

        Args:
            action: Flattened action array [dir_agent0, bid_agent0, dir_agent1, bid_agent1, ...]

        Returns:
            observations: Reordered observations for all agents
            rewards: Rewards for all agents
            terminated: Episode termination flag
            truncated: Episode truncation flag
            info: Additional information
        """
        # Convert flattened action array to BiddingGridworld format
        env_action = {}
        for agent_idx in range(self.num_agents):
            # Extract direction and bid for this agent
            direction = int(action[agent_idx * 2])
            bid = int(action[agent_idx * 2 + 1])

            env_action[f"agent_{agent_idx}"] = {
                "direction": direction,
                "bid": bid
            }

        # Execute step in underlying environment
        base_obs, rewards_dict, terminated, truncated, info = self.env.step(env_action)

        # Create reordered observations for all agents
        obs_list = []
        for agent_idx in range(self.num_agents):
            reordered_obs = self._reorder_observation(base_obs, agent_idx)
            obs_list.append(reordered_obs)

        stacked_obs = np.stack(obs_list, axis=0)

        # Extract rewards for all agents in order
        rewards_array = np.array([
            rewards_dict[f"agent_{i}"] for i in range(self.num_agents)
        ], dtype=np.float32)

        return stacked_obs, rewards_array, terminated, truncated, info


def make_env(args, idx, run_name):
    """Create a single BiddingGridworld environment with wrapper."""
    def thunk():
        # Create base environment (either static or moving targets)
        if args.moving_targets:
            env = MovingTargetBiddingGridworld(
                grid_size=args.grid_size,
                num_agents=args.num_agents,
                bid_upper_bound=args.bid_upper_bound,
                bid_penalty=args.bid_penalty,
                target_reward=args.target_reward,
                max_steps=args.max_steps,
                action_window=args.action_window,
                distance_reward_scale=args.distance_reward_scale,
                target_expiry_steps=args.target_expiry_steps,
                target_expiry_penalty=args.target_expiry_penalty,
                direction_change_prob=args.direction_change_prob,
                target_move_interval=args.target_move_interval
            )
        else:
            env = BiddingGridworld(
                grid_size=args.grid_size,
                num_agents=args.num_agents,
                bid_upper_bound=args.bid_upper_bound,
                bid_penalty=args.bid_penalty,
                target_reward=args.target_reward,
                max_steps=args.max_steps,
                action_window=args.action_window,
                distance_reward_scale=args.distance_reward_scale,
                target_expiry_steps=args.target_expiry_steps,
                target_expiry_penalty=args.target_expiry_penalty
            )

        # Wrap with our custom wrapper
        env = BiddingEnvWrapper(env, args.num_agents)

        return env

    return thunk


def reorder_observation_for_agent(base_obs: np.ndarray, target_index: int, num_agents: int) -> np.ndarray:
    """
    Reorder base observation so the specified agent's target appears first.

    This is a utility function that can be used anywhere we need to prepare
    observations for individual agents (training, evaluation, deployment, etc.).

    Args:
        base_obs: Base observation from BiddingGridworld environment
                 Structure: [agent_pos(2), all_targets(2*N), all_reached(N), all_counters(N), window_steps_remaining(1)]
        target_index: Which target this agent is pursuing (0 to num_agents-1)
        num_agents: Total number of agents/targets

    Returns:
        Reordered observation with the agent's target information first
        Structure: [agent_pos(2), target_idx(2), others(2*(N-1)),
                   target_idx_reached(1), others_reached(N-1),
                   target_idx_counter(1), others_counters(N-1),
                   window_steps_remaining(1)]
    """
    # Parse observation sections
    agent_pos = base_obs[0:2]  # First 2 values: agent position

    # Target positions: next 2*num_agents values
    target_pos_start = 2
    target_pos_end = 2 + 2 * num_agents
    all_target_positions = base_obs[target_pos_start:target_pos_end].reshape(num_agents, 2)

    # Target reached flags: next num_agents values
    reached_start = target_pos_end
    reached_end = reached_start + num_agents
    all_target_reached = base_obs[reached_start:reached_end]

    # Target step counters: next num_agents values
    counter_start = reached_end
    counter_end = counter_start + num_agents
    all_target_counters = base_obs[counter_start:counter_end]

    # Window steps remaining: last value
    window_steps_remaining = base_obs[-1]

    # Reorder each section so target_index comes first
    # Create index permutation: [target_index, other indices...]
    indices = [target_index] + [i for i in range(num_agents) if i != target_index]

    # Reorder target positions
    reordered_positions = all_target_positions[indices].flatten()

    # Reorder target reached flags
    reordered_reached = all_target_reached[indices]

    # Reorder target counters
    reordered_counters = all_target_counters[indices]

    # Reconstruct observation
    reordered_obs = np.concatenate([
        agent_pos,
        reordered_positions,
        reordered_reached,
        reordered_counters,
        [window_steps_remaining]  # Add window steps remaining at the end
    ])

    return reordered_obs


class SharedAgent(nn.Module):
    """
    Shared actor-critic network used by all agents.

    All agents use the same network parameters but receive different observations
    (targets reordered so each agent's target appears first). Each agent runs
    inference separately through this shared network.
    """

    def __init__(self, obs_dim, num_actions_per_agent):
        """
        Initialize shared actor-critic network.

        Args:
            obs_dim: Dimension of observation (targets reordered per agent)
            num_actions_per_agent: Number of action components per agent (2: direction + bid)
        """
        super().__init__()

        # Shared critic network: outputs single value estimate
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Shared actor network: outputs action logits
        # For bidding gridworld: outputs logits for direction (4 actions) and bid (bid_upper_bound+1 actions)
        # We'll use two separate heads for direction and bid
        self.actor_shared = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )

        # Separate heads for direction and bid actions
        self.direction_head = layer_init(nn.Linear(64, 4), std=0.01)  # 4 directions
        self.bid_head = None  # Will be set based on bid_upper_bound

    def set_bid_head(self, bid_upper_bound):
        """Set the bid head based on bid upper bound."""
        self.bid_head = layer_init(nn.Linear(64, bid_upper_bound + 1), std=0.01)
        # Move to same device as the rest of the model
        self.bid_head = self.bid_head.to(next(self.parameters()).device)

    def get_value(self, x):
        """
        Get value estimate for given observation.

        Args:
            x: Observation tensor (can be batched)

        Returns:
            Value estimate
        """
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        Get action and value for given observation.

        This is the core inference function. Each agent calls this separately
        with their reordered observation (their target appears first in the obs).

        Args:
            x: Observation tensor (can be batched)
            action: If provided, compute log prob for this action. Otherwise sample new action.
                   Action should be tensor of shape (..., 2) where last dim is [direction, bid]

        Returns:
            action: Sampled or provided action [direction, bid]
            log_prob: Log probability of the action
            entropy: Entropy of the action distribution
            value: Value estimate
        """
        # Get shared features
        shared_features = self.actor_shared(x)

        # Get logits for direction and bid separately
        direction_logits = self.direction_head(shared_features)
        bid_logits = self.bid_head(shared_features)

        # Create categorical distributions
        direction_dist = Categorical(logits=direction_logits)
        bid_dist = Categorical(logits=bid_logits)

        # Sample or use provided action
        if action is None:
            # Sample new actions
            direction = direction_dist.sample()
            bid = bid_dist.sample()
            action = torch.stack([direction, bid], dim=-1)
        else:
            # Use provided action
            direction = action[..., 0]
            bid = action[..., 1]

        # Compute log probabilities (sum of independent log probs)
        direction_log_prob = direction_dist.log_prob(direction)
        bid_log_prob = bid_dist.log_prob(bid)
        total_log_prob = direction_log_prob + bid_log_prob

        # Compute entropy (sum of independent entropies)
        entropy = direction_dist.entropy() + bid_dist.entropy()

        # Get value estimate
        value = self.critic(x)

        return action, total_log_prob, entropy, value


class ManualVectorEnv:
    """
    Manual vectorization for multi-agent environments.

    Handles environments that return multi-dimensional rewards and observations.
    """

    def __init__(self, env_fns):
        """
        Initialize manual vectorized environment.

        Args:
            env_fns: List of functions that create environments
        """
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space

    def reset(self, **kwargs):
        """Reset all environments."""
        observations = []
        infos = []
        for env in self.envs:
            obs, info = env.reset(**kwargs)
            observations.append(obs)
            infos.append(info)
        return np.array(observations), infos

    def step(self, actions):
        """Step all environments."""
        observations = []
        rewards = []
        terminations = []
        truncations = []
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)
            infos.append(info)

        return (
            np.array(observations),
            np.array(rewards),
            np.array(terminations),
            np.array(truncations),
            infos
        )

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


class PPOTrainer:
    """PPO Trainer for multi-agent bidding gridworld with shared networks."""

    def __init__(self, args: Args, callbacks: Optional[Dict] = None):
        """
        Initialize PPO Trainer.

        Args:
            args: Training configuration arguments
            callbacks: Optional dict of callback functions:
                - on_iteration_end(trainer, iteration, global_step): Called after each iteration
                - on_training_end(trainer, global_step): Called when training completes
        """
        self.args = args
        self.callbacks = callbacks or {}

        # Compute derived parameters
        self.args.batch_size = int(args.num_envs * args.num_steps * args.num_agents)
        self.args.minibatch_size = int(self.args.batch_size // args.num_minibatches)
        self.args.num_iterations = args.total_timesteps // (args.num_envs * args.num_steps * args.num_agents)

        self.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

        # Initialize wandb
        if self.args.track:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                config=vars(args),
                name=self.run_name,
                save_code=True,
            )

        # Seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # These will be initialized in setup()
        self.envs = None
        self.agent = None
        self.optimizer = None
        self.obs_dim = None

    def setup(self):
        """Setup environments, agent, and optimizer."""
        # Environment setup - use manual vectorization for multi-agent support
        self.envs = ManualVectorEnv(
            [make_env(self.args, i, self.run_name) for i in range(self.args.num_envs)]
        )

        # Create shared agent
        # Observation space is (num_agents, obs_dim), so we need shape[1] for per-agent obs dim
        self.obs_dim = self.envs.single_observation_space.shape[1]
        self.agent = SharedAgent(self.obs_dim, num_actions_per_agent=2).to(self.device)
        self.agent.set_bid_head(self.args.bid_upper_bound)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)

        print(f"🚀 PPO Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Observation dim: {self.obs_dim}")
        print(f"   Batch size: {self.args.batch_size}")
        print(f"   Num iterations: {self.args.num_iterations}")
        print(f"   Run name: {self.run_name}")

    def train(self):
        """Run the main training loop."""
        if self.envs is None:
            raise RuntimeError("Must call setup() before train()")

        # Storage setup
        # Observation space is (num_agents, obs_dim), so we need (num_steps, num_envs, num_agents, obs_dim)
        obs = torch.zeros((self.args.num_steps, self.args.num_envs, self.args.num_agents, self.obs_dim)).to(self.device)
        actions = torch.zeros((self.args.num_steps, self.args.num_envs, self.args.num_agents, 2)).to(self.device)
        logprobs = torch.zeros((self.args.num_steps, self.args.num_envs, self.args.num_agents)).to(self.device)
        rewards = torch.zeros((self.args.num_steps, self.args.num_envs, self.args.num_agents)).to(self.device)
        dones = torch.zeros((self.args.num_steps, self.args.num_envs, self.args.num_agents)).to(self.device)
        values = torch.zeros((self.args.num_steps, self.args.num_envs, self.args.num_agents)).to(self.device)

        # Start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.envs.reset(seed=self.args.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros((self.args.num_envs, self.args.num_agents)).to(self.device)

        # Track bidding-specific statistics (aggregated across rollout)
        episode_agent_wins = {f"agent_{i}": 0 for i in range(self.args.num_agents)}
        episode_bid_values = []

        print(f"\n{'='*60}")
        print(f"Starting training for {self.args.total_timesteps} timesteps")
        print(f"{'='*60}\n")

        for iteration in range(1, self.args.num_iterations + 1):
            # Annealing the learning rate
            if self.args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
                lrnow = frac * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            # Rollout phase
            for step in range(0, self.args.num_steps):
                global_step += self.args.num_envs * self.args.num_agents
                obs[step] = next_obs
                dones[step] = next_done

                # Get actions from shared network
                with torch.no_grad():
                    flat_obs = next_obs.reshape(-1, self.obs_dim)
                    action, logprob, _, value = self.agent.get_action_and_value(flat_obs)
                    action = action.reshape(self.args.num_envs, self.args.num_agents, 2)
                    logprob = logprob.reshape(self.args.num_envs, self.args.num_agents)
                    value = value.reshape(self.args.num_envs, self.args.num_agents)
                    values[step] = value

                actions[step] = action
                logprobs[step] = logprob

                # Execute actions in environment
                flat_action = action.reshape(self.args.num_envs, -1).cpu().numpy()
                next_obs, reward, terminations, truncations, infos = self.envs.step(flat_action)

                # Combine termination signals and broadcast to all agents
                # terminations and truncations are shape (num_envs,)
                # We need shape (num_envs, num_agents) since all agents share the same environment
                next_done_scalar = np.logical_or(terminations, truncations)  # shape: (num_envs,)
                next_done = np.broadcast_to(next_done_scalar[:, np.newaxis], (self.args.num_envs, self.args.num_agents))  # shape: (num_envs, num_agents)

                rewards[step] = torch.tensor(reward).to(self.device)
                next_obs = torch.Tensor(next_obs).to(self.device)
                next_done = torch.Tensor(next_done).to(self.device)

                # Extract bidding information for logging
                for env_idx in range(self.args.num_envs):
                    if isinstance(infos, dict) and "winning_agent" in infos:
                        winning_agent = infos["winning_agent"][env_idx] if hasattr(infos["winning_agent"], "__getitem__") else infos["winning_agent"]
                        if winning_agent is not None and winning_agent >= 0:
                            episode_agent_wins[f"agent_{winning_agent}"] += 1

                    if isinstance(infos, dict) and "bids" in infos:
                        bids = infos["bids"][env_idx] if hasattr(infos["bids"], "__getitem__") else infos["bids"]
                        if isinstance(bids, dict):
                            for agent_key, bid_value in bids.items():
                                episode_bid_values.append(bid_value)


            # Bootstrap value and compute advantages using shared utility
            with torch.no_grad():
                flat_next_obs = next_obs.reshape(-1, self.obs_dim)
                next_value = self.agent.get_value(flat_next_obs).reshape(self.args.num_envs, self.args.num_agents)

                advantages, returns = compute_gae(
                    rewards, values, dones, next_value, next_done,
                    self.args.gamma, self.args.gae_lambda
                )

            # Flatten for training
            # obs shape: (num_steps, num_envs, num_agents, obs_dim) -> (batch_size, obs_dim)
            b_obs = obs.reshape(-1, self.obs_dim)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1, 2)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimize policy and value network using shared utility
            b_inds = np.arange(self.args.batch_size)
            clipfracs = []

            for epoch in range(self.args.update_epochs):
                np.random.shuffle(b_inds)

                for start in range(0, self.args.batch_size, self.args.minibatch_size):
                    end = start + self.args.minibatch_size
                    mb_inds = b_inds[start:end]

                    # Use shared PPO update step
                    metrics = ppo_update_step(
                        self.agent,
                        self.optimizer,
                        b_obs[mb_inds],
                        b_actions[mb_inds],
                        b_logprobs[mb_inds],
                        b_advantages[mb_inds],
                        b_returns[mb_inds],
                        b_values[mb_inds],
                        self.args.clip_coef,
                        self.args.ent_coef,
                        self.args.vf_coef,
                        self.args.max_grad_norm,
                        self.args.norm_adv,
                        self.args.clip_vloss
                    )

                    clipfracs.append(metrics["clipfrac"])

                if self.args.target_kl is not None and metrics["approx_kl"] > self.args.target_kl:
                    break

            # Store final metrics for logging
            v_loss = metrics["v_loss"]
            pg_loss = metrics["pg_loss"]
            entropy_loss = metrics["entropy_loss"]
            old_approx_kl = metrics["old_approx_kl"]
            approx_kl = metrics["approx_kl"]

            # Compute explained variance using shared utility
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            explained_var = compute_explained_variance(y_pred, y_true)

            # Log training metrics
            sps = int(global_step / (time.time() - start_time))
            print(f"Iteration {iteration}/{self.args.num_iterations} - SPS: {sps} - Value Loss: {v_loss:.4f} - Policy Loss: {pg_loss:.4f}")

            if self.args.track:
                log_dict = {
                    "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "losses/value_loss": v_loss,
                    "losses/policy_loss": pg_loss,
                    "losses/entropy": entropy_loss,
                    "losses/old_approx_kl": old_approx_kl,
                    "losses/approx_kl": approx_kl,
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "charts/SPS": sps,
                    "charts/iteration": iteration,
                    "global_step": global_step,
                }

                # Add aggregate reward/value statistics
                log_dict["rewards/avg_step_reward"] = rewards.mean().item()
                log_dict["rewards/max_step_reward"] = rewards.max().item()
                log_dict["rewards/min_step_reward"] = rewards.min().item()
                log_dict["values/mean"] = values.mean().item()
                log_dict["values/std"] = values.std().item()
                log_dict["values/max"] = values.max().item()
                log_dict["values/min"] = values.min().item()
                log_dict["advantages/mean"] = advantages.mean().item()
                log_dict["advantages/std"] = advantages.std().item()

                # Add bidding statistics (aggregated over this rollout)
                if episode_bid_values:
                    log_dict["bidding/avg_bid_value"] = np.mean(episode_bid_values)
                    log_dict["bidding/max_bid_value"] = np.max(episode_bid_values)
                    log_dict["bidding/min_bid_value"] = np.min(episode_bid_values)

                # Add per-agent win rates (over this rollout)
                total_wins = sum(episode_agent_wins.values())
                if total_wins > 0:
                    for agent_idx in range(self.args.num_agents):
                        agent_key = f"agent_{agent_idx}"
                        win_rate = episode_agent_wins[agent_key] / total_wins
                        log_dict[f"agents/{agent_key}_win_rate"] = win_rate

                wandb.log(log_dict, step=global_step)

                # Reset rollout statistics for next iteration
                episode_bid_values = []
                episode_agent_wins = {f"agent_{i}": 0 for i in range(self.args.num_agents)}

            # Call iteration callback if provided
            if "on_iteration_end" in self.callbacks:
                self.callbacks["on_iteration_end"](self, iteration, global_step)

        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"{'='*60}\n")

        # Call training end callback if provided
        if "on_training_end" in self.callbacks:
            self.callbacks["on_training_end"](self, global_step)

    def save_model(self):
        """Save the trained model."""
        model_path = f"models/{self.run_name}"
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.agent.state_dict(), f"{model_path}/agent.pt")
        print(f"✅ Model saved to {model_path}/agent.pt")

        if self.args.track:
            wandb.save(f"{model_path}/agent.pt")

    def cleanup(self):
        """Cleanup resources."""
        if self.envs is not None:
            self.envs.close()

        if self.args.track:
            wandb.finish()

        print("🧹 Cleanup completed")


if __name__ == "__main__":
    # Create trainer and run
    args = Args()
    trainer = PPOTrainer(args)

    try:
        trainer.setup()
        trainer.train()
        trainer.save_model()
    finally:
        trainer.cleanup()
