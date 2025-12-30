# Single-agent PPO implementation for BiddingGridworld in single_agent_mode
# Simpler than multi-agent version - just learns to navigate and collect all targets

import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import wandb

from bidding_gridworld import BiddingGridworld, MovingTargetBiddingGridworld
from torch_batched_env import TorchBatchedBiddingGridworld, TorchBatchedConfig
from ppo_utils import layer_init, compute_gae, ppo_update_step, compute_explained_variance


@dataclass
class SingleAgentArgs:
    exp_name: str = "single_agent_ppo"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    use_torch_batched_env: bool = False
    """if toggled, use the GPU-native batched environment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "bidding-rl"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    # Environment specific arguments
    grid_size: int = 10
    """size of the gridworld"""
    num_targets: int = 3
    """number of targets to collect"""
    target_reward: float = 100.0
    """reward for reaching target"""
    max_steps: int = 100
    """maximum steps per episode"""
    distance_reward_scale: float = 0.1
    """reward scaling for distance improvements"""
    target_expiry_steps: Optional[int] = None
    """maximum steps allowed before target expiry penalty"""
    target_expiry_penalty: float = 5.0
    """penalty for not reaching target within expiry_steps"""
    reward_decay_factor: float = 0.0
    """reward decay based on relative target count (0.0 = no decay)"""
    moving_targets: bool = False
    """whether to use moving targets variant"""
    direction_change_prob: float = 0.1
    """probability of target direction change (for moving targets)"""
    target_move_interval: int = 1
    """steps between target movements (for moving targets)"""

    # Network architecture
    actor_hidden_sizes: Tuple[int, ...] = (128, 128, 128)
    """hidden layer sizes for the actor network"""
    critic_hidden_sizes: Tuple[int, ...] = (256, 256, 256)
    """hidden layer sizes for the critic network"""

    # Algorithm specific arguments
    num_iterations: int = 1000
    """the number of policy iterations to run"""
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
    total_timesteps: int = 0
    """total timesteps of the experiments (computed in runtime)"""


class SingleAgent(nn.Module):
    """
    Simple actor-critic network for single-agent navigation.

    Only outputs direction actions (no bidding).
    """

    def __init__(self, obs_dim, actor_hidden_sizes=None, critic_hidden_sizes=None):
        """
        Initialize single-agent actor-critic network.

        Args:
            obs_dim: Dimension of observation
        """
        super().__init__()

        actor_sizes = list(actor_hidden_sizes) if actor_hidden_sizes is not None else [128, 128, 128]
        critic_sizes = list(critic_hidden_sizes) if critic_hidden_sizes is not None else [256, 256, 256]

        # Critic network: outputs single value estimate
        critic_layers = []
        critic_in_dim = obs_dim
        for hidden_size in critic_sizes:
            critic_layers.append(layer_init(nn.Linear(critic_in_dim, hidden_size)))
            critic_layers.append(nn.ELU())
            critic_in_dim = hidden_size
        critic_layers.append(layer_init(nn.Linear(critic_in_dim, 1), std=1.0))
        self.critic = nn.Sequential(*critic_layers)

        # Actor network: outputs logits for direction (4 actions)
        actor_layers = []
        actor_in_dim = obs_dim
        for hidden_size in actor_sizes:
            actor_layers.append(layer_init(nn.Linear(actor_in_dim, hidden_size)))
            actor_layers.append(nn.ELU())
            actor_in_dim = hidden_size
        actor_layers.append(layer_init(nn.Linear(actor_in_dim, 4), std=0.01))  # 4 directions
        self.actor = nn.Sequential(*actor_layers)

    def get_value(self, x):
        """Get value estimate for given observation."""
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        Get action and value for given observation.

        Args:
            x: Observation tensor (can be batched)
            action: If provided, compute log prob for this action. Otherwise sample new action.

        Returns:
            action: Sampled or provided action (direction only)
            log_prob: Log probability of the action
            entropy: Entropy of the action distribution
            value: Value estimate
        """
        # Get action logits
        logits = self.actor(x)

        # Create categorical distribution
        probs = Categorical(logits=logits)

        # Sample or use provided action
        if action is None:
            action = probs.sample()

        # Compute log probability and entropy
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()

        # Get value estimate
        value = self.critic(x)

        return action, log_prob, entropy, value


def make_env(args, idx, run_name):
    """Create a single BiddingGridworld environment in single-agent mode."""
    def thunk():
        # Create base environment (either static or moving targets)
        # Note: num_agents parameter now determines number of targets
        if args.moving_targets:
            env = MovingTargetBiddingGridworld(
                grid_size=args.grid_size,
                num_agents=args.num_targets,  # num_agents = number of targets in single-agent mode
                target_reward=args.target_reward,
                max_steps=args.max_steps,
                distance_reward_scale=args.distance_reward_scale,
                target_expiry_steps=args.target_expiry_steps,
                target_expiry_penalty=args.target_expiry_penalty,
                reward_decay_factor=args.reward_decay_factor,
                direction_change_prob=args.direction_change_prob,
                target_move_interval=args.target_move_interval,
                single_agent_mode=True  # Enable single-agent mode
            )
        else:
            env = BiddingGridworld(
                grid_size=args.grid_size,
                num_agents=args.num_targets,  # num_agents = number of targets in single-agent mode
                target_reward=args.target_reward,
                max_steps=args.max_steps,
                distance_reward_scale=args.distance_reward_scale,
                target_expiry_steps=args.target_expiry_steps,
                target_expiry_penalty=args.target_expiry_penalty,
                reward_decay_factor=args.reward_decay_factor,
                single_agent_mode=True  # Enable single-agent mode
            )

        # Wrap with RecordEpisodeStatistics for automatic episode tracking
        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env

    return thunk


class SingleAgentPPOTrainer:
    """PPO Trainer for single-agent gridworld navigation."""

    def __init__(self, args: SingleAgentArgs, callbacks: Optional[Dict] = None):
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
        self.args.batch_size = int(args.num_envs * args.num_steps)
        self.args.minibatch_size = int(self.args.batch_size // args.num_minibatches)
        self.args.total_timesteps = self.args.num_iterations * self.args.num_envs * self.args.num_steps

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
        self.use_torch_env = False

    def setup(self):
        """Setup environments, agent, and optimizer."""
        # Environment setup
        self.use_torch_env = bool(self.args.use_torch_batched_env)
        if self.use_torch_env:
            env_config = TorchBatchedConfig(
                grid_size=self.args.grid_size,
                num_agents=self.args.num_targets,
                bid_upper_bound=0,
                bid_penalty=0.0,
                target_reward=self.args.target_reward,
                max_steps=self.args.max_steps,
                action_window=1,
                distance_reward_scale=self.args.distance_reward_scale,
                target_expiry_steps=self.args.target_expiry_steps,
                target_expiry_penalty=self.args.target_expiry_penalty,
                moving_targets=self.args.moving_targets,
                direction_change_prob=self.args.direction_change_prob,
                target_move_interval=self.args.target_move_interval,
                window_bidding=False,
                window_penalty=0.0,
                visible_targets=None,
                single_agent_mode=True,
                reward_decay_factor=self.args.reward_decay_factor,
            )
            self.envs = TorchBatchedBiddingGridworld(
                env_config,
                num_envs=self.args.num_envs,
                device=self.device,
                seed=self.args.seed,
            )
        else:
            # Environment setup - use SyncVectorEnv for single-agent
            self.envs = gym.vector.SyncVectorEnv(
                [make_env(self.args, i, self.run_name) for i in range(self.args.num_envs)]
            )

        # Create agent
        if self.use_torch_env:
            self.obs_dim = self.envs.obs_dim
        else:
            self.obs_dim = np.array(self.envs.single_observation_space.shape).prod()
        self.agent = SingleAgent(
            self.obs_dim,
            actor_hidden_sizes=self.args.actor_hidden_sizes,
            critic_hidden_sizes=self.args.critic_hidden_sizes,
        ).to(self.device)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)

        # Calculate expected observation dimension components for single-agent mode
        # Base: 2 (agent pos) + 2*num_targets (target pos) + num_targets (reached flags) +
        #       num_targets (step counters) + 1 (window steps) + num_targets (relative counts)
        expected_dim = 2 + 2*self.args.num_targets + self.args.num_targets + self.args.num_targets + 1 + self.args.num_targets

        print(f"🚀 Single-Agent PPO Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Observation dim: {self.obs_dim} (expected: {expected_dim})")
        print(f"   Includes relative target counts (count - min_count) for fair pursuit")
        print(f"   Batch size: {self.args.batch_size}")
        print(f"   Num iterations: {self.args.num_iterations}")
        print(f"   Run name: {self.run_name}")

    def train(self):
        """Run the main training loop."""
        if self.envs is None:
            raise RuntimeError("Must call setup() before train()")

        def format_duration(seconds: float) -> str:
            seconds = max(0.0, seconds)
            total = int(seconds)
            hours = total // 3600
            minutes = (total % 3600) // 60
            secs = total % 60
            if hours > 0:
                return f"{hours:d}:{minutes:02d}:{secs:02d}"
            return f"{minutes:d}:{secs:02d}"

        # Storage setup
        if self.use_torch_env:
            obs = torch.zeros((self.args.num_steps, self.args.num_envs, self.obs_dim), device=self.device)
            actions = torch.zeros((self.args.num_steps, self.args.num_envs), device=self.device)
        else:
            obs = torch.zeros((self.args.num_steps, self.args.num_envs) + self.envs.single_observation_space.shape).to(self.device)
            actions = torch.zeros((self.args.num_steps, self.args.num_envs) + self.envs.single_action_space.shape).to(self.device)
        logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)

        # Start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.envs.reset(seed=self.args.seed)
        if not self.use_torch_env:
            next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.args.num_envs).to(self.device)

        print(f"\n{'='*60}")
        print(f"Starting training for {self.args.num_iterations} iterations ({self.args.total_timesteps} timesteps)")
        print(f"{'='*60}\n")

        for iteration in range(1, self.args.num_iterations+1):
            iteration_start = time.time()
            # Annealing the learning rate
            if self.args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
                lrnow = frac * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            # Rollout phase
            for step in range(0, self.args.num_steps):
                global_step += self.args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # Get actions from network
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()

                actions[step] = action
                logprobs[step] = logprob

                # Execute actions in environment
                if self.use_torch_env:
                    next_obs, reward, terminations, truncations, infos = self.envs.step(action)
                    next_done = terminations | truncations
                    rewards[step] = reward.view(-1)
                    next_done = next_done.to(self.device, dtype=torch.float32)
                else:
                    next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
                    next_done = np.logical_or(terminations, truncations)
                    rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                    next_obs = torch.Tensor(next_obs).to(self.device)
                    next_done = torch.Tensor(next_done).to(self.device)

                # Log episode statistics
                if (not self.use_torch_env) and "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            # Extract target reach counts if available
                            targets_reached_count = info.get("targets_reached_count", None)
                            min_targets_reached = info.get("min_targets_reached", None)

                            # Build log message
                            log_msg = f"global_step={global_step}, episodic_return={info['episode']['r']:.2f}, episodic_length={info['episode']['l']}"
                            if targets_reached_count is not None:
                                log_msg += f", target_counts={targets_reached_count}, min_reached={min_targets_reached}"
                            print(log_msg)

                            if self.args.track:
                                log_dict = {
                                    "charts/episodic_return": info["episode"]["r"],
                                    "charts/episodic_length": info["episode"]["l"],
                                }
                                # Add target reach metrics if available
                                if targets_reached_count is not None:
                                    log_dict["charts/min_targets_reached"] = min_targets_reached
                                    # Log individual target counts
                                    for i, count in enumerate(targets_reached_count):
                                        log_dict[f"charts/target_{i}_reached_count"] = count
                                    # Log max-min spread (measure of balance)
                                    spread = max(targets_reached_count) - min(targets_reached_count)
                                    log_dict["charts/target_count_spread"] = spread

                                wandb.log(log_dict, step=global_step)

            # Bootstrap value and compute advantages using shared utility
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(-1)
                advantages, returns = compute_gae(
                    rewards, values, dones, next_value, next_done,
                    self.args.gamma, self.args.gae_lambda
                )

            # Flatten the batch
            if self.use_torch_env:
                b_obs = obs.reshape((-1, self.obs_dim))
                b_actions = actions.reshape(-1)
            else:
                b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
                b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
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
                        b_actions.long()[mb_inds],
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

            # Compute explained variance
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            explained_var = compute_explained_variance(y_pred, y_true)

            # Log training metrics
            sps = int(global_step / (time.time() - start_time))
            iter_time = time.time() - iteration_start
            remaining_iters = self.args.num_iterations - iteration
            eta = format_duration(remaining_iters * iter_time)
            print(
                f"Iteration {iteration}/{self.args.num_iterations} - SPS: {sps} - "
                f"Value Loss: {metrics['v_loss']:.4f} - Policy Loss: {metrics['pg_loss']:.4f} - ETA: {eta}"
            )

            if self.args.track:
                wandb.log({
                    "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "losses/value_loss": metrics["v_loss"],
                    "losses/policy_loss": metrics["pg_loss"],
                    "losses/entropy": metrics["entropy_loss"],
                    "losses/old_approx_kl": metrics["old_approx_kl"],
                    "losses/approx_kl": metrics["approx_kl"],
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "charts/SPS": sps,
                    "charts/iteration": iteration,
                    "global_step": global_step,
                }, step=global_step)

            # Call iteration callback if provided
            if "on_iteration_end" in self.callbacks:
                self.callbacks["on_iteration_end"](self, iteration, global_step)

        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"{'='*60}\n")

        # Call training end callback if provided
        if "on_training_end" in self.callbacks:
            self.callbacks["on_training_end"](self, global_step)

    def save_model(self, path: Optional[str] = None):
        """Save the trained model."""
        if path is None:
            path = f"models/{self.run_name}"

        os.makedirs(path, exist_ok=True)
        model_path = f"{path}/agent.pt"
        torch.save(self.agent.state_dict(), model_path)
        print(f"✅ Model saved to {model_path}")

        if self.args.track:
            wandb.save(model_path)

    def load_model(self, path: str):
        """Load a trained model."""
        self.agent.load_state_dict(torch.load(path, map_location=self.device))
        print(f"✅ Model loaded from {path}")

    def cleanup(self):
        """Cleanup resources."""
        if self.envs is not None:
            self.envs.close()

        if self.args.track:
            wandb.finish()

        print("🧹 Cleanup completed")


if __name__ == "__main__":
    # Create trainer and run
    args = SingleAgentArgs(
        num_iterations=1000,
        num_targets=3,
        grid_size=15,
        target_reward=100.0,
        distance_reward_scale=0.1,
        track=False  # Set to True to enable wandb tracking
    )

    trainer = SingleAgentPPOTrainer(args)

    try:
        trainer.setup()
        trainer.train()
        trainer.save_model()
    finally:
        trainer.cleanup()
