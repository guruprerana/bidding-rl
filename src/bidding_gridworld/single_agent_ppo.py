# Single-agent PPO implementation for BiddingGridworld in single_agent_mode
# Simpler than multi-agent version - just learns to navigate and collect all targets

import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import wandb

from bidding_gridworld.bidding_gridworld_torch import (
    BiddingGridworld,
    BiddingGridworldConfig,
)
from ppo_utils import layer_init
from ppo_trainer_base import SingleAgentPPOTrainerBase


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


class SingleAgentPPOTrainer(SingleAgentPPOTrainerBase):
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
        super().__init__(args, callbacks=callbacks)
        self.obs_dim = None

    def setup(self):
        """Setup environments, agent, and optimizer."""
        # Environment setup (torch batched only)
        env_config = BiddingGridworldConfig(
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
        self.envs = BiddingGridworld(
            env_config,
            num_envs=self.args.num_envs,
            device=self.device,
            seed=self.args.seed,
        )

        # Create agent
        self.obs_dim = self.envs.obs_dim
        self.agent = SingleAgent(
            self.obs_dim,
            actor_hidden_sizes=self.args.actor_hidden_sizes,
            critic_hidden_sizes=self.args.critic_hidden_sizes,
        ).to(self.device)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)

        self.args.batch_size = int(self.args.num_envs * self.args.num_steps)
        self.args.minibatch_size = int(self.args.batch_size // self.args.num_minibatches)

        # Calculate expected observation dimension components for single-agent mode
        # Base: 2 (agent pos) + 2*num_targets (target pos) + num_targets (step counters) +
        #       1 (window steps) + num_targets (relative counts) [+ num_targets (reached flags)]
        include_reached = not self.args.moving_targets
        expected_dim = 2 + 2 * self.args.num_targets + self.args.num_targets + 1 + self.args.num_targets
        if include_reached:
            expected_dim += self.args.num_targets

        print(f"🚀 Single-Agent PPO Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Observation dim: {self.obs_dim} (expected: {expected_dim})")
        print(f"   Includes relative target counts (count - min_count) for fair pursuit")
        print(f"   Batch size: {self.args.batch_size}")
        print(f"   Num iterations: {self.args.num_iterations}")
        print(f"   Run name: {self.run_name}")

    def _on_rollout_step(self, infos, global_step: int):
        if not isinstance(infos, dict):
            return
        if 'final_info' not in infos:
            return
        for info in infos['final_info']:
            if not info or 'episode' not in info:
                continue
            targets_reached_count = info.get('targets_reached_count', None)
            min_targets_reached = info.get('min_targets_reached', None)
            log_msg = (
                f"global_step={global_step}, episodic_return={info['episode']['r']:.2f}, "
                f"episodic_length={info['episode']['l']}"
            )
            if targets_reached_count is not None:
                log_msg += f", target_counts={targets_reached_count}, min_reached={min_targets_reached}"
            print(log_msg)

            if self.args.track:
                log_dict = {
                    'charts/episodic_return': info['episode']['r'],
                    'charts/episodic_length': info['episode']['l'],
                }
                if targets_reached_count is not None:
                    log_dict['charts/min_targets_reached'] = min_targets_reached
                    for i, count in enumerate(targets_reached_count):
                        log_dict[f'charts/target_{i}_reached_count'] = count
                    spread = max(targets_reached_count) - min(targets_reached_count)
                    log_dict['charts/target_count_spread'] = spread

                wandb.log(log_dict, step=global_step)

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
