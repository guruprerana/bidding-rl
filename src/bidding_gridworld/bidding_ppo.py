# PPO implementation for BiddingGridworld environments
# Based on https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
# Adapted for multi-agent bidding with shared actor-critic networks

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
from ppo_utils import build_mlp, layer_init, MaskedAttentionPooling
from ppo_trainer_base import MultiAgentPPOTrainerBase


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
    window_bidding: bool = False
    """whether agents can choose their control window length"""
    window_penalty: float = 0.0
    """penalty multiplier for chosen window length (only applies when window_bidding=True)"""
    visible_targets: Optional[int] = None
    """number of nearest other targets visible to each agent (None = all targets visible, centralized)"""
    bidding_mechanism: str = "all_pay"
    """bidding penalty mechanism: 'all_pay', 'winner_pays', or 'winner_pays_others_reward'"""

    # Target attention pooling
    use_target_attention_pooling: bool = False
    """whether to use masked attention pooling over target observations"""
    target_embed_dim: int = 64
    """embedding dimension for target attention pooling"""
    target_encoder_hidden_sizes: Tuple[int, ...] = (64, 64)
    """hidden layer sizes for per-target encoder used before pooling"""

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


class SharedAgent(nn.Module):
    """
    Shared actor-critic network used by all agents.

    All agents use the same network parameters but receive different observations
    (targets reordered so each agent's target appears first). Each agent runs
    inference separately through this shared network.
    """

    def __init__(
        self,
        obs_dim,
        num_actions_per_agent,
        window_bidding=False,
        actor_hidden_sizes=None,
        critic_hidden_sizes=None,
        use_target_attention_pooling: bool = False,
        target_embed_dim: int = 64,
        target_encoder_hidden_sizes: Optional[Tuple[int, ...]] = None,
        attention_pooling_layout: str = "centralized",
        include_target_reached: bool = True,
    ):
        """
        Initialize shared actor-critic network.

        Args:
            obs_dim: Dimension of observation (targets reordered per agent)
            num_actions_per_agent: Number of action components per agent (2 or 3: direction + bid [+ window])
            window_bidding: Whether window bidding is enabled
        """
        super().__init__()
        self.window_bidding = window_bidding
        self.use_target_attention_pooling = use_target_attention_pooling
        self.attention_pooling_layout = attention_pooling_layout
        self.include_target_reached = include_target_reached

        actor_sizes = list(actor_hidden_sizes) if actor_hidden_sizes is not None else [128, 128, 128]
        critic_sizes = list(critic_hidden_sizes) if critic_hidden_sizes is not None else [256, 256, 256]

        if self.use_target_attention_pooling:
            encoder_sizes = target_encoder_hidden_sizes if target_encoder_hidden_sizes is not None else (64, 64)
            target_feat_dim = 6 if self.include_target_reached else 5
            self.target_pool = MaskedAttentionPooling(
                input_dim=target_feat_dim,
                embed_dim=target_embed_dim,
                hidden_sizes=encoder_sizes,
            )
            own_feat_dim = target_feat_dim
            self.encoded_obs_dim = 3 + own_feat_dim + target_embed_dim
        else:
            self.encoded_obs_dim = obs_dim

        # Shared critic network: outputs single value estimate
        critic_layers = []
        critic_in_dim = self.encoded_obs_dim
        for hidden_size in critic_sizes:
            critic_layers.append(layer_init(nn.Linear(critic_in_dim, hidden_size)))
            critic_layers.append(nn.ELU())
            critic_in_dim = hidden_size
        critic_layers.append(layer_init(nn.Linear(critic_in_dim, 1), std=1.0))
        self.critic = nn.Sequential(*critic_layers)

        # Shared actor network: outputs action logits
        # For bidding gridworld: outputs logits for direction (4 actions) and bid (bid_upper_bound+1 actions)
        # If window_bidding: also outputs window (action_window actions)
        # We'll use separate heads for each action component
        actor_layers = []
        actor_in_dim = self.encoded_obs_dim
        for hidden_size in actor_sizes:
            actor_layers.append(layer_init(nn.Linear(actor_in_dim, hidden_size)))
            actor_layers.append(nn.ELU())
            actor_in_dim = hidden_size
        self.actor_shared = nn.Sequential(*actor_layers) if actor_layers else nn.Identity()
        self.actor_feature_dim = actor_in_dim

        # Separate heads for action components
        self.direction_head = layer_init(nn.Linear(self.actor_feature_dim, 4), std=0.01)  # 4 directions
        self.bid_head = None  # Will be set based on bid_upper_bound
        self.window_head = None  # Will be set based on action_window if window_bidding is True

    def _encode_obs(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_target_attention_pooling:
            return x

        if x.dim() == 1:
            x = x.unsqueeze(0)

        agent_pos = x[:, :2]
        window_steps = x[:, -1:]
        obs_dim = x.shape[1]

        if self.attention_pooling_layout == "centralized":
            target_block = x[:, 2:-1]
            block_width = 4 if self.include_target_reached else 3
            if target_block.shape[1] % block_width != 0:
                raise ValueError(f"Invalid centralized obs layout for attention pooling (obs_dim={obs_dim}).")
            num_targets = target_block.shape[1] // block_width
            target_pos = target_block[:, : 2 * num_targets].reshape(-1, num_targets, 2)
            if self.include_target_reached:
                target_reached = target_block[:, 2 * num_targets: 3 * num_targets].reshape(-1, num_targets, 1)
                target_counters = target_block[:, 3 * num_targets: 4 * num_targets].reshape(-1, num_targets, 1)
            else:
                target_reached = None
                target_counters = target_block[:, 2 * num_targets: 3 * num_targets].reshape(-1, num_targets, 1)
        elif self.attention_pooling_layout == "visible":
            if self.include_target_reached:
                if (obs_dim - 7) % 3 != 0:
                    raise ValueError(f"Invalid visible-targets obs layout for attention pooling (obs_dim={obs_dim}).")
                visible_targets = (obs_dim - 7) // 3
            else:
                if (obs_dim - 6) % 2 != 0:
                    raise ValueError(f"Invalid visible-targets obs layout for attention pooling (obs_dim={obs_dim}).")
                visible_targets = (obs_dim - 6) // 2
            num_targets = visible_targets + 1

            own_pos = x[:, 2:4].reshape(-1, 1, 2)
            if visible_targets > 0:
                vis_pos = x[:, 4:4 + 2 * visible_targets].reshape(-1, visible_targets, 2)
                target_pos = torch.cat([own_pos, vis_pos], dim=1)
            else:
                target_pos = own_pos

            if self.include_target_reached:
                own_reached = x[:, 4 + 2 * visible_targets:5 + 2 * visible_targets].reshape(-1, 1, 1)
                if visible_targets > 0:
                    vis_reached = x[:, 5 + 2 * visible_targets:5 + 2 * visible_targets + visible_targets].reshape(
                        -1, visible_targets, 1
                    )
                    target_reached = torch.cat([own_reached, vis_reached], dim=1)
                else:
                    target_reached = own_reached
                own_counter = x[:, 5 + 3 * visible_targets:6 + 3 * visible_targets].reshape(-1, 1, 1)
            else:
                target_reached = None
                own_counter = x[:, 4 + 2 * visible_targets:5 + 2 * visible_targets].reshape(-1, 1, 1)
            if visible_targets > 0:
                zeros = torch.zeros((x.shape[0], visible_targets, 1), device=x.device, dtype=x.dtype)
                target_counters = torch.cat([own_counter, zeros], dim=1)
            else:
                target_counters = own_counter
        else:
            raise ValueError(f"Unknown attention pooling layout: {self.attention_pooling_layout}")

        rel_pos = target_pos - agent_pos.unsqueeze(1)
        if self.include_target_reached:
            target_feats = torch.cat([target_pos, rel_pos, target_reached, target_counters], dim=-1)
        else:
            target_feats = torch.cat([target_pos, rel_pos, target_counters], dim=-1)
        pooled = self.target_pool(target_feats)
        own_feats = target_feats[:, 0, :]
        return torch.cat([agent_pos, window_steps, own_feats, pooled], dim=-1)

    def set_bid_head(self, bid_upper_bound):
        """Set the bid head based on bid upper bound."""
        self.bid_head = layer_init(nn.Linear(self.actor_feature_dim, bid_upper_bound + 1), std=0.01)
        # Move to same device as the rest of the model
        self.bid_head = self.bid_head.to(next(self.parameters()).device)

    def set_window_head(self, action_window):
        """Set the window head based on action window (only for window_bidding mode)."""
        if self.window_bidding:
            self.window_head = layer_init(nn.Linear(self.actor_feature_dim, action_window), std=0.01)
            # Move to same device as the rest of the model
            self.window_head = self.window_head.to(next(self.parameters()).device)

    def get_value(self, x):
        """
        Get value estimate for given observation.

        Args:
            x: Observation tensor (can be batched)

        Returns:
            Value estimate
        """
        encoded = self._encode_obs(x)
        return self.critic(encoded)

    def get_action_and_value(self, x, action=None):
        """
        Get action and value for given observation.

        This is the core inference function. Each agent calls this separately
        with their reordered observation (their target appears first in the obs).

        Args:
            x: Observation tensor (can be batched)
            action: If provided, compute log prob for this action. Otherwise sample new action.
                   Action should be tensor of shape (..., 2) or (..., 3) where last dim is [direction, bid] or [direction, bid, window]

        Returns:
            action: Sampled or provided action [direction, bid] or [direction, bid, window]
            log_prob: Log probability of the action
            entropy: Entropy of the action distribution
            value: Value estimate
        """
        x = self._encode_obs(x)
        # Get shared features
        shared_features = self.actor_shared(x)

        # Get logits for direction and bid separately
        direction_logits = self.direction_head(shared_features)
        bid_logits = self.bid_head(shared_features)

        # Create categorical distributions
        direction_dist = Categorical(logits=direction_logits)
        bid_dist = Categorical(logits=bid_logits)

        # Handle window if window_bidding is enabled
        if self.window_bidding:
            window_logits = self.window_head(shared_features)
            window_dist = Categorical(logits=window_logits)

        # Sample or use provided action
        if action is None:
            # Sample new actions
            direction = direction_dist.sample()
            bid = bid_dist.sample()
            if self.window_bidding:
                window = window_dist.sample()
                action = torch.stack([direction, bid, window], dim=-1)
            else:
                action = torch.stack([direction, bid], dim=-1)
        else:
            # Use provided action
            direction = action[..., 0]
            bid = action[..., 1]
            if self.window_bidding:
                window = action[..., 2]

        # Compute log probabilities (sum of independent log probs)
        direction_log_prob = direction_dist.log_prob(direction)
        bid_log_prob = bid_dist.log_prob(bid)
        total_log_prob = direction_log_prob + bid_log_prob

        # Compute entropy (sum of independent entropies)
        entropy = direction_dist.entropy() + bid_dist.entropy()

        if self.window_bidding:
            window_log_prob = window_dist.log_prob(window)
            total_log_prob = total_log_prob + window_log_prob
            entropy = entropy + window_dist.entropy()

        # Get value estimate
        value = self.critic(x)

        return action, total_log_prob, entropy, value


class PPOTrainer(MultiAgentPPOTrainerBase):
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
        super().__init__(args, callbacks=callbacks)
        self.obs_dim = None
        self.num_action_components = None
        self._episode_agent_wins = None
        self._episode_bid_sum = None
        self._episode_bid_count = None
        self._episode_bid_min = None
        self._episode_bid_max = None

    def setup(self):
        """Setup environments, agent, and optimizer."""
        # Environment setup (torch batched only)
        env_config = BiddingGridworldConfig(
            grid_size=self.args.grid_size,
            num_agents=self.args.num_agents,
            bid_upper_bound=self.args.bid_upper_bound,
            bid_penalty=self.args.bid_penalty,
            target_reward=self.args.target_reward,
            max_steps=self.args.max_steps,
            action_window=self.args.action_window,
            distance_reward_scale=self.args.distance_reward_scale,
            target_expiry_steps=self.args.target_expiry_steps,
            target_expiry_penalty=self.args.target_expiry_penalty,
            moving_targets=self.args.moving_targets,
            direction_change_prob=self.args.direction_change_prob,
            target_move_interval=self.args.target_move_interval,
            window_bidding=self.args.window_bidding,
            window_penalty=self.args.window_penalty,
            visible_targets=self.args.visible_targets,
            single_agent_mode=False,
            bidding_mechanism=self.args.bidding_mechanism,
        )
        self.envs = BiddingGridworld(
            env_config,
            num_envs=self.args.num_envs,
            device=self.device,
            seed=self.args.seed,
        )

        # Create shared agent
        # Observation space is (num_agents, obs_dim), so we need shape[1] for per-agent obs dim
        self.obs_dim = self.envs.per_agent_obs_dim
        num_actions_per_agent = 3 if self.args.window_bidding else 2
        self.num_action_components = num_actions_per_agent
        self.agent = SharedAgent(
            self.obs_dim,
            num_actions_per_agent=num_actions_per_agent,
            window_bidding=self.args.window_bidding,
            actor_hidden_sizes=self.args.actor_hidden_sizes,
            critic_hidden_sizes=self.args.critic_hidden_sizes,
            use_target_attention_pooling=self.args.use_target_attention_pooling,
            target_embed_dim=self.args.target_embed_dim,
            target_encoder_hidden_sizes=self.args.target_encoder_hidden_sizes,
            attention_pooling_layout="centralized" if self.args.visible_targets is None else "visible",
            include_target_reached=not self.args.moving_targets,
        ).to(self.device)
        self.agent.set_bid_head(self.args.bid_upper_bound)
        if self.args.window_bidding:
            self.agent.set_window_head(self.args.action_window)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)

        self.args.batch_size = int(self.args.num_envs * self.args.num_steps * self.args.num_agents)
        self.args.minibatch_size = int(self.args.batch_size // self.args.num_minibatches)

        print(f"🚀 PPO Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Observation dim: {self.obs_dim}")
        if self.args.visible_targets is None:
            print(f"   Observation mode: Centralized (all agents see all targets)")
        else:
            print(f"   Observation mode: Decentralized (visible_targets={self.args.visible_targets})")
        print(f"   Window bidding: {self.args.window_bidding}")
        if self.args.window_bidding:
            print(f"   Window penalty: {self.args.window_penalty}")
        print(f"   Target attention pooling: {self.args.use_target_attention_pooling}")
        if self.args.use_target_attention_pooling:
            layout = "centralized" if self.args.visible_targets is None else "visible"
            print(f"   Attention layout: {layout}")
            print(f"   Target embed dim: {self.args.target_embed_dim}")
        print(f"   Actions per agent: {num_actions_per_agent}")
        print(f"   Batch size: {self.args.batch_size}")
        print(f"   Num iterations: {self.args.num_iterations}")
        print(f"   Run name: {self.run_name}")

    def _on_iteration_start(self, iteration: int):
        if not self.args.track:
            return
        self._episode_agent_wins = torch.zeros(self.args.num_agents, device=self.device, dtype=torch.int64)
        self._episode_bid_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        self._episode_bid_count = torch.zeros((), device=self.device, dtype=torch.int64)
        self._episode_bid_min = None
        self._episode_bid_max = None

    def _on_rollout_step(self, infos, global_step: int):
        if not self.args.track or not isinstance(infos, dict):
            return
        winning_agent = infos.get('winning_agent', None)
        if torch.is_tensor(winning_agent):
            valid = winning_agent >= 0
            if torch.any(valid):
                counts = torch.bincount(
                    winning_agent[valid].to(torch.int64),
                    minlength=self.args.num_agents,
                )
                self._episode_agent_wins += counts

        bids = infos.get('bids', None)
        if torch.is_tensor(bids):
            bids_f = bids.to(torch.float32)
            self._episode_bid_sum += bids_f.sum()
            self._episode_bid_count += bids_f.numel()
            step_min = bids_f.min()
            step_max = bids_f.max()
            self._episode_bid_min = step_min if self._episode_bid_min is None else torch.minimum(self._episode_bid_min, step_min)
            self._episode_bid_max = step_max if self._episode_bid_max is None else torch.maximum(self._episode_bid_max, step_max)

    def _extra_log_dict(self, global_step: int) -> dict:
        if not self._last_rollout_stats:
            return {}
        rewards = self._last_rollout_stats['rewards']
        values = self._last_rollout_stats['values']
        advantages = self._last_rollout_stats['advantages']
        log_dict = {
            'rewards/avg_step_reward': rewards.mean().item(),
            'rewards/max_step_reward': rewards.max().item(),
            'rewards/min_step_reward': rewards.min().item(),
            'values/mean': values.mean().item(),
            'values/std': values.std().item(),
            'values/max': values.max().item(),
            'values/min': values.min().item(),
            'advantages/mean': advantages.mean().item(),
            'advantages/std': advantages.std().item(),
        }
        if self.args.track and self._episode_bid_count is not None and self._episode_bid_count.item() > 0:
            log_dict['bidding/avg_bid_value'] = (self._episode_bid_sum / self._episode_bid_count).item()
            log_dict['bidding/max_bid_value'] = self._episode_bid_max.item() if self._episode_bid_max is not None else 0.0
            log_dict['bidding/min_bid_value'] = self._episode_bid_min.item() if self._episode_bid_min is not None else 0.0
        total_wins = int(self._episode_agent_wins.sum().item()) if self._episode_agent_wins is not None else 0
        if total_wins > 0:
            for agent_idx in range(self.args.num_agents):
                agent_key = f'agent_{agent_idx}'
                win_rate = self._episode_agent_wins[agent_idx].item() / total_wins
                log_dict[f'agents/{agent_key}_win_rate'] = win_rate
        return log_dict

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
