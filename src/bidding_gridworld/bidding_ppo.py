# PPO implementation for BiddingGridworld environments
# Based on https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
# Adapted for multi-agent bidding with shared actor-critic networks

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
import wandb

from bidding_gridworld.bidding_gridworld_torch import (
    BiddingGridworld,
    BiddingGridworldConfig,
)
from ppo_utils import (
    build_mlp,
    layer_init,
    MaskedAttentionPooling,
    compute_gae,
    compute_explained_variance,
    format_duration,
    ppo_update_step,
)
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
    use_target_priorities: bool = True
    """include sampled target priorities in rewards and observations"""
    programmatic_bidding: str = "none"
    """optional deterministic bidding controller: 'none' or 'nearest_target'"""
    bid_only_ppo: bool = False
    """optimize learned bids while executing deterministic learned directions"""
    freeze_navigation_during_bid_only: bool = True
    """freeze the learned navigation encoder/trunk/head during bid-only PPO"""
    separate_bid_actor: bool = False
    """use a bid-specific actor trunk instead of the shared direction trunk"""
    bid_actor_hidden_sizes: Tuple[int, ...] = (128, 128)
    """hidden sizes for the optional bid-specific actor trunk"""
    ordinal_bid_head: bool = False
    """parameterize ordered bid logits through a learned center and scale"""
    policy_warm_start_checkpoint: Optional[str] = None
    """checkpoint used to initialize an end-to-end learned policy"""
    bid_credit_assignment: str = "individual"
    """bid PPO reward: individual, controller_team, shared_team, or mixed_team"""
    bid_other_reward_fraction: float = 1.0
    """with mixed_team credit, fraction of every other controller reward"""
    bid_mixed_reward_normalize: bool = False
    """rescale mixed credit to preserve cooperative team-reward magnitude"""
    factorized_auction_ppo: bool = False
    """mask PPO direction/bid losses to executed directions and real auctions"""
    battery_capacity: Optional[int] = None
    """shared robot battery capacity; None disables recharge mechanics"""
    recharge_station_positions: Optional[Tuple[Tuple[int, int], ...]] = None
    """fixed recharge-station (row, col) positions"""
    moving_recharge_stations: bool = False
    """whether recharge stations follow independent stochastic random walks"""
    recharge_station_direction_change_prob: float = 0.1
    """probability of changing station direction at each movement"""
    recharge_station_move_interval: int = 5
    """environment steps between recharge-station movements"""
    movement_energy_cost: int = 1
    """battery units consumed by each successful movement"""
    battery_depletion_penalty: float = 0.0
    """penalty charged to the controller when depletion triggers a tow"""
    charging_agent_enabled: bool = False
    """add a separately trained bidder responsible for recharging"""
    charging_low_battery_threshold: int = 20
    """battery level below which station-progress shaping becomes active"""
    charging_distance_reward_scale: float = 2.0
    """charging-agent reward per Manhattan step toward a station at full urgency"""
    charging_recharge_bonus: float = 20.0
    """charging-agent reward for a full-capacity refill, scaled by energy restored"""
    charging_depletion_penalty: float = 50.0
    """charging-agent penalty whenever the shared battery is depleted"""
    charging_high_battery_control_penalty: float = 0.0
    """penalty when the charging agent controls above its battery threshold"""
    feeder_low_battery_control_penalty: float = 0.0
    """per-step feeder penalty for controlling while charging is active"""
    feeder_yield_aux_coef: float = 0.0
    """auxiliary bid-zero loss coefficient on charging-active feeder observations"""
    feeder_yield_aux_bid_head_only: bool = False
    """stop auxiliary gradients before the feeder bid head"""
    feeder_yield_activation_margin: Optional[int] = None
    """optional feeder-yield margin; None uses charging_activation_margin"""
    charging_low_battery_bid_boost: int = 0
    """effective bid bonus for positive charging bids at low battery"""
    charging_bid_boost_threshold: Optional[int] = None
    """battery threshold for bid boost; None uses charging_low_battery_threshold"""
    charging_activation_margin: Optional[int] = None
    """gate charging bids until battery is within this margin of station distance"""
    charging_release_window_on_recharge: bool = False
    """release charging-agent control immediately after a refill"""
    charging_programmatic_navigation: bool = False
    """learn only charging bids and use shortest-path station navigation"""
    charging_greedy_navigation_eval: bool = False
    """use charger direction argmax during eval while retaining sampled bids"""
    charging_separate_direction_actor: bool = False
    """give learned charger navigation a trunk separate from its bidding trunk"""
    charging_ppo_bid_only: bool = False
    """update only charger bid/value policy with PPO after navigation pretraining"""
    charging_reserve_features_enabled: bool = False
    """add capacity-invariant physical battery and station-reserve features"""
    charging_nearest_station_features_enabled: bool = False
    """add nearest-station relative row/column to the charger observation"""
    charging_learning_rate: Optional[float] = None
    """charging-policy learning rate; None uses learning_rate"""
    charging_actor_hidden_sizes: Tuple[int, ...] = (128, 128, 128)
    """hidden layer sizes for the charging actor"""
    charging_critic_hidden_sizes: Tuple[int, ...] = (256, 256, 256)
    """hidden layer sizes for the charging critic"""
    feeder_warm_start_checkpoint: Optional[str] = None
    """priority-only feeder checkpoint used to initialize the feeder policy"""
    feeder_freeze_iterations: int = 0
    """number of initial iterations that update only the charging policy"""
    feeder_finetune_learning_rate: Optional[float] = None
    """feeder learning rate after unfreezing; None uses learning_rate"""
    charging_bc_updates: int = 0
    """number of shortest-path behavior-cloning updates for the charger"""
    charging_bc_batch_size: int = 4096
    """synthetic state batch size for charger behavior cloning"""
    charging_bc_learning_rate: float = 1e-3
    """learning rate for charger behavior cloning"""
    charging_bc_bid_loss_coef: float = 0.0
    """optional coefficient for cloning zero/max urgency bids"""
    charging_bc_bid_value: Optional[int] = None
    """positive BC bid target; None uses bid_upper_bound"""
    charging_bc_emergency_margin: Optional[int] = None
    """station-distance margin where BC escalates to an emergency bid"""
    charging_bc_emergency_bid_value: Optional[int] = None
    """emergency BC bid target; None uses bid_upper_bound"""
    charging_bc_refresh_updates: int = 0
    """synthetic BC updates applied after each charging PPO iteration"""
    charging_bc_refresh_learning_rate: Optional[float] = None
    """learning rate for recurring BC refresh; None uses BC pretraining LR"""

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
    lr_min: float = 0.0
    """Minimum learning rate floor when annealing (0.0 = anneal all the way to zero)"""
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
        energy_feature_dim: int = 0,
        separate_direction_actor: bool = False,
        use_target_priorities: bool = True,
        separate_bid_actor: bool = False,
        bid_actor_hidden_sizes: Optional[Tuple[int, ...]] = None,
        ordinal_bid_head: bool = False,
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
        self.energy_feature_dim = energy_feature_dim
        self.separate_direction_actor = separate_direction_actor
        self.separate_bid_actor = separate_bid_actor
        self.ordinal_bid_head = ordinal_bid_head
        self.bid_upper_bound = None
        self.use_target_priorities = use_target_priorities

        actor_sizes = list(actor_hidden_sizes) if actor_hidden_sizes is not None else [128, 128, 128]
        critic_sizes = list(critic_hidden_sizes) if critic_hidden_sizes is not None else [256, 256, 256]

        if self.use_target_attention_pooling:
            encoder_sizes = target_encoder_hidden_sizes if target_encoder_hidden_sizes is not None else (64, 64)
            target_feat_dim = (
                5
                + int(self.include_target_reached)
                + int(self.use_target_priorities)
            )
            self.target_pool = MaskedAttentionPooling(
                input_dim=target_feat_dim,
                embed_dim=target_embed_dim,
                hidden_sizes=encoder_sizes,
            )
            own_feat_dim = target_feat_dim
            self.encoded_obs_dim = (
                3 + self.energy_feature_dim + own_feat_dim + target_embed_dim
            )
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

        if self.separate_bid_actor:
            bid_sizes = list(
                bid_actor_hidden_sizes
                if bid_actor_hidden_sizes is not None
                else (128, 128)
            )
            bid_layers = []
            bid_in_dim = self.encoded_obs_dim
            for hidden_size in bid_sizes:
                bid_layers.append(
                    layer_init(nn.Linear(bid_in_dim, hidden_size))
                )
                bid_layers.append(nn.ELU())
                bid_in_dim = hidden_size
            self.bid_actor = (
                nn.Sequential(*bid_layers) if bid_layers else nn.Identity()
            )
            self.bid_feature_dim = bid_in_dim
        else:
            self.bid_actor = None
            self.bid_feature_dim = self.actor_feature_dim

        if separate_direction_actor:
            direction_layers = []
            direction_in_dim = self.encoded_obs_dim
            for hidden_size in actor_sizes:
                direction_layers.append(
                    layer_init(nn.Linear(direction_in_dim, hidden_size))
                )
                direction_layers.append(nn.ELU())
                direction_in_dim = hidden_size
            self.direction_actor = (
                nn.Sequential(*direction_layers)
                if direction_layers
                else nn.Identity()
            )
            direction_feature_dim = direction_in_dim
        else:
            self.direction_actor = None
            direction_feature_dim = self.actor_feature_dim

        # Separate heads for action components
        self.direction_head = layer_init(
            nn.Linear(direction_feature_dim, 4), std=0.01
        )
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
        target_block_end = obs_dim - 1 - self.energy_feature_dim
        energy_features = x[:, target_block_end:obs_dim - 1]

        if self.attention_pooling_layout == "centralized":
            target_block = x[:, 2:target_block_end]
            block_width = (
                3
                + int(self.include_target_reached)
                + int(self.use_target_priorities)
            )
            if target_block.shape[1] % block_width != 0:
                raise ValueError(f"Invalid centralized obs layout for attention pooling (obs_dim={obs_dim}).")
            num_targets = target_block.shape[1] // block_width
            target_pos = target_block[:, : 2 * num_targets].reshape(-1, num_targets, 2)
            cursor = 2 * num_targets
            if self.include_target_reached:
                target_reached = target_block[:, cursor:cursor + num_targets].reshape(-1, num_targets, 1)
                cursor += num_targets
            else:
                target_reached = None
            target_counters = target_block[:, cursor:cursor + num_targets].reshape(-1, num_targets, 1)
            cursor += num_targets
            target_priorities = (
                target_block[:, cursor:cursor + num_targets].reshape(-1, num_targets, 1)
                if self.use_target_priorities
                else None
            )
        elif self.attention_pooling_layout == "visible":
            layout_obs_dim = obs_dim - self.energy_feature_dim
            fixed_width = 6 + int(self.include_target_reached) + int(self.use_target_priorities)
            per_visible_width = 2 + int(self.include_target_reached) + int(self.use_target_priorities)
            if (layout_obs_dim - fixed_width) % per_visible_width != 0:
                raise ValueError(f"Invalid visible-targets obs layout for attention pooling (obs_dim={obs_dim}).")
            visible_targets = (layout_obs_dim - fixed_width) // per_visible_width
            num_targets = visible_targets + 1

            cursor = 2
            own_pos = x[:, cursor:cursor + 2].reshape(-1, 1, 2)
            cursor += 2
            if visible_targets > 0:
                vis_pos = x[:, cursor:cursor + 2 * visible_targets].reshape(-1, visible_targets, 2)
                cursor += 2 * visible_targets
                target_pos = torch.cat([own_pos, vis_pos], dim=1)
            else:
                target_pos = own_pos

            if self.include_target_reached:
                own_reached = x[:, cursor:cursor + 1].reshape(-1, 1, 1)
                cursor += 1
                if visible_targets > 0:
                    vis_reached = x[:, cursor:cursor + visible_targets].reshape(
                        -1, visible_targets, 1
                    )
                    cursor += visible_targets
                    target_reached = torch.cat([own_reached, vis_reached], dim=1)
                else:
                    target_reached = own_reached
            else:
                target_reached = None
            own_counter = x[:, cursor:cursor + 1].reshape(-1, 1, 1)
            cursor += 1
            if visible_targets > 0:
                zeros = torch.zeros((x.shape[0], visible_targets, 1), device=x.device, dtype=x.dtype)
                target_counters = torch.cat([own_counter, zeros], dim=1)
            else:
                target_counters = own_counter
            if self.use_target_priorities:
                own_priority = x[:, cursor:cursor + 1].reshape(-1, 1, 1)
                cursor += 1
                if visible_targets > 0:
                    vis_priorities = x[:, cursor:cursor + visible_targets].reshape(-1, visible_targets, 1)
                    target_priorities = torch.cat([own_priority, vis_priorities], dim=1)
                else:
                    target_priorities = own_priority
            else:
                target_priorities = None
        else:
            raise ValueError(f"Unknown attention pooling layout: {self.attention_pooling_layout}")

        rel_pos = target_pos - agent_pos.unsqueeze(1)
        target_feature_parts = [target_pos, rel_pos]
        if self.include_target_reached:
            target_feature_parts.append(target_reached)
        target_feature_parts.append(target_counters)
        if self.use_target_priorities:
            target_feature_parts.append(target_priorities)
        target_feats = torch.cat(target_feature_parts, dim=-1)
        pooled = self.target_pool(target_feats)
        own_feats = target_feats[:, 0, :]
        return torch.cat(
            [agent_pos, energy_features, window_steps, own_feats, pooled], dim=-1
        )

    def set_bid_head(self, bid_upper_bound):
        """Set the bid head based on bid upper bound."""
        self.bid_upper_bound = int(bid_upper_bound)
        output_dim = 2 if self.ordinal_bid_head else bid_upper_bound + 1
        self.bid_head = layer_init(
            nn.Linear(self.bid_feature_dim, output_dim), std=0.01
        )
        if self.ordinal_bid_head:
            # Start with a broad distribution centered in the valid range.
            # PPO then learns both the ordinal center and its uncertainty.
            with torch.no_grad():
                self.bid_head.bias[1] = torch.log(torch.expm1(torch.tensor(4.0)))
        # Move to same device as the rest of the model
        self.bid_head = self.bid_head.to(next(self.parameters()).device)

    def _bid_logits(self, bid_features: torch.Tensor) -> torch.Tensor:
        raw = self.bid_head(bid_features)
        if not self.ordinal_bid_head:
            return raw
        if self.bid_upper_bound is None:
            raise RuntimeError("set_bid_head must be called before bidding")
        center = self.bid_upper_bound * torch.sigmoid(raw[..., :1])
        scale = torch.nn.functional.softplus(raw[..., 1:2]) + 0.25
        levels = torch.arange(
            self.bid_upper_bound + 1,
            device=raw.device,
            dtype=raw.dtype,
        )
        return -0.5 * ((levels - center) / scale).square()

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

    def get_action_and_value(
        self,
        x,
        action=None,
        deterministic=False,
        deterministic_direction=False,
    ):
        """
        Get action and value for given observation.

        This is the core inference function. Each agent calls this separately
        with their reordered observation (their target appears first in the obs).

        Args:
            x: Observation tensor (can be batched)
            action: If provided, compute log prob for this action. Otherwise sample new action.
                   Action should be tensor of shape (..., 2) or (..., 3) where last dim is [direction, bid] or [direction, bid, window]
            deterministic: When no action is provided, choose every
                highest-logit action for evaluation instead of sampling.
            deterministic_direction: Choose only the highest-logit direction;
                bids and optional windows retain their normal sampling mode.

        Returns:
            action: Sampled or provided action [direction, bid] or [direction, bid, window]
            log_prob: Log probability of the action
            entropy: Entropy of the action distribution
            value: Value estimate
        """
        x = self._encode_obs(x)
        # Get shared features
        shared_features = self.actor_shared(x)
        direction_features = (
            self.direction_actor(x)
            if self.direction_actor is not None
            else shared_features
        )
        bid_features = (
            self.bid_actor(x)
            if self.bid_actor is not None
            else shared_features
        )

        # Get logits for direction and bid separately
        direction_logits = self.direction_head(direction_features)
        bid_logits = self._bid_logits(bid_features)

        # Create categorical distributions
        direction_dist = Categorical(logits=direction_logits)
        bid_dist = Categorical(logits=bid_logits)

        # Handle window if window_bidding is enabled
        if self.window_bidding:
            window_logits = self.window_head(shared_features)
            window_dist = Categorical(logits=window_logits)

        # Sample or use provided action
        if action is None:
            if deterministic:
                direction = direction_logits.argmax(dim=-1)
                bid = bid_logits.argmax(dim=-1)
            else:
                direction = (
                    direction_logits.argmax(dim=-1)
                    if deterministic_direction
                    else direction_dist.sample()
                )
                bid = bid_dist.sample()
            if self.window_bidding:
                window = (
                    window_logits.argmax(dim=-1)
                    if deterministic
                    else window_dist.sample()
                )
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

    def get_factorized_action_and_value(
        self,
        x,
        action=None,
        deterministic=False,
    ):
        """Return separate direction and bid statistics for auction PPO.

        Directions and bids affect the environment on different subsets of
        steps: only the controller's direction is executed, while bids affect
        only genuine auction rounds. Keeping their log probabilities separate
        lets the trainer apply the corresponding causal masks.
        """
        if self.window_bidding:
            raise ValueError(
                "factorized_auction_ppo does not support learned window lengths"
            )
        encoded = self._encode_obs(x)
        shared_features = self.actor_shared(encoded)
        direction_features = (
            self.direction_actor(encoded)
            if self.direction_actor is not None
            else shared_features
        )
        bid_features = (
            self.bid_actor(encoded)
            if self.bid_actor is not None
            else shared_features
        )
        direction_logits = self.direction_head(direction_features)
        bid_logits = self._bid_logits(bid_features)
        direction_dist = Categorical(logits=direction_logits)
        bid_dist = Categorical(logits=bid_logits)
        if action is None:
            if deterministic:
                direction = direction_logits.argmax(dim=-1)
                bid = bid_logits.argmax(dim=-1)
            else:
                direction = direction_dist.sample()
                bid = bid_dist.sample()
            action = torch.stack([direction, bid], dim=-1)
        else:
            direction = action[..., 0]
            bid = action[..., 1]
        return (
            action,
            direction_dist.log_prob(direction),
            bid_dist.log_prob(bid),
            direction_dist.entropy(),
            bid_dist.entropy(),
            self.critic(encoded),
        )

    def get_bid_action_and_value(self, x, action=None, deterministic=False):
        """Sample/evaluate bids while emitting a dummy direction action."""
        encoded = self._encode_obs(x)
        shared_features = self.actor_shared(encoded)
        bid_features = (
            self.bid_actor(encoded)
            if self.bid_actor is not None
            else shared_features
        )
        bid_logits = self._bid_logits(bid_features)
        bid_dist = Categorical(logits=bid_logits)

        if action is None:
            bid = (
                bid_logits.argmax(dim=-1)
                if deterministic
                else bid_dist.sample()
            )
        else:
            bid = action[..., 1]
        direction = torch.zeros_like(bid)
        log_prob = bid_dist.log_prob(bid)
        entropy = bid_dist.entropy()

        if self.window_bidding:
            window_logits = self.window_head(shared_features)
            window_dist = Categorical(logits=window_logits)
            if action is None:
                window = (
                    window_logits.argmax(dim=-1)
                    if deterministic
                    else window_dist.sample()
                )
            else:
                window = action[..., 2]
            result_action = torch.stack(
                [direction, bid, window], dim=-1
            )
            log_prob = log_prob + window_dist.log_prob(window)
            entropy = entropy + window_dist.entropy()
        else:
            result_action = torch.stack([direction, bid], dim=-1)

        value = self.critic(encoded)
        return result_action, log_prob, entropy, value

    def get_bid_action_and_value_with_direction(
        self,
        x,
        action=None,
        deterministic=False,
    ):
        """Learn bids while executing the policy's learned direction.

        Only the categorical bid contributes to PPO log-probability and
        entropy. Directions remain policy outputs and use argmax so frozen
        navigation stays stable during bidding specialization.
        """
        encoded = self._encode_obs(x)
        shared_features = self.actor_shared(encoded)
        direction_features = (
            self.direction_actor(encoded)
            if self.direction_actor is not None
            else shared_features
        )
        bid_features = (
            self.bid_actor(encoded)
            if self.bid_actor is not None
            else shared_features
        )
        direction_logits = self.direction_head(direction_features)
        bid_logits = self._bid_logits(bid_features)
        bid_dist = Categorical(logits=bid_logits)

        direction = direction_logits.argmax(dim=-1)
        if action is None:
            bid = bid_logits.argmax(dim=-1) if deterministic else bid_dist.sample()
        else:
            bid = action[..., 1]
        if self.window_bidding:
            result_action = torch.stack(
                [direction, bid, torch.zeros_like(direction)], dim=-1
            )
        else:
            result_action = torch.stack([direction, bid], dim=-1)
        return (
            result_action,
            bid_dist.log_prob(bid),
            bid_dist.entropy(),
            self.critic(encoded),
        )

    def get_direction_action_and_value(
        self,
        x,
        action=None,
        deterministic=False,
    ):
        """Sample/evaluate directions while emitting a dummy bid action."""
        encoded = self._encode_obs(x)
        shared_features = self.actor_shared(encoded)
        direction_features = (
            self.direction_actor(encoded)
            if self.direction_actor is not None
            else shared_features
        )
        direction_logits = self.direction_head(direction_features)
        direction_dist = Categorical(logits=direction_logits)

        if action is None:
            direction = (
                direction_logits.argmax(dim=-1)
                if deterministic
                else direction_dist.sample()
            )
        else:
            direction = action[..., 0]
        bid = torch.zeros_like(direction)
        if self.window_bidding:
            result_action = torch.stack(
                [direction, bid, torch.zeros_like(direction)], dim=-1
            )
        else:
            result_action = torch.stack([direction, bid], dim=-1)
        return (
            result_action,
            direction_dist.log_prob(direction),
            direction_dist.entropy(),
            self.critic(encoded),
        )


def load_feeder_checkpoint_with_energy_expansion(
    agent: SharedAgent,
    checkpoint_path: str,
    energy_feature_dim: int,
) -> Dict:
    """Load a feeder checkpoint, expanding only new encoded energy columns."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Feeder warm-start checkpoint not found: {path}")

    checkpoint_state = torch.load(path, map_location="cpu", weights_only=True)
    current_state = agent.state_dict()
    expanded_keys = {
        "actor_shared.0.weight",
        "critic.0.weight",
    }
    loaded_state = {}
    exact_keys = []
    expanded = []

    for key, current_value in current_state.items():
        if key not in checkpoint_state:
            raise ValueError(f"Warm-start checkpoint is missing parameter: {key}")
        source_value = checkpoint_state[key]
        if source_value.shape == current_value.shape:
            loaded_state[key] = source_value
            exact_keys.append(key)
            continue
        if (
            key in expanded_keys
            and source_value.ndim == 2
            and current_value.ndim == 2
            and source_value.shape[0] == current_value.shape[0]
            and source_value.shape[1] + energy_feature_dim
            == current_value.shape[1]
        ):
            value = torch.zeros_like(current_value)
            value[:, :2] = source_value[:, :2]
            value[:, 2 + energy_feature_dim:] = source_value[:, 2:]
            loaded_state[key] = value
            expanded.append(key)
            continue
        raise ValueError(
            f"Incompatible warm-start parameter {key}: checkpoint "
            f"{tuple(source_value.shape)} vs current {tuple(current_value.shape)}"
        )

    unexpected = sorted(set(checkpoint_state) - set(current_state))
    if unexpected:
        raise ValueError(
            "Warm-start checkpoint has unexpected parameters: "
            + ", ".join(unexpected)
        )
    agent.load_state_dict(loaded_state, strict=True)
    return {
        "checkpoint": str(path),
        "exact_parameter_tensors": len(exact_keys),
        "expanded_parameter_tensors": expanded,
        "new_energy_columns": energy_feature_dim,
        "energy_columns_initialized_to_zero": True,
    }


def load_policy_warm_start(
    agent: SharedAgent,
    checkpoint_path: str,
    reset_bid_policy: bool = False,
) -> Dict:
    """Load compatible learned-policy weights for bidding specialization."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Policy warm-start checkpoint not found: {path}")
    source = torch.load(path, map_location="cpu", weights_only=True)
    current = agent.state_dict()
    loaded = {}
    skipped = []
    for key, value in source.items():
        if reset_bid_policy and key.startswith("bid_head."):
            skipped.append(key)
            continue
        if key in current and current[key].shape == value.shape:
            loaded[key] = value
        else:
            skipped.append(key)
    incompatible = agent.load_state_dict(loaded, strict=False)
    return {
        "checkpoint": str(path),
        "loaded_keys": sorted(loaded),
        "skipped_source_keys": sorted(skipped),
        "missing_model_keys": sorted(incompatible.missing_keys),
        "unexpected_keys": sorted(incompatible.unexpected_keys),
    }


def pretrain_charging_policy(
    agent: SharedAgent,
    env: BiddingGridworld,
    updates: int,
    batch_size: int,
    learning_rate: float,
    bid_loss_coef: float = 0.0,
    bid_value: Optional[int] = None,
    emergency_margin: Optional[int] = None,
    emergency_bid_value: Optional[int] = None,
    direction_loss_coef: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> Dict:
    """Behavior-clone charging directions and optional urgency bids."""
    if updates < 0:
        raise ValueError("charging_bc_updates must be non-negative")
    if batch_size <= 0:
        raise ValueError("charging_bc_batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("charging_bc_learning_rate must be positive")
    if bid_loss_coef < 0:
        raise ValueError("charging_bc_bid_loss_coef must be non-negative")
    if direction_loss_coef < 0:
        raise ValueError("direction_loss_coef must be non-negative")
    if bid_value is None:
        bid_value = env.config.bid_upper_bound
    if not 0 <= bid_value <= env.config.bid_upper_bound:
        raise ValueError(
            "charging_bc_bid_value must be between 0 and bid_upper_bound"
        )
    if emergency_margin is not None and emergency_margin < 0:
        raise ValueError(
            "charging_bc_emergency_margin must be non-negative"
        )
    if emergency_bid_value is None:
        emergency_bid_value = env.config.bid_upper_bound
    if not 0 <= emergency_bid_value <= env.config.bid_upper_bound:
        raise ValueError(
            "charging_bc_emergency_bid_value must be between 0 and "
            "bid_upper_bound"
        )
    if updates == 0:
        return {"updates": 0}

    cfg = env.config
    device = env.device
    sampling_generator = env.gen if generator is None else generator
    denom = float(max(cfg.grid_size - 1, 1))
    capacity = int(cfg.battery_capacity)
    base_stations = env.recharge_station_pos
    parameters = list(agent.actor_shared.parameters()) + list(
        agent.bid_head.parameters()
    )
    if direction_loss_coef > 0:
        if agent.direction_actor is not None:
            parameters += list(agent.direction_actor.parameters())
        parameters += list(agent.direction_head.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate, eps=1e-5)
    direction_loss = torch.zeros((), device=device)
    bid_loss = torch.zeros((), device=device)
    direction_accuracy = 0.0
    bid_accuracy = 0.0

    for _ in range(updates):
        positions = torch.randint(
            0,
            cfg.grid_size,
            (batch_size, 2),
            generator=sampling_generator,
            device=device,
            dtype=torch.int32,
        )
        battery = torch.randint(
            1,
            capacity + 1,
            (batch_size,),
            generator=sampling_generator,
            device=device,
            dtype=torch.int32,
        )
        if cfg.moving_recharge_stations:
            stations = torch.randint(
                0,
                cfg.grid_size,
                (batch_size, env.num_recharge_stations, 2),
                generator=sampling_generator,
                device=device,
                dtype=torch.int32,
            )
        else:
            stations = base_stations.unsqueeze(0).expand(
                batch_size, -1, -1
            )
        relative = (
            stations.to(torch.float32)
            - positions.unsqueeze(1).to(torch.float32)
        ) / denom
        distances = relative.abs().sum(dim=-1) / 2.0
        nearest_idx = distances.argmin(dim=1)
        nearest_station = stations.gather(
            1, nearest_idx.view(-1, 1, 1).expand(-1, 1, 2)
        ).squeeze(1)
        delta = nearest_station - positions
        direction_target = torch.where(
            delta[:, 1] < 0,
            torch.zeros(batch_size, device=device, dtype=torch.long),
            torch.where(
                delta[:, 1] > 0,
                torch.ones(batch_size, device=device, dtype=torch.long),
                torch.where(
                    delta[:, 0] < 0,
                    torch.full(
                        (batch_size,), 2, device=device, dtype=torch.long
                    ),
                    torch.full(
                        (batch_size,), 3, device=device, dtype=torch.long
                    ),
                ),
            ),
        )
        physical_station_distance = (
            nearest_station - positions
        ).abs().sum(dim=1)
        station_energy_requirement = (
            physical_station_distance * int(cfg.movement_energy_cost)
        )
        if cfg.charging_activation_margin is None:
            charging_activation_limit = torch.full_like(
                battery, cfg.charging_low_battery_threshold
            )
        else:
            charging_activation_limit = (
                station_energy_requirement + cfg.charging_activation_margin
            )
        positive_bid_target = torch.full(
            (batch_size,), bid_value, device=device, dtype=torch.long
        )
        if emergency_margin is not None:
            emergency = battery <= (
                station_energy_requirement + emergency_margin
            )
            positive_bid_target = torch.where(
                emergency,
                torch.full_like(
                    positive_bid_target, emergency_bid_value
                ),
                positive_bid_target,
            )
        bid_target = torch.where(
            battery <= charging_activation_limit,
            positive_bid_target,
            torch.zeros(batch_size, device=device, dtype=torch.long),
        )
        window_remaining = torch.randint(
            0,
            max(cfg.action_window, 1),
            (batch_size,),
            generator=sampling_generator,
            device=device,
            dtype=torch.int32,
        )
        normalized_window = (
            window_remaining.to(torch.float32)
            / float(max(cfg.action_window, 1))
        ).unsqueeze(-1)
        charging_controls_window = (
            (window_remaining > 0)
            & (
                torch.rand(
                    batch_size,
                    generator=sampling_generator,
                    device=device,
                )
                < 0.5
            )
        ).to(torch.float32).unsqueeze(-1)
        obs = torch.cat(
            [
                positions.to(torch.float32) / denom,
                (battery.to(torch.float32) / capacity).unsqueeze(-1),
                *(
                    [
                        (
                            battery.to(torch.float32)
                            / float(
                                max(
                                    2
                                    * (cfg.grid_size - 1)
                                    * int(cfg.movement_energy_cost),
                                    1,
                                )
                            )
                        ).unsqueeze(-1),
                        (
                            battery.to(torch.float32)
                            - station_energy_requirement.to(torch.float32)
                        ).div(
                            float(
                                max(
                                    2
                                    * (cfg.grid_size - 1)
                                    * int(cfg.movement_energy_cost),
                                    1,
                                )
                            )
                        ).unsqueeze(-1),
                    ]
                    if cfg.charging_reserve_features_enabled
                    else []
                ),
                *(
                    [(nearest_station - positions).to(torch.float32) / denom]
                    if cfg.charging_nearest_station_features_enabled
                    else []
                ),
                relative.reshape(batch_size, -1),
                distances,
                normalized_window,
                charging_controls_window,
            ],
            dim=-1,
        )

        bid_features = agent.actor_shared(obs)
        direction_features = (
            agent.direction_actor(obs)
            if agent.direction_actor is not None
            else bid_features
        )
        direction_logits = agent.direction_head(direction_features)
        bid_logits = agent.bid_head(bid_features)
        direction_loss = F.cross_entropy(
            direction_logits, direction_target
        )
        bid_loss = F.cross_entropy(bid_logits, bid_target)
        loss = (
            direction_loss_coef * direction_loss
            + bid_loss_coef * bid_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        direction_accuracy = (
            direction_logits.argmax(dim=1) == direction_target
        ).to(torch.float32).mean().item()
        bid_accuracy = (
            bid_logits.argmax(dim=1) == bid_target
        ).to(torch.float32).mean().item()

    return {
        "updates": updates,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "bid_loss_coef": bid_loss_coef,
        "direction_loss_coef": direction_loss_coef,
        "bid_value": bid_value,
        "emergency_margin": emergency_margin,
        "emergency_bid_value": emergency_bid_value,
        "final_direction_loss": direction_loss.item(),
        "final_bid_loss": bid_loss.item(),
        "final_direction_accuracy": direction_accuracy,
        "final_bid_accuracy": bid_accuracy,
    }


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
        self.charging_agent = None
        self.charging_optimizer = None
        self.charging_obs_dim = None
        self._charging_metrics = None
        self.feeder_warm_start_report = None
        self.charging_bc_report = None
        self.charging_bc_refresh_report = None
        self.charging_bc_refresh_gen = None
        self._feeder_frozen_current = False

    def setup(self, *, skip_pretraining: bool = False):
        """Setup environments, agent, and optimizer.

        ``skip_pretraining`` is reserved for checkpoint loaders that only need to
        reconstruct the saved architecture before loading model state.
        """
        if self.args.charging_bc_refresh_updates < 0:
            raise ValueError(
                "charging_bc_refresh_updates must be non-negative"
            )
        if (
            self.args.charging_bc_refresh_learning_rate is not None
            and self.args.charging_bc_refresh_learning_rate <= 0
        ):
            raise ValueError(
                "charging_bc_refresh_learning_rate must be positive"
            )
        if self.args.feeder_yield_aux_coef < 0:
            raise ValueError(
                "feeder_yield_aux_coef must be non-negative"
            )
        if (
            self.args.charging_ppo_bid_only
            and not self.args.charging_separate_direction_actor
        ):
            raise ValueError(
                "charging_ppo_bid_only requires "
                "charging_separate_direction_actor"
            )
        if (
            self.args.charging_ppo_bid_only
            and not self.args.charging_programmatic_navigation
            and self.args.charging_bc_updates <= 0
            and not skip_pretraining
        ):
            raise ValueError(
                "learned bid-only charger PPO requires charging_bc_updates"
            )
        feeder_yield_margin = getattr(
            self.args, "feeder_yield_activation_margin", None
        )
        if feeder_yield_margin is not None and feeder_yield_margin < 0:
            raise ValueError(
                "feeder_yield_activation_margin must be non-negative"
            )
        if self.args.bid_only_ppo and self.args.programmatic_bidding != "none":
            raise ValueError(
                "bid_only_ppo requires policy-generated bids "
                "(programmatic_bidding='none')"
            )
        if self.args.bid_only_ppo and not self.args.policy_warm_start_checkpoint:
            raise ValueError(
                "bid_only_ppo requires policy_warm_start_checkpoint"
            )
        if self.args.bid_credit_assignment not in {
            "individual",
            "controller_team",
            "shared_team",
            "mixed_team",
        }:
            raise ValueError(
                "bid_credit_assignment must be 'individual', "
                "'controller_team', 'shared_team', or 'mixed_team'"
            )
        if not 0.0 <= self.args.bid_other_reward_fraction <= 1.0:
            raise ValueError(
                "bid_other_reward_fraction must be between 0 and 1"
            )
        if self.args.factorized_auction_ppo and self.args.window_bidding:
            raise ValueError(
                "factorized_auction_ppo requires window_bidding=False"
            )
        if self.args.factorized_auction_ppo and self.args.bid_only_ppo:
            raise ValueError(
                "factorized_auction_ppo is an end-to-end training mode"
            )
        if self.args.factorized_auction_ppo and self.args.charging_agent_enabled:
            raise ValueError(
                "factorized_auction_ppo currently supports target bidders only"
            )
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
            use_target_priorities=self.args.use_target_priorities,
            programmatic_bidding=self.args.programmatic_bidding,
            battery_capacity=self.args.battery_capacity,
            recharge_station_positions=self.args.recharge_station_positions,
            moving_recharge_stations=self.args.moving_recharge_stations,
            recharge_station_direction_change_prob=(
                self.args.recharge_station_direction_change_prob
            ),
            recharge_station_move_interval=(
                self.args.recharge_station_move_interval
            ),
            movement_energy_cost=self.args.movement_energy_cost,
            battery_depletion_penalty=self.args.battery_depletion_penalty,
            charging_agent_enabled=self.args.charging_agent_enabled,
            charging_low_battery_threshold=self.args.charging_low_battery_threshold,
            charging_distance_reward_scale=self.args.charging_distance_reward_scale,
            charging_recharge_bonus=self.args.charging_recharge_bonus,
            charging_depletion_penalty=self.args.charging_depletion_penalty,
            charging_high_battery_control_penalty=(
                self.args.charging_high_battery_control_penalty
            ),
            feeder_low_battery_control_penalty=(
                self.args.feeder_low_battery_control_penalty
            ),
            charging_low_battery_bid_boost=(
                self.args.charging_low_battery_bid_boost
            ),
            charging_bid_boost_threshold=(
                self.args.charging_bid_boost_threshold
            ),
            charging_activation_margin=self.args.charging_activation_margin,
            charging_release_window_on_recharge=(
                self.args.charging_release_window_on_recharge
            ),
            charging_programmatic_navigation=(
                self.args.charging_programmatic_navigation
            ),
            charging_reserve_features_enabled=(
                self.args.charging_reserve_features_enabled
            ),
            charging_nearest_station_features_enabled=(
                self.args.charging_nearest_station_features_enabled
            ),
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
            energy_feature_dim=self.envs.energy_feature_dim,
            use_target_priorities=self.args.use_target_priorities,
            separate_bid_actor=self.args.separate_bid_actor,
            bid_actor_hidden_sizes=self.args.bid_actor_hidden_sizes,
            ordinal_bid_head=self.args.ordinal_bid_head,
        ).to(self.device)
        self.agent.set_bid_head(self.args.bid_upper_bound)
        if self.args.window_bidding:
            self.agent.set_window_head(self.args.action_window)
        if self.args.feeder_warm_start_checkpoint:
            if not self.args.use_target_attention_pooling:
                raise ValueError(
                    "Feeder warm starts with changed battery observations require "
                    "use_target_attention_pooling=True"
                )
            self.feeder_warm_start_report = (
                load_feeder_checkpoint_with_energy_expansion(
                    self.agent,
                    self.args.feeder_warm_start_checkpoint,
                    self.envs.energy_feature_dim,
                )
            )
            print(
                "   Feeder warm start: "
                f"{self.feeder_warm_start_report['checkpoint']}"
            )

        self.policy_warm_start_report = None
        if self.args.policy_warm_start_checkpoint:
            self.policy_warm_start_report = load_policy_warm_start(
                self.agent,
                self.args.policy_warm_start_checkpoint,
                reset_bid_policy=self.args.separate_bid_actor,
            )
            print(
                "   Policy warm start: "
                f"{self.policy_warm_start_report['checkpoint']}"
            )

        if self.args.bid_only_ppo and self.args.freeze_navigation_during_bid_only:
            trainable_prefixes = ("bid_actor.", "bid_head.", "critic.")
            for name, parameter in self.agent.named_parameters():
                parameter.requires_grad_(name.startswith(trainable_prefixes))
        trainable_parameters = [
            parameter
            for parameter in self.agent.parameters()
            if parameter.requires_grad
        ]
        self.optimizer = optim.Adam(
            trainable_parameters, lr=self.args.learning_rate, eps=1e-5
        )
        if self.args.programmatic_bidding != "none":
            self.policy_action_value_fn = self.agent.get_direction_action_and_value
        elif self.args.bid_only_ppo:
            self.policy_action_value_fn = (
                self.agent.get_bid_action_and_value_with_direction
            )
        else:
            self.policy_action_value_fn = self.agent.get_action_and_value
        if self.args.charging_agent_enabled:
            self.charging_obs_dim = self.envs.charging_obs_dim
            self.charging_agent = SharedAgent(
                self.charging_obs_dim,
                num_actions_per_agent=num_actions_per_agent,
                window_bidding=self.args.window_bidding,
                actor_hidden_sizes=self.args.charging_actor_hidden_sizes,
                critic_hidden_sizes=self.args.charging_critic_hidden_sizes,
                separate_direction_actor=(
                    self.args.charging_separate_direction_actor
                ),
            ).to(self.device)
            self.charging_agent.set_bid_head(self.args.bid_upper_bound)
            if self.args.window_bidding:
                self.charging_agent.set_window_head(self.args.action_window)
            self.charging_bc_report = pretrain_charging_policy(
                self.charging_agent,
                self.envs,
                updates=self.args.charging_bc_updates,
                batch_size=self.args.charging_bc_batch_size,
                learning_rate=self.args.charging_bc_learning_rate,
                bid_loss_coef=self.args.charging_bc_bid_loss_coef,
                bid_value=self.args.charging_bc_bid_value,
                emergency_margin=(
                    self.args.charging_bc_emergency_margin
                ),
                emergency_bid_value=(
                    self.args.charging_bc_emergency_bid_value
                ),
            )
            self.charging_bc_refresh_gen = torch.Generator(device=self.device)
            self.charging_bc_refresh_gen.manual_seed(
                self.args.seed + 10_000_019
            )
            if self.args.charging_bc_updates:
                print(
                    "   Charging BC direction accuracy: "
                    f"{self.charging_bc_report['final_direction_accuracy']:.3f}"
                )
            charging_lr = (
                self.args.charging_learning_rate
                if self.args.charging_learning_rate is not None
                else self.args.learning_rate
            )
            self.charging_optimizer = optim.Adam(
                self.charging_agent.parameters(), lr=charging_lr, eps=1e-5
            )

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
        else:
            print(f"   Fixed action window: {self.args.action_window}")
        print(f"   Target attention pooling: {self.args.use_target_attention_pooling}")
        if self.args.use_target_attention_pooling:
            layout = "centralized" if self.args.visible_targets is None else "visible"
            print(f"   Attention layout: {layout}")
            print(f"   Target embed dim: {self.args.target_embed_dim}")
        print(f"   Actions per agent: {num_actions_per_agent}")
        print(f"   Separate charging agent: {self.args.charging_agent_enabled}")
        if self.args.charging_agent_enabled:
            print(f"   Charging observation dim: {self.charging_obs_dim}")
            print(
                "   Programmatic charging navigation: "
                f"{self.args.charging_programmatic_navigation}"
            )
            print(
                "   Separate charging direction actor: "
                f"{self.args.charging_separate_direction_actor}"
            )
            print(
                "   Charging PPO bid only: "
                f"{self.args.charging_ppo_bid_only}"
            )
            print(
                "   Feeder freeze iterations: "
                f"{self.args.feeder_freeze_iterations}"
            )
        print(f"   Batch size: {self.args.batch_size}")
        print(f"   Num iterations: {self.args.num_iterations}")
        print(f"   Run name: {self.run_name}")

    def train(self, start_iteration: int = 1, initial_global_step: int = 0):
        if not self.args.charging_agent_enabled:
            return super().train(start_iteration, initial_global_step)
        return self._train_with_charging_agent(start_iteration, initial_global_step)

    def _train_with_charging_agent(
        self, start_iteration: int, initial_global_step: int
    ):
        """Train feeder and charging policies from separate PPO reward streams."""
        if self.envs is None:
            raise RuntimeError("Must call setup() before train()")

        args = self.args
        if not 0 <= args.feeder_freeze_iterations <= args.num_iterations:
            raise ValueError(
                "feeder_freeze_iterations must be between 0 and num_iterations"
            )
        T, N, A = args.num_steps, args.num_envs, args.num_agents
        C = self.num_action_components
        feeder_batch_size = T * N * A
        charging_batch_size = T * N
        feeder_minibatch_size = feeder_batch_size // args.num_minibatches
        charging_minibatch_size = max(
            1, charging_batch_size // args.num_minibatches
        )
        args.batch_size = feeder_batch_size
        args.minibatch_size = feeder_minibatch_size
        args.total_timesteps = (
            args.num_iterations * T * N * self.envs.num_bidders
        )

        feeder_obs = torch.zeros(
            (T, N, A, self.obs_dim), device=self.device
        )
        charging_obs = torch.zeros(
            (T, N, self.charging_obs_dim), device=self.device
        )
        feeder_actions = torch.zeros((T, N, A, C), device=self.device)
        charging_actions = torch.zeros((T, N, C), device=self.device)
        feeder_logprobs = torch.zeros((T, N, A), device=self.device)
        charging_logprobs = torch.zeros((T, N), device=self.device)
        feeder_rewards = torch.zeros((T, N, A), device=self.device)
        charging_rewards = torch.zeros((T, N), device=self.device)
        feeder_dones = torch.zeros((T, N, A), device=self.device)
        charging_dones = torch.zeros((T, N), device=self.device)
        feeder_values = torch.zeros((T, N, A), device=self.device)
        charging_values = torch.zeros((T, N), device=self.device)

        global_step = initial_global_step
        start_time = time.time()
        next_feeder_obs, _ = self.envs.reset(seed=args.seed)
        next_charging_obs = self.envs.get_charging_observation()
        next_feeder_done = torch.zeros((N, A), device=self.device)
        next_charging_done = torch.zeros(N, device=self.device)

        for iteration in range(start_iteration, args.num_iterations + 1):
            iteration_start = time.time()
            self._on_iteration_start(iteration)
            feeder_frozen = iteration <= args.feeder_freeze_iterations
            self._feeder_frozen_current = feeder_frozen
            self.agent.requires_grad_(not feeder_frozen)

            feeder_base_lr = (
                args.feeder_finetune_learning_rate
                if args.feeder_finetune_learning_rate is not None
                else args.learning_rate
            )
            charging_base_lr = (
                args.charging_learning_rate
                if args.charging_learning_rate is not None
                else args.learning_rate
            )
            if args.anneal_lr:
                joint_iterations = max(
                    args.num_iterations - args.feeder_freeze_iterations, 1
                )
                joint_iteration = max(
                    iteration - args.feeder_freeze_iterations - 1, 0
                )
                feeder_frac = max(
                    0.0, 1.0 - joint_iteration / joint_iterations
                )
                feeder_lr = args.lr_min + feeder_frac * (
                    feeder_base_lr - args.lr_min
                )
                charging_frac = (
                    1.0 - (iteration - 1.0) / args.num_iterations
                )
                charging_lr = args.lr_min + charging_frac * (
                    charging_base_lr - args.lr_min
                )
            else:
                feeder_lr = feeder_base_lr
                charging_lr = charging_base_lr
            self.optimizer.param_groups[0]["lr"] = (
                0.0 if feeder_frozen else feeder_lr
            )
            self.charging_optimizer.param_groups[0]["lr"] = charging_lr

            for step in range(T):
                global_step += N * self.envs.num_bidders
                feeder_obs[step] = next_feeder_obs
                charging_obs[step] = next_charging_obs
                feeder_dones[step] = next_feeder_done
                charging_dones[step] = next_charging_done

                with torch.no_grad():
                    f_action, f_logprob, _, f_value = (
                        self.agent.get_action_and_value(
                            next_feeder_obs.reshape(-1, self.obs_dim)
                        )
                    )
                    f_action = f_action.reshape(N, A, C)
                    f_logprob = f_logprob.reshape(N, A)
                    f_value = f_value.reshape(N, A)
                    charging_action_fn = (
                        self.charging_agent.get_bid_action_and_value
                        if args.charging_programmatic_navigation
                        else self.charging_agent.get_action_and_value
                    )
                    c_action, c_logprob, _, c_value = (
                        charging_action_fn(next_charging_obs)
                    )
                    if (
                        args.charging_ppo_bid_only
                        and not args.charging_programmatic_navigation
                    ):
                        _, c_logprob, _, _ = (
                            self.charging_agent.get_bid_action_and_value(
                                next_charging_obs,
                                action=c_action,
                            )
                        )

                feeder_actions[step] = f_action
                charging_actions[step] = c_action
                feeder_logprobs[step] = f_logprob
                charging_logprobs[step] = c_logprob
                feeder_values[step] = f_value
                charging_values[step] = c_value.view(-1)

                joint_action = torch.cat(
                    [f_action, c_action.unsqueeze(1)], dim=1
                )
                next_feeder_obs, reward, terminations, truncations, infos = (
                    self.envs.step(joint_action)
                )
                next_charging_obs = self.envs.get_charging_observation()
                next_done = (terminations | truncations).to(
                    self.device, dtype=torch.float32
                )
                next_feeder_done = next_done.unsqueeze(1).expand(-1, A)
                next_charging_done = next_done
                feeder_rewards[step] = reward[:, :A]
                charging_rewards[step] = reward[:, A]
                self._on_rollout_step(infos, global_step)
                if torch.any(next_done > 0):
                    next_feeder_obs = self.envs.partial_reset(
                        next_done.bool()
                    )
                    next_charging_obs = (
                        self.envs.get_charging_observation()
                    )

            with torch.no_grad():
                next_feeder_value = self.agent.get_value(
                    next_feeder_obs.reshape(-1, self.obs_dim)
                ).reshape(N, A)
                next_charging_value = self.charging_agent.get_value(
                    next_charging_obs
                ).reshape(N)
                feeder_advantages, feeder_returns = compute_gae(
                    feeder_rewards,
                    feeder_values,
                    feeder_dones,
                    next_feeder_value,
                    next_feeder_done,
                    args.gamma,
                    args.gae_lambda,
                )
                charging_advantages, charging_returns = compute_gae(
                    charging_rewards,
                    charging_values,
                    charging_dones,
                    next_charging_value,
                    next_charging_done,
                    args.gamma,
                    args.gae_lambda,
                )

            self._last_rollout_stats = {
                "rewards": feeder_rewards.detach(),
                "values": feeder_values.detach(),
                "advantages": feeder_advantages.detach(),
            }
            if feeder_frozen:
                feeder_metrics = {
                    "v_loss": 0.0,
                    "pg_loss": 0.0,
                    "entropy_loss": 0.0,
                    "old_approx_kl": 0.0,
                    "approx_kl": 0.0,
                    "clipfrac": 0.0,
                }
                feeder_clipfracs = []
            else:
                feeder_metrics, feeder_clipfracs = self._update_policy(
                    self.agent,
                    self.optimizer,
                    feeder_obs.reshape(-1, self.obs_dim),
                    feeder_actions.reshape(-1, C),
                    feeder_logprobs.reshape(-1),
                    feeder_advantages.reshape(-1),
                    feeder_returns.reshape(-1),
                    feeder_values.reshape(-1),
                    feeder_batch_size,
                    feeder_minibatch_size,
                    auxiliary_loss_fn=(
                        self._feeder_yield_aux_loss
                        if args.feeder_yield_aux_coef > 0
                        else None
                    ),
                )
            charging_metrics, _ = self._update_policy(
                self.charging_agent,
                self.charging_optimizer,
                charging_obs.reshape(-1, self.charging_obs_dim),
                charging_actions.reshape(-1, C),
                charging_logprobs.reshape(-1),
                charging_advantages.reshape(-1),
                charging_returns.reshape(-1),
                charging_values.reshape(-1),
                charging_batch_size,
                charging_minibatch_size,
                bidding_only=(
                    args.charging_programmatic_navigation
                    or args.charging_ppo_bid_only
                ),
            )
            if args.charging_bc_refresh_updates > 0:
                refresh_lr = (
                    args.charging_bc_refresh_learning_rate
                    if args.charging_bc_refresh_learning_rate is not None
                    else args.charging_bc_learning_rate
                )
                self.charging_bc_refresh_report = (
                    pretrain_charging_policy(
                        self.charging_agent,
                        self.envs,
                        updates=args.charging_bc_refresh_updates,
                        batch_size=args.charging_bc_batch_size,
                        learning_rate=refresh_lr,
                        bid_loss_coef=args.charging_bc_bid_loss_coef,
                        bid_value=args.charging_bc_bid_value,
                        emergency_margin=(
                            args.charging_bc_emergency_margin
                        ),
                        emergency_bid_value=(
                            args.charging_bc_emergency_bid_value
                        ),
                        direction_loss_coef=(
                            0.0
                            if args.charging_programmatic_navigation
                            else 1.0
                        ),
                        generator=self.charging_bc_refresh_gen,
                    )
                )
            self._charging_metrics = {
                **charging_metrics,
                "mean_reward": charging_rewards.mean().item(),
                "mean_advantage": charging_advantages.mean().item(),
                "explained_variance": compute_explained_variance(
                    charging_values.reshape(-1).detach().cpu().numpy(),
                    charging_returns.reshape(-1).detach().cpu().numpy(),
                ),
            }

            iter_time = time.time() - iteration_start
            eta = format_duration(
                (args.num_iterations - iteration) * iter_time
            )
            sps = int(global_step / (time.time() - start_time))
            feeder_loss_text = (
                "frozen"
                if feeder_frozen
                else f"{feeder_metrics['pg_loss']:.4f}"
            )
            print(
                f"Iteration {iteration}/{args.num_iterations} - SPS: {sps} - "
                f"Feeder Loss: {feeder_loss_text} - "
                f"Charging Loss: {charging_metrics['pg_loss']:.4f} - "
                f"Charging Reward: {charging_rewards.mean().item():.4f} - "
                f"Iter Time: {format_duration(iter_time)} - ETA: {eta}"
            )

            self._maybe_log_iteration(
                global_step, feeder_metrics, feeder_clipfracs, start_time
            )
            if args.track:
                wandb.log(
                    {
                        "losses/explained_variance": compute_explained_variance(
                            feeder_values.reshape(-1).detach().cpu().numpy(),
                            feeder_returns.reshape(-1).detach().cpu().numpy(),
                        ),
                        "charts/iteration": iteration,
                        "training/feeder_frozen": float(feeder_frozen),
                        "training/feeder_learning_rate": (
                            self.optimizer.param_groups[0]["lr"]
                        ),
                        "training/charging_learning_rate": (
                            self.charging_optimizer.param_groups[0]["lr"]
                        ),
                        "training/charging_bc_refresh_updates": (
                            args.charging_bc_refresh_updates
                        ),
                    },
                    step=global_step,
                )
            if self.callbacks.get("on_iteration_end"):
                self.callbacks["on_iteration_end"](
                    self, iteration, global_step
                )
            self._on_iteration_end(iteration, global_step)

        if self.callbacks.get("on_training_end"):
            self.callbacks["on_training_end"](self, global_step)

    def _update_policy(
        self,
        agent,
        optimizer,
        obs,
        actions,
        old_logprobs,
        advantages,
        returns,
        old_values,
        batch_size,
        minibatch_size,
        bidding_only: bool = False,
        auxiliary_loss_fn=None,
    ):
        clipfracs = []
        metrics = None
        for _ in range(self.args.update_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = indices[start:start + minibatch_size]
                metrics = ppo_update_step(
                    agent,
                    optimizer,
                    obs[mb_inds],
                    actions[mb_inds],
                    old_logprobs[mb_inds],
                    advantages[mb_inds],
                    returns[mb_inds],
                    old_values[mb_inds],
                    self.args.clip_coef,
                    self.args.ent_coef,
                    self.args.vf_coef,
                    self.args.max_grad_norm,
                    self.args.norm_adv,
                    self.args.clip_vloss,
                    action_value_fn=(
                        agent.get_bid_action_and_value
                        if bidding_only
                        else None
                    ),
                    auxiliary_loss_fn=auxiliary_loss_fn,
                )
                clipfracs.append(metrics["clipfrac"])
            if (
                self.args.target_kl is not None
                and metrics["approx_kl"] > self.args.target_kl
            ):
                break
        return metrics, clipfracs

    def _feeder_yield_aux_loss(self, agent, obs):
        """Teach feeders to bid zero only when charging is active."""
        energy_start = obs.shape[1] - 1 - self.envs.energy_feature_dim
        energy = obs[:, energy_start:-1]
        battery = (
            energy[:, 0] * float(self.args.battery_capacity)
        )
        denom = float(max(self.args.grid_size - 1, 1))
        position = obs[:, :2] * denom
        stations = (
            energy[:, 1:]
            .reshape(-1, self.envs.num_recharge_stations, 2)
            * denom
        )
        nearest_distance = (
            stations - position.unsqueeze(1)
        ).abs().sum(dim=-1).min(dim=1).values
        nearest_energy_requirement = (
            nearest_distance
            * float(getattr(self.args, "movement_energy_cost", 1))
        )
        feeder_yield_margin = getattr(
            self.args, "feeder_yield_activation_margin", None
        )
        if feeder_yield_margin is None:
            feeder_yield_margin = self.args.charging_activation_margin
        if feeder_yield_margin is None:
            activation_limit = torch.full_like(
                battery,
                float(self.args.charging_low_battery_threshold),
            )
        else:
            activation_limit = (
                nearest_energy_requirement
                + float(feeder_yield_margin)
            )
        active = battery <= activation_limit
        if not torch.any(active):
            return torch.zeros((), device=obs.device)

        encoded = agent._encode_obs(obs[active])
        features = agent.actor_shared(encoded)
        if self.args.feeder_yield_aux_bid_head_only:
            features = features.detach()
        bid_logits = agent.bid_head(features)
        yield_target = torch.zeros(
            bid_logits.shape[0],
            device=obs.device,
            dtype=torch.long,
        )
        return self.args.feeder_yield_aux_coef * F.cross_entropy(
            bid_logits, yield_target
        )

    def _on_iteration_start(self, iteration: int):
        if not self.args.track:
            return
        self._episode_agent_wins = torch.zeros(
            self.envs.num_bidders, device=self.device, dtype=torch.int64
        )
        self._episode_bid_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        self._episode_bid_count = torch.zeros((), device=self.device, dtype=torch.int64)
        self._episode_bid_min = None
        self._episode_bid_max = None
        self._episode_reward_no_bid_sum = torch.zeros((), device=self.device, dtype=torch.float32)

    def _on_rollout_step(self, infos, global_step: int):
        if not self.args.track or not isinstance(infos, dict):
            return
        winning_agent = infos.get('winning_agent', None)
        if torch.is_tensor(winning_agent):
            valid = winning_agent >= 0
            if torch.any(valid):
                counts = torch.bincount(
                    winning_agent[valid].to(torch.int64),
                    minlength=self.envs.num_bidders,
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

        reward_no_bid_sum = infos.get('reward_no_bid_sum', None)
        if torch.is_tensor(reward_no_bid_sum):
            self._episode_reward_no_bid_sum += reward_no_bid_sum.sum()

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
        if self.args.track and hasattr(self, '_episode_reward_no_bid_sum'):
            n = self.args.num_envs * self.args.num_steps * self.envs.num_bidders
            log_dict['rewards/avg_step_reward_no_bid'] = (self._episode_reward_no_bid_sum / n).item()
        total_wins = int(self._episode_agent_wins.sum().item()) if self._episode_agent_wins is not None else 0
        if total_wins > 0:
            for agent_idx in range(self.args.num_agents):
                agent_key = f'agent_{agent_idx}'
                win_rate = self._episode_agent_wins[agent_idx].item() / total_wins
                log_dict[f'agents/{agent_key}_win_rate'] = win_rate
            if self.args.charging_agent_enabled:
                log_dict["charging/win_rate"] = (
                    self._episode_agent_wins[self.envs.charging_agent_idx].item()
                    / total_wins
                )
        if self._charging_metrics:
            for key, value in self._charging_metrics.items():
                log_dict[f"charging/{key}"] = value
        return log_dict

    def save_model(self, path: Optional[str] = None):
        """Save the trained model."""
        if path is None:
            path = f"models/{self.run_name}"
        os.makedirs(path, exist_ok=True)
        torch.save(self.agent.state_dict(), f"{path}/agent.pt")
        if self.charging_agent is not None:
            torch.save(
                self.charging_agent.state_dict(),
                f"{path}/charging_agent.pt",
            )
        print(f"✅ Model saved to {path}/agent.pt")

        if self.args.track:
            wandb.save(f"{path}/agent.pt")
            if self.charging_agent is not None:
                wandb.save(f"{path}/charging_agent.pt")

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
