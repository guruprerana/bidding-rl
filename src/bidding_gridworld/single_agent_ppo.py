# Single-agent PPO implementation for BiddingGridworld in single_agent_mode
# Simpler than multi-agent version - just learns to navigate and collect all targets

import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import wandb

from bidding_gridworld.bidding_gridworld_torch import (
    BiddingGridworld,
    BiddingGridworldConfig,
)
from ppo_utils import layer_init, MaskedAttentionPooling
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
    urgency_weighted_scalarization: bool = False
    """weight dense per-target rewards by normalized inverse time-to-live"""
    use_target_priorities: bool = True
    """include sampled target priorities in observations and completion rewards"""
    moving_targets: bool = False
    """whether to use moving targets variant"""
    direction_change_prob: float = 0.1
    """probability of target direction change (for moving targets)"""
    target_move_interval: int = 1
    """steps between target movements (for moving targets)"""
    nearest_target_shaping: bool = False
    """whether to shape reward toward nearest unreached target instead of summing over all"""
    nearest_expiry_shaping: bool = False
    """whether to shape reward toward the unreached target nearest to expiry instead of summing over all"""
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
    """penalty applied when depletion triggers a tow"""
    charging_low_battery_threshold: int = 20
    """battery threshold used to scale assisted charging rewards"""
    charging_distance_reward_scale: float = 0.0
    """reserve-aware reward scale for progress toward a recharge station"""
    charging_recharge_bonus: float = 0.0
    """assisted reward for recharging below the low-battery threshold"""
    charging_depletion_penalty: float = 0.0
    """additional assisted penalty when the battery depletes"""
    charging_activation_margin: Optional[int] = None
    """activate charging support when reserve is within this margin"""
    charging_bc_updates: int = 0
    """number of station-navigation behavior-cloning updates before PPO"""
    charging_bc_batch_size: int = 4096
    """number of synthetic unsafe-reserve observations per BC update"""
    charging_bc_learning_rate: float = 3e-3
    """learning rate for charging navigation behavior cloning"""

    # Network architecture
    actor_hidden_sizes: Tuple[int, ...] = (128, 128, 128)
    """hidden layer sizes for the actor network"""
    critic_hidden_sizes: Tuple[int, ...] = (256, 256, 256)
    """hidden layer sizes for the critic network"""
    use_target_attention_pooling: bool = False
    """whether to use masked attention pooling over target observations"""
    target_embed_dim: int = 64
    """embedding dimension for target attention pooling"""
    target_encoder_hidden_sizes: Optional[Tuple[int, ...]] = None
    """hidden sizes for the target encoder MLP (defaults to (64, 64))"""

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


class SingleAgent(nn.Module):
    """
    Actor-critic network for single-agent navigation.

    Only outputs direction actions (no bidding). Optionally uses masked
    attention pooling over per-target features for variable target counts.

    Single-agent observation layout (with include_target_reached):
      [agent_pos(2), target_pos(2*T), targets_reached(T),
       target_counters(T), target_priorities(T), window_steps(1), relative_counts(T)]
    Without include_target_reached (moving targets):
      [agent_pos(2), target_pos(2*T), target_counters(T),
       target_priorities(T), window_steps(1), relative_counts(T)]
    """

    def __init__(
        self,
        obs_dim,
        num_targets,
        actor_hidden_sizes=None,
        critic_hidden_sizes=None,
        use_target_attention_pooling: bool = False,
        target_embed_dim: int = 64,
        target_encoder_hidden_sizes=None,
        include_target_reached: bool = True,
        include_target_priority: bool = True,
        energy_feature_dim: int = 0,
    ):
        super().__init__()
        self.use_target_attention_pooling = use_target_attention_pooling
        self.include_target_reached = include_target_reached
        self.include_target_priority = include_target_priority
        self.num_targets = num_targets
        self.energy_feature_dim = energy_feature_dim

        actor_sizes = list(actor_hidden_sizes) if actor_hidden_sizes is not None else [128, 128, 128]
        critic_sizes = list(critic_hidden_sizes) if critic_hidden_sizes is not None else [256, 256, 256]

        if self.use_target_attention_pooling:
            encoder_sizes = target_encoder_hidden_sizes if target_encoder_hidden_sizes is not None else (64, 64)
            # Single-agent has an extra relative_count feature compared to multi-agent
            target_feat_dim = (
                7 if self.include_target_reached else 6
            ) + int(self.include_target_priority)
            self.target_pool = MaskedAttentionPooling(
                input_dim=target_feat_dim,
                embed_dim=target_embed_dim,
                hidden_sizes=encoder_sizes,
            )
            self.encoded_obs_dim = (
                3 + self.energy_feature_dim + target_feat_dim + target_embed_dim
            )
        else:
            self.encoded_obs_dim = obs_dim

        # Critic network: outputs single value estimate
        critic_layers = []
        critic_in_dim = self.encoded_obs_dim
        for hidden_size in critic_sizes:
            critic_layers.append(layer_init(nn.Linear(critic_in_dim, hidden_size)))
            critic_layers.append(nn.ELU())
            critic_in_dim = hidden_size
        critic_layers.append(layer_init(nn.Linear(critic_in_dim, 1), std=1.0))
        self.critic = nn.Sequential(*critic_layers)

        # Actor network: outputs logits for direction (4 actions)
        actor_layers = []
        actor_in_dim = self.encoded_obs_dim
        for hidden_size in actor_sizes:
            actor_layers.append(layer_init(nn.Linear(actor_in_dim, hidden_size)))
            actor_layers.append(nn.ELU())
            actor_in_dim = hidden_size
        actor_layers.append(layer_init(nn.Linear(actor_in_dim, 4), std=0.01))  # 4 directions
        self.actor = nn.Sequential(*actor_layers)

    def _encode_obs(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_target_attention_pooling:
            return x

        if x.dim() == 1:
            x = x.unsqueeze(0)

        T = self.num_targets
        agent_pos = x[:, :2]  # (B, 2)
        energy_features = (
            x[:, -self.energy_feature_dim:]
            if self.energy_feature_dim > 0
            else x[:, :0]
        )

        cursor = 2
        target_pos = x[:, cursor:cursor + 2 * T].reshape(-1, T, 2)
        cursor += 2 * T
        target_parts = [target_pos, target_pos - agent_pos.unsqueeze(1)]
        if self.include_target_reached:
            target_parts.append(x[:, cursor:cursor + T].reshape(-1, T, 1))
            cursor += T
        target_parts.append(x[:, cursor:cursor + T].reshape(-1, T, 1))
        cursor += T
        if self.include_target_priority:
            target_parts.append(x[:, cursor:cursor + T].reshape(-1, T, 1))
            cursor += T
        window_steps = x[:, cursor:cursor + 1]
        cursor += 1
        target_parts.append(x[:, cursor:cursor + T].reshape(-1, T, 1))
        target_feats = torch.cat(target_parts, dim=-1)

        pooled = self.target_pool(target_feats)          # (B, embed_dim)
        own_feats = target_feats[:, 0, :]               # (B, target_feat_dim)
        return torch.cat(
            [agent_pos, energy_features, window_steps, own_feats, pooled], dim=-1
        )

    def get_value(self, x):
        """Get value estimate for given observation."""
        return self.critic(self._encode_obs(x))

    def get_action_and_value(self, x, action=None, deterministic=False):
        """
        Get action and value for given observation.

        Args:
            x: Observation tensor (can be batched)
            action: If provided, compute log prob for this action. Otherwise sample new action.
            deterministic: When no action is provided, choose the highest-logit
                action for evaluation instead of sampling.

        Returns:
            action: Sampled or provided action (direction only)
            log_prob: Log probability of the action
            entropy: Entropy of the action distribution
            value: Value estimate
        """
        x = self._encode_obs(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else probs.sample()
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        value = self.critic(x)
        return action, log_prob, entropy, value


def pretrain_single_agent_charging_navigation(
    agent: SingleAgent,
    env: BiddingGridworld,
    updates: int,
    batch_size: int,
    learning_rate: float,
    activation_margin: int,
    seed: int,
) -> Dict:
    """Clone nearest-station directions on unsafe-reserve observations."""
    if updates < 0:
        raise ValueError("charging_bc_updates must be non-negative")
    if batch_size <= 0:
        raise ValueError("charging_bc_batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("charging_bc_learning_rate must be positive")
    if activation_margin < 0:
        raise ValueError("charging activation margin must be non-negative")
    if updates == 0:
        return {"updates": 0}
    if not env.battery_enabled:
        raise ValueError("charging navigation BC requires battery_capacity")

    bc_env = BiddingGridworld(
        env.config,
        num_envs=batch_size,
        device=env.device,
        seed=seed,
    )
    parameters = list(agent.actor.parameters())
    if agent.use_target_attention_pooling:
        parameters += list(agent.target_pool.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate, eps=1e-5)
    final_loss = 0.0
    final_accuracy = 0.0
    final_active_fraction = 0.0

    try:
        for _ in range(updates):
            bc_env.reset()
            bc_env.agent_pos = torch.randint(
                0,
                env.config.grid_size,
                (batch_size, 2),
                generator=bc_env.gen,
                device=env.device,
                dtype=torch.int32,
            )
            bc_env.battery_level = torch.randint(
                1,
                int(env.config.battery_capacity) + 1,
                (batch_size,),
                generator=bc_env.gen,
                device=env.device,
                dtype=torch.int32,
            )
            if env.config.moving_recharge_stations:
                bc_env.current_recharge_station_pos = torch.randint(
                    0,
                    env.config.grid_size,
                    (batch_size, bc_env.num_recharge_stations, 2),
                    generator=bc_env.gen,
                    device=env.device,
                    dtype=torch.int32,
                )

            energy_required = bc_env._nearest_recharge_energy_requirement(
                bc_env.agent_pos
            )
            active = (
                bc_env.battery_level <= energy_required + activation_margin
            ) & (energy_required > 0)
            obs = bc_env._get_observation()
            targets = bc_env._direction_to_nearest_recharge_station(
                bc_env.agent_pos
            )
            logits = agent.actor(agent._encode_obs(obs))
            if not torch.any(active):
                continue
            loss = F.cross_entropy(logits[active], targets[active])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                final_loss = float(loss.item())
                final_accuracy = float(
                    (logits[active].argmax(dim=-1) == targets[active])
                    .to(torch.float32)
                    .mean()
                    .item()
                )
                final_active_fraction = float(
                    active.to(torch.float32).mean().item()
                )
    finally:
        bc_env.close()

    return {
        "updates": updates,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "activation_margin": activation_margin,
        "final_loss": final_loss,
        "final_direction_accuracy": final_accuracy,
        "final_active_fraction": final_active_fraction,
    }


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
        self.charging_bc_report = None

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
            urgency_weighted_scalarization=(
                self.args.urgency_weighted_scalarization
            ),
            use_target_priorities=self.args.use_target_priorities,
            nearest_target_shaping=self.args.nearest_target_shaping,
            nearest_expiry_shaping=self.args.nearest_expiry_shaping,
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
            charging_low_battery_threshold=(
                self.args.charging_low_battery_threshold
            ),
            charging_distance_reward_scale=(
                self.args.charging_distance_reward_scale
            ),
            charging_recharge_bonus=self.args.charging_recharge_bonus,
            charging_depletion_penalty=(
                self.args.charging_depletion_penalty
            ),
            charging_activation_margin=self.args.charging_activation_margin,
        )
        self.envs = BiddingGridworld(
            env_config,
            num_envs=self.args.num_envs,
            device=self.device,
            seed=self.args.seed,
        )

        # Create agent
        self.obs_dim = self.envs.obs_dim
        include_reached = not self.args.moving_targets
        self.agent = SingleAgent(
            self.obs_dim,
            num_targets=self.args.num_targets,
            actor_hidden_sizes=self.args.actor_hidden_sizes,
            critic_hidden_sizes=self.args.critic_hidden_sizes,
            use_target_attention_pooling=self.args.use_target_attention_pooling,
            target_embed_dim=self.args.target_embed_dim,
            target_encoder_hidden_sizes=self.args.target_encoder_hidden_sizes,
            include_target_reached=include_reached,
            include_target_priority=self.args.use_target_priorities,
            energy_feature_dim=self.envs.energy_feature_dim,
        ).to(self.device)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)

        activation_margin = (
            0
            if self.args.charging_activation_margin is None
            else self.args.charging_activation_margin
        )
        self.charging_bc_report = pretrain_single_agent_charging_navigation(
            self.agent,
            self.envs,
            updates=self.args.charging_bc_updates,
            batch_size=self.args.charging_bc_batch_size,
            learning_rate=self.args.charging_bc_learning_rate,
            activation_margin=activation_margin,
            seed=self.args.seed + 100_000,
        )
        if self.args.charging_bc_updates:
            print(
                "   Charging navigation BC: "
                f"updates={self.args.charging_bc_updates}, "
                "direction_accuracy="
                f"{self.charging_bc_report['final_direction_accuracy']:.3f}"
            )

        self.args.batch_size = int(self.args.num_envs * self.args.num_steps)
        self.args.minibatch_size = int(self.args.batch_size // self.args.num_minibatches)

        # Calculate expected observation dimension components for single-agent mode
        # Base: agent/target positions, counters, priorities, window, relative counts,
        # and reached flags for the static-target variant.
        include_reached = not self.args.moving_targets
        expected_dim = 2 + 2 * self.args.num_targets + 3 * self.args.num_targets + 1
        if include_reached:
            expected_dim += self.args.num_targets
        if not self.args.use_target_priorities:
            expected_dim -= self.args.num_targets
        expected_dim += self.envs.energy_feature_dim

        print(f"🚀 Single-Agent PPO Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Observation dim: {self.obs_dim} (expected: {expected_dim})")
        print(f"   Includes relative target counts (count - min_count) for fair pursuit")
        print(
            "   Urgency-weighted scalarization: "
            f"{self.args.urgency_weighted_scalarization}"
        )
        print(f"   Target attention pooling: {self.args.use_target_attention_pooling}")
        if self.args.use_target_attention_pooling:
            print(f"   Target embed dim: {self.args.target_embed_dim}")
            print(f"   Encoded obs dim: {self.agent.encoded_obs_dim}")
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
