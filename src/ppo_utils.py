# Shared PPO utilities for both single-agent and multi-agent training
import torch
import torch.nn as nn
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer weights with orthogonal initialization."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Reward tensor (num_steps, num_envs, ...)
        values: Value estimates (num_steps, num_envs, ...)
        dones: Done flags (num_steps, num_envs, ...)
        next_value: Value estimate for next state (num_envs, ...)
        next_done: Done flag for next state (num_envs, ...)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: Computed advantages
        returns: Computed returns (advantages + values)
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(next_done)

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]

        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

    returns = advantages + values
    return advantages, returns


def ppo_update_step(
    agent: nn.Module,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    norm_adv: bool = True,
    clip_vloss: bool = True,
    action_value_fn=None,
    auxiliary_loss_fn=None,
) -> dict:
    """
    Perform a single PPO update step.

    Args:
        agent: The agent network (must have get_action_and_value method)
        optimizer: The optimizer
        obs: Observations
        actions: Actions taken
        logprobs: Log probabilities of actions
        advantages: Computed advantages
        returns: Computed returns
        values: Old value estimates
        clip_coef: PPO clipping coefficient
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Maximum gradient norm for clipping
        norm_adv: Whether to normalize advantages
        clip_vloss: Whether to clip value loss

    Returns:
        Dictionary of loss metrics
    """
    # Get new predictions
    if action_value_fn is None:
        action_value_fn = agent.get_action_and_value
    _, newlogprob, entropy, newvalue = action_value_fn(obs, actions)
    logratio = newlogprob - logprobs
    ratio = logratio.exp()

    # KL divergence approximation
    with torch.no_grad():
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()

    # Normalize advantages
    if norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss (PPO clipped objective)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = newvalue.view(-1)
    if clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = values + torch.clamp(
            newvalue - values,
            -clip_coef,
            clip_coef,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

    # Entropy loss
    entropy_loss = entropy.mean()

    # Total loss
    auxiliary_loss = (
        auxiliary_loss_fn(agent, obs)
        if auxiliary_loss_fn is not None
        else torch.zeros((), device=obs.device)
    )
    loss = (
        pg_loss
        - ent_coef * entropy_loss
        + v_loss * vf_coef
        + auxiliary_loss
    )

    # Optimization step
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
    optimizer.step()

    return {
        "pg_loss": pg_loss.item(),
        "v_loss": v_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "old_approx_kl": old_approx_kl.item(),
        "approx_kl": approx_kl.item(),
        "clipfrac": clipfrac.item(),
        "auxiliary_loss": auxiliary_loss.item(),
    }


def factorized_auction_ppo_update_step(
    agent,
    optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_direction_logprobs: torch.Tensor,
    old_bid_logprobs: torch.Tensor,
    direction_mask: torch.Tensor,
    bid_mask: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    norm_adv: bool,
    clip_vloss: bool,
):
    """PPO update with causal masks for auction action components.

    A direction contributes only when that agent controlled the shared body.
    A bid contributes only when the environment actually held an auction.
    This avoids policy gradients through actions ignored by the transition.
    """

    def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(value.dtype)
        return (value * mask).sum() / mask.sum().clamp_min(1.0)

    def masked_advantage(
        value: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if not norm_adv:
            return value
        mask_f = mask.to(value.dtype)
        count = mask_f.sum().clamp_min(1.0)
        mean = (value * mask_f).sum() / count
        variance = ((value - mean).square() * mask_f).sum() / count
        return (value - mean) / torch.sqrt(variance + 1e-8)

    (
        _,
        direction_logprobs,
        bid_logprobs,
        direction_entropy,
        bid_entropy,
        newvalue,
    ) = agent.get_factorized_action_and_value(obs, actions)

    direction_adv = masked_advantage(advantages, direction_mask)
    bid_adv = masked_advantage(advantages, bid_mask)

    direction_logratio = direction_logprobs - old_direction_logprobs
    direction_ratio = direction_logratio.exp()
    direction_pg_1 = -direction_adv * direction_ratio
    direction_pg_2 = -direction_adv * torch.clamp(
        direction_ratio, 1 - clip_coef, 1 + clip_coef
    )
    direction_pg = masked_mean(
        torch.maximum(direction_pg_1, direction_pg_2), direction_mask
    )

    bid_logratio = bid_logprobs - old_bid_logprobs
    bid_ratio = bid_logratio.exp()
    bid_pg_1 = -bid_adv * bid_ratio
    bid_pg_2 = -bid_adv * torch.clamp(
        bid_ratio, 1 - clip_coef, 1 + clip_coef
    )
    bid_pg = masked_mean(torch.maximum(bid_pg_1, bid_pg_2), bid_mask)
    pg_loss = direction_pg + bid_pg

    newvalue = newvalue.view(-1)
    if clip_vloss:
        value_loss = (newvalue - returns).square()
        value_clipped = values + torch.clamp(
            newvalue - values, -clip_coef, clip_coef
        )
        clipped_loss = (value_clipped - returns).square()
        v_loss = 0.5 * torch.maximum(value_loss, clipped_loss).mean()
    else:
        v_loss = 0.5 * (newvalue - returns).square().mean()

    entropy_loss = masked_mean(
        direction_entropy, direction_mask
    ) + masked_mean(bid_entropy, bid_mask)
    loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
    optimizer.step()

    active_count = (
        direction_mask.to(torch.float32).sum()
        + bid_mask.to(torch.float32).sum()
    ).clamp_min(1.0)
    with torch.no_grad():
        old_approx_kl = (
            masked_mean(-direction_logratio, direction_mask)
            + masked_mean(-bid_logratio, bid_mask)
        )
        approx_kl = (
            masked_mean(
                (direction_ratio - 1.0) - direction_logratio,
                direction_mask,
            )
            + masked_mean((bid_ratio - 1.0) - bid_logratio, bid_mask)
        )
        direction_clipped = (
            (direction_ratio - 1.0).abs() > clip_coef
        ).to(torch.float32)
        bid_clipped = ((bid_ratio - 1.0).abs() > clip_coef).to(torch.float32)
        clipfrac = (
            (direction_clipped * direction_mask).sum()
            + (bid_clipped * bid_mask).sum()
        ) / active_count

    return {
        "pg_loss": pg_loss.item(),
        "direction_pg_loss": direction_pg.item(),
        "bid_pg_loss": bid_pg.item(),
        "v_loss": v_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "old_approx_kl": old_approx_kl.item(),
        "approx_kl": approx_kl.item(),
        "clipfrac": clipfrac.item(),
        "auxiliary_loss": 0.0,
    }


def compute_explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute explained variance."""
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


def format_duration(seconds: float) -> str:
    """Format seconds into H:MM:SS or M:SS."""
    seconds = max(0.0, seconds)
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def build_mlp(input_dim: int, hidden_sizes: tuple[int, ...], output_dim: int) -> nn.Sequential:
    """Build a simple MLP with ELU activations."""
    layers = []
    in_dim = input_dim
    for hidden in hidden_sizes:
        layers.append(layer_init(nn.Linear(in_dim, hidden)))
        layers.append(nn.ELU())
        in_dim = hidden
    layers.append(layer_init(nn.Linear(in_dim, output_dim)))
    return nn.Sequential(*layers)


class MaskedAttentionPooling(nn.Module):
    """Masked attention pooling with a learned query over per-target embeddings."""

    def __init__(self, input_dim: int, embed_dim: int, hidden_sizes: tuple):
        super().__init__()
        self.encoder = build_mlp(input_dim, hidden_sizes, embed_dim)
        self.query = nn.Parameter(torch.randn(embed_dim))

    def forward(self, target_feats: torch.Tensor, target_mask=None) -> torch.Tensor:
        batch_size, num_targets, feat_dim = target_feats.shape
        flat = target_feats.reshape(batch_size * num_targets, feat_dim)
        embeddings = self.encoder(flat).reshape(batch_size, num_targets, -1)
        scores = torch.einsum("bnd,d->bn", embeddings, self.query)
        if target_mask is not None:
            scores = scores.masked_fill(~target_mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bnd,bn->bd", embeddings, weights)
        return pooled
