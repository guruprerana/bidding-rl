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
    clip_vloss: bool = True
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
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs, actions)
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
    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

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
    }


def compute_explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute explained variance."""
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
