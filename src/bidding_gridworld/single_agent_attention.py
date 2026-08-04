"""Richer attention encoders for the fully observable single-agent baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from bidding_gridworld.single_agent_ppo import SingleAgent
from ppo_utils import build_mlp, layer_init


FOCUS_MODES = ("first", "attention", "nearest", "soft_nearest")


class MultiQueryAttentionPooling(nn.Module):
    """Pool a target set into several learned query slots."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_sizes: Sequence[int],
        num_queries: int,
    ) -> None:
        super().__init__()
        if num_queries < 1:
            raise ValueError("num_queries must be positive")
        self.num_queries = int(num_queries)
        self.encoder = build_mlp(
            input_dim, tuple(hidden_sizes), embed_dim
        )
        self.queries = nn.Parameter(torch.randn(self.num_queries, embed_dim))

    def forward(
        self,
        target_feats: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        *,
        return_weights: bool = False,
    ):
        batch_size, num_targets, feat_dim = target_feats.shape
        embeddings = self.encoder(
            target_feats.reshape(batch_size * num_targets, feat_dim)
        ).reshape(batch_size, num_targets, -1)
        scores = torch.einsum("bnd,kd->bkn", embeddings, self.queries)
        if target_mask is not None:
            scores = scores.masked_fill(~target_mask.unsqueeze(1), -1e9)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bkn,bnd->bkd", weights, embeddings)
        if return_weights:
            return pooled, weights
        return pooled


def _mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    *,
    output_std: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for hidden in hidden_sizes:
        layers.extend(
            [layer_init(nn.Linear(current, hidden)), nn.ELU()]
        )
        current = hidden
    layers.append(layer_init(nn.Linear(current, output_dim), std=output_std))
    return nn.Sequential(*layers)


class MultiQuerySingleAgent(SingleAgent):
    """Single policy with several learned target summaries and a focus slot.

    The policy remains centralized and emits one direction. ``attention`` focus
    is fully learned. ``nearest`` and ``soft_nearest`` are useful diagnostics
    for separating representation limitations from optimization limitations.
    """

    def __init__(
        self,
        obs_dim: int,
        num_targets: int,
        actor_hidden_sizes: Sequence[int] = (128, 128, 128, 128),
        critic_hidden_sizes: Sequence[int] = (256, 256, 256, 256),
        target_embed_dim: int = 64,
        target_encoder_hidden_sizes: Sequence[int] = (64, 64),
        include_target_reached: bool = False,
        include_target_priority: bool = False,
        energy_feature_dim: int = 0,
        num_attention_queries: int = 4,
        focus_mode: str = "attention",
        soft_nearest_temperature: float = 20.0,
    ) -> None:
        if focus_mode not in FOCUS_MODES:
            raise ValueError(
                f"focus_mode must be one of {FOCUS_MODES}, got {focus_mode!r}"
            )
        if soft_nearest_temperature <= 0:
            raise ValueError("soft_nearest_temperature must be positive")
        super().__init__(
            obs_dim=obs_dim,
            num_targets=num_targets,
            actor_hidden_sizes=tuple(actor_hidden_sizes),
            critic_hidden_sizes=tuple(critic_hidden_sizes),
            use_target_attention_pooling=True,
            target_embed_dim=target_embed_dim,
            target_encoder_hidden_sizes=tuple(target_encoder_hidden_sizes),
            include_target_reached=include_target_reached,
            include_target_priority=include_target_priority,
            energy_feature_dim=energy_feature_dim,
        )
        self.num_attention_queries = int(num_attention_queries)
        self.focus_mode = focus_mode
        self.soft_nearest_temperature = float(soft_nearest_temperature)
        self.supports_variable_targets = True
        self.target_feat_dim = (
            7 if include_target_reached else 6
        ) + int(include_target_priority)
        self.target_embed_dim = int(target_embed_dim)
        self.target_pool = MultiQueryAttentionPooling(
            input_dim=self.target_feat_dim,
            embed_dim=self.target_embed_dim,
            hidden_sizes=tuple(target_encoder_hidden_sizes),
            num_queries=self.num_attention_queries,
        )
        self.encoded_obs_dim = (
            3
            + self.energy_feature_dim
            + self.target_feat_dim
            + self.num_attention_queries * self.target_embed_dim
        )
        self.actor = _mlp(
            self.encoded_obs_dim,
            tuple(actor_hidden_sizes),
            4,
            output_std=0.01,
        )
        self.critic = _mlp(
            self.encoded_obs_dim,
            tuple(critic_hidden_sizes),
            1,
            output_std=1.0,
        )

    def _infer_target_count(self, observation_width: int) -> int:
        fixed_width = 3 + self.energy_feature_dim
        per_target_width = (
            4
            + int(self.include_target_reached)
            + int(self.include_target_priority)
        )
        target_width = observation_width - fixed_width
        if target_width <= 0 or target_width % per_target_width != 0:
            raise ValueError(
                "Invalid single-agent observation width for variable-target "
                f"attention: width={observation_width}, fixed={fixed_width}, "
                f"per_target={per_target_width}"
            )
        return target_width // per_target_width

    def _split_obs(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        target_count = self._infer_target_count(x.shape[-1])
        agent_pos = x[:, :2]
        energy_features = (
            x[:, -self.energy_feature_dim :]
            if self.energy_feature_dim > 0
            else x[:, :0]
        )
        cursor = 2
        target_pos = x[:, cursor : cursor + 2 * target_count].reshape(
            -1, target_count, 2
        )
        cursor += 2 * target_count
        target_parts = [
            target_pos,
            target_pos - agent_pos.unsqueeze(1),
        ]
        if self.include_target_reached:
            target_parts.append(
                x[:, cursor : cursor + target_count].reshape(
                    -1, target_count, 1
                )
            )
            cursor += target_count
        target_parts.append(
            x[:, cursor : cursor + target_count].reshape(
                -1, target_count, 1
            )
        )
        cursor += target_count
        if self.include_target_priority:
            target_parts.append(
                x[:, cursor : cursor + target_count].reshape(
                    -1, target_count, 1
                )
            )
            cursor += target_count
        window_steps = x[:, cursor : cursor + 1]
        cursor += 1
        target_parts.append(
            x[:, cursor : cursor + target_count].reshape(
                -1, target_count, 1
            )
        )
        return (
            agent_pos,
            energy_features,
            window_steps,
            torch.cat(target_parts, dim=-1),
        )

    def _focus_features(
        self,
        target_feats: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.focus_mode == "first":
            return target_feats[:, 0, :]
        if self.focus_mode == "attention":
            return torch.einsum(
                "bn,bnd->bd", attention_weights[:, 0], target_feats
            )
        distance = target_feats[:, :, 2:4].abs().sum(dim=-1)
        if self.focus_mode == "nearest":
            index = distance.argmin(dim=-1)
            return target_feats.gather(
                1,
                index.view(-1, 1, 1).expand(
                    -1, 1, target_feats.shape[-1]
                ),
            ).squeeze(1)
        weights = torch.softmax(
            -self.soft_nearest_temperature * distance, dim=-1
        )
        return torch.einsum("bn,bnd->bd", weights, target_feats)

    def _encode_obs(self, x: torch.Tensor) -> torch.Tensor:
        (
            agent_pos,
            energy_features,
            window_steps,
            target_feats,
        ) = self._split_obs(x)
        pooled, weights = self.target_pool(
            target_feats, return_weights=True
        )
        focus = self._focus_features(target_feats, weights)
        return torch.cat(
            [
                agent_pos,
                energy_features,
                window_steps,
                focus,
                pooled.flatten(start_dim=1),
            ],
            dim=-1,
        )

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ):
        encoded = self._encode_obs(x)
        logits = self.actor(encoded)
        distribution = Categorical(logits=logits)
        if action is None:
            action = (
                logits.argmax(dim=-1)
                if deterministic
                else distribution.sample()
            )
        return (
            action,
            distribution.log_prob(action),
            distribution.entropy(),
            self.critic(encoded),
        )


def _copy_expanded_input_weight(
    destination: torch.Tensor,
    source: torch.Tensor,
    prefix_dim: int,
    embed_dim: int,
) -> None:
    destination[:, :prefix_dim].copy_(source[:, :prefix_dim])
    destination[:, prefix_dim : prefix_dim + embed_dim].copy_(
        source[:, prefix_dim : prefix_dim + embed_dim]
    )
    if destination.shape[1] > prefix_dim + embed_dim:
        destination[:, prefix_dim + embed_dim :].mul_(0.05)


def load_attention_warm_start(
    agent: MultiQuerySingleAgent,
    checkpoint: str | Path,
) -> dict:
    """Load a base or multi-query checkpoint, expanding input layers safely."""
    checkpoint = Path(checkpoint)
    source = torch.load(
        checkpoint, map_location=next(agent.parameters()).device, weights_only=True
    )
    current = agent.state_dict()
    if set(source) == set(current) and all(
        source[key].shape == current[key].shape for key in current
    ):
        agent.load_state_dict(source, strict=True)
        return {"checkpoint": str(checkpoint), "mode": "exact"}

    copied = []
    skipped = []
    with torch.no_grad():
        for key, value in source.items():
            if key in current and current[key].shape == value.shape:
                current[key].copy_(value)
                copied.append(key)
            else:
                skipped.append(key)

        old_query = source.get("target_pool.query")
        if old_query is not None:
            current["target_pool.queries"][0].copy_(old_query)
            if agent.num_attention_queries > 1:
                noise = torch.randn_like(
                    current["target_pool.queries"][1:]
                ) * 0.05
                current["target_pool.queries"][1:].copy_(
                    old_query.unsqueeze(0) + noise
                )
            copied.append("target_pool.query -> target_pool.queries")

        prefix_dim = (
            3 + agent.energy_feature_dim + agent.target_feat_dim
        )
        for network in ("actor", "critic"):
            key = f"{network}.0.weight"
            if key in source and key in current:
                _copy_expanded_input_weight(
                    current[key],
                    source[key],
                    prefix_dim=prefix_dim,
                    embed_dim=agent.target_embed_dim,
                )
                copied.append(f"{key} (expanded)")

    agent.load_state_dict(current, strict=True)
    return {
        "checkpoint": str(checkpoint),
        "mode": "expanded",
        "copied": sorted(set(copied)),
        "skipped": sorted(set(skipped)),
    }
