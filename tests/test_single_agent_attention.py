from __future__ import annotations

import torch

from bidding_gridworld.single_agent_attention import (
    MultiQuerySingleAgent,
    load_attention_warm_start,
)
from bidding_gridworld.single_agent_ppo import SingleAgent


def make_observation() -> torch.Tensor:
    # Moving-target, no-priority T=3 layout:
    # agent(2), target positions(6), counters(3), window(1), counts(3).
    return torch.tensor(
        [
            [
                0.5,
                0.5,
                0.9,
                0.9,
                0.55,
                0.5,
                0.1,
                0.1,
                0.2,
                0.8,
                0.4,
                0.0,
                0.0,
                0.1,
                0.2,
            ]
        ],
        dtype=torch.float32,
    )


def test_one_query_first_focus_preserves_base_checkpoint(tmp_path):
    torch.manual_seed(7)
    base = SingleAgent(
        obs_dim=15,
        num_targets=3,
        actor_hidden_sizes=(16, 16),
        critic_hidden_sizes=(24, 24),
        use_target_attention_pooling=True,
        target_embed_dim=8,
        target_encoder_hidden_sizes=(8,),
        include_target_reached=False,
        include_target_priority=False,
    )
    checkpoint = tmp_path / "base.pt"
    torch.save(base.state_dict(), checkpoint)
    enhanced = MultiQuerySingleAgent(
        obs_dim=15,
        num_targets=3,
        actor_hidden_sizes=(16, 16),
        critic_hidden_sizes=(24, 24),
        target_embed_dim=8,
        target_encoder_hidden_sizes=(8,),
        include_target_reached=False,
        include_target_priority=False,
        num_attention_queries=1,
        focus_mode="first",
    )

    report = load_attention_warm_start(enhanced, checkpoint)
    obs = make_observation()

    assert report["mode"] == "expanded"
    assert torch.allclose(base._encode_obs(obs), enhanced._encode_obs(obs))
    assert torch.allclose(
        base.actor(base._encode_obs(obs)),
        enhanced.actor(enhanced._encode_obs(obs)),
    )
    assert torch.allclose(
        base.critic(base._encode_obs(obs)),
        enhanced.critic(enhanced._encode_obs(obs)),
    )


def test_multi_query_attention_has_expected_shape_and_gradients():
    torch.manual_seed(8)
    agent = MultiQuerySingleAgent(
        obs_dim=15,
        num_targets=3,
        actor_hidden_sizes=(16,),
        critic_hidden_sizes=(24,),
        target_embed_dim=8,
        target_encoder_hidden_sizes=(8,),
        include_target_reached=False,
        include_target_priority=False,
        num_attention_queries=4,
        focus_mode="attention",
    )
    obs = make_observation().repeat(5, 1)
    encoded = agent._encode_obs(obs)
    loss = agent.actor(encoded).square().mean() + agent.critic(encoded).square().mean()
    loss.backward()

    assert encoded.shape == (5, agent.encoded_obs_dim)
    assert agent.target_pool.queries.grad is not None
    assert torch.isfinite(agent.target_pool.queries.grad).all()
    assert agent.target_pool.queries.grad.abs().sum() > 0


def test_nearest_focus_uses_nearest_target_features():
    agent = MultiQuerySingleAgent(
        obs_dim=15,
        num_targets=3,
        actor_hidden_sizes=(16,),
        critic_hidden_sizes=(24,),
        target_embed_dim=8,
        target_encoder_hidden_sizes=(8,),
        include_target_reached=False,
        include_target_priority=False,
        num_attention_queries=2,
        focus_mode="nearest",
    )
    encoded = agent._encode_obs(make_observation())
    # Prefix is agent(2), window(1), then the selected target's six features.
    focus = encoded[0, 3:9]
    expected = torch.tensor([0.55, 0.5, 0.05, 0.0, 0.8, 0.1])
    assert torch.allclose(focus, expected)


def test_attention_policy_accepts_more_targets_than_training_layout():
    agent = MultiQuerySingleAgent(
        obs_dim=35,
        num_targets=8,
        actor_hidden_sizes=(16,),
        critic_hidden_sizes=(24,),
        target_embed_dim=8,
        target_encoder_hidden_sizes=(8,),
        include_target_reached=False,
        include_target_priority=False,
        num_attention_queries=4,
        focus_mode="attention",
    )
    # Moving-target, no-priority observations have width 3 + 4*T.
    train_obs = torch.randn(2, 35)
    transfer_obs = torch.randn(2, 131)

    train_encoded = agent._encode_obs(train_obs)
    transfer_encoded = agent._encode_obs(transfer_obs)

    assert train_encoded.shape == (2, agent.encoded_obs_dim)
    assert transfer_encoded.shape == (2, agent.encoded_obs_dim)
    assert agent.get_action_and_value(transfer_obs, deterministic=True)[0].shape == (2,)


def test_variable_target_parser_rejects_invalid_observation_width():
    agent = MultiQuerySingleAgent(
        obs_dim=35,
        num_targets=8,
        actor_hidden_sizes=(16,),
        critic_hidden_sizes=(24,),
        target_embed_dim=8,
        target_encoder_hidden_sizes=(8,),
        include_target_reached=False,
        include_target_priority=False,
        num_attention_queries=4,
        focus_mode="attention",
    )
    invalid_obs = torch.randn(2, 130)

    try:
        agent._encode_obs(invalid_obs)
    except ValueError as error:
        assert "Invalid single-agent observation width" in str(error)
    else:
        raise AssertionError("invalid observation width should be rejected")
