import torch

from bidding_gridworld.bidding_gridworld_torch import (
    BiddingGridworld,
    BiddingGridworldConfig,
)
from bidding_gridworld.bidding_ppo import SharedAgent, load_policy_warm_start
from ppo_trainer_base import mix_bidder_rewards
from ppo_utils import factorized_auction_ppo_update_step


def _config(**overrides):
    values = dict(
        grid_size=7,
        num_agents=3,
        bid_upper_bound=6,
        bid_penalty=0.1,
        target_reward=50.0,
        max_steps=20,
        action_window=1,
        distance_reward_scale=0.6,
        target_expiry_steps=200,
        target_expiry_penalty=50.0,
        moving_targets=True,
        direction_change_prob=0.1,
        target_move_interval=5,
        window_bidding=False,
        window_penalty=0.0,
        visible_targets=None,
        use_target_priorities=False,
        programmatic_bidding="nearest_target",
        battery_capacity=None,
        charging_agent_enabled=False,
    )
    values.update(overrides)
    return BiddingGridworldConfig(**values)


def test_nearest_target_programmatic_bid_overrides_policy_bids():
    env = BiddingGridworld(
        _config(), num_envs=1, device=torch.device("cpu"), seed=1
    )
    env.reset()
    env.agent_pos[:] = torch.tensor([[3, 3]], dtype=torch.int32)
    env.target_pos[:] = torch.tensor(
        [[[3, 2], [6, 6], [0, 0]]], dtype=torch.int32
    )
    actions = torch.tensor([[[1, 6], [1, 6], [1, 6]]])

    _, _, _, _, info = env.step(actions)

    assert info["winning_agent"].tolist() == [0]
    assert info["bids"].tolist() == [[1, 0, 0]]


def test_direction_only_policy_ignores_bid_component_logprob():
    agent = SharedAgent(
        obs_dim=5,
        num_actions_per_agent=2,
        actor_hidden_sizes=(),
        critic_hidden_sizes=(),
    )
    agent.set_bid_head(6)
    obs = torch.zeros((4, 5))
    actions = torch.tensor([[0, 0], [1, 6], [2, 3], [3, 1]])

    emitted, logprob, entropy, value = agent.get_direction_action_and_value(
        obs, action=actions
    )

    assert emitted[:, 0].tolist() == [0, 1, 2, 3]
    assert emitted[:, 1].tolist() == [0, 0, 0, 0]
    assert logprob.shape == (4,)
    assert entropy.shape == (4,)
    assert value.shape == (4, 1)


def test_learned_bid_only_policy_keeps_direction_and_multilevel_bid_space():
    agent = SharedAgent(
        obs_dim=5,
        num_actions_per_agent=2,
        actor_hidden_sizes=(8,),
        critic_hidden_sizes=(8,),
        separate_bid_actor=True,
        bid_actor_hidden_sizes=(8,),
    )
    agent.set_bid_head(15)
    obs = torch.randn((16, 5))

    action, logprob, entropy, value = (
        agent.get_bid_action_and_value_with_direction(obs)
    )

    expected_direction = agent.direction_head(
        agent.actor_shared(obs)
    ).argmax(dim=-1)
    assert torch.equal(action[:, 0], expected_direction)
    assert action[:, 1].min() >= 0
    assert action[:, 1].max() <= 15
    assert logprob.shape == (16,)
    assert entropy.shape == (16,)
    assert value.shape == (16, 1)


def test_policy_warm_start_can_reset_only_bid_policy(tmp_path):
    source = SharedAgent(
        obs_dim=5,
        num_actions_per_agent=2,
        actor_hidden_sizes=(8,),
        critic_hidden_sizes=(8,),
    )
    source.set_bid_head(6)
    checkpoint = tmp_path / "agent.pt"
    torch.save(source.state_dict(), checkpoint)
    target = SharedAgent(
        obs_dim=5,
        num_actions_per_agent=2,
        actor_hidden_sizes=(8,),
        critic_hidden_sizes=(8,),
        separate_bid_actor=True,
        bid_actor_hidden_sizes=(8,),
    )
    target.set_bid_head(15)

    report = load_policy_warm_start(
        target, str(checkpoint), reset_bid_policy=True
    )

    assert "direction_head.weight" in report["loaded_keys"]
    assert "bid_head.weight" in report["skipped_source_keys"]
    assert torch.equal(
        target.direction_head.weight, source.direction_head.weight
    )


def test_controller_team_bid_credit_preserves_total_step_reward():
    env = BiddingGridworld(
        _config(programmatic_bidding="none"),
        num_envs=1,
        device=torch.device("cpu"),
        seed=3,
    )
    env.reset()
    actions = torch.tensor([[[1, 3], [0, 1], [2, 0]]])

    _, rewards, _, _, info = env.step(actions)

    bid_rewards = info["bid_policy_controller_team_rewards"]
    assert bid_rewards.shape == rewards.shape
    assert torch.allclose(bid_rewards.sum(dim=1), rewards.sum(dim=1))


def test_mixed_bid_credit_interpolates_selfish_and_cooperative_rewards():
    rewards = torch.tensor([[10.0, -2.0, 4.0], [1.0, 3.0, -5.0]])

    assert torch.equal(mix_bidder_rewards(rewards, 0.0), rewards)
    assert torch.equal(
        mix_bidder_rewards(rewards, 1.0),
        rewards.sum(dim=1, keepdim=True).expand_as(rewards),
    )
    assert torch.allclose(
        mix_bidder_rewards(rewards, 0.25),
        torch.tensor([[10.5, 1.5, 6.0], [0.5, 2.0, -4.0]]),
    )


def test_mixed_bid_credit_rejects_invalid_other_reward_fraction():
    rewards = torch.zeros((2, 3))
    for fraction in (-0.01, 1.01):
        try:
            mix_bidder_rewards(rewards, fraction)
        except ValueError as error:
            assert "between 0 and 1" in str(error)
        else:
            raise AssertionError("invalid other-reward fraction was accepted")


def test_mixed_bid_credit_can_preserve_team_reward_scale():
    rewards = torch.full((2, 8), 3.0)
    mixed = mix_bidder_rewards(
        rewards, 0.5, preserve_team_scale=True
    )

    assert torch.allclose(mixed, torch.full_like(rewards, 24.0))


def test_factorized_logprobs_reconstruct_joint_policy_logprob():
    agent = SharedAgent(
        obs_dim=5,
        num_actions_per_agent=2,
        actor_hidden_sizes=(8,),
        critic_hidden_sizes=(8,),
    )
    agent.set_bid_head(6)
    obs = torch.randn((12, 5))
    actions = torch.stack(
        [torch.arange(12) % 4, torch.arange(12) % 7], dim=1
    )

    _, joint_logprob, joint_entropy, joint_value = agent.get_action_and_value(
        obs, action=actions
    )
    (
        emitted,
        direction_logprob,
        bid_logprob,
        direction_entropy,
        bid_entropy,
        factorized_value,
    ) = agent.get_factorized_action_and_value(obs, action=actions)

    assert torch.equal(emitted, actions)
    torch.testing.assert_close(
        direction_logprob + bid_logprob, joint_logprob
    )
    torch.testing.assert_close(
        direction_entropy + bid_entropy, joint_entropy
    )
    torch.testing.assert_close(factorized_value, joint_value)


def test_ordinal_bid_head_emits_valid_ordered_categorical_bids():
    agent = SharedAgent(
        obs_dim=5,
        num_actions_per_agent=2,
        actor_hidden_sizes=(8,),
        critic_hidden_sizes=(8,),
        ordinal_bid_head=True,
    )
    agent.set_bid_head(15)
    obs = torch.randn((32, 5))
    encoded = agent._encode_obs(obs)
    features = agent.actor_shared(encoded)
    logits = agent._bid_logits(features)

    assert logits.shape == (32, 16)
    assert agent.bid_head.out_features == 2
    actions, _, _, _ = agent.get_action_and_value(obs)
    assert torch.all((0 <= actions[:, 1]) & (actions[:, 1] <= 15))

    loss = -logits.log_softmax(dim=-1)[:, 15].mean()
    loss.backward()
    assert agent.bid_head.weight.grad is not None
    assert torch.isfinite(agent.bid_head.weight.grad).all()


def test_factorized_ppo_update_accepts_sparse_causal_masks():
    agent = SharedAgent(
        obs_dim=5,
        num_actions_per_agent=2,
        actor_hidden_sizes=(8,),
        critic_hidden_sizes=(8,),
    )
    agent.set_bid_head(6)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    obs = torch.randn((16, 5))
    with torch.no_grad():
        (
            actions,
            direction_logprob,
            bid_logprob,
            _,
            _,
            values,
        ) = agent.get_factorized_action_and_value(obs)
    direction_mask = torch.zeros(16, dtype=torch.bool)
    direction_mask[::4] = True
    bid_mask = torch.zeros(16, dtype=torch.bool)
    bid_mask[::2] = True

    metrics = factorized_auction_ppo_update_step(
        agent,
        optimizer,
        obs,
        actions,
        direction_logprob,
        bid_logprob,
        direction_mask,
        bid_mask,
        advantages=torch.randn(16),
        returns=torch.randn(16),
        values=values.flatten(),
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        norm_adv=True,
        clip_vloss=False,
    )

    assert set(("direction_pg_loss", "bid_pg_loss")) <= metrics.keys()
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
