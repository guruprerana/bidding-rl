import pytest
import torch

from bidding_gridworld.bidding_gridworld_torch import (
    BiddingGridworld,
    BiddingGridworldConfig,
)


def make_config(**overrides):
    values = dict(
        grid_size=5,
        num_agents=2,
        bid_upper_bound=0,
        bid_penalty=0.0,
        target_reward=0.0,
        max_steps=20,
        action_window=1,
        distance_reward_scale=1.0,
        target_expiry_steps=4,
        target_expiry_penalty=0.0,
        moving_targets=False,
        direction_change_prob=0.0,
        target_move_interval=1,
        window_bidding=False,
        window_penalty=0.0,
        visible_targets=None,
        single_agent_mode=True,
        urgency_weighted_scalarization=True,
        use_target_priorities=False,
    )
    values.update(overrides)
    return BiddingGridworldConfig(**values)


def set_state(env, *, agent_pos, target_pos, counters, reached=None):
    env.agent_pos[:] = torch.tensor([agent_pos], device=env.device)
    env.target_pos[:] = torch.tensor([target_pos], device=env.device)
    env.target_counters[:] = torch.tensor([counters], device=env.device)
    if reached is None:
        reached = [0] * len(target_pos)
    env.targets_reached[:] = torch.tensor([reached], device=env.device)
    env.previous_distances = env._compute_distances()


def test_dense_objective_rewards_use_inverse_ttl_weights():
    env = BiddingGridworld(make_config(), 1, torch.device("cpu"), seed=3)
    env.reset()
    assert env.target_priorities.tolist() == [[1, 1]]
    assert env.obs_dim == 13
    set_state(
        env,
        agent_pos=[2, 2],
        target_pos=[[2, 4], [4, 2]],
        counters=[3, 0],
    )

    _, reward, _, _, info = env.step(torch.tensor([1]))  # right

    torch.testing.assert_close(
        info["urgency_weights"], torch.tensor([[0.8, 0.2]])
    )
    torch.testing.assert_close(
        info["per_objective_rewards"], torch.tensor([[1.0, -1.0]])
    )
    torch.testing.assert_close(reward, torch.tensor([0.6]))


def test_expiry_penalty_is_scalarized_with_pre_action_urgency():
    env = BiddingGridworld(
        make_config(distance_reward_scale=0.0, target_expiry_penalty=10.0),
        1,
        torch.device("cpu"),
        seed=7,
    )
    env.reset()
    set_state(
        env,
        agent_pos=[0, 0],
        target_pos=[[4, 4], [4, 3]],
        counters=[3, 0],
    )

    _, reward, _, _, info = env.step(torch.tensor([0]))  # left boundary

    torch.testing.assert_close(
        info["per_objective_rewards"], torch.tensor([[-10.0, 0.0]])
    )
    torch.testing.assert_close(reward, torch.tensor([-8.0]))


def test_reached_static_targets_are_excluded_from_normalization():
    env = BiddingGridworld(make_config(), 1, torch.device("cpu"), seed=11)
    env.reset()
    set_state(
        env,
        agent_pos=[2, 2],
        target_pos=[[2, 4], [4, 2]],
        counters=[3, 0],
        reached=[1, 0],
    )

    torch.testing.assert_close(env._urgency_weights(), torch.tensor([[0.0, 1.0]]))


def test_moving_target_completion_is_scalarized_then_respawned():
    env = BiddingGridworld(
        make_config(
            moving_targets=True,
            target_reward=10.0,
            distance_reward_scale=0.0,
        ),
        1,
        torch.device("cpu"),
        seed=13,
    )
    env.reset()
    set_state(
        env,
        agent_pos=[2, 2],
        target_pos=[[2, 3], [4, 4]],
        counters=[3, 0],
    )

    _, reward, terminated, truncated, info = env.step(torch.tensor([1]))

    torch.testing.assert_close(
        info["urgency_weights"], torch.tensor([[0.8, 0.2]])
    )
    torch.testing.assert_close(
        info["per_objective_rewards"], torch.tensor([[10.0, 0.0]])
    )
    torch.testing.assert_close(reward, torch.tensor([8.0]))
    assert env.targets_reached.tolist() == [[0, 0]]
    assert env.target_counters.tolist() == [[0, 1]]
    assert not terminated.item()
    assert not truncated.item()


def test_urgency_weighting_requires_single_agent_expiry_objectives():
    with pytest.raises(ValueError, match="requires single_agent_mode"):
        BiddingGridworld(
            make_config(single_agent_mode=False),
            1,
            torch.device("cpu"),
        )

    with pytest.raises(ValueError, match="requires target_expiry_steps"):
        BiddingGridworld(
            make_config(target_expiry_steps=None),
            1,
            torch.device("cpu"),
        )
