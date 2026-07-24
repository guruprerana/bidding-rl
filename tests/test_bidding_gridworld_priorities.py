import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bidding_gridworld.bidding_gridworld_torch import (
    BiddingGridworld,
    BiddingGridworldConfig,
    evaluate_multi_agent_policy,
    evaluate_multi_agent_policy_batched,
    evaluate_single_agent_policy,
    evaluate_single_agent_policy_batched,
)
from bidding_gridworld.bidding_ppo import SharedAgent
from bidding_gridworld.single_agent_ppo import SingleAgent


def make_config(
    *,
    num_agents=3,
    moving_targets=True,
    visible_targets=None,
    single_agent_mode=False,
):
    return BiddingGridworldConfig(
        grid_size=8,
        num_agents=num_agents,
        bid_upper_bound=2,
        bid_penalty=0.0,
        target_reward=10.0,
        max_steps=50,
        action_window=1,
        distance_reward_scale=0.0,
        target_expiry_steps=None,
        target_expiry_penalty=0.0,
        moving_targets=moving_targets,
        direction_change_prob=0.0,
        target_move_interval=100,
        window_bidding=False,
        window_penalty=0.0,
        visible_targets=visible_targets,
        single_agent_mode=single_agent_mode,
    )


def test_single_agent_reward_is_scaled_by_target_priority():
    env = BiddingGridworld(
        make_config(num_agents=2, single_agent_mode=True),
        num_envs=1,
        device=torch.device("cpu"),
        seed=1,
    )
    env.reset()
    env.target_pos[:] = torch.tensor([[[0, 1], [7, 7]]], dtype=torch.int32)
    env.target_priorities[:] = torch.tensor([[4, 2]], dtype=torch.int32)
    env.previous_distances = env._compute_distances()

    _, reward, _, _, info = env.step(torch.tensor([1]))

    assert reward.item() == 40.0
    assert info["per_objective_rewards"].tolist() == [[40.0, 0.0]]
    assert info["target_priorities_just_reached"].tolist() == [[4, 0]]


def test_multi_agent_reward_is_scaled_by_each_targets_priority():
    env = BiddingGridworld(
        make_config(num_agents=2),
        num_envs=1,
        device=torch.device("cpu"),
        seed=2,
    )
    env.reset()
    env.target_pos[:] = torch.tensor([[[0, 1], [7, 7]]], dtype=torch.int32)
    env.target_priorities[:] = torch.tensor([[3, 2]], dtype=torch.int32)
    env.previous_distances = env._compute_distances()
    actions = torch.tensor([[[1, 1], [0, 0]]])

    _, rewards, _, _, _ = env.step(actions)

    assert rewards.tolist() == [[30.0, 0.0]]


def test_priority_is_replaced_when_a_moving_target_respawns():
    env = BiddingGridworld(
        make_config(num_agents=2),
        num_envs=1,
        device=torch.device("cpu"),
        seed=3,
    )
    env.reset()
    env.target_priorities[:] = 1
    env._sample_target_priorities = lambda: torch.full_like(env.target_priorities, 4)

    env._move_targets(
        targets_just_reached=torch.tensor([[True, False]]),
        targets_expired=torch.tensor([[False, False]]),
    )

    assert env.target_priorities.tolist() == [[4, 1]]


def test_priorities_are_normalized_in_all_observation_layouts():
    single_env = BiddingGridworld(
        make_config(num_agents=3, single_agent_mode=True),
        num_envs=2,
        device=torch.device("cpu"),
        seed=4,
    )
    single_obs, _ = single_env.reset()
    priority_start = 2 + 3 * single_env.num_agents
    assert single_obs.shape == (2, 18)
    assert torch.equal(
        single_obs[:, priority_start:priority_start + single_env.num_agents],
        single_env.target_priorities.float() / 4.0,
    )

    central_env = BiddingGridworld(
        make_config(num_agents=3),
        num_envs=2,
        device=torch.device("cpu"),
        seed=5,
    )
    central_obs, _ = central_env.reset()
    priority_start = 2 + 3 * central_env.num_agents
    expected = central_env.target_priorities[:, central_env._reorder_idx].float() / 4.0
    assert central_obs.shape == (2, 3, 15)
    assert torch.equal(
        central_obs[:, :, priority_start:priority_start + central_env.num_agents],
        expected,
    )

    visible_env = BiddingGridworld(
        make_config(num_agents=3, visible_targets=1),
        num_envs=2,
        device=torch.device("cpu"),
        seed=6,
    )
    visible_obs, _ = visible_env.reset()
    assert visible_obs.shape == (2, 3, 10)
    assert torch.equal(visible_obs[:, :, 7], visible_env.target_priorities.float() / 4.0)


def test_attention_encoders_accept_priority_augmented_observations():
    for moving_targets in (False, True):
        central_env = BiddingGridworld(
            make_config(num_agents=3, moving_targets=moving_targets),
            num_envs=2,
            device=torch.device("cpu"),
            seed=7,
        )
        central_obs, _ = central_env.reset()
        central_agent = SharedAgent(
            obs_dim=central_env.per_agent_obs_dim,
            num_actions_per_agent=2,
            window_bidding=False,
            use_target_attention_pooling=True,
            attention_pooling_layout="centralized",
            include_target_reached=not moving_targets,
        )
        encoded = central_agent._encode_obs(
            central_obs.reshape(-1, central_env.per_agent_obs_dim)
        )
        assert encoded.shape == (6, central_agent.encoded_obs_dim)

        visible_env = BiddingGridworld(
            make_config(
                num_agents=3,
                moving_targets=moving_targets,
                visible_targets=1,
            ),
            num_envs=2,
            device=torch.device("cpu"),
            seed=8,
        )
        visible_obs, _ = visible_env.reset()
        visible_agent = SharedAgent(
            obs_dim=visible_env.per_agent_obs_dim,
            num_actions_per_agent=2,
            window_bidding=False,
            use_target_attention_pooling=True,
            attention_pooling_layout="visible",
            include_target_reached=not moving_targets,
        )
        encoded = visible_agent._encode_obs(
            visible_obs.reshape(-1, visible_env.per_agent_obs_dim)
        )
        assert encoded.shape == (6, visible_agent.encoded_obs_dim)

        single_env = BiddingGridworld(
            make_config(
                num_agents=3,
                moving_targets=moving_targets,
                single_agent_mode=True,
            ),
            num_envs=2,
            device=torch.device("cpu"),
            seed=9,
        )
        single_obs, _ = single_env.reset()
        single_agent = SingleAgent(
            obs_dim=single_env.obs_dim,
            num_targets=3,
            use_target_attention_pooling=True,
            include_target_reached=not moving_targets,
        )
        encoded = single_agent._encode_obs(single_obs)
        assert encoded.shape == (2, single_agent.encoded_obs_dim)


def _fix_one_step_resets(env, reached_priorities):
    original_reset = env.reset
    env.config.max_steps = 1

    def fixed_reset(seed=None):
        original_reset(seed)
        env.target_pos[:] = torch.tensor([7, 7], dtype=torch.int32)
        env.target_pos[:, 0] = torch.tensor([0, 1], dtype=torch.int32)
        env.target_priorities[:] = 1
        env.target_priorities[:, 0] = torch.tensor(reached_priorities, dtype=torch.int32)
        env.previous_distances = env._compute_distances()
        return env._get_observation(), {}

    env.reset = fixed_reset


def test_all_evaluators_sum_priorities_reached_before_respawn():
    single_env = BiddingGridworld(
        make_config(num_agents=2, single_agent_mode=True),
        num_envs=1,
        device=torch.device("cpu"),
        seed=10,
    )
    _fix_one_step_resets(single_env, [4])
    stats = evaluate_single_agent_policy(single_env, lambda _: 1, 1, verbose=False)
    assert stats["reached_priority_sum_per_episode"] == [4]
    assert stats["reached_priority_sum_per_target_per_episode"] == [[4, 0]]
    assert stats["reached_count_by_priority_per_episode"] == [[0, 0, 0, 1]]

    single_batched_env = BiddingGridworld(
        make_config(num_agents=2, single_agent_mode=True),
        num_envs=2,
        device=torch.device("cpu"),
        seed=11,
    )
    _fix_one_step_resets(single_batched_env, [4, 2])
    stats = evaluate_single_agent_policy_batched(
        single_batched_env,
        lambda obs: torch.ones(obs.shape[0], dtype=torch.int64),
        2,
        verbose=False,
    )
    assert stats["reached_priority_sum_per_episode"] == [4, 2]
    assert stats["reached_priority_sum_per_target_per_episode"] == [[4, 0], [2, 0]]
    assert stats["reached_count_by_priority_per_episode"] == [
        [0, 0, 0, 1],
        [0, 1, 0, 0],
    ]

    multi_env = BiddingGridworld(
        make_config(num_agents=2),
        num_envs=1,
        device=torch.device("cpu"),
        seed=12,
    )
    _fix_one_step_resets(multi_env, [3])
    multi_action = torch.tensor([[1, 1], [0, 0]], dtype=torch.int64)
    stats = evaluate_multi_agent_policy(
        multi_env,
        lambda _: multi_action,
        1,
        verbose=False,
    )
    assert stats["reached_priority_sum_per_episode"] == [3]
    assert stats["reached_priority_sum_per_target_per_episode"] == [[3, 0]]
    assert stats["reached_count_by_priority_per_episode"] == [[0, 0, 1, 0]]

    multi_batched_env = BiddingGridworld(
        make_config(num_agents=2),
        num_envs=2,
        device=torch.device("cpu"),
        seed=13,
    )
    _fix_one_step_resets(multi_batched_env, [3, 1])
    stats = evaluate_multi_agent_policy_batched(
        multi_batched_env,
        lambda obs: multi_action.unsqueeze(0).expand(obs.shape[0], -1, -1),
        2,
        verbose=False,
    )
    assert stats["reached_priority_sum_per_episode"] == [3, 1]
    assert stats["reached_priority_sum_per_target_per_episode"] == [[3, 0], [1, 0]]
    assert stats["reached_count_by_priority_per_episode"] == [
        [0, 0, 1, 0],
        [1, 0, 0, 0],
    ]


class TestBiddingGridworldPriorities(unittest.TestCase):
    def test_single_agent_reward_scaling(self):
        test_single_agent_reward_is_scaled_by_target_priority()

    def test_multi_agent_reward_scaling(self):
        test_multi_agent_reward_is_scaled_by_each_targets_priority()

    def test_priority_respawn(self):
        test_priority_is_replaced_when_a_moving_target_respawns()

    def test_observation_layouts(self):
        test_priorities_are_normalized_in_all_observation_layouts()

    def test_attention_encoders(self):
        test_attention_encoders_accept_priority_augmented_observations()

    def test_evaluation_priority_sums(self):
        test_all_evaluators_sum_priorities_reached_before_respawn()
