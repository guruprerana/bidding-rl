import sys
import tempfile
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
from bidding_gridworld.bidding_ppo import (
    Args,
    PPOTrainer,
    SharedAgent,
    load_feeder_checkpoint_with_energy_expansion,
    pretrain_charging_policy,
)
from bidding_gridworld.experiment import isolated_torch_rng
from bidding_gridworld.single_agent_ppo import (
    SingleAgent,
    pretrain_single_agent_charging_navigation,
)
from train_ppo_moving_targets import (
    parse_optional_int,
    parse_recharge_station_positions,
)


def make_config(
    *,
    grid_size=8,
    num_agents=3,
    moving_targets=True,
    visible_targets=None,
    single_agent_mode=False,
    battery_capacity=None,
    recharge_station_positions=None,
    moving_recharge_stations=False,
    recharge_station_direction_change_prob=0.1,
    recharge_station_move_interval=5,
    movement_energy_cost=1,
    battery_depletion_penalty=0.0,
    charging_agent_enabled=False,
    charging_low_battery_threshold=3,
    charging_distance_reward_scale=0.0,
    charging_recharge_bonus=0.0,
    charging_depletion_penalty=0.0,
    charging_high_battery_control_penalty=0.0,
    feeder_low_battery_control_penalty=0.0,
    charging_low_battery_bid_boost=0,
    charging_bid_boost_threshold=None,
    charging_activation_margin=None,
    charging_release_window_on_recharge=False,
    charging_programmatic_navigation=False,
    charging_reserve_features_enabled=False,
    charging_nearest_station_features_enabled=False,
):
    return BiddingGridworldConfig(
        grid_size=grid_size,
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
        battery_capacity=battery_capacity,
        recharge_station_positions=recharge_station_positions,
        moving_recharge_stations=moving_recharge_stations,
        recharge_station_direction_change_prob=(
            recharge_station_direction_change_prob
        ),
        recharge_station_move_interval=recharge_station_move_interval,
        movement_energy_cost=movement_energy_cost,
        battery_depletion_penalty=battery_depletion_penalty,
        charging_agent_enabled=charging_agent_enabled,
        charging_low_battery_threshold=charging_low_battery_threshold,
        charging_distance_reward_scale=charging_distance_reward_scale,
        charging_recharge_bonus=charging_recharge_bonus,
        charging_depletion_penalty=charging_depletion_penalty,
        charging_high_battery_control_penalty=(
            charging_high_battery_control_penalty
        ),
        feeder_low_battery_control_penalty=(
            feeder_low_battery_control_penalty
        ),
        charging_low_battery_bid_boost=charging_low_battery_bid_boost,
        charging_bid_boost_threshold=charging_bid_boost_threshold,
        charging_activation_margin=charging_activation_margin,
        charging_release_window_on_recharge=(
            charging_release_window_on_recharge
        ),
        charging_programmatic_navigation=charging_programmatic_navigation,
        charging_reserve_features_enabled=(
            charging_reserve_features_enabled
        ),
        charging_nearest_station_features_enabled=(
            charging_nearest_station_features_enabled
        ),
    )


def test_single_agent_reserve_aware_charging_reward():
    env = BiddingGridworld(
        make_config(
            num_agents=2,
            single_agent_mode=True,
            battery_capacity=50,
            recharge_station_positions=((0, 0),),
            charging_low_battery_threshold=35,
            charging_distance_reward_scale=4.0,
            charging_recharge_bonus=20.0,
            charging_depletion_penalty=100.0,
            charging_activation_margin=8,
        ),
        num_envs=1,
        device=torch.device("cpu"),
        seed=45,
    )
    env.reset()
    env.agent_pos[:] = torch.tensor([[0, 2]], dtype=torch.int32)
    env.battery_level[:] = 10
    env.target_pos[:] = torch.tensor([[[7, 7], [6, 6]]], dtype=torch.int32)
    env.previous_distances = env._compute_distances()

    _, reward, _, _, _ = env.step(torch.tensor([0]))

    expected = 4.0 * ((35.0 - 10.0) / 35.0)
    assert torch.allclose(reward, torch.tensor([expected]))


def test_single_agent_charging_navigation_bc_runs_on_unsafe_samples():
    env = BiddingGridworld(
        make_config(
            num_agents=2,
            single_agent_mode=True,
            battery_capacity=50,
            recharge_station_positions=((0, 0), (4, 4), (7, 7)),
            moving_recharge_stations=True,
        ),
        num_envs=64,
        device=torch.device("cpu"),
        seed=46,
    )
    obs, _ = env.reset()
    agent = SingleAgent(
        obs_dim=env.obs_dim,
        num_targets=2,
        actor_hidden_sizes=(32,),
        critic_hidden_sizes=(32,),
        use_target_attention_pooling=True,
        include_target_reached=False,
        energy_feature_dim=env.energy_feature_dim,
    )

    report = pretrain_single_agent_charging_navigation(
        agent,
        env,
        updates=5,
        batch_size=64,
        learning_rate=3e-3,
        activation_margin=8,
        seed=47,
    )

    assert obs.shape[-1] == env.obs_dim
    assert report["updates"] == 5
    assert report["final_active_fraction"] > 0
    assert 0 <= report["final_direction_accuracy"] <= 1


class StressConfigurationParserTests(unittest.TestCase):
    def test_stress_configuration_parsers(self):
        self.assertEqual(parse_optional_int("40"), 40)
        self.assertIsNone(parse_optional_int("none"))
        self.assertEqual(
            parse_recharge_station_positions("0,0; 15,15;29,29"),
            ((0, 0), (15, 15), (29, 29)),
        )

        with self.assertRaises(ValueError):
            parse_recharge_station_positions("0,0,1")

        with self.assertRaises(ValueError):
            parse_recharge_station_positions("")

    def test_shared_agent_deterministic_action_uses_logit_argmax(self):
        agent = SharedAgent(
            obs_dim=8,
            num_actions_per_agent=2,
            actor_hidden_sizes=(),
            critic_hidden_sizes=(8,),
        )
        agent.set_bid_head(2)
        with torch.no_grad():
            agent.direction_head.weight.zero_()
            agent.direction_head.bias.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0]))
            agent.bid_head.weight.zero_()
            agent.bid_head.bias.copy_(torch.tensor([3.0, 1.0, 0.0]))

        obs = torch.randn(4, 8)
        action, _, _, _ = agent.get_action_and_value(
            obs, deterministic=True
        )
        bid_action, _, _, _ = agent.get_bid_action_and_value(
            obs, deterministic=True
        )

        self.assertEqual(action.tolist(), [[3, 0]] * 4)
        self.assertEqual(bid_action.tolist(), [[0, 0]] * 4)

    def test_charger_can_argmax_direction_while_sampling_bids(self):
        agent = SharedAgent(
            obs_dim=8,
            num_actions_per_agent=2,
            actor_hidden_sizes=(),
            critic_hidden_sizes=(8,),
        )
        agent.set_bid_head(2)
        with torch.no_grad():
            agent.direction_head.weight.zero_()
            agent.direction_head.bias.copy_(
                torch.tensor([0.0, 1.0, 2.0, 3.0])
            )
            agent.bid_head.weight.zero_()
            agent.bid_head.bias.zero_()

        torch.manual_seed(34)
        action, _, _, _ = agent.get_action_and_value(
            torch.randn(128, 8), deterministic_direction=True
        )

        self.assertEqual(action[:, 0].unique().tolist(), [3])
        self.assertGreater(len(action[:, 1].unique()), 1)

    def test_single_agent_deterministic_action_uses_logit_argmax(self):
        agent = SingleAgent(
            obs_dim=8,
            num_targets=1,
            actor_hidden_sizes=(),
            critic_hidden_sizes=(8,),
        )
        with torch.no_grad():
            agent.actor[-1].weight.zero_()
            agent.actor[-1].bias.copy_(torch.tensor([0.0, 1.0, 3.0, 2.0]))

        action, _, _, _ = agent.get_action_and_value(
            torch.randn(4, 8), deterministic=True
        )

        self.assertEqual(action.tolist(), [2, 2, 2, 2])

    def test_evaluation_rng_is_reproducible_and_restores_training_rng(self):
        torch.manual_seed(123)
        training_state = torch.get_rng_state()
        with isolated_torch_rng(9001):
            first = torch.rand(5)
        self.assertTrue(torch.equal(torch.get_rng_state(), training_state))

        with isolated_torch_rng(9001):
            second = torch.rand(5)
        self.assertTrue(torch.equal(first, second))

    def test_bid_only_bc_refresh_does_not_update_direction_head(self):
        config = make_config(
            num_agents=2,
            battery_capacity=40,
            recharge_station_positions=((0, 0), (4, 4), (7, 7)),
            charging_agent_enabled=True,
            charging_low_battery_threshold=28,
            charging_activation_margin=4,
        )
        config.bid_upper_bound = 6
        env = BiddingGridworld(
            config,
            num_envs=1,
            device=torch.device("cpu"),
            seed=41,
        )
        agent = SharedAgent(
            obs_dim=env.charging_obs_dim,
            num_actions_per_agent=2,
            actor_hidden_sizes=(32,),
            critic_hidden_sizes=(32,),
        )
        agent.set_bid_head(config.bid_upper_bound)
        before = {
            name: value.detach().clone()
            for name, value in agent.direction_head.state_dict().items()
        }
        environment_rng_state = env.gen.get_state().clone()
        bc_generator = torch.Generator(device="cpu")
        bc_generator.manual_seed(9002)

        report = pretrain_charging_policy(
            agent,
            env,
            updates=20,
            batch_size=256,
            learning_rate=3e-3,
            bid_loss_coef=1.0,
            bid_value=6,
            direction_loss_coef=0.0,
            generator=bc_generator,
        )

        self.assertEqual(report["direction_loss_coef"], 0.0)
        for name, value in agent.direction_head.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        self.assertTrue(
            torch.equal(env.gen.get_state(), environment_rng_state)
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


def test_warm_start_expands_only_energy_inputs_and_preserves_outputs():
    torch.manual_seed(30)
    old_agent = SharedAgent(
        obs_dim=35,
        num_actions_per_agent=2,
        actor_hidden_sizes=(16, 16),
        critic_hidden_sizes=(24, 24),
        use_target_attention_pooling=True,
        target_embed_dim=8,
        target_encoder_hidden_sizes=(8,),
        attention_pooling_layout="centralized",
        include_target_reached=False,
        energy_feature_dim=0,
    )
    old_agent.set_bid_head(2)
    new_agent = SharedAgent(
        obs_dim=42,
        num_actions_per_agent=2,
        actor_hidden_sizes=(16, 16),
        critic_hidden_sizes=(24, 24),
        use_target_attention_pooling=True,
        target_embed_dim=8,
        target_encoder_hidden_sizes=(8,),
        attention_pooling_layout="centralized",
        include_target_reached=False,
        energy_feature_dim=7,
    )
    new_agent.set_bid_head(2)

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint = Path(tmp_dir) / "agent.pt"
        torch.save(old_agent.state_dict(), checkpoint)
        report = load_feeder_checkpoint_with_energy_expansion(
            new_agent, str(checkpoint), energy_feature_dim=7
        )

    assert set(report["expanded_parameter_tensors"]) == {
        "actor_shared.0.weight",
        "critic.0.weight",
    }
    old_encoded = torch.randn(5, old_agent.encoded_obs_dim)
    new_encoded = torch.cat(
        [
            old_encoded[:, :2],
            torch.randn(5, 7),
            old_encoded[:, 2:],
        ],
        dim=1,
    )
    assert torch.allclose(
        old_agent.actor_shared(old_encoded),
        new_agent.actor_shared(new_encoded),
    )
    assert torch.allclose(
        old_agent.critic(old_encoded),
        new_agent.critic(new_encoded),
    )
    assert torch.count_nonzero(
        new_agent.actor_shared[0].weight[:, 2:9]
    ).item() == 0
    assert torch.count_nonzero(
        new_agent.critic[0].weight[:, 2:9]
    ).item() == 0


def test_battery_movement_recharge_and_boundary_cost():
    env = BiddingGridworld(
        make_config(
            num_agents=2,
            single_agent_mode=True,
            battery_capacity=3,
            recharge_station_positions=((0, 0), (0, 2)),
        ),
        num_envs=1,
        device=torch.device("cpu"),
        seed=20,
    )
    env.reset()
    env.target_pos[:] = torch.tensor([[[7, 7], [6, 6]]], dtype=torch.int32)
    env.previous_distances = env._compute_distances()

    env.step(torch.tensor([0]))
    assert env.agent_pos.tolist() == [[0, 0]]
    assert env.battery_level.tolist() == [3]

    env.step(torch.tensor([1]))
    assert env.agent_pos.tolist() == [[0, 1]]
    assert env.battery_level.tolist() == [2]

    _, _, _, _, info = env.step(torch.tensor([1]))
    assert env.agent_pos.tolist() == [[0, 2]]
    assert env.battery_level.tolist() == [3]
    assert info["at_recharge_station"].tolist() == [True]


def test_battery_depletion_tows_and_penalizes_controller():
    env = BiddingGridworld(
        make_config(
            num_agents=2,
            battery_capacity=1,
            recharge_station_positions=((0, 0),),
            battery_depletion_penalty=50.0,
        ),
        num_envs=1,
        device=torch.device("cpu"),
        seed=21,
    )
    env.reset()
    env.target_pos[:] = torch.tensor([[[7, 7], [6, 6]]], dtype=torch.int32)
    env.previous_distances = env._compute_distances()
    actions = torch.tensor([[[1, 1], [0, 0]]])

    _, rewards, _, _, info = env.step(actions)

    assert info["battery_depleted"].tolist() == [True]
    assert env.agent_pos.tolist() == [[0, 0]]
    assert env.battery_level.tolist() == [1]
    assert rewards.tolist() == [[-50.0, 0.0]]


def test_battery_features_and_attention_encoder():
    stations = ((0, 0), (4, 4), (7, 7))
    env = BiddingGridworld(
        make_config(
            num_agents=3,
            battery_capacity=40,
            recharge_station_positions=stations,
        ),
        num_envs=2,
        device=torch.device("cpu"),
        seed=22,
    )
    obs, _ = env.reset()

    assert env.energy_feature_dim == 7
    assert obs.shape == (2, 3, 22)
    expected_energy = torch.tensor(
        [1.0, 0.0, 0.0, 4 / 7, 4 / 7, 1.0, 1.0]
    )
    assert torch.allclose(obs[0, 0, -8:-1], expected_energy)

    agent = SharedAgent(
        obs_dim=env.per_agent_obs_dim,
        num_actions_per_agent=2,
        window_bidding=False,
        use_target_attention_pooling=True,
        attention_pooling_layout="centralized",
        include_target_reached=False,
        energy_feature_dim=env.energy_feature_dim,
    )
    encoded = agent._encode_obs(obs.reshape(-1, env.per_agent_obs_dim))
    assert encoded.shape == (6, agent.encoded_obs_dim)

    single_env = BiddingGridworld(
        make_config(
            num_agents=3,
            single_agent_mode=True,
            battery_capacity=40,
            recharge_station_positions=stations,
        ),
        num_envs=2,
        device=torch.device("cpu"),
        seed=23,
    )
    single_obs, _ = single_env.reset()
    single_agent = SingleAgent(
        obs_dim=single_env.obs_dim,
        num_targets=3,
        use_target_attention_pooling=True,
        include_target_reached=False,
        energy_feature_dim=single_env.energy_feature_dim,
    )
    encoded = single_agent._encode_obs(single_obs)
    assert encoded.shape == (2, single_agent.encoded_obs_dim)


def test_separate_charging_bidder_gets_dense_station_rewards():
    env = BiddingGridworld(
        make_config(
            num_agents=2,
            battery_capacity=5,
            recharge_station_positions=((0, 0),),
            charging_agent_enabled=True,
            charging_low_battery_threshold=3,
            charging_distance_reward_scale=6.0,
            charging_recharge_bonus=10.0,
            charging_depletion_penalty=20.0,
            charging_high_battery_control_penalty=0.5,
        ),
        num_envs=1,
        device=torch.device("cpu"),
        seed=24,
    )
    feeder_obs, _ = env.reset()
    env.agent_pos[:] = torch.tensor([[0, 2]], dtype=torch.int32)
    env.battery_level[:] = 2
    env.target_pos[:] = torch.tensor([[[7, 7], [6, 6]]], dtype=torch.int32)
    env.previous_distances = env._compute_distances()

    assert feeder_obs.shape[1] == 2
    assert env.num_bidders == 3
    assert env.charging_agent_idx == 2
    charging_obs = env.get_charging_observation()
    assert charging_obs.shape == (1, env.charging_obs_dim)
    assert charging_obs.shape[1] == 8

    actions = torch.tensor([[[0, 0], [0, 0], [0, 2]]])
    _, rewards, _, _, info = env.step(actions)

    assert env.agent_pos.tolist() == [[0, 1]]
    assert info["winning_agent"].tolist() == [2]
    assert rewards.shape == (1, 3)
    assert torch.allclose(rewards, torch.tensor([[0.0, 0.0, 2.0]]))

    _, rewards, _, _, info = env.step(actions)
    assert env.agent_pos.tolist() == [[0, 0]]
    assert info["battery_recharged"].tolist() == [True]
    assert env.battery_level.tolist() == [5]
    assert torch.allclose(rewards, torch.tensor([[0.0, 0.0, 14.0]]))

    env.agent_pos[:] = torch.tensor([[0, 1]], dtype=torch.int32)
    env.battery_level[:] = 5
    env.previous_distances = env._compute_distances()
    _, rewards, _, _, info = env.step(actions)
    assert info["battery_recharged"].tolist() == [True]
    assert torch.allclose(rewards, torch.tensor([[0.0, 0.0, -0.5]]))


def test_rollout_logging_counts_charging_agent_wins():
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.args = type("Args", (), {"track": True})()
    trainer.envs = type("Env", (), {"num_bidders": 3})()
    trainer.device = torch.device("cpu")
    trainer._episode_agent_wins = torch.zeros(3, dtype=torch.int64)
    trainer._episode_bid_sum = torch.zeros(())
    trainer._episode_bid_count = torch.zeros((), dtype=torch.int64)
    trainer._episode_bid_min = None
    trainer._episode_bid_max = None
    trainer._episode_reward_no_bid_sum = torch.zeros(())

    trainer._on_rollout_step(
        {"winning_agent": torch.tensor([0, 2, 2, -1])},
        global_step=0,
    )

    assert trainer._episode_agent_wins.tolist() == [1, 0, 2]


def test_low_battery_charging_bid_boost_only_changes_emergency_auction():
    config = make_config(
        num_agents=2,
        battery_capacity=50,
        recharge_station_positions=((0, 0),),
        charging_agent_enabled=True,
        charging_low_battery_threshold=20,
        charging_low_battery_bid_boost=2,
    )
    config.bid_upper_bound = 6
    config.bid_penalty = 1.0
    env = BiddingGridworld(
        config,
        num_envs=1,
        device=torch.device("cpu"),
        seed=25,
    )
    env.reset()
    env.target_pos[:] = torch.tensor([[[7, 7], [6, 6]]], dtype=torch.int32)
    actions = torch.tensor([[[1, 6], [0, 0], [1, 5]]])

    env.battery_level[:] = 20
    _, rewards, _, _, info = env.step(actions)
    assert info["winning_agent"].tolist() == [2]
    assert info["effective_bids"].tolist() == [[6, 0, 7]]
    assert rewards.tolist() == [[-6.0, 0.0, -5.0]]

    env.battery_level[:] = 21
    _, _, _, _, info = env.step(actions)
    assert info["winning_agent"].tolist() == [0]
    assert info["effective_bids"].tolist() == [[6, 0, 5]]

    env.battery_level[:] = 20
    zero_charger_bid = actions.clone()
    zero_charger_bid[:, 2, 1] = 0
    _, _, _, _, info = env.step(zero_charger_bid)
    assert info["winning_agent"].tolist() == [0]
    assert info["effective_bids"].tolist() == [[6, 0, 0]]

    env.config.charging_bid_boost_threshold = 10
    env.battery_level[:] = 20
    _, _, _, _, info = env.step(actions)
    assert info["winning_agent"].tolist() == [0]
    assert info["effective_bids"].tolist() == [[6, 0, 5]]

    env.battery_level[:] = 10
    _, _, _, _, info = env.step(actions)
    assert info["winning_agent"].tolist() == [2]
    assert info["effective_bids"].tolist() == [[6, 0, 7]]


def test_distance_aware_charging_bid_gate():
    env = BiddingGridworld(
        make_config(
            num_agents=2,
            battery_capacity=10,
            recharge_station_positions=((0, 0),),
            movement_energy_cost=2,
            charging_agent_enabled=True,
            charging_activation_margin=0,
        ),
        num_envs=1,
        device=torch.device("cpu"),
        seed=27,
    )
    env.reset()
    env.agent_pos[:] = torch.tensor([[0, 4]], dtype=torch.int32)
    env.target_pos[:] = torch.tensor([[[7, 7], [6, 6]]], dtype=torch.int32)
    actions = torch.tensor([[[0, 1], [0, 0], [0, 2]]])

    env.battery_level[:] = 9
    _, _, _, _, info = env.step(actions)
    assert info["charging_bid_active"].tolist() == [False]
    assert info["effective_bids"].tolist() == [[1, 0, 0]]
    assert info["winning_agent"].tolist() == [0]

    env.window_steps_remaining.zero_()
    env.agent_pos[:] = torch.tensor([[0, 4]], dtype=torch.int32)
    env.battery_level[:] = 8
    _, _, _, _, info = env.step(actions)
    assert info["charging_bid_active"].tolist() == [True]
    assert info["effective_bids"].tolist() == [[1, 0, 2]]
    assert info["winning_agent"].tolist() == [2]


def test_charging_window_releases_immediately_after_recharge():
    config = make_config(
        num_agents=2,
        battery_capacity=10,
        recharge_station_positions=((0, 0),),
        charging_agent_enabled=True,
        charging_release_window_on_recharge=True,
    )
    config.action_window = 5
    env = BiddingGridworld(
        config,
        num_envs=1,
        device=torch.device("cpu"),
        seed=28,
    )
    env.reset()
    env.agent_pos[:] = torch.tensor([[0, 1]], dtype=torch.int32)
    env.battery_level[:] = 2
    env.target_pos[:] = torch.tensor([[[7, 7], [6, 6]]], dtype=torch.int32)

    actions = torch.tensor([[[0, 0], [0, 0], [0, 2]]])
    _, _, _, _, info = env.step(actions)

    assert info["battery_recharged"].tolist() == [True]
    assert info["winning_agent"].tolist() == [2]
    assert info["window_steps_remaining"].tolist() == [0]
    assert info["window_agent"].tolist() == [-1]


def test_programmatic_charging_navigation_overrides_only_charger_direction():
    env = BiddingGridworld(
        make_config(
            num_agents=2,
            battery_capacity=10,
            recharge_station_positions=((0, 0),),
            charging_agent_enabled=True,
            charging_programmatic_navigation=True,
        ),
        num_envs=1,
        device=torch.device("cpu"),
        seed=29,
    )
    env.reset()
    env.agent_pos[:] = torch.tensor([[0, 2]], dtype=torch.int32)
    env.battery_level[:] = 4
    env.target_pos[:] = torch.tensor([[[7, 7], [6, 6]]], dtype=torch.int32)

    # The charger proposes right (1), but its winning move is replaced with
    # the shortest-path direction left (0) toward the station.
    actions = torch.tensor([[[1, 0], [1, 0], [1, 2]]])
    _, _, _, _, info = env.step(actions)

    assert info["winning_agent"].tolist() == [2]
    assert info["programmatic_charging_direction"].tolist() == [0]
    assert env.agent_pos.tolist() == [[0, 1]]

    # A feeder winner retains its own requested direction.
    env.window_steps_remaining.zero_()
    feeder_actions = torch.tensor([[[1, 2], [0, 0], [0, 0]]])
    _, _, _, _, info = env.step(feeder_actions)
    assert info["winning_agent"].tolist() == [0]
    assert env.agent_pos.tolist() == [[0, 2]]


def test_feeder_is_penalized_for_controlling_during_charging_activation():
    env = BiddingGridworld(
        make_config(
            num_agents=2,
            battery_capacity=10,
            recharge_station_positions=((0, 0),),
            charging_agent_enabled=True,
            charging_activation_margin=0,
            feeder_low_battery_control_penalty=3.0,
        ),
        num_envs=1,
        device=torch.device("cpu"),
        seed=30,
    )
    env.reset()
    env.agent_pos[:] = torch.tensor([[0, 2]], dtype=torch.int32)
    env.battery_level[:] = 2
    env.target_pos[:] = torch.tensor([[[7, 7], [6, 6]]], dtype=torch.int32)

    actions = torch.tensor([[[1, 2], [0, 0], [0, 1]]])
    _, rewards, _, _, info = env.step(actions)

    assert info["charging_bid_active"].tolist() == [True]
    assert info["winning_agent"].tolist() == [0]
    assert torch.allclose(rewards, torch.tensor([[-3.0, 0.0, 0.0]]))


def test_bid_only_policy_ignores_direction_log_probability():
    agent = SharedAgent(
        obs_dim=8,
        num_actions_per_agent=2,
        actor_hidden_sizes=(32,),
        critic_hidden_sizes=(32,),
    )
    agent.set_bid_head(6)
    obs = torch.randn(4, 8)
    action_left = torch.tensor([[0, 3]]).expand(4, -1)
    action_right = torch.tensor([[1, 3]]).expand(4, -1)

    _, left_logprob, left_entropy, _ = agent.get_bid_action_and_value(
        obs, action_left
    )
    _, right_logprob, right_entropy, _ = agent.get_bid_action_and_value(
        obs, action_right
    )
    sampled_action, _, _, _ = agent.get_bid_action_and_value(obs)

    assert torch.allclose(left_logprob, right_logprob)
    assert torch.allclose(left_entropy, right_entropy)
    assert sampled_action[:, 0].tolist() == [0, 0, 0, 0]


def test_feeder_yield_auxiliary_loss_only_uses_charging_active_states():
    env = BiddingGridworld(
        make_config(
            num_agents=2,
            battery_capacity=10,
            recharge_station_positions=((0, 0),),
            charging_agent_enabled=True,
            charging_activation_margin=0,
        ),
        num_envs=1,
        device=torch.device("cpu"),
        seed=31,
    )
    agent = SharedAgent(
        obs_dim=env.per_agent_obs_dim,
        num_actions_per_agent=2,
        actor_hidden_sizes=(32,),
        critic_hidden_sizes=(32,),
    )
    agent.set_bid_head(env.config.bid_upper_bound)
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.envs = env
    trainer.args = type(
        "Args",
        (),
        {
            "battery_capacity": 10,
            "grid_size": 8,
            "charging_activation_margin": 0,
            "charging_low_battery_threshold": 3,
            "feeder_yield_aux_coef": 0.5,
            "feeder_yield_aux_bid_head_only": True,
            "feeder_yield_activation_margin": 0,
        },
    )()

    env.reset()
    env.agent_pos[:] = torch.tensor([[0, 2]], dtype=torch.int32)
    env.battery_level[:] = 2
    active_obs = env._get_observation().reshape(
        -1, env.per_agent_obs_dim
    )
    active_loss = trainer._feeder_yield_aux_loss(agent, active_obs)
    agent.zero_grad()
    active_loss.backward()

    env.battery_level[:] = 10
    inactive_obs = env._get_observation().reshape(
        -1, env.per_agent_obs_dim
    )
    inactive_loss = trainer._feeder_yield_aux_loss(
        agent, inactive_obs
    )

    assert active_loss.item() > 0
    assert inactive_loss.item() == 0
    assert agent.bid_head.weight.grad is not None
    assert agent.actor_shared[0].weight.grad is None

    # The feeder threshold can be tighter than charger activation.
    env.config.charging_activation_margin = 4
    env.battery_level[:] = 3
    tighter_obs = env._get_observation().reshape(
        -1, env.per_agent_obs_dim
    )
    tighter_loss = trainer._feeder_yield_aux_loss(agent, tighter_obs)
    assert tighter_loss.item() == 0


def test_charging_behavior_cloning_learns_station_navigation():
    config = make_config(
        num_agents=2,
        battery_capacity=50,
        recharge_station_positions=((0, 0), (4, 4), (7, 7)),
        charging_agent_enabled=True,
        charging_low_battery_threshold=25,
        charging_activation_margin=6,
        charging_reserve_features_enabled=True,
        charging_nearest_station_features_enabled=True,
        moving_recharge_stations=True,
    )
    config.bid_upper_bound = 6
    env = BiddingGridworld(
        config,
        num_envs=1,
        device=torch.device("cpu"),
        seed=26,
    )
    torch.manual_seed(26)
    agent = SharedAgent(
        obs_dim=env.charging_obs_dim,
        num_actions_per_agent=2,
        actor_hidden_sizes=(128, 128, 128),
        critic_hidden_sizes=(64,),
        separate_direction_actor=True,
    )
    agent.set_bid_head(env.config.bid_upper_bound)
    report = pretrain_charging_policy(
        agent,
        env,
        updates=1000,
        batch_size=1024,
        learning_rate=3e-3,
        bid_loss_coef=0.2,
        bid_value=4,
        emergency_margin=2,
        emergency_bid_value=6,
    )
    assert report["final_direction_accuracy"] > 0.9
    assert report["final_bid_accuracy"] > 0.9
    assert report["bid_value"] == 4
    assert report["emergency_margin"] == 2
    assert report["emergency_bid_value"] == 6

    validation_env = BiddingGridworld(
        config,
        num_envs=1024,
        device=torch.device("cpu"),
        seed=27,
    )
    validation_env.reset()
    torch.manual_seed(27)
    validation_env.agent_pos = torch.randint(
        0, config.grid_size, (1024, 2), dtype=torch.int32
    )
    validation_env.current_recharge_station_pos = torch.randint(
        0,
        config.grid_size,
        (1024, validation_env.num_recharge_stations, 2),
        dtype=torch.int32,
    )
    validation_env.battery_level = torch.randint(
        1, config.battery_capacity + 1, (1024,), dtype=torch.int32
    )
    validation_env.window_steps_remaining = torch.randint(
        0, config.action_window, (1024,), dtype=torch.int32
    )
    validation_env.window_agent = torch.where(
        validation_env.window_steps_remaining > 0,
        torch.full((1024,), validation_env.charging_agent_idx, dtype=torch.int32),
        torch.full((1024,), -1, dtype=torch.int32),
    )
    validation_obs = validation_env.get_charging_observation()
    with torch.no_grad():
        direction_features = agent.direction_actor(validation_obs)
        predicted_direction = agent.direction_head(
            direction_features
        ).argmax(dim=1)
    expected_direction = validation_env._direction_to_nearest_recharge_station(
        validation_env.agent_pos
    )
    context_accuracy = (
        predicted_direction == expected_direction
    ).to(torch.float32).mean().item()
    assert context_accuracy > 0.98


def test_separate_direction_actor_is_untouched_by_bid_only_loss():
    torch.manual_seed(35)
    agent = SharedAgent(
        obs_dim=10,
        num_actions_per_agent=2,
        actor_hidden_sizes=(16, 16),
        critic_hidden_sizes=(16,),
        separate_direction_actor=True,
    )
    agent.set_bid_head(3)
    obs = torch.randn(32, 10)

    _, log_prob, _, _ = agent.get_bid_action_and_value(obs)
    (-log_prob.mean()).backward()

    assert any(parameter.grad is not None for parameter in agent.actor_shared.parameters())
    assert all(parameter.grad is None for parameter in agent.direction_actor.parameters())
    assert all(parameter.grad is None for parameter in agent.direction_head.parameters())


def test_checkpoint_setup_skips_bid_only_pretraining_requirement():
    args = Args(
        cuda=False,
        track=False,
        num_envs=1,
        num_steps=2,
        num_agents=2,
        battery_capacity=50,
        recharge_station_positions=((0, 0), (4, 4), (7, 7)),
        charging_agent_enabled=True,
        charging_programmatic_navigation=False,
        charging_separate_direction_actor=True,
        charging_ppo_bid_only=True,
        charging_bc_updates=0,
    )

    with unittest.TestCase().assertRaises(ValueError):
        PPOTrainer(args).setup()

    trainer = PPOTrainer(args)
    trainer.setup(skip_pretraining=True)
    try:
        assert trainer.charging_agent is not None
        assert trainer.charging_agent.separate_direction_actor
    finally:
        trainer.envs.close()


def test_recharge_stations_move_per_environment_and_update_observations():
    config = make_config(
        num_agents=2,
        battery_capacity=50,
        recharge_station_positions=((3, 3),),
        charging_agent_enabled=True,
        moving_recharge_stations=True,
        recharge_station_direction_change_prob=0.0,
        recharge_station_move_interval=2,
    )
    env = BiddingGridworld(
        config,
        num_envs=2,
        device=torch.device("cpu"),
        seed=31,
    )
    env.reset()
    env.current_recharge_station_pos[:] = torch.tensor(
        [[[3, 3]], [[4, 4]]], dtype=torch.int32
    )
    env.recharge_station_directions[:] = torch.tensor(
        [[1], [2]], dtype=torch.int32
    )
    actions = torch.zeros((2, 3, 2), dtype=torch.int64)

    env.step(actions)
    assert env.current_recharge_station_pos.tolist() == [
        [[3, 3]],
        [[4, 4]],
    ]

    obs, _, _, _, info = env.step(actions)
    assert env.current_recharge_station_pos.tolist() == [
        [[3, 4]],
        [[3, 4]],
    ]
    assert info["recharge_station_positions"].tolist() == [
        [[3, 4]],
        [[3, 4]],
    ]
    energy_start = obs.shape[-1] - 1 - env.energy_feature_dim
    station_features = obs[:, 0, energy_start + 1 : energy_start + 3]
    assert torch.allclose(
        station_features,
        torch.tensor([[3 / 7, 4 / 7], [3 / 7, 4 / 7]]),
    )


def test_batched_video_capture_includes_priority_and_charging_state():
    config = make_config(
        grid_size=8,
        num_agents=2,
        battery_capacity=20,
        recharge_station_positions=((0, 0), (4, 4), (7, 7)),
        charging_agent_enabled=True,
        moving_recharge_stations=True,
        recharge_station_move_interval=1,
    )
    config.max_steps = 3
    env = BiddingGridworld(
        config,
        num_envs=2,
        device=torch.device("cpu"),
        seed=44,
    )

    def policy_fn(_obs):
        return torch.zeros((2, env.num_bidders, 2), dtype=torch.long)

    stats = evaluate_multi_agent_policy_batched(
        env,
        policy_fn,
        num_episodes=2,
        verbose=False,
        capture_episode_count=1,
    )
    episode = stats["episode_data_list"][0]
    assert len(episode["render_states"]) == 3
    first_state = episode["render_states"][0]
    second_state = episode["render_states"][1]
    assert len(first_state["target_priorities"]) == config.num_agents
    assert first_state["battery_level"] == config.battery_capacity
    assert len(first_state["recharge_station_positions"]) == 3
    assert (
        first_state["recharge_station_positions"]
        != second_state["recharge_station_positions"]
    )
    assert "charging_bid_active" in episode["step_details"][0]
    assert "effective_bids" in episode["step_details"][0]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "priority_charging.mp4"
        env.create_competition_gif(
            episode, output_path, fps=2, frame_stride=2
        )
        assert output_path.is_file()
        assert output_path.stat().st_size > 1000


def test_learned_charger_navigation_diagnostic_tracks_optimal_direction():
    config = make_config(
        num_agents=2,
        battery_capacity=50,
        recharge_station_positions=((0, 0),),
        charging_agent_enabled=True,
    )
    env = BiddingGridworld(
        config,
        num_envs=1,
        device=torch.device("cpu"),
        seed=33,
    )
    env.reset()
    env.agent_pos[:] = torch.tensor([[0, 2]], dtype=torch.int32)
    env.previous_distances = env._compute_distances()
    actions = torch.tensor([[[1, 0], [1, 0], [0, 2]]])

    _, _, _, _, info = env.step(actions)

    assert info["winning_agent"].tolist() == [2]
    assert info["optimal_charging_direction"].tolist() == [0]
    assert info["charging_navigation_active"].tolist() == [True]
    assert info["charging_direction_optimal"].tolist() == [True]


def test_partial_reset_restores_only_selected_station_trajectories():
    config = make_config(
        num_agents=2,
        battery_capacity=50,
        recharge_station_positions=((0, 0), (7, 7)),
        charging_agent_enabled=True,
        moving_recharge_stations=True,
    )
    env = BiddingGridworld(
        config,
        num_envs=2,
        device=torch.device("cpu"),
        seed=32,
    )
    env.reset()
    env.current_recharge_station_pos[:] = torch.tensor(
        [[[2, 2], [5, 5]], [[3, 3], [4, 4]]], dtype=torch.int32
    )
    env.recharge_station_move_counters[:] = 3
    second_directions = env.recharge_station_directions[1].clone()

    env.partial_reset(torch.tensor([True, False]))

    assert env.current_recharge_station_pos.tolist() == [
        [[0, 0], [7, 7]],
        [[3, 3], [4, 4]],
    ]
    assert env.recharge_station_move_counters.tolist() == [[0, 0], [3, 3]]
    assert torch.equal(env.recharge_station_directions[1], second_directions)


def test_charging_reserve_features_use_capacity_invariant_physical_slack():
    config = make_config(
        grid_size=8,
        num_agents=2,
        battery_capacity=40,
        recharge_station_positions=((0, 0),),
        movement_energy_cost=2,
        charging_agent_enabled=True,
        charging_low_battery_threshold=20,
        charging_reserve_features_enabled=True,
    )
    env = BiddingGridworld(
        config,
        num_envs=1,
        device=torch.device("cpu"),
        seed=29,
    )
    env.reset()
    env.agent_pos[:] = torch.tensor([[0, 2]], dtype=torch.int32)
    env.battery_level[:] = 4

    obs = env.get_charging_observation()

    assert env.charging_obs_dim == 10
    assert obs.shape == (1, 10)
    assert torch.allclose(obs[0, 3:5], torch.tensor([4 / 28, 0.0]))


def test_charging_observation_exposes_nearest_station_relative_position():
    config = make_config(
        grid_size=8,
        num_agents=2,
        battery_capacity=40,
        recharge_station_positions=((0, 0), (7, 7)),
        charging_agent_enabled=True,
        charging_nearest_station_features_enabled=True,
    )
    env = BiddingGridworld(
        config,
        num_envs=1,
        device=torch.device("cpu"),
        seed=36,
    )
    env.reset()
    env.agent_pos[:] = torch.tensor([[2, 5]], dtype=torch.int32)

    obs = env.get_charging_observation()

    assert env.charging_obs_dim == 13
    assert torch.allclose(obs[0, 3:5], torch.tensor([-2 / 7, -5 / 7]))


def test_activation_margin_covers_energy_loss_during_feeder_window():
    config = make_config(
        grid_size=8,
        num_agents=2,
        battery_capacity=70,
        recharge_station_positions=((0, 0),),
        movement_energy_cost=2,
        charging_agent_enabled=True,
        charging_low_battery_threshold=56,
        charging_activation_margin=12,
        charging_programmatic_navigation=True,
    )
    config.action_window = 3
    env = BiddingGridworld(
        config,
        num_envs=1,
        device=torch.device("cpu"),
        seed=30,
    )
    env.reset()
    env.agent_pos[:] = torch.tensor([[0, 2]], dtype=torch.int32)
    env.battery_level[:] = 18
    env.target_pos[:] = torch.tensor([[[7, 7], [6, 6]]], dtype=torch.int32)
    env.previous_distances = env._compute_distances()
    actions = torch.tensor([[[1, 1], [0, 0], [0, 2]]])

    depleted = []
    for _ in range(8):
        _, _, _, _, info = env.step(actions)
        depleted.append(bool(info["battery_depleted"].item()))

    assert not any(depleted)
    assert env.agent_pos.tolist() == [[0, 0]]
    assert env.battery_level.tolist() == [70]


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

    def test_warm_start_energy_expansion(self):
        test_warm_start_expands_only_energy_inputs_and_preserves_outputs()

    def test_evaluation_priority_sums(self):
        test_all_evaluators_sum_priorities_reached_before_respawn()

    def test_battery_movement_and_recharge(self):
        test_battery_movement_recharge_and_boundary_cost()

    def test_battery_depletion(self):
        test_battery_depletion_tows_and_penalizes_controller()

    def test_battery_observation_and_attention(self):
        test_battery_features_and_attention_encoder()

    def test_separate_charging_bidder(self):
        test_separate_charging_bidder_gets_dense_station_rewards()

    def test_charging_win_logging(self):
        test_rollout_logging_counts_charging_agent_wins()

    def test_charging_bid_boost(self):
        test_low_battery_charging_bid_boost_only_changes_emergency_auction()

    def test_charging_bid_gate(self):
        test_distance_aware_charging_bid_gate()

    def test_charging_window_release(self):
        test_charging_window_releases_immediately_after_recharge()

    def test_programmatic_charging_navigation(self):
        test_programmatic_charging_navigation_overrides_only_charger_direction()

    def test_feeder_low_battery_control_penalty(self):
        test_feeder_is_penalized_for_controlling_during_charging_activation()

    def test_bid_only_policy(self):
        test_bid_only_policy_ignores_direction_log_probability()

    def test_feeder_yield_auxiliary_loss(self):
        test_feeder_yield_auxiliary_loss_only_uses_charging_active_states()

    def test_charging_behavior_cloning(self):
        test_charging_behavior_cloning_learns_station_navigation()

    def test_separate_charger_direction_actor(self):
        test_separate_direction_actor_is_untouched_by_bid_only_loss()

    def test_checkpoint_setup_without_pretraining(self):
        test_checkpoint_setup_skips_bid_only_pretraining_requirement()

    def test_moving_recharge_stations(self):
        test_recharge_stations_move_per_environment_and_update_observations()

    def test_priority_charging_video_capture(self):
        test_batched_video_capture_includes_priority_and_charging_state()

    def test_learned_charger_navigation_diagnostic(self):
        test_learned_charger_navigation_diagnostic_tracks_optimal_direction()

    def test_moving_recharge_station_partial_reset(self):
        test_partial_reset_restores_only_selected_station_trajectories()

    def test_charging_reserve_features(self):
        test_charging_reserve_features_use_capacity_invariant_physical_slack()

    def test_charging_nearest_station_features(self):
        test_charging_observation_exposes_nearest_station_relative_position()

    def test_activation_margin_covers_feeder_window(self):
        test_activation_margin_covers_energy_loss_during_feeder_window()
