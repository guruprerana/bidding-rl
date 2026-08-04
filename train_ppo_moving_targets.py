#!/usr/bin/env python3
"""
PPO Training Script for Bidding Gridworld

This script trains PPO policies for both single-agent and multi-agent bidding.

Features:
- Single-agent mode: One agent navigates to collect all targets
- Multi-agent mode: Multiple agents bid for control to reach their targets
- Periodic checkpointing
- Regular rollout evaluations with MP4 generation
- Comprehensive wandb logging
- Moving target environment support
- All configuration in one place (no CLI arguments needed)

Usage:
    python train_ppo_moving_targets.py

Configure all parameters in the CONFIGURATION section of the main() function.
Set SINGLE_AGENT_MODE = True for single-agent navigation, False for multi-agent bidding.
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bidding_gridworld.bidding_ppo import Args
from bidding_gridworld.single_agent_ppo import SingleAgentArgs
from bidding_gridworld.experiment import PPOMovingTargetsExperiment


def parse_optional_int(value):
    """Parse an integer or a string that explicitly disables the setting."""
    if value.strip().lower() in {"none", "off", "disabled"}:
        return None
    return int(value)


def parse_recharge_station_positions(value):
    """Parse `row,col;row,col` station coordinates from an environment value."""
    positions = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        coordinates = tuple(int(part.strip()) for part in item.split(","))
        if len(coordinates) != 2:
            raise ValueError(
                "recharge station positions must use `row,col;row,col` format"
            )
        positions.append(coordinates)
    if not positions:
        raise ValueError("at least one recharge station position is required")
    return tuple(positions)


def main():
    """Main training function."""

    def env_value(name, default, cast):
        value = os.environ.get(name)
        return default if value is None else cast(value)

    # ========================================================================
    # CONFIGURATION - Modify parameters here
    # ========================================================================

    # Mode selection
    SINGLE_AGENT_MODE = env_value(
        "SINGLE_AGENT_MODE", False, lambda x: x.lower() == "true"
    )
    MOVING_TARGETS = env_value(
        "MOVING_TARGETS", True, lambda x: x.lower() == "true"
    )

    # Experiment settings
    EXPERIMENT_NAME = env_value(
        "EXPERIMENT_NAME",
        "gridw_multiagent_priority_charging_agent",
        str,
    )
    BASE_LOG_DIR = env_value(
        "BASE_LOG_DIR",
        "logs/cat_feeder_complexity_experiments",
        str,
    )
    CHECKPOINT_FREQ = env_value("CHECKPOINT_FREQ", 10, int)
    EVAL_FREQ = env_value("EVAL_FREQ", 10, int)
    NUM_EVAL_EPISODES = env_value("NUM_EVAL_EPISODES", 20, int)
    DETERMINISTIC_EVAL = env_value(
        "DETERMINISTIC_EVAL", False, lambda x: x.lower() == "true"
    )
    EVAL_SEED = env_value("EVAL_SEED", None, parse_optional_int)
    POLICY_SAMPLE_SEED = env_value(
        "POLICY_SAMPLE_SEED", None, parse_optional_int
    )
    FINAL_MODEL_IN_LOG_DIR = env_value(
        "FINAL_MODEL_IN_LOG_DIR", False, lambda x: x.lower() == "true"
    )
    NUM_VIDEO_EPISODES = 0  # Number of episodes to save as MP4s
    VIDEO_FREQ = 0  # Save video rollouts every N iterations (0 = use eval freq)
    EVAL_NUM_AGENTS = (
        env_value("EVAL_NUM_AGENTS", 0, int)
        if "EVAL_NUM_AGENTS" in os.environ
        else None
    )
    EVAL_NUM_TARGETS = (
        env_value("EVAL_NUM_TARGETS", 0, int)
        if "EVAL_NUM_TARGETS" in os.environ
        else None
    )

    # Environment parameters
    GRID_SIZE = env_value("GRID_SIZE", 30, int)
    NUM_AGENTS = env_value("NUM_AGENTS", 8, int)
    # E[priority] = 2.5, so coefficient 20 preserves the original mean feeding reward of 50.
    TARGET_REWARD = env_value("TARGET_REWARD", 20.0, float)
    MAX_STEPS = env_value("MAX_STEPS", 2000, int)
    EVAL_MAX_STEPS = env_value("EVAL_MAX_STEPS", 2000, int)
    DISTANCE_REWARD_SCALE = env_value(
        "DISTANCE_REWARD_SCALE", 0.6, float
    )
    TARGET_EXPIRY_STEPS = env_value("TARGET_EXPIRY_STEPS", 200, int)
    TARGET_EXPIRY_PENALTY = env_value(
        "TARGET_EXPIRY_PENALTY", 50.0, float
    )
    REWARD_DECAY_FACTOR = 0.0  # Single-agent only: decay rewards for over-visited targets (0.0 = no decay, 0.5 = moderate)
    URGENCY_WEIGHTED_SCALARIZATION = env_value(
        "URGENCY_WEIGHTED_SCALARIZATION",
        False,
        lambda x: x.lower() == "true",
    )
    USE_TARGET_PRIORITIES = env_value(
        "USE_TARGET_PRIORITIES", True, lambda x: x.lower() == "true"
    )

    # Shared feeder battery (None disables recharge mechanics and preserves legacy behavior)
    BATTERY_CAPACITY = env_value(
        "BATTERY_CAPACITY", 50, parse_optional_int
    )
    RECHARGE_STATION_POSITIONS = env_value(
        "RECHARGE_STATION_POSITIONS",
        ((0, 0), (15, 15), (29, 29)),
        parse_recharge_station_positions,
    )
    MOVING_RECHARGE_STATIONS = env_value(
        "MOVING_RECHARGE_STATIONS",
        False,
        lambda x: x.lower() == "true",
    )
    RECHARGE_STATION_DIRECTION_CHANGE_PROB = env_value(
        "RECHARGE_STATION_DIRECTION_CHANGE_PROB", 0.1, float
    )
    RECHARGE_STATION_MOVE_INTERVAL = env_value(
        "RECHARGE_STATION_MOVE_INTERVAL", 5, int
    )
    MOVEMENT_ENERGY_COST = env_value("MOVEMENT_ENERGY_COST", 1, int)
    BATTERY_DEPLETION_PENALTY = env_value(
        "BATTERY_DEPLETION_PENALTY", 10.0, float
    )
    CHARGING_AGENT_ENABLED = env_value(
        "CHARGING_AGENT_ENABLED", True, lambda x: x.lower() == "true"
    )
    CHARGING_LOW_BATTERY_THRESHOLD = env_value(
        "CHARGING_LOW_BATTERY_THRESHOLD", 20, int
    )
    CHARGING_DISTANCE_REWARD_SCALE = env_value(
        "CHARGING_DISTANCE_REWARD_SCALE", 2.0, float
    )
    CHARGING_RECHARGE_BONUS = env_value(
        "CHARGING_RECHARGE_BONUS", 20.0, float
    )
    CHARGING_DEPLETION_PENALTY = env_value(
        "CHARGING_DEPLETION_PENALTY", 50.0, float
    )
    CHARGING_HIGH_BATTERY_CONTROL_PENALTY = env_value(
        "CHARGING_HIGH_BATTERY_CONTROL_PENALTY", 0.0, float
    )
    FEEDER_LOW_BATTERY_CONTROL_PENALTY = env_value(
        "FEEDER_LOW_BATTERY_CONTROL_PENALTY", 0.0, float
    )
    FEEDER_YIELD_AUX_COEF = env_value(
        "FEEDER_YIELD_AUX_COEF", 0.0, float
    )
    FEEDER_YIELD_AUX_BID_HEAD_ONLY = env_value(
        "FEEDER_YIELD_AUX_BID_HEAD_ONLY",
        False,
        lambda x: x.lower() == "true",
    )
    FEEDER_YIELD_ACTIVATION_MARGIN = (
        env_value("FEEDER_YIELD_ACTIVATION_MARGIN", 0, int)
        if "FEEDER_YIELD_ACTIVATION_MARGIN" in os.environ
        else None
    )
    CHARGING_LOW_BATTERY_BID_BOOST = env_value(
        "CHARGING_LOW_BATTERY_BID_BOOST", 0, int
    )
    CHARGING_BID_BOOST_THRESHOLD = (
        env_value("CHARGING_BID_BOOST_THRESHOLD", 0, int)
        if "CHARGING_BID_BOOST_THRESHOLD" in os.environ
        else None
    )
    CHARGING_ACTIVATION_MARGIN = (
        env_value("CHARGING_ACTIVATION_MARGIN", 0, int)
        if "CHARGING_ACTIVATION_MARGIN" in os.environ
        else None
    )
    CHARGING_RELEASE_WINDOW_ON_RECHARGE = env_value(
        "CHARGING_RELEASE_WINDOW_ON_RECHARGE",
        False,
        lambda x: x.lower() == "true",
    )
    CHARGING_PROGRAMMATIC_NAVIGATION = env_value(
        "CHARGING_PROGRAMMATIC_NAVIGATION",
        False,
        lambda x: x.lower() == "true",
    )
    CHARGING_GREEDY_NAVIGATION_EVAL = env_value(
        "CHARGING_GREEDY_NAVIGATION_EVAL",
        False,
        lambda x: x.lower() == "true",
    )
    CHARGING_SEPARATE_DIRECTION_ACTOR = env_value(
        "CHARGING_SEPARATE_DIRECTION_ACTOR",
        False,
        lambda x: x.lower() == "true",
    )
    CHARGING_PPO_BID_ONLY = env_value(
        "CHARGING_PPO_BID_ONLY",
        False,
        lambda x: x.lower() == "true",
    )
    CHARGING_RESERVE_FEATURES_ENABLED = env_value(
        "CHARGING_RESERVE_FEATURES_ENABLED",
        False,
        lambda x: x.lower() == "true",
    )
    CHARGING_NEAREST_STATION_FEATURES_ENABLED = env_value(
        "CHARGING_NEAREST_STATION_FEATURES_ENABLED",
        False,
        lambda x: x.lower() == "true",
    )

    # Multi-agent specific parameters (ignored in single-agent mode)
    BID_UPPER_BOUND = env_value("BID_UPPER_BOUND", 6, int)
    BID_PENALTY = env_value("BID_PENALTY", 0.1, float)
    ACTION_WINDOW = env_value("ACTION_WINDOW", 5, int)
    WINDOW_BIDDING = env_value(
        "WINDOW_BIDDING",
        False,
        lambda x: x.lower() == "true",
    )
    WINDOW_PENALTY = env_value("WINDOW_PENALTY", 0.05, float)
    VISIBLE_TARGETS = None  # None = centralized (all targets visible); 0 = only own target; N = own target + N nearest others
    BIDDING_MECHANISM = env_value("BIDDING_MECHANISM", "all_pay", str)
    PROGRAMMATIC_BIDDING = env_value("PROGRAMMATIC_BIDDING", "none", str)
    BID_ONLY_PPO = env_value(
        "BID_ONLY_PPO", False, lambda x: x.lower() == "true"
    )
    FREEZE_NAVIGATION_DURING_BID_ONLY = env_value(
        "FREEZE_NAVIGATION_DURING_BID_ONLY",
        True,
        lambda x: x.lower() == "true",
    )
    SEPARATE_BID_ACTOR = env_value(
        "SEPARATE_BID_ACTOR", False, lambda x: x.lower() == "true"
    )
    ORDINAL_BID_HEAD = env_value(
        "ORDINAL_BID_HEAD", False, lambda x: x.lower() == "true"
    )
    POLICY_WARM_START_CHECKPOINT = (
        os.environ.get("POLICY_WARM_START_CHECKPOINT") or None
    )
    BID_CREDIT_ASSIGNMENT = env_value(
        "BID_CREDIT_ASSIGNMENT", "individual", str
    )
    BID_OTHER_REWARD_FRACTION = env_value(
        "BID_OTHER_REWARD_FRACTION", 1.0, float
    )
    BID_MIXED_REWARD_NORMALIZE = env_value(
        "BID_MIXED_REWARD_NORMALIZE",
        False,
        lambda x: x.lower() == "true",
    )
    FACTORIZED_AUCTION_PPO = env_value(
        "FACTORIZED_AUCTION_PPO", False, lambda x: x.lower() == "true"
    )
    COUNTERFACTUAL_BID_ADVANTAGES = env_value(
        "COUNTERFACTUAL_BID_ADVANTAGES",
        False,
        lambda x: x.lower() == "true",
    )
    COUNTERFACTUAL_BID_ADVANTAGE_MIX = env_value(
        "COUNTERFACTUAL_BID_ADVANTAGE_MIX", 1.0, float
    )
    BID_VF_COEF = (
        env_value("BID_VF_COEF", 0.0, float)
        if "BID_VF_COEF" in os.environ
        else None
    )

    # Moving targets parameters (only used if MOVING_TARGETS = True)
    DIRECTION_CHANGE_PROB = 0.1
    TARGET_MOVE_INTERVAL = 5

    # Training parameters
    NUM_ITERATIONS = env_value("NUM_ITERATIONS", 400, int)
    LEARNING_RATE = env_value("LEARNING_RATE", 0.00025, float)
    CHARGING_LEARNING_RATE = env_value(
        "CHARGING_LEARNING_RATE", 0.00025, float
    )
    FEEDER_WARM_START_CHECKPOINT = (
        os.environ.get("FEEDER_WARM_START_CHECKPOINT") or None
    )
    FEEDER_FREEZE_ITERATIONS = env_value(
        "FEEDER_FREEZE_ITERATIONS", 0, int
    )
    FEEDER_FINETUNE_LEARNING_RATE = env_value(
        "FEEDER_FINETUNE_LEARNING_RATE", 0.00005, float
    )
    CHARGING_BC_UPDATES = env_value("CHARGING_BC_UPDATES", 0, int)
    CHARGING_BC_BATCH_SIZE = env_value(
        "CHARGING_BC_BATCH_SIZE", 4096, int
    )
    CHARGING_BC_LEARNING_RATE = env_value(
        "CHARGING_BC_LEARNING_RATE", 0.001, float
    )
    CHARGING_BC_BID_LOSS_COEF = env_value(
        "CHARGING_BC_BID_LOSS_COEF", 0.0, float
    )
    CHARGING_BC_BID_VALUE = (
        env_value("CHARGING_BC_BID_VALUE", 0, int)
        if "CHARGING_BC_BID_VALUE" in os.environ
        else None
    )
    CHARGING_BC_EMERGENCY_MARGIN = (
        env_value("CHARGING_BC_EMERGENCY_MARGIN", 0, int)
        if "CHARGING_BC_EMERGENCY_MARGIN" in os.environ
        else None
    )
    CHARGING_BC_EMERGENCY_BID_VALUE = (
        env_value("CHARGING_BC_EMERGENCY_BID_VALUE", 0, int)
        if "CHARGING_BC_EMERGENCY_BID_VALUE" in os.environ
        else None
    )
    CHARGING_BC_REFRESH_UPDATES = env_value(
        "CHARGING_BC_REFRESH_UPDATES", 0, int
    )
    CHARGING_BC_REFRESH_LEARNING_RATE = (
        env_value("CHARGING_BC_REFRESH_LEARNING_RATE", 0.0, float)
        if "CHARGING_BC_REFRESH_LEARNING_RATE" in os.environ
        else None
    )
    LR_MIN = env_value("LR_MIN", 0.0, float)
    NUM_ENVS = env_value("NUM_ENVS", 4096, int)
    NUM_STEPS = env_value("NUM_STEPS", 256, int)
    NUM_MINIBATCHES = env_value("NUM_MINIBATCHES", 256, int)
    UPDATE_EPOCHS = env_value("UPDATE_EPOCHS", 4, int)
    ANNEAL_LR = env_value(
        "ANNEAL_LR", True, lambda x: x.lower() == "true"
    )
    GAMMA = env_value("GAMMA", 0.99, float)
    GAE_LAMBDA = env_value("GAE_LAMBDA", 0.95, float)
    NORM_ADV = env_value(
        "NORM_ADV", True, lambda x: x.lower() == "true"
    )
    CLIP_COEF = env_value("CLIP_COEF", 0.05, float)
    CLIP_VLOSS = env_value(
        "CLIP_VLOSS", False, lambda x: x.lower() == "true"
    )
    ENT_COEF = env_value("ENT_COEF", 0.03, float)
    VF_COEF = env_value("VF_COEF", 1.0, float)
    MAX_GRAD_NORM = env_value("MAX_GRAD_NORM", 0.5, float)
    TARGET_KL = None
    SEED = env_value("SEED", 1825, int)
    # Network architecture
    ACTOR_HIDDEN_SIZES = [128, 128, 128, 128]
    CRITIC_HIDDEN_SIZES = [256, 256, 256, 256]
    USE_TARGET_ATTENTION_POOLING = True
    TARGET_EMBED_DIM = 64
    TARGET_ENCODER_HIDDEN_SIZES = [64, 64]

    # Wandb tracking
    WANDB_PROJECT = "bidding-rl"
    WANDB_ENTITY = None
    TRACK = env_value("TRACK", True, lambda x: x.lower() == "true")
    LOG_VIDEOS_TO_WANDB = False  # Set to True to upload MP4s to wandb

    # ========================================================================
    # End of configuration
    # ========================================================================

    # Create appropriate Args based on mode
    if SINGLE_AGENT_MODE:
        args = SingleAgentArgs(
            exp_name=EXPERIMENT_NAME or "single_agent_ppo",
            seed=SEED,
            track=TRACK,
            wandb_project_name=WANDB_PROJECT,
            wandb_entity=WANDB_ENTITY,

            # Environment config
            grid_size=GRID_SIZE,
            num_targets=NUM_AGENTS,  # In single-agent mode, this is number of targets
            target_reward=TARGET_REWARD,
            max_steps=MAX_STEPS,
            distance_reward_scale=DISTANCE_REWARD_SCALE,
            target_expiry_steps=TARGET_EXPIRY_STEPS,
            target_expiry_penalty=TARGET_EXPIRY_PENALTY,
            reward_decay_factor=REWARD_DECAY_FACTOR,
            urgency_weighted_scalarization=(
                URGENCY_WEIGHTED_SCALARIZATION
            ),
            use_target_priorities=USE_TARGET_PRIORITIES,
            moving_targets=MOVING_TARGETS,
            direction_change_prob=DIRECTION_CHANGE_PROB,
            target_move_interval=TARGET_MOVE_INTERVAL,
            battery_capacity=BATTERY_CAPACITY,
            recharge_station_positions=RECHARGE_STATION_POSITIONS,
            moving_recharge_stations=MOVING_RECHARGE_STATIONS,
            recharge_station_direction_change_prob=(
                RECHARGE_STATION_DIRECTION_CHANGE_PROB
            ),
            recharge_station_move_interval=(
                RECHARGE_STATION_MOVE_INTERVAL
            ),
            movement_energy_cost=MOVEMENT_ENERGY_COST,
            battery_depletion_penalty=BATTERY_DEPLETION_PENALTY,

            # Training config
            num_iterations=NUM_ITERATIONS,
            learning_rate=LEARNING_RATE,
            lr_min=LR_MIN,
            num_envs=NUM_ENVS,
            num_steps=NUM_STEPS,
            num_minibatches=NUM_MINIBATCHES,
            update_epochs=UPDATE_EPOCHS,
            anneal_lr=ANNEAL_LR,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            norm_adv=NORM_ADV,
            clip_coef=CLIP_COEF,
            clip_vloss=CLIP_VLOSS,
            ent_coef=ENT_COEF,
            vf_coef=VF_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            target_kl=TARGET_KL,
            actor_hidden_sizes=ACTOR_HIDDEN_SIZES,
            critic_hidden_sizes=CRITIC_HIDDEN_SIZES,
            use_target_attention_pooling=USE_TARGET_ATTENTION_POOLING,
            target_embed_dim=TARGET_EMBED_DIM,
            target_encoder_hidden_sizes=TARGET_ENCODER_HIDDEN_SIZES,
        )
    else:
        args = Args(
            exp_name=EXPERIMENT_NAME or "multi_agent_ppo",
            seed=SEED,
            track=TRACK,
            wandb_project_name=WANDB_PROJECT,
            wandb_entity=WANDB_ENTITY,

            # Environment config
            grid_size=GRID_SIZE,
            num_agents=NUM_AGENTS,
            bid_upper_bound=BID_UPPER_BOUND,
            bid_penalty=BID_PENALTY,
            target_reward=TARGET_REWARD,
            max_steps=MAX_STEPS,
            action_window=ACTION_WINDOW,
            distance_reward_scale=DISTANCE_REWARD_SCALE,
            target_expiry_steps=TARGET_EXPIRY_STEPS,
            target_expiry_penalty=TARGET_EXPIRY_PENALTY,
            moving_targets=MOVING_TARGETS,
            direction_change_prob=DIRECTION_CHANGE_PROB,
            target_move_interval=TARGET_MOVE_INTERVAL,
            window_bidding=WINDOW_BIDDING,
            window_penalty=WINDOW_PENALTY,
            visible_targets=VISIBLE_TARGETS,
            bidding_mechanism=BIDDING_MECHANISM,
            use_target_priorities=USE_TARGET_PRIORITIES,
            programmatic_bidding=PROGRAMMATIC_BIDDING,
            bid_only_ppo=BID_ONLY_PPO,
            freeze_navigation_during_bid_only=(
                FREEZE_NAVIGATION_DURING_BID_ONLY
            ),
            separate_bid_actor=SEPARATE_BID_ACTOR,
            bid_actor_hidden_sizes=(128, 128),
            ordinal_bid_head=ORDINAL_BID_HEAD,
            policy_warm_start_checkpoint=POLICY_WARM_START_CHECKPOINT,
            bid_credit_assignment=BID_CREDIT_ASSIGNMENT,
            bid_other_reward_fraction=BID_OTHER_REWARD_FRACTION,
            bid_mixed_reward_normalize=BID_MIXED_REWARD_NORMALIZE,
            factorized_auction_ppo=FACTORIZED_AUCTION_PPO,
            counterfactual_bid_advantages=(
                COUNTERFACTUAL_BID_ADVANTAGES
            ),
            counterfactual_bid_advantage_mix=(
                COUNTERFACTUAL_BID_ADVANTAGE_MIX
            ),
            bid_vf_coef=BID_VF_COEF,
            battery_capacity=BATTERY_CAPACITY,
            recharge_station_positions=RECHARGE_STATION_POSITIONS,
            moving_recharge_stations=MOVING_RECHARGE_STATIONS,
            recharge_station_direction_change_prob=(
                RECHARGE_STATION_DIRECTION_CHANGE_PROB
            ),
            recharge_station_move_interval=(
                RECHARGE_STATION_MOVE_INTERVAL
            ),
            movement_energy_cost=MOVEMENT_ENERGY_COST,
            battery_depletion_penalty=BATTERY_DEPLETION_PENALTY,
            charging_agent_enabled=CHARGING_AGENT_ENABLED,
            charging_low_battery_threshold=CHARGING_LOW_BATTERY_THRESHOLD,
            charging_distance_reward_scale=CHARGING_DISTANCE_REWARD_SCALE,
            charging_recharge_bonus=CHARGING_RECHARGE_BONUS,
            charging_depletion_penalty=CHARGING_DEPLETION_PENALTY,
            charging_high_battery_control_penalty=(
                CHARGING_HIGH_BATTERY_CONTROL_PENALTY
            ),
            feeder_low_battery_control_penalty=(
                FEEDER_LOW_BATTERY_CONTROL_PENALTY
            ),
            feeder_yield_aux_coef=FEEDER_YIELD_AUX_COEF,
            feeder_yield_aux_bid_head_only=(
                FEEDER_YIELD_AUX_BID_HEAD_ONLY
            ),
            feeder_yield_activation_margin=(
                FEEDER_YIELD_ACTIVATION_MARGIN
            ),
            charging_low_battery_bid_boost=(
                CHARGING_LOW_BATTERY_BID_BOOST
            ),
            charging_bid_boost_threshold=(
                CHARGING_BID_BOOST_THRESHOLD
            ),
            charging_activation_margin=CHARGING_ACTIVATION_MARGIN,
            charging_release_window_on_recharge=(
                CHARGING_RELEASE_WINDOW_ON_RECHARGE
            ),
            charging_programmatic_navigation=(
                CHARGING_PROGRAMMATIC_NAVIGATION
            ),
            charging_greedy_navigation_eval=(
                CHARGING_GREEDY_NAVIGATION_EVAL
            ),
            charging_separate_direction_actor=(
                CHARGING_SEPARATE_DIRECTION_ACTOR
            ),
            charging_ppo_bid_only=CHARGING_PPO_BID_ONLY,
            charging_reserve_features_enabled=(
                CHARGING_RESERVE_FEATURES_ENABLED
            ),
            charging_nearest_station_features_enabled=(
                CHARGING_NEAREST_STATION_FEATURES_ENABLED
            ),
            charging_learning_rate=CHARGING_LEARNING_RATE,
            feeder_warm_start_checkpoint=FEEDER_WARM_START_CHECKPOINT,
            feeder_freeze_iterations=FEEDER_FREEZE_ITERATIONS,
            feeder_finetune_learning_rate=(
                FEEDER_FINETUNE_LEARNING_RATE
            ),
            charging_bc_updates=CHARGING_BC_UPDATES,
            charging_bc_batch_size=CHARGING_BC_BATCH_SIZE,
            charging_bc_learning_rate=CHARGING_BC_LEARNING_RATE,
            charging_bc_bid_loss_coef=CHARGING_BC_BID_LOSS_COEF,
            charging_bc_bid_value=CHARGING_BC_BID_VALUE,
            charging_bc_emergency_margin=(
                CHARGING_BC_EMERGENCY_MARGIN
            ),
            charging_bc_emergency_bid_value=(
                CHARGING_BC_EMERGENCY_BID_VALUE
            ),
            charging_bc_refresh_updates=CHARGING_BC_REFRESH_UPDATES,
            charging_bc_refresh_learning_rate=(
                CHARGING_BC_REFRESH_LEARNING_RATE
            ),

            # Training config
            num_iterations=NUM_ITERATIONS,
            learning_rate=LEARNING_RATE,
            num_envs=NUM_ENVS,
            num_steps=NUM_STEPS,
            num_minibatches=NUM_MINIBATCHES,
            update_epochs=UPDATE_EPOCHS,
            anneal_lr=ANNEAL_LR,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            norm_adv=NORM_ADV,
            clip_coef=CLIP_COEF,
            clip_vloss=CLIP_VLOSS,
            ent_coef=ENT_COEF,
            vf_coef=VF_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            target_kl=TARGET_KL,
            actor_hidden_sizes=ACTOR_HIDDEN_SIZES,
            critic_hidden_sizes=CRITIC_HIDDEN_SIZES,
            use_target_attention_pooling=USE_TARGET_ATTENTION_POOLING,
            target_embed_dim=TARGET_EMBED_DIM,
            target_encoder_hidden_sizes=TARGET_ENCODER_HIDDEN_SIZES,
        )

    # Create and run experiment
    experiment = PPOMovingTargetsExperiment(
        base_log_dir=BASE_LOG_DIR,
        experiment_name=EXPERIMENT_NAME,
        checkpoint_freq=CHECKPOINT_FREQ,
        eval_freq=EVAL_FREQ,
        video_freq=VIDEO_FREQ,
        num_eval_episodes=NUM_EVAL_EPISODES,
        num_video_episodes=NUM_VIDEO_EPISODES,
        log_videos_to_wandb=LOG_VIDEOS_TO_WANDB,
        single_agent_mode=SINGLE_AGENT_MODE,
        eval_max_steps=EVAL_MAX_STEPS,
        eval_num_agents=EVAL_NUM_AGENTS,
        eval_num_targets=EVAL_NUM_TARGETS,
        deterministic_eval=DETERMINISTIC_EVAL,
        eval_seed=EVAL_SEED,
        policy_sample_seed=POLICY_SAMPLE_SEED,
        final_model_in_log_dir=FINAL_MODEL_IN_LOG_DIR,
    )

    experiment.run(args)


if __name__ == "__main__":
    main()
