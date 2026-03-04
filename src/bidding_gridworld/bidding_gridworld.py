"""Deprecated: use src/bidding_gridworld/bidding_gridworld_torch.py (GPU-native) instead."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, List, Any, Optional
import random
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

import warnings

warnings.warn(
    "src/bidding_gridworld/bidding_gridworld.py is deprecated; "
    "use src/bidding_gridworld/bidding_gridworld_torch.py instead.",
    DeprecationWarning,
    stacklevel=2,
)


class BiddingGridworld(gym.Env):
    """
    A multi-objective gridworld environment where two agents compete through bidding
    to control movement towards their respective target destinations.
    
    The environment consists of:
    - A 10x10 gridworld
    - One shared agent that moves in the environment
    - Two objectives (destinations) that the agent needs to reach
    - Two competing agents that bid to control the shared agent's movement
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        grid_size: int = 10,
        num_agents: int = 2,
        target_positions: Optional[List[Tuple[int, int]]] = None,
        bid_upper_bound: int = 10,
        bid_penalty: float = 0.1,
        target_reward: float = 10.0,
        max_steps: int = 100,
        action_window: int = 1,
        distance_reward_scale: float = 0.0,
        target_expiry_steps: Optional[int] = None,
        target_expiry_penalty: float = 5.0,
        single_agent_mode: bool = False,
        window_bidding: bool = False,
        window_penalty: float = 0.0,
        reward_decay_factor: float = 0.0,
        visible_targets: Optional[int] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the BiddingGridworld environment.

        Args:
            grid_size: Size of the square gridworld (default: 10)
            num_agents: Number of agents/targets (default: 2)
            target_positions: Optional list of (row, col) tuples for target positions.
                            If None, random positions will be assigned. (default: None)
            bid_upper_bound: Maximum bid value (default: 10)
            bid_penalty: Penalty multiplier for bids (default: 0.1)
            target_reward: Reward for reaching target (default: 10.0)
            max_steps: Maximum number of steps per episode (default: 100)
            action_window: Number of steps a winning agent controls the action (default: 1)
                          When window_bidding is False, this is the fixed window length.
                          When window_bidding is True, this is the maximum window agents can bid for.
            distance_reward_scale: Reward scaling for distance improvements (default: 0.0, disabled)
                                  Positive values reward getting closer to target
            target_expiry_steps: Maximum steps allowed before target expiry penalty (default: None, disabled)
                                If set, agents receive penalty if target not reached within this many steps
            target_expiry_penalty: Penalty for not reaching target within expiry_steps (default: 5.0)
            single_agent_mode: If True, use single-agent mode with no bidding (default: False)
                              In this mode, a single agent controls movement and pursues all targets
                              (num_agents still determines the number of targets to reach)
            window_bidding: If True, agents can choose their control window length (1 to action_window) (default: False)
                           When enabled, each agent's action includes a 'window' component specifying
                           how many steps they want to control if they win the bid.
            window_penalty: Penalty multiplier for chosen window length (default: 0.0)
                           When window_bidding is enabled, agents pay window_penalty * chosen_window
                           Applied only on the first step of the window (similar to bid_penalty)
            reward_decay_factor: Reward decay based on relative target count (single-agent mode only) (default: 0.0)
                                When > 0, target rewards are multiplied by exp(-decay_factor * relative_count)
                                where relative_count = (times_target_reached - min_count_across_targets)
                                This incentivizes balanced pursuit of all targets.
            visible_targets: Number of nearest other targets visible to each agent (multi-agent mode only) (default: None)
                            When None, agents see all targets (centralized, current behavior).
                            When set to N, each agent sees their own target plus the N nearest other targets.
                            This creates decentralized observations where agents have limited visibility.
                            Only applies in multi-agent mode; ignored in single-agent mode.
            render_mode: Rendering mode (default: None)
        """
        super().__init__()

        self.single_agent_mode = single_agent_mode

        if not single_agent_mode:
            assert num_agents >= 2, "Must have at least 2 agents in multi-agent mode"
        else:
            assert num_agents >= 1, "Must have at least 1 target in single-agent mode"

        if target_positions is not None:
            assert len(target_positions) == num_agents, \
                f"target_positions must have {num_agents} positions, got {len(target_positions)}"
            for pos in target_positions:
                assert len(pos) == 2, f"Each position must be (row, col), got {pos}"
                assert 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size, \
                    f"Position {pos} out of bounds for grid_size {grid_size}"

        self.grid_size = grid_size
        self.num_agents = num_agents
        self.initial_target_positions = target_positions  # Store for reset
        self.bid_upper_bound = bid_upper_bound
        self.bid_penalty = bid_penalty
        self.target_reward = target_reward
        self.max_steps = max_steps
        self.action_window = action_window
        self.distance_reward_scale = distance_reward_scale
        self.target_expiry_steps = target_expiry_steps
        self.target_expiry_penalty = target_expiry_penalty
        self.window_bidding = window_bidding
        self.window_penalty = window_penalty
        self.reward_decay_factor = reward_decay_factor

        # Set visible_targets: default to seeing all targets
        if visible_targets is None:
            self.visible_targets = num_agents
        else:
            assert visible_targets >= 0, "visible_targets must be non-negative"
            assert visible_targets < num_agents, \
                f"visible_targets must be less than num_agents ({num_agents}), got {visible_targets}"
            self.visible_targets = visible_targets

        # Per-agent observations only used in multi-agent mode with limited visibility
        self.use_per_agent_obs = (not single_agent_mode) and (self.visible_targets < num_agents)

        self.render_mode = render_mode

        # Actions: 0=Left, 1=Right, 2=Up, 3=Down
        self.action_meanings = {0: "Left", 1: "Right", 2: "Up", 3: "Down"}

        # Action space
        if single_agent_mode:
            # Single agent: just choose a direction (no bidding)
            self.action_space = spaces.Discrete(4)
        else:
            # Multi-agent: each agent submits (direction, bid) or (direction, bid, window)
            # Each agent submits an action that consists of:
            # - direction: discrete action (0-3 for L,R,U,D)
            # - bid: integer value (0 to bid_upper_bound)
            # - window: (optional, if window_bidding=True) discrete value (0 to action_window-1, representing 1 to action_window)
            if window_bidding:
                self.action_space = spaces.Dict({
                    f"agent_{i}": spaces.Dict({
                        "direction": spaces.Discrete(4),
                        "bid": spaces.Discrete(bid_upper_bound + 1),
                        "window": spaces.Discrete(action_window)  # 0 to action_window-1 (add 1 to get actual window)
                    })
                    for i in range(num_agents)
                })
            else:
                self.action_space = spaces.Dict({
                    f"agent_{i}": spaces.Dict({
                        "direction": spaces.Discrete(4),
                        "bid": spaces.Discrete(bid_upper_bound + 1)
                    })
                    for i in range(num_agents)
                })
        
        # Observation space: Box vector for NN policies (normalized to [0, 1])
        # In multi-agent mode with limited visibility (use_per_agent_obs=True):
        #   Each agent has their own observation containing:
        #   [agent_pos, own_target_pos, nearest_visible_targets_pos, own_target_reached,
        #    visible_targets_reached, own_target_counter, window_remaining]
        # In centralized mode (use_per_agent_obs=False):
        #   Single shared observation containing all targets
        # Single-agent mode adds: [target0_count_relative, target1_count_relative, ...]
        #   where target_i_count_relative = (count[i] - min_count) normalized

        if self.use_per_agent_obs:
            # Per-agent observations: each agent sees own target + visible_targets nearest others
            # obs_dim = 2 (agent_pos) + 2 (own_target) + 2*visible_targets (other targets) +
            #          1 (own_reached) + visible_targets (others_reached) +
            #          1 (own_counter) + 1 (window_remaining)
            obs_dim_per_agent = 7 + 3 * self.visible_targets
            self.observation_space = spaces.Dict({
                f"agent_{i}": spaces.Box(low=0.0, high=1.0, shape=(obs_dim_per_agent,), dtype=np.float32)
                for i in range(num_agents)
            })
        else:
            # Centralized observation: all agents see all targets (current behavior)
            # Base: [agent_row_norm, agent_col_norm, target0_row_norm, target0_col_norm, ...,
            #        target0_reached, target1_reached, ..., target0_step_counter_norm, target1_step_counter_norm, ...,
            #        window_steps_remaining_norm]
            # Shape: 2 (agent position) + 2 * num_agents (target positions) + num_agents (target reached flags) +
            #        num_agents (step counters) + 1 (window steps remaining) + [num_agents (relative counts) if single_agent_mode else 0]
            obs_dim = 2 + 2 * num_agents + num_agents + num_agents + 1
            if single_agent_mode:
                obs_dim += num_agents  # Add relative target counts
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        
        # Initialize environment state
        self.reset()
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize positions
        self.agent_position = np.array([0, 0], dtype=np.int32)

        # Place targets - use specified positions or random
        self.target_positions = []
        if self.initial_target_positions is not None:
            # Use specified positions
            for pos in self.initial_target_positions:
                self.target_positions.append(np.array(pos, dtype=np.int32))
        else:
            # Generate random unique positions for targets
            available_positions = [(r, c) for r in range(self.grid_size)
                                 for c in range(self.grid_size)
                                 if (r, c) != (0, 0)]  # Exclude agent start position
            selected_positions = random.sample(available_positions, self.num_agents)
            for pos in selected_positions:
                self.target_positions.append(np.array(pos, dtype=np.int32))

        # Track which targets have been reached
        self.targets_reached = np.zeros(self.num_agents, dtype=np.int32)

        # Track how many times each target has been reached (for single-agent mode)
        self.targets_reached_count = np.zeros(self.num_agents, dtype=np.int32)

        # Track per-target step counters for expiry mechanism
        self.target_step_counters = np.zeros(self.num_agents, dtype=np.int32)
        self.targets_expired_this_step = {i: False for i in range(self.num_agents)}

        # Track previous distances for distance-based rewards
        self.previous_distances = np.zeros(self.num_agents, dtype=np.float32)
        for i in range(self.num_agents):
            self.previous_distances[i] = abs(self.agent_position[0] - self.target_positions[i][0]) + \
                                         abs(self.agent_position[1] - self.target_positions[i][1])

        # Step counter
        self.step_count = 0

        # Action window tracking
        self.window_agent = None  # Which agent currently has control
        self.window_steps_remaining = 0  # How many steps remain in window
        self.current_window_length = 0  # Length of current window (for penalty calculation)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: In single-agent mode: integer direction (0-3)
                   In multi-agent mode: dictionary containing actions for all agents,
                   each with 'direction' and 'bid' components

        Returns:
            observation: Current state observation
            reward: In single-agent mode: scalar reward
                   In multi-agent mode: dictionary of rewards for all agents
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        self.step_count += 1

        if self.single_agent_mode:
            # Single agent mode: action is just a direction
            winning_direction = action
            winning_agent = 0  # For tracking purposes
            apply_bid_penalty = False

            # Execute the action
            new_position = self._move_agent(self.agent_position, winning_direction)
            self.agent_position = new_position
        else:
            # Multi-agent mode: handle bidding
            # Extract actions and bids for all agents
            agent_actions = {}
            agent_bids = {}
            for i in range(self.num_agents):
                agent_key = f"agent_{i}"
                agent_actions[i] = action[agent_key]
                agent_bids[i] = action[agent_key]["bid"]

            # Check if we're in an action window
            apply_bid_penalty = False
            if self.window_steps_remaining > 0:
                # Use the locked-in agent's action
                winning_agent = self.window_agent
                winning_direction = agent_actions[winning_agent]["direction"]

                # Execute the action
                new_position = self._move_agent(self.agent_position, winning_direction)
                self.agent_position = new_position

                # Decrement window
                self.window_steps_remaining -= 1
            else:
                # Normal bidding
                max_bid = max(agent_bids.values())

                # Only move if at least one agent bid > 0
                if max_bid > 0:
                    winners = [agent_id for agent_id, bid in agent_bids.items() if bid == max_bid]

                    # If tie, randomly choose among winners
                    winning_agent = random.choice(winners)
                    winning_direction = agent_actions[winning_agent]["direction"]

                    # Execute the winning action
                    new_position = self._move_agent(self.agent_position, winning_direction)
                    self.agent_position = new_position

                    # Start new action window
                    self.window_agent = winning_agent

                    # Determine window length
                    if self.window_bidding:
                        # Use the winning agent's chosen window (add 1 since action space is 0 to action_window-1)
                        chosen_window = agent_actions[winning_agent]["window"] + 1
                        self.window_steps_remaining = chosen_window - 1  # -1 because current step counts
                        self.current_window_length = chosen_window  # Store for penalty calculation
                    else:
                        # Use the fixed action_window
                        self.window_steps_remaining = self.action_window - 1  # -1 because current step counts
                        self.current_window_length = self.action_window

                    # Apply bid penalty on first step of window
                    apply_bid_penalty = True
                else:
                    # All agents bid 0, no movement occurs
                    winning_agent = None
        
        # Check if any targets are reached for the first time this step
        targets_just_reached = {}
        for i in range(self.num_agents):
            if np.array_equal(self.agent_position, self.target_positions[i]) and self.targets_reached[i] == 0:
                self.targets_reached[i] = 1
                self.targets_reached_count[i] += 1  # Increment the count of times this target has been reached
                targets_just_reached[i] = True
                # Reset counter when target is reached
                self.target_step_counters[i] = 0
            else:
                targets_just_reached[i] = False

        # Update target step counters and check for expiries
        targets_expired = {}
        if self.target_expiry_steps is not None:
            for i in range(self.num_agents):
                # Only increment counter if target hasn't been reached yet
                if self.targets_reached[i] == 0:
                    self.target_step_counters[i] += 1
                    # Check if target has expired
                    if self.target_step_counters[i] >= self.target_expiry_steps:
                        targets_expired[i] = True
                        # Reset counter after expiry to allow continuous penalties
                        self.target_step_counters[i] = 0
                    else:
                        targets_expired[i] = False
                else:
                    targets_expired[i] = False
        else:
            # Expiry mechanism disabled
            for i in range(self.num_agents):
                targets_expired[i] = False

        # Store as instance variable for subclasses to access
        self.targets_expired_this_step = targets_expired

        # Calculate current distances for distance-based rewards
        current_distances = np.zeros(self.num_agents, dtype=np.float32)
        for i in range(self.num_agents):
            current_distances[i] = abs(self.agent_position[0] - self.target_positions[i][0]) + \
                                  abs(self.agent_position[1] - self.target_positions[i][1])

        # Calculate rewards
        if self.single_agent_mode:
            # Single agent mode: calculate scalar reward
            reward = self._calculate_single_agent_reward(
                targets_just_reached,
                targets_expired,
                self.previous_distances,
                current_distances
            )
        else:
            # Multi-agent mode: calculate rewards for all agents
            reward = self._calculate_rewards(
                agent_bids,
                winning_agent,
                targets_just_reached,
                targets_expired,
                apply_bid_penalty,
                self.previous_distances,
                current_distances,
                self.current_window_length
            )

        # Update previous distances for next step
        self.previous_distances = current_distances.copy()

        # Check termination conditions
        all_targets_reached = bool(np.all(self.targets_reached == 1))
        max_steps_reached = self.step_count >= self.max_steps

        terminated = all_targets_reached
        truncated = max_steps_reached and not all_targets_reached

        observation = self._get_observation()
        info = self._get_info()
        info["winning_agent"] = winning_agent
        info["targets_just_reached"] = targets_just_reached  # Add which targets were just reached
        if not self.single_agent_mode:
            info["bids"] = {f"agent_{i}": bid for i, bid in agent_bids.items()}
            info["window_agent"] = self.window_agent
            info["window_steps_remaining"] = self.window_steps_remaining
            info["bid_penalty_applied"] = apply_bid_penalty

        return observation, reward, terminated, truncated, info
    
    def _move_agent(self, position: np.ndarray, direction: int) -> np.ndarray:
        """Move agent in the specified direction, respecting grid boundaries."""
        new_position = position.copy()
        
        if direction == 0:  # Left
            new_position[1] = max(0, position[1] - 1)
        elif direction == 1:  # Right
            new_position[1] = min(self.grid_size - 1, position[1] + 1)
        elif direction == 2:  # Up
            new_position[0] = max(0, position[0] - 1)
        elif direction == 3:  # Down
            new_position[0] = min(self.grid_size - 1, position[0] + 1)
        
        return new_position
    
    def _calculate_rewards(
        self,
        agent_bids: Dict[int, int],
        winning_agent: int,
        targets_just_reached: Dict[int, bool],
        targets_expired: Dict[int, bool],
        apply_bid_penalty: bool,
        previous_distances: np.ndarray,
        current_distances: np.ndarray,
        current_window_length: int
    ) -> Dict[str, float]:
        """Calculate rewards for all agents."""
        rewards = {f"agent_{i}": 0.0 for i in range(self.num_agents)}

        # Bid penalties and window penalties (only apply on the first step of a window)
        if apply_bid_penalty:
            # Each agent pays penalty for their own bid
            for i in range(self.num_agents):
                # Pay for own bid
                rewards[f"agent_{i}"] -= self.bid_penalty * agent_bids[i]

            # Winning agent also pays penalty for chosen window length (if window_bidding enabled)
            if self.window_bidding and winning_agent is not None and winning_agent >= 0:
                rewards[f"agent_{winning_agent}"] -= self.window_penalty * current_window_length

        # Distance-based rewards (reward for getting closer to target)
        if self.distance_reward_scale > 0:
            for i in range(self.num_agents):
                # Skip if target already reached
                if self.targets_reached[i] == 0:
                    # Positive reward for decreasing distance (getting closer)
                    # Negative reward for increasing distance (getting farther)
                    distance_improvement = previous_distances[i] - current_distances[i]
                    rewards[f"agent_{i}"] += self.distance_reward_scale * distance_improvement

        # Target rewards (always apply when reached)
        for i in range(self.num_agents):
            if targets_just_reached[i]:
                rewards[f"agent_{i}"] += self.target_reward

        # Target expiry penalties (apply when target expires)
        for i in range(self.num_agents):
            if targets_expired[i]:
                rewards[f"agent_{i}"] -= self.target_expiry_penalty

        return rewards

    def _calculate_single_agent_reward(
        self,
        targets_just_reached: Dict[int, bool],
        targets_expired: Dict[int, bool],
        previous_distances: np.ndarray,
        current_distances: np.ndarray
    ) -> float:
        """Calculate scalar reward for single agent mode."""
        reward = 0.0

        # Distance-based rewards (sum across all targets)
        if self.distance_reward_scale > 0:
            for i in range(self.num_agents):
                # Skip if target already reached
                if self.targets_reached[i] == 0:
                    # Positive reward for decreasing distance (getting closer)
                    distance_improvement = previous_distances[i] - current_distances[i]
                    reward += self.distance_reward_scale * distance_improvement

        # Target rewards (sum for all targets reached this step)
        for i in range(self.num_agents):
            if targets_just_reached[i]:
                target_reward = self.target_reward

                # Apply reward decay based on relative count (only in single-agent mode)
                if self.single_agent_mode and self.reward_decay_factor > 0:
                    # Calculate relative count (how many more times this target was reached vs least-reached)
                    min_count = int(np.min(self.targets_reached_count))
                    relative_count = self.targets_reached_count[i] - min_count

                    # Apply exponential decay: reward = base_reward * exp(-decay_factor * relative_count)
                    decay_multiplier = np.exp(-self.reward_decay_factor * relative_count)
                    target_reward *= decay_multiplier

                reward += target_reward

        # Target expiry penalties (sum for all targets that expired)
        for i in range(self.num_agents):
            if targets_expired[i]:
                reward -= self.target_expiry_penalty

        return reward

    def _get_observation(self):
        """Get current observation as a normalized vector or dict of vectors in [0, 1]."""
        # Normalize positions to [0,1]; guard division for grid_size==1
        denom = float(self.grid_size - 1) if self.grid_size > 1 else 1.0
        agent_row_norm = float(self.agent_position[0]) / denom
        agent_col_norm = float(self.agent_position[1]) / denom

        # Normalization denominators
        counter_denom = float(self.target_expiry_steps) if self.target_expiry_steps is not None else float(self.max_steps)
        counter_denom = max(counter_denom, 1.0)
        window_denom = max(float(self.action_window), 1.0)
        window_steps_norm = float(self.window_steps_remaining) / window_denom

        if self.use_per_agent_obs:
            # Per-agent observations: each agent sees own target + visible_targets nearest others
            observations = {}

            for agent_id in range(self.num_agents):
                obs_list = [agent_row_norm, agent_col_norm]

                # Add own target position
                own_target_pos = self.target_positions[agent_id]
                own_t_row_norm = float(own_target_pos[0]) / denom
                own_t_col_norm = float(own_target_pos[1]) / denom
                obs_list.extend([own_t_row_norm, own_t_col_norm])

                # Find nearest visible_targets other targets (not including own target)
                other_targets = [(i, self.target_positions[i]) for i in range(self.num_agents) if i != agent_id]

                # Calculate distances to other targets (Manhattan distance)
                distances = []
                for other_id, other_pos in other_targets:
                    dist = abs(self.agent_position[0] - other_pos[0]) + abs(self.agent_position[1] - other_pos[1])
                    distances.append((dist, other_id, other_pos))

                # Sort by distance and take the nearest visible_targets targets
                distances.sort(key=lambda x: x[0])
                visible_other_targets = distances[:self.visible_targets]

                # Add visible target positions
                for _, other_id, other_pos in visible_other_targets:
                    other_t_row_norm = float(other_pos[0]) / denom
                    other_t_col_norm = float(other_pos[1]) / denom
                    obs_list.extend([other_t_row_norm, other_t_col_norm])

                # Add own target reached flag
                obs_list.append(float(self.targets_reached[agent_id]))

                # Add visible targets reached flags
                for _, other_id, _ in visible_other_targets:
                    obs_list.append(float(self.targets_reached[other_id]))

                # Add own target counter
                own_counter_norm = min(float(self.target_step_counters[agent_id]) / counter_denom, 1.0)
                obs_list.append(own_counter_norm)

                # Add window steps remaining
                obs_list.append(window_steps_norm)

                observations[f"agent_{agent_id}"] = np.array(obs_list, dtype=np.float32)

            return observations
        else:
            # Centralized observation: all agents see all targets (current behavior)
            # Build observation: [agent_pos, target_0_pos, ..., target_n_pos, target_0_reached, ..., target_n_reached, target_0_counter, ..., target_n_counter, window_steps_remaining]
            obs_list = [agent_row_norm, agent_col_norm]

            # Add all target positions
            for target_pos in self.target_positions:
                t_row_norm = float(target_pos[0]) / denom
                t_col_norm = float(target_pos[1]) / denom
                obs_list.extend([t_row_norm, t_col_norm])

            # Add all target reached flags
            for i in range(self.num_agents):
                obs_list.append(float(self.targets_reached[i]))

            # Add all target step counters (normalized)
            for i in range(self.num_agents):
                counter_norm = min(float(self.target_step_counters[i]) / counter_denom, 1.0)  # Clamp to [0, 1]
                obs_list.append(counter_norm)

            # Add window steps remaining (normalized by action_window)
            obs_list.append(window_steps_norm)

            # In single-agent mode, add relative target counts (count - min_count)
            if self.single_agent_mode:
                min_count = int(np.min(self.targets_reached_count))
                # Normalize by num_agents (reasonable upper bound for differences)
                count_denom = max(float(self.num_agents), 1.0)
                for i in range(self.num_agents):
                    relative_count = float(self.targets_reached_count[i] - min_count)
                    relative_count_norm = min(relative_count / count_denom, 1.0)  # Clamp to [0, 1]
                    obs_list.append(relative_count_norm)

            obs = np.array(obs_list, dtype=np.float32)
            return obs

    def _get_centralized_observation(self):
        """Get centralized observation regardless of visible_targets settings."""
        # Normalize positions to [0,1]; guard division for grid_size==1
        denom = float(self.grid_size - 1) if self.grid_size > 1 else 1.0
        agent_row_norm = float(self.agent_position[0]) / denom
        agent_col_norm = float(self.agent_position[1]) / denom

        # Normalization denominators
        counter_denom = float(self.target_expiry_steps) if self.target_expiry_steps is not None else float(self.max_steps)
        counter_denom = max(counter_denom, 1.0)
        window_denom = max(float(self.action_window), 1.0)
        window_steps_norm = float(self.window_steps_remaining) / window_denom

        # Centralized observation: all agents see all targets
        obs_list = [agent_row_norm, agent_col_norm]

        # Add all target positions
        for target_pos in self.target_positions:
            t_row_norm = float(target_pos[0]) / denom
            t_col_norm = float(target_pos[1]) / denom
            obs_list.extend([t_row_norm, t_col_norm])

        # Add all target reached flags
        for i in range(self.num_agents):
            obs_list.append(float(self.targets_reached[i]))

        # Add all target step counters (normalized)
        for i in range(self.num_agents):
            counter_norm = min(float(self.target_step_counters[i]) / counter_denom, 1.0)
            obs_list.append(counter_norm)

        # Add window steps remaining (normalized by action_window)
        obs_list.append(window_steps_norm)

        # In single-agent mode, add relative target counts (count - min_count)
        if self.single_agent_mode:
            min_count = int(np.min(self.targets_reached_count))
            count_denom = max(float(self.num_agents), 1.0)
            for i in range(self.num_agents):
                relative_count = float(self.targets_reached_count[i] - min_count)
                relative_count_norm = min(relative_count / count_denom, 1.0)
                obs_list.append(relative_count_norm)

        return np.array(obs_list, dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional information."""
        info = {
            "step_count": self.step_count,
            "max_steps": self.max_steps,
        }

        # Add manhattan distance to each target
        for i in range(self.num_agents):
            distance = abs(self.agent_position[0] - self.target_positions[i][0]) + \
                      abs(self.agent_position[1] - self.target_positions[i][1])
            info[f"manhattan_distance_to_target_{i}"] = distance

        # In single-agent mode, add target reach counts and min count
        if self.single_agent_mode:
            info["targets_reached_count"] = self.targets_reached_count.tolist()
            info["min_targets_reached"] = int(np.min(self.targets_reached_count))

        return info
    
    def render(self, mode: str = "human"):
        """Render the environment."""
        if mode == "human":
            self._render_text()
        elif mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_text(self):
        """Render the environment as text."""
        print(f"\nStep: {self.step_count}")
        print(f"Agent position: {self.agent_position}")

        # Print all target positions
        for i in range(self.num_agents):
            print(f"Target {i} position: {self.target_positions[i]} (reached: {bool(self.targets_reached[i])})")

        # Create grid visualization
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)

        # Place targets (use number or checkmark)
        for i in range(self.num_agents):
            target_pos = self.target_positions[i]
            if not self.targets_reached[i]:
                grid[target_pos[0], target_pos[1]] = str(i) if i < 10 else '*'
            else:
                grid[target_pos[0], target_pos[1]] = '✓'

        # Place agent (overwrites target if on same position)
        grid[self.agent_position[0], self.agent_position[1]] = 'A'

        print("\nGrid:")
        for row in grid:
            print(' '.join(row))
        print()
    
    def _render_rgb_array(self):
        """Render the environment as RGB array (for visualization)."""
        # This would create an actual image representation
        # For now, return a placeholder
        rgb_array = np.zeros((self.grid_size * 50, self.grid_size * 50, 3), dtype=np.uint8)
        return rgb_array

    @staticmethod
    def _direction_to_string(direction: int) -> str:
        """Convert direction number to readable string."""
        directions = ["Left", "Right", "Up", "Down"]
        return directions[direction] if 0 <= direction <= 3 else "Unknown"

    def create_episode_gif(
        self,
        episode_data: Dict[str, Any],
        output_path: Path,
        target_agent_id: int = 0,
        fps: int = 2
    ):
        """
        Create an animated GIF of an episode for a single agent's perspective.

        Args:
            episode_data: Dictionary containing episode information
            output_path: Path where to save the GIF
            target_agent_id: ID of the target being pursued
            fps: Frames per second for the animation
        """
        # Scale figure size based on grid size
        grid_size_inches = min(10, max(6, self.grid_size * 0.15))
        info_width = 4
        fig = plt.figure(figsize=(grid_size_inches + info_width, grid_size_inches))

        # Create grid for layout: [grid_ax, info_ax]
        gs = fig.add_gridspec(1, 2, width_ratios=[grid_size_inches, info_width], wspace=0.15)
        grid_ax = fig.add_subplot(gs[0])
        info_ax = fig.add_subplot(gs[1])

        def animate(frame):
            grid_ax.clear()
            info_ax.clear()

            if frame >= len(episode_data["states"]):
                return

            state = episode_data["states"][frame]
            denom = float(self.grid_size - 1) if self.grid_size > 1 else 1.0
            agent_row = int(state[0] * denom)
            agent_col = int(state[1] * denom)

            target_idx = 2 + target_agent_id * 2
            target_row = int(state[target_idx] * denom)
            target_col = int(state[target_idx + 1] * denom)
            target_reached_idx = 2 + 2 * self.num_agents + target_agent_id
            target_reached = int(state[target_reached_idx])

            # === Draw Grid ===
            grid_ax.set_xlim(-0.5, self.grid_size - 0.5)
            grid_ax.set_ylim(-0.5, self.grid_size - 0.5)
            grid_ax.set_aspect('equal')

            for i in range(self.grid_size + 1):
                grid_ax.axhline(i - 0.5, color='lightgray', linewidth=0.5)
                grid_ax.axvline(i - 0.5, color='lightgray', linewidth=0.5)

            stick_colors = ['royalblue', 'crimson', 'darkorange', 'forestgreen', 'purple']
            edge_colors = ['blue', 'red', 'orange', 'green', 'purple']

            def draw_stick_figure(ax, cx, cy, color, lw=1.5):
                ax.add_patch(plt.Circle((cx, cy - 0.22), 0.10, facecolor=color, edgecolor=color, linewidth=1))
                ax.plot([cx, cx], [cy - 0.12, cy + 0.08], color=color, linewidth=lw)
                ax.plot([cx - 0.18, cx + 0.18], [cy - 0.02, cy - 0.02], color=color, linewidth=lw)
                ax.plot([cx, cx - 0.16], [cy + 0.08, cy + 0.28], color=color, linewidth=lw)
                ax.plot([cx, cx + 0.16], [cy + 0.08, cy + 0.28], color=color, linewidth=lw)

            winning_agent = None
            if frame < len(episode_data.get("step_details", [])):
                step_detail = episode_data["step_details"][frame]
                winning_agent = step_detail.get("winning_agent", -1)

            is_controlling = (winning_agent == target_agent_id)

            if target_reached == 0:
                color = stick_colors[target_agent_id % len(stick_colors)]
                draw_stick_figure(grid_ax, target_col, target_row, color)
                if is_controlling:
                    grid_ax.text(target_col, target_row - 0.6, '⚡',
                           ha='center', va='center', fontsize=8, color='gold')
            else:
                draw_stick_figure(grid_ax, target_col, target_row, 'darkgreen')
                grid_ax.text(target_col, target_row - 0.5, '✓',
                       ha='center', va='center', fontsize=8, fontweight='bold', color='darkgreen')

            if winning_agent is not None and winning_agent >= 0:
                ring_color = edge_colors[winning_agent % len(edge_colors)]
                grid_ax.add_patch(plt.Circle((agent_col, agent_row), 0.35,
                                       facecolor='none', edgecolor=ring_color, linewidth=3))

            grid_ax.text(agent_col, agent_row, '☕',
                   ha='center', va='center', fontsize=16)

            # Set grid title and ticks
            grid_ax.set_title(f'Agent {target_agent_id} - Step {frame}', fontsize=11, fontweight='bold')

            # Optimize tick marks for large grids
            if self.grid_size <= 15:
                tick_step = 1
            elif self.grid_size <= 30:
                tick_step = 2
            else:
                tick_step = 5

            tick_positions = list(range(0, self.grid_size, tick_step))
            grid_ax.set_xticks(tick_positions)
            grid_ax.set_yticks(tick_positions)
            grid_ax.invert_yaxis()

            # === Draw Info Panel ===
            info_ax.axis('off')

            # Extract reward for this agent
            reward = 0
            if frame < len(episode_data["rewards"]):
                rewards_at_frame = episode_data["rewards"][frame]
                if isinstance(rewards_at_frame, dict):
                    reward = rewards_at_frame.get(f"agent_{target_agent_id}", 0)
                else:
                    reward = rewards_at_frame

            # Calculate cumulative reward
            total_reward = 0
            for f in range(frame + 1):
                if f < len(episode_data["rewards"]):
                    r = episode_data["rewards"][f]
                    if isinstance(r, dict):
                        total_reward += r.get(f"agent_{target_agent_id}", 0)
                    else:
                        total_reward += r

            # Build info text
            info_lines = []
            info_lines.append(f'AGENT ROLLOUT\n')
            info_lines.append(f'Grid: {self.grid_size}x{self.grid_size}')
            info_lines.append(f'Target: {target_agent_id}')
            info_lines.append(f'')
            info_lines.append(f'STATUS:')
            info_lines.append(f'  Target: {"✓ Reached" if target_reached else "✗ Not Reached"}')
            if is_controlling:
                info_lines.append(f'  Control: ⚡ Active')
            else:
                info_lines.append(f'  Control: Waiting')
            info_lines.append(f'')
            info_lines.append(f'REWARDS:')
            info_lines.append(f'  Step:  {reward:7.2f}')
            info_lines.append(f'  Total: {total_reward:7.2f}')

            # Add bid and action information if available
            if frame < len(episode_data.get("actions", [])):
                action = episode_data["actions"][frame]
                if isinstance(action, dict) and f"agent_{target_agent_id}" in action:
                    direction_names = {0: "Left ←", 1: "Right →", 2: "Up ↑", 3: "Down ↓"}
                    agent_action = action[f"agent_{target_agent_id}"]
                    direction = direction_names.get(agent_action.get("direction"), "?")
                    bid = agent_action.get("bid", 0)
                    desired_window = None
                    if self.window_bidding and "window" in agent_action:
                        desired_window = int(agent_action.get("window", 0)) + 1
                    info_lines.append(f'')
                    info_lines.append(f'ACTIONS:')
                    info_lines.append(f'  Bid:       {bid}')
                    if desired_window is not None:
                        info_lines.append(f'  Window:    {desired_window}')
                    info_lines.append(f'  Direction: {direction}')

            # Render text
            info_text = '\n'.join(info_lines)
            info_ax.text(0.05, 0.95, info_text,
                        transform=info_ax.transAxes,
                        fontfamily='monospace',
                        fontsize=10,
                        verticalalignment='top',
                        horizontalalignment='left')

        # Render frames to numpy arrays
        frames = []
        num_frames = len(episode_data["states"]) + 5
        for frame_idx in range(num_frames):
            animate(frame_idx)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            frames.append(frame)

        plt.close(fig)

        # Write video using OpenCV
        if len(frames) > 0:
            h, w = frames[0].shape[:2]
            output_path_mp4 = str(output_path).replace('.gif', '.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path_mp4, fourcc, fps, (w, h))

            try:
                for frame in frames:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                out.release()
                print(f"✅ Episode video saved: {output_path_mp4}")
            except Exception as e:
                print(f"⚠️  Could not save video {output_path_mp4}: {e}")
                out.release()

    def create_single_agent_gif(
        self,
        episode_data: Dict[str, Any],
        output_path: Path,
        fps: int = 2
    ):
        """
        Create an animated GIF for single-agent mode showing all targets.

        Args:
            episode_data: Dictionary containing episode information
            output_path: Path where to save the GIF
            fps: Frames per second for the animation
        """
        # Scale figure size based on grid size
        grid_size_inches = min(10, max(6, self.grid_size * 0.15))
        info_width = 4
        fig = plt.figure(figsize=(grid_size_inches + info_width, grid_size_inches))

        # Create grid for layout: [grid_ax, info_ax]
        gs = fig.add_gridspec(1, 2, width_ratios=[grid_size_inches, info_width], wspace=0.15)
        grid_ax = fig.add_subplot(gs[0])
        info_ax = fig.add_subplot(gs[1])

        def animate(frame):
            grid_ax.clear()
            info_ax.clear()

            if frame >= len(episode_data["states"]):
                return

            state = episode_data["states"][frame]
            denom = float(self.grid_size - 1) if self.grid_size > 1 else 1.0

            # Agent position
            agent_row = int(state[0] * denom)
            agent_col = int(state[1] * denom)

            # === Draw Grid ===
            grid_ax.set_xlim(-0.5, self.grid_size - 0.5)
            grid_ax.set_ylim(-0.5, self.grid_size - 0.5)
            grid_ax.set_aspect('equal')

            # Draw grid
            for i in range(self.grid_size + 1):
                grid_ax.axhline(i - 0.5, color='lightgray', linewidth=0.5)
                grid_ax.axvline(i - 0.5, color='lightgray', linewidth=0.5)

            stick_colors = ['royalblue', 'crimson', 'darkorange', 'forestgreen', 'purple']

            def draw_stick_figure(ax, cx, cy, color, lw=1.5):
                ax.add_patch(plt.Circle((cx, cy - 0.22), 0.10, facecolor=color, edgecolor=color, linewidth=1))
                ax.plot([cx, cx], [cy - 0.12, cy + 0.08], color=color, linewidth=lw)
                ax.plot([cx - 0.18, cx + 0.18], [cy - 0.02, cy - 0.02], color=color, linewidth=lw)
                ax.plot([cx, cx - 0.16], [cy + 0.08, cy + 0.28], color=color, linewidth=lw)
                ax.plot([cx, cx + 0.16], [cy + 0.08, cy + 0.28], color=color, linewidth=lw)

            # Draw all targets
            for target_id in range(self.num_agents):
                target_idx = 2 + target_id * 2
                target_row = int(state[target_idx] * denom)
                target_col = int(state[target_idx + 1] * denom)
                target_reached_idx = 2 + 2 * self.num_agents + target_id
                target_reached = int(state[target_reached_idx])

                if target_reached == 0:
                    draw_stick_figure(grid_ax, target_col, target_row, stick_colors[target_id % len(stick_colors)])
                else:
                    draw_stick_figure(grid_ax, target_col, target_row, 'darkgreen')
                    grid_ax.text(target_col, target_row - 0.5, '✓',
                           ha='center', va='center', fontsize=8, fontweight='bold', color='darkgreen')

            # Draw agent
            grid_ax.text(agent_col, agent_row, '☕',
                   ha='center', va='center', fontsize=16)

            # Set grid title and ticks
            grid_ax.set_title(f'Step {frame}', fontsize=11, fontweight='bold')

            # Optimize tick marks for large grids
            if self.grid_size <= 15:
                tick_step = 1
            elif self.grid_size <= 30:
                tick_step = 2
            else:
                tick_step = 5

            tick_positions = list(range(0, self.grid_size, tick_step))
            grid_ax.set_xticks(tick_positions)
            grid_ax.set_yticks(tick_positions)
            grid_ax.invert_yaxis()

            # === Draw Info Panel ===
            info_ax.axis('off')

            # Get reward for this frame
            reward = 0
            if frame < len(episode_data["rewards"]):
                reward = episode_data["rewards"][frame]

            # Calculate cumulative reward
            total_reward = sum(episode_data["rewards"][:frame + 1])

            # Count targets reached
            targets_reached = sum(1 for i in range(self.num_agents)
                                 if state[2 + 2 * self.num_agents + i] == 1)

            # Build info text
            info_lines = []
            info_lines.append(f'SINGLE AGENT MODE\n')
            info_lines.append(f'Grid: {self.grid_size}x{self.grid_size}')
            info_lines.append(f'Targets: {targets_reached}/{self.num_agents}')
            info_lines.append(f'')
            info_lines.append(f'REWARDS:')
            info_lines.append(f'  Step:  {reward:7.2f}')
            info_lines.append(f'  Total: {total_reward:7.2f}')

            # Add action information if available
            if frame < len(episode_data.get("actions", [])):
                action = episode_data["actions"][frame]
                direction_names = {0: "Left ←", 1: "Right →", 2: "Up ↑", 3: "Down ↓"}
                direction = direction_names.get(action, "?")
                info_lines.append(f'')
                info_lines.append(f'ACTION:')
                info_lines.append(f'  Direction: {direction}')

            # List target statuses
            info_lines.append(f'')
            info_lines.append(f'TARGET STATUS:')
            for target_id in range(self.num_agents):
                target_reached_idx = 2 + 2 * self.num_agents + target_id
                target_reached = int(state[target_reached_idx])
                status = '✓' if target_reached else '✗'
                info_lines.append(f'  {target_id}: {status}')

            # Render text
            info_text = '\n'.join(info_lines)
            info_ax.text(0.05, 0.95, info_text,
                        transform=info_ax.transAxes,
                        fontfamily='monospace',
                        fontsize=10,
                        verticalalignment='top',
                        horizontalalignment='left')

        # Render frames to numpy arrays
        frames = []
        num_frames = len(episode_data["states"]) + 5
        for frame_idx in range(num_frames):
            animate(frame_idx)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            frames.append(frame)

        plt.close(fig)

        # Write video using OpenCV
        if len(frames) > 0:
            h, w = frames[0].shape[:2]
            output_path_mp4 = str(output_path).replace('.gif', '.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path_mp4, fourcc, fps, (w, h))

            try:
                for frame in frames:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                out.release()
                print(f"✅ Single-agent video saved: {output_path_mp4}")
            except Exception as e:
                print(f"⚠️  Could not save video {output_path_mp4}: {e}")
                out.release()

    def create_competition_gif(
        self,
        episode_data: Dict[str, Any],
        output_path: Path,
        fps: int = 1
    ):
        """
        Create an animated GIF of a multi-agent competition episode.

        Args:
            episode_data: Dictionary containing episode information
            output_path: Path where to save the GIF
            fps: Frames per second for the animation
        """
        # Scale figure size based on grid size
        # Grid panel is square, info panel is fixed width
        grid_size_inches = min(10, max(6, self.grid_size * 0.15))
        info_width = 5
        fig = plt.figure(figsize=(grid_size_inches + info_width, grid_size_inches))

        # Create grid for layout: [grid_ax, info_ax]
        gs = fig.add_gridspec(1, 2, width_ratios=[grid_size_inches, info_width], wspace=0.15)
        grid_ax = fig.add_subplot(gs[0])
        info_ax = fig.add_subplot(gs[1])

        def animate(frame):
            grid_ax.clear()
            info_ax.clear()

            if frame >= len(episode_data["states"]):
                return

            state = episode_data["states"][frame]
            denom = float(self.grid_size - 1) if self.grid_size > 1 else 1.0
            agent_row = int(state[0] * denom)
            agent_col = int(state[1] * denom)

            target_positions = []
            targets_reached = []
            for i in range(self.num_agents):
                target_idx = 2 + i * 2
                target_positions.append((int(state[target_idx] * denom),
                                       int(state[target_idx + 1] * denom)))
                target_reached_idx = 2 + 2 * self.num_agents + i
                targets_reached.append(int(state[target_reached_idx]))

            step_detail = episode_data["step_details"][frame] if frame < len(episode_data["step_details"]) else None
            actions = episode_data["actions"][frame] if frame < len(episode_data["actions"]) else None
            rewards = episode_data["rewards"][frame] if frame < len(episode_data["rewards"]) else None

            # === Draw Grid ===
            grid_ax.set_xlim(-0.5, self.grid_size - 0.5)
            grid_ax.set_ylim(-0.5, self.grid_size - 0.5)
            grid_ax.set_aspect('equal')

            # Draw grid lines
            for i in range(self.grid_size + 1):
                grid_ax.axhline(i - 0.5, color='lightgray', linewidth=0.5)
                grid_ax.axvline(i - 0.5, color='lightgray', linewidth=0.5)

            stick_colors = ['royalblue', 'crimson', 'darkorange', 'forestgreen', 'purple']
            edge_colors = ['blue', 'red', 'orange', 'green', 'purple']
            winning_agent = step_detail.get("winning_agent", -1) if step_detail else None

            def draw_stick_figure(ax, cx, cy, color, lw=1.5):
                ax.add_patch(plt.Circle((cx, cy - 0.22), 0.10, facecolor=color, edgecolor=color, linewidth=1))
                ax.plot([cx, cx], [cy - 0.12, cy + 0.08], color=color, linewidth=lw)
                ax.plot([cx - 0.18, cx + 0.18], [cy - 0.02, cy - 0.02], color=color, linewidth=lw)
                ax.plot([cx, cx - 0.16], [cy + 0.08, cy + 0.28], color=color, linewidth=lw)
                ax.plot([cx, cx + 0.16], [cy + 0.08, cy + 0.28], color=color, linewidth=lw)

            # Draw targets
            for i in range(self.num_agents):
                target_row, target_col = target_positions[i]
                is_controlling = (winning_agent == i)

                if targets_reached[i] == 0:
                    draw_stick_figure(grid_ax, target_col, target_row, stick_colors[i % len(stick_colors)])
                    if is_controlling:
                        grid_ax.text(target_col, target_row - 0.6, '⚡',
                               ha='center', va='center', fontsize=8, color='gold')
                else:
                    draw_stick_figure(grid_ax, target_col, target_row, 'darkgreen')
                    grid_ax.text(target_col, target_row - 0.5, '✓',
                           ha='center', va='center', fontsize=8, fontweight='bold', color='darkgreen')

            # Draw agent with ring indicating controlling agent
            if winning_agent is not None and 0 <= winning_agent < self.num_agents:
                grid_ax.add_patch(plt.Circle((agent_col, agent_row), 0.35,
                                       facecolor='none', edgecolor=edge_colors[winning_agent % len(edge_colors)], linewidth=3))

            grid_ax.text(agent_col, agent_row, '☕',
                   ha='center', va='center', fontsize=16)

            # Set grid title and ticks
            grid_ax.set_title(f'Step {frame}', fontsize=11, fontweight='bold')

            # Optimize tick marks for large grids
            if self.grid_size <= 15:
                tick_step = 1
            elif self.grid_size <= 30:
                tick_step = 2
            else:
                tick_step = 5

            tick_positions = list(range(0, self.grid_size, tick_step))
            grid_ax.set_xticks(tick_positions)
            grid_ax.set_yticks(tick_positions)
            grid_ax.invert_yaxis()

            # === Draw Info Panel ===
            info_ax.axis('off')

            # Build info text
            info_lines = []
            info_lines.append(f'MULTI-AGENT COMPETITION\n')
            info_lines.append(f'Grid: {self.grid_size}x{self.grid_size}')
            info_lines.append(f'Agents: {self.num_agents}')
            controller_label = str(winning_agent) if winning_agent is not None and winning_agent >= 0 else "None"
            info_lines.append(f'Controller: {controller_label}')
            info_lines.append(f'')

            # Agent details
            if step_detail and actions and rewards:
                info_lines.append(f'AGENT DETAILS:')
                direction_names = {0: "←", 1: "→", 2: "↑", 3: "↓"}

                # Calculate cumulative rewards
                cumulative = {}
                for i in range(self.num_agents):
                    cumulative[f"agent_{i}"] = sum(
                        episode_data["rewards"][f].get(f"agent_{i}", 0)
                        for f in range(min(frame + 1, len(episode_data["rewards"])))
                    )

                for i in range(self.num_agents):
                    agent_key = f"agent_{i}"
                    if agent_key in actions:
                        action_data = actions[agent_key]
                        bid = action_data.get('bid', 0)
                        direction = direction_names.get(action_data.get('direction', 0), '?')
                        window_steps = None
                        if self.window_bidding and "window" in action_data:
                            window_steps = int(action_data.get("window", 0)) + 1
                        reward = cumulative.get(agent_key, 0)
                        window_str = f' W:{window_steps:2d}' if window_steps is not None else ''

                        # Highlight controlling agent
                        if i == winning_agent:
                            info_lines.append(f'  [{i}] ⚡ Bid:{bid:2d}{window_str} {direction} R:{reward:6.1f}')
                        else:
                            info_lines.append(f'  [{i}]   Bid:{bid:2d}{window_str} {direction} R:{reward:6.1f}')

            # Render text
            info_text = '\n'.join(info_lines)
            info_ax.text(0.05, 0.95, info_text,
                        transform=info_ax.transAxes,
                        fontfamily='monospace',
                        fontsize=9,
                        verticalalignment='top',
                        horizontalalignment='left')

        # Render frames to numpy arrays
        frames = []
        num_frames = len(episode_data["states"]) + 3
        for frame_idx in range(num_frames):
            animate(frame_idx)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            frames.append(frame)

        plt.close(fig)

        # Write video using OpenCV
        if len(frames) > 0:
            h, w = frames[0].shape[:2]
            output_path_mp4 = str(output_path).replace('.gif', '.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path_mp4, fourcc, fps, (w, h))

            try:
                for frame in frames:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                out.release()
                print(f"✅ Competition video saved: {output_path_mp4}")
            except Exception as e:
                print(f"⚠️  Could not save video {output_path_mp4}: {e}")
                out.release()

    def close(self):
        """Clean up rendering resources."""
        pass


# ============================================================================
# Evaluation Utilities
# ============================================================================

def evaluate_multi_agent_policy(
    env: 'BiddingGridworld',
    policy_fn,
    num_episodes: int,
    target_expiry_penalty: float = 0.0,
    verbose: bool = True
) -> Dict[str, List]:
    """
    Evaluate a multi-agent policy on the environment.

    Args:
        env: BiddingGridworld environment (multi-agent mode)
        policy_fn: Callable that takes observations (shape: [num_agents, obs_dim]) and returns
                  actions (shape: [num_agents, action_dim]). Should be deterministic for evaluation.
        num_episodes: Number of episodes to evaluate
        target_expiry_penalty: Target expiry penalty value (for counting expired targets)
        verbose: Whether to print progress

    Returns:
        Dictionary containing evaluation statistics:
        - episode_returns: List of total returns per episode
        - episode_lengths: List of episode lengths
        - targets_reached_per_episode: List of unique targets reached per episode
        - expired_targets_per_episode: List of expired targets per episode
        - min_targets_reached_per_episode: List of minimum reaches across agents
        - targets_reached_count_per_episode: List of per-agent reach counts
        - episode_data_list: List of episode data dicts (states, actions, rewards, step_details)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating multi-agent policy")
        print(f"Running {num_episodes} episodes")
        print(f"{'='*60}\n")

    eval_stats = {
        "episode_returns": [],
        "episode_lengths": [],
        "targets_reached_per_episode": [],
        "expired_targets_per_episode": [],
        "min_targets_reached_per_episode": [],
        "targets_reached_count_per_episode": [],
        "episode_data_list": [],
    }

    for episode_idx in range(num_episodes):
        # Reset environment
        base_obs, _ = env.reset()

        # Episode tracking
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_step_details = []
        episode_return = 0
        step_count = 0
        terminated = False
        truncated = False

        # Track expired targets and targets reached per agent
        episode_expired_count = 0
        targets_reached_count = np.zeros(env.num_agents, dtype=np.int32)

        while not (terminated or truncated):
            # Store state
            episode_states.append(env._get_centralized_observation().copy())

            # Get actions from policy
            actions = policy_fn(base_obs)  # Should return [num_agents, action_dim]

            # Convert to environment action format
            env_action = {}
            for agent_idx in range(env.num_agents):
                agent_action = {
                    "direction": int(actions[agent_idx, 0]),
                    "bid": int(actions[agent_idx, 1]),
                }
                if env.window_bidding:
                    agent_action["window"] = int(actions[agent_idx, 2])
                env_action[f"agent_{agent_idx}"] = agent_action

            episode_actions.append(env_action)

            # Step environment
            base_obs, rewards_dict, terminated, truncated, info = env.step(env_action)

            episode_return += sum(rewards_dict.values())
            episode_rewards.append(rewards_dict)

            # Track expired targets this step
            for agent_idx in range(env.num_agents):
                reward = rewards_dict[f"agent_{agent_idx}"]
                # If we see a large negative reward matching the expiry penalty, count it
                if target_expiry_penalty > 0:
                    if reward <= -target_expiry_penalty:
                        episode_expired_count += 1

            # Track targets reached using info dict
            targets_just_reached = info.get("targets_just_reached", {})
            for agent_idx in range(env.num_agents):
                if targets_just_reached.get(agent_idx, False):
                    targets_reached_count[agent_idx] += 1

            # Store step details
            episode_step_details.append({
                "winning_agent": info.get("winning_agent", -1),
                "bids": info.get("bids", {}),
                "window_agent": info.get("window_agent", None),
                "window_steps_remaining": info.get("window_steps_remaining", 0),
                "bid_penalty_applied": info.get("bid_penalty_applied", False),
            })

            step_count += 1

        # Count total targets reached (at least once during episode)
        targets_reached = sum(1 for count in targets_reached_count if count > 0)
        min_targets_reached = int(np.min(targets_reached_count))

        eval_stats["episode_returns"].append(episode_return)
        eval_stats["episode_lengths"].append(step_count)
        eval_stats["targets_reached_per_episode"].append(targets_reached)
        eval_stats["expired_targets_per_episode"].append(episode_expired_count)
        eval_stats["min_targets_reached_per_episode"].append(min_targets_reached)
        eval_stats["targets_reached_count_per_episode"].append(targets_reached_count.tolist())

        # Store episode data for visualization
        episode_data = {
            "states": episode_states,
            "actions": episode_actions,
            "rewards": episode_rewards,
            "step_details": episode_step_details,
        }
        eval_stats["episode_data_list"].append(episode_data)

        if verbose:
            print(f"  Episode {episode_idx + 1}: Return={episode_return:.2f}, "
                  f"Length={step_count}, Targets={targets_reached}/{env.num_agents}, "
                  f"Expired={episode_expired_count}, MinReached={min_targets_reached}")

    if verbose:
        avg_return = np.mean(eval_stats["episode_returns"])
        avg_length = np.mean(eval_stats["episode_lengths"])
        avg_targets = np.mean(eval_stats["targets_reached_per_episode"])
        avg_expired = np.mean(eval_stats["expired_targets_per_episode"])
        avg_min_reached = np.mean(eval_stats["min_targets_reached_per_episode"])
        success_rate = sum(1 for t in eval_stats["targets_reached_per_episode"]
                          if t == env.num_agents) / num_episodes

        print(f"\nEvaluation Summary:")
        print(f"  Average Return: {avg_return:.2f}")
        print(f"  Average Length: {avg_length:.1f}")
        print(f"  Average Targets: {avg_targets:.2f}/{env.num_agents}")
        print(f"  Average Expired: {avg_expired:.2f} ± {np.std(eval_stats['expired_targets_per_episode']):.2f}")
        print(f"  Average Min Reached: {avg_min_reached:.2f} ± {np.std(eval_stats['min_targets_reached_per_episode']):.2f}")
        print(f"  Success Rate: {success_rate*100:.1f}%\n")

    return eval_stats


def evaluate_single_agent_policy(
    env: 'BiddingGridworld',
    policy_fn,
    num_episodes: int,
    target_expiry_penalty: float = 0.0,
    verbose: bool = True
) -> Dict[str, List]:
    """
    Evaluate a single-agent policy on the environment.

    Args:
        env: BiddingGridworld environment (single-agent mode)
        policy_fn: Callable that takes observation (shape: [obs_dim]) and returns
                  action (scalar or shape: [1]). Should be deterministic for evaluation.
        num_episodes: Number of episodes to evaluate
        target_expiry_penalty: Target expiry penalty value (for counting expired targets)
        verbose: Whether to print progress

    Returns:
        Dictionary containing evaluation statistics:
        - episode_returns: List of total returns per episode
        - episode_lengths: List of episode lengths
        - targets_reached_per_episode: List of unique targets reached per episode
        - expired_targets_per_episode: List of expired targets per episode
        - min_targets_reached_per_episode: List of minimum reaches across targets
        - targets_reached_count_per_episode: List of per-target reach counts
        - episode_data_list: List of episode data dicts (states, actions, rewards)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating single-agent policy")
        print(f"Running {num_episodes} episodes")
        print(f"{'='*60}\n")

    eval_stats = {
        "episode_returns": [],
        "episode_lengths": [],
        "targets_reached_per_episode": [],
        "expired_targets_per_episode": [],
        "min_targets_reached_per_episode": [],
        "targets_reached_count_per_episode": [],
        "episode_data_list": [],
    }

    for episode_idx in range(num_episodes):
        # Reset environment
        obs, _ = env.reset()

        # Episode tracking
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_return = 0
        step_count = 0
        terminated = False
        truncated = False

        # Track expired targets and targets reached per target
        episode_expired_count = 0
        targets_reached_count = np.zeros(env.num_agents, dtype=np.int32)  # num_agents = num_targets in single mode

        while not (terminated or truncated):
            # Store state
            episode_states.append(obs.copy())

            # Get action from policy
            action = policy_fn(obs)  # Should return scalar or [1]
            if isinstance(action, np.ndarray):
                action = int(action.item())
            else:
                action = int(action)

            episode_actions.append(action)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            episode_return += reward
            episode_rewards.append(reward)

            # Track expired targets (if expiry is enabled)
            if target_expiry_penalty > 0:
                if reward <= -target_expiry_penalty:
                    episode_expired_count += 1

            # Track targets reached using info dict
            targets_just_reached = info.get("targets_just_reached", {})
            for target_idx in range(env.num_agents):
                if targets_just_reached.get(target_idx, False):
                    targets_reached_count[target_idx] += 1

            step_count += 1

        # Count total targets reached (at least once during episode)
        targets_reached = sum(1 for count in targets_reached_count if count > 0)
        min_targets_reached = int(np.min(targets_reached_count))

        eval_stats["episode_returns"].append(episode_return)
        eval_stats["episode_lengths"].append(step_count)
        eval_stats["targets_reached_per_episode"].append(targets_reached)
        eval_stats["expired_targets_per_episode"].append(episode_expired_count)
        eval_stats["min_targets_reached_per_episode"].append(min_targets_reached)
        eval_stats["targets_reached_count_per_episode"].append(targets_reached_count.tolist())

        # Store episode data for visualization
        episode_data = {
            "states": episode_states,
            "actions": episode_actions,
            "rewards": episode_rewards,
        }
        eval_stats["episode_data_list"].append(episode_data)

        if verbose:
            print(f"  Episode {episode_idx + 1}: Return={episode_return:.2f}, "
                  f"Length={step_count}, Targets={targets_reached}/{env.num_agents}, "
                  f"Expired={episode_expired_count}, MinReached={min_targets_reached}")

    if verbose:
        avg_return = np.mean(eval_stats["episode_returns"])
        avg_length = np.mean(eval_stats["episode_lengths"])
        avg_targets = np.mean(eval_stats["targets_reached_per_episode"])
        avg_expired = np.mean(eval_stats["expired_targets_per_episode"])
        avg_min_reached = np.mean(eval_stats["min_targets_reached_per_episode"])
        success_rate = sum(1 for t in eval_stats["targets_reached_per_episode"]
                          if t == env.num_agents) / num_episodes

        print(f"\nEvaluation Summary:")
        print(f"  Average Return: {avg_return:.2f}")
        print(f"  Average Length: {avg_length:.1f}")
        print(f"  Average Targets: {avg_targets:.2f}/{env.num_agents}")
        print(f"  Average Expired: {avg_expired:.2f} ± {np.std(eval_stats['expired_targets_per_episode']):.2f}")
        print(f"  Average Min Reached: {avg_min_reached:.2f} ± {np.std(eval_stats['min_targets_reached_per_episode']):.2f}")
        print(f"  Success Rate: {success_rate*100:.1f}%\n")

    return eval_stats


def create_random_action(num_agents: int = 2, bid_upper_bound: int = 10, window_bidding: bool = False, action_window: int = 1) -> Dict:
    """Helper function to create a random action for testing."""
    action = {}
    for i in range(num_agents):
        agent_action = {
            "direction": np.random.randint(0, 4),
            "bid": np.random.randint(0, bid_upper_bound + 1)
        }
        if window_bidding:
            agent_action["window"] = np.random.randint(1, action_window + 1)
        action[f"agent_{i}"] = agent_action
    return action


if __name__ == "__main__":
    # Example usage and testing
    print("Testing BiddingGridworld Environment")
    
    # Create environment
    env = BiddingGridworld(
        grid_size=10,
        bid_upper_bound=5,
        bid_penalty=0.1,
        target_reward=10.0
    )
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print("Initial observation (Box vector):")
    print(obs)
    env.render()
    
    # Run a few steps
    for step in range(10):
        action = create_random_action(env.num_agents, env.bid_upper_bound)
        print(f"\nStep {step + 1}")
        print(f"Actions: {action}")
        
        obs, rewards, terminated, truncated, info = env.step(action)
        
        print(f"Rewards: {rewards}")
        print(f"Winner: Agent {info['winning_agent']}")
        print(f"Bids: {info['bids']}")
        
        env.render()
        
        if terminated or truncated:
            print("Episode finished!")
            break
    
    env.close()


class MovingTargetBiddingGridworld(BiddingGridworld):
    """
    A variant of BiddingGridworld where targets move dynamically.

    Each target has a direction and moves at each step with the following logic:
    - With 0.1 probability, the target randomly changes direction
    - If a target hits a wall/edge, it must choose a new valid direction
    - Target positions are updated after agent movement
    """

    def __init__(
        self,
        grid_size: int = 10,
        num_agents: int = 2,
        target_positions: Optional[List[Tuple[int, int]]] = None,
        bid_upper_bound: int = 10,
        bid_penalty: float = 0.1,
        target_reward: float = 10.0,
        direction_change_prob: float = 0.1,
        target_move_interval: int = 1,
        max_steps: int = 100,
        action_window: int = 1,
        distance_reward_scale: float = 0.0,
        target_expiry_steps: Optional[int] = None,
        target_expiry_penalty: float = 5.0,
        single_agent_mode: bool = False,
        window_bidding: bool = False,
        window_penalty: float = 0.0,
        reward_decay_factor: float = 0.0,
        visible_targets: Optional[int] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the MovingTargetBiddingGridworld environment.

        Args:
            grid_size: Size of the square gridworld (default: 10)
            num_agents: Number of agents/targets (default: 2)
            target_positions: Optional list of (row, col) tuples for initial target positions.
                            If None, random positions will be assigned. (default: None)
            bid_upper_bound: Maximum bid value (default: 10)
            bid_penalty: Penalty multiplier for bids (default: 0.1)
            target_reward: Reward for reaching target (default: 10.0)
            direction_change_prob: Probability of randomly changing direction (default: 0.1)
            target_move_interval: Number of steps between target movements (default: 1)
                                 Setting to 1 means targets move every step.
                                 Setting to N means targets move every N steps.
            max_steps: Maximum number of steps per episode (default: 100)
            action_window: Number of steps a winning agent controls the action (default: 1)
                          When window_bidding is False, this is the fixed window length.
                          When window_bidding is True, this is the maximum window agents can bid for.
            distance_reward_scale: Reward scaling for distance improvements (default: 0.0, disabled)
                                  Positive values reward getting closer to target
            target_expiry_steps: Maximum steps allowed before target expiry penalty (default: None, disabled)
                                If set, agents receive penalty if target not reached within this many steps
            target_expiry_penalty: Penalty for not reaching target within expiry_steps (default: 5.0)
            single_agent_mode: If True, use single-agent mode with no bidding (default: False)
            window_bidding: If True, agents can choose their control window length (1 to action_window) (default: False)
            window_penalty: Penalty multiplier for chosen window length (default: 0.0)
            reward_decay_factor: Reward decay based on relative target count (single-agent mode only) (default: 0.0)
            visible_targets: Number of nearest other targets visible to each agent (multi-agent mode only) (default: None)
            render_mode: Rendering mode (default: None)
        """
        super().__init__(
            grid_size=grid_size,
            num_agents=num_agents,
            target_positions=target_positions,
            bid_upper_bound=bid_upper_bound,
            bid_penalty=bid_penalty,
            target_reward=target_reward,
            max_steps=max_steps,
            action_window=action_window,
            distance_reward_scale=distance_reward_scale,
            target_expiry_steps=target_expiry_steps,
            target_expiry_penalty=target_expiry_penalty,
            single_agent_mode=single_agent_mode,
            window_bidding=window_bidding,
            window_penalty=window_penalty,
            reward_decay_factor=reward_decay_factor,
            visible_targets=visible_targets,
            render_mode=render_mode
        )

        self.direction_change_prob = direction_change_prob
        self.target_move_interval = target_move_interval

        # Initialize target directions (will be set in reset)
        self.target_directions = []

        # Track steps since last target movement for each target
        self.steps_since_target_move = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment with moving targets."""
        obs, info = super().reset(seed=seed, options=options)

        # Initialize random directions for each target
        self.target_directions = []
        self.steps_since_target_move = []
        for _ in range(self.num_agents):
            # Random initial direction: 0=Left, 1=Right, 2=Up, 3=Down
            self.target_directions.append(np.random.randint(0, 4))
            # Initialize step counter for each target
            self.steps_since_target_move.append(0)

        return obs, info

    def step(self, action: Dict) -> Tuple[np.ndarray, Dict, bool, bool, Dict]:
        """
        Execute one step with moving targets.

        The step sequence is:
        1. Agent moves based on bidding
        2. Check if any targets are reached
        3. Targets move
        4. Return observation with updated target positions
        """
        # Execute parent step (agent movement and reward calculation)
        obs, rewards, terminated, truncated, info = super().step(action)

        # Save winning_agent, bids, and targets_just_reached from parent step
        # targets_just_reached indicates which targets were reached BEFORE they moved/respawned
        winning_agent = info.get("winning_agent", -1)
        bids = info.get("bids", {})
        targets_just_reached = info.get("targets_just_reached", {})

        # Move targets after agent has moved
        self._move_targets()

        # Update observation and info with new target positions
        obs = self._get_observation()
        info = self._get_info()

        # Restore winning_agent, bids, and targets_just_reached from parent step
        info["winning_agent"] = winning_agent
        info["bids"] = bids
        info["targets_just_reached"] = targets_just_reached

        # terminated is always false in this moving targets version
        terminated = False

        return obs, rewards, terminated, truncated, info

    def _move_targets(self):
        """Move all targets according to their directions."""
        for i in range(self.num_agents):
            # If target was just reached or expired, respawn it at a random position
            if self.targets_reached[i] == 1 or self.targets_expired_this_step.get(i, False):
                # Get all available positions (excluding agent position)
                available_positions = [
                    (r, c) for r in range(self.grid_size)
                    for c in range(self.grid_size)
                    if not np.array_equal(np.array([r, c]), self.agent_position)
                ]
                # Respawn at random position
                new_pos = random.choice(available_positions)
                self.target_positions[i] = np.array(new_pos, dtype=np.int32)
                # Reset the reached flag
                self.targets_reached[i] = 0
                # Assign new random direction
                self.target_directions[i] = np.random.randint(0, 4)
                # Reset the step counter for this target
                self.target_step_counters[i] = 0
                # Reset movement interval counter
                self.steps_since_target_move[i] = 0
                continue

            # Increment step counter for this target
            self.steps_since_target_move[i] += 1

            # Only move target if interval has been reached
            if self.steps_since_target_move[i] >= self.target_move_interval:
                # Reset movement counter
                self.steps_since_target_move[i] = 0

                # With probability direction_change_prob, randomly change direction
                if np.random.random() < self.direction_change_prob:
                    self.target_directions[i] = np.random.randint(0, 4)

                # Calculate new position
                current_pos = self.target_positions[i]
                new_pos = self._move_target_in_direction(current_pos, self.target_directions[i])

                # Check if hit a wall (new_pos == current_pos means we couldn't move)
                if np.array_equal(new_pos, current_pos):
                    # Hit a wall, choose a new valid direction
                    valid_directions = self._get_valid_directions(current_pos)
                    if valid_directions:
                        self.target_directions[i] = np.random.choice(valid_directions)
                        new_pos = self._move_target_in_direction(current_pos, self.target_directions[i])

                # Update target position
                self.target_positions[i] = new_pos

    def _move_target_in_direction(self, position: np.ndarray, direction: int) -> np.ndarray:
        """Move target in the specified direction, respecting grid boundaries."""
        new_position = position.copy()

        if direction == 0:  # Left
            new_position[1] = max(0, position[1] - 1)
        elif direction == 1:  # Right
            new_position[1] = min(self.grid_size - 1, position[1] + 1)
        elif direction == 2:  # Up
            new_position[0] = max(0, position[0] - 1)
        elif direction == 3:  # Down
            new_position[0] = min(self.grid_size - 1, position[0] + 1)

        return new_position

    def _get_valid_directions(self, position: np.ndarray) -> List[int]:
        """Get list of valid directions from current position (not hitting walls)."""
        valid = []

        # Check each direction
        if position[1] > 0:  # Can go left
            valid.append(0)
        if position[1] < self.grid_size - 1:  # Can go right
            valid.append(1)
        if position[0] > 0:  # Can go up
            valid.append(2)
        if position[0] < self.grid_size - 1:  # Can go down
            valid.append(3)

        return valid


if __name__ == "__main__":
    # Test the MovingTargetBiddingGridworld
    print("\n" + "="*60)
    print("Testing MovingTargetBiddingGridworld")
    print("="*60)

    env = MovingTargetBiddingGridworld(
        grid_size=5,
        num_agents=2,
        bid_upper_bound=3,
        bid_penalty=0.1,
        target_reward=10.0,
        direction_change_prob=0.2,  # Higher probability for testing
        target_move_interval=1  # Move targets every step
    )

    obs, info = env.reset(seed=42)
    print("\nInitial state:")
    env.render()

    # Run a few steps to see targets moving
    for step in range(10):
        action = create_random_action(env.num_agents, env.bid_upper_bound)
        print(f"\nStep {step + 1}")
        print(f"Actions: {action}")

        obs, rewards, terminated, truncated, info = env.step(action)

        print(f"Rewards: {rewards}")
        print(f"Winner: Agent {info['winning_agent']}")

        env.render()

        if terminated or truncated:
            print("Episode finished!")
            break

    env.close()
    print("\n✅ MovingTargetBiddingGridworld testing completed!")

    # Test single-agent mode
    print("\n" + "="*60)
    print("Testing Single-Agent Mode")
    print("="*60)

    env_single = BiddingGridworld(
        grid_size=5,
        num_agents=3,  # 3 targets to reach
        target_reward=10.0,
        max_steps=50,
        single_agent_mode=True,
        distance_reward_scale=0.1
    )

    obs, info = env_single.reset(seed=123)
    print("\nInitial state (Single Agent with 3 targets):")
    env_single.render()

    # Run a few steps with random actions
    total_reward = 0
    for step in range(15):
        action = np.random.randint(0, 4)  # Just a direction
        print(f"\nStep {step + 1}")
        print(f"Action: {env_single.action_meanings[action]}")

        obs, reward, terminated, truncated, info = env_single.step(action)

        total_reward += reward
        print(f"Step Reward: {reward:.2f}")
        print(f"Total Reward: {total_reward:.2f}")

        env_single.render()

        if terminated or truncated:
            print("Episode finished!")
            break

    env_single.close()
    print("\n✅ Single-agent mode testing completed!")
