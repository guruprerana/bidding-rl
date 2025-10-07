import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, List, Any, Optional
import random


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
            render_mode: Rendering mode (default: None)
        """
        super().__init__()

        assert num_agents >= 2, "Must have at least 2 agents"

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
        self.render_mode = render_mode
        
        # Actions: 0=Left, 1=Right, 2=Up, 3=Down
        self.action_meanings = {0: "Left", 1: "Right", 2: "Up", 3: "Down"}
        
        # Action space for each agent: (direction, bid)
        # Each agent submits an action that consists of:
        # - direction: discrete action (0-3 for L,R,U,D)
        # - bid: integer value (0 to bid_upper_bound)
        self.action_space = spaces.Dict({
            f"agent_{i}": spaces.Dict({
                "direction": spaces.Discrete(4),
                "bid": spaces.Discrete(bid_upper_bound + 1)
            })
            for i in range(num_agents)
        })
        
        # Observation space: Box vector for NN policies (normalized to [0, 1])
        # [agent_row_norm, agent_col_norm, target0_row_norm, target0_col_norm, ..., target0_reached, target1_reached, ...]
        # Shape: 2 (agent position) + 2 * num_agents (target positions) + num_agents (target reached flags)
        obs_dim = 2 + 2 * num_agents + num_agents
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

        # Step counter
        self.step_count = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: Dict) -> Tuple[np.ndarray, Dict, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Dictionary containing actions for both agents
                   Each action has 'direction' and 'bid' components
        
        Returns:
            observation: Current state observation
            rewards: Rewards for both agents
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        self.step_count += 1

        # Extract actions and bids for all agents
        agent_actions = {}
        agent_bids = {}
        for i in range(self.num_agents):
            agent_key = f"agent_{i}"
            agent_actions[i] = action[agent_key]
            agent_bids[i] = action[agent_key]["bid"]

        # Determine winner of the bid (highest bidder)
        max_bid = max(agent_bids.values())
        winners = [agent_id for agent_id, bid in agent_bids.items() if bid == max_bid]

        # If tie, randomly choose among winners
        winning_agent = random.choice(winners)
        winning_direction = agent_actions[winning_agent]["direction"]
        
        # Execute the winning action
        new_position = self._move_agent(self.agent_position, winning_direction)
        self.agent_position = new_position
        
        # Check if any targets are reached for the first time this step
        targets_just_reached = {}
        for i in range(self.num_agents):
            if np.array_equal(self.agent_position, self.target_positions[i]) and self.targets_reached[i] == 0:
                self.targets_reached[i] = 1
                targets_just_reached[i] = True
            else:
                targets_just_reached[i] = False

        # Calculate rewards for all agents
        rewards = self._calculate_rewards(agent_bids, winning_agent, targets_just_reached)

        # Check termination conditions
        all_targets_reached = bool(np.all(self.targets_reached == 1))
        max_steps_reached = self.step_count >= self.max_steps

        terminated = all_targets_reached
        truncated = max_steps_reached and not all_targets_reached
        
        observation = self._get_observation()
        info = self._get_info()
        info["winning_agent"] = winning_agent
        info["bids"] = {f"agent_{i}": bid for i, bid in agent_bids.items()}

        return observation, rewards, terminated, truncated, info
    
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
        targets_just_reached: Dict[int, bool]
    ) -> Dict[str, float]:
        """Calculate rewards for all agents."""
        rewards = {f"agent_{i}": 0.0 for i in range(self.num_agents)}

        # Bid penalties and rewards
        # Each agent pays penalty for their own bid and receives reward from all other agents' bids
        for i in range(self.num_agents):
            # Pay for own bid
            rewards[f"agent_{i}"] -= self.bid_penalty * agent_bids[i]
            # Receive reward from other agents' bids (disabled currently)
            # for j in range(self.num_agents):
            #     if i != j:
            #         rewards[f"agent_{i}"] += self.bid_penalty * agent_bids[j]

            if targets_just_reached[i]:
                rewards[f"agent_{i}"] += self.target_reward

        return rewards
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation as a normalized vector in [0, 1]."""
        # Normalize positions to [0,1]; guard division for grid_size==1
        denom = float(self.grid_size - 1) if self.grid_size > 1 else 1.0
        agent_row_norm = float(self.agent_position[0]) / denom
        agent_col_norm = float(self.agent_position[1]) / denom

        # Build observation: [agent_pos, target_0_pos, ..., target_n_pos, target_0_reached, ..., target_n_reached]
        obs_list = [agent_row_norm, agent_col_norm]

        # Add all target positions
        for target_pos in self.target_positions:
            t_row_norm = float(target_pos[0]) / denom
            t_col_norm = float(target_pos[1]) / denom
            obs_list.extend([t_row_norm, t_col_norm])

        # Add all target reached flags
        for i in range(self.num_agents):
            obs_list.append(float(self.targets_reached[i]))

        obs = np.array(obs_list, dtype=np.float32)
        return obs
    
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
    
    def close(self):
        """Clean up rendering resources."""
        pass


def create_random_action(num_agents: int = 2, bid_upper_bound: int = 10) -> Dict:
    """Helper function to create a random action for testing."""
    action = {}
    for i in range(num_agents):
        action[f"agent_{i}"] = {
            "direction": np.random.randint(0, 4),
            "bid": np.random.randint(0, bid_upper_bound + 1)
        }
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
        max_steps: int = 100,
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
            max_steps: Maximum number of steps per episode (default: 100)
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
            render_mode=render_mode
        )

        self.direction_change_prob = direction_change_prob

        # Initialize target directions (will be set in reset)
        self.target_directions = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment with moving targets."""
        obs, info = super().reset(seed=seed, options=options)

        # Initialize random directions for each target
        self.target_directions = []
        for _ in range(self.num_agents):
            # Random initial direction: 0=Left, 1=Right, 2=Up, 3=Down
            self.target_directions.append(np.random.randint(0, 4))

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

        # Move targets after agent has moved
        self._move_targets()

        # Update observation with new target positions
        obs = self._get_observation()
        info = self._get_info()
        info["winning_agent"] = info.get("winning_agent", -1)
        info["bids"] = info.get("bids", {})

        # terminated is always false in this moving targets version
        terminated = False

        return obs, rewards, terminated, truncated, info

    def _move_targets(self):
        """Move all targets according to their directions."""
        for i in range(self.num_agents):
            # If target was just reached, respawn it at a random position
            if self.targets_reached[i] == 1:
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
                continue

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
        direction_change_prob=0.2  # Higher probability for testing
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