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
        bid_upper_bound: int = 10,
        bid_penalty: float = 0.1,
        target_reward: float = 10.0,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the BiddingGridworld environment.
        
        Args:
            grid_size: Size of the square gridworld (default: 10)
            bid_upper_bound: Maximum bid value (default: 10)
            bid_penalty: Penalty multiplier for bids (default: 0.1)
            target_reward: Reward for reaching target (default: 10.0)
            render_mode: Rendering mode (default: None)
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.bid_upper_bound = bid_upper_bound
        self.bid_penalty = bid_penalty
        self.target_reward = target_reward
        self.render_mode = render_mode
        
        # Actions: 0=Left, 1=Right, 2=Up, 3=Down
        self.action_meanings = {0: "Left", 1: "Right", 2: "Up", 3: "Down"}
        
        # Action space for each agent: (direction, bid)
        # Each agent submits an action that consists of:
        # - direction: discrete action (0-3 for L,R,U,D)
        # - bid: integer value (0 to bid_upper_bound)
        self.action_space = spaces.Dict({
            "agent_0": spaces.Dict({
                "direction": spaces.Discrete(4),
                "bid": spaces.Discrete(bid_upper_bound + 1)
            }),
            "agent_1": spaces.Dict({
                "direction": spaces.Discrete(4),
                "bid": spaces.Discrete(bid_upper_bound + 1)
            })
        })
        
        # Observation space: Box vector for NN policies (normalized to [0, 1])
        # [agent_row_norm, agent_col_norm, target0_row_norm, target0_col_norm,
        #  target1_row_norm, target1_col_norm, target0_reached, target1_reached]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        
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
        
        # Place targets at different locations to create interesting dynamics
        # Target 0 (for agent 0) - bottom right corner
        self.target_0_position = np.array([self.grid_size-1, self.grid_size-1], dtype=np.int32)
        
        # Target 1 (for agent 1) - top right corner
        self.target_1_position = np.array([0, self.grid_size-1], dtype=np.int32)
        
        # Track which targets have been reached
        self.targets_reached = np.array([0, 0], dtype=np.int32)
        
        # Step counter
        self.step_count = 0
        self.max_steps = self.grid_size * self.grid_size * 2  # Reasonable episode length
        
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
        
        # Extract actions and bids
        agent_0_action = action["agent_0"]
        agent_1_action = action["agent_1"]
        
        direction_0 = agent_0_action["direction"]
        bid_0 = agent_0_action["bid"]
        
        direction_1 = agent_1_action["direction"] 
        bid_1 = agent_1_action["bid"]
        
        # Determine winner of the bid
        if bid_0 > bid_1:
            winning_direction = direction_0
            winning_agent = 0
        elif bid_1 > bid_0:
            winning_direction = direction_1
            winning_agent = 1
        else:
            # Tie - randomly choose winner
            if np.random.random() < 0.5:
                winning_direction = direction_0
                winning_agent = 0
            else:
                winning_direction = direction_1
                winning_agent = 1
        
        # Execute the winning action
        new_position = self._move_agent(self.agent_position, winning_direction)
        self.agent_position = new_position
        
        # Check if targets are reached for the first time this step
        target_0_just_reached = False
        target_1_just_reached = False
        
        if np.array_equal(self.agent_position, self.target_0_position) and self.targets_reached[0] == 0:
            self.targets_reached[0] = 1
            target_0_just_reached = True
        
        if np.array_equal(self.agent_position, self.target_1_position) and self.targets_reached[1] == 0:
            self.targets_reached[1] = 1
            target_1_just_reached = True
        
        # Calculate rewards for both agents
        rewards = self._calculate_rewards(bid_0, bid_1, winning_agent, target_0_just_reached, target_1_just_reached)
        
        # Check termination conditions
        # terminated = bool(np.all(self.targets_reached == 1))  # Both targets reached
        terminated = truncated = self.step_count >= self.max_steps
        
        observation = self._get_observation()
        info = self._get_info()
        info["winning_agent"] = winning_agent
        info["bids"] = {"agent_0": bid_0, "agent_1": bid_1}
        
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
        bid_0: int, 
        bid_1: int, 
        winning_agent: int,
        target_0_just_reached: bool,
        target_1_just_reached: bool
    ) -> Dict[str, float]:
        """Calculate rewards for both agents."""
        rewards = {"agent_0": 0.0, "agent_1": 0.0}
        
        # Bid penalties and rewards
        # Each agent pays penalty for their own bid and receives reward from opponent's bid
        rewards["agent_0"] -= self.bid_penalty * bid_0  # Agent 0 pays for their bid
        rewards["agent_0"] += self.bid_penalty * bid_1  # Agent 0 gets reward from Agent 1's bid
        
        rewards["agent_1"] -= self.bid_penalty * bid_1  # Agent 1 pays for their bid
        rewards["agent_1"] += self.bid_penalty * bid_0  # Agent 1 gets reward from Agent 0's bid
        
        # Target rewards (only given when target is reached for the first time)
        if target_0_just_reached:
            rewards["agent_0"] += self.target_reward
        
        if target_1_just_reached:
            rewards["agent_1"] += self.target_reward
        
        return rewards
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation as a normalized vector in [0, 1]."""
        # Normalize positions to [0,1]; guard division for grid_size==1
        denom = float(self.grid_size - 1) if self.grid_size > 1 else 1.0
        agent_row_norm = float(self.agent_position[0]) / denom
        agent_col_norm = float(self.agent_position[1]) / denom
        t0_row_norm = float(self.target_0_position[0]) / denom
        t0_col_norm = float(self.target_0_position[1]) / denom
        t1_row_norm = float(self.target_1_position[0]) / denom
        t1_col_norm = float(self.target_1_position[1]) / denom

        obs = np.array([
            agent_row_norm,
            agent_col_norm,
            t0_row_norm,
            t0_col_norm,
            t1_row_norm,
            t1_col_norm,
            float(self.targets_reached[0]),
            float(self.targets_reached[1]),
        ], dtype=np.float32)
        return obs
    
    def _get_info(self) -> Dict:
        """Get additional information."""
        return {
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "manhattan_distance_to_target_0": abs(self.agent_position[0] - self.target_0_position[0]) + 
                                            abs(self.agent_position[1] - self.target_0_position[1]),
            "manhattan_distance_to_target_1": abs(self.agent_position[0] - self.target_1_position[0]) + 
                                            abs(self.agent_position[1] - self.target_1_position[1])
        }
    
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
        print(f"Target 0 position: {self.target_0_position} (reached: {bool(self.targets_reached[0])})")
        print(f"Target 1 position: {self.target_1_position} (reached: {bool(self.targets_reached[1])})")
        
        # Create grid visualization
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        
        # Place targets
        grid[self.target_0_position[0], self.target_0_position[1]] = '0' if not self.targets_reached[0] else '✓'
        grid[self.target_1_position[0], self.target_1_position[1]] = '1' if not self.targets_reached[1] else '✓'
        
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


def create_random_action(bid_upper_bound: int = 10) -> Dict:
    """Helper function to create a random action for testing."""
    return {
        "agent_0": {
            "direction": np.random.randint(0, 4),
            "bid": np.random.randint(0, bid_upper_bound + 1)
        },
        "agent_1": {
            "direction": np.random.randint(0, 4), 
            "bid": np.random.randint(0, bid_upper_bound + 1)
        }
    }


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
        action = create_random_action(env.bid_upper_bound)
        print(f"\nStep {step + 1}")
        print(f"Actions: Agent 0: {action['agent_0']}, Agent 1: {action['agent_1']}")
        
        obs, rewards, terminated, truncated, info = env.step(action)
        
        print(f"Rewards: {rewards}")
        print(f"Winner: Agent {info['winning_agent']}")
        print(f"Bids: {info['bids']}")
        
        env.render()
        
        if terminated or truncated:
            print("Episode finished!")
            break
    
    env.close()