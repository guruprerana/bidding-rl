import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Union
from bidding_gridworld import BiddingGridworld


class ZeroSumBiddingWrapper(gym.Env):
    """
    A wrapper around BiddingGridworld that creates a two-player zero-sum Markov game
    focused on a single target. This is used to train individual agents in a 
    decentralized manner.
    
    In this setup:
    - One agent (protagonist) tries to reach their target
    - The other agent (adversary) tries to prevent the protagonist from reaching the target
    - The reward structure is zero-sum: protagonist's reward = -adversary's reward
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        target_agent_id: int = 0,
        grid_size: int = 10,
        bid_upper_bound: int = 10,
        bid_penalty: float = 0.1,
        target_reward: float = 10.0,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the zero-sum wrapper.
        
        Args:
            target_agent_id: Which agent (0 or 1) is the protagonist trying to reach their target
            grid_size: Size of the square gridworld
            bid_upper_bound: Maximum bid value
            bid_penalty: Penalty multiplier for bids
            target_reward: Reward for reaching target
            render_mode: Rendering mode
        """
        super().__init__()
        
        assert target_agent_id in [0, 1], "target_agent_id must be 0 or 1"
        
        self.target_agent_id = target_agent_id
        self.adversary_agent_id = 1 - target_agent_id
        
        # Create the underlying BiddingGridworld environment
        self.env = BiddingGridworld(
            grid_size=grid_size,
            bid_upper_bound=bid_upper_bound,
            bid_penalty=bid_penalty,
            target_reward=target_reward,
            render_mode=render_mode
        )
        
        # Action space: each player submits direction and bid
        self.action_space = spaces.Dict({
            "protagonist": spaces.Dict({
                "direction": spaces.Discrete(4),
                "bid": spaces.Discrete(bid_upper_bound + 1)
            }),
            "adversary": spaces.Dict({
                "direction": spaces.Discrete(4),
                "bid": spaces.Discrete(bid_upper_bound + 1)
            })
        })
        
        # Observation space: same as underlying environment but from perspective of both players
        self.observation_space = spaces.Dict({
            "agent_position": spaces.Discrete(grid_size * grid_size),
            "target_reached": spaces.Discrete(2)  # Only track the target of interest
        })
        
        # Store target position for the protagonist
        if self.target_agent_id == 0:
            self.target_position = self.env.target_0_position.copy()
        else:
            self.target_position = self.env.target_1_position.copy()
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """Reset the environment."""
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Update target position in case it changed (though it's fixed in current implementation)
        if self.target_agent_id == 0:
            self.target_position = self.env.target_0_position.copy()
        else:
            self.target_position = self.env.target_1_position.copy()
        
        zero_sum_obs = self._convert_observation(obs)
        zero_sum_info = self._convert_info(info)
        
        return zero_sum_obs, zero_sum_info
    
    def step(
        self, 
        action: Dict
    ) -> Tuple[Dict, Dict, bool, bool, Dict]:
        """
        Execute one step in the zero-sum game.
        
        Args:
            action: Dictionary with 'protagonist' and 'adversary' actions
        
        Returns:
            observation: Converted observation
            rewards: Zero-sum rewards for both players
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Convert zero-sum actions to BiddingGridworld format
        bidding_action = self._convert_action(action)
        
        # Execute step in underlying environment
        obs, rewards, terminated, truncated, info = self.env.step(bidding_action)
        
        # Convert to zero-sum rewards
        zero_sum_rewards = self._convert_rewards(rewards)
        
        # Convert observation and info
        zero_sum_obs = self._convert_observation(obs)
        zero_sum_info = self._convert_info(info)
        
        # Termination condition: only when protagonist's target is reached
        # (we ignore the other target for this zero-sum game)
        target_reached = obs["targets_reached"][self.target_agent_id] == 1
        terminated = target_reached
        
        return zero_sum_obs, zero_sum_rewards, terminated, truncated, zero_sum_info
    
    def _convert_action(self, action: Dict) -> Dict:
        """Convert zero-sum action format to BiddingGridworld format."""
        bidding_action = {"agent_0": None, "agent_1": None}
        
        # Assign protagonist and adversary actions to correct agent slots
        bidding_action[f"agent_{self.target_agent_id}"] = action["protagonist"]
        bidding_action[f"agent_{self.adversary_agent_id}"] = action["adversary"]
        
        return bidding_action
    
    def _convert_observation(self, obs: Dict) -> Dict:
        """Convert BiddingGridworld observation to zero-sum format."""
        return {
            "agent_position": obs["agent_position"],
            "target_reached": obs["targets_reached"][self.target_agent_id]
        }
    
    def _convert_rewards(self, rewards: Dict) -> Dict:
        """Convert BiddingGridworld rewards to zero-sum format."""
        protagonist_reward = rewards[f"agent_{self.target_agent_id}"]
        
        # Pure zero-sum: adversary reward is simply negative of protagonist reward
        adversary_reward = -protagonist_reward
        
        return {
            "protagonist": protagonist_reward,
            "adversary": adversary_reward
        }
    
    def _convert_info(self, info: Dict) -> Dict:
        """Convert BiddingGridworld info to zero-sum format."""
        # Add distance to target for the protagonist
        if self.target_agent_id == 0:
            distance_key = "manhattan_distance_to_target_0"
        else:
            distance_key = "manhattan_distance_to_target_1"
        
        return {
            "step_count": info["step_count"],
            "max_steps": info["max_steps"],
            "distance_to_target": info[distance_key],
            "winning_agent": info.get("winning_agent", -1),
            "bids": info.get("bids", {}),
            "target_agent_id": self.target_agent_id,
            "underlying_rewards": info.get("underlying_rewards", {})
        }
    
    def render(self, mode: str = "human"):
        """Render the environment."""
        if mode == "human":
            print(f"Zero-Sum Game - Target Agent: {self.target_agent_id}")
            print(f"Target Position: {self.target_position}")
        return self.env.render(mode)
    
    def observation_to_state_key(self, observation: Dict) -> Tuple[int, int]:
        """
        Convert observation to a consistent state key tuple.
        
        Args:
            observation: Observation dict with 'agent_position' and 'target_reached'
            
        Returns:
            Tuple of (agent_position, target_reached) as integers
        """
        return (int(observation["agent_position"]), int(observation["target_reached"]))
    
    def close(self):
        """Close the environment."""
        return self.env.close()
    
    @property
    def grid_size(self):
        """Get grid size from underlying environment."""
        return self.env.grid_size
    
    @property
    def bid_upper_bound(self):
        """Get bid upper bound from underlying environment."""
        return self.env.bid_upper_bound


def create_random_zero_sum_action(bid_upper_bound: int = 10) -> Dict:
    """Helper function to create a random action for the zero-sum game."""
    return {
        "protagonist": {
            "direction": np.random.randint(0, 4),
            "bid": np.random.randint(0, bid_upper_bound + 1)
        },
        "adversary": {
            "direction": np.random.randint(0, 4),
            "bid": np.random.randint(0, bid_upper_bound + 1)
        }
    }


if __name__ == "__main__":
    # Example usage and testing
    print("Testing ZeroSumBiddingWrapper")
    
    # Test for agent 0 (trying to reach target 0)
    print("\n=== Testing Agent 0 vs Adversary ===")
    env_0 = ZeroSumBiddingWrapper(
        target_agent_id=0,
        grid_size=5,
        bid_upper_bound=3,
        bid_penalty=0.1,
        target_reward=10.0
    )
    
    obs, info = env_0.reset(seed=42)
    print("Initial observation:", obs)
    print("Initial info:", info)
    env_0.render()
    
    # Run a few steps
    for step in range(5):
        action = create_random_zero_sum_action(env_0.bid_upper_bound)
        print(f"\nStep {step + 1}")
        print(f"Action: {action}")
        
        obs, rewards, terminated, truncated, info = env_0.step(action)
        
        print(f"Observation: {obs}")
        print(f"Rewards: {rewards}")
        print(f"Terminated: {terminated}")
        
        if terminated or truncated:
            print("Episode finished!")
            break
    
    env_0.close()
    
    # Test for agent 1 (trying to reach target 1)
    print("\n\n=== Testing Agent 1 vs Adversary ===")
    env_1 = ZeroSumBiddingWrapper(
        target_agent_id=1,
        grid_size=5,
        bid_upper_bound=3,
        bid_penalty=0.1,
        target_reward=10.0
    )
    
    obs, info = env_1.reset(seed=42)
    print("Initial observation:", obs)
    print("Target agent 1 trying to reach:", env_1.target_position)
    env_1.render()
    
    env_1.close()
    
    print("\n✅ ZeroSumBiddingWrapper testing completed!")
