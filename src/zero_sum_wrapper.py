import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Union
from bidding_gridworld import BiddingGridworld, MovingTargetBiddingGridworld


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
        num_agents: int = 2,
        grid_size: int = 10,
        target_positions: Optional[List[Tuple[int, int]]] = None,
        bid_upper_bound: int = 10,
        bid_penalty: float = 0.1,
        target_reward: float = 10.0,
        env_class = None,
        render_mode: Optional[str] = None,
        **env_kwargs
    ):
        """
        Initialize the zero-sum wrapper.

        Args:
            target_agent_id: Which agent is the protagonist trying to reach their target
            num_agents: Number of agents/targets in the underlying environment
            grid_size: Size of the square gridworld
            target_positions: Optional list of target positions for all agents
            bid_upper_bound: Maximum bid value
            bid_penalty: Penalty multiplier for bids
            target_reward: Reward for reaching target
            env_class: Environment class to use (BiddingGridworld or MovingTargetBiddingGridworld).
                      If None, defaults to BiddingGridworld.
            render_mode: Rendering mode
            **env_kwargs: Additional keyword arguments to pass to the environment
                         (e.g., direction_change_prob for MovingTargetBiddingGridworld)
        """
        super().__init__()

        assert target_agent_id < num_agents, f"target_agent_id must be < num_agents ({num_agents})"

        self.target_agent_id = target_agent_id
        self.num_agents = num_agents

        # Default to BiddingGridworld if no class specified
        if env_class is None:
            env_class = BiddingGridworld

        # Create the underlying environment (BiddingGridworld or MovingTargetBiddingGridworld)
        self.env = env_class(
            grid_size=grid_size,
            num_agents=num_agents,
            target_positions=target_positions,
            bid_upper_bound=bid_upper_bound,
            bid_penalty=bid_penalty,
            target_reward=target_reward,
            render_mode=render_mode,
            **env_kwargs
        )
        
        # Store bid bound for action conversion
        self._bid_upper_bound = bid_upper_bound
        
        # Action space: flattened for DQN compatibility
        # Each player has 4 directions * (bid_upper_bound + 1) bids possible actions
        actions_per_player = 4 * (bid_upper_bound + 1)
        # Total action space is all combinations of protagonist and adversary actions
        total_actions = actions_per_player * actions_per_player
        self.action_space = spaces.Discrete(total_actions)
        
        # Store for action conversion
        self.actions_per_player = actions_per_player

        # Observation space: same as underlying environment but from perspective of both players
        # [agent_row, agent_col, target_row, target_col, target_reached, target_step_counter]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

        # Target position will be set during reset
        self.target_position = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        obs, info = self.env.reset(seed=seed, options=options)

        # Get target position for the protagonist
        self.target_position = self.env.target_positions[self.target_agent_id].copy()

        zero_sum_obs = self._convert_observation(obs)
        zero_sum_info = self._convert_info(info)

        return zero_sum_obs, zero_sum_info
    
    def step(
        self, 
        action: int
    ) -> Tuple[Dict, Dict, bool, bool, Dict]:
        """
        Execute one step in the zero-sum game.
        
        Args:
            action: Discrete action index
        
        Returns:
            observation: Converted observation
            rewards: Zero-sum rewards for both players
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Convert discrete action to dict format
        action_dict = self._discrete_to_dict_action(action)
            
        # Convert zero-sum actions to BiddingGridworld format
        bidding_action = self._convert_action(action_dict)
        
        # Execute step in underlying environment
        obs, rewards, terminated, truncated, info = self.env.step(bidding_action)
        
        # Convert to zero-sum rewards (return scalar reward for protagonist)
        protagonist_reward = rewards[f"agent_{self.target_agent_id}"]

        # Convert observation and info
        zero_sum_obs = self._convert_observation(obs)
        zero_sum_info = self._convert_info(info)

        # Add detailed rewards to info for analysis
        zero_sum_info["rewards"] = {
            "protagonist": protagonist_reward,
            "adversary": -protagonist_reward
        }

        # Termination condition: episode ends when protagonist's target is reached OR expired
        target_reached = self.env.targets_reached[self.target_agent_id] == 1
        target_expired = self.env.targets_expired_this_step.get(self.target_agent_id, False)
        terminated = target_reached or target_expired

        # Add termination reason to info
        if target_reached:
            zero_sum_info["termination_reason"] = "target_reached"
        elif target_expired:
            zero_sum_info["termination_reason"] = "target_expired"
        else:
            zero_sum_info["termination_reason"] = None

        return zero_sum_obs, protagonist_reward, terminated, truncated, zero_sum_info
    
    def _discrete_to_dict_action(self, action: int) -> Dict:
        """Convert discrete action index to dict format."""
        # Extract protagonist and adversary action indices
        prot_action_idx = action // self.actions_per_player
        adv_action_idx = action % self.actions_per_player
        
        # Convert each action index to direction and bid
        def idx_to_direction_bid(idx):
            direction = idx // (self._bid_upper_bound + 1)
            bid = idx % (self._bid_upper_bound + 1)
            return {"direction": direction, "bid": bid}
        
        return {
            "protagonist": idx_to_direction_bid(prot_action_idx),
            "adversary": idx_to_direction_bid(adv_action_idx)
        }
    
    def _convert_action(self, action: Dict) -> Dict:
        """Convert zero-sum action format to BiddingGridworld format.

        The protagonist action goes to target_agent_id.
        The adversary action is copied to ALL other agents.
        """
        bidding_action = {}

        # Assign protagonist action to target agent
        bidding_action[f"agent_{self.target_agent_id}"] = action["protagonist"]

        # Assign adversary action to ALL other agents
        for i in range(self.num_agents):
            if i != self.target_agent_id:
                bidding_action[f"agent_{i}"] = action["adversary"]

        return bidding_action
    
    def _convert_observation(self, obs):
        """Convert BiddingGridworld observation to zero-sum format."""
        # obs format from BiddingGridworld (variable length):
        # [agent_row_norm, agent_col_norm,
        #  target0_row_norm, target0_col_norm, ..., targetN_row_norm, targetN_col_norm,
        #  target0_reached, ..., targetN_reached,
        #  target0_step_counter_norm, ..., targetN_step_counter_norm]
        #
        # Structure: 2 + 2*num_agents + num_agents + num_agents elements

        # Extract protagonist's target position
        target_idx_base = 2 + 2 * self.target_agent_id  # Index of target row
        target_row = obs[target_idx_base]
        target_col = obs[target_idx_base + 1]

        # Extract protagonist's target reached flag
        target_reached_idx = 2 + 2 * self.num_agents + self.target_agent_id
        target_reached = obs[target_reached_idx]

        # Extract protagonist's target step counter
        target_counter_idx = 2 + 3 * self.num_agents + self.target_agent_id
        target_step_counter = obs[target_counter_idx]

        # Return observation focused on protagonist's target
        # [agent_row, agent_col, target_row, target_col, target_reached, target_step_counter]
        return np.array([
            obs[0],  # agent_row_norm
            obs[1],  # agent_col_norm
            target_row,
            target_col,
            target_reached,
            target_step_counter
        ], dtype=np.float32)

    def _convert_info(self, info: Dict) -> Dict:
        """Convert BiddingGridworld info to zero-sum format."""
        # Add distance to target for the protagonist
        distance_key = f"manhattan_distance_to_target_{self.target_agent_id}"

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
        # Sample random discrete action
        action = env_0.action_space.sample()
        print(f"\nStep {step + 1}")
        print(f"Action (discrete): {action}")

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
    
    # Test ZeroSumBiddingWrapper with MovingTargetBiddingGridworld
    print("\n" + "="*60)
    print("Testing ZeroSumBiddingWrapper with Moving Targets")
    print("="*60)

    env_moving = ZeroSumBiddingWrapper(
        target_agent_id=0,
        grid_size=5,
        num_agents=2,
        bid_upper_bound=3,
        bid_penalty=0.1,
        target_reward=10.0,
        env_class=MovingTargetBiddingGridworld,
        direction_change_prob=0.2,
        max_steps=30
    )

    obs, info = env_moving.reset(seed=42)
    print("\nInitial observation:", obs)
    print("Initial info:", info)
    env_moving.render()

    # Run a few steps to see targets moving
    for step in range(5):
        action = env_moving.action_space.sample()
        print(f"\nStep {step + 1}")

        obs, reward, terminated, truncated, info = env_moving.step(action)

        print(f"Observation: {obs}")
        print(f"Reward: {reward}")

        env_moving.render()

        if terminated or truncated:
            print("Episode finished!")
            break

    env_moving.close()

    print("\n✅ All ZeroSumBiddingWrapper tests completed!")

