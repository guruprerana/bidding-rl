import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, List, Any, Optional
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
            distance_reward_scale: Reward scaling for distance improvements (default: 0.0, disabled)
                                  Positive values reward getting closer to target
            target_expiry_steps: Maximum steps allowed before target expiry penalty (default: None, disabled)
                                If set, agents receive penalty if target not reached within this many steps
            target_expiry_penalty: Penalty for not reaching target within expiry_steps (default: 5.0)
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
        self.action_window = action_window
        self.distance_reward_scale = distance_reward_scale
        self.target_expiry_steps = target_expiry_steps
        self.target_expiry_penalty = target_expiry_penalty
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
        # [agent_row_norm, agent_col_norm, target0_row_norm, target0_col_norm, ...,
        #  target0_reached, target1_reached, ..., target0_step_counter_norm, target1_step_counter_norm, ...]
        # Shape: 2 (agent position) + 2 * num_agents (target positions) + num_agents (target reached flags) + num_agents (step counters)
        obs_dim = 2 + 2 * num_agents + num_agents + num_agents
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
                self.window_steps_remaining = self.action_window - 1  # -1 because current step counts

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

        # Calculate rewards for all agents
        rewards = self._calculate_rewards(
            agent_bids,
            winning_agent,
            targets_just_reached,
            targets_expired,
            apply_bid_penalty,
            self.previous_distances,
            current_distances
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
        info["bids"] = {f"agent_{i}": bid for i, bid in agent_bids.items()}
        info["window_agent"] = self.window_agent
        info["window_steps_remaining"] = self.window_steps_remaining
        info["bid_penalty_applied"] = apply_bid_penalty

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
        targets_just_reached: Dict[int, bool],
        targets_expired: Dict[int, bool],
        apply_bid_penalty: bool,
        previous_distances: np.ndarray,
        current_distances: np.ndarray
    ) -> Dict[str, float]:
        """Calculate rewards for all agents."""
        rewards = {f"agent_{i}": 0.0 for i in range(self.num_agents)}

        # Bid penalties and rewards (only apply on the first step of a window)
        if apply_bid_penalty:
            # Each agent pays penalty for their own bid and receives reward from all other agents' bids
            for i in range(self.num_agents):
                # Pay for own bid
                rewards[f"agent_{i}"] -= self.bid_penalty * agent_bids[i]
                # Receive reward from other agents' bids (disabled currently)
                # for j in range(self.num_agents):
                #     if i != j:
                #         rewards[f"agent_{i}"] += self.bid_penalty * agent_bids[j]

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
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation as a normalized vector in [0, 1]."""
        # Normalize positions to [0,1]; guard division for grid_size==1
        denom = float(self.grid_size - 1) if self.grid_size > 1 else 1.0
        agent_row_norm = float(self.agent_position[0]) / denom
        agent_col_norm = float(self.agent_position[1]) / denom

        # Build observation: [agent_pos, target_0_pos, ..., target_n_pos, target_0_reached, ..., target_n_reached, target_0_counter, ..., target_n_counter]
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
        # Normalize by target_expiry_steps if set, otherwise by max_steps
        counter_denom = float(self.target_expiry_steps) if self.target_expiry_steps is not None else float(self.max_steps)
        counter_denom = max(counter_denom, 1.0)  # Avoid division by zero
        for i in range(self.num_agents):
            counter_norm = min(float(self.target_step_counters[i]) / counter_denom, 1.0)  # Clamp to [0, 1]
            obs_list.append(counter_norm)

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
        fig, ax = plt.subplots(figsize=(10, 8))

        def animate(frame):
            ax.clear()
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

            ax.set_xlim(-0.5, self.grid_size - 0.5)
            ax.set_ylim(-0.5, self.grid_size - 0.5)
            ax.set_aspect('equal')

            for i in range(self.grid_size + 1):
                ax.axhline(i - 0.5, color='lightgray', linewidth=0.5)
                ax.axvline(i - 0.5, color='lightgray', linewidth=0.5)

            target_colors = ['lightblue', 'lightcoral', 'lightyellow']
            edge_colors = ['blue', 'red', 'orange']

            winning_agent = None
            if frame < len(episode_data.get("step_details", [])):
                step_detail = episode_data["step_details"][frame]
                winning_agent = step_detail.get("winning_agent", -1)

            is_controlling = (winning_agent == target_agent_id)
            edge_width = 4 if is_controlling else 2
            edge_color = 'gold' if is_controlling else edge_colors[target_agent_id % 3]

            if target_reached == 0:
                ax.add_patch(plt.Rectangle(
                    (target_col - 0.4, target_row - 0.4), 0.8, 0.8,
                    facecolor=target_colors[target_agent_id % 3],
                    edgecolor=edge_color, linewidth=edge_width
                ))
                ax.text(target_col, target_row, str(target_agent_id),
                       ha='center', va='center', fontsize=12, fontweight='bold')
                if is_controlling:
                    ax.text(target_col, target_row - 0.6, '⚡',
                           ha='center', va='center', fontsize=10, color='gold')
            else:
                ax.add_patch(plt.Rectangle(
                    (target_col - 0.4, target_row - 0.4), 0.8, 0.8,
                    facecolor='lightgreen', edgecolor='green', linewidth=2
                ))
                ax.text(target_col, target_row, '✓',
                       ha='center', va='center', fontsize=12, fontweight='bold')

            if winning_agent is not None and winning_agent >= 0:
                ring_color = edge_colors[winning_agent % 3] if winning_agent != target_agent_id else 'gold'
                ax.add_patch(plt.Circle((agent_col, agent_row), 0.35,
                                       facecolor='none', edgecolor=ring_color, linewidth=3))

            ax.add_patch(plt.Circle((agent_col, agent_row), 0.3,
                                   facecolor='yellow', edgecolor='orange'))
            ax.text(agent_col, agent_row, 'A',
                   ha='center', va='center', fontsize=10, fontweight='bold')

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

            title = f'Agent {target_agent_id} - Step {frame} - Reward: {reward:.2f}\n'
            title += f'Total Reward: {total_reward:.2f}'

            # Add bid and action information if available
            if frame < len(episode_data.get("actions", [])):
                action = episode_data["actions"][frame]
                if isinstance(action, dict) and f"agent_{target_agent_id}" in action:
                    direction_names = {0: "Left ←", 1: "Right →", 2: "Up ↑", 3: "Down ↓"}
                    agent_action = action[f"agent_{target_agent_id}"]
                    direction = direction_names.get(agent_action["direction"], "?")
                    bid = agent_action["bid"]
                    title += f'\nBid: {bid} | Action: {direction}'

            ax.set_title(title, fontsize=11)
            ax.set_xticks(range(self.grid_size))
            ax.set_yticks(range(self.grid_size))
            ax.invert_yaxis()

        anim = animation.FuncAnimation(fig, animate,
                                      frames=len(episode_data["states"]) + 5,
                                      interval=1000//fps, repeat=True)
        try:
            anim.save(str(output_path), writer='pillow', fps=fps)
            print(f"✅ Episode GIF saved: {output_path}")
        except Exception as e:
            print(f"⚠️  Could not save GIF {output_path}: {e}")
        finally:
            plt.close(fig)

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
        fig, ax = plt.subplots(figsize=(10, 8))

        def animate(frame):
            ax.clear()
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

            ax.set_xlim(-0.5, self.grid_size - 0.5)
            ax.set_ylim(-0.5, self.grid_size - 0.5)
            ax.set_aspect('equal')

            for i in range(self.grid_size + 1):
                ax.axhline(i - 0.5, color='lightgray', linewidth=0.5)
                ax.axvline(i - 0.5, color='lightgray', linewidth=0.5)

            target_colors = ['lightblue', 'lightcoral', 'lightyellow']
            edge_colors = ['blue', 'red', 'orange']
            winning_agent = step_detail.get("winning_agent", -1) if step_detail else None

            for i in range(self.num_agents):
                target_row, target_col = target_positions[i]
                is_controlling = (winning_agent == i)
                edge_width = 4 if is_controlling else 2
                edge_color = 'gold' if is_controlling else edge_colors[i]

                if targets_reached[i] == 0:
                    ax.add_patch(plt.Rectangle(
                        (target_col - 0.4, target_row - 0.4), 0.8, 0.8,
                        facecolor=target_colors[i], edgecolor=edge_color, linewidth=edge_width
                    ))
                    ax.text(target_col, target_row, str(i),
                           ha='center', va='center', fontsize=12, fontweight='bold')
                    if is_controlling:
                        ax.text(target_col, target_row - 0.6, '⚡',
                               ha='center', va='center', fontsize=10, color='gold')
                else:
                    ax.add_patch(plt.Rectangle(
                        (target_col - 0.4, target_row - 0.4), 0.8, 0.8,
                        facecolor='lightgreen', edgecolor='green', linewidth=2
                    ))
                    ax.text(target_col, target_row, '✓',
                           ha='center', va='center', fontsize=12, fontweight='bold')

            if winning_agent is not None and 0 <= winning_agent < self.num_agents:
                ax.add_patch(plt.Circle((agent_col, agent_row), 0.35,
                                       facecolor='none', edgecolor=edge_colors[winning_agent], linewidth=3))

            ax.add_patch(plt.Circle((agent_col, agent_row), 0.3,
                                   facecolor='yellow', edgecolor='orange'))
            ax.text(agent_col, agent_row, 'A',
                   ha='center', va='center', fontsize=10, fontweight='bold')

            title = f'Competition Episode - Step {frame}\n'
            if step_detail:
                targets_status = ", ".join([f"{i}={'✓' if targets_reached[i] else '✗'}" for i in range(self.num_agents)])
                title += f'Targets: {targets_status}\n'
                if rewards:
                    cumulative = {f"agent_{i}": sum(episode_data["rewards"][f].get(f"agent_{i}", 0)
                                                    for f in range(frame + 1)) for i in range(self.num_agents)}
                    rewards_str = ", ".join([f"{i}={cumulative[f'agent_{i}']:.2f}" for i in range(self.num_agents)])
                    title += f'Rewards: {rewards_str}\n'

                # Add bids and actions
                if actions:
                    direction_names = {0: "←", 1: "→", 2: "↑", 3: "↓"}
                    bids_str = ", ".join([f"{i}={actions[f'agent_{i}']['bid']}" for i in range(self.num_agents)])
                    title += f'Bids: {bids_str}\n'
                    actions_str = ", ".join([f"{i}={direction_names.get(actions[f'agent_{i}']['direction'], '?')}"
                                            for i in range(self.num_agents)])
                    title += f'Actions: {actions_str}'

            ax.set_title(title, fontsize=10)
            ax.set_xticks(range(self.grid_size))
            ax.set_yticks(range(self.grid_size))
            ax.invert_yaxis()

        anim = animation.FuncAnimation(fig, animate,
                                      frames=len(episode_data["states"]) + 3,
                                      interval=1000//fps, repeat=True)
        try:
            anim.save(str(output_path), writer='pillow', fps=fps)
            print(f"✅ Competition GIF saved: {output_path}")
        except Exception as e:
            print(f"⚠️  Could not save GIF {output_path}: {e}")
        finally:
            plt.close(fig)

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
        target_move_interval: int = 1,
        max_steps: int = 100,
        action_window: int = 1,
        distance_reward_scale: float = 0.0,
        target_expiry_steps: Optional[int] = None,
        target_expiry_penalty: float = 5.0,
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
            distance_reward_scale: Reward scaling for distance improvements (default: 0.0, disabled)
                                  Positive values reward getting closer to target
            target_expiry_steps: Maximum steps allowed before target expiry penalty (default: None, disabled)
                                If set, agents receive penalty if target not reached within this many steps
            target_expiry_penalty: Penalty for not reaching target within expiry_steps (default: 5.0)
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

        # Save winning_agent and bids from parent step
        winning_agent = info.get("winning_agent", -1)
        bids = info.get("bids", {})

        # Move targets after agent has moved
        self._move_targets()

        # Update observation and info with new target positions
        obs = self._get_observation()
        info = self._get_info()

        # Restore winning_agent and bids from parent step
        info["winning_agent"] = winning_agent
        info["bids"] = bids

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