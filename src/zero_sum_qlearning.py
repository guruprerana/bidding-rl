import numpy as np
from typing import Dict, Tuple, Optional, Any
import scipy.optimize
from collections import defaultdict
import pickle


class ZeroSumQLearning:
    """
    Q-learning algorithm for two-player zero-sum Markov games.
    
    This implements Nash-Q learning where:
    1. We maintain ONE Q-table storing protagonist's payoffs (adversary gets -payoff)
    2. Q-value updates require solving a matrix game at each state
    3. The value of the next state is determined by the Nash equilibrium
    
    In zero-sum games, we only need one payoff matrix since:
    - Protagonist's payoff: Q[s][a_p][a_a] 
    - Adversary's payoff: -Q[s][a_p][a_a]
    """
    
    def __init__(
        self,
        protagonist_action_space_size: int,
        adversary_action_space_size: int,
        observation_space_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        """
        Initialize the Zero-Sum Q-Learning algorithm.
        
        Args:
            protagonist_action_space_size: Number of actions for protagonist
            adversary_action_space_size: Number of actions for adversary
            observation_space_size: Size of observation space
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            epsilon: Initial epsilon for epsilon-greedy exploration
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        self.protagonist_actions = protagonist_action_space_size
        self.adversary_actions = adversary_action_space_size
        self.observation_space_size = observation_space_size
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Single Q-table for zero-sum game: Q[state][protagonist_action][adversary_action] 
        # Stores protagonist's expected payoff; adversary's payoff is implicitly -Q[s][a_p][a_a]
        self.q_table = defaultdict(lambda: np.zeros((self.protagonist_actions, self.adversary_actions)))
        
        # Value function: V[state] = Nash equilibrium value for protagonist
        self.value_function = defaultdict(float)
        
        # Mixed strategies from Nash equilibrium for both players
        self.protagonist_policy = defaultdict(lambda: np.ones(self.protagonist_actions) / self.protagonist_actions)
        self.adversary_policy = defaultdict(lambda: np.ones(self.adversary_actions) / self.adversary_actions)
        
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        
    def _state_to_key(self, observation: Dict) -> str:
        """Convert observation to a hashable key for Q-table indexing."""
        # Flatten the observation dictionary into a tuple
        items = []
        for key in sorted(observation.keys()):
            value = observation[key]
            if isinstance(value, np.ndarray):
                items.append(tuple(value.flatten()))
            else:
                items.append(value)
        return str(tuple(items))
    
    def _action_to_index(self, action: Dict) -> Tuple[int, int]:
        """Convert action dictionary to action indices."""
        # Assuming actions are structured as: {"direction": int, "bid": int}
        # Convert to single index: direction * (bid_upper_bound + 1) + bid
        protagonist_action = action["protagonist"]
        adversary_action = action["adversary"]
        
        # Convert 2D action (direction, bid) to 1D index
        prot_idx = protagonist_action["direction"] * (max(self.protagonist_actions // 4, 1)) + protagonist_action["bid"]
        adv_idx = adversary_action["direction"] * (max(self.adversary_actions // 4, 1)) + adversary_action["bid"]
        
        return prot_idx, adv_idx
    
    def _index_to_action(self, protagonist_idx: int, adversary_idx: int, bid_upper_bound: int) -> Dict:
        """Convert action indices back to action dictionary."""
        # Reverse the conversion from _action_to_index
        directions_per_bid = bid_upper_bound + 1
        
        prot_direction = protagonist_idx // directions_per_bid
        prot_bid = protagonist_idx % directions_per_bid
        
        adv_direction = adversary_idx // directions_per_bid
        adv_bid = adversary_idx % directions_per_bid
        
        return {
            "protagonist": {"direction": prot_direction, "bid": prot_bid},
            "adversary": {"direction": adv_direction, "bid": adv_bid}
        }
    
    def _solve_matrix_game(self, payoff_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve the zero-sum matrix game to find Nash equilibrium.
        
        In zero-sum games, we only need to solve ONE linear program. The protagonist's
        strategy comes from the primal solution, and the adversary's strategy comes
        from the dual solution of the same LP.
        
        Args:
            payoff_matrix: Matrix of shape (protagonist_actions, adversary_actions)
                          representing protagonist's payoffs (adversary gets -payoff_matrix)
        
        Returns:
            protagonist_strategy: Mixed strategy for protagonist (row player)
            adversary_strategy: Mixed strategy for adversary (column player)
            game_value: Value of the game for protagonist (adversary gets -game_value)
        """
        try:
            m, n = payoff_matrix.shape
            
            # Solve the protagonist's problem: max v subject to:
            # sum_i x_i * payoff_matrix[i,j] >= v for all j
            # sum_i x_i = 1
            # x_i >= 0
            
            c = np.zeros(m + 1)
            c[-1] = -1  # Maximize v (minimize -v)
            
            # Constraints: sum_i x_i * payoff_matrix[i,j] >= v for all j
            # Rewritten as: -sum_i x_i * payoff_matrix[i,j] + v <= 0
            A_ub = np.zeros((n, m + 1))
            for j in range(n):
                A_ub[j, :m] = -payoff_matrix[:, j]
                A_ub[j, -1] = 1
            b_ub = np.zeros(n)
            
            # Equality constraint: sum_i x_i = 1
            A_eq = np.zeros((1, m + 1))
            A_eq[0, :m] = 1
            b_eq = np.array([1])
            
            # Bounds: x_i >= 0, v unbounded
            bounds = [(0, None)] * m + [(None, None)]
            
            result = scipy.optimize.linprog(
                c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                bounds=bounds, method='highs'
            )
            
            if result.success:
                protagonist_strategy = result.x[:m]
                game_value = result.x[-1]
                
                # Extract adversary's strategy from dual variables
                # The dual variables correspond to the inequality constraints
                if hasattr(result, 'ineqlin') and result.ineqlin is not None:
                    # Dual variables from inequality constraints
                    adversary_strategy = result.ineqlin.marginals
                    # Normalize to ensure it's a valid probability distribution
                    if np.sum(adversary_strategy) > 0:
                        adversary_strategy = adversary_strategy / np.sum(adversary_strategy)
                    else:
                        adversary_strategy = np.ones(n) / n
                else:
                    # Fallback: solve adversary's problem if dual not available
                    adversary_strategy = self._solve_adversary_fallback(payoff_matrix)
                    
            else:
                # Fallback to uniform strategies
                protagonist_strategy = np.ones(m) / m
                adversary_strategy = np.ones(n) / n
                game_value = 0.0
                
            return protagonist_strategy, adversary_strategy, game_value
            
        except Exception as e:
            print(f"Warning: Matrix game solving failed: {e}")
            # Fallback to uniform strategies
            m, n = payoff_matrix.shape
            protagonist_strategy = np.ones(m) / m
            adversary_strategy = np.ones(n) / n
            game_value = 0.0
            return protagonist_strategy, adversary_strategy, game_value
    
    def _solve_adversary_fallback(self, payoff_matrix: np.ndarray) -> np.ndarray:
        """
        Fallback method to solve for adversary's strategy if dual solution unavailable.
        
        Args:
            payoff_matrix: Protagonist's payoff matrix
            
        Returns:
            adversary_strategy: Optimal mixed strategy for adversary
        """
        try:
            m, n = payoff_matrix.shape
            
            # Adversary minimizes protagonist's payoff: min u subject to:
            # sum_j y_j * payoff_matrix[i,j] <= u for all i
            # sum_j y_j = 1
            # y_j >= 0
            
            c = np.zeros(n + 1)
            c[-1] = 1  # Minimize u
            
            # Constraints: sum_j y_j * payoff_matrix[i,j] <= u for all i
            A_ub = np.zeros((m, n + 1))
            for i in range(m):
                A_ub[i, :n] = payoff_matrix[i, :]
                A_ub[i, -1] = -1
            b_ub = np.zeros(m)
            
            # Equality constraint: sum_j y_j = 1
            A_eq = np.zeros((1, n + 1))
            A_eq[0, :n] = 1
            b_eq = np.array([1])
            
            # Bounds: y_j >= 0, u unbounded
            bounds = [(0, None)] * n + [(None, None)]
            
            result = scipy.optimize.linprog(
                c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method='highs'
            )
            
            if result.success:
                return result.x[:n]
            else:
                return np.ones(n) / n
                
        except Exception:
            return np.ones(n) / n
    
    def get_action(self, observation: Dict, bid_upper_bound: int, training: bool = True) -> Dict:
        """
        Get action using epsilon-greedy policy based on Nash equilibrium.
        
        Args:
            observation: Current state observation
            bid_upper_bound: Maximum bid value (needed for action conversion)
            training: Whether in training mode (affects exploration)
        
        Returns:
            Action dictionary with protagonist and adversary actions
        """
        state_key = self._state_to_key(observation)
        
        if training and np.random.random() < self.epsilon:
            # Random exploration
            prot_action = np.random.randint(self.protagonist_actions)
            adv_action = np.random.randint(self.adversary_actions)
        else:
            # Use mixed strategy from Nash equilibrium
            protagonist_strategy = self.protagonist_policy[state_key]
            adversary_strategy = self.adversary_policy[state_key]
            
            prot_action = np.random.choice(self.protagonist_actions, p=protagonist_strategy)
            adv_action = np.random.choice(self.adversary_actions, p=adversary_strategy)
        
        return self._index_to_action(prot_action, adv_action, bid_upper_bound)
    
    def update(
        self, 
        observation: Dict, 
        action: Dict, 
        reward: float, 
        next_observation: Dict, 
        terminated: bool
    ) -> None:
        """
        Update Q-values using the zero-sum Q-learning update rule.
        
        Args:
            observation: Current state
            action: Action taken (joint action of both players)
            reward: Reward received by protagonist (adversary gets -reward)
            next_observation: Next state
            terminated: Whether episode terminated
        """
        state_key = self._state_to_key(observation)
        next_state_key = self._state_to_key(next_observation)
        
        prot_action_idx, adv_action_idx = self._action_to_index(action)
        
        # Current Q-value
        current_q = self.q_table[state_key][prot_action_idx, adv_action_idx]
        
        # Compute next state value by solving matrix game
        if terminated:
            next_value = 0.0
        else:
            next_payoff_matrix = self.q_table[next_state_key]
            prot_strategy, adv_strategy, game_value = self._solve_matrix_game(next_payoff_matrix)
            next_value = game_value
            
            # Update policies for next state
            self.protagonist_policy[next_state_key] = prot_strategy
            self.adversary_policy[next_state_key] = adv_strategy
            self.value_function[next_state_key] = game_value
        
        # Q-learning update
        target = reward + self.discount_factor * next_value
        new_q = current_q + self.learning_rate * (target - current_q)
        self.q_table[state_key][prot_action_idx, adv_action_idx] = new_q
        
        # Update current state's Nash equilibrium
        current_payoff_matrix = self.q_table[state_key]
        prot_strategy, adv_strategy, game_value = self._solve_matrix_game(current_payoff_matrix)
        self.protagonist_policy[state_key] = prot_strategy
        self.adversary_policy[state_key] = adv_strategy
        self.value_function[state_key] = game_value
        
        self.step_count += 1
    
    def end_episode(self) -> None:
        """Called at the end of each episode to update exploration parameters."""
        self.episode_count += 1
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
    
    def save_model(self, filepath: str) -> None:
        """Save the learned Q-table and policies."""
        model_data = {
            'q_table': dict(self.q_table),
            'value_function': dict(self.value_function),
            'protagonist_policy': dict(self.protagonist_policy),
            'adversary_policy': dict(self.adversary_policy),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'hyperparameters': {
                'protagonist_actions': self.protagonist_actions,
                'adversary_actions': self.adversary_actions,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a previously saved Q-table and policies."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: np.zeros((self.protagonist_actions, self.adversary_actions)))
        self.q_table.update(model_data['q_table'])
        
        self.value_function = defaultdict(float)
        self.value_function.update(model_data['value_function'])
        
        self.protagonist_policy = defaultdict(lambda: np.ones(self.protagonist_actions) / self.protagonist_actions)
        self.protagonist_policy.update(model_data['protagonist_policy'])
        
        self.adversary_policy = defaultdict(lambda: np.ones(self.adversary_actions) / self.adversary_actions)
        self.adversary_policy.update(model_data['adversary_policy'])
        
        self.epsilon = model_data['epsilon']
        self.episode_count = model_data['episode_count']
        self.step_count = model_data['step_count']
        
        print(f"Model loaded from {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'average_q_value': np.mean([np.mean(q_matrix) for q_matrix in self.q_table.values()]) if self.q_table else 0.0
        }
    
    def get_adversary_payoff_matrix(self, observation: Dict) -> np.ndarray:
        """
        Get the adversary's payoff matrix for a given state.
        In zero-sum games, this is simply the negative of the protagonist's payoffs.
        
        Args:
            observation: State observation
            
        Returns:
            Adversary's payoff matrix (negative of protagonist's)
        """
        state_key = self._state_to_key(observation)
        protagonist_payoffs = self.q_table[state_key]
        return -protagonist_payoffs


if __name__ == "__main__":
    # Example usage
    print("Testing ZeroSumQLearning")
    
    # Example parameters for BiddingGridworld with 4 directions and 3 bid levels
    protagonist_actions = 4 * 4  # 4 directions * 4 bids (0-3)
    adversary_actions = 4 * 4    # Same for adversary
    observation_space_size = 100  # Example: 10x10 grid = 100 positions
    
    learner = ZeroSumQLearning(
        protagonist_action_space_size=protagonist_actions,
        adversary_action_space_size=adversary_actions,
        observation_space_size=observation_space_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.3
    )
    
    # Example observation and action
    obs = {"agent_position": 5, "target_reached": 0, "target_position": 99}
    action = {
        "protagonist": {"direction": 1, "bid": 2},
        "adversary": {"direction": 0, "bid": 1}
    }
    
    # Test getting action
    selected_action = learner.get_action(obs, bid_upper_bound=3)
    print("Selected action:", selected_action)
    
    # Test update
    reward = 1.0
    next_obs = {"agent_position": 6, "target_reached": 0, "target_position": 99}
    learner.update(obs, action, reward, next_obs, terminated=False)
    
    print("Stats:", learner.get_stats())
    print("✅ ZeroSumQLearning test completed!")
