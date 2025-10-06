import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, Type, Union
import scipy.optimize
import warnings

from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.buffers import ReplayBuffer
import gymnasium as gym


class ZeroSumQNetwork(QNetwork):
    """
    Q-Network for zero-sum matrix games.
    
    Outputs a flattened payoff matrix that can be reshaped to 
    [protagonist_actions x adversary_actions] for Nash equilibrium solving.
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        features_extractor: nn.Module,
        features_dim: int,
        protagonist_actions: int,
        adversary_actions: int,
        net_arch: Optional[list] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        # Initialize parent with total matrix size as action space
        matrix_size = protagonist_actions * adversary_actions
        super().__init__(
            observation_space,
            action_space,
            features_extractor,
            features_dim,
            net_arch,
            activation_fn,
        )
        
        self.protagonist_actions = protagonist_actions
        self.adversary_actions = adversary_actions
        self.matrix_size = matrix_size
        
        # Override the final layer to output payoff matrix
        self.q_net = nn.Sequential(
            *[layer for layer in self.q_net[:-1]],  # All layers except last
            nn.Linear(self.q_net[-1].in_features, matrix_size)  # New output layer
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning flattened payoff matrix.
        
        Args:
            obs: Batch of observations [batch_size, obs_dim]
            
        Returns:
            Flattened payoff matrices [batch_size, protagonist_actions * adversary_actions]
        """
        features = self.features_extractor(obs)
        return self.q_net(features)
    
    def get_payoff_matrix(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get payoff matrix reshaped to proper dimensions.
        
        Args:
            obs: Batch of observations [batch_size, obs_dim]
            
        Returns:
            Payoff matrices [batch_size, protagonist_actions, adversary_actions]
        """
        q_flat = self.forward(obs)
        return q_flat.view(-1, self.protagonist_actions, self.adversary_actions)


class ZeroSumDQNPolicy(DQNPolicy):
    """
    Policy for Zero-Sum DQN that handles matrix game action selection.
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule,
        protagonist_actions: int,
        adversary_actions: int,
        net_arch: Optional[list] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        **kwargs,
    ):
        self.protagonist_actions = protagonist_actions
        self.adversary_actions = adversary_actions
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            **kwargs,
        )
    
    def make_q_net(self) -> ZeroSumQNetwork:
        """Create the Q-network."""
        # Create features extractor
        features_extractor = self.make_features_extractor()
        
        return ZeroSumQNetwork(
            self.observation_space,
            self.action_space,
            features_extractor=features_extractor,
            features_dim=features_extractor.features_dim,
            protagonist_actions=self.protagonist_actions,
            adversary_actions=self.adversary_actions,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
        ).to(self.device)
    
    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Predict joint actions using Nash equilibrium.

        Args:
            observation: Batch of observations
            deterministic: Unused (kept for API compatibility)

        Returns:
            Joint action indices [batch_size]
        """
        with torch.no_grad():
            payoff_matrices = self.q_net.get_payoff_matrix(observation)
            batch_size = payoff_matrices.shape[0]
            joint_actions = []
            
            for i in range(batch_size):
                payoff_matrix = payoff_matrices[i].cpu().numpy()
                prot_strategy, adv_strategy, _ = self._solve_matrix_game(payoff_matrix)
                
                # Sample from Nash equilibrium strategies
                prot_action = np.random.choice(self.protagonist_actions, p=prot_strategy)
                adv_action = np.random.choice(self.adversary_actions, p=adv_strategy)
                
                # Convert joint action to single index
                joint_action_idx = prot_action * self.adversary_actions + adv_action
                joint_actions.append(joint_action_idx)
            
            return torch.tensor(joint_actions, device=self.device)
    
    def _solve_matrix_game(self, payoff_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve zero-sum matrix game to find Nash equilibrium.
        Reuses logic from ZeroSumQLearning.
        """
        try:
            m, n = payoff_matrix.shape
            
            # Solve protagonist's problem
            c = np.zeros(m + 1)
            c[-1] = -1  # Maximize v
            
            A_ub = np.zeros((n, m + 1))
            for j in range(n):
                A_ub[j, :m] = -payoff_matrix[:, j]
                A_ub[j, -1] = 1
            b_ub = np.zeros(n)
            
            A_eq = np.zeros((1, m + 1))
            A_eq[0, :m] = 1
            b_eq = np.array([1])
            
            bounds = [(0, None)] * m + [(None, None)]
            
            result = scipy.optimize.linprog(
                c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                bounds=bounds, method='highs'
            )
            
            if result.success:
                protagonist_strategy = result.x[:m]
                game_value = result.x[-1]
                adversary_strategy = self._solve_adversary_fallback(payoff_matrix)
                
                protagonist_strategy = self._normalize_strategy(protagonist_strategy)
                adversary_strategy = self._normalize_strategy(adversary_strategy)
                
                return protagonist_strategy, adversary_strategy, game_value
            else:
                # Fallback to uniform strategies
                return (np.ones(m) / m, np.ones(n) / n, 0.0)
                
        except Exception as e:
            warnings.warn(f"Matrix game solving failed: {e}")
            m, n = payoff_matrix.shape
            return (np.ones(m) / m, np.ones(n) / n, 0.0)
    
    def _solve_adversary_fallback(self, payoff_matrix: np.ndarray) -> np.ndarray:
        """Solve for adversary's optimal strategy."""
        try:
            m, n = payoff_matrix.shape
            
            c = np.zeros(n + 1)
            c[-1] = 1  # Minimize u
            
            A_ub = np.zeros((m, n + 1))
            for i in range(m):
                A_ub[i, :n] = payoff_matrix[i, :]
                A_ub[i, -1] = -1
            b_ub = np.zeros(m)
            
            A_eq = np.zeros((1, n + 1))
            A_eq[0, :n] = 1
            b_eq = np.array([1])
            
            bounds = [(0, None)] * n + [(None, None)]
            
            result = scipy.optimize.linprog(
                c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method='highs'
            )
            
            if result.success:
                return self._normalize_strategy(result.x[:n])
            else:
                return np.ones(n) / n
                
        except Exception:
            return np.ones(n) / n
    
    def _normalize_strategy(self, strategy: np.ndarray) -> np.ndarray:
        """Normalize strategy to valid probability distribution."""
        strategy = np.maximum(strategy, 0.0)
        
        if np.sum(strategy) == 0:
            return np.ones(len(strategy)) / len(strategy)
        
        strategy = strategy / np.sum(strategy)
        
        return strategy


class ZeroSumDQN(DQN):
    """
    Deep Q-Network for zero-sum matrix games.
    
    Extends stable-baselines3 DQN to handle competitive bidding scenarios
    where actions are selected via Nash equilibrium rather than epsilon-greedy.
    """
    
    def __init__(
        self,
        policy: Union[str, Type[ZeroSumDQNPolicy]],
        env: Union[GymEnv, str],
        protagonist_actions: int,
        adversary_actions: int,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.protagonist_actions = protagonist_actions
        self.adversary_actions = adversary_actions
        
        # Update policy kwargs to include action space dimensions
        if policy_kwargs is None:
            policy_kwargs = {}
        policy_kwargs.update({
            "protagonist_actions": protagonist_actions,
            "adversary_actions": adversary_actions,
        })

        super().__init__(
            policy=policy,
            env=env,
            policy_kwargs=policy_kwargs,
            **kwargs,
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Train the model with custom loss for matrix games.
        """
        # Switch to train mode
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                # Compute the next Q-values using the target network
                target_payoff_matrices = self.q_net_target.get_payoff_matrix(replay_data.next_observations)
                target_values = []
                
                # Solve Nash equilibrium for each next state
                for i in range(batch_size):
                    payoff_matrix = target_payoff_matrices[i].cpu().numpy()
                    _, _, game_value = self.policy._solve_matrix_game(payoff_matrix)
                    target_values.append(game_value)
                
                target_values = torch.tensor(target_values, device=self.device).unsqueeze(1)
                
                # Compute the target Q values
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_values

            # Get current Q-values estimates
            current_payoff_matrices = self.q_net.get_payoff_matrix(replay_data.observations)
            
            # Extract Q-values for the taken joint actions
            current_q_values = []
            for i in range(batch_size):
                action_idx = replay_data.actions[i].item()
                prot_idx = action_idx // self.adversary_actions
                adv_idx = action_idx % self.adversary_actions
                q_val = current_payoff_matrices[i, prot_idx, adv_idx]
                current_q_values.append(q_val)
            
            current_q_values = torch.stack(current_q_values).unsqueeze(1)

            # Compute Huber loss
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
    
    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Predict action using Nash equilibrium instead of epsilon-greedy.
        """
        # Convert to tensor if needed
        observation_tensor = self.policy.obs_to_tensor(observation)[0]

        with torch.no_grad():
            # Get payoff matrix
            payoff_matrix = self.q_net.get_payoff_matrix(observation_tensor.unsqueeze(0))[0]
            payoff_matrix_np = payoff_matrix.cpu().numpy()

            # Solve Nash equilibrium
            prot_strategy, adv_strategy, _ = self.policy._solve_matrix_game(payoff_matrix_np)

            # Sample actions from mixed strategies
            prot_action = np.random.choice(self.protagonist_actions, p=prot_strategy)
            adv_action = np.random.choice(self.adversary_actions, p=adv_strategy)

            # Convert to joint action index
            joint_action_idx = prot_action * self.adversary_actions + adv_action

            return np.array([joint_action_idx]), state
    
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """Get constructor parameters for saving/loading."""
        data = super()._get_constructor_parameters()
        data.update({
            "protagonist_actions": self.protagonist_actions,
            "adversary_actions": self.adversary_actions,
        })
        return data

