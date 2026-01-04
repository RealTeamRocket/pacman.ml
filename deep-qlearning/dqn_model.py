"""
Deep Q-Network (DQN) Model with Dueling Architecture

This module implements the neural network architecture for the DQN agent.
Uses a Dueling DQN architecture which separates state value and action advantages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture.
    
    Separates the Q-value into:
    - V(s): Value of being in state s
    - A(s, a): Advantage of taking action a in state s
    
    Q(s, a) = V(s) + A(s, a) - mean(A(s, :))
    
    This helps the network learn which states are valuable
    regardless of the action taken.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize the Dueling DQN.
        
        Args:
            state_dim: Dimension of the input state
            action_dim: Number of possible actions
            hidden_dim: Size of hidden layers
        """
        super(DuelingDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Value stream - estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream - estimates A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor of shape (batch_size, state_dim)
        
        Returns:
            Q-values for each action, shape (batch_size, action_dim)
        """
        features = self.feature_layer(state)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages using the dueling formula
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_action(self, state: torch.Tensor, action_mask: torch.Tensor = None) -> int:
        """
        Get the best action for a given state.
        
        Args:
            state: Input state tensor of shape (state_dim,) or (1, state_dim)
            action_mask: Optional mask of valid actions (1 = valid, 0 = invalid)
        
        Returns:
            Action index with highest Q-value
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.forward(state)
            
            # Apply action mask if provided
            if action_mask is not None:
                if action_mask.dim() == 1:
                    action_mask = action_mask.unsqueeze(0)
                # Set invalid actions to very negative Q-value
                q_values = q_values.masked_fill(action_mask == 0, -1e9)
            
            return q_values.argmax(dim=1).item()


class SimpleDQN(nn.Module):
    """
    Simple DQN architecture without dueling.
    
    A straightforward feedforward network that maps states to Q-values.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize the Simple DQN.
        
        Args:
            state_dim: Dimension of the input state
            action_dim: Number of possible actions
            hidden_dim: Size of hidden layers
        """
        super(SimpleDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor of shape (batch_size, state_dim)
        
        Returns:
            Q-values for each action, shape (batch_size, action_dim)
        """
        return self.network(state)
    
    def get_action(self, state: torch.Tensor, action_mask: torch.Tensor = None) -> int:
        """
        Get the best action for a given state.
        
        Args:
            state: Input state tensor of shape (state_dim,) or (1, state_dim)
            action_mask: Optional mask of valid actions (1 = valid, 0 = invalid)
        
        Returns:
            Action index with highest Q-value
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.forward(state)
            
            # Apply action mask if provided
            if action_mask is not None:
                if action_mask.dim() == 1:
                    action_mask = action_mask.unsqueeze(0)
                q_values = q_values.masked_fill(action_mask == 0, -1e9)
            
            return q_values.argmax(dim=1).item()


def create_model(state_dim: int, action_dim: int, hidden_dim: int = 256, 
                 dueling: bool = True) -> nn.Module:
    """
    Factory function to create a DQN model.
    
    Args:
        state_dim: Dimension of the input state
        action_dim: Number of possible actions
        hidden_dim: Size of hidden layers
        dueling: Whether to use dueling architecture
    
    Returns:
        DQN model instance
    """
    if dueling:
        return DuelingDQN(state_dim, action_dim, hidden_dim)
    else:
        return SimpleDQN(state_dim, action_dim, hidden_dim)