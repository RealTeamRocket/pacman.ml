"""
Deep Q-Learning Agent with Double DQN

This module implements the DQN agent with:
- Double DQN to reduce Q-value overestimation
- Target network for stable learning
- Epsilon-greedy exploration with decay
- Experience replay integration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from typing import Optional, Tuple, Dict, Any

from dqn_model import create_model, DuelingDQN
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent:
    """
    Deep Q-Learning Agent with Double DQN.
    
    Double DQN uses the online network to select actions but the
    target network to evaluate them, reducing overestimation bias.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        target_update_freq: int = 1000,
        buffer_size: int = 100000,
        batch_size: int = 64,
        dueling: bool = True,
        prioritized_replay: bool = False,
        device: str = None
    ):
        """
        Initialize the DQN Agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Number of possible actions
            hidden_dim: Size of hidden layers in the network
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Multiplicative decay per step
            target_update_freq: Steps between target network updates
            buffer_size: Size of experience replay buffer
            batch_size: Mini-batch size for training
            dueling: Use dueling network architecture
            prioritized_replay: Use prioritized experience replay
            device: Device to use (cuda/cpu/mps)
        """
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"[DQN Agent] Using device: {self.device}")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Epsilon for exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Create online and target networks
        self.online_net = create_model(state_dim, action_dim, hidden_dim, dueling).to(self.device)
        self.target_net = create_model(state_dim, action_dim, hidden_dim, dueling).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Target network is never trained directly
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        
        # Learning rate scheduler - VERY gentle, only reduce after long plateau
        # Factor 0.7 (not 0.5) and patience 200 (not 50) to avoid killing learning
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=200, 
            min_lr=1e-5  # Don't go below 1e-5 (was 1e-6 which is too low)
        )
        self.recent_losses = []  # Track recent losses for scheduler
        
        # Loss function
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for stability
        
        # Replay buffer
        self.prioritized_replay = prioritized_replay
        if prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training stats
        self.steps_done = 0
        self.updates_done = 0
        self.total_loss = 0
        self.loss_count = 0
    
    def select_action(self, state: np.ndarray, action_mask: np.ndarray = None, 
                      training: bool = True, dot_bias_strength: float = 1.0) -> int:
        """
        Select an action using epsilon-greedy policy with dot-hunting bias.
        
        Args:
            state: Current state as numpy array
            action_mask: Optional mask of valid actions
            training: If True, use epsilon-greedy; if False, be greedy
            dot_bias_strength: How much to bias Q-values toward dot-rich directions
                              (higher = more strongly follow dots)
        
        Returns:
            Selected action index
        """
        # Extract dot-per-direction features from state (indices -9 to -5)
        # These are normalized: what fraction of reachable dots are in each direction
        dots_per_dir = state[-9:-5]  # [up, down, left, right] normalized
        
        # Epsilon-greedy exploration - but bias toward dot-rich directions
        if training and np.random.random() < self.epsilon:
            # Biased random action: more likely to pick directions with more dots
            if action_mask is not None:
                valid_actions = np.where(action_mask > 0)[0]
                if len(valid_actions) > 0:
                    # Weight by dot density - STRONGLY prefer directions with dots
                    dot_weights = np.array([dots_per_dir[a] for a in valid_actions])
                    # Square the weights to make differences more pronounced
                    dot_weights = dot_weights ** 2
                    # Add tiny constant to avoid zero weights, but keep near-zero for no-dot dirs
                    dot_weights = dot_weights + 0.01
                    # Normalize to probabilities
                    probs = dot_weights / dot_weights.sum()
                    return np.random.choice(valid_actions, p=probs)
            return np.random.randint(self.action_dim)
        
        # Greedy action from network with dot bias
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.online_net(state_tensor).squeeze(0)  # [4]
            
            # Add dot bias to Q-values: boost directions with more dots
            # AND penalize directions with NO dots (like going to the middle)
            dot_bias = torch.tensor(dots_per_dir, dtype=torch.float32).to(self.device)
            
            # Strong penalty for zero-dot directions (e.g., going to middle at start)
            # If a direction has 0% of dots, it gets a -10 penalty
            zero_dot_penalty = torch.where(
                dot_bias < 0.01,  # If less than 1% of dots in this direction
                torch.tensor(-10.0, device=self.device),  # Strong penalty
                torch.tensor(0.0, device=self.device)
            )
            
            # Apply positive bias for dot-rich directions AND penalty for empty directions
            q_values = q_values + dot_bias * dot_bias_strength + zero_dot_penalty
            
            # Apply action mask
            if action_mask is not None:
                mask_tensor = torch.tensor(action_mask, dtype=torch.float32).to(self.device)
                # Set invalid actions to very negative value
                q_values = q_values + (mask_tensor - 1) * 1e9
            
            return q_values.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """
        Store an experience in the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step (sample batch and update network).
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # Sample from replay buffer
        if self.prioritized_replay:
            states, actions, rewards, next_states, dones, indices, weights = \
                self.replay_buffer.sample(self.batch_size)
            weights = weights.to(self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
            weights = None
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Clip rewards to prevent exploding gradients
        rewards = torch.clamp(rewards, -100, 100)
        
        # Current Q-values
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: Use online network to select actions, target network to evaluate
        with torch.no_grad():
            # Online network selects best actions for next states
            next_actions = self.online_net(next_states).argmax(dim=1)
            # Target network evaluates those actions
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # Compute target Q-values
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss
        if weights is not None:
            # Weighted loss for prioritized replay
            td_errors = (current_q - target_q).detach().cpu().numpy()
            loss = (weights * (current_q - target_q).pow(2)).mean()
            # Update priorities
            self.replay_buffer.update_priorities(indices, td_errors)
        else:
            loss = self.loss_fn(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Tighter gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Track stats
        self.updates_done += 1
        self.total_loss += loss.item()
        self.loss_count += 1
        
        # Track recent losses for LR scheduler
        self.recent_losses.append(loss.item())
        if len(self.recent_losses) > 100:
            self.recent_losses.pop(0)
        
        # Update learning rate based on recent loss (every 500 updates, not 100)
        # Less frequent updates = more stable learning
        if self.updates_done % 500 == 0 and len(self.recent_losses) >= 100:
            avg_loss = sum(self.recent_losses) / len(self.recent_losses)
            self.scheduler.step(avg_loss)
        
        # Update target network
        if self.updates_done % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy online network weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps_done += 1
    
    def get_average_loss(self) -> float:
        """Get average loss since last call."""
        if self.loss_count == 0:
            return 0.0
        avg_loss = self.total_loss / self.loss_count
        self.total_loss = 0
        self.loss_count = 0
        return avg_loss
    
    def save(self, filepath: str):
        """
        Save agent state to file.
        
        Args:
            filepath: Path to save file
        """
        torch.save({
            'online_net_state_dict': self.online_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'updates_done': self.updates_done,
        }, filepath)
        print(f"[SAVE] Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load agent state from file.
        
        Args:
            filepath: Path to load file
        """
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.online_net.load_state_dict(checkpoint['online_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.steps_done = checkpoint.get('steps_done', 0)
            self.updates_done = checkpoint.get('updates_done', 0)
            print(f"[LOAD] Model loaded from {filepath}, ε={self.epsilon:.3f}, steps={self.steps_done}")
        else:
            print(f"[LOAD] No checkpoint found at {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        current_lr = self.optimizer.param_groups[0]['lr']
        return {
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'updates_done': self.updates_done,
            'buffer_size': len(self.replay_buffer),
            'device': str(self.device),
            'learning_rate': current_lr
        }
    
    def print_stats(self):
        """Print agent statistics."""
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print(f"DQN AGENT STATS")
        print(f"{'='*60}")
        print(f"Device: {stats['device']}")
        print(f"Epsilon: {stats['epsilon']:.4f}")
        print(f"Learning Rate: {stats['learning_rate']:.2e}")
        print(f"Steps: {stats['steps_done']}")
        print(f"Network Updates: {stats['updates_done']}")
        print(f"Buffer Size: {stats['buffer_size']}")
        print(f"{'='*60}\n")