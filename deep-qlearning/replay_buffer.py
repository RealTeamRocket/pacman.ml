"""
Replay Buffer for Deep Q-Learning

Experience replay is a key component of DQN that:
1. Breaks correlation between consecutive samples
2. Allows reuse of past experiences
3. Stabilizes training

This module implements both a standard replay buffer and a
prioritized replay buffer for improved sample efficiency.
"""

import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional
import torch

# Named tuple for storing experiences
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Standard Experience Replay Buffer.
    
    Stores transitions and samples random mini-batches for training.
    Uses a circular buffer (deque) for memory efficiency.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode ended
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = torch.tensor(
            np.array([e.state for e in experiences]), 
            dtype=torch.float32
        )
        actions = torch.tensor(
            [e.action for e in experiences], 
            dtype=torch.long
        )
        rewards = torch.tensor(
            [e.reward for e in experiences], 
            dtype=torch.float32
        )
        next_states = torch.tensor(
            np.array([e.next_state for e in experiences]), 
            dtype=torch.float32
        )
        dones = torch.tensor(
            [e.done for e in experiences], 
            dtype=torch.float32
        )
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch."""
        return len(self.buffer) >= batch_size


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    
    Samples experiences based on their TD-error priority.
    Experiences with higher TD-error (more surprising) are sampled more often.
    
    Uses a sum-tree data structure for efficient O(log n) sampling.
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, 
                 beta_start: float = 0.4, beta_frames: int = 100000):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames to anneal beta to 1.0
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        # Sum tree for efficient priority sampling
        self.tree_size = 1
        while self.tree_size < capacity:
            self.tree_size *= 2
        
        self.sum_tree = np.zeros(2 * self.tree_size)
        self.min_tree = np.full(2 * self.tree_size, float('inf'))
        self.data = [None] * self.tree_size
        
        self.position = 0
        self.size = 0
        self.max_priority = 1.0
    
    def _update_tree(self, idx: int, priority: float):
        """Update the sum tree and min tree at given index."""
        tree_idx = idx + self.tree_size
        self.sum_tree[tree_idx] = priority
        self.min_tree[tree_idx] = priority
        
        # Propagate changes up the tree
        while tree_idx > 1:
            tree_idx //= 2
            left = 2 * tree_idx
            right = left + 1
            self.sum_tree[tree_idx] = self.sum_tree[left] + self.sum_tree[right]
            self.min_tree[tree_idx] = min(self.min_tree[left], self.min_tree[right])
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """
        Add an experience with maximum priority.
        
        New experiences are added with max priority to ensure they're sampled at least once.
        """
        experience = Experience(state, action, reward, next_state, done)
        
        priority = self.max_priority ** self.alpha
        
        self.data[self.position] = experience
        self._update_tree(self.position, priority)
        
        self.position = (self.position + 1) % self.tree_size
        self.size = min(self.size + 1, self.capacity)
    
    def _get_priority_sum(self) -> float:
        """Get total priority sum."""
        return self.sum_tree[1]
    
    def _get_min_priority(self) -> float:
        """Get minimum priority."""
        return self.min_tree[1]
    
    def _sample_index(self, priority_value: float) -> int:
        """Sample an index based on priority value."""
        idx = 1
        while idx < self.tree_size:
            left = 2 * idx
            right = left + 1
            if priority_value <= self.sum_tree[left]:
                idx = left
            else:
                priority_value -= self.sum_tree[left]
                idx = right
        return idx - self.tree_size
    
    @property
    def beta(self) -> float:
        """Calculate current beta value (annealed from beta_start to 1.0)."""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a prioritized batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        indices = []
        experiences = []
        priorities = []
        
        priority_sum = self._get_priority_sum()
        segment = priority_sum / batch_size
        
        for i in range(batch_size):
            # Sample from each segment for more uniform coverage
            low = segment * i
            high = segment * (i + 1)
            priority_value = random.uniform(low, high)
            
            idx = self._sample_index(priority_value)
            
            # Ensure we don't sample empty slots
            while self.data[idx] is None:
                priority_value = random.uniform(0, priority_sum)
                idx = self._sample_index(priority_value)
            
            indices.append(idx)
            experiences.append(self.data[idx])
            priorities.append(self.sum_tree[idx + self.tree_size])
        
        # Calculate importance sampling weights
        min_priority = self._get_min_priority()
        max_weight = (min_priority / priority_sum * self.size) ** (-self.beta)
        
        weights = []
        for priority in priorities:
            prob = priority / priority_sum
            weight = (prob * self.size) ** (-self.beta)
            weights.append(weight / max_weight)
        
        self.frame += 1
        
        states = torch.tensor(
            np.array([e.state for e in experiences]),
            dtype=torch.float32
        )
        actions = torch.tensor(
            [e.action for e in experiences],
            dtype=torch.long
        )
        rewards = torch.tensor(
            [e.reward for e in experiences],
            dtype=torch.float32
        )
        next_states = torch.tensor(
            np.array([e.next_state for e in experiences]),
            dtype=torch.float32
        )
        dones = torch.tensor(
            [e.done for e in experiences],
            dtype=torch.float32
        )
        weights = torch.tensor(weights, dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """
        Update priorities based on TD-errors.
        
        Args:
            indices: Indices of experiences to update
            td_errors: TD-errors for priority calculation
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self._update_tree(idx, priority)
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch."""
        return self.size >= batch_size