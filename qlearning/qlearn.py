import numpy as np
import pickle
import os
import random
from collections import defaultdict
import config

class QLearningAgent:
    def __init__(self, actions, alpha=config.ALPHA, gamma=config.GAMMA, 
                 epsilon=config.EPSILON_START, epsilon_decay=config.EPSILON_DECAY, 
                 epsilon_min=config.EPSILON_MIN):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Tabular Q-table: Q[state][action] = value
        self.Q = defaultdict(lambda: {a: 0.0 for a in actions})
        
        self.states_visited = 0
        self.updates_count = 0
        
        # Logging
        self.last_state_key = None
        self.last_q_values = {}
        
        print(f"[AGENT INIT] Alpha={alpha}, Gamma={gamma}, Epsilon={epsilon}")
    
    def get_state_key(self, state):
        """
        Convert state to simple tuple for Q-table lookup.
        """
        if not state:
            return (0, 0, 0, 0, False)
        
        # Extract pacman position (in tiles)
        pacman = state.get('pacman', {})
        px = pacman.get('x', 0) // 8
        py = pacman.get('y', 0) // 8
        
        # Find nearest dangerous ghost
        ghosts = state.get('ghosts', [])
        min_ghost_dist = 20  # Far away
        ghost_dir = 4  # None
        
        dangerous_ghost_count = 0
        
        for ghost in ghosts:
            ghost_state = ghost.get('state', 0)
            if ghost_state not in (2, 3):
                dangerous_ghost_count += 1
                gx = ghost.get('x', 0) // 8
                gy = ghost.get('y', 0) // 8
                
                dist = abs(gx - px) + abs(gy - py)
                
                if dist < min_ghost_dist:
                    min_ghost_dist = dist
                    
                    # Direction to ghost
                    dx = gx - px
                    dy = gy - py
                    if abs(dx) > abs(dy):
                        ghost_dir = 2 if dx < 0 else 3  # left:right
                    else:
                        ghost_dir = 0 if dy < 0 else 1  # up:down
        
        # Bucket ghost distance
        if min_ghost_dist < 3:
            ghost_dist_bucket = 0  # Very close
        elif min_ghost_dist < 6:
            ghost_dist_bucket = 1  # Close
        elif min_ghost_dist < 12:
            ghost_dist_bucket = 2  # Medium
        else:
            ghost_dist_bucket = 3  # Far
        
        # Check for nearby food
        game_map = state.get('map', [])
        food_nearby = False
        
        if game_map and 0 <= py < 36 and 0 <= px < 28:
            # Check surrounding tiles
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    cx, cy = px + dx, py + dy
                    if 0 <= cy < 36 and 0 <= cx < 28:
                        if game_map[cy][cx] in (16, 20):
                            food_nearby = True
                            break
                if food_nearby:
                    break
        
        # Discretize pacman position to reduce state space
        px_bucket = px // 2  
        py_bucket = py // 2 
        
        state_key = (px_bucket, py_bucket, ghost_dist_bucket, ghost_dir, food_nearby)
        
        # Log occasionally
        if random.random() < 0.01:  # 1% of the time
            print(f"[STATE] Pacman=({px},{py}→{px_bucket},{py_bucket}), "
                  f"Ghost(dist={min_ghost_dist}→bucket{ghost_dist_bucket}, dir={ghost_dir}, count={dangerous_ghost_count}), "
                  f"Food={food_nearby}")
        
        return state_key
    
    def get_legal_actions(self, state):
        """Get legal actions (no walls)"""
        if not state:
            return self.actions
        
        pacman = state.get('pacman', {})
        px = pacman.get('x', 0) // 8
        py = pacman.get('y', 0) // 8
        
        game_map = state.get('map', [])
        legal = []
        
        directions = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0)
        }
        
        for action, (dx, dy) in directions.items():
            nx, ny = px + dx, py + dy
            
            if game_map and 0 <= ny < 36 and 0 <= nx < 28:
                if game_map[ny][nx] in (64, 16, 20):
                    legal.append(action)
        
        if not legal:
            print(f"[WARNING] No legal actions at ({px},{py})! Returning all.")
            return self.actions
        
        return legal
    
    def get_action(self, state):
        """Epsilon-greedy action selection"""
        legal_actions = self.get_legal_actions(state)
        
        if not legal_actions:
            print(f"[ERROR] No legal actions!")
            return 'up'
        
        # Explore
        if np.random.rand() < self.epsilon:
            action = random.choice(legal_actions)
            if random.random() < 0.01:
                print(f"[ACTION] EXPLORING: chose {action} randomly")
            return action
        
        # Exploit
        state_key = self.get_state_key(state)
        q_values = self.Q[state_key]
        
        # Get best legal action
        legal_q = {a: q_values[a] for a in legal_actions}
        best_action = max(legal_q, key=legal_q.get)
        
        # Log occasionally
        if random.random() < 0.01:
            print(f"[ACTION] EXPLOITING: Q-values={legal_q}, chose {best_action}")
        
        self.last_state_key = state_key
        self.last_q_values = legal_q
        
        return best_action
    
    def learn(self, state, action, reward, next_state):
        """Q-learning update with adaptive learning rate for dangerous situations"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Current Q-value
        current_q = self.Q[state_key][action]
        
        # Max next Q-value
        legal_next = self.get_legal_actions(next_state)
        if legal_next:
            next_q_values = [self.Q[next_state_key][a] for a in legal_next]
            max_next_q = max(next_q_values)
        else:
            max_next_q = 0.0
        
        # Adaptive learning rate: learn faster from dangerous situations
        # Use higher alpha for deaths and ghost proximity penalties
        if reward < -50:  # Death or major penalty
            alpha = min(0.4, self.alpha * 2.0)  # Double learning rate, capped at 0.4
        elif reward > 50:  # Major reward (power pill, ghost eaten)
            alpha = min(0.3, self.alpha * 1.5)  # 1.5x learning rate for good outcomes
        else:
            alpha = self.alpha  # Normal learning rate
            
            # Q-learning update
        target_q = reward + self.gamma * max_next_q
        new_q = current_q + alpha * (target_q - current_q)
        self.Q[state_key][action] = new_q
        
        self.updates_count += 1
        self.states_visited = len(self.Q)
        
        # Log significant updates
        if abs(reward) > 50 or abs(new_q - current_q) > 10:
            alpha_used = alpha if alpha != self.alpha else self.alpha
            print(f"[LEARN] State={state_key}, Action={action}")
            print(f"        Reward={reward:.1f}, Q: {current_q:.2f} → {new_q:.2f} (Δ={new_q-current_q:.2f})")
            print(f"        Target={target_q:.2f}, MaxNextQ={max_next_q:.2f}, Alpha={alpha_used:.2f}")
    
    def decay_epsilon(self):
        old_epsilon = self.epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        if old_epsilon != self.epsilon and random.random() < 0.1:
            print(f"[EPSILON] Decayed: {old_epsilon:.3f} → {self.epsilon:.3f}")
    
    def save(self, filename):
        data = {
            'Q': dict(self.Q),
            'epsilon': self.epsilon,
            'states_visited': self.states_visited,
            'updates_count': self.updates_count
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"[SAVE] Q-table saved: {len(self.Q)} states, {self.updates_count} updates")
    
    def load(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.Q = defaultdict(lambda: {a: 0.0 for a in self.actions}, data['Q'])
                self.epsilon = data.get('epsilon', self.epsilon)
                self.states_visited = data.get('states_visited', 0)
                self.updates_count = data.get('updates_count', 0)
            print(f"[LOAD] Loaded Q-table: {len(self.Q)} states, {self.updates_count} updates, epsilon={self.epsilon:.3f}")
    
    def map_state(self, api_state):
        return api_state
    
    def get_weights(self):
        """For compatibility - return Q-table stats"""
        # Sample some Q-values
        sample_states = list(self.Q.keys())[:5]
        sample_q = {}
        for s in sample_states:
            sample_q[s] = self.Q[s]
        
        return {
            'states_explored': len(self.Q),
            'updates_count': self.updates_count,
            'epsilon': self.epsilon,
            'sample_q_values': sample_q
        }
    
    def print_stats(self):
        """Print detailed statistics"""
        print(f"\n{'='*60}")
        print(f"Q-LEARNING AGENT STATISTICS")
        print(f"{'='*60}")
        print(f"States explored: {len(self.Q)}")
        print(f"Total updates: {self.updates_count}")
        print(f"Current epsilon: {self.epsilon:.3f}")
        
        if len(self.Q) > 0:
            # Find states with highest/lowest Q-values
            all_q_values = []
            for state, actions in self.Q.items():
                for action, q in actions.items():
                    all_q_values.append((state, action, q))
            
            all_q_values.sort(key=lambda x: x[2], reverse=True)
            
            print(f"\nTop 5 Q-values:")
            for i, (state, action, q) in enumerate(all_q_values[:5]):
                print(f"  {i+1}. State={state}, Action={action}, Q={q:.2f}")
            
            print(f"\nBottom 5 Q-values:")
            for i, (state, action, q) in enumerate(all_q_values[-5:]):
                print(f"  {i+1}. State={state}, Action={action}, Q={q:.2f}")
            
            # Average Q-value
            avg_q = sum(q for _, _, q in all_q_values) / len(all_q_values)
            print(f"\nAverage Q-value: {avg_q:.2f}")
        
        print(f"{'='*60}\n")
