import numpy as np
import pickle
import os
import random
from collections import defaultdict, deque

class QLearningAgent:
    """
    Q-learning agent that only makes decisions at junctions.
    
    A junction is a tile where:
    - There are 2+ FORWARD directions (excluding reverse)
    
    NOT a junction:
    - Corridor (1 forward direction)
    - L-bend/corner (1 forward direction after current is blocked)
    """
    
    def __init__(self, actions, alpha=0.3, gamma=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.03):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table
        self.Q = defaultdict(lambda: {a: 0.0 for a in actions})
        
        # Current direction Pacman is moving
        self.current_direction = None
        
        # Stats
        self.updates_count = 0
        self.decisions_made = 0
        
        # For learning: store the last junction state/action
        self.last_junction_state = None
        self.last_junction_action = None
        self.reward_since_junction = 0
        self.steps_since_junction = 0
        
        print(f"[JUNCTION AGENT] α={alpha}, γ={gamma}, ε={epsilon}")
    
    def get_open_directions(self, state):
        """Get list of all open (passable) directions"""
        if not state:
            return self.actions
        
        pacman = state.get('pacman', {})
        px = pacman.get('x', 0) // 8
        py = pacman.get('y', 0) // 8
        game_map = state.get('map', [])
        
        if not game_map:
            return self.actions
        
        open_dirs = []
        dir_map = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}
        
        for action, (dx, dy) in dir_map.items():
            nx, ny = px + dx, py + dy
            if 0 <= ny < 36 and 0 <= nx < 28:
                tile = game_map[ny][nx]
                if tile in (64, 16, 20):  # Passable: space, dot, pill
                    open_dirs.append(action)
        
        return open_dirs if open_dirs else self.actions
    
    def get_forward_directions(self, open_dirs):
        """
        Get directions that are NOT reversing.
        Forward = all open directions except the reverse of current direction.
        """
        if self.current_direction is None:
            return open_dirs
        
        reverse_map = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
        reverse = reverse_map.get(self.current_direction)
        
        forward = [d for d in open_dirs if d != reverse]
        
        # If no forward directions, we're at a dead end - must reverse
        return forward if forward else open_dirs
    
    def is_junction(self, state):
        """
        Determine if current position is a TRUE junction (decision point).
        
        Junction = 2+ forward directions available (a real choice)
        NOT junction = 0 or 1 forward direction (no choice to make)
        """
        if not state:
            return True
        
        # First move of the game - must decide
        if self.current_direction is None:
            return True
        
        open_dirs = self.get_open_directions(state)
        forward_dirs = self.get_forward_directions(open_dirs)
        
        # 2+ forward directions = TRUE junction (T, cross, fork)
        # 0 or 1 forward direction = NOT a junction (corridor, corner, dead end)
        return len(forward_dirs) >= 2
    
    def get_legal_actions(self, state):
        """Get passable directions (alias for compatibility)"""
        return self.get_open_directions(state)
    
    def get_state_key(self, state):
        """
        State representation for junction-based learning.
        
        Since we only decide at junctions, we can use a richer state:
        - Junction identity (position)
        - Ghost threat level and direction
        - Available exits
        - Food direction
        """
        if not state:
            return (0, 0, 0, 0, 0, 0)
        
        pacman = state.get('pacman', {})
        px = pacman.get('x', 0) // 8
        py = pacman.get('y', 0) // 8
        game_map = state.get('map', [])
        ghosts = state.get('ghosts', [])
        
        # === JUNCTION POSITION (exact, since there are only ~50-60 junctions) ===
        junction_id = py * 28 + px  # Unique ID for this tile
        
        # === AVAILABLE EXITS (bitmask) ===
        open_dirs = self.get_open_directions(state)
        exits = 0
        for i, d in enumerate(['up', 'down', 'left', 'right']):
            if d in open_dirs:
                exits |= (1 << i)
        
        # === GHOST THREAT ===
        min_dist = 100
        threat_dir = 4  # No threat
        frightened_count = 0
        
        for ghost in ghosts:
            gs = ghost.get('state', 0)
            gx = ghost.get('x', 0) // 8
            gy = ghost.get('y', 0) // 8
            dist = abs(gx - px) + abs(gy - py)
            
            if gs == 3:  # Frightened
                if dist < 8:
                    frightened_count += 1
            elif gs in (1, 2):  # Dangerous
                if dist < min_dist:
                    min_dist = dist
                    # Direction ghost is coming from
                    dx, dy = gx - px, gy - py
                    if abs(dx) > abs(dy):
                        threat_dir = 3 if dx > 0 else 2  # right or left
                    elif abs(dy) > 0:
                        threat_dir = 1 if dy > 0 else 0  # down or up
        
        # Bucket ghost distance
        if min_dist <= 3:
            ghost_danger = 0  # Critical
        elif min_dist <= 6:
            ghost_danger = 1  # Close
        elif min_dist <= 10:
            ghost_danger = 2  # Medium
        else:
            ghost_danger = 3  # Safe
            threat_dir = 4  # Don't care about direction
        
        # === FOOD DIRECTION ===
        food_dir = self._find_food_direction(px, py, game_map)
        
        # === POWER MODE ===
        power_mode = 1 if frightened_count > 0 else 0
        
        # State: (junction_id, exits, ghost_danger, threat_dir, food_dir, power_mode)
        return (junction_id, exits, ghost_danger, threat_dir, food_dir, power_mode)
    
    def _find_food_direction(self, px, py, game_map):
        """Find direction to nearest food using simple search"""
        if not game_map:
            return 4
        
        # Check each direction
        dirs = [(0, -1, 0), (0, 1, 1), (-1, 0, 2), (1, 0, 3)]  # up, down, left, right
        
        best_dir = 4
        best_dist = 100
        
        for dx, dy, dir_code in dirs:
            # Search up to 10 tiles in this direction
            for dist in range(1, 12):
                nx, ny = px + dx * dist, py + dy * dist
                
                if not (0 <= ny < 36 and 0 <= nx < 28):
                    break
                
                tile = game_map[ny][nx]
                
                if tile in (16, 20):  # Dot or pill
                    if dist < best_dist:
                        best_dist = dist
                        best_dir = dir_code
                    break
                elif tile not in (64,):  # Wall
                    break
        
        # If no food in straight lines, do a quick BFS
        if best_dir == 4:
            best_dir = self._bfs_food_direction(px, py, game_map)
        
        return best_dir
    
    def _bfs_food_direction(self, px, py, game_map):
        """BFS to find food direction"""
        from collections import deque
        
        visited = {(px, py)}
        queue = deque()
        
        dirs = [(0, -1, 0), (0, 1, 1), (-1, 0, 2), (1, 0, 3)]
        
        # Initialize with adjacent cells
        for dx, dy, dir_code in dirs:
            nx, ny = px + dx, py + dy
            if 0 <= ny < 36 and 0 <= nx < 28:
                tile = game_map[ny][nx]
                if tile in (16, 20):
                    return dir_code
                if tile in (64, 16, 20):
                    queue.append((nx, ny, dir_code))
                    visited.add((nx, ny))
        
        # BFS
        while queue:
            x, y, origin_dir = queue.popleft()
            
            for dx, dy, _ in dirs:
                nx, ny = x + dx, y + dy
                if (nx, ny) in visited:
                    continue
                if not (0 <= ny < 36 and 0 <= nx < 28):
                    continue
                
                tile = game_map[ny][nx]
                if tile in (16, 20):
                    return origin_dir
                if tile in (64,):
                    visited.add((nx, ny))
                    queue.append((nx, ny, origin_dir))
        
        return 4
    
    def get_action(self, state):
        """
        Get action for current state.
        
        At TRUE junction: Make Q-learning decision from forward directions
        At corridor/corner: Automatically take the only forward direction
        At dead end: Reverse
        """
        open_dirs = self.get_open_directions(state)
        forward_dirs = self.get_forward_directions(open_dirs)
        
        # Not at a junction - take the only available forward direction
        if not self.is_junction(state):
            if len(forward_dirs) == 1:
                # Corridor or L-bend: exactly one way to go
                self.current_direction = forward_dirs[0]
                return self.current_direction
            elif len(forward_dirs) == 0:
                # Dead end: must reverse (open_dirs has the reverse)
                if open_dirs:
                    self.current_direction = open_dirs[0]
                    return self.current_direction
        
        # At a TRUE junction - make a Q-learning decision
        self.decisions_made += 1
        
        # Epsilon-greedy selection from FORWARD directions only
        if random.random() < self.epsilon:
            # Exploration: random choice from forward directions
            action = random.choice(forward_dirs)
        else:
            # Exploitation: best Q-value from forward directions
            state_key = self.get_state_key(state)
            q_vals = {a: self.Q[state_key][a] for a in forward_dirs}
            action = max(q_vals, key=q_vals.get)
        
        self.current_direction = action
        return action
    
    def accumulate_reward(self, reward):
        """Accumulate reward between junctions"""
        self.reward_since_junction += reward
        self.steps_since_junction += 1
    
    def learn_at_junction(self, state, action, done=False):
        """
        Learn when reaching a new junction.
        Uses accumulated reward since last junction.
        """
        if self.last_junction_state is None:
            # First junction - just record
            self.last_junction_state = self.get_state_key(state)
            self.last_junction_action = action
            self.reward_since_junction = 0
            self.steps_since_junction = 0
            return
        
        # Learn from transition: last_junction -> this_junction
        current_state_key = self.get_state_key(state)
        
        if done:
            target = self.reward_since_junction
        else:
            forward_dirs = self.get_forward_directions(self.get_open_directions(state))
            if forward_dirs:
                max_next_q = max(self.Q[current_state_key][a] for a in forward_dirs)
            else:
                max_next_q = 0
            target = self.reward_since_junction + self.gamma * max_next_q
        
        # Adaptive learning rate
        if self.reward_since_junction < -100:
            alpha = min(0.6, self.alpha * 2)
        elif self.reward_since_junction > 50:
            alpha = min(0.5, self.alpha * 1.5)
        else:
            alpha = self.alpha
        
        # Update
        old_q = self.Q[self.last_junction_state][self.last_junction_action]
        self.Q[self.last_junction_state][self.last_junction_action] += alpha * (target - old_q)
        self.updates_count += 1
        
        # Log significant updates
        if abs(self.reward_since_junction) > 100:
            new_q = self.Q[self.last_junction_state][self.last_junction_action]
            print(f"[LEARN] R={self.reward_since_junction:.1f} over {self.steps_since_junction} steps, "
                  f"Q: {old_q:.1f}→{new_q:.1f}")
        
        # Update for next junction
        self.last_junction_state = current_state_key
        self.last_junction_action = action
        self.reward_since_junction = 0
        self.steps_since_junction = 0
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_episode(self):
        """Reset for new episode"""
        self.current_direction = None
        self.last_junction_state = None
        self.last_junction_action = None
        self.reward_since_junction = 0
        self.steps_since_junction = 0
        self.decisions_made = 0
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'Q': dict(self.Q),
                'epsilon': self.epsilon,
                'updates': self.updates_count
            }, f)
        print(f"[SAVE] {len(self.Q)} states, {self.updates_count} updates, ε={self.epsilon:.3f}")
    
    def load(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.Q = defaultdict(lambda: {a: 0.0 for a in self.actions}, data['Q'])
                self.epsilon = data.get('epsilon', self.epsilon)
                self.updates_count = data.get('updates', 0)
            print(f"[LOAD] {len(self.Q)} states, ε={self.epsilon:.3f}")
    
    def print_stats(self):
        print(f"\n{'='*60}")
        print(f"JUNCTION Q-LEARNING STATS")
        print(f"{'='*60}")
        print(f"States: {len(self.Q)}")
        print(f"Updates: {self.updates_count}")
        print(f"Epsilon: {self.epsilon:.3f}")
        
        if self.Q:
            all_q = [(s, a, q) for s, acts in self.Q.items() for a, q in acts.items()]
            all_q.sort(key=lambda x: x[2], reverse=True)
            
            print(f"\nTop 5 Q-values:")
            for s, a, q in all_q[:5]:
                jid, exits, gdanger, gdir, food, power = s
                jx, jy = jid % 28, jid // 28
                print(f"  Q={q:7.2f} {a:5s} @ ({jx},{jy}) exits={exits:04b} ghost={gdanger} food={food}")
            
            print(f"\nBottom 5 Q-values:")
            for s, a, q in all_q[-5:]:
                jid, exits, gdanger, gdir, food, power = s
                jx, jy = jid % 28, jid // 28
                print(f"  Q={q:7.2f} {a:5s} @ ({jx},{jy}) exits={exits:04b} ghost={gdanger}")
        
        print(f"{'='*60}\n")
