import numpy as np
import pickle
import os
import random
from collections import defaultdict, deque

class QLearningAgent:
    """
    Improved Q-learning agent with better ghost avoidance.
    
    Key improvements:
    - Better state encoding with escape route awareness
    - Ghost direction tracking relative to available exits
    - Danger-aware action selection
    - Optimistic initialization for better exploration
    """
    
    def __init__(self, actions, alpha=0.2, gamma=0.95, 
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table with optimistic initialization (small positive values encourage exploration)
        self.Q = defaultdict(lambda: {a: 10.0 for a in actions})
        
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
        
        # Direction mappings
        self.dir_to_vec = {
            'up': (0, -1), 'down': (0, 1), 
            'left': (-1, 0), 'right': (1, 0)
        }
        self.reverse_dir = {
            'up': 'down', 'down': 'up', 
            'left': 'right', 'right': 'left'
        }
        
        print(f"[IMPROVED AGENT] α={alpha}, γ={gamma}, ε={epsilon}")
    
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
        
        for action, (dx, dy) in self.dir_to_vec.items():
            nx, ny = px + dx, py + dy
            if 0 <= ny < 36 and 0 <= nx < 28:
                tile = game_map[ny][nx]
                if tile in (64, 16, 20):  # Passable: space, dot, pill
                    open_dirs.append(action)
        
        return open_dirs if open_dirs else self.actions
    
    def get_forward_directions(self, open_dirs):
        """Get directions that are NOT reversing."""
        if self.current_direction is None:
            return open_dirs
        
        reverse = self.reverse_dir.get(self.current_direction)
        forward = [d for d in open_dirs if d != reverse]
        
        return forward if forward else open_dirs
    
    def is_junction(self, state):
        """Determine if current position is a TRUE junction (decision point)."""
        if not state:
            return True
        
        if self.current_direction is None:
            return True
        
        open_dirs = self.get_open_directions(state)
        forward_dirs = self.get_forward_directions(open_dirs)
        
        return len(forward_dirs) >= 2
    
    def get_ghost_info(self, state, px, py):
        """
        Get detailed ghost information for state encoding.
        Returns: (min_dist, danger_dirs_bitmask, frightened_nearby, closest_ghost_quadrant)
        """
        ghosts = state.get('ghosts', [])
        
        min_dangerous_dist = 100
        frightened_nearby = 0
        danger_dirs = 0  # Bitmask: up=1, down=2, left=4, right=8
        closest_quadrant = 4  # 0=up-left, 1=up-right, 2=down-left, 3=down-right, 4=none
        
        for ghost in ghosts:
            gs = ghost.get('state', 0)
            gx = ghost.get('x', 0) // 8
            gy = ghost.get('y', 0) // 8
            dist = abs(gx - px) + abs(gy - py)
            
            if gs == 3:  # Frightened
                if dist <= 8:
                    frightened_nearby += 1
            elif gs in (0, 1, 2):  # Dangerous (scatter, chase, or house-leaving)
                if dist < min_dangerous_dist:
                    min_dangerous_dist = dist
                    # Determine quadrant of closest ghost
                    if gy <= py and gx <= px:
                        closest_quadrant = 0  # up-left
                    elif gy <= py and gx > px:
                        closest_quadrant = 1  # up-right
                    elif gy > py and gx <= px:
                        closest_quadrant = 2  # down-left
                    else:
                        closest_quadrant = 3  # down-right
                
                # Mark dangerous directions (directions toward this ghost)
                if dist <= 6:  # Only care about close ghosts
                    if gy < py:  # Ghost is above
                        danger_dirs |= 1  # up is dangerous
                    elif gy > py:  # Ghost is below
                        danger_dirs |= 2  # down is dangerous
                    if gx < px:  # Ghost is left
                        danger_dirs |= 4  # left is dangerous
                    elif gx > px:  # Ghost is right
                        danger_dirs |= 8  # right is dangerous
        
        return min_dangerous_dist, danger_dirs, frightened_nearby, closest_quadrant
    
    def get_state_key(self, state):
        """
        Improved state representation with escape awareness.
        """
        if not state:
            return (0, 0, 0, 0, 0, 0, 0)
        
        pacman = state.get('pacman', {})
        px = pacman.get('x', 0) // 8
        py = pacman.get('y', 0) // 8
        game_map = state.get('map', [])
        
        # === POSITION (coarser buckets to reduce state space) ===
        # Divide map into zones for generalization
        zone_x = px // 7  # 4 horizontal zones (0-3)
        zone_y = py // 9  # 4 vertical zones (0-3)
        zone_id = zone_y * 4 + zone_x
        
        # === AVAILABLE EXITS ===
        open_dirs = self.get_open_directions(state)
        exits = 0
        for i, d in enumerate(['up', 'down', 'left', 'right']):
            if d in open_dirs:
                exits |= (1 << i)
        
        # === GHOST ANALYSIS ===
        min_dist, danger_dirs, frightened_count, ghost_quadrant = self.get_ghost_info(state, px, py)
        
        # Bucket ghost distance
        if min_dist <= 2:
            ghost_danger = 0  # Critical - immediate threat
        elif min_dist <= 4:
            ghost_danger = 1  # High danger
        elif min_dist <= 7:
            ghost_danger = 2  # Medium danger
        elif min_dist <= 12:
            ghost_danger = 3  # Low danger
        else:
            ghost_danger = 4  # Safe
            danger_dirs = 0  # Don't care about direction when safe
            ghost_quadrant = 4
        
        # === SAFE EXITS (exits that don't lead toward ghosts) ===
        safe_exits = exits & (~danger_dirs) & 0xF
        if safe_exits == 0:
            safe_exits = exits  # If no safe exits, all exits are "safe"
        
        # === FOOD DIRECTION ===
        food_dir = self._find_food_direction(px, py, game_map)
        
        # === POWER MODE ===
        power_mode = 1 if frightened_count > 0 else 0
        
        # State tuple - designed for ghost avoidance
        return (zone_id, exits, ghost_danger, safe_exits, ghost_quadrant, food_dir, power_mode)
    
    def _find_food_direction(self, px, py, game_map):
        """Find direction to nearest food"""
        if not game_map:
            return 4
        
        dirs = [(0, -1, 0), (0, 1, 1), (-1, 0, 2), (1, 0, 3)]
        
        best_dir = 4
        best_dist = 100
        
        for dx, dy, dir_code in dirs:
            for dist in range(1, 12):
                nx, ny = px + dx * dist, py + dy * dist
                
                if not (0 <= ny < 36 and 0 <= nx < 28):
                    break
                
                tile = game_map[ny][nx]
                
                if tile in (16, 20):
                    if dist < best_dist:
                        best_dist = dist
                        best_dir = dir_code
                    break
                elif tile not in (64,):
                    break
        
        if best_dir == 4:
            best_dir = self._bfs_food_direction(px, py, game_map)
        
        return best_dir
    
    def _bfs_food_direction(self, px, py, game_map):
        """BFS to find food direction"""
        visited = {(px, py)}
        queue = deque()
        
        dirs = [(0, -1, 0), (0, 1, 1), (-1, 0, 2), (1, 0, 3)]
        
        for dx, dy, dir_code in dirs:
            nx, ny = px + dx, py + dy
            if 0 <= ny < 36 and 0 <= nx < 28:
                tile = game_map[ny][nx]
                if tile in (16, 20):
                    return dir_code
                if tile in (64, 16, 20):
                    queue.append((nx, ny, dir_code))
                    visited.add((nx, ny))
        
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
    
    def get_safe_actions(self, state, forward_dirs):
        """
        Filter actions to prefer safe directions when ghosts are near.
        Returns actions sorted by safety (safest first).
        """
        if not state:
            return forward_dirs
        
        pacman = state.get('pacman', {})
        px = pacman.get('x', 0) // 8
        py = pacman.get('y', 0) // 8
        
        min_dist, danger_dirs, frightened, _ = self.get_ghost_info(state, px, py)
        
        # If ghosts are frightened or far away, don't filter
        if frightened > 0 or min_dist > 6:
            return forward_dirs
        
        # Map directions to bitmask
        dir_bits = {'up': 1, 'down': 2, 'left': 4, 'right': 8}
        
        safe_dirs = []
        risky_dirs = []
        
        for d in forward_dirs:
            if (dir_bits.get(d, 0) & danger_dirs) == 0:
                safe_dirs.append(d)
            else:
                risky_dirs.append(d)
        
        # Return safe dirs first, then risky if no safe options
        return safe_dirs + risky_dirs if safe_dirs else forward_dirs
    
    def get_action(self, state):
        """
        Get action with improved ghost avoidance.
        """
        open_dirs = self.get_open_directions(state)
        forward_dirs = self.get_forward_directions(open_dirs)
        
        # Not at a junction - take the only available forward direction
        if not self.is_junction(state):
            if len(forward_dirs) == 1:
                self.current_direction = forward_dirs[0]
                return self.current_direction
            elif len(forward_dirs) == 0:
                if open_dirs:
                    self.current_direction = open_dirs[0]
                    return self.current_direction
        
        # At a junction - make a decision
        self.decisions_made += 1
        
        # Get safety-sorted actions
        sorted_actions = self.get_safe_actions(state, forward_dirs)
        
        # Check ghost proximity for emergency escape
        if state:
            pacman = state.get('pacman', {})
            px = pacman.get('x', 0) // 8
            py = pacman.get('y', 0) // 8
            min_dist, danger_dirs, frightened, _ = self.get_ghost_info(state, px, py)
            
            # Emergency escape mode - heavily favor safe directions when ghost is very close
            if min_dist <= 3 and frightened == 0 and len(sorted_actions) > 0:
                # 70% chance to take safest action when in immediate danger
                if random.random() < 0.7:
                    self.current_direction = sorted_actions[0]
                    return self.current_direction
        
        # Epsilon-greedy with safety awareness
        if random.random() < self.epsilon:
            # Exploration: weighted random favoring safe directions
            if len(sorted_actions) > 1 and random.random() < 0.6:
                # 60% chance to pick from safer half of options
                safe_half = sorted_actions[:max(1, len(sorted_actions)//2 + 1)]
                action = random.choice(safe_half)
            else:
                action = random.choice(forward_dirs)
        else:
            # Exploitation: best Q-value, but consider safety
            state_key = self.get_state_key(state)
            q_vals = {a: self.Q[state_key][a] for a in forward_dirs}
            
            # Add safety bonus to Q-values when ghost is close
            if state:
                min_dist, _, frightened, _ = self.get_ghost_info(state, px, py)
                if min_dist <= 5 and frightened == 0:
                    dir_bits = {'up': 1, 'down': 2, 'left': 4, 'right': 8}
                    for d in q_vals:
                        if (dir_bits.get(d, 0) & danger_dirs) == 0:
                            q_vals[d] += 50  # Bonus for safe directions
            
            action = max(q_vals, key=q_vals.get)
        
        self.current_direction = action
        return action
    
    def accumulate_reward(self, reward):
        """Accumulate reward between junctions"""
        self.reward_since_junction += reward
        self.steps_since_junction += 1
    
    def learn_at_junction(self, state, action, done=False):
        """Learn when reaching a new junction."""
        if self.last_junction_state is None:
            self.last_junction_state = self.get_state_key(state)
            self.last_junction_action = action
            self.reward_since_junction = 0
            self.steps_since_junction = 0
            return
        
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
        
        # Adaptive learning rate based on reward magnitude
        if self.reward_since_junction < -200:
            alpha = min(0.5, self.alpha * 2)  # Learn faster from deaths
        elif self.reward_since_junction < -50:
            alpha = min(0.4, self.alpha * 1.5)
        elif self.reward_since_junction > 50:
            alpha = min(0.35, self.alpha * 1.3)
        else:
            alpha = self.alpha
        
        # Q-learning update
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
                # Load with optimistic default for new states
                self.Q = defaultdict(lambda: {a: 10.0 for a in self.actions}, data['Q'])
                self.epsilon = data.get('epsilon', self.epsilon)
                self.updates_count = data.get('updates', 0)
            print(f"[LOAD] {len(self.Q)} states, ε={self.epsilon:.3f}")
    
    def print_stats(self):
        print(f"\n{'='*60}")
        print(f"IMPROVED Q-LEARNING STATS")
        print(f"{'='*60}")
        print(f"States: {len(self.Q)}")
        print(f"Updates: {self.updates_count}")
        print(f"Epsilon: {self.epsilon:.3f}")
        
        if self.Q:
            all_q = [(s, a, q) for s, acts in self.Q.items() for a, q in acts.items()]
            all_q.sort(key=lambda x: x[2], reverse=True)
            
            print(f"\nTop 5 Q-values:")
            for s, a, q in all_q[:5]:
                zone, exits, gdanger, safe_exits, gquad, food, power = s
                print(f"  Q={q:7.2f} {a:5s} zone={zone} exits={exits:04b} danger={gdanger} safe={safe_exits:04b}")
            
            print(f"\nBottom 5 Q-values:")
            for s, a, q in all_q[-5:]:
                zone, exits, gdanger, safe_exits, gquad, food, power = s
                print(f"  Q={q:7.2f} {a:5s} zone={zone} exits={exits:04b} danger={gdanger}")
        
        print(f"{'='*60}\n")