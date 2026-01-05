"""
Pacman Environment for Deep Q-Learning

This environment provides a rich state representation for the neural network:
- Pacman position (normalized)
- Ghost positions and states
- Local dot map around Pacman
- Direction information

Reward system inspired by the tabular Q-learning experiments:
- Survival-time based death penalty
- Progressive dot milestone bonuses
- Endgame incentives for collecting final dots
"""

import requests
import numpy as np
from typing import Optional, Tuple, Dict, Any, List


class PacmanEnv:
    """
    Deep Q-Learning environment for Pacman.
    
    Provides a tensor-based state representation suitable for neural networks.
    """
    
    # Map dimensions
    MAP_WIDTH = 28
    MAP_HEIGHT = 36
    TOTAL_DOTS = 244
    
    # Action space
    ACTIONS = ['up', 'down', 'left', 'right']
    NUM_ACTIONS = 4
    ACTION_TO_IDX = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
    IDX_TO_ACTION = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
    
    def __init__(self, url: str = "http://127.0.0.1:8080", local_view_size: int = 7):
        self.base_url = url
        self.local_view_size = local_view_size  # Size of local view around Pacman
        self.session = requests.Session()
        
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry = Retry(total=3, backoff_factor=0.1)
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        
        # Episode tracking
        self.episode = 0
        self.steps = 0
        self.prev_score = 0
        self.prev_lives = 2
        self.deaths_this_episode = 0
        
        # Per-life tracking for survival-based rewards
        self.steps_this_life = 0
        self.dots_this_life = 0
        
        # Dot milestone tracking
        self.dots_eaten_total = 0
        self.milestone_200_given = False
        self.milestone_220_given = False
        self.milestone_235_given = False
        
        # Track previous distance to nearest dot for reward shaping
        self.prev_nearest_dot_dist = None
        
        # Position history tracking to detect oscillation/stuck behavior
        # Track TILE positions (not sub-tile), and only when tile changes
        self.tile_history = []  # List of recent tile (x, y) positions
        self.tile_history_max = 30  # Track last 30 tile positions
        self.last_tile = None  # Last tile position to detect tile changes
        self.visited_tiles = set()  # All tiles visited this life
        self.oscillation_count = 0  # Count consecutive oscillations
        
        # State dimensions
        self.state_dim = self._calculate_state_dim()

        # Junction decision tracking
        self.current_direction = None
        self.last_junction_action = None
        self.last_decision_pos = None  # Track position where last decision was made
        self.arrival_direction = None  # Direction we were traveling when we arrived at current junction
        
        # Direction vectors for map checking
        self.dir_to_vec = {
            'up': (0, -1), 'down': (0, 1),
            'left': (-1, 0), 'right': (1, 0)
        }
        self.reverse_dir = {
            'up': 'down', 'down': 'up',
            'left': 'right', 'right': 'left'
        }
        
    def _calculate_state_dim(self) -> int:
        """Calculate the dimension of the state vector."""
        # Pacman position (x, y normalized) = 2
        # Pacman direction (one-hot) = 4
        # Ghost info per ghost (x, y, dangerous, frightened, distance) = 5 * 4 = 20
        # Local dot map (local_view_size x local_view_size) = 49 (for 7x7)
        # Global info (dots_remaining normalized, lives, power_mode) = 3
        # Distance to nearest dot in each direction = 4
        # Danger features (danger_level, ghost_dir_x, ghost_dir_y, raw_dist) = 4
        # DOT HUNTING features (quadrant_dots x 4, best_quadrant_dir x 2, nearest_dot_dist) = 7
        # ENDGAME features (urgency, direction_to_nearest_dot_x, direction_to_nearest_dot_y) = 3
        # NEW: Dots reachable per direction (4 directions) = 4
        # NEW: Best dot path direction (one-hot 4) = 4
        # NEW: Scattered dot penalty (1) = 1
        
        local_map_size = self.local_view_size * self.local_view_size
        return 2 + 4 + 20 + local_map_size + 3 + 4 + 4 + 7 + 3 + 4 + 4 + 1
    
    def reset(self) -> np.ndarray:
        """Reset the environment and return initial state."""
        try:
            self.session.post(f"{self.base_url}/api/restart", timeout=5)
            self.session.post(f"{self.base_url}/api/start", timeout=5)
            
            state_dict = self._get_state()
            
            if state_dict:
                current_lives = state_dict.get('status', {}).get('lives', 2)
                current_score = state_dict.get('status', {}).get('score', 0)
                
                # Detect new episode
                if current_lives == 2 and current_score == 0:
                    self.episode += 1
                    self.deaths_this_episode = 0
                    self.dots_eaten_total = 0
                    self.milestone_200_given = False
                    self.milestone_220_given = False
                    self.milestone_235_given = False
                    # Reset additional milestones
                    if hasattr(self, 'milestone_50_given'):
                        delattr(self, 'milestone_50_given')
                    if hasattr(self, 'milestone_100_given'):
                        delattr(self, 'milestone_100_given')
                
                self.prev_lives = current_lives
                self.prev_score = current_score
            else:
                self.prev_lives = 2
                self.prev_score = 0
            
            self.steps = 0
            self.steps_this_life = 0
            self.dots_this_life = 0
            
            # Reset position tracking
            self.tile_history = []
            self.last_tile = None
            self.visited_tiles = set()
            self.oscillation_count = 0
            self.prev_nearest_dot_dist = None
            self.last_decision_pos = None  # Reset decision position tracking
            self.arrival_direction = None  # Reset arrival direction
            
            return self._state_to_tensor(state_dict)
            
        except Exception as e:
            print(f"[ERROR] Reset: {e}")
            return np.zeros(self.state_dim, dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take an action and return (next_state, reward, done, info).
        
        Args:
            action: Integer action (0=up, 1=down, 2=left, 3=right)
        
        Returns:
            next_state: numpy array of the next state
            reward: float reward value
            done: boolean indicating if episode/life ended
            info: dictionary with additional information
        """
        action_name = self.IDX_TO_ACTION[action]
        
        try:
            resp = self.session.post(
                f"{self.base_url}/api/step",
                json={"direction": action_name},
                timeout=5
            )
            state_dict = resp.json()
            
            reward, death_occurred, game_over = self._calculate_reward(state_dict)
            
            # Episode ends on death or round won
            done = death_occurred or state_dict.get('status', {}).get('round_won', False)
            
            # MASSIVE Win bonus - this is the ultimate goal!
            if state_dict.get('status', {}).get('round_won', False):
                lives_remaining = state_dict.get('status', {}).get('lives', 0)
                win_bonus = 5000 + (lives_remaining * 2000)  # Huge bonus for winning!
                reward += win_bonus
                print(f"[🏆🏆🏆 WIN! 🏆🏆🏆] Level complete with {lives_remaining} spare lives! Bonus: {win_bonus}")
            
            self.steps += 1
            self.steps_this_life += 1
            self.prev_score = state_dict.get('status', {}).get('score', 0)
            self.prev_lives = state_dict.get('status', {}).get('lives', 2)
            
            if death_occurred:
                self.deaths_this_episode += 1
                self.steps_this_life = 0
                self.dots_this_life = 0
                # Reset position tracking on death
                self.tile_history = []
                self.last_tile = None
                self.visited_tiles = set()
                self.oscillation_count = 0
                self.prev_nearest_dot_dist = None
            
            info = {
                'deaths': self.deaths_this_episode,
                'game_over': game_over,
                'lives': self.prev_lives,
                'score': self.prev_score,
                'dots_eaten': self.dots_eaten_total,
                'round_won': state_dict.get('status', {}).get('round_won', False)
            }
            
            return self._state_to_tensor(state_dict), reward, done, info
            
        except Exception as e:
            print(f"[ERROR] Step: {e}")
            return np.zeros(self.state_dim, dtype=np.float32), -100, True, {'error': str(e)}
    
    def _get_state(self) -> Optional[Dict]:
        """Get current game state from API."""
        try:
            return self.session.get(f"{self.base_url}/api/state", timeout=5).json()
        except:
            return None
    
    def _state_to_tensor(self, state_dict: Optional[Dict]) -> np.ndarray:
        """
        Convert game state to a neural network input tensor.
        
        This creates a rich representation including:
        - Normalized Pacman position
        - Ghost positions and states
        - Local view of dots around Pacman
        - Global game information
        """
        if not state_dict:
            return np.zeros(self.state_dim, dtype=np.float32)
        
        features = []
        
        pacman = state_dict.get('pacman', {})
        ghosts = state_dict.get('ghosts', [])
        status = state_dict.get('status', {})
        game_map = state_dict.get('map', [])
        
        # === Pacman Position (normalized to 0-1) ===
        px = pacman.get('x', 0) / 8  # Convert to tile coords
        py = pacman.get('y', 0) / 8
        features.append(px / self.MAP_WIDTH)
        features.append(py / self.MAP_HEIGHT)
        
        # === Pacman Direction (one-hot) ===
        direction = pacman.get('dir', 0)
        dir_onehot = [0, 0, 0, 0]
        if 0 <= direction < 4:
            dir_onehot[direction] = 1
        features.extend(dir_onehot)
        
        # === Ghost Information ===
        # For each of 4 possible ghosts: (x, y, is_dangerous, is_frightened, distance)
        # Also track nearest dangerous ghost for danger features
        nearest_dangerous_dist = 100
        nearest_dangerous_dx = 0
        nearest_dangerous_dy = 0
        
        for i in range(4):
            if i < len(ghosts):
                ghost = ghosts[i]
                gx_tile = ghost.get('x', 0) / 8
                gy_tile = ghost.get('y', 0) / 8
                gx = gx_tile / self.MAP_WIDTH
                gy = gy_tile / self.MAP_HEIGHT
                gs = ghost.get('state', 0)
                is_dangerous = 1.0 if gs in (0, 1, 2) else 0.0
                is_frightened = 1.0 if gs == 3 else 0.0
                
                # Distance to this ghost (normalized, closer = higher value for danger)
                dist = abs(gx_tile - px) + abs(gy_tile - py)
                dist_normalized = max(0, 1.0 - dist / 15.0)  # 1.0 = very close, 0.0 = far
                
                features.extend([gx, gy, is_dangerous, is_frightened, dist_normalized])
                
                # Track nearest dangerous ghost
                if is_dangerous and dist < nearest_dangerous_dist:
                    nearest_dangerous_dist = dist
                    nearest_dangerous_dx = gx_tile - px
                    nearest_dangerous_dy = gy_tile - py
            else:
                features.extend([0, 0, 0, 0, 0])
        
        # === Local Dot Map ===
        # Create a local view around Pacman showing dots/pills/walls
        px_tile = int(px)
        py_tile = int(py)
        half_size = self.local_view_size // 2
        
        local_map = []
        for dy in range(-half_size, half_size + 1):
            for dx in range(-half_size, half_size + 1):
                tx, ty = px_tile + dx, py_tile + dy
                if 0 <= ty < len(game_map) and 0 <= tx < len(game_map[0]):
                    tile = game_map[ty][tx]
                    # Encode: 0=wall/empty, 0.5=dot, 1.0=power pill
                    if tile == 16:  # Dot
                        local_map.append(0.5)
                    elif tile == 20:  # Power pill
                        local_map.append(1.0)
                    elif tile == 64:  # Empty (passable)
                        local_map.append(0.1)
                    else:  # Wall
                        local_map.append(0.0)
                else:
                    local_map.append(0.0)  # Out of bounds = wall
        features.extend(local_map)
        
        # === Global Information ===
        dots_remaining = status.get('dots_remaining', 244)
        features.append(dots_remaining / self.TOTAL_DOTS)  # Normalized dots remaining
        features.append(status.get('lives', 2) / 3.0)  # Normalized lives
        
        # Power mode (any ghost frightened?)
        power_mode = any(g.get('state', 0) == 3 for g in ghosts)
        features.append(1.0 if power_mode else 0.0)
        
        # === Distance to nearest dot in each direction ===
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # up, down, left, right
            dist = self._find_nearest_dot_distance(px_tile, py_tile, dx, dy, game_map)
            features.append(dist / 15.0)  # Normalize by max search distance
        
        # === DOT HUNTING FEATURES ===
        # Count dots in each quadrant of the map to guide exploration
        quadrant_dots = self._count_dots_per_quadrant(game_map)
        total_quadrant_dots = sum(quadrant_dots) + 0.001  # Avoid div by zero
        
        # Normalized dot count per quadrant (0-1, higher = more dots there)
        for q_dots in quadrant_dots:
            features.append(q_dots / total_quadrant_dots if total_quadrant_dots > 0 else 0.25)
        
        # Direction to the quadrant with the most dots
        # Quadrants: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
        best_quadrant = quadrant_dots.index(max(quadrant_dots))
        quadrant_centers = [
            (7, 9),   # top-left center
            (21, 9),  # top-right center
            (7, 27),  # bottom-left center
            (21, 27)  # bottom-right center
        ]
        best_center = quadrant_centers[best_quadrant]
        
        # Direction to best quadrant (normalized -1 to 1)
        dir_to_dots_x = (best_center[0] - px_tile) / 14.0  # Normalize by half map width
        dir_to_dots_y = (best_center[1] - py_tile) / 18.0  # Normalize by half map height
        features.append(max(-1, min(1, dir_to_dots_x)))
        features.append(max(-1, min(1, dir_to_dots_y)))
        
        # Distance to actual nearest dot (BFS-based, more accurate)
        nearest_dot_dist, nearest_dot_pos = self._find_nearest_dot_bfs_with_pos(px_tile, py_tile, game_map)
        features.append(1.0 - min(1.0, nearest_dot_dist / 30.0))  # Inverted: 1=dot nearby, 0=far
        
        # === DANGER FEATURES - Critical for ghost avoidance ===
        # Danger level (1.0 = ghost very close, 0.0 = safe)
        if nearest_dangerous_dist <= 1:
            danger_level = 1.0
        elif nearest_dangerous_dist <= 2:
            danger_level = 0.8
        elif nearest_dangerous_dist <= 3:
            danger_level = 0.6
        elif nearest_dangerous_dist <= 5:
            danger_level = 0.4
        elif nearest_dangerous_dist <= 8:
            danger_level = 0.2
        else:
            danger_level = 0.0
        features.append(danger_level)
        
        # Direction to nearest dangerous ghost (helps agent learn escape directions)
        # Normalized: -1 to 1 range
        if nearest_dangerous_dist < 100:
            ghost_dir_x = max(-1, min(1, nearest_dangerous_dx / 5.0))
            ghost_dir_y = max(-1, min(1, nearest_dangerous_dy / 5.0))
        else:
            ghost_dir_x = 0
            ghost_dir_y = 0
        features.append(ghost_dir_x)
        features.append(ghost_dir_y)
        
        # Raw distance to nearest ghost (inverted: higher = more danger)
        features.append(max(0, 1.0 - nearest_dangerous_dist / 10.0))
        
        # === ENDGAME HUNTING FEATURES ===
        # Urgency level - how important it is to hunt dots (higher when fewer remain)
        if dots_remaining <= 10:
            urgency = 1.0
        elif dots_remaining <= 20:
            urgency = 0.8
        elif dots_remaining <= 40:
            urgency = 0.6
        elif dots_remaining <= 80:
            urgency = 0.4
        elif dots_remaining <= 120:
            urgency = 0.2
        else:
            urgency = 0.0
        features.append(urgency)
        
        # Direction to nearest dot (actual BFS-found dot, not quadrant center)
        # This gives the agent a direct signal of where to go
        if nearest_dot_pos is not None:
            dot_dir_x = (nearest_dot_pos[0] - px_tile) / 14.0  # Normalize
            dot_dir_y = (nearest_dot_pos[1] - py_tile) / 18.0
            features.append(max(-1, min(1, dot_dir_x)))
            features.append(max(-1, min(1, dot_dir_y)))
        else:
            features.append(0.0)
            features.append(0.0)
        
        # === NEW ENDGAME HUNTING FEATURES ===
        # Count dots reachable if we go in each direction (up, down, left, right)
        # This directly answers "which way has more dots to collect?"
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
        dots_per_direction = self._count_dots_per_direction(px_tile, py_tile, game_map, directions)
        
        # Normalize by remaining dots (so features are 0-1)
        max_dots = max(dots_per_direction) if dots_per_direction else 1
        total_dots_reachable = sum(dots_per_direction)
        for dot_count in dots_per_direction:
            # Normalize: what fraction of reachable dots are in this direction?
            if total_dots_reachable > 0:
                features.append(dot_count / total_dots_reachable)
            else:
                features.append(0.25)  # Equal if no dots found
        
        # One-hot encoding of best direction to go (most dots reachable)
        best_dir_idx = dots_per_direction.index(max_dots) if max_dots > 0 else 0
        for i in range(4):
            features.append(1.0 if i == best_dir_idx else 0.0)
        
        # Scattered dot penalty - how spread out are the remaining dots?
        # High value = dots are scattered (harder to collect efficiently)
        # Low value = dots are clustered (easier to collect)
        if total_dots_reachable > 0:
            # Calculate how evenly distributed dots are across directions
            # If perfectly even (0.25, 0.25, 0.25, 0.25), scattered = 1.0
            # If all in one direction (1, 0, 0, 0), scattered = 0.0
            evenness = sum((d / total_dots_reachable - 0.25) ** 2 for d in dots_per_direction)
            # evenness ranges from 0 (perfectly even) to 0.1875 (all in one direction)
            scattered = 1.0 - (evenness / 0.1875)  # Invert: 0 = clustered, 1 = scattered
            features.append(scattered)
        else:
            features.append(0.5)
        
        return np.array(features, dtype=np.float32)
    
    def _count_dots_per_quadrant(self, game_map: List) -> List[int]:
        """Count remaining dots in each quadrant of the map."""
        if not game_map:
            return [0, 0, 0, 0]
        
        # Quadrants: top-left, top-right, bottom-left, bottom-right
        # Map is 28 wide, 36 tall
        mid_x, mid_y = 14, 18
        quadrant_dots = [0, 0, 0, 0]
        
        for y, row in enumerate(game_map):
            for x, tile in enumerate(row):
                if tile in (16, 20):  # Dot or power pill
                    if x < mid_x and y < mid_y:
                        quadrant_dots[0] += 1  # top-left
                    elif x >= mid_x and y < mid_y:
                        quadrant_dots[1] += 1  # top-right
                    elif x < mid_x and y >= mid_y:
                        quadrant_dots[2] += 1  # bottom-left
                    else:
                        quadrant_dots[3] += 1  # bottom-right
        
        return quadrant_dots
    
    def _find_nearest_dot_bfs(self, px: int, py: int, game_map: List, max_dist: int = 30) -> int:
        """Find distance to nearest dot using BFS (actual pathfinding)."""
        dist, _ = self._find_nearest_dot_bfs_with_pos(px, py, game_map, max_dist)
        return dist
    
    def _find_nearest_dot_bfs_with_pos(self, px: int, py: int, game_map: List, max_dist: int = 30) -> tuple:
        """Find distance and position of nearest dot using BFS (actual pathfinding).
        
        Returns:
            (distance, (dot_x, dot_y)) or (max_dist, None) if no dot found
        """
        if not game_map:
            return max_dist, None
        
        from collections import deque
        
        visited = {(px, py)}
        queue = deque([(px, py, 0)])  # (x, y, distance)
        
        while queue:
            x, y, dist = queue.popleft()
            
            if dist > max_dist:
                return max_dist, None
            
            # Check if this tile has a dot
            if 0 <= y < len(game_map) and 0 <= x < len(game_map[0]):
                tile = game_map[y][x]
                if tile in (16, 20) and dist > 0:  # Found a dot (not starting position)
                    return dist, (x, y)
            
            # Explore neighbors
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited:
                    if 0 <= ny < len(game_map) and 0 <= nx < len(game_map[0]):
                        tile = game_map[ny][nx]
                        if tile in (64, 16, 20):  # Passable
                            visited.add((nx, ny))
                            queue.append((nx, ny, dist + 1))
        
        return max_dist, None
    
    def _count_dots_per_direction(self, px: int, py: int, game_map: List, 
                                   directions: List[tuple], search_depth: int = 20) -> List[int]:
        """Count how many dots are reachable if we first step in each direction.
        
        This helps the agent decide which direction to go at a junction by showing
        how many dots can be collected in each direction.
        
        Args:
            px, py: Current position
            directions: List of (dx, dy) direction tuples to check
            search_depth: How far to search in each direction
            
        Returns:
            List of dot counts for each direction
        """
        from collections import deque
        
        if not game_map:
            return [0] * len(directions)
        
        dot_counts = []
        
        for dx, dy in directions:
            # First step in this direction
            start_x, start_y = px + dx, py + dy
            
            # Check if first step is valid
            if not (0 <= start_y < len(game_map) and 0 <= start_x < len(game_map[0])):
                dot_counts.append(0)
                continue
            
            tile = game_map[start_y][start_x]
            if tile not in (64, 16, 20):  # Not passable
                dot_counts.append(0)
                continue
            
            # BFS from this starting point to count reachable dots
            visited = {(px, py), (start_x, start_y)}
            queue = deque([(start_x, start_y, 1)])
            dots_found = 0
            
            # Count dot at starting position
            if tile in (16, 20):
                dots_found += 1
            
            while queue:
                x, y, dist = queue.popleft()
                
                if dist >= search_depth:
                    continue
                
                # Explore neighbors
                for ndx, ndy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nx, ny = x + ndx, y + ndy
                    if (nx, ny) not in visited:
                        if 0 <= ny < len(game_map) and 0 <= nx < len(game_map[0]):
                            ntile = game_map[ny][nx]
                            if ntile in (64, 16, 20):  # Passable
                                visited.add((nx, ny))
                                queue.append((nx, ny, dist + 1))
                                if ntile in (16, 20):  # Dot or power pill
                                    dots_found += 1
            
            dot_counts.append(dots_found)
        
        return dot_counts
    
    def _find_nearest_dot_distance(self, px: int, py: int, dx: int, dy: int,
                                    game_map: List, max_dist: int = 15) -> float:
        """Find distance to nearest dot in given direction."""
        for dist in range(1, max_dist + 1):
            tx, ty = px + dx * dist, py + dy * dist
            if not (0 <= ty < len(game_map) and 0 <= tx < len(game_map[0])):
                return max_dist
            tile = game_map[ty][tx]
            if tile in (16, 20):  # Dot or pill
                return dist
            if tile not in (64, 16, 20):  # Wall
                return max_dist
        return max_dist
    
    def _calculate_reward(self, state_dict: Dict) -> Tuple[float, bool, bool]:
        """
        Calculate reward using lessons from tabular Q-learning.
        
        Key principles:
        - Strong positive rewards for eating dots (primary objective)
        - Death penalty based on INACTIVITY, not death itself
          (dying while productive is fine, dying while doing nothing is bad)
        - Gentle ghost proximity penalties (only very close)
        - Progressive milestone bonuses
        """
        if not state_dict:
            return -50, True, True
        
        reward = 0.0
        death_occurred = False
        game_over = False
        
        pacman = state_dict.get('pacman', {})
        status = state_dict.get('status', {})
        
        # === Death Handling ===
        if not pacman.get('alive', True) or pacman.get('just_died', False):
            death_occurred = True
            
            dots_remaining = status.get('dots_remaining', 244)
            dots_eaten_total = 244 - dots_remaining
            
            if self.prev_lives <= 0:
                game_over = True
            
            # Death penalty scales based on:
            # 1. How close we were to winning (dying with few dots left = BAD)
            # 2. How productive this life was
            
            # Base penalty for inactivity
            if self.dots_this_life >= 10:
                base_penalty = 0
            elif self.dots_this_life >= 5:
                base_penalty = -5
            elif self.dots_this_life >= 1:
                base_penalty = -10
            elif self.steps_this_life >= 50:
                base_penalty = -20
            else:
                base_penalty = -50
            
            # Extra penalty for dying when close to winning
            # (moderate - don't make agent too scared to try)
            if dots_remaining <= 10:
                endgame_penalty = -40  # Close to winning
            elif dots_remaining <= 20:
                endgame_penalty = -25
            elif dots_remaining <= 40:
                endgame_penalty = -15
            else:
                endgame_penalty = 0
            
            death_penalty = base_penalty + endgame_penalty
            reward = death_penalty
            
            if game_over:
                print(f"[GAME OVER] Deaths: {self.deaths_this_episode + 1}, Total dots: {dots_eaten_total}, "
                      f"Dots this life: {self.dots_this_life}, Penalty: {death_penalty:.0f}")
            
            return reward, death_occurred, game_over
        
        # === Living Rewards ===
        curr_score = status.get('score', 0)
        score_diff = curr_score - self.prev_score
        dots_remaining = status.get('dots_remaining', 244)
        self.dots_eaten_total = 244 - dots_remaining
        
        # === Position tracking and oscillation detection ===
        pacman_x = pacman.get('x', 0) // 8
        pacman_y = pacman.get('y', 0) // 8
        current_tile = (pacman_x, pacman_y)
        
        oscillation_penalty = 0
        exploration_bonus = 0
        
        # Only process when we've moved to a NEW tile
        if current_tile != self.last_tile:
            # Mild oscillation detection - since we block reverse, this should be rare
            if current_tile in self.tile_history[-4:]:
                self.oscillation_count += 1
                # Small penalty - reverse is already blocked so this is less critical
                oscillation_penalty = -1.0 * min(self.oscillation_count, 5)
            else:
                self.oscillation_count = max(0, self.oscillation_count - 1)
            
            # Small reward for visiting new tiles
            if current_tile not in self.visited_tiles:
                exploration_bonus = 1.0  # Modest bonus for new tile
                self.visited_tiles.add(current_tile)
                self.oscillation_count = max(0, self.oscillation_count - 1)
            
            # Track nearest dot for movement reward
            game_map = state_dict.get('map', [])
            nearest_dot = self._find_nearest_dot_bfs(pacman_x, pacman_y, game_map, max_dist=15)
            
            # Reward for moving toward dots - STRONGER when fewer dots remain
            if self.prev_nearest_dot_dist is not None and nearest_dot < 15:
                # Calculate urgency multiplier based on dots remaining
                dots_remaining = 244 - self.dots_eaten_total
                if dots_remaining <= 20:
                    urgency = 4.0  # Very strong hunting in endgame
                elif dots_remaining <= 40:
                    urgency = 3.0
                elif dots_remaining <= 60:
                    urgency = 2.0
                elif dots_remaining <= 100:
                    urgency = 1.5
                else:
                    urgency = 1.0
                
                if nearest_dot < self.prev_nearest_dot_dist:
                    reward += 0.5 * urgency  # Getting closer to a dot
                elif nearest_dot > self.prev_nearest_dot_dist + 1:
                    reward -= 0.3 * urgency  # Moving away from dots (penalize more in endgame)
            
            self.prev_nearest_dot_dist = nearest_dot
            
            # === ENDGAME HUNTING REWARD ===
            # In endgame (< 50 dots), give bonus for moving toward direction with more dots
            dots_remaining = 244 - self.dots_eaten_total
            if dots_remaining <= 50 and dots_remaining > 0:
                # Calculate dots per direction from new position
                directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
                dots_per_dir = self._count_dots_per_direction(pacman_x, pacman_y, game_map, directions, search_depth=15)
                
                # Determine which direction we moved
                if self.last_tile is not None:
                    move_dx = pacman_x - self.last_tile[0]
                    move_dy = pacman_y - self.last_tile[1]
                    
                    # Map movement to direction index
                    dir_map = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
                    moved_dir = dir_map.get((move_dx, move_dy), -1)
                    
                    if moved_dir >= 0 and sum(dots_per_dir) > 0:
                        # Calculate what fraction of dots are in the direction we moved
                        total_dots = sum(dots_per_dir)
                        dots_in_moved_dir = dots_per_dir[moved_dir]
                        best_dots = max(dots_per_dir)
                        
                        # Reward for moving toward more dots
                        if dots_in_moved_dir == best_dots and best_dots > 0:
                            # Moved toward the best direction!
                            endgame_hunt_bonus = 3.0 * (1.0 - dots_remaining / 50.0)  # Stronger as fewer dots
                            reward += endgame_hunt_bonus
                        elif dots_in_moved_dir < best_dots * 0.3:
                            # Moved toward a direction with few dots - mild penalty
                            endgame_hunt_penalty = -1.0 * (1.0 - dots_remaining / 50.0)
                            reward += endgame_hunt_penalty
            
            # Update tile history
            self.tile_history.append(current_tile)
            if len(self.tile_history) > self.tile_history_max:
                self.tile_history.pop(0)
            
            self.last_tile = current_tile
        
        reward += oscillation_penalty + exploration_bonus
        
        if score_diff == 10:  # Dot - PRIMARY OBJECTIVE
            # Base dot reward scales up as fewer dots remain (hunt them down!)
            dots_remaining = 244 - self.dots_eaten_total
            if dots_remaining <= 10:
                dot_reward = 25  # Final dots are very valuable
            elif dots_remaining <= 20:
                dot_reward = 20
            elif dots_remaining <= 40:
                dot_reward = 15
            elif dots_remaining <= 80:
                dot_reward = 12
            else:
                dot_reward = 10
            
            reward += dot_reward
            self.dots_this_life += 1
            
            # Milestone bonuses
            if self.dots_eaten_total >= 235 and not self.milestone_235_given:
                reward += 100  # Big bonus for getting close to winning
                self.milestone_235_given = True
                print(f"[MILESTONE] 235+ dots! +100 bonus - FINISH IT!")
            elif self.dots_eaten_total >= 220 and not self.milestone_220_given:
                reward += 50
                self.milestone_220_given = True
                print(f"[MILESTONE] 220+ dots! +50 bonus")
            elif self.dots_eaten_total >= 200 and not self.milestone_200_given:
                reward += 30
                self.milestone_200_given = True
                print(f"[MILESTONE] 200+ dots! +30 bonus")
            elif self.dots_eaten_total >= 100 and not hasattr(self, 'milestone_100_given'):
                reward += 20
                self.milestone_100_given = True
            elif self.dots_eaten_total >= 50 and not hasattr(self, 'milestone_50_given'):
                reward += 10
                self.milestone_50_given = True
                
        elif score_diff == 50:  # Power pill
            reward += 20
            self.dots_this_life += 1
        elif score_diff >= 200:  # Ghost eaten
            reward += 30
        elif score_diff > 0:  # Fruit
            reward += score_diff // 10
        
        # Ghost proximity - moderate penalties (not too harsh to avoid over-caution)
        # Only scale up slightly in endgame
        dots_remaining = 244 - self.dots_eaten_total
        if dots_remaining <= 20:
            danger_multiplier = 1.5  # Slightly more careful in endgame
        elif dots_remaining <= 50:
            danger_multiplier = 1.2
        else:
            danger_multiplier = 1.0
        
        min_dangerous_dist = 100
        frightened_nearby = 0
        num_close_ghosts = 0  # Track how many ghosts are nearby
        
        for ghost in state_dict.get('ghosts', []):
            gs = ghost.get('state', 0)
            gx = ghost.get('x', 0) // 8
            gy = ghost.get('y', 0) // 8
            dist = abs(gx - pacman_x) + abs(gy - pacman_y)
            
            if gs == 3:  # Frightened - encourage chasing
                if dist <= 4:
                    frightened_nearby += 1
                    reward += 3  # Encourage chasing frightened ghosts
            elif gs in (0, 1, 2):  # Dangerous
                if dist < min_dangerous_dist:
                    min_dangerous_dist = dist
                if dist <= 4:
                    num_close_ghosts += 1
        
        # Ghost proximity penalties - moderate (don't overwhelm dot rewards)
        if frightened_nearby == 0:
            if min_dangerous_dist <= 1:
                reward -= 8 * danger_multiplier   # DANGER! Very close
            elif min_dangerous_dist <= 2:
                reward -= 4 * danger_multiplier   # Close danger
            elif min_dangerous_dist <= 3:
                reward -= 2 * danger_multiplier   # Getting close
            # No penalty for distance 4 - let agent take some risks
            
            # Small extra penalty for multiple ghosts nearby
            if num_close_ghosts >= 2:
                reward -= 3 * danger_multiplier
        
        # Survival bonus - reward for staying alive, especially in endgame
        if dots_remaining <= 50:
            reward += 0.5  # Small bonus per step when close to winning
        if dots_remaining <= 20:
            reward += 1.0  # Extra bonus in final stretch
        
        return reward, death_occurred, game_over
    
    def is_oscillating(self) -> bool:
        """Check if agent is currently stuck in an oscillation pattern."""
        return self.oscillation_count >= 3
    
    def get_anti_oscillation_mask(self, state_dict: Optional[Dict] = None) -> np.ndarray:
        """
        Get action mask that discourages oscillation by blocking reverse direction
        when oscillation is detected.
        """
        if state_dict is None:
            state_dict = self._get_state()
        
        # Start with normal open directions mask
        mask = self.get_action_mask(state_dict)
        
        # If oscillating and we have a current direction, discourage reverse
        if self.oscillation_count >= 2 and self.current_direction is not None:
            reverse = self.reverse_dir.get(self.current_direction)
            if reverse:
                reverse_idx = self.ACTION_TO_IDX[reverse]
                # Set reverse direction to very low probability (but not zero)
                # so agent can still escape if truly trapped
                mask[reverse_idx] = 0.1
                
                # Renormalize if needed
                if mask.sum() > 0:
                    # Boost non-reverse directions
                    for i in range(len(mask)):
                        if i != reverse_idx and mask[i] > 0:
                            mask[i] = 1.0
        
        return mask
    
    def _get_open_directions(self, state_dict: Optional[Dict] = None) -> List[str]:
        """
        Get list of all open (passable) directions from current position.
        Uses map tiles to determine which directions are passable.
        """
        if state_dict is None:
            state_dict = self._get_state()
        
        if not state_dict:
            return self.ACTIONS
        
        pacman = state_dict.get('pacman', {})
        px = pacman.get('x', 0) // 8  # Convert to tile coords
        py = pacman.get('y', 0) // 8
        game_map = state_dict.get('map', [])
        
        if not game_map:
            return self.ACTIONS
        
        open_dirs = []
        
        for action, (dx, dy) in self.dir_to_vec.items():
            nx, ny = px + dx, py + dy
            if 0 <= ny < self.MAP_HEIGHT and 0 <= nx < self.MAP_WIDTH:
                tile = game_map[ny][nx]
                if tile in (64, 16, 20):  # Passable: space, dot, pill
                    open_dirs.append(action)
        
        return open_dirs if open_dirs else self.ACTIONS
    
    def _get_forward_directions(self, open_dirs: List[str]) -> List[str]:
        """Get directions that are NOT reversing."""
        if self.current_direction is None:
            return open_dirs
        
        reverse = self.reverse_dir.get(self.current_direction)
        forward = [d for d in open_dirs if d != reverse]
        
        return forward if forward else open_dirs
    
    def is_junction(self, state_dict: Optional[Dict] = None) -> bool:
        """
        Determine if Pacman is at a junction (decision point).
        
        A junction is where Pacman has 2+ valid directions to choose from
        (not counting going backwards).
        
        This helps the agent focus learning on important decision points.
        """
        if state_dict is None:
            state_dict = self._get_state()
        
        if not state_dict:
            return True  # Assume junction if no state
        
        open_dirs = self._get_open_directions(state_dict)
        forward_dirs = self._get_forward_directions(open_dirs)
        
        # Junction = 2+ forward options
        return len(forward_dirs) >= 2
    
    def is_new_junction(self, state_dict: Optional[Dict] = None) -> bool:
        """
        Determine if Pacman is at a NEW junction (hasn't made a decision here yet).
        
        This prevents the agent from making multiple decisions at the same
        position while waiting for the game to move Pacman.
        """
        if state_dict is None:
            state_dict = self._get_state()
        
        if not state_dict:
            return True
        
        # First check if it's a junction at all
        if not self.is_junction(state_dict):
            return False
        
        # Get current tile position
        pacman = state_dict.get('pacman', {})
        px = pacman.get('x', 0) // 8
        py = pacman.get('y', 0) // 8
        current_pos = (px, py)
        
        # Check if we already made a decision at this position
        if self.last_decision_pos == current_pos:
            return False  # Same position, already decided
        
        return True  # New junction position
    
    def mark_decision_made(self, state_dict: Optional[Dict] = None, chosen_direction: str = None):
        """
        Mark that we've made a decision at the current position.
        Call this after making a junction decision.
        
        NOTE: We do NOT update arrival_direction here! That should only happen
        when we actually LEAVE this position and arrive at a new tile.
        Otherwise the action mask gets confused while we're still waiting at the junction.
        
        Args:
            state_dict: Game state
            chosen_direction: The direction chosen at this junction (stored for when we move)
        """
        if state_dict is None:
            state_dict = self._get_state()
        
        if state_dict:
            pacman = state_dict.get('pacman', {})
            px = pacman.get('x', 0) // 8
            py = pacman.get('y', 0) // 8
            self.last_decision_pos = (px, py)
            
            # Store the chosen direction - arrival_direction will be updated
            # when we actually move to a new tile (in the training loop)
            # DO NOT update arrival_direction here!
    
    def get_forward_direction(self, state_dict: Optional[Dict] = None) -> Optional[str]:
        """
        Get the automatic direction for hallways and corners.
        
        Returns:
            - The only forward direction (hallway)
            - The only available direction when at a corner/dead-end (must turn)
            - None if at a junction (2+ forward options, needs DQN decision)
        
        This handles:
        1. Hallways: Only 1 forward option -> take it
        2. Corners: 0 forward options but 1 available (must turn) -> take it
        3. Dead ends: Must reverse -> take the only option
        """
        if state_dict is None:
            state_dict = self._get_state()
        
        if not state_dict:
            return None
        
        # Get all available directions from map
        available = self._get_open_directions(state_dict)
        forward_options = self._get_forward_directions(available)
        
        # Case 1: Exactly 1 forward option (hallway) -> take it automatically
        if len(forward_options) == 1:
            return forward_options[0]
        
        # Case 2: 0 forward options (corner/dead-end) -> must turn or reverse
        # This handles corners where the hallway bends (e.g., going right but must turn up)
        if len(forward_options) == 0:
            # Take any available direction (there should be at least 1)
            if len(available) >= 1:
                # Prefer non-reverse if possible, but if only reverse exists, take it
                return available[0]
        
        # Case 3: 2+ forward options (junction) -> needs DQN decision
        return None
    
    def get_action_mask(self, state_dict: Optional[Dict] = None, allow_reverse: bool = False) -> np.ndarray:
        """
        Get a mask of valid actions (1 = valid, 0 = invalid).
        
        By default, blocks the reverse direction (like real Pacman where you
        can't instantly turn around). This prevents useless back-and-forth movement.
        
        Args:
            state_dict: Game state dict
            allow_reverse: If True, allow reversing direction (only for special cases)
        """
        if state_dict is None:
            state_dict = self._get_state()
        
        if not state_dict:
            return np.ones(self.NUM_ACTIONS, dtype=np.float32)
        
        # Get open directions from map
        open_dirs = self._get_open_directions(state_dict)
        
        # Block the reverse of ARRIVAL direction (not current_direction)
        # This prevents going back the way we came to this junction
        # Example: if we arrived going UP, we can't go DOWN (back where we came from)
        if not allow_reverse and self.arrival_direction is not None:
            reverse_of_arrival = self.reverse_dir.get(self.arrival_direction)
            if reverse_of_arrival and reverse_of_arrival in open_dirs:
                forward_dirs = [d for d in open_dirs if d != reverse_of_arrival]
                # Only use forward_dirs if there's at least one option
                if forward_dirs:
                    open_dirs = forward_dirs
        
        mask = np.zeros(self.NUM_ACTIONS, dtype=np.float32)
        
        # Set mask based on passable directions
        for direction in open_dirs:
            idx = self.ACTION_TO_IDX[direction]
            mask[idx] = 1.0
        
        # If no valid moves (shouldn't happen), allow all
        if mask.sum() == 0:
            mask = np.ones(self.NUM_ACTIONS, dtype=np.float32)
        
        return mask
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics for the current episode."""
        return {
            'episode': self.episode,
            'steps': self.steps,
            'deaths': self.deaths_this_episode,
            'score': self.prev_score,
            'lives': self.prev_lives,
            'dots_eaten': self.dots_eaten_total
        }