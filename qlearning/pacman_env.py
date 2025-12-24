import requests
import json
import time
import config

class PacmanEnv:
    def __init__(self, url="http://127.0.0.1:8080"):
        self.base_url = url
        self.session = requests.Session()
        from requests.adapters import HTTPAdapter
        from requests.packages.urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.0,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.step_count = 0
        self.episode_count = 0
        
        # Track previous state for shaped rewards
        self.prev_pacman_pos = None
        self.prev_nearest_ghost_dist = None
        self.prev_nearest_dot_dist = None
        
        print(f"[ENV INIT] Connected to {url}")

    def reset(self):
        """Resets the game and returns the initial state."""
        try:
            # Restart game
            self.session.post(f"{self.base_url}/api/restart", timeout=5.0)
            # Start game
            self.session.post(f"{self.base_url}/api/start", timeout=5.0)
            
            state = self.get_state()
            self.step_count = 0
            self.episode_count += 1
            
            # Reset tracking for shaped rewards
            self.prev_pacman_pos = None
            self.prev_nearest_ghost_dist = None
            self.prev_nearest_dot_dist = None
            
            if state:
                print(f"\n[RESET] Episode {self.episode_count} started")
                print(f"        Pacman at ({state.get('pacman',{}).get('x',0)}, {state.get('pacman',{}).get('y',0)})")
                print(f"        Lives: {state.get('status',{}).get('lives',0)}, Score: {state.get('status',{}).get('score',0)}")
            
            return state
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Reset failed: {e}")
            return None

    def step(self, action):
        """
        Takes a step in the environment.
        """
        try:
            # Get state before move to calculate reward
            prev_state = self.get_state()
            
            # Send action
            payload = {"direction": action}
            response = self.session.post(f"{self.base_url}/api/step", json=payload, timeout=5.0)
            response.raise_for_status()
            
            new_state = response.json()
            
            # Calculate reward
            reward = self._get_reward(prev_state, new_state)
            
            # Check if done
            done = new_state.get('status', {}).get('game_over', False)
            if not new_state.get('pacman', {}).get('alive', True):
                done = True
            
            self.step_count += 1
            
            # Log significant events
            if abs(reward) > 50:
                print(f"[STEP {self.step_count}] Action={action}, Reward={reward:.1f}, Done={done}")
                if reward > 100:
                    print(f"             ✓ Positive event! Score={new_state.get('status',{}).get('score',0)}")
                elif reward < -100:
                    print(f"             ✗ Negative event! Lives={new_state.get('status',{}).get('lives',0)}")
            
            return new_state, reward, done, {}
            
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Step failed: {e}")
            return None, 0, True, {}

    def get_state(self):
        """Fetches the current game state."""
        try:
            response = self.session.get(f"{self.base_url}/api/state", timeout=5.0)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Get state failed: {e}")
            return None

    def _get_reward(self, prev_state, current_state):
        """
        Calculates reward based on game events, ghost proximity, and shaped rewards.
        """
        if not prev_state or not current_state:
            return 0
        
        # Base reward
        reward = config.REWARDS['step']  # -1
        
        # Small survival bonus
        if current_state.get('pacman', {}).get('alive', True):
            reward += 0.5
        
        # Score changes
        prev_score = prev_state.get('status', {}).get('score', 0)
        curr_score = current_state.get('status', {}).get('score', 0)
        score_diff = curr_score - prev_score
        
        reward_breakdown = {"base": config.REWARDS['step'], "survival": 0.5}
        
        if score_diff == 10:
            reward += config.REWARDS['dot']
            reward_breakdown['dot'] = config.REWARDS['dot']
        elif score_diff == 50:
            reward += config.REWARDS['pill']
            reward_breakdown['pill'] = config.REWARDS['pill']
        elif score_diff >= 200:
            reward += config.REWARDS['ghost_eaten']
            reward += 50
            reward_breakdown['ghost'] = config.REWARDS['ghost_eaten'] + 50
        
        # Death
        if not current_state.get('pacman', {}).get('alive', True):
            reward += config.REWARDS['death']
            reward_breakdown['death'] = config.REWARDS['death']
        
        # Get positions
        prev_pacman_x = prev_state.get('pacman', {}).get('x', 0)
        prev_pacman_y = prev_state.get('pacman', {}).get('y', 0)
        curr_pacman_x = current_state.get('pacman', {}).get('x', 0)
        curr_pacman_y = current_state.get('pacman', {}).get('y', 0)
        
        # Ghost proximity penalties (continuous danger signal)
        ghost_penalty = 0
        nearest_ghost_dist = float('inf')
        
        for ghost in current_state.get('ghosts', []):
            ghost_state = ghost.get('state', 0)
            
            if ghost_state not in (2, 3):  # Not frightened, not eaten
                ghost_x = ghost.get('x', 0)
                ghost_y = ghost.get('y', 0)
                
                dist_pixels = abs(ghost_x - curr_pacman_x) + abs(ghost_y - curr_pacman_y)
                dist_tiles = dist_pixels / 8
                
                nearest_ghost_dist = min(nearest_ghost_dist, dist_tiles)
                
                if dist_tiles < 2:
                    ghost_penalty += config.REWARDS['ghost_death_zone']
                elif dist_tiles < 5:
                    ghost_penalty += config.REWARDS['ghost_danger']
        
        reward += ghost_penalty
        if ghost_penalty != 0:
            reward_breakdown['ghost_proximity'] = ghost_penalty
        
        # === SHAPED REWARDS: Reward improvement in position ===
        
        # 1. Reward for moving away from dangerous ghosts
        if self.prev_nearest_ghost_dist is not None and nearest_ghost_dist != float('inf'):
            if nearest_ghost_dist > self.prev_nearest_ghost_dist + 0.5:  # Moved at least 0.5 tiles away
                reward += 2
                reward_breakdown['ghost_escape'] = 2
            elif nearest_ghost_dist < self.prev_nearest_ghost_dist - 0.5:  # Moved closer
                reward -= 1
                reward_breakdown['ghost_approach'] = -1
        
        # 2. Reward for moving toward nearest dot
        nearest_dot_dist = self._find_nearest_dot_distance(current_state)
        
        if self.prev_nearest_dot_dist is not None and nearest_dot_dist is not None:
            if nearest_dot_dist < self.prev_nearest_dot_dist - 0.5:  # Moved closer to food
                reward += 1
                reward_breakdown['food_approach'] = 1
        
        # Update tracking variables for next step
        self.prev_pacman_pos = (curr_pacman_x, curr_pacman_y)
        self.prev_nearest_ghost_dist = nearest_ghost_dist if nearest_ghost_dist != float('inf') else None
        self.prev_nearest_dot_dist = nearest_dot_dist
        
        # Log reward breakdown for significant events
        if abs(reward) > 10 and random.random() < 0.1:
            print(f"[REWARD] Total={reward:.1f}, Breakdown={reward_breakdown}")
        
        return reward
    
    def _find_nearest_dot_distance(self, state):
        """
        Find the Manhattan distance to the nearest dot or pill.
        Returns None if no dots found.
        """
        if not state:
            return None
        
        pacman_x = state.get('pacman', {}).get('x', 0)
        pacman_y = state.get('pacman', {}).get('y', 0)
        
        game_map = state.get('map', [])
        if not game_map:
            return None
        
        min_dist = float('inf')
        
        # Search in a reasonable radius (not the whole map for performance)
        search_radius = 20  # tiles
        
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                tile_x = (pacman_x // 8) + dx
                tile_y = (pacman_y // 8) + dy
                
                # Bounds check
                if 0 <= tile_y < len(game_map) and 0 <= tile_x < len(game_map[0]):
                    tile = game_map[tile_y][tile_x]
                    
                    # Check if it's a dot (16) or pill (20)
                    if tile in (16, 20):
                        dist = abs(dx) + abs(dy)
                        min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else None

import random  # Add at top of file if not already there
