import requests
from collections import deque

class PacmanEnv:
    """
    Environment for junction-based agent.
    Tracks rewards and signals junction arrivals.
    """
    
    def __init__(self, url="http://127.0.0.1:8080"):
        self.base_url = url
        self.session = requests.Session()
        
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry = Retry(total=3, backoff_factor=0.1)
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        
        self.episode = 0
        self.steps = 0
        self.prev_score = 0
        self.prev_lives = 3
    
    def reset(self):
        """Reset game"""
        try:
            self.session.post(f"{self.base_url}/api/restart", timeout=5)
            self.session.post(f"{self.base_url}/api/start", timeout=5)
            
            state = self._get_state()
            
            self.episode += 1
            self.steps = 0
            self.prev_score = 0
            self.prev_lives = state.get('status', {}).get('lives', 3) if state else 3
            
            return state
        except Exception as e:
            print(f"[ERROR] Reset: {e}")
            return None
    
    def step(self, action):
        """Take action"""
        try:
            resp = self.session.post(
                f"{self.base_url}/api/step",
                json={"direction": action},
                timeout=5
            )
            state = resp.json()
            
            reward = self._calculate_reward(state)
            
            done = (
                state.get('status', {}).get('game_over', False) or
                not state.get('pacman', {}).get('alive', True) or
                state.get('status', {}).get('round_won', False)
            )
            
            if state.get('status', {}).get('round_won', False):
                reward += 1000
                print(f"[WIN!] Level complete!")
            
            self.steps += 1
            self.prev_score = state.get('status', {}).get('score', 0)
            self.prev_lives = state.get('status', {}).get('lives', 3)
            
            return state, reward, done, {}
            
        except Exception as e:
            print(f"[ERROR] Step: {e}")
            return None, -200, True, {}
    
    def _get_state(self):
        try:
            return self.session.get(f"{self.base_url}/api/state", timeout=5).json()
        except:
            return None
    
    def _calculate_reward(self, state):
        """
        Reward function optimized for junction-based learning.
        Rewards are accumulated between junctions.
        """
        if not state:
            return -200
        
        reward = 0
        
        # Death
        if not state.get('pacman', {}).get('alive', True):
            return -300  # Strong death penalty
        
        # Life lost (but not dead yet)
        curr_lives = state.get('status', {}).get('lives', 3)
        if curr_lives < self.prev_lives:
            reward -= 200
        
        # Score changes
        curr_score = state.get('status', {}).get('score', 0)
        diff = curr_score - self.prev_score
        
        if diff == 10:  # Dot
            reward += 10
        elif diff == 50:  # Power pill
            reward += 30
        elif diff >= 200:  # Ghost
            reward += 150
        elif diff > 0:  # Fruit
            reward += diff // 2
        
        # Small step cost (encourages efficiency)
        reward -= 0.5
        
        # Ghost proximity danger
        pacman = state.get('pacman', {})
        px = pacman.get('x', 0) // 8
        py = pacman.get('y', 0) // 8
        
        for ghost in state.get('ghosts', []):
            gs = ghost.get('state', 0)
            if gs in (1, 2):  # Dangerous
                gx = ghost.get('x', 0) // 8
                gy = ghost.get('y', 0) // 8
                dist = abs(gx - px) + abs(gy - py)
                
                if dist <= 2:
                    reward -= 15
                elif dist <= 4:
                    reward -= 5
        
        return reward
