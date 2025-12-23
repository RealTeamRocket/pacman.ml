import requests
import json
import time

class PacmanEnv:
    def __init__(self, url="http://127.0.0.1:8080"):
        self.base_url = url
        self.session = requests.Session()
        # Use Keep-Alive (default) but add retries for robustness
        from requests.adapters import HTTPAdapter
        from requests.packages.urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.0, # Fast retries for local server
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def reset(self):
        """Resets the game and returns the initial state."""
        try:
            # Restart game
            self.session.post(f"{self.base_url}/api/restart", timeout=5.0)
            # Start game
            self.session.post(f"{self.base_url}/api/start", timeout=5.0)
            return self.get_state()
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Pacman API: {e}")
            return None

    def step(self, action):
        """
        Takes a step in the environment.
        Args:
            action: str, one of "up", "down", "left", "right", "none"
        Returns:
            state: dict, the new game state
            reward: float, the reward for this step
            done: bool, whether the game is over
            info: dict, additional info
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
                done = True # Treat death as terminal for this episode? Or just a life lost?
                            # User said: "restart for restarting the game after player dies for next try"
                            # So death is likely terminal for the episode logic here.
            
            return new_state, reward, done, {}
            
        except requests.exceptions.RequestException as e:
            print(f"Error during step: {e}")
            return None, 0, True, {}

    def get_state(self):
        """Fetches the current game state."""
        try:
            response = self.session.get(f"{self.base_url}/api/state", timeout=5.0)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching state: {e}")
            return None

    def _get_reward(self, prev_state, current_state):
        """
        Calculates reward based on score difference, ghost eating, and death.
        """
        if not prev_state or not current_state:
            return 0
            
        prev_score = prev_state.get('status', {}).get('score', 0)
        curr_score = current_state.get('status', {}).get('score', 0)
        
        score_diff = curr_score - prev_score
        
        # Base reward from score
        reward = score_diff
        
        # Bonus for eating ghosts (score increase of 200, 400, 800, 1600)
        # We can just rely on the score diff for this, but maybe add extra incentive?
        # User said "eating ghosts so collecion the big ones".
        # If score_diff is >= 200, it's likely a ghost eat or fruit.
        # Let's add a multiplier or bonus for "interesting" events.
        if score_diff >= 200:
            reward += 50 # Extra bonus for significant achievements
            
        # Penalize death
        if not current_state.get('pacman', {}).get('alive', True):
            reward -= 500 # Massive penalty for dying to prioritize survival
            
        # Small penalty for time passing disabled to encourage exploration
        # reward -= 1
            
        return reward
