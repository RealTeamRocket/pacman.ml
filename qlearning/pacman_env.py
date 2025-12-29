import requests

class PacmanEnv:
    """
    Environment for junction-based agent with 3-lives system.

    An episode spans all 3 lives (2 spare + 1 current):
    - Death (losing a life) is penalized but recoverable
    - Game over (all 3 lives lost) is severely penalized
    - Completing level with any lives is excellent
    - Score and progress accumulate across lives within an episode
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
        self.prev_lives = 2  # 2 spare lives at start
        self.deaths_this_episode = 0

    def reset(self):
        """
        Reset game - called at start of episode or after death.

        The backend handles the 3-lives logic:
        - First reset: full reset (2 spare lives, score=0, all dots)
        - After death with lives remaining: soft reset (positions only, keep score/dots)
        - After 3rd death: full reset
        """
        try:
            self.session.post(f"{self.base_url}/api/restart", timeout=5)
            self.session.post(f"{self.base_url}/api/start", timeout=5)

            state = self._get_state()

            if state:
                current_lives = state.get('status', {}).get('lives', 2)
                current_score = state.get('status', {}).get('score', 0)

                # Detect if this is a new episode (full reset happened)
                # Full reset: lives back to 2 (spare) and score reset to 0
                if current_lives == 2 and current_score == 0:
                    self.episode += 1
                    self.deaths_this_episode = 0

                self.prev_lives = current_lives
                self.prev_score = current_score
            else:
                self.prev_lives = 2
                self.prev_score = 0

            self.steps = 0

            return state
        except Exception as e:
            print(f"[ERROR] Reset: {e}")
            return None

    def step(self, action):
        """
        Take action and return (state, reward, done, info).

        done=True when:
        - Pacman dies (triggers soft/full reset on next reset() call)
        - Round is won (all dots eaten)

        The reward system accounts for the 3-lives structure.
        """
        try:
            resp = self.session.post(
                f"{self.base_url}/api/step",
                json={"direction": action},
                timeout=5
            )
            state = resp.json()

            reward, death_occurred, game_over = self._calculate_reward(state)

            # Episode ends on death or round won
            done = (
                death_occurred or
                state.get('status', {}).get('round_won', False)
            )

            # Bonus for winning the round
            if state.get('status', {}).get('round_won', False):
                # Scale bonus by remaining lives (more lives = better performance)
                lives_remaining = state.get('status', {}).get('lives', 0)
                reward += 500 + (lives_remaining * 200)  # 500-900 based on lives
                print(f"[WIN!] Level complete with {lives_remaining} spare lives!")

            self.steps += 1
            self.prev_score = state.get('status', {}).get('score', 0)
            self.prev_lives = state.get('status', {}).get('lives', 2)

            # Track deaths for info
            if death_occurred:
                self.deaths_this_episode += 1

            info = {
                'deaths_this_episode': self.deaths_this_episode,
                'game_over': game_over,
                'lives': self.prev_lives
            }

            return state, reward, done, info

        except Exception as e:
            print(f"[ERROR] Step: {e}")
            return None, -100, True, {'error': str(e)}

    def _get_state(self):
        try:
            return self.session.get(f"{self.base_url}/api/state", timeout=5).json()
        except:
            return None

    def _calculate_reward(self, state):
        """
        Reward function optimized for 3-lives learning.

        Key insights:
        - Death is bad but recoverable (you have more lives)
        - Game over (3 deaths) is VERY bad - wasted all lives
        - Accumulating score across lives is good
        - Dying late (after collecting lots of dots) is better than dying early

        Returns: (reward, death_occurred, game_over)
        """
        if not state:
            return -100, True, True

        reward = 0
        death_occurred = False
        game_over = False

        pacman = state.get('pacman', {})
        status = state.get('status', {})

        # Check for death - use just_died flag or alive check
        if not pacman.get('alive', True) or pacman.get('just_died', False):
            death_occurred = True

            # Game over is when we die with 0 spare lives (this was our last life)
            # We use prev_lives because that's what we had BEFORE dying
            if self.prev_lives <= 0:
                # Game over - all lives lost - SEVERE penalty
                # This is the worst outcome
                game_over = True
                reward = -500
                print(f"[GAME OVER] All lives lost! Deaths: {self.deaths_this_episode + 1}")
            else:
                # Lost a life but have more - moderate penalty
                # Scale penalty: dying early is worse (less progress made)
                dots_remaining = status.get('dots_remaining', 244)
                dots_eaten = 244 - dots_remaining

                # Base death penalty reduced since we can continue
                # Dying after eating many dots is less bad (progress was made)
                progress_factor = min(1.0, dots_eaten / 100.0)  # 0 to 1
                death_penalty = -150 * (1.0 - 0.3 * progress_factor)  # -150 to -105
                reward = death_penalty

                print(f"[DEATH] Life lost! Spare lives remaining: {self.prev_lives - 1}, dots eaten: {dots_eaten}")

            return reward, death_occurred, game_over

        # === LIVING REWARDS ===

        # Score changes (dots, pills, ghosts, fruit)
        curr_score = status.get('score', 0)
        score_diff = curr_score - self.prev_score

        if score_diff == 10:  # Dot
            reward += 10
        elif score_diff == 50:  # Power pill
            reward += 25
        elif score_diff >= 200:  # Ghost eaten
            reward += 100 + (score_diff - 200) // 2  # More for combo
        elif score_diff > 0:  # Fruit or other
            reward += score_diff // 10

        # Life change detection (gained a life from score?)
        curr_lives = status.get('lives', 2)
        if curr_lives > self.prev_lives:
            reward += 100  # Extra life bonus
            print(f"[EXTRA LIFE] Spare lives: {curr_lives}")

        # Small step cost (encourages efficiency)
        reward -= 0.3

        # Ghost proximity - danger awareness
        px = pacman.get('x', 0) // 8
        py = pacman.get('y', 0) // 8

        min_dangerous_dist = 100
        frightened_nearby = 0

        for ghost in state.get('ghosts', []):
            gs = ghost.get('state', 0)
            gx = ghost.get('x', 0) // 8
            gy = ghost.get('y', 0) // 8
            dist = abs(gx - px) + abs(gy - py)

            if gs == 3:  # Frightened - opportunity!
                if dist <= 5:
                    frightened_nearby += 1
                    reward += 3  # Encourage hunting
            elif gs in (1, 2):  # Dangerous (scatter/chase)
                min_dangerous_dist = min(min_dangerous_dist, dist)

        # Danger penalties (but less severe since death isn't episode-ending)
        if min_dangerous_dist <= 2:
            reward -= 8  # Very close - high danger
        elif min_dangerous_dist <= 4:
            reward -= 3  # Medium danger
        
        return reward, death_occurred, game_over
    
    def get_episode_stats(self):
        """Get statistics for the current episode."""
        return {
            'episode': self.episode,
            'steps': self.steps,
            'deaths': self.deaths_this_episode,
            'score': self.prev_score,
            'lives': self.prev_lives
        }