import requests

class PacmanEnv:
    """
    Environment for junction-based agent with 3-lives system.

    Reward function focused on COLLECTING ALL DOTS:
    - Progressive bonuses for dot milestones (200+, 220+, 235+ dots)
    - Massive bonus for winning (eating all 244 dots)
    - Death penalty scales with how FAST you died
    - The goal is to collect ALL dots and win the game
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
        self.dots_at_episode_start = 244

        # Track survival time per life
        self.steps_this_life = 0
        self.dots_this_life = 0

        # Track dot milestones for progressive bonuses
        self.dots_eaten_total = 0
        self.milestone_200_given = False
        self.milestone_220_given = False
        self.milestone_235_given = False

    def reset(self):
        """Reset game - called at start of episode or after death."""
        try:
            self.session.post(f"{self.base_url}/api/restart", timeout=5)
            self.session.post(f"{self.base_url}/api/start", timeout=5)

            state = self._get_state()

            if state:
                current_lives = state.get('status', {}).get('lives', 2)
                current_score = state.get('status', {}).get('score', 0)

                # Detect if this is a new episode (full reset happened)
                if current_lives == 2 and current_score == 0:
                    self.episode += 1
                    self.deaths_this_episode = 0
                    self.dots_at_episode_start = state.get('status', {}).get('dots_remaining', 244)
                    # Reset milestone tracking for new episode
                    self.dots_eaten_total = 0
                    self.milestone_200_given = False
                    self.milestone_220_given = False
                    self.milestone_235_given = False

                self.prev_lives = current_lives
                self.prev_score = current_score
            else:
                self.prev_lives = 2
                self.prev_score = 0

            self.steps = 0
            self.steps_this_life = 0
            self.dots_this_life = 0

            return state
        except Exception as e:
            print(f"[ERROR] Reset: {e}")
            return None

    def step(self, action):
        """Take action and return (state, reward, done, info)."""
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

            # Bonus for winning the round - THIS IS THE ULTIMATE GOAL!
            if state.get('status', {}).get('round_won', False):
                lives_remaining = state.get('status', {}).get('lives', 0)
                # MASSIVE bonus for winning - this is what we want!
                win_bonus = 2000 + (lives_remaining * 1000)
                reward += win_bonus
                print(f"[🏆 WIN! 🏆] Level complete with {lives_remaining} spare lives! Bonus: {win_bonus}")

            self.steps += 1
            self.steps_this_life += 1
            self.prev_score = state.get('status', {}).get('score', 0)
            self.prev_lives = state.get('status', {}).get('lives', 2)

            if death_occurred:
                self.deaths_this_episode += 1
                # Reset per-life counters on death
                self.steps_this_life = 0
                self.dots_this_life = 0

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
        Survival-time based reward function.

        Key insight: Death is inevitable unless you win.
        So we don't punish death itself - we punish dying FAST.

        - Die after 10 steps: BIG penalty (you didn't learn to survive)
        - Die after 500 steps: Small penalty (you survived well, death was coming anyway)
        - Die after 1000+ steps: Nearly no penalty (excellent survival)

        This teaches the agent to SURVIVE LONGER rather than fear death.
        """
        if not state:
            return -100, True, True

        reward = 0
        death_occurred = False
        game_over = False

        pacman = state.get('pacman', {})
        status = state.get('status', {})

        # === DEATH HANDLING - SURVIVAL TIME BASED ===
        if not pacman.get('alive', True) or pacman.get('just_died', False):
            death_occurred = True

            dots_remaining = status.get('dots_remaining', 244)
            dots_eaten_total = 244 - dots_remaining

            # Game over check - we had 0 spare lives before dying
            if self.prev_lives <= 0:
                game_over = True

            # === SURVIVAL-BASED DEATH PENALTY ===
            # The longer you survived, the smaller the penalty
            #
            # Formula: penalty = -max_penalty * (1 - survival_factor)
            # where survival_factor = min(1.0, steps / target_steps)
            #
            # Target: survive ~500 steps per life = minimal penalty
            # Die at 0 steps: -200 penalty
            # Die at 250 steps: -100 penalty
            # Die at 500+ steps: ~0 penalty

            target_survival_steps = 500
            survival_factor = min(1.0, self.steps_this_life / target_survival_steps)

            # Also consider dots eaten this life as survival metric
            # If you ate many dots, you were being productive
            dots_factor = min(1.0, self.dots_this_life / 80)  # 80 dots = good progress

            # Combined survival score (weighted average)
            combined_survival = 0.6 * survival_factor + 0.4 * dots_factor

            # Base penalty that decreases with survival
            max_penalty = 200
            death_penalty = -max_penalty * (1.0 - combined_survival)

            # Minimum penalty even for good survival (death should still be slightly negative)
            death_penalty = min(death_penalty, -10)

            reward = death_penalty

            if game_over:
                print(f"[GAME OVER] Deaths: {self.deaths_this_episode + 1}, Total dots: {dots_eaten_total}, "
                      f"Last life: {self.steps_this_life} steps, {self.dots_this_life} dots, Penalty: {death_penalty:.0f}")
            else:
                print(f"[DEATH] Life lost! Steps survived: {self.steps_this_life}, Dots this life: {self.dots_this_life}, "
                      f"Penalty: {death_penalty:.0f}, Survival: {combined_survival:.1%}")

            return reward, death_occurred, game_over

        # === LIVING REWARDS ===

        # Score changes (dots, pills, ghosts, fruit)
        curr_score = status.get('score', 0)
        score_diff = curr_score - self.prev_score

        # Track total dots eaten this episode
        dots_remaining = status.get('dots_remaining', 244)
        self.dots_eaten_total = 244 - dots_remaining
        
        if score_diff == 10:  # Dot
            reward += 15
            self.dots_this_life += 1
            
            # === PROGRESSIVE DOT MILESTONE BONUSES ===
            # These incentivize the agent to push for more dots
            
            if self.dots_eaten_total >= 235 and not self.milestone_235_given:
                # SO CLOSE to winning! Massive bonus
                reward += 500
                self.milestone_235_given = True
                print(f"[MILESTONE] 235+ dots! Almost there! +500 bonus")
            elif self.dots_eaten_total >= 220 and not self.milestone_220_given:
                # Getting very close
                reward += 200
                self.milestone_220_given = True
                print(f"[MILESTONE] 220+ dots! Keep going! +200 bonus")
            elif self.dots_eaten_total >= 200 and not self.milestone_200_given:
                # Good progress
                reward += 100
                self.milestone_200_given = True
                print(f"[MILESTONE] 200+ dots! Great progress! +100 bonus")
            
            # Extra per-dot bonus in endgame (last 30 dots)
            if dots_remaining <= 30:
                reward += 10  # Extra incentive for each dot in endgame
            elif dots_remaining <= 50:
                reward += 5   # Moderate extra incentive
                
        elif score_diff == 50:  # Power pill
            reward += 40
            self.dots_this_life += 1
        elif score_diff >= 200:  # Ghost eaten
            ghost_bonus = 100 + score_diff  # 300-1700 for ghosts
            reward += ghost_bonus
            print(f"[GHOST EATEN] +{ghost_bonus}")
        elif score_diff > 0:  # Fruit
            reward += score_diff // 5

        # Small step cost to encourage efficiency
        reward -= 0.2

        # === GHOST PROXIMITY PENALTIES ===
        # Still penalize being close to ghosts to encourage avoidance
        # But these are smaller than before since death penalty is now survival-based
        px = pacman.get('x', 0) // 8
        py = pacman.get('y', 0) // 8

        min_dangerous_dist = 100
        frightened_nearby = 0

        for ghost in state.get('ghosts', []):
            gs = ghost.get('state', 0)
            gx = ghost.get('x', 0) // 8
            gy = ghost.get('y', 0) // 8
            dist = abs(gx - px) + abs(gy - py)

            if gs == 3:  # Frightened
                if dist <= 6:
                    frightened_nearby += 1
                    reward += 5  # Encourage hunting
            elif gs in (0, 1, 2):  # Dangerous
                min_dangerous_dist = min(min_dangerous_dist, dist)

        # Graduated danger penalties - but smaller now
        # These serve as immediate feedback to help learn ghost avoidance
        if frightened_nearby == 0:  # Only penalize if not in power mode
            if min_dangerous_dist <= 1:
                reward -= 15  # Very close
            elif min_dangerous_dist <= 2:
                reward -= 8   # Dangerous
            elif min_dangerous_dist <= 3:
                reward -= 4   # Risky
            elif min_dangerous_dist <= 5:
                reward -= 1   # Slightly risky

        # === SURVIVAL BONUS ===
        # Small bonus for staying alive longer - reinforces survival behavior
        if self.steps_this_life > 100:
            # Tiny bonus that accumulates
            reward += 0.5
        if self.steps_this_life > 300:
            reward += 0.5

        return reward, death_occurred, game_over

    def get_episode_stats(self):
        """Get statistics for the current episode."""
        return {
            'episode': self.episode,
            'steps': self.steps,
            'deaths': self.deaths_this_episode,
            'score': self.prev_score,
            'lives': self.prev_lives,
            'steps_this_life': self.steps_this_life,
            'dots_this_life': self.dots_this_life
        }
