# qlearning/config.py

# Training Hyperparameters
ALPHA = 0.2          # Learning rate
GAMMA = 0.95         # Discount factor
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_MIN = 0.05   # Minimum exploration rate
EPSILON_DECAY = 0.995 # Decay rate per episode (slower for thorough exploration)
LEARNING_RATE_DANGER = 0.4

# Reward Configuration
REWARDS = {
    # Core Game Events
    'step': -1,              # Living penalty (encourages speed)
    'dot': 15,               # Eating a dot
    'pill': 50,              # Eating a power pill
    'ghost_eaten': 200,      # Eating a ghost
    'death': -200,           # Dying
    'level_complete': 1000,  # Clearing the level
    
    # Movement & Exploration (Anti-Stuck)
    'stuck_penalty': -10,    # Penalty for staying in same 1-tile radius for too long
    'wall_collision': -2,    # Trying to move into a wall
    'reverse_penalty': -5,   # Reversing direction immediately
    'new_tile': 2,           # Visiting a tile not visited recently
    
    # Ghost Avoidance (Distance based)
    'ghost_death_zone': -5, # Ghost within 2 tiles
    'ghost_danger': -2,      # Ghost within 5 tiles
    'ghost_safe': 0,         # Ghost further away
    
    # Ghost Hunting (When powered)
    'hunt_bonus': 10         # Bonus for being close to frightened ghost
}

# Training Settings
EPISODES = 1000
STUCK_WINDOW = 10        # Number of steps to check for being stuck
STUCK_THRESHOLD = 2      # Max unique tiles visited in window to count as stuck
