# Training Hyperparameters
ALPHA = 0.15          # Learning rate (slightly lower for stability)
GAMMA = 0.95          # Discount factor  
EPSILON_START = 1.0   # Initial exploration rate
EPSILON_MIN = 0.05    # Minimum exploration rate
EPSILON_DECAY = 0.997 # Slower decay for more exploration

# Reward Configuration - Carefully tuned for Pacman
REWARDS = {
    # === Core Game Events ===
    'step': -0.5,            # Small living penalty (encourages speed but not too harsh)
    'dot': 10,               # Eating a dot - primary objective
    'pill': 30,              # Power pill - strategic value
    'ghost_eaten': 150,      # Eating a ghost - high reward for risk
    'death': -200,           # Dying - severe penalty
    'level_complete': 500,   # Clearing the level
    
    # === Movement & Exploration ===
    'new_tile': 1,           # Small bonus for exploring new areas
    'stuck_penalty': -8,     # Penalty for oscillating in place
    'wall_collision': -1,    # Trying to move into a wall
    
    # === Ghost Proximity (Distance-based danger signals) ===
    'ghost_death_zone': -8,  # Ghost within 2 tiles - HIGH danger
    'ghost_danger': -3,      # Ghost within 4 tiles - medium danger
    'ghost_safe': 0,         # Ghost further away
    
    # === Ghost Hunting (When powered) ===
    'hunt_bonus': 8          # Bonus for being close to frightened ghost
}

# Training Settings
EPISODES = 2000          # More episodes for better learning
MAX_STEPS_PER_EPISODE = 2000  # Prevent infinite episodes

# State discretization
POSITION_BUCKET_SIZE = 4  # Tiles per zone for position bucketing
