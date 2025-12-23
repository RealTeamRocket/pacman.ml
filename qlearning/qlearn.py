import numpy as np
import pickle
import os
import random

class QLearningAgent:
    def __init__(self, actions, alpha=0.01, gamma=0.8, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.weights = {
            'bias': 1.0,
            'eats_food': 500.0,    # Massive reward for actually eating
            'food_score': 200.0,   # Strong pull towards food
            'ghost_proximity': -1000.0, # High penalty, but only when close (<5 tiles)
            'lives': 0.0
        }
    
    def get_weights(self):
        return self.weights

    def get_features(self, state, action):
        """
        Extract features for (state, action) pair.
        State is assumed to be a dictionary with 'pacman', 'ghosts', 'map'.
        """
        if not state: return {}
        
        features = {}
        features['bias'] = 1.0
        
        pacman = state.get('pacman', {})
        px, py = pacman.get('x', 0), pacman.get('y', 0)
        tx, ty = px // 8, py // 8
        
        # Calculate next position based on action
        dx, dy = 0, 0
        if action == 'up': dy = -1
        elif action == 'down': dy = 1
        elif action == 'left': dx = -1
        elif action == 'right': dx = 1
        
        # Next tile
        nx, ny = tx + dx, ty + dy
        
        game_map = state.get('map', [])
        GRID_W = 28
        GRID_H = 36
        
        # Check walls
        is_wall = False
        if not game_map:
            is_wall = False
        elif nx < 0 or nx >= GRID_W or ny < 0 or ny >= GRID_H:
            is_wall = True
        else:
            tile = game_map[ny][nx]
            # TILE_SPACE(64), TILE_DOT(16), TILE_PILL(20) are walkable
            if tile not in (64, 16, 20):
                is_wall = True
                
        if is_wall:
            # Action hits a wall. Current pos logic applies but with penalty?
            # Ideally we predict we stay in place
            nx, ny = tx, ty
            
        features['stop'] = 1.0 if action == 'none' else 0.0

        # Feature: Eats Food
        eats_food = 0.0
        if game_map and 0 <= nx < GRID_W and 0 <= ny < GRID_H:
             if game_map[ny][nx] in (16, 20): # DOT or PILL
                 eats_food = 1.0
        features['eats_food'] = eats_food
        
        # Feature: Closest Food (Inverse Distance)
        # BFS to find nearest food from (nx, ny)
        min_dist = 999
        if game_map:
            queue = [(nx, ny, 0)]
            visited = set([(nx, ny)])
            found = False
            
            head = 0
            while head < len(queue):
                cx, cy, dist = queue[head]
                head += 1
                
                # Limit depth
                if dist > 30: break 
                
                if 0 <= cy < GRID_H and 0 <= cx < GRID_W:
                    if game_map[cy][cx] in (16, 20):
                        min_dist = dist
                        found = True
                        break
                
                # Neighbors
                for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    mx, my = cx + ddx, cy + ddy
                    if (mx, my) not in visited:
                        # Check wall
                        w = False
                        if mx < 0 or mx >= GRID_W or my < 0 or my >= GRID_H: w = True
                        elif game_map[my][mx] not in (64, 16, 20): w = True
                        
                        if not w:
                            visited.add((mx, my))
                            queue.append((mx, my, dist+1))
                            
        # Inverse distance: 1.0 is on top of food, 0.5 is 1 step away, etc.
        features['food_score'] = 1.0 / (min_dist + 1.0)
        
        # Feature: Ghost Proximity (Inverse)
        ghosts = state.get('ghosts', [])
        ghost_danger = 0.0
        
        for g in ghosts:
            if g.get('state') == 1: # Chasing
                gx, gy = g.get('x', 0), g.get('y', 0)
                gtx, gty = gx // 8, gy // 8
                
                # Manhattan distance to next pos
                dist = abs(gtx - nx) + abs(gty - ny)
                
                if dist < 5:
                    ghost_danger = max(ghost_danger, 1.0 / (dist + 0.1))
                    
        features['ghost_proximity'] = ghost_danger
        
        return features
        
        return features

    def get_q_value(self, state, action):
        features = self.get_features(state, action)
        q_value = 0.0
        for feature, value in features.items():
            q_value += self.weights.get(feature, 0.0) * value
        return q_value

    def get_legal_actions(self, state):
        if not state: return self.actions
        
        pacman = state.get('pacman', {})
        px, py = pacman.get('x', 0), pacman.get('y', 0)
        tx, ty = px // 8, py // 8
        
        game_map = state.get('map', [])
        GRID_W = 28
        GRID_H = 36
        
        legal = []
        for action in self.actions:
            dx, dy = 0, 0
            if action == 'up': dy = -1
            elif action == 'down': dy = 1
            elif action == 'left': dx = -1
            elif action == 'right': dx = 1
            
            nx, ny = tx + dx, ty + dy
            
            # Check map bounds and walls
            is_legal = True
            if not game_map:
                is_legal = True # Assume all reasonable if no map
            elif nx < 0 or nx >= GRID_W or ny < 0 or ny >= GRID_H:
                is_legal = False # Out of bounds
            else:
                tile = game_map[ny][nx]
                # TILE_SPACE(64), TILE_DOT(16), TILE_PILL(20) are walkable
                if tile not in (64, 16, 20):
                    is_legal = False
            
            if is_legal:
                legal.append(action)
                
        if not legal:
            return self.actions # Fallback if stuck (shouldn't happen)
            
        return legal

    def get_action(self, state):
        legal_actions = self.get_legal_actions(state)
        
        if np.random.rand() < self.epsilon:
            return random.choice(legal_actions)
        
        q_values = [(action, self.get_q_value(state, action)) for action in legal_actions]
        
        if not q_values: return random.choice(self.actions)
        
        max_q = max(q_values, key=lambda x: x[1])[1]
        best_actions = [action for action, q in q_values if q == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state):
        features = self.get_features(state, action)
        current_q = self.get_q_value(state, action)
        
        # Max Q for next state (over legal actions only)
        legal_next = self.get_legal_actions(next_state)
        next_q_values = [self.get_q_value(next_state, a) for a in legal_next]
        next_max_q = max(next_q_values) if next_q_values else 0.0
        
        difference = (reward + self.gamma * next_max_q) - current_q
        
        # Update weights
        for feature, value in features.items():
            self.weights[feature] = self.weights.get(feature, 0.0) + self.alpha * difference * value

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.weights, f)

    def load(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                # Need to handle if loading old q_table vs new weights
                try:
                    loaded = pickle.load(f)
                    if isinstance(loaded, dict) and 'bias' in loaded:
                        self.weights = loaded
                    else:
                        print("Old Q-table format detected, starting fresh.")
                except:
                    pass

    def map_state(self, api_state):
        # Pass through the raw state or slightly processed
        # We need the map for features
        return api_state


