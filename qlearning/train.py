#!/usr/bin/env python3
"""
Junction-based Q-Learning Training

Key difference: Agent only makes decisions at junctions (intersections).
Between junctions, it continues in the same direction.

This reduces decisions from ~800/episode to ~50-100/episode,
making each decision more meaningful and learning faster.
"""

import argparse
import time
import os
import json
from pacman_env import PacmanEnv
from qlearn import QLearningAgent

def train(episodes=1000, port=8080):
    print(f"\n{'='*60}")
    print(f"JUNCTION-BASED Q-LEARNING")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
    print(f"Agent only decides at junctions (intersections)")
    print(f"{'='*60}\n")
    
    env = PacmanEnv(url=f"http://127.0.0.1:{port}")
    agent = QLearningAgent(
        actions=["up", "down", "left", "right"],
        alpha=0.3,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.03
    )
    
    qtable_file = "qtable.pkl"
    if os.path.exists(qtable_file):
        agent.load(qtable_file)
    
    # Stats
    scores = []
    rewards = []
    dots = []
    decisions_list = []
    
    best_score = 0
    best_ep = 0
    
    try:
        for ep in range(1, episodes + 1):
            state = env.reset()
            agent.reset_episode()
            
            if not state:
                print("[ERROR] No state, retrying...")
                time.sleep(1)
                state = env.reset()
                if not state:
                    return
            
            total_reward = 0
            step = 0
            done = False
            init_dots = state.get('status', {}).get('dots_remaining', 244)
            
            # Initial action at starting position
            action = agent.get_action(state)
            if agent.is_junction(state):
                agent.learn_at_junction(state, action, done=False)
            
            while not done and step < 3000:
                next_state, reward, done, _ = env.step(action)
                
                if next_state is None:
                    break
                
                # Accumulate reward
                agent.accumulate_reward(reward)
                total_reward += reward
                step += 1
                
                # Check if we reached a junction
                if agent.is_junction(next_state) or done:
                    # Get new action at this junction
                    new_action = agent.get_action(next_state)
                    
                    # Learn from the junction-to-junction transition
                    agent.learn_at_junction(next_state, new_action, done=done)
                    
                    action = new_action
                else:
                    # Not a junction - continue with same action
                    action = agent.get_action(next_state)  # Will return current_direction
                
                state = next_state
            
            agent.decay_epsilon()
            
            # Stats
            final_score = state.get('status', {}).get('score', 0) if state else 0
            final_dots = state.get('status', {}).get('dots_remaining', 244) if state else 244
            dots_eaten = init_dots - final_dots
            
            scores.append(final_score)
            rewards.append(total_reward)
            dots.append(dots_eaten)
            decisions_list.append(agent.decisions_made)
            
            if final_score > best_score:
                best_score = final_score
                best_ep = ep
            
            # Print episode
            print(f"Ep {ep:4d}: Score={final_score:4d} Dots={dots_eaten:3d} "
                  f"Steps={step:4d} Decisions={agent.decisions_made:3d} "
                  f"R={total_reward:7.1f} ε={agent.epsilon:.3f}")
            
            # Summary every 50
            if ep % 50 == 0:
                n = 50
                print(f"\n--- Last {n} episodes ---")
                print(f"Avg Score: {sum(scores[-n:])/n:.1f} (max: {max(scores[-n:])})")
                print(f"Avg Dots:  {sum(dots[-n:])/n:.1f}")
                print(f"Avg Decisions: {sum(decisions_list[-n:])/n:.1f}")
                print(f"States: {len(agent.Q)}")
                print(f"Best Ever: {best_score} (Ep {best_ep})")
                print()
            
            # Save every 100
            if ep % 100 == 0:
                agent.save(qtable_file)
                agent.print_stats()
    
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
    
    agent.save(qtable_file)
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Episodes: {len(scores)}")
    print(f"Avg Score: {sum(scores)/len(scores):.1f}")
    print(f"Avg Dots: {sum(dots)/len(dots):.1f}")
    print(f"Avg Decisions/ep: {sum(decisions_list)/len(decisions_list):.1f}")
    print(f"Best: {best_score} (Ep {best_ep})")
    print(f"States: {len(agent.Q)}")
    print(f"{'='*60}")
    
    agent.print_stats()
    
    with open('training.json', 'w') as f:
        json.dump({
            'scores': scores,
            'rewards': rewards,
            'dots': dots,
            'decisions': decisions_list,
            'best_score': best_score
        }, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    
    train(episodes=args.episodes, port=args.port)
