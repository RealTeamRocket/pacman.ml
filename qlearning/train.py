#!/usr/bin/env python3
"""
Junction-based Q-Learning Training with 3-Lives Episodes

Key features:
- An episode spans all 3 lives (2 spare + 1 current)
- Death is penalized but learning continues with remaining lives
- Game over (3 deaths) ends the episode
- Score and progress accumulate across lives
"""

import argparse
import time
import os
import json
from pacman_env import PacmanEnv
from qlearn import QLearningAgent


def train(episodes=1000, port=8080):
    print(f"\n{'='*60}")
    print(f"JUNCTION-BASED Q-LEARNING (3-LIVES EPISODES)")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
    print(f"Each episode uses all 3 lives (2 spare + 1 current)")
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
    deaths_list = []
    
    best_score = 0
    best_ep = 0
    
    try:
        for ep in range(1, episodes + 1):
            # Start new episode
            state = env.reset()
            agent.reset_episode()  # Reset agent ONCE per episode, not per life
            
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
            
            # Main game loop - continues across all lives
            while not done and step < 10000:
                next_state, reward, life_done, info = env.step(action)
                
                if next_state is None:
                    break
                
                # Accumulate reward
                agent.accumulate_reward(reward)
                total_reward += reward
                step += 1
                
                # Check if we reached a junction
                if agent.is_junction(next_state) or life_done:
                    # Get new action at this junction
                    new_action = agent.get_action(next_state)
                    
                    # Learn from the junction-to-junction transition
                    agent.learn_at_junction(next_state, new_action, done=life_done)
                    
                    action = new_action
                else:
                    # Not a junction - continue with same action
                    action = agent.get_action(next_state)
                
                state = next_state
                
                # Check if life ended
                if life_done:
                    if info.get('game_over', False):
                        # All lives lost - episode truly over
                        done = True
                    elif next_state.get('status', {}).get('round_won', False):
                        # Won the round!
                        done = True
                    else:
                        # Lost a life but have more - soft reset and continue
                        state = env.reset()
                        if state is None:
                            break
                        
                        # Reset direction tracking for new life position
                        agent.current_direction = None
                        
                        # Get action for respawn position
                        action = agent.get_action(state)
                        if agent.is_junction(state):
                            agent.learn_at_junction(state, action, done=False)
            
            agent.decay_epsilon()
            
            # Stats
            final_score = state.get('status', {}).get('score', 0) if state else 0
            final_dots = state.get('status', {}).get('dots_remaining', 244) if state else 244
            dots_eaten = init_dots - final_dots
            episode_deaths = env.deaths_this_episode
            round_won = state.get('status', {}).get('round_won', False) if state else False
            
            scores.append(final_score)
            rewards.append(total_reward)
            dots.append(dots_eaten)
            decisions_list.append(agent.decisions_made)
            deaths_list.append(episode_deaths)
            
            if final_score > best_score:
                best_score = final_score
                best_ep = ep
            
            # Print episode
            win_marker = " 🏆" if round_won else ""
            death_markers = "💀" * episode_deaths
            print(f"Ep {ep:4d}: Score={final_score:5d} Dots={dots_eaten:3d} "
                  f"Steps={step:5d} Dec={agent.decisions_made:3d} "
                  f"Deaths={episode_deaths} R={total_reward:8.1f} ε={agent.epsilon:.3f} "
                  f"{death_markers}{win_marker}")
            
            # Summary every 50
            if ep % 50 == 0:
                n = min(50, len(scores))
                recent_scores = scores[-n:]
                recent_dots = dots[-n:]
                recent_deaths = deaths_list[-n:]
                recent_decisions = decisions_list[-n:]
                
                print(f"\n{'='*60}")
                print(f"SUMMARY - Last {n} episodes")
                print(f"{'='*60}")
                print(f"Avg Score:     {sum(recent_scores)/n:7.1f} (max: {max(recent_scores)})")
                print(f"Avg Dots:      {sum(recent_dots)/n:7.1f}")
                print(f"Avg Deaths:    {sum(recent_deaths)/n:7.2f} / 3")
                print(f"Avg Decisions: {sum(recent_decisions)/n:7.1f}")
                print(f"States in Q:   {len(agent.Q)}")
                print(f"Best Ever:     {best_score} (Ep {best_ep})")
                print(f"{'='*60}\n")
            
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
    if scores:
        print(f"Episodes:      {len(scores)}")
        print(f"Avg Score:     {sum(scores)/len(scores):.1f}")
        print(f"Avg Dots:      {sum(dots)/len(dots):.1f}")
        print(f"Avg Deaths:    {sum(deaths_list)/len(deaths_list):.2f} / 3")
        print(f"Best Score:    {best_score} (Ep {best_ep})")
        print(f"States in Q:   {len(agent.Q)}")
    print(f"{'='*60}")
    
    agent.print_stats()
    
    with open('training.json', 'w') as f:
        json.dump({
            'scores': scores,
            'rewards': rewards,
            'dots': dots,
            'decisions': decisions_list,
            'deaths': deaths_list,
            'best_score': best_score,
            'best_episode': best_ep
        }, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pacman Q-Learning Agent")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of episodes to train (each uses 3 lives)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port for Pacman API server")
    args = parser.parse_args()
    
    train(episodes=args.episodes, port=args.port)