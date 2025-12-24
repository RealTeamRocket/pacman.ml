import argparse
import time
import os
import requests
from pacman_env import PacmanEnv
from qlearn import QLearningAgent
import config

def train(episodes=config.EPISODES, port=8080):
    print(f"\n{'='*70}")
    print(f"PACMAN Q-LEARNING TRAINING")
    print(f"{'='*70}")
    print(f"Episodes: {episodes}")
    print(f"Server: http://127.0.0.1:{port}")
    print(f"Config: Alpha={config.ALPHA}, Gamma={config.GAMMA}, Epsilon={config.EPSILON_START}→{config.EPSILON_MIN}")
    print(f"{'='*70}\n")
    
    env = PacmanEnv(url=f"http://127.0.0.1:{port}")
    agent = QLearningAgent(actions=["up", "down", "left", "right"])
    
    # Load existing Q-table if available
    if os.path.exists("qtable.pkl"):
        agent.load("qtable.pkl")
    
    # Statistics tracking
    episode_rewards = []
    episode_scores = []
    episode_lengths = []
    
    try:
        for episode in range(episodes):
            state_api = env.reset()
            
            if not state_api:
                print("[ERROR] Failed to get initial state. Is server running?")
                time.sleep(1)
                state_api = env.reset()
                if not state_api:
                    print("[ERROR] Could not connect. Exiting.")
                    return
            
            # Track last action
            last_action = "none"
            state_api['last_action'] = last_action
            state = agent.map_state(state_api)
            
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                action = agent.get_action(state)
                next_state_api, reward, done, _ = env.step(action)
                
                if next_state_api is None:
                    print("[ERROR] Lost connection to server.")
                    done = True
                    break
                
                next_state_api['last_action'] = action
                next_state = agent.map_state(next_state_api)
                
                agent.learn(state, action, reward, next_state)
                
                state = next_state
                state_api = next_state_api
                last_action = action
                total_reward += reward
                steps += 1
                
                # Prevent infinite loops
                if steps > 1000:
                    print(f"[WARNING] Episode exceeded 1000 steps, forcing termination")
                    done = True
            
            agent.decay_epsilon()
            
            # Store stats
            final_score = state_api.get('status', {}).get('score', 0) if state_api else 0
            episode_rewards.append(total_reward)
            episode_scores.append(final_score)
            episode_lengths.append(steps)
            
            # Print episode summary
            print(f"Episode {episode+1}/{episodes}: Reward={total_reward:.1f}, Score={final_score}, Steps={steps}, Epsilon={agent.epsilon:.2f}")
            
            # Detailed stats every 10 episodes
            if (episode + 1) % 10 == 0:
                avg_reward_10 = sum(episode_rewards[-10:]) / 10
                avg_score_10 = sum(episode_scores[-10:]) / 10
                avg_length_10 = sum(episode_lengths[-10:]) / 10
                
                print(f"\n{'─'*70}")
                print(f"Episodes {episode-9}-{episode+1} Average:")
                print(f"  Reward: {avg_reward_10:.1f}")
                print(f"  Score: {avg_score_10:.1f}")
                print(f"  Length: {avg_length_10:.1f} steps")
                print(f"  Epsilon: {agent.epsilon:.3f}")
                print(f"  States explored: {len(agent.Q)}")
                print(f"{'─'*70}\n")
            
            # Comprehensive stats every 50 episodes
            if (episode + 1) % 50 == 0:
                agent.print_stats()
            
            # Save checkpoint
            if (episode + 1) % 100 == 0:
                agent.save("qtable.pkl")
                
                # Save training log
                import json
                with open('training_log.json', 'w') as f:
                    json.dump({
                        'rewards': episode_rewards,
                        'scores': episode_scores,
                        'lengths': episode_lengths
                    }, f, indent=2)
                print("[SAVE] Training log saved\n")
    
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training interrupted by user")
        agent.save("qtable.pkl")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Average reward: {sum(episode_rewards)/len(episode_rewards):.1f}")
    print(f"Average score: {sum(episode_scores)/len(episode_scores):.1f}")
    print(f"Best score: {max(episode_scores)}")
    print(f"States explored: {len(agent.Q)}")
    print(f"{'='*70}\n")
    
    agent.print_stats()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--port", type=str, default="8080")
    args = parser.parse_args()
    
    train(episodes=args.episodes, port=args.port)
