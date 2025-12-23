import argparse
import time
import os
import requests
from pacman_env import PacmanEnv
from qlearn import QLearningAgent

def train(episodes=1000, port=8080):
    print(f"Connecting to Pacman server at http://127.0.0.1:{port}...")
    
    env = PacmanEnv(url=f"http://127.0.0.1:{port}")
    agent = QLearningAgent(actions=["up", "down", "left", "right"])
    
    # Load existing Q-table if available
    if os.path.exists("qtable.pkl"):
        agent.load("qtable.pkl")
        print("Loaded existing Q-table.")

    try:
        for episode in range(episodes):
            state_api = env.reset()
            if not state_api:
                print("Failed to get initial state. Is the server running? (./build/pacman --api)")
                # Retry a few times before giving up? Or just break.
                # User says: "game needs to be started seperatly".
                # If reset fails, we can try waiting a bit or just fail.
                # Let's try to wait once, then fail.
                time.sleep(1)
                state_api = env.reset()
                if not state_api:
                    print("Could not connect. Exiting.")
                    return

            state = agent.map_state(state_api)
            total_reward = 0
            done = False
            
            while not done:
                action = agent.get_action(state)
                next_state_api, reward, done, _ = env.step(action)
                
                if next_state_api is None:
                    print("Lost connection to server.")
                    done = True
                    break
                    
                next_state = agent.map_state(next_state_api)
                
                agent.learn(state, action, reward, next_state)
                
                state = next_state
                state_api = next_state_api
                total_reward += reward
                
            agent.decay_epsilon()
            print(f"Episode {episode+1}/{episodes}: Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            
            if (episode + 1) % 100 == 0:
                agent.save("qtable.pkl")
                print("Q-table saved.")

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        agent.save("qtable.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--port", type=str, default="8080")
    args = parser.parse_args()
    
    train(episodes=args.episodes, port=args.port)
