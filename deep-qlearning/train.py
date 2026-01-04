#!/usr/bin/env python3
"""
Deep Q-Learning Training Script for Pacman

This script trains a DQN agent to play Pacman using:
- Double DQN with target network
- Dueling network architecture
- Experience replay
- Epsilon-greedy exploration with decay

The reward system is based on lessons from tabular Q-learning:
- Survival-time based death penalty
- Progressive dot milestone bonuses
- Endgame incentives

Usage from project root:
    poetry run python deep-qlearning/train.py --episodes 5000
"""

import argparse
import time
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any

# Add the deep-qlearning directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import torch
from tqdm import tqdm

from pacman_env import PacmanEnv
from dqn_agent import DQNAgent


def train(
    episodes: int = 5000,
    port: int = 8080,
    hidden_dim: int = 256,
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,  # Much faster decay: reaches 0.1 in ~450 episodes
    target_update_freq: int = 500,
    buffer_size: int = 100000,
    batch_size: int = 64,
    train_freq: int = 4,
    save_freq: int = 100,
    checkpoint_dir: str = None,
    resume: bool = True
):
    """
    Train the DQN agent.
    
    Args:
        episodes: Number of episodes to train
        port: Port for Pacman API server
        hidden_dim: Hidden layer size for neural network
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Minimum exploration rate
        epsilon_decay: Epsilon decay per episode
        target_update_freq: Steps between target network updates
        buffer_size: Experience replay buffer size
        batch_size: Training batch size
        train_freq: Steps between training updates
        save_freq: Episodes between checkpoint saves
        checkpoint_dir: Directory for saving checkpoints
        resume: Whether to resume from checkpoint
    """
    
    print(f"\n{'='*60}")
    print(f"DEEP Q-LEARNING TRAINING")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Learning rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Epsilon: {epsilon_start} → {epsilon_end} (decay={epsilon_decay})")
    print(f"Buffer size: {buffer_size}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Create checkpoint directory (default to deep-qlearning/checkpoints)
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize environment
    env = PacmanEnv(url=f"http://127.0.0.1:{port}")
    
    # Initialize agent
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.NUM_ACTIONS,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update_freq=target_update_freq,
        buffer_size=buffer_size,
        batch_size=batch_size,
        dueling=True,
        prioritized_replay=False
    )
    
    # Resume from checkpoint if available
    checkpoint_path = os.path.join(checkpoint_dir, "dqn_latest.pt")
    start_episode = 1
    
    if resume and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        # Try to load training state
        state_path = os.path.join(checkpoint_dir, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                training_state = json.load(f)
                start_episode = training_state.get('episode', 1) + 1
                print(f"[RESUME] Resuming from episode {start_episode}")
    
    # Statistics tracking
    scores: List[int] = []
    dots_eaten: List[int] = []
    deaths_list: List[int] = []
    wins_list: List[int] = []
    rewards_list: List[float] = []
    losses: List[float] = []
    
    best_score = 0
    best_dots = 0
    best_ep = 0
    total_wins = 0
    total_steps = 0
    
    try:
        for ep in range(start_episode, episodes + 1):
            # Reset environment
            state = env.reset()
            
            # Reset direction tracking for new episode
            env.current_direction = None
            env.last_decision_pos = None
            
            if state is None or np.all(state == 0):
                print(f"[ERROR] Failed to get initial state, retrying...")
                time.sleep(1)
                state = env.reset()
                env.current_direction = None
                env.last_decision_pos = None
                if state is None or np.all(state == 0):
                    continue
            
            episode_reward = 0
            episode_steps = 0
            done = False
            junctions_encountered = 0
            auto_moves = 0  # Hallways + corners (no DQN needed)
            fallback_moves = 0  # Times we fell back to DQN (shouldn't happen)
            
            # Debug: verbose output for first few episodes
            debug_verbose = (ep <= 3)
            
            # Track if we made a decision this step (for experience storage)
            made_decision = False
            
            # Track last tile position to detect when we actually move
            last_tile_pos = None
            pending_direction = None  # Direction we chose, to become arrival_direction when we move
            
            # Episode loop - continues across all lives
            while not done and episode_steps < 10000:
                # Get current game state
                state_dict = env._get_state()
                
                # Get current tile position
                pacman_data = state_dict.get('pacman', {}) if state_dict else {}
                current_tile = (pacman_data.get('x', 0) // 8, pacman_data.get('y', 0) // 8)
                
                # Update arrival_direction only when we actually move to a new tile
                if last_tile_pos is not None and current_tile != last_tile_pos:
                    # We moved! Update arrival_direction to the direction we were going
                    if pending_direction is not None:
                        env.arrival_direction = pending_direction
                
                # Debug: show available directions
                if debug_verbose and episode_steps < 20:
                    open_dirs = env._get_open_directions(state_dict)
                    forward_dirs = env._get_forward_directions(open_dirs)
                    pacman = state_dict.get('pacman', {}) if state_dict else {}
                    px = pacman.get('x', 0) // 8
                    py = pacman.get('y', 0) // 8
                    is_jnct = env.is_junction(state_dict)
                    is_new = env.is_new_junction(state_dict)
                    jnct_status = "NEW_JNCT" if is_new else ("WAITING" if is_jnct else "hallway")
                    # Show which direction is blocked (reverse of arrival)
                    blocked = env.reverse_dir.get(env.arrival_direction) if env.arrival_direction else None
                    action_mask = env.get_action_mask(state_dict)
                    valid_actions = [env.IDX_TO_ACTION[i] for i in range(4) if action_mask[i] > 0]
                    print(f"  [DEBUG] Step {episode_steps}: pos=({px},{py}) "
                          f"[{jnct_status}] open={open_dirs} valid={valid_actions} "
                          f"arrived_from={env.arrival_direction} blocked={blocked}")
                
                # Check if at a NEW junction (decision point we haven't decided at yet)
                # This prevents making multiple decisions at the same position while
                # waiting for the game to move Pacman
                at_new_junction = env.is_new_junction(state_dict)
                made_decision = False
                
                if at_new_junction:
                    # At a NEW junction - use DQN to make a decision
                    junctions_encountered += 1
                    made_decision = True
                    
                    # Get action mask - reverse direction is always blocked (like real Pacman)
                    action_mask = env.get_action_mask(state_dict)
                    
                    action_idx = agent.select_action(state, action_mask=action_mask, training=True)
                    action_name = env.IDX_TO_ACTION[action_idx]
                    
                    # Store this decision for continuing in hallways
                    env.current_direction = action_name
                    env.last_junction_action = action_idx
                    
                    # Mark that we've made a decision at this position
                    env.mark_decision_made(state_dict)
                    
                    # Store the pending direction - will become arrival_direction when we move
                    pending_direction = action_name
                elif env.is_junction(state_dict):
                    # At the SAME junction we already decided at - continue with last decision
                    auto_moves += 1
                    action_name = env.current_direction if env.current_direction else 'right'
                    action_idx = env.ACTION_TO_IDX[action_name]
                else:
                    # In a hallway - continue in the current direction
                    forward_dir = env.get_forward_direction(state_dict)
                    
                    if forward_dir is not None:
                        # Single valid forward direction (hallway or corner) - continue automatically
                        auto_moves += 1
                        action_name = forward_dir
                        action_idx = env.ACTION_TO_IDX[action_name]
                        env.current_direction = action_name
                        # Store pending direction for when we move (handles corners)
                        pending_direction = action_name
                    else:
                        # Fallback: use DQN (shouldn't happen often)
                        fallback_moves += 1
                        if debug_verbose and fallback_moves <= 5:
                            print(f"  [DEBUG] Fallback at step {episode_steps}: "
                                  f"not junction but no forward_dir")
                        action_mask = env.get_action_mask(state_dict)
                        action_idx = agent.select_action(state, action_mask=action_mask, training=True)
                        action_name = env.IDX_TO_ACTION[action_idx]
                        env.current_direction = action_name
                
                # Take action (using action index)
                next_state, reward, life_done, info = env.step(action_idx)
                
                # Update last_tile_pos for next iteration
                last_tile_pos = current_tile
                
                # Store experience only if we made a NEW junction decision
                # This focuses learning on important decisions, not hallway continuations
                if made_decision:
                    agent.store_experience(state, action_idx, reward, next_state, life_done)
                
                # Train
                if total_steps % train_freq == 0:
                    loss = agent.train_step()
                    if loss is not None:
                        losses.append(loss)
                
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                # Check if episode is truly over
                if life_done:
                    if info.get('game_over', False) or info.get('round_won', False):
                        done = True
                    else:
                        # Lost a life but have more - reset and continue
                        state = env.reset()
                        # Reset direction tracking on death
                        env.current_direction = None
                        env.last_decision_pos = None  # Reset so we make fresh decision at respawn
                        env.arrival_direction = None  # Reset arrival direction for respawn
                        last_tile_pos = None  # Reset tile tracking
                        pending_direction = None
                        if state is None or np.all(state == 0):
                            break
                else:
                    state = next_state
            
            # Decay epsilon at end of episode
            agent.decay_epsilon()
            
            # Episode statistics
            final_score = info.get('score', 0)
            final_dots = info.get('dots_eaten', 0)
            final_deaths = info.get('deaths', 3)
            round_won = info.get('round_won', False)
            
            scores.append(final_score)
            dots_eaten.append(final_dots)
            deaths_list.append(final_deaths)
            wins_list.append(1 if round_won else 0)
            rewards_list.append(episode_reward)
            
            if round_won:
                total_wins += 1
            
            if final_score > best_score:
                best_score = final_score
                best_ep = ep
                # Save best model
                agent.save(os.path.join(checkpoint_dir, "dqn_best.pt"))
            
            if final_dots > best_dots:
                best_dots = final_dots
            
            # Print episode summary
            win_marker = " 🏆 WIN!" if round_won else ""
            death_markers = "💀" * final_deaths
            
            avg_loss = agent.get_average_loss()
            
            auto_pct = 100 * auto_moves / max(1, episode_steps)
            fallback_info = f" Fallback={fallback_moves}" if fallback_moves > 0 else ""
            print(f"Ep {ep:5d}: Score={final_score:5d} Dots={final_dots:3d} "
                  f"Steps={episode_steps:5d} Jnct={junctions_encountered:3d} Auto={auto_pct:4.1f}% Deaths={final_deaths} "
                  f"R={episode_reward:8.1f} ε={agent.epsilon:.3f} "
                  f"Loss={avg_loss:.4f}{fallback_info} {death_markers}{win_marker}")
            
            # Periodic summary
            if ep % 50 == 0:
                n = min(50, len(scores))
                recent_scores = scores[-n:]
                recent_dots = dots_eaten[-n:]
                recent_deaths = deaths_list[-n:]
                recent_wins = wins_list[-n:]
                recent_rewards = rewards_list[-n:]
                
                print(f"\n{'='*60}")
                print(f"SUMMARY - Last {n} episodes (Episode {ep})")
                print(f"{'='*60}")
                print(f"Avg Score:     {sum(recent_scores)/n:7.1f} (max: {max(recent_scores)})")
                print(f"Avg Dots:      {sum(recent_dots)/n:7.1f} (max: {max(recent_dots)})")
                print(f"Avg Deaths:    {sum(recent_deaths)/n:7.2f} / 3")
                print(f"Avg Reward:    {sum(recent_rewards)/n:7.1f}")
                print(f"Wins:          {sum(recent_wins)} / {n}")
                print(f"Epsilon:       {agent.epsilon:.4f}")
                print(f"Buffer Size:   {len(agent.replay_buffer)}")
                print(f"Total Steps:   {total_steps}")
                print(f"Best Ever:     {best_score} (Ep {best_ep}), {best_dots} dots")
                print(f"Total Wins:    {total_wins}")
                print(f"{'='*60}\n")
            
            # Save checkpoint
            if ep % save_freq == 0:
                agent.save(checkpoint_path)
                
                # Save training state
                training_state = {
                    'episode': ep,
                    'best_score': best_score,
                    'best_dots': best_dots,
                    'best_ep': best_ep,
                    'total_wins': total_wins,
                    'total_steps': total_steps
                }
                with open(os.path.join(checkpoint_dir, "training_state.json"), 'w') as f:
                    json.dump(training_state, f)
                
                # Save training history
                history = {
                    'scores': scores,
                    'dots': dots_eaten,
                    'deaths': deaths_list,
                    'wins': wins_list,
                    'rewards': rewards_list,
                    'best_score': best_score,
                    'best_dots': best_dots,
                    'best_episode': best_ep,
                    'total_wins': total_wins
                }
                with open(os.path.join(checkpoint_dir, "training_history.json"), 'w') as f:
                    json.dump(history, f)
    
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving checkpoint...")
    
    # Final save
    agent.save(checkpoint_path)
    agent.print_stats()
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    if scores:
        print(f"Episodes:      {len(scores)}")
        print(f"Avg Score:     {sum(scores)/len(scores):.1f}")
        print(f"Avg Dots:      {sum(dots_eaten)/len(dots_eaten):.1f}")
        print(f"Avg Deaths:    {sum(deaths_list)/len(deaths_list):.2f} / 3")
        print(f"Total Wins:    {total_wins} ({100*total_wins/len(scores):.1f}%)")
        print(f"Best Score:    {best_score} (Ep {best_ep})")
        print(f"Best Dots:     {best_dots}")
    print(f"{'='*60}")
    
    # Save final history
    history = {
        'scores': scores,
        'dots': dots_eaten,
        'deaths': deaths_list,
        'wins': wins_list,
        'rewards': rewards_list,
        'best_score': best_score,
        'best_dots': best_dots,
        'best_episode': best_ep,
        'total_wins': total_wins
    }
    with open(os.path.join(checkpoint_dir, "training_history.json"), 'w') as f:
        json.dump(history, f)


def main():
    parser = argparse.ArgumentParser(description="Train Pacman DQN Agent")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=5000,
                        help="Number of episodes to train")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port for Pacman API server")
    
    # Network parameters
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden layer dimension")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    
    # RL parameters
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0,
                        help="Initial exploration rate")
    parser.add_argument("--epsilon-end", type=float, default=0.05,
                        help="Minimum exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.995,
                        help="Epsilon decay rate per episode (0.995 reaches 0.1 in ~450 eps)")
    
    # Buffer parameters
    parser.add_argument("--buffer-size", type=int, default=100000,
                        help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size")
    
    # Training frequency
    parser.add_argument("--train-freq", type=int, default=4,
                        help="Steps between training updates")
    parser.add_argument("--target-update", type=int, default=500,
                        help="Steps between target network updates")
    
    # Checkpointing
    parser.add_argument("--save-freq", type=int, default=100,
                        help="Episodes between checkpoint saves")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory for checkpoints (default: deep-qlearning/checkpoints)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't resume from checkpoint")
    
    args = parser.parse_args()
    
    train(
        episodes=args.episodes,
        port=args.port,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update_freq=args.target_update,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        save_freq=args.save_freq,
        checkpoint_dir=args.checkpoint_dir,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()