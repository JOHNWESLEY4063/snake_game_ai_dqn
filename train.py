import pygame
import torch
import numpy as np
import argparse
import random
import os

# Import your project modules
from snake_gameai import SnakeGameAI
from agent import Agent, BATCH_SIZE, UPDATE_TARGET_EVERY_GAMES
from model import Linear_QNet, QTrainer
# Import the Dashboard class for plotting
from plots import Dashboard 

# A helper function for reproducibility
def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    # --- Setup ---
    parser = argparse.ArgumentParser(description="Train a Deep Q-Learning agent for Snake Game.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last saved checkpoint.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    set_seeds(args.seed)

    # --- Initialize Game, Agent, and Plotting Dashboard ---
    game = SnakeGameAI()
    agent = Agent()
    dashboard = Dashboard()

    # --- Initialize lists for all metrics ---
    plot_scores = []
    plot_mean_scores = []
    plot_avg_losses = []
    plot_episode_lengths = []
    plot_efficiencies = []
    plot_avg_q_values = []
    
    total_score = 0
    record_score = 0
    
    # --- Add accumulators for the current episode's metrics ---
    current_episode_loss = 0
    current_episode_q_sum = 0
    q_value_steps = 0

    print("\n--- Starting Deep Q-Learning Training ---")

    # --- Main Training Loop ---
    while True:
        state_old = agent.get_state(game)
        final_move, max_q = agent.get_action(state_old)
        
        if max_q > 0:
            current_episode_q_sum += max_q
            q_value_steps += 1

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        loss = agent.train_short_memory(state_old, final_move, reward, state_new, done)
        current_episode_loss += loss

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            episode_length = game.frame_iteration
            game.reset()
            agent.n_games += 1

            if len(agent.memory) > BATCH_SIZE * 2:
                long_mem_loss = agent.train_long_memory()
                current_episode_loss += long_mem_loss
            
            if agent.n_games % UPDATE_TARGET_EVERY_GAMES == 0:
                agent.update_target_network()

            # --- Calculate and append all metrics for plotting ---
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot_episode_lengths.append(episode_length)
            
            avg_loss = current_episode_loss / episode_length if episode_length > 0 else 0
            plot_avg_losses.append(avg_loss)
            
            efficiency = (score / episode_length) * 100 if episode_length > 0 else 0
            plot_efficiencies.append(efficiency)
            
            avg_q = current_episode_q_sum / q_value_steps if q_value_steps > 0 else 0
            plot_avg_q_values.append(avg_q)

            # --- Reset per-episode accumulators ---
            current_episode_loss = 0
            current_episode_q_sum = 0
            q_value_steps = 0

            if score > record_score:
                record_score = score
                agent.save_data()

            # --- Log metrics to console ---
            print(f'Game {agent.n_games:5} | Score: {score:3} | Record: {record_score:3} | Mean Score: {mean_score:.2f}')
            
            # --- Call the update method on the dashboard object ---
            dashboard.update(plot_scores, plot_mean_scores, plot_avg_losses, plot_episode_lengths, plot_efficiencies, plot_avg_q_values)

if __name__ == '__main__':
    train()
