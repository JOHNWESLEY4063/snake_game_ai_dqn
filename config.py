# config.py
import torch
import os
from datetime import datetime

# --- General Project Settings ---
PROJECT_NAME = "Snake_DQL_AI"
MODEL_DIR = './model_weights'
LOG_DIR = './runs' # For TensorBoard logs

# Ensure model and log directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Set device for PyTorch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- Agent and Training Hyperparameters ---
class AgentConfig:
    MAX_MEMORY = 100_000 # Max size of the replay buffer
    BATCH_SIZE = 1000    # Number of samples to train on from replay buffer
    LR = 0.0005          # Learning rate for the optimizer
    GAMMA = 0.9          # Discount factor for future rewards
    EPSILON_START = 1.0  # Initial exploration rate
    EPSILON_MIN = 0.01   # Minimum exploration rate
    EPSILON_DECAY_RATE = 0.995 # Epsilon decay rate per game
    UPDATE_TARGET_EVERY_GAMES = 100 # How often to update the target network
    SAVE_CHECKPOINT_EVERY_GAMES = 500 # How often to save model checkpoints

    # File names for saving/loading
    MAIN_MODEL_FILE = 'snake_q_model.pth'
    TARGET_MODEL_FILE = 'snake_target_q_model.pth'
    REPLAY_MEMORY_FILE = 'replay_memory.pkl'
    TRAINING_STATE_FILE = 'training_state.pkl' # To save n_games, scores etc.

# --- Network Architecture ---
class NetworkConfig:
    # State representation:
    # 8-direction danger (straight, right, left, back, diag_fr, diag_fl, diag_br, diag_bl)
    # 4-direction current heading (up, down, left, right)
    # 4-direction food location (up, down, left, right)
    INPUT_SIZE = 8 + 4 + 4 # Total 16 features
    HIDDEN_SIZE_1 = 256
    HIDDEN_SIZE_2 = 128 # Added a second hidden layer for more capacity
    OUTPUT_SIZE = 3     # 3 actions: [straight, right, left]

# --- Game Settings ---
class GameConfig:
    WIDTH = 640
    HEIGHT = 480
    BLOCK_SIZE = 20
    SPEED = 40 # Game visualization speed

    # Reward values
    REWARD_FOOD = 10
    REWARD_COLLISION = -10
    REWARD_STEP = -0.01 # Small negative reward for each step to encourage efficiency
    REWARD_CLOSER = 0.05 # Small positive reward for getting closer to food
    REWARD_FURTHER = -0.05 # Small negative reward for getting further from food

    # Timeout to prevent infinite loops when not eating
    COLLISION_TIMEOUT_MULTIPLIER = 150

    # Obstacle settings
    # Set to True to enable obstacles game mode
    # This will be controlled by a command-line argument in train.py
    # but the default here is for internal use if not overridden.
    ENABLE_OBSTACLES = False
    NUM_STATIC_OBSTACLES = 5 # Number of static obstacles
    STATIC_OBSTACLE_LENGTH_MIN = 2 # Min length of a static obstacle segment
    STATIC_OBSTACLE_LENGTH_MAX = 5 # Max length of a static obstacle segment
    # Dynamic obstacles are more complex to implement and manage state for DQL.
    # For now, we'll focus on static obstacles. Dynamic obstacles would require
    # a more sophisticated state representation and potentially a different
    # approach to ensure learnability.
    # DYNAMIC_OBSTACLE_CHANCE = 0.01 # Chance per step for a dynamic obstacle to appear/move
    # DYNAMIC_OBSTACLE_COUNT = 0 # Max number of dynamic obstacles at once
