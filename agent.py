import torch
import random
import numpy as np
import math
from collections import deque
from snake_gameai import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
# --- FIX: Removed the unnecessary import of 'plot' ---
# from plots import plot 
import os
import pickle

# --- Constants & Hyperparameters ---
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0005
GAMMA = 0.9

MODEL_DIR = './model_weights' # Dedicated directory for saving models and memory
MAIN_MODEL_FILE = 'snake_q_model.pth'
TARGET_MODEL_FILE = 'snake_target_q_model.pth'
MEMORY_FILE = 'replay_memory.pkl'

UPDATE_TARGET_EVERY_GAMES = 100

EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY_RATE = 0.995

# Set device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = EPSILON_START
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = Linear_QNet(11, 256, 128, 3).to(device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        self.target_model = Linear_QNet(11, 256, 128, 3).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self._load_data()

    def _load_data(self):
        """Load pre-trained models and replay memory if they exist."""
        main_model_path = os.path.join(MODEL_DIR, MAIN_MODEL_FILE)
        target_model_path = os.path.join(MODEL_DIR, TARGET_MODEL_FILE)
        memory_path = os.path.join(MODEL_DIR, MEMORY_FILE)

        if os.path.exists(main_model_path):
            try:
                self.model.load_state_dict(torch.load(main_model_path, map_location=device))
                print(f"✅ Main Q-Network loaded from {main_model_path}")
            except Exception as e:
                print(f"❌ Error loading main Q-Network: {e}. Starting fresh.")
        else:
            print("🆕 No saved main Q-Network found. Starting fresh.")

        if os.path.exists(target_model_path):
            try:
                self.target_model.load_state_dict(torch.load(target_model_path, map_location=device))
                self.target_model.eval()
                print(f"✅ Target Q-Network loaded from {target_model_path}")
            except Exception as e:
                print(f"❌ Error loading target Q-Network: {e}. Reinitializing from main model.")
                self.target_model.load_state_dict(self.model.state_dict())
                self.target_model.eval()
        else:
            print("🆕 No saved target Q-Network found. Initializing from main Q-Network.")
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()

        if os.path.exists(memory_path):
            try:
                with open(memory_path, 'rb') as f:
                    loaded_memory_list = pickle.load(f)
                    self.memory = deque(loaded_memory_list, maxlen=MAX_MEMORY)
                    print(f"✅ Replay memory loaded from {memory_path} ({len(self.memory)} experiences)")
            except Exception as e:
                print(f"❌ Error loading replay memory: {e}. Starting with empty memory.")
        else:
            print("🆕 No saved replay memory found. Starting with empty memory.")

    def get_state(self, game):
        """
        Extracts the current state of the game relevant to the agent.
        Returns a numpy array of 11 boolean values.
        """
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)) or (dir_l and game.is_collision(point_l)) or (dir_u and game.is_collision(point_u)) or (dir_d and game.is_collision(point_d)),
            (dir_u and game.is_collision(point_r)) or (dir_d and game.is_collision(point_l)) or (dir_l and game.is_collision(point_u)) or (dir_r and game.is_collision(point_d)),
            (dir_d and game.is_collision(point_r)) or (dir_u and game.is_collision(point_l)) or (dir_r and game.is_collision(point_u)) or (dir_l and game.is_collision(point_d)),
            dir_l, dir_r, dir_u, dir_d,
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience in the replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """
        Trains the Q-network using a random batch from the replay memory.
        Returns the loss from this training step.
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones, target_model=self.target_model)
        return loss

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Trains the Q-network on the single most recent experience.
        Returns the loss from this training step.
        """
        loss = self.trainer.train_step(state, action, reward, next_state, done, target_model=self.target_model)
        return loss

    def get_action(self, state):
        """
        Selects an action using an epsilon-greedy strategy.
        Returns the action and the max Q-value for metrics.
        """
        self.epsilon = max(EPSILON_MIN, EPSILON_START * (EPSILON_DECAY_RATE ** self.n_games))
        final_move = [0, 0, 0]
        
        max_q = 0

        if random.random() < self.epsilon:
            move = random.randint(0, 2)
        else:
            state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            max_q = torch.max(prediction).item()

        final_move[move] = 1
        return final_move, max_q

    def save_data(self):
        """Save the main Q-network, target Q-network, and replay memory."""
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        torch.save(self.model.state_dict(), os.path.join(MODEL_DIR, MAIN_MODEL_FILE))
        torch.save(self.target_model.state_dict(), os.path.join(MODEL_DIR, TARGET_MODEL_FILE))
        print(f"🧠 Models saved: {MAIN_MODEL_FILE}, {TARGET_MODEL_FILE}")

        try:
            with open(os.path.join(MODEL_DIR, MEMORY_FILE), 'wb') as f:
                pickle.dump(list(self.memory), f)
            print(f"💾 Replay memory saved to {os.path.join(MODEL_DIR, MEMORY_FILE)}")
        except Exception as e:
            print(f"❌ Error saving replay memory: {e}")

    def update_target_network(self):
        """Copies weights from the main Q-network to the target Q-network."""
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        print(f"🎯 Target network updated at game {self.n_games}")

