import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math

# Initialize all imported Pygame modules
pygame.init()

# Use a system default font as a fallback if 'arial.ttf' is not found
try:
    font = pygame.font.Font('arial.ttf', 25)
except FileNotFoundError:
    font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    """Enumeration for the four possible directions of movement."""
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# A named tuple for representing 2D points, making code more readable
Point = namedtuple('Point', 'x, y')

# RGB color constants
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)

BLOCK_SIZE = 20
SPEED = 40 # Controls the visualization speed of the Pygame window

class SnakeGameAI:
    """
    Manages the Snake game environment, including state, rules, and rendering.
    This class serves as the 'Environment' in the Reinforcement Learning paradigm.
    """
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # Initialize display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI Training')
        self.clock = pygame.time.Clock()
        self.reset() # Initialize the game state for the first episode

    def reset(self):
        """
        Resets the game to its initial state for a new episode.
        This is a critical function for RL, called at the start of every new game.
        """
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        # frame_iteration tracks the number of steps in the current episode.
        # It's used for metrics (episode length) and to prevent infinite loops.
        self.frame_iteration = 0
        self._place_food()
        self.prev_distance = self._get_food_distance()

    def _place_food(self):
        """Places food randomly on the grid, ensuring it doesn't overlap with the snake."""
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    def _get_food_distance(self):
        """Calculates the Manhattan distance from the snake's head to the food."""
        return abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

    def play_step(self, action):
        """
        Performs one step of the game based on the AI's chosen action.
        This is the main interface between the agent and the environment.
        MODIFIED to include more sophisticated reward shaping.

        Args:
            action (list): A one-hot encoded list [straight, right, left].

        Returns:
            tuple: (reward, game_over, score)
        """
        self.frame_iteration += 1

        # Handle Pygame events (e.g., closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Update the snake's head position based on the action
        self._move(action)
        self.snake.insert(0, self.head)

        # --- Enhanced Reward Engineering ---
        reward = 0
        game_over = False

        # Check for game-ending collisions or timeouts
        if self.is_collision() or self.frame_iteration > 150 * len(self.snake):
            game_over = True
            reward = -10 # Large negative reward for dying
            return reward, game_over, self.score

        # Check for food consumption
        if self.head == self.food:
            self.score += 1
            reward = 10 # Large positive reward for eating food
            self._place_food()
            self.prev_distance = self._get_food_distance()
        else:
            # --- Reward Shaping for non-food steps ---
            self.snake.pop() # Remove tail if no food is eaten
            new_distance = self._get_food_distance()
            # If the snake gets closer to the food, give a small positive reward
            if new_distance < self.prev_distance:
                reward = 0.1
            # If the snake moves away from the food, give a small negative reward
            else:
                reward = -0.15
            
            self.prev_distance = new_distance


        # Update UI and control game speed
        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score

    def _update_ui(self):
        """Updates the Pygame display with the current game state."""
        self.display.fill(BLACK)

        # Draw grid lines for better visual context
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, GRAY, (x, 0), (x, self.h))
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, GRAY, (0, y), (self.w, y))

        # Draw snake
        for idx, pt in enumerate(self.snake):
            color = BLUE2 if idx == 0 else BLUE1
            pygame.draw.rect(self.display, color, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Render score text
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])

        pygame.display.flip()

    def _move(self, action):
        """
        Updates the snake's direction and head based on the relative action.
        Action is a one-hot encoded list: [1,0,0] -> Straight, [0,1,0] -> Right, [0,0,1] -> Left.
        """
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]): # Go Straight
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]): # Turn Right (clockwise)
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else: # [0, 0, 1] Turn Left (counter-clockwise)
            next_idx = (idx - 1 + 4) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def is_collision(self, pt=None):
        """
        Checks for collision with game boundaries or the snake's own body.
        """
        if pt is None:
            pt = self.head

        # Hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True

        return False

if __name__ == '__main__':
    # This block allows for independent testing of the game environment.
    game = SnakeGameAI()
    while True:
        # Simulate a basic action (always go straight)
        action_for_test = [1, 0, 0]
        reward, done, score = game.play_step(action_for_test)

        if done:
            print(f'Final Score: {score}, Reward: {reward}')
            game.reset()
