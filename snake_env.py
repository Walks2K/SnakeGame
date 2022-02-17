"""
Custom environment for Snake
"""

import random
from collections import deque

import pygame
import numpy as np
import gym
from gym import spaces
import cv2


SNAKE_LEN_GOAL = 30


class Snake:
    """
    Snake class
    """

    def __init__(self, x, y, length=3):
        self.x = x
        self.y = y
        self.length = length
        self.start_length = length

        self.direction = "RIGHT"
        self.body = []
        self.body.append([self.x, self.y])
        for i in range(1, self.length):
            self.body.append([self.x - i, self.y])

    def update(self):
        """
        Update snake position
        """
        if self.direction == "RIGHT":
            self.x += 1
        elif self.direction == "LEFT":
            self.x -= 1
        elif self.direction == "UP":
            self.y -= 1
        elif self.direction == "DOWN":
            self.y += 1

        self.body.insert(0, [self.x, self.y])
        if len(self.body) > self.length:
            self.body.pop()

    def collide_boundary(self, grid_size):
        """
        Check if snake hits the boundary
        """
        if self.x >= grid_size[0] or self.x < 0 or self.y >= grid_size[1] or self.y < 0:
            return True
        return False

    def collide_snake(self):
        """
        Check if snake hits itself
        """
        for i in range(1, len(self.body)):
            if self.x == self.body[i][0] and self.y == self.body[i][1]:
                return True
        return False


class Food:
    """
    Food class
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y


def get_empty_space(snake, grid_size):
    """
    Get empty space on the screen
    """
    empty_space = []
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            if (x, y) not in snake.body:
                empty_space.append([x, y])
    return empty_space


class SnakeEnv(gym.Env):
    """
    Custom Snake environment that follows gym interface
    """

    def __init__(self, grid_size=(15, 15), fps=30, max_steps=200):
        """
        Initialize Snake environment
        """
        super(SnakeEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(12,), dtype=np.float32)

        self.grid_size = grid_size
        self.fps = fps
        self.max_steps = max_steps

    def step(self, action):
        """
        Step through the environment
        """
        if action == 0:
            self.snake.direction = "RIGHT"
        elif action == 1:
            self.snake.direction = "LEFT"
        elif action == 2:
            self.snake.direction = "UP"
        elif action == 3:
            self.snake.direction = "DOWN"

        self.snake.update()

        done = self.snake.collide_boundary(
            self.grid_size) or self.snake.collide_snake() or self.steps <= 0

        if not done:
            if self.snake.x == self.food.x and self.snake.y == self.food.y:
                self.snake.length += 1
                food_pos = random.choice(
                    get_empty_space(self.snake, self.grid_size))
                self.food = Food(food_pos[0], food_pos[1])
                reward = 10
                self.steps = self.max_steps
            else:
                reward = 0
        else:
            reward = -10

        observation = self.get_observation()
        return observation, reward, done, {}

    def reset(self):
        """
        Reset/initialize the environment
        """
        self.snake = Snake(self.grid_size[0] // 2, self.grid_size[1] // 2)
        food_pos = random.choice(get_empty_space(self.snake, self.grid_size))
        self.food = Food(food_pos[0], food_pos[1])
        self.cell_size = 20
        self.screen = pygame.display.set_mode(
            (self.grid_size[0] * self.cell_size, self.grid_size[1] * self.cell_size))
        pygame.display.set_caption("Snake")
        pygame.init()
        pygame.font.init()
        self.clock = pygame.time.Clock()
        self.steps = self.max_steps

        observation = self.get_observation()
        return observation

    def render(self):
        """
        Render the environment
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.screen.fill((0, 0, 0))
        for i in range(len(self.snake.body)):
            if i == 0:
                pygame.draw.rect(
                    self.screen, (0, 255, 0), (self.snake.body[i][0] * self.cell_size, self.snake.body[i][1] * self.cell_size, self.cell_size, self.cell_size))
            else:
                pygame.draw.rect(
                    self.screen, (255, 255, 255), (self.snake.body[i][0] * self.cell_size, self.snake.body[i][1] * self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(
            self.screen, (255, 0, 0), (self.food.x * self.cell_size, self.food.y * self.cell_size, self.cell_size, self.cell_size))
        pygame.display.update()
        self.clock.tick(self.fps)

    def get_observation(self):
        """
        Get observation of environment
        11 inputs:
        Danger Left, Danger Right, Danger Up, Danger Down - one hot
        Direction Left, Direction Right, Direction Up, Direction Down - one hot
        Food Left, Food Right, Food Up, Food Down - one hot

        Return a numpy array
        """
        observation = np.zeros(12)

        # Danger Left
        if self.snake.x - 1 == -1:
            observation[0] = 1
        elif [self.snake.x - 1, self.snake.y] in self.snake.body:
            observation[0] = 1

        # Danger Right
        if self.snake.x + 1 == self.grid_size[0]:
            observation[1] = 1
        elif [self.snake.x + 1, self.snake.y] in self.snake.body:
            observation[1] = 1

        # Danger Up
        if self.snake.y - 1 == -1:
            observation[2] = 1
        elif [self.snake.x, self.snake.y - 1] in self.snake.body:
            observation[2] = 1

        # Danger Down
        if self.snake.y + 1 == self.grid_size[1]:
            observation[3] = 1
        elif [self.snake.x, self.snake.y + 1] in self.snake.body:
            observation[3] = 1

        # Direction one-hot
        if self.snake.direction == "LEFT":
            observation[4] = 1
        elif self.snake.direction == "RIGHT":
            observation[5] = 1
        elif self.snake.direction == "UP":
            observation[6] = 1
        elif self.snake.direction == "DOWN":
            observation[7] = 1

        # Food direction one hot
        food_dist_x = self.snake.x - self.food.x
        food_dist_y = self.snake.y - self.food.y

        if food_dist_x < 0:
            observation[8] = 1
        elif food_dist_x > 0:
            observation[9] = 1

        if food_dist_y < 0:
            observation[10] = 1
        elif food_dist_y > 0:
            observation[11] = 1

        return observation
