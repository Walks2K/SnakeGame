"""
Custom environment for Snake using OpenAI Gym

Actions:
0: left
1: right
2: up
3: down

Observations:
food above: 0/1 | 0
food below: 0/1 | 1
food left: 0/1 | 2
food right: 0/1 | 3
obstacle (wall/own body) above: 0/1 | 4
obstacle (wall/own body) below: 0/1 | 5
obstacle (wall/own body) left: 0/1 | 6
obstacle (wall/own body) right: 0/1 | 7
direction: 0/1/2/3 | 8

Rewards:
-1: moved away from food
+1: moved towards food
-10: hit wall/own body
+10: ate food
"""

import random

import pygame
import numpy as np
import gym
from gym import spaces
import cv2


class SnakeEnv(gym.Env):
    """
    Custom Environment that follows gym interface

    Args:
        width (int): width of the grid
        height (int): height of the grid
        fps (int): frames per second
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, width=10, height=10, fps=10):
        super(SnakeEnv, self).__init__()

        self.width = width
        self.height = height
        self.fps = fps

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(9,), dtype=np.uint8)

        self.viewer = None

        self.reset()

    def reset(self):
        self.snake = [[self.width // 2, self.height // 2]]
        self.food = self._new_food()
        self.direction = [0, 0]
        self.score = 0
        self.done = False

        return self._get_obs()

    def step(self, action):
        self._action(action)
        self._move()
        reward, done = self._get_reward()
        obs = self._get_obs()

        return obs, reward, done, {}

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = pygame.display.set_mode((self.width * 10, self.height * 10))

        self.viewer.fill((0, 0, 0))

        for x, y in self.snake:
            pygame.draw.rect(
                self.viewer,
                (0, 255, 0),
                pygame.Rect(x * 10, y * 10, 10, 10),
            )

        pygame.draw.rect(
            self.viewer,
            (255, 0, 0),
            pygame.Rect(self.food[0] * 10, self.food[1] * 10, 10, 10),
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        pygame.display.flip()
        pygame.time.Clock().tick(self.fps)

    def close(self):
        if self.viewer:
            pygame.display.quit()
            self.viewer = None

    def _action(self, action):
        if action == 0:
            self.direction = [-1, 0]
        elif action == 1:
            self.direction = [1, 0]
        elif action == 2:
            self.direction = [0, -1]
        elif action == 3:
            self.direction = [0, 1]

    def _move(self):
        new_head = [
            self.snake[0][0] + self.direction[0],
            self.snake[0][1] + self.direction[1],
        ]

        if (
            new_head[0] < 0
            or new_head[0] >= self.width
            or new_head[1] < 0
            or new_head[1] >= self.height
        ):
            self.done = True
            return

        if new_head in self.snake:
            self.done = True
            return

        self.snake.insert(0, new_head)

        if new_head != self.food:
            self.snake.pop()
        else:
            self.food = self._new_food()

    def _new_food(self):
        while True:
            food = [random.randrange(self.width), random.randrange(self.height)]
            if food not in self.snake:
                return food

    def _get_reward(self):
        if self.done:
            return -10, True

        if self.snake[0] == self.food:
            return 10, False

        if self.direction == [1, 0]:
            if self.food[0] > self.snake[0][0]:
                return 1, False
            else:
                return -1, False
        elif self.direction == [-1, 0]:
            if self.food[0] < self.snake[0][0]:
                return 1, False
            else:
                return -1, False
        elif self.direction == [0, 1]:
            if self.food[1] > self.snake[0][1]:
                return 1, False
            else:
                return -1, False
        elif self.direction == [0, -1]:
            if self.food[1] < self.snake[0][1]:
                return 1, False
            else:
                return -1, False

    def _get_obs(self):
        obs = np.zeros(9, dtype=np.uint8)

        if self.food[0] < self.snake[0][0]:
            obs[0] = 1
        elif self.food[0] > self.snake[0][0]:
            obs[1] = 1

        if self.food[1] < self.snake[0][1]:
            obs[2] = 1
        elif self.food[1] > self.snake[0][1]:
            obs[3] = 1

        if (
            self.snake[0][0] == 0
            or [self.snake[0][0] - 1, self.snake[0][1]] in self.snake
        ):
            obs[4] = 1
        if (
            self.snake[0][0] == self.width - 1
            or [self.snake[0][0] + 1, self.snake[0][1]] in self.snake
        ):
            obs[5] = 1
        if (
            self.snake[0][1] == 0
            or [self.snake[0][0], self.snake[0][1] - 1] in self.snake
        ):
            obs[6] = 1
        if (
            self.snake[0][1] == self.height - 1
            or [self.snake[0][0], self.snake[0][1] + 1] in self.snake
        ):
            obs[7] = 1

        if self.direction == [-1, 0]:
            obs[8] = 0
        elif self.direction == [1, 0]:
            obs[8] = 1
        elif self.direction == [0, -1]:
            obs[8] = 2
        elif self.direction == [0, 1]:
            obs[8] = 3

        return obs

    def _get_image(self):
        img = np.zeros((self.width * 10, self.height * 10, 3), dtype=np.uint8)
        for x, y in self.snake:
            img[x * 10 : x * 10 + 10, y * 10 : y * 10 + 10] = (0, 255, 0)
        img[
            self.food[0] * 10 : self.food[0] * 10 + 10,
            self.food[1] * 10 : self.food[1] * 10 + 10,
        ] = (255, 0, 0)
        return img

    def get_image(self):
        return cv2.cvtColor(self._get_image(), cv2.COLOR_RGB2BGR)
