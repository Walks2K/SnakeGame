"""
Custom Snake environment built to be played by a Reinforcement Learning agent.

To solve, we will implement an Actor-Critic model.

    Actor:
        The actor is a neural network that maps states to actions.
        The actor is trained to output the best action for a given state.

    Critic:
        The critic is a neural network that maps states to values.
        The critic is trained to output the value of a given state.

    Actor-Critic:
        The actor-critic model is a combination of the actor and critic.
        The actor-critic model is trained to maximize the value of the state.
"""

import random
import numpy as np
import pygame

# Neural Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# COLOURS    R    G    B
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


class DQN(nn.Module):
    """
    Deep Q-Network (DQN)
    """

    def __init__(self, state_size, action_size):
        """
        Initialize the DQN
        """
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Define the neural network layers
        self.fc1 = nn.Linear(state_size, 128, dtype=torch.float)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, action_size)

    def forward(self, state):
        """
        Forward propagation of the neural network
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


class SnakeEnv:
    """
    Custom Snake environment built to be played by a Reinforcement Learning agent.
    """

    def __init__(self,
                 width=40,
                 height=40,
                 n_snakes=1,
                 n_food=1,
                 n_obstacles=0,
                 n_steps=200,
                 n_episodes=1000,
                 max_steps=200,
                 max_episode_length=200,
                 min_episode_length=0,
                 min_episode_reward=0,
                 max_episode_reward=0,
                 render=False,
                 render_delay=0.1,
                 verbose=False,
                 seed=None):
        """
        Initialize the environment.

        :param width: int
            The width of the environment.
        :param height: int
            The height of the environment.
        :param n_snakes: int
            The number of snakes in the environment.
        :param n_food: int
            The number of food items in the environment.
        :param n_obstacles: int
            The number of obstacles in the environment.
        :param n_steps: int
            The number of steps to run the environment for.
        :param n_episodes: int
            The number of episodes to run the environment for.
        :param max_steps: int
            The maximum number of steps to run the environment for.
        :param max_episode_length: int
            The maximum number of steps to run an episode for.
        :param min_episode_length: int
            The minimum number of steps to run an episode for.
        :param min_episode_reward: float
            The minimum reward to be achieved by an episode.
        :param max_episode_reward: float
            The maximum reward to be achieved by an episode.
        :param render: bool
            Whether to render the environment.
        :param render_delay: float
            The delay between rendering the environment.
        :param verbose: bool
            Whether to print information about the environment.
        :param seed: int
            The random seed to use.
        """
        self.width = width
        self.height = height
        self.n_snakes = n_snakes
        self.n_food = n_food
        self.n_obstacles = n_obstacles
        self.n_steps = n_steps
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.max_episode_length = max_episode_length
        self.min_episode_length = min_episode_length
        self.min_episode_reward = min_episode_reward
        self.max_episode_reward = max_episode_reward
        self.render_enabled = render
        self.render_delay = render_delay
        self.verbose = verbose
        self.seed = seed
        self.state_size = (self.width, self.height, self.n_snakes, self.n_food,
                           self.n_obstacles)
        self.action_size = 4
        self.model = DQN(state_size=self.state_size,
                         action_size=self.action_size)

        # Initialize the environment.
        self.init()

    def select_action(self, state):
        """
        Select an action.

        :param state: np.array
            The current state.
        :return: int
            The selected action.
        """
        # Select an action.
        return self.model(torch.from_numpy(state).float()).argmax().item()

    def init(self):
        """
        Initialize the environment.
        """
        # Set the random seed.
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # Initialize the environment.
        self.reset()

    def reset(self):
        """
        Reset the environment.
        """
        # Initialize the environment.
        self.snakes = [
            Snake(self.width // 2, self.height // 2)
            for _ in range(self.n_snakes)
        ]
        self.food = [Food(self.width, self.height) for _ in range(self.n_food)]
        self.obstacles = [
            Obstacle(self.width, self.height) for _ in range(self.n_obstacles)
        ]

        # Initialize the environment.
        self.step_count = 0
        self.episode_count = 0
        self.episode_length = 0
        self.episode_reward = 0
        self.episode_lengths = []
        self.episode_rewards = []

        # Initialize the environment.
        self.done = False
        self.terminal = False
        self.reward = 0
        self.info = {}

        # Initialize the environment.
        self.state = self.get_state()

    def snake_checks(self):
        """
        Check for collisions between the snakes and the food.
        """
        # Loop through the snakes
        for snake in self.snakes:
            # Loop through the food
            for food in self.food:
                snake.check_food(food)

            # Check for collisions between the snakes and the walls.
            if snake.check_collision():
                # Set the terminal flag.
                self.terminal = True

            # Loop through the obstacles
            for obstacle in self.obstacles:
                # Check for collisions
                if snake.check_obstacle(obstacle):
                    # Set the collision flag
                    self.terminal = True

    def step(self, action):
        """
        Perform a step in the environment.

        :param action: int
            The action to perform.
        """
        # Turn the snake.
        self.snakes[0].turn(action)

        # Move the snake.
        self.snakes[0].move()

        # Perform a step in the environment.
        self.step_count += 1
        self.episode_length += 1
        self.episode_reward += self.reward

        # Run snake checks.
        self.snake_checks()

        # Check if the episode is done.
        self.done = self.episode_length >= self.max_episode_length or self.terminal

        # Check if the episode is done.
        if self.done:
            # Add the episode length to the episode lengths.
            self.episode_lengths.append(self.episode_length)

            # Add the episode reward to the episode rewards.
            self.episode_rewards.append(self.episode_reward)

            # Reset the episode.
            self.reset()

        # Get the state.
        self.state = self.get_state()

    def get_state(self):
        """
        Get the current state of the environment.

        :return: numpy.ndarray
            The current state of the environment.
        """
        # Get the current state of the environment.
        state = np.zeros((self.height, self.width,
                          self.n_snakes + self.n_food + self.n_obstacles + 1))

        # Get the current state of the environment.
        for snake in self.snakes:
            state[snake.y, snake.x, 0] = 1
        for food in self.food:
            state[food.y, food.x, 1] = 1
        for obstacle in self.obstacles:
            state[obstacle.y, obstacle.x, 2] = 1

        # Get the current state of the environment.
        return state

    def render(self):
        """
        Render the environment.
        """
        if self.render_enabled:
            pygame.init()
            screen = pygame.display.set_mode(
                (self.width * 10, self.height * 10))
            pygame.display.set_caption('Snake')
            clock = pygame.time.Clock()

            # Event checks.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Render the objects.
            for snake in self.snakes:
                snake.render(screen)

            for food in self.food:
                food.render(screen)

            for obstacle in self.obstacles:
                obstacle.render(screen)

            pygame.display.flip()
            clock.tick(30)

    def close(self):
        """
        Close the environment.
        """
        # Close the environment.
        pass

    def seed(self, seed=None):
        """
        Set the random seed.

        :param seed: int
            The random seed to use.
        """
        # Set the random seed.
        self.seed = seed

    def run(self):
        """
        Run the environment.
        """
        # Loop through non-terminal snakes
        for snake in self.snakes:
            if snake.terminal:
                continue

            # Loop through the steps
            for _ in range(self.n_steps):
                # Get the action
                action = self.select_action(self.state)

                # Step the environment
                self.step(action)

                # Render the environment
                self.render()

                # Check if the episode is done
                if self.done:
                    break


class Snake(object):
    """
    The snake class.
    """

    def __init__(self, width, height):
        """
        Initialize the snake.

        :param width: int
            The width of the environment.
        :param height: int
            The height of the environment.
        """
        # Initialize the snake.
        self.x = random.randint(0, width - 1)
        self.y = random.randint(0, height - 1)
        self.dx = 0
        self.dy = 0
        self.width = width
        self.height = height
        self.length = 4
        self.tail = []
        self.terminal = False

    def turn(self, action):
        """
        Turn the snake.

        :param action: int
            The action to perform.
        """
        # Turn the snake.
        if action == 0:
            self.dx = 0
            self.dy = -1
        elif action == 1:
            self.dx = 1
            self.dy = 0
        elif action == 2:
            self.dx = 0
            self.dy = 1
        elif action == 3:
            self.dx = -1
            self.dy = 0

    def move(self):
        """
        Move the snake.
        """
        # Move the snake.
        self.x += self.dx
        self.y += self.dy

        # Move the snake.
        self.tail.insert(0, (self.x, self.y))
        if len(self.tail) > self.length:
            self.tail.pop()

    def check_collision(self):
        """
        Check if the snake has collided with the environment or itself
        """
        # Check if the snake has collided with the environment.
        if self.x <= 0 or self.x >= self.width:
            self.terminal = True
        if self.y <= 0 or self.y >= self.height:
            self.terminal = True

        # Check if the snake has collided with itself.
        for tail in self.tail:
            if self.x == tail[0] and self.y == tail[1]:
                self.terminal = True

    def check_food(self, food):
        """
        Check if the snake has eaten the food.

        :param food: Food
            The food to check.
        """
        # Check if the snake has eaten the food.
        if self.x == food.x and self.y == food.y:
            self.length += 1
            food.respawn()
            return True
        return False

    def check_obstacle(self, obstacle):
        """
        Check if the snake has collided with the obstacle.

        :param obstacle: Obstacle
            The obstacle to check.
        """
        # Check if the snake has collided with the obstacle.
        if self.x == obstacle.x and self.y == obstacle.y:
            self.terminal = True

    def reset(self):
        """
        Reset the snake.
        """
        # Reset the snake.
        self.x = random.randint(0, self.width - 1)
        self.y = random.randint(0, self.height - 1)
        self.dx = 0
        self.dy = 0

    def render(self, surface):
        """
        Render the snake.

        :param screen: pygame.Surface
            The screen to render to.
        """
        pygame.draw.rect(surface, BLUE, (self.x * 10, self.y * 10, 10, 10))
        for tail in self.tail[1:]:
            pygame.draw.rect(surface, WHITE,
                             (tail[0] * 10, tail[1] * 10, 10, 10))


class Food(object):
    """
    The food class.
    """

    def __init__(self, width, height):
        """
        Initialize the food.

        :param width: int
            The width of the environment.
        :param height: int
            The height of the environment.
        """
        # Initialize the food.
        self.x = random.randint(0, width - 1)
        self.y = random.randint(0, height - 1)
        self.width = width
        self.height = height

    def respawn(self):
        """
        Respawn the food.
        """
        # Respawn the food.
        self.x = random.randint(0, self.width - 1)
        self.y = random.randint(0, self.height - 1)

    def render(self, surface):
        """
        Render the food.

        :param screen: pygame.Surface
            The screen to render to.
        """
        pygame.draw.rect(surface, GREEN, (self.x * 10, self.y * 10, 10, 10))


class Obstacle(object):
    """
    The obstacle class.
    """

    def __init__(self, width, height):
        """
        Initialize the obstacle.

        :param width: int
            The width of the environment.
        :param height: int
            The height of the environment.
        """
        # Initialize the obstacle.
        self.x = random.randint(0, width - 1)
        self.y = random.randint(0, height - 1)

    def render(self, surface):
        """
        Render the obstacle.

        :param screen: pygame.Surface
            The screen to render to.
        """
        pygame.draw.rect(surface, RED, (self.x * 10, self.y * 10, 10, 10))


if __name__ == '__main__':
    # Create the environment.
    environment = SnakeEnv(width=30,
                           height=30,
                           n_snakes=1,
                           n_food=1,
                           n_obstacles=5,
                           render=True)

    # Run the environment.
    environment.run()

    # Close the environment.
    environment.close()