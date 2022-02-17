"""
Snake to be played by NEAT algorithm.
"""

import multiprocessing
import os
import random

import neat
import numpy as np
import pygame

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

WIN_WIDTH = 300
WIN_HEIGHT = 300
CELL_SIZE = 20
FPS = 30

GENS = 250  # Set to 0 to run until fitness_threshold in config-feedforward.txt is reached
MAX_MOVES = 300
USE_CHECKPOINT = False
CHECKPOINT_GEN = '249'
CHECKPOINT_EVERY = 50
BINARY_VISION = False

assert WIN_WIDTH % CELL_SIZE == 0, "Window width must be a multiple of cell size."
assert WIN_HEIGHT % CELL_SIZE == 0, "Window height must be a multiple of cell size."


def distance(x1_pos, y1_pos, x2_pos, y2_pos):
    """
    Finds distance between two points.
    """
    return np.sqrt((x1_pos - x2_pos)**2 + (y1_pos - y2_pos)**2)


class Snake:
    """
    Snake class
    """

    def __init__(self, x_pos, y_pos, length):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.length = length
        self.direction = (1, 0)
        self.tail = [(x_pos, y_pos), (x_pos - 1, y_pos), (x_pos - 2, y_pos),
                     (x_pos - 3, y_pos)]
        self.score = 0
        self.hunger = MAX_MOVES

    def move(self):
        """
        Move the snake and reduce hunger.
        """
        self.x_pos += self.direction[0]
        self.y_pos += self.direction[1]
        self.hunger -= 1

    def add_cell(self):
        """
        Adds a cell to the snake's tail and removes the last cell if it exceeds the length.
        """
        self.tail.insert(0, (self.x_pos, self.y_pos))
        if self.length < len(self.tail) + 1:
            self.tail.pop()

    def eat(self, food):
        """
        Food handler.

        Args:
            food (Food): Food class instance to use.

        Returns:
            Boolean: True if food is eaten else false.
        """
        if self.x_pos == food.x_pos and self.y_pos == food.y_pos:
            self.score += 1
            self.length += 1
            self.hunger = MAX_MOVES
            return True
        return False

    def check_collision(self):
        """
        Check if snake has collided with anything

        Returns:
            Boolean: True if collision else false.
        """
        if self.hunger <= 0:
            return True
        if self.x_pos < 0 or self.x_pos >= WIN_WIDTH // CELL_SIZE or self.y_pos < 0 or self.y_pos \
                >= WIN_HEIGHT // CELL_SIZE:
            return True
        for cell in self.tail[1:]:
            if self.x_pos == cell[0] and self.y_pos == cell[1]:
                return True
        return False

    def draw(self, win):
        """
        Draw the snake

        Args:
            win (pygame window): Window to draw in
        """
        for cell in self.tail:
            pygame.draw.rect(win, GREEN if cell == self.tail[0] else WHITE,
                             (cell[0] * CELL_SIZE, cell[1] * CELL_SIZE,
                              CELL_SIZE, CELL_SIZE))

    def change_direction(self, direction):
        """
        Change the snake's direction.
        """
        # if direction[0] != -self.direction[0] or direction[1] != -self.direction[1]:
        self.direction = direction

    def vision(self, food_obj):
        """
        Calculate vision
        """
        # Fire a ray in 8 directions and return:
        # - distance to wall in that direction,
        # - 0/1 (binary) or dist if there is food in that direction
        # - 0/1 (binary) or dist if tail in that direction
        vision = []
        for direction in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1),
                          (1, -1), (-1, 1)]:
            dist = 0
            food = 0
            tail = 0
            x_pos = self.x_pos + direction[0]
            y_pos = self.y_pos + direction[1]
            while True:
                if x_pos < 0 or x_pos >= WIN_WIDTH // CELL_SIZE or y_pos < 0 or y_pos \
                        >= WIN_HEIGHT // CELL_SIZE:
                    dist = distance(self.x_pos, self.y_pos, x_pos, y_pos)
                    # scale between 0 and 1 and round to 2 decimal places
                    dist = round(
                        dist / (WIN_WIDTH // CELL_SIZE if direction[1] == 0 else WIN_HEIGHT // CELL_SIZE), 2)
                    break
                for cell in self.tail[1:]:
                    if x_pos == cell[0] and y_pos == cell[1]:
                        tail = 1 if BINARY_VISION else round(distance(self.x_pos, self.y_pos, x_pos, y_pos) /
                                                             (WIN_WIDTH // CELL_SIZE + WIN_HEIGHT // CELL_SIZE), 2)
                if food == 0:
                    if x_pos == food_obj.x_pos and y_pos == food_obj.y_pos:
                        food = 1 if BINARY_VISION else round(distance(self.x_pos, self.y_pos, x_pos, y_pos) /
                                                             (WIN_WIDTH // CELL_SIZE + WIN_HEIGHT // CELL_SIZE), 2)
                x_pos += direction[0]
                y_pos += direction[1]

            vision.append([dist, food, tail])
        return vision


class Food:
    """
    Food class
    """

    def __init__(self):
        self.x_pos = random.randint(0, WIN_WIDTH / CELL_SIZE - 1)
        self.y_pos = random.randint(0, WIN_HEIGHT / CELL_SIZE - 1)

    def draw(self, win):
        """
        Draw food

        Args:
            win (pygame window): Window to draw to
        """
        pygame.draw.rect(
            win, RED,
            (self.x_pos * CELL_SIZE, self.y_pos * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def spawn(self, snake):
        """
        Spawns new food

        Args:
            snake (Snake): Snake class instance
        """
        while True:
            self.x_pos = random.randint(0, WIN_WIDTH // CELL_SIZE - 1)
            self.y_pos = random.randint(0, WIN_HEIGHT // CELL_SIZE - 1)
            if (self.x_pos, self.y_pos) not in snake.tail:
                break


def draw_window(win, snake, food):
    """
    Draw the window
    """
    win.fill(BLACK)
    snake.draw(win)
    food.draw(win)
    pygame.display.update()


def neat_inputs(snake, food):
    """
    Calculate inputs for NEAT

    Args:
        snake (Snake): Snake class instance
        food (Food): Food class instance

    Returns:
        List: list of inputs for NEAT
    """
    inputs = []

    # get vision from snake - 24 inputs (8 directions * 3 values)
    vision = snake.vision(food)
    # enumerate vision
    for _, vis in enumerate(vision):
        inputs.append(vis[0])
        inputs.append(vis[1])
        inputs.append(vis[2])

    # flatten direction vector for one hot variables - 4 inputs
    if snake.direction == (1, 0):  # right
        inputs.append(1)
        inputs.append(0)
        inputs.append(0)
        inputs.append(0)
    elif snake.direction == (-1, 0):  # left
        inputs.append(0)
        inputs.append(1)
        inputs.append(0)
        inputs.append(0)
    elif snake.direction == (0, 1):  # down
        inputs.append(0)
        inputs.append(0)
        inputs.append(1)
        inputs.append(0)
    elif snake.direction == (0, -1):  # up
        inputs.append(0)
        inputs.append(0)
        inputs.append(0)
        inputs.append(1)

    return inputs


def eval_genome(genome, config):
    """
    Evaluates genomes
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitness = 0
    snake = Snake((WIN_WIDTH / CELL_SIZE) // 2, (WIN_HEIGHT / CELL_SIZE) // 2,
                  4)
    food = Food()
    done = False

    while not done:

        inputs = neat_inputs(snake, food)
        outputs = net.activate(inputs)

        # Get direction from max of output vector
        if max(outputs) == outputs[0]:
            snake.change_direction((1, 0))
        elif max(outputs) == outputs[1]:
            snake.change_direction((-1, 0))
        elif max(outputs) == outputs[2]:
            snake.change_direction((0, 1))
        elif max(outputs) == outputs[3]:
            snake.change_direction((0, -1))

        snake.move()
        snake.add_cell()

        if snake.check_collision():
            fitness -= 100
            done = True
            break

        if snake.eat(food):
            food = Food()
            food.spawn(snake)
            fitness += 10

        fitness += 0.01
        # draw_window(WIN, snake, food)

    return fitness


def eval_genomes(genomes, config):
    """
    Wrapper for evaluating genomes
    """
    for _, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run(config_path):
    """
    Run program
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    pop = neat.Population(config)

    if USE_CHECKPOINT:
        pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-' +
                                                   str(CHECKPOINT_GEN))

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(CHECKPOINT_EVERY))

    # winner = p.run(eval_genomes, 50)
    para_eval = neat.ParallelEvaluator(
        multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(para_eval.evaluate, GENS if GENS > 0 else None)
    del para_eval

    window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Snake")

    print(f"Best genome: {winner}")

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    snake = Snake((WIN_WIDTH / CELL_SIZE) // 2, (WIN_HEIGHT / CELL_SIZE) // 2,
                  4)
    food = Food()
    done = False
    clock = pygame.time.Clock()
    wait = True

    while wait:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    wait = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                pygame.quit()
                quit()

        clock.tick(FPS)

        inputs = neat_inputs(snake, food)
        outputs = winner_net.activate(inputs)

        # Get direction from max of output vector
        if max(outputs) == outputs[0]:
            snake.change_direction((1, 0))
        elif max(outputs) == outputs[1]:
            snake.change_direction((-1, 0))
        elif max(outputs) == outputs[2]:
            snake.change_direction((0, 1))
        elif max(outputs) == outputs[3]:
            snake.change_direction((0, -1))

        snake.move()
        snake.add_cell()

        if snake.check_collision():
            done = True
            print(f"Score: {snake.score}")
            print(inputs)
            print(outputs)
            break

        if snake.eat(food):
            food = Food()
            food.spawn(snake)

        draw_window(window, snake, food)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    cfg_path = os.path.join(local_dir, "config-feedforward.txt")
    run(cfg_path)
