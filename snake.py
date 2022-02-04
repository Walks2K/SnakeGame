import pygame
import neat
import time
import os
import random
import numpy as np

pygame.font.init()

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

WIN_WIDTH = 600
WIN_HEIGHT = 600
CELL_SIZE = 20
FPS = 24

assert WIN_WIDTH % CELL_SIZE == 0, "Window width must be a multiple of cell size."
assert WIN_HEIGHT % CELL_SIZE == 0, "Window height must be a multiple of cell size."

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Snake")

CLOCK = pygame.time.Clock()


def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class Snake:

    def __init__(self, x, y, length):
        self.x = x
        self.y = y
        self.length = length
        self.direction = (1, 0)
        self.tail = []
        self.score = 0
        self.fitness = 0
        self.hunger = 100

    def move(self):
        self.x += self.direction[0]
        self.y += self.direction[1]
        self.hunger -= 1

    def add_cell(self):
        if self.length < len(self.tail) + 1:
            self.tail.pop()
        self.tail.insert(0, (self.x, self.y))

    def eat(self, food):
        if self.x == food.x and self.y == food.y:
            self.score += 1
            self.length += 1
            self.hunger += 100
            return True
        return False

    def check_collision(self):
        if self.hunger <= 0:
            return True
        if self.x < 0 or self.x >= WIN_WIDTH // CELL_SIZE or self.y < 0 or self.y >= WIN_HEIGHT // CELL_SIZE:
            return True
        for cell in self.tail[1:]:
            if self.x == cell[0] and self.y == cell[1]:
                return True
        return False

    def draw(self, win):
        for cell in self.tail:
            pygame.draw.rect(win, GREEN if cell == self.tail[0] else WHITE,
                             (cell[0] * CELL_SIZE, cell[1] * CELL_SIZE,
                              CELL_SIZE, CELL_SIZE))

    def change_direction(self, direction):
        self.direction = direction

    def straight_obstacle_dist(self):
        """
        Fire a raytrace from x, y in direction we are facing to find nearest obstacle.
        """
        x = self.x
        y = self.y
        while True:
            x += self.direction[0]
            y += self.direction[1]
            """
            Boundary check
            """
            if x < 0 or x >= WIN_WIDTH // CELL_SIZE or y < 0 or y >= WIN_HEIGHT // CELL_SIZE:
                return distance(self.x, self.y, x, y)
            """
            Tail check
            """
            for cell in self.tail[1:]:
                if x == cell[0] and y == cell[1]:
                    return distance(self.x, self.y, x, y)


class Food:

    def __init__(self):
        self.x = random.randint(0, WIN_WIDTH / CELL_SIZE - 1)
        self.y = random.randint(0, WIN_HEIGHT / CELL_SIZE - 1)

    def draw(self, win):
        pygame.draw.rect(
            win, RED,
            (self.x * CELL_SIZE, self.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))


def draw_window(win, snake, food):
    win.fill((0, 0, 0))
    snake.draw(win)
    food.draw(win)
    pygame.display.update()


def neat_inputs(snake, food):
    inputs = []
    inputs.append((food.x - snake.x))
    inputs.append((food.y - snake.y))
    inputs.append(snake.straight_obstacle_dist())

    return inputs


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitness = 0
    snake = Snake((WIN_WIDTH / CELL_SIZE) // 2, (WIN_HEIGHT / CELL_SIZE) // 2,
                  4)
    food = Food()
    run = True
    clock = pygame.time.Clock()
    while run:
        clock.tick()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        if snake.check_collision():
            run = False
            fitness -= (100 - snake.score)

        inputs = neat_inputs(snake, food)
        outputs = net.activate(inputs)
        if outputs[0] > 0.5:
            snake.change_direction((1, 0))
        elif outputs[1] > 0.5:
            snake.change_direction((-1, 0))
        elif outputs[2] > 0.5:
            snake.change_direction((0, 1))
        elif outputs[3] > 0.5:
            snake.change_direction((0, -1))

        snake.move()
        snake.add_cell()

        if snake.eat(food):
            food = Food()
            fitness += 100

        fitness += 0.1
        draw_window(WIN, snake, food)

    return fitness


def eval_genomes(genomes, config):
    for _, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, 15)
    print(f"Best genome: {winner}")

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    snake = Snake((WIN_WIDTH / CELL_SIZE) // 2, (WIN_HEIGHT / CELL_SIZE) // 2,
                  4)
    food = Food()
    run = True
    clock = pygame.time.Clock()
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        if snake.check_collision():
            run = False
            print(f"Score: {snake.score}")

        inputs = neat_inputs(snake, food)
        outputs = winner_net.activate(inputs)
        if outputs[0] > 0.5:
            snake.change_direction((1, 0))
        elif outputs[1] > 0.5:
            snake.change_direction((-1, 0))
        elif outputs[2] > 0.5:
            snake.change_direction((0, 1))
        elif outputs[3] > 0.5:
            snake.change_direction((0, -1))

        snake.move()
        snake.add_cell()

        if snake.eat(food):
            food = Food()

        draw_window(WIN, snake, food)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
