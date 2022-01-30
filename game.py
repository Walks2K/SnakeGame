"""
Snake Game using pygame
We will define a grid of cells, and each cell will be a square.
The snake will move around the grid, and the food will be placed randomly.
The snake will grow as it eats the food.
The snake will die if it runs into itself or the edge of the grid.
"""

import pygame
import random
import sys
import time

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Define some global variables
CELL_SIZE = 20
ROWS = 30
COLUMNS = 30
SCREEN_WIDTH = COLUMNS * CELL_SIZE
SCREEN_HEIGHT = ROWS * CELL_SIZE
FPS = 30


# Define the snake class
class Snake:

    def __init__(self):
        self.position = [COLUMNS // 2, ROWS // 2]
        self.body = []
        self.length = 4
        self.direction = "RIGHT"

    def update(self):
        if self.direction == "RIGHT":
            self.position[0] += 1
        elif self.direction == "LEFT":
            self.position[0] -= 1
        elif self.direction == "UP":
            self.position[1] -= 1
        elif self.direction == "DOWN":
            self.position[1] += 1

        self.body.insert(0, list(self.position))
        if len(self.body) > self.length:
            self.body.pop()

    def draw(self, surface):
        for part in self.body:
            pygame.draw.rect(surface, GREEN,
                             (part[0] * CELL_SIZE, part[1] * CELL_SIZE,
                              CELL_SIZE, CELL_SIZE))

    def check_collision(self):
        if self.position in self.body[1:]:
            return True
        elif self.position[0] > COLUMNS - 1 or self.position[0] < 0:
            return True
        elif self.position[1] > ROWS - 1 or self.position[1] < 0:
            return True
        else:
            return False


# Define the food class
class Food:

    def __init__(self):
        self.position = [0, 0]
        self.eaten = False

    def spawn(self):
        self.position = [
            random.randint(0, COLUMNS - 1),
            random.randint(0, ROWS - 1)
        ]
        self.eaten = False

    def draw(self, surface):
        if not self.eaten:
            pygame.draw.rect(surface, RED,
                             (self.position[0] * CELL_SIZE, self.position[1] *
                              CELL_SIZE, CELL_SIZE, CELL_SIZE))


# Define the game class
class Game:

    def __init__(self):
        self.snake = Snake()
        self.food = Food()
        self.food.spawn()

    def update(self):
        self.snake.update()

        if self.snake.position == self.food.position:
            self.food.eaten = True
            self.snake.length += 1
            self.food.spawn()

        if self.check_collision():
            self.gameover(screen)
            self.reset()

    def check_collision(self):
        return self.snake.check_collision()

    def draw(self, surface):
        surface.fill(BLACK)
        self.snake.draw(surface)
        self.food.draw(surface)

    def gameover(self, surface):
        restart_delay = time.time() + 3
        while time.time() < restart_delay:
            surface.fill(BLACK)

            font = pygame.font.SysFont("monospace", 72)
            text = font.render("Game Over", True, WHITE)
            text_rect = text.get_rect()
            text_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
            surface.blit(text, text_rect)

            font = pygame.font.SysFont("monospace", 36)
            text = font.render(
                "Restarting in " + str(int(restart_delay - time.time()) + 1),
                True, WHITE)
            text_rect = text.get_rect()
            text_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50)
            surface.blit(text, text_rect)

            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

    def reset(self):
        self.snake = Snake()
        self.food = Food()
        self.food.spawn()


# Initialize pygame
pygame.init()

# Create the game object
game = Game()

# Create the window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake Game")

# Create the clock
clock = pygame.time.Clock()

# Loop until the user clicks the close button.
done = False

# -------- Main Program Loop -----------
while not done:
    # --- Main event loop
    # Ensure only one event is processed at a time
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                done = True
            if event.key == pygame.K_RIGHT:
                if (game.snake.direction != "LEFT"):
                    game.snake.direction = "RIGHT"
            elif event.key == pygame.K_LEFT:
                if (game.snake.direction != "RIGHT"):
                    game.snake.direction = "LEFT"
            elif event.key == pygame.K_UP:
                if (game.snake.direction != "DOWN"):
                    game.snake.direction = "UP"
            elif event.key == pygame.K_DOWN:
                if (game.snake.direction != "UP"):
                    game.snake.direction = "DOWN"

    game.update()
    game.draw(screen)
    pygame.display.flip()

    # --- Limit to FPS
    clock.tick(FPS)