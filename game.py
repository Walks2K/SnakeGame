import pygame
import random
import sys
import time

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

CELL_SIZE = 20
ROWS = 30
COLUMNS = 30
SCREEN_WIDTH = COLUMNS * CELL_SIZE
SCREEN_HEIGHT = ROWS * CELL_SIZE
FPS = 30


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
        # first part is green, second part is white
        for i in range(len(self.body)):
            pygame.draw.rect(surface, GREEN if i == 0 else WHITE, [
                self.body[i][0] * CELL_SIZE, self.body[i][1] * CELL_SIZE,
                CELL_SIZE, CELL_SIZE
            ])

    def check_collision(self):
        if self.position in self.body[1:]:
            return True
        elif self.position[0] > COLUMNS - 1 or self.position[0] < 0:
            return True
        elif self.position[1] > ROWS - 1 or self.position[1] < 0:
            return True
        else:
            return False


class Food:

    def __init__(self):
        self.position = [0, 0]
        self.eaten = False

    def update(self, snake, player):
        if self.position == snake.position:
            self.eaten = True
            snake.length += 1
            player.score += 1
        if self.eaten:
            self.spawn()

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


class Player:

    def __init__(self):
        self.score = 0

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    if (game.snake.direction != "LEFT"):
                        game.snake.direction = "RIGHT"
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    if (game.snake.direction != "RIGHT"):
                        game.snake.direction = "LEFT"
                elif event.key == pygame.K_UP or event.key == pygame.K_w:
                    if (game.snake.direction != "DOWN"):
                        game.snake.direction = "UP"
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    if (game.snake.direction != "UP"):
                        game.snake.direction = "DOWN"


class Game:

    def __init__(self):
        self.snake = Snake()
        self.food = Food()
        self.player = Player()
        self.food.spawn()

    def update(self):
        self.player.update()
        self.snake.update()
        self.food.update(self.snake, self.player)

        if self.check_collision():
            self.gameover(screen)
            self.reset()

    def check_collision(self):
        return self.snake.check_collision()

    def draw(self, surface):
        surface.fill(BLACK)
        self.snake.draw(surface)
        self.food.draw(surface)
        font = pygame.font.SysFont("monospace", 24)
        text = font.render("Score: " + str(self.player.score), True, WHITE)
        text_rect = text.get_rect()
        text_rect.topleft = (0, 0)
        surface.blit(text, text_rect)

    def gameover(self, surface):
        restart_delay = time.time() + 3
        while time.time() < restart_delay:
            surface.fill(BLACK)

            font = pygame.font.SysFont("monospace", 40)
            text = font.render("GAME OVER", True, WHITE)
            text_rect = text.get_rect()
            text_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 45)
            surface.blit(text, text_rect)

            font = pygame.font.SysFont("monospace", 25)
            text = font.render(
                "Restarting in " + str(int(restart_delay - time.time() + 1)),
                True, WHITE)
            text_rect = text.get_rect()
            text_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
            surface.blit(text, text_rect)

            text = font.render("Score: " + str(self.player.score), True, WHITE)
            text_rect = text.get_rect()
            text_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30)
            surface.blit(text, text_rect)

            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

    def reset(self):
        self.snake = Snake()
        self.food = Food()
        self.player = Player()
        self.food.spawn()


pygame.init()
game = Game()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake Game")
clock = pygame.time.Clock()

while True:
    game.update()
    game.draw(screen)
    pygame.display.flip()

    clock.tick(FPS)