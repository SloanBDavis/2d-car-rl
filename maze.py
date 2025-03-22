import pygame
import random
import sys

# Constants
TILE_SIZE = 40
GRID_WIDTH = 10
GRID_HEIGHT = 10
WINDOW_WIDTH = TILE_SIZE * GRID_WIDTH
WINDOW_HEIGHT = TILE_SIZE * GRID_HEIGHT


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY  = (200, 200, 200)
BLUE  = (50, 50, 255)
GREEN = (50, 255, 50)
RED   = (255, 50, 50)

# --- Maze Map (0=empty, 1=wall, 2=goal) ---
maze = [
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 2],
    [1, 1, 1, 0, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 1, 1, 1, 1, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 0, 0, 0, 1, 0],
]

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Maze")

clock = pygame.time.Clock()

agent_pos = [0,0]

def draw_maze():
    screen.fill(WHITE)
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            if maze[y][x] == 1:
                pygame.draw.rect(screen, BLACK, rect)
            elif maze[y][x] == 2:
                pygame.draw.rect(screen, GREEN, rect)
            pygame.draw.rect(screen, GRAY, rect, 1)

    agent_rect = pygame.Rect(agent_pos[0]*TILE_SIZE, agent_pos[1]*TILE_SIZE, TILE_SIZE, TILE_SIZE)
    pygame.draw.rect(screen, BLUE, agent_rect)

def move_agent(action):
    dx, dy = 0, 0
    if action == 0:
        dy = -1
    elif action == 1:
        dy = 1
    elif action == 2:
        dx = -1
    elif action == 3:
        dx = 1

    new_x = agent_pos[0] + dx
    new_y = agent_pos[1] + dy

    if 0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT and maze[new_y][new_x] != 1:
        agent_pos[0] = new_x
        agent_pos[1] = new_y

running = True
while running:
    clock.tick(10)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    move_agent(random.randint(0, 3))

    draw_maze()
    pygame.display.flip()

pygame.quit()
sys.exit()
            
            







