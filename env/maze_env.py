import pygame
from utils.constants import *

class MazeEnv:
    def __init__(self, headless=False):
        if not headless:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("RL Maze")
            self.clock = pygame.time.Clock()
        self.maze = MAZE
        self.goal_pos = self.find_goal_position()
        self.visited = set()
        self.reset()
        self.headless = headless

    def find_goal_position(self):
        for y in range(len(self.maze)):
            for x in range(len(self.maze[0])):
                if self.maze[y][x] == 2:
                    return [x, y]
        return None

    def reset(self):
        self.agent_pos = [0, 0]
        self.prev_pos = self.agent_pos.copy()
        self.hit_wall = False
        self.visited.clear()
        self.visited.add(tuple(self.agent_pos))
        return self.get_state()

    def step(self, action):
        self.prev_pos = self.agent_pos.copy()
        dx, dy = 0, 0
        if action == 0: dy = -1
        if action == 1: dy = 1
        if action == 2: dx = -1
        if action == 3: dx = 1

        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy

        self.hit_wall = False
        done = False
        reward = -0.01

        if 0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT:
            if self.maze[new_y][new_x] != 1:
                self.agent_pos = [new_x, new_y]
                
                pos_tuple = tuple(self.agent_pos)
                if pos_tuple not in self.visited:
                    reward += 0.2
                    self.visited.add(pos_tuple)
                
                curr_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
                prev_dist = abs(self.prev_pos[0] - self.goal_pos[0]) + abs(self.prev_pos[1] - self.goal_pos[1])
                
                if curr_dist < prev_dist:
                    reward += 0.1
                
                if self.maze[self.agent_pos[1]][self.agent_pos[0]] == 2:
                    reward += 100.0
                    done = True
            else:
                self.hit_wall = True
                reward -= 0.5

        return self.get_state(), reward, done

    def get_state(self):
        x_norm = self.agent_pos[0] / (GRID_WIDTH - 1)
        y_norm = self.agent_pos[1] / (GRID_HEIGHT - 1)
        
        dist_to_goal = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        dist_norm = dist_to_goal / (GRID_WIDTH + GRID_HEIGHT)
        
        walls = [0, 0, 0, 0]
        for i, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
            new_x = self.agent_pos[0] + dx
            new_y = self.agent_pos[1] + dy
            if (0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT and 
                self.maze[new_y][new_x] == 1):
                walls[i] = 1
                
        return (x_norm, y_norm, dist_norm, *walls)

    def render(self):
        if self.headless:
            return
        self.screen.fill(WHITE)
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                if self.maze[y][x] == 1:
                    pygame.draw.rect(self.screen, BLACK, rect)
                elif self.maze[y][x] == 2:
                    pygame.draw.rect(self.screen, GREEN, rect)
                pygame.draw.rect(self.screen, GRAY, rect, 1)
        agent_rect = pygame.Rect(self.agent_pos[0]*TILE_SIZE, self.agent_pos[1]*TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(self.screen, BLUE, agent_rect)
        pygame.display.flip()

    def close(self):
        pygame.quit()
