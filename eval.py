# main.py

from env.maze_env import MazeEnv
from agent.dqn_agent import DQNAgent
import pygame
import numpy as np

env = MazeEnv(headless=False)
state = env.reset()

# New state space has 7 dimensions: x, y, distance to goal, and 4 wall sensors
agent = DQNAgent(state_dim=7, action_dim=4)

episodes = 500
best_reward = float('-inf')
solved_steps = 100  # Consider maze solved if done in less steps
success_count = 0  # Track number of successful episodes

print("Collecting initial experiences...")
while len(agent.memory) < agent.min_memory_size:
    state = env.reset()
    done = False
    while not done and len(agent.memory) < agent.min_memory_size:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
print(f"Initial collection done. Starting training with {len(agent.memory)} experiences")

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    steps_this_episode = 0
    agent.episode_losses = []

    for step in range(solved_steps):
        if not env.headless:
            env.clock.tick(20)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    exit()

        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        steps_this_episode += 1

        agent.remember(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        env.render()

        if done:
            if steps_this_episode < solved_steps:  # Actually reached the goal
                success_count += 1
            break

    avg_loss = np.mean(agent.episode_losses) if agent.episode_losses else 0
    if total_reward > best_reward:
        best_reward = total_reward
        
    print(f"Episode {episode+1}: Steps = {steps_this_episode} | "
          f"Reward = {total_reward:.2f} | Best = {best_reward:.2f} | "
          f"Epsilon = {agent.epsilon:.3f} | Success Rate = {success_count/(episode+1):.3f} | "
          f"Avg Loss = {avg_loss:.4f}")

env.close()
