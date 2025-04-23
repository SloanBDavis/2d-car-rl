# 2D Car Reinforcement Learning (2d-car-rl)

## Overview

This project trains a simple agent (a "car") to navigate a 2D maze using Reinforcement Learning (RL). The core algorithm implemented is a Deep Q-Network (DQN). The project utilizes:

*   **Python:** For the core logic and RL algorithm.
*   **Pygame:** For creating and visualizing the 2D maze environment.
*   **PyTorch:** For building and training the DQN neural network.

## Features

*   **Deep Q-Network (DQN) Agent:** Implements a DQN agent with experience replay and a target network (using Double DQN logic for target calculation).
*   **Pygame Environment:** A visual 2D grid-based maze environment.
*   **Defined Maze Structure:** A 10x10 maze layout is defined in `utils/constants.py`.
*   **State Representation:** The agent perceives its state as a 7-dimensional vector:
    *   Normalized X-coordinate
    *   Normalized Y-coordinate
    *   Normalized Manhattan distance to the goal
    *   Binary sensor readings for walls immediately North, South, West, and East.
*   **Action Space:** The agent can perform 4 discrete actions: Up, Down, Left, Right.
*   **Reward Shaping:** The reward function encourages reaching the goal, penalizes hitting walls or taking steps, and incentivizes exploring new cells and moving closer to the goal.

## Requirements

*   Python (3.8+ recommended)
*   Pygame
*   PyTorch
*   NumPy

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd 2d-car-rl
    ```
2.  **(Recommended) Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    # venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

*   **Training:**
    ```bash
    python main.py
    ```
    *Note: The training currently runs headlessly (without visualization) by default as configured in `main.py`.*

*   **Evaluation:**
    ```bash
    python eval.py
    ```
    *Note: Currently, `eval.py` appears to be a copy of `main.py`. To properly evaluate a trained model, this script needs modification to:*
    1.  *Load saved model weights.*
    2.  *Run the agent deterministically (e.g., set epsilon = 0).*
    3.  *Potentially run with visualization enabled (`headless=False` in `MazeEnv` initialization).*

## Project Structure

```
.
├── .gitignore
├── main.py             # Main training script
├── eval.py             # Evaluation script (currently same as main.py)
├── README.md           # This file
├── agent/
│   └── dqn_agent.py    # DQN model and agent logic (PyTorch)
├── env/
│   └── maze_env.py     # Maze environment definition (Pygame)
├── assets/
│   └── .gitkeep        # Placeholder for potential assets (images, sounds)
└── utils/
    └── constants.py    # Maze layout, colors, grid dimensions
```

## Implementation Details

*   **DQN Network:** A simple Multi-Layer Perceptron (MLP) with two hidden layers (64 neurons each, ReLU activation, Dropout).
*   **Hyperparameters:** Key parameters like learning rate, gamma (discount factor), and epsilon (exploration rate) are defined in `agent/dqn_agent.py`.
*   **Optimization:** Uses Adam optimizer and Huber loss.

## Future Work Ideas

*   Implement saving and loading of trained model weights.
*   Refactor `eval.py` for proper model evaluation.
*   Add command-line arguments for configuration (e.g., `--headless`, `--load_model`, hyperparameters).
*   Experiment with different maze layouts or sizes.
*   Try alternative RL algorithms (e.g., A2C, PPO).
*   Improve visualization (e.g., show Q-values, agent's path).
