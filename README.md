## Overview

This project consists of a Tic-Tac-Toe game implemented in Python with a Q-learning AI agent. The AI is trained to play Tic-Tac-Toe using a Q-learning approach and can compete against human players, a random move generator, or another AI.

## Project Structure

The project is organized into the following files:

1. `TicTacToe.py` - Contains the Tic-Tac-Toe game logic and interaction with the AI.
2. `QLearningAgent.py` - Contains the Q-learning agent implementation, including model creation and training.
3. `DataGenerator.py` - Simulates games to generate state-action data for training the Q-learning agent.

## Dependencies

The project requires the following Python packages:

- numpy
- keras
- random

You can install the required packages using pip:
```bash
pip install numpy keras
```

## Usage

### Playing the Game

To play the game, run the `TicTacToe.py` file. You can specify the type of players (Human, Random, AI) in the `__main__` section.

```python
if __name__ == "__main__":
    ttt = TicTacToe("Human", "AI")
    ttt.play_game()
```

### Training the Q-learning Agent

1. **Simulate Games**:
   Generate state-action data by simulating a number of games using `DataGenerator.py`.
   ```python
   if __name__ == "__main__":
       dg = DataGenerator()
       dg.simulate(1000000)
       dg.print_store()
       dg.save_to_npy()
   ```

2. **Train the Q-learning Model**:
   Use the generated data to train the Q-learning model by running `QLearningAgent.py`.
   ```python
   if __name__ == "__main__":
       qla = QLearningAgent(np.load("states.npy", allow_pickle=True), np.load("values.npy", allow_pickle=True))
       qla.train_q(epochs=5)
   ```

### File Descriptions

- **TicTacToe.py**:
  - Defines the `TicTacToe` class for managing game state, player moves, and checking win/draw conditions.
  - Implements methods for human, random, and AI moves.

- **QLearningAgent.py**:
  - Defines the `QLearningAgent` class for creating and training a neural network model using Keras.
  - Trains the model on state-action values generated from simulated games.

- **DataGenerator.py**:
  - Defines the `DataGenerator` class for simulating games and generating state-action data.
  - Increments counters for wins, draws, and losses based on the game outcomes.
  - Saves the generated data to `.npy` files for training the Q-learning agent.

## Running the Code

To play the game or train the AI, run the respective Python files from the command line.

### Example Commands

- Playing the game:
  ```bash
  python TicTacToe.py
  ```

- Simulating games to generate data:
  ```bash
  python DataGenerator.py
  ```

- Training the Q-learning model:
  ```bash
  python QLearningAgent.py
  ```
