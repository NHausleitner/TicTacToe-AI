import random
import numpy as np
from TicTacToe import TicTacToe


class DataGenerator:
    def __init__(self):
        self.states = []
        self.counters = []
        self.temp_store = list()

    def increment_key(self, key, win_increment, draw_increment, lose_increment):
        key = key.tolist() if isinstance(key, np.ndarray) else key
        if key not in self.states:
            self.states.append(key)
            self.counters.append([0, 0, 0])  # [wins, draws, loses]

        index = self.states.index(key)
        self.counters[index][0] += win_increment
        self.counters[index][1] += draw_increment
        self.counters[index][2] += lose_increment

    def print_store(self):
        print("States:", self.states)
        print("Counters:", self.counters)

    def calculate_value(self):
        calculated_values = []
        for counters in self.counters:
            wins, draws, loses = counters
            total = wins + draws + loses
            if total > 0:
                value = (wins - loses) / total
                calculated_values.append(value)
            else:
                calculated_values.append(0)
        return np.array(calculated_values)

    def save_to_npy(self, filename_states='states.npy', filename_counters='values.npy'):
        np.save(filename_states, np.array(self.states))
        np.save(filename_counters, self.calculate_value())

    def simulate(self, games):
        ttt = TicTacToe("None", "None")

        for game in range(1, games+1):
            ttt.reset()
            while not ttt.is_game_over():
                valid_moves = ttt.valid_moves()
                move = random.choice(valid_moves)
                ttt.make_move(move)
                self.temp_store.append(ttt.board_to_state("nparray"))

                if ttt.check_winner():
                    if ttt.winner == 1:
                        for k in self.temp_store:
                            self.increment_key(k, 1, 0, 0)
                        self.temp_store.clear()
                    elif ttt.winner == 2:
                        for k in self.temp_store:
                            self.increment_key(k, 0, 0, 1)
                        self.temp_store.clear()
                elif ttt.check_draw():
                    for k in self.temp_store:
                        self.increment_key(k, 0, 1, 0)
                    self.temp_store.clear()
            print(f"Game {game}/{games} done")
