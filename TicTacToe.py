import random
import numpy as np
from keras.src.saving import load_model


class TicTacToe:
    def __init__(self, player_one, player_two):
        self.board = [0 for _ in range(9)]
        self.winner = None
        self.player_one = player_one
        self.player_two = player_two
        self.current_player = None
        self.old_board = None
        self.ai = load_model('Q.keras')

    def board_to_state(self, data_type):
        if data_type == "nparray":
            state = []
            for value in self.board:
                if value == 0:
                    state.extend([1, 0, 0])  # Empty
                elif value == 1:
                    state.extend([0, 1, 0])  # X
                elif value == 2:
                    state.extend([0, 0, 1])  # O
            state.extend([0] if self.current_player == 1 else [1])
            return np.array(state)
        else:
            raise ValueError("Invalid board type")

    def print_board(self):
        for i in range(3):
            row = self.board[3 * i: 3 * i + 3]
            print('|'.join([self.get_symbol(x) for x in row]))
            if i < 2:
                print('-----')

    @staticmethod
    def get_symbol(value):
        if value == 1:
            return 'X'
        elif value == 2:
            return 'O'
        else:
            return ' '

    def is_game_over(self):
        return self.check_draw() or self.check_winner()

    def check_winner(self):
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        for combo in winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != 0:
                self.winner = self.board[combo[0]]
                return True
        return False

    def check_draw(self):
        return all(x != 0 for x in self.board)

    def make_move(self, position):
        if self.board[position] == 0:
            self.old_board = self.board.copy()
            self.board[position] = self.current_player

            self.check_winner()
            self.check_draw()

            self.current_player = 2 if self.current_player == 1 else 1
            return True
        return False

    def get_human_move(self):
        while True:
            try:
                move = int(input("Enter your move (1-9): ")) - 1
                if move in range(9) and self.board[move] == 0:
                    return move
                else:
                    print("Invalid move. Try again.")
            except ValueError:
                print("Please enter a number between 1 and 9.")

    def get_random_move(self):
        valid_moves = self.valid_moves()
        return random.choice(valid_moves)

    def get_ai_move(self, player):
        valid_moves = self.valid_moves()
        states = []
        for move in valid_moves:
            self.make_move(move)
            states.append(self.board_to_state("nparray"))
            self.pop()

        states = np.vstack(states)
        pred = self.ai.predict(states)
        if player == 1:
            move = valid_moves[np.argmax(pred)]
        else:
            move = valid_moves[np.argmin(pred)]
        return move

    def pop(self):
        self.board = self.old_board
        self.old_board = None
        self.winner = None
        self.current_player = 1 if self.current_player == 2 else 2

    def valid_moves(self):
        return [i for i, x in enumerate(self.board) if x == 0]

    def play_game(self):
        self.current_player = 1
        while True:
            self.print_board()
            move = None
            if self.current_player == 1:
                print("Player 1's turn (X)")
                if self.player_one == "Human":
                    move = self.get_human_move()
                elif self.player_one == "Random":
                    move = self.get_random_move()
                elif self.player_one == "AI":
                    move = self.get_ai_move(1)
            else:
                print("Player 2's turn (O)")
                if self.player_two == "Human":
                    move = self.get_human_move()
                elif self.player_two == "Random":
                    move = self.get_random_move()
                elif self.player_two == "AI":
                    move = self.get_ai_move(2)

            self.make_move(move)

            if self.check_winner():
                self.print_board()
                print(f"Player {self.get_symbol(self.current_player)} wins!")
                break
            elif self.check_draw():
                self.print_board()
                print("It's a draw!")
                break

    def reset(self):
        self.board = [0 for _ in range(9)]
        self.winner = None
        self.current_player = 1
        self.old_board = None


if __name__ == "__main__":
    ttt = TicTacToe("Human", "AI")
    ttt.play_game()
