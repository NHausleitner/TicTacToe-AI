import numpy as np
from keras import Sequential
from keras.src.layers import InputLayer, Dense
from TicTacToeDataGenerator import DataGenerator


class QLearningAgent:
    def __init__(self, states, values):
        self.x = states
        self.y = values
        self.model = self.create_model()

    @staticmethod
    def create_model():
        model = Sequential([
            InputLayer(shape=(28,)),
            Dense(400, activation='relu'),
            Dense(200, activation='relu'),
            Dense(100, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer="adam", loss='mse', metrics=['mse'])
        return model

    def train_q(self, epochs):
        self.model.fit(self.x, self.y, epochs=epochs, batch_size=8, verbose=1)
        self.model.save('Q.keras')


if __name__ == "__main__":
    dg = DataGenerator()
    dg.simulate(1000000)
    dg.print_store()
    dg.save_to_npy()

    qla = QLearningAgent(np.load("states.npy", allow_pickle=True), np.load("values.npy", allow_pickle=True))
    qla.train_q(epochs=5)
