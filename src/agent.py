import random
from collections import deque

import flappy_bird_gym
import numpy as np
from tensorflow.keras.models import load_model

from .model import NeuralNet
from .settings import ARTIFACT_DIR, MODEL_NAME, console
from rich.table import Table
from rich.progress import track


class DQNAgent:
    def __init__(self):
        self.env = flappy_bird_gym.make("FlappyBird-v0")
        self.episodes = 1000
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_number = 64

        self.train_start = 1000
        self.jump_probability = 0.9
        self.model = NeuralNet.build_model(
            input_shape=(self.state_space,), output_shape=self.action_space
        )

    def take_action(self, state):
        if np.random.random() > self.epsilon:
            return np.argmax(self.model.predict(state))
        else:
            return 1 if np.random.random() < self.jump_probability else 0

    def learn(self):
        if len(self.memory) < self.train_start:
            return

        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_number))
        state = np.zeros((self.batch_number, self.state_space))
        next_state = np.zeros((self.batch_number, self.state_space))
        action, reward, done = [], [], []

        for i in range(self.batch_number):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_number):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (
                    np.amax(target_next[i])
                )

        self.model.fit(state, target, batch_size=self.batch_number, verbose=0)

    def train(self):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Episode", style="dim")
        table.add_column("Score")
        table.add_column("Epsilon")

        for i in track(range(self.episodes), description="Training"):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_space])
            done = False
            score = 0
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            while not done:
                # self.env.render()
                action = self.take_action(state)
                next_state, reward, done, info = self.env.step(action)

                next_state = np.reshape(next_state, [1, self.state_space])
                score += 1

                if done:
                    reward -= 100

                self.memory.append((state, action, reward, next_state, done))
                state = next_state

                if done:
                    if i % 100 == 0:
                        table.add_row(str(i), str(score), str(self.epsilon))
                        console.print("\n")
                        console.print(table)
                    if score >= 1000:
                        console.print(
                            f"Saving the best model as {MODEL_NAME}", style="bold green"
                        )
                        self.model.save_model(ARTIFACT_DIR / f"{MODEL_NAME}.h5")
                        return

                self.learn()

    def perform(self):
        console.print("Loading best model!", style="bold blue")
        self.model = load_model(ARTIFACT_DIR / f"{MODEL_NAME}.h5")
        while 1:
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_space])
            done = False
            score = 0

            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, info = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_space])
                score += 1

                console.print("Current Score: {}".format(score), end="\r")

                if done:
                    console.print("Bird is dead!", style="bold red")
                    break


if __name__ == "__main__":
    agent = DQNAgent()
    agent.train()
    # agent.perform()
