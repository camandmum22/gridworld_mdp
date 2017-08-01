from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import deque
from random import sample
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import *
import numpy as np
import gym

BUFFER_SIZE = 1000
MINI_BATCH = 50
DISCOUNT = 0.99
ALPHA_LEARNING = 0.0001
STEPS_LIMIT = 500
EPISODES_LIMIT = 1000
EPSILON = 0.05

class Neural_Network:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.net = self.create_net(states, actions)
        self.target_net = self.create_net(states, actions)

    def create_net(self, states, actions):
        size_hidden = 10
        act_func = 'relu'
        net = Sequential()
        net.add(Dense(size_hidden, input_dim=states[0], activation=act_func))
        net.add(Dense(size_hidden, activation=act_func))
        net.add(Dense(actions, activation='linear'))
        optimizer = Adam(lr=ALPHA_LEARNING)
        net.compile(loss='mse', optimizer=optimizer)
        return net

    def train_net(self, data, labels):
        data = np.vstack(data)
        labels = np.vstack(labels)
        loss = self.net.fit(data, labels, batch_size=MINI_BATCH, epochs=1, verbose=0, shuffle=True)
        return loss

    def predict_q(self, state, target=False):
        state = np.asarray(state).reshape((1, self.states[0]))
        if target:
            q_values = self.target_net.predict(state)
        else:
            q_values = self.net.predict(state)
        return q_values[0]

    def optimal_action(self, state, target=False):
        state = np.asarray(state).reshape((1, self.states[0]))
        q_vals = self.predict_q(state=state, target=target)
        best_action = np.argmax(q_vals)
        return best_action


class DQN:
    def __init__(self, game):
        self.game = game
        self.buffer = deque([], maxlen=BUFFER_SIZE)
        self.nets = Neural_Network(self.game.observation_space.shape, self.game.action_space.n)
        self.target = False
        self.is_learning = False
        self.episode_count = 0
        self.step_count = 0

    def new_observation(self, observation, target_rate, use_batch):
        self.buffer.append(observation)
        if self.is_learning:
            self.init_replay(target_rate=target_rate, use_batch=use_batch)
        if self.step_count >= MINI_BATCH:
            self.is_learning = True

    def init_replay(self, target_rate, use_batch):
        if use_batch:
            data, labels = self.get_batch()
            self.nets.train_net(data, labels)
        if self.episode_count % target_rate == 0:
            w = self.nets.net.get_weights()
            self.nets.target_net.set_weights(w)
            self.target = True

    def e_greedy(self, state):
        random_num = np.random.rand()
        if random_num <= EPSILON:
            choice = np.random.randint(0, self.game.action_space.n)
        else:
            choice = self.nets.optimal_action(state=state, target=False)
        return choice

    def get_batch(self):
        data = []
        labels = []
        mini_batch = sample(self.buffer, min(MINI_BATCH, len(self.buffer)))

        for s1, action, s2, reward, done in mini_batch:
            data.append(s1)
            old_q = self.nets.predict_q(state=s1, target=False)
            new_q = self.nets.predict_q(state=s2, target=self.target)
            optimal_action = self.nets.optimal_action(state=s2, target=False)
            if done:
                old_q[action] = reward
            else:
                old_q[action] = reward + DISCOUNT * new_q[optimal_action]
            labels.append(old_q)
        return data, labels


def run():
    #np.random.seed(0)
    game = gym.make('CartPole-v0')
    set_rewards = deque([], maxlen=100)
    dqn = DQN(game)

    # Change to adapt the update rate for the target network
    target_rate = EPISODES_LIMIT+1
    use_batch = False

    for n in range(EPISODES_LIMIT):
        reward = 0
        s = game.reset()
        done = False
        steps = 0
        dqn.episode_count += 1
        while not done and steps < STEPS_LIMIT:
            dqn.step_count += 1
            steps += 1
            # game.render()
            a = dqn.e_greedy(s)
            s2, rew, done, _ = game.step(a)
            dqn.new_observation((s, a, s2, rew, done), target_rate=target_rate, use_batch=use_batch)
            reward += rew
            s = s2

        set_rewards.append(reward)
        rate_rewards = sum(set_rewards) / len(set_rewards)

        print("Episode {}\tScore {}\tAverage {}\tsteps {}\tdone {}".format(
            (n + 1), reward, rate_rewards, dqn.step_count, done))


if __name__ == "__main__":
    run()
