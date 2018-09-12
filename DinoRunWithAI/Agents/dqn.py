import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K
from collections import deque
from Agents.agent_class import Agent


class DQNAgent(Agent):
    def __init__(self, action_size, input_shape):
        self.input_shape = input_shape
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99982
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.weight_name = 'dino-dqn.h5'

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        rand_action = [0 for _ in range(self.action_size)]
        if np.random.rand() <= self.epsilon:
            self.update_eps()
            rand_action[random.randrange(self.action_size)] = 1
            return rand_action
        self.update_eps()
        act_values = self.model.predict(np.array([state]))
        rand_action[np.argmax(act_values[0])] = 1
        return rand_action  # returns action

    def update_eps(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def clear_act(self, state):
        rand_action = [0 for _ in range(self.action_size)]
        act_values = self.model.predict(np.array([state]))
        rand_action[np.argmax(act_values[0])] = 1
        return rand_action  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(np.array([state]))
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.model.predict(np.array([next_state]))[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            states.append(state)
            targets.append(target[0])
        # self.model.fit(np.array([state]), target, epochs=1, verbose=0)
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

    def load(self):
        self.model.load_weights(self.weight_name)

    def save(self):
        self.model.save_weights(self.weight_name)
