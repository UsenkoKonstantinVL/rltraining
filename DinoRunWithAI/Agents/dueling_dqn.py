import random
import numpy as np
from keras.models import Sequential
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Flatten, Convolution2D, merge, Input, Lambda, Add
from keras.optimizers import Adam
from keras import backend as K
from collections import deque
from Agents.agent_class import Agent


class DuelingDQNAgent(Agent):
    def __init__(self, action_size, input_shape):
        self.input_shape = input_shape
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.00003
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.update_target_model()
        self.weight_name = 'dino-duelingdqn.h5'

    def _build_model(self):
        state_input = Input(shape=self.input_shape)
        x = Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu')(state_input)
        x = Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu')(x)
        x = Convolution2D(64, 3, 3, activation='relu')(x)
        x = Flatten()(x)

        # state value tower - V
        state_value = Dense(256, activation='relu')(x)
        state_value = Dense(1, init='uniform')(state_value)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], axis=-1), output_shape=(self.action_size,))(state_value)

        # action advantage tower - A
        action_advantage = Dense(256, activation='relu')(x)
        action_advantage = Dense(self.action_size)(action_advantage)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
                                  output_shape=(self.action_size,))(action_advantage)

        # merge to state-action value function Q
        state_action_value = Add()([state_value, action_advantage])

        model = Model(input=state_input, output=state_action_value)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)

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
            self.epsilon -= self.epsilon_decay

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
