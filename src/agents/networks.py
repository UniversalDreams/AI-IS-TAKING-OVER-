import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512,
                 name='critic', chkpt_dir='tmp/ddpg'):
        super().__init__(name=name)
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, f"{self.model_name}_ddpg.weights.h5"
        )

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q   = Dense(1, activation=None)

    def call(self, state, action):
        # state: (B, S), action: (B, A)
        x = tf.concat([state, action], axis=1)
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.q(x)
        return q


class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, n_actions=2,
                 name='actor', chkpt_dir='tmp/ddpg'):
        super().__init__(name=name)
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, f"{self.model_name}_ddpg.weights.h5"
        )

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu  = Dense(self.n_actions, activation='tanh')

    def call(self, state):
        x  = self.fc1(state)
        x  = self.fc2(x)
        mu = self.mu(x)                 # in [-1, 1]
        mu_scaled = (mu + 1.0) * 0.5    # -> [0, 1] percentage action
        return mu_scaled
