import os
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time

class Actions():
    # Define 11 choices of actions to be:
    # [v_pref,      [-pi/6, -pi/12, 0, pi/12, pi/6]]
    # [0.5*v_pref,  [-pi/6, 0, pi/6]]
    # [0,           [-pi/6, 0, pi/6]]
    def __init__(self):
        self.actions = np.mgrid[1.0:1.1:0.5, -np.pi/6:np.pi/6+0.01:np.pi/12].reshape(2, -1).T
        self.actions = np.vstack([self.actions,np.mgrid[0.5:0.6:0.5, -np.pi/6:np.pi/6+0.01:np.pi/6].reshape(2, -1).T])
        self.actions = np.vstack([self.actions,np.mgrid[0.0:0.1:0.5, -np.pi/6:np.pi/6+0.01:np.pi/6].reshape(2, -1).T])
        self.num_actions = len(self.actions)

class NetworkVPCore(object):
    def __init__(self, device, model_name, num_actions):
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions

        # Build the model
        with tf.device(self.device):
            self._create_model()

    def _create_model(self):
        """Create the Keras model"""
        pass  # Implemented in child class

    def _create_graph_outputs(self, fc1_output):
        """Create output layers"""
        # Policy head
        logits_p = keras.layers.Dense(self.num_actions, name='logits_p')(fc1_output)
        softmax_p = tf.nn.softmax(logits_p)
        softmax_p = (softmax_p + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * self.num_actions)

        # Value head
        logits_v = keras.layers.Dense(1, name='logits_v')(fc1_output)
        logits_v = tf.squeeze(logits_v, axis=[1])

        return softmax_p, logits_v

    def predict_p(self, x):
        """Predict policy"""
        return self.model(x, training=False)[0].numpy()

    def predict_v(self, x):
        """Predict value"""
        return self.model(x, training=False)[1].numpy()

    def get_lstm_output(self, x):
        """Get LSTM hidden state output"""
        return self.model.get_lstm_output(x).numpy()

    def simple_load(self, filename=None):
        """Load model weights"""
        if filename is None:
            print("[network.py] Didn't define simple_load filename")
            return
        self.model.load_weights(filename)


class NetworkVP_rnn(NetworkVPCore):
    def __init__(self, device, model_name, num_actions):
        super(NetworkVP_rnn, self).__init__(device, model_name, num_actions)

    def _create_model(self):
        """Create the full Keras model"""
        # Input layer
        input_layer = keras.Input(shape=(Config.NN_INPUT_SIZE,), name='X')

        # Normalization
        if Config.NORMALIZE_INPUT:
            avg_vec = tf.constant(Config.NN_INPUT_AVG_VECTOR, dtype=tf.float32)
            std_vec = tf.constant(Config.NN_INPUT_STD_VECTOR, dtype=tf.float32)
            x_normalized = (input_layer - avg_vec) / std_vec
        else:
            x_normalized = input_layer

        # Regularizer
        if Config.USE_REGULARIZATION:
            regularizer = keras.regularizers.l2(0.0)
        else:
            regularizer = None

        # Multi-agent architecture
        if Config.MULTI_AGENT_ARCH == 'RNN':
            num_hidden = 64
            max_length = Config.MAX_NUM_OTHER_AGENTS_OBSERVED

            # Split input into components
            num_other_agents = input_layer[:, 0]
            host_agent_vec = x_normalized[:, Config.FIRST_STATE_INDEX:Config.HOST_AGENT_STATE_SIZE+Config.FIRST_STATE_INDEX]
            other_agent_vec = x_normalized[:, Config.HOST_AGENT_STATE_SIZE+Config.FIRST_STATE_INDEX:]

            # Reshape for LSTM
            other_agent_seq = tf.reshape(other_agent_vec,
                                        [-1, max_length, Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH])

            # LSTM layer
            lstm_layer = keras.layers.LSTM(num_hidden, return_state=True)
            rnn_outputs, state_h, state_c = lstm_layer(other_agent_seq)

            # Concatenate host agent and LSTM output
            layer1_input = tf.concat([host_agent_vec, state_h], 1)
            layer1 = keras.layers.Dense(256, activation='relu',
                                       kernel_regularizer=regularizer,
                                       name='layer1')(layer1_input)

        # Additional dense layer
        layer2 = keras.layers.Dense(256, activation='relu', name='layer2')(layer1)
        final_flat = keras.layers.Flatten()(layer2)

        # FCN
        fc1 = keras.layers.Dense(256, activation='relu', name='fullyconnected1')(final_flat)

        # Output heads
        softmax_p, logits_v = self._create_graph_outputs(fc1)

        # Create model
        self.model = keras.Model(inputs=input_layer, outputs=[softmax_p, logits_v])

        # Store LSTM output model for get_lstm_output method
        self.lstm_output_model = keras.Model(inputs=input_layer, outputs=state_h)

    def get_lstm_output(self, x):
        """Get LSTM hidden state output"""
        return self.lstm_output_model(x, training=False).numpy()


class Config:
    #########################################################################
    # GENERAL PARAMETERS
    NORMALIZE_INPUT     = True
    USE_DROPOUT         = False
    USE_REGULARIZATION  = True
    ROBOT_MODE          = True
    EVALUATE_MODE       = True

    SENSING_HORIZON     = 8.0

    MIN_POLICY = 1e-4

    MAX_NUM_AGENTS_IN_ENVIRONMENT = 20
    MULTI_AGENT_ARCH = 'RNN'

    DEVICE                        = '/cpu:0' # Device

    HOST_AGENT_OBSERVATION_LENGTH = 4 # dist to goal, heading to goal, pref speed, radius
    OTHER_AGENT_OBSERVATION_LENGTH = 7 # other px, other py, other vx, other vy, other radius, combined radius, distance between
    RNN_HELPER_LENGTH = 1 # num other agents
    AGENT_ID_LENGTH = 1 # id
    IS_ON_LENGTH = 1 # 0/1 binary flag

    HOST_AGENT_AVG_VECTOR = np.array([0.0, 0.0, 1.0, 0.5]) # dist to goal, heading to goal, pref speed, radius
    HOST_AGENT_STD_VECTOR = np.array([5.0, 3.14, 1.0, 1.0]) # dist to goal, heading to goal, pref speed, radius
    OTHER_AGENT_AVG_VECTOR = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0]) # other px, other py, other vx, other vy, other radius, combined radius, distance between
    OTHER_AGENT_STD_VECTOR = np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0]) # other px, other py, other vx, other vy, other radius, combined radius, distance between
    RNN_HELPER_AVG_VECTOR = np.array([0.0])
    RNN_HELPER_STD_VECTOR = np.array([1.0])
    IS_ON_AVG_VECTOR = np.array([0.0])
    IS_ON_STD_VECTOR = np.array([1.0])

    if MAX_NUM_AGENTS_IN_ENVIRONMENT > 2:
        if MULTI_AGENT_ARCH == 'RNN':
            # NN input:
            # [num other agents, dist to goal, heading to goal, pref speed, radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius]
            MAX_NUM_OTHER_AGENTS_OBSERVED = 10
            OTHER_AGENT_FULL_OBSERVATION_LENGTH = OTHER_AGENT_OBSERVATION_LENGTH
            HOST_AGENT_STATE_SIZE = HOST_AGENT_OBSERVATION_LENGTH
            FULL_STATE_LENGTH = RNN_HELPER_LENGTH + HOST_AGENT_OBSERVATION_LENGTH + MAX_NUM_OTHER_AGENTS_OBSERVED * OTHER_AGENT_FULL_OBSERVATION_LENGTH
            FIRST_STATE_INDEX = 1

            NN_INPUT_AVG_VECTOR = np.hstack([RNN_HELPER_AVG_VECTOR,HOST_AGENT_AVG_VECTOR,np.tile(OTHER_AGENT_AVG_VECTOR,MAX_NUM_OTHER_AGENTS_OBSERVED)])
            NN_INPUT_STD_VECTOR = np.hstack([RNN_HELPER_STD_VECTOR,HOST_AGENT_STD_VECTOR,np.tile(OTHER_AGENT_STD_VECTOR,MAX_NUM_OTHER_AGENTS_OBSERVED)])

    FULL_LABELED_STATE_LENGTH = FULL_STATE_LENGTH + AGENT_ID_LENGTH
    NN_INPUT_SIZE = FULL_STATE_LENGTH



if __name__ == '__main__':
    actions = Actions().actions
    num_actions = Actions().num_actions
    nn = NetworkVP_rnn(Config.DEVICE, 'network', num_actions)
    # nn.simple_load('path/to/checkpoint')  # Uncomment and provide path to load weights

    obs = np.zeros((1, Config.FULL_STATE_LENGTH), dtype=np.float32)

    num_queries = 10000
    t_start = time.time()
    for i in range(num_queries):
        obs[0,0] = 10 # num other agents
        obs[0,1] = np.random.uniform(0.5, 10.0) # dist to goal
        obs[0,2] = np.random.uniform(-np.pi, np.pi) # heading to goal
        obs[0,3] = np.random.uniform(0.2, 2.0) # pref speed
        obs[0,4] = np.random.uniform(0.2, 1.5) # radius
        predictions = nn.predict_p(obs)[0]
    t_end = time.time()
    print("avg query time:", (t_end - t_start)/num_queries)
    print("total time:", t_end - t_start)
    # action = actions[np.argmax(predictions)]
    # print("action:", action)