import gym
import numpy as np
import time
import random
import matplotlib.pyplot as plt

import tensorflow as tf

class nl_Agent_NN:

    def __init__(self, sess, observation_size, action_size,
                    batch_size = 256, max_history = 10000,
                    learning_rate = 0.0005, ql_gamma = 0.9):
        self.sess = sess
        self.observation_size = observation_size
        self.action_size = action_size

        self.batch_size = batch_size
        self.max_history = max_history
        self.learning_rate = learning_rate
        self.ql_gamma = ql_gamma

        self.setup_agent()
        self.sess.run(tf.global_variables_initializer())
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord = self.coord)

    def setup_agent(self):
        self.obs_in = tf.placeholder(tf.float32, shape=[None, self.observation_size])
        self.Qs_gt = tf.placeholder(tf.float32, shape=[None, self.action_size])
        self.act_num = tf.placeholder(tf.int32, shape=[None])
        self.keep_prob = tf.placeholder(tf.float32)

        Q_mask = tf.one_hot(self.act_num, self.action_size)

        # vars
        W_fc1 = tf.Variable(tf.truncated_normal([self.observation_size, 32], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[32]))

        W_fc2 = tf.Variable(tf.truncated_normal([32, 32], stddev=0.1))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[32]))

        W_fc3 = tf.Variable(tf.truncated_normal([32, self.action_size], stddev=0.1))
        b_fc3 = tf.Variable(tf.constant(0.1, shape=[self.action_size]))

        # network
        h_fc1 = tf.nn.relu(tf.matmul(self.obs_in, W_fc1) + b_fc1)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)
        self.Qs_nn = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

        self.Q_masked = tf.multiply(self.Qs_nn, Q_mask)

        self.loss = tf.reduce_mean(
                        tf.square(tf.subtract(self.Q_masked, self.Qs_gt)))

        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def get_action(self, observation):
        predicted_Qs = self.sess.run(self.Qs_nn, feed_dict={self.obs_in:[observation], 
                                                            self.keep_prob:1.})
        action = np.argmax(predicted_Qs)
        return action, predicted_Qs

    def train_agent(self, replay_memory):
        b_size = min(self.batch_size, len(replay_memory))

        batch_memory = random.sample(replay_memory, b_size)

        max_target_Q = np.zeros(b_size)
        target_Qs = np.zeros((b_size, self.action_size))
        target_actions = np.zeros(b_size)
        for i in range(b_size):
            state_obs, action, reward, next_state_obs, is_done = batch_memory[i]

            max_target_Q[i] = reward
            if not is_done:
                act, qs = self.get_action(next_state_obs)
                max_target_Q[i] += self.ql_gamma * np.max(qs)
            target_Qs[i][action] = max_target_Q[i]
            target_actions[i] = action

        features = [f[0] for f in batch_memory]

        _, loss, Qs_nn, Q_masked = self.sess.run([self.train_step, self.loss, 
                                                    self.Qs_nn, self.Q_masked], 
                    feed_dict={self.obs_in:features, self.Qs_gt:target_Qs,
                                self.act_num:target_actions,
                                self.keep_prob:0.5})

        return loss, Qs_nn, target_Qs, Q_masked
















