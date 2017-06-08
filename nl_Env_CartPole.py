import gym
import numpy as np
import time
from datetime import datetime
import random
import matplotlib.pyplot as plt

import tensorflow as tf

from nl_Agents import nl_Agent_NN

class nl_Env_CartPole:
    MAX_ENV_STEPS = 1000
    MAX_REPLAY_MEMORY = 100000
    RANDOM_ACTION_PROB = 0.3
    WIN_STEPS = 200

    def __init__(self, sess):
        self.env = gym.make('CartPole-v0')
        self.observation_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.agent = nl_Agent_NN(sess, self.observation_size, self.action_size)

        self.memory_shape = 6 # [state_obs, action, reward, next_state_obs, is_done]
        self.replay_memory = []
        self.replay_memory_index = 0

    def agent_play(self, show_game = True):
        total_steps = 0
        observation = self.env.reset()
        for step in range(self.MAX_ENV_STEPS):
            if show_game:
                self.env.render()
            action, _ = self.agent.get_action(observation)
            new_observation, reward, done, info = self.env.step(action)
            if done:
                total_steps = step+1
                if show_game:
                    print("Play steps: {}".format(total_steps))
                break

            observation = new_observation
            if show_game:
                time.sleep(1.0/30) # 30fps
        return total_steps

    def get_train_action(self, observation):
        if random.random() < self.RANDOM_ACTION_PROB:
            action = self.env.action_space.sample() # random action
        else:
            action, predicted_Qs = self.agent.get_action(observation)

        return action

    def save_memory(self, state_obs, action, reward, next_state_obs, is_done):
        memory = [state_obs, action, reward, next_state_obs, is_done]
        if len(self.replay_memory) < self.MAX_REPLAY_MEMORY:
            self.replay_memory.append(memory)
        else:
            self.replay_memory[self.replay_memory_index] = memory

        self.replay_memory_index += 1
        if self.replay_memory_index >= self.MAX_REPLAY_MEMORY:
            self.replay_memory_index = 0

    def train_agent(self, num_episodes, watch_interval = 10, val_plays = 10):
        start_time = datetime.now()

        train_agent_steps = np.zeros(num_episodes)
        val_agent_steps = np.zeros(num_episodes)

        for ep in range(num_episodes):
            if ep % watch_interval == 0:
                avg_val_steps = 0.0
                for i in range(val_plays):
                    avg_val_steps += self.agent_play(show_game=(i == val_plays-1))
                avg_val_steps /= val_plays
                val_agent_steps[ep] = avg_val_steps

                print ep, datetime.now() - start_time, "Average", avg_val_steps

                plt.close()
                plt.plot(range(0, ep*watch_interval+1, watch_interval), val_agent_steps[:ep+1],
                    range(0, ep*watch_interval, watch_interval), train_agent_steps[:ep])
                plt.show(block=False)
            else:
                val_agent_steps[ep] = val_agent_steps[ep-1]

            observation = self.env.reset()
            for step in range(self.MAX_ENV_STEPS):
                action = self.get_train_action(observation)
                next_observation, reward, is_done, info = self.env.step(action)
                if is_done and step < self.WIN_STEPS-1:
                    reward = -100

                self.save_memory(observation, action, reward, next_observation, is_done)

                if is_done:
                    train_agent_steps[ep] = step+1
                    break

                observation = next_observation

                loss, Qs_nn, target_Qs, Q_masked = self.agent.train_agent(self.replay_memory)

        print "Done", datetime.now() - start_time
        plt.close()
        plt.plot(range(len(val_agent_steps)), val_agent_steps,
                    range(len(train_agent_steps)), train_agent_steps)
        plt.show()






if __name__ == "__main__":
    with tf.Session() as sess:
        nl_env = nl_Env_CartPole(sess)
        # nl_env.agent_play()
        nl_env.train_agent(200)













