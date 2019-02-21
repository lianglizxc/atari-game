"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
import pandas as pd
import time

start = time.clock()


class DDPG(object):

    def __init__(self, a_dim, s_dim, a_bound, lr_a = 1e-3, lr_q = 1e-3, gamma = 0.9, tau = 0.01, MEMORY_CAPACITY = 10000):


        self.memory_size = MEMORY_CAPACITY
        self.memory = np.zeros((self.memory_size, s_dim * 2 + a_dim + 2), dtype=np.float32)
        self.pointer = 0
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.lr_a, self.lr_q, self.gamma, self.tau = lr_a, lr_q,gamma, tau
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.s = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.s_next = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.r = tf.placeholder(tf.float32, [None, 1], 'r')
        self.done = tf.placeholder(tf.float32, [None, 1], 'done')

        self.q, self.a = self._actor_critic_net(self.s, 'eval')
        q_, a_ = self._actor_critic_net(self.s_next, 'target', trainable=False)

        # networks parameters
        self.actor_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval/actor')
        self.critic_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval/critic')
        self.actor_para_target = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target/actor')
        self.critic_para_target = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target/critic')

        q_target = self.r + gamma * q_ * (1 - self.done)
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)
        self.train_op_q = tf.train.AdamOptimizer(self.lr_q).minimize(self.td_error, var_list=self.critic_para)

        a_loss = - tf.reduce_mean(self.q)
        self.train_op_a = tf.train.AdamOptimizer(self.lr_a).minimize(a_loss, var_list=self.actor_para)

        self.soft_replace = [tf.assign(target, var) for target, var in zip(self.actor_para_target, self.actor_para)] + \
                            [tf.assign(target, var) for target, var in zip(self.critic_para_target, self.critic_para)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def sample_action(self, s, var):
        a = self.sess.run(self.a, feed_dict={self.s: s[None, :]})[0]
        a = np.clip(np.random.normal(a, var), -2, 2)
        return a

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(self.memory_size, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        s = bt[:, :self.s_dim]
        a = bt[:, self.s_dim: self.s_dim + self.a_dim]
        r = bt[:, self.s_dim + self.a_dim: self.s_dim + self.a_dim + 1]
        s_next = bt[:, -self.s_dim-1: -1]
        done = bt[:,[-1]]

        self.sess.run(self.train_op_a, {self.s: s})
        _ , lose = self.sess.run([self.train_op_q, self.td_error], {self.s: s, self.a: a,
                                                                    self.r: r, self.s_next: s_next,
                                                                    self.done: done})
        return lose

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, a, [r], s_, [done]))
        index = self.pointer % self.memory_size  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _actor_critic_net(self, s, scope, trainable=True):

        with tf.variable_scope(scope):
            a = self._actor_net(s, trainable)
            q = self._critic_net(s, a, trainable)
        return q, a

    def _actor_net(self, s, trainable):
        hidden_units = [10, 10]
        activation = [tf.nn.relu, tf.nn.relu]
        layer = s
        with tf.variable_scope('actor'):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)
            for units, activ in zip(hidden_units, activation):
                layer = tf.layers.dense(layer, units, activation=activ,
                                      kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)
            a = tf.layers.dense(layer, self.a_dim, activation=tf.nn.tanh,
                                kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)

            return self.a_bound * a

    def _critic_net(self, s, a, trainable):
        hidden_units = [40, 20, 10]
        activation = [tf.nn.relu, tf.nn.relu]
        layer = tf.concat([s, a], axis=1)
        with tf.variable_scope('critic'):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)
            for units, activ in zip(hidden_units, activation):
                layer = tf.layers.dense(layer, units, activation=activ,
                                        kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)
            q = tf.layers.dense(layer, 1, activation=None,
                                kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)
            return q


if __name__ == '__main__':
    MAX_EPISODES = 500
    MAX_EP_STEPS = 200
    BATCH_SIZE = 32

    RENDER = False
    ENV_NAME = 'Pendulum-v0'
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    ddpg = DDPG(a_dim, s_dim, a_bound)

    var = 3  # control exploration
    episode_reward = []

    for i in range(MAX_EPISODES+1):
        s = env.reset()
        ep_reward = 0

        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            if var >= 0.01:
                var *= 0.98

            a = ddpg.sample_action(s, var)
            s_, r, done, info = env.step(a)

            ddpg.store_transition(s, a, r, s_, done)

            if ddpg.pointer > ddpg.memory_size:
                ddpg.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS-1 or done:
                episode_reward.append((i,ep_reward))

                if i % 10 == 0:
                    print('Episode:', i, ' Reward: %i' % ep_reward, ' exploration: %.2f' % var)

                if ep_reward > 10:RENDER = True
                break

    end = time.clock()
    print('Running time: %s Seconds' % (end - start))
    filename = ENV_NAME + '_experiment_%s' % 1
    pd.DataFrame(episode_reward, columns = ['episode','reward']).to_csv(filename,index= False)
