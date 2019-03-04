import tensorflow as tf
import numpy as np
import gym
import seaborn as sns
import matplotlib.pyplot as plt

import time

start = time.clock()

class DQN(object):

    def __init__(self, n_a, s_dim, r_q = 1e-3, gamma = 0.9, tau = 0.01,
                 batch_size =32, memory_capacity = 10000, double_q = False):

        self.a_dim = 1
        self.memory = np.zeros((memory_capacity, s_dim * 2 + self.a_dim + 2), dtype=np.float32)
        self.memory_size = memory_capacity
        self.pointer = 0
        self.batch_size = batch_size

        self.lr_q, self.gamma, self.tau = r_q, gamma, tau
        self.n_a, self.s_dim = n_a, s_dim
        self.s = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.a = tf.placeholder(tf.int32, [None], 'a')
        self.s_next = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.r = tf.placeholder(tf.float32, [None, 1], 'r')
        self.done = tf.placeholder(tf.float32, [None, 1], 'done')
        self.double_q = double_q

        with tf.variable_scope('eval'):
            self.q = self.build_DQN(self.s)

        with tf.variable_scope('target'):
            q_ = self.build_DQN(self.s_next, False)

        self.sample_a = tf.argmax(self.q, axis=1)

        # if self.double_q:
        #     with tf.variable_scope('eval', reuse=True):
        #         q_reuse = self.build_DQN(self.s_next)
        #     max_a = tf.argmax(q_reuse, axis=1)
        #     one_hot_mask = tf.one_hot(max_a, depth=self.n_a)
        #     q_target = self.r + gamma * tf.reduce_sum(q_ * one_hot_mask, axis=1, keepdims=True) * (1.0 - self.done)
        # else:
        q_target = self.r + gamma * tf.reduce_max(q_, axis=1) * (1.0 - self.done)

        q_eval = tf.reduce_sum(self.q * tf.one_hot(self.a, depth = self.n_a), axis=1, keepdims=True)
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q_eval)

        # networks parameters
        self.eval_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval')
        self.target_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')

        self.train_op_q = tf.train.AdamOptimizer(self.lr_q).minimize(self.td_error, var_list=self.eval_para)

        self.soft_replace = [tf.assign(target, var) for target, var in zip(self.target_para, self.eval_para)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def sample_action(self, s):
        a = self.sess.run(self.sample_a, feed_dict={self.s: s[None, :]})[0]
        return a


    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(self.memory_size, size=self.batch_size)
        bt = self.memory[indices, :]
        s = bt[:, :self.s_dim]
        a = bt[:, self.s_dim: self.s_dim + self.a_dim]
        r = bt[:, -self.s_dim - 1: -self.s_dim]
        s_next = bt[:, -self.s_dim-1:-1]
        done = bt[:, [-1]]

        _ , lose = self.sess.run([self.train_op_q, self.td_error], {self.s: s, self.a: a.ravel(),
                                                                    self.r: r, self.s_next: s_next,
                                                                    self.done: done})
        return lose

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, [a], [r], s_, [done]))
        index = self.pointer % self.memory_size  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def build_DQN(self, s,  trainable = True):
        hidden_units = [20, 10]
        activation = [tf.nn.relu, tf.nn.relu]
        layer = s
        for units, activ in zip(hidden_units, activation):
            layer = tf.layers.dense(layer, units, activation=activ, trainable=trainable)
        q = tf.layers.dense(layer, self.n_a, activation=None, trainable=trainable)
        return q

if __name__ == '__main__':
    MAX_EPISODES = 500
    MAX_EP_STEPS = 200

    RENDER = False
    ENV_NAME = 'Boxing-ram-v0'

    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    dqn = DQN(a_dim, s_dim)

    episode_reward = []
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0

        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            a = dqn.sample_action(s)
            s_, r, done, info = env.step(a)
            dqn.store_transition(s, a, r, s_, done)

            if dqn.pointer > dqn.memory_size:
                loss = dqn.learn()
                print('loss at epoch %s is %s' %(i, loss))

            s = s_
            ep_reward += r
            if done: break

        episode_reward.append(ep_reward)
        print('Episode:', i, ' Reward: %i' % int(ep_reward))
        if ep_reward > 10:
            RENDER = True

    end = time.clock()
    print('Running time: %s Seconds' % (end - start))

    sns.set(style="darkgrid")
    plt.figure(1)
    plt.plot(episode_reward, label='DDPG')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(loc='best')

    plt.show()