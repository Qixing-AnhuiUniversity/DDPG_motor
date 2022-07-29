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
import time
from pandas import Series, DataFrame
from collections import deque
import matlab
import matlab.engine
import matplotlib.pyplot as plt

#####################  hyper parameters  ####################
eng = matlab.engine.start_matlab()
#eng.workspace['Rr'] = Rr_real
eng.workspace['id'] = 40.0
eng.workspace['iq'] = 50.0
eng.workspace['del_Rr'] = 0.0
eng.workspace['Rr'] = 0.2
Rr = 0.2
MAX_EPISODES = 1
MAX_EP_STEPS = 3000
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 1000
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'Pendulum-v0'
Reward_batch = []
Rr_batch = []
torque_batch = []
Reward_batch_iter = []
Rr_batch_iter = []
torque_batch_iter = []
###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')
            #return a
        
    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

###############################  training  ####################################

#env = gym.make(ENV_NAME)
#env = env.unwrapped
#env.seed(1)

s_dim = 6
a_dim = 1
a_bound = 0.5
#print(env.action_space.low)
#print('a_bound is '+str(a_bound))

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 0.5  # control exploration
t1 = time.time()
for i in range(MAX_EPISODES):
    #s = np.array([40.0,50.0,0.0,0.0,0.0,200])
    s = np.array([0.04,0.05,0.0,0.0,0.0,0.2])
    ep_reward = 0
    a = 0.1
    for j in range(MAX_EP_STEPS):
        #if RENDER:
            #env.render()
        #eng.workspace['Rr'] = Rr
      #state = [old_vdreal, old_vqreal]
      #next_state,reward,done,_ = env.step(action)
        #Rr = float((a+1)/2)+0.0001
        #eng.workspace['Rr'] = float(Rr)
        a = ddpg.choose_action(s)
        print('action is'+ str(a))
        a = np.clip(np.random.normal(a, var), -1, 1)    # add randomness to action selection for exploration
        #s_, r, done, info = env.step(a)
        print('real action is'+ str(a))
        Rr = float((a+1)/2)+0.0001
        Rr_batch.append(Rr)
        print('Rr is'+str(float(Rr)))
        eng.workspace['Rr'] = float(Rr)
        #eng.warning('off')
        eng.sim('qxenv.mdl')
        #spd = eng.workspace['speed']
        tor = eng.workspace['torque']
        vd = eng.workspace['vd']
        vq = eng.workspace['vq']
        power = eng.workspace['power']
        vdnp = np.array(vd)
        vdnp_reshape = vdnp.reshape(vdnp.shape[0])
        vdpd = DataFrame(vdnp_reshape)
        vdcut = vdpd.ix[4000:5000]
        vdreal = vdcut.mean()
        old_vdreal = vdreal
        vqnp = np.array(vq)
        vqnp_reshape = vqnp.reshape(vqnp.shape[0])
        vqpd = DataFrame(vqnp_reshape)
        vqcut = vqpd.ix[4000:5000]
        vqreal = vqcut.mean()
        old_vqreal = vqreal
        powernp = np.array(power)
        powernp_reshape = powernp.reshape(powernp.shape[0])
        powerpd = DataFrame(powernp_reshape)
        powercut = powerpd.ix[4000:5000]
        powerreal = powercut.mean()
        old_powerreal = powerreal
        #spdnp = np.array(spd)
        #spdnp_reshape = spdnp.reshape(spdnp.shape[0])
        #spdpd = DataFrame(spdnp_reshape)
        #spdpd.columns = ['speed']
        tornp = np.array(tor)
        tornp_reshape = tornp.reshape(tornp.shape[0])
        torpd = DataFrame(tornp_reshape)
        #torpd.columns = ['torque']
        torcut = torpd.ix[4000:5000]
        torreal = torcut.mean()
        
        # Add exploration noise
        #a = ddpg.choose_action(s)
        #print('action is'+ str(a))
        #a = np.clip(np.random.normal(a, var), -1, 1)    # add randomness to action selection for exploration
        #s_, r, done, info = env.step(a)
        #print('real action is'+ str(a))
        #Rr = float((a+1)/2)+0.0001
        #eng.workspace['Rr'] = float(Rr)
        s_ = np.array([0.04,0.05,vdreal*0.001,vqreal*0.001,powerreal*0.001,torreal*0.001]).reshape(1,6)[0]
        tor_now = float(torreal)
        torque_batch.append(tor_now)
        #print('Rr is'+str(float(Rr)))
          #reward = (tor_now - tor_max)*10
          #if tor_now > tor_max:
            #tor_max = tor_now
        r = (tor_now - 200)/200.0
        if tor_now > 200:
            r = (tor_now-200)/200.0 + 1.0
        print('Reward is'+ str(r))
        ddpg.store_transition(s, a, r / 10, s_)
        print('point is'+ str(ddpg.pointer))

        if ddpg.pointer > MEMORY_CAPACITY:
            print("begin to learn")
            var *= .995   # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r
        Reward_batch.append(ep_reward)
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            Reward_batch_iter.append(ep_reward)
            torque_batch_iter.append(tor_now)
            Rr_batch_iter.append(Rr)
            DataFrame(Reward_batch).to_csv('ddpg_reward.csv')
            DataFrame(Rr_batch).to_csv('ddpg_Rr.csv')
            DataFrame(torque_batch).to_csv('ddpg_torque.csv')
            DataFrame(Reward_batch_iter).to_csv('ddpg_reward_iter.csv')
            DataFrame(Rr_batch_iter).to_csv('ddpg_Rr_iter.csv')
            DataFrame(torque_batch_iter).to_csv('ddpg_torque_iter.csv')
            #with open('data.txt','w') as f:
                #f.write(str(Reward_batch)) 
            # if ep_reward > -300:RENDER = True
            break
print('Running time: ', time.time() - t1)