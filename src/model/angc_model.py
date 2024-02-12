import sys, getopt, optparse
import numpy as np
from jax import jit, numpy as jnp, random, nn, lax
from functools import partial
import time
import math
from ngc import NGC
from mlp import MLP
from buffer import Buffer

@partial(jit, static_argnums=[1])
def one_hot(inp, nC):
    p_t = jnp.argmax(inp, axis=1)
    return nn.one_hot(p_t, num_classes=nC, dtype=jnp.float32)

class ANGC:
    """
    Implementation of simple active neural generative coding (ANGC) agent.

    @author: Alexander Ororbia
    """
    def __init__(self, n_x, n_a, batch_size=1, seed=42): #key=None):
        self.key = random.PRNGKey(seed)
        #self.key = random.PRNGKey(time.time_ns()) if key is None else key
        self.n_x = n_x
        self.n_a = n_a
        self.batch_size = batch_size

        self.gamma = 0.99 #0.999 #0.9 #0.99 #0.95 #0.99 # 0.95 #0.98
        self.eps = 0.3 #0.9 #0.2 #0.3 #1. #0.95 #0.05 # for eps-greedy policy
        self.eps_decay = 0.998 #0.999 #0.9995 #0.975 #0.9997 #0.9925 # 0.97
        self.eps_min = 0.01 #0.05 #0.1
        self.normalize_reward = True
        self.epi_w = 1. #1. #0.95
        self.r_epi_mu = 0.
        self.r_epi_var = 0.
        self.r_epi_N = 0.
        # 50-100 w/ 1e5

        # self.EPS_START = 0.9
        # self.EPS_END = 0.05
        # self.EPS_DECAY = 1000
        # self.eps = self.EPS_START
        # self.steps_done = 0.
        # self.eps_start = 0.9
        # self.eps_end = 0.1
        # self.eps_decay = 200
        # self.steps = 0

        ## actor model
        self.key, *subkeys = random.split(self.key, 3)
        # TODO: find old ngc code on cpole (in TF2 on flashdrive in office??)
        # TODO: add in converge check for d1.t/d2.t in ngc model (cut off inf if < tol)
        Ka = 10 #40 # <-- more seemed to help
        #Ka = 30 #20 #20 #10 #40 #30 #25 #15 #8 #30 # 12 #10 #8 #10
        Kg = 6 #10 #8 #10
        # [64,128]
        # beta = 0.05
        #'''
        beta = 0.05 # 0.1 # for GPC/Z-IL
        self.actor = NGC(n_x, n_a, n_z=[50,100], update_clip=100., beta=beta, eta=0.001, K=Ka,
                         batch_size=batch_size, key=subkeys[0])
        self.target_actor = NGC(n_x, n_a, n_z=[50,100], beta=0., eta=0.0, K=1,
                                batch_size=batch_size, key=subkeys[0])
        #'''
        '''
        self.actor = MLP(n_x, n_a, n_z=[50,100], eta=0.001,
                         batch_size=batch_size, grad_clip=10., key=subkeys[0])
        self.target_actor = MLP(n_x, n_a, n_z=[50,100], eta=0.0,
                                batch_size=batch_size, key=subkeys[0])
        '''
        self.target_actor.theta[0] = self.actor.theta[0] + 0
        self.target_actor.theta[1] = self.actor.theta[1] + 0
        self.target_actor.theta[2] = self.actor.theta[2] + 0
        self.target_actor.theta[3] = self.actor.theta[3] + 0
        self.target_actor.theta[4] = self.actor.theta[4] + 0
        self.target_actor.theta[5] = self.actor.theta[5] + 0

        ## generative model synapses
        # self.world_model = NGC(n_x, n_x, n_z=[256,128], beta=0.05, eta=0.001, K=10,
        #                      batch_size=batch_size, key=subkeys[0])
        grad_clip = 10. #2.
        self.world_model = MLP(n_x, n_x, n_z=[256,128], eta=0.001,
                               batch_size=batch_size, grad_clip=10., key=subkeys[1])
        #self.world_model = NGC(n_x, n_x, n_z=[128,128], update_clip=10., beta=0.05,  eta=0.001, #0.0002,
        #                       K=Kg, batch_size=batch_size, key=subkeys[1])
        #n_mem = 1000000 #
        n_mem = 100000
        self.memory = Buffer(buffer_capacity=n_mem, batch_size=self.batch_size, seed=seed)

        self.rEpi_min = 0.
        self.rEpi_max = None

        '''
        TODO:
        also put back in RMSprop
        '''

    def normalize_signal(self, r):
        r = (r - self.rEpi_min)/(self.rEpi_max - self.rEpi_min)
        return r

    def get_action(self, z_t):
        qVals = self.policy(z_t)
        #print(qVals.shape)
        a_t = int(jnp.argmax(qVals,axis=1))
        return a_t

    def policy(self, z_t):
        H, Z = self.actor._project(z_t)
        qVals = Z[len(Z)-1]
        return qVals

    def decay_epsilon(self):
        #self.eps = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        #self.steps_done += 1
        if self.eps_decay > 0.0:
            self.eps = max(self.eps * self.eps_decay, self.eps_min) #0.01)
        #    #self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def random_policy(self, state):
        """
        Outputs a random action

        :param state: not used
        :return: action
        """
        #self.decay_epsilon()
        self.key, *subkeys = random.split(self.key, 2)
        a = random.randint(subkeys[0], (1,1), 0, self.n_a)
        #return np.random.randint(0, self.act_dim)
        return int(a)

    def collect_policy(self, s_t):
        """
        Acquire a noisy action

        :param state: the game state
        :return: action
        """
        #print("S: ",s_t.shape)
        self.key, *subkeys = random.split(self.key, 2)
        r_samp = random.uniform(subkeys[0], (1, 1), minval=0., maxval=1., dtype=jnp.float32)

        # self.steps = self.steps + 1
        #
        # eps_thresh = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps / self.eps_decay)
        # self.eps = eps_thresh

        # if r_samp > eps_thresh:
        #     return self.get_action(s_t)
        # return self.random_policy(s_t)
        self.decay_epsilon()
        if float(r_samp) < self.eps:
            #return self.get_action(s_t)
            return self.random_policy(s_t)
        return self.get_action(s_t)

    def update(self):
        if self.memory.get_current_capacity() >= self.batch_size:
            sT, a, r, r_epi, sTP1, D = self.memory.sample(batch_size=self.batch_size)
            #r_epi = 0.
            if self.normalize_reward == True:
                r_epi = self.normalize_signal(r_epi) * self.epi_w
            self._update(sT, a, r, r_epi, sTP1, D)

    def process(self, s_t, a_t, r_t, r_epi, s_tp1, D_t): ## processes a current transition
        #r_epi = 0.
        #if r_t <= 0.:
        #    r_t = -1.
        '''
        TODO: check that the below normalization code works the way I expect it to...
        '''
        if self.normalize_reward == True:
            self.rEpi_min = min(self.rEpi_min, float(jnp.amin(r_epi)))
            if self.rEpi_max is not None:
                self.rEpi_max = max(self.rEpi_max, float(jnp.amax(r_epi)))
            else:
                self.rEpi_max = float(jnp.amax(r_epi))

        self.memory.record((s_t, jnp.asarray([[a_t]]), jnp.asarray([[r_t]]), r_epi, s_tp1, jnp.asarray([[D_t]])))
        return float(r_epi)

    def _update(self, s_t, a_t, r_t, r_epi, s_tp1, D_t): ## conducts a step of memory-induced learning

        r = r_t + r_epi # full EFE is instrumental + epistemic signals
        _action = nn.one_hot(jnp.squeeze(a_t), self.n_a)

        ## run actor
        H_t, Z_t = self.actor._project(s_t)
        q_t = Z_t[len(Z_t)-1]
        ## update actor
        #H_tp1, Z_tp1 = self.target_actor._project(s_tp1)
        H_tp1, Z_tp1 = self.target_actor._project(s_tp1)
        q_tp1 = Z_tp1[len(Z_tp1)-1]
        qMax = jnp.amax(q_tp1, axis=1, keepdims=True)
        #print(qMax.shape)
        #sys.exit(0)
        y = (r_t + (qMax * self.gamma) * (1. - D_t))
        target = _action * y + (q_t * (1. - _action))

        ## adjust synaptic parameters of agent's modules
        self.actor._settle(s_t, target, m=_action)      ## update actor model
        self.world_model._settle(s_t, s_tp1) ## update generative model

    def update_target(self):
        self.target_actor.theta[0] = self.actor.theta[0] + 0
        self.target_actor.theta[1] = self.actor.theta[1] + 0
        self.target_actor.theta[2] = self.actor.theta[2] + 0
        self.target_actor.theta[3] = self.actor.theta[3] + 0
        self.target_actor.theta[4] = self.actor.theta[4] + 0
        self.target_actor.theta[5] = self.actor.theta[5] + 0
