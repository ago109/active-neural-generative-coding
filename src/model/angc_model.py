import sys, getopt, optparse
import numpy as np
from jax import jit, numpy as jnp, random, nn, lax
from functools import partial
import time
import math
from ngc import NGC
from buffer import Buffer

class ANGC:
    """
    Implementation of simple active neural generative coding (ANGC) agent.

    -- Arguments --
    n_x: observation state dimensionality
    n_a: action dimensionality (number of discrete actions agent can take)
    eta: global learning rate for actor/world modules
    actor_n_z: 2-item list w/ # of neuronal units in actor circuit's internal layers
    world_n_z: 2-item list w/ # of neuronal units in world model circuit's internal layers
    n_mem: # of transitions that replay memory can hold
    gamma: actor/policy discounted return factor
    eps_i: initial epsilon value
    eps_f: final epsilon value
    eps_decay: epsilon decay coefficient
    batch_size: # of samples in a batch for sampling from agent's memory
    seed: integer seeding value for this agent

    @author: Alexander G. Ororbia II, Ankur Mali
    """
    def __init__(self, n_x, n_a, eta=0.002, actor_n_z=[256,256], world_n_z=[256,256],
                 n_mem=1000000, gamma=0.99, eps_i=0.3, eps_f=0.01, eps_decay=0.998,
                 batch_size=1, seed=42):
        self.seed = seed
        self.key = random.PRNGKey(seed)
        self.n_x = n_x
        self.n_a = n_a
        self.n_mem = n_mem ## number of transitions to store in experiential memory
        self.batch_size = batch_size
        self.actor_n_z = actor_n_z
        self.world_n_z = world_n_z

        ## meta-parameters to control effect of epsilon-greedy initial policy
        self.gamma = gamma # 0.95 #0.98
        self.eps = eps_i # starting point for eps-greedy policy
        self.eps_decay = eps_decay
        self.eps_min = eps_f # ending point for eps-greedy policy

        ## statistics/control factors for dynamically normalized/scaled epistemic signal
        self.normalize_reward = True ## clamped to True; want epistemic value normalized
        self.epi_w = 1. ## controls effect of epistemic modulation
        self.rEpi_min = 0.
        self.rEpi_max = None

        ## policy/actor model
        self.key, *subkeys = random.split(self.key, 3)
        self.actor = NGC(n_x, n_a, n_z=self.actor_n_z, update_clip=100., eta=eta, key=subkeys[0])
        ## target stability actor model
        self.target_actor = NGC(n_x, n_a, n_z=self.actor_n_z, eta=0.0, key=subkeys[0])
        self.target_actor.theta[0] = self.actor.theta[0] + 0
        self.target_actor.theta[1] = self.actor.theta[1] + 0
        self.target_actor.theta[2] = self.actor.theta[2] + 0
        self.target_actor.theta[3] = self.actor.theta[3] + 0
        self.target_actor.theta[4] = self.actor.theta[4] + 0
        self.target_actor.theta[5] = self.actor.theta[5] + 0
        ## generative/world transition model
        self.world_model = NGC(n_x + n_a, n_x, n_z=self.world_n_z, update_clip=100.,
                               eta=eta, key=subkeys[1])
        ## experience replay memory model
        self.memory = Buffer(buffer_capacity=self.n_mem, batch_size=self.batch_size, seed=seed)

    def normalize_signal(self, epi_term):
        """
        Normalizes current raw epistemic signal according to current agent bound parameters

        -- Arguments --
        :param epi_term: un-normalized, raw epistemic signal

        :return: normalized epistemic signal
        """
        _epi_term = (epi_term - self.rEpi_min)/(self.rEpi_max - self.rEpi_min)
        return _epi_term

    def policy(self, s_t):
        """
        Acquire this agent's direct policy's current output values for each action

        -- Arguments --
        :param s_t: observation/state of world at time t

        :return: policy output values for all actions
        """
        H, Z = self.actor._project(s_t)
        outVals = Z[len(Z)-1]
        return outVals

    def get_action(self, s_t):
        """
        Get current (discrete/integer) action from this agent

        -- Arguments --
        :param s_t: observation/state of world at time t

        :return: agent's chosen action given state(t)
        """
        outVals = self.policy(s_t)
        a_t = int(jnp.argmax(outVals,axis=1))
        return a_t

    def decay_epsilon(self):
        """
        Decay epsilon factor eps-greedy initial policy one step in time
        """
        if self.eps_decay > 0.0:
            self.eps = max(self.eps * self.eps_decay, self.eps_min)

    def random_policy(self, s_t):
        """
        Outputs a random action

        -- Arguments --
        :param s_t: observation/state of world at time t (Not used!)

        :return: action for time t
        """
        #self.decay_epsilon()
        self.key, *subkeys = random.split(self.key, 2)
        a = random.randint(subkeys[0], (1,1), 0, self.n_a)
        #return np.random.randint(0, self.act_dim)
        return int(a)

    def collect_policy(self, s_t):
        """
        Acquire a noisy action from the actor/policy

        :param s_t: observation/state of world at time t

        :return: action for time t
        """
        self.key, *subkeys = random.split(self.key, 2)
        r_samp = random.uniform(subkeys[0], (1, 1), minval=0., maxval=1., dtype=jnp.float32)

        self.decay_epsilon()
        if float(r_samp) < self.eps:
            return self.random_policy(s_t)
        return self.get_action(s_t)

    def update(self):
        """
        Runs a memory-driven update of the synapses of internal module circuits.
        """
        if self.memory.get_current_capacity() >= self.batch_size:
            sT, a, r, r_epi, sTP1, D = self.memory.sample(batch_size=self.batch_size)
            if self.normalize_reward == True:
                r_epi = self.normalize_signal(r_epi) * self.epi_w
            self._update(sT, a, r, r_epi, sTP1, D)

    def process(self, s_t, a_t, r_t, r_epi, s_tp1, D_t):
        """
        Processes a current transition of the form:
        [state(t), action(t), reward(t), epistemic_value(t), state(t+1), done(t) ]

        -- Arguments --
        :param s_t: world observation/state at time t
        :param a_t: action taken at time t
        :param r_t: reward received at time t
        :param r_epi: epistemic signal produced at time t
        :param s_tp1: world observation/state at time t+1
        :param D_t: done signal

        :return: batch of epistemic signals
        """
        if self.normalize_reward == True:
            self.rEpi_min = min(self.rEpi_min, float(jnp.amin(r_epi)))
            if self.rEpi_max is not None:
                self.rEpi_max = max(self.rEpi_max, float(jnp.amax(r_epi)))
            else:
                self.rEpi_max = float(jnp.amax(r_epi))

        self.memory.record((s_t, jnp.asarray([[a_t]]), jnp.asarray([[r_t]]), r_epi, s_tp1, jnp.asarray([[D_t]])))
        return float(r_epi)

    def _update(self, s_t, a_t, r_t, r_epi, s_tp1, D_t):
        """
        Conducts a singel step of memory-induced learning

        -- Arguments --
        :param s_t: world observation/state at time t
        :param a_t: action taken at time t
        :param r_t: reward received at time t
        :param r_epi: (normalized) epistemic signal produced at time t
        :param s_tp1: world observation/state at time t+1
        :param D_t: done signal
        """
        r = r_t + r_epi # full EFE is instrumental + epistemic signals
        _action = nn.one_hot(jnp.squeeze(a_t), self.n_a)

        ## run actor
        H_t, Z_t = self.actor._project(s_t)
        q_t = Z_t[len(Z_t)-1]
        ## update actor
        H_tp1, Z_tp1 = self.target_actor._project(s_tp1)
        q_tp1 = Z_tp1[len(Z_tp1)-1]
        qMax = jnp.amax(q_tp1, axis=1, keepdims=True)
        y = (r_t + (qMax * self.gamma) * (1. - D_t))
        target = _action * y + (q_t * (1. - _action))

        ## adjust synaptic parameters of agent's modules (action and world model)
        self.actor._settle(s_t, target)      ## update actor model
        c_t = jnp.concatenate((s_t, _action), axis=1)
        self.world_model._settle(c_t, s_tp1) ## update generative model

    def update_target(self):
        """
        Adjusts agent's internal target actor/policy circuit
        """
        self.target_actor.theta[0] = self.actor.theta[0] + 0
        self.target_actor.theta[1] = self.actor.theta[1] + 0
        self.target_actor.theta[2] = self.actor.theta[2] + 0
        self.target_actor.theta[3] = self.actor.theta[3] + 0
        self.target_actor.theta[4] = self.actor.theta[4] + 0
        self.target_actor.theta[5] = self.actor.theta[5] + 0
