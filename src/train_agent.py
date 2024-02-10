seed = 69 ## program noise seed
import random
from random import shuffle
random.seed(seed)
import numpy as np

import sys, getopt, optparse
#import numpy as np
from jax import jit, numpy as jnp, random, nn
from functools import partial
import time
sys.path.insert(0, 'model')

from angc_model import ANGC

import gymnasium as gym
#env = gym.make("CartPole-v1", render_mode=None) #render_mode="human")
env = gym.envs.make("CartPole-v1")

seed = 69 #42 # 69
key = random.PRNGKey(seed)
n_e = 200 #150 #500 #2000 #600 #3000 #150 # number of episodes


obs_t, info = env.reset(seed=seed) #42)
obs_t = jnp.expand_dims(obs_t, axis=0)
o_dim = obs_t.shape[1]
a_dim = env.action_space.n

bsize = 128 #20 #32 #128 #20 #512 #20 #100 #20 # 128
agent = ANGC(o_dim, a_dim, batch_size=bsize, seed=seed) #key=key)
targ_shift = 40 #5 #100 #10 #10 #100 ## num transitions before target shifted

r_win = []
n_trans = 0
for e in range(n_e):
    ep_R = 0. ## episodic return
    ep_Repi = 0.
    done = False
    t = 0
    while(done == False):
        #print("\r Step {}".format(t),end="")
        #act_t = agent.get_action(obs_t)
        act_t = agent.collect_policy(obs_t)
        #action = env.action_space.sample()  # this is where you would insert your policy

        obs_tp1, reward, terminated, truncated, info = env.step(act_t)
        obs_tp1 = jnp.expand_dims(obs_tp1, axis=0)
        ep_R += reward
        #done = terminated * 1.
        D_t = 0.
        if terminated or truncated:
            D_t = 1.

        ## query generative model
        H, Z = agent.world_model._project(obs_t)
        mu_tp1 = Z[len(Z)-1]
        r_epi = jnp.linalg.norm(obs_tp1 - mu_tp1, axis=1, keepdims=True)

        r_epist = agent.process(obs_t, act_t, reward, r_epi, obs_tp1, D_t)
        agent.update()
        ep_Repi += r_epist
        #agent.decay_epsilon()

        if t % targ_shift == 0:
            agent.update_target()

        obs_t = obs_tp1 + 0 ## transition to next state
        t += 1
        n_trans += 1
        if terminated or truncated:
            obs_t, info = env.reset() ## set to initial state/condition
            obs_t = jnp.expand_dims(obs_t, axis=0)
            done = True
    # for _ in range(10):
    #     agent.update()
    '''
    print("-----------------------")
    print("   {}  {}  {}".format(jnp.linalg.norm(agent.actor.theta[0]),
                              jnp.linalg.norm(agent.actor.theta[1]),
                              jnp.linalg.norm(agent.actor.theta[2])))
    print("-----------------------")
    '''
    #agent.decay_epsilon()
    # if e % targ_shift == 0:
    #     agent.update_target()
    if len(r_win) > 100:
        r_win.pop(0)
    r_win.append(ep_R)
    r_mu = jnp.mean(jnp.asarray(r_win))
    print("{}: mu: {}  R: {}  Repi: {}  nS: {}  eps: {}".format(e, r_mu, ep_R, ep_Repi, t, float(agent.eps)))
print()
env.close()
