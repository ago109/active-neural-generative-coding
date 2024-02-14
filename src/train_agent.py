import sys, getopt, optparse
sys.path.insert(0, 'model') ## link this script to model sub-directory

################################################################################
## collect program external arguments
options, remainder = getopt.getopt(sys.argv[1:], '', ["seed=", "n_episodes=",
                                                      "trial_id=", "results_dir="])
# Collect arguments from argv
seed = 42 ## program noise seed
trial_id = 0
n_episodes = 700 # 1000 ## number of episodes to simulate
results_dir = "./"
for opt, arg in options:
    if opt in ("--seed"):
        seed = int(arg.strip())
    elif opt in ("--trial_id"):
        trial_id = int(arg.strip())
    elif opt in ("--results_dir"):
        results_dir = arg.strip()
    elif opt in ("--n_episodes"):
        n_episodes = int(arg.strip())
print(" >> Running trial {} w/ seed = {}".format(trial_id, seed))
################################################################################

import random
from random import shuffle
random.seed(seed) ## set random's seed
import numpy as np

#import numpy as np
from jax import jit, numpy as jnp, random, nn
from functools import partial
import time
import math
from angc_model import ANGC
import gymnasium as gym

env_name = "MountainCar-v0"
#env = gym.make(env_name, render_mode=None) #render_mode="human")
env = gym.envs.make(env_name)
key = random.PRNGKey(seed) ## set Jax seed

################################################################################
## construct agent model architecture
obs_t, info = env.reset(seed=seed) ## also set gym's seed
obs_t = jnp.expand_dims(obs_t, axis=0)
o_dim = obs_t.shape[1]
a_dim = env.action_space.n
agent = ANGC(o_dim, a_dim, eta=0.002,
             actor_n_z=[256,256],
             world_n_z=[256,256],
             n_mem=1000000,
             gamma=0.99, #0.98
             eps_i=1.,
             eps_f=0.01,
             eps_decay=0.99991,
             batch_size=512,
             seed=seed
        )
targ_shift = 50 ## num transitions before target shifted
################################################################################
save_chkpt = 100 ## periodically save arrays every so many episodes

################################################################################
############## Run simulation of agent's interaction w/ its world ##############

returns = [] ## raw episodic instrumental reward signals
r_win = [] ## Rainbow-RL avg window
epist_returns = [] ## raw episodic epistemic signals
x_positions = [] ## raw episodic final positions (for mcar only)
n_trans = 0 ## num total transitions taken
for e in range(n_episodes):
    ep_R = 0. ## episodic return
    ep_Repi = 0.
    done = False
    t = 0
    pos = 0.
    if "MountainCar" in env_name:
        pos = obs_t[0,0]
    while(done == False):
        act_t = agent.collect_policy(obs_t)
        _act = nn.one_hot(act_t, agent.n_a) ## encode agent's action at time t

        obs_tp1, reward, terminated, truncated, info = env.step(act_t)
        obs_tp1 = jnp.expand_dims(obs_tp1, axis=0)
        ep_R += reward
        D_t = 0. ## the "done signal"
        if terminated or truncated: ## end of episode or goal reached - D_t = True
            D_t = 1.

        ## query generative model to produce an estimate of the epistemic signal
        H, Z = agent.world_model._project(obs_t)
        mu_tp1 = Z[len(Z)-1]
        r_epi = jnp.linalg.norm(obs_tp1 - mu_tp1, axis=1, keepdims=True)

        r_t = reward
        if "MountainCar" in env_name:
            if r_t >= 0.:
                r_t = 1. # give small "bonus" when goal state is reached in mcar
        r_epist = agent.process(obs_t, act_t, reward, r_epi, obs_tp1, D_t)
        agent.update()
        r_epist = agent.normalize_signal(r_epist)
        ep_Repi += r_epist

        if t % targ_shift == 0:
            agent.update_target()

        obs_t = obs_tp1 + 0 ## transition to next state
        if "MountainCar" in env_name:
            pos = max(pos, obs_tp1[0,0]) ## track agent position in mcar
        t += 1
        n_trans += 1
        if terminated or truncated:
            obs_t, info = env.reset() ## set to initial state/condition
            obs_t = jnp.expand_dims(obs_t, axis=0)
            done = True

    if len(r_win) > 100: ## update Rainbow-RL averaging window
        r_win.pop(0)
    r_win.append(ep_R)
    returns.append(ep_R)
    epist_returns.append(ep_Repi)
    if "MountainCar" in env_name:
        x_positions.append(pos)
    r_mu = jnp.mean(jnp.asarray(r_win))

    if "MountainCar" in env_name:
        print("{}: mu: {}  R: {}  Repi: {}  nS: {}  ( Pos: {})".format(e, r_mu, ep_R, ep_Repi, t, pos))
        #print("{}: mu: {}  R: {}  Repi: {}  nS: {}  eps: {} ( Pos: {})".format(e, r_mu, ep_R, ep_Repi, t, float(agent.eps), pos))
    else:
        print("{}: mu: {}  R: {}  Repi: {}  nS: {} ".format(e, r_mu, ep_R, ep_Repi, t))
    if e % save_chkpt == 0:
        np.save("{}rewards{}.npy".format(results_dir,trial_id), np.asarray(returns)) # save intermediate reward values
        np.save("{}epi_rewards{}.npy".format(results_dir,trial_id), np.asarray(epist_returns))
        if "MountainCar" in env_name:
            np.save("{}x_positions{}.npy".format(results_dir,trial_id), np.asarray(x_positions))
print()
################################################################################
env.close()

## save final returns to disk
np.save("{}rewards{}.npy".format(results_dir,trial_id), np.asarray(returns))
np.save("{}epi_rewards{}.npy".format(results_dir,trial_id), np.asarray(epist_returns))
if "MountainCar" in env_name:
    np.save("{}x_positions{}.npy".format(results_dir,trial_id), np.asarray(x_positions))
