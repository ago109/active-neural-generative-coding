import sys, getopt, optparse
import numpy as np
from jax import jit, numpy as jnp, random, nn, lax
from functools import partial
import time
import math
from adam import Adam
from sgd import SGD

####################### requisite JIT-i-fied co-routines #######################
@jit
def constrain(W, norm_bound): ## enforce a norm constrain to synaptic weight vectors
    wnorm = jnp.linalg.norm(W, axis=0, keepdims=True)
    mask = (wnorm > norm_bound).astype(jnp.float32)
    factor = (norm_bound/(wnorm + 1e-5)) * mask + (1. - mask)
    _W = W * factor
    return _W

@partial(jit, static_argnums=[4,5,6])
def adjust_state(h_l, e_lp1, e_l, W_lp1, beta, leak=0., zeta=0.):
    ## compute adjustment to neural activity states
    d_l = run_syn(e_lp1, W_lp1.T, 0.)
    dh_dt = -e_l + d_l * dfx(h_l)
    if leak > 0.:
        dh_dt = dh_dt - h_l * leak
    if zeta > 0.:
        dh_dt = dh_dt - jnp.sign(h_l) * zeta
    _h_l = h_l + (dh_dt) * beta
    _z_l = fx(_h_l)
    return _h_l, _z_l, dh_dt

@partial(jit, static_argnums=[3,4,5])
def calc_syn_adjust(pre, post, params, w_b, leak, clip_val=200.):
    ## compute adjustment to synaptic weight parameters
    dW = -jnp.matmul((pre).T, post) * (1./post.shape[0])
    if w_b > 0.:
        dW = dW * (w_b - jnp.abs(params))
    if leak > 0.:
        dW = -params * leak + dW
    if clip_val < 200.: # TODO: retry for other models
        dW = jnp.clip(dW, -clip_val, clip_val)
    return dW # flip the sign since grad descent being used outside

@jit
def calc_bias_adjust(post):
    ## compute adjustment to bias parameters
    db = -jnp.sum(post, axis=0, keepdims=True) * (1./post.shape[0])
    return db

@jit
def run_syn(inp, W, b): ## run current from inp to output j_curr
    j_curr = jnp.matmul(inp, W) + b
    return j_curr

@jit
def fx(x): ## activation fx
    return nn.leaky_relu(x)

@jit
def dfx(x): ## deriv of fx (dampening function)
    m = (x >= 0.).astype(jnp.float32)
    dx = m + (1. - m) * 0.01
    return dx
################################################################################

class NGC:
    """
    Implementation of a simple, fast x-to-y neural generative coding (NGC) circuit.

    -- Arguments --
    n_x: number of input neuronal units
    n_y: number of output neuronal units
    n_z: a 2-item list containing numbers of 1st and 2nd internal layers of neurons,
         i.e., [256,128]
    eta: learning rate of circuit (for controlling rate of synaptic adjustments)
    update_clip: upper bound on absolute magnitude of synaptic adjustments (Default: 200)
    key: Jax seeding key to init this model with

    @author: Alexander Ororbia, Ankur Mali
    """
    def __init__(self, n_x, n_y, n_z, eta=0.001, update_clip=200., key=None):
        self.key = random.PRNGKey(time.time_ns()) if key is None else key
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z
        if len(self.n_z) != 2:
            print("ERROR: number of internal layers is {} but must be two!".format(len(self.n_z)))

        self.K = 3 ## number of state updates to take (fixed to 3 for this model)
        self.beta = 1. ## latent neural state update
        self.sleak = 0. ## state leak
        self.eta = eta ## learning rate of circuit (parameter update step-size)
        self.update_clip = update_clip
        self.w_b = 0. ## synaptic soft bound
        self.nw_norm = -0. ## synaptic norm constraint
        self.wleak = 0. ## synaptic leak
        self.zeta = 0. ## strength of kurtotic prior

        ## circuit synapses
        self.key, *subkeys = random.split(self.key, 4)
        k = 1./(self.n_z[0]) #(self.n_x)
        W1 = random.uniform(subkeys[0], (self.n_x, self.n_z[0]), minval=-math.sqrt(k), maxval=math.sqrt(k), dtype=jnp.float32)
        b1 = random.uniform(subkeys[0], (1, self.n_z[0]), minval=-math.sqrt(k), maxval=math.sqrt(k), dtype=jnp.float32)
        k = 1./(self.n_z[1]) #(self.n_z[0])
        W2 = random.uniform(subkeys[1], (self.n_z[0], self.n_z[1]), minval=-math.sqrt(k), maxval=math.sqrt(k), dtype=jnp.float32)
        b2 = random.uniform(subkeys[0], (1, self.n_z[1]), minval=-math.sqrt(k), maxval=math.sqrt(k), dtype=jnp.float32)
        k = 1./(self.n_y) #(self.n_z[1])
        W3 = random.uniform(subkeys[2], (self.n_z[1], self.n_y), minval=-math.sqrt(k), maxval=math.sqrt(k), dtype=jnp.float32)
        b3 = random.uniform(subkeys[0], (1, self.n_y), minval=-math.sqrt(k), maxval=math.sqrt(k), dtype=jnp.float32)

        self.theta = [W1, W2, W3, b1, b2, b3]
        self.opt = Adam(learning_rate=self.eta)
        #self.opt = SGD(learning_rate=self.eta)
        self.eta_decay = 1. #0.998 # learning rate decay

    def _project(self, z_t): ## fast ancestral projection routine for NGC circuit
        """
        Runs fast ancestral projection routine for NGC circuit.

        z_t: external state input to clamp to circuit
        """
        ## in effect, a forward pass as in any feedforward neural structure
        W1 = self.theta[0]
        W2 = self.theta[1]
        W3 = self.theta[2]
        b1 = self.theta[3]
        b2 = self.theta[4]
        b3 = self.theta[5]
        h1 = run_syn(z_t, W1, b1)
        z1 = fx(h1)
        h2 = run_syn(z1, W2, b2)
        z2 = fx(h2)
        h3 = run_syn(z2, W3, b3)
        z3 = h3 + 0
        return [0., h1, h2, h3], [z_t, z1, z2, z3]


    def _compute_update(self, z_t, y_t, z_init, m=None, verbose=False): ## internal routine
        self.K = 3
        self.beta = 1.
        W1 = self.theta[0]
        W2 = self.theta[1]
        W3 = self.theta[2]
        b1 = self.theta[3]
        b2 = self.theta[4]
        b3 = self.theta[5]
        ####
        # Several approximations are used below to speed up PC/NGC simulation
        # as well as ensure additional stability in mapping from x to y:
        # 0) we tie error feedback to generative synapses
        # 1) we use scheduled synaptic updates (t = layer_index)
        # 2) we set state update to 1
        # The above allows to avoid having to run multiple steps of state inference
        # to get to equilibrium to ensure updates are not too noisy in the face of a
        # an already very noisy RL problem and to avoid making each step in time too
        # expensive
        ####
        H, Z = self._project(z_t) # get initial conditions for states
        H[1] = H[1] * z_init
        Z[1] = Z[1] * z_init
        H[2] = H[2] * z_init
        Z[2] = Z[2] * z_init
        Namp = 1. #30. #10.
        E = [0., Z[1] * 0, Z[2] * 0, (y_t - Z[len(Z)-1])] # init error units

        _z1 = Z[1] + 0
        _z2 = Z[2] + 0
        mu1 = H[1] + 0
        mu2 = H[2] + 0
        mu3 = H[3] + 0
        dW1 = None
        dW2 = None
        dW3 = None
        db1 = None
        db2 = None
        db3 = None
        for k in range(self.K):
            e1 = E[1]
            e2 = E[2]
            e3 = E[3]
            h1 = H[1]
            h2 = H[2]
            h3 = H[3]
            z1 = Z[1]
            z2 = Z[2]
            z3 = Z[3]

            ## compute scheduled synaptic updates
            if k == 0: ## update output layer
                dW3 = calc_syn_adjust(z2, e3, W3, w_b=self.w_b, leak=self.wleak, clip_val=self.update_clip)
                db3 = calc_bias_adjust(e3)
            elif k == 1: ## update internal layer
                dW2 = calc_syn_adjust(z1, e2, W2, w_b=self.w_b, leak=self.wleak, clip_val=self.update_clip)
                db2 = calc_bias_adjust(e2)
            elif k == 2: ## update sensory layer
                dW1 = calc_syn_adjust(z_t, e1, W1, w_b=self.w_b, leak=self.wleak, clip_val=self.update_clip)
                db1 = calc_bias_adjust(e1)

            ## adjust states
            h1, z1, dt1 = adjust_state(h1, e2, e1, W2, 1., leak=self.sleak, zeta=self.zeta)
            H[1] = h1 + 0
            Z[1] = z1 + 0
            h2, z2, dt2 = adjust_state(h2, e3, e2, W3, 1., leak=self.sleak, zeta=self.zeta)
            H[2] = h2 + 0
            Z[2] = z2 + 0

            ## make predictions
            mu1 = run_syn(z_t, W1, b1)
            e1 = (h1 - mu1)
            mu2 = run_syn(z1, W2, b2)
            e2 = (h2 - mu2)
            mu3 = run_syn(z2, W3, b3)
            e3 = (y_t - mu3)
            E[1] = e1
            E[2] = e2
            E[3] = e3
            if verbose == True:
                print("+++++++++")
                print("d1.t {}  d2.t {}".format(jnp.linalg.norm(dt1),jnp.linalg.norm(dt2)))
        update = [dW1, dW2, dW3, db1, db2, db3]
        return H, Z, E, update

    def _settle(self, z_t, y_t):
        """
        Runs internal settling routine of this neural circuit.

        z_t: external state input to clamp to circuit
        y_t: target output value to clamp to circuit
        """
        H, Z, E, update = self._compute_update(z_t, y_t, z_init=1.)
        dW1, dW2, dW3, db1, db2, db3 = update
        self.opt.update(self.theta, [dW1, dW2, dW3, db1, db2, db3])
        if self.nw_norm > 0.:
            for i in range(len(self.theta)):
                if self.theta[i].shape[0] > 1:
                    self.theta[i] = constrain(self.theta[i], self.nw_norm)
        return H, Z, E

    def decay_eta(self): ## decays learning rate/step-size one step in time
        lower_bound_lr = 1e-7
        if self.eta_decay > 0.0:
            self.eta = (max(lower_bound_lr, float(self.eta * self.eta_decay)))
