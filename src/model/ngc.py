import sys, getopt, optparse
import numpy as np
from jax import jit, numpy as jnp, random, nn, lax
from functools import partial
import time
import math
from adam import Adam
from sgd import SGD

@partial(jit, static_argnums=[1])
def bkwta(x, nWTA=5): #5 10 15 #K=50):
    values, indices = lax.top_k(x, nWTA) # Note: we do not care to sort the indices
    kth = jnp.expand_dims(jnp.min(values,axis=1),axis=1) # must do comparison per sample in potential mini-batch
    topK = jnp.greater_equal(x, kth).astype(jnp.float32) # cast booleans to floats
    return topK

@partial(jit, static_argnums=[1])
def kwta(x, nWTA=5): #5 10 15 #K=50):
    """
        K-winners-take-all competitive activation function
    """
    values, indices = lax.top_k(x, nWTA) # Note: we do not care to sort the indices
    kth = jnp.expand_dims(jnp.min(values,axis=1),axis=1) # must do comparison per sample in potential mini-batch
    topK = jnp.greater_equal(x, kth).astype(jnp.float32) # cast booleans to floats
    return topK * x

@jit
def constrain(W, norm):
    _W = W * (norm/(jnp.linalg.norm(W, axis=0, keepdims=True) + 1e-5))
    return _W

@jit
def normalize(W, wnorm):
    wAbsSum = jnp.sum(jnp.abs(W), axis=0, keepdims=True)
    m = (wAbsSum == 0.).astype(dtype=jnp.float32)
    wAbsSum = wAbsSum * (1. - m) + m
    #wAbsSum[wAbsSum == 0.] = 1.
    _W = W * (wnorm/wAbsSum)
    return _W

@partial(jit, static_argnums=[5,6])
def adjust_state(h_l, e_lp1, e_l, W_lp1, beta, leak=0., zeta=0.): #0.01):
    d_l = run_syn(e_lp1, W_lp1.T, 0.)
    dh_dt = -e_l + d_l * dfx(h_l)
    if leak > 0.:
        dh_dt = dh_dt - h_l * leak
    if zeta > 0.:
        dh_dt = dh_dt - jnp.sign(h_l) * zeta
    _h_l = h_l + (dh_dt) * beta
    z_l = fx(_h_l)
    return _h_l, z_l, dh_dt

#@partial(jit, static_argnums=[3,4,5,6])
#def adjust_synap(pre, post, params, eta, leak=0.0, w_b=0., wnorm=0.):
#    clip_val = 5.
#    dW = jnp.clip(jnp.matmul((pre).T, post), -clip_val, clip_val)
#    if w_b > 0.:
#        dW = dW * (w_b - jnp.abs(params))
#    _params = params + (-params * leak) + dW * (eta/pre.shape[0])
#    if wnorm > 0.:
#        _params = constrain(_params, wnorm)
#    return _params

# @partial(jit, static_argnums=[3,4,5,6])
# def calc_syn_adjust(pre, post, params, eta, leak=0.01, w_b=0., clip_val=100.):
#     #clip_val = 1. #5.
#     dW = -jnp.matmul((pre).T, post) #* (1./post.shape[0])
#     #if w_b > 0.:
#     #    dW = dW * (w_b - jnp.abs(params))
#     #dW = -dW
#     #dW = jnp.clip(dW, -clip_val, clip_val)
#     dW = -params * leak + dW #* (1./pre.shape[0])
#     return dW # flip the sign since grad descent being used outside

@partial(jit, static_argnums=[3,4,5])
def calc_syn_adjust(pre, post, params, w_b, leak, clip_val=200.):
    dW = -jnp.matmul((pre).T, post) #* (1./post.shape[0])
    if w_b > 0.:
        dW = dW * (w_b - jnp.abs(params))
    if leak > 0.:
        dW = -params * leak + dW
    if clip_val >= 200.:
        dW = jnp.clip(dW, -clip_val, clip_val)
    return dW # flip the sign since grad descent being used outside

# @jit
# def calc_bias_adjust(post, params, scale):
#     #clip_val = 100. #5.
#     db = -jnp.sum(post, axis=0, keepdims=True) * scale #* (1./post.shape[0])
#     #db = db * (1./post.shape[0])
#     #db = jnp.clip(db, -clip_val, clip_val)
#     return db

@jit
def calc_bias_adjust(post):
    db = -jnp.sum(post, axis=0, keepdims=True) #* (1./post.shape[0])
    return db

@jit
def run_syn(inp, W, b):
    j_curr = jnp.matmul(inp, W) + b
    return j_curr

@partial(jit, static_argnums=[4])
def update_mu(mu_t, z_in, W, b, gamma=1.):
    mu_tp1 = run_syn(z_in, W, b)
    mu = mu_t * (1. - gamma) + mu_tp1 * gamma
    return mu

@jit
def fx(x):
    #return nn.leaky_relu(x)
    #return nn.tanh(x)
    return nn.relu(x)

@jit
def dfx(x): ## this is needed to avoid bad initial conditions
    #tanh_x = nn.tanh(x)
    #return -(tanh_x * tanh_x) + 1.0
    return (x >= 0.).astype(jnp.float32) #return (x >= 0.).astype(jnp.float32)
    #m = (x > 0.).astype(jnp.float32)
    #dx = m + (1. - m) * 0.01
    #return dx
    #return x + 0

class NGC:
    """
    Implementation of an x-to-y neural generative coding (NGC) circuit.

    TODO:
    try using Jax grad to get grad wrt z and theta using EFE instead
    try using leak on states and/or synapse
    try different beta for state updates
    try a weighting on error neurons (to get near bp direction); esp e3 (via 1/Sigma3)
    try weight normalizations?

    @author: Alexander Ororbia
    """
    def __init__(self, n_x, n_y, n_z, beta=0.1, eta=0.01, K=10, update_clip=200., batch_size=1, key=None):
        self.key = random.PRNGKey(time.time_ns()) if key is None else key
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z

        self.K = K
        self.beta = beta # = dt/tau_m
        self.sleak = 0.001 ## state leak
        self.eta = eta
        self.update_clip = update_clip
        self.w_scale = 0. #n_x/10. #5. # 10.
        self.w_b = 0.
        self.nw_norm = -0. # 2.
        self.wleak = 0. #0.002 #0.2 ## synaptic leak
        self.zeta = 0. ## strength of kurtotic prior
        self.mu_gamma = 0.01

        ## circuit synapses
        self.key, *subkeys = random.split(self.key, 4)
        '''
        lb = -0.025 #-0.3
        ub = 0.025 #0.3
        W1 = random.uniform(subkeys[0], (self.n_x, self.n_z[0]), minval=lb, maxval=ub, dtype=jnp.float32)
        W2 = random.uniform(subkeys[1], (self.n_z[0], self.n_z[1]), minval=lb, maxval=ub, dtype=jnp.float32)
        W3 = random.uniform(subkeys[2], (self.n_z[1], self.n_y), minval=lb, maxval=ub, dtype=jnp.float32)
        '''
        #'''
        k = 1./(self.n_z[0]) #(self.n_x)
        W1 = random.uniform(subkeys[0], (self.n_x, self.n_z[0]), minval=-math.sqrt(k), maxval=math.sqrt(k), dtype=jnp.float32)
        b1 = random.uniform(subkeys[0], (1, self.n_z[0]), minval=-math.sqrt(k), maxval=math.sqrt(k), dtype=jnp.float32)
        k = 1./(self.n_z[1]) #(self.n_z[0])
        W2 = random.uniform(subkeys[1], (self.n_z[0], self.n_z[1]), minval=-math.sqrt(k), maxval=math.sqrt(k), dtype=jnp.float32)
        b2 = random.uniform(subkeys[0], (1, self.n_z[1]), minval=-math.sqrt(k), maxval=math.sqrt(k), dtype=jnp.float32)
        k = 1./(self.n_y) #(self.n_z[1])
        W3 = random.uniform(subkeys[2], (self.n_z[1], self.n_y), minval=-math.sqrt(k), maxval=math.sqrt(k), dtype=jnp.float32)
        b3 = random.uniform(subkeys[0], (1, self.n_y), minval=-math.sqrt(k), maxval=math.sqrt(k), dtype=jnp.float32)
        #'''
        '''
        b1 = jnp.zeros((1, self.n_z[0]))
        b2 = jnp.zeros((1, self.n_z[1]))
        b3 = jnp.zeros((1, self.n_y))
        '''
        self.theta = [W1, W2, W3, b1, b2, b3]
        self.opt = Adam(learning_rate=self.eta)
        #self.opt = SGD(learning_rate=self.eta)
        self.eta_decay = 1. #0.998 # learning rate decay

    def _project(self, z_t):
        W1 = self.theta[0]
        W2 = self.theta[1]
        W3 = self.theta[2]
        b1 = self.theta[3] #* 0
        b2 = self.theta[4] #* 0
        b3 = self.theta[5] #* 0
        h1 = run_syn(z_t, W1, b1)
        z1 = fx(h1)
        h2 = run_syn(z1, W2, b2)
        z2 = fx(h2)
        h3 = run_syn(z2, W3, b3)
        z3 = h3 + 0
        return [0., h1, h2, h3], [z_t, z1, z2, z3]

    def _compute_update(self, z_t, y_t, z_init, m=None, verbose=False):
        W1 = self.theta[0]
        W2 = self.theta[1]
        W3 = self.theta[2]
        b1 = self.theta[3] #* 0
        b2 = self.theta[4] #* 0
        b3 = self.theta[5] #* 0
        H, Z = self._project(z_t) # get initial conditions
        H[1] = H[1] * z_init
        Z[1] = Z[1] * z_init
        H[2] = H[2] * z_init
        Z[2] = Z[2] * z_init
        E = [0., Z[1] * 0, Z[2] * 0, (y_t - Z[len(Z)-1])] # init error units
        if m is not None:
           E[3] = E[3] * m
        ## collect initial neural statistics to seed E-step
        _z1 = Z[1] + 0
        _z2 = Z[2] + 0
        mu1 = H[1] + 0
        mu2 = H[2] + 0
        mu3 = H[3] + 0

        ## run E-step K times to get latents
        for k in range(self.K):
            e1 = E[1]
            e2 = E[2]
            e3 = E[3] #* 0.0025 #* (1./y_t.shape[0]) # dL/d_out
            h1 = H[1]
            h2 = H[2]
            ## adjust states
            h1, z1, dt1 = adjust_state(h1, e2, e1, W2, self.beta, leak=self.sleak, zeta=self.zeta)
            H[1] = h1 + 0
            Z[1] = z1 + 0
            h2, z2, dt2 = adjust_state(h2, e3, e2, W3, self.beta, leak=self.sleak, zeta=self.zeta)
            H[2] = h2 + 0
            Z[2] = z2 + 0
            ## make predictions
            #mu1 = run_syn(z_t, W1, b1)
            mu1 = update_mu(mu1, z_t, W1, b1, gamma=self.mu_gamma)
            e1 = (h1 - mu1)
            #mu2 = run_syn(z1, W2, b2)
            mu2 = update_mu(mu2, z1, W2, b2, gamma=self.mu_gamma)
            e2 = (h2 - mu2)
            #mu3 = run_syn(z2, W3, b3)
            mu3 = update_mu(mu3, z2, W3, b3, gamma=self.mu_gamma)
            e3 = (y_t - mu3)
            if m is not None:
               e3 = e3 * m
            E[1] = e1
            E[2] = e2
            E[3] = e3

        ## run M-step once given latents -- make synaptic adjustments
        dW1 = calc_syn_adjust(z_t, E[1], W1, w_b=self.w_b, leak=self.wleak, clip_val=self.update_clip)
        dW2 = calc_syn_adjust(_z1, E[2], W2, w_b=self.w_b, leak=self.wleak, clip_val=self.update_clip)
        dW3 = calc_syn_adjust(_z2, E[3], W3, w_b=self.w_b, leak=self.wleak, clip_val=self.update_clip)
        db1 = calc_bias_adjust(E[1])
        db2 = calc_bias_adjust(E[2])
        db3 = calc_bias_adjust(E[3])
        update = [dW1, dW2, dW3, db1, db2, db3]
        return H, Z, E, update

    def _settle(self, z_t, y_t, z_init=1., m=None):
        H, Z, E, update = self._compute_update(z_t, y_t, z_init=z_init, m=m)
        dW1, dW2, dW3, db1, db2, db3 = update
        self.opt.update(self.theta, [dW1, dW2, dW3, db1, db2, db3])
        if self.nw_norm > 0.:
            for i in range(len(self.theta)):
                if self.theta[i].shape[0] > 1:
                    self.theta[i] = constrain(self.theta[i], self.nw_norm)
        '''
        if self.w_scale > 0.:
            for i in range(len(self.theta)):
                self.theta[i] = normalize(self.theta[i], self.w_scale)
        '''
        return H, Z, E

    def decay_eta(self):
        lower_bound_lr = 1e-7
        if self.eta_decay > 0.0:
            self.eta = (max(lower_bound_lr, float(self.eta * self.eta_decay)))
