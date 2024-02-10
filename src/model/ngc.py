import sys, getopt, optparse
import numpy as np
from jax import jit, numpy as jnp, random, nn, lax
from functools import partial
import time
from adam import Adam

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
    d_l = run_syn(e_lp1, W_lp1.T)
    _h_l = h_l + (-h_l * leak + d_l * dfx(h_l) - e_l - jnp.sign(h_l) * zeta) * beta
    z_l = fx(_h_l)
    return _h_l, z_l

@partial(jit, static_argnums=[3,4,5,6])
def adjust_synap(pre, post, params, eta, leak=0.0, w_b=0., wnorm=0.):
    clip_val = 5.
    dW = jnp.clip(jnp.matmul((pre).T, post), -clip_val, clip_val)
    if w_b > 0.:
        dW = dW * (w_b - jnp.abs(params))
    _params = params + (-params * leak) + dW * (eta/pre.shape[0])
    if wnorm > 0.:
        _params = constrain(_params, wnorm)
    return _params

@partial(jit, static_argnums=[3,4,5])
def calc_syn_adjust(pre, post, params, eta, leak=0.0, w_b=0.):
    clip_val = 1. #5.
    dW = jnp.clip(jnp.matmul((pre).T, post), -clip_val, clip_val)
    if w_b > 0.:
        dW = dW * (w_b - jnp.abs(params))
    dW = -params * leak + dW #* (1./pre.shape[0])
    return -dW # flip the sign since grad descent being used outside

@jit
def run_syn(inp, W):
    j_curr = jnp.matmul(inp, W)
    return j_curr

@jit
def fx(x):
    return nn.relu(x)

@jit
def dfx(x): ## this is needed to avoid bad initial conditions
    m = (x >= 0.).astype(jnp.float32)
    return m

class NGC:
    """
    Implementation of an x-to-y neural generative coding (NGC) circuit.

    @author: Alexander Ororbia
    """
    def __init__(self, n_x, n_y, n_z, beta=0.1, eta=0.01, K=10, batch_size=1, key=None):
        self.key = random.PRNGKey(time.time_ns()) if key is None else key
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z

        self.K = K
        self.beta = beta # = dt/tau_m
        self.eta = eta
        self.w_scale = 0. #n_x/10. #5. # 10.
        self.w_b = 0.
        self.nw_norm = 2. #1. #0.
        self.wleak = 0. #0.002 #0.2
        self.zeta = 0.
        ## Also have normalization and synaptic scaling to try
        ## try random conditions

        ## circuit synapses
        self.key, *subkeys = random.split(self.key, 4)
        lb = -0.05 #-0.3
        ub = 0.05 #0.3
        W1 = random.uniform(subkeys[0], (self.n_x, self.n_z[0]), minval=lb, maxval=ub, dtype=jnp.float32)
        W2 = random.uniform(subkeys[1], (self.n_z[0], self.n_z[1]), minval=lb, maxval=ub, dtype=jnp.float32)
        W3 = random.uniform(subkeys[2], (self.n_z[1], self.n_y), minval=lb, maxval=ub, dtype=jnp.float32)
        self.theta = [W1, W2, W3]
        self.opt = Adam(learning_rate=self.eta)
        self.eta_decay = 0.998 # learning rate decay

    def _project(self, z_t):
        W1 = self.theta[0]
        W2 = self.theta[1]
        W3 = self.theta[2]
        h1 = run_syn(z_t, W1)
        z1 = fx(h1)
        h2 = run_syn(z1, W2)
        z2 = fx(h2)
        h3 = run_syn(z2, W3)
        z3 = h3 + 0
        return [0., h1, h2, h3], [z_t, z1, z2, z3]

    def _settle(self, z_t, y_t, z_init=1.):
        W1 = self.theta[0]
        W2 = self.theta[1]
        W3 = self.theta[2]
        H, Z = self._project(z_t) # get initial conditions
        H[1] = H[1] * z_init
        Z[1] = Z[1] * z_init
        H[2] = H[2] * z_init
        Z[2] = Z[2] * z_init
        E = [0., Z[1] * 0, Z[2], y_t - Z[len(Z)-1]] # init error units
        ## run E-step K times to get latents
        for k in range(self.K):
            e1 = E[1]
            e2 = E[2]
            e3 = E[3]
            h1 = H[1]
            h2 = H[2]
            ## adjust states
            h1, z1 = adjust_state(h1, e2, e1, W2, self.beta, self.zeta)
            H[1] = h1
            Z[1] = z1
            h2, z2 = adjust_state(h2, e3, e2, W3, self.beta, self.zeta)
            H[2] = h2
            Z[2] = z2
            ## make predictions
            mu1 = run_syn(z_t, W1)
            e1 = h1 - mu1
            mu2 = run_syn(z1, W2)
            e2 = h2 - mu2
            mu3 = run_syn(z2, W3)
            e3 = y_t - mu3
            E[1] = e1
            E[2] = e2
            E[3] = e3
        ## run M-step once given latents
        ## make synaptic adjustments
        #self.W1 = adjust_synap(Z[0], E[1], self.W1, self.eta, w_b=self.w_b, leak=self.wleak, wnorm=self.nw_norm)
        #self.W2 = adjust_synap(Z[1], E[2], self.W2, self.eta, w_b=self.w_b, leak=self.wleak, wnorm=self.nw_norm)
        dW1 = calc_syn_adjust(Z[0], E[1], W1, self.eta, w_b=self.w_b, leak=self.wleak)
        dW2 = calc_syn_adjust(Z[1], E[2], W2, self.eta, w_b=self.w_b, leak=self.wleak)
        dW3 = calc_syn_adjust(Z[2], E[3], W3, self.eta, w_b=self.w_b, leak=self.wleak)
        # print(jnp.linalg.norm(dW1))
        # print(jnp.linalg.norm(dW2))
        self.opt.update(self.theta, [dW1, dW2, dW3])
        if self.nw_norm > 0.:
            for i in range(len(self.theta)):
                self.theta[i] = constrain(self.theta[i], self.nw_norm)
        if self.w_scale > 0.:
            for i in range(len(self.theta)):
                self.theta[i] = normalize(self.theta[i], self.w_scale)
        #self.theta = [W1, W2]
        return H, Z, E

    def decay_eta(self):
        lower_bound_lr = 1e-7
        if self.eta_decay > 0.0:
            self.eta = (max(lower_bound_lr, float(self.eta * self.eta_decay)))
