import sys, getopt, optparse
import numpy as np
from jax import jit, numpy as jnp, random, nn, lax
from functools import partial
import time
import math
from adam import Adam

@partial(jit, static_argnums=[1])
def one_hot(inp, nC):
    p_t = jnp.argmax(inp, axis=1)
    return nn.one_hot(p_t, num_classes=nC, dtype=jnp.float32)

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

@jit
def calc_teach_signal(h_out, d_in, W):
    d_out = run_syn(d_in, W.T) * dfx(h_out)
    return d_out

@jit
def calc_syn_adjust(pre, post, params):
    clip_val = 100. #5.
    # dW = jnp.clip(jnp.matmul((pre).T, post), -clip_val, clip_val)
    # if w_b > 0.:
    #     dW = dW * (w_b - jnp.abs(params))
    #dW = -params * leak +
    #dW = jnp.clip(jnp.matmul((pre).T, post) * (1./pre.shape[0]), -clip_val, clip_val)
    dW = jnp.matmul((pre).T, post)
    #dW = dW * (1./pre.shape[0])
    #dW = jnp.clip(dW, -clip_val, clip_val)
    return dW # flip the sign since grad descent being used outside

@jit
def calc_bias_adjust(post, params):
    clip_val = 100. #5.
    db = jnp.sum(post, axis=0, keepdims=True)
    #db = db * (1./post.shape[0])
    #db = jnp.clip(db, -clip_val, clip_val)
    return db

@jit
def run_syn(inp, W):
    j_curr = jnp.matmul(inp, W)
    return j_curr

@jit
def fx(x):
    return nn.leaky_relu(x)
    #return nn.tanh(x)
    #return nn.relu(x)

@jit
def dfx(x): ## this is needed to avoid bad initial conditions
    #tanh_x = nn.tanh(x)
    #return -(tanh_x * tanh_x) + 1.0
    #return (x > 0.).astype(jnp.float32) #return (x >= 0.).astype(jnp.float32)
    m = (x > 0.).astype(jnp.float32)
    dx = m + (1. - m) * 0.01
    return dx

class MLP:
    """
    Implementation of an x-to-y neural generative coding (NGC) circuit.

    @author: Alexander Ororbia
    """
    def __init__(self, n_x, n_y, n_z, eta=0.002, batch_size=1, key=None):
        self.key = random.PRNGKey(time.time_ns()) if key is None else key
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z

        self.eta = eta
        self.w_scale = 0. #n_x/10. #5. # 10.
        self.w_b = 0.
        self.nw_norm = 0. #2. #1. #0.
        self.wleak = 0.

        ## circuit synapses
        self.key, *subkeys = random.split(self.key, 4)
        k = 1./(self.n_x)
        W1 = random.uniform(subkeys[0], (self.n_x, self.n_z[0]), minval=-math.sqrt(k), maxval=math.sqrt(k), dtype=jnp.float32)
        b1 = random.uniform(subkeys[0], (1, self.n_z[0]), minval=-math.sqrt(k), maxval=math.sqrt(k), dtype=jnp.float32)
        k = 1./(self.n_z[0])
        W2 = random.uniform(subkeys[1], (self.n_z[0], self.n_z[1]), minval=-math.sqrt(k), maxval=math.sqrt(k), dtype=jnp.float32)
        b2 = random.uniform(subkeys[0], (1, self.n_z[1]), minval=-math.sqrt(k), maxval=math.sqrt(k), dtype=jnp.float32)
        k = 1./(self.n_z[1])
        W3 = random.uniform(subkeys[2], (self.n_z[1], self.n_y), minval=-math.sqrt(k), maxval=math.sqrt(k), dtype=jnp.float32)
        b3 = random.uniform(subkeys[0], (1, self.n_y), minval=-math.sqrt(k), maxval=math.sqrt(k), dtype=jnp.float32)
        #b1 = jnp.zeros((1, self.n_z[0]))
        #b2 = jnp.zeros((1, self.n_z[1]))
        #b3 = jnp.zeros((1, self.n_y))
        self.theta = [W1, W2, W3, b1, b2, b3]
        '''
        FIXME: is this Adam broken?
        '''
        self.opt = Adam(learning_rate=self.eta)
        self.eta_decay = 1. # 0.9998 # 0.998 # learning rate decay

    def _project(self, z_t):
        W1 = self.theta[0]
        W2 = self.theta[1]
        W3 = self.theta[2]
        b1 = self.theta[3]
        b2 = self.theta[4]
        b3 = self.theta[5]
        h1 = run_syn(z_t, W1) + b1
        z1 = fx(h1)
        h2 = run_syn(z1, W2) + b2
        z2 = fx(h2)
        h3 = run_syn(z2, W3) + b3
        z3 = h3 + 0
        return [0., h1, h2, h3], [z_t, z1, z2, z3]

    def _settle(self, z_t, y_t, z_init=1.):
        W1 = self.theta[0]
        W2 = self.theta[1]
        W3 = self.theta[2]
        b1 = self.theta[3]
        b2 = self.theta[4]
        b3 = self.theta[5]
        H, Z = self._project(z_t) # get initial conditions

        ## do backprop
        e3 = (Z[len(Z)-1] - y_t) * (1./(z_t.shape[0] * 1.)) #* 0.5
        d2 = calc_teach_signal(H[2], e3, W3)
        d1 = calc_teach_signal(H[1], d2, W2)
        ## compute adjustments w/ teaching signals
        dW1 = calc_syn_adjust(Z[0], d1, W1)
        dW2 = calc_syn_adjust(Z[1], d2, W2)
        dW3 = calc_syn_adjust(Z[2], e3, W3)
        db1 = calc_bias_adjust(d1, W1)
        db2 = calc_bias_adjust(d2, W2)
        db3 = calc_bias_adjust(e3, W3)

        # print(jnp.linalg.norm(dW1))
        # print(jnp.linalg.norm(dW2))
        self.opt.update(self.theta, [dW1, dW2, dW3, db1, db2, db3])
        # if self.nw_norm > 0.:
        #     for i in range(len(self.theta)):
        #         self.theta[i] = constrain(self.theta[i], self.nw_norm)
        # if self.w_scale > 0.:
        #     for i in range(len(self.theta)):
        #         self.theta[i] = normalize(self.theta[i], self.w_scale)
        #self.theta = [W1, W2]
        E = [0., d1, d2, e3]
        return H, Z, E

    def decay_eta(self):
        lower_bound_lr = 1e-7
        if self.eta_decay > 0.0:
            self.eta = (max(lower_bound_lr, float(self.eta * self.eta_decay)))
