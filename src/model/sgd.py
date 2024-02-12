import sys, getopt, optparse
import numpy as np
from jax import jit, numpy as jnp, random, nn, lax
from functools import partial
import time

@jit
def _update(param, grad, lr):
    _param = param - lr * grad
    return _param

class SGD():
    def __init__(self, learning_rate=0.001):
        self.eta = learning_rate
        self.time = 0.

    def update(self, theta, grads):
        self.time += 1
        for i in range(len(theta)):
            px_i = _update(theta[i], grads[i], self.eta)
            theta[i] = px_i
