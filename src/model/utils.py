import sys, getopt, optparse
#import numpy as np
from jax import jit, numpy as jnp, random, nn, lax
from functools import partial
import time

import math
import matplotlib.pyplot as plt
from matplotlib import gridspec

#import tensorflow as tf
import numpy as np
import io

@jit
def flatten(matrix_set):
    vec = []
    for i in range(len(matrix_set)):
        mat = matrix_set[i]
        flat_mat = jnp.reshape(mat, (1, mat.shape[0] * mat.shape[1])) #, dtype=jnp.float32)
        vec.append(flat_mat)
        #if i > 0:
        #    vec = jnp.concatenate((vec, flat_mat), axis=1)
        #else:
        #    vec = flat_mat
    _vec = jnp.concatenate(vec, axis=1)
    return _vec # flattened super vector

@jit
def cos_sim(v1, v2):
    prod = jnp.matmul(v1, v2.T)
    prod = prod/(jnp.linalg.norm(v1) * jnp.linalg.norm(v2) + 1e-6)
    return prod

@jit
def idfx(v):
    return v + 0

@jit
def kwta(x, nWTA=5): #5 10 15 #K=50):
    """
        K-winners-take-all competitive activation function
    """
    values, indices = lax.top_k(x, nWTA) # Note: we do not care to sort the indices
    kth = jnp.expand_dims(jnp.min(values,axis=1),axis=1) # must do comparison per sample in potential mini-batch
    topK = jnp.greater_equal(x, kth).astype(jnp.float32) # cast booleans to floats
    return topK * x

@jit
def bkwta(x, nWTA=12): #5 10 15 #K=50):
    """
        K-winners-take-all competitive activation function
    """
    values, indices = lax.top_k(x, nWTA) # Note: we do not care to sort the indices
    kth = jnp.expand_dims(jnp.min(values,axis=1),axis=1) # must do comparison per sample in potential mini-batch
    topK = jnp.greater_equal(x, kth).astype(jnp.float32) # cast booleans to floats
    return topK


def visualize(thetas, sizes, prefix, suffix='.jpg'):
    """

    Args:
        thetas:

        sizes:

        prefix:

        suffix:
    """
    Ts = [t.T for t in thetas] # [tf.transpose(t) for t in thetas]
    num_filters = [T.shape[0] for T in Ts]
    n_cols = [math.ceil(math.sqrt(nf)) for nf in num_filters]
    n_rows = [math.ceil(nf / c) for nf, c in zip(num_filters, n_cols)]

    starts = [sum(n_cols[:i]) + i for i in range(len(n_cols))]
    max_size = max(sizes)

    spacers = len(sizes) - 1
    n_cols_total = sum(n_cols) + spacers
    n_rows_total = max(n_rows)

    plt.figure(figsize=(n_cols_total, n_rows_total))
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    for idx in range(len(Ts)):
        T = Ts[idx]
        size = n_cols[idx]
        start = starts[idx]
        for i in range(num_filters[idx]):
            r = math.floor(i / n_cols[idx]) #math.sqrt(num_filters[idx]))
            extra = n_cols_total - size

            point = start + 1 + i + (r * extra)
            plt.subplot(n_rows_total, n_cols_total, point)
            filter = T[i, :]
            plt.imshow(jnp.reshape(filter, (sizes[idx][0], sizes[idx][1])), cmap=plt.cm.bone, interpolation='nearest')
            plt.axis("off")

    plt.subplots_adjust(top=0.9)
    plt.savefig(prefix+suffix, bbox_inches='tight')
    plt.close()


class DataLoader(object):
    """
        A data loader object, meant to allow sampling w/o replacement of one or
        more named design matrices. Note that this object is iterable (and
        implements an __iter__() method).

        Args:
            design_matrices:  list of named data design matrices - [("name", matrix), ...]

            batch_size:  number of samples to place inside a mini-batch

            disable_shuffle:  if True, turns off sample shuffling (thus no sampling w/o replacement)

            ensure_equal_batches: if True, ensures sampled batches are equal in size (Default = True).
                Note that this means the very last batch, if it's not the same size as the rest, will
                reuse random samples from previously seen batches (yielding a batch with a mix of
                vectors sampled with and without replacement).
    """
    def __init__(self, design_matrices, batch_size, disable_shuffle=False,
                 ensure_equal_batches=True, seed=69):
        np.random.seed(seed)
        self.batch_size = batch_size
        self.ensure_equal_batches = ensure_equal_batches
        self.disable_shuffle = disable_shuffle
        self.design_matrices = design_matrices
        if len(design_matrices) < 1:
            print(" ERROR: design_matrices must contain at least one design matrix!")
            sys.exit(1)
        self.data_len = len( self.design_matrices[0][1] )
        self.ptrs = np.arange(0, self.data_len, 1)
        if self.data_len < self.batch_size:
            print("ERROR: batch size {} is > total number data samples {}".format(
                  self.batch_size, self.data_len))
            sys.exit(1)

    def __iter__(self):
        """
            Yields a mini-batch of the form:  [("name", batch),("name",batch),...]
        """
        if self.disable_shuffle == False:
            self.ptrs = np.random.permutation(self.data_len)
        idx = 0
        while idx < len(self.ptrs): # go through each sample via the sampling pointer
            e_idx = idx + self.batch_size
            if e_idx > len(self.ptrs): # prevents reaching beyond length of dataset
                e_idx = len(self.ptrs)
            # extract sampling integer pointers
            indices = self.ptrs[idx:e_idx]
            if self.ensure_equal_batches == True:
                if indices.shape[0] < self.batch_size:
                    diff = self.batch_size - indices.shape[0]
                    indices = np.concatenate((indices, self.ptrs[0:diff]))
            # create the actual pattern vector batch block matrices
            data_batch = []
            for dname, dmatrix in self.design_matrices:
                x_batch = dmatrix[indices]
                data_batch.append((dname, x_batch))
            yield data_batch
            idx = e_idx
