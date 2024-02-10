import random
#import tensorflow as tf
import numpy as np
#from collections import deque

class Buffer:
    """
    Generalized experience replay buffer to hold transitions of arbitrary length.
    Note that this also offers functionality for computing the expected returns
    across episodes (allowing it to serve as a rollout buffer if needed).

    This code was extracted from the CogNGen library, where the module was
    first proposed.

    @author - Alexander G. Ororbia II
    """
    def __init__(self, buffer_capacity=100000, batch_size=64, seed=69):
        random.seed(seed)
        np.random.seed(seed)
        # num of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # num of tuples to train on.
        self.batch_size = batch_size
        # this tells us num of times record() was called.
        self.buffer_counter = 0
        # internal memory to hold the matrix blocks of data
        self.memory = []
        self.advantage = None

    def reset(self):
        """
        Resets the buffer to be empty but w/ pre-allocated memory blocks
        """
        self.advantage = None
        self.buffer_counter = 0
        for i in range(len(self.memory)):
            # create new memory block at start
            dim = self.memory[i].shape[1]
            block = np.zeros((self.buffer_capacity, dim))
            block[index] = item # insert item
            self.memory[i] = block


    def get_current_capacity(self):
        """
        Returns many sampled transitions are stored in memory currently
        """
        return self.buffer_counter

    def calc_advantage(self, rew_slot_idx, value_slot_idx, done_slot_idx, gamma, gae_lambda=0.95):
        """
        Calculates the generalized expected advantage and discounted returns across
        episodes (accounting for terminal states). Note that calling this
        return will re-compute the advantages and override any existing values
        currently w/in the self.advantage construct
        """
        rewards = self.memory[rew_slot_idx]
        values = self.memory[value_slot_idx]
        dones = self.memory[done_slot_idx].astype(np.float32)

        adv = np.zeros((rewards.shape[0],1))
        returns = np.zeros((rewards.shape[0],1))

        # Accumulate discounted returns
        discount = gamma
        lmbda = gae_lambda
        g = 0
        returns_current = values[-1]
        for i in reversed(range(rewards.shape[0])):
            gamma = discount * (1.0 - dones[i])
            if i != rewards.shape[0] - 1:
                td_error = rewards[i] + gamma * values[i-1] - values[i]
            else:
                td_error = rewards[i] - values[i]
            g = td_error * gamma * lmbda * g
            returns_current = rewards[i] + gamma * returns_current
            adv[i] = g
            returns[i] = returns_current
        adv = (adv - np.mean(adv))/(np.std(adv) + 1e-10)
        self.advantage = adv
        self.returns = returns


    # Takes observed transition tuple as input
    def record(self, obs_tuple):
        """
        Record the current transition to the memory buffer
        """
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        mem_check = len(self.memory) # check if memory is empty
        for i in range(len(obs_tuple)):
            item = obs_tuple[i]
            if len(item.shape) == 1:
                dim = item.shape[0]
            else:
                dim = item.shape[1]
            if mem_check > 0:
                self.memory[i][index] = item # insert item
            else:
                # create new memory block at start
                block = np.zeros((self.buffer_capacity, dim))
                block[index] = item # insert item
                self.memory.append( block )

        self.buffer_counter += 1

    def sample(self, batch_size=-1, sample_noreplace=False):
        """
        Samples a mini-batch of transitions from the current memory buffer
        """
        # get sampling range
        sample_size = batch_size
        if sample_size < 0:
            sample_size = self.batch_size

        # calculate the sampling indices
        if sample_noreplace is True:
            if self.buffer_counter < self.buffer_capacity:
                indices = np.random.permutation(self.buffer_counter)
            else:
                indices = np.random.permutation(self.buffer_capacity)
            if self.buffer_counter < sample_size:
                batch_indices = indices
            else:
                batch_indices = indices[0:sample_size]
        else:
            record_range = min(self.buffer_counter, self.buffer_capacity)
            if self.buffer_counter < sample_size:
                batch_indices = np.random.choice(record_range, self.buffer_counter)
            else:
                batch_indices = np.random.choice(record_range, sample_size)

        # extract the data batch blocks given the sampling indices
        batch_set = []
        for i in range(len(self.memory)):
            block = self.memory[i]
            batch_i = block[batch_indices] #tf.convert_to_tensor(block[batch_indices], dtype=tf.float32)
            #batch_i = jnp.asarray(batch_i, dtype=jnp.float32)
            #print(type(batch_i))
            #sys.exit(0)
            batch_set.append(batch_i)
        """
        if self.advantage is not None:
            batch_returns = tf.convert_to_tensor(self.advantage[batch_indices], dtype=tf.float32)
            batch_set.append(batch_returns)
            batch_returns = tf.convert_to_tensor(self.returns[batch_indices], dtype=tf.float32)
            batch_set.append(batch_returns)
        """
        batch_set = tuple(batch_set) # convert list to tuple of batch sample blocks

        return batch_set

    def sample_indices(self, batch_size=-1, sample_noreplace=False):
        sample_size = batch_size
        if sample_size < 0:
            sample_size = self.batch_size

        #print("Counter: ",self.buffer_counter)
        if sample_noreplace is True:
            if self.buffer_counter < self.buffer_capacity:
                indices = np.random.permutation(self.buffer_counter)
            else:
                indices = np.random.permutation(self.buffer_capacity)
            if self.buffer_counter < sample_size:
                batch_indices = indices
            else:
                batch_indices = indices[0:sample_size]
        else:
            record_range = min(self.buffer_counter, self.buffer_capacity)
            if self.buffer_counter < sample_size:
                batch_indices = np.random.choice(record_range, self.buffer_counter)
            else:
                batch_indices = np.random.choice(record_range, sample_size)
        return batch_indices
