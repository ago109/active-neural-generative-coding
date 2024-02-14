import random
import numpy as np

class Buffer:
    """
    Generalized experience replay buffer to hold transitions of arbitrary length.
    Note that this also offers functionality for computing the expected returns
    across episodes (allowing it to serve as a rollout buffer if needed).

    This code was extracted from the work on CogNGen
    (Ororbia & Kelly, 2022; https://escholarship.org/uc/item/35j3v2kh), where
    the module was first developed/used.

    -- Arguments --
    :param buffer_capacity: how many total memories can be held before memory
                            forgets its earliest transition memory
    :param batch_size: number of samples memory should produce when sampled
    :param seed: integer to seed memory's randomness with

    @author: Alexander G. Ororbia II
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

        :return: current replay memory usage
        """
        return self.buffer_counter

    # Takes observed transition tuple as input
    def record(self, obs_tuple):
        """
        Record the current transition to the memory buffer

        -- Arguments --
        :param obs_tuple: an environmental transition
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

        -- Arguments --
        :param batch_size: # of samples to place in a sampled memory batch
        :param sample_noreplace: if True, sample w/o replacement from memory

        :return: a batch of transition samples extracted from memory
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
