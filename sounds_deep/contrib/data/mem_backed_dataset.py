""" Dataset that stores data in VRAM """
import math

import tensorflow as tf


class MemBackedDataset(object):
    """ Stores entire dataset in memory for better performance.

    Attributes:
        - size (int): the number of examples in this dataset
        - data (dict): see constructor
        - names (list of str): the names of the parts of the data
        - variables (dict): the variables for each part of the data (these
          contain the entire dataset)
        - placeholders (dict): the placeholders used for the init ops for the
          variables
        - slices (dict): ops for slicing the variables
    """

    def __init__(self, data, shuffle=True, num_epochs=None):
        """ Constructor

        Args:
            - data (dict of e.g. np.ndarray): the keys should be strings naming
              the parts of the data (e.g., 'images', 'labels', etc.)
            - shuffle (bool): whether or not to shuffle the dataset
            - num_epochs (int, optional): the number of epochs worth of data to
              provide. If None is passed the dataset will keep providing data
              forever.
        """
        # save data and names
        self.names = list(data.keys())
        self.data = data

        # check for consistency in number of examples and save size
        self._check_example_numbers(data)
        self.size = self.data[self.names[0]].shape[0]

        self.placeholders = {
            name: tf.placeholder(
                dtype=self.data[name].dtype, shape=self.data[name].shape)
            for name in self.names
        }
        self.variables = {
            name: tf.Variable(
                self.placeholders[name], trainable=False, collections=[])
            for name in self.names
        }

        # create slice ops
        slice_list = tf.train.slice_input_producer(
            [self.variables[name] for name in self.names],
            num_epochs=num_epochs,
            shuffle=shuffle,
            capacity=self.size)
        self.slices = {
            self.names[i]: slice_list[i]
            for i in range(len(self.names))
        }

    def load_data(self, session, data=None):
        """ Injects data into a given session.

        Args:
            - session (tf.Session): the session to inject data into
            - data (dict, optional): see constructor. If not passed, the data
              used to initialize this object will be used.
        """
        # resolve data
        if data is None:
            data = self.data

        # check data sizes
        self._check_example_numbers(data)
        assert data[self.names[0]].shape[0] == self.size

        # initialize locals
        type(tf.local_variables_initializer())
        session.run(tf.local_variables_initializer())

        # inject data
        for name in data:
            session.run(
                self.variables[name].initializer,
                feed_dict={self.placeholders[name]: data[name]})

    def _check_example_numbers(self, data):
        """ Checks if all elements of data have the same first dimension.

        Args:
            - data (dict): see constructor
        """
        for name in data:
            assert data[self.names[0]].shape[0] == data[name].shape[0]

    def get_batch_tensors(self, batch_size, num_threads=4):
        """ Returns dict of batch Tensors ready to be plugged into a model

        Args:
            - batch_size (int): how many examples to include in a batch
            - num_threads (int): how many threads to use for batching
        """
        return tf.train.batch(
            self.slices,
            batch_size,
            capacity=self.size,
            num_threads=num_threads,
            allow_smaller_final_batch=True)

    def calc_batches_per_epoch(self, batch_size):
        return int(math.ceil(float(self.size) / float(batch_size)))