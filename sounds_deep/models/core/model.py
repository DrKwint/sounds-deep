from abc import ABCMeta, abstractmethod
import os
import six

import tensorflow as tf
import numpy as np

from sounds_deep.util.lazy_property import lazy_property


@six.add_metaclass(ABCMeta)
class Model(object):
    """ Contains everything that lives in the TF graph. """

    def __init__(self,
                 session=None,
                 stage=0,
                 optimizer=tf.train.AdamOptimizer,
                 optimizer_args={},
                 save_dir='./'):
        """
        Args
            - session (tf.Session instance, optional): pass a tf.Session
              instance for this model to live on, otherwise a new tf.Session
              will be created at init time.
            - stage (int, optional): 0 for training, 1 for everything else
            - optimizer (tf.train.Optimizer): optimizer to add to the TF graph
            - optimizer_args (dict): arguments used to initialize the optimizer
            - save_dir (str, optional): directory to save model checkpoints

        Attributes:
            - session
            - global_step
            - stage
            - saver

        Lazy-loaded properties:
            - loss
            - init
        """
        # resolve session
        if session is None:
            self.session = tf.Session()
        else:
            self.session = session

        # create special tensors for global step and stage
        self.global_step = tf.Variable(
            initial_value=0, trainable=False, name='global_step')
        self.stage = tf.Variable(
            initial_value=stage, trainable=False, name='stage')

        # instantiate optimizer
        self.optimizer = optimizer(**optimizer_args)

        # build graph and initialize
        self.save_dir = save_dir
        self.build_graph()
        self.saver = tf.train.Saver()
        self.session.run(self.init)

    def build_graph(self):
        """ Implementations must construct the entire graph.

        Implementations may use any tensors saved to ``self`` during a
        subclass's initialization.
        """
        pass

    @lazy_property
    @abstractmethod
    def loss(self):
        """ Returns the graph node representing the loss to be optimized. """
        pass

    @lazy_property
    def init(self):
        """ Defines the init op for the model."""
        return tf.global_variables_initializer()

    def save(self,
             checkpoint_file=None,
             run_name=None,
             saver=None,
             verbose=True):
        """ Saves the model's variables to a checkpoint file.

        Args:
            - checkpoint_file (str, optional): specify a checkpoint file to save
              to. If None is passed, the checkpoint will be saved to
              '<log_dir>/model.cpkt'.
            - run_name (str, optional): you can omit checkpoint_file and pass
              this to save to '<log_dir>/<run_name>/model.cpkt'
            - saver (tf.train.Saver instance, optional): by default we will use
              self.saver to do the saving, but you can pass a custom Saver
              instance here to override.

            Note that at least one of run_name or checkpoint_file needs to be specified
        """
        if checkpoint_file is None and run_name is None:
            raise Exception("Unsupported configuration to Model.save():"
                            "must specify at least one of `run_name` or"
                            "`checkpoint_file`'")

        if saver is None:
            saver = self.saver
        if checkpoint_file is None:
            checkpoint_file = os.path.join(self.save_dir, run_name,
                                           'model.ckpt')
        if not os.path.exists(os.path.split(checkpoint_file)[0]):
            os.mkdir(os.path.split(checkpoint_file)[0])
        saver.save(self.session, checkpoint_file, global_step=self.global_step)
        if verbose:
            print('Saved at global step %04d' % self.get_global_step())

    def load(self,
             checkpoint_file=None,
             run_name=None,
             saver=None,
             verbose=True):
        """ Restores the model's variables from a checkpoint file.

        Args:
            - checkpoint_file (str, optional): specify a checkpoint file to
              restore from. If None is passed, the latest checkpoint file in the
              log_dir will be loaded. If no checkpoint file is found, a
              FileNotFoundError will be raised.
            - run_name (str, optional): you can omit checkpoint_file and pass
              this to load from the latest checkpoint file in
              '<log_dir>/<run_name>/'
            - saver (tf.train.Saver instance, optional): by default we will use
              self.saver to do the restoring, but you can pass a custom Saver
              instance here to override.
        """
        if saver is None:
            saver = self.saver
        if checkpoint_file is None:
            ckpt = tf.train.get_checkpoint_state(self.save_dir(run_name))
            if ckpt and ckpt.model_checkpoint_path:
                checkpoint_file = ckpt.model_checkpoint_path
            else:
                try:
                    raise FileNotFoundError('no checkpoint file found')
                except NameError:
                    FileNotFoundError = IOError
        saver.restore(self.session, checkpoint_file)
        if verbose:
            print('Restored to global step %04d' % self.get_global_step())

    @lazy_property
    def train_op(self):
        """ Defines the train op for the model."""
        return self.optimizer.minimize(self.loss, global_step=self.global_step)

    def get_global_step(self):
        """ Exposes the global_step tensor in a useful int format.

        Returns:
            The current global step as an int.
        """
        return int(self.session.run(self.global_step))

    def run_with_queue(self, function):
        """ Utility function holding boilerplate for queue processing.

        It sets up the coordinator, starts the queue runners, handles the
        exception thrown when the dataset is exhausted, and cleans up the
        threads.

        Args:
            - function (callable): should accept coordinator instance as a
              positional argument

        Returns:
            Whatever `function` returns, unless the end of the dataset is
            encountered.
        """
        coord = tf.train.Coordinator()
        _ = tf.train.start_queue_runners(sess=self.session, coord=coord)
        try:
            return function(coord)
        except tf.errors.OutOfRangeError:
            print('Reached end of dataset')

    def loop_with_queue(self, function, max_step=None):
        """ Runs a function through run_with_queue() in a loop.

        As a caveat, the inner function cannot return anything. Clients should
        contrive functions that write to mutable state variables.

        Args:
            - function (callable): the function to put in the loop.
            - max_step (int, optional): the number of minibatches to process
              before terminating the loop. Pass None to loop forever or until
              the dataset is exhausted.
        """

        def looped_function(coord):
            i = 0
            while not coord.should_stop() and (not max_step or i < max_step):
                function()
                i += 1

        self.run_with_queue(looped_function)

    def batch_map(self, batch_tensor, fn=None, cat_axis=0, max_step=None):
        """ Map a function over the entire dataset, one batch at a time.

        Args:
            - batch_tensor: Tensor(s) representing one batch
            - fn: A function with the signature fn(minibatch) -> np.ndarray.
              Pass None to assume the identity function.
            - cat_axis: Which axis to concatenate the results with.
            - max_step (int, optional): the number of minibatches to process.
              Pass None to loop until the dataset is exhausted.

        Returns:
            The result of applying fn to the entire dataset, returned as a
            single np.ndarray.
        """
        # resolve fn
        if fn is None:
            fn = lambda x: x

        # list to store results
        results = []

        def batch_map_function():
            minibatch = self.session.run(batch_tensor)
            results.append(fn(minibatch))

        self.loop_with_queue(batch_map_function, max_step=max_step)

        # syntactic sugar: if fn returns a tuple, we return a tuple of arrays
        if type(results[0]) == tuple:
            return tuple(
                np.concatenate(
                    [result[i] for result in results], axis=cat_axis)
                for i in range(len(results[0])))
        return np.concatenate(results, axis=cat_axis)
