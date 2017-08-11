from abc import abstractmethod
import tensorflow as tf
import sonnet as snt

from sounds_deep.util.lazy_property import lazy_property

class Model(snt.AbstractModule):
    def __init__(self, name, hyperparameter_dict):
        super(Model, self).__init__(name=name)
        self.hyperparameter_dict = hyperparameter_dict

    @lazy_property
    @abstractmethod
    def loss(self):
        """ Returns the graph node representing the loss to be optimized. """
        pass

    @lazy_property
    def losses(self):
        return {'total loss': tf.reduce_mean(self.loss)}

    def add_metrics(self):
        """ Adds streaming metrics to the graph.

        Default implementation computes a streaming mean for every loss.
        Subclasses are free to override this with custom streaming metrics.

        Returns:
            - metrics (dict of Tensors): named dict of metric Tensors
            - update_op (op): operation to update all metrics
        """
        metrics = {}
        update_ops = []
        with tf.name_scope('metrics'):
            for loss in self.losses:
                mean_value, update_op = tf.contrib.metrics.streaming_mean(
                    self.losses[loss])
                metrics[loss] = mean_value
                update_ops.append(update_op)
        return metrics, tf.group(*update_ops)

    @lazy_property
    def metric_reset_op(self):
        metric_vars = [i for i in tf.local_variables()
                       if i.name.split('/')[0] == 'metrics']
        return tf.variables_initializer(metric_vars)
