import sonnet as snt
import tensorflow as tf

tfd = tf.contrib.distributions


class SoftplusLogisticDistribution(snt.AbstractModule):
    def __init__(self, name="parameterized_logistic_distribution"):
        super(SoftplusLogisticDistribution, self).__init__(name=name)

    @property
    def scale(self):
        return tf.nn.softplus(self._inv_softplus_scale)

    def _build(self, inputs):
        self._inv_softplus_scale = tf.get_variable(
            "inv_softplus_scale", shape=inputs.shape, dtype=tf.float32)
        return tfd.Logistic(
            loc=inputs, scale=tf.nn.softplus(self._inv_softplus_scale))
