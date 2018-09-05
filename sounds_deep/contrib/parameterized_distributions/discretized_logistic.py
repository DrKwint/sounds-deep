import sonnet as snt
import tensorflow as tf

tfd = tf.contrib.distributions


class DiscretizedLogistic(snt.AbstractModule, tfd.Distribution):
    def __init__(self, loc, batch_dims=3, name="discretized_logistic"):
        super(DiscretizedLogistic, self).__init__(name=name)
        self._name = name
        self._dtype = tf.float32
        self._reparameterization_type = tfd.NOT_REPARAMETERIZED
        self._allow_nan_stats = False
        self._graph_parents = []
        self._loc = loc
        self._batch_dims = batch_dims
        self._log_scale = tf.get_variable(
            "log_scale",
            initializer=tf.zeros(loc.get_shape().as_list()[-batch_dims:]),
            dtype=tf.float32)

    def mean(self):
        return self._loc

    @property
    def scale(self):
        return tf.exp(self._log_scale)

    def _build(self):
        pass

    def log_prob(self, sample, binsize=1 / 256.0):
        scale = tf.exp(self._log_scale)
        mean = self._loc
        sample = (tf.floor(sample / binsize) * binsize - mean) / scale
        logp = tf.log(
            tf.sigmoid(sample + binsize / scale) - tf.sigmoid(sample) + 1e-7)
        return logp # tf.reduce_sum(logp, [2, 3, 4])
