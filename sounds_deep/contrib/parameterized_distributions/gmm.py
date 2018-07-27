import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.distributions as tfd

import distributions.gaussian as gaussian


class GMM(snt.AbstractModule):
    def __init__(self, nb_components, dimension, mu_init=None, cov_init=None, trainable=False, name='gmm'):
        super(GMM, self).__init__(name=name)
        with self._enter_variable_scope():
            self.pi = tf.get_variable(
                "pi", shape=(nb_components), dtype=tf.float32, trainable=trainable)

            if mu_init is not None:
                assert mu_init.get_shape().as_list() == [nb_components, dimension]
                self.mu = tf.get_variable(
                    "mixture_mu", initializer=mu_init, dtype=tf.float32, trainable=trainable)
            else:
                self.mu = tf.get_variable(
                    "mixture_mu",
                    shape=(nb_components, dimension),
                    dtype=tf.float32,
                    trainable=trainable)

            if cov_init is not None:
                assert cov_init.get_shape().as_list() == [nb_components, dimension, dimension]
                self._L_k_raw = tf.get_variable(
                    "mixture_lower_cov",
                    initializer=tf.cholesky(cov_init),
                    dtype=tf.float32,
                    trainable=trainable)
            else:
                self._L_k_raw = tf.get_variable(
                    "mixture_lower_cov",
                    shape=(nb_components, dimension, dimension),
                    dtype=tf.float32,
                    trainable=trainable)

            self.model = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=self.pi),
                components_distribution=tfd.MultivariateNormalFullCovariance(
                    loc=self.mu, covariance_matrix=self._L_k_raw))

    def precision(self):
        L_k = tf.linalg.LinearOperatorLowerTriangular(self._L_k_raw, name='to_triL').to_dense()
        L_k = tf.matrix_set_diag(
            L_k, tf.nn.softplus(tf.matrix_diag_part(L_k), name='softplus_diag'), name='L')
        P = tf.matmul(L_k, tf.matrix_transpose(L_k), name='precision')
        return P

    def _build(self):
        pi = tf.nn.softmax(self.pi)
        eta1 = self.mu
        eta2 = tf.multiply(tf.constant(-0.5, dtype=tf.float32), self.precision())
        return pi, eta1, eta2

    def standard_parameters(self):
        pi, eta1, eta2 = self._build()
        mu, sigma = gaussian.natural_to_standard(eta1, eta2)
        return pi, mu, sigma

    def sample_per_component(self, sample_shape):
        return self.model.components_distribution.sample(sample_shape)
