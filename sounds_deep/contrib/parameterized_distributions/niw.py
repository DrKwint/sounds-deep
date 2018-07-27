import sonnet as snt
import tensorflow as tf

import distributions.dirichlet as dirichlet
import distributions.niw as niw


class NormalInverseWishart(snt.AbstractModule):
    """ Normal Inverse Wishart distribution with natural parameters

    Includes a dirichlet to be a GMM conjugate prior.
    """

    def __init__(self,
                 nb_components,
                 dimension,
                 alpha_scale=0.1,
                 beta_scale=1e-5,
                 v_init=10.,
                 m_scale=1.,
                 C_scale=10.,
                 name='normal_inverse_wishart'):
        super(NormalInverseWishart, self).__init__(name=name)
        with self._enter_variable_scope():
            alpha_init = alpha_scale * tf.ones((nb_components, ))
            beta_init = beta_scale * tf.ones((nb_components, ))
            v_init = tf.tile([float(dimension + v_init)], [nb_components])
            means_init = m_scale * tf.random_uniform(
                (nb_components, dimension), minval=-1, maxval=1)
            covariance_init = tf.matrix_inverse(C_scale * tf.tile(
                tf.expand_dims(tf.eye(dimension), axis=0), [nb_components, 1, 1]))

            A, b, beta, v_hat = niw.standard_to_natural(beta_init, means_init, covariance_init,
                                                        v_init)
            alpha = dirichlet.standard_to_natural(alpha_init)

            self.alpha = tf.get_variable(
                'alpha_k',
                # shape=(nb_components),
                dtype=tf.float32,
                initializer=alpha,
                trainable=False)
            self.A = tf.get_variable(
                "beta_k",
                # shape=(nb_components, dimension),
                dtype=tf.float32,
                initializer=A,
                trainable=False)
            self.b = tf.get_variable(
                "m_k",
                # shape=(nb_components),
                dtype=tf.float32,
                initializer=b,
                trainable=False)
            self.beta = tf.get_variable(
                "C_k",
                # shape=(nb_components, dimension, dimension),
                initializer=beta,
                dtype=tf.float32,
                trainable=False)
            self.v_hat = tf.get_variable(
                "v_k",
                # shape=(nb_components),
                dtype=tf.float32,
                initializer=v_hat,
                trainable=False)

    def _build(self):
        pass

    def expected_values(self):
        _, m, C, v = niw.natural_to_standard(self.A, self.b, self.beta, self.v_hat)
        exp_log_pi = dirichlet.expected_log_pi(
            dirichlet.natural_to_standard(self.alpha))
        with tf.name_scope('niw_expectation'):
            exp_m = tf.identity(m, 'expected_mean')
            C_inv = tf.matrix_inverse(C)
            C_inv_sym = tf.divide(
                tf.add(C_inv, tf.matrix_transpose(C_inv)), 2., name='C_inv_symmetrised')
            exp_C = tf.matrix_inverse(
                tf.multiply(
                    C_inv_sym, tf.expand_dims(tf.expand_dims(v, 1), 2), name='expected_precision'),
                name='expected_covariance')
            return exp_log_pi, exp_m, exp_C

    def m_step_op(self, prior, samples, r_nk, step_size):
        """ 
        Update parameters as the M step of an EM process

        Args:
          prior:
          samples:
        1-dim latent variable is assumed
        """
        # Bishop eq 10.51
        N_k = tf.reduce_sum(r_nk, axis=0)

        # Bishop eq 10.52
        xbar_k = (tf.expand_dims(r_nk, axis=-1) * samples) / tf.expand_dims(N_k, axis=-1)

        # Bishop eq 10.53
        x_xk = tf.reshape(samples - xbar_k, samples.get_shape().as_list() + [-1])
        S_k = tf.expand_dims(tf.expand_dims(r_nk, -1), -1) * tf.matmul(
            x_xk, x_xk, transpose_b=True)

        # rename for easy and clarity
        beta, m, C, v = niw.natural_to_standard(prior.A, prior.b, prior.beta, prior.v_hat)
        alpha_0 = prior.alpha
        m_0 = m
        beta_0 = beta
        v_0 = v
        W_0 = C

        # Bishop eq 10.58
        alpha_k = alpha_0 + N_k
        # Bishop eq 10.60
        beta_k = beta_0 + N_k
        # Bishop eq 10.61
        m_k = (tf.expand_dims(beta_0, -1) +
               tf.expand_dims(N_k, axis=-1) * xbar_k) / tf.expand_dims(beta_k, -1)
        # Bishop eq 10.62
        W_k_2nd = tf.expand_dims(tf.expand_dims(N_k, axis=-1), -1) * S_k
        xbar_m0 = tf.reshape(xbar_k - m_0, xbar_k.get_shape().as_list() + [-1])
        W_k_3rd = tf.expand_dims(tf.expand_dims(
            ((beta_0 * N_k) / (beta_0 * N_k)), -1), -1) * tf.matmul(
                xbar_m0, xbar_m0, transpose_b=True)
        W_k = W_0 + W_k_2nd + W_k_3rd
        # Bishop eq 10.63
        v_k = v_0 + N_k

        # create update op
        current_vars = (self.alpha, self.A, self.b, self.beta, self.v_hat)
        updated_params = (alpha_k, ) + niw.standard_to_natural(beta_k, tf.reduce_mean(m_k, axis=0),
                                                               tf.reduce_mean(W_k, axis=0), v_k)

        return tf.group([
            tf.assign(initial, tf.add(((1 - step_size) * initial), (step_size * updated)))
            for initial, updated in zip(current_vars, updated_params)
        ])
