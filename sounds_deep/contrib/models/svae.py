import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.distributions as tfd

import sounds_deep.contrib.distributions.gaussian as gaussian
import sounds_deep.contrib.parameterized_distributions.gmm as gmm
import sounds_deep.contrib.parameterized_distributions.niw as param_niw
import sounds_deep.contrib.util


# class weight assignments given data
def compute_log_z_given_y(eta1_phi1,
                          eta2_phi1,
                          eta1_phi2,
                          eta2_phi2,
                          pi_phi2,
                          name='log_q_z_given_y_phi'):
    """
    Args:
        eta1_phi1: encoder output; shape = N, K, L
        eta2_phi1: encoder output; shape = N, K, L, L
        eta1_phi2: GMM-EM parameter; shape = K, L
        eta2_phi2: GMM-EM parameter; shape = K, L, L
        name: tensorflow name scope
    Returns:
        log q(z|y, phi)
    """
    with tf.name_scope(name):
        N, L = eta1_phi1.get_shape().as_list()
        assert eta2_phi1.get_shape() == (N, L, L)
        K, L2 = eta1_phi2.get_shape().as_list()
        assert L2 == L
        assert eta2_phi2.get_shape() == (K, L, L)

        # combine eta2_phi1 and eta2_phi2
        eta2_phi_tilde = tf.add(
            tf.expand_dims(eta2_phi1, axis=1), tf.expand_dims(
                eta2_phi2, axis=0))

        # w_eta2 = -0.5 * inv(sigma_phi1 + sigma_phi2)
        solved = tf.matrix_solve(
            eta2_phi_tilde,
            tf.tile(tf.expand_dims(eta2_phi2, axis=0), [N, 1, 1, 1]))
        w_eta2 = tf.einsum('nju,nkui->nkij', eta2_phi1, solved)

        # for nummerical stability...
        w_eta2 = tf.divide(
            w_eta2 + tf.matrix_transpose(w_eta2), 2., name='symmetrised')

        # w_eta1 = inv(sigma_phi1 + sigma_phi2) * mu_phi2
        w_eta1 = tf.einsum(
            'nuj,nkuv->nkj',
            eta2_phi1,
            tf.matrix_solve(
                eta2_phi_tilde,
                tf.tile(
                    tf.expand_dims(tf.expand_dims(eta1_phi2, axis=0), axis=-1),
                    [N, 1, 1, 1]))  # shape inside solve= N, K, D, 1
        )  # w_eta1.shape = N, K, D

        # compute means
        mu_phi1, _ = gaussian.natural_to_standard(eta1_phi1, eta2_phi1)

        # compute log_z_given_y_phi
        return gaussian.log_probability_nat(mu_phi1, w_eta1, w_eta2,
                                            pi_phi2)  #, (w_eta1, w_eta2)


def subsample_x(x_k_samples, log_q_z_given_y, seed=0):
    """
    Given S samples for each of the K components for N datapoints (x_k_samples) and q(z_n=k|y), subsample S samples for
    each data point
    Args:
        x_k_samples: sample matrix of shape (N, K, S, L)
        log_q_z_given_y: probability q(z_n=k|y_n, phi)
        seed: random seed
    Returns:
        x_samples: a sample matrix of shape (N, S, L)
    """
    with tf.name_scope('subsample_x'):
        N, K, S, L = x_k_samples.get_shape().as_list()

        # prepare indices for N and S dimension
        # tf can't tile int32 tensors on the GPU. Therefore, tile it as float and convert to int afterwards
        n_idx = tf.to_int32(
            tf.tile(
                tf.reshape(tf.range(N, dtype=tf.float32), (-1, 1)),
                multiples=[1, S]))
        s_idx = tf.to_int32(
            tf.tile(
                tf.reshape(tf.range(S, dtype=tf.float32), (1, -1)),
                multiples=[N, 1]))

        # sample S times z ~ q(z|y, phi) for each N.
        z_samps = tf.multinomial(
            logits=log_q_z_given_y, num_samples=S, seed=seed, name='z_samples')
        z_samps = tf.cast(z_samps, dtype=tf.int32)

        # tensor of shape (N, S, 3), containing indices of all chosen samples
        choices = tf.concat([
            tf.expand_dims(n_idx, 2),
            tf.expand_dims(z_samps, 2),
            tf.expand_dims(s_idx, 2)
        ],
                            axis=2)

        return tf.gather_nd(x_k_samples, choices, name='x_samples')


class GMM_SVAE(snt.AbstractModule):
    def __init__(self,
                 latent_dimension,
                 nb_components,
                 encoder,
                 decoder,
                 name='gmm_svae'):
        super(GMM_SVAE, self).__init__(name=name)
        self._latent_dimension = latent_dimension
        self._nb_components = nb_components
        self._encoder = encoder
        self._decoder = decoder

        with self._enter_variable_scope():
            self._mu_net = snt.Sequential(
                [tf.layers.Flatten(),
                 snt.Linear(self._latent_dimension)])
            self._sigma_net = snt.Sequential([
                tf.layers.Flatten(),
                snt.Linear(self._latent_dimension), tf.nn.softplus
            ])

            self._theta_prior = param_niw.NormalInverseWishart(
                nb_components,
                latent_dimension,
                alpha_scale=0.05 / nb_components,
                beta_scale=0.5,
                m_scale=0,
                C_scale=latent_dimension + 0.5,
                v_init=latent_dimension + 0.5)
            self._theta = param_niw.NormalInverseWishart(
                nb_components,
                latent_dimension,
                alpha_scale=1.,
                beta_scale=1.,
                m_scale=5.,
                C_scale=2 * latent_dimension,
                v_init=latent_dimension + 1)
            _, exp_mu, exp_cov = self._theta.expected_values()
            self.phi_gmm = gmm.GMM(
                nb_components,
                latent_dimension,
                mu_init=exp_mu,
                cov_init=exp_cov,
                trainable=True)

    def _build(self,
               inputs,
               nb_samples=10,
               seed=0,
               encoder_param_type='natural'):
        ### vae encode
        emb = self._encoder(inputs)
        enc_eta1 = self._mu_net(emb)
        enc_eta2_diag = self._sigma_net(emb)
        if encoder_param_type == 'natural':
            enc_eta2_diag *= -1. / 2
            # enc_eta2_diag -= 1e-8
        enc_eta2 = tf.matrix_diag(enc_eta2_diag)

        ### GMM natural parameters
        gmm_pi, gmm_eta1, gmm_eta2 = self.phi_gmm()

        ### combined GMM and VAE latent parameters
        # eta1_tilde.shape = (N, K, D); eta2_tsilde.shape = (N, K, D, D)
        # with tf.control_dependencies([util.matrix_is_pos_def_op(-2 * enc_eta2)]):
        eta1_tilde = tf.expand_dims(
            enc_eta1, axis=1) + tf.expand_dims(
                gmm_eta1, axis=0)
        eta2_tilde = tf.expand_dims(
            enc_eta2, axis=1) + tf.expand_dims(
                gmm_eta2, axis=0)
        log_z_given_y_phi = compute_log_z_given_y(enc_eta1, enc_eta2, gmm_eta1,
                                                  gmm_eta2, gmm_pi)
        # with tf.control_dependencies([util.matrix_is_pos_def_op(-2 * gmm_eta2)]):
        mu, cov = gaussian.natural_to_standard(eta1_tilde, eta2_tilde)
        posterior_mixture_distribution = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(tf.exp(log_z_given_y_phi)),
            components_distribution=tfd.MultivariateNormalFullCovariance(
                loc=mu, covariance_matrix=cov))

        # sample x for each of the K components
        # latent_k_samples.shape == nb_samples, batch_size, nb_components, latent_dim
        latent_k_samples = posterior_mixture_distribution.components_distribution.sample(
            [nb_samples])

        ### vae decode
        output_mean = snt.BatchApply(self._decoder, n_dims=3)(latent_k_samples)
        output_variance = tf.get_variable(
            'output_variance',
            dtype=tf.float32,
            initializer=tf.zeros(output_mean.get_shape().as_list()),
            trainable=True)  # learned parameter for output distribution
        output_distribution = tfd.Independent(
            tfd.MultivariateNormalDiagWithSoftplusScale(
                loc=output_mean, scale_diag=output_variance),
            reinterpreted_batch_ndims=2)

        # subsample for each datum in minibatch (go from `nb_samples` per component to `nb_samples` total)
        latent_samples = subsample_x(
            tf.transpose(latent_k_samples, [1, 0, 2, 3]), log_z_given_y_phi,
            seed)

        return output_distribution, posterior_mixture_distribution, latent_k_samples, latent_samples, log_z_given_y_phi

    def compute_elbo(self, data, output_distribution,
                     posterior_mixture_distribution, latent_k_samples,
                     log_z_given_y_phi):
        nb_samples = output_distribution.batch_shape[0]
        r_nk = tf.exp(log_z_given_y_phi)

        # compute negative reconstruction error
        with tf.name_scope('compute_reconstruction_err'):
            shaped_data = tf.tile(
                tf.expand_dims(tf.expand_dims(data, axis=1), axis=0),
                [nb_samples, 1, self._nb_components, 1, 1, 1])
            neg_reconstruction_error = tf.reduce_mean(
                tf.reduce_sum(
                    output_distribution.log_prob(shaped_data) * r_nk, axis=2))

        # compute E[log q_phi(x,z=k|y)]
        with tf.name_scope('compute_regularizer'):
            with tf.name_scope('log_numerator'):
                log_N_x_given_phi = posterior_mixture_distribution.components_distribution.log_prob(
                    latent_k_samples)
                log_numerator = log_N_x_given_phi + log_z_given_y_phi

            with tf.name_scope('log_denominator'):
                with tf.name_scope('theta_expected_vals'):
                    log_pi, mu, sigma = self._theta.expected_values()
                    # mu = tf.Print(mu, [tf.norm(mu, axis=1)], summarize=10, message='gmm_mu: ')
                    # mu = tf.Print(mu, [tf.norm(sigma, axis=[1, 2])], summarize=10, message='gmm_sigma: ')
                    # mu = tf.Print(mu, [tf.nn.softmax(log_pi)], summarize=10, message='gmm_log_pi: ')
                    mu = tf.stop_gradient(mu)
                    sigma = tf.stop_gradient(sigma)
                    log_pi = tf.stop_gradient(log_pi)
                    theta_gaussian_dist = tfd.MultivariateNormalFullCovariance(
                        mu, sigma)

                log_N_x_given_theta = theta_gaussian_dist.log_prob(
                    latent_k_samples)
                log_denominator = log_N_x_given_theta + log_pi

            # log_denominator = tf.Print(log_denominator, [latent_k_samples, mu, sigma])
            # log_denominator = tf.Print(log_denominator, [log_N_x_given_theta, tf.log(pi)])

        # weighted sum using r_nk over components, then mean over samples and batch
        regularizer_term = tf.reduce_mean(
            tf.reduce_sum(r_nk * (log_numerator - log_denominator), axis=2))

        elbo = neg_reconstruction_error - regularizer_term

        details = (neg_reconstruction_error,
                   tf.reduce_mean(
                       tf.reduce_sum(
                           tf.multiply(r_nk, log_numerator), axis=-1),
                       axis=0),
                   tf.reduce_mean(
                       tf.reduce_sum(
                           tf.multiply(r_nk, log_denominator), axis=-1),
                       axis=0), regularizer_term)

        return elbo, details

    def m_step_op(self, latent_posterior_samples, r_nk, step_size):
        return self._theta.m_step_op(self._theta_prior,
                                     latent_posterior_samples, r_nk, step_size)
