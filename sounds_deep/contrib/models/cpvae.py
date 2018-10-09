import os

import numpy as np
import sonnet as snt
import tensorflow as tf

import sounds_deep.contrib.models.vae as vae
import sounds_deep.contrib.ops.ddt as ddt
from sklearn import tree

tfd = tf.contrib.distributions
tfb = tfd.bijectors


def std_gaussian_KL_divergence(mu, sigma):
    """ Analytic KL-div between N(mu, sigma) and N(0, 1) """
    return -0.5 * tf.reduce_sum(
        1 + tf.log(tf.square(sigma)) - tf.square(mu) - tf.square(sigma), 1)


def diagonal_gaussian_kl(mu_p, sigma_p, mu_q, sigma_q):
    return -tf.reduce_sum(
        tf.log(sigma_q / sigma_p) + tf.square(sigma_p) +
        (tf.square(mu_p - mu_q) / (2 * tf.square(sigma_q))) - 0.5)


class CPVAE(snt.AbstractModule):
    """

    Note: Make sure learning rate is inversely correlated with data size. If you're getting
    myserious crashes, try an order of magnitude smaller.

    Attributes:
        latent_prior (tfd.Distribution): Prior latent distribution.
        latent_posterior (tfd Distribution): Posterior latent distribution.
        output_distribution (tfd Distribution): VAE output distribution.
        distortion (Tensor): Also called reconstruction error.
        rate (Tensor): Also called latent loss.
        elbo (Tensor): Evidence Lower Bound. Minimize to train a VAE.
        importance_weighted_elbo (Tensor): importance weighted ELBO. Minimize to train an IWAE.
    """

    def __init__(self,
                 latent_dimension,
                 box_num,
                 class_num,
                 decision_tree,
                 encoder_net,
                 decoder_net,
                 beta,
                 gamma,
                 delta,
                 output_dist_fn=vae.BERNOULLI_FN,
                 name='vae'):
        """
        Args:
            latent_dimension (int): Dimension of the latent variable.
            encoder_net (snt.Module): Encoder mapping from rank 4 input to rank 2 output.
            decoder_net (Tensor -> Tensor): Decoder mapping from a rank 2 input to rank 4 output.
            prior_fn (int -> tfd.Distribution): Callable which takes an integral dimension size.
            posterior_fn (Tensor -> Tensor -> tfd.Distribution): Callable which takes location and
                                                                 scale and returns a tfd distribution.
            output_dist_fn (Tensor -> tfd.Distribution): Callable from loc to a tfd distribution.
        """
        super(CPVAE, self).__init__(name=name)
        self._latent_dimension = latent_dimension
        self._box_num = box_num
        self._class_num = class_num
        self._decision_tree = decision_tree
        self._encoder = encoder_net
        self._decoder = decoder_net
        self._output_dist_fn = output_dist_fn
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        with self._enter_variable_scope():
            self._loc = snt.Linear(latent_dimension)
            self._scale = snt.Sequential(
                [snt.Linear(latent_dimension), tf.nn.softplus])

            self.class_locs = tf.Variable(
                np.zeros([self._class_num, self._latent_dimension],
                         dtype=np.float32))
            self.class_scales = tf.Variable(
                np.ones([self._class_num, self._latent_dimension],
                        dtype=np.float32))

            # declare variables for gaussian box inference
            self._lower = tf.Variable(
                np.empty([box_num, latent_dimension], dtype=np.float32),
                trainable=False)
            self._upper = tf.Variable(
                np.empty([box_num, latent_dimension], dtype=np.float32),
                trainable=False)
            self._values = tf.Variable(
                np.empty([box_num, class_num], dtype=np.float32),
                trainable=False)
            self._inference = ddt.TransductiveBoxInference()

    def _build(self, data, labels, n_samples=1, analytic_kl=True):
        """Builds VAE (or IWAE depending on arguments).

        Args:
            inputs (Tensor): A rank 4 tensor with NHWC shape. Values of this tensor are assumed to
                             be in `[0,1]`.
            n_samples (int): Number of samples to use in importance weighting. Model is a VAE if
                             `n_samples == 1` and an IWAE if `n_samples > 1`.
            analytic_kl (bool): Whether to use a built-in analytic calculation of KL-divergence if
                                it's available. This setting is treated as `False` if `n_samples > 1`.
        Returns:
            Tensor: result of encoding, sampling, and decoding inputs in [0,1]
        """
        x = data
        encoder_repr = self._encoder(x)
        loc = self._loc(encoder_repr)
        self.z_mu = loc
        scale = self._scale(encoder_repr)
        self.z_sigma = scale
        self.latent_posterior = tfd.MultivariateNormalDiag(loc, scale)
        latent_posterior_sample = self.latent_posterior.sample(n_samples)
        self.latent_posterior_sample = latent_posterior_sample
        sample_decoder = snt.BatchApply(self._decoder)
        output = sample_decoder(latent_posterior_sample)
        self.output_distribution = tfd.Independent(
            self._output_dist_fn(output), reinterpreted_batch_ndims=3)

        distortion = -self.output_distribution.log_prob(x)
        mean_adjustment = tf.matmul(labels, self.class_locs, a_is_sparse=True)
        # scale_adjustment = tf.matmul(labels, self.class_scales, a_is_sparse=True)
        rate = self.beta * std_gaussian_KL_divergence(loc - mean_adjustment,
                                                      scale)
        # rate = self.beta * diagonal_gaussian_kl(loc, scale, mean_adjustment, scale_adjustment)
        elbo_local = -(rate + distortion)

        y_pred = self._inference(loc, scale, self._lower, self._upper,
                                 self._values)
        classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y_pred, labels=tf.argmax(labels, axis=1))

        drift_loss = tf.norm(self.z_mu, ord=2, axis=1)

        self.classification_loss = classification_loss
        self.distortion = distortion
        self.rate = rate
        self.posterior_logp = self.latent_posterior.log_prob(
            latent_posterior_sample)
        self.elbo = tf.reduce_mean(tf.reduce_logsumexp(elbo_local, axis=0))
        self.importance_weighted_elbo = tf.reduce_mean(
            tf.reduce_logsumexp(elbo_local, axis=0) -
            tf.log(tf.to_float(n_samples)))

        objective = -self.elbo + self.gamma * classification_loss + self.delta * drift_loss

        return objective

    def posterior_parameters(self, session, label_tensor, batch_num,
                             feed_dict_fn):
        codes = []
        labels = []
        mu = []
        sigma = []
        for _ in range(batch_num):
            c, m, s, l = session.run([
                self.latent_posterior_sample, self.z_mu, self.z_sigma,
                label_tensor
            ],
                                     feed_dict=feed_dict_fn())
            codes.append(c)
            labels.append(l)
            mu.append(m)
            sigma.append(s)
        mu = np.concatenate(mu)
        sigma = np.concatenate(sigma)
        codes = np.squeeze(np.concatenate(codes, axis=1))
        labels = np.argmax(np.concatenate(labels), axis=1)
        sigma = np.array(sigma)
        return mu, sigma, codes, labels

    def aggregate_posterior_parameters(self, session, label_tensor, batch_num,
                                       feed_dict_fn):
        mu, sigma, _, labels = self.posterior_parameters(
            session, label_tensor, batch_num, feed_dict_fn)
        if len(labels.shape) > 1: labels = np.argmax(labels, axis=1)
        class_locs = np.empty([self._class_num, self._latent_dimension])
        class_scales = np.empty([self._class_num, self._latent_dimension])
        mu_sq = np.square(mu)
        sigma_sq = np.square(sigma)
        sum_sq = sigma_sq + mu_sq
        for l in range(self._class_num):
            idxs = np.nonzero(labels == l)[0]
            class_locs[l] = np.mean(mu[idxs], axis=0)
            class_scales[l] = np.mean(
                sum_sq[idxs], axis=0) - np.square(class_locs[l])
        return class_locs, class_scales

    def update(self,
               session,
               label_tensor,
               batch_num,
               feed_dict_fn,
               epoch,
               output_dir=''):
        """
        Args:
        - label_tensor: each label should be one-hot encoded
        """
        # run data
        codes = []
        labels = []
        mu = []
        sigma = []
        for _ in range(batch_num):
            c, m, s, l = session.run([
                self.latent_posterior_sample, self.z_mu, self.z_sigma,
                label_tensor
            ],
                                     feed_dict=feed_dict_fn())
            codes.append(c)
            labels.append(l)
            mu.append(m)
            sigma.append(s)
        mu = np.concatenate(mu)
        sigma = np.concatenate(sigma)
        codes = np.squeeze(np.concatenate(codes, axis=1))
        labels = np.argmax(np.concatenate(labels), axis=1)
        sigma = np.array(sigma)

        # update class stats
        if len(labels.shape) > 1: labels = np.argmax(labels, axis=1)
        class_locs = np.empty([self._class_num, self._latent_dimension])
        class_scales = np.empty([self._class_num, self._latent_dimension])
        mu_sq = np.square(mu)
        sigma_sq = np.square(sigma)
        sum_sq = sigma_sq + mu_sq
        for l in range(self._class_num):
            idxs = np.nonzero(labels == l)[0]
            class_locs[l] = np.mean(mu[idxs], axis=0)
            class_scales[l] = np.mean(
                sum_sq[idxs], axis=0) - np.square(class_locs[l])
            # print('{}: {}'.format(l, np.linalg.norm(class_locs[l])))
        session.run([
            self.class_locs.assign(class_locs),
            self.class_scales.assign(class_scales)
        ])

        # train ensemble
        self._decision_tree.fit(codes, labels)
        lower_, upper_, values_ = ddt.get_decision_tree_boundaries(
            self._decision_tree, self._latent_dimension, self._class_num)
        # ensure arrays are of correct size, even if tree is too small
        lower, upper, values = session.run(
            [self._lower, self._upper, self._values])
        lower[:lower_.shape[0], :lower_.shape[1]] = lower_
        upper[:upper_.shape[0], :upper_.shape[1]] = upper_
        values[:values_.shape[0], :values_.shape[1]] = values_
        # set tree values into tf graph
        session.run([
            self._lower.assign(lower),
            self._upper.assign(upper),
            self._values.assign(values)
        ])
        tree.export_graphviz(
            self._decision_tree,
            out_file=os.path.join(output_dir, 'ddt_epoch{}.dot'.format(epoch)),
            filled=True,
            rounded=True)

        predicted_labels = self._decision_tree.predict(codes)
        return np.mean(predicted_labels != labels)

    def sample(self, batch_size, cluster_ids=None, latent_code=None):
        """Generate samples of the specified shape.

        `self._build` must be called before this function. 
        Note that a call to sample() without arguments will generate a single sample

        Args:
            sample_shape: 0D or 1D int32 Tensor. Shape of the generated samples.
            seed: Python integer seed for RNG
            name: name to give to the op.
        Returns:
            Tensor: a sample with prepended dimensions sample_shape.
        """
        assert self.is_connected, 'Must call `build` before this function'
        if latent_code is None:
            epsilon = tf.random_normal([batch_size, self._latent_dimension])
            if cluster_ids is None:
                cluster_ids = tf.squeeze(
                    tf.multinomial(tf.ones([batch_size, self._class_num]), 1))
                cluster_ids = tf.one_hot(cluster_ids, self._class_num)
            loc = tf.matmul(cluster_ids, self.class_locs, a_is_sparse=True)
            scale = tf.matmul(cluster_ids, self.class_scales, a_is_sparse=True)
            latent_code = epsilon * scale + loc
        output = self._decoder(latent_code)
        return self._output_dist_fn(output).mean()
