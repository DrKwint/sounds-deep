"""
Semi-supervised VAE (from Kigma and Welling, and Jang et al.)
"""
import functools
import operator

import numpy as np
import sonnet as snt
import tensorflow as tf

import sounds_deep.contrib.models.vae as vae
import sounds_deep.contrib.util.util as util

tfd = tf.contrib.distributions

UNIFORM_RELAXED_DISCRETE_FN = lambda shape, temperature: tfd.ExpRelaxedOneHotCategorical(temperature, logits=tf.ones(shape))
STD_GAUSSIAN_FN = lambda latent_dimension: tfd.MultivariateNormalDiag(
    loc=tf.zeros(latent_dimension), scale_diag=tf.ones(latent_dimension))


class M2(snt.AbstractModule):
    """
    Revised version of Kingma & Welling's 2014 M2 model with Jang et al.'s
    Gumbel-Softmax distribution for the y variable.

    Uses a uniform Gumbel-Softmax as the prior on y.
    """

    def __init__(self,
                 z_shape,
                 y_shape,
                 z_net,
                 y_net,
                 x_hat_net,
                 y_prior_temperature,
                 z_prior_fn=vae.STD_GAUSSIAN_FN,
                 z_posterior_fn=vae.SOFTPLUS_GAUSSIAN_FN,
                 x_hat_dist_fn=vae.BERNOULLI_FN,
                 name='named_latent_vae'):
        """
        Args:
            z_shape (int or [int]): shape of the z latent variable
            y_shape (int or [int]): shape of the y latent variable
            z_net (Tensor -> Tensor): fn taking rank 4 input
            y_net (Tensor -> Tensor): fn taking rank 4 input
            x_hat_net (Tensor -> Tensor): fn taking rank 2 input
            y_prior_temperature (float or Tensor): temperature on G-S y prior distribution
            z_prior_fn ([int] -> TFD): fn which takes event_shape and returns a TFD
            z_posterior_fn (Tensor -> Tensor -> TFD): fn which takes loc and scale and returns a TFD
            x_hat_dist_fn (Tensor -> TFD): fn which takes logits and returns a TFD
            name (str): name for this module in the TF graph
        """
        super(M2, self).__init__(name=name)
        if type(z_shape) is int: z_shape = [z_shape]
        if type(y_shape) is int: y_shape = [y_shape]
        self._z_net = z_net
        self._y_net = y_net
        self._x_hat_net = x_hat_net
        self._z_posterior_fn = z_posterior_fn
        self._x_hat_dist_fn = x_hat_dist_fn

        product = lambda x: functools.reduce(operator.mul, x)
        with self._enter_variable_scope():
            # input to _loc and _scale is shape
            # (batch_size, _)
            # want output shape (batch_size) + z_shape
            self._loc = snt.Sequential([
                    snt.Linear(product(z_shape)),
                    lambda x: tf.reshape(x, [-1] + z_shape)
                ])
            self._scale = snt.Sequential([
                    snt.Linear(product(z_shape)),
                    lambda x: tf.reshape(x, [-1] + z_shape)
                ])
            # input to _y_logits is shape
            # (n_samples, batch_size, height, width, channels)
            # want output shape (n_samples, batch_size) + y_shape
            self._y_logits = snt.Sequential([
                snt.BatchFlatten(),
                snt.Linear(product(y_shape)),
                lambda x: tf.reshape(x, [-1] + y_shape)
            ])
            self.z_prior = z_prior_fn(z_shape)
            self.y_prior = tfd.ExpRelaxedOneHotCategorical(
                y_prior_temperature, tf.ones(y_shape))

    def _build(self,
               unlabeled_input,
               labeled_input,
               hvar_labels,
               y_posterior_temperature,
               classification_loss_coeff=0.8,
               n_samples=1):
        """
        Builds the model and returns the training objective

        Args:
            unlabeled_input (Tensor): unsupervised data of shape NHWC
            labeled_input (Tensor): supervised data parallel to y_labels of
                shape NHWC. HWC must be equal to that of unlabeled input, but
                not N.
            y_labels (Tensor): labels of shape (N,) + y_shape
            y_posterior_temperature (float or Tensor): temperature on G-S y
                prior distribution
            class_loss_coeff (float or Tensor): called $alpha$ in Jang et al.,
                modifies effect of y log_prob term on loss
            n_samples (int): number of samples in importance sampling
        Returns:
            (Tensor, dict): loss tensor and stats dict of values for observability
        """

        def infer_y_posterior(x, temperature):
            logits = self._y_logits(self._y_net(x))
            return tfd.ExpRelaxedOneHotCategorical(temperature, logits=logits)

        def infer_z_posterior(x, y):
            """x should be rank 4 and y should be rank 2 or 3"""
            z_repr = self._z_net(x, y)
            z_repr = snt.BatchFlatten()(z_repr)
            print(z_repr)
            return self._z_posterior_fn(self._loc(z_repr), self._scale(z_repr))
            # data_shape = util.int_shape(x)
            # print(y)
            # y_channel = tf.tile(
            #     tf.expand_dims(tf.expand_dims(tf.expand_dims(y, 0), 2), 2),
            #     [n_samples, 1, data_shape[1], data_shape[2], 1])
            # print(y_channel)
            # z_encoder_input = tf.concat(
            #     [
            #         tf.tile(tf.expand_dims(x, 0), [n_samples, 1, 1, 1, 1]),
            #         y_channel
            #     ],
            #     axis=4)
            # print(z_encoder_input)
            # batch_encoder = snt.BatchApply(self._z_net)
            # z_encoder_repr = batch_encoder(z_encoder_input)
            # print(z_encoder_repr)
            # z_encoder_repr = snt.BatchFlatten(preserve_dims=2)(z_encoder_repr)
            # print(z_encoder_repr)

            # y_shape = util.int_shape(y)
            # if len(y_shape) == 2:
            #     y = tf.tile(tf.expand_dims(y, 0), [n_samples, 1, 1])
            # z_logits = snt.BatchFlatten()(self._z_net(x))
            # z_logits = tf.tile(tf.expand_dims(z_logits, 0), [n_samples, 1, 1])
            # z_encoder_repr = tf.concat([z_logits, y], -1)
            # return self._z_posterior_fn(
            #     self._loc(z_encoder_repr), self._scale(z_encoder_repr))

        y_posterior_labeled = infer_y_posterior(labeled_input,
                                                y_posterior_temperature)
        y_sample_labeled = y_posterior_labeled.sample()
        y_posterior_unlabeled = infer_y_posterior(unlabeled_input,
                                                  y_posterior_temperature)
        y_sample_unlabeled = y_posterior_unlabeled.sample()

        z_posterior_labeled = infer_z_posterior(labeled_input, hvar_labels)
        z_sample_labeled = z_posterior_labeled.sample()
        z_posterior_unlabeled = infer_z_posterior(unlabeled_input,
                                                  tf.exp(y_sample_unlabeled))
        z_sample_unlabeled = z_posterior_unlabeled.sample()

        x_hat_labeled = self._infer_x_hat(hvar_labels, z_sample_labeled)
        x_hat_unlabeled = self._infer_x_hat(
            tf.exp(y_sample_unlabeled), z_sample_unlabeled)

        supervised_distortion = -x_hat_labeled.log_prob(labeled_input)
        unsupervised_distortion = -x_hat_unlabeled.log_prob(unlabeled_input)

        supervised_rate = (z_posterior_labeled.log_prob(z_sample_labeled) -
                           self.z_prior.log_prob(z_sample_labeled) -
                           self.y_prior.log_prob(hvar_labels))
        unsupervised_rate = (z_posterior_unlabeled.log_prob(z_sample_unlabeled)
                             - self.z_prior.log_prob(z_sample_unlabeled) -
                             self.y_prior.log_prob(y_sample_unlabeled))

        y_logits = self._y_logits(self._y_net(unlabeled_input))
        unsupervised_y_entropy = -tf.reduce_sum(
            tf.exp(y_logits) * y_logits, axis=-1)
        supervised_y_log_prob = tf.reduce_sum(
            hvar_labels * y_sample_labeled, axis=-1)

        supervised_elbo_local = -(supervised_distortion + supervised_rate)
        supervised_elbo = tf.reduce_mean(supervised_elbo_local, axis=-1)
        unsupervised_elbo_local = -(unsupervised_distortion +
                                    unsupervised_rate) + unsupervised_y_entropy
        unsupervised_elbo = tf.reduce_mean(unsupervised_elbo_local, axis=0)

        elbo = supervised_elbo + unsupervised_elbo + (
            classification_loss_coeff * supervised_y_log_prob)

        stats_dict = dict()
        stats_dict['y_sample_unlabeled'] = y_sample_unlabeled
        stats_dict['supervised_rate'] = supervised_rate
        stats_dict['unsupervised_rate'] = unsupervised_rate
        stats_dict['supervised_distortion'] = supervised_distortion
        stats_dict['unsupervised_distortion'] = unsupervised_distortion
        stats_dict['supervised_elbo'] = supervised_elbo
        stats_dict['unsupervised_elbo'] = unsupervised_elbo
        stats_dict['supervised_y_log_prob'] = supervised_y_log_prob
        stats_dict['unsupervised_y_entropy'] = unsupervised_y_entropy

        return elbo, stats_dict

    def _infer_x_hat(self, y, z):
        """z should be of rank 3 and y should be of rank 2 or 3"""
        z_shape = util.int_shape(z)
        y_shape = util.int_shape(y)
        if len(z_shape) == 2:
            z = tf.expand_dims(z, 0)
        if len(y_shape) == 2:
            y = tf.tile(tf.expand_dims(y, 0), [tf.shape(z)[0], 1, 1])
        joint_yz = tf.concat([y, z], axis=-1)
        sample_decoder = snt.BatchApply(self._x_hat_net)
        output = sample_decoder(joint_yz)
        return tfd.Independent(
            self._x_hat_dist_fn(output), reinterpreted_batch_ndims=3)

    def sample(self,
               sample_shape=(),
               seed=None,
               z_value=None,
               y_value=None,
               name='sample'):
        """Generate samples of the specified shape.

        `self._build` must be called before this function. 
        Note that a call to sample() without arguments will generate a single sample.
        Note also that either y_temperature or y_sample must be set.

        Args:
            sample_shape: 0D or 1D int32 Tensor. Shape of the generated samples.
            seed: Python integer seed for RNG
            name: name to give to the op.
        Returns:
            Tensor: a sample with prepended dimensions sample_shape.
        """
        assert self.is_connected, 'Must call `build` before this function'
        if sample_shape == (): sample_shape = 1
        with self._enter_variable_scope():
            with tf.variable_scope(name):
                if y_value is None:
                    y_value = tf.exp(
                        self.y_prior.sample(sample_shape, seed, 'y_sample'))
                if z_value is None:
                    z_value = self.z_prior.sample(sample_shape, seed,
                                                  'z_sample')
                    # this reshape -> conv id -> reshape bullshit is here because
                    # some bug in Sonnet is keeping the output of a tfd distribution
                    # sample from going right to a linear
                    z_value = tf.reshape(z_value, [-1, 4, 4, 1])
                    z_value = tf.layers.conv2d(
                        z_value,
                        1,
                        1,
                        use_bias=False,
                        kernel_initializer=tf.ones_initializer,
                        padding='same',
                        trainable=False)
                    z_value = tf.reshape(z_value, sample_shape + [-1])

                return self._infer_x_hat(y_value, z_value).mean()
