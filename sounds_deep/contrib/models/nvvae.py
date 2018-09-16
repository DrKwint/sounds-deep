import sonnet as snt
import sounds_deep.contrib.models.vae as vae
import tensorflow as tf
import sounds_deep.contrib.util.util as util

tfd = tf.contrib.distributions


class NamedLatentVAE(snt.AbstractModule):
    def __init__(self,
                 z_shape,
                 y_shape,
                 y_net,
                 encoder_net,
                 decoder_net,
                 prior_fn=vae.STD_GAUSSIAN_FN,
                 posterior_fn=vae.SOFTPLUS_GAUSSIAN_FN,
                 output_dist_fn=vae.BERNOULLI_FN,
                 name='named_latent_vae'):
        super(NamedLatentVAE, self).__init__(name=name)
        self._y_net = y_net
        self._z_net = encoder_net
        self._x_hat_net = decoder_net
        self._z_posterior_fn = posterior_fn
        self._output_dist_fn = output_dist_fn

        with self._enter_variable_scope():
            self._loc = snt.Linear(z_shape)
            self._scale = snt.Linear(z_shape)
            self._y_logits = snt.Linear(y_shape)
            self.latent_prior = prior_fn(z_shape)

    def _build(self,
               unlabeled_input,
               labeled_input,
               hvar_labels,
               y_posterior_temperature,
               classification_loss_coeff=0.8,
               n_samples=1,
               analytic_kl=True):
        """data must be NHWC"""
        data = tf.concat([labeled_input, unlabeled_input], axis=0)

        # predict y
        y_posterior_labeled = self.infer_y_posterior(labeled_input,
                                                     y_posterior_temperature)
        y_posterior_sample_labeled = y_posterior_labeled.sample(n_samples)
        y_posterior_unlabeled = self.infer_y_posterior(
            unlabeled_input, y_posterior_temperature)
        y_posterior_sample_unlabeled = y_posterior_unlabeled.sample(n_samples)
        self.y_posterior_sample_unlabeled = y_posterior_sample_unlabeled 

        nv_predicted = tf.concat(
            [
                tf.tile(
                    tf.expand_dims(hvar_labels, axis=0), [n_samples, 1, 1]),
                tf.exp(y_posterior_sample_unlabeled)
            ],
            axis=1)
        self.nv_latent_prior = tfd.ExpRelaxedOneHotCategorical(
            y_posterior_temperature, logits=tf.ones_like(nv_predicted))

        # machine latent
        self.latent_posterior = self.infer_z_posterior(data, nv_predicted)

        # draw latent posterior sample
        latent_posterior_sample = self.latent_posterior.sample()

        # define output distribution
        x_hat = self._infer_x_hat(nv_predicted, latent_posterior_sample)

        # loss calculation
        # supervised:
        distortion = -x_hat.log_prob(data)
        rate = (self.latent_posterior.log_prob(latent_posterior_sample) -
                self.latent_prior.log_prob(latent_posterior_sample) -
                self.nv_latent_prior.log_prob(nv_predicted))
        supervised_distortion, unsupervised_distortion = tf.split(
            distortion, 2, axis=1)
        supervised_rate, unsupervised_rate = tf.split(rate, 2, axis=1)
        nv_logits = y_posterior_unlabeled.parameters['logits']
        nv_entropy = -tf.reduce_sum(tf.exp(nv_logits) * nv_logits, axis=-1)
        nv_log_prob = tf.reduce_sum(
            hvar_labels * y_posterior_sample_labeled, axis=-1)

        supervised_local_elbo = -(supervised_distortion + supervised_rate)
        unsupervised_local_elbo = -(
            unsupervised_distortion + unsupervised_rate) + nv_entropy

        elbo_local = supervised_local_elbo + unsupervised_local_elbo + classification_loss_coeff * nv_log_prob

        self.distortion = distortion
        self.rate = rate
        self.nv_entropy = nv_entropy
        self.nv_log_prob = nv_log_prob

        self.prior_logp = self.latent_prior.log_prob(latent_posterior_sample)
        self.posterior_logp = self.latent_posterior.log_prob(
            latent_posterior_sample)
        self.nv_prior_logp = self.nv_latent_prior.log_prob(nv_predicted)
        self.nv_posterior_logp = y_posterior_unlabeled.log_prob(
            y_posterior_sample_unlabeled)
        self.elbo = tf.reduce_mean(tf.reduce_logsumexp(elbo_local, axis=0))

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
            self._output_dist_fn(output), reinterpreted_batch_ndims=3)

    def infer_y_posterior(self, x, temperature):
        logits = self._y_logits(snt.BatchFlatten()(self._y_net(x)))
        return tfd.ExpRelaxedOneHotCategorical(temperature, logits=logits)

    def infer_z_posterior(self, x, y):
        """x should be rank 4 and y should be rank 2 or 3"""
        x_shape = util.int_shape(x)
        y_shape = util.int_shape(y)
        y_channel = tf.tile(
            tf.expand_dims(tf.expand_dims(y, 2), 2),
            [1, 1, x_shape[1], x_shape[2], 1])
        z_encoder_input = tf.concat(
            [
                tf.tile(tf.expand_dims(x, 0), [y_shape[0], 1, 1, 1, 1]),
                y_channel
            ],
            axis=4)
        z_repr = snt.BatchApply(self._z_net)(z_encoder_input)
        z_repr = snt.BatchFlatten(preserve_dims=2)(z_repr)
        batch_loc = snt.BatchApply(self._loc)
        batch_scale = snt.BatchApply(self._scale)
        return self._z_posterior_fn(batch_loc(z_repr), batch_scale(z_repr))

    def sample(self,
               sample_shape=(),
               seed=None,
               temperature=0.01,
               prior_sample=None,
               nv_prior_sample=None,
               name='sample'):
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
        if sample_shape == (): sample_shape = 1
        with self._enter_variable_scope():
            with tf.variable_scope(name):

                if prior_sample is None:
                    prior_sample = self.latent_prior.sample(
                        sample_shape, seed, 'prior_sample')
                    # this reshape -> conv id -> reshape bullshit is here because some bug in Sonnet
                    # is keeping the output of a tfd distribution sample from going right to a linear
                    prior_sample = tf.reshape(prior_sample, [-1, 4, 4, 1])
                    prior_sample = tf.layers.conv2d(
                        prior_sample,
                        1,
                        1,
                        use_bias=False,
                        kernel_initializer=tf.ones_initializer,
                        padding='same',
                        trainable=False)
                    prior_sample = tf.reshape(prior_sample,
                                              sample_shape + [-1])

                if nv_prior_sample is None:
                    nv_prior_sample = tfd.ExpRelaxedOneHotCategorical(
                        temperature=temperature, logits=tf.ones(10)).sample(
                            sample_shape, seed, 'nv_prior_sample')

                return self._infer_x_hat(nv_prior_sample, prior_sample).mean()
