import sonnet as snt
import sounds_deep.contrib.models.vae as vae
import tensorflow as tf

tfd = tf.contrib.distributions


class NamedLatentVAE(snt.AbstractModule):
    def __init__(self,
                 latent_dimension,
                 nv_latent_dimension,
                 nv_encoder_net,
                 encoder_net,
                 decoder_net,
                 prior_fn=vae.STD_GAUSSIAN_FN,
                 posterior_fn=vae.SOFTPLUS_GAUSSIAN_FN,
                 output_dist_fn=vae.BERNOULLI_FN,
                 name='named_latent_vae'):
        super(NamedLatentVAE, self).__init__(name=name)
        self._nv_encoder = nv_encoder_net
        self._encoder = encoder_net
        self._decoder = decoder_net
        self._latent_posterior_fn = posterior_fn
        self._output_dist_fn = output_dist_fn

        with self._enter_variable_scope():
            self._loc = snt.Sequential([
                snt.BatchFlatten(preserve_dims=2),
                snt.BatchApply(snt.Linear(latent_dimension))
            ])
            self._scale = snt.Sequential([
                snt.BatchFlatten(preserve_dims=2),
                snt.BatchApply(snt.Linear(latent_dimension))
            ])
            self._nv_logits = snt.Sequential(
                [snt.BatchFlatten(),
                 snt.Linear(nv_latent_dimension)])
            self.latent_prior = prior_fn(latent_dimension)

    def _build(self,
               unlabeled_input,
               labeled_input,
               hvar_labels,
               temperature,
               n_samples=3,
               analytic_kl=True):
        """data must be NHWC"""
        data = tf.concat([labeled_input, unlabeled_input], axis=0)

        # predict y
        nv_logits = self._nv_logits(self._nv_encoder(unlabeled_input))
        self.nv_latent_prior = tfd.ExpRelaxedOneHotCategorical(
            temperature, logits=tf.ones_like(nv_logits))
        self.nv_latent_posterior = tfd.ExpRelaxedOneHotCategorical(
            temperature, logits=nv_logits)
        nv_latent_posterior_sample = self.nv_latent_posterior.sample(n_samples)
        predicted_nv = tf.concat(
            [
                tf.tile(
                    tf.expand_dims(hvar_labels, axis=0), [n_samples, 1, 1]),
                tf.exp(nv_latent_posterior_sample)
            ],
            axis=1)

        # machine latent
        data_shape = data.get_shape().as_list()
        img_predicted_nv = tf.tile(
            tf.expand_dims(tf.expand_dims(predicted_nv, 2), 2),
            [1, 1, data_shape[1], data_shape[2], 1])
        z_encoder_input = tf.concat(
            [
                tf.tile(tf.expand_dims(data, 0), [n_samples, 1, 1, 1, 1]),
                img_predicted_nv
            ],
            axis=4)
        batch_encoder = snt.BatchApply(self._encoder)
        z_encoder_repr = batch_encoder(z_encoder_input)
        self.latent_posterior = self._latent_posterior_fn(
            self._loc(z_encoder_repr), self._scale(z_encoder_repr))

        # draw latent posterior sample
        latent_posterior_sample = self.latent_posterior.sample()
        joint_latent_posterior_sample = tf.concat(
            [predicted_nv, latent_posterior_sample], axis=2)

        # define output distribution
        sample_decoder = snt.BatchApply(self._decoder)
        output = sample_decoder(joint_latent_posterior_sample)
        self.output_distribution = tfd.Independent(
            self._output_dist_fn(output), reinterpreted_batch_ndims=3)

        distortion = -self.output_distribution.log_prob(data)
        if analytic_kl and n_samples == 1:
            rate = tfd.kl_divergence(self.latent_posterior, self.latent_prior)
        else:
            rate = (self.latent_posterior.log_prob(latent_posterior_sample) -
                    self.latent_prior.log_prob(latent_posterior_sample))
        nv_entropy = -tf.reduce_sum(tf.exp(nv_logits) * nv_logits, axis=-1)
        nv_rate = self.nv_latent_posterior.log_prob(
            nv_latent_posterior_sample) - self.nv_latent_prior.log_prob(
                nv_latent_posterior_sample)
        with tf.control_dependencies([  #tf.assert_positive(rate),
                tf.assert_positive(distortion)
        ]):
            elbo_local = -(rate + distortion)
            labeled_terms = nv_entropy + 0.8 * nv_rate

        self.distortion = distortion
        self.rate = rate
        self.prior_logp = self.latent_prior.log_prob(latent_posterior_sample)
        self.posterior_logp = self.latent_posterior.log_prob(
            latent_posterior_sample)
        self.nv_rate = nv_rate
        self.nv_prior_logp = self.nv_latent_prior.log_prob(
            nv_latent_posterior_sample)
        self.nv_posterior_logp = self.nv_latent_posterior.log_prob(
            nv_latent_posterior_sample)
        self.elbo = tf.reduce_mean(tf.reduce_logsumexp(
            elbo_local, axis=0)) + tf.reduce_mean(nv_entropy, axis=0)
        self.importance_weighted_elbo = tf.reduce_mean(
            tf.reduce_logsumexp(elbo_local, axis=0) -
            tf.log(tf.to_float(n_samples))) + tf.reduce_mean(labeled_terms)
        return output

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

                joint_prior_sample = tf.concat(
                    [tf.exp(nv_prior_sample), prior_sample], axis=-1)
                output = self._decoder(joint_prior_sample)
                return self._output_dist_fn(output).mean()
