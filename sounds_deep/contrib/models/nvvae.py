import sonnet as snt
import sounds_deep.contrib.models.vae as vae
import tensorflow as tf

tfd = tf.contrib.distributions


class NamedLatentVAE(snt.AbstractModule):
    def __init__(self,
                 latent_dimension,
                 nv_latent_dimension,
                 encoder_net,
                 decoder_net,
                 prior_fn=vae.STD_GAUSSIAN_FN,
                 posterior_fn=vae.SOFTPLUS_GAUSSIAN_FN,
                 output_dist_fn=vae.BERNOULLI_FN,
                 name='named_latent_vae'):
        super(NamedLatentVAE, self).__init__(name=name)
        self._encoder = encoder_net
        self._decoder = decoder_net
        self._latent_posterior_fn = posterior_fn
        self._output_dist_fn = output_dist_fn

        with self._enter_variable_scope():
            self._loc = snt.Linear(latent_dimension)
            self._scale = snt.Linear(latent_dimension)
            self._nv_logits = snt.Linear(nv_latent_dimension)
            self.latent_prior = prior_fn(latent_dimension)

    def _build(self,
               inputs,
               hvar_labels,
               temperature,
               n_samples=1,
               analytic_kl=True):
        x = inputs
        encoder_repr = self._encoder(x)

        # define latent distributions
        self.latent_posterior = self._latent_posterior_fn(
            self._loc(encoder_repr), self._scale(encoder_repr))
        self.nv_latent_prior = tfd.ExpRelaxedOneHotCategorical(
            temperature, logits=hvar_labels)
        self.nv_latent_posterior = tfd.ExpRelaxedOneHotCategorical(
            temperature, logits=self._nv_logits(encoder_repr))

        # draw latent posterior sample
        latent_posterior_sample = self.latent_posterior.sample(n_samples)
        nv_latent_posterior_sample = self.nv_latent_posterior.sample(n_samples)
        joint_latent_posterior_sample = tf.concat(
            [tf.exp(nv_latent_posterior_sample), latent_posterior_sample],
            axis=2)

        # define output distribution
        sample_decoder = snt.BatchApply(self._decoder)
        output = sample_decoder(joint_latent_posterior_sample)
        self.output_distribution = tfd.Independent(
            self._output_dist_fn(output), reinterpreted_batch_ndims=3)

        distortion = -self.output_distribution.log_prob(x)
        if analytic_kl and n_samples == 1:
            rate = tfd.kl_divergence(self.latent_posterior, self.latent_prior)
        else:
            rate = (self.latent_posterior.log_prob(latent_posterior_sample) -
                    self.latent_prior.log_prob(latent_posterior_sample))
        nv_rate = self.nv_latent_posterior.log_prob(
            nv_latent_posterior_sample) - self.nv_latent_prior.log_prob(
                nv_latent_posterior_sample)
        with tf.control_dependencies([
                tf.assert_positive(rate),
                tf.assert_positive(distortion)
                # tf.assert_positive(nv_rate)
        ]):
            elbo_local = -(rate + 16 * nv_rate + distortion)

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
        self.elbo = tf.reduce_mean(tf.reduce_logsumexp(elbo_local, axis=0))
        self.importance_weighted_elbo = tf.reduce_mean(
            tf.reduce_logsumexp(elbo_local, axis=0) -
            tf.log(tf.to_float(n_samples)))
        return output

    def sample(self,
               sample_shape=(),
               seed=None,
               temperature=0.01,
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
                prior_sample = self.latent_prior.sample(
                    sample_shape, seed, 'prior_sample')
                prior_sample = tf.reshape(prior_sample, [-1, 4, 4, 1])
                prior_sample = tf.layers.conv2d(
                    prior_sample,
                    1,
                    1,
                    use_bias=False,
                    kernel_initializer=tf.ones_initializer,
                    padding='same',
                    trainable=False)
                prior_sample = tf.reshape(prior_sample, [1, -1])
                nv_prior_sample = tfd.ExpRelaxedOneHotCategorical(
                    temperature=temperature, logits=tf.ones(10)).sample(
                        sample_shape, seed, 'nv_prior_sample')
                joint_prior_sample = tf.concat(
                    [tf.exp(nv_prior_sample), prior_sample], axis=-1)
                output = self._decoder(joint_prior_sample)
                return self._output_dist_fn(output).mean()
