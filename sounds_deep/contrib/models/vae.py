import tensorflow as tf
import sonnet as snt

tfd = tf.contrib.distributions
tfb = tfd.bijectors

STD_GAUSSIAN_FN = lambda latent_dimension: tfd.MultivariateNormalDiag(
    loc=tf.zeros(latent_dimension), scale_diag=tf.ones(latent_dimension))
SOFTPLUS_GAUSSIAN_FN = lambda loc, scale: tfd.MultivariateNormalDiagWithSoftplusScale(
            loc=loc, scale_diag=scale)
BERNOULLI_FN = lambda loc: tfd.Bernoulli(loc)


class VAE(snt.AbstractModule):
    """Variational Autoencoder (VAE) as first introduced by Kingma and Welling (2014)

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
                 encoder_net,
                 decoder_net,
                 prior_fn=STD_GAUSSIAN_FN,
                 posterior_fn=SOFTPLUS_GAUSSIAN_FN,
                 output_dist_fn=BERNOULLI_FN,
                 name='vae'):
        """Initializes a Variational Autoencoder (VAE) as a callable.

        The expected range of input pixels (this class mostly expects image inputs) is [0,1].

        Example:
            A small VAE::

                import sonnet as snt
                import tensorflow as tf

                import sounds_deep.contrib.distributions.discretized_logistic as discretized_logistic
                import sounds_deep.contrib.models.vae as vae

                latent_dimension = 50
                encoder_module = snt.Sequential([
                    snt.nets.ConvNet2D([16, 32, 64], [3], [1], [snt.SAME]),
                    tf.keras.layers.Flatten()
                ])
                decoder_module = snt.Sequential([
                    snt.Linear(49), lambda x: tf.reshape(x, [-1, 7, 7, 1]),
                    snt.nets.ConvNet2DTranspose([32, 16, 1], [(14, 14), (28,28), (28,28)],
                                                [3], [2, 2, 1], [snt.SAME])
                ])
                model = vae.VAE(
                    latent_dimension,
                    encoder_module,
                    decoder_module,
                    output_dist_fn=discretized_logistic.DiscretizedLogistic)

        Args:
            latent_dimension (int): Dimension of the latent variable.
            encoder_net (snt.Module): Encoder mapping from rank 4 input to rank 2 output.
            decoder_net (Tensor -> Tensor): Decoder mapping from a rank 2 input to rank 4 output.
            prior_fn (int -> tfd.Distribution): Callable which takes an integral dimension size.
            posterior_fn (Tensor -> Tensor -> tfd.Distribution): Callable which takes location and
                                                                 scale and returns a tfd distribution.
            output_dist_fn (Tensor -> tfd.Distribution): Callable from loc to a tfd distribution.
        """
        super(VAE, self).__init__(name=name)
        self._encoder = encoder_net
        self._decoder = decoder_net
        self._latent_posterior_fn = posterior_fn
        self._output_dist_fn = output_dist_fn

        with self._enter_variable_scope():
            self._loc = snt.Linear(latent_dimension)
            self._scale = snt.Linear(latent_dimension)
            self.latent_prior = prior_fn(latent_dimension)

    def _build(self, inputs, n_samples=1, analytic_kl=True):
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
        x = inputs
        encoder_repr = self._encoder(x)
        self.latent_posterior = self._latent_posterior_fn(
            self._loc(encoder_repr), self._scale(encoder_repr))
        latent_posterior_sample = self.latent_posterior.sample(n_samples)
        sample_decoder = snt.BatchApply(self._decoder)
        output = sample_decoder(latent_posterior_sample)
        self.output_distribution = tfd.Independent(
            self._output_dist_fn(output), reinterpreted_batch_ndims=3)

        distortion = -self.output_distribution.log_prob(x)
        if analytic_kl and n_samples == 1:
            rate = tfd.kl_divergence(self.latent_posterior, self.latent_prior)
        else:
            rate = (self.latent_posterior.log_prob(latent_posterior_sample) -
                    self.latent_prior.log_prob(latent_posterior_sample))
        with tf.control_dependencies(
            [tf.assert_positive(rate),
             tf.assert_positive(distortion)]):
            elbo_local = -(rate + distortion)

        self.distortion = distortion
        self.rate = rate
        self.prior_logp = self.latent_prior.log_prob(latent_posterior_sample)
        self.posterior_logp = self.latent_posterior.log_prob(
            latent_posterior_sample)
        self.elbo = tf.reduce_mean(tf.reduce_logsumexp(elbo_local, axis=0))
        self.importance_weighted_elbo = tf.reduce_mean(
            tf.reduce_logsumexp(elbo_local, axis=0) -
            tf.log(tf.to_float(n_samples)))
        return output

    def sample(self, sample_shape=(), seed=None, name='sample'):
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
                output = self._decoder(prior_sample)
                return self._output_dist_fn(output).mean()
