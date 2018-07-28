import tensorflow as tf
import sonnet as snt
import tensorflow.contrib.distributions as tfd

STD_GAUSSIAN_FN = lambda latent_dimension: tfd.MultivariateNormalDiag(
    loc=tf.zeros(latent_dimension), scale_diag=tf.ones(latent_dimension))
SOFTPLUS_GAUSSIAN_FN = lambda loc, scale: tfd.MultivariateNormalDiagWithSoftplusScale(
            loc=loc, scale_diag=scale)


class VAE(snt.AbstractModule):
    """
    Members:
      latent_prior: tfd Distribution
      latent_posterior: tfd Distribution
      output_distribution: tfd Distribution
      elbo: Tensor ()
      importance_weighted_elbo: Tensor ()
    """

    def __init__(self,
                 latent_dimension,
                 encoder_net,
                 decoder_net,
                 prior_fn=STD_GAUSSIAN_FN,
                 posterior_fn=SOFTPLUS_GAUSSIAN_FN,
                 name='vae'):
        """prior should be a callable taking an integer for latent dimensionality and return a TF Distribution"""
        super(VAE, self).__init__(name=name)
        self._encoder = encoder_net
        self._decoder = decoder_net
        self._latent_posterior_fn = posterior_fn

        with self._enter_variable_scope():
            self._loc = snt.Linear(latent_dimension)
            self._scale = snt.Linear(latent_dimension)
            # Consider using a parameterized GMM prior learned with backprop
            self.latent_prior = prior_fn(latent_dimension)

    def _build(self, inputs, n_samples=10, analytic_kl=True):
        datum_shape = inputs.get_shape().as_list()[1:]
        enc_repr = self._encoder(inputs)
        self.latent_posterior = self._latent_posterior_fn(
            self._loc(enc_repr), self._scale(enc_repr))
        latent_posterior_sample = self.latent_posterior.sample(n_samples)
        sample_decoder = snt.BatchApply(self._decoder)
        self.output_distribution = tfd.Independent(
            tfd.Bernoulli(logits=sample_decoder(latent_posterior_sample)),
            reinterpreted_batch_ndims=len(datum_shape))

        distortion = -self.output_distribution.log_prob(inputs)
        if analytic_kl:
            rate = tfd.kl_divergence(self.latent_posterior, self.latent_prior)
        else:
            rate = (self.latent_posterior.log_prob(latent_posterior_sample) -
                    self.latent_prior.log_prob(latent_posterior_sample))
        elbo_local = -(rate + distortion)
        self.elbo = tf.reduce_mean(elbo_local)
        self.importance_weighted_elbo = tf.reduce_mean(
            tf.reduce_logsumexp(elbo_local, axis=0) -
            tf.log(tf.to_float(n_samples)))
