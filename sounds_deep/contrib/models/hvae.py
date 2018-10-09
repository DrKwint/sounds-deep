import tensorflow as tf
import sonnet as snt
import tensorflow.contrib.distributions as tfd

import sounds_deep.contrib.util as util

STD_GAUSSIAN_FN = lambda latent_dimension: tfd.MultivariateNormalDiag(
    loc=tf.zeros(latent_dimension), scale_diag=tf.ones(latent_dimension))
SOFTPLUS_GAUSSIAN_FN = lambda loc, scale: tfd.MultivariateNormalDiagWithSoftplusScale(
            loc=loc, scale_diag=scale)


class HVAE(snt.AbstractModule):
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
                 hvar_shape,
                 prior_fn=STD_GAUSSIAN_FN,
                 posterior_fn=SOFTPLUS_GAUSSIAN_FN,
                 temperature=1.0,
                 name='vae'):
        """prior should be a callable taking an integer for latent dimensionality and return a TF Distribution"""
        super(HVAE, self).__init__(name=name)
        self._encoder = encoder_net
        self._decoder = decoder_net
        self._latent_posterior_fn = posterior_fn
        self._temperature = temperature

        with self._enter_variable_scope():
            self._loc = snt.Linear(latent_dimension)
            self._scale = snt.Linear(latent_dimension)
            self._hvar = snt.Linear(hvar_shape)
            # Consider using a parameterized GMM prior learned with backprop
            self.latent_prior = prior_fn(latent_dimension)

    def _build(self, inputs, hvar_labels, n_samples=10, analytic_kl=True):
        datum_shape = inputs.get_shape().as_list()[1:]
        enc_repr = self._encoder(inputs)

        self.hvar_prior = tfd.ExpRelaxedOneHotCategorical(
            temperature=self._temperature, logits=hvar_labels)
        self.hvar_posterior = tfd.ExpRelaxedOneHotCategorical(
            temperature=self._temperature, logits=self._hvar(enc_repr))
        hvar_sample_shape = [n_samples
                             ] + self.hvar_posterior.batch_shape.as_list(
                             ) + self.hvar_posterior.event_shape.as_list()
        hvar_sample = tf.reshape(
            self.hvar_posterior.sample(n_samples), hvar_sample_shape)

        self.latent_posterior = self._latent_posterior_fn(
            self._loc(enc_repr), self._scale(enc_repr))
        latent_posterior_sample = self.latent_posterior.sample(n_samples)

        joint_sample = tf.concat([hvar_sample, latent_posterior_sample],
                                 axis=-1)

        sample_decoder = snt.BatchApply(self._decoder)
        self.output_distribution = tfd.Independent(
            tfd.Bernoulli(logits=sample_decoder(joint_sample)),
            reinterpreted_batch_ndims=len(datum_shape))

        distortion = -self.output_distribution.log_prob(inputs)
        if analytic_kl and n_samples == 1:
            rate = tfd.kl_divergence(self.latent_posterior, self.latent_prior)
        else:
            rate = (self.latent_posterior.log_prob(latent_posterior_sample) -
                    self.latent_prior.log_prob(latent_posterior_sample))
        hrate = self.hvar_posterior.log_prob(
            hvar_sample) - self.hvar_prior.log_prob(hvar_sample)
        # hrate = tf.Print(hrate, [temperature])
        # hrate = tf.Print(hrate, [hvar_sample], summarize=10)
        # hrate = tf.Print(hrate, [self.hvar_posterior.log_prob(hvar_sample)])
        # hrate = tf.Print(hrate, [self.hvar_prior.log_prob(hvar_sample)])
        # hrate = tf.Print(hrate, [hrate], summarize=10)
        elbo_local = -(rate + hrate + distortion)
        self.elbo = tf.reduce_mean(elbo_local)
        self.importance_weighted_elbo = tf.reduce_mean(
            tf.reduce_logsumexp(elbo_local, axis=0) -
            tf.log(tf.to_float(n_samples)))

        self.hvar_sample = tf.exp(tf.split(hvar_sample, n_samples)[0])
        self.hvar_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=hvar_labels, logits=tf.split(hvar_sample, n_samples)[0])
        self.hvar_labels = hvar_labels
        self.distortion = distortion
        self.rate = rate
        self.hrate = hrate

    def sample(self):
        hvar_prior = tfd.ExpRelaxedOneHotCategorical(
            temperature=self._temperature, logits=tf.ones(10))
        hvar_sample = hvar_prior.sample()
        hvar_sample = tf.Print(
            hvar_sample, [tf.exp(hvar_sample)], summarize=10)
        latent_posterior_sample = self.latent_prior.sample()
        joint_sample = tf.concat([hvar_sample, latent_posterior_sample],
                                 axis=-1)
        joint_sample = tf.expand_dims(joint_sample, 0)
        return self._decoder(joint_sample)
