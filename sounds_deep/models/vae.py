import numpy as np
import tensorflow as tf

from sounds_deep.models.core.simple_model import Model
from sounds_deep.util.distributions import (bernoulli_joint_log_likelihood,
                                            gaussian_sample_log_likelihood,
                                            std_gaussian_KL_divergence)
from sounds_deep.util.lazy_property import lazy_property
from sounds_deep.util.shaping import shape


class VAE(Model):
    def __init__(self, name, hyperparameter_dict):
        """
        Args:
            - x_in (Tensor)
            - latent_dim: dimensionality of the latent variable
            - encoder (snt.Module): module whose output should be
                [batch_size, 2*latent_dim]
            - decoder
        """
        super(VAE, self).__init__(
            name=name, hyperparameter_dict=hyperparameter_dict)
        self.latent_dim = hyperparameter_dict['latent_dim']
        self.encoder = hyperparameter_dict['encoder']
        self.decoder = hyperparameter_dict['decoder']

    def _build(self, x_in):
        """
        Adds Tensor Attributes:
            - z_mu
            - z_log_sigma
            - z_sample
            - re_x
        """
        self.x_in = x_in
        self.z_mu, self.z_log_sigma = tf.split(
            self.encoder(self.x_in), 2, axis=1)
        if shape(self.z_mu)[1] != self.latent_dim:
            pass  # complain and crash here

        # reparameterization trick
        self.z_sample = tf.add(self.z_mu,
                               tf.multiply(tf.exp(self.z_log_sigma), self.eps))

        # Alias for `AutoencoderModel` as VAE is stochastic and base AE isn't
        self.data_code = self.z_sample
        self.re_x = self.decoder(self.z_sample)

        self.losses
        return self

    @lazy_property
    def loss(self):
        # log p(x|z)
        self.reconstr_loss = bernoulli_joint_log_likelihood(
            self.x_in, self.re_x)

        # KL( q(z|X) || p(z) )
        self.kl_div = std_gaussian_KL_divergence(self.z_mu, self.z_log_sigma)

        elbo = tf.reduce_mean(self.reconstr_loss - self.kl_div)
        return -elbo

    @lazy_property
    def losses(self):
        return {
            'total': tf.reduce_mean(self.loss),
            'reconstruction': tf.reduce_mean(self.reconstr_loss),
            'latent': tf.reduce_mean(self.kl_div)
        }

    @lazy_property
    def eps(self):
        """ The gaussian noise for reparameterization"""
        return tf.random_normal([self.latent_dim], 0, 1, dtype=tf.float32)
