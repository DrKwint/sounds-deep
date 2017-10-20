import numpy as np
import tensorflow as tf
import sonnet as snt

from sounds_deep.util.shaping import flatten

def bernoulli_joint_log_likelihood(x_in, re_x):
    '''
    Bernoulli log-likelihood reconstruction loss that is used in the VAE
    Args must be in range [0,1]
    '''
    x_in = flatten(x_in)
    re_x = tf.nn.relu(flatten(re_x))
    return tf.reduce_sum(x_in * tf.log(1e-10 + re_x) + (
        (1 - x_in) * tf.log(1e-10 + 1 - re_x)), 1)


def gaussian_sample_log_likelihood(sample, mu, sigma):
    log2pi = tf.constant(np.log(2.0 * np.pi), dtype=tf.float32)
    lg_sigma_sq = tf.log(tf.square(sigma))
    other_term = tf.square(tf.divide(tf.subtract(sample, mu), sigma))
    term_sum = tf.reduce_sum(
        tf.add(log2pi, tf.add(lg_sigma_sq, other_term)), 1)
    log_likelihood = tf.multiply(0.5, term_sum)
    return log_likelihood


def std_gaussian_KL_divergence(mu, log_sigma):
    """ Analytic KL-div between N(mu, e^log_sigma) and N(0, 1) """
    return -0.5 * tf.reduce_sum(
        1 + log_sigma - tf.square(mu) - tf.exp(log_sigma), 1)

class DiagonalGaussian(snt.AbstractModule):
    def __init__(self, mean, logvar):
        self._mean = mean
        self._logvar = logvar

    def _build(self):
        noise = tf.random_normal(tf.shape(self._mean))
        sample = mean + tf.exp(self._logvar) * noise
        return sample

    # def log_prob(self, sample):
    #     return -0.5 * (np.log(2 * np.pi) + self._logvar + (tf.square(sample - self._mean) / tf.exp(self._logvar))
