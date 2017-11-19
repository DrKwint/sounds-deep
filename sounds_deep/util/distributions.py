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

def discretized_logistic(mean, logscale, binsize=1 / 256.0, sample=None):
    scale = tf.exp(logscale)
    sample = (tf.floor(sample / binsize) * binsize - mean) / scale
    logp = tf.log(tf.sigmoid(sample + binsize / scale) - tf.sigmoid(sample) + 1e-7)

    if logp.shape.ndims == 4:
        logp = tf.reduce_sum(logp, [1, 2, 3])
    elif logp.shape.ndims == 2:
        logp = tf.reduce_sum(logp, 1)
    return logp

def std_gaussian_KL_divergence(mu, log_sigma):
    """ Analytic KL-div between N(mu, e^log_sigma) and N(0, 1) """
    sigma = tf.exp(log_sigma)
    return -0.5 * tf.reduce_sum(1 + tf.log(tf.square(sigma)) - tf.square(mu) - tf.square(sigma), 1)

class DiagonalGaussian(snt.AbstractModule):
    def __init__(self, mean, logvar):
        self._mean = mean
        self._logvar = logvar

    def _build(self):
        noise = tf.random_normal(tf.shape(self._mean))
        sample = mean + tf.exp(self._logvar) * noise
        return sample

    def log_prob(self, data):
        return diag_gaussian_log_likelihood(self._mean, self._logvar, data)
