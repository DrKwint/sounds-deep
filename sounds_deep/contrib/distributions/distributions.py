import numpy as np
import tensorflow as tf

import sonnet as snt


def bernoulli_joint_log_likelihood(x_in, re_x):
    '''
    Bernoulli log-likelihood reconstruction loss that is used in the VAE
    Args must be in range [0,1]
    '''
    x_in = flatten(x_in)
    re_x = tf.nn.relu(flatten(re_x))
    return tf.reduce_sum(x_in * tf.log(1e-10 + re_x) +
                         ((1 - x_in) * tf.log(1e-10 + 1 - re_x)), 1)


def discretized_logistic(mean, logscale, binsize=1 / 256.0, sample=None):
    scale = tf.exp(logscale)
    sample = (tf.floor(sample / binsize) * binsize - mean) / scale
    logp = tf.log(
        tf.sigmoid(sample + binsize / scale) - tf.sigmoid(sample) + 1e-7)

    if logp.shape.ndims == 4:
        logp = tf.reduce_sum(logp, [1, 2, 3])
    elif logp.shape.ndims == 2:
        logp = tf.reduce_sum(logp, 1)
    return logp


def std_gaussian_KL_divergence(mu, sigma):
    """ Analytic KL-div between N(mu, sigma) and N(0, 1) """
    return -0.5 * tf.reduce_sum(
        1 + tf.log(tf.square(sigma)) - tf.square(mu) - tf.square(sigma), 1)


def flatten(tensor):
    """
    Flattens a tensor along all non-batch dimensions.
    This is correctly a NOP if the input is already flat.
    """
    if len(tensor.get_shape()) == 2:
        return tensor
    else:
        return tf.layers.Flatten()(tensor)


def expected_bernoulli_loglike(y_binary,
                               logits,
                               r_nk=None,
                               name='bernoulli_expct_loglike'):
    # E[log p(y|x)]
    with tf.name_scope(name):
        if r_nk is None:
            N, S, D = logits.get_shape().as_list()
            assert y_binary.get_shape() == (N, D)
        else:
            N, K, S, D = logits.get_shape().as_list()
            assert y_binary.get_shape() == (N, D)
            assert r_nk.get_shape() == (N, K)

        # add dimensions for K and S
        y_binary = tf.expand_dims(y_binary, 1)  # N, 1, D
        if r_nk is not None:
            y_binary = tf.expand_dims(y_binary, 1)  # N, 1, 1, D

        pixel_log_probs = -tf.log(
            tf.add(1., tf.exp(tf.multiply(-logits, y_binary))))
        sample_log_probs = tf.reduce_sum(
            pixel_log_probs, axis=-1)  # sum over pixels
        img_log_probs = tf.reduce_mean(
            sample_log_probs, axis=-1)  # average over samples

        if r_nk is not None:
            img_log_probs = tf.reduce_sum(
                tf.multiply(r_nk, img_log_probs),
                axis=1)  # average over components

        return tf.reduce_sum(
            img_log_probs, name='expct_bernoulli_loglik')  # sum over minibatch


def expected_diagonal_gaussian_loglike(y,
                                       means,
                                       vars,
                                       weights=None,
                                       name='diag_gauss_expct'):
    """
    computes expected diagonal log-likelihood SUM_{n=1} E_{q(z)}[log N(x_n|mu(z), sigma(z))]
    Args:
        y: data
        means: predicted means; shape (size_minibatch, nb_samples, dims) or (size_minimbatch, nb_comps, nb_samps, dims)
        vars: predicted variances; shape is same as for means
        weights: None or matrix of shape (N, K) containing normalized weights
    Returns:
    """
    # todo refactor (merge the ifs)
    with tf.name_scope(name):
        if weights is None:
            # required dimension: size_minibatch, nb_samples, dims
            means = means if len(means.get_shape()) == 3 else tf.expand_dims(
                means, axis=1)
            vars = vars if len(vars.get_shape()) == 3 else tf.expand_dims(
                vars, axis=1)
            M, S, L = means.get_shape()
            assert y.get_shape() == (M, L)

            sample_mean = tf.reduce_sum(
                tf.pow(tf.expand_dims(y, axis=1) - means, 2) /
                vars) + tf.reduce_sum(tf.log(vars))

            S = tf.constant(int(S), dtype=tf.float32, name='number_samples')
            M = tf.constant(int(M), dtype=tf.float32, name='size_minibatch')
            L = tf.constant(int(L), dtype=tf.float32, name='latent_dimensions')
            pi = tf.constant(np.pi, dtype=tf.float32, name='pi')

            sample_mean /= S
            loglik = -1 / 2 * sample_mean - M * L / 2. * tf.log(2. * pi)

        else:
            M, K, S, L = means.get_shape()
            assert vars.get_shape() == means.get_shape()
            print(weights)
            print(means.get_shape())
            assert weights.get_shape() == (M, K)

            # adjust y's shape (add component and sample dimensions)
            y = tf.expand_dims(tf.expand_dims(y, axis=1), axis=1)

            sample_mean = tf.einsum(
                'nksd,nk->',
                tf.square(y - means) / vars + tf.log(vars + 1e-8), weights)

            M = tf.constant(int(M), dtype=tf.float32, name='size_minibatch')
            S = tf.constant(int(S), dtype=tf.float32, name='number_samples')
            L = tf.constant(int(L), dtype=tf.float32, name='latent_dimensions')
            pi = tf.constant(np.pi, dtype=tf.float32, name='pi')

            sample_mean /= S
            loglik = -1 / 2 * sample_mean - M * L / 2. * tf.log(2. * pi)

        return tf.identity(loglik, name='expct_gaussian_loglik')
