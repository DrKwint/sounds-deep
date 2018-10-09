from __future__ import absolute_import, division, print_function

import argparse

import numpy as np
import tensorflow as tf

import sounds_deep.contrib.data.data as data
import sounds_deep.contrib.models.cpvae as cpvae
import sounds_deep.contrib.models.vae as vae
import sounds_deep.contrib.parameterized_distributions.discretized_logistic as discretized_logistic
import sounds_deep.contrib.util.plot as plot
import sounds_deep.contrib.util.scaling as scaling
import sounds_deep.contrib.util.util as util


def convolve_gaussians(mu, sigma):
    #Takes np.arrays as input, assumes the distributions are equally weighted.
    mu_hat = np.mean(mu)
    sigma_hat = np.mean(np.square(mu) + np.square(sigma)) - np.square(mu_hat)
    return mu_hat, sigma_hat


def starting_point(start_classes, c_means, c_stddev):
    #start_classes - np array of classes to average for starting point
    start_c_mus = c_means[start_classes]
    start_c_sigmas = np.sqrt(c_stddev[start_classes])
    convolved_start_mu, convolved_start_sigma = convolve_gaussians(
        start_c_mus, start_c_sigmas)
    return convolved_start_mu, convolved_start_sigma
    #Gives the starting "average digit" of the classees given in start_classes


def evaluation_spacing(mu, sigma, active_dims, target_mu=None, num_steps=5):
    """
    If target_mu is None, standard deviations are used, otherwise steps are
    uniform between mu and target
    """
    # generate integer steps
    idxs = np.linspace(-1 * num_steps, num_steps, num=(2 * num_steps + 1))

    # calculate the delta each step should take
    step_delta = (target_mu - mu) / num_steps if target_mu else sigma

    # zero for non-active dimensions
    mask = np.zeros_like(step_delta)
    for i in active_dims:
        mask[i] = 1
    step_delta *= mask

    # calculate total displacement for each number of steps
    displacement = np.stack([step_delta * i for i in idxs])

    # apply displacement
    step_vals = mu + displacement

    return step_vals
