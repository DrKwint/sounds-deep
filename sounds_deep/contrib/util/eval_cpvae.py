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


def evaluation_spacing(start_mu, start_sigma, spacing='sd', num_steps=5):
    if spacing == 'uniform':
        assert False, "This is unimplemented!"
    else:
        sds = np.linspace(-1 * num_steps, num_steps, num=(2 * num_steps + 1))
        #e.g. numsteps=3, sds=[-3,-2,1,0,1,2,3]
        intermediate_dim_values = np.full_like(sds, start_mu)
        for x in range(len(intermediate_dim_values)):
            intermediate_dim_values[x] += start_sigma * sds[x]

        return intermediate_dim_values
