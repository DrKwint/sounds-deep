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
    """
    Combine gaussian parameters with assumption that each gaussian is weighted
    equally

    Args
        mu: list of np.array
        sigma: list of np.array
    """
    mu_hat = np.mean(mu, axis=0)
    sigma_hat = np.mean(
        np.square(mu) + np.square(sigma), axis=0) - np.square(mu_hat)
    return mu_hat, sigma_hat


def starting_point(start_classes, c_means, c_stddev):
    #start_classes - np array of classes to average for starting point
    start_c_mus = c_means[start_classes]
    start_c_sigmas = np.sqrt(c_stddev[start_classes])
    convolved_start_mu, convolved_start_sigma = convolve_gaussians(
        start_c_mus, start_c_sigmas)
    return convolved_start_mu, convolved_start_sigma
    #Gives the starting "average digit" of the classees given in start_classes


def evaluation_spacing(mu,
                       sigma,
                       active_dims=None,
                       target_mu=None,
                       num_steps=3):
    """
    If target_mu is None, standard deviations are used, otherwise steps are
    uniform between mu and target
    """
    # generate integer steps
    idxs = np.linspace(-1 * num_steps, num_steps, num=(2 * num_steps + 1))

    # calculate the delta each step should take
    step_delta = (
        target_mu - mu) / num_steps if target_mu is not None else sigma

    # zero for non-active dimensions
    mask = np.zeros_like(step_delta)
    if active_dims:
        if type(active_dims) is not list: active_dims = [active_dims]
        for i in active_dims:
            mask[i] = 1
    else:
        mask = np.ones_like(mask)
    step_delta *= mask
    # calculate total displacement for each number of steps
    displacement = np.stack([step_delta * i for i in idxs])

    # apply displacement
    step_vals = mu + displacement

    return step_vals


def two_leaf_visualization(c_means,
                           c_sds,
                           classes,
                           active_dims=None,
                           num_steps=3):
    assert len(classes) == 2
    #Start from the average of two classes, varry active_dim(s).
    if active_dims is not None:
        initial_mu, initial_sigma = starting_point(classes, c_means, c_sds)
        target_mu = None
    else:
        initial_mu = c_means[classes[0]]
        target_mu = c_means[classes[1]]
        initial_sigma = None
    latent_code = evaluation_spacing(initial_mu, initial_sigma, active_dims,
                                     target_mu, num_steps)
    if active_dims is None:
        base_filename = 'AllDim_'
    elif active_dims.size == 1:
        base_filename = 'Dim_{}'.format(active_dims)
    else:
        base_filename = 'Dims' + ['_{}'.format(i) for i in active_dims]
    base_filename += '_Classes'
    for i in classes:
        base_filename += '_{}'.format(i)
    #base_filename += ['_{}'.format(i) for i in classes]

    sds = np.linspace(-1 * num_steps, num_steps, 2 * num_steps + 1)

    filenames = [base_filename + '_SDs_{}'.format(i) for i in sds]
    print(filenames)
    return latent_code, filenames


def mean_digit_dim_visualization(c_means, c_sds, active_dims=None,
                                 num_steps=3):
    #Start from "mean digit", show impact of varrying a single dimension in latent space.
    classes = np.arange(len(c_means))
    filenames = [
        'all_class_dim{}_{}'.format(active_dims, i)
        for i in range(2 * num_steps + 1)
    ]
    all_class_mu, all_class_sigma = starting_point(classes, c_means, c_sds)
    latent_code = evaluation_spacing(all_class_mu, all_class_sigma,
                                     active_dims, num_steps)
    return latent_code, filenames


def instance_to_class_visualization(instance_mu,
                                    instance_sigma,
                                    c_means,
                                    c_sds,
                                    target_c,
                                    active_dim=None,
                                    num_steps=3):
    #Celebrity baby from instance to actual class.
    target_mu = c_means[target_c]
    return evaluation_spacing(instance_mu, instance_sigma, active_dim,
                              target_mu, num_steps)
