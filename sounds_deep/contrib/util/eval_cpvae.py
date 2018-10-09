from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf

import sounds_deep.contrib.data.data as data
import sounds_deep.contrib.util.scaling as scaling
import sounds_deep.contrib.util.util as util
import sounds_deep.contrib.models.cpvae as cpvae
import sounds_deep.contrib.models.vae as vae
import sounds_deep.contrib.parameterized_distributions.discretized_logistic as discretized_logistic
import sounds_deep.contrib.util.plot as plot

# parser = argparse.ArgumentParser(description='Evaluate using a trained VAE model.')
# #parser.add_argument('--batch_size', type=int, default=32) Template
# parser.add_argument("--dataset", type=str, default='mnist', help="Current options are mnist and cifar.")
# parser.add_argument("--gendim", type=int)
# parser.add_argument("--genval", nargs="*", type=float)
# parser.add_argument("--startclass", nargs="*", type=int, help="Single input for one class, multiple for an average of the given classes.")
# parser.add_argument("--target", type=int)
# parser.add_argument("--numsteps", nargs="*", type=int, default=3, help="Number of images generated to each side of the mean.")
# parser.add_argument("--spacing", type=str, default='sd')
# parser.add_argument("--model_dir", type=str)
# parser.add_argument("--data_dir", type=str, default='./data/')

# args = parser.parse_args()


def convolve_gaussians(mu, sigma):
  #Takes np.arrays as input, assumes the distributions are equally weighted.
  mu_hat = np.mean(mu)
  sigma_hat = np.mean(np.square(mu) + np.square(sigma)) - np.square(mu_hat)
  return mu_hat, sigma_hat




def starting_point(start_classes, c_means, c_variances):
  #start_classes - np array of classes to average for starting point
  start_c_mus = c_means[start_classes]
  start_c_sigmas = np.sqrt(c_variances[start_classes])
  convolved_start_mu, convolved_start_sigma = convolve_gaussians(start_c_mus, start_c_sigmas)
  return convolved_start_mu, convolved_start_sigma
  #Gives the starting "average digit" of the classees given in start_classes


def evaluation_spacing(start_sigma, start_mu, spacing='sd', num_steps=5):
  if spacing == 'linear':
    intermediate_dim_values = np.linspace(c_means[c],c_means[target],args.numsteps) #Replace with standard deviations.
  else:
    sds = np.linspace(-1*num_steps, num_steps, num=(2*num_steps + 1))
    #e.g. numsteps=3, sds=[-3,-2,1,0,1,2,3]
    intermediate_dim_values = np.full_like(sds, start_mu)
    for x in range(len(intermediate_dim_values)):
      intermediate_dim_values[x] += start_sigma*sds[x]

    return intermediate_dim_values








  # DEPRICATED #
  # for val in intermediate_dim_values:
  #   #Generate the image.
  #   generated_im = vae.generate(np.expand_dims(code, 0)).reshape(IMG_X, IMG_Y, IMG_C)
  #   #Replaced with cpvae.sample()
  #   if IMG_C == 1: canvas = np.squeeze(generated_im)
  #   im = toimage(canvas, mode='L')
  #   filename = "generated_class{}_dim{}_val{}".format(c, args.gendim, val)
  #   im.save('{}.png'.format(filename))


#  for c in args.genclass:
